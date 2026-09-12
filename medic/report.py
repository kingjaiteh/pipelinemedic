"""The triage report: what the graph concluded, in JSON and on the terminal."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from medic.graph.state import (
    Approval,
    Budget,
    Evidence,
    FixAttempt,
    Hallucination,
    Hypothesis,
    Incident,
    MedicState,
    PullRequest,
    Triage,
    Validation,
)


def _is_test_path(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    return "tests" in parts or "macros" in parts or parts[-1] == "dbt_project.yml"


def root_cause_matches(text: str, terms: tuple[str, ...] | list[str]) -> bool:
    """Every term must match; a term is a set of alternatives separated by '|'."""
    hay = text.lower()
    for term in terms:
        alternatives = [a.strip().lower() for a in term.split("|") if a.strip()]
        if not any(re.search(re.escape(a), hay) for a in alternatives):
            return False
    return bool(terms)


class TriageReport(BaseModel):
    created_at: str
    model: str = "unknown"
    incident: Incident
    triage: Triage | None
    hypotheses: list[Hypothesis]
    evidence: list[Evidence]
    hallucinations: list[Hallucination]
    budget: Budget
    status: str
    escalation_reason: str | None = None
    ground_truth: dict[str, Any] = Field(default_factory=dict)
    # Fix phase, present once `medic fix` has run on the incident.
    thread_id: str | None = None
    fix_attempts: list[FixAttempt] = Field(default_factory=list)
    validation: Validation | None = None
    approval: Approval | None = None
    pull_request: PullRequest | None = None

    @classmethod
    def from_state(
        cls,
        state: MedicState,
        ground_truth: dict[str, Any] | None = None,
        model: str = "unknown",
        thread_id: str | None = None,
    ) -> TriageReport:
        return cls(
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            model=model,
            incident=state["incident"],
            triage=state.get("triage"),
            hypotheses=list(state.get("hypotheses") or []),
            evidence=list(state.get("evidence") or []),
            hallucinations=list(state.get("hallucinations") or []),
            budget=state["budget"],
            status=state.get("status") or "unknown",
            escalation_reason=state.get("escalation_reason"),
            ground_truth=ground_truth or {},
            thread_id=thread_id,
            fix_attempts=list(state.get("fix_attempts") or []),
            validation=state.get("validation"),
            approval=state.get("approval"),
            pull_request=state.get("pull_request"),
        )

    @classmethod
    def load(cls, path: Path) -> TriageReport:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    @property
    def top(self) -> Hypothesis | None:
        return self.hypotheses[0] if self.hypotheses else None

    @property
    def last_attempt(self) -> FixAttempt | None:
        return self.fix_attempts[-1] if self.fix_attempts else None

    def with_ground_truth(
        self, category: str, fix_kind: str, terms: tuple[str, ...]
    ) -> TriageReport:
        top = self.hypotheses[0] if self.hypotheses else None
        text = f"{top.root_cause} {top.recommended_action}" if top else ""
        self.ground_truth = {
            "category": category,
            "fix_kind": fix_kind,
            "root_cause_terms": list(terms),
            "top1_root_cause_correct": bool(top) and root_cause_matches(text, terms),
            "top1_category_correct": bool(top) and top.category == category,
            "top1_fix_kind_correct": bool(top) and top.fix_kind == fix_kind,
            "triage_category_correct": bool(self.triage) and self.triage.category == category,
        }
        return self

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=1), encoding="utf-8")
        return path

    def render(self) -> str:
        lines = [
            f"status: {self.status}"
            + (f" ({self.escalation_reason})" if self.escalation_reason else "")
            + f"  model: {self.model}"
        ]
        if self.triage:
            lines.append(
                f"triage: {self.triage.category} ({self.triage.confidence:.2f}) {self.triage.rationale}"
            )
        lines.append(
            f"budget: {self.budget.tool_calls} tool calls, {self.budget.llm_calls} LLM calls, "
            f"{self.budget.tokens} tokens"
        )
        lines.append("")
        for h in self.hypotheses:
            lines.append(
                f"#{h.rank} [{h.category}, {h.fix_kind}, {h.confidence:.2f}] {h.root_cause}"
            )
            lines.append(f"    action: {h.recommended_action}")
            lines.append(f"    evidence: {', '.join(h.evidence_ids)}")
        if not self.hypotheses:
            lines.append("no hypotheses survived")
        lines.append("")
        for e in self.evidence:
            excerpt = e.excerpt if len(e.excerpt) <= 160 else e.excerpt[:157] + "..."
            lines.append(f"{e.id} [{e.tool} {e.tool_call_id}] {e.claim}")
            lines.append(f'    "{excerpt}"')
        if self.hallucinations:
            lines.append("")
            lines.append(f"stripped by critic ({len(self.hallucinations)}):")
            for hal in self.hallucinations:
                lines.append(f"  {hal.kind} {hal.ref}: {hal.reason}")
        if self.fix_attempts:
            lines.append("")
            lines.extend(self.render_fix())
        if self.ground_truth:
            g = self.ground_truth
            lines.append("")
            lines.append(
                "ground truth: top-1 root cause "
                + ("CORRECT" if g["top1_root_cause_correct"] else "WRONG")
                + f", category {'ok' if g['top1_category_correct'] else 'off'} (expected {g['category']})"
                + f", fix kind {'ok' if g['top1_fix_kind_correct'] else 'off'} (expected {g['fix_kind']})"
            )
            if "fixer_kind_correct" in g:
                lines.append(
                    f"fixer: kind {g['fixer_kind']} "
                    + ("matches" if g["fixer_kind_correct"] else "does not match")
                    + f" the expected {g['fix_kind']}"
                    + (", tests untouched" if g.get("tests_untouched") else ", TESTS TOUCHED")
                )
        return "\n".join(lines)

    def render_fix(self) -> list[str]:
        lines: list[str] = []
        for a in self.fix_attempts:
            lines.append(f"attempt {a.iteration}: {a.outcome}; {a.proposal.kind}")
            lines.append(f"    {a.proposal.explanation}")
            if a.proposal.recommended_action:
                lines.append(f"    action: {a.proposal.recommended_action}")
            if a.apply is not None and not a.apply.ok:
                for o in a.apply.outcomes:
                    if not o.ok:
                        lines.append(f"    rejected: {o.error}")
            if a.validation is not None:
                lines.append(f"    validation: {a.validation.summary()}")
        last = self.last_attempt
        if last and last.apply and last.apply.ok and last.apply.diff:
            lines.append("")
            lines.append(last.apply.diff.rstrip())
        if self.approval:
            lines.append("")
            lines.append(
                f"review: {self.approval.status}"
                + (f" ({self.approval.note})" if self.approval.note else "")
            )
        if self.pull_request:
            pr = self.pull_request
            lines.append(f"pull request ({pr.mode}): {pr.url or pr.path}")
            lines.append(f"    {pr.title}")
        if self.thread_id and self.status == "awaiting_review":
            lines.append("")
            lines.append("waiting for review. Resume with one of:")
            lines.append(f"  medic resume {self.thread_id} --approve")
            lines.append(f'  medic resume {self.thread_id} --reject "why"')
        return lines

    def with_fix_ground_truth(self, fix_kind: str) -> TriageReport:
        """Grade what the fixer did: kind, and whether tests were left alone."""
        last = self.last_attempt
        touched = any(
            _is_test_path(p)
            for a in self.fix_attempts
            if a.apply and a.apply.ok
            for p in a.apply.edited_paths
        )
        self.ground_truth.update(
            {
                "fixer_kind": last.proposal.kind if last else None,
                "fixer_kind_correct": bool(last) and last.proposal.kind == fix_kind,
                "sandbox_passed": bool(self.validation) and self.validation.passed,
                "tests_untouched": not touched,
            }
        )
        return self


def report_path(sandbox_root: Path, key: str) -> Path:
    return sandbox_root / "incidents" / f"{key}.triage.json"


def fix_report_path(sandbox_root: Path, key: str) -> Path:
    return sandbox_root / "incidents" / f"{key}.fix.json"


def summary_row(report: TriageReport) -> dict[str, Any]:
    top = report.hypotheses[0] if report.hypotheses else None
    g = report.ground_truth
    return {
        "scenario": report.incident.scenario,
        "model": report.model,
        "key": report.incident.scenario_key,
        "status": report.status,
        "triage": report.triage.category if report.triage else None,
        "top1_category": top.category if top else None,
        "top1_fix_kind": top.fix_kind if top else None,
        "top1_correct": g.get("top1_root_cause_correct"),
        "tool_calls": report.budget.tool_calls,
        "llm_calls": report.budget.llm_calls,
        "tokens": report.budget.tokens,
        "evidence": len(report.evidence),
        "stripped": len(report.hallucinations),
    }


def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=1, default=str)
