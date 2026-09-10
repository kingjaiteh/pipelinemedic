"""Evidence citation enforcement.

A claim survives only if its evidence id exists and its excerpt is a
substring of the stored output of the tool call it cites. Whitespace runs
and JSON string escapes are normalized on both sides before comparing,
because tool outputs are JSON and models unescape when they quote; nothing
else is forgiven. A hypothesis keeps only the evidence ids that survived and
is dropped if none did.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from medic.graph.state import Evidence, Hallucination, Hypothesis, ToolRecord

_WS = re.compile(r"\s+")
MIN_EXCERPT_CHARS = 8


def normalize(text: str) -> str:
    """Collapse whitespace and undo JSON string escaping, lowercase untouched."""
    unescaped = (
        text.replace('\\"', '"').replace("\\n", " ").replace("\\t", " ").replace("\\\\", "\\")
    )
    return _WS.sub(" ", unescaped).strip()


def excerpt_matches(excerpt: str, output: str) -> str | None:
    """'exact', 'normalized', or None."""
    if not excerpt.strip():
        return None
    if excerpt in output:
        return "exact"
    if normalize(excerpt) in normalize(output):
        return "normalized"
    return None


@dataclass
class CriticResult:
    evidence: list[Evidence]
    hypotheses: list[Hypothesis]
    hallucinations: list[Hallucination] = field(default_factory=list)
    match_kinds: dict[str, str] = field(default_factory=dict)

    @property
    def stripped_evidence(self) -> int:
        return sum(1 for h in self.hallucinations if h.kind == "evidence")

    @property
    def stripped_hypotheses(self) -> int:
        return sum(1 for h in self.hallucinations if h.kind == "hypothesis")


def check_evidence(evidence: list[Evidence], tool_records: list[ToolRecord]) -> CriticResult:
    by_id = {r.api_call_id: r for r in tool_records if r.api_call_id}
    by_id.update({r.id: r for r in tool_records})
    kept: list[Evidence] = []
    halluc: list[Hallucination] = []
    kinds: dict[str, str] = {}
    for e in evidence:
        record = by_id.get(e.tool_call_id)
        if record is None:
            halluc.append(
                Hallucination(
                    kind="evidence",
                    ref=e.id,
                    claim=e.claim,
                    reason=f"cites unknown tool_call_id {e.tool_call_id!r}",
                )
            )
            continue
        if len(e.excerpt.strip()) < MIN_EXCERPT_CHARS:
            halluc.append(
                Hallucination(
                    kind="evidence",
                    ref=e.id,
                    claim=e.claim,
                    reason=f"excerpt too short to verify ({len(e.excerpt.strip())} chars)",
                )
            )
            continue
        kind = excerpt_matches(e.excerpt, record.output)
        if kind is None:
            halluc.append(
                Hallucination(
                    kind="evidence",
                    ref=e.id,
                    claim=e.claim,
                    reason=f"excerpt is not a substring of the {record.tool} output for {e.tool_call_id}",
                )
            )
            continue
        kinds[e.id] = kind
        kept.append(e if e.tool == record.tool else e.model_copy(update={"tool": record.tool}))
    return CriticResult(evidence=kept, hypotheses=[], hallucinations=halluc, match_kinds=kinds)


def check_hypotheses(
    hypotheses: list[Hypothesis], surviving_evidence: list[Evidence]
) -> CriticResult:
    valid_ids = {e.id for e in surviving_evidence}
    kept: list[Hypothesis] = []
    halluc: list[Hallucination] = []
    for h in hypotheses:
        cited = [i for i in h.evidence_ids if i in valid_ids]
        dropped = [i for i in h.evidence_ids if i not in valid_ids]
        if not cited:
            halluc.append(
                Hallucination(
                    kind="hypothesis",
                    ref=f"rank {h.rank}",
                    claim=h.root_cause,
                    reason=(
                        "cites no evidence"
                        if not h.evidence_ids
                        else f"none of its evidence survived: {', '.join(h.evidence_ids)}"
                    ),
                )
            )
            continue
        if dropped:
            halluc.append(
                Hallucination(
                    kind="hypothesis",
                    ref=f"rank {h.rank}",
                    claim=h.root_cause,
                    reason=f"dropped citations that did not survive: {', '.join(dropped)}",
                )
            )
        kept.append(h.model_copy(update={"evidence_ids": cited}))
    for i, h in enumerate(kept, start=1):
        h.rank = i
    return CriticResult(evidence=list(surviving_evidence), hypotheses=kept, hallucinations=halluc)


def run_critic(
    evidence: list[Evidence], hypotheses: list[Hypothesis], tool_records: list[ToolRecord]
) -> CriticResult:
    ev = check_evidence(evidence, tool_records)
    hy = check_hypotheses(hypotheses, ev.evidence)
    return CriticResult(
        evidence=ev.evidence,
        hypotheses=hy.hypotheses,
        hallucinations=ev.hallucinations + hy.hallucinations,
        match_kinds=ev.match_kinds,
    )


def as_json(result: CriticResult) -> str:
    return json.dumps(
        {
            "evidence_kept": [e.id for e in result.evidence],
            "hypotheses_kept": [h.rank for h in result.hypotheses],
            "hallucinations": [h.model_dump() for h in result.hallucinations],
        },
        indent=1,
    )
