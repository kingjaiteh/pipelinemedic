"""Record a run so it can be replayed with no API key and no sandbox.

A transcript is the incident, every model turn, and every tool call with
its output. `ReplayChatModel` serves the model turns in order;
`replay_tools` answers each tool call from the recorded outputs. Running the
real graph over them exercises routing, budget accounting, evidence
recording and the critic exactly as the live run did.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field, create_model

from medic.graph.state import (
    ApplyResult,
    Approval,
    Budget,
    EditSpec,
    Evidence,
    FixAttempt,
    Hallucination,
    Hypothesis,
    Incident,
    LLMTurn,
    MedicState,
    PullRequest,
    ToolRecord,
    Triage,
    Validation,
)
from medic.tools.registry import READ_ONLY_TOOL_NAMES

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


class Transcript(BaseModel):
    scenario_key: str | None = None
    model: str = "unknown"
    incident: Incident
    llm_turns: list[LLMTurn]
    tool_records: list[ToolRecord]
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    hallucinations: list[Hallucination] = Field(default_factory=list)
    budget: Budget
    status: str
    # Fix runs. A fix transcript is seeded from a finished investigation, so
    # it carries the evidence and hypotheses the fixer started from and the
    # seed budget; `budget` above is the final one.
    triage: Triage | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    seed_budget: Budget | None = None
    fix_attempts: list[FixAttempt] = Field(default_factory=list)
    approval: Approval | None = None
    pull_request: PullRequest | None = None
    escalation_reason: str | None = None

    @classmethod
    def from_state(
        cls, state: MedicState, model: str = "unknown", seed_budget: Budget | None = None
    ) -> Transcript:
        incident = state["incident"]
        return cls(
            scenario_key=incident.scenario_key,
            model=model,
            incident=incident,
            llm_turns=list(state.get("llm_turns") or []),
            tool_records=list(state.get("tool_records") or []),
            hypotheses=list(state.get("hypotheses") or []),
            hallucinations=list(state.get("hallucinations") or []),
            budget=state["budget"],
            status=state.get("status") or "unknown",
            triage=state.get("triage"),
            evidence=list(state.get("evidence") or []),
            seed_budget=seed_budget,
            fix_attempts=list(state.get("fix_attempts") or []),
            approval=state.get("approval"),
            pull_request=state.get("pull_request"),
            escalation_reason=state.get("escalation_reason"),
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> Transcript:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    @property
    def is_fix_run(self) -> bool:
        return self.seed_budget is not None


def fixture_path(key: str) -> Path:
    return FIXTURES_DIR / key / "transcript.json"


def fix_fixture_path(key: str) -> Path:
    return FIXTURES_DIR / key / "fix_transcript.json"


class ReplayChatModel(BaseChatModel):
    """Replays recorded AI turns in order. bind_tools records what was bound."""

    script: list[AIMessage]
    turn: int = 0
    bound: list[list[str]] = Field(default_factory=list)
    tool_choices: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "replay"

    def bind_tools(self, tools: list[Any], *, tool_choice: Any = None, **kwargs: Any):  # type: ignore[override]
        self.bound.append(
            [getattr(t, "name", None) or getattr(t, "__name__", str(t)) for t in tools]
        )
        self.tool_choices.append(tool_choice)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if self.turn >= len(self.script):
            raise IndexError(f"replay script exhausted after {len(self.script)} turns")
        msg = self.script[self.turn]
        self.turn += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @classmethod
    def from_turns(cls, turns: list[LLMTurn]) -> ReplayChatModel:
        script = [
            AIMessage(
                content=t.content,
                tool_calls=[
                    {"name": c["name"], "args": c["args"], "id": c.get("id")} for c in t.tool_calls
                ],
                usage_metadata=(
                    {
                        "input_tokens": t.usage.get("input_tokens", 0),
                        "output_tokens": t.usage.get("output_tokens", 0),
                        "total_tokens": t.usage.get("total_tokens", 0),
                    }
                    if t.usage
                    else None
                ),
            )
            for t in turns
        ]
        return cls(script=script)


class AnyArgs(BaseModel):
    model_config = ConfigDict(extra="allow")


TAG_LINE = re.compile(r"^\[T\d+ [A-Za-z_]+\]\n")


def strip_tag(output: str) -> str:
    """Recorded outputs carry the [T3 tool] tag the graph adds; the tool returns the body."""
    return TAG_LINE.sub("", output, count=1)


def replay_tools(
    records: list[ToolRecord], names: tuple[str, ...] = READ_ONLY_TOOL_NAMES
) -> list[BaseTool]:
    """One tool per name that answers from the recorded outputs, by call id then by args."""
    by_id = {r.id: r for r in records}
    by_args: dict[tuple[str, str], list[ToolRecord]] = {}
    for r in records:
        if r.repeat_of:
            continue  # the graph answers repeats itself; only originals are replayed
        by_args.setdefault((r.tool, json.dumps(r.args, sort_keys=True, default=str)), []).append(r)

    arg_names: dict[str, set[str]] = {n: set() for n in names}
    for r in records:
        arg_names.setdefault(r.tool, set()).update(r.args)

    def make(name: str) -> BaseTool:
        def func(**kwargs: Any) -> str:
            given = {k: v for k, v in kwargs.items() if v is not None}
            key = (name, json.dumps(given, sort_keys=True, default=str))
            queue = by_args.get(key)
            if queue:
                return strip_tag(queue.pop(0).output)
            return json.dumps({"error": f"replay: no recorded output for {name}({given})"})

        # LangChain drops every argument unless the schema declares it, so the
        # schema is built from the argument names this tool was recorded with.
        fields: dict[str, Any] = {k: (Any, None) for k in sorted(arg_names.get(name, ()))}
        schema = create_model(f"{name}_replay_args", **fields) if fields else AnyArgs
        return StructuredTool(
            name=name, description=f"replay of {name}", args_schema=schema, func=func
        )

    tools = [make(n) for n in names]
    # Allow lookups by id too, for callers that know the call id.
    for t in tools:
        t.metadata = {"records_by_id": by_id}
    return tools


class ReplaySandboxOps:
    """Answers the fix phase's sandbox operations from a recorded run.

    Apply and validation results come back per attempt, in recorded order;
    the files the fixer was shown first are the recorded pre-read records,
    which `replay_tools` then serves by path; the pull request is the one
    recorded.
    """

    def __init__(self, transcript: Transcript):
        self.transcript = transcript
        self.attempts = list(transcript.fix_attempts)
        self.applied: list[list[EditSpec]] = []
        self.resets = 0

    def files_to_read(self, incident: Incident, hypotheses: list[Hypothesis]) -> list[str]:
        return [
            str(r.args.get("path"))
            for r in self.transcript.tool_records
            if r.origin == "fixer_preread"
        ]

    def _attempt(self, what: str) -> FixAttempt:
        idx = len(self.applied) - 1
        if idx < 0 or idx >= len(self.attempts):
            raise IndexError(f"replay: no recorded attempt for {what} number {idx + 1}")
        return self.attempts[idx]

    def apply(self, edits: list[EditSpec], kind: str) -> ApplyResult:
        self.applied.append(list(edits))
        result = self._attempt("apply").apply
        if result is None:
            raise AssertionError("replay: the recorded attempt applied no edits")
        return result

    def validate(self, edited_paths: list[str]) -> Validation:
        result = self._attempt("validation").validation
        if result is None:
            raise AssertionError("replay: the recorded attempt was not validated")
        return result

    def diff(self) -> str:
        last = self.attempts[-1] if self.attempts else None
        return last.apply.diff if last and last.apply else ""

    def reset(self) -> None:
        self.resets += 1

    def open_pr(self, top, evidence, attempt, validation, reviewer_note) -> PullRequest:
        if self.transcript.pull_request is None:
            raise AssertionError("replay: no pull request was recorded")
        return self.transcript.pull_request
