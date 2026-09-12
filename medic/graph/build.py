"""The graph: triage -> investigate (tool loop) -> hypothesize -> critic -> fix -> review.

Nodes are closures over the model, the tools and the sandbox operations, so
the same graph runs live (Gemini plus a real sandbox) and in tests (a
scripted model, replay tools, recorded sandbox results).

Investigation (Phase 1b): the investigator is one model turn per node visit;
`run_tools` executes the calls it made; a conditional edge decides whether to
go around again, hand over to the hypothesis step, or escalate when a budget
is exceeded.

Fix (Phase 2): the fixer is also one model turn per visit and may read a few
files before it submits a proposal. `apply_fix` applies the edits through the
guard, `validator_node` rebuilds the affected models in the sandbox, and a
failure feeds the output back to the fixer up to `max_fix_iterations` times.
Anything that reaches `human_review` stops there on `interrupt()` and waits
for `Command(resume=...)`; approval opens the pull request (dry-run by
default), rejection escalates. Without sandbox operations the graph ends
after the critic, which is what `medic triage` and the replay tests use.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import ValidationError

from medic.agents import prompts
from medic.agents.schemas import (
    EvidenceItem,
    HypothesisSet,
    ProposeFix,
    SubmitEvidence,
    TriageVerdict,
)
from medic.config import LLM
from medic.critic import run_critic
from medic.graph.ops import SandboxOps
from medic.graph.state import (
    Approval,
    Budget,
    Evidence,
    FixAttempt,
    Hallucination,
    Hypothesis,
    LLMTurn,
    MedicState,
    ProposedFix,
    ToolRecord,
    Triage,
)
from medic.llm import invoke_with_backoff

Log = Callable[[str], None]
# Identical tool calls tolerated before the investigator is told to submit.
MAX_REPEATED_CALLS = 2
# The incident summary shown to the investigator is citable under this tag.
INCIDENT_TAG = "T0"
FIXER_TOOL_NAMES = ("read_pipeline_file", "list_pipeline_files")


def _turn(node: str, ai: AIMessage) -> LLMTurn:
    usage = ai.usage_metadata or {}
    return LLMTurn(
        node=node,
        content=ai.content if isinstance(ai.content, str) else json.dumps(ai.content),
        tool_calls=[
            {"name": c["name"], "args": c.get("args") or {}, "id": c.get("id")}
            for c in ai.tool_calls
        ],
        usage={
            k: int(v)
            for k, v in usage.items()
            if isinstance(v, int | float) and not isinstance(v, bool)
        },
    )


def _charge(budget: Budget, ai: AIMessage, **counters: int) -> Budget:
    usage = ai.usage_metadata or {}
    return budget.model_copy(
        update={
            "llm_calls": budget.llm_calls + 1,
            "tokens": budget.tokens + int(usage.get("total_tokens", 0)),
            **counters,
        }
    )


def _signature(name: str, args: dict[str, Any]) -> tuple[str, str]:
    return name, json.dumps(args, sort_keys=True, default=str)


def _records_by_id(records: list[ToolRecord]) -> dict[str, ToolRecord]:
    """Look up by the visible tag (T3) first, then by the API's own call id."""
    by_id: dict[str, ToolRecord] = {}
    for r in records:
        if r.api_call_id:
            by_id.setdefault(r.api_call_id, r)
    for r in records:
        by_id[r.id] = r
    return by_id


def _first_call(ai: AIMessage, name: str) -> dict[str, Any] | None:
    for call in ai.tool_calls:
        if call["name"] == name:
            return call.get("args") or {}
    return None


def _last_turn_names(state: MedicState, node: str) -> list[str]:
    """Tool names on the last model turn, independent of how the reducer stored it."""
    turns = state.get("llm_turns") or []
    if turns and turns[-1].node == node:
        return [c["name"] for c in turns[-1].tool_calls]
    return []


def review_payload(state: MedicState) -> dict[str, Any]:
    """What the reviewer is shown at the interrupt, and what the CLI prints."""
    incident = state["incident"]
    attempts = state.get("fix_attempts") or []
    last = attempts[-1] if attempts else None
    top = (state.get("hypotheses") or [None])[0]
    validation = state.get("validation")
    return {
        "run_id": incident.run_id,
        "scenario": incident.scenario,
        "failing_nodes": [n.name for n in incident.failing_nodes],
        "root_cause": top.root_cause if top else None,
        "kind": last.proposal.kind if last else None,
        "explanation": last.proposal.explanation if last else None,
        "recommended_action": last.proposal.recommended_action if last else None,
        "diff": last.apply.diff if last and last.apply else "",
        "validation": validation.summary() if validation else None,
        "attempts": [a.outcome for a in attempts],
        "budget": state["budget"].model_dump(),
    }


def build_graph(
    model: BaseChatModel,
    tools: list[BaseTool],
    *,
    ops: SandboxOps | None = None,
    budget: Budget | None = None,
    callbacks: list[Any] | None = None,
    log: Log | None = None,
    sleep: Callable[[float], None] = time.sleep,
    min_interval_s: float = LLM.min_interval_s,
    checkpointer: Any = None,
):
    """Compile the graph. `budget` sets the limits; the counters start at zero.

    `ops` enables the fix phase; without it the graph ends after the critic.
    `checkpointer` is required for the fix phase to pause at human review.
    `min_interval_s` spaces model calls out (the free Gemini tier allows ten a
    minute); pass 0 and a no-op `sleep` in tests.
    """
    limits = budget or Budget()
    config = {"callbacks": callbacks or []}
    by_name = {t.name: t for t in tools}
    fixer_tools = [by_name[n] for n in FIXER_TOOL_NAMES if n in by_name]
    say: Log = log or (lambda _msg: None)
    last_call: list[float] = []

    def invoke(runnable, messages) -> AIMessage:
        if last_call and min_interval_s > 0:
            wait = min_interval_s - (time.monotonic() - last_call[0])
            if wait > 0:
                sleep(wait)
        try:
            return invoke_with_backoff(runnable, messages, config=config, sleep=sleep, log=log)
        finally:
            last_call[:] = [time.monotonic()]

    def read_file_record(budget: Budget, path: str, origin: str) -> tuple[ToolRecord, Budget]:
        """Read a file through the guarded tool on the graph's own initiative."""
        budget = budget.model_copy(update={"tool_calls": budget.tool_calls + 1})
        ref = f"T{budget.tool_calls}"
        tool = by_name.get("read_pipeline_file")
        error = False
        if tool is None:
            body, error = json.dumps({"error": "read_pipeline_file is not available"}), True
        else:
            try:
                out = tool.invoke({"path": path}, config=config)
                body = out if isinstance(out, str) else json.dumps(out, default=str)
                error = body.startswith('{"error"')
            except Exception as exc:  # noqa: BLE001 - shown to the model instead
                body, error = json.dumps({"error": f"{type(exc).__name__}: {exc}"}), True
        record = ToolRecord(
            id=ref,
            tool="read_pipeline_file",
            args={"path": path},
            output=f"[{ref} read_pipeline_file]\n{body}",
            error=error,
            origin=origin,
        )
        return record, budget

    # ------------------------------------------------- investigation nodes
    def triage_agent(state: MedicState) -> dict:
        incident = state["incident"]
        llm = model.bind_tools([TriageVerdict], tool_choice=TriageVerdict.__name__)
        ai = invoke(
            llm,
            [
                SystemMessage(prompts.TRIAGE_SYSTEM),
                HumanMessage(prompts.incident_message(incident)),
            ],
        )
        args = _first_call(ai, TriageVerdict.__name__)
        triage: Triage | None = None
        if args is not None:
            try:
                v = TriageVerdict.model_validate(args)
                triage = Triage(category=v.category, confidence=v.confidence, rationale=v.rationale)
            except ValidationError as exc:
                say(f"triage: invalid verdict ({exc.error_count()} errors); continuing without")
        else:
            say("triage: model returned no verdict; continuing without")
        if triage:
            say(f"triage: {triage.category} ({triage.confidence:.2f}) {triage.rationale}")
        return {
            "triage": triage,
            "llm_turns": [_turn("triage_agent", ai)],
            "budget": _charge(state.get("budget") or limits, ai),
            "status": "investigating",
        }

    def investigator_agent(state: MedicState) -> dict:
        budget = state["budget"]
        incident = state["incident"]
        history = list(state.get("messages") or [])
        system = SystemMessage(
            prompts.investigator_system(state.get("triage"), budget.tool_calls_left)
        )
        new_messages: list = []
        if not history:
            new_messages.append(HumanMessage(prompts.incident_message(incident, tagged=True)))
            convo = [*new_messages]
        else:
            convo = history
        last = convo[-1] if convo else None
        stalled = isinstance(last, AIMessage) and not last.tool_calls
        looping = budget.repeated_calls >= MAX_REPEATED_CALLS
        must_submit = stalled or looping or budget.tool_calls_left <= 0
        if must_submit:
            llm = model.bind_tools([*tools, SubmitEvidence], tool_choice=SubmitEvidence.__name__)
            if history:
                why = (
                    "You are repeating tool calls that return the same result."
                    if looping
                    else "Your tool budget is spent."
                    if budget.tool_calls_left <= 0
                    else ""
                )
                nudge = HumanMessage(
                    f"{why} Submit your evidence now with SubmitEvidence, quoting only from "
                    "results you already have.".strip()
                )
                new_messages.append(nudge)
                convo = [*convo, nudge]
        else:
            llm = model.bind_tools([*tools, SubmitEvidence])
        ai = invoke(llm, [system, *convo])
        names = [c["name"] for c in ai.tool_calls]
        say(f"investigator: {', '.join(names) if names else 'no tool call'}")
        return {
            "messages": [*new_messages, ai],
            "llm_turns": [_turn("investigator_agent", ai)],
            "budget": _charge(budget, ai),
        }

    def execute_calls(
        state: MedicState, ai: AIMessage, *, dedupe: bool, skip: tuple[str, ...]
    ) -> tuple[list[ToolMessage], list[ToolRecord], Budget]:
        budget = state["budget"]
        records: list[ToolRecord] = []
        replies: list[ToolMessage] = []
        # First occurrence wins, so a repeat always points at the original result.
        seen: dict[tuple[str, str], ToolRecord] = {}
        if dedupe:
            for r in state.get("tool_records") or []:
                seen.setdefault(_signature(r.tool, r.args), r)
        for call in ai.tool_calls:
            name, args, api_id = (
                call["name"],
                call.get("args") or {},
                call.get("id") or f"call_{budget.tool_calls}",
            )
            if name in skip:
                continue
            budget = budget.model_copy(update={"tool_calls": budget.tool_calls + 1})
            # The id the model cites. Gemini never shows the model the API's
            # own call ids, so every result is tagged with this one instead.
            ref = f"T{budget.tool_calls}"
            tool = by_name.get(name)
            error = False
            previous = seen.get(_signature(name, args)) if dedupe else None
            if previous is not None:
                # Same tool, same arguments: the answer cannot change. Hand the
                # old result back rather than spend the call, and count it.
                budget = budget.model_copy(update={"repeated_calls": budget.repeated_calls + 1})
                body = (
                    f"REPEATED CALL: identical to {previous.id}; the result is the same and will "
                    f"not change. Do not call this again. Cite {previous.id} for it."
                )
                error = previous.error
            elif tool is None:
                body = json.dumps({"error": f"unknown tool {name!r}; available: {sorted(by_name)}"})
                error = True
            else:
                try:
                    out = tool.invoke(args, config=config)
                    body = out if isinstance(out, str) else json.dumps(out, default=str)
                except Exception as exc:  # noqa: BLE001 - surface to the model, keep going
                    body = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
                    error = True
            output = f"[{ref} {name}]\n{body}"
            record = ToolRecord(
                id=ref,
                api_call_id=api_id,
                tool=name,
                args=args,
                output=output,
                error=error,
                repeat_of=previous.id if previous is not None else None,
            )
            records.append(record)
            if dedupe:
                seen.setdefault(_signature(name, args), record)
            replies.append(ToolMessage(content=output, tool_call_id=api_id, name=name))
            say(
                f"  {name}({json.dumps(args, default=str)[:120]}) -> {len(output)} chars{' [error]' if error else ''}"
            )
        return replies, records, budget

    def run_tools(state: MedicState) -> dict:
        ai = state["messages"][-1]
        assert isinstance(ai, AIMessage)
        replies, records, budget = execute_calls(
            state, ai, dedupe=True, skip=(SubmitEvidence.__name__,)
        )
        return {"messages": replies, "tool_records": records, "budget": budget}

    def record_evidence(state: MedicState) -> dict:
        ai = state["messages"][-1]
        assert isinstance(ai, AIMessage)
        args = _first_call(ai, SubmitEvidence.__name__) or {}
        by_id = _records_by_id(state.get("tool_records") or [])
        existing = list(state.get("evidence") or [])
        # Validate item by item: one malformed entry must not discard the rest.
        raw_items = args.get("evidence")
        if not isinstance(raw_items, list):
            say("evidence: submission carried no evidence list")
            raw_items = []
        items: list[EvidenceItem] = []
        malformed: list[Hallucination] = []
        for n, raw in enumerate(raw_items, start=1):
            try:
                items.append(EvidenceItem.model_validate(raw))
            except ValidationError as exc:
                fields = ", ".join(str(e["loc"][0]) for e in exc.errors() if e.get("loc"))
                claim = str(raw.get("claim", "")) if isinstance(raw, dict) else str(raw)[:200]
                malformed.append(
                    Hallucination(
                        kind="evidence",
                        ref=f"submitted item {n}",
                        claim=claim,
                        reason=f"malformed evidence item ({fields or 'invalid'})",
                    )
                )
        if malformed:
            say(f"evidence: {len(malformed)} malformed item(s) dropped")
        new: list[Evidence] = []
        for i, item in enumerate(items, start=len(existing) + 1):
            record = by_id.get(item.tool_call_id.strip())
            if record is not None and record.repeat_of:
                # A citation of a repeated call means the original result.
                record = by_id.get(record.repeat_of, record)
            new.append(
                Evidence(
                    id=f"E{i}",
                    claim=item.claim,
                    tool=record.tool if record else "unknown",
                    tool_call_id=record.id if record else item.tool_call_id,
                    excerpt=item.excerpt,
                )
            )
        say(f"evidence: {len(new)} items submitted")
        return {"evidence": existing + new, "hallucinations": malformed, "status": "hypothesizing"}

    def hypothesis_agent(state: MedicState) -> dict:
        budget = state["budget"]
        llm = model.bind_tools([HypothesisSet], tool_choice=HypothesisSet.__name__)
        human = HumanMessage(
            prompts.hypothesis_message(
                state["incident"], state.get("triage"), state.get("evidence") or []
            )
        )
        ai = invoke(llm, [SystemMessage(prompts.HYPOTHESIS_SYSTEM), human])
        args = _first_call(ai, HypothesisSet.__name__)
        hypotheses: list[Hypothesis] = []
        if args is not None:
            try:
                parsed = HypothesisSet.model_validate(args)
                hypotheses = [
                    Hypothesis(rank=i, **item.model_dump())
                    for i, item in enumerate(parsed.hypotheses, start=1)
                ]
            except ValidationError as exc:
                say(f"hypothesis: invalid set ({exc.error_count()} errors)")
        say(f"hypothesis: {len(hypotheses)} ranked")
        return {
            "hypotheses": hypotheses,
            "llm_turns": [_turn("hypothesis_agent", ai)],
            "budget": _charge(budget, ai),
        }

    def critic_node(state: MedicState) -> dict:
        result = run_critic(
            state.get("evidence") or [],
            state.get("hypotheses") or [],
            state.get("tool_records") or [],
        )
        say(
            f"critic: kept {len(result.evidence)} evidence, {len(result.hypotheses)} hypotheses; "
            f"stripped {result.stripped_evidence} evidence, {result.stripped_hypotheses} hypothesis issues"
        )
        return {
            "evidence": result.evidence,
            "hypotheses": result.hypotheses,
            "hallucinations": result.hallucinations,
            "status": "investigated",
        }

    # ------------------------------------------------------------ fix nodes
    def fixer_agent(state: MedicState) -> dict:
        assert ops is not None
        budget = state["budget"]
        history = list(state.get("fix_messages") or [])
        new_messages: list = []
        records: list[ToolRecord] = []
        if not history:
            # Show the fixer the current text of the files most likely involved,
            # read through the same guarded tool the model would use.
            file_records: list[ToolRecord] = []
            for path in ops.files_to_read(state["incident"], state.get("hypotheses") or []):
                record, budget = read_file_record(budget, path, "fixer_preread")
                records.append(record)
                if not record.error:
                    file_records.append(record)
            new_messages.append(
                HumanMessage(
                    prompts.fixer_message(
                        state["incident"],
                        state.get("triage"),
                        state.get("hypotheses") or [],
                        state.get("evidence") or [],
                        file_records,
                    )
                )
            )
            convo = [*new_messages]
        else:
            convo = history
        last = convo[-1] if convo else None
        stalled = isinstance(last, AIMessage) and not last.tool_calls
        turns_left = max(0, budget.max_fixer_turns - budget.fixer_turns)
        must_submit = stalled or turns_left <= 0
        system = SystemMessage(
            prompts.fixer_system(budget.fix_iterations_left, turns_left if not must_submit else 0)
        )
        if must_submit:
            llm = model.bind_tools([*fixer_tools, ProposeFix], tool_choice=ProposeFix.__name__)
            if history:
                nudge = HumanMessage("Submit your proposal now with ProposeFix.")
                new_messages.append(nudge)
                convo = [*convo, nudge]
        else:
            llm = model.bind_tools([*fixer_tools, ProposeFix])
        ai = invoke(llm, [system, *convo])
        names = [c["name"] for c in ai.tool_calls]
        say(f"fixer: {', '.join(names) if names else 'no tool call'}")
        return {
            "fix_messages": [*new_messages, ai],
            "tool_records": records,
            "llm_turns": [_turn("fixer_agent", ai)],
            "budget": _charge(budget, ai, fixer_turns=budget.fixer_turns + 1),
            "status": "fixing",
        }

    def run_fix_tools(state: MedicState) -> dict:
        ai = state["fix_messages"][-1]
        assert isinstance(ai, AIMessage)
        # No dedupe here: the fixer may legitimately re-read a file it changed.
        replies, records, budget = execute_calls(
            state, ai, dedupe=False, skip=(ProposeFix.__name__,)
        )
        return {"fix_messages": replies, "tool_records": records, "budget": budget}

    def apply_fix(state: MedicState) -> dict:
        assert ops is not None
        ai = state["fix_messages"][-1]
        assert isinstance(ai, AIMessage)
        budget = state["budget"]
        attempts = list(state.get("fix_attempts") or [])
        iteration = budget.fix_iterations + 1
        budget = budget.model_copy(update={"fix_iterations": iteration, "fixer_turns": 0})
        args = _first_call(ai, ProposeFix.__name__) or {}
        try:
            parsed = ProposeFix.model_validate(args)
            proposal = ProposedFix(**parsed.model_dump())
        except ValidationError as exc:
            say(f"fixer: malformed proposal ({exc.error_count()} errors)")
            proposal = ProposedFix(
                kind="needs_human",
                explanation=f"malformed proposal: {exc.errors()[0].get('msg', 'invalid')}",
            )
        if proposal.kind in ("code_patch", "test_change") and not proposal.edits:
            proposal = proposal.model_copy(
                update={
                    "kind": "needs_human",
                    "explanation": f"{proposal.kind} proposed with no edits. {proposal.explanation}",
                }
            )
        attempt = FixAttempt(iteration=iteration, proposal=proposal)
        out: dict[str, Any] = {"proposed_fix": proposal, "validation": None}
        if not proposal.is_code_change:
            if any(a.apply and a.apply.ok for a in attempts):
                ops.reset()
                say("fixer: declined after earlier edits; sandbox reset")
            say(f"fixer: {proposal.kind}; {proposal.recommended_action or proposal.explanation}")
            attempts.append(attempt)
            return {**out, "fix_attempts": attempts, "budget": budget, "status": "awaiting_review"}
        result = ops.apply(proposal.edits, proposal.kind)
        attempt.apply = result
        attempts.append(attempt)
        if result.ok:
            say(f"fixer: {proposal.kind}, edited {', '.join(result.edited_paths)}")
            return {**out, "fix_attempts": attempts, "budget": budget, "status": "validating"}
        errors = "; ".join(o.error or "" for o in result.outcomes if not o.ok)
        say(f"fixer: edits rejected: {errors[:300]}")
        messages: list = []
        if budget.fix_iterations_left > 0 and not budget.exceeded():
            messages.append(HumanMessage(prompts.fix_feedback_message(attempt, [])))
        return {
            **out,
            "fix_attempts": attempts,
            "budget": budget,
            "fix_messages": messages,
            "status": "fixing",
        }

    def validator_node(state: MedicState) -> dict:
        assert ops is not None
        budget = state["budget"]
        attempts = list(state.get("fix_attempts") or [])
        attempt = attempts[-1]
        assert attempt.apply is not None
        validation = ops.validate(attempt.apply.edited_paths)
        attempt.validation = validation
        say(f"validator: {'passed' if validation.passed else 'failed'}; {validation.summary()}")
        if validation.passed:
            return {
                "fix_attempts": attempts,
                "validation": validation,
                "status": "awaiting_review",
            }
        records: list[ToolRecord] = []
        messages: list = []
        if budget.fix_iterations_left > 0 and not budget.exceeded():
            current: list[ToolRecord] = []
            for path in attempt.apply.edited_paths:
                record, budget = read_file_record(budget, path, "fixer_feedback")
                records.append(record)
                if not record.error:
                    current.append(record)
            messages.append(HumanMessage(prompts.fix_feedback_message(attempt, current)))
        return {
            "fix_attempts": attempts,
            "validation": validation,
            "tool_records": records,
            "fix_messages": messages,
            "budget": budget,
            "status": "validating",
        }

    def human_review(state: MedicState) -> dict:
        assert ops is not None
        decision = interrupt(review_payload(state))
        try:
            approval = Approval.model_validate(
                {
                    **(decision if isinstance(decision, dict) else {"status": decision}),
                    "decided_at": datetime.now(UTC).isoformat(timespec="seconds"),
                }
            )
        except ValidationError as exc:
            raise ValueError(
                f"resume value must be {{status: approved|rejected, note}}: {exc}"
            ) from exc
        proposal = state.get("proposed_fix")
        validation = state.get("validation")
        say(f"review: {approval.status}{' - ' + approval.note if approval.note else ''}")
        if approval.status == "rejected":
            ops.reset()
            return {"approval": approval, "status": "rejected"}
        if proposal and proposal.is_code_change and validation and validation.passed:
            return {"approval": approval, "status": "approved"}
        return {"approval": approval, "status": "done"}

    def open_pr(state: MedicState) -> dict:
        assert ops is not None
        attempt = (state.get("fix_attempts") or [])[-1]
        top = (state.get("hypotheses") or [None])[0]
        pr = ops.open_pr(
            top,
            state.get("evidence") or [],
            attempt,
            state.get("validation"),
            (state.get("approval") or Approval(status="approved")).note,
        )
        say(f"pull request ({pr.mode}): {pr.url or pr.path}")
        return {"pull_request": pr, "status": "done"}

    def escalate_report(state: MedicState) -> dict:
        budget = state["budget"]
        approval = state.get("approval")
        attempts = state.get("fix_attempts") or []
        if approval and approval.status == "rejected":
            reason = f"rejected by reviewer: {approval.note or 'no note'}"
        elif attempts and budget.fix_iterations_left == 0 and attempts[-1].outcome != "passed":
            reason = f"fix did not pass after {len(attempts)} attempt(s): {attempts[-1].outcome}"
        else:
            reason = budget.exceeded() or "the investigator produced no usable turn"
        say(f"escalate: {reason}")
        return {"status": "escalated", "escalation_reason": reason}

    # ------------------------------------------------------------ routing
    def entry(state: MedicState) -> Literal["triage_agent", "fixer_agent"]:
        # A thread seeded from a finished investigation starts at the fixer.
        if ops is not None and state.get("status") == "investigated":
            return "fixer_agent"
        return "triage_agent"

    def after_investigator(
        state: MedicState,
    ) -> Literal["run_tools", "record_evidence", "escalate_report", "investigator_agent"]:
        names = _last_turn_names(state, "investigator_agent")
        if SubmitEvidence.__name__ in names:
            # The investigation is over and its evidence is already paid for;
            # ranking it costs one more call, so a budget overrun does not
            # discard it. Escalation is for an investigation that would go on.
            return "record_evidence"
        if state["budget"].exceeded() or not state.get("llm_turns"):
            return "escalate_report"
        if names:
            return "run_tools"
        return "investigator_agent"

    def after_tools(state: MedicState) -> Literal["investigator_agent", "escalate_report"]:
        return "escalate_report" if state["budget"].exceeded() else "investigator_agent"

    def after_fixer(
        state: MedicState,
    ) -> Literal["apply_fix", "run_fix_tools", "escalate_report", "fixer_agent"]:
        names = _last_turn_names(state, "fixer_agent")
        if ProposeFix.__name__ in names:
            return "apply_fix"
        if state["budget"].exceeded():
            return "escalate_report"
        if names:
            return "run_fix_tools"
        return "fixer_agent"

    def after_fix_tools(state: MedicState) -> Literal["fixer_agent", "escalate_report"]:
        return "escalate_report" if state["budget"].exceeded() else "fixer_agent"

    def can_retry(state: MedicState) -> bool:
        budget = state["budget"]
        return budget.fix_iterations_left > 0 and not budget.exceeded()

    def after_apply(
        state: MedicState,
    ) -> Literal["human_review", "validator_node", "fixer_agent", "escalate_report"]:
        attempt = state["fix_attempts"][-1]
        if not attempt.proposal.is_code_change:
            return "human_review"
        if attempt.apply is not None and attempt.apply.ok:
            return "validator_node"
        return "fixer_agent" if can_retry(state) else "escalate_report"

    def after_validation(
        state: MedicState,
    ) -> Literal["human_review", "fixer_agent", "escalate_report"]:
        validation = state.get("validation")
        if validation is not None and validation.passed:
            return "human_review"
        return "fixer_agent" if can_retry(state) else "escalate_report"

    def after_review(state: MedicState) -> Literal["open_pr", "escalate_report", "__end__"]:
        status = state.get("status")
        if status == "approved":
            return "open_pr"
        if status == "rejected":
            return "escalate_report"
        return END

    graph = StateGraph(MedicState)
    graph.add_node("triage_agent", triage_agent)
    graph.add_node("investigator_agent", investigator_agent)
    graph.add_node("run_tools", run_tools)
    graph.add_node("record_evidence", record_evidence)
    graph.add_node("hypothesis_agent", hypothesis_agent)
    graph.add_node("critic_node", critic_node)
    graph.add_node("escalate_report", escalate_report)

    graph.add_edge("triage_agent", "investigator_agent")
    graph.add_conditional_edges("investigator_agent", after_investigator)
    graph.add_conditional_edges("run_tools", after_tools)
    graph.add_edge("record_evidence", "hypothesis_agent")
    graph.add_edge("hypothesis_agent", "critic_node")
    graph.add_edge("escalate_report", END)

    if ops is None:
        graph.add_edge(START, "triage_agent")
        graph.add_edge("critic_node", END)
    else:
        graph.add_conditional_edges(START, entry)
        graph.add_edge("critic_node", "fixer_agent")
        graph.add_node("fixer_agent", fixer_agent)
        graph.add_node("run_fix_tools", run_fix_tools)
        graph.add_node("apply_fix", apply_fix)
        graph.add_node("validator_node", validator_node)
        graph.add_node("human_review", human_review)
        graph.add_node("open_pr", open_pr)
        graph.add_conditional_edges("fixer_agent", after_fixer)
        graph.add_conditional_edges("run_fix_tools", after_fix_tools)
        graph.add_conditional_edges("apply_fix", after_apply)
        graph.add_conditional_edges("validator_node", after_validation)
        graph.add_conditional_edges("human_review", after_review)
        graph.add_edge("open_pr", END)
    return graph.compile(checkpointer=checkpointer)


def incident_record(incident) -> ToolRecord:
    """The incident summary as record T0, so quotes from it can be verified too."""
    return ToolRecord(
        id=INCIDENT_TAG,
        tool="incident",
        args={},
        output=prompts.incident_message(incident, tagged=True),
    )


def initial_state(incident, budget: Budget | None = None) -> MedicState:
    return {
        "incident": incident,
        "triage": None,
        "messages": [],
        "tool_records": [incident_record(incident)],
        "llm_turns": [],
        "hallucinations": [],
        "evidence": [],
        "hypotheses": [],
        "budget": budget or Budget(),
        "status": "triaging",
        "escalation_reason": None,
        "fix_messages": [],
        "proposed_fix": None,
        "fix_attempts": [],
        "validation": None,
        "approval": None,
        "pull_request": None,
    }


def investigated_state(
    incident,
    *,
    triage: Triage | None,
    evidence: list[Evidence],
    hypotheses: list[Hypothesis],
    hallucinations: list[Hallucination] | None = None,
    budget: Budget | None = None,
) -> MedicState:
    """A finished investigation, ready to enter the graph at the fixer."""
    state = initial_state(incident, budget)
    state.update(
        {
            "triage": triage,
            "evidence": list(evidence),
            "hypotheses": list(hypotheses),
            "hallucinations": list(hallucinations or []),
            "status": "investigated",
        }
    )
    return state
