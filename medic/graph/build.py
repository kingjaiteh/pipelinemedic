"""The Phase 1b graph: triage -> investigate (tool loop) -> hypothesize -> critic.

Nodes are closures over the model and the tools so the same graph runs live
(Gemini plus sandbox tools) and in tests (a scripted model plus replay tools).
The investigator is one model turn per node visit; `run_tools` executes the
calls it made; a conditional edge decides whether to go around again, hand
over to the hypothesis step, or escalate when a budget is exceeded.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from medic.agents import prompts
from medic.agents.schemas import EvidenceItem, HypothesisSet, SubmitEvidence, TriageVerdict
from medic.config import LLM
from medic.critic import run_critic
from medic.graph.state import (
    Budget,
    Evidence,
    Hallucination,
    Hypothesis,
    LLMTurn,
    MedicState,
    ToolRecord,
    Triage,
)
from medic.llm import invoke_with_backoff

Log = Callable[[str], None]
# Identical tool calls tolerated before the investigator is told to submit.
MAX_REPEATED_CALLS = 2
# The incident summary shown to the investigator is citable under this tag.
INCIDENT_TAG = "T0"


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


def _charge(budget: Budget, ai: AIMessage) -> Budget:
    usage = ai.usage_metadata or {}
    return budget.model_copy(
        update={
            "llm_calls": budget.llm_calls + 1,
            "tokens": budget.tokens + int(usage.get("total_tokens", 0)),
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


def build_graph(
    model: BaseChatModel,
    tools: list[BaseTool],
    *,
    budget: Budget | None = None,
    callbacks: list[Any] | None = None,
    log: Log | None = None,
    sleep: Callable[[float], None] = time.sleep,
    min_interval_s: float = LLM.min_interval_s,
):
    """Compile the graph. `budget` sets the limits; the counters start at zero.

    `min_interval_s` spaces model calls out (the free Gemini tier allows ten a
    minute); pass 0 and a no-op `sleep` in tests.
    """
    limits = budget or Budget()
    config = {"callbacks": callbacks or []}
    by_name = {t.name: t for t in tools}
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

    # ------------------------------------------------------------- nodes
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

    def run_tools(state: MedicState) -> dict:
        ai = state["messages"][-1]
        assert isinstance(ai, AIMessage)
        budget = state["budget"]
        records: list[ToolRecord] = []
        replies: list[ToolMessage] = []
        # First occurrence wins, so a repeat always points at the original result.
        seen: dict[tuple[str, str], ToolRecord] = {}
        for r in state.get("tool_records") or []:
            seen.setdefault(_signature(r.tool, r.args), r)
        for call in ai.tool_calls:
            name, args, api_id = (
                call["name"],
                call.get("args") or {},
                call.get("id") or f"call_{budget.tool_calls}",
            )
            if name == SubmitEvidence.__name__:
                continue
            budget = budget.model_copy(update={"tool_calls": budget.tool_calls + 1})
            # The id the model cites. Gemini never shows the model the API's
            # own call ids, so every result is tagged with this one instead.
            ref = f"T{budget.tool_calls}"
            tool = by_name.get(name)
            error = False
            previous = seen.get(_signature(name, args))
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
            seen.setdefault(_signature(name, args), record)
            replies.append(ToolMessage(content=output, tool_call_id=api_id, name=name))
            say(
                f"  {name}({json.dumps(args, default=str)[:120]}) -> {len(output)} chars{' [error]' if error else ''}"
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
            "status": "done",
        }

    def escalate_report(state: MedicState) -> dict:
        reason = state["budget"].exceeded() or "the investigator produced no usable turn"
        say(f"escalate: {reason}")
        return {"status": "escalated", "escalation_reason": reason}

    # ------------------------------------------------------------ routing
    def after_investigator(
        state: MedicState,
    ) -> Literal["run_tools", "record_evidence", "escalate_report", "investigator_agent"]:
        # The last model turn, independent of how the message reducer stored it.
        turns = state.get("llm_turns") or []
        names = (
            [c["name"] for c in turns[-1].tool_calls]
            if turns and turns[-1].node == "investigator_agent"
            else []
        )
        if SubmitEvidence.__name__ in names:
            # The investigation is over and its evidence is already paid for;
            # ranking it costs one more call, so a budget overrun does not
            # discard it. Escalation is for an investigation that would go on.
            return "record_evidence"
        if state["budget"].exceeded() or not turns:
            return "escalate_report"
        if names:
            return "run_tools"
        return "investigator_agent"

    def after_tools(state: MedicState) -> Literal["investigator_agent", "escalate_report"]:
        return "escalate_report" if state["budget"].exceeded() else "investigator_agent"

    graph = StateGraph(MedicState)
    graph.add_node("triage_agent", triage_agent)
    graph.add_node("investigator_agent", investigator_agent)
    graph.add_node("run_tools", run_tools)
    graph.add_node("record_evidence", record_evidence)
    graph.add_node("hypothesis_agent", hypothesis_agent)
    graph.add_node("critic_node", critic_node)
    graph.add_node("escalate_report", escalate_report)

    graph.add_edge(START, "triage_agent")
    graph.add_edge("triage_agent", "investigator_agent")
    graph.add_conditional_edges("investigator_agent", after_investigator)
    graph.add_conditional_edges("run_tools", after_tools)
    graph.add_edge("record_evidence", "hypothesis_agent")
    graph.add_edge("hypothesis_agent", "critic_node")
    graph.add_edge("critic_node", END)
    graph.add_edge("escalate_report", END)
    return graph.compile()


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
    }
