"""Phase 0 spike: one tool-calling loop that names the failing model and its upstreams.

This is deliberately not the graph. It proves the tools, the provider, and the
tracing work end to end before any routing exists. The loop is model-agnostic
so tests can drive it with a scripted fake and no API key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """You are PipelineMedic, an on-call data engineer investigating a failed dbt run.

Work only from tool results. Never guess a model name or an upstream that a tool did not return.

Procedure:
1. Call get_run_results to see which nodes failed and the error messages.
2. Pick the failing model closest to the source of the problem. A test that
   failed because its model errored is a symptom, not the cause.
3. Call get_lineage on that model with direction "upstream" to learn what it reads from.
4. Call SpikeReport exactly once with your findings. Quote the error message
   as the tool returned it."""

USER_PROMPT = "A dbt build just failed. Investigate and submit your report."


class SpikeReport(BaseModel):
    """Final answer. Submit this once; the investigation ends when you do."""

    failing_model: str = Field(description="dbt name of the model that broke, e.g. dim_channels")
    upstream_models: list[str] = Field(
        description="Names of models and sources it reads from, as returned by get_lineage"
    )
    error_summary: str = Field(description="The error message, quoted from get_run_results")
    reasoning: str = Field(description="One or two sentences on why this is the root model")


@dataclass
class SpikeResult:
    report: SpikeReport | None
    messages: list[BaseMessage]
    steps: int
    tool_calls: int
    final_text: str | None = None
    exhausted: bool = False
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)


def run_spike(
    model: BaseChatModel,
    tools: list[BaseTool],
    *,
    max_steps: int = 8,
    callbacks: list[Any] | None = None,
) -> SpikeResult:
    """Drive the model until it submits a SpikeReport or runs out of steps."""
    by_name = {t.name: t for t in tools}
    llm = model.bind_tools([*tools, SpikeReport])
    config = {"callbacks": callbacks or []}

    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT), HumanMessage(USER_PROMPT)]
    tool_calls = 0
    tool_outputs: list[dict[str, Any]] = []

    for step in range(1, max_steps + 1):
        ai: AIMessage = llm.invoke(messages, config=config)  # type: ignore[assignment]
        messages.append(ai)

        if not ai.tool_calls:
            text = ai.content if isinstance(ai.content, str) else str(ai.content)
            return SpikeResult(
                None, messages, step, tool_calls, final_text=text, tool_outputs=tool_outputs
            )

        for call in ai.tool_calls:
            name, args, call_id = call["name"], call.get("args") or {}, call.get("id")
            if name == SpikeReport.__name__:
                report = SpikeReport.model_validate(args)
                return SpikeResult(report, messages, step, tool_calls, tool_outputs=tool_outputs)

            tool_calls += 1
            tool = by_name.get(name)
            if tool is None:
                output = f"error: unknown tool {name!r}; available: {sorted(by_name)}"
            else:
                try:
                    output = tool.invoke(args, config=config)
                except Exception as exc:  # noqa: BLE001 - surface to the model, do not crash
                    output = f"error: {type(exc).__name__}: {exc}"
            output = output if isinstance(output, str) else str(output)
            tool_outputs.append({"id": call_id, "tool": name, "args": args, "output": output})
            messages.append(ToolMessage(content=output, tool_call_id=call_id or name, name=name))

    return SpikeResult(
        None, messages, max_steps, tool_calls, exhausted=True, tool_outputs=tool_outputs
    )
