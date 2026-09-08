"""The spike loop, driven by a scripted model. No API key, no network."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from medic.agents.spike import SpikeReport, run_spike
from medic.tools.dbt_artifacts import make_dbt_artifact_tools


class ScriptedChatModel(BaseChatModel):
    """Replays a fixed list of AI turns. bind_tools records the tool names."""

    script: list[AIMessage]
    turn: int = 0
    bound_tools: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: list[Any], **kwargs: Any):  # type: ignore[override]
        self.bound_tools = [
            getattr(t, "name", None) or getattr(t, "__name__", str(t)) for t in tools
        ]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = self.script[min(self.turn, len(self.script) - 1)]
        self.turn += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def test_loop_runs_tools_then_stops_on_report(manifest_path, run_results_path):
    tools = make_dbt_artifact_tools(manifest_path, run_results_path)
    model = ScriptedChatModel(
        script=[
            call("get_run_results", {}, "c1"),
            call(
                "get_lineage", {"node": "dim_channels", "direction": "upstream", "depth": 2}, "c2"
            ),
            call(
                "SpikeReport",
                {
                    "failing_model": "dim_channels",
                    "upstream_models": ["int_touchpoints_sessionized", "int_journeys"],
                    "error_summary": 'Binder Error: Referenced column "touch_cnt" not found',
                    "reasoning": "Only model that errored; the test was skipped because of it.",
                },
                "c3",
            ),
        ]
    )

    result = run_spike(model, tools, max_steps=5)

    assert model.bound_tools == ["get_run_results", "get_lineage", "SpikeReport"]
    assert isinstance(result.report, SpikeReport)
    assert result.report.failing_model == "dim_channels"
    assert result.steps == 3 and result.tool_calls == 2 and not result.exhausted

    tool_msgs = [m for m in result.messages if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_msgs] == ["c1", "c2"]
    assert "Binder Error" in tool_msgs[0].content
    assert "int_journeys" in tool_msgs[1].content
    assert [o["tool"] for o in result.tool_outputs] == ["get_run_results", "get_lineage"]


def test_unknown_tool_is_reported_back_to_the_model(manifest_path, run_results_path):
    tools = make_dbt_artifact_tools(manifest_path, run_results_path)
    model = ScriptedChatModel(
        script=[
            call("query_duckdb", {"sql": "select 1"}, "c1"),
            AIMessage(content="I cannot continue."),
        ]
    )
    result = run_spike(model, tools, max_steps=5)
    assert result.report is None
    assert result.final_text == "I cannot continue."
    assert "unknown tool 'query_duckdb'" in result.tool_outputs[0]["output"]


def test_step_budget_is_enforced(manifest_path, run_results_path):
    tools = make_dbt_artifact_tools(manifest_path, run_results_path)
    model = ScriptedChatModel(script=[call("get_run_results", {}, "loop")])
    result = run_spike(model, tools, max_steps=3)
    assert result.exhausted and result.report is None
    assert result.steps == 3 and result.tool_calls == 3
