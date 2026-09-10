"""Graph routing with a scripted model and fake tools. No API, no sandbox."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from medic.graph.build import MAX_REPEATED_CALLS, build_graph, initial_state
from medic.graph.state import Budget, FailingNode, Incident
from medic.transcript import ReplayChatModel

RUN_RESULTS = json.dumps(
    {
        "failing": [
            {
                "name": "dim_channels",
                "status": "error",
                "message": 'Binder Error: Values list "c" does not have a column named "touch_cnt"',
            }
        ]
    }
)
MODEL_SQL = (
    "# dbt_marketing/models/marts/dim_channels.sql (3 lines)\nselect\n    c.touch_cnt,\nfrom c\n"
)


@pytest.fixture
def incident() -> Incident:
    return Incident(
        source="chaos",
        run_id="chaos-bad_commit",
        scenario=5,
        scenario_key="bad_commit",
        dbt_args=["build"],
        artifact="x/run_results.json",
        manifest="x/manifest.json",
        sandbox_dir="x",
        duckdb="x/marketing.duckdb",
        repo="x/repo",
        failing_nodes=[
            FailingNode(
                unique_id="model.pkg.dim_channels",
                name="dim_channels",
                resource_type="model",
                status="error",
                message="Binder Error: touch_cnt",
            )
        ],
        status_counts={"error": 1, "skipped": 21, "success": 12, "pass": 95},
    )


@pytest.fixture
def fake_tools():
    calls: list[tuple[str, dict]] = []

    @tool
    def get_run_results() -> str:
        """run results"""
        calls.append(("get_run_results", {}))
        return RUN_RESULTS

    @tool
    def read_pipeline_file(path: str) -> str:
        """read a file"""
        calls.append(("read_pipeline_file", {"path": path}))
        return MODEL_SQL

    @tool
    def query_duckdb(sql: str) -> str:
        """query"""
        calls.append(("query_duckdb", {"sql": sql}))
        raise RuntimeError("database is on fire")

    return [get_run_results, read_pipeline_file, query_duckdb], calls


def call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id}


def ai(*calls: dict, tokens: int = 100) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=list(calls),
        usage_metadata={"input_tokens": tokens - 10, "output_tokens": 10, "total_tokens": tokens},
    )


TRIAGE = call(
    "TriageVerdict",
    {"category": "code_error", "confidence": 0.8, "rationale": "Binder Error names touch_cnt"},
    "t1",
)


def hypothesis_turn(ids: list[str]) -> AIMessage:
    return ai(
        call(
            "HypothesisSet",
            {
                "hypotheses": [
                    {
                        "root_cause": "dim_channels.sql references c.touch_cnt, which does not exist",
                        "category": "code_error",
                        "evidence_ids": ids,
                        "confidence": 0.9,
                        "fix_kind": "code_patch",
                        "recommended_action": "revert to c.touch_count",
                    }
                ]
            },
            "h1",
        )
    )


def run(model: ReplayChatModel, tools, incident: Incident, budget: Budget | None = None):
    graph = build_graph(model, tools, budget=budget, sleep=lambda _s: None, min_interval_s=0)
    return graph.invoke(initial_state(incident, budget), config={"recursion_limit": 100})


def test_happy_path_records_evidence_and_survives_critic(incident, fake_tools):
    tools, calls = fake_tools
    model = ReplayChatModel(
        script=[
            ai(TRIAGE),
            ai(
                call("get_run_results", {}, "c1"),
                call("read_pipeline_file", {"path": "m.sql"}, "c2"),
            ),
            ai(call("query_duckdb", {"sql": "select 1"}, "c3")),
            ai(
                call(
                    "SubmitEvidence",
                    {
                        "summary": "typo",
                        "evidence": [
                            {
                                "claim": "the model errored",
                                "tool_call_id": "c1",
                                "excerpt": "Binder Error: Values list",
                            },
                            {
                                "claim": "the sql has touch_cnt",
                                "tool_call_id": "c2",
                                "excerpt": "c.touch_cnt,",
                            },
                            {"claim": "made up", "tool_call_id": "c2", "excerpt": "c.touch_count,"},
                            {
                                "claim": "query failed",
                                "tool_call_id": "c3",
                                "excerpt": "database is on fire",
                            },
                        ],
                    },
                    "s1",
                )
            ),
            hypothesis_turn(["E1", "E2", "E3"]),
        ]
    )
    final = run(model, tools, incident)

    assert final["status"] == "done"
    assert final["triage"].category == "code_error"
    assert [c[0] for c in calls] == ["get_run_results", "read_pipeline_file", "query_duckdb"]
    assert [r.id for r in final["tool_records"]] == ["T0", "T1", "T2", "T3"]
    assert [r.api_call_id for r in final["tool_records"][1:]] == ["c1", "c2", "c3"]
    assert final["tool_records"][0].tool == "incident"
    assert final["tool_records"][1].output.startswith("[T1 get_run_results]\n")
    assert (
        final["tool_records"][3].error and "database is on fire" in final["tool_records"][3].output
    )
    assert [e.id for e in final["evidence"]] == ["E1", "E2", "E4"]  # E3 was not verbatim
    assert final["evidence"][1].tool == "read_pipeline_file"
    assert [h.evidence_ids for h in final["hypotheses"]] == [["E1", "E2"]]
    assert [h.ref for h in final["hallucinations"]] == ["E3", "rank 1"]
    assert final["budget"].tool_calls == 3 and final["budget"].llm_calls == 5
    assert final["budget"].tokens == 500
    assert [t.node for t in final["llm_turns"]] == [
        "triage_agent",
        "investigator_agent",
        "investigator_agent",
        "investigator_agent",
        "hypothesis_agent",
    ]
    # the investigator saw the incident, then tool replies, in order
    kinds = [type(m).__name__ for m in final["messages"]]
    assert kinds[:5] == ["HumanMessage", "AIMessage", "ToolMessage", "ToolMessage", "AIMessage"]
    assert model.tool_choices[0] == "TriageVerdict" and model.tool_choices[-1] == "HypothesisSet"
    assert model.tool_choices[1] is None


def test_repeated_calls_are_not_rerun_and_force_submission(incident, fake_tools):
    tools, calls = fake_tools
    same = call("read_pipeline_file", {"path": "m.sql"}, "c1")
    model = ReplayChatModel(
        script=[
            ai(TRIAGE),
            ai(same),
            ai({**same, "id": "c2"}),
            ai({**same, "id": "c3"}),
            ai(
                call(
                    "SubmitEvidence",
                    {
                        "summary": "s",
                        "evidence": [
                            {"claim": "sql", "tool_call_id": "c3", "excerpt": "c.touch_cnt,"}
                        ],
                    },
                    "s1",
                )
            ),
            hypothesis_turn(["E1"]),
        ]
    )
    final = run(model, tools, incident)
    assert final["status"] == "done"
    assert len(calls) == 1  # the tool itself ran once
    assert final["budget"].tool_calls == 3 and final["budget"].repeated_calls == MAX_REPEATED_CALLS
    assert "REPEATED CALL: identical to T1" in final["tool_records"][3].output
    # the third repeat crossed the threshold, so the next investigator turn was forced
    assert model.tool_choices[4] == "SubmitEvidence"
    nudges = [m for m in final["messages"] if isinstance(m, HumanMessage)]
    assert any("repeating tool calls" in m.content for m in nudges)
    # an excerpt cited against the repeat verifies against the original result
    assert [e.id for e in final["evidence"]] == ["E1"]
    assert final["evidence"][0].tool_call_id == "T1"  # resolved from the repeat T3


def test_model_that_stops_talking_is_told_to_submit(incident, fake_tools):
    tools, _ = fake_tools
    model = ReplayChatModel(
        script=[
            ai(TRIAGE),
            ai(call("get_run_results", {}, "c1")),
            AIMessage(content="I think it is a typo."),
            ai(
                call(
                    "SubmitEvidence",
                    {
                        "summary": "s",
                        "evidence": [
                            {
                                "claim": "err",
                                "tool_call_id": "c1",
                                "excerpt": "Binder Error: Values list",
                            }
                        ],
                    },
                    "s1",
                )
            ),
            hypothesis_turn(["E1"]),
        ]
    )
    final = run(model, tools, incident)
    assert final["status"] == "done" and len(final["hypotheses"]) == 1
    assert model.tool_choices[3] == "SubmitEvidence"


def test_tool_budget_exceeded_escalates(incident, fake_tools):
    tools, calls = fake_tools
    script = [ai(TRIAGE)] + [
        ai(call("query_duckdb", {"sql": f"select {i}"}, f"c{i}")) for i in range(10)
    ]
    model = ReplayChatModel(script=script)
    final = run(model, tools, incident, Budget(max_tool_calls=3))
    assert final["status"] == "escalated"
    assert final["escalation_reason"].startswith("tool calls 4 > 3")
    assert final["hypotheses"] == [] and len(calls) == 4


def test_llm_budget_exceeded_escalates(incident, fake_tools):
    tools, _ = fake_tools
    model = ReplayChatModel(script=[ai(TRIAGE)] + [AIMessage(content="hmm") for _ in range(10)])
    final = run(model, tools, incident, Budget(max_llm_calls=3))
    assert final["status"] == "escalated" and "LLM calls" in final["escalation_reason"]


def test_unknown_tool_is_reported_and_charged(incident, fake_tools):
    tools, _ = fake_tools
    model = ReplayChatModel(
        script=[
            ai(TRIAGE),
            ai(call("get_dagster_run", {"run_id": "x"}, "c1")),
            ai(call("SubmitEvidence", {"summary": "s", "evidence": []}, "s1")),
            hypothesis_turn([]),
        ]
    )
    final = run(model, tools, incident)
    assert "unknown tool 'get_dagster_run'" in final["tool_records"][1].output
    assert final["budget"].tool_calls == 1
    assert final["status"] == "done" and final["hypotheses"] == []
    assert final["hallucinations"][0].reason == "cites no evidence"
    tool_msgs = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs[0].tool_call_id == "c1"


def test_malformed_evidence_items_are_dropped_not_the_submission(incident, fake_tools):
    tools, _ = fake_tools
    model = ReplayChatModel(
        script=[
            ai(TRIAGE),
            ai(call("read_pipeline_file", {"path": "m.sql"}, "c1")),
            ai(
                call(
                    "SubmitEvidence",
                    {
                        "summary": "s",
                        "evidence": [
                            {"claim": "good", "tool_call_id": "T1", "excerpt": "c.touch_cnt,"},
                            {
                                "claim": "rows instead of excerpt",
                                "tool_call_id": "T1",
                                "rows": [[1, 2]],
                            },
                            "not even a dict",
                        ],
                    },
                    "s1",
                )
            ),
            hypothesis_turn(["E1"]),
        ]
    )
    final = run(model, tools, incident)
    assert [e.id for e in final["evidence"]] == ["E1"]
    assert [h.evidence_ids for h in final["hypotheses"]] == [["E1"]]
    reasons = [h.reason for h in final["hallucinations"]]
    assert reasons == ["malformed evidence item (excerpt)", "malformed evidence item (invalid)"]


def test_incident_summary_is_citable_as_t0(incident, fake_tools):
    tools, _ = fake_tools
    model = ReplayChatModel(
        script=[
            ai(TRIAGE),
            ai(
                call(
                    "SubmitEvidence",
                    {
                        "summary": "s",
                        "evidence": [
                            {
                                "claim": "from the summary",
                                "tool_call_id": "T0",
                                "excerpt": '"name": "dim_channels"',
                            },
                            {
                                "claim": "not in the summary",
                                "tool_call_id": "T0",
                                "excerpt": "touch_count is fine",
                            },
                        ],
                    },
                    "s1",
                )
            ),
            hypothesis_turn(["E1", "E2"]),
        ]
    )
    final = run(model, tools, incident)
    kinds = [type(m).__name__ for m in final["messages"]]
    assert kinds[0] == "HumanMessage"
    assert final["messages"][0].content.startswith("[T0 incident]\n")
    assert [e.id for e in final["evidence"]] == ["E1"]
    assert final["evidence"][0].tool == "incident"
    assert final["budget"].tool_calls == 0
