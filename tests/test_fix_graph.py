"""Fix phase routing: fixer, apply, validator, retry, interrupt, resume, PR, escalate.

A scripted model, fake tools and fake sandbox operations; no API, no dbt.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from medic.graph.build import build_graph, investigated_state, review_payload
from medic.graph.checkpoints import list_threads, open_saver, thread_config, thread_state
from medic.graph.state import (
    ApplyResult,
    Budget,
    EditOutcome,
    Evidence,
    FailingNode,
    Hypothesis,
    Incident,
    PullRequest,
    Validation,
)
from medic.transcript import ReplayChatModel

MODEL_SQL = (
    "# dbt_marketing/models/marts/dim_channels.sql (3 lines)\nselect\n    c.touch_cnt,\nfrom c\n"
)
GOOD_EDIT = {
    "path": "dbt_marketing/models/marts/dim_channels.sql",
    "find": "c.touch_cnt,",
    "replace": "c.touch_count,",
}


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
def seeded(incident):
    hyp = Hypothesis(
        rank=1,
        root_cause="dim_channels.sql references c.touch_cnt, which does not exist",
        category="code_error",
        evidence_ids=["E1"],
        confidence=0.9,
        fix_kind="code_patch",
        recommended_action="revert to c.touch_count",
    )
    ev = Evidence(
        id="E1", claim="typo", tool="read_pipeline_file", tool_call_id="T3", excerpt="c.touch_cnt,"
    )
    seed = Budget(tool_calls=8, llm_calls=9, tokens=55_000)
    return investigated_state(incident, triage=None, evidence=[ev], hypotheses=[hyp], budget=seed)


@pytest.fixture
def tools():
    reads: list[str] = []

    @tool
    def read_pipeline_file(path: str) -> str:
        """read"""
        reads.append(path)
        return MODEL_SQL

    @tool
    def list_pipeline_files(subdir: str = "dbt_marketing") -> str:
        """list"""
        return '["dbt_marketing/models/marts/dim_channels.sql"]'

    return [read_pipeline_file, list_pipeline_files], reads


class FakeOps:
    """Records what the graph asked for; answers from queues."""

    def __init__(self, applies=None, validations=None, preread=None):
        self.applies = list(applies or [])
        self.validations = list(validations or [])
        self.preread = list(preread or ["dbt_marketing/models/marts/dim_channels.sql"])
        self.calls: list[tuple[str, object]] = []

    def files_to_read(self, incident, hypotheses):
        return self.preread

    def apply(self, edits, kind):
        self.calls.append(("apply", [e.model_dump() for e in edits], kind))
        return self.applies.pop(0)

    def validate(self, edited_paths):
        self.calls.append(("validate", list(edited_paths)))
        return self.validations.pop(0)

    def diff(self):
        return ""

    def reset(self):
        self.calls.append(("reset",))

    def open_pr(self, top, evidence, attempt, validation, reviewer_note):
        self.calls.append(("open_pr", reviewer_note))
        return PullRequest(
            mode="dry-run", title="medic: fix dim_channels", branch="medic/bad_commit", path="pr.md"
        )


def call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id}


def ai(*calls: dict, tokens: int = 100) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=list(calls),
        usage_metadata={"input_tokens": tokens - 10, "output_tokens": 10, "total_tokens": tokens},
    )


def propose(kind="code_patch", edits=None, explanation="revert the typo", action="", call_id="p1"):
    return ai(
        call(
            "ProposeFix",
            {
                "kind": kind,
                "explanation": explanation,
                "edits": [GOOD_EDIT] if edits is None else edits,
                "recommended_action": action,
            },
            call_id,
        )
    )


APPLIED = ApplyResult(
    ok=True,
    outcomes=[EditOutcome(path="dbt_marketing/models/marts/dim_channels.sql", ok=True)],
    diff="--- a/x\n+++ b/x\n@@\n-    c.touch_cnt,\n+    c.touch_count,\n",
    edited_paths=["dbt_marketing/models/marts/dim_channels.sql"],
)
REJECTED = ApplyResult(
    ok=False,
    outcomes=[
        EditOutcome(
            path="tests/x.sql", ok=False, error="tests/x.sql defines what the pipeline checks"
        )
    ],
)
PASSED = Validation(
    passed=True,
    dbt_args=["build", "--select", "dim_channels+"],
    status_counts={"success": 1, "pass": 3},
)
FAILED = Validation(
    passed=False,
    dbt_args=["build", "--select", "dim_channels+"],
    returncode=1,
    status_counts={"error": 1},
    failing=[
        FailingNode(
            unique_id="model.pkg.dim_channels",
            name="dim_channels",
            resource_type="model",
            status="error",
            message="still broken",
        )
    ],
    stdout_tail="1 error",
)


def make(model, tools, ops, saver=None, budget=None):
    return build_graph(
        model,
        tools,
        ops=ops,
        budget=budget,
        checkpointer=saver or InMemorySaver(),
        sleep=lambda _s: None,
        min_interval_s=0,
    )


def test_patch_validates_pauses_and_opens_pr_on_approval(seeded, tools):
    tool_list, reads = tools
    ops = FakeOps(applies=[APPLIED], validations=[PASSED])
    model = ReplayChatModel(script=[propose()])
    graph = make(model, tool_list, ops)
    cfg = thread_config("t1")

    out = graph.invoke(seeded, cfg)

    # the fixer was shown the failing model's file, read through the tool, as a record
    assert reads == ["dbt_marketing/models/marts/dim_channels.sql"]
    pre = [r for r in out["tool_records"] if r.origin == "fixer_preread"]
    assert [r.id for r in pre] == ["T9"] and pre[0].output.startswith("[T9 read_pipeline_file]")
    first = out["fix_messages"][0]
    assert isinstance(first, HumanMessage) and "[T9 read_pipeline_file]" in first.content
    assert "#1 [code_error, code_patch, 0.90]" in first.content
    # applied, validated, paused
    assert out["status"] == "awaiting_review"
    assert ops.calls[0][0] == "apply" and ops.calls[0][2] == "code_patch"
    assert ops.calls[1] == ("validate", ["dbt_marketing/models/marts/dim_channels.sql"])
    assert out["fix_attempts"][0].outcome == "passed"
    assert out["budget"].fix_iterations == 1 and out["budget"].llm_calls == 10
    assert out["budget"].tool_calls == 9
    payload = out["__interrupt__"][0].value
    assert payload == review_payload(out)
    assert payload["kind"] == "code_patch" and "c.touch_count" in payload["diff"]
    assert payload["validation"].startswith("dbt build --select dim_channels+ exit 0")

    final = graph.invoke(Command(resume={"status": "approved", "note": "looks right"}), cfg)
    assert final["status"] == "done"
    assert final["approval"].status == "approved" and final["approval"].decided_at
    assert final["pull_request"].path == "pr.md"
    assert ops.calls[-1] == ("open_pr", "looks right")
    assert [t.node for t in final["llm_turns"]] == ["fixer_agent"]


def test_rejection_resets_the_sandbox_and_escalates(seeded, tools):
    tool_list, _ = tools
    ops = FakeOps(applies=[APPLIED], validations=[PASSED])
    graph = make(ReplayChatModel(script=[propose()]), tool_list, ops)
    cfg = thread_config("t2")
    graph.invoke(seeded, cfg)
    final = graph.invoke(Command(resume={"status": "rejected", "note": "wrong column"}), cfg)
    assert final["status"] == "escalated"
    assert final["escalation_reason"] == "rejected by reviewer: wrong column"
    assert ("reset",) in ops.calls and not any(c[0] == "open_pr" for c in ops.calls)


def test_failed_validation_feeds_back_then_passes(seeded, tools):
    tool_list, reads = tools
    ops = FakeOps(applies=[APPLIED, APPLIED], validations=[FAILED, PASSED])
    second = {**GOOD_EDIT, "find": "from c", "replace": "from by_channel c"}
    model = ReplayChatModel(script=[propose(), propose(edits=[second], call_id="p2")])
    graph = make(model, tool_list, ops)
    out = graph.invoke(seeded, thread_config("t3"))
    assert out["status"] == "awaiting_review"
    assert [a.outcome for a in out["fix_attempts"]] == ["failed validation", "passed"]
    assert out["budget"].fix_iterations == 2
    # the feedback quoted the dbt failure and re-read the edited file for the fixer
    feedback = [m for m in out["fix_messages"] if isinstance(m, HumanMessage)][1]
    assert "Attempt 1 did not pass" in feedback.content and "still broken" in feedback.content
    assert "Current text of the edited files" in feedback.content
    assert reads == ["dbt_marketing/models/marts/dim_channels.sql"] * 2
    assert [r.origin for r in out["tool_records"][1:]] == ["fixer_preread", "fixer_feedback"]
    assert ops.calls[2][1][0]["find"] == "from c"


def test_three_failed_attempts_escalate(seeded, tools):
    tool_list, _ = tools
    ops = FakeOps(applies=[APPLIED] * 3, validations=[FAILED] * 3)
    model = ReplayChatModel(script=[propose(call_id=f"p{i}") for i in range(3)])
    graph = make(model, tool_list, ops)
    out = graph.invoke(seeded, thread_config("t4"))
    assert out["status"] == "escalated"
    assert out["escalation_reason"] == "fix did not pass after 3 attempt(s): failed validation"
    assert "__interrupt__" not in out
    assert model.turn == 3


def test_rejected_edit_counts_as_an_attempt_and_is_fed_back(seeded, tools):
    tool_list, _ = tools
    ops = FakeOps(applies=[REJECTED, APPLIED], validations=[PASSED])
    bad = {"path": "tests/x.sql", "find": "1", "replace": "2"}
    model = ReplayChatModel(script=[propose(edits=[bad]), propose(call_id="p2")])
    graph = make(model, tool_list, ops)
    out = graph.invoke(seeded, thread_config("t5"))
    assert out["status"] == "awaiting_review"
    assert [a.outcome for a in out["fix_attempts"]] == ["edits did not apply", "passed"]
    feedback = [m for m in out["fix_messages"] if isinstance(m, HumanMessage)][1]
    assert "could not be applied" in feedback.content
    assert "defines what the pipeline checks" in feedback.content
    assert not any(c[0] == "validate" and c[1] == [] for c in ops.calls)


def test_decline_goes_straight_to_review_and_approval_ends_without_pr(seeded, tools):
    tool_list, _ = tools
    ops = FakeOps()
    model = ReplayChatModel(
        script=[propose(kind="upstream_data_issue", edits=[], action="ask the source owner")]
    )
    graph = make(model, tool_list, ops)
    cfg = thread_config("t6")
    out = graph.invoke(seeded, cfg)
    assert out["status"] == "awaiting_review"
    assert out["fix_attempts"][0].outcome == "declined (upstream_data_issue)"
    assert out["__interrupt__"][0].value["recommended_action"] == "ask the source owner"
    assert not any(c[0] in ("apply", "validate") for c in ops.calls)
    final = graph.invoke(Command(resume={"status": "approved"}), cfg)
    assert final["status"] == "done" and final["pull_request"] is None


def test_decline_after_an_applied_attempt_resets_the_sandbox(seeded, tools):
    tool_list, _ = tools
    ops = FakeOps(applies=[APPLIED], validations=[FAILED])
    model = ReplayChatModel(
        script=[
            propose(),
            propose(kind="needs_human", edits=[], action="check with data owner", call_id="p2"),
        ]
    )
    graph = make(model, tool_list, ops)
    out = graph.invoke(seeded, thread_config("t7"))
    assert out["status"] == "awaiting_review"
    assert [a.outcome for a in out["fix_attempts"]] == [
        "failed validation",
        "declined (needs_human)",
    ]
    assert ("reset",) in ops.calls


def test_code_patch_without_edits_becomes_needs_human(seeded, tools):
    tool_list, _ = tools
    graph = make(ReplayChatModel(script=[propose(edits=[])]), tool_list, FakeOps())
    out = graph.invoke(seeded, thread_config("t8"))
    assert out["proposed_fix"].kind == "needs_human"
    assert out["proposed_fix"].explanation.startswith("code_patch proposed with no edits")
    assert out["status"] == "awaiting_review"


def test_fixer_may_read_files_first_then_is_forced_to_submit(seeded, tools):
    tool_list, reads = tools
    ops = FakeOps(applies=[APPLIED], validations=[PASSED])
    read = call(
        "read_pipeline_file", {"path": "dbt_marketing/models/marts/_marts__models.yml"}, "r1"
    )
    model = ReplayChatModel(
        script=[ai(read), ai({**read, "id": "r2"}), ai({**read, "id": "r3"}), propose()]
    )
    graph = make(model, tool_list, ops, budget=Budget(max_fixer_turns=3))
    out = graph.invoke(seeded, thread_config("t9"))
    assert out["status"] == "awaiting_review"
    # three reads through run_fix_tools (no dedupe), then a forced proposal
    assert reads.count("dbt_marketing/models/marts/_marts__models.yml") == 3
    assert model.tool_choices[:3] == [None, None, None] and model.tool_choices[3] == "ProposeFix"
    assert model.bound[0] == ["read_pipeline_file", "list_pipeline_files", "ProposeFix"]
    nudges = [m for m in out["fix_messages"] if isinstance(m, HumanMessage)]
    assert any("Submit your proposal now" in m.content for m in nudges)
    assert out["budget"].tool_calls == 8 + 1 + 3


def test_llm_budget_exceeded_in_fix_phase_escalates(seeded, tools):
    tool_list, _ = tools
    model = ReplayChatModel(script=[AIMessage(content="thinking") for _ in range(5)])
    seeded["budget"] = Budget(llm_calls=9, max_llm_calls=10)
    graph = make(model, tool_list, FakeOps(), budget=seeded["budget"])
    out = graph.invoke(seeded, thread_config("t10"))
    assert out["status"] == "escalated" and "LLM calls" in out["escalation_reason"]


def test_resume_from_a_fresh_process_via_sqlite(seeded, tools, tmp_path: Path):
    tool_list, _ = tools
    db = tmp_path / "cp.sqlite"
    ops = FakeOps(applies=[APPLIED], validations=[PASSED])
    cfg = thread_config("bad_commit-20260912-000000")
    with open_saver(db) as saver:
        graph = make(ReplayChatModel(script=[propose()]), tool_list, ops, saver=saver)
        out = graph.invoke(seeded, cfg)
        assert out["status"] == "awaiting_review"
        info = thread_state(saver, "bad_commit-20260912-000000")
        assert info.waiting and info.run_id == "chaos-bad_commit" and info.scenario == 5
        assert [t.thread_id for t in list_threads(saver)] == ["bad_commit-20260912-000000"]
    # a new saver, a new graph, a model with nothing to say: only the review resumes
    with open_saver(db) as saver:
        ops2 = FakeOps()
        graph = make(ReplayChatModel(script=[]), tool_list, ops2, saver=saver)
        final = graph.invoke(Command(resume={"status": "approved", "note": "ship it"}), cfg)
        assert final["status"] == "done" and final["pull_request"].title.startswith("medic:")
        assert ops2.calls == [("open_pr", "ship it")]
        assert thread_state(saver, "bad_commit-20260912-000000").status == "done"
        assert thread_state(saver, "nope") is None


def test_investigation_only_graph_ends_after_critic(seeded, tools):
    tool_list, _ = tools
    graph = build_graph(
        ReplayChatModel(script=[]), tool_list, sleep=lambda _s: None, min_interval_s=0
    )
    assert "fixer_agent" not in graph.get_graph().nodes
