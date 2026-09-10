"""Replay every recorded live run through the real graph with no API and no sandbox.

Each transcript under tests/fixtures/<scenario>/ came from `medic triage
--scenario N` against Gemini. Replaying it exercises routing, budget
accounting, evidence recording and the critic exactly as the live run did,
and pins the outcome the live run produced.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from medic.graph.build import build_graph, initial_state
from medic.graph.state import Budget
from medic.transcript import FIXTURES_DIR, ReplayChatModel, Transcript, replay_tools

FIXTURES = sorted(FIXTURES_DIR.glob("*/transcript.json"))


def replay(path: Path):
    transcript = Transcript.load(path)
    model = ReplayChatModel.from_turns(transcript.llm_turns)
    tools = replay_tools(transcript.tool_records)
    limits = transcript.budget.model_copy(
        update={"tool_calls": 0, "repeated_calls": 0, "llm_calls": 0, "tokens": 0}
    )
    graph = build_graph(model, tools, budget=limits, sleep=lambda _s: None, min_interval_s=0)
    start = initial_state(transcript.incident, Budget(**limits.model_dump()))
    if not any(r.tool == "incident" for r in transcript.tool_records):
        # Recorded before the incident summary became citable record T0:
        # replay under the conditions the model actually saw.
        start["tool_records"] = []
    final = graph.invoke(start, config={"recursion_limit": 200})
    return transcript, model, final


@pytest.mark.skipif(not FIXTURES, reason="no recorded transcripts yet")
@pytest.mark.parametrize("path", FIXTURES, ids=[p.parent.name for p in FIXTURES])
def test_replay_reproduces_the_live_run(path: Path):
    transcript, model, final = replay(path)

    assert model.turn == len(transcript.llm_turns), "every recorded model turn was consumed"
    assert final["status"] == transcript.status
    assert final["budget"].tool_calls == transcript.budget.tool_calls
    assert final["budget"].llm_calls == transcript.budget.llm_calls
    assert final["budget"].tokens == transcript.budget.tokens
    # Transcripts recorded before the incident summary became record T0 lack it.
    live = [r for r in final["tool_records"] if r.tool != "incident"]
    recorded = [r for r in transcript.tool_records if r.tool != "incident"]
    assert [r.id for r in live] == [r.id for r in recorded]
    assert [r.output for r in live] == [r.output for r in recorded]
    assert [h.model_dump() for h in final["hypotheses"]] == [
        h.model_dump() for h in transcript.hypotheses
    ]
    assert [h.model_dump() for h in final["hallucinations"]] == [
        h.model_dump() for h in transcript.hallucinations
    ]


@pytest.mark.skipif(not FIXTURES, reason="no recorded transcripts yet")
@pytest.mark.parametrize("path", FIXTURES, ids=[p.parent.name for p in FIXTURES])
def test_no_uncited_claim_survives(path: Path):
    _, _, final = replay(path)
    by_id = {r.id: r for r in final["tool_records"]}
    for e in final["evidence"]:
        assert e.tool_call_id in by_id, e
        assert e.tool == by_id[e.tool_call_id].tool
    kept = {e.id for e in final["evidence"]}
    for h in final["hypotheses"]:
        assert h.evidence_ids and set(h.evidence_ids) <= kept, h


@pytest.mark.skipif(not FIXTURES, reason="no recorded transcripts yet")
@pytest.mark.parametrize("path", FIXTURES, ids=[p.parent.name for p in FIXTURES])
def test_answer_key_never_queried(path: Path):
    transcript = Transcript.load(path)
    for r in transcript.tool_records:
        if r.tool == "query_duckdb":
            sql = str(r.args.get("sql", "")).lower()
            assert "true_effects" not in sql and "channel_ground_truth" not in sql, r.args
