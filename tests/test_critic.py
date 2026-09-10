"""The critic: verbatim excerpts survive, everything else is stripped."""

from __future__ import annotations

from medic.critic import check_evidence, check_hypotheses, excerpt_matches, normalize, run_critic
from medic.graph.state import Evidence, Hypothesis, ToolRecord

OUTPUT = (
    '{"failing":[{"name":"dim_channels","message":"Runtime Error in model dim_channels\\n'
    '  Binder Error: Values list \\"c\\" does not have a column named \\"touch_cnt\\""}]}'
)
RECORDS = [
    ToolRecord(id="c1", tool="get_run_results", args={}, output=OUTPUT),
    ToolRecord(
        id="c2", tool="read_pipeline_file", args={"path": "x"}, output="select\n    c.touch_cnt,\n"
    ),
]


def ev(id_: str, call: str, excerpt: str, tool: str = "?") -> Evidence:
    return Evidence(id=id_, claim=f"claim {id_}", tool=tool, tool_call_id=call, excerpt=excerpt)


def hyp(rank: int, ids: list[str]) -> Hypothesis:
    return Hypothesis(
        rank=rank,
        root_cause=f"cause {rank}",
        category="code_error",
        evidence_ids=ids,
        confidence=0.5,
        fix_kind="code_patch",
        recommended_action="fix it",
    )


def test_excerpt_match_kinds():
    assert excerpt_matches("c.touch_cnt,", RECORDS[1].output) == "exact"
    # the model unescaped the JSON quotes and collapsed the newline
    assert (
        excerpt_matches(
            'Binder Error: Values list "c" does not have a column named "touch_cnt"', OUTPUT
        )
        == "normalized"
    )
    assert excerpt_matches('Values  list "c"', OUTPUT) == "normalized"
    assert excerpt_matches("column named touch_count", OUTPUT) is None
    assert excerpt_matches("   ", OUTPUT) is None
    assert normalize('a \\"b\\"\\n  c') == 'a "b" c'


def test_check_evidence_strips_unknown_id_short_and_unmatched():
    evidence = [
        ev("E1", "c1", 'Binder Error: Values list "c" does not have a column named "touch_cnt"'),
        ev("E2", "c9", "anything at all here"),
        ev("E3", "c2", "cnt"),
        ev("E4", "c2", "select c.touch_count from c"),
        ev("E5", "c2", "c.touch_cnt,", tool="wrong_name"),
    ]
    result = check_evidence(evidence, RECORDS)
    assert [e.id for e in result.evidence] == ["E1", "E5"]
    assert result.evidence[1].tool == "read_pipeline_file"  # corrected from the record
    assert result.match_kinds == {"E1": "normalized", "E5": "exact"}
    reasons = {h.ref: h.reason for h in result.hallucinations}
    assert "unknown tool_call_id" in reasons["E2"]
    assert "too short" in reasons["E3"]
    assert "not a substring" in reasons["E4"]


def test_check_hypotheses_drops_uncited_and_reranks():
    surviving = [ev("E1", "c1", "x" * 10), ev("E5", "c2", "y" * 10)]
    hypotheses = [hyp(1, ["E2"]), hyp(2, ["E1", "E4"]), hyp(3, []), hyp(4, ["E5"])]
    result = check_hypotheses(hypotheses, surviving)
    assert [(h.rank, h.root_cause, h.evidence_ids) for h in result.hypotheses] == [
        (1, "cause 2", ["E1"]),
        (2, "cause 4", ["E5"]),
    ]
    reasons = [h.reason for h in result.hallucinations]
    assert any("none of its evidence survived: E2" in r for r in reasons)
    assert any("dropped citations that did not survive: E4" in r for r in reasons)
    assert any(r == "cites no evidence" for r in reasons)


def test_run_critic_end_to_end():
    evidence = [ev("E1", "c2", "c.touch_cnt,"), ev("E2", "c1", "made up text that is long")]
    result = run_critic(evidence, [hyp(1, ["E1", "E2"]), hyp(2, ["E2"])], RECORDS)
    assert [e.id for e in result.evidence] == ["E1"]
    assert [(h.rank, h.evidence_ids) for h in result.hypotheses] == [(1, ["E1"])]
    assert result.stripped_evidence == 1 and result.stripped_hypotheses == 2
