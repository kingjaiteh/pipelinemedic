"""The edit guard: exact replacement, atomic apply, and what it refuses."""

from __future__ import annotations

from pathlib import Path

import pytest

from medic.graph.state import EditSpec
from medic.tools.sandbox_edit import EditRejected, apply_edits, collect_tests, resolve_editable
from medic.tools.sandbox_edit import tests_weakened as weakened

MODEL = "select\n    lower(trim(channel)) as channel,\n    cast(cost as double) as touch_cost_usd\nfrom source\n"
SCHEMA = """version: 2
models:
  - name: stg_touchpoints
    columns:
      - name: touchpoint_id
        tests:
          - unique
          - not_null
      - name: touch_cost_usd
        tests:
          - not_null
"""
SOURCES = """version: 2
sources:
  - name: raw
    tables:
      - name: touchpoints
        freshness:
          warn_after: {count: 2, period: day}
          error_after: {count: 7, period: day}
        columns:
          - name: channel
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    dbt = tmp_path / "dbt_marketing"
    (dbt / "models" / "staging").mkdir(parents=True)
    (dbt / "tests").mkdir()
    (dbt / "macros").mkdir()
    (dbt / "target").mkdir()
    (dbt / "models" / "staging" / "stg_touchpoints.sql").write_text(MODEL, encoding="utf-8")
    (dbt / "models" / "staging" / "_staging__models.yml").write_text(SCHEMA, encoding="utf-8")
    (dbt / "models" / "staging" / "_sources.yml").write_text(SOURCES, encoding="utf-8")
    (dbt / "tests" / "assert_volume.sql").write_text("select 1 where false\n", encoding="utf-8")
    (dbt / "macros" / "helpers.sql").write_text("{% macro x() %}{% endmacro %}\n", encoding="utf-8")
    (dbt / "dbt_project.yml").write_text(
        "name: marketing\nvars:\n  data_as_of: '2026-04-14'\n", encoding="utf-8"
    )
    (dbt / "profiles.yml").write_text("marketing:\n  target: dev\n", encoding="utf-8")
    (dbt / "target" / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "README.md").write_text("outside the project\n", encoding="utf-8")
    return tmp_path


def edit(path: str, find: str, replace: str) -> EditSpec:
    return EditSpec(path=path, find=find, replace=replace)


def test_exact_replacement_writes_and_diffs(repo: Path):
    result = apply_edits(
        repo,
        [
            edit(
                "dbt_marketing/models/staging/stg_touchpoints.sql",
                "trim(channel)",
                "trim(marketing_channel)",
            )
        ],
        "code_patch",
    )
    assert result.ok and result.edited_paths == ["dbt_marketing/models/staging/stg_touchpoints.sql"]
    assert "-    lower(trim(channel)) as channel," in result.diff
    assert "+    lower(trim(marketing_channel)) as channel," in result.diff
    text = (repo / "dbt_marketing/models/staging/stg_touchpoints.sql").read_text(encoding="utf-8")
    assert "marketing_channel" in text


def test_project_relative_path_is_accepted(repo: Path):
    result = apply_edits(
        repo, [edit("models/staging/stg_touchpoints.sql", "trim(channel)", "trim(c)")], "code_patch"
    )
    assert result.ok and result.edited_paths == ["dbt_marketing/models/staging/stg_touchpoints.sql"]


def test_find_must_match_exactly_once(repo: Path):
    missing = apply_edits(
        repo, [edit("models/staging/stg_touchpoints.sql", "trim( channel )", "x")], "code_patch"
    )
    assert not missing.ok and "not found" in missing.outcomes[0].error
    ambiguous = apply_edits(
        repo, [edit("models/staging/stg_touchpoints.sql", "channel", "x")], "code_patch"
    )
    assert not ambiguous.ok and "occurs 2 times" in ambiguous.outcomes[0].error
    # nothing was written by either
    text = (repo / "dbt_marketing/models/staging/stg_touchpoints.sql").read_text(encoding="utf-8")
    assert text == MODEL


def test_apply_is_all_or_nothing(repo: Path):
    result = apply_edits(
        repo,
        [
            edit("models/staging/stg_touchpoints.sql", "trim(channel)", "trim(marketing_channel)"),
            edit("models/staging/stg_touchpoints.sql", "does not exist", "x"),
        ],
        "code_patch",
    )
    assert not result.ok
    assert [o.ok for o in result.outcomes] == [True, False]
    text = (repo / "dbt_marketing/models/staging/stg_touchpoints.sql").read_text(encoding="utf-8")
    assert text == MODEL


def test_two_edits_to_one_file_chain_and_diff_against_disk(repo: Path):
    result = apply_edits(
        repo,
        [
            edit("models/staging/stg_touchpoints.sql", "trim(channel)", "trim(marketing_channel)"),
            edit(
                "models/staging/stg_touchpoints.sql",
                "cast(cost as double)",
                "try_cast(cost as double)",
            ),
        ],
        "code_patch",
    )
    assert result.ok and result.diff.count("@@") >= 1
    assert "-    lower(trim(channel)) as channel," in result.diff
    assert "+    try_cast(cost as double) as touch_cost_usd" in result.diff


@pytest.mark.parametrize(
    "path, reason",
    [
        ("tests/assert_volume.sql", "defines what the pipeline checks"),
        ("macros/helpers.sql", "defines what the pipeline checks"),
        ("dbt_project.yml", "defines what the pipeline checks"),
        ("profiles.yml", "never edited"),
        ("target/manifest.json", "generated output"),
        ("../README.md", "outside"),
        ("README.md", "does not exist"),
    ],
)
def test_protected_paths_are_refused_for_code_patches(repo: Path, path: str, reason: str):
    result = apply_edits(repo, [edit(path, "1", "2")], "code_patch")
    assert not result.ok
    assert reason in result.outcomes[0].error


def test_test_change_may_edit_a_singular_test(repo: Path):
    result = apply_edits(
        repo, [edit("tests/assert_volume.sql", "where false", "where 1 = 0")], "test_change"
    )
    assert result.ok


def test_absolute_and_escaping_paths_are_refused(repo: Path):
    with pytest.raises(EditRejected):
        resolve_editable(repo, str(repo / "dbt_marketing" / "models" / "x.sql"))
    with pytest.raises(EditRejected):
        resolve_editable(repo, "models/../../../etc/passwd.sql")
    with pytest.raises(EditRejected):
        resolve_editable(repo, "models/staging/data.duckdb")


def test_removing_a_test_from_yml_is_refused(repo: Path):
    result = apply_edits(
        repo,
        [edit("models/staging/_staging__models.yml", "          - unique\n", "")],
        "code_patch",
    )
    assert not result.ok
    assert "removes or moves test declarations" in result.outcomes[0].error
    assert 'stg_touchpoints/touchpoint_id: "unique"' in result.outcomes[0].error


def test_moving_a_test_between_columns_is_refused(repo: Path):
    before = SCHEMA
    after = SCHEMA.replace(
        "      - name: touch_cost_usd\n        tests:\n          - not_null\n",
        "      - name: touch_cost_usd\n      - name: other\n        tests:\n          - not_null\n",
    )
    assert weakened(before, after).startswith("removes or moves")


def test_weakening_a_test_is_refused(repo: Path):
    result = apply_edits(
        repo,
        [
            edit(
                "models/staging/_staging__models.yml",
                "      - name: touch_cost_usd\n        tests:\n          - not_null\n",
                "      - name: touch_cost_usd\n        tests:\n          - not_null:\n"
                "              config:\n                severity: warn\n",
            )
        ],
        "code_patch",
    )
    assert not result.ok
    # the not_null entry changed shape, so it counts as removed, and severity was added
    assert "only a test_change proposal" in result.outcomes[0].error


def test_adding_a_test_to_yml_is_allowed(repo: Path):
    result = apply_edits(
        repo,
        [
            edit(
                "models/staging/_staging__models.yml",
                "      - name: touch_cost_usd\n        tests:\n          - not_null\n",
                "      - name: touch_cost_usd\n        tests:\n          - not_null\n          - positive_values\n",
            )
        ],
        "code_patch",
    )
    assert result.ok


def test_freshness_block_counts_as_a_test(repo: Path):
    result = apply_edits(
        repo,
        [edit("models/staging/_sources.yml", "count: 7", "count: 70")],
        "code_patch",
    )
    assert not result.ok and "freshness" in result.outcomes[0].error
    # renaming a documented column in the same file is fine
    ok = apply_edits(
        repo,
        [edit("models/staging/_sources.yml", "- name: channel", "- name: marketing_channel")],
        "code_patch",
    )
    assert ok.ok


def test_unparseable_yml_after_edit_is_refused(repo: Path):
    result = apply_edits(
        repo,
        [edit("models/staging/_staging__models.yml", "version: 2", "version: [2")],
        "code_patch",
    )
    assert not result.ok and "does not parse" in result.outcomes[0].error


def test_new_file_needs_empty_find_and_may_add_a_singular_test_only_as_test_change(repo: Path):
    created = apply_edits(
        repo, [edit("models/staging/stg_new.sql", "", "select 1 as x\n")], "code_patch"
    )
    assert created.ok and (repo / "dbt_marketing/models/staging/stg_new.sql").exists()
    assert created.diff.startswith("--- a/dbt_marketing/models/staging/stg_new.sql")
    wrong = apply_edits(repo, [edit("models/staging/stg_other.sql", "x", "y")], "code_patch")
    assert not wrong.ok and "does not exist" in wrong.outcomes[0].error
    blocked = apply_edits(repo, [edit("tests/assert_new.sql", "", "select 1")], "code_patch")
    assert not blocked.ok


def test_collect_tests_tags_by_model_and_column():
    found = collect_tests(SCHEMA)
    assert found == [
        '/stg_touchpoints/touchpoint_id: "unique"',
        '/stg_touchpoints/touchpoint_id: "not_null"',
        '/stg_touchpoints/touch_cost_usd: "not_null"',
    ]


def test_empty_edit_list_is_reported(repo: Path):
    result = apply_edits(repo, [], "code_patch")
    assert not result.ok and result.outcomes[-1].error == "no edits given"
