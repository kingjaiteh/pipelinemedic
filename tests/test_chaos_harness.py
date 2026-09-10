"""Sandbox lifecycle, scenario registry, expectation check, freshness parsing. No dbt."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import duckdb
import pytest

from chaos.harness import DbtRun, Sandbox, SandboxError, sha256
from chaos.runner import IncidentRecord, check_expectation, incident_path
from chaos.scenarios import Scenario, get_scenario, load_scenarios
from medic.config import PipelineTarget
from medic.tools.dbt_artifacts import summarize_run_results, summarize_source_freshness

PKG = "marketing_attribution"


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


def test_registry_has_seven_scenarios_with_callable_mutations():
    scenarios = load_scenarios()
    assert sorted(scenarios) == [1, 2, 3, 4, 5, 6, 7]
    assert len({s.key for s in scenarios.values()}) == 7
    for s in scenarios.values():
        assert callable(s.apply)
        assert s.dbt_args[0] in {"build", "source"}
        assert s.category in {
            "schema_drift",
            "data_quality",
            "code_error",
            "source_freshness",
            "volume_anomaly",
        }
        assert s.fix_kind in {"code_patch", "upstream_data_issue"}
        assert s.expected_failing and s.error_contains


def test_fix_kinds_match_the_plan():
    kinds = {n: s.fix_kind for n, s in load_scenarios().items()}
    assert [n for n, k in kinds.items() if k == "code_patch"] == [1, 4, 5]
    assert [n for n, k in kinds.items() if k == "upstream_data_issue"] == [2, 3, 6, 7]


def test_get_scenario_unknown():
    with pytest.raises(KeyError):
        get_scenario(99)


# --------------------------------------------------------------------------- #
# sandbox lifecycle against a throwaway "pipeline" repo (real git, no dbt)
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PipelineTarget:
    for var, value in {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }.items():
        monkeypatch.setenv(var, value)
    repo = tmp_path / "pipeline"
    (repo / "dbt_marketing" / "models").mkdir(parents=True)
    (repo / "dbt_marketing" / "dbt_project.yml").write_text("name: x\n", encoding="utf-8")
    # git does not track empty directories, so the worktree needs a file under models/
    (repo / "dbt_marketing" / "models" / "stg.sql").write_text("select 1", encoding="utf-8")
    (repo / "data").mkdir()
    con = duckdb.connect(str(repo / "data" / "marketing.duckdb"))
    con.execute("create schema raw; create table raw.t as select 1 as x")
    con.close()

    def git(*a: str) -> None:
        subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    git("add", ".")
    git("commit", "-q", "-m", "seed")
    return PipelineTarget(repo=repo)


def _worktrees(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "worktree", "list"], capture_output=True, text=True, check=True
    ).stdout


def _branches(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "branch"], capture_output=True, text=True, check=True
    ).stdout


def test_sandbox_create_copies_db_and_adds_worktree(fake_pipeline: PipelineTarget, tmp_path):
    root = tmp_path / "sandboxes"
    before = sha256(fake_pipeline.duckdb)

    sb = Sandbox.create("probe", sandbox_root=root, target=fake_pipeline)
    assert sb.exists()
    assert sb.duckdb.name == "marketing.duckdb"
    assert sb.duckdb.parent == root / "probe"
    assert sha256(sb.duckdb) == before
    assert (sb.repo / "dbt_marketing" / "dbt_project.yml").exists()
    assert "medic/probe" in _branches(fake_pipeline.repo)
    assert str(sb.repo.as_posix()) in _worktrees(fake_pipeline.repo).replace("\\", "/")

    # mutate the copy and commit in the worktree: production stays byte-identical
    con = sb.connect()
    con.execute("insert into raw.t values (2)")
    con.close()
    (sb.repo / "dbt_marketing" / "models" / "m.sql").write_text("select 1", encoding="utf-8")
    sb.git("add", ".")
    sb.git("commit", "-q", "-m", "chaos")
    assert sha256(fake_pipeline.duckdb) == before
    assert sb.git("log", "--format=%s", "-1") == "chaos"
    main_log = subprocess.run(
        ["git", "-C", str(fake_pipeline.repo), "log", "--format=%s", "-1"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert main_log == "seed"

    with pytest.raises(SandboxError):
        Sandbox.create("probe", sandbox_root=root, target=fake_pipeline)

    sb.teardown()
    assert not root.joinpath("probe").exists()
    assert "medic/probe" not in _branches(fake_pipeline.repo)
    assert "probe" not in _worktrees(fake_pipeline.repo)
    sb.teardown()  # idempotent


def test_sandbox_replace_rebuilds_from_main(fake_pipeline: PipelineTarget, tmp_path):
    root = tmp_path / "sandboxes"
    sb = Sandbox.create("again", sandbox_root=root, target=fake_pipeline)
    (sb.repo / "junk.txt").write_text("x", encoding="utf-8")
    sb2 = Sandbox.create("again", sandbox_root=root, target=fake_pipeline, replace=True)
    assert not (sb2.repo / "junk.txt").exists()
    sb2.teardown()


def test_sandbox_names_are_restricted(fake_pipeline: PipelineTarget, tmp_path):
    for bad in ("", "Has Space", "../escape", "UPPER", "a/b"):
        with pytest.raises(SandboxError):
            Sandbox.at(bad, sandbox_root=tmp_path, target=fake_pipeline)


def test_dbt_env_and_command_point_at_the_sandbox(fake_pipeline: PipelineTarget, tmp_path):
    sb = Sandbox.at("env", sandbox_root=tmp_path, target=fake_pipeline)
    env = sb.dbt_env()
    assert env["MARKETING_DUCKDB"] == str(tmp_path / "env" / "marketing.duckdb")
    assert env["DBT_PROFILES_DIR"] == str(sb.dbt_dir)
    cmd = sb.dbt_command("build", "--select", "dim_channels+")
    assert cmd[0] == str(fake_pipeline.dbt_exe)
    assert cmd[1:4] == ["build", "--select", "dim_channels+"]
    assert "--project-dir" in cmd and str(sb.dbt_dir) in cmd
    with pytest.raises(SandboxError):
        sb.run_dbt("build")  # no dbt.exe in the fake pipeline


# --------------------------------------------------------------------------- #
# expectation check and incident record
# --------------------------------------------------------------------------- #


def _scenario(**overrides) -> Scenario:
    base = {
        "number": 5,
        "key": "bad_commit",
        "title": "Bad commit",
        "mutation": "typo",
        "dbt_args": ("build",),
        "category": "code_error",
        "fix_kind": "code_patch",
        "expected_failing": ("dim_channels",),
        "error_contains": "Binder Error",
        "apply": lambda ctx: None,
    }
    base.update(overrides)
    return Scenario(**base)


def test_check_expectation_matches_on_name_or_unique_id(run_results_path):
    summary = summarize_run_results(run_results_path)
    run = DbtRun(("build",), 1, "", "", 1.0, run_results_path, summary)

    ok = check_expectation(_scenario(), run)
    assert ok.matched and ok.found == ["dim_channels"] and ok.missing == []

    by_id = check_expectation(_scenario(expected_failing=(f"model.{PKG}.dim_channels",)), run)
    assert by_id.matched

    wrong_node = check_expectation(_scenario(expected_failing=("stg_touchpoints",)), run)
    assert not wrong_node.matched and wrong_node.missing == ["stg_touchpoints"]

    wrong_text = check_expectation(_scenario(error_contains="Conversion Error"), run)
    assert not wrong_text.matched and not wrong_text.error_found

    passed = DbtRun(("build",), 0, "", "", 1.0, run_results_path, summary)
    assert not check_expectation(_scenario(), passed).matched


def test_check_expectation_without_artifact():
    run = DbtRun(("build",), 2, "", "boom", 0.1, None, None)
    exp = check_expectation(_scenario(), run)
    assert not exp.matched and exp.missing == ["dim_channels"]


def test_incident_record_roundtrip(tmp_path: Path):
    record = IncidentRecord(
        scenario=5,
        key="bad_commit",
        title="Bad commit",
        category="code_error",
        fix_kind="code_patch",
        created_at="2026-09-10T00:00:00+00:00",
        sandbox="s",
        duckdb="d",
        repo="r",
        manifest="m",
        run={"returncode": 1},
        expectation={"found": ["dim_channels"]},
        matched=True,
        production_sha256_before="a",
        production_sha256_after="a",
    )
    path = record.write(incident_path("bad_commit", tmp_path))
    assert path == tmp_path / "incidents" / "bad_commit.json"
    back = IncidentRecord.read(path)
    assert back == record and back.production_untouched and not back.torn_down
    changed = IncidentRecord(**{**json.loads(path.read_text()), "production_sha256_after": "b"})
    assert not changed.production_untouched


# --------------------------------------------------------------------------- #
# sources.json
# --------------------------------------------------------------------------- #


def test_summarize_source_freshness(tmp_path: Path):
    day = 86400.0
    data = {
        "metadata": {"generated_at": "2026-09-10T12:00:00Z"},
        "results": [
            {
                "unique_id": f"source.{PKG}.raw.touchpoints",
                "status": "error",
                "max_loaded_at": "2026-08-10T12:00:00",
                "max_loaded_at_time_ago_in_s": 31 * day,
                "criteria": {
                    "warn_after": {"count": 2, "period": "day"},
                    "error_after": {"count": 7, "period": "day"},
                },
            },
            {
                "unique_id": f"source.{PKG}.raw.conversions",
                "status": "warn",
                "max_loaded_at_time_ago_in_s": 3 * day,
                "criteria": {"warn_after": {"count": 2, "period": "day"}},
            },
            {"unique_id": f"source.{PKG}.raw.kaggle_ads", "status": "pass"},
            {"unique_id": f"source.{PKG}.raw.broken", "status": "runtime error", "error": "boom"},
        ],
    }
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    s = summarize_source_freshness(path)
    assert s.total == 4
    assert s.status_counts == {"error": 1, "warn": 1, "pass": 1, "runtime error": 1}
    assert [f.name for f in s.failing] == ["raw.touchpoints", "raw.broken"]
    stale = s.failing[0]
    assert stale.age_days == 31.0 and stale.error_after == "7 day" and stale.warn_after == "2 day"
    assert "31.0 days old" in stale.message
    assert s.failing[1].message == "boom"
    assert [w.name for w in s.warning] == ["raw.conversions"]
    assert s.warning[0].error_after is None
