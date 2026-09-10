"""Phase 1 tool guards: SQL, file paths, git history. No sandbox, no API."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import duckdb
import pytest

from medic.tools.git_history import make_git_tool, recent_changes
from medic.tools.pipeline_files import PathRejected, list_files, make_file_tools, read_file
from medic.tools.warehouse import SqlRejected, guard_sql, make_query_tool, run_query

# --------------------------------------------------------------------------- #
# query_duckdb
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "sql",
    [
        "delete from raw.touchpoints",
        "select 1; select 2",
        "update raw.touchpoints set cost = 0",
        "select * from raw.true_effects",
        "select * from raw.channel_ground_truth limit 5",
        "select * from read_csv('C:/secrets.csv')",
        "select * from 'https://example.com/x.parquet'",
        "create table t as select 1",
        "attach 'other.duckdb'",
        "pragma database_list",
        "",
        "   ;  ",
        "with x as (select 1) insert into t select * from x",
    ],
)
def test_guard_rejects(sql):
    with pytest.raises(SqlRejected):
        guard_sql(sql)


def test_guard_adds_or_clamps_limit():
    assert guard_sql("select 1").endswith("limit 200")
    assert guard_sql("select 1;").endswith("limit 200")
    assert guard_sql("select 1 limit 5").endswith("limit 5")
    assert guard_sql("select 1 LIMIT 9999").endswith("limit 200")
    assert guard_sql("with a as (select 1 as x) select x from a").startswith("with")


def test_run_query_and_tool(tmp_path: Path):
    db = tmp_path / "marketing.duckdb"
    con = duckdb.connect(str(db))
    con.execute("create schema raw; create table raw.t as select range as x from range(500)")
    con.close()

    result = run_query(db, "select x from raw.t order by x")
    assert result.columns == ["x"] and len(result.rows) == 200
    assert "row limit 200 reached" in result.to_json()

    tool = make_query_tool(db)
    assert tool.name == "query_duckdb"
    ok = json.loads(tool.invoke({"sql": "select count(*) as n from raw.t"}))
    assert ok["rows"] == [[500]] and ok["columns"] == ["n"]
    bad = json.loads(tool.invoke({"sql": "drop table raw.t"}))
    assert "rejected" in bad["error"]
    missing = json.loads(tool.invoke({"sql": "select * from raw.nope"}))
    assert "CatalogException" in missing["error"]
    # the guard ran before the connection opened, so the file is still intact
    assert run_query(db, "select count(*) from raw.t").rows == [[500]]


def test_run_query_timeout(tmp_path: Path):
    db = tmp_path / "marketing.duckdb"
    duckdb.connect(str(db)).close()
    slow = "select count(*) from range(200000000) a cross join range(200000000) b"
    with pytest.raises(TimeoutError):
        run_query(db, slow, timeout_s=0.2)


# --------------------------------------------------------------------------- #
# read_pipeline_file / list_pipeline_files
# --------------------------------------------------------------------------- #


@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    models = repo / "dbt_marketing" / "models" / "marts"
    models.mkdir(parents=True)
    (models / "dim_channels.sql").write_text(
        "select\n    c.touch_count,\nfrom c\n", encoding="utf-8"
    )
    (repo / "dbt_marketing" / "dbt_project.yml").write_text("name: x\n", encoding="utf-8")
    (repo / "dbt_marketing" / "target").mkdir()
    (repo / "dbt_marketing" / "target" / "manifest.json").write_text("{}", encoding="utf-8")
    (repo / ".git").mkdir()
    (repo / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (repo / "data.duckdb").write_bytes(b"\x00")
    (tmp_path / "outside.sql").write_text("select 1", encoding="utf-8")
    return repo


def test_read_file_accepts_repo_and_project_relative_paths(worktree: Path):
    a = read_file(worktree, "dbt_marketing/models/marts/dim_channels.sql")
    b = read_file(worktree, "models/marts/dim_channels.sql")
    c = read_file(worktree, "models\\marts\\dim_channels.sql")
    assert a == b == c
    assert a.startswith("# dbt_marketing/models/marts/dim_channels.sql (3 lines)\n")
    assert "c.touch_count" in a


@pytest.mark.parametrize(
    "path",
    ["../outside.sql", "dbt_marketing/../../outside.sql", ".git/config", "data.duckdb"],
)
def test_read_file_refuses(worktree: Path, path: str):
    with pytest.raises(PathRejected):
        read_file(worktree, path)


def test_read_file_absolute_and_missing(worktree: Path):
    with pytest.raises(PathRejected):
        read_file(worktree, str(worktree / "dbt_marketing" / "dbt_project.yml"))
    with pytest.raises(FileNotFoundError):
        read_file(worktree, "models/nope.sql")


def test_read_file_truncates(worktree: Path):
    big = worktree / "dbt_marketing" / "big.sql"
    big.write_text("x" * 100, encoding="utf-8")
    out = read_file(worktree, "big.sql", max_chars=40)
    assert out.endswith("...[truncated at 40 chars]") and "x" * 41 not in out


def test_list_files_skips_target_and_git(worktree: Path):
    files = list_files(worktree)
    assert "dbt_marketing/models/marts/dim_channels.sql" in files
    assert "dbt_marketing/dbt_project.yml" in files
    assert not any("target" in f or ".git" in f or f.endswith(".duckdb") for f in files)
    with pytest.raises(PathRejected):
        list_files(worktree, "..")


def test_file_tools_wrap_errors_as_json(worktree: Path):
    read_tool, list_tool = make_file_tools(worktree)
    assert read_tool.name == "read_pipeline_file" and list_tool.name == "list_pipeline_files"
    assert "c.touch_count" in read_tool.invoke({"path": "models/marts/dim_channels.sql"})
    assert "error" in json.loads(read_tool.invoke({"path": "../outside.sql"}))
    assert "list_pipeline_files" in json.loads(read_tool.invoke({"path": "models/x.sql"}))["error"]
    listed = json.loads(list_tool.invoke({"subdir": "dbt_marketing/models"}))
    assert listed == ["dbt_marketing/models/marts/dim_channels.sql"]


# --------------------------------------------------------------------------- #
# get_recent_changes
# --------------------------------------------------------------------------- #


@pytest.fixture
def history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for var, value in {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }.items():
        monkeypatch.setenv(var, value)
    repo = tmp_path / "repo"
    model = repo / "dbt_marketing" / "models" / "marts" / "dim_channels.sql"
    model.parent.mkdir(parents=True)

    def git(*a: str) -> None:
        subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)

    model.write_text("select c.touch_count from c\n", encoding="utf-8")
    git("init", "-q", "-b", "main")
    git("add", ".")
    git("commit", "-q", "-m", "seed")
    model.write_text("select c.touch_cnt from c\n", encoding="utf-8")
    git("commit", "-q", "-am", "Tidy the dim_channels column list")
    return repo


def test_recent_changes_shows_newest_first_with_diff(history: Path):
    out = recent_changes(history, "models/marts/dim_channels.sql", n=5)
    assert out.index("Tidy the dim_channels column list") < out.index("seed")
    assert "-select c.touch_count from c" in out and "+select c.touch_cnt from c" in out
    whole = recent_changes(history, None, n=1)
    assert "Tidy" in whole and "seed" not in whole


def test_recent_changes_guards_and_tool(history: Path):
    with pytest.raises(PathRejected):
        recent_changes(history, "../elsewhere")
    tool = make_git_tool(history)
    assert tool.name == "get_recent_changes"
    assert "Tidy" in tool.invoke({"path": "dbt_marketing/models/marts/dim_channels.sql", "n": 2})
    assert "no commits touch" in tool.invoke({"path": "models/nothing.sql"})
    assert "error" in json.loads(tool.invoke({"path": "../x"}))
