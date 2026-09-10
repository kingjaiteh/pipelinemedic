"""get_test_failures runs a failing test's compiled SQL against a tiny warehouse."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from medic.tools.failing_rows import TestNotFound, compiled_test_sql, make_test_failures_tool

PKG = "marketing_attribution"


@pytest.fixture
def warehouse(tmp_path: Path) -> Path:
    db = tmp_path / "marketing.duckdb"
    con = duckdb.connect(str(db))
    con.execute("create schema raw")
    con.execute(
        "create table raw.touchpoints as select * from (values (1, 'a'), (2, null), (3, null)) t(user_id, channel)"
    )
    con.close()
    return db


@pytest.fixture
def run_results(tmp_path: Path) -> Path:
    data = {
        "results": [
            {
                "unique_id": f"model.{PKG}.stg_touchpoints",
                "status": "success",
                "compiled_code": "select 1",
            },
            {
                "unique_id": f"test.{PKG}.not_null_stg_touchpoints_channel.abc",
                "status": "fail",
                "message": "Got 2 results, configured to fail if != 0",
                "compiled_code": "select user_id, channel from raw.touchpoints where channel is null",
            },
            {
                "unique_id": f"test.{PKG}.assert_something",
                "status": "fail",
                "compiled_code": "delete from raw.touchpoints",
            },
            {
                "unique_id": f"test.{PKG}.unique_stg_touchpoints_user_id.def",
                "status": "pass",
                "compiled_code": None,
            },
        ]
    }
    path = tmp_path / "run_results.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_compiled_sql_by_name_or_unique_id(run_results: Path):
    uid, sql = compiled_test_sql(run_results, "not_null_stg_touchpoints_channel")
    assert uid.endswith("not_null_stg_touchpoints_channel.abc") and sql.startswith("select user_id")
    assert compiled_test_sql(run_results, uid)[0] == uid
    with pytest.raises(TestNotFound) as exc:
        compiled_test_sql(run_results, "nope")
    assert "assert_something" in str(exc.value) and "not_null_stg_touchpoints_channel" in str(
        exc.value
    )
    with pytest.raises(TestNotFound):
        compiled_test_sql(run_results, "unique_stg_touchpoints_user_id")  # passed, no compiled SQL


def test_tool_returns_flagged_rows_and_guards(run_results: Path, warehouse: Path):
    tool = make_test_failures_tool(run_results, warehouse)
    assert tool.name == "get_test_failures"
    out = json.loads(tool.invoke({"test": "not_null_stg_touchpoints_channel"}))
    assert out["columns"] == ["user_id", "channel"]
    assert out["rows"] == [[2, None], [3, None]]
    assert out["test"].startswith(f"test.{PKG}.not_null")

    guarded = json.loads(tool.invoke({"test": "assert_something"}))
    assert "rejected by the guard" in guarded["error"]
    missing = json.loads(tool.invoke({"test": "nothing"}))
    assert "no test named" in missing["error"]
    # the guard fired before anything ran: the table is intact
    con = duckdb.connect(str(warehouse), read_only=True)
    assert con.execute("select count(*) from raw.touchpoints").fetchone()[0] == 3
    con.close()
