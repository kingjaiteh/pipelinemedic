"""get_test_failures: the rows a failing dbt test flagged.

dbt writes every test's compiled SQL into run_results.json (`compiled_code`).
A test that failed with "Got N results" is a SELECT that returned N rows, so
running that SELECT against the sandbox copy shows exactly which rows it
objected to. The query goes through the same guard as query_duckdb.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import BaseTool, tool

from medic.tools.dbt_artifacts import _name_and_type
from medic.tools.warehouse import SqlRejected, run_query


class TestNotFound(LookupError):
    pass


def compiled_test_sql(run_results_path: Path, test: str) -> tuple[str, str]:
    """(unique_id, compiled SQL) for a test named by unique_id or bare name."""
    data = json.loads(Path(run_results_path).read_text(encoding="utf-8"))
    tests = [r for r in data.get("results", []) if r["unique_id"].startswith("test.")]
    for r in tests:
        if r["unique_id"] == test or _name_and_type(r["unique_id"])[0] == test:
            sql = r.get("compiled_code")
            if not sql:
                raise TestNotFound(f"{test!r} has no compiled SQL in run_results.json")
            return r["unique_id"], sql
    names = sorted(
        _name_and_type(r["unique_id"])[0] for r in tests if r.get("status") in ("fail", "error")
    )
    raise TestNotFound(f"no test named {test!r} in run_results.json; failing tests: {names}")


def failing_rows(run_results_path: Path, db_path: Path, test: str) -> str:
    try:
        uid, sql = compiled_test_sql(run_results_path, test)
    except TestNotFound as exc:
        return json.dumps({"error": str(exc)})
    try:
        result = run_query(db_path, sql)
    except SqlRejected as exc:
        return json.dumps({"error": f"compiled test SQL rejected by the guard: {exc}"})
    except TimeoutError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:  # noqa: BLE001 - DuckDB errors are the finding here
        return json.dumps({"error": f"{type(exc).__name__}: {exc}", "test": uid})
    payload = json.loads(result.to_json())
    payload["test"] = uid
    return json.dumps(payload, default=str)


def make_test_failures_tool(run_results_path: Path, db_path: Path) -> BaseTool:
    @tool
    def get_test_failures(test: str) -> str:
        """Run a failing dbt test's own compiled SQL against the sandbox copy and
        return the rows it flagged (up to 200). `test` is the test's name as shown
        by get_run_results, for example not_null_stg_touchpoints_user_id or
        assert_daily_touchpoint_volume. For a test that failed with "Got N
        results" this is the most direct evidence there is."""
        return failing_rows(run_results_path, db_path, test)

    return get_test_failures
