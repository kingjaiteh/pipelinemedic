"""query_duckdb: one guarded SELECT against the sandbox copy.

The guard is the point. A single statement, SELECT or WITH only, no
writes, no file reads, no answer-key tables, a row limit stamped on, and a
timeout enforced by interrupting the connection from a timer thread. The
connection is opened per call and closed after, so dbt can still run in the
sandbox between queries (DuckDB allows one writer process per file).
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
from langchain_core.tools import BaseTool, tool

from medic.config import BUDGET

ANSWER_KEY = re.compile(r"\b(true_effects|channel_ground_truth|answer_key)\b", re.IGNORECASE)
WRITE_OR_ESCAPE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|export|import|install|load|"
    r"pragma|set|reset|call|vacuum|checkpoint|begin|commit|rollback|use)\b",
    re.IGNORECASE,
)
FILE_READ = re.compile(r"\b(read_\w+|glob|parquet_scan|sniff_csv)\s*\(|\w+://", re.IGNORECASE)
TRAILING_LIMIT = re.compile(r"\blimit\s+(\d+)\s*$", re.IGNORECASE)


class SqlRejected(ValueError):
    pass


def guard_sql(sql: str, row_limit: int = BUDGET.sql_row_limit) -> str:
    """Return the statement that will run, or raise SqlRejected with the reason."""
    s = sql.strip()
    while s.endswith(";"):
        s = s[:-1].rstrip()
    if not s:
        raise SqlRejected("empty statement")
    if ";" in s:
        raise SqlRejected("one statement per call")
    if not re.match(r"^(select|with)\b", s, re.IGNORECASE):
        raise SqlRejected("SELECT (or WITH ... SELECT) only")
    if ANSWER_KEY.search(s):
        raise SqlRejected(
            "raw.true_effects and raw.channel_ground_truth are the answer key; off limits"
        )
    if FILE_READ.search(s):
        raise SqlRejected("file and URL reads are not allowed; query tables only")
    hit = WRITE_OR_ESCAPE.search(s)
    if hit:
        raise SqlRejected(f"'{hit.group(1)}' is not allowed in a read-only query")
    m = TRAILING_LIMIT.search(s)
    if m:
        if int(m.group(1)) > row_limit:
            s = s[: m.start()] + f"limit {row_limit}"
    else:
        s = f"{s}\nlimit {row_limit}"
    return s


@dataclass
class QueryResult:
    sql: str
    columns: list[str]
    rows: list[list[Any]]
    elapsed_s: float
    row_limit: int
    note: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = {
            "columns": self.columns,
            "rows": self.rows,
            "row_count": len(self.rows),
            "elapsed_s": round(self.elapsed_s, 3),
        }
        if len(self.rows) >= self.row_limit:
            payload["note"] = f"row limit {self.row_limit} reached; aggregate or filter for more"
        elif self.note:
            payload["note"] = self.note
        return json.dumps(payload, default=str)


def run_query(
    db_path: Path,
    sql: str,
    *,
    row_limit: int = BUDGET.sql_row_limit,
    timeout_s: float = BUDGET.sql_timeout_s,
) -> QueryResult:
    guarded = guard_sql(sql, row_limit)
    con = duckdb.connect(str(db_path), read_only=True)
    timer = threading.Timer(timeout_s, con.interrupt)
    started = time.monotonic()
    try:
        timer.start()
        cur = con.execute(guarded)
        columns = [d[0] for d in cur.description or []]
        rows = [list(r) for r in cur.fetchall()]
    except duckdb.InterruptException as exc:
        raise TimeoutError(f"query exceeded {timeout_s} s and was interrupted") from exc
    finally:
        timer.cancel()
        con.close()
    return QueryResult(guarded, columns, rows, time.monotonic() - started, row_limit)


def make_query_tool(db_path: Path) -> BaseTool:
    @tool
    def query_duckdb(sql: str) -> str:
        """Run one read-only SELECT against the sandbox copy of the warehouse and
        return columns and rows as JSON. Rules: a single SELECT or WITH statement,
        no writes, a limit of 200 rows is applied, 10 second timeout, and the
        answer-key tables raw.true_effects and raw.channel_ground_truth are
        refused. Schemas: raw (touchpoints, conversions, kaggle_ads), staging
        (stg_*, int_* views), marts (dim_*, fct_* tables). Useful starts:
        select column_name, data_type from information_schema.columns where
        table_schema = 'raw' and table_name = 'touchpoints'."""
        try:
            return run_query(db_path, sql).to_json()
        except SqlRejected as exc:
            return json.dumps({"error": f"query rejected: {exc}"})
        except TimeoutError as exc:
            return json.dumps({"error": str(exc)})
        except duckdb.Error as exc:
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    return query_duckdb
