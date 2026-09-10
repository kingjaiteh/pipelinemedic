"""What a scenario is allowed to touch: one DuckDB connection and one worktree.

Scenario functions take a MutationContext rather than a Sandbox so the unit
tests can hand them an in-memory database and a throwaway git repo instead of
the 40 MB warehouse copy.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

# The day scenarios 3 and 7 operate on. Inside the steady-state window of the
# generator (first 54 days) with seven full days before it, so the volume test
# has a trailing median to compare against.
TARGET_DAY = "2026-02-10"


class MutationError(RuntimeError):
    """The sandbox does not look like the pipeline this scenario was written for."""


@dataclass
class MutationContext:
    con: duckdb.DuckDBPyConnection
    repo: Path | None = None

    def sql(self, query: str, params: list[Any] | None = None) -> list[tuple]:
        return self.con.execute(query, params or []).fetchall()

    def scalar(self, query: str, params: list[Any] | None = None) -> Any:
        row = self.con.execute(query, params or []).fetchone()
        return None if row is None else row[0]

    def columns(self, schema: str, table: str) -> list[str]:
        rows = self.sql(
            "select column_name from information_schema.columns "
            "where table_schema = ? and table_name = ? order by ordinal_position",
            [schema, table],
        )
        return [r[0] for r in rows]

    def require_column(self, schema: str, table: str, column: str) -> None:
        if column not in self.columns(schema, table):
            raise MutationError(f"{schema}.{table} has no column {column!r}")

    def git(self, *args: str) -> str:
        if self.repo is None:
            raise MutationError("this scenario needs a git worktree and none was provided")
        proc = subprocess.run(
            ["git", "-C", str(self.repo), *args], capture_output=True, text=True, check=False
        )
        if proc.returncode != 0:
            raise MutationError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
        return proc.stdout.strip()
