"""Run one scenario end to end: sandbox, mutation, dbt, record, check.

The record written to sandbox/incidents/<key>.json is the incident that
`medic triage --scenario N` will pick up in Phase 1b.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chaos.harness import DbtRun, Sandbox, sha256
from chaos.mutation import MutationContext
from chaos.scenarios import Scenario
from medic.config import SANDBOX_ROOT, TARGET, PipelineTarget

INCIDENTS_DIR_NAME = "incidents"


@dataclass
class Expectation:
    """Did the sandbox fail the way scenarios.yaml says it should?"""

    expected_failing: list[str]
    found: list[str]
    missing: list[str]
    error_found: bool
    dbt_failed: bool

    @property
    def matched(self) -> bool:
        return self.dbt_failed and bool(self.found) and self.error_found


def check_expectation(scenario: Scenario, run: DbtRun) -> Expectation:
    ids = set(run.failing_ids)
    names = set(run.failing_names)
    found = [e for e in scenario.expected_failing if e in ids or e in names]
    missing = [e for e in scenario.expected_failing if e not in found]
    messages = [n.message or "" for n in run.summary.failing] if run.summary else []
    error_found = any(scenario.error_contains in m for m in messages)
    return Expectation(list(scenario.expected_failing), found, missing, error_found, not run.ok)


@dataclass
class IncidentRecord:
    scenario: int
    key: str
    title: str
    category: str
    fix_kind: str
    created_at: str
    sandbox: str
    duckdb: str
    repo: str
    manifest: str
    run: dict[str, Any]
    expectation: dict[str, Any]
    matched: bool
    production_sha256_before: str
    production_sha256_after: str
    torn_down: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def production_untouched(self) -> bool:
        return self.production_sha256_before == self.production_sha256_after

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: Path) -> IncidentRecord:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def incident_path(key: str, sandbox_root: Path = SANDBOX_ROOT) -> Path:
    return sandbox_root / INCIDENTS_DIR_NAME / f"{key}.json"


def run_scenario(
    scenario: Scenario,
    *,
    sandbox_root: Path = SANDBOX_ROOT,
    target: PipelineTarget = TARGET,
    teardown: bool = False,
    dbt_timeout: int = 600,
    log: Callable[[str], None] = lambda _: None,
) -> IncidentRecord:
    before = sha256(target.duckdb)
    log(f"[{scenario.number}] {scenario.title}")
    log(f"    production sha256 {before[:12]}")

    sb = Sandbox.create(scenario.key, sandbox_root=sandbox_root, target=target, replace=True)
    log(f"    sandbox {sb.root}")

    con = sb.connect()
    try:
        scenario.apply(MutationContext(con=con, repo=sb.repo))
    finally:
        con.close()
    log(f"    mutation applied: {scenario.mutation}")

    run = sb.run_dbt(*scenario.dbt_args, timeout=dbt_timeout)
    log(f"    dbt {' '.join(scenario.dbt_args)}: exit {run.returncode} in {run.elapsed_s:.0f}s")
    if run.summary is None:
        log("    no artifact written; dbt stderr/stdout tail follows")
        log(run.stderr.strip() or run.stdout_tail())
    else:
        counts = ", ".join(f"{k}={v}" for k, v in sorted(run.summary.status_counts.items()))
        log(f"    {counts}")
        for node in run.summary.failing:
            first_line = (node.message or "").strip().splitlines()[:1]
            log(f"    {node.status:>6}  {node.name}  {first_line[0] if first_line else ''}")

    expectation = check_expectation(scenario, run)
    log(f"    expected {expectation.expected_failing} -> found {expectation.found}")
    log(
        f"    error text {'found' if expectation.error_found else 'NOT found'}: {scenario.error_contains!r}"
    )

    after = sha256(target.duckdb)
    record = IncidentRecord(
        scenario=scenario.number,
        key=scenario.key,
        title=scenario.title,
        category=scenario.category,
        fix_kind=scenario.fix_kind,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        sandbox=str(sb.root),
        duckdb=str(sb.duckdb),
        repo=str(sb.repo),
        manifest=str(sb.manifest),
        run=run.to_dict(),
        expectation=asdict(expectation),
        matched=expectation.matched,
        production_sha256_before=before,
        production_sha256_after=after,
    )
    if teardown:
        sb.teardown()
        record.torn_down = True
        log("    sandbox torn down")
    path = record.write(incident_path(scenario.key, sandbox_root))
    log(f"    record {path}")
    log(
        f"    production untouched: {record.production_untouched}; expectation matched: {record.matched}"
    )
    return record
