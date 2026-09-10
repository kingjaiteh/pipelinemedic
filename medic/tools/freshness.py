"""get_source_freshness: run `dbt source freshness` in the sandbox and summarize."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import BaseTool, tool

from medic.tools.dbt_artifacts import summarize_source_freshness

TIMEOUT_S = 180


def source_freshness(sandbox_dir: Path, name: str = "incident") -> str:
    from chaos.harness import Sandbox, SandboxError

    sb = Sandbox(name=name, root=Path(sandbox_dir))
    try:
        run = sb.run_dbt("source", "freshness", timeout=TIMEOUT_S)
    except SandboxError as exc:
        return json.dumps({"error": str(exc)})
    if not sb.sources_json.exists():
        return json.dumps(
            {
                "error": "dbt wrote no sources.json",
                "stderr": run.stderr[-2000:],
                "stdout_tail": run.stdout_tail(15),
            }
        )
    summary = summarize_source_freshness(sb.sources_json)
    payload = summary.model_dump()
    payload["dbt_exit_code"] = run.returncode
    for entry in payload["failing"] + payload["warning"]:
        if entry.get("age_s") is not None:
            entry["age_days"] = round(entry["age_s"] / 86400, 1)
    return json.dumps(payload)


def make_freshness_tool(sandbox_dir: Path, name: str = "incident") -> BaseTool:
    @tool
    def get_source_freshness() -> str:
        """Run `dbt source freshness` in the sandbox and return, per source table,
        the status (pass, warn, error), how old the newest row is in days, and the
        warn and error thresholds. Takes several seconds. Use it when a source
        may be stale or a freshness check is what failed."""
        return source_freshness(sandbox_dir, name)

    return get_source_freshness
