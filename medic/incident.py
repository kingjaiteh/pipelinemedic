"""Build an Incident from a chaos record or from a run_results path plus a sandbox."""

from __future__ import annotations

import json
from pathlib import Path

from medic.graph.state import FailingNode, Incident
from medic.tools.dbt_artifacts import summarize_run_results, summarize_source_freshness


def failing_nodes_from_artifact(artifact: Path) -> tuple[list[FailingNode], dict[str, int]]:
    if artifact.name == "sources.json":
        s = summarize_source_freshness(artifact)
        nodes = [
            FailingNode(
                unique_id=f.unique_id,
                name=f.name,
                resource_type="source",
                status=f.status,
                message=f.message,
            )
            for f in s.failing
        ]
        return nodes, s.status_counts
    s = summarize_run_results(artifact)
    nodes = [
        FailingNode(
            unique_id=f.unique_id,
            name=f.name,
            resource_type=f.resource_type,
            status=f.status,
            message=f.message,
        )
        for f in s.failing
    ]
    return nodes, s.status_counts


def incident_from_record(record_path: Path) -> Incident:
    """From sandbox/incidents/<key>.json written by `medic chaos run`."""
    data = json.loads(record_path.read_text(encoding="utf-8"))
    if data.get("torn_down"):
        raise FileNotFoundError(
            f"sandbox for scenario {data['scenario']} was torn down; run `medic chaos run {data['scenario']}` first"
        )
    artifact = Path(data["run"]["artifact"])
    if not artifact.exists():
        raise FileNotFoundError(f"artifact missing: {artifact}")
    nodes, counts = failing_nodes_from_artifact(artifact)
    return Incident(
        source="chaos",
        run_id=f"chaos-{data['key']}",
        scenario=data["scenario"],
        scenario_key=data["key"],
        dbt_args=list(data["run"]["args"]),
        artifact=str(artifact),
        manifest=data["manifest"],
        sandbox_dir=data["sandbox"],
        duckdb=data["duckdb"],
        repo=data["repo"],
        failing_nodes=nodes,
        status_counts=counts,
    )


def incident_from_paths(
    artifact: Path,
    sandbox_dir: Path,
    manifest: Path | None = None,
    dbt_args: list[str] | None = None,
) -> Incident:
    """From an artifact and a sandbox directory holding marketing.duckdb and repo/."""
    sandbox_dir = sandbox_dir.resolve()
    duckdb = sandbox_dir / "marketing.duckdb"
    repo = sandbox_dir / "repo"
    if not duckdb.exists() or not repo.exists():
        raise FileNotFoundError(
            f"{sandbox_dir} needs marketing.duckdb and repo/ (see medic chaos run)"
        )
    manifest = manifest or artifact.parent / "manifest.json"
    nodes, counts = failing_nodes_from_artifact(artifact)
    args = dbt_args or (["source", "freshness"] if artifact.name == "sources.json" else ["build"])
    return Incident(
        source="run_results",
        run_id=artifact.parent.name,
        dbt_args=args,
        artifact=str(artifact),
        manifest=str(manifest),
        sandbox_dir=str(sandbox_dir),
        duckdb=str(duckdb),
        repo=str(repo),
        failing_nodes=nodes,
        status_counts=counts,
    )
