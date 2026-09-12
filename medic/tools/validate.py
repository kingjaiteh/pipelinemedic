"""Validate a fix: rebuild the affected part of the pipeline in the sandbox.

The selection is derived, never chosen by the model: every failing model
(or the model a failing test belongs to) plus every model whose file the
fixer edited, each with its descendants, so the tests on them run too. A
freshness incident is re-checked with `dbt source freshness` instead.
Artifacts go to <sandbox>/validation/target so the incident's own
run_results.json is left as it was.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from medic.graph.state import FailingNode, Incident, Validation
from medic.tools.dbt_artifacts import Manifest

VALIDATION_DIR = "validation"
MODEL_FILE = re.compile(r"^dbt_marketing/models/.+/([A-Za-z0-9_]+)\.sql$")


def _tested_models(manifest: Manifest, unique_id: str) -> list[str]:
    node = manifest.nodes.get(unique_id) or {}
    if node.get("resource_type") in ("model", "seed", "snapshot"):
        return [node["name"]]
    out: list[str] = []
    for parent in node.get("depends_on", {}).get("nodes", []):
        p = manifest.nodes.get(parent) or {}
        if p.get("resource_type") in ("model", "seed", "snapshot"):
            out.append(p["name"])
    return out


def selection_for(incident: Incident, manifest: Manifest, edited_paths: list[str]) -> list[str]:
    """dbt arguments for the validation run."""
    if incident.dbt_args[:2] == ["source", "freshness"]:
        return ["source", "freshness"]
    names: list[str] = []
    for node in incident.failing_nodes:
        for name in _tested_models(manifest, node.unique_id):
            if name not in names:
                names.append(name)
    for path in edited_paths:
        m = MODEL_FILE.match(path.replace("\\", "/"))
        if m and m.group(1) not in names:
            names.append(m.group(1))
    if not names:
        return ["build"]
    return ["build", "--select", " ".join(f"{n}+" for n in names)]


def validate_in_sandbox(
    sandbox_dir: Path, sandbox_name: str, dbt_args: list[str], timeout: int = 600
) -> Validation:
    from chaos.harness import Sandbox, SandboxError

    sb = Sandbox(name=sandbox_name, root=Path(sandbox_dir))
    target_path = sb.root / VALIDATION_DIR / "target"
    try:
        run = sb.run_dbt(*dbt_args, timeout=timeout, target_path=target_path)
    except SandboxError as exc:
        return Validation(passed=False, dbt_args=dbt_args, error=str(exc))
    except Exception as exc:  # noqa: BLE001 - a timeout or a missing exe is the finding
        return Validation(passed=False, dbt_args=dbt_args, error=f"{type(exc).__name__}: {exc}")
    if run.summary is None:
        tail = run.stderr.strip()[-2000:] or run.stdout_tail(20)
        return Validation(
            passed=False,
            dbt_args=dbt_args,
            returncode=run.returncode,
            elapsed_s=run.elapsed_s,
            stdout_tail=tail,
            error=f"dbt exited {run.returncode} and wrote no artifact",
        )
    failing = [
        FailingNode(
            unique_id=n.unique_id,
            name=n.name,
            resource_type=getattr(n, "resource_type", "source"),
            status=n.status,
            message=n.message,
        )
        for n in run.summary.failing
    ]
    return Validation(
        passed=run.ok and not failing,
        dbt_args=dbt_args,
        returncode=run.returncode,
        elapsed_s=round(run.elapsed_s, 1),
        status_counts=run.summary.status_counts,
        failing=failing,
        stdout_tail=run.stdout_tail(20),
    )


def as_json(v: Validation) -> str:
    return json.dumps(v.model_dump(), default=str)
