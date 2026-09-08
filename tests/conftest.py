"""Hand-built dbt artifacts. Small enough to reason about, shaped like the real ones.

    raw.touchpoints ─> stg_touchpoints ─> int_touchpoints_sessionized ─┐
    raw.conversions ─> stg_conversions ─> int_journeys ───────────────┴─> dim_channels

dim_channels errored (a Binder Error, as in chaos scenario 5); the test on it
was skipped; everything else passed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PKG = "marketing_attribution"


def _model(name: str, parents: list[str], path: str) -> dict:
    return {
        "unique_id": f"model.{PKG}.{name}",
        "name": name,
        "resource_type": "model",
        "original_file_path": path,
        "depends_on": {"macros": [], "nodes": parents},
    }


def _source(name: str) -> dict:
    return {
        "unique_id": f"source.{PKG}.raw.{name}",
        "name": name,
        "source_name": "raw",
        "resource_type": "source",
        "original_file_path": "models/staging/_sources.yml",
        "depends_on": {"macros": [], "nodes": []},
    }


def _test(name: str, parent: str) -> dict:
    return {
        "unique_id": f"test.{PKG}.{name}.abc123",
        "name": name,
        "resource_type": "test",
        "original_file_path": "models/marts/_marts__models.yml",
        "depends_on": {"macros": [], "nodes": [parent]},
    }


def build_manifest() -> dict:
    src_t, src_c = _source("touchpoints"), _source("conversions")
    stg_t = _model("stg_touchpoints", [src_t["unique_id"]], "models/staging/stg_touchpoints.sql")
    stg_c = _model("stg_conversions", [src_c["unique_id"]], "models/staging/stg_conversions.sql")
    int_s = _model(
        "int_touchpoints_sessionized",
        [stg_t["unique_id"]],
        "models/intermediate/int_touchpoints_sessionized.sql",
    )
    int_j = _model(
        "int_journeys",
        [int_s["unique_id"], stg_c["unique_id"]],
        "models/intermediate/int_journeys.sql",
    )
    dim = _model(
        "dim_channels", [int_s["unique_id"], int_j["unique_id"]], "models/marts/dim_channels.sql"
    )
    t_dim = _test("not_null_dim_channels_channel", dim["unique_id"])
    t_stg = _test("not_null_stg_touchpoints_user_id", stg_t["unique_id"])

    nodes = {n["unique_id"]: n for n in (stg_t, stg_c, int_s, int_j, dim, t_dim, t_stg)}
    sources = {s["unique_id"]: s for s in (src_t, src_c)}
    everything = {**nodes, **sources}

    parent_map = {uid: list(n["depends_on"]["nodes"]) for uid, n in everything.items()}
    child_map: dict[str, list[str]] = {uid: [] for uid in everything}
    for uid, parents in parent_map.items():
        for p in parents:
            child_map[p].append(uid)

    return {
        "metadata": {"dbt_schema_version": "https://schemas.getdbt.com/dbt/manifest/v12.json"},
        "nodes": nodes,
        "sources": sources,
        "parent_map": parent_map,
        "child_map": child_map,
    }


BINDER_ERROR = (
    "Runtime Error in model dim_channels (models/marts/dim_channels.sql)\n"
    '  Binder Error: Referenced column "touch_cnt" not found in FROM clause!\n'
    '  Candidate bindings: "touch_count"'
)


def build_run_results() -> dict:
    def r(uid: str, status: str, message: str | None = None, secs: float = 0.1) -> dict:
        return {
            "unique_id": uid,
            "status": status,
            "message": message,
            "execution_time": secs,
            "failures": None,
            "timing": [],
        }

    m = f"model.{PKG}."
    return {
        "metadata": {
            "dbt_schema_version": "https://schemas.getdbt.com/dbt/run-results/v6.json",
            "generated_at": "2026-09-08T12:00:00Z",
        },
        "args": {"which": "build", "invocation_command": "dbt build --project-dir dbt_marketing"},
        "elapsed_time": 3.2,
        "results": [
            r(m + "stg_touchpoints", "success", "CREATE VIEW"),
            r(m + "stg_conversions", "success", "CREATE VIEW"),
            r(f"test.{PKG}.not_null_stg_touchpoints_user_id.abc123", "pass"),
            r(m + "int_touchpoints_sessionized", "success", "CREATE VIEW"),
            r(m + "int_journeys", "success", "CREATE VIEW"),
            r(m + "dim_channels", "error", BINDER_ERROR, 0.4),
            r(f"test.{PKG}.not_null_dim_channels_channel.abc123", "skipped"),
        ],
    }


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(build_manifest()), encoding="utf-8")
    return p


@pytest.fixture
def run_results_path(tmp_path: Path) -> Path:
    p = tmp_path / "run_results.json"
    p.write_text(json.dumps(build_run_results()), encoding="utf-8")
    return p
