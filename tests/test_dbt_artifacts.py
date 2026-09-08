import json

import pytest

from medic.tools.dbt_artifacts import (
    MAX_DEPTH,
    Manifest,
    NodeNotFound,
    make_dbt_artifact_tools,
    summarize_run_results,
)

PKG = "marketing_attribution"


def test_run_results_lists_failures_models_first(run_results_path):
    summary = summarize_run_results(run_results_path)
    assert summary.total == 7
    assert summary.status_counts == {"success": 4, "pass": 1, "error": 1, "skipped": 1}
    assert [f.name for f in summary.failing] == ["dim_channels"]
    assert summary.failing[0].resource_type == "model"
    assert "Binder Error" in summary.failing[0].message
    assert summary.skipped == [f"test.{PKG}.not_null_dim_channels_channel.abc123"]
    assert summary.invocation.startswith("dbt build")


def test_upstream_lineage_by_bare_name(manifest_path):
    m = Manifest.load(manifest_path)
    lin = m.lineage("dim_channels", "upstream", depth=2)
    names = {n.name: n.depth for n in lin.nodes}
    assert names["dim_channels"] == 0
    assert names["int_touchpoints_sessionized"] == 1
    assert names["int_journeys"] == 1
    assert names["stg_touchpoints"] == 2
    assert names["stg_conversions"] == 2
    # depth 2 stops before the sources
    assert "touchpoints" not in names
    assert (f"model.{PKG}.int_journeys", f"model.{PKG}.dim_channels") in lin.edges
    assert not lin.truncated


def test_upstream_reaches_sources_at_depth_three(manifest_path):
    lin = Manifest.load(manifest_path).lineage("dim_channels", "upstream", depth=3)
    sources = [n for n in lin.nodes if n.resource_type == "source"]
    assert {s.name for s in sources} == {"touchpoints", "conversions"}


def test_depth_is_capped(manifest_path):
    lin = Manifest.load(manifest_path).lineage("dim_channels", "upstream", depth=99)
    assert lin.depth == MAX_DEPTH


def test_downstream_excludes_tests_by_default(manifest_path):
    m = Manifest.load(manifest_path)
    lin = m.lineage("stg_touchpoints", "downstream", depth=1)
    assert [n.name for n in lin.nodes if n.depth == 1] == ["int_touchpoints_sessionized"]
    with_tests = m.lineage("stg_touchpoints", "downstream", depth=1, include_tests=True)
    assert any(n.resource_type == "test" for n in with_tests.nodes)


def test_source_addressed_as_source_dot_table(manifest_path):
    m = Manifest.load(manifest_path)
    assert m.resolve("raw.touchpoints") == f"source.{PKG}.raw.touchpoints"
    assert m.resolve("touchpoints") == f"source.{PKG}.raw.touchpoints"


def test_unknown_node_raises(manifest_path):
    with pytest.raises(NodeNotFound):
        Manifest.load(manifest_path).lineage("fct_nothing")


def test_tools_return_json_strings(manifest_path, run_results_path):
    get_run_results, get_lineage = make_dbt_artifact_tools(manifest_path, run_results_path)
    assert get_run_results.name == "get_run_results"
    assert get_lineage.name == "get_lineage"

    run = json.loads(get_run_results.invoke({}))
    assert run["failing"][0]["unique_id"] == f"model.{PKG}.dim_channels"

    lin = json.loads(
        get_lineage.invoke({"node": "dim_channels", "direction": "upstream", "depth": 1})
    )
    assert lin["node"] == f"model.{PKG}.dim_channels"
    assert {n["name"] for n in lin["nodes"] if n["depth"] == 1} == {
        "int_touchpoints_sessionized",
        "int_journeys",
    }

    missing = json.loads(get_lineage.invoke({"node": "nope"}))
    assert "error" in missing
