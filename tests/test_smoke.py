"""Phase 0 precondition: the agent can see the pipeline it is meant to fix."""

from medic.config import TARGET


def test_target_repo_exists():
    assert TARGET.repo.is_dir(), f"pipeline repo not found at {TARGET.repo}"


def test_dbt_artifacts_present():
    assert TARGET.manifest.is_file(), "run the pipeline once to produce target/manifest.json"
    assert TARGET.run_results.is_file()


def test_duckdb_present():
    assert TARGET.duckdb.is_file()
