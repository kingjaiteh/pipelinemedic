"""All read-only tools for one incident, bound to its sandbox."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool

from medic.graph.state import Incident
from medic.tools.dbt_artifacts import make_dbt_artifact_tools
from medic.tools.failing_rows import make_test_failures_tool
from medic.tools.freshness import make_freshness_tool
from medic.tools.git_history import make_git_tool
from medic.tools.pipeline_files import make_file_tools
from medic.tools.warehouse import make_query_tool

READ_ONLY_TOOL_NAMES = (
    "get_run_results",
    "get_lineage",
    "read_pipeline_file",
    "list_pipeline_files",
    "get_recent_changes",
    "query_duckdb",
    "get_test_failures",
    "get_source_freshness",
)


def make_incident_tools(incident: Incident) -> list[BaseTool]:
    repo = Path(incident.repo)
    tools = [
        *make_dbt_artifact_tools(Path(incident.manifest), Path(incident.artifact)),
        *make_file_tools(repo),
        make_git_tool(repo),
        make_query_tool(Path(incident.duckdb)),
        make_test_failures_tool(Path(incident.artifact), Path(incident.duckdb)),
        make_freshness_tool(Path(incident.sandbox_dir), incident.scenario_key or "incident"),
    ]
    assert tuple(t.name for t in tools) == READ_ONLY_TOOL_NAMES
    return tools
