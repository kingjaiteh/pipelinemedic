"""Read-only tools over dbt's target/ artifacts.

run_results.json says which nodes failed and why; manifest.json is the DAG.
Everything here is pure file parsing with no warehouse access, so the agent's
first two tools work against any dbt project and can be unit-tested against a
hand-built fixture.

Paths are bound when the tools are built (`make_dbt_artifact_tools`), not
passed by the model. The agent should not be choosing files on disk.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Literal

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

# Lineage answers stay small on purpose. The manifest holds the whole DAG and
# the model must never receive all of it in one tool result.
MAX_DEPTH = 3
MAX_NODES = 40
MAX_MESSAGE_CHARS = 800

FAILING_STATUSES = frozenset({"error", "fail", "runtime error"})

Direction = Literal["upstream", "downstream"]


class NodeResult(BaseModel):
    unique_id: str
    name: str
    resource_type: str
    status: str
    message: str | None = None
    execution_time: float = 0.0


class RunSummary(BaseModel):
    invocation: str | None = None
    generated_at: str | None = None
    total: int
    status_counts: dict[str, int]
    failing: list[NodeResult]
    skipped: list[str] = Field(default_factory=list)


class LineageNode(BaseModel):
    unique_id: str
    name: str
    resource_type: str
    depth: int
    original_file_path: str | None = None


class Lineage(BaseModel):
    node: str
    direction: Direction
    depth: int
    nodes: list[LineageNode]
    edges: list[tuple[str, str]] = Field(
        default_factory=list, description="(parent, child) pairs within the subgraph"
    )
    truncated: bool = False


class NodeNotFound(LookupError):
    pass


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _clip(text: str | None, limit: int = MAX_MESSAGE_CHARS) -> str | None:
    if text is None:
        return None
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + " ...[truncated]"


def _name_and_type(unique_id: str) -> tuple[str, str]:
    parts = unique_id.split(".")
    return parts[-1], parts[0]


# --------------------------------------------------------------------------- #
# run_results.json
# --------------------------------------------------------------------------- #


def summarize_run_results(path: Path) -> RunSummary:
    """Failing nodes with their messages, plus status counts for the whole run."""
    data = _load(path)
    results = data.get("results", [])
    counts: dict[str, int] = {}
    failing: list[NodeResult] = []
    skipped: list[str] = []
    for r in results:
        status = r.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
        uid = r["unique_id"]
        name, rtype = _name_and_type(uid)
        if status in FAILING_STATUSES:
            failing.append(
                NodeResult(
                    unique_id=uid,
                    name=name,
                    resource_type=rtype,
                    status=status,
                    message=_clip(r.get("message")),
                    execution_time=float(r.get("execution_time") or 0.0),
                )
            )
        elif status == "skipped":
            skipped.append(uid)
    # Tests come after models in the manifest hash order, so sort failures
    # models-first: the model that broke is more useful than the tests it
    # cascaded into.
    failing.sort(key=lambda n: (n.resource_type != "model", n.unique_id))
    args = data.get("args", {})
    return RunSummary(
        invocation=args.get("invocation_command"),
        generated_at=data.get("metadata", {}).get("generated_at"),
        total=len(results),
        status_counts=counts,
        failing=failing,
        skipped=skipped,
    )


# --------------------------------------------------------------------------- #
# manifest.json
# --------------------------------------------------------------------------- #


class Manifest:
    """Thin index over manifest.json: node lookup by id or name, and both maps."""

    def __init__(self, data: dict):
        self.nodes: dict[str, dict] = {**data.get("nodes", {}), **data.get("sources", {})}
        self.parent_map: dict[str, list[str]] = data.get("parent_map", {})
        self.child_map: dict[str, list[str]] = data.get("child_map", {})

    @classmethod
    def load(cls, path: Path) -> Manifest:
        return cls(_load(path))

    def resolve(self, node: str) -> str:
        """Accept a unique_id or a bare name; models win over sources over tests."""
        if node in self.nodes:
            return node
        matches = [uid for uid, n in self.nodes.items() if n.get("name") == node]
        if not matches:
            # Sources are addressed as source_name.table_name in dbt.
            matches = [
                uid
                for uid, n in self.nodes.items()
                if n.get("resource_type") == "source"
                and f"{n.get('source_name')}.{n.get('name')}" == node
            ]
        if not matches:
            raise NodeNotFound(f"no node named {node!r} in the manifest")
        rank = {"model": 0, "source": 1, "seed": 2, "snapshot": 3}
        matches.sort(key=lambda uid: rank.get(self.nodes[uid].get("resource_type"), 9))
        return matches[0]

    def lineage(
        self,
        node: str,
        direction: Direction = "upstream",
        depth: int = 2,
        include_tests: bool = False,
    ) -> Lineage:
        """Bounded breadth-first walk. Never returns the whole DAG."""
        root = self.resolve(node)
        depth = max(1, min(int(depth), MAX_DEPTH))
        neighbours = self.parent_map if direction == "upstream" else self.child_map

        seen: dict[str, int] = {root: 0}
        edges: list[tuple[str, str]] = []
        queue: deque[str] = deque([root])
        truncated = False
        while queue:
            current = queue.popleft()
            level = seen[current]
            if level >= depth:
                continue
            for nxt in neighbours.get(current, []):
                info = self.nodes.get(nxt, {})
                if not include_tests and info.get("resource_type") == "test":
                    continue
                edge = (nxt, current) if direction == "upstream" else (current, nxt)
                if edge not in edges:
                    edges.append(edge)
                if nxt in seen:
                    continue
                if len(seen) >= MAX_NODES:
                    truncated = True
                    continue
                seen[nxt] = level + 1
                queue.append(nxt)

        nodes = [
            LineageNode(
                unique_id=uid,
                name=self.nodes.get(uid, {}).get("name", _name_and_type(uid)[0]),
                resource_type=self.nodes.get(uid, {}).get("resource_type", _name_and_type(uid)[1]),
                depth=lvl,
                original_file_path=self.nodes.get(uid, {}).get("original_file_path"),
            )
            for uid, lvl in sorted(seen.items(), key=lambda kv: (kv[1], kv[0]))
        ]
        return Lineage(
            node=root,
            direction=direction,
            depth=depth,
            nodes=nodes,
            edges=edges,
            truncated=truncated,
        )


# --------------------------------------------------------------------------- #
# LangChain tool wrappers
# --------------------------------------------------------------------------- #


def make_dbt_artifact_tools(manifest_path: Path, run_results_path: Path) -> list[BaseTool]:
    """Build the two Phase 0 tools with their file paths bound in.

    The manifest is loaded lazily on first use so building the tool list never
    touches disk (tests and the CLI both construct tools before deciding to run).
    """
    manifest_holder: dict[str, Manifest] = {}

    def manifest() -> Manifest:
        if "m" not in manifest_holder:
            manifest_holder["m"] = Manifest.load(manifest_path)
        return manifest_holder["m"]

    @tool
    def get_run_results() -> str:
        """Summarize the latest dbt run: which nodes failed, their error messages,
        which were skipped as a consequence, and status counts. Call this first."""
        summary = summarize_run_results(run_results_path)
        return summary.model_dump_json(indent=None)

    @tool
    def get_lineage(node: str, direction: Direction = "upstream", depth: int = 2) -> str:
        """Walk the dbt DAG from one node. `node` is a model or source name (for
        example "dim_channels" or "raw.touchpoints") or a full unique_id.
        `direction` is "upstream" (what it reads from) or "downstream" (what reads
        it). `depth` is capped at 3. Tests are excluded. Returns the nodes reached
        and the edges between them, never the whole graph."""
        try:
            return manifest().lineage(node, direction, depth).model_dump_json(indent=None)
        except NodeNotFound as exc:
            return json.dumps({"error": str(exc)})

    return [get_run_results, get_lineage]
