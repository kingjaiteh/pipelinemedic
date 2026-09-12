"""What the fix phase does to the sandbox, behind one interface.

The graph calls these five operations and nothing else with side effects:
apply edits, run dbt, show the diff, reset the worktree, open the pull
request. `LiveSandboxOps` does them on the real sandbox; the replay
counterpart in `medic.transcript` answers from a recorded run.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Protocol

from medic.config import GITHUB, GitHubSettings
from medic.graph.state import (
    ApplyResult,
    EditSpec,
    Evidence,
    FixAttempt,
    Hypothesis,
    Incident,
    PullRequest,
    Validation,
)
from medic.tools.dbt_artifacts import Manifest
from medic.tools.github import pr_body, pr_title, push_and_open, write_dry_run
from medic.tools.sandbox_edit import apply_edits
from medic.tools.validate import selection_for, validate_in_sandbox

DBT_SUBDIR = "dbt_marketing"
# Files the fixer is shown before its first turn, at most.
MAX_PREREAD = 3
# A path the hypothesis text names: models/staging/stg_touchpoints.sql, _sources.yml.
PATH_IN_TEXT = re.compile(r"(?:[\w.-]+[/\\])*[\w.-]+\.(?:sql|yml|yaml)")


def _models_behind(manifest: Manifest, unique_id: str) -> list[str]:
    """The model itself, or the models a failing test depends on."""
    node = manifest.nodes.get(unique_id) or {}
    if node.get("resource_type") in ("model", "seed", "snapshot"):
        return [unique_id]
    return [
        p
        for p in node.get("depends_on", {}).get("nodes", [])
        if (manifest.nodes.get(p) or {}).get("resource_type") in ("model", "seed", "snapshot")
    ]


class SandboxOps(Protocol):
    def files_to_read(self, incident: Incident, hypotheses: list[Hypothesis]) -> list[str]: ...

    def apply(self, edits: list[EditSpec], kind: str) -> ApplyResult: ...

    def validate(self, edited_paths: list[str]) -> Validation: ...

    def diff(self) -> str: ...

    def reset(self) -> None: ...

    def open_pr(
        self,
        top: Hypothesis | None,
        evidence: list[Evidence],
        attempt: FixAttempt,
        validation: Validation | None,
        reviewer_note: str,
    ) -> PullRequest: ...


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    return proc.stdout.strip()


class LiveSandboxOps:
    def __init__(
        self,
        incident: Incident,
        *,
        pr_mode: str | None = None,
        pr_dir: Path | None = None,
        settings: GitHubSettings = GITHUB,
        dbt_timeout: int = 600,
    ):
        self.incident = incident
        self.repo = Path(incident.repo)
        self.sandbox_dir = Path(incident.sandbox_dir)
        self.name = incident.scenario_key or "incident"
        self.settings = settings
        self.pr_mode = pr_mode or settings.pr_mode
        self.pr_dir = pr_dir or self.sandbox_dir.parent / "incidents"
        self.dbt_timeout = dbt_timeout
        self._manifest: Manifest | None = None

    @property
    def manifest(self) -> Manifest:
        if self._manifest is None:
            self._manifest = Manifest.load(Path(self.incident.manifest))
        return self._manifest

    @property
    def branch(self) -> str:
        return _git(self.repo, "rev-parse", "--abbrev-ref", "HEAD") or f"medic/{self.name}"

    def files_to_read(self, incident: Incident, hypotheses: list[Hypothesis]) -> list[str]:
        """The failing models' files, then any file the top hypothesis names."""
        paths: list[str] = []
        for node in incident.failing_nodes:
            for uid in _models_behind(self.manifest, node.unique_id):
                rel = self.manifest.nodes[uid].get("original_file_path")
                if rel:
                    rel = f"{DBT_SUBDIR}/{rel.replace(chr(92), '/')}"
                    if rel not in paths:
                        paths.append(rel)
        if hypotheses:
            top = hypotheses[0]
            for m in PATH_IN_TEXT.finditer(f"{top.root_cause} {top.recommended_action}"):
                rel = m.group(0).replace("\\", "/")
                if not rel.startswith(f"{DBT_SUBDIR}/"):
                    rel = f"{DBT_SUBDIR}/{rel}"
                if rel not in paths and (self.repo / rel).is_file():
                    paths.append(rel)
        return paths[:MAX_PREREAD]

    def apply(self, edits: list[EditSpec], kind: str) -> ApplyResult:
        return apply_edits(self.repo, edits, kind)

    def validate(self, edited_paths: list[str]) -> Validation:
        args = selection_for(self.incident, self.manifest, edited_paths)
        return validate_in_sandbox(self.sandbox_dir, self.name, args, timeout=self.dbt_timeout)

    def diff(self) -> str:
        _git(self.repo, "add", "-N", "--", DBT_SUBDIR)
        return _git(self.repo, "diff", "--", DBT_SUBDIR)

    def reset(self) -> None:
        """Back to the branch head: drops every edit the fixer made."""
        _git(self.repo, "reset", "--hard", "--quiet")
        _git(self.repo, "clean", "-fdq", "--", DBT_SUBDIR)

    def open_pr(
        self,
        top: Hypothesis | None,
        evidence: list[Evidence],
        attempt: FixAttempt,
        validation: Validation | None,
        reviewer_note: str,
    ) -> PullRequest:
        title = pr_title(self.incident, attempt.proposal.explanation)
        body = pr_body(self.incident, top, evidence, attempt, validation, reviewer_note)
        branch = self.branch
        if self.pr_mode == "push":
            return push_and_open(self.repo, branch, title, body, self.settings)
        pr = PullRequest(mode="dry-run", title=title, branch=branch, body=body)
        path = write_dry_run(self.pr_dir / f"{self.name}.pr.md", pr)
        return pr.model_copy(update={"path": str(path)})
