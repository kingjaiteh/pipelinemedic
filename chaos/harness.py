"""Sandbox: a DuckDB copy plus a git worktree of the pipeline, and dbt runs against them.

Nothing here opens the production database or writes to the pipeline's working
tree. The copy is made with a file copy, the worktree is created from the
pipeline's own git, and dbt is the pipeline's own executable so the version
matches what produced the artifacts.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb

from medic.config import SANDBOX_ROOT, TARGET, PipelineTarget
from medic.tools.dbt_artifacts import (
    FreshnessSummary,
    RunSummary,
    summarize_run_results,
    summarize_source_freshness,
)

# dbt's compiled SQL names the catalog after the file ("marketing".staging...),
# so a copy must keep this exact file name. Copies get their own directory.
DB_NAME = "marketing.duckdb"
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")
# Commands whose outcome lands in run_results.json.
RUN_RESULTS_COMMANDS = frozenset({"build", "run", "test", "seed", "snapshot", "compile"})


class SandboxError(RuntimeError):
    pass


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    if check and proc.returncode != 0:
        raise SandboxError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc


def _branch_exists(repo: Path, branch: str) -> bool:
    return (
        _git(
            repo, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}", check=False
        ).returncode
        == 0
    )


def _force_writable(func: Any, path: str, _exc: BaseException) -> None:
    os.chmod(path, stat.S_IWRITE)
    func(path)


@dataclass
class DbtRun:
    """One dbt invocation inside the sandbox and the artifact it wrote."""

    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float
    artifact: Path | None = None
    summary: RunSummary | FreshnessSummary | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def failing_ids(self) -> list[str]:
        return [] if self.summary is None else [n.unique_id for n in self.summary.failing]

    @property
    def failing_names(self) -> list[str]:
        return [] if self.summary is None else [n.name for n in self.summary.failing]

    def stdout_tail(self, lines: int = 30) -> str:
        return "\n".join(self.stdout.rstrip().splitlines()[-lines:])

    def to_dict(self) -> dict[str, Any]:
        return {
            "args": list(self.args),
            "returncode": self.returncode,
            "elapsed_s": round(self.elapsed_s, 1),
            "artifact": str(self.artifact) if self.artifact else None,
            "failing": [n.model_dump() for n in self.summary.failing] if self.summary else [],
            "status_counts": self.summary.status_counts if self.summary else {},
            "stdout_tail": self.stdout_tail(),
        }


@dataclass
class Sandbox:
    name: str
    root: Path
    target: PipelineTarget = field(default_factory=lambda: TARGET)

    # ------------------------------------------------------------------ paths
    @property
    def duckdb(self) -> Path:
        return self.root / DB_NAME

    @property
    def repo(self) -> Path:
        return self.root / "repo"

    @property
    def branch(self) -> str:
        return f"medic/{self.name}"

    @property
    def dbt_dir(self) -> Path:
        return self.repo / "dbt_marketing"

    @property
    def target_dir(self) -> Path:
        return self.dbt_dir / "target"

    @property
    def manifest(self) -> Path:
        return self.target_dir / "manifest.json"

    @property
    def run_results(self) -> Path:
        return self.target_dir / "run_results.json"

    @property
    def sources_json(self) -> Path:
        return self.target_dir / "sources.json"

    def exists(self) -> bool:
        return self.duckdb.exists() and self.repo.exists()

    # ---------------------------------------------------------- lifecycle
    @classmethod
    def at(
        cls, name: str, sandbox_root: Path = SANDBOX_ROOT, target: PipelineTarget = TARGET
    ) -> Sandbox:
        """Address a sandbox by name without creating anything."""
        if not NAME_RE.match(name):
            raise SandboxError(f"bad sandbox name {name!r}: use lowercase letters, digits, - or _")
        return cls(name=name, root=sandbox_root / name, target=target)

    @classmethod
    def create(
        cls,
        name: str,
        *,
        sandbox_root: Path = SANDBOX_ROOT,
        target: PipelineTarget = TARGET,
        base: str = "main",
        replace: bool = False,
    ) -> Sandbox:
        """Copy the DuckDB file and add a worktree on a fresh `medic/<name>` branch."""
        sb = cls.at(name, sandbox_root, target)
        if sb.root.exists() or _branch_exists(target.repo, sb.branch):
            if not replace:
                raise SandboxError(f"sandbox {name!r} already exists; tear it down or pass replace")
            sb.teardown()
        if not target.duckdb.exists():
            raise SandboxError(f"no warehouse at {target.duckdb}")
        if not (target.repo / ".git").exists():
            raise SandboxError(f"{target.repo} is not a git repository")

        sb.root.mkdir(parents=True)
        shutil.copy2(target.duckdb, sb.duckdb)
        wal = target.duckdb.with_name(target.duckdb.name + ".wal")
        if wal.exists():
            shutil.copy2(wal, sb.duckdb.with_name(sb.duckdb.name + ".wal"))
        _git(target.repo, "worktree", "add", str(sb.repo), "-b", sb.branch, base)
        return sb

    def teardown(self) -> None:
        """Remove the worktree, its branch, and the directory. Safe to call twice."""
        repo = self.target.repo
        if self.repo.exists():
            _git(repo, "worktree", "remove", "--force", str(self.repo), check=False)
        _git(repo, "worktree", "prune", check=False)
        if _branch_exists(repo, self.branch):
            _git(repo, "branch", "-D", self.branch, check=False)
        if self.root.exists():
            shutil.rmtree(self.root, onexc=_force_writable)

    # ---------------------------------------------------------------- use
    def connect(self, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
        """Open the copy. Close it before running dbt: DuckDB allows one writer process."""
        return duckdb.connect(str(self.duckdb), read_only=read_only)

    def git(self, *args: str) -> str:
        return _git(self.repo, *args).stdout.strip()

    def dbt_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["MARKETING_DUCKDB"] = str(self.duckdb)
        env["DBT_PROFILES_DIR"] = str(self.dbt_dir)
        env.pop("DBT_TARGET_PATH", None)
        return env

    def dbt_command(self, *args: str) -> list[str]:
        return [
            str(self.target.dbt_exe),
            *args,
            "--project-dir",
            str(self.dbt_dir),
            "--no-use-colors",
        ]

    def run_dbt(self, *args: str, timeout: int = 600) -> DbtRun:
        """Run the pipeline's dbt inside the worktree against the copy."""
        if not self.target.dbt_exe.exists():
            raise SandboxError(f"dbt not found at {self.target.dbt_exe}")
        if not self.exists():
            raise SandboxError(f"sandbox {self.name!r} is not set up")
        started = time.monotonic()
        proc = subprocess.run(
            self.dbt_command(*args),
            env=self.dbt_env(),
            cwd=str(self.dbt_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        elapsed = time.monotonic() - started
        artifact, summary = self._artifact_for(args)
        return DbtRun(
            tuple(args), proc.returncode, proc.stdout, proc.stderr, elapsed, artifact, summary
        )

    def _artifact_for(
        self, args: tuple[str, ...]
    ) -> tuple[Path | None, RunSummary | FreshnessSummary | None]:
        if args[:2] == ("source", "freshness"):
            path = self.sources_json
            return (path, summarize_source_freshness(path)) if path.exists() else (path, None)
        if args and args[0] in RUN_RESULTS_COMMANDS:
            path = self.run_results
            return (path, summarize_run_results(path)) if path.exists() else (path, None)
        return None, None
