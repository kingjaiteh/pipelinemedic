"""get_recent_changes: `git log -p` on the sandbox worktree.

The worktree's branch carries the chaos commit (scenario 5) on top of the
pipeline's own history, so the investigator sees both the recent real
changes and anything the scenario planted.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from langchain_core.tools import BaseTool, tool

from medic.tools.pipeline_files import DBT_SUBDIR, PathRejected

MAX_CHARS = 8_000
FORMAT = "commit %h%nauthor %an%ndate %ad%nsubject %s%n"


def _inside(repo: Path, path: str) -> str:
    root = repo.resolve()
    if Path(path).is_absolute():
        raise PathRejected("use a path relative to the repository root")
    for candidate in (root / path, root / DBT_SUBDIR / path):
        target = candidate.resolve()
        if target != root and root not in target.parents:
            raise PathRejected(f"{path!r} resolves outside the sandbox worktree")
        if target.exists():
            return target.relative_to(root).as_posix()
    # A deleted file still has history; let git decide.
    return path.replace("\\", "/")


def recent_changes(
    repo: Path, path: str | None = None, n: int = 5, max_chars: int = MAX_CHARS
) -> str:
    n = max(1, min(int(n), 20))
    args = ["git", "-C", str(repo), "log", f"-n{n}", "--date=short", f"--format={FORMAT}", "-p"]
    if path:
        args += ["--", _inside(repo, path)]
    proc = subprocess.run(
        args, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git log exited {proc.returncode}")
    out = proc.stdout.strip()
    if not out:
        return f"no commits touch {path!r}" if path else "no commits"
    if len(out) > max_chars:
        out = out[:max_chars] + f"\n...[truncated at {max_chars} chars]"
    return out


def make_git_tool(repo: Path) -> BaseTool:
    @tool
    def get_recent_changes(path: str = "", n: int = 5) -> str:
        """Show the last `n` commits (default 5, max 20) with their diffs, from the
        sandbox checkout. Pass `path` (relative to the repository root, or to
        the dbt project) to restrict to commits touching one file or directory;
        leave it empty for the whole repository. The newest commit is first."""
        try:
            return recent_changes(repo, path or None, n)
        except (PathRejected, RuntimeError) as exc:
            return json.dumps({"error": str(exc)})

    return get_recent_changes
