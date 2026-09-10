"""read_pipeline_file and list_pipeline_files: the sandbox worktree, nothing outside it."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import BaseTool, tool

MAX_CHARS = 20_000
ALLOWED_SUFFIXES = frozenset(
    {".sql", ".yml", ".yaml", ".md", ".txt", ".py", ".toml", ".ps1", ".cfg"}
)
SKIP_DIRS = frozenset({".git", ".venv", "target", "logs", "__pycache__", "dbt_packages"})
DBT_SUBDIR = "dbt_marketing"


class PathRejected(ValueError):
    pass


def resolve_inside(repo: Path, path: str) -> Path:
    """Resolve `path` against the worktree, then against its dbt project directory.

    dbt's manifest reports paths relative to the project (models/...), while
    the worktree root is one level up, so both spellings are accepted, with
    either slash. Anything that resolves outside the worktree, or into .git,
    is refused.
    """
    root = repo.resolve()
    raw = Path(path.replace("\\", "/"))
    if raw.is_absolute():
        raise PathRejected("use a path relative to the repository root")
    for candidate in (root / raw, root / DBT_SUBDIR / raw):
        target = candidate.resolve()
        if target != root and root not in target.parents:
            raise PathRejected(f"{path!r} resolves outside the sandbox worktree")
        if ".git" in target.relative_to(root).parts:
            raise PathRejected("the .git directory is not readable")
        if target.is_file():
            return target
    raise FileNotFoundError(path)


def read_file(repo: Path, path: str, max_chars: int = MAX_CHARS) -> str:
    target = resolve_inside(repo, path)
    if target.suffix.lower() not in ALLOWED_SUFFIXES:
        raise PathRejected(f"{target.suffix!r} files are not readable; text sources only")
    text = target.read_text(encoding="utf-8", errors="replace")
    rel = target.relative_to(repo.resolve()).as_posix()
    lines = text.count("\n") + (0 if text.endswith("\n") else 1)
    header = f"# {rel} ({lines} lines)\n"
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n...[truncated at {max_chars} chars]"
    return header + text


def list_files(repo: Path, subdir: str = "", max_entries: int = 200) -> list[str]:
    root = repo.resolve()
    base = (root / subdir.replace("\\", "/")).resolve() if subdir else root
    if base != root and root not in base.parents:
        raise PathRejected(f"{subdir!r} resolves outside the sandbox worktree")
    if not base.is_dir():
        raise FileNotFoundError(subdir)
    out: list[str] = []
    for p in sorted(base.rglob("*")):
        rel = p.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES:
            out.append(rel.as_posix())
        if len(out) >= max_entries:
            break
    return out


def make_file_tools(repo: Path) -> list[BaseTool]:
    @tool
    def read_pipeline_file(path: str) -> str:
        """Read one text file from the sandbox checkout of the pipeline: model SQL,
        schema and source yml, singular tests, macros, dbt_project.yml. `path`
        is relative to the repository root (dbt_marketing/models/staging/
        stg_touchpoints.sql) or to the dbt project (models/staging/
        stg_touchpoints.sql); both work. Paths outside the checkout are refused."""
        try:
            return read_file(repo, path)
        except FileNotFoundError:
            return json.dumps({"error": f"no file at {path!r}; try list_pipeline_files"})
        except PathRejected as exc:
            return json.dumps({"error": str(exc)})

    @tool
    def list_pipeline_files(subdir: str = "dbt_marketing") -> str:
        """List the text files under a directory of the sandbox checkout, relative
        to the repository root. Defaults to the dbt project."""
        try:
            return json.dumps(list_files(repo, subdir))
        except FileNotFoundError:
            return json.dumps({"error": f"no directory {subdir!r}"})
        except PathRejected as exc:
            return json.dumps({"error": str(exc)})

    return [read_pipeline_file, list_pipeline_files]
