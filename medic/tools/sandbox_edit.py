"""Apply the fixer's edits inside the sandbox worktree, and refuse the ones that hide problems.

An edit is an exact text replacement: `find` must occur once in the file's
current text. The model never writes a diff; the diff is generated here for
the record and the pull request. All edits of one proposal are checked and
then written together, so a bad one leaves the worktree untouched.

The guard has three layers. Paths must resolve inside the dbt project of the
worktree (never profiles.yml, never target/ or logs/). Test definitions
(tests/, macros/, dbt_project.yml) are refused unless the proposal declared
itself a test_change. A yml edit that removes a test, moves it, or weakens
one (severity, warn_if, error_if, where, enabled: false) is refused on the
same condition, because that is how a fixer papers over a data problem.
"""

from __future__ import annotations

import difflib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from medic.graph.state import ApplyResult, EditOutcome, EditSpec
from medic.tools.pipeline_files import ALLOWED_SUFFIXES, DBT_SUBDIR

MAX_FILE_CHARS = 200_000
# Editing any of these is a change to what the pipeline checks, not a fix.
PROTECTED_DIRS = ("tests", "macros")
PROTECTED_FILES = ("dbt_project.yml",)
NEVER_EDIT = ("profiles.yml",)
SKIP_DIRS = ("target", "logs", "dbt_packages", ".git")
TEST_KEYS = ("tests", "data_tests", "unit_tests")
WEAKENING_KEYS = ("severity", "warn_if", "error_if", "where", "enabled", "fail_calc", "limit")


class EditRejected(ValueError):
    pass


def resolve_editable(repo: Path, path: str) -> Path:
    """Resolve `path` inside the worktree's dbt project. The file may not exist yet."""
    root = repo.resolve()
    raw = Path(path.replace("\\", "/"))
    if raw.is_absolute():
        raise EditRejected("use a path relative to the repository root")
    parts = raw.parts
    if parts and parts[0] != DBT_SUBDIR:
        raw = Path(DBT_SUBDIR) / raw
    target = (root / raw).resolve()
    if root not in target.parents:
        raise EditRejected(f"{path!r} resolves outside the sandbox worktree")
    rel = target.relative_to(root)
    if rel.parts[0] != DBT_SUBDIR:
        raise EditRejected(f"edits are confined to {DBT_SUBDIR}/; {path!r} is outside it")
    inner = rel.parts[1:]
    if not inner:
        raise EditRejected("path names the project directory, not a file")
    if inner[0] in SKIP_DIRS:
        raise EditRejected(f"{inner[0]}/ is generated output, not source")
    if target.name in NEVER_EDIT:
        raise EditRejected(f"{target.name} points the pipeline at a database; never edited")
    if target.suffix.lower() not in ALLOWED_SUFFIXES:
        raise EditRejected(f"{target.suffix!r} files are not editable; text sources only")
    return target


def is_test_definition(repo: Path, target: Path) -> bool:
    inner = target.resolve().relative_to(repo.resolve() / DBT_SUBDIR).parts
    return inner[0] in PROTECTED_DIRS or (len(inner) == 1 and inner[0] in PROTECTED_FILES)


# ----------------------------------------------------------------- yml tests
def _walk_tests(node, where: str, out: list[str]) -> None:
    """Every test declaration under `node`, tagged with the model or column it hangs on."""
    if isinstance(node, dict):
        name = node.get("name")
        here = f"{where}/{name}" if isinstance(name, str) else where
        for key in TEST_KEYS:
            tests = node.get(key)
            if isinstance(tests, list):
                for t in tests:
                    out.append(f"{here}: {json.dumps(t, sort_keys=True, default=str)}")
        fresh = node.get("freshness")
        if isinstance(fresh, dict):
            out.append(f"{here}: freshness {json.dumps(fresh, sort_keys=True, default=str)}")
        for key, value in node.items():
            if key in TEST_KEYS:
                continue
            _walk_tests(value, here, out)
    elif isinstance(node, list):
        for item in node:
            _walk_tests(item, where, out)


def collect_tests(text: str) -> list[str]:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise EditRejected(f"the edited yml does not parse: {exc}") from exc
    out: list[str] = []
    _walk_tests(data, "", out)
    return out


def _weakening_count(text: str) -> int:
    lowered = text.lower()
    return sum(lowered.count(f"{k}:") for k in WEAKENING_KEYS)


def tests_weakened(before: str, after: str) -> str | None:
    """Why the yml change weakens what the pipeline checks, or None."""
    removed = Counter(collect_tests(before)) - Counter(collect_tests(after))
    if removed:
        gone = "; ".join(sorted(removed))
        return f"removes or moves test declarations: {gone}"
    if _weakening_count(after) > _weakening_count(before):
        return "adds a severity, where, enabled, warn_if, error_if, limit or fail_calc setting"
    return None


# --------------------------------------------------------------------- apply
@dataclass
class _Pending:
    path: Path
    rel: str
    before: str
    after: str


def _replace_once(text: str, find: str, replace: str, rel: str) -> str:
    if not find:
        raise EditRejected(
            f"{rel}: empty `find` on an existing file; give the exact text to replace"
        )
    n = text.count(find)
    if n == 0:
        raise EditRejected(
            f"{rel}: `find` text not found; it must match the current file exactly, "
            "including indentation and line breaks"
        )
    if n > 1:
        raise EditRejected(f"{rel}: `find` text occurs {n} times; include more context")
    return text.replace(find, replace, 1)


def unified_diff(rel: str, before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
        )
    )


def check_edit(repo: Path, edit: EditSpec, kind: str, current: dict[Path, str]) -> _Pending:
    target = resolve_editable(repo, edit.path)
    rel = target.relative_to(repo.resolve()).as_posix()
    if is_test_definition(repo, target) and kind != "test_change":
        raise EditRejected(
            f"{rel} defines what the pipeline checks; only a test_change proposal may edit it"
        )
    if target in current:
        before = current[target]
    elif target.exists():
        before = target.read_text(encoding="utf-8")
        if len(before) > MAX_FILE_CHARS:
            raise EditRejected(f"{rel} is too large to edit ({len(before)} chars)")
    else:
        if edit.find:
            raise EditRejected(f"{rel} does not exist; use an empty `find` to create it")
        before = None
    after = edit.replace if before is None else _replace_once(before, edit.find, edit.replace, rel)
    if target.suffix.lower() in (".yml", ".yaml") and kind != "test_change":
        reason = tests_weakened(before or "", after)
        if reason:
            raise EditRejected(f"{rel}: {reason}; only a test_change proposal may do that")
    return _Pending(target, rel, before or "", after)


def apply_edits(repo: Path, edits: list[EditSpec], kind: str) -> ApplyResult:
    """Check every edit against the current text, then write them all, or none."""
    outcomes: list[EditOutcome] = []
    pending: list[_Pending] = []
    current: dict[Path, str] = {}
    ok = True
    for edit in edits:
        try:
            p = check_edit(repo, edit, kind, current)
        except EditRejected as exc:
            outcomes.append(EditOutcome(path=edit.path, ok=False, error=str(exc)))
            ok = False
            continue
        current[p.path] = p.after
        pending.append(p)
        outcomes.append(EditOutcome(path=p.rel, ok=True))
    if not ok or not pending:
        if ok:
            outcomes.append(EditOutcome(path="", ok=False, error="no edits given"))
        return ApplyResult(ok=False, outcomes=outcomes)
    # Diff per file against the text on disk, not against intermediate edits.
    diffs: list[str] = []
    final: dict[Path, tuple[str, str]] = {}
    for p in pending:
        on_disk = final.get(p.path, (p.before, ""))[0]
        final[p.path] = (on_disk, p.after)
    for path, (before, after) in final.items():
        rel = path.relative_to(repo.resolve()).as_posix()
        diffs.append(unified_diff(rel, before, after))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(after, encoding="utf-8")
    return ApplyResult(
        ok=True,
        outcomes=outcomes,
        diff="".join(diffs),
        edited_paths=[path.relative_to(repo.resolve()).as_posix() for path in final],
    )
