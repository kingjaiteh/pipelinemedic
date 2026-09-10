"""Scenario 5: a committed typo in a marts model.

The only scenario that changes code rather than data. The commit lands on the
sandbox branch only, so `get_recent_changes` on the worktree can find it and
the fix is a revert.
"""

from __future__ import annotations

import re
from pathlib import Path

from chaos.mutation import MutationContext, MutationError

MODEL_PATH = Path("dbt_marketing/models/marts/dim_channels.sql")
# The final select's `c.touch_count,` line, whatever its indentation and line ending.
PATTERN = re.compile(r"^(?P<indent>[ \t]+)c\.touch_count,", re.MULTILINE)
REPLACEMENT = r"\g<indent>c.touch_cnt,"
COMMIT_MESSAGE = "Tidy the dim_channels column list"


def apply(ctx: MutationContext) -> None:
    if ctx.repo is None:
        raise MutationError("scenario 5 needs a worktree")
    path = ctx.repo / MODEL_PATH
    if not path.exists():
        raise MutationError(f"{MODEL_PATH} not found in the worktree")
    text = path.read_text(encoding="utf-8", newline="")
    new_text, count = PATTERN.subn(REPLACEMENT, text, count=1)
    if count != 1:
        raise MutationError(f"could not find `c.touch_count,` in {MODEL_PATH}")
    path.write_text(new_text, encoding="utf-8", newline="")
    ctx.git("add", MODEL_PATH.as_posix())
    ctx.git("commit", "--quiet", "--no-verify", "-m", COMMIT_MESSAGE)
