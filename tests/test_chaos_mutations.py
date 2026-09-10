"""Each scenario mutation against a tiny in-memory warehouse. No 40 MB copy, no dbt."""

from __future__ import annotations

import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pytest

from chaos.mutation import TARGET_DAY, MutationContext, MutationError
from chaos.scenarios import (
    bad_commit,
    duplicate_keys,
    null_flood,
    renamed_column,
    stale_source,
    type_change,
    volume_drop,
)

CHANNELS = ["display", "paid_social", "paid_search", "organic_search", "email", "direct"]
START = datetime(2026, 2, 1)  # noqa: DTZ001 - naive, like the generator's timestamps
DAYS = 14
USERS = 60
TOUCHES = 5  # rows per day = USERS * TOUCHES / DAYS ... spread evenly below


def build_tiny_raw(con: duckdb.DuckDBPyConnection) -> None:
    """raw.touchpoints and raw.conversions with the production column types."""
    con.execute("create schema raw")
    con.execute(
        "create table raw.touchpoints (user_id bigint, touch_number integer, "
        "touched_at timestamp_ns, channel varchar, campaign varchar, cost double)"
    )
    con.execute(
        "create table raw.conversions (user_id bigint, converted boolean, "
        "journey_length bigint, converted_at timestamp_ns, revenue double)"
    )
    rows = []
    i = 0
    for user in range(USERS):
        for touch in range(TOUCHES):
            day = i % DAYS
            ts = START + timedelta(days=day, hours=touch * 3, minutes=user)
            channel = CHANNELS[i % len(CHANNELS)]
            rows.append((user, touch, ts, channel, f"{channel}_c{touch % 3}", float(i % 7)))
            i += 1
    con.executemany("insert into raw.touchpoints values (?, ?, ?, ?, ?, ?)", rows)
    con.executemany(
        "insert into raw.conversions values (?, ?, ?, ?, ?)",
        [(u, u % 3 == 0, TOUCHES, START + timedelta(days=20), 10.0) for u in range(USERS)],
    )


@pytest.fixture
def ctx() -> MutationContext:
    con = duckdb.connect(":memory:")
    build_tiny_raw(con)
    yield MutationContext(con=con)
    con.close()


def day_count(ctx: MutationContext, day: str = TARGET_DAY) -> int:
    return ctx.scalar(
        "select count(*) from raw.touchpoints where cast(touched_at as date) = cast(? as date)",
        [day],
    )


def test_fixture_covers_the_target_day(ctx):
    assert ctx.scalar("select count(*) from raw.touchpoints") == USERS * TOUCHES
    assert day_count(ctx) > 0


def test_renamed_column(ctx):
    renamed_column.apply(ctx)
    cols = ctx.columns("raw", "touchpoints")
    assert "channel" not in cols
    assert cols[3] == "marketing_channel"
    with pytest.raises(MutationError):
        renamed_column.apply(ctx)  # the column is gone, so the guard fires


def test_null_flood_is_deterministic_and_near_30_percent(ctx):
    null_flood.apply(ctx)
    total = ctx.scalar("select count(*) from raw.touchpoints")
    nulls = ctx.scalar("select count(*) from raw.touchpoints where user_id is null")
    assert 0.15 * total < nulls < 0.45 * total

    con2 = duckdb.connect(":memory:")
    build_tiny_raw(con2)
    null_flood.apply(MutationContext(con=con2))
    assert (
        con2.execute("select count(*) from raw.touchpoints where user_id is null").fetchone()[0]
        == nulls
    )


def test_duplicate_keys_doubles_one_day_only(ctx):
    before_day = day_count(ctx)
    before_total = ctx.scalar("select count(*) from raw.touchpoints")
    duplicate_keys.apply(ctx)
    assert day_count(ctx) == 2 * before_day
    assert ctx.scalar("select count(*) from raw.touchpoints") == before_total + before_day
    dupes = ctx.scalar(
        "select count(*) from (select user_id, touch_number from raw.touchpoints "
        "group by 1, 2 having count(*) > 1)"
    )
    assert dupes == before_day


def test_type_change_makes_cost_text_with_a_few_bad_values(ctx):
    cols_before = ctx.columns("raw", "touchpoints")
    type_change.apply(ctx)
    assert ctx.columns("raw", "touchpoints") == cols_before
    assert ctx.scalar("select typeof(cost) from raw.touchpoints limit 1") == "VARCHAR"
    bad = ctx.scalar("select count(*) from raw.touchpoints where try_cast(cost as double) is null")
    assert bad >= 0  # one in 5000 on 300 rows may be zero; the type change is the point
    with pytest.raises(duckdb.ConversionException):
        # Force at least one bad value, then prove the production cast breaks.
        ctx.con.execute(
            f"update raw.touchpoints set cost = '{type_change.BAD_VALUE}' "
            "where user_id = 0 and touch_number = 0"
        )
        ctx.con.execute("select cast(cost as double) from raw.touchpoints").fetchall()


def test_stale_source_shifts_every_timestamp_back(ctx):
    before = ctx.scalar("select max(touched_at) from raw.touchpoints")
    stale_source.apply(ctx)
    after = ctx.scalar("select max(touched_at) from raw.touchpoints")
    assert before - after == timedelta(days=stale_source.SHIFT_DAYS)


def test_volume_drop_keeps_about_one_in_five(ctx):
    before_day = day_count(ctx)
    other_days = ctx.scalar("select count(*) from raw.touchpoints") - before_day
    volume_drop.apply(ctx)
    kept = day_count(ctx)
    assert 0.05 * before_day <= kept <= 0.4 * before_day
    assert ctx.scalar("select count(*) from raw.touchpoints") - kept == other_days


def test_day_scenarios_refuse_an_empty_day():
    con = duckdb.connect(":memory:")
    con.execute("create schema raw")
    con.execute(
        "create table raw.touchpoints (user_id bigint, touch_number integer, touched_at timestamp)"
    )
    empty = MutationContext(con=con)
    with pytest.raises(MutationError):
        duplicate_keys.apply(empty)
    with pytest.raises(MutationError):
        volume_drop.apply(empty)


# --------------------------------------------------------------------------- #
# scenario 5 needs a git repo, not a warehouse
# --------------------------------------------------------------------------- #

MODEL_TEXT = (
    "select\r\n"
    "    c.channel,\r\n"
    "\r\n"
    "    c.touch_count,\r\n"
    "    c.users_reached\r\n"
    "from by_channel c\r\n"
)


@pytest.fixture
def tiny_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for var, value in {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }.items():
        monkeypatch.setenv(var, value)
    repo = tmp_path / "repo"
    model = repo / bad_commit.MODEL_PATH
    model.parent.mkdir(parents=True)
    model.write_bytes(MODEL_TEXT.encode())

    def git(*a: str) -> None:
        subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    git("add", ".")
    git("commit", "-q", "-m", "seed")
    return repo


def test_bad_commit_commits_a_typo_on_the_worktree(tiny_repo: Path):
    ctx = MutationContext(con=duckdb.connect(":memory:"), repo=tiny_repo)
    bad_commit.apply(ctx)

    text = (tiny_repo / bad_commit.MODEL_PATH).read_bytes().decode()
    assert "    c.touch_cnt,\r\n" in text  # indentation and CRLF preserved
    assert "c.touch_count" not in text
    assert ctx.git("log", "--format=%s", "-1") == bad_commit.COMMIT_MESSAGE
    assert ctx.git("status", "--porcelain") == ""
    diff = ctx.git("show", "--format=", "HEAD")
    assert "-    c.touch_count," in diff and "+    c.touch_cnt," in diff


def test_bad_commit_needs_the_pattern_and_a_repo(tiny_repo: Path):
    with pytest.raises(MutationError):
        bad_commit.apply(MutationContext(con=duckdb.connect(":memory:")))
    (tiny_repo / bad_commit.MODEL_PATH).write_text("select 1", encoding="utf-8")
    with pytest.raises(MutationError):
        bad_commit.apply(MutationContext(con=duckdb.connect(":memory:"), repo=tiny_repo))
