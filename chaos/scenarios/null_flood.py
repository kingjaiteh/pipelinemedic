"""Scenario 2: 30% of raw touchpoints arrive with no user_id."""

from chaos.mutation import MutationContext

# hash() is deterministic, so the same rows go null on every run.
NULL_SHARE_PCT = 30


def apply(ctx: MutationContext) -> None:
    ctx.require_column("raw", "touchpoints", "user_id")
    ctx.con.execute(
        "update raw.touchpoints set user_id = null where hash(user_id, touch_number) % 100 < ?",
        [NULL_SHARE_PCT],
    )
