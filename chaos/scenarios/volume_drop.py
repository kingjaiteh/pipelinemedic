"""Scenario 7: most of one day's touchpoints never arrived."""

from chaos.mutation import TARGET_DAY, MutationContext, MutationError

# Keep one row in five: an 80% drop against the trailing median.
KEEP_ONE_IN = 5


def apply(ctx: MutationContext) -> None:
    n = ctx.scalar(
        "select count(*) from raw.touchpoints where cast(touched_at as date) = cast(? as date)",
        [TARGET_DAY],
    )
    if not n:
        raise MutationError(f"no touchpoints on {TARGET_DAY}; nothing to drop")
    ctx.con.execute(
        "delete from raw.touchpoints where cast(touched_at as date) = cast(? as date) "
        "and hash(user_id, touch_number) % ? <> 0",
        [TARGET_DAY, KEEP_ONE_IN],
    )
