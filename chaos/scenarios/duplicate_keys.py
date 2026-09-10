"""Scenario 3: one day of raw touchpoints is loaded twice."""

from chaos.mutation import TARGET_DAY, MutationContext, MutationError


def apply(ctx: MutationContext) -> None:
    n = ctx.scalar(
        "select count(*) from raw.touchpoints where cast(touched_at as date) = cast(? as date)",
        [TARGET_DAY],
    )
    if not n:
        raise MutationError(f"no touchpoints on {TARGET_DAY}; nothing to duplicate")
    ctx.con.execute(
        "insert into raw.touchpoints select * from raw.touchpoints "
        "where cast(touched_at as date) = cast(? as date)",
        [TARGET_DAY],
    )
