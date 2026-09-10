"""Scenario 6: the latest load carries timestamps 30 days in the past."""

from chaos.mutation import MutationContext

SHIFT_DAYS = 30


def apply(ctx: MutationContext) -> None:
    ctx.require_column("raw", "touchpoints", "touched_at")
    ctx.con.execute(
        f"update raw.touchpoints set touched_at = touched_at - interval {SHIFT_DAYS} day"
    )
