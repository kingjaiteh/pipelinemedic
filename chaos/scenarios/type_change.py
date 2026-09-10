"""Scenario 4: cost arrives as text, and a handful of rows are not numbers."""

from chaos.mutation import MutationContext

BAD_VALUE = "n/a"
# One row in this many gets the bad value: about 34 of 170k in the real data.
BAD_ONE_IN = 5000


def apply(ctx: MutationContext) -> None:
    ctx.require_column("raw", "touchpoints", "cost")
    cols = ctx.columns("raw", "touchpoints")
    select = ", ".join("cast(cost as varchar) as cost" if c == "cost" else c for c in cols)
    ctx.con.execute(
        f"create or replace table raw.touchpoints as select {select} from raw.touchpoints"
    )
    ctx.con.execute(
        "update raw.touchpoints set cost = ? where hash(user_id, touch_number) % ? = 0",
        [BAD_VALUE, BAD_ONE_IN],
    )
