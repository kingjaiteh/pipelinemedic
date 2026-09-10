"""Scenario 1: a source column is renamed under the staging model."""

from chaos.mutation import MutationContext


def apply(ctx: MutationContext) -> None:
    ctx.require_column("raw", "touchpoints", "channel")
    ctx.con.execute("alter table raw.touchpoints rename column channel to marketing_channel")
