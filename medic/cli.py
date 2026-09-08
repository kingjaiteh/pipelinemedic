"""medic command line. Phase 0 ships `medic spike`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from medic import tracing
from medic.config import LLM, TARGET

app = typer.Typer(
    help="PipelineMedic: agentic incident triage for dbt + Dagster.", no_args_is_help=True
)


@app.callback()
def main() -> None:
    """Keep `medic <command>` as the shape even while there is one command."""


@app.command()
def spike(
    run_results: Annotated[
        Path,
        typer.Option(
            "--run-results",
            exists=True,
            dir_okay=False,
            help="run_results.json from a failed build",
        ),
    ],
    manifest: Annotated[
        Path | None,
        typer.Option(
            "--manifest",
            exists=True,
            dir_okay=False,
            help="manifest.json for the same project. Defaults to the target pipeline's.",
        ),
    ] = None,
    max_steps: Annotated[int, typer.Option(help="Model turns before giving up")] = 8,
) -> None:
    """One tool-calling loop: name the failing model and what it reads from."""
    from medic.agents.spike import run_spike
    from medic.llm import make_chat_model
    from medic.tools.dbt_artifacts import make_dbt_artifact_tools

    manifest_path = manifest or TARGET.manifest
    tools = make_dbt_artifact_tools(manifest_path, run_results)
    model = make_chat_model()
    callbacks = tracing.make_callbacks()

    typer.echo(f"provider={LLM.provider} model={LLM.model} tracing={'on' if callbacks else 'off'}")
    with tracing.trace("spike", run_results=str(run_results)) as handle:
        result = run_spike(model, tools, max_steps=max_steps, callbacks=callbacks)

    for out in result.tool_outputs:
        typer.echo(f"-> {out['tool']}({json.dumps(out['args'])}) [{len(out['output'])} chars]")

    if result.report is None:
        typer.echo("no report submitted" + (" (step budget exhausted)" if result.exhausted else ""))
        if result.final_text:
            typer.echo(result.final_text)
        raise typer.Exit(code=1)

    typer.echo(json.dumps(result.report.model_dump(), indent=2))
    typer.echo(f"steps={result.steps} tool_calls={result.tool_calls}")
    if handle.url:
        typer.echo(f"trace: {handle.url}")
