"""medic command line: `spike` (Phase 0) and `chaos` (Phase 1a)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from medic import tracing
from medic.config import LLM, SANDBOX_ROOT, TARGET

app = typer.Typer(
    help="PipelineMedic: agentic incident triage for dbt + Dagster.", no_args_is_help=True
)
chaos_app = typer.Typer(
    help="Chaos suite: inject known failures into a sandbox.", no_args_is_help=True
)
app.add_typer(chaos_app, name="chaos")


@app.callback()
def main() -> None:
    """PipelineMedic."""


# --------------------------------------------------------------------------- #
# spike
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# chaos
# --------------------------------------------------------------------------- #


def _parse_selection(scenario: str) -> list[int]:
    from chaos.scenarios import load_scenarios

    numbers = sorted(load_scenarios())
    if scenario.lower() == "all":
        return numbers
    try:
        picked = [int(part) for part in scenario.split(",")]
    except ValueError as exc:
        raise typer.BadParameter(
            f"expected a number, a comma list, or 'all'; got {scenario!r}"
        ) from exc
    unknown = [n for n in picked if n not in numbers]
    if unknown:
        raise typer.BadParameter(f"no scenario {unknown}; have {numbers}")
    return picked


@chaos_app.command("list")
def chaos_list() -> None:
    """Show the seven scenarios and what each is expected to break."""
    from chaos.runner import incident_path
    from chaos.scenarios import load_scenarios

    for n, s in load_scenarios().items():
        record = incident_path(s.key)
        state = "not run"
        if record.exists():
            data = json.loads(record.read_text(encoding="utf-8"))
            state = ("reproduced" if data.get("matched") else "MISMATCH") + (
                "" if data.get("torn_down") else ", sandbox kept"
            )
        typer.echo(f"{n}  {s.key:<16} {s.category:<17} {s.fix_kind:<21} {state}")
        typer.echo(f"   {s.mutation}")
        typer.echo(
            f"   dbt {' '.join(s.dbt_args)} -> {', '.join(s.expected_failing)} ({s.error_contains!r})"
        )


@chaos_app.command("run")
def chaos_run(
    scenario: Annotated[str, typer.Argument(help="Scenario number, comma list, or 'all'")],
    teardown: Annotated[
        bool, typer.Option("--teardown", help="Remove the sandbox after recording the incident")
    ] = False,
    dbt_timeout: Annotated[int, typer.Option(help="Seconds to allow the dbt command")] = 600,
) -> None:
    """Create a sandbox, apply the mutation, run dbt, record the incident, check the outcome."""
    from chaos.runner import run_scenario
    from chaos.scenarios import load_scenarios

    scenarios = load_scenarios()
    failed: list[int] = []
    for n in _parse_selection(scenario):
        record = run_scenario(
            scenarios[n], teardown=teardown, dbt_timeout=dbt_timeout, log=typer.echo
        )
        if not (record.matched and record.production_untouched):
            failed.append(n)
    if failed:
        typer.echo(f"scenarios not reproduced as expected: {failed}")
        raise typer.Exit(code=1)


@chaos_app.command("teardown")
def chaos_teardown(
    scenario: Annotated[str, typer.Argument(help="Scenario number, comma list, or 'all'")],
) -> None:
    """Remove a scenario's worktree, branch, and warehouse copy. The incident record stays."""
    from chaos.harness import Sandbox
    from chaos.runner import IncidentRecord, incident_path
    from chaos.scenarios import load_scenarios

    scenarios = load_scenarios()
    for n in _parse_selection(scenario):
        sb = Sandbox.at(scenarios[n].key)
        existed = sb.exists()
        sb.teardown()
        record = incident_path(sb.name)
        if record.exists():
            data = IncidentRecord.read(record)
            data.torn_down = True
            data.write(record)
        typer.echo(f"{n}  {sb.name}: {'removed' if existed else 'nothing to remove'} ({sb.root})")
    typer.echo(f"sandbox root: {SANDBOX_ROOT}")
