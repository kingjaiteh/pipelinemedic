"""medic command line: spike, chaos, triage, fix, resume, threads."""

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
# triage
# --------------------------------------------------------------------------- #


@app.command()
def triage(
    scenario: Annotated[
        int | None,
        typer.Option("--scenario", help="Chaos scenario number; needs `medic chaos run N` first"),
    ] = None,
    run_results: Annotated[
        Path | None,
        typer.Option(
            "--run-results", exists=True, dir_okay=False, help="run_results.json or sources.json"
        ),
    ] = None,
    sandbox: Annotated[
        Path | None,
        typer.Option(
            "--sandbox",
            exists=True,
            file_okay=False,
            help="Directory with marketing.duckdb and repo/",
        ),
    ] = None,
    manifest: Annotated[
        Path | None,
        typer.Option(
            "--manifest", exists=True, dir_okay=False, help="Defaults to next to the artifact"
        ),
    ] = None,
    record: Annotated[
        bool,
        typer.Option(
            "--record/--no-record", help="Save the transcript under tests/fixtures/<key>/"
        ),
    ] = True,
    max_tool_calls: Annotated[
        int | None, typer.Option(help="Override the tool-call budget")
    ] = None,
    model_name: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="Model to use for this run (the free tier quota is per model per day)",
        ),
    ] = None,
) -> None:
    """Investigate one incident: triage, tool loop, ranked hypotheses, critic."""
    from medic.config import LLMSettings
    from medic.graph.build import build_graph, initial_state
    from medic.graph.state import Budget
    from medic.incident import incident_from_paths, incident_from_record
    from medic.llm import make_chat_model
    from medic.report import TriageReport, report_path
    from medic.tools.registry import make_incident_tools
    from medic.transcript import Transcript, fixture_path

    settings = LLMSettings(model=model_name) if model_name else LLM

    ground_truth = None
    if scenario is not None:
        from chaos.runner import incident_path
        from chaos.scenarios import get_scenario

        spec = get_scenario(scenario)
        ground_truth = spec
        try:
            incident = incident_from_record(incident_path(spec.key))
        except FileNotFoundError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=2) from exc
    elif run_results is not None and sandbox is not None:
        incident = incident_from_paths(run_results, sandbox, manifest)
    else:
        raise typer.BadParameter("give --scenario N, or --run-results <path> with --sandbox <dir>")

    tools = make_incident_tools(incident)
    model = make_chat_model(settings)
    callbacks = tracing.make_callbacks()
    budget = Budget(max_tool_calls=max_tool_calls) if max_tool_calls else Budget()
    typer.echo(
        f"incident {incident.run_id}: {len(incident.failing_nodes)} failing node(s); "
        f"provider={settings.provider} model={settings.model} tracing={'on' if callbacks else 'off'}"
    )
    graph = build_graph(model, tools, budget=budget, callbacks=callbacks, log=typer.echo)
    with tracing.trace("triage", run_id=incident.run_id, model=settings.model) as handle:
        final = graph.invoke(initial_state(incident, budget), config={"recursion_limit": 120})

    report = TriageReport.from_state(final, model=settings.model)
    if ground_truth is not None:
        report.with_ground_truth(
            ground_truth.category, ground_truth.fix_kind, ground_truth.root_cause_terms
        )
    typer.echo("")
    typer.echo(report.render())

    key = incident.scenario_key or incident.run_id
    saved = report.save(report_path(SANDBOX_ROOT, key))
    typer.echo(f"\nreport {saved}")
    if record:
        path = Transcript.from_state(final, model=settings.model).save(fixture_path(key))
        typer.echo(f"transcript {path}")
    if handle.url:
        typer.echo(f"trace: {handle.url}")
    if report.status != "investigated" or not report.hypotheses:
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# fix / resume / threads
# --------------------------------------------------------------------------- #


@app.command()
def fix(
    scenario: Annotated[
        int | None,
        typer.Option(
            "--scenario", help="Chaos scenario number; needs `medic triage --scenario N` first"
        ),
    ] = None,
    report: Annotated[
        Path | None,
        typer.Option(
            "--report",
            exists=True,
            dir_okay=False,
            help="A .triage.json written by `medic triage`, for incidents that are not scenarios",
        ),
    ] = None,
    record: Annotated[
        bool,
        typer.Option(
            "--record/--no-record", help="Save the transcript under tests/fixtures/<key>/"
        ),
    ] = True,
    model_name: Annotated[
        str | None,
        typer.Option(
            "--model", help="Model for this run (the free tier quota is per model per day)"
        ),
    ] = None,
    pr_mode: Annotated[
        str | None,
        typer.Option("--pr-mode", help="dry-run or push; defaults to MEDIC_PR_MODE"),
    ] = None,
) -> None:
    """Propose a fix from a finished investigation, validate it, and stop for review."""
    from chaos.scenarios import get_scenario
    from medic.config import LLMSettings
    from medic.graph.build import build_graph, investigated_state
    from medic.graph.checkpoints import new_thread_id, open_saver, thread_config
    from medic.graph.ops import LiveSandboxOps
    from medic.graph.state import Budget
    from medic.llm import make_chat_model
    from medic.report import TriageReport, fix_report_path, report_path
    from medic.tools.registry import make_incident_tools
    from medic.transcript import Transcript, fix_fixture_path

    settings = LLMSettings(model=model_name) if model_name else LLM
    spec = None
    if scenario is not None:
        spec = get_scenario(scenario)
        report = report_path(SANDBOX_ROOT, spec.key)
    if report is None or not report.exists():
        typer.echo(
            f"no triage report at {report}; run `medic triage --scenario {scenario}` first"
            if scenario is not None
            else "give --scenario N or --report <path>"
        )
        raise typer.Exit(code=2)
    triage_report = TriageReport.load(report)
    incident = triage_report.incident
    if not Path(incident.repo).exists() or not Path(incident.duckdb).exists():
        typer.echo(f"sandbox for {incident.run_id} is gone; run `medic chaos run` again")
        raise typer.Exit(code=2)

    key = incident.scenario_key or incident.run_id
    tools = make_incident_tools(incident)
    ops = LiveSandboxOps(incident, pr_mode=pr_mode)
    ops.reset()
    model = make_chat_model(settings)
    callbacks = tracing.make_callbacks()
    # The incident's counters carry over; the limits are the configured ones,
    # not whatever `--max-tool-calls` the triage run was given.
    seed_budget = Budget(
        tool_calls=triage_report.budget.tool_calls,
        repeated_calls=triage_report.budget.repeated_calls,
        llm_calls=triage_report.budget.llm_calls,
        tokens=triage_report.budget.tokens,
    )
    thread_id = new_thread_id(key)
    typer.echo(
        f"incident {incident.run_id}: fixing from {report.name}; provider={settings.provider} "
        f"model={settings.model} tracing={'on' if callbacks else 'off'} thread={thread_id}"
    )
    start = investigated_state(
        incident,
        triage=triage_report.triage,
        evidence=triage_report.evidence,
        hypotheses=triage_report.hypotheses,
        hallucinations=triage_report.hallucinations,
        budget=seed_budget,
    )
    with open_saver() as saver:
        graph = build_graph(
            model,
            tools,
            ops=ops,
            budget=seed_budget,
            callbacks=callbacks,
            log=typer.echo,
            checkpointer=saver,
        )
        with tracing.trace("fix", run_id=incident.run_id, model=settings.model) as handle:
            final = graph.invoke(start, config={**thread_config(thread_id), "recursion_limit": 120})

    out = TriageReport.from_state(final, model=settings.model, thread_id=thread_id)
    if spec is not None:
        out.with_ground_truth(spec.category, spec.fix_kind, spec.root_cause_terms)
        out.with_fix_ground_truth(spec.fix_kind)
    typer.echo("")
    typer.echo("\n".join(out.render_fix()))
    saved = out.save(fix_report_path(SANDBOX_ROOT, key))
    typer.echo(f"\nreport {saved}")
    if record:
        path = Transcript.from_state(final, model=settings.model, seed_budget=seed_budget).save(
            fix_fixture_path(key)
        )
        typer.echo(f"transcript {path}")
    if handle.url:
        typer.echo(f"trace: {handle.url}")
    if out.status == "escalated":
        raise typer.Exit(code=1)


@app.command()
def resume(
    thread_id: Annotated[str, typer.Argument(help="Thread id printed by `medic fix`")],
    approve: Annotated[bool, typer.Option("--approve", help="Approve the proposal")] = False,
    reject: Annotated[
        str | None, typer.Option("--reject", help="Reject, with a note for the report")
    ] = None,
    note: Annotated[str, typer.Option("--note", help="Reviewer note to record")] = "",
    push: Annotated[
        bool, typer.Option("--push", help="Open a real pull request instead of writing pr.md")
    ] = False,
) -> None:
    """Resume a run paused at human review with a decision."""
    from langgraph.types import Command

    from medic.graph.build import build_graph
    from medic.graph.checkpoints import open_saver, thread_config, thread_state
    from medic.graph.ops import LiveSandboxOps
    from medic.llm import make_chat_model
    from medic.report import TriageReport, fix_report_path
    from medic.tools.registry import make_incident_tools
    from medic.transcript import Transcript, fix_fixture_path

    if approve == (reject is not None):
        raise typer.BadParameter("give exactly one of --approve or --reject NOTE")
    decision = (
        {"status": "approved", "note": note} if approve else {"status": "rejected", "note": reject}
    )

    with open_saver() as saver:
        info = thread_state(saver, thread_id)
        if info is None:
            typer.echo(f"no thread {thread_id!r}; `medic threads` lists them")
            raise typer.Exit(code=2)
        if info.status != "awaiting_review":
            typer.echo(f"thread {thread_id} is {info.status}, not waiting for review")
            raise typer.Exit(code=2)
        incident = info.values["incident"]
        tools = make_incident_tools(incident)
        ops = LiveSandboxOps(incident, pr_mode="push" if push else None)
        graph = build_graph(make_chat_model(), tools, ops=ops, log=typer.echo, checkpointer=saver)
        final = graph.invoke(Command(resume=decision), config=thread_config(thread_id))

    key = incident.scenario_key or incident.run_id
    previous_path = fix_report_path(SANDBOX_ROOT, key)
    previous = TriageReport.load(previous_path) if previous_path.exists() else None
    out = TriageReport.from_state(
        final, model=previous.model if previous else "unknown", thread_id=thread_id
    )
    if previous:
        out.ground_truth = previous.ground_truth
    typer.echo("")
    typer.echo("\n".join(out.render_fix()))
    typer.echo(
        f"status: {out.status}" + (f" ({out.escalation_reason})" if out.escalation_reason else "")
    )
    out.save(previous_path)
    fixture = fix_fixture_path(key)
    if fixture.exists():
        recorded = Transcript.load(fixture)
        Transcript.from_state(final, model=recorded.model, seed_budget=recorded.seed_budget).save(
            fixture
        )
        typer.echo(f"transcript {fixture}")


@app.command()
def threads() -> None:
    """List checkpointed runs and whether they are waiting for review."""
    from medic.graph.checkpoints import list_threads, open_saver

    with open_saver() as saver:
        rows = list_threads(saver)
    if not rows:
        typer.echo("no threads yet; `medic fix --scenario N` starts one")
        return
    for t in rows:
        flag = "  <- waiting for review" if t.waiting else ""
        typer.echo(f"{t.thread_id:<40} {t.status or '?':<16} {t.run_id or ''}{flag}")


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
