# PipelineMedic

Agentic incident triage for a dbt + Dagster pipeline. When a model or test
fails, the agent reads the run artifacts, walks the lineage graph, queries the
warehouse, ranks root causes with cited evidence, drafts a fix, validates it in
a sandbox, and waits for a human before opening a pull request.

Target pipeline: [marketing-attribution-platform](https://github.com/kingjaiteh/marketing-attribution-platform),
14 dbt models and 115 tests on DuckDB, orchestrated by Dagster.

## What exists (Phase 0, spike)

- `medic/tools/dbt_artifacts.py`: two read-only tools over dbt's `target/`
  artifacts. `get_run_results` lists the failing nodes with their error
  messages, what was skipped as a consequence, and status counts.
  `get_lineage` walks the DAG from one node, upstream or downstream, capped
  at depth 3 and 40 nodes, tests excluded. It never returns the whole graph.
- `medic/llm.py`: chat model factory. Gemini 2.5 Flash by default, Anthropic
  as an optional extra. Provider and model come from `.env`.
- `medic/tracing.py`: Langfuse tracing. One trace per run with every model
  and tool call inside it. A no-op when the keys are absent.
- `medic spike --run-results <path>`: a single tool-calling loop. The model
  calls the two tools, then submits a structured report naming the failing
  model, what it reads from, and the error it saw.
- 19 tests that run with no API key and no warehouse, against hand-built
  artifacts and a scripted model.

The target pipeline gained two guards for this project: a source freshness
rule on the raw touchpoints table and a daily volume test. Those make the
stale-source and volume-drop incidents in the plan observable at build time.

Not built yet: the chaos harness, the LangGraph graph with triage,
investigator, hypothesis and critic nodes, the fixer and validator, the human
approval gate, the eval suite, the API, and the console.

## Setup

```powershell
cd pipelinemedic
uv sync
copy .env.example .env   # fill in GEMINI_API_KEY; Langfuse keys are optional
uv run pytest
```

The pipeline repo is expected one directory up at `../marketing-attribution`
with a built `dbt_marketing/target/` and `data/marketing.duckdb`. Override
with `PIPELINE_REPO` and `PIPELINE_DUCKDB` in `.env`.

## Running the spike

Point it at a `run_results.json` from a failed build. To make one by hand,
break a model in a sandbox rather than in the pipeline's working tree:

```powershell
# from the pipeline repo
git worktree add ..\pipelinemedic\sandbox\spike\repo -b medic/spike main
copy data\marketing.duckdb ..\pipelinemedic\sandbox\spike\marketing.duckdb

# in the worktree: introduce a typo in a marts model, then build against the copy
$env:MARKETING_DUCKDB = "<absolute path to the copy>"
$env:DBT_PROFILES_DIR = "<worktree>\dbt_marketing"
.venv\Scripts\dbt.exe build --project-dir <worktree>\dbt_marketing

# back here
uv run medic spike --run-results sandbox\spike\repo\dbt_marketing\target\run_results.json `
                   --manifest    sandbox\spike\repo\dbt_marketing\target\manifest.json
```

Output is the sequence of tool calls, a JSON report, and the Langfuse trace
URL when tracing is on. The report has the shape below. The values are what
the two tools returned for the sandbox build above, not a recorded model run:

```json
{
  "failing_model": "dim_channels",
  "upstream_models": ["int_journeys", "int_touchpoints_sessionized", "stg_conversions", "stg_touchpoints"],
  "error_summary": "Binder Error: Values list \"c\" does not have a column named \"touch_cnt\"",
  "reasoning": "dim_channels is the only node that errored; the 21 skipped nodes are downstream of it."
}
```

Tear the sandbox down from the pipeline repo with `git worktree remove` and
`git branch -D medic/spike`. `sandbox/` is gitignored.

## Guardrails that already hold

- The production DuckDB file is never opened by this project. Sandbox builds
  run against a copy.
- Tool file paths are bound when the tools are built, not chosen by the model.
- The API key is read from `GEMINI_API_KEY` and passed to the client
  explicitly. Nothing here reads the provider library's default variable.
