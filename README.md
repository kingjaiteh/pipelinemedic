# PipelineMedic

Agentic incident triage for a dbt + Dagster pipeline. When a model or test
fails, the agent reads the run artifacts, walks the lineage graph, queries the
warehouse, ranks root causes with cited evidence, drafts a fix, validates it in
a sandbox, and waits for a human before opening a pull request.

Target pipeline: [marketing-attribution-platform](https://github.com/kingjaiteh/marketing-attribution-platform),
14 dbt models and 115 tests on DuckDB, orchestrated by Dagster.

## What exists

Phase 0, the spike:

- `medic/tools/dbt_artifacts.py`: read-only parsers over dbt's `target/`
  artifacts, exposed as two LangChain tools. `get_run_results` lists the
  failing nodes with their error messages, what was skipped as a consequence,
  and status counts. `get_lineage` walks the DAG from one node, upstream or
  downstream, capped at depth 3 and 40 nodes, tests excluded. It never
  returns the whole graph. `summarize_source_freshness` does the same for
  `sources.json`.
- `medic/llm.py`: chat model factory. Gemini 2.5 Flash by default, Anthropic
  as an optional extra. Provider and model come from `.env`.
- `medic/tracing.py`: Langfuse tracing. One trace per run with every model
  and tool call inside it. A no-op when the keys are absent.
- `medic spike --run-results <path>`: a single tool-calling loop. The model
  calls the two tools, then submits a structured report naming the failing
  model, what it reads from, and the error it saw.

Phase 1a, the chaos harness:

- `chaos/harness.py`: a `Sandbox` is a copy of the warehouse file plus a git
  worktree of the pipeline on a `medic/<name>` branch. `run_dbt` calls the
  pipeline's own dbt executable inside the worktree against the copy and
  parses the artifact it wrote. `teardown` removes the worktree, the branch,
  and the directory.
- `chaos/scenarios/`: seven scripted failures, one module each, listed in
  `chaos/scenarios.yaml` with the node and error text each is expected to
  produce. Row selection uses a hash of the key columns, so a scenario picks
  the same rows every time.
- `medic chaos list`, `medic chaos run <n|all> [--teardown]`, `medic chaos
  teardown <n|all>`. A run checks the production file's checksum before and
  after, applies the mutation, runs dbt, compares the outcome with the YAML,
  and writes `sandbox/incidents/<key>.json`.
- 40 tests that run with no API key and no warehouse: hand-built artifacts,
  a scripted chat model, an in-memory DuckDB for the mutations, and a
  throwaway git repo for the sandbox lifecycle.

The target pipeline gained two guards for this project: a source freshness
rule on the raw touchpoints table and a daily volume test. Those make the
stale-source and volume-drop incidents observable at build time.

Not built yet: the LangGraph graph with triage, investigator, hypothesis and
critic nodes, the fixer and validator, the human approval gate, the eval
suite, the API, and the console.

## The chaos suite

Each scenario mutates a sandbox and runs dbt there. The table is what the
harness observed on 2026-09-10, not a prediction.

| # | Scenario | Mutation | Failing node | Error |
|---|---|---|---|---|
| 1 | Renamed source column | `raw.touchpoints.channel` renamed to `marketing_channel` | `stg_touchpoints` | Binder Error at view creation |
| 2 | Null flood | `user_id` set to null on 30% of rows | `not_null_stg_touchpoints_user_id` | 51,464 failing rows |
| 3 | Duplicate keys | one day re-inserted | `unique_stg_touchpoints_touchpoint_id` | 3,286 failing rows |
| 4 | Type change | `cost` recreated as text with a few `n/a` values | `not_null_stg_touchpoints_touch_cost_usd` | Conversion Error |
| 5 | Bad commit | a committed typo in `dim_channels.sql` | `dim_channels` | Binder Error |
| 6 | Stale source | `touched_at` shifted back 30 days | `raw.touchpoints` | source freshness error, 31 days old |
| 7 | Volume drop | 80% of one day deleted | `assert_daily_touchpoint_volume` | 1 failing day |

Scenarios 1, 4 and 5 have valid code fixes. Scenarios 2, 3, 6 and 7 are
upstream data problems; the correct response is to say so, not to loosen a
test. Scenario 6 runs `dbt source freshness` instead of `dbt build`, because
a build passes on backdated data.

Two things the observed results changed. In scenario 1, DuckDB reports the
missing column as `Column "channel" referenced that exists in the SELECT
clause - but this column cannot be referenced before it is defined`, because
the staging model aliases its output as `channel` too. The message points at
the model, and the agent will have to look at the source table to see the
rename. In scenario 4 the view builds fine and only one of the eight tests on
it errors, the one that reads the cast column.

## Setup

```powershell
cd pipelinemedic
uv sync
copy .env.example .env   # fill in GEMINI_API_KEY; Langfuse keys are optional
uv run pytest
```

The pipeline repo is expected one directory up at `../marketing-attribution`
with a built `dbt_marketing/target/`, a `data/marketing.duckdb`, and its own
`.venv` with dbt installed. Override the location with `PIPELINE_REPO` and
`PIPELINE_DUCKDB` in `.env`. Sandboxes go under `./sandbox` (about 40 MB
each) unless `MEDIC_SANDBOX_ROOT` says otherwise.

## Running a scenario and the spike

```powershell
uv run medic chaos run 5
uv run medic spike --run-results sandbox\bad_commit\repo\dbt_marketing\target\run_results.json `
                   --manifest    sandbox\bad_commit\repo\dbt_marketing\target\manifest.json
uv run medic chaos teardown 5
```

The chaos run prints the mutation, the dbt exit code, the failing nodes, and
whether the outcome matched the YAML and the production file was untouched.
The spike prints the tool calls it made, a JSON report, and the Langfuse
trace URL when tracing is on. This is the report Gemini 2.5 Flash returned on
scenario 5, after two tool calls:

```json
{
  "failing_model": "dim_channels",
  "upstream_models": ["int_journeys", "int_touchpoints_sessionized", "stg_conversions", "stg_touchpoints"],
  "error_summary": "Runtime Error in model dim_channels (models\\marts\\dim_channels.sql)\n  Binder Error: Values list \"c\" does not have a column named \"touch_cnt\"\n  \n  LINE 86:     c.touch_cnt,\n               ^",
  "reasoning": "The model `dim_channels` is the only failing model. The error message indicates that a column named `touch_cnt` is missing from one of its upstream inputs, which `dim_channels` expects to find."
}
```

The reasoning is wrong in an instructive way: the column is missing from the
model's own select list, not from an upstream. That is the gap the
investigator and critic nodes in the next phase exist to close, with file
reads and git history as evidence.

## Guardrails that already hold

- The production DuckDB file is never opened by this project. Every scenario
  and every dbt run targets a copy, and the harness proves it with a checksum
  before and after.
- Code changes happen in a git worktree on a `medic/` branch. The pipeline's
  working tree and `main` are never touched.
- Tool file paths are bound when the tools are built, not chosen by the model.
- The API key is read from `GEMINI_API_KEY` and passed to the client
  explicitly. Nothing here reads the provider library's default variable.
