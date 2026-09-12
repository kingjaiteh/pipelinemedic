# PipelineMedic

Agentic incident triage for a dbt + Dagster pipeline. When a model or test
fails, the agent reads the run artifacts, walks the lineage graph, queries a
sandbox copy of the warehouse, and ranks root causes where every claim quotes
a tool result verbatim. It then proposes a fix or declines to change code,
validates the fix with dbt in the sandbox, and stops for a human decision
before a pull request is drafted.

Target pipeline: [marketing-attribution-platform](https://github.com/kingjaiteh/marketing-attribution-platform),
14 dbt models and 115 tests on DuckDB, orchestrated by Dagster.

## Results so far

Seven scripted failures injected into a sandbox, each investigated once by
`medic triage`. Root cause is graded by terms in `chaos/scenarios.yaml`;
category and fix kind are reported, not graded. Runs are from 2026-09-10.
The free Gemini tier allows 20 requests per model per day, so the runs are
spread over several Gemini 3.x flash models; each report records its model.

| # | Scenario | Top-1 root cause | Category | Fix kind | Tool calls | Model calls | Tokens | Evidence kept / stripped |
|---|---|---|---|---|---|---|---|---|
| 1 | Renamed source column | correct | code_error (expected schema_drift) | code_patch | 12 | 13 | 150,345 | 4 / 0 |
| 2 | Null flood | correct | data_quality | upstream_data_issue | 5 | 5 | 19,592 | 2 / 1 |
| 3 | Duplicate keys | correct | data_quality | upstream_data_issue | 8 | 5 | 18,199 | 2 / 1 |
| 4 | Type change | correct | data_quality (expected schema_drift) | upstream_data_issue (expected code_patch) | 4 | 7 | 14,530 | 2 / 1 |
| 5 | Bad commit | correct | code_error | code_patch | 8 | 9 | 55,279 | 4 / 0 |
| 6 | Stale source | correct | source_freshness | upstream_data_issue | 8 | 7 | 32,978 | 4 / 0 |
| 7 | Volume drop | correct | volume_anomaly | upstream_data_issue | 8 | 7 | 24,534 | 3 / 0 |

Every surviving evidence item is a verbatim substring of the tool result it
cites; the critic stripped three that were not, and one malformed item. No
run queried the answer-key tables. Two scenarios needed a second run: the
first run of scenario 3 lost its whole submission to one malformed item, and
the first run of scenario 7 blamed the generator's natural tail-off instead
of the one bad day. Both fixes are in the code, not the prompts for those
cases: per-item validation, and a tool that runs the failing test's own SQL.

What the top hypothesis said, shortened:

- 1: `stg_touchpoints.sql` references `channel`, but the column in
  `raw.touchpoints` is now `marketing_channel`.
- 2: `raw.touchpoints` has 51,464 null `user_id` values, which also nulls
  `touchpoint_id`.
- 3: `raw.touchpoints` has duplicate `(user_id, touch_number)` rows, so the
  composite key is not unique.
- 4: 37 rows of `raw.touchpoints.cost` hold the string `n/a`, and the cast
  to double fails. Ranked as an upstream data issue with a defensive cast as
  the second hypothesis. The plan called it a code patch at the time; the
  fixing section below explains why that changed.
- 5: commit ab4cfdb changed `c.touch_count` to `c.touch_cnt` in
  `dim_channels.sql`; revert it.
- 6: `raw.touchpoints` is 31 days stale against the dataset's reference
  date, past the 7-day error threshold.
- 7: 2026-02-10 has 684 touchpoints against a trailing median of 3,175,
  about 21 percent.

Scenario 1 is the instructive one. DuckDB reports the missing column as
`Column "channel" referenced that exists in the SELECT clause - but this
column cannot be referenced before it is defined`, because the staging model
aliases its output as `channel` too. The message points at the model; the
agent had to list the source table's columns to see the rename.

### Fixing

Each investigation above was then handed to the fixer with `medic fix`
(runs from 2026-09-12, seeded from the saved reports, so the investigation
was not repeated). The fixer either edits the sandbox checkout and has dbt
rebuild the affected models, or declines and says what the data owner should
do. Every run stopped at the human review gate and was approved from a
separate process.

| # | Scenario | Fixer decision | Expected | Sandbox build | Model calls | Files read first | PR draft |
|---|---|---|---|---|---|---|---|
| 1 | Renamed source column | code_patch: alias `marketing_channel` in staging, rename the column in `_sources.yml` | code_patch | passed, 105 tests | 3 | listed staging, read `_sources.yml` | yes |
| 2 | Null flood | upstream_data_issue: 51,464 null `user_id`, fix ingestion | upstream_data_issue | not attempted | 3 | listed project, read `_sources.yml` | no |
| 3 | Duplicate keys | upstream_data_issue: duplicate `(user_id, touch_number)` rows, request a clean reload | upstream_data_issue | not attempted | 3 | listed project, read the staging tests | no |
| 4 | Type change | upstream_data_issue: 37 `'n/a'` costs, correct at the source | upstream_data_issue | not attempted | 2 | read the staging tests | no |
| 5 | Bad commit | code_patch: revert `c.touch_cnt` to `c.touch_count` | code_patch | passed, 20 tests | 1 | none | yes |
| 6 | Stale source | upstream_data_issue: newest row 31 days before the reference date, restart ingestion | upstream_data_issue | not attempted | 3 | listed project, read `_sources.yml` | no |
| 7 | Volume drop | upstream_data_issue: 684 touchpoints on 2026-02-10 against a median of 3,175 | upstream_data_issue | not attempted | 4 | listed tests, read the volume test | no |

Seven of seven decisions match the expected fix kind, both code patches
passed their sandbox build on the first attempt, and no run edited a test,
a macro, or `dbt_project.yml`. The fixer is shown the failing model's file
before its first turn; the "files read first" column is what it chose to
read on top of that.

Scenario 4 changed the ground truth. The plan expected a code patch (a
defensive cast). The fixer read the staging tests, saw the `not_null` test
on `touch_cost_usd`, and declined: a `try_cast` turns `'n/a'` into null and
that test still fails, while defaulting the value to 0 would hide bad source
data. That reasoning is right for this pipeline, so scenario 4 now expects
`upstream_data_issue`. The fixer's explanation, as recorded:

> The source table raw.touchpoints contains 37 rows where the cost column
> has the string literal 'n/a' instead of numeric values, causing the
> staging cast to double to fail. Casting 'n/a' to NULL would still violate
> the not_null test on touch_cost_usd, and replacing bad values with
> defaults like 0 would improperly mask invalid source data.

Scenario 1's patch, as applied and validated:

```diff
--- a/dbt_marketing/models/staging/stg_touchpoints.sql
+++ b/dbt_marketing/models/staging/stg_touchpoints.sql
@@ -30,7 +30,7 @@
         -- nanoseconds, and the wider type propagates awkwardly into joins.
         cast(touched_at as timestamp)    as touched_at,

-        lower(trim(channel))             as channel,
+        lower(trim(marketing_channel))   as channel,
         lower(trim(campaign))            as campaign,

         cast(cost as double)             as touch_cost_usd
--- a/dbt_marketing/models/staging/_sources.yml
+++ b/dbt_marketing/models/staging/_sources.yml
@@ -34,7 +34,7 @@
             description: 0-based position of this touch within the user's journey.
           - name: touched_at
             description: When the exposure happened.
-          - name: channel
+          - name: marketing_channel
             description: Marketing channel (six values, see channels.py).
           - name: campaign
             description: Campaign within the channel, three per channel.
```

One budget artifact showed up: scenario 2's investigation had been run with
a tool-call cap of 8 to save quota, the cap carried into the fix run, and
the fixer's exploratory reads pushed the count to 9, which escalated the run
before it proposed anything. The fix run now inherits the incident's counters
but takes its limits from configuration; the rerun is the row above.

## What exists

Phase 0, the spike:

- `medic/tools/dbt_artifacts.py`: parsers over dbt's `target/` artifacts.
  `get_run_results` lists failing nodes with messages, skips, and counts, or
  the per-source freshness result when the failed command was `dbt source
  freshness`. `get_lineage` walks the DAG from one node, capped at depth 3
  and 40 nodes, tests excluded.
- `medic/llm.py`: chat model factory (Gemini by default, Anthropic optional),
  with backoff for per-minute limits and overloads and an immediate stop on
  the daily quota.
- `medic/tracing.py`: Langfuse tracing, a no-op when the keys are absent.
- `medic spike`: a single tool-calling loop ending in a structured report.

Phase 1a, the chaos harness:

- `chaos/harness.py`: a `Sandbox` is a copy of the warehouse file plus a git
  worktree of the pipeline on a `medic/<name>` branch. `run_dbt` calls the
  pipeline's own dbt executable against the copy. `teardown` removes all of
  it. The production file's checksum is compared before and after every run.
- `chaos/scenarios/`: seven scripted failures, one module each, with the
  expected node, error text, and root-cause terms in `chaos/scenarios.yaml`.
- `medic chaos list|run|teardown`.

Phase 1b, the read-only investigator:

- Eight read-only tools, all bound to the sandbox: `get_run_results`,
  `get_lineage`, `read_pipeline_file`, `list_pipeline_files`,
  `get_recent_changes` (git log with diffs on the worktree), `query_duckdb`,
  `get_test_failures` (runs the failing test's compiled SQL and returns the
  rows it flagged), and `get_source_freshness`.
- Guards: SQL must be a single SELECT, writes and file reads are rejected,
  the answer-key tables are refused by name, a 200-row limit is stamped on,
  and a timer interrupts the connection after 10 seconds. File reads resolve
  inside the worktree or are refused. Tool file paths are bound at
  construction, never chosen by the model.
- `medic/graph/`: a hand-built LangGraph state graph. `triage_agent`
  classifies; `investigator_agent` makes one model turn per visit;
  `run_tools` executes its calls; a conditional edge loops, hands over to
  `hypothesis_agent` when the model submits evidence, or escalates when a
  budget is exceeded (25 tool calls, 20 model calls, 200k tokens). Identical
  tool calls are answered from the first result and, after two, force a
  submission. `critic_node` runs last.
- The critic: a claim survives only if its cited tool result exists and its
  excerpt is a substring of that result (exact, or with whitespace and JSON
  escapes normalized). A hypothesis keeps only surviving citations and is
  dropped with none. Stripped items are listed in the report.
- Every tool result the model sees starts with a tag such as
  `[T3 read_pipeline_file]`, and the incident summary is `T0`. The model
  cites tags. This exists because the Gemini API never shows the model the
  tool-call ids the client generates; the first live run cited invented ids
  and the critic stripped everything.
- `medic triage --scenario N [--model M]`, or `--run-results <path>
  --sandbox <dir>` for any incident. Writes a report to
  `sandbox/incidents/<key>.triage.json` and a transcript to
  `tests/fixtures/<key>/transcript.json`.
- Replay: each transcript (model turns, tool outputs) is replayed through the
  real graph by `tests/test_replay.py` with a scripted model and replay
  tools, asserting the same hypotheses, stripped items, and budget counts.
  102 tests run with no API key and no warehouse.

Phase 2, fix, validate, approve:

- `fixer_agent`: one model turn per visit, like the investigator. It is
  given the ranked hypotheses with their evidence and the current text of
  the failing model's file, may read a few more files, and submits
  `ProposeFix`: a kind (`code_patch`, `test_change`, `upstream_data_issue`,
  `needs_human`), an explanation, and a list of exact text replacements.
  The model never writes a diff; the tool generates it.
- `medic/tools/sandbox_edit.py`: applies the replacements inside the
  worktree's dbt project, all or nothing. `find` must match the current
  file once. It refuses `profiles.yml` always, and `tests/`, `macros/` and
  `dbt_project.yml` unless the proposal is a `test_change`. A yml edit that
  removes or moves a test declaration or a freshness block, or adds a
  `severity`, `where`, `enabled`, `warn_if`, `error_if`, `limit` or
  `fail_calc` setting, is refused on the same condition. Adding a test is
  allowed.
- `validator_node`: runs `dbt build --select <models>+` in the sandbox, where
  the models are the failing ones (or the model a failing test belongs to)
  plus any model whose file was edited. Artifacts go to a separate
  `validation/` directory so the incident's own `run_results.json` survives.
  A freshness incident is re-checked with `dbt source freshness`.
- A failed validation, or an edit the guard refused, goes back to the fixer
  with the dbt output and the current text of the edited files, up to three
  attempts. A later attempt may decline instead; the worktree is then reset.
- `human_review`: LangGraph `interrupt()`. The graph state is checkpointed
  to SQLite after every node, so `medic fix` exits at the gate and `medic
  resume <thread> --approve|--reject "note"` continues it from a new process.
  Approval of a validated patch opens the pull request; approval of a
  decline just closes the incident; rejection escalates with the note.
- `open_pr`: dry-run by default, writes `sandbox/incidents/<key>.pr.md` with
  the incident, the root cause and its quoted evidence, the diff, the
  validation result and the reviewer's note. `--push` commits the sandbox
  branch, pushes it, and opens the pull request through the GitHub REST API
  with a fine-grained token. Never to main.
- `medic fix --scenario N [--model M]` seeds a new thread from the saved
  triage report, so the investigation is not repeated; `medic threads`
  lists threads and which are waiting. Fix runs are recorded to
  `tests/fixtures/<key>/fix_transcript.json` and replayed by the same tests,
  including the interrupt and the recorded decision. 157 tests, no API key,
  no warehouse.

The target pipeline gained two guards for this project: a source freshness
rule on the raw touchpoints table and a daily volume test.

Not built yet: the eval suite, the API, and the console.

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

## Running a scenario end to end

```powershell
uv run medic chaos run 5
uv run medic triage --scenario 5 --model gemini-3.6-flash
uv run medic fix --scenario 5 --model gemini-3.8-flash
uv run medic resume bad_commit-20260912-224334 --approve --note "one-line revert, build green"
uv run medic chaos teardown 5
```

The chaos run prints the mutation, the dbt exit code, the failing nodes, and
whether the outcome matched the YAML and the production file was untouched.
The triage run prints each model turn and tool call as it happens, then the
report: triage verdict, ranked hypotheses with their cited evidence, the
excerpts, anything the critic stripped, and how the top hypothesis compares
with the scenario's ground truth. This is the scenario 5 report, as printed
(the status label was `done` when this was recorded; it is `investigated`
now that the graph continues into the fix phase):

```
status: done  model: gemini-3.6-flash
triage: code_error (0.95) Binder Error: Values list "c" does not have a column named "touch_cnt"
budget: 8 tool calls, 9 LLM calls, 55279 tokens

#1 [code_error, code_patch, 0.95] Commit ab4cfdb renamed c.touch_count to c.touch_cnt in models/marts/dim_channels.sql, creating a mismatch with the CTE alias touch_count defined in by_channel and expected in _marts__models.yml.
    action: Revert c.touch_cnt back to c.touch_count in models/marts/dim_channels.sql.
    evidence: E1, E2, E3, E4

E1 [get_run_results T1] The dbt build run failed on model dim_channels with a DuckDB Binder Error indicating column touch_cnt does not exist in CTE alias c.
    ""message":"Runtime Error in model dim_channels (models\\marts\\dim_channels.sql)\n  Binder Error: Values list \"c\" does not have a column named \"touch_cnt\..."
E2 [get_recent_changes T4] Recent commit ab4cfdb modified dim_channels.sql by changing c.touch_count to c.touch_cnt in the outer SELECT statement.
    "-    c.touch_count,
E3 [read_pipeline_file T3] The upstream CTE by_channel in dim_channels.sql defines the column alias as touch_count.
    "count(*)                                        as touch_count,"
E4 [read_pipeline_file T7] The schema configuration in _marts__models.yml expects the model column to be named touch_count.
    "- name: touch_count"

ground truth: top-1 root cause CORRECT, category ok (expected code_error), fix kind ok (expected code_patch)
```

The fix run starts from that report, edits the sandbox checkout, rebuilds
the affected models, and stops. This is scenario 5's fix run, as printed:

```
incident chaos-bad_commit: fixing from bad_commit.triage.json; provider=gemini model=gemini-3.8-flash tracing=off thread=bad_commit-20260912-224334
fixer: ProposeFix
fixer: code_patch, edited dbt_marketing/models/marts/dim_channels.sql
validator: passed; dbt build --select dim_channels+ exit 0 (pass=20, success=2)

attempt 1: passed; code_patch
    Revert `c.touch_cnt` to `c.touch_count` in `dim_channels.sql`. The upstream CTE `by_channel` defines the column as `touch_count`, causing a DuckDB binder error when referenced as `touch_cnt`.
    validation: dbt build --select dim_channels+ exit 0 (pass=20, success=2)

--- a/dbt_marketing/models/marts/dim_channels.sql
+++ b/dbt_marketing/models/marts/dim_channels.sql
@@ -73,7 +73,7 @@
 select
     c.channel,

-    c.touch_cnt,
+    c.touch_count,
     c.users_reached,
     c.campaign_count,
     c.converting_users_reached,

waiting for review. Resume with one of:
  medic resume bad_commit-20260912-224334 --approve
  medic resume bad_commit-20260912-224334 --reject "why"
```

`medic resume ... --approve` from another shell prints the decision and
writes `sandbox/incidents/bad_commit.pr.md`: the incident, the root cause
with its four quoted evidence items, the diff, the validation line and the
reviewer's note. Add `--push` to commit the sandbox branch, push it, and open
the pull request for real; that needs `GITHUB_TOKEN` in `.env`.

## Guardrails that already hold

- The production DuckDB file is never opened by this project. Every scenario,
  every query and every dbt run targets a copy, and the harness proves it
  with a checksum before and after.
- Code changes happen in a git worktree on a `medic/` branch. The pipeline's
  working tree and `main` are never touched.
- The model never chooses a file on disk: tool paths are bound when the tools
  are built, and file reads that resolve outside the worktree are refused.
- SQL from the model is one SELECT with a row limit and a timeout, and the
  answer-key tables are refused by name. No run has queried them.
- Nothing reaches the report without a verbatim quote from a tool result.
- Edits are confined to the dbt project inside the worktree. Test
  definitions, macros, `dbt_project.yml` and `profiles.yml` cannot be edited
  by a code patch, and a yml edit that removes or weakens a test is refused.
  No recorded run touched any of them.
- Nothing leaves the sandbox without a human decision. The graph stops at
  `human_review` and only `medic resume --approve` continues it; a pull
  request is a dry-run file unless `--push` is given, and it targets a
  `medic/` branch, never main.
- The API key is read from `GEMINI_API_KEY` and passed to the client
  explicitly. Nothing here reads the provider library's default variable.
