# PipelineMedic

Agentic incident triage for a dbt + Dagster pipeline. When a model or test
fails, the agent reads the run artifacts, walks the lineage graph, queries the
warehouse, ranks root causes with cited evidence, drafts a fix, validates it in
a sandbox, and waits for a human before opening a pull request.

Target pipeline: [marketing-attribution-platform](https://github.com/kingjaiteh/marketing-attribution-platform).

Status: scaffold only. Phase 0 (parse dbt artifacts, lineage tool, one traced
LLM call) is next. See PLAN.md for scope and sequencing.

## Setup

```powershell
cd pipelinemedic
uv sync
copy .env.example .env   # then fill in GEMINI_API_KEY
uv run pytest
```
