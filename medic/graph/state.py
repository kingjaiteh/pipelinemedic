"""Graph state. Sub-objects are Pydantic; the state itself is a TypedDict.

Lists that several nodes append to (messages, tool records, LLM turns,
hallucinations) carry a reducer. Everything else is last write wins.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from medic.config import BUDGET

Category = Literal[
    "schema_drift",
    "data_quality",
    "dependency_failure",
    "code_error",
    "source_freshness",
    "volume_anomaly",
]
CATEGORIES: tuple[str, ...] = (
    "schema_drift",
    "data_quality",
    "dependency_failure",
    "code_error",
    "source_freshness",
    "volume_anomaly",
)
FixKind = Literal["code_patch", "test_change", "upstream_data_issue", "needs_human"]


class FailingNode(BaseModel):
    unique_id: str
    name: str
    resource_type: str
    status: str
    message: str | None = None


class Incident(BaseModel):
    """Where the failure came from and where the sandbox for it lives."""

    source: Literal["chaos", "run_results", "dagster"]
    run_id: str
    scenario: int | None = None
    scenario_key: str | None = None
    dbt_args: list[str] = Field(default_factory=list)
    artifact: str
    manifest: str
    sandbox_dir: str
    duckdb: str
    repo: str
    failing_nodes: list[FailingNode] = Field(default_factory=list)
    status_counts: dict[str, int] = Field(default_factory=dict)

    @property
    def raw_error(self) -> str:
        return "\n".join(f"{n.name}: {n.message or n.status}" for n in self.failing_nodes)

    def summary(self) -> dict[str, Any]:
        """What the model is shown: the failure, not the file system."""
        return {
            "dbt_command": "dbt " + " ".join(self.dbt_args),
            "status_counts": self.status_counts,
            "failing_nodes": [n.model_dump() for n in self.failing_nodes],
        }


class Triage(BaseModel):
    category: Category
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class ToolRecord(BaseModel):
    """One tool call as it happened. The critic checks excerpts against `output`."""

    id: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    output: str
    error: bool = False
    api_call_id: str | None = None
    # Set when the call duplicated an earlier one; the real output lives there.
    repeat_of: str | None = None
    # "model" when the model asked for it; the fix phase also reads files on
    # its own to show the fixer current text ("fixer_preread", "fixer_feedback").
    origin: str = "model"


class Evidence(BaseModel):
    id: str
    claim: str
    tool: str
    tool_call_id: str
    excerpt: str


class Hypothesis(BaseModel):
    rank: int
    root_cause: str
    category: Category
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    fix_kind: FixKind
    recommended_action: str


class Hallucination(BaseModel):
    kind: Literal["evidence", "hypothesis"]
    ref: str
    claim: str
    reason: str


class LLMTurn(BaseModel):
    """One model response, kept so a run can be replayed without the API."""

    node: str
    content: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)


class EditSpec(BaseModel):
    """One exact text replacement inside the sandbox worktree."""

    path: str
    find: str
    replace: str


class ProposedFix(BaseModel):
    kind: FixKind
    explanation: str
    edits: list[EditSpec] = Field(default_factory=list)
    recommended_action: str = ""

    @property
    def is_code_change(self) -> bool:
        return self.kind in ("code_patch", "test_change") and bool(self.edits)


class EditOutcome(BaseModel):
    path: str
    ok: bool
    error: str | None = None


class ApplyResult(BaseModel):
    """What happened when a proposal's edits were applied. All or nothing."""

    ok: bool
    outcomes: list[EditOutcome] = Field(default_factory=list)
    diff: str = ""
    edited_paths: list[str] = Field(default_factory=list)


class Validation(BaseModel):
    """One dbt run in the sandbox after a fix was applied."""

    passed: bool
    dbt_args: list[str] = Field(default_factory=list)
    returncode: int = 0
    elapsed_s: float = 0.0
    status_counts: dict[str, int] = Field(default_factory=dict)
    failing: list[FailingNode] = Field(default_factory=list)
    stdout_tail: str = ""
    error: str | None = None

    def summary(self) -> str:
        if self.error:
            return self.error
        counts = ", ".join(f"{k}={v}" for k, v in sorted(self.status_counts.items()))
        head = f"dbt {' '.join(self.dbt_args)} exit {self.returncode} ({counts})"
        if self.passed:
            return head
        names = "; ".join(f"{n.name}: {(n.message or n.status)[:200]}" for n in self.failing[:5])
        return f"{head}: {names}"


class FixAttempt(BaseModel):
    iteration: int
    proposal: ProposedFix
    apply: ApplyResult | None = None
    validation: Validation | None = None

    @property
    def outcome(self) -> str:
        if not self.proposal.is_code_change:
            return f"declined ({self.proposal.kind})"
        if self.apply is not None and not self.apply.ok:
            return "edits did not apply"
        if self.validation is None:
            return "not validated"
        return "passed" if self.validation.passed else "failed validation"


class Approval(BaseModel):
    status: Literal["approved", "rejected"]
    note: str = ""
    decided_at: str = ""


class PullRequest(BaseModel):
    mode: Literal["dry-run", "push"]
    title: str
    branch: str
    body: str = ""
    path: str | None = None
    url: str | None = None
    commit: str | None = None


class Budget(BaseModel):
    tool_calls: int = 0
    repeated_calls: int = 0
    llm_calls: int = 0
    tokens: int = 0
    # Fix phase. One iteration is one proposal applied and validated; the
    # fixer may read a few files before each proposal, which is its turn count.
    fix_iterations: int = 0
    fixer_turns: int = 0
    max_tool_calls: int = BUDGET.max_tool_calls
    max_llm_calls: int = BUDGET.max_llm_calls
    max_tokens: int = BUDGET.max_tokens
    max_fix_iterations: int = BUDGET.max_fix_iterations
    max_fixer_turns: int = BUDGET.max_fixer_turns

    def exceeded(self) -> str | None:
        if self.tool_calls > self.max_tool_calls:
            return f"tool calls {self.tool_calls} > {self.max_tool_calls}"
        if self.llm_calls > self.max_llm_calls:
            return f"LLM calls {self.llm_calls} > {self.max_llm_calls}"
        if self.tokens > self.max_tokens:
            return f"tokens {self.tokens} > {self.max_tokens}"
        return None

    @property
    def tool_calls_left(self) -> int:
        return max(0, self.max_tool_calls - self.tool_calls)

    @property
    def fix_iterations_left(self) -> int:
        return max(0, self.max_fix_iterations - self.fix_iterations)


Status = Literal[
    "triaging",
    "investigating",
    "hypothesizing",
    "investigated",
    "fixing",
    "validating",
    "awaiting_review",
    "approved",
    "rejected",
    "done",
    "escalated",
]


class MedicState(TypedDict, total=False):
    incident: Incident
    triage: Triage | None
    messages: Annotated[list[BaseMessage], add_messages]
    tool_records: Annotated[list[ToolRecord], operator.add]
    llm_turns: Annotated[list[LLMTurn], operator.add]
    hallucinations: Annotated[list[Hallucination], operator.add]
    evidence: list[Evidence]
    hypotheses: list[Hypothesis]
    budget: Budget
    status: Status
    escalation_reason: str | None
    # Fix phase. The fixer has its own conversation. Attempts are rewritten
    # whole by the two nodes that own them, in sequence, so no reducer.
    fix_messages: Annotated[list[BaseMessage], add_messages]
    proposed_fix: ProposedFix | None
    fix_attempts: list[FixAttempt]
    validation: Validation | None
    approval: Approval | None
    pull_request: PullRequest | None
