"""Structured outputs the agents submit as forced tool calls.

Using a tool call rather than free text means every model turn, including
triage and hypothesis ranking, is an AIMessage with tool_calls. That keeps
one code path for live runs and for replay from a recorded transcript.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from medic.graph.state import Category, FixKind


class TriageVerdict(BaseModel):
    """Classify the incident. Submit exactly once."""

    category: Category = Field(description="One category for the incident")
    confidence: float = Field(ge=0.0, le=1.0, description="0 to 1")
    rationale: str = Field(
        description="One sentence quoting the decisive part of the error message"
    )


class EvidenceItem(BaseModel):
    claim: str = Field(description="One sentence stating a fact the excerpt supports")
    tool_call_id: str = Field(
        description="The tag at the start of the tool result the excerpt comes from, for example T3"
    )
    excerpt: str = Field(
        description="Copied verbatim from that tool result, under 300 characters. Do not paraphrase."
    )


class SubmitEvidence(BaseModel):
    """End the investigation and hand over the evidence. Submit exactly once."""

    evidence: list[EvidenceItem] = Field(description="3 to 8 items, most decisive first")
    summary: str = Field(description="Two sentences: what broke and what the evidence shows")


class HypothesisItem(BaseModel):
    root_cause: str = Field(
        description="One or two sentences naming the exact table, column, file or commit involved"
    )
    category: Category
    evidence_ids: list[str] = Field(description="Evidence ids (E1, E2, ...) that support this")
    confidence: float = Field(ge=0.0, le=1.0)
    fix_kind: FixKind = Field(
        description=(
            "code_patch if changing pipeline code fixes it without hiding a data problem; "
            "upstream_data_issue if the data is wrong and the pipeline correctly caught it; "
            "test_change only if the test itself is wrong; needs_human if unclear"
        )
    )
    recommended_action: str = Field(description="One sentence: what to do next")


class HypothesisSet(BaseModel):
    """Ranked root-cause hypotheses. Submit exactly once."""

    hypotheses: list[HypothesisItem] = Field(description="1 to 3 hypotheses, most likely first")
