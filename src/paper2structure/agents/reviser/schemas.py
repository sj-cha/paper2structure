from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RevisionAction = Literal["accept_candidates", "revisit_evidence", "stop_inconclusive"]


class StructureVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    structure_id: str
    resembles_paper: bool
    issues: list[str] = Field(default_factory=list)


class RevisionDecision(BaseModel):
    """Agent 5 output. revisit_evidence routes back through Agent 2 -> Agent 3 -> Agent 4."""
    model_config = ConfigDict(extra="forbid")
    action: RevisionAction
    reasoning: str
    verdicts: list[StructureVerdict] = Field(default_factory=list)
    accepted_structure_ids: list[str] = Field(default_factory=list)
    feedback: str = Field("", description="concrete instructions for re-examining evidence and re-planning")
    unresolved: list[str] = Field(default_factory=list, description="observations/assumptions left unexplained")


class ReportNarrative(BaseModel):
    """The reviewer's own text in final_report.md; the write_report tool adds the deterministic sections."""
    model_config = ConfigDict(extra="forbid")
    summary: str
    assessment: str
    next_steps: list[str] = Field(default_factory=list)
