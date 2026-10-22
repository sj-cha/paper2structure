from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RevisionAction = Literal["accept_candidates", "rebuild_structures", "revisit_evidence", "stop_inconclusive"]


class StructureVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    structure_id: str
    resembles_paper: bool
    issues: list[str] = Field(default_factory=list)


class RevisionDecision(BaseModel):
    """Agent 5 output. rebuild_structures sends structures back to Agent 4 (the plan stands);
    revisit_evidence sends the feedback to Agent 3 (the characterization reports are kept), then
    Agent 4 builds the new plans."""
    model_config = ConfigDict(extra="forbid")
    action: RevisionAction
    reasoning: str
    verdicts: list[StructureVerdict] = Field(default_factory=list)
    accepted_structure_ids: list[str] = Field(default_factory=list)
    rebuild_structure_ids: list[str] = Field(default_factory=list,
                                             description="structures to rebuild under their plans (rebuild_structures)")
    feedback: str = Field("", description="concrete instructions: what is wrong and what the rebuilt structures or "
                                          "the new plans must show")
    unresolved: list[str] = Field(default_factory=list, description="observations/assumptions left unexplained")


class ReportNarrative(BaseModel):
    """The reviewer's own text in final_report.md; the write_report tool adds the deterministic sections."""
    model_config = ConfigDict(extra="forbid")
    summary: str
    assessment: str
    next_steps: list[str] = Field(default_factory=list)


class ReviewCheck(BaseModel):
    """A claim check the reviewer ran itself (check_claim) and its result."""
    review: int = Field(description="index of the review (revision_history) it was run in")
    structure_id: str
    check_id: str
    kind: str
    claim: str
    status: Literal["pass", "warn", "fail", "skip"]
    detail: str = ""
