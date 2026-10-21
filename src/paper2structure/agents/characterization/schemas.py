from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from paper2structure.agents.summarizer.schemas import SourceReference


class EvidenceItem(BaseModel):
    """One experimental observation, kept separate from its interpretation."""
    model_config = ConfigDict(extra="forbid")
    evidence_id: str
    technique: str
    observation: str
    value: Optional[float] = None
    unit: Optional[str] = None
    authors_interpretation: Optional[str] = None
    structural_implication: Optional[str] = None
    confidence: Literal["high", "medium", "low"] = "medium"
    classification: Literal["reported", "inferred"] = "reported"
    source: SourceReference = Field(default_factory=SourceReference)


class StructuralConstraint(BaseModel):
    """What atomic configurations the evidence supports, excludes or requires."""
    model_config = ConfigDict(extra="forbid")
    constraint_id: str
    statement: str
    evidence_ids: list[str]
    effect: Literal["supports", "excludes", "requires"] = "supports"


class ExperimentalEvidence(BaseModel):
    """Agent 2 output. Minimal placeholder schema while Agent 2 is a fixture loader."""
    model_config = ConfigDict(extra="forbid")
    synthetic: bool = Field(False, description="True for hand-written fixtures, not extracted from a real paper")
    items: list[EvidenceItem]
    constraints: list[StructuralConstraint] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list, description="structural information the paper does not provide")

    @model_validator(mode="after")
    def _ids(self) -> ExperimentalEvidence:
        ids = [e.evidence_id for e in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence IDs must be unique")
        for c in self.constraints:
            unknown = set(c.evidence_ids) - set(ids)
            if unknown:
                raise ValueError(f"constraint {c.constraint_id} cites unknown evidence {sorted(unknown)}")
        return self

    @property
    def ids(self) -> set[str]:
        return {e.evidence_id for e in self.items} | {c.constraint_id for c in self.constraints}
