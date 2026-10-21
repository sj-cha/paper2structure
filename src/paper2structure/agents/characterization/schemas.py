"""Output of the characterization agents.

The inputs are the summarizer's stored models (``MaterialSystem`` and ``Evidence`` from
``paper2structure.agents.summarizer.schemas``): text evidence carries what the paper states,
figure evidence carries the image, caption and citing sentences. IDs in this report point
back to them: ``source_id`` is a figure ID (image file stem, e.g. 'p2_f1') or a text
evidence ID (e.g. 'E003'), and ``system_ids`` are the summarizer's system IDs ('S1', ...).
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


Confidence = Literal["high", "medium", "low"]


class Observation(BaseModel):
    observation_id: str = Field(description="Unique ID within the report, e.g. 'O1'.")
    source_id: str = Field(description="Figure ID of the image or evidence ID of the text record this observation comes from.")
    system_ids: list[str] = Field(default_factory=list, description="Systems the observation refers to, e.g. ['S2'].")
    quantity: str = Field(description="What was read, e.g. '2theta peak position', 'FWHM', 'd-spacing', 'chemical shift'.")
    value: str = Field(description="Value as read or stated; kept as text to allow ranges and '~'.")
    unit: Optional[str] = None
    read_from: Literal["printed_label", "axis_estimate", "visual", "stated_in_text"] = Field(
        description="'printed_label' if the value is written on the figure; 'axis_estimate' if read against axis ticks; "
                    "'visual' for qualitative features (e.g. a shoulder, an extra peak, peak broadening); "
                    "'stated_in_text' if the value or claim comes from a text record.",
    )
    uncertainty: Optional[str] = Field(default=None, description="Reading precision, e.g. '±0.1 deg' from tick spacing, or the error bar stated in the text.")
    note: str = Field(default="", description="Where in the source, e.g. 'trace 2 of 3, labelled (002)' or 'Rietveld result in Results'.")

    @property
    def source_type(self) -> Literal["figure", "text"]:
        return "text" if self.read_from == "stated_in_text" else "figure"


class Finding(BaseModel):
    claim: str = Field(description="One interpretive conclusion drawn from the data.")
    system_ids: list[str] = Field(default_factory=list, description="Systems the claim applies to.")
    evidence: list[str] = Field(description="Observation IDs supporting the claim.")
    confidence: Confidence


class StructuralConstraint(BaseModel):
    property: str = Field(description="Structural property constrained by the data, e.g. 'interlayer spacing', 'phase'.")
    system_ids: list[str] = Field(default_factory=list, description="Systems the constraint applies to; several for a series-wide constraint.")
    value: str = Field(description="Constrained value or range, as text.")
    unit: Optional[str] = None
    derivation: str = Field(default="", description="Relation or reasoning used, e.g. 'Bragg, lambda=1.5406 A, d = 2*d(002)'.")
    evidence: list[str] = Field(description="Observation IDs supporting the constraint.")
    confidence: Confidence


class CharacterizationReport(BaseModel):
    technique: str
    observations: list[Observation] = Field(
        default_factory=list,
        description="Values and features read from the figure images or stated in the text records.",
    )
    findings: list[Finding] = Field(default_factory=list)
    structural_constraints: list[StructuralConstraint] = Field(
        default_factory=list,
        description="Constraints usable as inputs to atomistic structure generation.",
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="Unreadable regions, ambiguities, inconsistencies with the text, or limits of the technique.",
    )
