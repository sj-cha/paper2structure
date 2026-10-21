from typing import Literal, Optional

from pydantic import BaseModel, Field


Confidence = Literal["high", "medium", "low"]


# ---------------------------------------------------------------------------
# Input: one figure image plus the text that accompanies it.
# The upstream agent locates and tags figures; it does NOT extract values.
# ---------------------------------------------------------------------------

class FigureInput(BaseModel):
    figure_id: str = Field(description="Figure/panel identifier, e.g. 'Fig. 2a'.")
    image_path: str = Field(description="Path to the figure image (PNG/JPEG/WebP/GIF).")
    technique: str = Field(description="Technique tag used for routing, e.g. 'XRD'.")
    caption: str = Field(default="", description="Figure caption text from the paper.")
    material: Optional[str] = Field(default=None, description="Material/sample the figure refers to, if stated.")
    context: str = Field(default="", description="Relevant paper text, e.g. instrument settings or the paragraph citing the figure.")


# ---------------------------------------------------------------------------
# Input: what the paper's text says about one technique.
# Produced by the upstream summarizer; one record per passage or evidence item.
# ---------------------------------------------------------------------------

class TextInput(BaseModel):
    text_id: str = Field(description="Identifier for the passage, e.g. 'T1' or an upstream evidence ID.")
    technique: str = Field(description="Technique tag used for routing, e.g. 'XRD'.")
    text: str = Field(description="Excerpt or summary of what the paper states: results, refined values, assignments, instrument settings.")
    material: Optional[str] = Field(default=None, description="Material/sample the text refers to, if stated.")
    source: str = Field(default="", description="Where in the paper, e.g. 'page 3, Results' or 'Methods'.")
    figure_refs: list[str] = Field(default_factory=list, description="Figures the text cites, e.g. ['Fig. 1a'].")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    observation_id: str = Field(description="Unique ID within the report, e.g. 'O1'.")
    source_id: str = Field(description="figure_id of the image or text_id of the passage this observation comes from.")
    quantity: str = Field(description="What was read, e.g. '2theta peak position', 'FWHM', 'd-spacing', 'chemical shift'.")
    value: str = Field(description="Value as read or stated; kept as text to allow ranges and '~'.")
    unit: Optional[str] = None
    read_from: Literal["printed_label", "axis_estimate", "visual", "stated_in_text"] = Field(
        description="'printed_label' if the value is written on the figure; 'axis_estimate' if read against axis ticks; "
                    "'visual' for qualitative features (e.g. a shoulder, an extra peak, peak broadening); "
                    "'stated_in_text' if the value or claim comes from a text passage.",
    )
    uncertainty: Optional[str] = Field(default=None, description="Reading precision, e.g. '±0.1 deg' from tick spacing, or the error bar stated in the text.")
    note: str = Field(default="", description="Where in the source, e.g. 'trace 2 of 3, labelled (002)' or 'Rietveld result in Results'.")

    @property
    def source_type(self) -> Literal["figure", "text"]:
        return "text" if self.read_from == "stated_in_text" else "figure"


class Finding(BaseModel):
    claim: str = Field(description="One interpretive conclusion drawn from the data.")
    evidence: list[str] = Field(description="Observation IDs supporting the claim.")
    confidence: Confidence


class StructuralConstraint(BaseModel):
    property: str = Field(description="Structural property constrained by the data, e.g. 'interlayer spacing', 'phase'.")
    value: str = Field(description="Constrained value or range, as text.")
    unit: Optional[str] = None
    derivation: str = Field(default="", description="Relation or reasoning used, e.g. 'Bragg, lambda=1.5406 A, d = 2*d(002)'.")
    evidence: list[str] = Field(description="Observation IDs supporting the constraint.")
    confidence: Confidence


class CharacterizationReport(BaseModel):
    technique: str
    material: Optional[str] = None
    observations: list[Observation] = Field(
        default_factory=list,
        description="Values and features read from the figure images or stated in the text inputs.",
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
