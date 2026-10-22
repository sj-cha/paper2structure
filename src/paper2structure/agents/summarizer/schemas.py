"""Pydantic data models for the paper summarizer.

Two kinds of models live here:

1. LLM output models (``*Output``): what we ask the LLM to return. pydantic-ai turns
   these into a JSON schema, sends it to the model, and validates the reply.
2. Stored models (``MaterialSystem``, ``Evidence``, ``PaperGraph``...): what we save to
   disk so that other agents (e.g. an XRD-analysis agent) can load them later.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ...config import TECHNIQUES, TechniqueCategory, canonical_technique


# --------------------------------------------------------------------------------------
# Small building blocks
# --------------------------------------------------------------------------------------


class Measurement(BaseModel):
    """One number (or short value) read from the text or a figure."""

    quantity: str = Field(description="What was measured, e.g. 'lattice parameter a', 'space group', 'grain size'.")
    value: str = Field(description="The value as written, e.g. '5.48', 'Pnma', '~1'. Keep as text to avoid losing ranges or '~'.")
    unit: str | None = Field(default=None, description="Unit, e.g. 'Å', 'nm', 'K'. None if unitless.")
    uncertainty: str | None = Field(default=None, description="Error bar if reported, e.g. '0.002'.")
    approximate: bool = Field(default=False, description="True if read off a plot by eye rather than stated in text/table.")


class SynthesisInfo(BaseModel):
    method: str | None = Field(default=None, description="e.g. 'solid-state reaction', 'sol-gel', 'hydrothermal'.")
    precursors: list[str] = Field(default_factory=list, description="Starting materials with purity if given.")
    conditions: list[str] = Field(default_factory=list, description="Temperatures, times, atmospheres, pressures, e.g. 'fired at 1250 °C for 5-8 h in air'.")
    notes: str | None = None


class StructureClaims(BaseModel):
    """Structural facts the paper states for a material (the 'answer' structure generation needs)."""

    crystal_system: str | None = Field(default=None, description="e.g. 'orthorhombic'.")
    space_group: str | None = Field(default=None, description="e.g. 'Pnma'.")
    lattice_parameters: list[Measurement] = Field(default_factory=list, description="a, b, c, alpha, beta, gamma, volume.")
    site_occupancy: list[str] = Field(default_factory=list, description="e.g. 'Bi substitutes on La site'.")
    phase_purity: str | None = Field(default=None, description="e.g. 'single phase', 'Bi2O3 impurity on surface'.")
    other: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# LLM output models
# --------------------------------------------------------------------------------------


class PaperMetadata(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    doi: str | None = None
    year: int | None = None
    summary: str = Field(description="3-5 sentence plain-language summary of the paper.")


class MaterialSystemCandidate(BaseModel):
    name: str = Field(description="Specific formula or name as used in the paper, e.g. 'La0.8Bi0.2CrO3'.")
    family: str | None = Field(default=None, description="Parent family/series, e.g. 'La1-xBixCrO3'.")
    composition_variables: dict[str, str] = Field(default_factory=dict, description="Variables that define this member, e.g. {'x': '0.2'}.")
    material_class: str | None = Field(default=None, description="e.g. 'perovskite oxide', 'MOF', 'thin film', 'alloy'.")
    form: str | None = Field(default=None, description="e.g. 'polycrystalline pellet', 'thin film on SrTiO3', 'nanoparticles'.")
    why_distinct: str = Field(description="Short reason this counts as its own system (different composition, phase, substrate, ...).")


class SystemsOutput(BaseModel):
    """Output of step 1: paper metadata + every distinct material system that was evaluated."""

    metadata: PaperMetadata
    systems: list[MaterialSystemCandidate]


class TextEvidenceItem(BaseModel):
    system_ids: list[str] = Field(description="IDs (e.g. 'S1') of the systems this evidence applies to. Use several IDs if it applies to a whole series.")
    technique: str = Field(description="Canonical technique name from the list in the instructions, e.g. 'XRD', 'TEM', 'SEM', 'XPS', 'Raman', 'magnetization'.")
    technique_category: TechniqueCategory
    summary: str = Field(description="What this evidence says, in one or two sentences.")
    measurements: list[Measurement] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list, description="Instrument/settings, e.g. 'Cu-Kα', '2θ 10-110°', 'FullProf', '200 kV'.")
    figure_refs: list[str] = Field(default_factory=list, description="Figures the text points to, e.g. ['Fig. 1a'].")
    quote: str | None = Field(default=None, description="Short verbatim quote from the paper supporting this item.")
    page: int | None = Field(default=None, description="1-based page number.")


class TextEvidenceOutput(BaseModel):
    """Output of step 2 for ONE system: synthesis, structure claims, and per-technique evidence."""

    synthesis: SynthesisInfo
    structure: StructureClaims
    evidence: list[TextEvidenceItem]


class FigurePanel(BaseModel):
    panel_label: str | None = Field(default=None, description="'a', 'b', ... or None if single-panel.")
    plot_type: str = Field(description="e.g. 'XRD pattern', 'Rietveld fit', 'lattice parameter vs composition', 'TEM micrograph', 'SAED pattern'.")
    technique: str
    technique_category: TechniqueCategory
    system_ids: list[str] = Field(description="Systems shown in this panel (use the given IDs).")
    description: str = Field(description="What the panel shows (samples, plot layout, insets), without reading any values off it.")
    axes: list[str] = Field(default_factory=list, description="Axis labels with units, e.g. ['2θ (deg)', 'Intensity (arb.)'].")


class FigureOutput(BaseModel):
    """Output of step 3 for ONE figure image."""

    figure_label: str = Field(description="e.g. 'Fig. 2'.")
    panels: list[FigurePanel]
    useful_for_structure: bool = Field(description="True if this figure carries information useful to build an atomic structure.")


# --------------------------------------------------------------------------------------
# Stored models (what other agents read)
# --------------------------------------------------------------------------------------


class MaterialSystem(BaseModel):
    system_id: str
    name: str
    family: str | None = None
    composition_variables: dict[str, str] = Field(default_factory=dict)
    material_class: str | None = None
    form: str | None = None
    synthesis: SynthesisInfo = Field(default_factory=SynthesisInfo)
    structure: StructureClaims = Field(default_factory=StructureClaims)


class Evidence(BaseModel):
    """One text statement or one figure panel about one technique.

    Text evidence carries what the paper states (``summary``, ``quote``, ``measurements``,
    ``conditions``). Figure evidence carries only the image, caption (``quote``) and the
    sentences citing the figure (``context``); values are read from the image by the
    characterization agents, not here.
    """

    evidence_id: str
    system_ids: list[str]
    technique: str = Field(description="Canonical technique name from paper2structure.config.TECHNIQUES when one fits.")
    technique_as_written: str | None = Field(default=None, description="Technique name as the paper/LLM wrote it, e.g. 'HRTEM' for technique 'TEM'.")
    technique_category: TechniqueCategory
    source_type: Literal["text", "figure"]
    source_ref: str = Field(description="Where it came from, e.g. 'page 2' or 'Fig. 1 (b)'.")
    summary: str
    measurements: list[Measurement] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    figure_refs: list[str] = Field(default_factory=list)
    quote: str | None = None
    context: list[str] = Field(default_factory=list, description="Body-text sentences that cite this figure, for figure evidence.")
    image_path: str | None = Field(default=None, description="Path (relative to the output folder) of the figure image, for figure evidence.")

    @model_validator(mode="before")
    @classmethod
    def _canonical_technique(cls, data):
        """Map aliases (HRTEM, Rietveld refinement, FT-IR, ...) to the canonical name and its category."""
        if isinstance(data, dict) and data.get("technique"):
            canonical = canonical_technique(data["technique"])
            if canonical:
                data = dict(data)
                if canonical != data["technique"]:
                    data.setdefault("technique_as_written", data["technique"])
                data["technique"] = canonical
                data["technique_category"] = TECHNIQUES[canonical][0]
        return data


class PaperGraph(BaseModel):
    """Everything extracted from one paper. Saved as ``paper_graph.json``. The knowledge graph
    (paper -> system -> evidence -> technique / figure) is implicit: evidence names its systems,
    technique and figure image. Files written by older versions also carry ``nodes``/``edges``,
    which are ignored on loading.
    """

    source_pdf: str
    model: str
    metadata: PaperMetadata
    systems: list[MaterialSystem]
    evidence: list[Evidence]

    def evidence_for(self, technique: str | None = None, system_id: str | None = None, category: str | None = None) -> list[Evidence]:
        """Filter evidence, e.g. ``graph.evidence_for(technique="XRD")``. Matching is case-insensitive substring."""
        out = self.evidence
        if technique:
            out = [e for e in out if technique.lower() in e.technique.lower()]
        if category:
            out = [e for e in out if e.technique_category == category]
        if system_id:
            out = [e for e in out if system_id in e.system_ids]
        return out
