"""Pydantic data models for the paper summarizer.

Two kinds of models live here:

1. LLM output models (``*Output``): what we ask the LLM to return. pydantic-ai turns
   these into a JSON schema, sends it to the model, and validates the reply.
2. Stored models (``MaterialSystem``, ``Evidence``, ``PaperGraph``...): what we save to
   disk so that other agents (e.g. an XRD-analysis agent) can load them later.
"""

from __future__ import annotations

import json
from pathlib import PurePath
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ...config import TechniqueCategory, canonical_technique, technique_category


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
    technique: str = Field(description="A canonical technique name from the instructions, otherwise the paper's name (e.g. 'magnetization').")
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
    technique: str = Field(description="A canonical technique name from the instructions, otherwise the paper's name.")
    system_ids: list[str] = Field(description="Systems shown in this panel (use the given IDs).")
    description: str = Field(description="What the panel shows (samples, plot layout, insets), without reading any values off it.")
    axes: list[str] = Field(default_factory=list, description="Axis labels with units, e.g. ['2θ (deg)', 'Intensity (arb.)'].")
    image_id: str | None = Field(default=None, description="The id of the image the panel is in, when the figure comes as several images.")


class FigureOutput(BaseModel):
    """Output of step 3 for ONE figure (one image, or the crops of one figure)."""

    figure_label: str = Field(description="e.g. 'Fig. 2'.")
    panels: list[FigurePanel]


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
    technique_category: TechniqueCategory = Field(default="other", description="From the technique vocabulary; 'other' outside it.")
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
                data["technique_category"] = technique_category(canonical)
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

    def prompt_json(self) -> str:
        """The summary as Agents 3-5 read it: compact JSON without empty or default fields and without
        the run's provenance (source PDF, model). Text evidence is listed item by item; figure evidence
        is grouped per image under its figure ID (the source_id the characterization reports cite),
        with the caption and the sentences citing it once, then its panels."""
        derived = {"source_type", "technique_category", "image_path"}  # implied by where an item is listed
        text, figures = [], {}
        for e in self.evidence:
            if not e.image_path:
                text.append(e.model_dump(mode="json", exclude_defaults=True, exclude=derived))
                continue
            fig = figures.setdefault(e.image_path, {"figure": PurePath(e.image_path).stem, "caption": e.quote,
                                                    "cited_by": e.context, "panels": []})
            fig["panels"].append(e.model_dump(mode="json", exclude_defaults=True, exclude=derived | {"quote", "context"}))
        view = {"metadata": self.metadata.model_dump(mode="json", exclude_defaults=True),
                "systems": [s.model_dump(mode="json", exclude_defaults=True) for s in self.systems],
                "text_evidence": text,
                "figures": [{k: v for k, v in f.items() if v} for f in figures.values()]}
        return json.dumps(view, ensure_ascii=False, separators=(",", ":"))
