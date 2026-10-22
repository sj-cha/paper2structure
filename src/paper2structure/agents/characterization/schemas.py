"""Output of the characterization agents.

The inputs are the summarizer's stored models (``MaterialSystem`` and ``Evidence`` from
``paper2structure.agents.summarizer.schemas``): text evidence carries what the paper states,
figure evidence carries the image, caption and citing sentences. IDs in this report point
back to them: ``source_id`` is a figure ID (image file stem, e.g. 'p2_f1') or a text
evidence ID (e.g. 'E003'), and ``system_ids`` are the summarizer's system IDs ('S1', ...).
"""

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel, Field, model_validator
from pydantic.json_schema import SkipJsonSchema

if TYPE_CHECKING:
    from ..summarizer.schemas import PaperGraph

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


class Finding(BaseModel):
    claim: str = Field(description="One interpretive conclusion drawn from the data.")
    system_ids: list[str] = Field(default_factory=list, description="Systems the claim applies to.")
    evidence: list[str] = Field(description="Observation IDs supporting the claim.")
    confidence: Confidence


class StructuralConstraint(BaseModel):
    # 'C<k>', its position in the report as the characterization agent wrote it; kept when the reports
    # are filtered, so the claim IDs plans cite ('XRD:C3') never change. Not part of the model's output
    # schema.
    constraint_id: SkipJsonSchema[Optional[str]] = None
    property: str = Field(description="Structural property constrained by the data, e.g. 'interlayer spacing', 'phase'.")
    system_ids: list[str] = Field(default_factory=list, description="Systems the constraint applies to; several for a series-wide constraint.")
    value: str = Field(description="Constrained value or range, as text.")
    unit: Optional[str] = None
    derivation: str = Field(default="", description="Relation or reasoning used, e.g. 'Bragg, lambda=1.5406 A, d = 2*d(002)'.")
    evidence: list[str] = Field(description="Observation IDs supporting the constraint.")
    confidence: Confidence


class CharacterizationReport(BaseModel):
    technique: SkipJsonSchema[str] = ""  # the agent's spec name, set by CharacterizationAgent.analyze
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

    @model_validator(mode="after")
    def _number_constraints(self) -> "CharacterizationReport":
        for k, c in enumerate(self.structural_constraints, 1):
            if c.constraint_id is None:
                c.constraint_id = f"C{k}"
        return self


def _concerns(system_ids: list[str], keep: set[str]) -> bool:
    return not system_ids or bool(keep & set(system_ids))


class ExperimentalEvidence(BaseModel):
    """The characterization agents' output as the rest of the workflow sees it: one CharacterizationReport per
    technique (canonical name) and the techniques found in the paper but not analyzed."""

    reports: dict[str, CharacterizationReport] = Field(default_factory=dict)
    skipped: dict[str, str] = Field(default_factory=dict, description="Technique -> why it was not analyzed.")
    errors: list[str] = Field(default_factory=list)

    @property
    def ids(self) -> set[str]:
        """Citable observation IDs, qualified by technique: 'XRD:O3'."""
        return {f"{t}:{o.observation_id}" for t, r in self.reports.items() for o in r.observations}

    def prompt_json(self) -> str:
        """The reports as the Planner and the Reviewer read them: compact JSON without default fields and without
        the structural constraints, which the claims list gives in full (ID, systems, value,
        confidence, cited observations, derivation)."""
        return self.model_dump_json(exclude_defaults=True, exclude={"reports": {"__all__": {"structural_constraints"}}})

    def focused(self, system_ids: set[str]) -> "ExperimentalEvidence":
        """The reports restricted to these systems: the observations, findings and constraints about
        them or about no system in particular, plus every observation a kept finding or constraint
        cites. IDs (observation, constraint) are unchanged, so claim and citation IDs stay valid."""
        reports = {}
        for technique, r in self.reports.items():
            findings = [f for f in r.findings if _concerns(f.system_ids, system_ids)]
            constraints = [c for c in r.structural_constraints if _concerns(c.system_ids, system_ids)]
            cited = {o for item in findings + constraints for o in item.evidence}
            observations = [o for o in r.observations
                            if _concerns(o.system_ids, system_ids) or o.observation_id in cited]
            reports[technique] = r.model_copy(update={"observations": observations, "findings": findings,
                                                      "structural_constraints": constraints})
        return self.model_copy(update={"reports": reports})

    def mismatches(self, summary: "PaperGraph") -> tuple[list[str], list[str]]:
        """Whether saved reports belong to this summary: (errors, warnings). Reports cite the
        summary's evidence and figure IDs, which are numbered per summarizer run. An error: a report
        whose observations cite more of another technique's evidence than its own (reports made from
        another summary). Warnings, counted per kind: sources and systems the summary does not have
        (expected when the summary was narrowed to some systems)."""
        techniques: dict[str, set[str]] = {}
        for e in summary.evidence:
            techniques.setdefault(e.evidence_id, set()).add(e.technique)
            if e.image_path:
                techniques.setdefault(Path(e.image_path).stem, set()).add(e.technique)
        systems = {s.system_id for s in summary.systems}
        errors, counts = [], Counter()
        for technique, r in sorted(self.reports.items()):
            own = foreign = 0
            for o in r.observations:
                found = techniques.get(o.source_id)
                if found is None:
                    counts["observations cite evidence the summary does not have"] += 1
                elif technique in found:
                    own += 1
                else:
                    foreign += 1
            if foreign > own:
                errors.append(f"{technique}: {foreign} of {len(r.observations)} observations cite other techniques' "
                              "evidence in this summary; the reports were made from another summary")
            items = r.observations + r.findings + r.structural_constraints
            counts["report items concern systems the summary does not have"] += sum(
                1 for x in items if x.system_ids and not set(x.system_ids) & systems)
        warnings = [f"{n} {kind}" for kind, n in counts.items() if n]
        return errors, warnings

    @classmethod
    def load(cls, folder: str | Path) -> "ExperimentalEvidence":
        """The reports saved in `folder` (one CharacterizationReport JSON per technique)."""
        paths = sorted(Path(folder).glob("*.json"))
        if not paths:
            raise FileNotFoundError(f"no characterization reports (*.json) in {folder}")
        reports = [CharacterizationReport.model_validate_json(p.read_text()) for p in paths]
        return cls(reports={r.technique: r for r in reports})
