"""Agent 2, characterization: one agent per technique (techniques/, keyed by the canonical names in
paper2structure.config.TECHNIQUES) reads the figures and text the summarizer tagged with it and
returns a CharacterizationReport; ExperimentalEvidence bundles the reports for Agents 3-5."""
from .agent import CharacterizationAgent
from .schemas import (
    CharacterizationReport,
    ExperimentalEvidence,
    Finding,
    Observation,
    StructuralConstraint,
)
from .spec import GENERIC, TechniqueSpec, generate_system_prompt
from .techniques import FTIR, REGISTRY, SEM, TEM, XRD, find_spec

__all__ = [
    "CharacterizationAgent",
    "CharacterizationReport",
    "ExperimentalEvidence",
    "Finding",
    "FTIR",
    "GENERIC",
    "Observation",
    "REGISTRY",
    "SEM",
    "StructuralConstraint",
    "TEM",
    "TechniqueSpec",
    "XRD",
    "find_spec",
    "generate_system_prompt",
]
