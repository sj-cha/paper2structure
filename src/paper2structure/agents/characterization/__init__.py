"""Agent 2, characterization: turns the paper's experimental data into an evidence snapshot
(ExperimentalEvidence). Technique agents (XRD, NMR, XPS, IR, TEM, ...) go here as one module or
subpackage each, every one contributing EvidenceItems; until they exist, `analyze_evidence` is a
placeholder that loads a snapshot written beforehand."""
from paper2structure.agents.characterization.agent import analyze_evidence
from paper2structure.agents.characterization.schemas import EvidenceItem, ExperimentalEvidence, StructuralConstraint

__all__ = ["analyze_evidence", "EvidenceItem", "ExperimentalEvidence", "StructuralConstraint"]
