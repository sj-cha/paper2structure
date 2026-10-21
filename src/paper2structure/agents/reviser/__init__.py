"""Agent 5, the scientific reviser: judges the structures and writes the final report."""
from paper2structure.agents.reviser.agent import LLMReviser, RevisionDeps, save_report, save_structures_readme
from paper2structure.agents.reviser.schemas import ReportNarrative, RevisionAction, RevisionDecision, StructureVerdict

__all__ = ["LLMReviser", "RevisionDeps", "save_report", "save_structures_readme", "ReportNarrative", "RevisionAction",
           "RevisionDecision", "StructureVerdict"]
