"""Agent 4, the Construction agent: a pydantic-ai-harness Coder in a sandboxed workspace (workspace.py)
with its skills library (skills/); its reports must pass the validation gate (tools.geometry_checks).
The agent (agent.py: the harness Coder, the toolkit) is imported on first use, so the schemas and the
workspace helpers stay light."""
from paper2structure.agents.constructor.schemas import (BuiltStructure, ConstructionReport, FailedPlan,
                                                        StructureArtifact, TargetCheck)

__all__ = ["ConstructionDeps", "LLMConstructor", "BuiltStructure", "ConstructionReport", "FailedPlan",
           "StructureArtifact", "TargetCheck"]


def __getattr__(name: str):
    if name in ("ConstructionDeps", "LLMConstructor"):
        from paper2structure.agents.constructor import agent

        return getattr(agent, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
