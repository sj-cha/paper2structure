from __future__ import annotations

from dataclasses import dataclass

from pydantic_graph import BaseNode, GraphRunContext

from ...agents.characterization import CharacterizationReport
from ...agents.summarizer.knowledge_graph import load_graph
from ..state import OrchestratorDeps, OrchestratorState
from .characterize import Characterize

Ctx = GraphRunContext[OrchestratorState, OrchestratorDeps]


@dataclass
class LoadPaper(BaseNode[OrchestratorState, OrchestratorDeps, dict[str, CharacterizationReport]]):
    """Load the summarizer's output and any reports saved by earlier runs. No LLM."""

    async def run(self, ctx: Ctx) -> Characterize:
        state = ctx.state
        if not state.paper_path.exists():
            raise FileNotFoundError(f"{state.paper_path} not found; run the summarizer on the paper first.")
        state.paper = load_graph(state.paper_path)
        for path in sorted(state.characterization_dir.glob("*.json")):
            report = CharacterizationReport.model_validate_json(path.read_text())
            state.reports[report.technique] = report
        ctx.deps.on_progress(
            f"Loaded {state.paper.metadata.title!r}: {len(state.paper.systems)} systems, "
            f"{len(state.paper.evidence)} evidence items, {len(state.reports)} saved reports."
        )
        return Characterize()
