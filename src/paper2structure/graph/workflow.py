from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Optional

from pydantic_graph import GraphBuilder, StepContext

from paper2structure.agents.constructor import LLMConstructor
from paper2structure.agents.planner import LLMPlanner
from paper2structure.agents.reviser import LLMReviser
from paper2structure.config import Paper2StructureConfig
from paper2structure.graph.nodes import (AnalyzeEvidence, ConstructStructures, FinalizeResults, PlanStructures,
                                         ScientificRevision, SummarizePaper, ValidateStructures)
from paper2structure.graph.schemas import ReconstructionResult
from paper2structure.graph.state import ArtifactStore, ReconstructionState, WorkflowDeps

NODES = (SummarizePaper, AnalyzeEvidence, PlanStructures, ConstructStructures, ValidateStructures,
         ScientificRevision, FinalizeResults)


def build_graph():
    """SummarizePaper (PDF -> PaperGraph) -> AnalyzeEvidence (characterization reports) -> PlanStructures
    -> ConstructStructures <-> ValidateStructures -> ScientificRevision -> (AnalyzeEvidence | FinalizeResults)."""
    g = GraphBuilder(state_type=ReconstructionState, deps_type=WorkflowDeps, output_type=ReconstructionResult)

    @g.step
    async def start(ctx: StepContext[ReconstructionState, WorkflowDeps, None]) -> SummarizePaper:
        return SummarizePaper()

    g.add(*(g.node(n) for n in NODES), g.edge_from(g.start_node).to(start))
    return g.build()


def build_deps(cfg: Paper2StructureConfig, output_dir: str | Path, seed: int, max_revisions: int = 1,
               max_repairs: int = 2, models: Optional[dict] = None) -> WorkflowDeps:
    """Agents and run directory for one run. `models` may map "paper" (Agents 1-2), "planner",
    "constructor" and "reviser" to pydantic-ai Model instances or names (tests, "dry-run");
    otherwise cfg.campaign.model is used."""
    models = models or {}
    c = cfg.campaign
    planner = LLMPlanner(models.get("planner", c.model), c.reasoning_effort, max_plans=cfg.planning.max_plans,
                         max_atoms=cfg.planning.max_atoms, min_vacuum=cfg.validation.min_vacuum,
                         overrides=cfg.overrides, max_retry=c.max_retry)
    constructor = LLMConstructor(models.get("constructor", c.model), c.reasoning_effort, cfg)
    reviser = LLMReviser(models.get("reviser", c.model), c.reasoning_effort, max_retry=c.max_retry,
                         read_scripts=cfg.revision.read_scripts)
    return WorkflowDeps(cfg=cfg, store=ArtifactStore(output_dir), planner=planner, constructor=constructor,
                        reviser=reviser, seed=int(seed), paper_model=models.get("paper"),
                        max_revisions=int(max_revisions), max_repairs=int(max_repairs))


async def run_workflow(deps: WorkflowDeps, run_id: Optional[str] = None) -> ReconstructionResult:
    graph = build_graph()
    deps.store.path("workflow.mmd").write_text(graph.render(direction="TB"))
    state = ReconstructionState(run_id=run_id or f"{deps.store.root.name}-{_dt.datetime.now():%Y%m%d-%H%M%S}")
    return await graph.run(state=state, deps=deps)
