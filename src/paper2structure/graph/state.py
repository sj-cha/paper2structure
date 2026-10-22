from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from pydantic_ai.models import Model

from pydantic import BaseModel, Field
from pydantic_core import to_json

from paper2structure.agents.characterization.schemas import ExperimentalEvidence
from paper2structure.agents.constructor.schemas import ConstructionReport, StructureArtifact
from paper2structure.agents.planner.schemas import PlanSet
from paper2structure.agents.reviser.schemas import ReportNarrative, RevisionDecision
from paper2structure.agents.summarizer.schemas import PaperGraph
from paper2structure.config import Paper2StructureConfig
from paper2structure.tools.schemas import ValidationReport

if TYPE_CHECKING:
    from paper2structure.agents.constructor import LLMConstructor
    from paper2structure.agents.planner import LLMPlanner
    from paper2structure.agents.reviser import LLMReviser


class ReconstructionState(BaseModel):
    """Typed, JSON-serializable graph state. Large data (structures, scripts) lives in the run
    directory and is referenced by path; agents and tools live in WorkflowDeps."""
    run_id: str
    summary: Optional[PaperGraph] = None  # Agent 1: metadata, material systems, text and figure evidence
    paper_dir: Optional[str] = None  # folder of paper_graph.json; its figure image paths are relative to it
    evidence: Optional[ExperimentalEvidence] = None  # Agent 2: one characterization report per technique
    evidence_version: int = 0
    plan_history: list[PlanSet] = Field(default_factory=list)
    revision: int = 0  # loops taken back through evidence -> planning -> construction
    attempt: int = 0  # construction attempts within the current revision
    pending_plan_ids: list[str] = Field(default_factory=list)
    artifacts: list[StructureArtifact] = Field(default_factory=list)  # every version ever built
    current_ids: list[str] = Field(default_factory=list)  # structures of the current revision
    failed_plans: dict[str, str] = Field(default_factory=dict)
    validation_reports: dict[str, ValidationReport] = Field(default_factory=dict)
    construction_reports: list[ConstructionReport] = Field(default_factory=list)
    revision_history: list[RevisionDecision] = Field(default_factory=list)
    report: Optional[ReportNarrative] = None  # the reviewer's text for final_report.md (write_report tool)
    review_feedback: str = ""
    repair_feedback: str = ""
    node_history: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    status: str = "running"

    @property
    def plans(self) -> Optional[PlanSet]:
        return self.plan_history[-1] if self.plan_history else None

    def artifact(self, structure_id: str) -> StructureArtifact:
        return next(a for a in self.artifacts if a.structure_id == structure_id)


class ArtifactStore:
    """Run directory: inputs, plans, construction workspaces, exported structures, reports and a
    JSON snapshot of the graph state after every node (pydantic-graph V2 has no built-in
    persistence). Nothing is overwritten across revisions: files carry revision/attempt labels."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self._snapshots = len(list((self.root / "state").glob("*.json")))  # continue numbering in a reused dir

    def path(self, *parts: str) -> Path:
        """A path in the run directory, with its parent directories created."""
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def rel(self, path: str | Path) -> str:
        return str(Path(path).resolve().relative_to(self.root))

    def write_json(self, rel_path: str, obj: Any) -> Path:
        p = self.path(rel_path)
        p.write_bytes(to_json(obj, indent=2))
        return p

    def snapshot(self, state: BaseModel, node: str) -> Path:
        self._snapshots += 1
        return self.write_json(f"state/{self._snapshots:02d}_{node}.json", state)


@dataclass
class WorkflowDeps:
    cfg: Paper2StructureConfig
    store: ArtifactStore
    planner: LLMPlanner
    constructor: LLMConstructor
    reviser: LLMReviser
    seed: int
    paper_model: Model | str | None = None  # Agents 1-2 (summarizer, characterization); None = cfg.campaign.model
    max_revisions: int = 1
    max_repairs: int = 2
