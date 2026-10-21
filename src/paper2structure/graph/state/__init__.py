"""State and settings shared by the orchestrator's nodes.

``OrchestratorState`` is what the nodes read and write. Everything in it is also saved under
``out_dir`` (the summarizer's output folder), so a later run can pick up where this one stopped:
for example rerun only the XRD agent and keep the other reports.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai.models import Model

from ...agents.characterization import CharacterizationReport
from ...agents.summarizer.schemas import PaperGraph


@dataclass
class OrchestratorState:
    out_dir: Path
    techniques: list[str] | None = None  # techniques to (re)run; None = every technique with evidence
    include_stubs: bool = False  # also run techniques whose spec is not filled in yet
    paper: PaperGraph | None = None
    reports: dict[str, CharacterizationReport] = field(default_factory=dict)  # canonical technique -> report
    errors: list[str] = field(default_factory=list)

    @property
    def paper_path(self) -> Path:
        return self.out_dir / "paper_graph.json"

    @property
    def characterization_dir(self) -> Path:
        return self.out_dir / "characterization"


@dataclass
class OrchestratorDeps:
    model: Model | str | None = None  # None = paper2structure.config default
    on_progress: Callable[[str], None] = print


__all__ = ["OrchestratorDeps", "OrchestratorState"]
