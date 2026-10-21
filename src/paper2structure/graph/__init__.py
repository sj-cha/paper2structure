"""The orchestrator: a plain (non-LLM) pydantic-graph that runs the agents after the summarizer.

    LoadPaper -> Characterize -> End        (hypothesis -> constructor -> revision to follow)

The summarizer runs once per paper, outside this graph, and writes ``<out_dir>/paper_graph.json``.
The orchestrator reads that folder and can be rerun as often as needed, e.g. for one technique::

    python -m paper2structure.graph outputs/<paper>                 # every implemented technique
    python -m paper2structure.graph outputs/<paper> --techniques XRD
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from pydantic_ai.models import Model
from pydantic_graph import GraphBuilder

from ..agents.characterization import CharacterizationReport
from .nodes import Characterize, LoadPaper
from .state import OrchestratorDeps, OrchestratorState

_builder = GraphBuilder(
    name="orchestrator",
    state_type=OrchestratorState,
    deps_type=OrchestratorDeps,
    input_type=LoadPaper,
    output_type=dict[str, CharacterizationReport],
)
_builder.add(
    _builder.edge_from(_builder.start_node).to(LoadPaper),
    _builder.node(LoadPaper),
    _builder.node(Characterize),
)
orchestrator_graph = _builder.build()


async def run_orchestrator(
    out_dir: str | Path,
    techniques: list[str] | None = None,
    include_stubs: bool = False,
    model: Model | str | None = None,
    on_progress: Callable[[str], None] = print,
) -> OrchestratorState:
    """Run the orchestrator on a summarizer output folder and return the final state."""
    state = OrchestratorState(out_dir=Path(out_dir), techniques=techniques, include_stubs=include_stubs)
    deps = OrchestratorDeps(model=model, on_progress=on_progress)
    await orchestrator_graph.run(state=state, deps=deps, inputs=LoadPaper())
    return state


def run_orchestrator_sync(*args, **kwargs) -> OrchestratorState:
    return asyncio.run(run_orchestrator(*args, **kwargs))


__all__ = ["OrchestratorDeps", "OrchestratorState", "orchestrator_graph", "run_orchestrator", "run_orchestrator_sync"]
