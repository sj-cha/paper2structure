from __future__ import annotations

import asyncio
from dataclasses import dataclass

from pydantic_graph import BaseNode, End, GraphRunContext

from ...agents.characterization import CharacterizationAgent, CharacterizationReport, TechniqueSpec, find_spec
from ...config import canonical_technique
from ..state import OrchestratorDeps, OrchestratorState

Ctx = GraphRunContext[OrchestratorState, OrchestratorDeps]


@dataclass
class Characterize(BaseNode[OrchestratorState, OrchestratorDeps, dict[str, CharacterizationReport]]):
    """Run one characterization agent per technique found in the paper, concurrently.

    Techniques without a spec, and stub specs (unless ``include_stubs``), are skipped.
    The hypothesis -> constructor -> revision nodes will follow this one.
    """

    async def run(self, ctx: Ctx) -> End[dict[str, CharacterizationReport]]:
        state = ctx.state
        specs = self._select_specs(ctx)
        if specs:
            ctx.deps.on_progress(f"Characterizing: {', '.join(s.name for s in specs)}")
            results = await asyncio.gather(*(self._run_one(ctx, spec) for spec in specs), return_exceptions=True)
            for spec, result in zip(specs, results):
                if isinstance(result, BaseException):
                    state.errors.append(f"{spec.name} characterization failed: {result!r}")
                    ctx.deps.on_progress(f"  ! {state.errors[-1]}")
                else:
                    state.reports[spec.name] = result
        else:
            ctx.deps.on_progress("No techniques to characterize.")
        return End(state.reports)

    @staticmethod
    def _select_specs(ctx: Ctx) -> list[TechniqueSpec]:
        state = ctx.state
        found = sorted({e.technique for e in state.paper.evidence})
        if state.techniques is not None:
            requested = {canonical_technique(t) or t for t in state.techniques}
            found = [t for t in found if t in requested]

        specs: list[TechniqueSpec] = []
        for technique in found:
            spec = find_spec(technique)
            if spec is None:
                ctx.deps.on_progress(f"  - skipping {technique}: not in the technique vocabulary")
            elif not spec.implemented and not state.include_stubs:
                ctx.deps.on_progress(f"  - skipping {technique}: spec is a stub")
            else:
                specs.append(spec)
        return specs

    @staticmethod
    async def _run_one(ctx: Ctx, spec: TechniqueSpec) -> CharacterizationReport:
        state = ctx.state
        report = await CharacterizationAgent(spec, model=ctx.deps.model).analyze(
            state.paper.evidence,
            state.paper.systems,
            base_dir=state.out_dir,
            filename=state.characterization_dir / f"{spec.name.replace(' ', '_')}.json",
        )
        ctx.deps.on_progress(
            f"  {spec.name}: {len(report.observations)} observations, "
            f"{len(report.structural_constraints)} constraints"
        )
        return report
