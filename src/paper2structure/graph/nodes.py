"""Workflow nodes. Each node's `run` return annotation declares its outgoing edges."""
from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from ase.io import read
from pydantic_ai.models import Model
from pydantic_graph import BaseNode, End, GraphRunContext

from paper2structure.agents.characterization import CharacterizationAgent, ExperimentalEvidence, find_spec
from paper2structure.agents.constructor.agent import WORKSPACE, ConstructionDeps
from paper2structure.agents.constructor.schemas import ConstructionReport, StructureArtifact
from paper2structure.agents.reviser import save_report, save_structures_readme
from paper2structure.agents.summarizer import PaperGraph, load_graph, summarize_paper
from paper2structure.config import CharacterizationConfig, canonical_technique, get_model
from paper2structure.graph.schemas import ReconstructionResult
from paper2structure.graph.state import ReconstructionState, WorkflowDeps
from paper2structure.tools._geometry import get_molecules
from paper2structure.tools.geometry_checks import check_roundtrip
from paper2structure.tools.render import render_png
from paper2structure.tools.structure_export import describe, export_structure

log = logging.getLogger(__name__)
Ctx = GraphRunContext[ReconstructionState, WorkflowDeps]


def _enter(ctx: Ctx, node: str) -> None:
    ctx.state.node_history.append(node)
    log.info("[%s] revision %d, attempt %d", node, ctx.state.revision, ctx.state.attempt)


def _checkpoint(ctx: Ctx, node: str) -> None:
    ctx.deps.store.snapshot(ctx.state, node)


def _paper_model(ctx: Ctx) -> Model | str:
    model = ctx.deps.paper_model
    return model if isinstance(model, Model) else get_model(model or ctx.deps.cfg.campaign.model)


@dataclass
class SummarizePaper(BaseNode[ReconstructionState, WorkflowDeps, ReconstructionResult]):
    """Agent 1: reads the PDF (text, figures, material systems) into a PaperGraph saved under
    paper/, or loads the paper_graph.json of an earlier run. Runs once."""

    async def run(self, ctx: Ctx) -> AnalyzeEvidence | FinalizeResults:
        _enter(ctx, "SummarizePaper")
        st, deps, inputs = ctx.state, ctx.deps, ctx.deps.cfg.inputs
        try:
            if inputs.paper_graph:
                path = Path(inputs.paper_graph).resolve()
                st.summary, st.paper_dir = load_graph(path), str(path.parent)
            elif inputs.pdf:
                out = deps.store.path("paper", "paper_graph.json").parent
                (out / "errors.txt").unlink(missing_ok=True)  # the summarizer only writes it when steps fail
                sc = deps.cfg.summarizer
                st.summary = await summarize_paper(inputs.pdf, out, _paper_model(ctx), on_progress=log.info,
                                                   analyze_figures=sc.analyze_figures, max_figures=sc.max_figures)
                st.paper_dir = str(out)
            else:
                raise ValueError("the config needs inputs.pdf or inputs.paper_graph")
            if not st.summary.systems:
                raise ValueError("the summarizer found no material systems in the paper")
        except Exception as e:  # noqa: BLE001
            st.errors.append(f"summarization failed: {type(e).__name__}: {e}")
            _checkpoint(ctx, "SummarizePaper")
            return FinalizeResults(status="failed")
        failed_steps = Path(st.paper_dir) / "errors.txt"  # per-system or per-figure steps the summarizer skipped
        if failed_steps.exists():
            st.errors += [f"summarizer: {line}" for line in failed_steps.read_text().splitlines() if line.strip()]
        if not st.summary.evidence:
            st.errors.append("summarizer: no evidence was extracted from the paper")
        log.info("[SummarizePaper] %r: %d systems, %d evidence items", st.summary.metadata.title,
                 len(st.summary.systems), len(st.summary.evidence))
        _checkpoint(ctx, "SummarizePaper")
        return AnalyzeEvidence()


async def characterize(paper: PaperGraph, paper_dir: Path, out_dir: Path, model: Model | str,
                       cc: CharacterizationConfig) -> ExperimentalEvidence:
    """One characterization agent per technique found in the paper's evidence, concurrently; each
    report is also saved as out_dir/<technique>.json. Techniques outside the vocabulary, and stub
    specs unless cc.include_stubs, are skipped."""
    evidence = ExperimentalEvidence()
    found = sorted({e.technique for e in paper.evidence})
    if cc.techniques is not None:
        requested = {canonical_technique(t) or t for t in cc.techniques}
        found = [t for t in found if t in requested]
    specs = []
    for technique in found:
        spec = find_spec(technique)
        if spec is None:
            evidence.skipped[technique] = "not in the technique vocabulary"
        elif not spec.implemented and not cc.include_stubs:
            evidence.skipped[technique] = "spec is a stub"
        else:
            specs.append(spec)
    log.info("[AnalyzeEvidence] characterizing %s; skipped %s", [s.name for s in specs] or "nothing",
             evidence.skipped or "none")
    results = await asyncio.gather(*(
        CharacterizationAgent(s, model=model, reasoning_effort=cc.reasoning_effort, output_retries=cc.output_retries)
        .analyze(paper.evidence, paper.systems, base_dir=paper_dir, filename=out_dir / f"{s.name.replace(' ', '_')}.json")
        for s in specs), return_exceptions=True)
    for spec, result in zip(specs, results):
        if isinstance(result, BaseException):
            evidence.errors.append(f"{spec.name} characterization failed: {type(result).__name__}: {result}")
            log.warning("[AnalyzeEvidence] %s", evidence.errors[-1])
        else:
            evidence.reports[spec.name] = result
            log.info("[AnalyzeEvidence] %s: %d observations, %d structural constraints", spec.name,
                     len(result.observations), len(result.structural_constraints))
    return evidence


@dataclass
class AnalyzeEvidence(BaseNode[ReconstructionState, WorkflowDeps, ReconstructionResult]):
    """Agent 2: one characterization agent per technique in the paper's evidence (or the reports
    saved in inputs.characterization). Re-entered when the reviewer finds that the structures do not
    resemble the paper: the reports are kept and the reviewer's feedback travels on to the planner."""

    async def run(self, ctx: Ctx) -> PlanStructures | FinalizeResults:
        _enter(ctx, "AnalyzeEvidence")
        st, deps = ctx.state, ctx.deps
        try:
            if st.evidence is not None:
                log.info("[AnalyzeEvidence] revision %d: characterization reports kept; reviewer feedback "
                         "forwarded to the planner", st.revision)
            elif deps.cfg.inputs.characterization:
                st.evidence = ExperimentalEvidence.load(deps.cfg.inputs.characterization)
            else:
                st.evidence = await characterize(st.summary, Path(st.paper_dir), deps.store.root / "characterization",
                                                 _paper_model(ctx), deps.cfg.characterization)
        except Exception as e:  # noqa: BLE001
            st.errors.append(f"characterization failed: {type(e).__name__}: {e}")
            _checkpoint(ctx, "AnalyzeEvidence")
            return FinalizeResults(status="failed")
        st.errors.extend(e for e in st.evidence.errors if e not in st.errors)
        st.evidence_version += 1
        deps.store.write_json(f"inputs/experimental_evidence_v{st.evidence_version}.json", st.evidence)
        _checkpoint(ctx, "AnalyzeEvidence")
        return PlanStructures()


@dataclass
class PlanStructures(BaseNode[ReconstructionState, WorkflowDeps, ReconstructionResult]):
    """Agent 3: structural hypotheses as construction plans."""

    async def run(self, ctx: Ctx) -> ConstructStructures | FinalizeResults:
        _enter(ctx, "PlanStructures")
        st = ctx.state
        try:
            plans = await ctx.deps.planner.plan(st.summary, st.evidence, st.plan_history, st.review_feedback)
        except Exception as e:  # noqa: BLE001
            st.errors.append(f"planning failed: {type(e).__name__}: {e}")
            _checkpoint(ctx, "PlanStructures")
            return FinalizeResults(status="failed")
        st.plan_history.append(plans)
        st.pending_plan_ids = [p.plan_id for p in plans.plans]
        st.attempt, st.repair_feedback, st.current_ids, st.failed_plans = 0, "", [], {}
        ctx.deps.store.write_json(f"plans/plans_r{st.revision}.json", plans)
        _checkpoint(ctx, "PlanStructures")
        return ConstructStructures()


@dataclass
class ConstructStructures(BaseNode[ReconstructionState, WorkflowDeps, ReconstructionResult]):
    """Agent 4: one coding-agent run per pending plan, concurrently, each in its own sandboxed
    workspace with its own request budget; every reported structure is the output of its script
    re-run from scratch, and that script is saved with it. A plan whose run fails, or that the agent
    reports as not built, is recorded with the reason; ValidateStructures sends it back (bounded
    repair loop) with the scripts of the attempt."""

    async def run(self, ctx: Ctx) -> ValidateStructures:
        _enter(ctx, "ConstructStructures")
        st = ctx.state
        pending = list(st.pending_plan_ids)
        results = await asyncio.gather(*(self._build(ctx, pid) for pid in pending), return_exceptions=True)
        for pid, result in zip(pending, results):
            if isinstance(result, BaseException):
                st.failed_plans[pid] = f"construction failed: {type(result).__name__}: {result}"
                st.errors.append(f"{pid} (revision {st.revision}, attempt {st.attempt}): {st.failed_plans[pid]}")
            else:
                self._register(ctx, *result)
        log.info("[ConstructStructures] built %d structures; not built: %s", len(st.current_ids),
                 sorted(p for p in pending if p in st.failed_plans) or "none")
        ctx.deps.store.write_json("artifacts.json", st.artifacts)
        _checkpoint(ctx, "ConstructStructures")
        return ValidateStructures()

    @staticmethod
    async def _build(ctx: Ctx, plan_id: str) -> tuple[ConstructionReport, ConstructionDeps]:
        st, store = ctx.state, ctx.deps.store
        previous = {}
        if st.repair_feedback:
            previous = {f"{a.structure_id}.py": (store.root / a.script).read_text()
                        for a in map(st.artifact, st.current_ids) if a.plan_id == plan_id}
            if not previous and st.attempt > 0:  # not built last time: resume from that attempt's scripts
                last = store.root / "construction" / f"r{st.revision}_a{st.attempt - 1}" / plan_id / WORKSPACE
                previous = {p.name: p.read_text() for p in sorted(last.glob("*.py"))}
        workdir = store.path("construction", f"r{st.revision}_a{st.attempt}", plan_id)
        return await ctx.deps.constructor.construct(st.plans, [plan_id], workdir, ctx.deps.seed, st.repair_feedback,
                                                    previous, st.summary, st.evidence)

    @staticmethod
    def _register(ctx: Ctx, report: ConstructionReport, cdeps: ConstructionDeps) -> None:
        """Export the verified structures of one plan's report and record its unbuilt plans."""
        st, store, cfg = ctx.state, ctx.deps.store, ctx.deps.cfg
        st.construction_reports.append(report)
        rebuilt = {b.plan_id for b in report.structures}
        st.current_ids = [i for i in st.current_ids if st.artifact(i).plan_id not in rebuilt]
        for pid in rebuilt:
            st.failed_plans.pop(pid, None)
        taken = {a.structure_id for a in st.artifacts}
        for built in report.structures:
            sid = f"{built.name}_r{st.revision}a{st.attempt}"
            if sid in taken:  # two plans' runs chose the same name
                sid = f"{built.plan_id}_{sid}"
            try:
                source = cdeps.verified[built.name]
                atoms = read(source)
                files = export_structure(atoms, store.path("structures"), sid, formats=cfg.export.formats)
                image = store.path("structures", f"{sid}.png")
                render_png(atoms, image)
                script = store.path("structures", f"{sid}.py")
                shutil.copyfile(cdeps.scripts[built.name], script)
                artifact = StructureArtifact(
                    structure_id=sid, plan_id=built.plan_id, name=built.name, revision=st.revision,
                    attempt=st.attempt, files={k: store.rel(v) for k, v in files.items()}, source=store.rel(source),
                    script=store.rel(script), image=store.rel(image), method=built.method,
                    target_checks=built.target_checks, formula=atoms.get_chemical_formula(), n_atoms=len(atoms),
                    cell=atoms.cell.array.round(6).tolist(), pbc=tuple(bool(p) for p in atoms.pbc),
                    n_molecules=len(get_molecules(atoms)), description=describe(atoms), deviations=built.deviations,
                    notes=built.notes)
            except Exception as e:  # noqa: BLE001 - one unexportable structure must not end the run
                st.failed_plans[built.plan_id] = f"{built.name} could not be exported: {type(e).__name__}: {e}"
                st.errors.append(f"{built.name} (plan {built.plan_id}): {st.failed_plans[built.plan_id]}")
                continue
            st.validation_reports[sid] = cdeps.reports[built.name].model_copy(update={"structure_id": sid})
            st.artifacts.append(artifact)
            st.current_ids.append(sid)
            taken.add(sid)
        for f in report.failed:
            st.failed_plans[f.plan_id] = f.reason


@dataclass
class ValidateStructures(BaseNode[ReconstructionState, WorkflowDeps, ReconstructionResult]):
    """Routes on the validation gate's reports (computed when construction accepted each
    structure), adding the export round trip of the written files; hard failures and plans that
    were not built go back to construction (bounded repair loop)."""

    async def run(self, ctx: Ctx) -> ConstructStructures | ScientificRevision:
        _enter(ctx, "ValidateStructures")
        st, store = ctx.state, ctx.deps.store
        failing, notes = set(), []
        for sid in st.current_ids:
            a = st.artifact(sid)
            rep = st.validation_reports[sid]
            if not any(c.name == "export_roundtrip" for c in rep.checks):
                files = {k: str(store.root / p) for k, p in a.files.items()}
                roundtrip = check_roundtrip(read(store.root / a.source), files)
                rep = st.validation_reports[sid] = rep.with_check(roundtrip)
            if not rep.passed:
                failing.add(a.plan_id)
                notes.append(f"{a.name} (plan {a.plan_id}): {rep.summary()}")
        unbuilt = [pid for pid in st.pending_plan_ids if pid in st.failed_plans]
        failing.update(unbuilt)
        store.write_json("validation_reports.json", st.validation_reports)
        if failing and st.attempt < ctx.deps.max_repairs:
            st.attempt += 1
            st.pending_plan_ids = sorted(failing)
            parts = (["Deterministic validation failed:"] + notes if notes else []) + (
                ["Plans that were not built (their scripts from the last attempt are in previous/):"]
                + [f"- {pid}: {st.failed_plans[pid]}" for pid in unbuilt] if unbuilt else [])
            st.repair_feedback = "\n".join(parts)
            _checkpoint(ctx, "ValidateStructures")
            return ConstructStructures()
        _checkpoint(ctx, "ValidateStructures")
        return ScientificRevision()


@dataclass
class ScientificRevision(BaseNode[ReconstructionState, WorkflowDeps, ReconstructionResult]):
    """Agent 5: accept or stop (after writing the final report), or send the run back through
    evidence -> planning -> construction."""

    async def run(self, ctx: Ctx) -> AnalyzeEvidence | FinalizeResults:
        _enter(ctx, "ScientificRevision")
        st, deps = ctx.state, ctx.deps
        current = [st.artifact(i) for i in st.current_ids]
        images = None
        if deps.cfg.campaign.use_images:
            images = {a.structure_id: (deps.store.root / a.image).read_bytes() for a in current[:4] if a.image}
        left = deps.max_revisions - st.revision
        try:
            decision = await deps.reviser.review(st, deps.store, left, images)
        except Exception as e:  # noqa: BLE001
            st.errors.append(f"review failed: {type(e).__name__}: {e}")
            _checkpoint(ctx, "ScientificRevision")
            return FinalizeResults(status="failed")
        st.revision_history.append(decision)
        deps.store.write_json("revision_history.json", st.revision_history)
        _checkpoint(ctx, "ScientificRevision")
        if decision.action == "revisit_evidence":
            if left > 0:
                st.revision += 1
                st.review_feedback = decision.feedback
                if st.failed_plans:  # the planner sees the builder's own reasons, not just the reviewer's
                    st.review_feedback += "\n\nPlans that could not be built:\n" + "\n".join(
                        f"- {k}: {v}" for k, v in st.failed_plans.items())
                return AnalyzeEvidence()
            return FinalizeResults(status="revision_budget_exhausted")
        return FinalizeResults(status="accepted" if decision.action == "accept_candidates" else "inconclusive")


@dataclass
class FinalizeResults(BaseNode[ReconstructionState, WorkflowDeps, ReconstructionResult]):
    """Write the run summary, token log and final report (re-rendered with the final status and
    decision; without a reviewer narrative when the run ended before a final review); end the run."""
    status: str = "failed"

    async def run(self, ctx: Ctx) -> End[ReconstructionResult]:
        _enter(ctx, "FinalizeResults")
        st, deps, store = ctx.state, ctx.deps, ctx.deps.store
        st.status = self.status
        last = st.revision_history[-1] if st.revision_history else None
        accepted = list(last.accepted_structure_ids) if last and self.status == "accepted" else []
        store.write_json("plans.json", st.plan_history)
        store.write_json("artifacts.json", st.artifacts)
        store.write_json("validation_reports.json", st.validation_reports)
        store.write_json("revision_history.json", st.revision_history)
        store.write_json("construction_reports.json", st.construction_reports)
        store.write_json("token_log.json", {"planner": deps.planner.usage, "constructor": deps.constructor.usage,
                                            "reviser": deps.reviser.usage})
        report = save_report(st, store)  # final status and decision around the reviewer's text
        save_structures_readme(st, store)
        result = ReconstructionResult(
            run_id=st.run_id, status=self.status, run_dir=str(store.root), accepted_structure_ids=accepted,
            structure_files={a.structure_id: a.files for a in st.artifacts}, report=store.rel(report),
            revisions=st.revision, errors=list(st.errors))
        store.write_json("result.json", result)
        _checkpoint(ctx, "FinalizeResults")
        return End(result)
