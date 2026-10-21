"""Workflow nodes. Each node's `run` return annotation declares its outgoing edges."""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass

from ase.io import read
from pydantic_graph import BaseNode, End, GraphRunContext

from paper2structure.agents.characterization import analyze_evidence
from paper2structure.agents.constructor.schemas import StructureArtifact
from paper2structure.agents.reviser import save_report, save_structures_readme
from paper2structure.agents.summarizer import summarize_paper
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


@dataclass
class SummarizePaper(BaseNode[ReconstructionState, WorkflowDeps, ReconstructionResult]):
    """Agent 1 placeholder: loads the paper-summary fixture named in the config. Runs once."""

    async def run(self, ctx: Ctx) -> AnalyzeEvidence:
        _enter(ctx, "SummarizePaper")
        ctx.state.summary = summarize_paper(ctx.deps.cfg.inputs.paper_summary)
        ctx.deps.store.write_json("inputs/paper_summary.json", ctx.state.summary)
        _checkpoint(ctx, "SummarizePaper")
        return AnalyzeEvidence()


@dataclass
class AnalyzeEvidence(BaseNode[ReconstructionState, WorkflowDeps, ReconstructionResult]):
    """Agent 2 placeholder: loads the evidence fixture. Re-entered when the reviewer finds that the
    structures do not resemble the paper; the placeholder returns the same snapshot and the
    reviewer's feedback travels on to the planner."""

    async def run(self, ctx: Ctx) -> PlanStructures:
        _enter(ctx, "AnalyzeEvidence")
        st = ctx.state
        st.evidence = analyze_evidence(ctx.deps.cfg.inputs.experimental_evidence, st.review_feedback)
        st.evidence_version += 1
        ctx.deps.store.write_json(f"inputs/experimental_evidence_v{st.evidence_version}.json", st.evidence)
        if st.review_feedback:
            log.info("[AnalyzeEvidence] placeholder agent: evidence fixture unchanged; reviewer feedback "
                     "forwarded to the planner")
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
    """Agent 4: a coding agent builds the pending plans in a sandboxed workspace; every reported
    structure is the output of its script re-run from scratch, and that script is saved with it."""

    async def run(self, ctx: Ctx) -> ValidateStructures | ScientificRevision | FinalizeResults:
        _enter(ctx, "ConstructStructures")
        st, store, cfg = ctx.state, ctx.deps.store, ctx.deps.cfg
        workdir = store.path("construction", f"r{st.revision}_a{st.attempt}")
        previous = {}
        if st.repair_feedback:
            for sid in st.current_ids:
                a = st.artifact(sid)
                if a.plan_id in st.pending_plan_ids:
                    previous[f"{a.structure_id}.py"] = (store.root / a.script).read_text()
        try:
            report, cdeps = await ctx.deps.constructor.construct(st.plans, st.pending_plan_ids, workdir, ctx.deps.seed,
                                                                 st.repair_feedback, previous, st.summary, st.evidence)
        except Exception as e:  # noqa: BLE001
            st.errors.append(f"construction failed (revision {st.revision}, attempt {st.attempt}): "
                             f"{type(e).__name__}: {e}")
            _checkpoint(ctx, "ConstructStructures")
            return ScientificRevision() if st.current_ids else FinalizeResults(status="failed")
        st.construction_reports.append(report)
        rebuilt = {b.plan_id for b in report.structures}
        st.current_ids = [i for i in st.current_ids if st.artifact(i).plan_id not in rebuilt]
        for built in report.structures:
            source = cdeps.verified[built.name]
            atoms = read(source)
            sid = f"{built.name}_r{st.revision}a{st.attempt}"
            files = export_structure(atoms, store.path("structures"), sid, formats=cfg.export.formats)
            image = store.path("structures", f"{sid}.png")
            render_png(atoms, image)
            script = store.path("structures", f"{sid}.py")
            shutil.copyfile(cdeps.scripts[built.name], script)
            st.validation_reports[sid] = cdeps.reports[built.name].model_copy(update={"structure_id": sid})
            st.artifacts.append(StructureArtifact(
                structure_id=sid, plan_id=built.plan_id, name=built.name, revision=st.revision, attempt=st.attempt,
                files={k: store.rel(v) for k, v in files.items()}, source=store.rel(source),
                script=store.rel(script), image=store.rel(image), method=built.method,
                target_checks=built.target_checks, formula=atoms.get_chemical_formula(),
                n_atoms=len(atoms), cell=atoms.cell.array.round(6).tolist(), pbc=tuple(bool(p) for p in atoms.pbc),
                n_molecules=len(get_molecules(atoms)), description=describe(atoms), deviations=built.deviations,
                notes=built.notes))
            st.current_ids.append(sid)
        for f in report.failed:
            st.failed_plans[f.plan_id] = f.reason
        store.write_json("artifacts.json", st.artifacts)
        _checkpoint(ctx, "ConstructStructures")
        return ValidateStructures()


@dataclass
class ValidateStructures(BaseNode[ReconstructionState, WorkflowDeps, ReconstructionResult]):
    """Routes on the validation gate's reports (computed when construction accepted each
    structure), adding the export round trip of the written files; hard failures go back to
    construction (bounded repair loop)."""

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
        store.write_json("validation_reports.json", st.validation_reports)
        if failing and st.attempt < ctx.deps.max_repairs:
            st.attempt += 1
            st.pending_plan_ids = sorted(failing)
            st.repair_feedback = "Deterministic validation failed:\n" + "\n".join(notes)
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
