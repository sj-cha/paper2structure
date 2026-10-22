"""Workflow nodes. Each node's `run` return annotation declares its outgoing edges."""
from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ase.io import read
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage
from pydantic_graph import BaseNode, End, GraphRunContext

from paper2structure.agents.characterization import CharacterizationAgent, ExperimentalEvidence, find_spec
from paper2structure.agents.constructor.schemas import ConstructionReport, StructureArtifact
from paper2structure.agents.reviser import save_report, save_structures_readme
from paper2structure.agents.structured_agent import add_usage
from paper2structure.agents.summarizer import PaperGraph, load_graph
from paper2structure.config import CharacterizationConfig, canonical_technique, get_model
from paper2structure.graph.schemas import ReconstructionResult
from paper2structure.graph.state import ReconstructionState, WorkflowDeps
from paper2structure.tools.geometry_checks import check_roundtrip
from paper2structure.tools.render import render_png
from paper2structure.tools.structure_export import describe, export_structure

if TYPE_CHECKING:  # the constructor's agent loads with the workflow (WorkflowDeps), not with these nodes
    from paper2structure.agents.constructor.agent import ConstructionDeps

log = logging.getLogger(__name__)
Ctx = GraphRunContext[ReconstructionState, WorkflowDeps]


def _enter(ctx: Ctx, node: str) -> None:
    ctx.state.node_history.append(node)
    log.info("[%s] revision %d, attempt %d", node, ctx.state.revision, ctx.state.attempt)


def _checkpoint(ctx: Ctx, node: str, full: bool = False) -> None:
    ctx.deps.store.snapshot(ctx.state, node, full=full)


def _fail(ctx: Ctx, node: str, what: str, e: Exception) -> FinalizeResults:
    """End the run: the error goes to the state as one line and to the log with its traceback."""
    ctx.state.errors.append(f"{what} failed: {type(e).__name__}: {e}")
    log.error("[%s] %s failed", node, what, exc_info=e)
    _checkpoint(ctx, node)
    return FinalizeResults(status="failed")


def _paper_model(ctx: Ctx) -> Model | str:
    model = ctx.deps.paper_model
    return model if isinstance(model, Model) else get_model(model or ctx.deps.cfg.campaign.model)


@dataclass
class SummarizePaper(BaseNode[ReconstructionState, WorkflowDeps, ReconstructionResult]):
    """Agent 1: reads the PDF (text, figures, material systems) into a PaperGraph saved under
    paper/, or loads the paper_graph.json of an earlier run (copied to inputs/). Runs once."""

    async def run(self, ctx: Ctx) -> AnalyzeEvidence | FinalizeResults:
        _enter(ctx, "SummarizePaper")
        st, deps, inputs = ctx.state, ctx.deps, ctx.deps.cfg.inputs
        try:
            if inputs.pdf and inputs.characterization:
                raise ValueError("inputs.characterization cannot be combined with inputs.pdf: saved reports cite the "
                                 "system and evidence IDs of the summary they were made from, and a new summary "
                                 "renumbers them; pass that summary as inputs.paper_graph (--paper_graph)")
            if inputs.paper_graph:
                path = Path(inputs.paper_graph).resolve()
                st.summary, st.paper_dir = load_graph(path), str(path.parent)
                if not path.is_relative_to(deps.store.root):  # the run keeps its own copy of what it used
                    deps.store.write_json("inputs/paper_graph.json", st.summary)
            elif inputs.pdf:
                from paper2structure.agents.summarizer import summarize_paper  # the PDF reader, only when needed

                out = deps.store.path("paper", "paper_graph.json").parent
                (out / "errors.txt").unlink(missing_ok=True)  # the summarizer only writes it when steps fail
                sc, usage = deps.cfg.summarizer, RunUsage()
                try:
                    st.summary = await summarize_paper(inputs.pdf, out, _paper_model(ctx), on_progress=log.info,
                                                       analyze_figures=sc.analyze_figures, max_figures=sc.max_figures,
                                                       usage=usage)
                finally:
                    add_usage(deps.paper_usage["summarizer"], usage)
                st.paper_dir = str(out)
            else:
                raise ValueError("the config needs inputs.pdf or inputs.paper_graph")
            if not st.summary.systems:
                raise ValueError("the summarizer found no material systems in the paper")
        except Exception as e:  # noqa: BLE001
            return _fail(ctx, "SummarizePaper", "summarization", e)
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
                       cc: CharacterizationConfig, usage: Optional[dict[str, int]] = None) -> ExperimentalEvidence:
    """One characterization agent per technique found in the paper's evidence, concurrently; each
    report is also saved as out_dir/<technique>.json (earlier reports there are removed first).
    Techniques outside the vocabulary are skipped. The agents' token usage is added to `usage`
    (token_log.json's keys)."""
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
        else:
            specs.append(spec)
    log.info("[AnalyzeEvidence] characterizing %s; skipped %s", [s.name for s in specs] or "nothing",
             evidence.skipped or "none")
    for stale in Path(out_dir).glob("*.json"):  # a reused folder must not mix in another run's reports
        stale.unlink()
    agents = [CharacterizationAgent(s, model=model, reasoning_effort=cc.reasoning_effort, output_retries=cc.output_retries)
              for s in specs]
    results = await asyncio.gather(*(
        agent.analyze(paper.evidence, paper.systems, base_dir=paper_dir, filename=out_dir / f"{s.name.replace(' ', '_')}.json")
        for agent, s in zip(agents, specs)), return_exceptions=True)
    for agent, spec, result in zip(agents, specs, results):
        if usage is not None:
            for key, n in agent.usage.items():
                usage[key] += n
        if isinstance(result, BaseException):
            evidence.errors.append(f"{spec.name} characterization failed: {type(result).__name__}: {result}")
            log.warning("[AnalyzeEvidence] %s", evidence.errors[-1], exc_info=result)
        else:
            evidence.reports[spec.name] = result
            log.info("[AnalyzeEvidence] %s: %d observations, %d structural constraints", spec.name,
                     len(result.observations), len(result.structural_constraints))
    return evidence


@dataclass
class AnalyzeEvidence(BaseNode[ReconstructionState, WorkflowDeps, ReconstructionResult]):
    """Agent 2: one characterization agent per technique in the paper's evidence, or the reports
    saved in inputs.characterization (checked against the summary). The reports are restricted to
    the summary's systems and saved once (inputs/experimental_evidence.json). Re-entered when the
    reviewer revisits the evidence: the reports are kept and the reviewer's feedback travels on to
    the planner."""

    async def run(self, ctx: Ctx) -> PlanStructures | FinalizeResults:
        _enter(ctx, "AnalyzeEvidence")
        st, deps = ctx.state, ctx.deps
        if st.evidence is not None:
            log.info("[AnalyzeEvidence] revision %d: characterization reports kept; reviewer feedback "
                     "forwarded to the planner", st.revision)
            _checkpoint(ctx, "AnalyzeEvidence")
            return PlanStructures()
        folder = deps.cfg.inputs.characterization
        try:
            if folder:
                evidence = ExperimentalEvidence.load(folder)
                log.info("[AnalyzeEvidence] loaded the reports %s from %s", sorted(evidence.reports), folder)
                errors, warnings = evidence.mismatches(st.summary)
                for w in warnings:  # expected when the summary was narrowed to some systems
                    log.warning("[AnalyzeEvidence] saved reports: %s", w)
                if errors:
                    raise ValueError(f"the reports in {folder} do not belong to this summary: " + "; ".join(errors))
            else:
                evidence = await characterize(st.summary, Path(st.paper_dir), deps.store.root / "characterization",
                                              _paper_model(ctx), deps.cfg.characterization,
                                              usage=deps.paper_usage["characterization"])
        except Exception as e:  # noqa: BLE001
            return _fail(ctx, "AnalyzeEvidence", "characterization", e)
        st.evidence = evidence.focused({s.system_id for s in st.summary.systems})
        kept = sum(len(r.observations) for r in st.evidence.reports.values())
        total = sum(len(r.observations) for r in evidence.reports.values())
        if kept < total:
            log.info("[AnalyzeEvidence] %d of %d observations concern the summary's systems; the rest are left out",
                     kept, total)
        st.errors.extend(e for e in st.evidence.errors if e not in st.errors)
        deps.store.write_json("inputs/experimental_evidence.json", st.evidence)
        _checkpoint(ctx, "AnalyzeEvidence")
        return PlanStructures()


def _replan_context(st: ReconstructionState) -> str:
    """What the planner learns on revisit: for every earlier plan set, how each plan's structures
    fared (validation, failing claim checks, the reviewer's verdicts) and the review that closed it,
    then the latest feedback."""
    if not st.plan_history:
        return st.review_feedback
    closing = [d for d in st.revision_history if d.action == "revisit_evidence"]
    parts = []
    for k, plans in enumerate(st.plan_history):
        decision = closing[k] if k < len(closing) else None
        verdicts = {v.structure_id: v for v in decision.verdicts} if decision else {}
        lines = [f"### Plan set {k}"]
        for p in plans.plans:
            built = [a for a in st.artifacts if a.revision == k and a.plan_id == p.plan_id]
            if not built:
                lines.append(f"- {p.plan_id}: no structure was built")
            for a in built:
                rep = st.validation_reports.get(a.structure_id)
                claims = [f"{c.name.removeprefix('claim ')} {c.status}: {c.detail}" for c in (rep.checks if rep else [])
                          if c.name.startswith("claim ") and c.status in ("fail", "warn")]
                v = verdicts.get(a.structure_id)
                lines.append(f"- {p.plan_id}: {a.structure_id}, validation {'PASS' if rep and rep.passed else 'FAIL'}"
                             + (f"; claim checks: {'; '.join(claims)}" if claims else "")
                             + (f"; reviewer: {'resembles' if v.resembles_paper else 'does not resemble'} the paper"
                                + (f" ({'; '.join(v.issues)})" if v.issues else "") if v else ""))
        if decision and decision.unresolved:
            lines.append("Unresolved: " + "; ".join(decision.unresolved))
        parts.append("\n".join(lines))
    return "\n\n".join(parts + ["### Feedback on the latest plan set", st.review_feedback])


@dataclass
class PlanStructures(BaseNode[ReconstructionState, WorkflowDeps, ReconstructionResult]):
    """Agent 3: structural hypotheses as construction plans (plans.json holds every plan set)."""

    async def run(self, ctx: Ctx) -> ConstructStructures | FinalizeResults:
        _enter(ctx, "PlanStructures")
        st = ctx.state
        try:
            plans = await ctx.deps.planner.plan(st.summary, st.evidence, st.plan_history,
                                                _replan_context(st) if st.review_feedback else "")
        except Exception as e:  # noqa: BLE001
            return _fail(ctx, "PlanStructures", "planning", e)
        st.plan_history.append(plans)
        st.pending_plan_ids = [p.plan_id for p in plans.plans]
        st.attempt, st.repairs, st.current_ids, st.failed_plans = 0, 0, [], {}
        st.rebuild_ids, st.rebuild_request, st.plan_feedback, st.repair_feedback = [], {}, {}, ""
        ctx.deps.store.write_json("plans.json", st.plan_history)
        _checkpoint(ctx, "PlanStructures")
        return ConstructStructures()


@dataclass
class ConstructStructures(BaseNode[ReconstructionState, WorkflowDeps, ReconstructionResult]):
    """Agent 4: one coding-agent run per pending plan, concurrently, each in its own sandboxed
    workspace with its own request budget and its own plan's feedback; every reported structure is
    the output of its script re-run from scratch, and that script is saved with it. A plan whose run
    fails, or that the agent reports as not built, is recorded with the reason; ValidateStructures
    sends it back (bounded repair loop) with the scripts of the attempt."""

    async def run(self, ctx: Ctx) -> ValidateStructures:
        _enter(ctx, "ConstructStructures")
        st = ctx.state
        pending = list(st.pending_plan_ids)
        results = await asyncio.gather(*(self._build(ctx, pid) for pid in pending), return_exceptions=True)
        for pid, result in zip(pending, results):
            if isinstance(result, BaseException):
                st.failed_plans[pid] = f"construction failed: {type(result).__name__}: {result}"
                st.errors.append(f"{pid} (revision {st.revision}, attempt {st.attempt}): {st.failed_plans[pid]}")
                log.warning("[ConstructStructures] %s", st.errors[-1], exc_info=result)
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
        feedback = st.plan_feedback.get(plan_id, "")
        previous = {}
        if feedback:
            scripts: dict[str, str] = {}  # one copy per distinct script, named after the first structure it built
            for a in map(st.artifact, st.current_ids):
                if a.plan_id == plan_id:
                    scripts.setdefault((store.root / a.script).read_text(), a.structure_id)
            previous = {f"{sid}.py": code for code, sid in scripts.items()}
            if plan_id in st.failed_plans:  # not built last time: that attempt's scripts too
                from paper2structure.agents.constructor.agent import WORKSPACE

                last = store.root / "construction" / f"r{st.revision}_a{st.attempt - 1}" / plan_id / WORKSPACE
                previous.update({f"last_attempt/{p.name}": p.read_text() for p in sorted(last.glob("*.py"))})
        workdir = store.path("construction", f"r{st.revision}_a{st.attempt}", plan_id)
        return await ctx.deps.constructor.construct(st.plans.get(plan_id), workdir, ctx.deps.seed, feedback,
                                                    previous, st.summary, st.evidence)

    @staticmethod
    def _register(ctx: Ctx, report: ConstructionReport, cdeps: ConstructionDeps) -> None:
        """Export the verified structures of one plan's report and record its unbuilt plans. A plan
        the report built replaces its current structures that the reviewer asked to rebuild, that
        failed validation, or that were emitted again under the same name; the others stay. A plan
        the report lists only as failed keeps its structures."""
        st, store, cfg = ctx.state, ctx.deps.store, ctx.deps.cfg
        st.construction_reports.append(report)
        manifest = cdeps.manifest()
        built_plans = {b.plan_id for b in report.structures}
        names = {b.name for b in report.structures}

        def replaced(sid: str) -> bool:
            a, rep = st.artifact(sid), st.validation_reports.get(sid)
            return a.plan_id in built_plans and (sid in st.rebuild_ids or a.name in names
                                                 or (rep is not None and not rep.passed))

        st.current_ids = [i for i in st.current_ids if not replaced(i)]
        st.rebuild_ids = [i for i in st.rebuild_ids if st.artifact(i).plan_id not in built_plans]
        for pid in built_plans:
            st.failed_plans.pop(pid, None)
        taken = {a.structure_id for a in st.artifacts}
        for built in report.structures:
            sid = f"{built.name}_r{st.revision}a{st.attempt}"
            if sid in taken:  # two plans' runs chose the same name
                sid = f"{built.plan_id}_{sid}"
            emitted = manifest.get(built.name, {})
            try:
                source = cdeps.verified[built.name]
                atoms = read(source)
                files = export_structure(atoms, store.path("structures"), sid, formats=cfg.export.formats)
                image = store.path("structures", f"{sid}.png")
                render_png(atoms, image)
                script = store.path("structures", f"{sid}.py")
                shutil.copyfile(cdeps.scripts[built.name], script)
                artifact = StructureArtifact(
                    structure_id=sid, plan_id=built.plan_id, name=built.name,
                    system_ids=emitted.get("system_ids") or built.system_ids, revision=st.revision,
                    files={k: store.rel(v) for k, v in files.items()}, source=store.rel(source),
                    script=store.rel(script), image=store.rel(image), method=built.method,
                    target_checks=built.target_checks, formula=atoms.get_chemical_formula(), n_atoms=len(atoms),
                    description=describe(atoms), deviations=built.deviations,
                    notes="; ".join(dict.fromkeys(n for n in (emitted.get("notes", ""), built.notes) if n)),
                    construction_summary=report.summary)
            except Exception as e:  # noqa: BLE001 - one unexportable structure must not end the run
                st.failed_plans[built.plan_id] = f"{built.name} could not be exported: {type(e).__name__}: {e}"
                st.errors.append(f"{built.name} (plan {built.plan_id}): {st.failed_plans[built.plan_id]}")
                log.warning("[ConstructStructures] %s", st.errors[-1], exc_info=True)
                continue
            st.validation_reports[sid] = cdeps.reports[built.name].model_copy(update={"structure_id": sid})
            st.artifacts.append(artifact)
            st.current_ids.append(sid)
            taken.add(sid)
        for f in report.failed:
            st.failed_plans[f.plan_id] = f.reason


def _repair_feedback(st: ReconstructionState, plan_id: str, notes: list[str]) -> str:
    """A plan's feedback for its repair run: the reviewer's pending rebuild request, which still
    applies, then what failed."""
    parts = [st.rebuild_request[plan_id]] if st.rebuild_request.get(plan_id) else []
    if notes:
        parts += ["Deterministic validation failed:"] + notes
    if plan_id in st.failed_plans:
        parts += ["The plan was not built in the last attempt (any scripts it wrote are in previous/last_attempt/):",
                  f"- {plan_id}: {st.failed_plans[plan_id]}"]
    return "\n".join(parts)


@dataclass
class ValidateStructures(BaseNode[ReconstructionState, WorkflowDeps, ReconstructionResult]):
    """Routes on the validation gate's reports (computed when construction accepted each
    structure), adding the export round trip of the written files; the plans built in this round
    whose structures fail, or that were not built, go back to construction (bounded repair loop),
    each with its own feedback."""

    async def run(self, ctx: Ctx) -> ConstructStructures | ScientificRevision:
        _enter(ctx, "ValidateStructures")
        st, store = ctx.state, ctx.deps.store
        new = [st.artifact(i) for i in st.current_ids
               if not any(c.name == "export_roundtrip" for c in st.validation_reports[i].checks)]

        def roundtrip(a: StructureArtifact):
            return check_roundtrip(read(store.root / a.source), {k: str(store.root / p) for k, p in a.files.items()})

        for a, check in zip(new, await asyncio.gather(*(asyncio.to_thread(roundtrip, a) for a in new))):
            st.validation_reports[a.structure_id] = st.validation_reports[a.structure_id].with_check(check)
        failing, notes = set(), {}
        for sid in st.current_ids:
            a, rep = st.artifact(sid), st.validation_reports[sid]
            if not rep.passed and a.plan_id in st.pending_plan_ids:  # plans not rebuilt now stay with the reviewer
                failing.add(a.plan_id)
                notes.setdefault(a.plan_id, []).append(f"{a.name} (plan {a.plan_id}): {rep.summary()}")
        failing.update(pid for pid in st.pending_plan_ids if pid in st.failed_plans)
        store.write_json("validation_reports.json", st.validation_reports)
        if failing and st.repairs < ctx.deps.max_repairs:
            st.repairs += 1
            st.attempt += 1
            st.pending_plan_ids = sorted(failing)
            st.plan_feedback = {pid: _repair_feedback(st, pid, notes.get(pid, [])) for pid in st.pending_plan_ids}
            st.repair_feedback = "\n\n".join(st.plan_feedback.values())
            _checkpoint(ctx, "ValidateStructures")
            return ConstructStructures()
        _checkpoint(ctx, "ValidateStructures")
        return ScientificRevision()


@dataclass
class ScientificRevision(BaseNode[ReconstructionState, WorkflowDeps, ReconstructionResult]):
    """Agent 5: accept or stop (after writing the final report), send structures whose plan stands
    back to construction (rebuild), or send the run back through evidence -> planning ->
    construction (revisit). The reviewer's output validator keeps rebuilds and revisits within
    max_rebuilds and max_revisions."""

    async def run(self, ctx: Ctx) -> AnalyzeEvidence | ConstructStructures | FinalizeResults:
        _enter(ctx, "ScientificRevision")
        st, deps = ctx.state, ctx.deps
        current = [st.artifact(i) for i in st.current_ids]
        images = None
        if deps.cfg.campaign.use_images:
            images = {a.structure_id: (deps.store.root / a.image).read_bytes() for a in current[:4]}
        left = deps.max_revisions - st.revision
        try:
            decision = await deps.reviser.review(st, deps.store, left, images, deps.max_rebuilds - st.rebuilds)
        except Exception as e:  # noqa: BLE001
            return _fail(ctx, "ScientificRevision", "review", e)
        st.revision_history.append(decision)
        deps.store.write_json("revision_history.json", st.revision_history)
        st.rebuild_request, st.plan_feedback = {}, {}  # a new decision replaces any earlier request
        if decision.action == "rebuild_structures":
            st.rebuilds += 1
            st.attempt += 1
            st.repairs = 0
            st.rebuild_ids = list(decision.rebuild_structure_ids)
            st.pending_plan_ids = sorted({st.artifact(i).plan_id for i in st.rebuild_ids})
            for pid in st.pending_plan_ids:  # each plan's run gets the reviewer's text and its own structures' issues
                ids = [i for i in st.rebuild_ids if st.artifact(i).plan_id == pid]
                issues = [f"- {v.structure_id}: {'; '.join(v.issues)}" for v in decision.verdicts
                          if v.structure_id in ids and v.issues]
                st.rebuild_request[pid] = "\n".join([
                    f"The reviewer asks to rebuild {', '.join(ids)}: the plan stands, the built structure does not "
                    "realize it. Fix the scripts (in previous/) accordingly; the plan's structures you do not "
                    "report again are kept.", decision.feedback, *issues])
            st.plan_feedback = dict(st.rebuild_request)
            st.repair_feedback = "\n\n".join(st.plan_feedback.values())
            _checkpoint(ctx, "ScientificRevision")
            return ConstructStructures()
        if decision.action == "revisit_evidence":  # the reviewer's validator keeps it within max_revisions
            st.revision += 1
            st.candidate_ids += [i for i in st.current_ids if i not in st.candidate_ids
                                 and st.validation_reports[i].passed]  # still acceptable after re-planning
            st.review_feedback = decision.feedback
            if st.failed_plans:  # the planner sees the builder's own reasons, not just the reviewer's
                st.review_feedback += "\n\nPlans that could not be built:\n" + "\n".join(
                    f"- {k}: {v}" for k, v in st.failed_plans.items())
            _checkpoint(ctx, "ScientificRevision")
            return AnalyzeEvidence()
        _checkpoint(ctx, "ScientificRevision")
        return FinalizeResults(status="accepted" if decision.action == "accept_candidates" else "inconclusive")


@dataclass
class FinalizeResults(BaseNode[ReconstructionState, WorkflowDeps, ReconstructionResult]):
    """Write the run summary, token log and final report (re-rendered with the final status and
    decision; without a reviewer narrative when the run ended before a final review or failed); end
    the run."""
    status: str = "failed"

    async def run(self, ctx: Ctx) -> End[ReconstructionResult]:
        _enter(ctx, "FinalizeResults")
        st, deps, store = ctx.state, ctx.deps, ctx.deps.store
        st.status = self.status
        if self.status == "failed":  # a narrative written for an earlier decision does not describe this end
            st.report = None
        store.write_json("plans.json", st.plan_history)
        store.write_json("artifacts.json", st.artifacts)
        store.write_json("validation_reports.json", st.validation_reports)
        store.write_json("revision_history.json", st.revision_history)
        store.write_json("construction_reports.json", st.construction_reports)
        store.write_json("token_log.json", {**deps.paper_usage, "planner": deps.planner.usage,
                                            "constructor": deps.constructor.usage, "reviser": deps.reviser.usage})
        report = save_report(st, store)  # final status and decision around the reviewer's text
        save_structures_readme(st, store)
        result = ReconstructionResult(
            run_id=st.run_id, status=self.status, run_dir=str(store.root), accepted_structure_ids=st.accepted_ids,
            current_structure_ids=list(st.current_ids), structure_files={a.structure_id: a.files for a in st.artifacts},
            report=store.rel(report), revisions=st.revision, errors=list(st.errors))
        store.write_json("result.json", result)
        _checkpoint(ctx, "FinalizeResults", full=True)
        return End(result)
