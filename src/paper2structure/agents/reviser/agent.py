from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai import BinaryContent, ModelRetry, RunContext
from pydantic_ai.workspaces import LocalWorkspaceBackend, ReadOnlyWorkspace, Workspace
from pydantic_ai_harness import FileSystem

from paper2structure.agents.planner.claims import claims_about, constraint_value, structural_claims
from paper2structure.agents.reviser.schemas import ReportNarrative, ReviewCheck, RevisionDecision
from paper2structure.agents.structured_agent import StructuredAgent
from paper2structure.tools.schemas import ClaimCheck

if TYPE_CHECKING:  # graph.state imports this package's schemas
    from paper2structure.graph.state import ArtifactStore, ReconstructionState

log = logging.getLogger(__name__)


@dataclass
class RevisionDeps:
    state: ReconstructionState
    store: ArtifactStore
    revisions_left: int
    rebuilds_left: int = 0
    report_written: bool = False  # write_report was called in this review

    @property
    def structure_ids(self) -> set[str]:
        """The current structures: the ones a decision may rebuild."""
        return set(self.state.current_ids)

    @property
    def candidate_ids(self) -> set[str]:
        """Earlier plan sets' structures that passed validation: they may be accepted, not rebuilt."""
        return set(self.state.candidate_ids) - self.structure_ids


class LLMReviser(StructuredAgent):
    """Agent 5: judges whether the built structures represent what the paper describes, and writes
    the final report when its decision ends the run."""
    _output_type = RevisionDecision
    _deps_type = RevisionDeps
    _preflight_probe = ("Health check: call write_report with the summary 'health check', the assessment 'health "
                        "check' and no next steps, then return action 'stop_inconclusive' with reasoning 'health check'.")

    def __init__(self, model, effort: str | None, max_retry: int = 3, read_scripts: bool = True,
                 constraints: str = ""):
        """`constraints`: what the run asks of the structures (purpose, size budget, user pins),
        appended to the system prompt."""
        super().__init__(model, effort, reviser_system_prompt(read_scripts), system_context=constraints,
                         max_retry=max_retry)
        self.read_scripts = read_scripts

    def _capabilities(self) -> list:
        # read-only file tools over the run's structures/ directory (scripts, sidecars)
        return [FileSystem(read_only=True, tools=("read_file", "list_files", "grep"))] if self.read_scripts else []

    @staticmethod
    def _workspace(directory: Path) -> ReadOnlyWorkspace:
        return ReadOnlyWorkspace(Workspace(LocalWorkspaceBackend(directory)))

    def preflight(self, **run_kwargs) -> tuple[bool, str]:
        """A health-check review in a throwaway run directory, through the report tool."""
        from paper2structure.graph.state import ArtifactStore, ReconstructionState  # graph imports this package

        with tempfile.TemporaryDirectory(prefix="p2s-preflight-") as tmp:
            deps = RevisionDeps(state=ReconstructionState(run_id="preflight"), store=ArtifactStore(tmp),
                                revisions_left=0)
            if self.read_scripts:
                run_kwargs["workspace"] = self._workspace(Path(tmp))
            return super().preflight(deps=deps, **run_kwargs)

    def _register(self, agent) -> None:
        @agent.tool
        def write_report(ctx: RunContext[RevisionDeps], summary: str, assessment: str, next_steps: list[str]) -> str:
            """Write the run's final report (final_report.md). Your text: `summary` (what was modeled
            and the bottom line, two to five sentences), `assessment` (how each accepted structure
            relates to the evidence, and the caveats a user must know before simulating it) and
            `next_steps` (what would resolve the open questions: relaxations, alternative models,
            further measurements). The tool adds the paper, the evidence table, the plans, every
            structure with its validation, measured targets and script, the revision history and
            the limitations. Call it before a decision that ends the run."""
            if not summary.strip() or not assessment.strip():
                raise ModelRetry("summary and assessment must not be empty")
            deps = ctx.deps
            deps.state.report = ReportNarrative(summary=summary.strip(), assessment=assessment.strip(),
                                                next_steps=[n.strip() for n in next_steps if n.strip()])
            deps.report_written = True
            path = save_report(deps.state, deps.store)
            return (f"wrote {deps.store.rel(path)}: your summary, assessment and {len(deps.state.report.next_steps)} "
                    "next steps with the run's evidence, plans, structures, validation and scripts. It is "
                    "finalized with your decision when the run ends.")

        @agent.tool
        def check_claim(ctx: RunContext[RevisionDeps], structure_id: str, check: ClaimCheck) -> str:
            """Run one deterministic check of a claim on a constructed structure, the same checks the
            validation gate runs for the plans: lattice, space_group, composition, bond_length,
            bond_angle, coordination, hydrogen_bonds or layers. Use it to test what the plan's checks
            leave open (a claim listed as unchecked, a reported bond angle) instead of judging it
            from the description."""
            from ase.io import read

            from paper2structure.tools.evidence_checks import run_claim_checks

            deps = ctx.deps
            known = deps.structure_ids | deps.candidate_ids
            if structure_id not in known:
                raise ModelRetry(f"unknown structure {structure_id!r} (valid: {sorted(known)})")
            a = deps.state.artifact(structure_id)
            atoms = read(deps.store.root / a.source)
            result = run_claim_checks(atoms, [check.model_copy(update={"system_ids": []})])[0]
            deps.state.review_checks.append(ReviewCheck(
                review=len(deps.state.revision_history), structure_id=structure_id, check_id=check.id,
                kind=check.kind, claim=check.claim, status=result.status, detail=result.detail))
            return f"{result.status}: {result.detail}"

        @agent.output_validator
        def _validate(ctx: RunContext[RevisionDeps], output: RevisionDecision) -> RevisionDecision:
            deps = ctx.deps
            problems = []
            known, candidates = deps.structure_ids, deps.candidate_ids
            unknown = (set(output.accepted_structure_ids) | set(output.rebuild_structure_ids)
                       | {v.structure_id for v in output.verdicts}) - known - candidates
            if unknown:
                problems.append(f"unknown structure IDs {sorted(unknown)} (valid: {sorted(known | candidates)})")
            if set(output.rebuild_structure_ids) & candidates:
                problems.append(f"{sorted(set(output.rebuild_structure_ids) & candidates)} are earlier candidates: "
                                "their plans were replaced, so they can be accepted but not rebuilt")
            if output.action == "revisit_evidence" and deps.revisions_left <= 0:
                problems.append("no revision budget is left: choose accept_candidates or stop_inconclusive")
            if output.action == "revisit_evidence" and not output.feedback.strip():
                problems.append("revisit_evidence needs concrete feedback for the planning and construction agents")
            if output.action == "rebuild_structures":
                if deps.rebuilds_left <= 0:
                    problems.append("no rebuild is left: choose accept_candidates, revisit_evidence or stop_inconclusive")
                if not output.rebuild_structure_ids:
                    problems.append("rebuild_structures needs rebuild_structure_ids")
                if not output.feedback.strip():
                    problems.append("rebuild_structures needs feedback saying what is wrong in each structure and what "
                                    "the rebuilt one must show")
            if output.action == "accept_candidates" and not output.accepted_structure_ids:
                problems.append("accept_candidates needs the accepted structure IDs")
            if (not known and not candidates and deps.state.failed_plans and deps.revisions_left > 0
                    and output.action != "revisit_evidence"):
                problems.append("no structure could be built but a revision is left: choose revisit_evidence and "
                                "explain the construction failures in feedback so the planner can fix the plans")
            if output.action in ("accept_candidates", "stop_inconclusive") and not deps.report_written:
                problems.append(f"{output.action} ends the run: call write_report first")
            if problems:
                raise ModelRetry("Fix the decision:\n- " + "\n- ".join(problems))
            return output

    async def review(self, state: ReconstructionState, store: ArtifactStore, revisions_left: int,
                     images: dict[str, bytes] | None = None, rebuilds_left: int = 0) -> RevisionDecision:
        """Review the current structures in `state`; `images` maps structure IDs to rendered views."""
        deps = RevisionDeps(state=state, store=store, revisions_left=revisions_left, rebuilds_left=rebuilds_left)
        text = reviser_prompt(state, revisions_left, rebuilds_left)
        prompt: list = [text]
        for sid, png in (images or {}).items():
            prompt += [f"Rendered top and side views of {sid}:", BinaryContent(data=png, media_type="image/png")]
        extra = {}
        if self.read_scripts:
            structures = store.root / "structures"
            structures.mkdir(parents=True, exist_ok=True)
            extra["workspace"] = self._workspace(structures)
        return await self.arun(prompt if images else text, deps=deps, **extra)


def reviser_prompt(state: ReconstructionState, revisions_left: int, rebuilds_left: int = 0) -> str:
    plans = state.plans.plans if state.plans else []
    parts = ["## Paper summary (material systems, text and figure evidence)",
             state.summary.model_dump_json(exclude_none=True),
             "## Characterization reports (per technique)", state.evidence.model_dump_json(exclude_none=True),
             "## Plans", state.plans.model_dump_json(exclude_none=True)]
    claims = claims_about(structural_claims(state.summary, state.evidence), {i for p in plans for i in p.system_ids})
    if claims:
        tested = {i: p.plan_id for p in plans for c in p.checks for i in c.claim_ids}
        unchecked = {i: f"{p.plan_id}: {u.reason}" for p in plans for u in p.unchecked for i in u.claim_ids}
        parts += ["## Structural claims about the plans' systems (the claim IDs the plans' checks cite)", "\n".join(
            f"- {c.claim_id} [{', '.join(c.system_ids) or 'all systems'}] {c.text}"
            + ("" if c.claim_id in tested else f" -- UNCHECKED ({unchecked[c.claim_id]})" if c.claim_id in unchecked
               else " -- no plan tests it") for c in claims)]
    if state.revision_history:
        lines = ["## Your earlier reviews in this run (oldest first)"]
        for k, d in enumerate(state.revision_history):
            lines.append(f"- Review {k}: {d.action}: {d.reasoning}")
            lines += [f"  - feedback: {d.feedback}"] if d.feedback else []
            lines += [f"  - {v.structure_id}: {'resembles' if v.resembles_paper else 'does not resemble'} the paper"
                      + (f" ({'; '.join(v.issues)})" if v.issues else "") for v in d.verdicts]
            lines += [f"  - check_claim on {c.structure_id}, {c.check_id} {c.kind} ({c.claim}): {c.status}: {c.detail}"
                      for c in state.review_checks if c.review == k]
            lines += [f"  - unresolved: {'; '.join(d.unresolved)}"] if d.unresolved else []
        parts.append("\n".join(lines))
    parts.append("## Constructed structures")
    current = [state.artifact(i) for i in state.current_ids]
    if not current:
        parts.append("No structure was built.")
    runs_shown = set()
    for a in current:
        rep = state.validation_reports.get(a.structure_id)
        claim_checks = [c for c in rep.checks if c.name.startswith("claim ")] if rep else []
        run_summary = a.construction_summary if a.construction_summary not in runs_shown else "(as above)"
        runs_shown.add(a.construction_summary)
        parts.append("\n".join([
            f"### {a.structure_id} (plan {a.plan_id}" + (f", models {', '.join(a.system_ids)})" if a.system_ids else ")"),
            a.description,
            f"built by {Path(a.script).name}: {a.method or 'no method reported'}",
            "measured targets: " + ("; ".join(f"{t.property} {t.measured} ({'met' if t.satisfied else 'NOT met'})"
                                              for t in a.target_checks) or "none reported"),
            f"deviations from plan: {'; '.join(a.deviations) or 'none reported'}",
            *([f"constructor's notes: {a.notes}"] if a.notes else []),
            *([f"construction run summary: {run_summary}"] if a.construction_summary else []),
            f"deterministic validation: {rep.summary(include_skipped=True) if rep else 'not run'}",
            *(["paper claims tested (the plan's checks):"]
              + [f"- {c.name.removeprefix('claim ')} {c.status}: {c.detail}" for c in claim_checks] if claim_checks else []),
        ]))
    earlier = [state.artifact(i) for i in state.candidate_ids if i not in state.current_ids]
    if earlier:
        parts += ["## Earlier candidates (built from earlier plan sets and passed validation: they can be accepted, "
                  "not rebuilt)", "\n".join(
                      f"- {a.structure_id} (plan {a.plan_id} of plan set {a.revision}"
                      + (f", models {', '.join(a.system_ids)}" if a.system_ids else "") + f"): "
                      + (a.description.splitlines()[0] if a.description else f"{a.formula}, {a.n_atoms} atoms")
                      + f"; validation {state.validation_reports[a.structure_id].summary()}" for a in earlier)]
    if state.failed_plans:
        parts += ["## Plans that could not be built"] + [f"- {k}: {v}" for k, v in state.failed_plans.items()]
    parts.append(f"Rebuilds left: {rebuilds_left}" + ("" if rebuilds_left > 0 else " (rebuild_structures is not available)"))
    parts.append(f"Revisions left: {revisions_left}"
                 + ("" if revisions_left > 0 else " (revisit_evidence is not available)"))
    return "\n\n".join(parts)


def reviser_system_prompt(read_scripts: bool = True) -> str:
    scripts = ("""
- Each structure was built by the standalone script named in its section (`<structure id>.py`).
  Read it with read_file when the summary leaves open how a feature was made (how the shell
  was assigned, where a defect sits, how a molecule was placed); judge the structure, not the
  coding style.""" if read_scripts else "")
    return """\
You are the Scientific Revision agent of Paper2Structure. You judge whether the constructed atomic
structures plausibly represent the system the paper studies: the material and phase, the exposed
surface or particle, the adsorbate identity and protonation state, the binding mode and
site, coverage, and any solvent. Compare every structure with the paper summary, the evidence
snapshot and the plan it realizes.

Rules
- Deterministic validation results are authoritative for geometry (overlaps, broken molecules,
  vacuum, bond-valence sums of the framework cations, like-charge contacts). Also check what they
  cannot see: compare the bond lengths and coordination numbers in each structure's description
  with what the material requires (e.g. octahedral cations six-coordinated, sensible
  cation-anion distances). Judge resemblance to the paper qualitatively; never invent numerical agreement scores
  and never treat the authors' proposed mechanism as proven.
- Accepting a structure means it is a reasonable starting structure for simulation, not that it is
  experimentally established. List the observations it does not explain and the assumptions it
  rests on under `unresolved`.
- Give a verdict for every structure. Check the measured targets against the plan and the
  evidence. The plans' checks test the paper's claims on every structure (listed under each
  structure); a failing claim means the structure contradicts the paper. The claims have their
  own IDs (`S1:lattice`, `XRD:C2`); a plan's checks are numbered within the plan (C1, C2, ...)
  and name the claim IDs they test. check_claim runs any such check yourself: use it for claims
  marked UNCHECKED or untested that a check can test, and for quantitative statements in the
  evidence the plans did not turn into checks.
- Your earlier reviews in this run (decisions, verdicts, checks) are listed: build on them, and
  say what changed when you depart from them.""" + scripts + """

Actions
- accept_candidates: one or more structures pass validation and represent the paper's system;
  list them in accepted_structure_ids. Keep several when the evidence cannot tell them apart.
  Earlier candidates (structures of plan sets a revisit replaced) may be accepted too.
- rebuild_structures: a plan is right but a built structure is not: wrong bond lengths or
  coordination, wrong composition or charge, missing or misplaced components, a geometry that
  does not realize the plan's description. List the structures in rebuild_structure_ids and write
  feedback naming, for each, what is wrong (with the measured values) and what the rebuilt
  structure must show. The Construction agent rebuilds their plans with your feedback and its
  previous scripts, without re-planning; structures you do not list are kept. Prefer it to
  revisit_evidence whenever the hypothesis stands.
- revisit_evidence: an important feature of the paper is missing or wrong in the structures, or
  plans could not be built, and re-planning from the same evidence could fix it. The planner gets
  your feedback with the paper summary, the characterization reports (which are kept, not
  re-analyzed) and how each earlier plan's structures fared (validation, failing claims, your
  verdicts), then planning and construction re-run; the current structures that pass validation
  stay acceptable as earlier candidates. Write concrete feedback naming what is wrong and what
  the paper indicates; for plans that could not be built, restate why.
- stop_inconclusive: the evidence cannot support a meaningful model, the structures cannot be
  repaired with the information available, or no revisions are left.

Report
- accept_candidates and stop_inconclusive end the run: call write_report first with your summary,
  assessment and next steps. Write them for a computational scientist who will simulate the
  accepted structures. rebuild_structures and revisit_evidence do not need a report."""


def save_report(state: ReconstructionState, store: ArtifactStore) -> Path:
    """final_report.md: the reviewer's summary, assessment and next steps (state.report) around
    the deterministic record: paper, evidence, plans, structures with validation and scripts,
    revision decisions, reproduction and limitations."""
    s, ev, narrative = state.summary, state.evidence, state.report
    claims = {c.claim_id: c for c in structural_claims(s, ev)} if s is not None and ev is not None else {}
    last = state.revision_history[-1] if state.revision_history else None
    accepted = set(last.accepted_structure_ids) if last and last.action == "accept_candidates" else set()
    status = "under review" if state.status == "running" else state.status
    lines = [f"# Paper2Structure report - {state.run_id}", "", f"**Status:** {status}", "",
             "## 1. Summary",
             narrative.summary if narrative else "No reviewer summary: the run ended before a final review.", ""]
    if s is not None:
        m = s.metadata
        lines += ["## 2. Paper and material systems",
                  f"- {m.title}" + (f" ({', '.join(m.authors[:3])}{' et al.' if len(m.authors) > 3 else ''}"
                                    f"{', ' + str(m.year) if m.year else ''})" if m.authors else ""),
                  f"- DOI: {m.doi or 'n/a'}; source: {s.source_pdf}", f"- {m.summary}", "",
                  "| system | name | form | space group | lattice parameters |", "|---|---|---|---|---|"]
        for sy in s.systems:
            lat = ", ".join(f"{x.quantity}={x.value}{' ' + x.unit if x.unit else ''}" for x in sy.structure.lattice_parameters)
            lines.append(f"| {sy.system_id} | {sy.name} | {sy.form or ''} | {sy.structure.space_group or ''} | {lat} |")
        lines.append("")
    if ev is not None:
        lines += ["## 3. Characterization (structural constraints per technique)",
                  "| claim | technique | property | systems | value | confidence |", "|---|---|---|---|---|---|"]
        for tech, r in sorted(ev.reports.items()):
            for c in r.structural_constraints:
                lines.append(f"| {tech}:{c.constraint_id} | {tech} | {c.property} | {', '.join(c.system_ids)} | "
                             f"{constraint_value(c)} | {c.confidence} |")
        if ev.skipped:
            lines += ["", "Techniques not analyzed: " + "; ".join(f"{k} ({v})" for k, v in sorted(ev.skipped.items()))]
        lines.append("")
    lines.append("## 4. Structural hypotheses and plans")
    for r, ps in enumerate(state.plan_history):
        lines.append(f"### Plan set {r}")
        for p in ps.plans:
            lines += [f"- **{p.plan_id}** ({p.model_type}): {p.hypothesis}",
                      f"  - model: {p.description}",
                      f"  - targets: {'; '.join(f'{t.property} {t.value}' for t in p.targets) or 'none'}",
                      f"  - claim checks: {'; '.join(f'{c.id} {c.kind}: {c.claim}' for c in p.checks) or 'none'}",
                      f"  - unchecked claims: {'; '.join(_unchecked(u, claims) for u in p.unchecked) or 'none'}",
                      f"  - evidence: {', '.join(p.evidence_ids) or 'none'}",
                      f"  - assumptions: {'; '.join(p.assumptions) or 'none stated'}"]
    lines += ["", "## 5. Constructed structures and validation",
              "Current: built for the latest plan set and not rebuilt since; candidate: an earlier plan set's structure "
              "that passed validation. The others were superseded.", "",
              "| structure | plan | models | formula | atoms | validation | current | accepted | files | script |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for a in state.artifacts:
        rep = state.validation_reports.get(a.structure_id)
        files = ", ".join(f"[{k}]({v})" for k, v in a.files.items())
        current = ("yes" if a.structure_id in state.current_ids
                   else "candidate" if a.structure_id in state.candidate_ids else "")
        lines.append(f"| {a.structure_id} | {a.plan_id} | {', '.join(a.system_ids)} | {a.formula} | {a.n_atoms} | "
                     f"{rep.summary(include_skipped=True) if rep else 'not validated'} | {current} | "
                     f"{'yes' if a.structure_id in accepted else ''} | "
                     f"{files} | [{Path(a.script).name}]({a.script}) |")
    claim_rows = [(a.structure_id, c) for a in state.artifacts if (rep := state.validation_reports.get(a.structure_id))
                  for c in rep.checks if c.name.startswith("claim ")]
    if claim_rows:
        lines += ["", "Paper claims tested on each structure (the plans' checks; skip: not applicable to that kind of "
                  "structure):", "", "| structure | check | result | detail |", "|---|---|---|---|"]
        lines += [f"| {sid} | {c.name.removeprefix('claim ')} | {c.status} | {c.detail} |" for sid, c in claim_rows]
    for a in state.artifacts:
        lines += ["", f"**{a.structure_id}** - {a.method or 'no method reported'}"]
        lines += [f"- target {t.property}: {t.measured} ({'met' if t.satisfied else 'NOT met'}"
                  + (f"; {t.method}" if t.method else "") + ")" for t in a.target_checks]
        lines += [f"- deviation: {d}" for d in a.deviations]
        lines += [f"- notes: {a.notes}"] if a.notes else []
    if state.failed_plans:
        lines += ["", "Plans that could not be built: " + "; ".join(f"{k}: {v}" for k, v in state.failed_plans.items())]
    lines += ["", "## 6. Revision decisions"]
    for k, d in enumerate(state.revision_history):
        lines += [f"- Review {k}: **{d.action}** - {d.reasoning}"]
        if d.rebuild_structure_ids:
            lines.append(f"  - rebuild: {', '.join(d.rebuild_structure_ids)}")
        if d.feedback:
            lines.append(f"  - feedback: {d.feedback}")
        for v in d.verdicts:
            lines.append(f"  - {v.structure_id}: {'resembles' if v.resembles_paper else 'does not resemble'} the paper"
                         + (f" ({'; '.join(v.issues)})" if v.issues else ""))
    if state.review_checks:
        lines += ["", "Claims the reviewer checked itself (check_claim):", "",
                  "| review | structure | check | claim | result | detail |", "|---|---|---|---|---|---|"]
        lines += [f"| {c.review} | {c.structure_id} | {c.check_id} {c.kind} | {c.claim} | {c.status} | {c.detail} |"
                  for c in state.review_checks]
    if narrative:
        lines += ["", "## 7. Assessment", narrative.assessment]
    if last and last.unresolved:
        lines += ["", "Unexplained observations and unverified assumptions:"] + [f"- {u}" for u in last.unresolved]
    if narrative and narrative.next_steps:
        lines += ["", "## 8. Next steps"] + [f"- {n}" for n in narrative.next_steps]
    lines += ["", "## 9. Reproduce and modify",
              "Each `structures/<id>.py` is the standalone script that generated `<id>`: it was re-run alone in an "
              "empty directory and reproduced the structure exactly. `python structures/<id>.py` rebuilds it into "
              "`structures/` of the current directory (needs the paper2structure package); edit the parameters at "
              "the top to explore variants. The agent's full workspaces, with every intermediate script, are in "
              "`construction/r<revision>_a<attempt>/<plan>/workspace/`."]
    lines += ["", "## 10. Limitations",
              "- The characterization reports are not re-analyzed on revision; the reviewer's feedback goes to the planner.",
              "- Techniques without a filled-in specification are not analyzed (see section 3).",
              "- Structures are unrelaxed starting geometries; no energies, relaxations or simulated observables "
              "(XPS, IR, XRD) were computed, so agreement with experiment is qualitative only.",
              "- Acceptance means a plausible simulation starting point, not an experimentally established structure."]
    if state.errors:
        lines += ["", "## Errors"] + [f"- {e}" for e in state.errors]
    path = store.path("final_report.md")
    path.write_text("\n".join(lines) + "\n")
    return path


def _unchecked(u, claims: dict) -> str:
    """An unchecked-claims entry with the claims' text."""
    return "; ".join(f"{i} ({claims[i].text})" if i in claims else i for i in u.claim_ids) + f" - {u.reason}"


def save_structures_readme(state: ReconstructionState, store: ArtifactStore) -> Path:
    """structures/README.md: which script rebuilds which structure, and how to run it."""
    lines = ["# Structures", "",
             "Every structure here comes with the standalone script that generated it (verified by re-running it "
             "alone in an empty directory). To rebuild or modify one:", "",
             "```bash", "pip install -e <paper2structure checkout>   # once", "python <id>.py   # writes structures/<name>.extxyz and .vasp", "```",
             "", "| structure | plan | formula | atoms | script | built as |", "|---|---|---|---|---|---|"]
    for a in state.artifacts:
        lines.append(f"| {a.structure_id} | {a.plan_id} | {a.formula} | {a.n_atoms} | {Path(a.script).name} | {a.name} |")
    path = store.path("structures", "README.md")
    path.write_text("\n".join(lines) + "\n")
    return path
