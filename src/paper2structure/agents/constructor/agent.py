"""Agent 4, the Construction agent: a pydantic-ai-harness Coder that writes and runs its own Python
in a sandboxed workspace, with paper2structure.tools as an optional helper library. Every
structure it reports is rebuilt by running its script alone in an empty directory, so the script
handed to the user is exactly the code that generated the structure."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from ase.io import read
from pydantic_ai import BinaryContent, ModelRetry, RunContext, UsageLimits
from pydantic_ai.messages import ToolReturn
from pydantic_ai.workspaces import CommandResult, LocalWorkspaceBackend
from pydantic_ai_harness import Memory, Planning, Skills, SystemReminders
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.memory import FileStore
from pydantic_ai_harness.system_reminders import Reminder

from paper2structure.agents.characterization.schemas import ExperimentalEvidence
from paper2structure.agents.constructor import skills
from paper2structure.agents.constructor.schemas import ConstructionReport
from paper2structure.agents.constructor.workspace import make_workspace, replay
from paper2structure.agents.planner.schemas import StructurePlan
from paper2structure.agents.structured_agent import StructuredAgent
from paper2structure.agents.summarizer.schemas import PaperGraph
from paper2structure.config import Paper2StructureConfig, ValidationConfig, simulation_brief
from paper2structure.tools.geometry_checks import validate_structure
from paper2structure.tools.render import render_png
from paper2structure.tools.schemas import CheckResult, ValidationReport
from paper2structure.tools.structure_export import EMIT_DIR, MANIFEST, NAME, describe, structure_path

log = logging.getLogger(__name__)

WORKSPACE = "workspace"  # the agent's directory inside an attempt directory
REPLAYS = "replay"  # from-scratch runs of the final scripts, next to it
MAX_PARALLEL_REPLAYS = 4


@dataclass
class ConstructionDeps:
    workdir: Path  # attempt directory holding workspace/ (the agent's) and replay/ (verification)
    plan: StructurePlan  # the plan this run builds
    cfg: Paper2StructureConfig
    verified: dict[str, Path] = field(default_factory=dict)  # structure name -> extxyz from the replay
    scripts: dict[str, Path] = field(default_factory=dict)  # structure name -> its script in the workspace
    reports: dict[str, ValidationReport] = field(default_factory=dict)  # structure name -> gate report
    replays: dict[Path, tuple[str, CommandResult]] = field(default_factory=dict)  # script -> (content hash, result)
    gate_rejections: int = 0  # final reports sent back by the validation gate

    @property
    def workspace(self) -> Path:
        return self.workdir / WORKSPACE

    def manifest(self) -> dict[str, dict]:
        """emit()'s records by structure name. The agent can edit the file: a broken one is sent
        back to it (ModelRetry), since every caller runs inside a tool or the output validator."""
        path = self.workspace / EMIT_DIR / MANIFEST
        if not path.exists():
            return {}
        try:
            entries = json.loads(path.read_text())
            out = {e["name"]: e for e in entries}
            if any(not {"plan_id", "script"} <= set(e) for e in entries):
                raise KeyError("an entry lacks plan_id or script")
            return out
        except (ValueError, TypeError, KeyError) as e:
            raise ModelRetry(f"{EMIT_DIR}/{MANIFEST} is malformed ({type(e).__name__}: {e}): delete it and run your "
                             "build scripts again") from e

    def emitted(self, name: str) -> Path:
        return structure_path(self.workspace, name)

    def not_emitted(self, name: str) -> Optional[str]:
        """The message for a structure name the agent never emitted, else None."""
        if self.emitted(name).exists():
            return None
        return f"no structure {name!r} in {EMIT_DIR}/; emitted so far: {sorted(self.manifest())}"

    def replay_dir(self, script: Path) -> Path:
        """Replay directory named after the script's workspace-relative path (a/b/build.py -> a__b__build)."""
        rel = script.relative_to(self.workspace.resolve()).with_suffix("")
        return self.workdir / REPLAYS / "__".join(rel.parts)


def _tail(text: str, n: int = 2000) -> str:
    text = text.strip()
    return text if len(text) <= n else "...\n" + text[-n:]


def _read(path: Path):
    """The atoms in an extxyz file, or None when it is missing or unreadable (the agent can
    overwrite structure files with anything)."""
    try:
        return read(path)
    except Exception:  # noqa: BLE001
        return None


def _same_atoms(x, y, tol: float = 1e-3) -> bool:
    return (x is not None and y is not None and len(x) == len(y)
            and x.get_chemical_symbols() == y.get_chemical_symbols()
            and np.allclose(x.cell.array, y.cell.array, atol=tol) and np.allclose(x.positions, y.positions, atol=tol))


def _validate(deps: ConstructionDeps, atoms, name: str, system_ids: Optional[list[str]] = None) -> ValidationReport:
    """The validation gate on one structure: tools.geometry_checks with the campaign's `validation`
    settings, the plan's oxidation states for the optional charge check and the plan's checks of the
    paper's claims for the systems the structure models (as reported or emitted; none: all the
    plan's). A structure the checks cannot process fails."""
    systems = system_ids or deps.manifest().get(name, {}).get("system_ids") or []
    try:
        return validate_structure(atoms, name, max_atoms=deps.cfg.planning.max_atoms,
                                  oxidation_states=deps.plan.oxidation_states(), claim_checks=deps.plan.checks,
                                  system_ids=systems, **deps.cfg.validation.model_dump())
    except Exception as e:  # noqa: BLE001 - a broken structure is the agent's to fix, not the end of the run
        log.warning("validating %s failed: %s", name, e, exc_info=True)
        return ValidationReport(structure_id=name, passed=False, checks=[CheckResult(
            name="validation_error", status="fail", detail=f"{type(e).__name__}: {e}")])


def check_emitted(deps: ConstructionDeps, name: str) -> str:
    """Summary and gate checks of an emitted structure (check_structure's body)."""
    missing = deps.not_emitted(name)
    if missing:
        return missing
    atoms = _read(deps.emitted(name))
    if atoms is None:
        return f"could not read {EMIT_DIR}/{name}.extxyz"
    report = _validate(deps, atoms, name)
    try:
        summary = describe(atoms)
    except Exception as e:  # noqa: BLE001
        summary = f"could not describe {name}: {type(e).__name__}: {e}"
    emitted_for = deps.manifest()[name]["plan_id"]
    other = ("" if emitted_for == deps.plan.plan_id else f"\nnote: emitted for plan {emitted_for!r}: emit it with "
             f"plan_id={deps.plan.plan_id!r}, or the final report is rejected")
    return f"{summary}\nvalidation: {report.summary()}{other}"


def check_report(deps: ConstructionDeps, output: ConstructionReport) -> tuple[list[str], dict[Path, list[str]]]:
    """Bookkeeping problems of a final report (another plan, invalid names, structures never emitted,
    emitted by inline code or by another script, an empty report), and the reported structures by
    script."""
    problems = []
    manifest = deps.manifest()
    workspace = deps.workspace.resolve()
    by_script: dict[Path, list[str]] = {}
    pid = deps.plan.plan_id
    for f in output.failed:
        if f.plan_id != pid:
            problems.append(f"failed plan {f.plan_id!r} is not the plan to build ({pid})")
    for s in output.structures:
        entry = manifest.get(s.name)
        if s.plan_id != pid:
            problems.append(f"{s.name!r}: {s.plan_id!r} is not the plan to build ({pid})")
        if not NAME.match(s.name):
            problems.append(f"{s.name!r} is not a valid structure name (letters, digits, '_', '-', '.'; max 64)")
            continue
        if entry is None:
            problems.append(f"{s.name!r} was never emitted (emitted: {sorted(manifest)})")
            continue
        if entry["plan_id"] != s.plan_id:
            problems.append(f"{s.name!r} was emitted for plan {entry['plan_id']}, not {s.plan_id}")
        if entry["script"] == "<inline>":
            problems.append(f"{s.name!r} was emitted by inline code; final structures come from a standalone "
                            "script file run as `python <script>`")
            continue
        script = (workspace / s.script).resolve()
        if script != (workspace / entry["script"]).resolve():
            problems.append(f"{s.name!r} was last emitted by {entry['script']}, not by {s.script}")
        elif not script.is_file() or workspace not in script.parents:  # deleted, or run from outside
            problems.append(f"script {s.script!r} is not a file in the workspace")
        else:
            by_script.setdefault(script, []).append(s.name)
    if not output.structures and not output.failed:
        problems.append(f"build plan {pid} or report it as failed with the reason")
    return problems, by_script


async def replay_scripts(deps: ConstructionDeps, scripts: list[Path]) -> dict[Path, CommandResult]:
    """Run each script alone in an empty directory, up to MAX_PARALLEL_REPLAYS at once. A script
    whose content already reproduced in this run is not run again (its replay directory is kept)."""
    limit = asyncio.Semaphore(MAX_PARALLEL_REPLAYS)

    async def run(script: Path) -> CommandResult:
        digest = hashlib.sha256(script.read_bytes()).hexdigest()
        cached = deps.replays.get(script)
        if cached and cached[0] == digest:
            return cached[1]
        async with limit:
            res = await replay(script, deps.replay_dir(script), deps.cfg.construction)
        if res.exit_code == 0:
            deps.replays[script] = (digest, res)
        else:
            deps.replays.pop(script, None)
        return res

    return dict(zip(scripts, await asyncio.gather(*(run(s) for s in scripts))))


async def verify_report(deps: ConstructionDeps, output: ConstructionReport) -> list[str]:
    """Problems with a final report: its bookkeeping (check_report), scripts that do not reproduce
    their structures when run alone from scratch, then the validation gate on the replayed ones."""
    problems, by_script = check_report(deps, output)
    if problems:
        return problems
    results = await replay_scripts(deps, list(by_script))
    fresh_atoms = {}
    for script, names in by_script.items():
        res = results[script]
        if res.exit_code != 0:
            problems.append(f"{script.name} fails when run alone in an empty directory (exit code {res.exit_code}):\n"
                            f"{_tail(res.stderr)}")
            continue
        for name in names:
            fresh = structure_path(deps.replay_dir(script), name)
            atoms = _read(fresh)
            if atoms is None:
                problems.append(f"{script.name} run alone in an empty directory does not emit {name!r}")
            elif not _same_atoms(atoms, _read(deps.emitted(name))):
                problems.append(f"{name!r} from a fresh run of {script.name} differs from {EMIT_DIR}/{name}.extxyz: "
                                "make the script deterministic (SEED passed to every random step) and independent "
                                "of files from earlier runs, then run it again")
            else:
                deps.verified[name] = fresh
                deps.scripts[name] = script
                fresh_atoms[name] = atoms
    if problems:
        return problems
    return gate_report(deps, output, fresh_atoms)


def gate_report(deps: ConstructionDeps, output: ConstructionReport, atoms: dict) -> list[str]:
    """The validation gate on the replayed structures; their reports go to the workflow. It sends a
    report back at most `construction.gate_retries` times; after that, failures pass on to
    ValidateStructures (repair loop, reviewer) so the gate never stalls a run."""
    failing = {}
    for s in output.structures:
        report = deps.reports[s.name] = _validate(deps, atoms[s.name], s.name, s.system_ids)
        if not report.passed:
            failing[s.name] = "; ".join(f"{c.name}: {c.detail}" for c in report.failures())
    if not failing:
        return []
    if deps.gate_rejections >= deps.cfg.construction.gate_retries:
        log.warning("validation gate still failing after %d rejections; passing on to validation: %s",
                    deps.gate_rejections, sorted(failing))
        return []
    deps.gate_rejections += 1
    return [f"validation gate: {name} fails - {why}" for name, why in failing.items()] + [
        "fix the script and re-run it (check_structure shows the same checks), or report the plan as failed"]


class LLMConstructor(StructuredAgent):
    """Agent 4: realizes structure plans as standalone Python scripts, written and run by a coding
    agent in a sandboxed workspace."""
    _output_type = ConstructionReport
    _deps_type = ConstructionDeps
    _preflight_probe = "Health check: do not call any tool; return an empty report with summary 'ok'."

    def __init__(self, model, effort: str | None, cfg: Paper2StructureConfig):
        super().__init__(model, effort, constructor_system_prompt(cfg), max_retry=cfg.campaign.max_retry,
                         output_retries=cfg.campaign.max_retry + cfg.construction.gate_retries)
        self.cfg = cfg

    def _capabilities(self) -> list:
        cc = self.cfg.construction
        caps = [Coder(instructions=CODER_ADDENDUM, sub_agents=cc.delegation), Skills(skills.WORKSPACE_LIBRARY)]
        if cc.planning_tool:
            caps.append(Planning())
        if cc.reminder_every > 0:
            caps.append(SystemReminders(reminders=[Reminder(REMINDER, interval=cc.reminder_every)]))
        if cc.memory_dir:
            store = Path(cc.memory_dir).expanduser().resolve()
            store.mkdir(parents=True, exist_ok=True)
            caps.append(Memory(FileStore(".", workspace=LocalWorkspaceBackend(store))))
        return caps

    def _register(self, agent) -> None:
        @agent.tool
        def check_structure(ctx: RunContext[ConstructionDeps], name: str) -> str:
            """Summarize an emitted structure (structures/<name>.extxyz) and run the workflow's
            deterministic validation on it (the gate your final report must pass): overlaps,
            molecule integrity against the SMILES, adsorbate-substrate binding, close and
            like-charge contacts, vacuum, the atom budget and the plan's checks of the paper's
            claims ("claim C1", ...). Fix every FAIL."""
            return check_emitted(ctx.deps, name)

        if self.cfg.campaign.use_images:
            @agent.tool
            def view_structure(ctx: RunContext[ConstructionDeps], name: str) -> ToolReturn | str:
                """Show rendered top and side views of an emitted structure (by its emit() name)."""
                missing = ctx.deps.not_emitted(name)
                if missing:
                    return missing
                try:
                    atoms = read(ctx.deps.emitted(name))
                    png = render_png(atoms)
                except Exception as e:  # noqa: BLE001
                    return f"could not render {EMIT_DIR}/{name}.extxyz: {type(e).__name__}: {e}"
                return ToolReturn(return_value=f"top and side views of {name} ({atoms.get_chemical_formula()})",
                                  content=[BinaryContent(data=png, media_type="image/png")])

        @agent.output_validator
        async def _validate(ctx: RunContext[ConstructionDeps], output: ConstructionReport) -> ConstructionReport:
            if ctx.deps is None:
                return output
            problems = await verify_report(ctx.deps, output)
            if problems:
                log.info("construction report rejected: %s", problems)
                raise ModelRetry("Fix these problems, then give the final report again:\n- " + "\n- ".join(problems))
            return output

    async def construct(self, plan: StructurePlan, workdir: Path, seed: int, feedback: str = "",
                        previous_scripts: Optional[dict[str, str]] = None, summary: Optional[PaperGraph] = None,
                        evidence: Optional[ExperimentalEvidence] = None) -> tuple[ConstructionReport, ConstructionDeps]:
        """Build `plan` in a fresh workspace under `workdir`; returns the verified report and the deps
        holding the replayed structures, their scripts and their gate reports."""
        deps = ConstructionDeps(workdir=Path(workdir), plan=plan, cfg=self.cfg)
        for leftover in (deps.workspace, deps.workdir / REPLAYS):  # a reused run directory: start clean
            shutil.rmtree(leftover, ignore_errors=True)
        previous_scripts = previous_scripts or {}
        prepare_workspace(deps.workspace, plan, feedback, previous_scripts, summary, evidence, self.cfg.validation)
        cc = self.cfg.construction
        report = await self.arun(constructor_prompt(plan, seed, feedback, previous_scripts),
                                 deps=deps, usage_limits=UsageLimits(request_limit=cc.max_requests,
                                                                     tool_calls_limit=cc.max_tool_calls),
                                 workspace=make_workspace(deps.workspace, cc))
        return report, deps

    def preflight(self, **run_kwargs) -> tuple[bool, str]:
        with tempfile.TemporaryDirectory(prefix="p2s-preflight-") as tmp:
            skills.install(Path(tmp), self.cfg.validation)
            return super().preflight(workspace=make_workspace(Path(tmp), self.cfg.construction), **run_kwargs)


def prepare_workspace(workspace: Path, plan: StructurePlan, feedback: str, previous_scripts: dict[str, str],
                      summary: Optional[PaperGraph], evidence: Optional[ExperimentalEvidence],
                      validation: ValidationConfig) -> None:
    """Task files, the previous attempt's scripts, the skills library (with the run's gate
    thresholds) and a README for the user."""
    task = workspace / "task"
    task.mkdir(parents=True, exist_ok=True)
    (task / "plan.json").write_text(plan.model_dump_json(indent=1))
    if summary is not None:
        (task / "paper_summary.json").write_text(summary.model_dump_json(exclude_none=True))
    if evidence is not None:
        (task / "evidence.json").write_text(evidence.model_dump_json(exclude_none=True))
    if feedback:
        (task / "feedback.md").write_text(feedback + "\n")
    for name, code in previous_scripts.items():  # name may hold a subfolder, e.g. last_attempt/build.py
        path = workspace / "previous" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)
    skills.install(workspace, validation)
    (workspace / "README.md").write_text(WORKSPACE_README)


WORKSPACE_README = """\
# Construction workspace

Written by the Paper2Structure Construction agent.

- `task/`: the plan it was asked to build (plan.json), the paper summary and evidence it was given,
  and feedback from a previous attempt (feedback.md), if any.
- `*.py`: its scripts. Each final script is standalone: `python <script>.py` rebuilds its structures
  into `structures/` (`<name>.extxyz` with per-atom roles and molecules, `<name>.vasp` POSCAR).
  Edit the parameters at the top to explore variants. Needs the paper2structure package installed.
- `.agents/skills/`: the guidance the agent loaded (the paper2structure.tools reference and recipes).
- `structures/manifest.json`: every emitted structure and the script that emitted it.
"""

CODER_ADDENDUM = """\
In this workspace the software you write builds atomic structures: follow the Structure
Construction contract above. Scratch checks may run inline (`python - <<'EOF' ... EOF`); final
structures come only from the standalone build scripts, which stay in the workspace."""

REMINDER = ("Final structures come from standalone scripts (one file, SEED passed to every random step, "
            "emit() for each structure), pass check_structure (overlaps, adsorbate binding, ...), "
            "and are re-run from scratch and validated again before the report is accepted.")


def constructor_prompt(plan: StructurePlan, seed: int, feedback: str, previous_scripts: dict[str, str]) -> str:
    parts = ["## Plan to build (also in task/plan.json)", plan.model_dump_json(indent=1), f"Use SEED = {seed}."]
    if feedback:
        parts += ["## Feedback on the previous attempt (also in task/feedback.md)", feedback]
    if previous_scripts:
        parts.append("The scripts of the plan's current structures (named after the first structure each built) and "
                     "of an attempt that built nothing (last_attempt/) are in previous/: "
                     + ", ".join(sorted(previous_scripts)) + ". Start from them and address the feedback above.")
    parts.append("Build the plan above (or report why it cannot be built), then give the final report.")
    return "\n\n".join(parts)


def constructor_system_prompt(cfg: Paper2StructureConfig) -> str:
    charge = ""
    if cfg.validation.charge_neutrality:
        charge = (", and zero net formal charge (substrate oxidation states plus the SMILES charges: balance "
                  "anionic adsorbates with excess surface cations, or the reverse)")
    views = " view_structure(name) shows rendered views." if cfg.campaign.use_images else ""
    pins = cfg.overrides.pins()
    technical = "search settings, rounding of thickness" + ("" if cfg.overrides.supercell else " or supercell")
    pinned = (f"\n- User-pinned choices, already in the plan: {pins}. Never change them, not even as technical "
              "parameters; apply them also when the plan describes the model only in its description; if the plan "
              "cannot be built with them, report it as failed." if pins else "")
    return f"""\
You are the Structure Construction agent of Paper2Structure. You turn a construction plan into
atomic structures by writing and running Python in this workspace; {simulation_brief(cfg.campaign.simulation)}.
Any Python with ASE, pymatgen, RDKit, numpy and scipy is
available. paper2structure.tools is a helper library for the common steps (crystals from Wyckoff
positions, molecules in crystal cavities, slab terminations, surface sites, adsorbate placement, particles, solvent, bookkeeping):
use it where it fits and write your own code for everything else (core-shell particles,
interfaces, defects, supported particles, ...). The skills describe the library and recipes; load
structure-toolkit before writing the first script.

The workspace
- task/plan.json holds the plan; task/paper_summary.json (the paper's material systems and
  evidence) and task/evidence.json (the characterization reports) the context it comes from;
  task/feedback.md and previous/ the feedback on and the scripts of a previous attempt.
- Commands run in a sandbox: no network, and only this workspace is writable.

The deliverable: one standalone build script for the plan (or one per group of closely related
structures), e.g. build_P1_core_shell.py. Users will run and modify these scripts, so:
- One file that imports only installed packages and reads no files. Named parameters at the top
  (lattice constants, sizes, coverage, SEED = the given seed) with comments tying them to the plan,
  then clear steps. Pass SEED to every random step.
- Atomic positions come from code: crystal symmetry, lattice vectors, geometry and searches.
  Never type coordinate lists from memory. Crystals come from the plan's Wyckoff sites
  (build_bulk); the molecules of a hybrid or molecular crystal go in with fill_cavities.
- Molecules are placed with place_adsorbates or add_solvent, or registered with add_molecule, so
  validation can check their bonds.
- Each finished structure is registered with emit(atoms, name=..., plan_id=..., notes=...), which
  writes structures/<name>.extxyz and .vasp. When a plan models several systems, pass the ones
  each structure models: emit(..., system_ids=["S2"]).

How to work
- Run each script with the shell (`python build_P1_....py`), print what you need (describe(),
  measurements), then call check_structure(name) on every emitted structure and fix every
  FAIL. The same deterministic checks gate your final report: no overlapping atoms, molecules
  matching their SMILES, every adsorbate bound to the substrate (an electronegative donor on an
  ionic surface binds a cation), vacuum and the atom budget{charge}, and the plan's checks: the
  paper's claims (lattice, space group, composition, bond lengths and angles, coordination,
  hydrogen bonds, layer counts). A failing claim check means the structure contradicts the
  paper: change how it is built (e.g. Wyckoff coordinates that give the reported space group and
  bond angles), never the check.{views}
- Measure every plan target on the built structure with code and report the values.
- Change only technical parameters ({technical}, a truncated adsorbate) and report every
  deviation from the plan. If the plan cannot be built as specified, report it as failed with the
  reason instead of building something else.{pinned}
- Limits: at most {cfg.planning.max_atoms} atoms per structure; at least {cfg.validation.min_vacuum} A of
  vacuum between periodic images of slabs and particles (unless solvent fills it).
- Pass notes=... to emit() for what the reviewer must know about a structure (simplifications,
  choices the plan left open); they are shown with the structure.

Final report: for every structure that realizes the plan, its emit() name, plan_id, the script
that builds it, how it was built (toolkit steps and your own code), the measured targets and the
deviations; or the plan as failed, with the reason. Each listed script is then run alone in an
empty directory and must reproduce its structures exactly."""
