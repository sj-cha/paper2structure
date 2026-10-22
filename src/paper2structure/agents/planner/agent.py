from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pydantic_ai import ModelRetry, RunContext
from pymatgen.core import Composition
from rdkit import Chem

from paper2structure.agents.characterization.schemas import ExperimentalEvidence
from paper2structure.agents.planner.schemas import BulkSpec, PlanSet
from paper2structure.agents.structured_agent import StructuredAgent
from paper2structure.agents.summarizer.schemas import PaperGraph
from paper2structure.config import OverridesConfig
from paper2structure.tools.bulk_builder import build_bulk

log = logging.getLogger(__name__)


@dataclass
class PlanningDeps:
    evidence_ids: set[str]
    max_plans: int
    overrides: OverridesConfig
    max_atoms: int = 5000
    system_ids: set[str] = field(default_factory=set)  # the summarizer's material systems; empty skips the check


def _smiles_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.AddHs(mol).GetNumAtoms() if mol is not None else 0


def _smiles_elements(smiles: str) -> set[str] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms()}


def _check_bulk(bulk: BulkSpec) -> list[str]:
    """Build the bulk exactly as specified; a plan whose crystal cannot be built (wrong Wyckoff
    setting, overlapping sites, wrong composition) goes back to the planner, not to construction."""
    lat = bulk.lattice
    try:
        atoms = build_bulk(bulk.space_group, [s.species for s in bulk.sites], [list(s.coords) for s in bulk.sites],
                           a=lat.a, b=lat.b, c=lat.c, alpha=lat.alpha, beta=lat.beta, gamma=lat.gamma,
                           formula_units=bulk.formula_units)
    except Exception as e:  # noqa: BLE001 - every builder error is a planning error here
        return [f"bulk {bulk.formula} ({bulk.space_group}) cannot be built: {e}"]
    try:
        want = Composition(bulk.formula).reduced_formula
    except Exception:  # noqa: BLE001
        return [f"bulk formula {bulk.formula!r} is not a chemical formula"]
    got = Composition(atoms.get_chemical_formula()).reduced_formula
    if got != want:
        return [f"bulk sites generate {got}, not the stated formula {want}; check the Wyckoff positions"]
    return []


def check_plan_set(plans: PlanSet, deps: PlanningDeps) -> list[str]:
    """Deterministic checks of a plan set against the evidence snapshot and user overrides."""
    problems = []
    if len(plans.plans) > deps.max_plans:
        problems.append(f"{len(plans.plans)} plans given; at most {deps.max_plans} allowed")
    ov = deps.overrides
    for p in plans.plans:
        cited = (set(p.evidence_ids) | {e for c in p.crystals for e in c.lattice.evidence_ids}
                 | {e for t in p.targets for e in t.evidence_ids})
        unknown = cited - deps.evidence_ids
        if unknown:
            problems.append(f"{p.plan_id}: unknown evidence IDs {sorted(unknown)} (valid: {sorted(deps.evidence_ids)})")
        if not p.evidence_ids and not p.assumptions:
            problems.append(f"{p.plan_id}: cite supporting evidence IDs or state the assumptions it rests on")
        unknown_systems = set(p.system_ids) - deps.system_ids if deps.system_ids else set()
        if unknown_systems:
            problems.append(f"{p.plan_id}: unknown system IDs {sorted(unknown_systems)} (valid: {sorted(deps.system_ids)})")
        for crystal in p.crystals:
            if crystal.lattice.source == "paper" and not crystal.lattice.evidence_ids:
                problems.append(f"{p.plan_id}: lattice of {crystal.name!r} has source 'paper' but cites no evidence")
            problems.extend(f"{p.plan_id}: crystal {crystal.name!r}: {msg}" for msg in _check_bulk(crystal))
        if (p.surface is not None or p.nanocrystal is not None) and not p.crystals:
            problems.append(f"{p.plan_id}: a surface or nanocrystal spec needs the crystal it is cut from in `crystals`")
        if ov.model_type and p.model_type.strip().lower() != ov.model_type.strip().lower():
            problems.append(f"{p.plan_id}: user override requires model_type {ov.model_type!r}")
        if p.surface is not None:
            if ov.miller and tuple(p.surface.miller) != tuple(ov.miller):
                problems.append(f"{p.plan_id}: user override requires Miller index {tuple(ov.miller)}")
            if ov.supercell and tuple(p.surface.supercell) != tuple(ov.supercell):
                problems.append(f"{p.plan_id}: user override requires supercell {tuple(ov.supercell)}")
        if ov.lattice and p.crystals:
            for k, v in ov.lattice.items():
                got = getattr(p.crystals[0].lattice, k, None)
                if got is None or abs(float(got) - float(v)) > 1e-6:
                    problems.append(f"{p.plan_id}: user override requires lattice {k} = {v} (first crystal)")
        for ads in p.adsorbates:
            elements = _smiles_elements(ads.smiles)
            if elements is None:
                problems.append(f"{p.plan_id}: unparseable SMILES {ads.smiles!r} for {ads.name}")
                continue
            missing = set(ads.binding_atoms) - elements
            if missing:
                problems.append(f"{p.plan_id}: binding atoms {sorted(missing)} are not in {ads.smiles!r}")
            if ads.anchor_smarts:
                patt = Chem.MolFromSmarts(ads.anchor_smarts)
                if patt is None:
                    problems.append(f"{p.plan_id}: invalid anchor SMARTS {ads.anchor_smarts!r}")
                elif not Chem.AddHs(Chem.MolFromSmiles(ads.smiles)).HasSubstructMatch(patt):
                    problems.append(f"{p.plan_id}: anchor SMARTS {ads.anchor_smarts!r} does not match {ads.smiles!r}")
        molecular = (sum(a.count * _smiles_atoms(a.smiles) for a in p.adsorbates)
                     + sum((v.n_molecules or 0) * _smiles_atoms(v.smiles) for v in p.solvent))
        if molecular > deps.max_atoms:
            problems.append(f"{p.plan_id}: the adsorbates and solvent alone have {molecular} atoms (hydrogens included), "
                            f"over the {deps.max_atoms}-atom budget before any substrate: use fewer molecules, a smaller "
                            "model or a truncated adsorbate (e.g. a shorter alkyl chain) and state it as an assumption")
        for solvent in p.solvent:
            if _smiles_elements(solvent.smiles) is None:
                problems.append(f"{p.plan_id}: unparseable solvent SMILES {solvent.smiles!r}")
            if solvent.density is None and solvent.n_molecules is None:
                problems.append(f"{p.plan_id}: solvent {solvent.name!r} needs a density or a number of molecules")
    return problems


class LLMPlanner(StructuredAgent):
    """Agent 3: turns the paper summary and evidence snapshot into competing structure plans."""
    _output_type = PlanSet
    _deps_type = PlanningDeps
    _preflight_probe = ("Health check: return one plan with plan_id 'P1', model_type 'bulk', description 'fcc Cu "
                        "bulk', one crystal (space group Fm-3m, Cu at (0, 0, 0), a = 3.615 A, source literature).")

    def __init__(self, model, effort: str | None, *, max_plans: int, max_atoms: int, min_vacuum: float,
                 overrides: OverridesConfig, max_retry: int = 3):
        super().__init__(model, effort,
                         planner_system_prompt(max_plans, max_atoms, min_vacuum, overrides.describe()),
                         max_retry=max_retry)
        self.max_plans = max_plans
        self.max_atoms = max_atoms
        self.overrides = overrides

    def _register(self, agent) -> None:
        @agent.output_validator
        def _validate(ctx: RunContext[PlanningDeps], output: PlanSet) -> PlanSet:
            if ctx.deps is None:
                return output
            problems = check_plan_set(output, ctx.deps)
            if problems:
                log.info("planner output rejected: %s", problems)
                raise ModelRetry("Fix these problems and return the complete plan set again:\n- "
                                 + "\n- ".join(problems))
            return output

    async def plan(self, summary: PaperGraph, evidence: ExperimentalEvidence,
                   history: list[PlanSet], feedback: str) -> PlanSet:
        deps = PlanningDeps(evidence_ids={e.evidence_id for e in summary.evidence} | evidence.ids,
                            max_plans=self.max_plans, overrides=self.overrides, max_atoms=self.max_atoms,
                            system_ids={s.system_id for s in summary.systems})
        return await self.arun(planner_prompt(summary, evidence, history, feedback), deps=deps)


def planner_prompt(summary: PaperGraph, evidence: ExperimentalEvidence, history: list[PlanSet],
                   feedback: str) -> str:
    parts = ["## Paper summary (Agent 1: metadata, material systems, text and figure evidence)",
             summary.model_dump_json(indent=1, exclude={"nodes", "edges"}),
             "## Characterization reports (Agent 2: observations, findings and structural constraints per technique)",
             evidence.model_dump_json(indent=1)]
    if history:
        parts += ["## Previous plan sets (oldest first)"] + [h.model_dump_json(indent=1) for h in history]
    if feedback:
        parts += ["## Reviewer feedback on the structures built from the latest plans", feedback,
                  "Revise the plans to address this feedback."]
    else:
        parts.append("Write the plan set.")
    return "\n\n".join(parts)


def planner_system_prompt(max_plans: int, max_atoms: int, min_vacuum: float, overrides: str) -> str:
    return f"""\
You are the Structural Planning agent of Paper2Structure. From the paper summary and the
characterization reports you decide which atomic models best represent the system the paper
studies, and you write one construction plan per model. A separate Construction agent builds each
plan with pymatgen, ASE and RDKit; the models are starting structures for machine-learned
interatomic potential (MLIP) simulations.

Your inputs
- The paper summary (Agent 1): metadata, the material systems studied (IDs S1, S2, ...; each with
  synthesis and the structural facts the paper states: crystal system, space group, lattice
  parameters, site occupancy, phase purity) and evidence items (IDs E001, ...) extracted from the
  text and figures, tagged by technique.
- The characterization reports (Agent 2), one per technique: observations read from the figures
  and text (IDs O1, O2, ... within each report), findings, and structural constraints with their
  confidence.
- Cite evidence as summarizer evidence IDs ("E004") or as characterization observation IDs
  qualified by their technique ("XRD:O3"); no other IDs exist.
- A paper may study several material systems (a doping or homologous series, several phases or
  substrates). Give every plan the system_ids it models, and spend the plan limit on the systems
  and hypotheses that matter most for the paper's question.

The Construction agent is a coding agent: it writes its own Python (ASE, pymatgen, RDKit) on top
of a helper library, so any well-defined atomistic model can be built - core-shell or alloyed
particles, supported particles, interfaces and heterostructures, defects and dopants, stepped
surfaces, solvent mixtures with ions. Never simplify a model because no dedicated builder exists;
simplify only for scientific reasons or the atom budget, and list each simplification as an
assumption.

What each plan decides
- model_type: a short label ("slab", "nanocrystal", "core-shell nanocrystal", "supported
  nanoparticle", "interface", "defective slab", "bulk", ...). Choose what represents the experiment
  within the atom budget: particles much larger than a few nm are usually better represented by
  slabs of their dominant facets (with shell layers for a core-shell particle); small particles,
  or questions about edges, corners, interfaces or shell thickness, call for an explicit particle.
- description: what to build, component by component, and how the components are arranged
  (which crystal forms the core, the shell, the support; orientation relationships; where
  defects sit), precise enough to build without the paper.
- crystals: every crystalline component, each named ("bulk", "core", "shell", "support") as the
  space group (Hermann-Mauguin symbol as pymatgen spells it, e.g. "Fm-3m", "Pm-3m", "P4_2/mnm",
  "I4_1/amd", "P6_3mc", "F-43m"), its asymmetric unit as Wyckoff representatives with fractional
  coordinates in the ITA standard setting (origin choice 1 where a group has two origins;
  hexagonal axes for rhombohedral groups) or in the paper's own setting written as its symbol
  ("Pbnm", "Pcab", "Cc2m"; lattice parameters and coordinates in that same setting), lattice parameters and formula units per conventional
  cell. Use lattice parameters reported in the evidence (source "paper", cite the evidence ID);
  otherwise literature values (source "literature", listed as an assumption). Each crystal is
  built from its sites exactly as written and must be complete: every atom of its formula, bonded
  neighbours, no overlaps. A crystal you cannot give complete Wyckoff sites for - hybrid
  organic-inorganic or molecular crystals (e.g. organic cations between inorganic layers),
  frameworks with guest molecules, cells the paper only partly determines - stays out of
  `crystals`: give its space group, lattice parameters, layer stacking or connectivity and its
  molecules (SMILES with formal charges, number per cell) in the description, and the Construction
  agent builds it in code. Never list a partial sublattice or an isolated layer as a crystal.
- surface (optional helper): Miller index, termination described by its exposed species, minimum
  thickness (A), in-plane supercell, vacuum (at least {min_vacuum} A) and a fixed bottom region (A).
- nanocrystal (optional helper): shape and size (closed-shell shapes for elemental metals; spheres
  or faceted polyhedra cut from the crystal for compounds), facets, and oxidation states when the
  particle must be charge-neutral. For multi-component particles give the overall size here and
  the regions in the description.
- adsorbates: SMILES with explicit formal charges encoding the protonation state, the
  binding atoms (one element for monodentate binding, two for bidentate; keep this consistent with
  the hypothesis) and an anchor SMARTS for the binding group, mode "adsorb" (on top of surface
  atoms) or "replace" (the binding group takes the place of one surface atom, or of two for a
  bridge site, e.g. a deprotonated acid substituting surface halides), site kind (top/bridge/
  hollow), site elements (for "replace", the elements being replaced), number of molecules, and an
  expected donor-surface distance. Co-adsorbed fragments such as a transferred proton ("[H+]")
  are separate monatomic adsorbates.
- solvent: only when explicit solvent is essential to the question; one entry per species (ions
  included) with a density or a number of molecules, and a layer thickness for slabs.
- targets: the quantitative features the evidence fixes and the built structure must show (core
  diameter, shell thickness, coverage, exposed sites, dopant fraction, ...), each with its value,
  tolerance and evidence IDs. The Construction agent measures them on the built structure.

Scientific rules
- Distinguish observations from interpretations. Every plan states its hypothesis, cites the
  evidence IDs that support it and lists every assumption the evidence does not establish
  (literature lattice values, termination, coverage, protonation, binding mode, simplifications).
- When the evidence admits several structural interpretations (binding mode, site, termination,
  protonation, coverage), write competing plans instead of forcing one answer: at most {max_plans}.
- Ordered models only: for partial occupancy, doping or mixing, model an ordered representative,
  e.g. a supercell with explicit substitutions at the reported fraction (spread out, not clustered
  unless the evidence says so), or an end-member when the atom budget forbids it; state the choice
  as an assumption.
- Keep every model below {max_atoms} atoms including adsorbates and solvent.
- Never invent experimental values; if the evidence is silent, say so.

User overrides (hard constraints): {overrides}

On revision you receive your previous plans and the reviewer's feedback. Change what the feedback
identifies; do not resubmit a plan that was judged not to resemble the paper."""
