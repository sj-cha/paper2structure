from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pydantic_ai import ModelRetry, RunContext
from pymatgen.core import Composition
from rdkit import Chem

from paper2structure.agents.characterization.schemas import ExperimentalEvidence
from paper2structure.agents.planner.claims import Claim, claims_about, structural_claims
from paper2structure.agents.planner.schemas import BulkSpec, PlanSet, StructurePlan
from paper2structure.agents.structured_agent import StructuredAgent
from paper2structure.agents.summarizer.schemas import PaperGraph
from paper2structure.config import OverridesConfig, simulation_brief
from paper2structure.tools._symmetry import resolve, space_group_number
from paper2structure.tools.bulk_builder import build_bulk
from paper2structure.tools.structure_export import NAME

log = logging.getLogger(__name__)


@dataclass
class PlanningDeps:
    evidence_ids: set[str]
    max_plans: int
    overrides: OverridesConfig
    max_atoms: int = 5000
    system_ids: set[str] = field(default_factory=set)  # the summarizer's material systems; empty skips the check
    claims: dict[str, Claim] = field(default_factory=dict)  # every plan checks the claims about its systems


def _smiles_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.AddHs(mol).GetNumAtoms() if mol is not None else 0


def _smiles_elements(smiles: str) -> set[str] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms()}


def _build(bulk: BulkSpec):
    lat = bulk.lattice
    return build_bulk(bulk.space_group, [s.species for s in bulk.sites], [list(s.coords) for s in bulk.sites],
                      a=lat.a, b=lat.b, c=lat.c, alpha=lat.alpha, beta=lat.beta, gamma=lat.gamma,
                      formula_units=None if bulk.molecules else bulk.formula_units, check_bonded=not bulk.molecules)


def describe_crystal(bulk: BulkSpec) -> str:
    """What a crystal spec builds (the try_crystal tool): the problems the plan check would report,
    or the generated cell's composition, symmetry, coordination with bond lengths, and the angles
    at atoms bridging two others of one element (M-O-M, Si-O-Si, ...)."""
    import io
    from contextlib import redirect_stdout

    with redirect_stdout(io.StringIO()):
        problems, atoms = _built(bulk)
    if problems:
        return "PROBLEMS:\n- " + "\n- ".join(problems)
    try:
        return _crystal_report(bulk, atoms)
    except Exception as e:  # noqa: BLE001 - a failed analysis is feedback, not the end of the planning run
        log.warning("try_crystal analysis failed for %s: %s", bulk.space_group, e)
        return f"the crystal builds and passes the plan check ({atoms.get_chemical_formula()}), but its analysis failed: {e}"


def _crystal_report(bulk: BulkSpec, atoms) -> str:
    import numpy as np
    import spglib
    from ase.neighborlist import neighbor_list

    from paper2structure.tools._geometry import rcov

    sym = np.array(atoms.get_chemical_symbols())
    found = {p: spglib.get_spacegroup((atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers), symprec=p)
             for p in (0.01, 0.1)}
    setting = resolve(bulk.space_group)
    lines = [f"framework {atoms.get_chemical_formula()} ({len(atoms)} atoms), "
             + (f"plus {', '.join(f'{m.per_cell} {m.name}' for m in bulk.molecules)} per cell; " if bulk.molecules else "")
             + f"requested {bulk.space_group} = {setting.label}, found {found[0.01]} at "
               f"symprec 0.01 A and {found[0.1]} at 0.1 A"]
    i, j, d, D = neighbor_list("ijdD", atoms, 1.25 * rcov(sym))
    for el in sorted(set(sym)):
        parts = []
        for nb in sorted(set(sym[j[sym[i] == el]])):
            m = (sym[i] == el) & (sym[j] == nb)
            cn = np.bincount(i[m], minlength=len(atoms))[sym == el]
            parts.append(f"{cn.min()}-{cn.max()} {nb} at {d[m].min():.2f}-{d[m].max():.2f} A")
        lines.append(f"{el}: " + ("; ".join(parts) if parts else "no neighbour within 1.25 x covalent radii"))
    for el in sorted(set(sym)):
        angles = {}
        for k in np.where(sym == el)[0]:
            m = i == k
            for nb in set(sym[j[m]]):
                v = D[m][sym[j[m]] == nb]
                if len(v) == 2:
                    cos = v[0] @ v[1] / (np.linalg.norm(v[0]) * np.linalg.norm(v[1]))
                    angles.setdefault(nb, []).append(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
        lines += [f"{nb}-{el}-{nb} bridging angles: {min(a):.1f}-{max(a):.1f} deg" for nb, a in sorted(angles.items())]
    return "\n".join(lines)


def _check_bulk(bulk: BulkSpec) -> list[str]:
    """Build the bulk exactly as specified; a plan whose crystal cannot be built (wrong Wyckoff
    setting, overlapping sites, wrong composition) goes back to the planner, not to construction."""
    return _built(bulk)[0]


def _built(bulk: BulkSpec):
    """(problems, atoms): the plan check's problems with the crystal, and the crystal when it builds."""
    try:
        atoms = _build(bulk)
    except Exception as e:  # noqa: BLE001 - every builder error is a planning error here
        return [f"bulk {bulk.formula} ({bulk.space_group}) cannot be built: {e}"], None
    if bulk.molecules:
        return _check_molecular_crystal(bulk, Composition(atoms.get_chemical_formula())), atoms
    ordered = ("a crystal is the ordered cell its sites generate: give that formula and put doping, substitution "
               "or partial occupancy in the description (the Construction agent makes it in a supercell)")
    try:
        stated = Composition(bulk.formula)
    except Exception:  # noqa: BLE001
        return [f"bulk formula {bulk.formula!r} is not a chemical formula; {ordered}"], atoms
    generated = Composition(atoms.get_chemical_formula())
    if generated.reduced_formula != stated.reduced_formula:
        doped = (not set(stated.elements) <= set(generated.elements)
                 or any(abs(n - round(n)) > 1e-6 for n in stated.values()))
        return [f"bulk sites generate {generated.reduced_formula}, not the stated formula {stated.reduced_formula}; "
                + (ordered if doped else "check the Wyckoff positions")], atoms
    return [], atoms


def _check_claims(p: StructurePlan, deps: PlanningDeps) -> list[str]:
    """The plan's claim checks are well formed, and every listed claim about its systems is checked
    or listed as unchecked with a reason."""
    from ase.data import chemical_symbols

    problems = []
    for c in p.checks:
        where = f"{p.plan_id}: check {c.id}"
        unknown = set(c.claim_ids) - set(deps.claims) if deps.claims else set()
        if unknown:
            problems.append(f"{where}: unknown claim IDs {sorted(unknown)}")
        if set(c.system_ids) - set(p.system_ids):
            problems.append(f"{where}: system_ids {sorted(set(c.system_ids) - set(p.system_ids))} are not the plan's")
        elements = [*getattr(c, "elements", ()), *(getattr(c, k, None) for k in ("center", "neighbor", "element", "donor")),
                    *getattr(c, "acceptors", ())]
        bad = sorted({e for e in elements if e is not None and e not in chemical_symbols[1:]})
        if bad:
            problems.append(f"{where}: unknown elements {bad}")
        if c.kind == "space_group":
            try:
                space_group_number(c.symbol)
            except ValueError as e:
                problems.append(f"{where}: {e}")
        if c.kind == "composition":
            try:
                Composition(c.formula)
            except Exception:  # noqa: BLE001
                problems.append(f"{where}: {c.formula!r} is not a chemical formula")
        if c.kind in ("bond_length", "bond_angle") and c.min > c.max:
            problems.append(f"{where}: min {c.min} is above max {c.max}")
        if c.kind == "lattice" and all(getattr(c, k) is None for k in ("a", "b", "c", "alpha", "beta", "gamma")):
            problems.append(f"{where}: give at least one lattice parameter")
    for u in p.unchecked:
        unknown = set(u.claim_ids) - set(deps.claims) if deps.claims else set()
        if unknown:
            problems.append(f"{p.plan_id}: unchecked claims {sorted(unknown)} are not listed claims")
    covered = {i for c in p.checks for i in c.claim_ids} | {i for u in p.unchecked for i in u.claim_ids}
    missing = [c for c in claims_about(list(deps.claims.values()), p.system_ids) if c.claim_id not in covered]
    if missing:
        problems.append(f"{p.plan_id}: {len(missing)} claims about its systems are neither tested by a check "
                        "(claim_ids) nor listed in unchecked with a reason: "
                        + "; ".join(f"{c.claim_id} ({c.text[:80]})" for c in missing))
    return problems


def _check_molecular_crystal(bulk: BulkSpec, framework: Composition) -> list[str]:
    """Framework atoms from the sites plus the molecules per cell make the stated formula (times
    the formula units)."""
    total = framework
    for m in bulk.molecules:
        mol = Chem.MolFromSmiles(m.smiles)
        if mol is None:
            return [f"molecule {m.name}: unparseable SMILES {m.smiles!r}"]
        counts: dict[str, int] = {}
        for a in Chem.AddHs(mol).GetAtoms():
            counts[a.GetSymbol()] = counts.get(a.GetSymbol(), 0) + 1
        total = total + Composition(counts) * m.per_cell
    try:
        stated = Composition(bulk.formula)
    except Exception:  # noqa: BLE001
        return [f"bulk formula {bulk.formula!r} is not a chemical formula"]
    reduced, n_fu = total.get_reduced_composition_and_factor()
    made = (f"the framework sites ({framework.alphabetical_formula}) and the molecules per cell "
            f"({', '.join(f'{m.per_cell} {m.name}' for m in bulk.molecules)}) make {total.alphabetical_formula}")
    if reduced.reduced_formula != stated.reduced_formula:
        z = bulk.formula_units or 1
        return [f"{made}, but {bulk.formula} x {z} formula units per cell is {(stated * z).alphabetical_formula}: "
                "check the framework sites and the molecules per cell"]
    units = total.num_atoms / stated.num_atoms
    if bulk.formula_units is not None and abs(units - bulk.formula_units) > 1e-6:
        return [f"{made}: {units:g} formula units of {bulk.formula}, not {bulk.formula_units}"]
    return []


def check_plan_set(plans: PlanSet, deps: PlanningDeps) -> list[str]:
    """Deterministic checks of a plan set against the evidence snapshot and user overrides."""
    problems = []
    if len(plans.plans) > deps.max_plans:
        problems.append(f"{len(plans.plans)} plans given; at most {deps.max_plans} allowed")
    ov = deps.overrides
    for p in plans.plans:
        if not NAME.match(p.plan_id):
            problems.append(f"plan ID {p.plan_id!r} names files and folders: use letters, digits, '_', '-', '.' "
                            "(e.g. 'P1')")
        if deps.system_ids and not p.system_ids:
            problems.append(f"{p.plan_id}: give the system_ids it models (valid: {sorted(deps.system_ids)})")
        cited = (set(p.evidence_ids) | {e for c in p.crystals for e in c.lattice.evidence_ids}
                 | {e for t in p.targets for e in t.evidence_ids} | {e for c in p.checks for e in c.evidence_ids})
        unknown = cited - deps.evidence_ids - set(deps.claims)
        if unknown:
            problems.append(f"{p.plan_id}: unknown evidence IDs {sorted(unknown)} (valid: {sorted(deps.evidence_ids)})")
        if not p.evidence_ids and not p.assumptions:
            problems.append(f"{p.plan_id}: cite supporting evidence IDs or state the assumptions it rests on")
        unknown_systems = set(p.system_ids) - deps.system_ids if deps.system_ids else set()
        if unknown_systems:
            problems.append(f"{p.plan_id}: unknown system IDs {sorted(unknown_systems)} (valid: {sorted(deps.system_ids)})")
        problems.extend(_check_claims(p, deps))
        for crystal in p.crystals:
            if crystal.lattice.source == "paper" and not crystal.lattice.evidence_ids:
                problems.append(f"{p.plan_id}: lattice of {crystal.name!r} has source 'paper' but cites no evidence")
            problems.extend(f"{p.plan_id}: crystal {crystal.name!r}: {msg}" for msg in _check_bulk(crystal))
        if (p.surface is not None or p.nanocrystal is not None) and not p.crystals:
            problems.append(f"{p.plan_id}: a surface or nanocrystal spec needs the crystal it is cut from in `crystals`; "
                            "(a hybrid crystal with its molecules); if it has no complete Wyckoff sites, drop the spec and describe the slab or "
                            "particle (orientation, thickness, vacuum, size) in the description")
        if ov.model_type and p.model_type.strip().lower() != ov.model_type.strip().lower():
            problems.append(f"{p.plan_id}: user override requires model_type {ov.model_type!r}")
        slab_like = any(w in p.model_type.lower() for w in ("slab", "surface"))
        if (ov.miller or ov.supercell) and slab_like and p.surface is None and p.crystals:
            problems.append(f"{p.plan_id}: the user pinned the slab's Miller index or supercell: give the `surface` "
                            "spec (cut from the first crystal) so the construction follows the pins")
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
                 overrides: OverridesConfig, max_retry: int = 3, simulation: str = "MLIP"):
        super().__init__(model, effort,
                         planner_system_prompt(max_plans, max_atoms, min_vacuum, overrides.describe(), simulation),
                         max_retry=max_retry, output_retries=max_retry + 2)
        self.max_plans = max_plans
        self.max_atoms = max_atoms
        self.overrides = overrides

    def _register(self, agent) -> None:
        @agent.tool_plain
        def try_crystal(crystal: BulkSpec) -> str:
            """Build one crystal spec as the plan check will: the problems to fix, or the generated
            framework's composition, symmetry, coordination numbers with bond lengths and bridging
            angles, to compare with what the paper reports before you put it in a plan."""
            return describe_crystal(crystal)

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
        claims = structural_claims(summary, evidence)
        deps = PlanningDeps(evidence_ids={e.evidence_id for e in summary.evidence} | evidence.ids,
                            max_plans=self.max_plans, overrides=self.overrides, max_atoms=self.max_atoms,
                            system_ids={s.system_id for s in summary.systems},
                            claims={c.claim_id: c for c in claims})
        return await self.arun(planner_prompt(summary, evidence, history, feedback, claims), deps=deps)


def planner_prompt(summary: PaperGraph, evidence: ExperimentalEvidence, history: list[PlanSet],
                   feedback: str, claims: list[Claim] | None = None) -> str:
    claims = structural_claims(summary, evidence) if claims is None else claims
    parts = ["## Paper summary (Agent 1: metadata, material systems, text and figure evidence)",
             summary.model_dump_json(exclude_none=True),
             "## Characterization reports (Agent 2: observations, findings and structural constraints per technique)",
             evidence.model_dump_json(exclude_none=True)]
    if claims:
        parts += ["## Structural claims (every plan tests each claim about its systems with a check, "
                  "or lists it in unchecked with the reason)",
                  "\n".join(f"- {c.claim_id} [{', '.join(c.system_ids) or 'all systems'}] {c.text}" for c in claims)]
    if history:
        parts += ["## Previous plan sets (oldest first)"] + [h.model_dump_json(exclude_none=True) for h in history]
    if feedback:
        parts += ["## Review of the structures built from the previous plans", feedback,
                  "Revise the plans to address this review."]
    else:
        parts.append("Write the plan set.")
    return "\n\n".join(parts)


def planner_system_prompt(max_plans: int, max_atoms: int, min_vacuum: float, overrides: str,
                          simulation: str = "MLIP") -> str:
    return f"""\
You are the Structural Planning agent of Paper2Structure. From the paper summary and the
characterization reports you decide which atomic models best represent the system the paper
studies, and you write one construction plan per model. A separate Construction agent builds each
plan with pymatgen, ASE and RDKit; {simulation_brief(simulation)}.

Your inputs
- The paper summary (Agent 1): metadata, the material systems studied (IDs S1, S2, ...; each with
  synthesis and the structural facts the paper states: crystal system, space group, lattice
  parameters, site occupancy, phase purity) and evidence items (IDs E001, ...) extracted from the
  text and figures, tagged by technique.
- The characterization reports (Agent 2), one per technique: observations read from the figures
  and text (IDs O1, O2, ... within each report), findings, and structural constraints with their
  confidence.
- Cite evidence as summarizer evidence IDs ("E004") or as characterization observation IDs
  qualified by their technique ("XRD:O3"), or the structural claims below by their IDs; no other
  IDs exist.
- The structural claims: the facts recorded for each system and the reports' structural
  constraints, each with an ID ("S2:space_group", "XRD:C3"). The structures must agree with them,
  so every plan turns each claim about its systems into checks.
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
  space group, its asymmetric unit as Wyckoff representatives with fractional coordinates, lattice
  parameters and formula units per conventional cell. Write the space group so that it names the
  setting the coordinates are in, as the paper reports them: the Hermann-Mauguin symbol ("Fm-3m",
  "P4_2/mnm", "P6_3mc", "Pnma", or a non-standard setting such as "Pbnm", "P2_1/n", "C2cb"), with
  ":2" for origin choice 2 and ":R" for rhombohedral axes ("Fd-3m:2", "R-3m:R"); without a suffix a
  symbol means origin choice 1 and hexagonal axes. Never transform a paper's coordinates by hand:
  name its setting instead. Lattice parameters are those of that setting. Use lattice parameters
  reported in the evidence (source "paper", cite the evidence ID); otherwise literature values (source "literature", listed as an assumption). Each crystal is
  built from its sites exactly as written and must be complete: every atom of its formula, bonded
  neighbours, no overlaps. A hybrid organic-inorganic or molecular crystal (organic cations
  between or inside inorganic layers, a framework with guest molecules) is a crystal too: its
  sites are the framework atoms and `molecules` lists the molecules (SMILES with formal charges,
  number per conventional cell); the formula is the whole crystal's, and the Construction agent
  fills the framework's cavities with the molecules, keeping the space group. The framework
  coordinates fix its distortions (polyhedral tilts, bond angles) and its symmetry, so take them
  from the paper; when the
  paper does not report them, from the literature for the same compound, or idealized positions
  whose free parameters reproduce the reported bond lengths and angles (state which as an
  assumption). Test every crystal with try_crystal and adjust the sites until it builds with the
  reported composition, symmetry, coordination, bond lengths and angles. Never list a partial sublattice or an isolated layer as a crystal, and never leave
  a crystalline component out of `crystals`.
- surface (optional helper, for a crystal listed in `crystals`): Miller index, termination
  described by its exposed species, minimum thickness (A), in-plane supercell, vacuum (at least
  {min_vacuum} A) and a fixed bottom region (A).
- nanocrystal (optional helper, for a crystal listed in `crystals`): shape and size (closed-shell
  shapes for elemental metals; spheres or faceted polyhedra cut from the crystal for compounds),
  facets, and oxidation states when the particle must be charge-neutral. For multi-component
  particles give the overall size here and the regions in the description.
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
- checks: the paper's claims as tests the validation gate runs on every structure built from the
  plan; a failing check sends the structure back to construction. Each check cites the listed
  claims it tests (claim_ids) and the evidence IDs, takes its values from the claim (never
  invented) and a tolerance from the reported uncertainty or the precision of the value. Kinds:
  lattice (a, b, c, alpha, beta, gamma of a bulk model, relative tolerance), space_group (the
  symbol in the paper's setting; scope "framework" when the experiment averages over disordered
  molecules), composition (a bulk model's formula), bond_length (an element pair with the reported
  min and max), bond_angle (A, vertex, C with the reported range, e.g. M-O-M bridge angles),
  coordination (center, neighbor, number), hydrogen_bonds (donor, acceptors and hydrogen bonds per
  donor, when the paper reports hydrogen bonding, e.g. N-H...O or O-H...Cl) and layers (element,
  number of planes per framework block, stacking axis: the slab thickness of a layered
  material). Also turn the quantitative structural facts in the evidence (tables of bond lengths
  and angles, layer spacings) into checks. When a plan models several systems, scope each check
  to its systems with system_ids. Severity "warn" only for approximate or low-confidence claims.
- unchecked: every listed structural claim about the plan's systems that no check can test (phase
  purity, morphology, disorder an ordered model cannot show, a technique's statement that it does
  not constrain the structure), with the reason. Every claim about the plan's systems appears in a
  check's claim_ids or in unchecked.
- targets: the other quantitative features the evidence fixes (core diameter, shell thickness,
  coverage, exposed sites, dopant fraction, ...), each with its value, tolerance and evidence IDs.
  The Construction agent measures them on the built structure.

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

On revision you receive your previous plans, how the structures built from each plan fared
(validation, failing claim checks, the reviewer's verdicts) and the reviewer's feedback. Change
what the review identifies; do not resubmit a plan that was judged not to resemble the paper. The
user overrides stay in force on revision: where the feedback conflicts with them, keep the
override and record the conflict as an assumption."""
