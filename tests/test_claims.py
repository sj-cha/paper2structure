"""The paper's claims as checks: the planner must address every claim about its systems, and the
validation gate tests the built structure against each check (tools.evidence_checks)."""
import copy

import numpy as np
import pytest
from ase import Atoms

from helpers import PT_PLAN
from paper2structure.agents.planner import PlanningDeps, PlanSet, check_plan_set
from paper2structure.agents.planner.claims import Claim
from paper2structure.config import OverridesConfig
from paper2structure.tools import Adsorbate, BindingMotif, add_molecule, build_bulk, build_slab
from paper2structure.tools._symmetry import space_group_number, symmetry_operations
from paper2structure.tools.evidence_checks import run_claim_checks
from paper2structure.tools.schemas import (BondAngleCheck, BondLengthCheck, CompositionCheck, CoordinationCheck,
                                           HydrogenBondCheck, LatticeCheck, LayerCheck, SpaceGroupCheck)


@pytest.fixture(scope="module")
def cs_bulk():
    return build_bulk("Pm-3m", ["Cs", "Pb", "Br"], [[0.5, 0.5, 0.5], [0, 0, 0], [0.5, 0, 0]], a=5.87, formula_units=1)


def _status(atoms, *checks, systems=None):
    return {r.name.removeprefix("claim "): r.status for r in run_claim_checks(atoms, checks, systems)}


def test_space_group_symbols_in_any_setting():
    assert [space_group_number(s) for s in ("Cc2m", "C2cb", "Aba2", "Pbnm", "P 1 1 2_1/b", "I4_1/amd")] == \
        [40, 41, 41, 62, 14, 141]
    assert len(symmetry_operations("C2cb")[0]) == 8  # centring translations included
    with pytest.raises(ValueError, match="unknown space group"):
        space_group_number("Xyz")


def test_claims_on_a_cubic_perovskite(cs_bulk):
    supercell = cs_bulk.repeat((2, 2, 2))
    right = [LatticeCheck(id="L", claim="a = 5.87", a=5.87, b=5.87, c=5.87),
             SpaceGroupCheck(id="SG", claim="Pm-3m", symbol="Pm-3m"),
             CompositionCheck(id="X", claim="CsPbBr3", formula="CsPbBr3"),
             BondLengthCheck(id="B", claim="Pb-Br 2.935", elements=("Pb", "Br"), min=2.93, max=2.94),
             BondAngleCheck(id="A", claim="linear Pb-Br-Pb", elements=("Pb", "Br", "Pb"), min=178, max=180),
             CoordinationCheck(id="CN", claim="PbBr6", center="Pb", neighbor="Br", number=6)]
    assert set(_status(supercell, *right).values()) == {"pass"}  # a 2x2x2 supercell matches the cell
    wrong = [SpaceGroupCheck(id="SG", claim="tilted", symbol="I4/mcm"),
             BondAngleCheck(id="A", claim="tilted", elements=("Pb", "Br", "Pb"), min=150, max=160),
             CompositionCheck(id="X", claim="Cs2PbBr4", formula="Cs2PbBr4"),
             LatticeCheck(id="L", claim="a = 6.2", a=6.2, severity="warn")]
    assert _status(supercell, *wrong) == {"SG": "fail", "A": "fail", "X": "fail", "L": "warn"}
    detail = run_claim_checks(supercell, [wrong[0].model_copy(update={"evidence_ids": ["XRD:O2"]})])[0].detail
    assert "built Pm-3m (#221)" in detail and "paper I4/mcm (#140)" in detail and "XRD:O2" in detail


def test_layer_counts_and_bulk_only_checks(cs_bulk):
    slab = build_slab(cs_bulk, (0, 0, 1), termination=0, min_thickness=12.0, vacuum=15.0, supercell=(1, 1))
    pb_planes = len({round(z, 1) for z in slab.positions[slab.symbols == "Pb", 2]})
    got = _status(slab, LayerCheck(id="n", claim="slab", element="Pb", count=pb_planes, axis="c"),
                  LatticeCheck(id="L", claim="a", a=5.87), SpaceGroupCheck(id="SG", claim="Pm-3m", symbol="Pm-3m"))
    assert got == {"n": "pass", "L": "skip", "SG": "skip"}
    # the 3D network is bonded to its own image along c: not a layer
    result = run_claim_checks(cs_bulk, [LayerCheck(id="n", claim="n=1", element="Pb", count=1, axis="c")])[0]
    assert result.status == "fail" and "connected through the cell" in result.detail


def test_checks_scoped_to_other_systems_are_not_applied(cs_bulk):
    checks = [SpaceGroupCheck(id="S2", claim="S2 phase", symbol="Pm-3m", system_ids=["S2"]),
              SpaceGroupCheck(id="S3", claim="S3 phase", symbol="P4/mbm", system_ids=["S3"])]
    assert _status(cs_bulk, *checks, systems=["S2"]) == {"S2": "pass"}
    assert set(_status(cs_bulk, *checks)) == {"S2", "S3"}  # a structure that declares no system gets all


def test_hydrogen_bonds_need_each_nh_to_point_at_an_acceptor():
    ma = Adsorbate.from_smiles("C[NH3+]", BindingMotif(["N"], anchor="[NH3+]"), name="MA", seed=0).atoms
    sym = ma.get_chemical_symbols()
    n = sym.index("N")
    hs = [k for k in range(len(ma)) if sym[k] == "H" and np.linalg.norm(ma.positions[k] - ma.positions[n]) < 1.2]
    iodides = [ma.positions[h] + 2.7 * (ma.positions[h] - ma.positions[n]) / np.linalg.norm(ma.positions[h] - ma.positions[n])
               for h in hs]
    host = Atoms("I3", positions=iodides, cell=[20, 20, 20], pbc=True)
    shift = np.array([10.0, 10.0, 10.0]) - ma.positions[n]
    host.translate(shift)
    placed = Atoms(sym, positions=ma.positions + shift)
    check = HydrogenBondCheck(id="H", claim="three N-H...I per NH3+", acceptors=["I"], per_donor=3)
    model = add_molecule(host, placed, "C[NH3+]", "MA", role="guest")
    result = run_claim_checks(model, [check])[0]
    assert result.status == "pass" and "3 H-bonds: 1" in result.detail
    # the same molecule turned 60 degrees about its C-N axis: no N-H points at an iodide
    c = sym.index("C")
    axis = placed.positions[n] - placed.positions[c]
    turned = placed.copy()
    turned.rotate(60, axis, center=placed.positions[n])
    result = run_claim_checks(add_molecule(host, turned, "C[NH3+]", "MA", role="guest"), [check])[0]
    assert result.status == "fail" and "0 H-bonds: 1" in result.detail


def _deps(**claims) -> PlanningDeps:
    return PlanningDeps({"E1", "E2", "E3"}, 3, OverridesConfig(),
                        claims={k: Claim(k, s, f"claim {k}", "XRD") for k, s in claims.items()})


def test_planner_must_check_or_explain_every_claim_about_its_systems():
    claims = {"S1:space_group": ("S1",), "S1:lattice": ("S1",), "S1:other1": ("S1",), "XRD:C1": ("S1",),
              "S1:occupancy1": ("S1",), "S1:occupancy2": ("S1",), "S1:other2": ("S1",), "FTIR:C1": ("S1",),
              "FTIR:C2": ("S1",), "FTIR:C3": ("S1",), "FTIR:C4": ("S1",), "LEED:C1": ("S1",), "LEED:C2": ("S1",),
              "XPS:C1": ("S1",), "S2:lattice": ("S2",)}  # S2 is not modeled by the plan: no need to address it
    assert check_plan_set(PlanSet.model_validate(PT_PLAN), _deps(**claims)) == []
    missing = copy.deepcopy(PT_PLAN)
    missing["plans"][0]["unchecked"] = []
    problems = check_plan_set(PlanSet.model_validate(missing), _deps(**claims))
    assert any("10 claims about its systems" in p and "LEED:C2" in p for p in problems), problems
    bad = copy.deepcopy(PT_PLAN)
    bad["plans"][0]["checks"] += [
        {"kind": "space_group", "id": "C2", "claim": "x", "symbol": "Q4/xyz", "claim_ids": ["XRD:C9"]},
        {"kind": "bond_angle", "id": "C3", "claim": "x", "elements": ["Pb", "Xx", "Pb"], "min": 170, "max": 160,
         "system_ids": ["S7"]}]
    problems = check_plan_set(PlanSet.model_validate(bad), _deps(**claims))
    for text in ("unknown claim IDs ['XRD:C9']", "unknown space group symbol 'Q4/xyz'", "unknown elements ['Xx']",
                 "min 170.0 is above max 160.0", "system_ids ['S7'] are not the plan's"):
        assert any(text in p for p in problems), (text, problems)


BA2PBI4 = dict(space_group="Pbca", species=["Pb", "I", "I"], coords=[[0, 0, 0], [-0.03, 0.02, 0.115], [0.2, 0.3, 0.02]],
               a=8.863, b=8.682, c=27.57)


def test_fill_cavities_keeps_the_space_group_and_hydrogen_bonds_the_heads():
    from paper2structure.tools import fill_cavities, get_molecules

    framework = build_bulk(BA2PBI4["space_group"], BA2PBI4["species"], BA2PBI4["coords"],
                           a=BA2PBI4["a"], b=BA2PBI4["b"], c=BA2PBI4["c"])
    crystal = fill_cavities(framework, "CCCC[NH3+]", 8, name="BA", seed=0)
    assert len(get_molecules(crystal)) == 8 and {m["role"] for m in get_molecules(crystal)} == {"guest"}
    checks = [SpaceGroupCheck(id="SG", claim="Pbca", symbol="Pbca"),
              CompositionCheck(id="X", claim="BA2PbI4", formula="(C4H9NH3)2PbI4"),
              HydrogenBondCheck(id="H", claim="3 N-H...I per head", acceptors=["I"], per_donor=3),
              LayerCheck(id="n", claim="n = 1", element="Pb", count=1, axis="c")]
    assert set(_status(crystal, *checks).values()) == {"pass"}
    from paper2structure.tools.geometry_checks import validate_structure
    assert validate_structure(crystal, "BA2PbI4", max_atoms=5000, min_vacuum=10.0).passed
    with pytest.raises(ValueError, match="not invariant"):  # a supercell is not the conventional cell
        fill_cavities(framework.repeat((2, 1, 1)), "CCCC[NH3+]", 16, name="BA", seed=0)


def test_build_bulk_takes_settings_pymatgen_does_not_know():
    atoms = build_bulk("C2cb", ["Pb", "I"], [[0.01, 0.1, 0.02], [0.27, 0.11, 0.24]], a=8.93, b=51.96, c=8.88)
    assert atoms.info["p2s_space_group"] == "C2cb" and "p2s_warning" not in atoms.info
    assert _status(atoms, SpaceGroupCheck(id="SG", claim="C2cb", symbol="C2cb")) == {"SG": "pass"}


def test_planner_checks_the_formula_of_a_crystal_with_molecules():
    plan = copy.deepcopy(PT_PLAN)
    hybrid = {"name": "bulk", "formula": "(C4H9NH3)2PbI4", "space_group": "Pbca", "formula_units": 4,
              "sites": [{"species": s, "coords": c} for s, c in zip(BA2PBI4["species"], BA2PBI4["coords"])],
              "molecules": [{"name": "BA", "smiles": "CCCC[NH3+]", "per_cell": 8}],
              "lattice": {"a": 8.863, "b": 8.682, "c": 27.57, "source": "literature"}}
    plan["plans"][0].update({"model_type": "bulk", "surface": None, "adsorbates": [], "crystals": [hybrid],
                             "checks": [], "unchecked": []})
    deps = PlanningDeps({"E1", "E2", "E3"}, 3, OverridesConfig())
    assert check_plan_set(PlanSet.model_validate(plan), deps) == []
    plan["plans"][0]["crystals"][0]["molecules"][0]["per_cell"] = 6
    problems = check_plan_set(PlanSet.model_validate(plan), deps)
    assert any("6 BA" in p and "not the stated formula" in p for p in problems), problems


def test_fill_cavities_is_not_specific_to_perovskites():
    from paper2structure.tools import fill_cavities

    # ammonium chloride (CsCl type): NH4+ at the cube centre, each N-H pointing at a corner Cl at ~2.3 A,
    # a hydrogen bond shorter than the plain H...Cl clash limit
    chloride = build_bulk("Pm-3m", ["Cl"], [[0, 0, 0]], a=3.87, check_bonded=False)
    salt = fill_cavities(chloride, "[NH4+]", 1, name="NH4", seed=0)
    four = HydrogenBondCheck(id="H", claim="4 N-H...Cl", acceptors=["Cl"], per_donor=4)
    assert _status(salt, four, CompositionCheck(id="X", claim="NH4Cl", formula="NH4Cl")) == {"H": "pass", "X": "pass"}
    # a neutral molecule without donors goes where it touches the framework without clashing
    host = build_bulk("Pm-3m", ["Cl"], [[0, 0, 0]], a=7.0, check_bonded=False)
    with_co2 = fill_cavities(host, "O=C=O", 1, name="CO2", seed=0)
    from paper2structure.tools.geometry_checks import validate_structure
    report = {c.name: c.status for c in validate_structure(with_co2, "CO2", max_atoms=5000, min_vacuum=10.0).checks}
    assert report["overlaps"] == report["molecule_integrity"] == report["molecule_geometry"] == "pass"
