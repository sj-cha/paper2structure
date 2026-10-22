"""Structure-building tools and deterministic validation (no LLM)."""
import importlib.util
import json

import numpy as np
import pytest
from ase import Atoms
from ase.geometry import get_distances
from ase.io import read
from pymatgen.core import Composition

from paper2structure.tools import (Adsorbate, BindingMotif, add_solvent, build_bulk, build_cluster, build_slab,
                                   carve_nanocrystal, list_terminations, place_adsorbates, surface_atoms, surface_sites,
                                   unique_sites)
from paper2structure.tools._geometry import get_molecules, neighbor_pairs, rcov
from paper2structure.tools.geometry_checks import validate_structure
from paper2structure.tools.structure_export import describe, export_structure, grouped_order


@pytest.fixture(scope="module")
def pt_bulk():
    return build_bulk("Fm-3m", ["Pt"], [[0, 0, 0]], a=3.924, formula_units=4)


@pytest.fixture(scope="module")
def cs_bulk():
    return build_bulk("Pm-3m", ["Cs", "Pb", "Br"], [[0.5, 0.5, 0.5], [0, 0, 0], [0.5, 0, 0]], a=5.87, formula_units=1)


@pytest.fixture(scope="module")
def anatase():
    return build_bulk("I4_1/amd", ["Ti", "O"], [[0, 0, 0], [0, 0, 0.2081]], a=3.785, c=9.514, formula_units=4)


@pytest.fixture(scope="module")
def co():
    return Adsorbate.from_smiles("[C-]#[O+]", BindingMotif(["C"], anchor="[C-]#[O+]"), name="CO")


def _min_dist(atoms, i, others):
    _, d = get_distances(atoms.positions[[i]], atoms.positions[others], cell=atoms.cell, pbc=atoms.pbc)
    return d[0]


def _validate(atoms, **kw):
    return validate_structure(atoms, "t", max_atoms=5000, min_vacuum=kw.pop("min_vacuum", 12.0), **kw)


def test_build_bulk_symmetry_and_origin_hint(anatase):
    assert len(anatase) == 12 and anatase.get_chemical_formula() == "O8Ti4"
    with pytest.raises(ValueError, match="origin choice"):
        build_bulk("I4_1/amd", ["Ti", "O"], [[0, 0.75, 0.125], [0, 0.25, 0.0833]], a=3.785, c=9.514)
    with pytest.raises(ValueError, match="formula units"):
        build_bulk("Fm-3m", ["Pt"], [[0, 0, 0]], a=3.924, formula_units=1)
    with pytest.raises(ValueError, match="must be given"):
        build_bulk("P4_2/mnm", ["Ti", "O"], [[0, 0, 0], [0.305, 0.305, 0]], a=4.594)


def test_build_bulk_rejects_swapped_perovskite_sites():
    # the live planner once put Br around Cs (Cs at 1a, Br at 3d): right formula, Z and no overlap,
    # but Pb left without any Br
    with pytest.raises(ValueError, match=r"no bonded neighbour for Pb \(nearest Br at 4.15 A\)"):
        build_bulk("Pm-3m", ["Cs", "Pb", "Br"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0]], a=5.87)
    ok = build_bulk("Pm-3m", ["Cs", "Pb", "Br"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]], a=5.87)
    assert "Pb CN 6 (Br 2.94 A)" in describe(ok) and "Cs CN 12 (Br 4.15 A)" in describe(ok)


def test_build_bulk_honors_a_non_standard_setting():
    # GdFeO3-type LaCrO3 as papers often write it (Pbnm, c the long axis) and as its Pnma transform
    sites = (["La", "Cr", "O", "O"], [[0.996, 0.018, 0.25], [0.5, 0, 0], [0.07, 0.49, 0.25], [0.72, 0.28, 0.04]])
    pbnm = build_bulk("Pbnm", *sites, a=5.48, b=5.52, c=7.76, formula_units=4)
    pnma = build_bulk("Pnma", sites[0], [[y, z, x] for x, y, z in sites[1]], a=5.52, b=7.76, c=5.48, formula_units=4)
    assert np.allclose(pbnm.cell.lengths(), [5.48, 5.52, 7.76])
    first = lambda at: np.sort(at.get_all_distances(mic=True)[np.triu_indices(len(at), 1)])[:12]
    assert np.allclose(first(pbnm), first(pnma), atol=2e-3)


def test_empty_molecule_records_survive_extxyz(tmp_path, pt_bulk):
    # ASE reads an empty JSON list back as an empty array; that must still mean "no molecules"
    atoms = pt_bulk.copy()
    atoms.info["p2s_molecules"] = []
    atoms.write(tmp_path / "x.extxyz")
    back = read(tmp_path / "x.extxyz")
    assert isinstance(back.info["p2s_molecules"], np.ndarray) and get_molecules(back) == []
    assert describe(back)


def test_anion_contacts_are_flagged(cs_bulk):
    k = next(t.index for t in list_terminations(cs_bulk, (0, 0, 1), min_thickness=10) if t.top == "PbBr2")
    slab = build_slab(cs_bulk, (0, 0, 1), k, min_thickness=10, vacuum=15, supercell=(2, 2))
    fo = Adsorbate.from_smiles("[O-]C=O", BindingMotif(["O", "O"], anchor="C(=O)[O-]"), name="formate")
    crowded = place_adsorbates(slab, fo, surface_sites(slab, kinds=("top",), elements=["Br"])[:1], bond_length=1.9)
    check = next(c for c in _validate(crowded).checks if c.name == "anion_contacts")
    assert check.status == "warn" and "-Br" in check.detail


def test_cspbbr3_terminations_both_faces(cs_bulk):
    terms = list_terminations(cs_bulk, (0, 0, 1), min_thickness=10)
    assert {t.top for t in terms} == {"PbBr2", "CsBr"}
    for t in terms:
        slab = build_slab(cs_bulk, (0, 0, 1), t.index, min_thickness=10, vacuum=15, supercell=(2, 2))
        z = slab.positions[:, 2]
        top = {slab[i].symbol for i in range(len(slab)) if z[i] > z.max() - 1.0}
        assert top == {str(e) for e in Composition(t.top).elements}
        assert np.linalg.det(slab.cell.array) > 0  # flipped slabs stay right-handed
        assert slab.cell[2, 2] - (z.max() - z.min()) == pytest.approx(15.0, abs=1e-6)


def test_pt111_sites_and_co_geometry(pt_bulk, co):
    slab = build_slab(pt_bulk, (1, 1, 1), 0, min_thickness=6, vacuum=15, supercell=(3, 3), fix_bottom=2.0)
    labels = {s.kind + s.subtype for s in unique_sites(surface_sites(slab, kinds=("top", "bridge", "hollow")))}
    assert labels == {"top", "bridge", "hollowfcc", "hollowhcp"}
    pt = [i for i in range(len(slab))]
    top = place_adsorbates(slab, co, surface_sites(slab, kinds=("top",)), count=1, bond_length=1.85)
    c = len(top) - 2
    assert _min_dist(top, c, pt).min() == pytest.approx(1.85, abs=1e-6)
    assert top[c].symbol == "C" and top.positions[c + 1, 2] > top.positions[c, 2]  # carbon-down
    fcc = [s for s in surface_sites(slab, kinds=("hollow",)) if s.subtype == "fcc"]
    hol = place_adsorbates(slab, co, fcc, count=1, bond_length=2.05)
    assert np.sort(_min_dist(hol, len(hol) - 2, pt))[:3] == pytest.approx([2.05] * 3, abs=1e-6)
    assert _validate(top).passed and _validate(hol).passed


def test_phosphonate_replaces_surface_br(cs_bulk):
    slab = build_slab(cs_bulk, (0, 0, 1), 0, min_thickness=10, vacuum=15, supercell=(2, 2))
    br = surface_sites(slab, kinds=("top",), elements=["Br"])
    hpa = Adsorbate.from_smiles("CCCCCCP(=O)([O-])O", BindingMotif(["O", "O"], anchor="P(=O)([O-])O"), name="HPA")
    out = place_adsorbates(slab, hpa, br, count=2, mode="replace")
    assert len(out) == len(slab) - 2 + 2 * len(hpa.atoms)
    mols = get_molecules(out)
    assert [m["mode"] for m in mols] == ["replace", "replace"] and all(m["charge"] == -1 for m in mols)
    report = _validate(out)
    assert report.passed, report.summary()
    assert report.metrics["min_distance_ratio"] > 0.6


def test_formate_bridges_two_ti5c(anatase):
    terms = list_terminations(anatase, (1, 0, 1), min_thickness=10)
    k = next(t.index for t in terms if t.symmetric and t.stoichiometric and t.top == "TiO2"
             and {s.coordination for s in surface_sites(build_slab(anatase, (1, 0, 1), t.index, min_thickness=10,
                                                                   vacuum=15), elements=["Ti"])} == {(5,)})
    slab = build_slab(anatase, (1, 0, 1), k, min_thickness=10, vacuum=15, supercell=(3, 2))
    bridges = [s for s in surface_sites(slab, kinds=("bridge",), elements=["Ti"], max_pair_distance=4.0)
               if s.coordination == (5, 5)]
    fo = Adsorbate.from_smiles("[O-]C=O", BindingMotif(["O", "O"], anchor="C(=O)[O-]"), name="formate")
    out = place_adsorbates(slab, fo, bridges[:1], count=1)
    o = out.positions[[len(out) - len(fo.atoms) + b for b in fo.binding_atoms]]
    ti = bridges[0].position + bridges[0].offsets  # the two Ti5c of the site
    d = np.linalg.norm(o[:, None, :] - ti[None, :, :], axis=2)
    assert sorted(np.argmin(d, axis=1)) == [0, 1]  # each oxygen binds its own Ti5c
    assert d.min(axis=1) == pytest.approx([1.6 + 0.66] * 2, abs=0.05)  # covalent-sum default distance
    assert _validate(out).passed


def test_clusters_and_carved_nanocrystal(cs_bulk):
    pt55 = build_cluster("Pt", "cuboctahedron", 3, 3.924)
    assert len(pt55) == 55 and len(surface_atoms(pt55)) == 42
    assert len(build_cluster("Pt", "icosahedron", 3, 3.924)) == 55
    nc = carve_nanocrystal(cs_bulk, 20.0, facets={(1, 0, 0): 1.0}, oxidation_states={"Cs": 1, "Pb": 2, "Br": -1},
                           center_element="Cs")
    assert nc.info["p2s_charge"] == 0
    assert _validate(nc).passed


def test_solvent_layer(pt_bulk, co):
    slab = build_slab(pt_bulk, (1, 1, 1), 0, min_thickness=6, vacuum=15, supercell=(5, 5))
    wet = add_solvent(place_adsorbates(slab, co, surface_sites(slab), count=1), "CC#N", density=0.786, thickness=10.0)
    mols = get_molecules(wet)
    solvent = [m for m in mols if m["role"] == "solvent"]
    assert len(solvent) >= 15  # ~0.786 g/cm^3 over 5x5 Pt(111) x 10 A
    ids = wet.arrays["mol_id"]
    is_solvent = np.isin(ids, [m["mol_id"] for m in solvent])
    i, j, D = neighbor_pairs(wet, 1.99)
    clash = (ids[i] != ids[j]) & (is_solvent[i] | is_solvent[j])  # packing tolerance is 2.0 A
    assert not clash.any()
    assert next(c for c in _validate(wet).checks if c.name == "molecule_integrity").status == "pass"


def test_export_poscar_default_and_roundtrip(tmp_path, pt_bulk, co):
    slab = build_slab(pt_bulk, (1, 1, 1), 0, min_thickness=6, vacuum=15, supercell=(2, 2), fix_bottom=2.0)
    model = place_adsorbates(slab, co, surface_sites(slab), count=1, bond_length=1.85)
    files = export_structure(model, tmp_path, "s1")
    assert set(files) == {"poscar", "metadata"}
    poscar = (tmp_path / "s1.vasp").read_text().splitlines()
    assert poscar[5].split() == ["Pt", "C", "O"] and poscar[7].strip() == "Selective dynamics"
    back = read(tmp_path / "s1.vasp")
    assert back.get_chemical_symbols() == model[grouped_order(model)].get_chemical_symbols()
    meta = json.loads((tmp_path / "s1.json").read_text())
    assert meta["mol_id"].count(0) == 2 and meta["fixed"]
    report = validate_structure(model, "s1", max_atoms=5000, min_vacuum=12.0, files=files)
    assert next(c for c in report.checks if c.name == "export_roundtrip").status == "pass"


def test_validation_flags_overlaps_and_broken_molecules(pt_bulk, co):
    slab = build_slab(pt_bulk, (1, 1, 1), 0, min_thickness=6, vacuum=15, supercell=(2, 2))
    bad = place_adsorbates(slab, co, surface_sites(slab), count=1, bond_length=0.4)
    assert next(c for c in _validate(bad).checks if c.name == "overlaps").status == "fail"
    broken = place_adsorbates(slab, co, surface_sites(slab), count=1, bond_length=1.85)
    broken.positions[-1, 2] += 1.5  # pull O off its C
    report = _validate(broken)
    assert not report.passed and next(c for c in report.checks if c.name == "molecule_integrity").status == "fail"
    thin = build_slab(pt_bulk, (1, 1, 1), 0, min_thickness=6, vacuum=5, supercell=(2, 2))
    assert next(c for c in _validate(thin).checks if c.name == "vacuum").status == "fail"


def test_top_sites_on_thick_slab():
    # regression: the periodic image of the slab bottom across the vacuum once "shadowed" every
    # top atom of thick slabs
    cs = build_bulk("Pm-3m", ["Cs", "Pb", "Br"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]], a=5.87, formula_units=1)
    k = next(t.index for t in list_terminations(cs, (1, 0, 0), min_thickness=25.0) if t.top == "PbBr2")
    slab = build_slab(cs, (1, 0, 0), k, min_thickness=25.0, vacuum=18.0, supercell=(2, 2), fix_bottom=8.0)
    sites = surface_sites(slab, kinds=("top",), elements=["Br"])
    z = slab.positions[:, 2]
    assert len(sites) == 8 and all(s.position[2] > z.max() - 0.5 for s in sites)


def test_replace_mode_honours_bond_length(cs_bulk):
    # CsBr-terminated (001): each surface Br caps a Pb directly below; a monodentate phosphonate O
    # taking that site must end at the requested Pb-O distance, not at the old Pb-Br distance
    k = next(t.index for t in list_terminations(cs_bulk, (0, 0, 1), min_thickness=10) if t.top == "CsBr")
    slab = build_slab(cs_bulk, (0, 0, 1), k, min_thickness=10, vacuum=15, supercell=(2, 2))
    br = surface_sites(slab, kinds=("top",), elements=["Br"])
    mono = Adsorbate.from_smiles("CCP(=O)(O)[O-]", BindingMotif(["O"], anchor="P[O-]"), name="EPA-")
    out = place_adsorbates(slab, mono, br, count=1, mode="replace", bond_length=2.3)
    o = len(out) - len(mono.atoms) + mono.binding_atoms[0]
    pb = [i for i in range(len(out)) if out[i].symbol == "Pb"]
    assert _min_dist(out, o, pb).min() == pytest.approx(2.3, abs=0.02)


def test_monodentate_replacing_bridging_bromide_binds_one_pb(cs_bulk):
    # PbBr2-terminated (001): the in-plane Br bridges two Pb at 2.94 A; a monodentate O taking its
    # place must form a terminal Pb-O bond at the requested length (the live run left it at 2.94 A)
    k = next(t.index for t in list_terminations(cs_bulk, (0, 0, 1), min_thickness=10) if t.top == "PbBr2")
    slab = build_slab(cs_bulk, (0, 0, 1), k, min_thickness=10, vacuum=15, supercell=(2, 2))
    mono = Adsorbate.from_smiles("CCP(=O)(O)[O-]", BindingMotif(["O"], anchor="P[O-]"), name="EPA-")
    out = place_adsorbates(slab, mono, surface_sites(slab, kinds=("top",), elements=["Br"]), count=1,
                        mode="replace", bond_length=2.3)
    o = len(out) - len(mono.atoms) + mono.binding_atoms[0]
    d = np.sort(_min_dist(out, o, [i for i in range(len(out)) if out[i].symbol == "Pb"]))
    assert d[0] == pytest.approx(2.3, abs=0.02) and d[1] > 2.9
    assert _validate(out).passed


def test_dianion_replaces_two_surface_bromides(cs_bulk):
    k = next(t.index for t in list_terminations(cs_bulk, (0, 0, 1), min_thickness=10) if t.top == "PbBr2")
    slab = build_slab(cs_bulk, (0, 0, 1), k, min_thickness=10, vacuum=15, supercell=(2, 2))
    pairs = surface_sites(slab, kinds=("bridge",), elements=["Br"], max_pair_distance=4.3)
    assert pairs and all(s.symbols == ("Br", "Br") for s in pairs)
    di = Adsorbate.from_smiles("CCP(=O)([O-])[O-]", BindingMotif(["O", "O"], anchor="P([O-])[O-]"), name="EP2-")
    out = place_adsorbates(slab, di, pairs, count=1, mode="replace", bond_length=2.3)
    assert out.get_chemical_symbols().count("Br") == slab.get_chemical_symbols().count("Br") - 2
    first = len(out) - len(di.atoms)
    pb = [i for i in range(len(out)) if out[i].symbol == "Pb"]
    assert min(_min_dist(out, first + b, pb).min() for b in di.binding_atoms) == pytest.approx(2.3, abs=0.05)
    assert _validate(out).passed


def test_monatomic_proton_on_bridging_oxygen(anatase):
    slab = build_slab(anatase, (1, 0, 1), 5, min_thickness=10, vacuum=15, supercell=(2, 2))
    o2c = [s for s in surface_sites(slab, kinds=("top",), elements=["O"]) if s.coordination == (2,)]
    h = Adsorbate.from_smiles("[H+]", name="H+")
    assert h.binding_atoms == [0] and h.charge == 1
    out = place_adsorbates(slab, h, o2c[:1], bond_length=0.98)
    assert _min_dist(out, len(out) - 1, list(o2c[0].indices)).min() == pytest.approx(0.98, abs=1e-6)
    mol = get_molecules(out)[-1]
    assert mol["role"] == "adsorbate" and mol["charge"] == 1 and out.get_tags()[-1] == 2
    assert _validate(out).passed


def test_neighbor_pairs_nvalchemiops_matches_ase(monkeypatch, pt_bulk):
    if importlib.util.find_spec("nvalchemiops") is None:
        pytest.skip("nvalchemi-toolkit-ops not installed")
    slab = build_slab(pt_bulk, (1, 1, 1), 0, min_thickness=8, vacuum=10, supercell=(4, 4))
    slab.positions += np.array([3.1, -7.7, 0.2])
    r = 1.2 * rcov(slab.get_chemical_symbols())
    ref = {(int(i), int(j), tuple(np.round(d, 3))) for i, j, d in zip(*neighbor_pairs(slab, r))}
    monkeypatch.setenv("P2S_NEIGHBOR_BACKEND", "nvalchemiops")
    got = {(int(i), int(j), tuple(np.round(d, 3))) for i, j, d in zip(*neighbor_pairs(slab, r))}
    assert got == ref


def test_custom_structures_get_kind_profile_and_molecule_checks(tmp_path, monkeypatch):
    """Structures built with plain ASE (no toolkit builder) are still understood by describe/validation."""
    from ase.build import fcc111, molecule
    from ase.cluster import Octahedron

    from paper2structure.tools import add_molecule, composition_profile, emit, structure_kind

    core_shell = Octahedron("Pd", 5, cutoff=2, latticeconstant=3.89)
    core_shell.center(vacuum=8.0)
    core_shell.pbc = True
    r = np.linalg.norm(core_shell.positions - core_shell.positions.mean(axis=0), axis=1)
    core_shell.symbols[r > r.max() - 1.5] = "Pt"
    assert structure_kind(core_shell) == "nanocrystal"
    assert composition_profile(core_shell).startswith("radial composition from the center: 0-3 A Pd13")

    slab = fcc111("Pt", (2, 2, 4), vacuum=8.0)
    slab.pbc = True
    co = molecule("CO")
    top = slab.positions[np.argmax(slab.positions[:, 2])]
    c = [a.index for a in co if a.symbol == "C"][0]
    co.positions += top + [0, 0, 1.85] - co.positions[c]
    if co.positions[1 - c, 2] < co.positions[c, 2]:  # O up
        co.positions[1 - c, 2] += 2 * (co.positions[c, 2] - co.positions[1 - c, 2])
    model = add_molecule(slab, co, "[C-]#[O+]", "CO", site="top")
    assert structure_kind(model) == "slab" and model.get_tags()[-1] == 2
    report = validate_structure(model, "custom", max_atoms=5000, min_vacuum=10.0)
    assert report.passed and any(ch.name == "molecule_integrity" and ch.status == "pass" for ch in report.checks)
    stretched = model.copy()
    stretched.positions[[a.index for a in model if a.symbol == "O"][0]] += [0, 0, 1.5]  # C-O bond broken
    broken = validate_structure(stretched, "broken", max_atoms=5000, min_vacuum=10.0)
    assert any(ch.name == "molecule_integrity" and ch.status == "fail" for ch in broken.checks)
    with pytest.raises(ValueError, match="do not match"):
        add_molecule(slab, co, "O=C=O", "CO2")

    monkeypatch.chdir(tmp_path)
    emit(model, name="pt_co_custom", plan_id="P1")
    entry = json.loads((tmp_path / "structures" / "manifest.json").read_text())[0]
    assert entry["name"] == "pt_co_custom" and entry["script"]
    assert (tmp_path / "structures" / "pt_co_custom.vasp").exists()


def test_molecule_records_with_backslash_smiles_survive_extxyz(tmp_path):
    # cis double bonds (C/C=C\\C) broke the live core-shell run: ASE's extxyz writer does not escape backslashes
    from paper2structure.tools._geometry import get_molecules, set_molecules

    records = [{"mol_id": 0, "name": 'oleate "cis" 100%', "smiles": "[O-]C(=O)CCCCCCC/C=C\\CCCCCCCC"},
               {"mol_id": 1, "name": "ring", "smiles": "C%10CCCCCCCCC%10"}]
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[8, 8, 8], pbc=True)
    set_molecules(atoms, records)
    atoms.write(tmp_path / "m.extxyz")
    assert get_molecules(read(tmp_path / "m.extxyz")) == records
    legacy = Atoms("H", cell=[8, 8, 8], pbc=True)
    legacy.info["p2s_molecules"] = '[{"mol_id": 0, "smiles": "C/C=C\\C"}]'  # as an older file reads back
    assert get_molecules(legacy)[0]["smiles"] == "C/C=C\\C"


def test_gate_checks_binding_and_optional_charge(pt_bulk, cs_bulk):
    """The validation gate's adsorbate-binding check, and the charge check it runs when enabled."""
    from ase.build import molecule

    from paper2structure.tools import add_molecule
    from paper2structure.tools.geometry_checks import formal_charge

    def status(atoms, name, **kw):
        rep = validate_structure(atoms, "t", max_atoms=5000, min_vacuum=10.0, **kw)
        return next(c for c in rep.checks if c.name == name).status

    slab = build_slab(pt_bulk, (1, 1, 1), 0, min_thickness=6, vacuum=15, supercell=(2, 2))
    top = slab.positions[np.argmax(slab.positions[:, 2])]
    co = molecule("CO")
    co.positions -= co.positions[[a.index for a in co if a.symbol == "C"][0]]
    if co.positions[:, 2].min() < 0:
        co.positions[:, 2] *= -1  # O up
    bound = add_molecule(slab, Atoms(co.get_chemical_symbols(), co.positions + top + [0, 0, 1.85]), "[C-]#[O+]", "CO")
    floating = add_molecule(slab, Atoms(co.get_chemical_symbols(), co.positions + top + [0, 0, 6.0]), "[C-]#[O+]", "CO")
    assert status(bound, "adsorbate_binding") == "pass" and status(floating, "adsorbate_binding") == "fail"
    hydroxide = add_molecule(slab, Atoms("OH", [top + [0, 0, 2.0], top + [0, 0, 2.97]]), "[OH-]", "OH")
    assert validate_structure(hydroxide, "t", max_atoms=5000, min_vacuum=10.0).passed  # charge not checked by default
    assert formal_charge(hydroxide)[0] == -1 and status(hydroxide, "charge_neutrality", charge_neutrality=True) == "fail"

    # an O donor touching a surface Br instead of Pb (the live P2 failure) fails on an ionic slab
    cs = build_slab(cs_bulk, (1, 0, 0), 0, min_thickness=10, vacuum=15, supercell=(2, 2))
    zt = cs.positions[:, 2].max()
    br = [a.index for a in cs if a.symbol == "Br" and a.position[2] > zt - 0.5]
    water = Atoms("OH2", [[0, 0, 0], [0.76, 0.59, 0.3], [-0.76, 0.59, 0.3]])
    on_br = add_molecule(cs, Atoms(water.get_chemical_symbols(), water.positions + cs.positions[br[0]] + [0, 0, 2.6]),
                         "O", "water")
    assert status(on_br, "adsorbate_binding") == "fail"
    assert formal_charge(cs)[0] == formal_charge(cs, {"Pb": 2})[0]  # CsPbBr3 defaults: Cs+1 Pb+2 Br-1

    # mixed valence (Fe3O4) counts as neutral; a wrong plan oxidation state does not
    fe3o4 = Atoms("Fe3O4", positions=np.random.default_rng(0).random((7, 3)) * 8, cell=[20, 20, 20], pbc=True)
    assert formal_charge(fe3o4)[0] == 0
    assert formal_charge(fe3o4, {"Fe": 2})[0] == -2


def test_validation_is_fast_for_thousands_of_atoms_and_many_molecules(pt_bulk, co):
    import time

    slab = build_slab(pt_bulk, (1, 1, 1), 0, min_thickness=10, vacuum=20, supercell=(12, 12))
    covered = place_adsorbates(slab, co, surface_sites(slab, kinds=("top",), elements=["Pt"]), count=72,
                            coarse_step_deg=90.0, fine_step_deg=30.0, max_refine=0)
    t = time.perf_counter()
    rep = validate_structure(covered, "big", max_atoms=5000, min_vacuum=15.0)
    assert time.perf_counter() - t < 3.0, "validation must not be a bottleneck"
    assert len(covered) > 800 and rep.passed, rep.summary()


BRUCITE = dict(space_group="P-3m1", species=["Mg", "O", "H"],
               coords=[[0, 0, 0], [1 / 3, 2 / 3, 0.2216], [1 / 3, 2 / 3, 0.4303]], a=3.142, c=4.766)


def test_coordination_counts_bonds_inside_polyatomic_anions(pt_bulk, cs_bulk):
    """Only metal-metal contacts are excluded in compounds: C-O, O-H and S-S bonds count, and
    monatomic-anion materials keep their coordination numbers."""
    from paper2structure.tools._geometry import bulk_coordination

    def cn(bulk):
        return json.loads(bulk_coordination(bulk))

    assert cn(build_bulk(**BRUCITE)) == {"Mg": 6, "O": 4, "H": 1}  # rejected as "wrong sites" before
    assert cn(build_bulk("R-3c", ["Ca", "C", "O"], [[0, 0, 0], [0, 0, 0.25], [0.2578, 0, 0.25]],
                         a=4.99, c=17.06)) == {"Ca": 6, "C": 3, "O": 3}
    assert cn(build_bulk("Pa-3", ["Fe", "S"], [[0, 0, 0], [0.385, 0.385, 0.385]], a=5.417)) == {"Fe": 6, "S": 4}
    assert cn(build_bulk("I4_1/amd", ["Ti", "O"], [[0, 0, 0], [0, 0, 0.208]], a=3.784, c=9.515)) == {"Ti": 6, "O": 3}
    assert cn(cs_bulk) == {"Cs": 12, "Pb": 6, "Br": 6} and cn(pt_bulk) == {"Pt": 12}


def test_carved_hydroxide_particle_keeps_its_hydrogens():
    from paper2structure.tools._geometry import coordination_pairs

    nc = carve_nanocrystal(build_bulk(**BRUCITE), 14.0, oxidation_states={"Mg": 2, "O": -2, "H": 1})
    i, j, _ = coordination_pairs(nc)
    cn = np.bincount(i, minlength=len(nc))
    sym = np.array(nc.get_chemical_symbols())
    assert (sym == "H").sum() > 20 and not (cn == 0).any() and nc.info["p2s_charge"] == 0
    assert all(sym[j[i == k]].tolist() == ["O"] for k in np.where(sym == "H")[0])  # each H on its O


def test_hydrogen_bond_to_a_surface_hydroxyl_is_not_a_like_charge_contact():
    from paper2structure.tools import add_molecule

    bulk = build_bulk(**BRUCITE)
    oh_top = next(t for t in list_terminations(bulk, (0, 0, 1), min_thickness=8.0) if t.top == "H2O2" and t.symmetric)
    slab = build_slab(bulk, (0, 0, 1), termination=oh_top.index, min_thickness=8.0, vacuum=15.0, supercell=(3, 3))
    h = slab.positions[np.argmax(slab.positions[:, 2])]
    water = Atoms("OH2", [h + [0, 0, 1.8], h + [0.76, 0, 2.39], h + [-0.76, 0, 2.39]])  # O accepts the surface H
    report = validate_structure(add_molecule(slab, water, "O", "water"), "t", max_atoms=5000, min_vacuum=10.0)
    binding = next(c for c in report.checks if c.name == "adsorbate_binding")
    assert binding.status != "fail", binding.detail  # a surface H is not an anion


def test_van_der_waals_basal_planes_have_surface_sites():
    """Layered materials cleave without breaking a bond; their outermost layer is still exposed,
    while the fully coordinated lower atoms of a polar zinc-blende (111) bilayer are not."""
    from collections import Counter

    mos2 = build_bulk("P6_3/mmc", ["Mo", "S"], [[1 / 3, 2 / 3, 0.25], [1 / 3, 2 / 3, 0.621]], a=3.16, c=12.29)
    basal = next(t for t in list_terminations(mos2, (0, 0, 1), min_thickness=6.0) if t.symmetric)
    slab = build_slab(mos2, (0, 0, 1), termination=basal.index, min_thickness=6.0, vacuum=15.0, supercell=(2, 2))
    assert {s.symbols for s in surface_sites(slab, kinds=("top",))} == {("S",)} and "S(CN 3)" in basal.exposed
    brucite = build_bulk(**BRUCITE)
    oh = next(t for t in list_terminations(brucite, (0, 0, 1), min_thickness=8.0) if t.symmetric)
    slab = build_slab(brucite, (0, 0, 1), termination=oh.index, min_thickness=8.0, vacuum=15.0, supercell=(2, 2))
    assert Counter(slab[i].symbol for i in surface_atoms(slab)) == {"H": 4}
    cdse = build_bulk("F-43m", ["Cd", "Se"], [[0, 0, 0], [0.25, 0.25, 0.25]], a=6.08)
    polar = build_slab(cdse, (1, 1, 1), termination=0, min_thickness=8.0, vacuum=15.0, supercell=(2, 2))
    assert Counter(polar[i].symbol for i in surface_atoms(polar)) == {"Cd": 4}  # Se 0.88 A below stays inside
