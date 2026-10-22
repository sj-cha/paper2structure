"""Structure-building tools and deterministic validation (no LLM)."""
import importlib.util
import json
from pathlib import Path

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
from paper2structure.tools.evidence_checks import run_claim_checks
from paper2structure.tools.geometry_checks import validate_structure
from paper2structure.tools.schemas import CoordinationCheck
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


def test_cationic_molecule_atoms_are_not_like_charge_contacts(cs_bulk):
    # an ammonium N close to surface Br is an attractive contact (how ammonium ligands bind), not like charges
    k = next(t.index for t in list_terminations(cs_bulk, (0, 0, 1), min_thickness=10) if t.top == "PbBr2")
    slab = build_slab(cs_bulk, (0, 0, 1), k, min_thickness=10, vacuum=15, supercell=(2, 2))
    ma = Adsorbate.from_smiles("C[NH3+]", BindingMotif(["N"], anchor="[NH3+]"), name="MA")
    near = place_adsorbates(slab, ma, surface_sites(slab, kinds=("top",), elements=["Br"])[:1], bond_length=2.3)
    check = next(c for c in _validate(near).checks if c.name == "anion_contacts")
    assert check.status == "pass", check.detail


def test_guest_molecules_are_part_of_the_crystal(cs_bulk):
    # organic cations inside a crystal: registered, checked for integrity, not for surface binding
    from paper2structure.tools import add_molecule

    ma = Adsorbate.from_smiles("C[NH3+]", BindingMotif(["N"], anchor="[NH3+]"), name="MA")
    host = cs_bulk.repeat((2, 2, 2))
    del host[[a.index for a in host if a.symbol == "Cs"][:1]]
    centre = host.cell.array.sum(axis=0) / 4  # an emptied A site of the 2x2x2 cell
    molecule = ma.atoms.copy()
    molecule.translate(centre - molecule.positions.mean(axis=0))
    with_guest = add_molecule(host, molecule, "C[NH3+]", "MA", role="guest")
    assert get_molecules(with_guest)[0]["role"] == "guest" and "1 x MA (guest)" in describe(with_guest)
    names = {c.name for c in _validate(with_guest).checks}
    assert "adsorbate_binding" not in names and "molecule_integrity" in names
    with pytest.raises(ValueError, match="'guest'"):
        add_molecule(host, molecule, "C[NH3+]", "MA", role="ligand")


def _check(atoms, name):
    return next(c for c in _validate(atoms).checks if c.name == name)


def test_bond_valence_passes_real_crystals_and_flags_distorted_ones(cs_bulk, anatase):
    for good in (cs_bulk.repeat(2), anatase, build_bulk("Fm-3m", ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]], a=4.21)):
        assert _check(good, "bond_valence").status == "pass", _check(good, "bond_valence").detail
        assert _check(good, "like_charge_contacts").status == "pass"
    squeezed, stretched = cs_bulk.repeat(2), cs_bulk.repeat(2)
    squeezed.set_cell(squeezed.cell * 0.8, scale_atoms=True)  # Pb-Br 2.35 A instead of 2.94
    stretched.set_cell(stretched.cell * 1.25, scale_atoms=True)
    for bad, sign in ((squeezed, "+"), (stretched, "-")):
        check = _check(bad, "bond_valence")
        assert check.status == "fail" and f"Pb" in check.detail and f"({sign}" in check.detail, check.detail
    # only the framework cation counts: Cs in its cuboctahedral cage is under-bonded by nature
    assert "Cs" not in _check(cs_bulk.repeat(2), "bond_valence").detail


def test_one_overly_short_bond_and_like_charge_contacts(anatase):
    moved = anatase.repeat((2, 2, 1))
    ti = [a.index for a in moved if a.symbol == "Ti"]
    o = min((a.index for a in moved if a.symbol == "O"), key=lambda k: moved.get_distance(ti[0], k, mic=True))
    moved.set_distance(ti[0], o, 1.55, fix=0, mic=True)  # one Ti-O bond squeezed from 1.93 A
    assert "one Ti-O bond of 1.55 A" in _check(moved, "bond_valence").detail
    close = anatase.repeat((2, 2, 1))
    ti0, ti1 = [a.index for a in close if a.symbol == "Ti"][:2]
    close.set_distance(ti0, ti1, 2.0, fix=0, mic=True)  # Ti-Ti 2.0 A: x0.63 of the covalent sum
    check = _check(close, "like_charge_contacts")
    assert check.status == "fail" and "Ti" in check.detail, check.detail


def test_unregistered_organics_are_flagged_not_assessed(cs_bulk):
    ma = Adsorbate.from_smiles("C[NH3+]", BindingMotif(["N"], anchor="[NH3+]"), name="MA")
    host = cs_bulk.repeat(2)
    cs = next(a.index for a in host if a.symbol == "Cs")
    centre = host.positions[cs].copy()
    del host[cs]
    host.extend(Atoms(ma.atoms.get_chemical_symbols(), positions=ma.atoms.positions - ma.atoms.positions.mean(0) + centre))
    host.arrays["mol_id"][:] = -1  # no add_molecule: the MA atoms are plain substrate atoms
    check = _check(host, "bond_valence")
    assert check.status == "warn" and "role='guest'" in check.detail


def test_molecules_must_be_rdkit_conformers(cs_bulk):
    from paper2structure.tools import add_molecule

    host = cs_bulk.repeat(2)
    cs = next(a.index for a in host if a.symbol == "Cs")
    centre = host.positions[cs].copy()
    del host[cs]
    ba = Adsorbate.from_smiles("CCCC[NH3+]", name="BA", seed=3).atoms
    ba.rotate(70, "x", center="COM")
    ba.translate(centre - ba.get_center_of_mass())
    ok = add_molecule(host, ba, "CCCC[NH3+]", "BA", role="guest")  # moved rigidly: accepted
    assert _check(ok, "molecule_geometry").status == "pass"

    bent = ba.copy()  # one C-C-C angle squeezed by hand
    c = [a.index for a in bent if a.symbol == "C"]
    bent.set_angle(c[0], c[1], c[2], 80.0, indices=[c[0]] + [a.index for a in bent if a.symbol == "H"
                                                             and bent.get_distance(a.index, c[0]) < 1.2])
    with pytest.raises(ValueError, match="not the RDKit conformer"):
        add_molecule(host, bent, "CCCC[NH3+]", "BA", role="guest")
    # registered without add_molecule's check (e.g. set_molecules by hand): the gate fails it
    sneaked = ok.copy()
    sneaked.positions[-len(ba):] = bent.positions
    assert _check(sneaked, "molecule_geometry").status == "fail"


def test_unregistered_organics_and_isolated_atoms_fail(cs_bulk):
    ma = Adsorbate.from_smiles("C[NH3+]", name="MA", seed=1).atoms
    host = cs_bulk.repeat(2)
    cs = next(a.index for a in host if a.symbol == "Cs")
    centre = host.positions[cs].copy()
    del host[cs]
    host.extend(Atoms(ma.get_chemical_symbols(), positions=ma.positions - ma.positions.mean(0) + centre))
    host.arrays["mol_id"][:] = -1  # organic atoms left in the substrate
    assert _check(host, "unregistered_molecules").status == "fail"
    lone = cs_bulk.repeat(2)
    lone.center(vacuum=8.0, axis=2)
    top = lone.positions[:, 2].max()
    lone += Atoms("C", positions=[[1.0, 1.0, top + 4.0]])  # a placeholder atom in the vacuum, 4 A from everything
    lone.arrays["mol_id"][-1] = -1
    assert _check(lone, "isolated_atoms").status == "fail"
    assert _check(cs_bulk.repeat(2), "isolated_atoms").status == "pass"  # Cs in its cage is bonded


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


def test_cif_round_trip_reads_the_p1_sites_as_written(tmp_path, pt_bulk):
    import re

    from paper2structure.tools.geometry_checks import check_roundtrip

    atoms = pt_bulk.repeat((8, 8, 8))  # 2048 atoms: ASE's symmetry-expanding CIF reader takes about 16 s
    files = {k: str(v) for k, v in export_structure(atoms, tmp_path, "pt", formats=("poscar", "cif", "extxyz")).items()}
    check = check_roundtrip(atoms, files)
    assert check.status == "pass" and "cif" in check.detail
    cif = Path(files["cif"])
    cif.write_text(re.sub(r"(_cell_length_a\s+)\S+", r"\g<1>40.0", cif.read_text()))
    assert check_roundtrip(atoms, files).detail == "cif: cell changed"


def test_the_gpu_neighbor_list_waits_for_torch_or_very_large_structures(monkeypatch, pt_bulk):
    import sys
    import types

    import paper2structure.tools._geometry as geo
    from ase.neighborlist import neighbor_list

    calls = []

    def fake(atoms, cutoff):
        calls.append(len(atoms))
        return neighbor_list("ijD", atoms, cutoff)

    monkeypatch.setattr(geo, "_nvalchemi_pairs", fake)
    monkeypatch.setattr(geo, "_backend_failed", False)
    monkeypatch.setenv("P2S_NEIGHBOR_BACKEND", "auto")
    big = pt_bulk.repeat((8, 8, 8))  # 2048 atoms
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    geo.neighbor_pairs(big, 3.0)
    assert calls == []  # importing torch and warp would cost more than the GPU saves here
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    geo.neighbor_pairs(big, 3.0)
    assert calls == [2048]


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


MOF5 = [("Zn", [0.20639, 0.29361, 0.20639]), ("O", [0.25, 0.25, 0.25]), ("O", [0.21964, 0.28036, 0.13389]),
        ("C", [0.25, 0.25, 0.1114]), ("C", [0.25, 0.25, 0.0540]), ("C", [0.21875, 0.28125, 0.0270]),
        ("H", [0.1965, 0.3035, 0.0447])]  # Zn4O(BDC)3, Fm-3m (literature coordinates)


def test_linkers_bonded_into_a_framework_are_not_loose_molecules():
    mof = build_bulk("Fm-3m", [s for s, _ in MOF5], [c for _, c in MOF5], a=25.755, formula_units=8)
    report = _validate(mof)
    checks = {c.name: c for c in report.checks}
    assert report.passed and "unregistered_molecules" not in checks
    assert checks["bond_valence"].status == "pass" and "Zn 1.6" in checks["bond_valence"].detail
    # a formic acid dropped into the pore without registering it is still a loose molecule
    loose = Adsorbate.from_smiles("OC=O", seed=0).atoms
    loose.translate(mof.cell.array.sum(axis=0) / 2 - loose.positions.mean(axis=0))
    with_loose = mof + loose
    with_loose.set_array("mol_id", np.full(len(with_loose), -1))
    assert _check(with_loose, "unregistered_molecules").status == "fail"


# ---------------------------------------------------------------------------------------------
# Periodicity: every cell vector is measured as the exported (fully periodic) files will have it
# ---------------------------------------------------------------------------------------------


def test_ase_slabs_left_non_periodic_along_the_normal_are_measured_like_their_poscar():
    from ase.build import fcc111

    from paper2structure.tools import composition_profile, structure_kind, vacuum_gap

    slab = fcc111("Pt", (3, 3, 4), vacuum=10.0)  # ASE leaves pbc (T, T, F)
    assert structure_kind(slab) == "slab" and vacuum_gap(slab) == pytest.approx(20.0, abs=0.05)
    assert composition_profile(slab) == "layers from the top (4 in total): Pt9 | Pt9 | Pt9 | Pt9"
    assert _validate(slab, min_vacuum=15.0).passed
    tight = fcc111("Pt", (3, 3, 4), vacuum=2.0)  # its POSCAR has 4 A between images
    assert next(c for c in _validate(tight, min_vacuum=15.0).checks if c.name == "vacuum").status == "fail"
    wrapped = slab.copy()  # a slab shifted across the cell boundary is still one slab
    wrapped.pbc = True
    wrapped.translate([0, 0, -slab.positions[:, 2].mean()])
    wrapped.wrap()
    assert structure_kind(wrapped) == "slab" and vacuum_gap(wrapped) == pytest.approx(20.0, abs=0.05)
    assert composition_profile(wrapped) == composition_profile(slab)


def test_a_bulk_label_holds_only_for_the_crystals_own_lattice(pt_bulk):
    from ase.build import surface

    from paper2structure.tools import structure_kind

    assert structure_kind(pt_bulk.repeat((2, 2, 2))) == "bulk"
    opened = pt_bulk.repeat((3, 3, 3))
    opened.center(vacuum=10.0, axis=2)
    cut = surface(pt_bulk, (1, 1, 1), 4, vacuum=10.0)
    assert structure_kind(opened) == structure_kind(cut) == "slab"  # both carry the stale "bulk" label
    assert "vacuum" in {c.name for c in _validate(opened, min_vacuum=15.0).checks}


def test_wire_interiors_are_measured_across_the_axis(cs_bulk):
    from paper2structure.tools import structure_kind

    wire = cs_bulk.repeat((4, 4, 1))
    wire.center(vacuum=10.0, axis=(0, 1))
    wire.info = {}
    assert structure_kind(wire) == "wire"
    check = CoordinationCheck(id="C", claim="PbBr6 octahedra", center="Pb", neighbor="Br", number=6)
    [result] = run_claim_checks(wire, [check])
    assert result.status == "pass", result.detail  # was "no interior framework Pb": the periodic ends are no surface
    exposed = set(surface_atoms(wire))
    assert not exposed & {a.index for a in wire if a.symbol == "Pb" and 5 < a.position[0] - wire.positions[:, 0].min() < 13
                         and 5 < a.position[1] - wire.positions[:, 1].min() < 13}
    with pytest.raises(ValueError, match="no surface"):
        surface_atoms(cs_bulk.repeat(2))


def test_a_particle_on_a_support_keeps_the_support_periodic(pt_bulk, co):
    support = build_slab(pt_bulk, (1, 1, 1), min_thickness=6, vacuum=20, supercell=(6, 6))
    particle = build_cluster("Pt", "cuboctahedron", 2, 3.924)
    top = support.positions[:, 2].max()
    particle.translate(support.cell.array[:2].sum(axis=0) / 2 + [0, 0, top + 2.4] - particle.positions.mean(axis=0)
                       + [0, 0, np.ptp(particle.positions[:, 2]) / 2])
    model = particle + support  # the order the review found: the particle's label and vacuum come along
    model.set_cell(support.cell)
    model.pbc = True
    sites = [s for s in surface_sites(model) if s.position[2] > top + 4][:1]
    covered = place_adsorbates(model, co, sites, bond_length=1.85, seed=0)
    assert np.allclose(covered.cell.array[:2], support.cell.array[:2])  # was re-boxed into a finite flake


# ---------------------------------------------------------------------------------------------
# Placement, solvent and the molecule bookkeeping
# ---------------------------------------------------------------------------------------------


def test_crowded_adsorbates_are_rescored_against_each_other(pt_bulk):
    slab = build_slab(pt_bulk, (1, 1, 1), min_thickness=6, vacuum=15, supercell=(3, 3))
    acetate = Adsorbate.from_smiles("CC(=O)[O-]", BindingMotif(["O"], anchor="C(=O)[O-]"), name="acetate", seed=0)
    out = place_adsorbates(slab, acetate, surface_sites(slab), count=8, seed=0)
    assert _check(out, "overlaps").status == "pass"  # the early molecules used to be scored without the later ones


def test_long_chains_are_not_bent_by_the_slab_image_across_the_vacuum(pt_bulk):
    slab = build_slab(pt_bulk, (1, 1, 1), min_thickness=8, vacuum=15, supercell=(3, 3))
    thiolate = Adsorbate.from_smiles("CCCCCCCCCCCC[S-]", BindingMotif(["S"], anchor="[S-]"), name="C12S", seed=0)
    hollow = [s for s in surface_sites(slab, kinds=("hollow",)) if s.subtype == "fcc"][:1]
    out = place_adsorbates(slab, thiolate, hollow, seed=0)
    assert get_molecules(out)[0]["clearance"] > 1.2  # was 0.6 against the image the returned cell no longer has
    assert out.cell[2, 2] > slab.cell[2, 2]  # the vacuum above the chain is restored


def test_sites_follow_a_particle_recentered_by_earlier_placements(co):
    pt55 = build_cluster("Pt", "cuboctahedron", 3, 3.924)
    sites = surface_sites(pt55, kinds=("top",))
    covered = place_adsorbates(pt55, co, sites[:3], bond_length=1.85, seed=0)
    proton = Adsorbate.from_smiles("[H+]", name="H")
    site = sites[20]
    out = place_adsorbates(covered, proton, [site], bond_length=1.55, seed=0)
    assert np.linalg.norm(out.positions[-1] - out.positions[site.indices[0]]) == pytest.approx(1.55, abs=0.02)
    with pytest.raises(ValueError, match="does not match this structure"):  # atoms removed since surface_sites
        place_adsorbates(covered[:site.indices[0]], proton, [site], bond_length=1.55, seed=0)


def test_ammonium_hydrogen_bonded_to_surface_bromide_is_bound(cs_bulk):
    k = next(t.index for t in list_terminations(cs_bulk, (0, 0, 1), min_thickness=10) if t.top == "PbBr2")
    slab = build_slab(cs_bulk, (0, 0, 1), k, min_thickness=10, vacuum=15, supercell=(2, 2))
    ma = Adsorbate.from_smiles("C[NH3+]", BindingMotif(["N"], anchor="[NH3+]"), name="MA")
    site = surface_sites(slab, kinds=("top",), elements=["Br"])[:1]
    for bond in (2.3, 3.3):  # an N-H...Br contact, from close to the usual 3.3 A
        report = _validate(place_adsorbates(slab, ma, site, bond_length=bond))
        binding = next(c for c in report.checks if c.name == "adsorbate_binding")
        assert binding.status == "pass" and report.passed, binding.detail
    water = Adsorbate.from_smiles("O", BindingMotif(["O"], anchor="[OH2]"), name="water")
    wet = place_adsorbates(slab, water, site, bond_length=2.3)  # a neutral O on Br is still a like-charge contact
    assert next(c for c in _validate(wet).checks if c.name == "adsorbate_binding").status == "fail"


def test_add_molecule_keeps_a_builder_slabs_vacuum(pt_bulk, co):
    from paper2structure.tools import add_molecule

    slab = build_slab(pt_bulk, (1, 1, 1), min_thickness=6, vacuum=15, supercell=(3, 3))
    site = surface_sites(slab)[0]
    molecule = co.atoms.copy()
    molecule.translate(site.position + [0, 0, 1.85] - molecule.positions[co.binding_atoms[0]])
    model = add_molecule(slab, molecule, co.smiles, "CO")
    assert _validate(model, min_vacuum=15.0).passed  # was 12 A: the molecule ate into the vacuum


def test_solvent_surrounds_adsorbates_and_fills_an_empty_box(pt_bulk, co):
    slab = build_slab(pt_bulk, (1, 1, 1), min_thickness=6, vacuum=15, supercell=(4, 4))
    covered = place_adsorbates(slab, co, surface_sites(slab)[:1], bond_length=1.85, seed=0)
    wet = add_solvent(covered, "O", n_molecules=40, thickness=12, seed=0)
    water = wet.positions[[a.index for a in wet if a.tag == 3]]
    assert water[:, 2].min() - slab.positions[:, 2].max() < 3.5  # was above the CO, a dry gap over the surface
    box = Atoms(cell=[15.0] * 3, pbc=True)
    liquid = add_solvent(box, "O", density=1.0, seed=0)
    assert len(get_molecules(liquid)) > 80  # 113 at 1 g/cm^3 in 3375 A^3; was a crash on the empty cell


def test_slabs_and_particles_of_filled_crystals_are_refused(cs_bulk):
    from paper2structure.tools import add_molecule

    ma = Adsorbate.from_smiles("C[NH3+]", BindingMotif(["N"], anchor="[NH3+]"), name="MA")
    host = cs_bulk.repeat((2, 2, 2))
    del host[[a.index for a in host if a.symbol == "Cs"][:1]]
    molecule = ma.atoms.copy()
    molecule.translate(host.cell.array.sum(axis=0) / 4 - molecule.positions.mean(axis=0))
    filled = add_molecule(host, molecule, "C[NH3+]", "MA", role="guest")
    with pytest.raises(ValueError, match="framework only"):
        build_slab(filled, (0, 0, 1))
    with pytest.raises(ValueError, match="framework only"):
        carve_nanocrystal(filled, 12.0)


def test_exported_files_keep_the_binding_atoms_of_reordered_molecules(tmp_path, anatase):
    terms = list_terminations(anatase, (1, 0, 1), min_thickness=10)
    k = next(t.index for t in terms if t.symmetric and t.stoichiometric and t.top == "TiO2"
             and {s.coordination for s in surface_sites(build_slab(anatase, (1, 0, 1), t.index, min_thickness=10,
                                                                   vacuum=15), elements=["Ti"])} == {(5,)})
    slab = build_slab(anatase, (1, 0, 1), k, min_thickness=10, vacuum=15, supercell=(3, 2))
    bridges = [s for s in surface_sites(slab, kinds=("bridge",), elements=["Ti"], max_pair_distance=4.0)
               if s.coordination == (5, 5)]
    fo = Adsorbate.from_smiles("[O-]C=O", BindingMotif(["O", "O"], anchor="C(=O)[O-]"), name="formate")
    out = place_adsorbates(slab, fo, bridges[:1], count=1)
    files = export_structure(out, tmp_path, "fo", formats=("extxyz",))
    back = read(files["extxyz"])  # POSCAR order: each element in one block, so the formate reads O, O, C, H
    [record] = get_molecules(back)
    members = np.where(back.arrays["mol_id"] == record["mol_id"])[0]
    assert [back[int(members[b])].symbol for b in record["binding_atoms"]] == ["O", "O"]


# ---------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------


def test_bond_valence_tolerates_vacancies_and_still_passes_substituted_frameworks():
    sno2 = build_bulk("P4_2/mnm", ["Sn", "O"], [[0, 0, 0], [0.307, 0.307, 0]], a=4.737, c=3.186).repeat((3, 3, 4))
    del sno2[next(a.index for a in sno2 if a.symbol == "O")]
    check = _check(sno2, "bond_valence")
    assert check.status != "fail" and "Sn 4." in check.detail and "vs +4" in check.detail, check.detail  # was vs +2
    co_mof = build_bulk("Fm-3m", ["Co" if s == "Zn" else s for s, _ in MOF5], [c for _, c in MOF5], a=25.755)
    assert _check(co_mof, "bond_valence").status == "pass"  # uncounted linker C/H: the lowest common state fits


def test_radical_and_isotope_molecules_match_their_rdkit_conformers(pt_bulk):
    from paper2structure.tools import add_molecule

    slab = build_slab(pt_bulk, (1, 1, 1), min_thickness=6, vacuum=15, supercell=(3, 3))
    for smiles in ("[O]", "[OH]"):
        ads = Adsorbate.from_smiles(smiles, BindingMotif(["O"], anchor="[O]") if smiles == "[OH]" else None, name=smiles)
        out = place_adsorbates(slab, ads, surface_sites(slab)[:1], seed=0)
        assert _check(out, "molecule_geometry").status == "pass"  # was "its bond graph does not match"
    d2o = Adsorbate.from_smiles("[2H]O[2H]", seed=0).atoms
    d2o.translate([5.0, 5.0, slab.positions[:, 2].max() + 6.0])
    assert len(get_molecules(add_molecule(slab, d2o, "[2H]O[2H]", "D2O", role="solvent"))) == 1


def test_checks_that_could_not_run_say_so(anatase):
    terms = list_terminations(anatase, (1, 0, 1), min_thickness=4)
    thin = build_slab(anatase, (1, 0, 1), terms[0].index, min_thickness=4, vacuum=15)
    report = _validate(thin)
    check = next(c for c in report.checks if c.name == "bond_valence")
    assert check.status == "skip" and "not assessed" in check.detail  # was "pass"
    assert "bond_valence not assessed" in report.summary(include_skipped=True)
    assert "bond_valence" not in report.summary()  # the builder's feedback stays actionable
