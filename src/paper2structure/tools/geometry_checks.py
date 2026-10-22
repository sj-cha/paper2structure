"""Deterministic structure validation (no LLM). Molecule-integrity checks follow PerovAgent's
bond-graph comparison."""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

import numpy as np
from ase import Atoms
from ase.io import read
from rdkit import Chem

from paper2structure.tools.schemas import CheckResult, ValidationReport
from paper2structure.tools._geometry import (ANION_FORMERS, get_molecules, mol_ids, neighbor_pairs, rcov,
                                             surface_anions)
from paper2structure.tools.structure_export import grouped_order, structure_kind, vacuum_gap


def infer_bonds(atoms: Atoms, scale: float = 1.2) -> frozenset:
    """Bonds from geometry (minimum image): pairs closer than scale x the covalent-radius sum."""
    if len(atoms) < 2:
        return frozenset()
    i, j, _ = neighbor_pairs(atoms, scale * rcov(atoms.get_chemical_symbols()))
    return frozenset((int(min(a, b)), int(max(a, b))) for a, b in zip(i, j) if a != b)


def wl_certificate(symbols: list[str], bonds) -> tuple:
    """Weisfeiler-Lehman isomorphism certificate of an element-labelled graph."""
    adj: dict[int, list[int]] = {k: [] for k in range(len(symbols))}
    for a, b in bonds:
        adj[a].append(b)
        adj[b].append(a)
    lab = {k: symbols[k] for k in adj}
    for _ in range(3):
        lab = {k: lab[k] + "(" + ",".join(sorted(lab[n] for n in adj[k])) + ")" for k in adj}
    return tuple(sorted(lab.values()))


def _pair_text(atoms: Atoms, i: int, j: int, d: float) -> str:
    return f"{atoms[i].symbol}{i}-{atoms[j].symbol}{j} {d:.2f} A"


def check_roundtrip(atoms: Atoms, files: dict[str, str]) -> CheckResult:
    """Re-read every exported file (`files`: format -> path) and compare species order, cell and positions."""
    ordered = atoms[grouped_order(atoms)]
    problems = []
    for fmt, path in files.items():
        if fmt == "metadata":
            continue
        try:
            back = read(path)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{fmt}: unreadable ({e})")
            continue
        if back.get_chemical_symbols() != ordered.get_chemical_symbols():
            problems.append(f"{fmt}: species/order changed")
            continue
        # CIF stores cell parameters, not vectors, so compare orientation-independent quantities
        if not np.allclose(back.cell.cellpar(), ordered.cell.cellpar(), atol=1e-3):
            problems.append(f"{fmt}: cell changed")
            continue
        frac = back.get_scaled_positions(wrap=False) - ordered.get_scaled_positions(wrap=False)
        frac -= np.round(frac)
        if np.abs(frac @ ordered.cell.array).max() > 1e-2:
            problems.append(f"{fmt}: positions changed")
    if problems:
        return CheckResult(name="export_roundtrip", status="fail", detail="; ".join(problems))
    return CheckResult(name="export_roundtrip", status="pass",
                       detail="re-read " + ", ".join(k for k in files if k != "metadata"))


_OXI = re.compile(r"([A-Z][a-z]?):([+-]?\d+)")


def parse_oxidation_states(text: str) -> dict[str, int]:
    """'Cd:+2 Se:-2' (atoms.info["p2s_oxidation_states"]) -> {"Cd": 2, "Se": -2}."""
    return {e: int(q) for e, q in _OXI.findall(text or "")}


def formal_charge(atoms: Atoms, oxidation_states: Optional[dict[str, int]] = None) -> tuple[Optional[int], str]:
    """Net formal charge: molecule charges from their SMILES plus the substrate's ionic charges.

    Substrate oxidation states come from `oxidation_states`, then the structure's own record
    (atoms.info["p2s_oxidation_states"], e.g. from carve_nanocrystal), then defaults: anion formers
    (O, S, Se, halogens, N, ...) at their most negative common state, metals at the per-element
    common positive states closest to neutral (mixed valence allowed between consecutive states).
    Metallic and covalent substrates carry no ionic charge. Returns (net charge or None when it
    cannot be assigned, how it was assigned)."""
    from itertools import product

    from pymatgen.core import Element

    ids = mol_ids(atoms)
    molecular = int(sum(int(m.get("charge", 0)) for m in get_molecules(atoms)))
    counts = Counter(s for s, m in zip(atoms.get_chemical_symbols(), ids) if m < 0)
    given = {**parse_oxidation_states(str(atoms.info.get("p2s_oxidation_states", ""))), **(oxidation_states or {})}
    anions = [e for e in counts if e in ANION_FORMERS and e not in given]
    metals = [e for e in counts if e not in given and e not in ANION_FORMERS
              and (Element(e).is_metal or Element(e).is_metalloid)]
    others = [e for e in counts if e not in given and e not in anions and e not in metals]
    fixed = sum(given[e] * n for e, n in counts.items() if e in given)
    if not anions and not any(given.get(e, 0) < 0 for e in counts):  # metallic or covalent substrate
        net = molecular + fixed
        return net, f"non-ionic substrate (charge {fixed:+d}); molecules {molecular:+d}"
    if others:
        return None, f"cannot assign oxidation states to {', '.join(sorted(others))} in the substrate"
    for e in anions:
        neg = [q for q in Element(e).common_oxidation_states if q < 0]
        if not neg:
            return None, f"no negative oxidation state for {e}"
        given[e] = min(neg)
        fixed += given[e] * counts[e]
    options = {}
    for e in metals:
        pos = [q for q in Element(e).common_oxidation_states if q > 0] or [q for q in Element(e).oxidation_states if q > 0][:3]
        if not pos:
            return None, f"no positive oxidation state for {e}"
        options[e] = pos
    best = None
    for combo in product(*(options[e] for e in metals)):
        net = molecular + fixed + sum(q * counts[e] for e, q in zip(metals, combo))
        if best is None or abs(net) < abs(best[0]):
            best = (net, dict(zip(metals, combo)))
    net, chosen = best if best else (molecular + fixed, {})
    states = {**given, **chosen}
    text = " ".join(f"{e}{q:+d}" for e, q in sorted(states.items())) + f"; molecules {molecular:+d}"
    for e in metals:  # mixed valence between consecutive states absorbs up to one charge per atom
        pos = options[e]
        q = chosen[e]
        room = counts[e] * ((max(pos) - q) if net < 0 else (q - min(pos)))
        if net and any(abs(a - b) == 1 for a in pos for b in pos) and abs(net) <= room:
            return 0, text + f"; neutral with mixed-valence {e}"
    return net, text


def _check_binding(atoms: Atoms, adsorbates: list[dict], ids: np.ndarray, r: np.ndarray, binding_scale: float,
                   detached_scale: float) -> CheckResult:
    """Closest substrate contact of each adsorbate (of its recorded binding atoms, else of any atom)."""
    sym = atoms.get_chemical_symbols()
    sub = ids < 0
    anions = surface_anions(s for s, k in zip(sym, sub) if k)
    i, j, D = neighbor_pairs(atoms, detached_scale * r)
    keep = (ids[i] >= 0) & sub[j]
    i, j, d = i[keep], j[keep], np.linalg.norm(D[keep], axis=1)
    ratio = d / (r[i] + r[j])
    failed, weak, ok = [], [], 0
    for m in adsorbates:
        members = np.where(ids == m["mol_id"])[0]
        binding = [int(members[k]) for k in m.get("binding_atoms") or [] if k < len(members)] or list(members)
        sel = np.isin(i, binding)
        label = f"mol {m['mol_id']} {m['name']}"
        if not sel.any():
            failed.append(f"{label}: detached (no substrate atom within {detached_scale} x covalent sum)")
            continue
        k = np.where(sel)[0][np.argmin(ratio[sel])]
        a, b = int(i[k]), int(j[k])
        contact = f"{sym[a]}{a}-{sym[b]}{b} {d[k]:.2f} A ({ratio[k]:.2f}x)"
        if sym[a] in ANION_FORMERS and sym[b] in anions:
            failed.append(f"{label}: donor {contact} touches a surface {sym[b]} (like charges), not a cation")
        elif ratio[k] > binding_scale:
            weak.append(f"{label}: closest contact {contact}")
        else:
            ok += 1
    status = "fail" if failed else ("warn" if weak else "pass")
    detail = "; ".join((failed + weak)[:5]) if failed or weak else f"{ok} adsorbates bound to the substrate"
    if failed or weak:
        detail = f"{len(failed)} unbound or misplaced, {len(weak)} weakly bound of {len(adsorbates)}: " + detail
    return CheckResult(name="adsorbate_binding", status=status, detail=detail)


_POSITIVE: dict[str, frozenset[str]] = {}


def _positive_elements(smiles: str) -> frozenset[str]:
    """Elements carrying a positive formal charge in a SMILES (the N of an ammonium or pyridinium)."""
    if smiles not in _POSITIVE:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        atoms = mol.GetAtoms() if mol is not None else ()
        _POSITIVE[smiles] = frozenset(a.GetSymbol() for a in atoms if a.GetFormalCharge() > 0)
    return _POSITIVE[smiles]


def validate_structure(atoms: Atoms, structure_id: str, *, max_atoms: int, min_vacuum: float,
                       overlap_scale: float = 0.6, close_contact_scale: float = 0.85, bond_scale: float = 1.2,
                       anion_contact_scale: float = 1.3, binding_scale: float = 1.5, detached_scale: float = 2.0,
                       charge_neutrality: bool = False, oxidation_states: Optional[dict[str, int]] = None,
                       files: Optional[dict[str, str]] = None) -> ValidationReport:
    """Geometric and chemical sanity checks. Pairs closer than overlap_scale x the covalent-radius
    sum fail; contacts involving a molecule closer than close_contact_scale x the sum (other than
    the molecule's own bonds) warn, as do electronegative molecule atoms (O, N, S, halogens) closer
    than anion_contact_scale x the sum to a substrate anion; each molecule's bond graph must match
    its SMILES. Each adsorbate must touch the substrate (closest contact within binding_scale x the
    covalent sum; beyond detached_scale fails) and, on metal/metalloid compounds, an
    electronegative donor atom must not touch an electronegative substrate atom (surface_anions). With charge_neutrality the net formal charge must be zero
    (formal_charge, using `oxidation_states` where given)."""
    checks: list[CheckResult] = []
    metrics: dict[str, float | int | str] = {"formula": atoms.get_chemical_formula(), "n_atoms": len(atoms)}
    pos = atoms.get_positions()

    finite = bool(np.all(np.isfinite(pos)))
    checks.append(CheckResult(name="finite_coordinates", status="pass" if finite else "fail",
                              detail="" if finite else "NaN/inf coordinates"))
    if not finite:
        return ValidationReport(structure_id=structure_id, passed=False, checks=checks, metrics=metrics)

    vol = abs(atoms.cell.volume) if atoms.cell.rank == 3 else 0.0
    cell_ok = vol > 1.0 or not any(atoms.pbc)
    checks.append(CheckResult(name="cell", status="pass" if cell_ok else "fail",
                              detail=f"volume {vol:.1f} A^3" if cell_ok else "degenerate periodic cell"))
    metrics["cell_volume"] = round(float(vol), 2)
    checks.append(CheckResult(name="atom_count", status="pass" if len(atoms) <= max_atoms else "fail",
                              detail=f"{len(atoms)} atoms (limit {max_atoms})"))

    r = rcov(atoms.get_chemical_symbols())
    ids = mol_ids(atoms)
    # one neighbor list for overlaps, close contacts, molecule bonds and like-charge contacts
    i, j, D = neighbor_pairs(atoms, max(1.0, bond_scale, anion_contact_scale) * r)
    keep = i < j
    i, j, d = i[keep], j[keep], np.linalg.norm(D[keep], axis=1)
    ratio = d / (r[i] + r[j]) if len(d) else np.array([])
    if len(ratio):
        k = int(np.argmin(ratio))
        metrics["min_distance_ratio"] = round(float(ratio[k]), 3)
        metrics["closest_pair"] = _pair_text(atoms, int(i[k]), int(j[k]), float(d[k]))
    bad = np.where(ratio < overlap_scale)[0] if len(ratio) else []
    checks.append(CheckResult(
        name="overlaps", status="fail" if len(bad) else "pass",
        detail=(f"{len(bad)} pairs below {overlap_scale} x covalent sum: " +
                ", ".join(_pair_text(atoms, int(i[k]), int(j[k]), float(d[k])) for k in bad[:5]))
        if len(bad) else f"no pair below {overlap_scale} x covalent sum"))

    molecules = get_molecules(atoms)
    intra = (ids[i] >= 0) & (ids[i] == ids[j]) & (ratio < bond_scale) if len(ratio) else np.array([], bool)
    intra_bonds: set[tuple[int, int]] = {(int(a), int(b)) for a, b in zip(i[intra], j[intra])}
    bonds_of: dict[int, list[tuple[int, int]]] = {}
    for a, b in intra_bonds:
        bonds_of.setdefault(int(ids[a]), []).append((a, b))
    integrity_problems = []
    for m in molecules:
        members = np.where(ids == m["mol_id"])[0]
        if len(members) == 0:
            integrity_problems.append(f"mol {m['mol_id']} {m['name']}: no atoms")
            continue
        local = {int(g): k for k, g in enumerate(members)}
        bonds = [(local[a], local[b]) for a, b in bonds_of.get(int(m["mol_id"]), [])]
        smiles = m.get("smiles")
        if not smiles:
            continue
        ref = Chem.AddHs(Chem.MolFromSmiles(smiles))
        want = wl_certificate([a.GetSymbol() for a in ref.GetAtoms()],
                              [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in ref.GetBonds()])
        got = wl_certificate([atoms[int(g)].symbol for g in members], bonds)
        if len(members) != ref.GetNumAtoms() or got != want:
            integrity_problems.append(f"mol {m['mol_id']} {m['name']}: {len(bonds)} bonds found vs "
                                      f"{ref.GetNumBonds()} in {smiles}")
    if molecules:
        checks.append(CheckResult(name="molecule_integrity", status="fail" if integrity_problems else "pass",
                                  detail="; ".join(integrity_problems[:5]) if integrity_problems
                                  else f"{len(molecules)} molecules match their SMILES"))
    metrics["n_molecules"] = len(molecules)
    metrics["molecular_charge"] = int(sum(int(m.get("charge", 0)) for m in molecules))

    if len(ratio) and molecules:
        involved = (ids[i] >= 0) | (ids[j] >= 0)
        close = np.where((ratio < close_contact_scale) & involved)[0]
        close = [k for k in close if (int(i[k]), int(j[k])) not in intra_bonds]
        checks.append(CheckResult(
            name="close_contacts", status="warn" if close else "pass",
            detail=(f"{len(close)} contacts below {close_contact_scale} x covalent sum: " +
                    ", ".join(_pair_text(atoms, int(i[k]), int(j[k]), float(d[k])) for k in close[:5]))
            if close else "no close non-bonded contacts"))

    if molecules:
        sym = atoms.get_chemical_symbols()
        anions = surface_anions(s for s, m in zip(sym, ids) if m < 0)
        if anions:
            near = ratio < anion_contact_scale
            cationic = {m["mol_id"]: _positive_elements(m.get("smiles", "")) for m in molecules}  # [NH3+]: no anion
            hits = set()
            for a, b, dist in zip(i[near], j[near], d[near]):
                for x, y in ((a, b), (b, a)):  # molecule atom x, substrate atom y
                    if (ids[x] >= 0 and sym[x] in ANION_FORMERS and sym[x] not in cationic.get(int(ids[x]), ())
                            and ids[y] < 0 and sym[y] in anions):
                        hits.add((int(x), int(y), round(float(dist), 2)))
            hits = sorted(hits)
            checks.append(CheckResult(
                name="anion_contacts", status="warn" if hits else "pass",
                detail=(f"{len(hits)} molecule-anion contacts below {anion_contact_scale} x covalent sum "
                        "(like charges): " + ", ".join(f"{sym[a]}{a}-{sym[b]}{b} {d:.2f} A" for a, b, d in hits[:5]))
                if hits else "no electronegative molecule atom crowds a surface anion"))

    adsorbates = [m for m in molecules if m.get("role") == "adsorbate"]
    if adsorbates:
        checks.append(_check_binding(atoms, adsorbates, ids, r, binding_scale, detached_scale))

    if charge_neutrality:
        net, how = formal_charge(atoms, oxidation_states)
        metrics["net_charge"] = "n/a" if net is None else net
        checks.append(CheckResult(
            name="charge_neutrality", status="warn" if net is None else ("pass" if net == 0 else "fail"),
            detail=f"not assessed: {how}" if net is None else f"net formal charge {net:+d} ({how})"))

    kind = structure_kind(atoms)
    metrics["kind"] = kind
    if kind in ("slab", "wire", "nanocrystal") and any(atoms.pbc):
        gap = vacuum_gap(atoms)
        metrics["vacuum_gap"] = round(gap, 2)
        solvated = any(m.get("role") == "solvent" for m in molecules)
        ok = gap >= min_vacuum - 0.1
        status = "pass" if ok else ("warn" if solvated else "fail")
        checks.append(CheckResult(name="vacuum", status=status,
                                  detail=f"{gap:.1f} A between periodic images (minimum {min_vacuum})"
                                  + ("; solvent fills the gap" if solvated and not ok else "")))

    counts = Counter(atoms.get_chemical_symbols())
    metrics["composition"] = " ".join(f"{e}{n}" for e, n in sorted(counts.items()))
    if files:
        checks.append(check_roundtrip(atoms, files))
    passed = not any(c.status == "fail" for c in checks)
    return ValidationReport(structure_id=structure_id, passed=passed, checks=checks, metrics=metrics)
