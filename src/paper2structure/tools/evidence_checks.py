"""The paper's claims as deterministic tests of a built structure: the planner turns each claim of
the summary and the characterization reports into a typed check (tools.schemas.ClaimCheck) and
the validation gate runs them here. Each result names the claim, the built value and the evidence."""
from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import spglib
from ase import Atoms
from ase.data import atomic_numbers, vdw_radii
from ase.neighborlist import neighbor_list
from pymatgen.core import Composition
from rdkit import Chem
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from paper2structure.tools._geometry import get_molecules, mol_ids, rcov
from paper2structure.tools._symmetry import space_group_number
from paper2structure.tools.schemas import (BondAngleCheck, BondLengthCheck, CheckResult, ClaimCheck,
                                           CompositionCheck, CoordinationCheck, HydrogenBondCheck, LatticeCheck,
                                           LayerCheck, SpaceGroupCheck)
from paper2structure.tools.structure_export import structure_kind, substrate_depth

BOND_SCALE = 1.25  # x covalent-radius sum: the first coordination shell
INTERIOR_DEPTH = 4.0  # A below the surface of slabs and particles
LAYER_TOL = 1.0  # A along the stacking axis within one plane


def _vdw(symbol: str) -> float:
    return float(vdw_radii[atomic_numbers[symbol]])


def _max_bond(a: str, b: str, given: Optional[float]) -> float:
    return float(given) if given else BOND_SCALE * float(rcov([a, b]).sum())


def _subset(atoms: Atoms, scope: str) -> Atoms:
    return atoms[mol_ids(atoms) < 0] if scope == "framework" else atoms


def _canonical(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def _range(values) -> str:
    return f"{min(values):.2f}-{max(values):.2f}" if len(values) else "none"


def _lattice(atoms: Atoms, c: LatticeCheck) -> tuple[str, str]:
    par = atoms.cell.cellpar()
    built = "a={:.4f}, b={:.4f}, c={:.4f} A, angles {:.2f}/{:.2f}/{:.2f}".format(*par)
    off = []
    for k, name in enumerate("abc"):
        want = getattr(c, name)
        if want is None:
            continue
        m = max(1, int(round(par[k] / want)))
        if abs(par[k] / m - want) > c.tolerance * want:
            off.append(f"{name} {par[k]:.4f} vs {want}" + (f" (x{m})" if m > 1 else ""))
    for k, name in enumerate(("alpha", "beta", "gamma")):
        want = getattr(c, name)
        if want is not None and abs(par[3 + k] - want) > c.angle_tolerance:
            off.append(f"{name} {par[3 + k]:.2f} vs {want}")
    return ("fail" if off else "pass"), (f"{'; '.join(off)} (built {built})" if off else f"built {built}")


def _space_group(atoms: Atoms, c: SpaceGroupCheck) -> tuple[str, str]:
    want = space_group_number(c.symbol)
    sub = _subset(atoms, c.scope)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = spglib.get_symmetry_dataset((sub.cell.array, sub.get_scaled_positions(), sub.numbers),
                                           symprec=c.symprec)
    got, name = (int(data.number), data.international) if data is not None else (1, "P1")
    scope = " (framework without molecules)" if c.scope == "framework" else ""
    detail = f"built {name} (#{got}) at symprec {c.symprec} A{scope}, paper {c.symbol} (#{want})"
    return ("pass" if got == want else "fail"), detail


def _composition(atoms: Atoms, c: CompositionCheck) -> tuple[str, str]:
    sub = _subset(atoms, c.scope)
    built = Composition(sub.get_chemical_formula()).fractional_composition
    want = Composition(c.formula).fractional_composition
    elements = set(built.elements) | set(want.elements)
    diff = max(abs(built[e] - want[e]) for e in elements)
    detail = (f"built {Composition(sub.get_chemical_formula()).reduced_formula}, paper {Composition(c.formula).reduced_formula} "
              f"(largest atomic-fraction difference {diff:.3f})")
    return ("pass" if diff <= c.tolerance else "fail"), detail


def _pairs(atoms: Atoms, a: str, b: str, cutoff: float):
    """(i, j, d, D) for the a-b pairs within cutoff, i of element a (each unordered pair once when a == b)."""
    if len(atoms) < 2:
        return np.array([], int), np.array([], int), np.array([]), np.zeros((0, 3))
    i, j, d, D = neighbor_list("ijdD", atoms, cutoff)
    sym = np.array(atoms.get_chemical_symbols())
    keep = (sym[i] == a) & (sym[j] == b)
    if a == b:
        keep &= i < j
    return i[keep], j[keep], d[keep], D[keep]


def _bond_length(atoms: Atoms, c: BondLengthCheck) -> tuple[str, str]:
    a, b = c.elements
    _, _, d, _ = _pairs(atoms, a, b, c.max + 0.5)
    if not len(d):
        return "fail", f"no {a}-{b} pair within {c.max + 0.5:.2f} A"
    out = d[(d < c.min - c.tolerance) | (d > c.max + c.tolerance)]
    detail = f"built {a}-{b} {_range(d)} A over {len(d)} pairs, paper {c.min}-{c.max} A (+-{c.tolerance})"
    return ("fail" if len(out) else "pass"), (f"{len(out)} pairs outside: " if len(out) else "") + detail


def _bond_angle(atoms: Atoms, c: BondAngleCheck) -> tuple[str, str]:
    a, v, b = c.elements
    sym = np.array(atoms.get_chemical_symbols())
    cut = max(_max_bond(a, v, c.max_bond), _max_bond(b, v, c.max_bond))
    i, j, d, D = neighbor_list("ijdD", atoms, cut)
    angles = []
    for center in np.where(sym == v)[0]:
        m = i == center
        jj, dd, DD = j[m], d[m], D[m]
        ok_a = (sym[jj] == a) & (dd <= _max_bond(a, v, c.max_bond))
        ok_b = (sym[jj] == b) & (dd <= _max_bond(b, v, c.max_bond))
        for x in np.where(ok_a)[0]:
            for y in np.where(ok_b)[0]:
                if x == y or (a == b and y <= x):
                    continue
                u, w = DD[x], DD[y]
                angles.append(float(np.degrees(np.arccos(np.clip(u @ w / (np.linalg.norm(u) * np.linalg.norm(w)), -1, 1)))))
    if not angles:
        return "fail", f"no {a}-{v}-{b} angle (no {v} with bonded {a} and {b})"
    angles = np.array(angles)
    out = angles[(angles < c.min - c.tolerance) | (angles > c.max + c.tolerance)]
    detail = (f"built {a}-{v}-{b} {angles.min():.1f}-{angles.max():.1f} deg over {len(angles)} angles, "
              f"paper {c.min}-{c.max} (+-{c.tolerance})")
    return ("fail" if len(out) else "pass"), (f"{len(out)} angles outside: " if len(out) else "") + detail


def _interior(atoms: Atoms) -> np.ndarray:
    sub = np.where(mol_ids(atoms) < 0)[0]
    return substrate_depth(atoms, sub) >= INTERIOR_DEPTH


def _coordination(atoms: Atoms, c: CoordinationCheck) -> tuple[str, str]:
    sym = np.array(atoms.get_chemical_symbols())
    ids = mol_ids(atoms)
    centers = np.where((sym == c.center) & (ids < 0) & _interior(atoms))[0]
    if not len(centers):
        return "fail", f"no interior framework {c.center}"
    i, j, d = neighbor_list("ijd", atoms, _max_bond(c.center, c.neighbor, c.max_bond))
    cn = np.array([int(np.sum((i == k) & (sym[j] == c.neighbor))) for k in centers])
    values, counts = np.unique(cn, return_counts=True)
    dist = ", ".join(f"CN {v}: {n}" for v, n in zip(values, counts))
    detail = f"{c.center} by {c.neighbor} within {_max_bond(c.center, c.neighbor, c.max_bond):.2f} A: {dist}; paper {c.number}"
    return ("pass" if np.all(cn == c.number) else "fail"), detail


def _hydrogen_bonds(atoms: Atoms, c: HydrogenBondCheck) -> tuple[str, str]:
    sym = np.array(atoms.get_chemical_symbols())
    ids = mol_ids(atoms)
    keep = {int(m["mol_id"]) for m in get_molecules(atoms)
            if c.smiles is None or _canonical(m.get("smiles", "")) == _canonical(c.smiles)}
    donors = [k for k in np.where((sym == c.donor) & (ids >= 0))[0] if int(ids[k]) in keep]
    cut = max(c.max_distance or _vdw("H") + _vdw(a) - 0.1 for a in c.acceptors)
    i, j, d, D = neighbor_list("ijdD", atoms, max(cut, 1.2))
    counts = []
    for n in donors:
        hs = j[(i == n) & (sym[j] == "H") & (d < 1.2) & (ids[j] == ids[n])]
        if not len(hs):
            continue
        bonded = 0
        for h in hs:
            vh = D[(i == n) & (j == h)][0]
            m = (i == h) & np.isin(sym[j], c.acceptors) & (ids[j] != ids[h])
            for x, dx, Dx in zip(j[m], d[m], D[m]):
                limit = c.max_distance or _vdw("H") + _vdw(sym[x]) - 0.1
                cos = float(-vh @ Dx / (np.linalg.norm(vh) * np.linalg.norm(Dx)))
                if dx <= limit and np.degrees(np.arccos(np.clip(cos, -1, 1))) >= c.min_angle:
                    bonded += 1
                    break
        counts.append(bonded)
    which = f"{c.donor}-H...{'/'.join(c.acceptors)}" + (f" in {c.smiles}" if c.smiles else "")
    if not counts:
        return "fail", f"no {c.donor} donor with hydrogens in the molecules ({which})"
    values, n = np.unique(counts, return_counts=True)
    dist = ", ".join(f"{v} H-bonds: {k}" for v, k in zip(values, n))
    detail = f"{which} per donor: {dist} ({len(counts)} donors); paper at least {c.per_donor} per donor"
    return ("pass" if min(counts) >= c.per_donor else "fail"), detail


def _layers(atoms: Atoms, c: LayerCheck) -> tuple[str, str]:
    sub = _subset(atoms, "framework")
    k = "abc".index(c.axis)
    if len(sub) < 2:
        return "fail", "no framework"
    i, j, S = neighbor_list("ijS", sub, BOND_SCALE * rcov(sub.get_chemical_symbols()))  # within 1.25 x r_i + r_j
    g = coo_matrix((np.ones(len(i)), (i, j)), shape=(len(sub), len(sub)))
    n_blocks, label = connected_components(g, directed=False)
    normal = sub.cell.reciprocal()[k] / np.linalg.norm(sub.cell.reciprocal()[k])
    frac = sub.get_scaled_positions()
    sym = np.array(sub.get_chemical_symbols())
    nbrs: dict[int, list[tuple[int, np.ndarray]]] = {}
    for a, b, s in zip(i, j, S):
        nbrs.setdefault(int(a), []).append((int(b), s))
    found = []
    for blk in range(n_blocks):
        members = np.where(label == blk)[0]
        if not np.any(sym[members] == c.element):
            continue
        # unwrap the block: an atom reached again through a different image along the axis means
        # the block is bonded to its own periodic image, i.e. not a layer
        image = {int(members[0]): np.zeros(3, int)}
        queue, periodic = [int(members[0])], False
        while queue and not periodic:
            a = queue.pop()
            for b, s in nbrs.get(a, []):
                want = image[a] + s
                if b not in image:
                    image[b] = want
                    queue.append(b)
                elif want[k] != image[b][k]:
                    periodic = True
                    break
        if periodic:
            found.append("connected through the cell")
            continue
        heights = np.array([(frac[m] + image[int(m)]) @ sub.cell.array @ normal
                            for m in members if sym[m] == c.element])
        found.append(1 + int(np.sum(np.diff(np.sort(heights)) > LAYER_TOL)))
    if not found:
        return "fail", f"no framework block contains {c.element}"
    ok = all(n == c.count for n in found)
    detail = f"{c.element} planes along {c.axis} per framework block: {found}; paper {c.count}"
    return ("pass" if ok else "fail"), detail


_RUN = {"lattice": _lattice, "space_group": _space_group, "composition": _composition, "bond_length": _bond_length,
        "bond_angle": _bond_angle, "coordination": _coordination, "hydrogen_bonds": _hydrogen_bonds,
        "layers": _layers}
BULK_ONLY = {"lattice", "space_group", "composition"}  # slabs and particles have their own cells and terminations


def applies(check: ClaimCheck, system_ids: Optional[Sequence[str]]) -> bool:
    """A check scoped to systems applies to a structure that models one of them (or declares none)."""
    return not check.system_ids or not system_ids or bool(set(check.system_ids) & set(system_ids))


def run_claim_checks(atoms: Atoms, checks: Sequence[ClaimCheck],
                     system_ids: Optional[Sequence[str]] = None) -> list[CheckResult]:
    """One CheckResult per applicable check, named "claim <id>"; the detail gives the claim, the built
    value, the paper's value and the evidence. A failing check of severity "warn" warns."""
    kind = structure_kind(atoms)
    results = []
    for c in checks:
        if not applies(c, system_ids):
            continue
        name = f"claim {c.id}"
        head = f"{c.claim} [{', '.join(c.evidence_ids + c.claim_ids) or 'no evidence cited'}]: "
        if c.kind in BULK_ONLY and kind != "bulk":
            results.append(CheckResult(name=name, status="skip", detail=head + f"not checked on a {kind}"))
            continue
        try:
            status, detail = _RUN[c.kind](atoms, c)
        except Exception as e:  # noqa: BLE001 - a check that cannot run must not hide the others
            status, detail = "fail", f"could not be checked: {type(e).__name__}: {e}"
        if status == "fail" and c.severity == "warn":
            status = "warn"
        results.append(CheckResult(name=name, status=status, detail=head + detail))
    return results
