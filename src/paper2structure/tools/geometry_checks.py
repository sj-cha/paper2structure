"""Deterministic structure validation (no LLM). Molecule-integrity checks follow PerovAgent's
bond-graph comparison."""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional, Sequence

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


_MOLECULE_REF: dict[str, Optional[tuple]] = {}


def _molecule_reference(smiles: str) -> Optional[tuple]:
    """(match template, reference positions) for `smiles`: the toolkit's RDKit conformer
    (Adsorbate.from_smiles: ETKDG + UFF) and, where MMFF has parameters, the same conformer
    MMFF-optimized (the force fields differ by up to 0.2 A on resonance groups such as phosphonates).
    The template drops bond orders, charges and aromaticity so it matches a bond graph read from
    coordinates. None when RDKit cannot embed it."""
    if smiles not in _MOLECULE_REF:
        from rdkit.Chem import AllChem

        from paper2structure.tools.adsorbate_builder import Adsorbate

        try:
            ref = Adsorbate.from_smiles(smiles, seed=0)
        except Exception:  # noqa: BLE001 - not embeddable: geometry not assessed
            _MOLECULE_REF[smiles] = None
            return None
        references = [ref.mol.GetConformer().GetPositions()]
        mmff = Chem.Mol(ref.mol)
        try:
            if AllChem.MMFFHasAllMoleculeParams(mmff) and AllChem.MMFFOptimizeMolecule(mmff, maxIters=2000) >= 0:
                references.append(mmff.GetConformer().GetPositions())
        except Exception:  # noqa: BLE001 - UFF reference only
            pass
        template = Chem.RWMol(ref.mol)
        for a in template.GetAtoms():
            a.SetFormalCharge(0)
            a.SetIsAromatic(False)
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(0)
        for b in template.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE)
            b.SetIsAromatic(False)
        _MOLECULE_REF[smiles] = (template.GetMol(), references)
    return _MOLECULE_REF[smiles]


def _angle(a: np.ndarray, center: np.ndarray, b: np.ndarray) -> float:
    u, v = a - center, b - center
    return float(np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0))))


def molecule_geometry(symbols: list[str], positions: np.ndarray, bonds, smiles: str, bond_tol: float = 0.12,
                      angle_tol: float = 20.0) -> list[str]:
    """Bond lengths and bond angles of a molecule against RDKit conformers of its SMILES (UFF and
    MMFF); the problems against the closer one (empty when it matches or RDKit cannot embed the
    SMILES). Position, orientation, atom order and torsions are free: turning a torsion leaves bond
    lengths and angles unchanged. The tolerances cover RDKit's own spread over seeds and methods
    (ETKDG, UFF, MMFF: up to 0.09 A and 15 degrees). `positions` must be unwrapped (whole
    molecule); `bonds` are local index pairs."""
    ref = _molecule_reference(smiles)
    if ref is None:
        return []
    template, references = ref
    found = [_geometry_problems(template, symbols, positions, bonds, smiles, r, bond_tol, angle_tol) for r in references]
    return min(found, key=len)


def _geometry_problems(template, symbols, positions, bonds, smiles, ref_pos, bond_tol, angle_tol) -> list[str]:
    rw = Chem.RWMol()
    for s in symbols:
        atom = Chem.Atom(s)
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    for a, b in bonds:
        rw.AddBond(int(a), int(b), Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    match = mol.GetSubstructMatch(template)
    if len(match) != template.GetNumAtoms() or len(symbols) != template.GetNumAtoms():
        return [f"its bond graph does not match {smiles}"]
    pos = np.asarray(positions, dtype=float)[list(match)]  # in template order
    sym = [a.GetSymbol() for a in template.GetAtoms()]
    # compare within symmetry classes of the bond graph, sorted: the match may pair equivalent atoms
    # (hydrogens of a CH3, the oxygens of a phosphonate) differently from the reference conformer
    rank = list(Chem.CanonicalRankAtoms(template, breakTies=False))
    bonds_by_class: dict[tuple, list] = {}
    for b in template.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        key = (tuple(sorted((rank[i], rank[j]))), "-".join(sorted((sym[i], sym[j]))))
        bonds_by_class.setdefault(key, []).append((np.linalg.norm(pos[i] - pos[j]), np.linalg.norm(ref_pos[i] - ref_pos[j])))
    angles_by_class: dict[tuple, list] = {}
    for center in template.GetAtoms():
        c, nbs = center.GetIdx(), [n.GetIdx() for n in center.GetNeighbors()]
        for k, i in enumerate(nbs):
            for j in nbs[k + 1:]:
                ends = sorted(((rank[i], sym[i]), (rank[j], sym[j])))
                key = (rank[c], tuple(ends), f"{ends[0][1]}-{sym[c]}-{ends[1][1]}")
                angles_by_class.setdefault(key, []).append(
                    (_angle(pos[i], pos[c], pos[j]), _angle(ref_pos[i], ref_pos[c], ref_pos[j])))
    problems = []
    for key, pairs in bonds_by_class.items():
        got, want = sorted(g for g, _ in pairs), sorted(w for _, w in pairs)
        worst = int(np.argmax(np.abs(np.subtract(got, want))))
        if abs(got[worst] - want[worst]) > bond_tol:
            problems.append(f"{key[1]} {got[worst]:.2f} A (RDKit {want[worst]:.2f})")
    for key, pairs in angles_by_class.items():
        got, want = sorted(g for g, _ in pairs), sorted(w for _, w in pairs)
        worst = int(np.argmax(np.abs(np.subtract(got, want))))
        if abs(got[worst] - want[worst]) > angle_tol:
            problems.append(f"{key[2]} {got[worst]:.0f} deg (RDKit {want[worst]:.0f})")
    return problems


def _unwrapped(atoms: Atoms, members: np.ndarray, bonds: list[tuple[int, int]]) -> np.ndarray:
    """Positions of a molecule's atoms made whole across periodic boundaries along its bonds."""
    pos = atoms.positions[members].copy()
    if not any(atoms.pbc):
        return pos
    adjacent: dict[int, list[int]] = {}
    for a, b in bonds:
        adjacent.setdefault(a, []).append(b)
        adjacent.setdefault(b, []).append(a)
    seen, stack = {0}, [0]
    while stack:
        a = stack.pop()
        for b in adjacent.get(a, []):
            if b not in seen:
                seen.add(b)
                pos[b] = pos[a] + atoms.get_distance(int(members[a]), int(members[b]), mic=True, vector=True)
                stack.append(b)
    return pos


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
    ids = mol_ids(atoms)
    molecular = int(sum(int(m.get("charge", 0)) for m in get_molecules(atoms)))
    counts = Counter(s for s, m in zip(atoms.get_chemical_symbols(), ids) if m < 0)
    given = {**parse_oxidation_states(str(atoms.info.get("p2s_oxidation_states", ""))), **(oxidation_states or {})}
    net, _, how = _oxidation_states(counts, given, molecular)
    return net, how


def _oxidation_states(counts: Counter, given: dict[str, int], molecular: int
                      ) -> tuple[Optional[int], dict[str, float], str]:
    """(net charge or None, oxidation state per substrate element, how) for formal_charge and the
    bond-valence check; a mixed-valence element gets its average state."""
    from itertools import product

    from pymatgen.core import Element

    given = dict(given)
    anions = [e for e in counts if e in ANION_FORMERS and e not in given]
    metals = [e for e in counts if e not in given and e not in ANION_FORMERS
              and (Element(e).is_metal or Element(e).is_metalloid)]
    others = [e for e in counts if e not in given and e not in anions and e not in metals]
    fixed = sum(given[e] * n for e, n in counts.items() if e in given)
    if not anions and not any(given.get(e, 0) < 0 for e in counts):  # metallic or covalent substrate
        net = molecular + fixed
        return net, dict(given), f"non-ionic substrate (charge {fixed:+d}); molecules {molecular:+d}"
    if others:
        return None, {}, f"cannot assign oxidation states to {', '.join(sorted(others))} in the substrate"
    for e in anions:
        neg = [q for q in Element(e).common_oxidation_states if q < 0]
        if not neg:
            return None, {}, f"no negative oxidation state for {e}"
        given[e] = min(neg)
        fixed += given[e] * counts[e]
    options = {}
    for e in metals:
        pos = [q for q in Element(e).common_oxidation_states if q > 0] or [q for q in Element(e).oxidation_states if q > 0][:3]
        if not pos:
            return None, {}, f"no positive oxidation state for {e}"
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
            return 0, {**states, e: q - net / counts[e]}, text + f"; neutral with mixed-valence {e}"
    return net, states, text


_BV_B = 0.31  # A, softness of the generic bond-valence parameters (O'Keeffe & Brese 1991)
_BV_REACH = 1.5  # x covalent sum: bonds counted in bond-valence sums (a 25%-stretched lattice still bonds)
_R0: dict[tuple[str, str], Optional[float]] = {}


def _r0(cation: str, anion: str) -> Optional[float]:
    """Generic bond-valence R0 of a cation-anion pair (pymatgen's O'Keeffe-Brese parameters)."""
    if (cation, anion) not in _R0:
        from pymatgen.analysis.bond_valence import BV_PARAMS
        from pymatgen.core import Element

        p, q = BV_PARAMS.get(Element(cation)), BV_PARAMS.get(Element(anion))
        if p is None or q is None:
            _R0[(cation, anion)] = None
        else:
            r1, r2, c1, c2 = p["r"], q["r"], p["c"], q["c"]
            _R0[(cation, anion)] = r1 + r2 - r1 * r2 * (np.sqrt(c1) - np.sqrt(c2)) ** 2 / (c1 * r1 + c2 * r2)
    return _R0[(cation, anion)]


def _depth(atoms: Atoms, sub: np.ndarray) -> np.ndarray:
    """Distance (A) of each atom below the substrate's surface: along the normal for slabs, below
    the convex hull for wires and particles, infinite for bulk."""
    from paper2structure.tools.adsorption_builder import _hull_depth
    from paper2structure.tools.structure_export import _axis_gaps

    kind = structure_kind(atoms)
    if kind == "bulk":
        return np.full(len(atoms), np.inf)
    pos = atoms.positions
    if kind == "slab":
        gaps = _axis_gaps(atoms)
        k = max(gaps, key=gaps.get) if gaps else 2
        cell = atoms.cell.array
        n = np.cross(cell[(k + 1) % 3], cell[(k + 2) % 3])
        h = pos @ (n / np.linalg.norm(n))
        return np.minimum(h - h[sub].min(), h[sub].max() - h)
    depth = np.zeros(len(atoms))
    depth[sub] = _hull_depth(pos[sub])
    return depth


def _loose_organics(sym: np.ndarray, i: np.ndarray, j: np.ndarray, ratio: np.ndarray, sub: np.ndarray,
                    bond_scale: float) -> list[tuple[int, int]]:
    """C-H and N-H bonds of the substrate in organic fragments that no heavy atom bonds to a metal:
    molecules left unregistered. Organic parts bonded into the framework (the linkers of a
    metal-organic framework, the ligands of a coordination polymer) belong to it. Fragments are
    joined through heavy atoms only, so a hydrogen crowding an anion does not bind its molecule."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from paper2structure.tools._geometry import cation_elements

    bonded = ratio < bond_scale
    metal = sub & np.isin(sym, list(cation_elements(sym[sub])))
    heavy = sub & ~metal & (sym != "H")
    hh = bonded & heavy[i] & heavy[j]
    _, label = connected_components(coo_matrix((np.ones(int(hh.sum())), (i[hh], j[hh])), shape=(len(sym),) * 2),
                                    directed=False)
    bound = set(label[i[bonded & heavy[i] & metal[j]]]) | set(label[j[bonded & heavy[j] & metal[i]]])
    out = []
    for a, b in zip(i[bonded], j[bonded]):
        if sub[a] and sub[b] and {sym[a], sym[b]} in ({"C", "H"}, {"N", "H"}):
            if label[a if sym[a] != "H" else b] not in bound:
                out.append((int(a), int(b)))
    return out


def _crystal_chemistry(atoms: Atoms, i: np.ndarray, j: np.ndarray, d: np.ndarray, ratio: np.ndarray, ids: np.ndarray,
                       given: dict[str, int], bvs_warn: float, bvs_fail: float, bond_share_warn: float,
                       like_contact_warn: float, like_contact_fail: float, min_depth: float = 4.0,
                       min_cn: int = 3, loose_organics: bool = False) -> list[CheckResult]:
    """Crystal-chemistry checks of an ionic substrate (one with anion formers), from the shared
    neighbor list (unordered pairs i < j):

    - like_charge_contacts: cation-cation pairs (metal atoms of the substrate or of different
      molecules) or substrate anion-anion pairs closer than like_contact_fail x the covalent sum
      fail (warn below like_contact_warn); covalent sums put real bonds near 1, and like charges
      never come that close.
    - bond_valence: bond-valence sums of the framework cations (those with the strongest average
      bond, valence / coordination, within 80% of the strongest) at interior ionic sites (at least
      min_cn anion neighbours, min_depth A below the surface) against their oxidation states:
      beyond bvs_fail fails, beyond bvs_warn warns (without charge-balanced oxidation states each
      element takes its most common one, and only beyond 100% fails). A single bond carrying
      more than bond_share_warn of its cation's valence warns, more than all of it fails. Large
      cations in big cages (Cs, Sr, Ba, Bi on perovskite A sites) are under-bonded with generic
      parameters and are not framework cations. Bonds to the electronegative atoms of registered
      molecules count (a carboxylate O on a metal), and organic parts bonded into the framework
      take part as anions (their O, N) or not at all (C, H); unregistered molecules in the substrate
      (loose_organics) leave it unassessed."""
    sym = np.array(atoms.get_chemical_symbols())
    sub = ids < 0
    anion_el = {e for e in set(sym[sub]) if e in ANION_FORMERS}
    if not anion_el:
        return []
    organic = {"C", "H"}
    is_anion = np.isin(sym, list(anion_el)) & sub
    is_cation = sub & ~is_anion & ~np.isin(sym, list(organic))
    metal = ~np.isin(sym, list(ANION_FORMERS | organic))  # a cation wherever it was registered
    checks = []

    same_molecule = (ids[i] >= 0) & (ids[i] == ids[j])
    like = ((metal[i] & metal[j] & ~same_molecule) | (is_anion[i] & is_anion[j])) & (ratio < like_contact_warn)
    if len(ratio):
        hits = np.where(like)[0]
        status = "fail" if (ratio[hits] < like_contact_fail).any() else ("warn" if len(hits) else "pass")
        hits = hits[np.argsort(ratio[hits])]
        checks.append(CheckResult(
            name="like_charge_contacts", status=status,
            detail=(f"{len(hits)} cation-cation or anion-anion pairs below {like_contact_warn} x covalent sum "
                    f"(fail below {like_contact_fail}): " + ", ".join(
                        f"{_pair_text(atoms, int(i[k]), int(j[k]), float(d[k]))} (x{ratio[k]:.2f})" for k in hits[:5]))
            if len(hits) else f"no like-charge pair below {like_contact_warn} x covalent sum"))

    if loose_organics:
        checks.append(CheckResult(
            name="bond_valence", status="warn",
            detail="not assessed: unregistered organic molecules are part of the substrate; register them with "
                   "add_molecule(..., role='guest') (or as adsorbates) so the inorganic framework can be checked"))
        return checks

    # cation -> anion bonds, both orientations of the unordered pairs; the anion may be a molecule's
    # electronegative atom bound to the framework cation (a carboxylate O on a metal)
    a = np.concatenate([i, j]); b = np.concatenate([j, i]); dist = np.concatenate([d, d])
    bond = is_cation[a] & (is_anion[b] | ((ids[b] >= 0) & np.isin(sym[b], list(ANION_FORMERS))))
    a, b, dist = a[bond], b[bond], dist[bond]
    r0 = np.array([_r0(x, y) or np.nan for x, y in zip(sym[a], sym[b])])
    s = np.where(np.isfinite(r0), np.exp((r0 - dist) / _BV_B), 0.0)
    cn = np.bincount(a, minlength=len(atoms))
    bvs = np.bincount(a, weights=s, minlength=len(atoms))
    strongest = np.zeros(len(atoms))
    np.maximum.at(strongest, a, s)
    ionic = is_cation & (cn >= min_cn)
    counts = Counter(sym[k] for k in np.where(is_anion | ionic)[0])
    molecular = int(sum(int(m.get("charge", 0)) for m in get_molecules(atoms)))
    net, states, how = _oxidation_states(counts, given, molecular)
    if net is None:
        checks.append(CheckResult(name="bond_valence", status="pass", detail=f"not assessed: {how}"))
        return checks
    if net != 0:  # no charge balance: the most common state of each element, not the least-charged mix
        from pymatgen.core import Element

        for e in states:
            common = [q for q in Element(e).common_oxidation_states if q > 0]
            if e not in given and states[e] > 0 and common:
                states[e] = common[0]
    deep = _depth(atoms, sub) >= min_depth
    sites = {e: np.where(ionic & deep & (sym == e))[0] for e in set(sym[ionic]) if states.get(e, 0) > 0}
    sites = {e: k for e, k in sites.items() if len(k)}
    if not sites:
        checks.append(CheckResult(name="bond_valence", status="pass",
                                  detail="not assessed: no interior ionic cation sites"))
        return checks
    strength = {e: states[e] / float(np.median(cn[k])) for e, k in sites.items()}
    top = max(strength.values())
    framework = sorted(e for e in sites if strength[e] >= 0.8 * top)
    neutral = net == 0
    fail_at = bvs_fail if neutral else max(bvs_fail, 1.0)
    status, lines = "pass", []
    for e in framework:
        dev = (float(np.median(bvs[sites[e]])) - states[e]) / states[e]
        level = "fail" if abs(dev) > fail_at else ("warn" if abs(dev) > bvs_warn else "pass")
        status = level if level == "fail" or (level == "warn" and status == "pass") else status
        lines.append(f"{e} {np.median(bvs[sites[e]]):.2f} vs {states[e]:+g} ({dev:+.0%}, {len(sites[e])} interior sites, "
                     f"CN {int(np.median(cn[sites[e]]))})")
    share = {e: float(strongest[k].max()) / states[e] for e, k in sites.items()}
    heavy = sorted((e for e in share if share[e] > bond_share_warn), key=lambda e: -share[e])
    if any(share[e] > 1.0 for e in heavy):
        status = "fail"  # one bond stronger than the whole valence of its cation
    elif heavy and status == "pass":
        status = "warn"
    for e in heavy:
        k = sites[e][int(np.argmax(strongest[sites[e]]))]
        partner = b[a == k][int(np.argmax(s[a == k]))]
        lines.append(f"one {e}-{sym[partner]} bond of {atoms.get_distance(int(k), int(partner), mic=True):.2f} A "
                     f"carries {share[e]:.0%} of {e}'s valence (too short)")
    detail = "bond-valence sums of the framework cations: " + "; ".join(lines)
    if not neutral:
        detail += f" (no charge-balanced oxidation states: {how}; fails only beyond {fail_at:.0%})"
    checks.append(CheckResult(name="bond_valence", status=status, detail=detail))
    return checks


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
                       bvs_warn: float = 0.30, bvs_fail: float = 0.40, bond_share_warn: float = 0.45,
                       like_contact_warn: float = 0.8, like_contact_fail: float = 0.7,
                       molecule_bond_tol: float = 0.12, molecule_angle_tol: float = 20.0,
                       files: Optional[dict[str, str]] = None, claim_checks: Sequence = (),
                       system_ids: Optional[Sequence[str]] = None) -> ValidationReport:
    """Geometric and chemical sanity checks. Pairs closer than overlap_scale x the covalent-radius
    sum fail; contacts involving a molecule closer than close_contact_scale x the sum (other than
    the molecule's own bonds) warn, as do electronegative molecule atoms (O, N, S, halogens) closer
    than anion_contact_scale x the sum to a substrate anion; each molecule's bond graph must match
    its SMILES. Each adsorbate must touch the substrate (closest contact within binding_scale x the
    covalent sum; beyond detached_scale fails) and, on metal/metalloid compounds, an
    electronegative donor atom must not touch an electronegative substrate atom (surface_anions). With charge_neutrality the net formal charge must be zero
    (formal_charge, using `oxidation_states` where given). Ionic substrates also get the
    crystal-chemistry checks of _crystal_chemistry: like-charge contacts, and bond-valence sums of
    the framework cations (bvs_warn / bvs_fail, bond_share_warn). Molecules must have the geometry
    of the RDKit conformer of their SMILES (bond lengths within molecule_bond_tol A, angles within
    molecule_angle_tol degrees); C-H and N-H bonds in the substrate (molecules not registered) and
    substrate atoms without any neighbour fail. `claim_checks` are the plan's tests of the paper's
    claims (evidence_checks.run_claim_checks), for the structure's `system_ids`."""
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
    # one neighbor list for overlaps, close contacts, molecule bonds, like-charge contacts and bond valences
    i, j, D = neighbor_pairs(atoms, max(1.0, bond_scale, anion_contact_scale, _BV_REACH) * r)
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
    integrity_problems, geometry_problems = [], []
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
            continue
        off = molecule_geometry([atoms[int(g)].symbol for g in members], _unwrapped(atoms, members, bonds), bonds,
                                smiles, molecule_bond_tol, molecule_angle_tol)
        if off:
            geometry_problems.append(f"mol {m['mol_id']} {m['name']}: {', '.join(off[:3])}")
    if molecules:
        checks.append(CheckResult(name="molecule_integrity", status="fail" if integrity_problems else "pass",
                                  detail="; ".join(integrity_problems[:5]) if integrity_problems
                                  else f"{len(molecules)} molecules match their SMILES"))
        checks.append(CheckResult(
            name="molecule_geometry", status="fail" if geometry_problems else "pass",
            detail=(f"{len(geometry_problems)} molecules differ from the RDKit conformer of their SMILES (build them "
                    "with Adsorbate.from_smiles, then move them rigidly or turn torsions): "
                    + "; ".join(geometry_problems[:4]))
            if geometry_problems else "bond lengths and angles match the RDKit conformers"))

    sub = ids < 0
    sym = atoms.get_chemical_symbols()
    organic = _loose_organics(np.array(sym), i, j, ratio, sub, bond_scale)
    if organic:
        a, b = organic[0]
        checks.append(CheckResult(
            name="unregistered_molecules", status="fail",
            detail=f"{len(organic)} C-H/N-H bonds in substrate fragments not bonded to a metal, e.g. "
                   f"{_pair_text(atoms, a, b, float(atoms.get_distance(a, b, mic=True)))}: molecules must be RDKit "
                   "conformers registered with place_adsorbates, add_solvent, add_molecule (role 'adsorbate', "
                   "'solvent' or 'guest') or fill_cavities; organic parts bonded into the framework (linkers of a "
                   "metal-organic framework) may stay in it"))
    if len(atoms) > 1:
        near = ratio < _BV_REACH
        linked = np.bincount(np.concatenate([i[near], j[near]]), minlength=len(atoms))
        lonely = np.where(sub & (linked == 0))[0]
        checks.append(CheckResult(
            name="isolated_atoms", status="fail" if len(lonely) else "pass",
            detail=(f"{len(lonely)} substrate atoms with no neighbour within {_BV_REACH} x covalent sum, e.g. "
                    + ", ".join(f"{sym[k]}{k}" for k in lonely[:5])) if len(lonely)
            else "every substrate atom has a neighbour"))
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

    given = {**parse_oxidation_states(str(atoms.info.get("p2s_oxidation_states", ""))), **(oxidation_states or {})}
    checks.extend(_crystal_chemistry(atoms, i, j, d, ratio, ids, given, bvs_warn, bvs_fail, bond_share_warn,
                                     like_contact_warn, like_contact_fail, loose_organics=bool(organic)))

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

    if claim_checks:
        from paper2structure.tools.evidence_checks import run_claim_checks

        checks.extend(run_claim_checks(atoms, claim_checks, system_ids))

    counts = Counter(atoms.get_chemical_symbols())
    metrics["composition"] = " ".join(f"{e}{n}" for e, n in sorted(counts.items()))
    if files:
        checks.append(check_roundtrip(atoms, files))
    passed = not any(c.status == "fail" for c in checks)
    return ValidationReport(structure_id=structure_id, passed=passed, checks=checks, metrics=metrics)
