"""Surface sites and adsorbate placement. Placement generalizes PerovKit's coarse-to-fine rotational
search (https://github.com/sj-cha/PerovKit) from perovskite X/A sites to arbitrary surface sites."""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from ase import Atoms
from ase.geometry import get_distances

from paper2structure.tools._geometry import (ANION_FORMERS, ROLE_ADSORBATE, ROLE_SOLVENT, PeriodicTree,
                                             coordination_pairs, default_seed, ensure_bookkeeping,
                                             farthest_point_sampling, get_molecules, mol_ids, neighbor_pairs, rcov,
                                             rotation_about_axis, rotation_from_u_to_v, set_molecules, signed_angle,
                                             substrate_mask, surface_anions, unit)
from paper2structure.tools.structure_export import structure_kind, vacuum_gap

Z = np.array([0.0, 0.0, 1.0])
ANION_CONTACT_SCALE = 1.3  # adsorbate O/N/S/halogen - substrate anion pairs count as clashes below this x radii


@dataclass
class Site:
    """A candidate binding site on a surface or nanocrystal."""
    kind: str                      # "top", "bridge" or "hollow"
    indices: tuple[int, ...]       # substrate atoms forming the site
    symbols: tuple[str, ...]
    position: np.ndarray           # centroid of the site atoms (A)
    normal: np.ndarray             # outward unit normal
    coordination: tuple[int, ...]  # coordination numbers of the site atoms
    subtype: str = ""              # hollow: "fcc"/"hcp" (3-fold) or "4-fold"
    offsets: np.ndarray = field(default_factory=lambda: np.zeros((1, 3)), repr=False)

    @property
    def label(self) -> str:
        sub = f"-{self.subtype}" if self.subtype else ""
        cn = "/".join(str(c) for c in self.coordination)
        return f"{self.kind}{sub}:{'-'.join(self.symbols)}(CN {cn})"

    def __str__(self) -> str:
        p = ", ".join(f"{x:.2f}" for x in self.position)
        n = ", ".join(f"{x:.2f}" for x in self.normal)
        return f"Site {self.label} atoms {list(self.indices)} at ({p}) normal ({n})"


def _environment(atoms: Atoms, idx: np.ndarray, cutoff_scale: float):
    """Coordination numbers and outward normals (minus the bond-vector sum) of atoms[idx],
    counting only neighbours among atoms[idx]."""
    sub = atoms[idx]
    i, j, D = coordination_pairs(sub, cutoff_scale)
    n = len(sub)
    cn = np.bincount(i, minlength=n)
    S = np.zeros((n, 3))
    np.add.at(S, i, D)
    norms = np.linalg.norm(S, axis=1)
    normals = np.where(norms[:, None] > 1e-6, -S / np.maximum(norms, 1e-12)[:, None], 0.0)
    if np.any(norms <= 1e-6):
        center = sub.positions.mean(axis=0)
        for k in np.where(norms <= 1e-6)[0]:
            r = sub.positions[k] - center
            normals[k] = r / np.linalg.norm(r) if np.linalg.norm(r) > 1e-6 else Z
    return cn, normals, (i, j, D)


def _reference_cn(atoms: Atoms, symbols: list[str], cn: np.ndarray) -> dict[str, int]:
    ref = atoms.info.get("p2s_bulk_cn")
    if ref:
        ref = json.loads(ref) if isinstance(ref, str) else dict(ref)
        return {s: int(ref.get(s, 0)) for s in set(symbols)}
    out: dict[str, int] = {}
    for s, c in zip(symbols, cn):
        out[s] = max(out.get(s, 0), int(c))
    return out


def _hull_depth(positions: np.ndarray) -> np.ndarray:
    """Distance (A) of each point below the convex hull of all points (0 on the hull)."""
    from scipy.spatial import ConvexHull

    try:
        eq = ConvexHull(positions).equations
    except Exception:  # noqa: BLE001 - flat or tiny particle: every atom is on the outside
        return np.zeros(len(positions))
    # inside a convex hull every facet's signed distance n.x + d is <= 0; the nearest facet is the largest
    return np.maximum(-(positions @ eq[:, :3].T + eq[:, 3]).max(axis=1), 0.0)


def surface_atoms(atoms: Atoms, side: str = "auto", cutoff_scale: float = 1.2, layer_tol: float = 0.5) -> list[int]:
    """Indices of exposed substrate atoms: atoms with fewer neighbours than the same element in the
    bulk (broken bonds), plus the outermost layer - the atoms within `layer_tol` A of the top (or
    bottom) of a slab or of a particle's convex hull - so that van der Waals basal planes (MoS2,
    graphite, hydroxide layers), which cleave without breaking a bond, are exposed too. The
    tolerance keeps fully coordinated atoms of polar bilayers (0.9 A below a zinc-blende (111)
    top) out.

    side: "top" or "bottom" for slabs, "all" for nanocrystals; "auto" picks "top" for slabs and
    "all" otherwise. Atoms with another substrate atom stacked right above them along the slab
    normal are never exposed. Adsorbate and solvent atoms are ignored.
    """
    idx = np.where(substrate_mask(atoms))[0]
    if len(idx) == 0:
        return []
    cn, normals, _ = _environment(atoms, idx, cutoff_scale)
    symbols = [atoms[i].symbol for i in idx]
    ref = _reference_cn(atoms, symbols, cn)
    under = np.array([c < ref[s] for s, c in zip(symbols, cn)])
    if side == "auto":
        side = "top" if structure_kind(atoms) == "slab" else "all"
    if side in ("top", "bottom"):
        sign = 1.0 if side == "top" else -1.0
        h = sign * atoms.positions[idx, 2]
        mid = 0.5 * (h.max() + h.min())
        keep = (under & (sign * normals[:, 2] > 0.3) & (h > mid)) | (h >= h.max() - layer_tol)
        keep &= ~_shadowed(atoms, idx, keep, sign)
    elif side == "all":
        keep = under | (_hull_depth(atoms.positions[idx]) <= layer_tol)
    else:
        raise ValueError(f"side must be 'auto', 'top', 'bottom' or 'all'; got {side!r}")
    return [int(i) for i in idx[keep]]


def _in_plane_pbc(atoms: Atoms) -> list[bool]:
    """Periodicity for slab geometry tests: never across the vacuum, where the image of the slab
    bottom would otherwise appear just above the top surface."""
    return [bool(atoms.pbc[0]), bool(atoms.pbc[1]), False]


def _shadowed(atoms: Atoms, idx: np.ndarray, cand: np.ndarray, sign: float) -> np.ndarray:
    """Candidates with another substrate atom stacked directly over them (along the slab normal)."""
    out = np.zeros(len(idx), dtype=bool)
    pos = atoms.positions[idx]
    r = rcov(atoms.get_chemical_symbols())[idx]
    for k in np.where(cand)[0]:
        D, _ = get_distances(pos[k:k + 1], pos, cell=atoms.cell, pbc=_in_plane_pbc(atoms))
        D = D[0]
        above = sign * D[:, 2] > 0.5
        lateral = np.linalg.norm(D[:, :2], axis=1) < 0.6 * (r[k] + r)
        out[k] = bool(np.any(above & lateral))
    return out


def _wrap_key(atoms: Atoms, pos: np.ndarray, kind: str) -> tuple:
    """Hashable position key that identifies periodic images of the same site."""
    if atoms.cell.rank < 3:
        return (kind,) + tuple(float(x) for x in np.round(pos, 1))
    frac = np.round(np.linalg.solve(atoms.cell.array.T, pos), 2)
    for ax in range(3):
        if atoms.pbc[ax]:
            frac[ax] = round(float(frac[ax]) % 1.0, 2) % 1.0
    return (kind,) + tuple(float(x) for x in frac)


def surface_sites(atoms: Atoms, kinds: Sequence[str] = ("top",), elements: Optional[Sequence[str]] = None,
                  side: str = "auto", max_pair_distance: Optional[float] = None,
                  cutoff_scale: float = 1.2) -> list[Site]:
    """Binding sites formed by exposed substrate atoms.

    Args:
        kinds: any of "top", "bridge", "hollow" (3-fold "fcc"/"hcp" and 4-fold sites).
        elements: only build sites from these elements (e.g. ["Ti"] for Ti-Ti bridges).
        side: see surface_atoms.
        max_pair_distance: maximum distance (A) between two atoms of a bridge/hollow site; default
            1.25 x the covalent-radius sum. Raise it for sites across non-bonded atoms: e.g. ~4 A
            for Ti5c-Ti5c on anatase (101), ~4.3 A for neighbouring surface Br on a CsPbBr3
            PbBr2-terminated (001) face (a replace-mode bridge for an adsorbate taking two halide sites).
    Slab sites point along +z (top) or -z (bottom); nanocrystal sites use local outward normals.
    """
    surf = surface_atoms(atoms, side=side, cutoff_scale=cutoff_scale)
    if elements:
        surf = [i for i in surf if atoms[i].symbol in set(elements)]
    if not surf:
        return []
    sub_all = np.where(substrate_mask(atoms))[0]
    cn_all, normals_all, _ = _environment(atoms, sub_all, cutoff_scale)
    cn_of = {int(o): int(c) for o, c in zip(sub_all, cn_all)}
    normal_of = {int(o): n for o, n in zip(sub_all, normals_all)}
    resolved = side if side != "auto" else ("top" if structure_kind(atoms) == "slab" else "all")
    fixed_normal = Z if resolved == "top" else (-Z if resolved == "bottom" else None)
    symbols = atoms.get_chemical_symbols()

    surf = np.asarray(surf, dtype=int)
    sub = atoms[surf]
    i, j, D = neighbor_pairs(sub, float(max_pair_distance) if max_pair_distance is not None
                             else 1.25 * rcov(sub.get_chemical_symbols()))
    adj: dict[int, list[tuple[int, np.ndarray]]] = {k: [] for k in range(len(sub))}
    for a, b, d in zip(i, j, D):
        adj[int(a)].append((int(b), d))

    sites: list[Site] = []
    seen: set = set()

    def add(kind: str, members: list[int], vecs: list[np.ndarray], subtype: str = "") -> None:
        base = sub.positions[members[0]]
        pts = np.array([base + v for v in vecs])
        center = pts.mean(axis=0)
        key = _wrap_key(atoms, center, kind + subtype)
        if key in seen:
            return
        seen.add(key)
        orig = tuple(int(surf[m]) for m in members)
        normal = fixed_normal if fixed_normal is not None else unit(sum(normal_of[o] for o in orig))
        sites.append(Site(kind=kind, indices=orig, symbols=tuple(symbols[o] for o in orig),
                          position=center, normal=np.array(normal, dtype=float),
                          coordination=tuple(cn_of[o] for o in orig), subtype=subtype,
                          offsets=pts - center))

    if "top" in kinds:
        for k in range(len(sub)):
            add("top", [k], [np.zeros(3)])
    if "bridge" in kinds:
        for a in adj:
            for b, d in adj[a]:
                add("bridge", [a, b], [np.zeros(3), d])
    if "hollow" in kinds:
        tree = PeriodicTree(sub.positions, np.ones(len(sub)), sub.cell.array, sub.pbc)
        for a in adj:
            nbrs = adj[a]
            for x in range(len(nbrs)):
                for y in range(x + 1, len(nbrs)):
                    (b, dab), (c, dac) = nbrs[x], nbrs[y]
                    cosang = float(np.dot(dab, dac) / (np.linalg.norm(dab) * np.linalg.norm(dac)))
                    closes = any(cc == c and np.allclose(dbc, dac - dab, atol=0.2) for cc, dbc in adj[b])
                    if closes and cosang > 0.0:  # acute corner of a closed triangle
                        sub_t = _hollow_subtype(atoms, sub.positions[a] + (dab + dac) / 3.0, fixed_normal)
                        add("hollow", [a, b, c], [np.zeros(3), dab, dac], sub_t)
                    elif not closes and abs(cosang) < 0.35:
                        corner = sub.positions[a] + dab + dac
                        dist, hit = tree.tree.query(corner)
                        if dist < 0.4:
                            d_idx = int(tree.index[hit])
                            add("hollow", [a, b, c, d_idx], [np.zeros(3), dab, dac, dab + dac], "4-fold")
    order = {"top": 0, "bridge": 1, "hollow": 2}
    sites.sort(key=lambda s: (order.get(s.kind, 9), -float(s.position[2]), s.indices))
    return sites


def _hollow_subtype(atoms: Atoms, center: np.ndarray, normal: Optional[np.ndarray]) -> str:
    """"hcp" if a substrate atom sits right below a slab 3-fold hollow, else "fcc"."""
    if normal is None:
        return ""
    idx = np.where(substrate_mask(atoms))[0]
    D, _ = get_distances(center[None, :], atoms.positions[idx], cell=atoms.cell, pbc=_in_plane_pbc(atoms))
    D = D[0]
    depth = -(D @ normal)
    lateral = np.linalg.norm(D - np.outer(D @ normal, normal), axis=1)
    return "hcp" if np.any((depth > 0.5) & (depth < 3.5) & (lateral < 0.7)) else "fcc"


def unique_sites(sites: Sequence[Site]) -> list[Site]:
    """One representative per distinct site type (kind, subtype, elements, coordination)."""
    seen: set = set()
    out = []
    for s in sites:
        key = (s.kind, s.subtype, tuple(sorted(s.symbols)), tuple(sorted(s.coordination)))
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _anchor_position(site: Site, local: np.ndarray, binding: list[int], symbols: list[str], mode: str,
                     bond_length: Optional[float], offset: float) -> np.ndarray:
    n = site.normal
    if mode == "replace":
        return site.position + n * offset
    d = float(bond_length) if bond_length else float(rcov([symbols[b] for b in binding]).mean()
                                                       + rcov(site.symbols).mean())
    lateral_off = site.offsets - np.outer(site.offsets @ n, n)
    rho = np.linalg.norm(lateral_off, axis=1)
    if len(binding) == 1:
        lat = float(rho.mean())
    else:
        span_l = float(np.linalg.norm(local[binding[1]] - local[binding[0]]))
        if site.kind == "bridge":
            lat = abs(float(np.linalg.norm(lateral_off[1] - lateral_off[0])) - span_l) / 2.0
        elif site.kind == "top":
            lat = span_l / 2.0
        else:
            lat = max(0.0, float(rho.mean()) - span_l / 2.0)
    h = math.sqrt(max(d * d - lat * lat, (0.5 * d) ** 2))
    return site.position + n * (h + offset)


def _pose_candidates(site: Site, aligned: np.ndarray, binding: list[int], coarse_step_deg: float,
                     tilts: Sequence[float], rng: random.Random):
    """(theta, tilt_axis, tilt_angle) triples to score."""
    n = site.normal
    perp = unit(np.cross(n, np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])))
    if len(binding) == 2 and site.kind == "bridge":
        v = site.offsets[1] - site.offsets[0]
        theta0 = signed_angle(aligned[binding[1]] - aligned[binding[0]], v, n)
        axis = unit(v - np.dot(v, n) * n)
        return [(theta0 + flip, axis, math.radians(t)) for flip in (0.0, math.pi)
                for t in sorted({0.0, *tilts, *(-t for t in tilts)})]
    start = rng.random() * 2.0 * math.pi
    thetas = [start + 2.0 * math.pi * k / max(1, round(360.0 / coarse_step_deg))
              for k in range(max(1, round(360.0 / coarse_step_deg)))]
    out = []
    for th in thetas:
        out.append((th, perp, 0.0))
        for t in tilts:
            if t > 0:
                for psi in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
                    out.append((th, rotation_about_axis(n, psi) @ perp, math.radians(t)))
    return out


def _coords(aligned: np.ndarray, anchor: np.ndarray, n: np.ndarray, theta: float, axis: np.ndarray,
            tilt: float) -> np.ndarray:
    R = rotation_about_axis(axis, tilt) @ rotation_about_axis(n, theta) if tilt else rotation_about_axis(n, theta)
    return anchor + aligned @ R.T


def place_adsorbates(atoms: Atoms, adsorbate, sites: Sequence[Site], count: Optional[int] = None,
                  coverage: Optional[float] = None, mode: str = "adsorb", bond_length: Optional[float] = None,
                  offset: float = 0.0, min_clearance: float = 1.2, coarse_step_deg: float = 15.0,
                  fine_step_deg: float = 2.0, window_deg: float = 12.0, tilts: Sequence[float] = (15.0, 30.0),
                  seed: Optional[int] = None, max_refine: int = 2) -> Atoms:
    """Place copies of `adsorbate` (an Adsorbate with a BindingMotif) on `sites` and return a new structure.

    Args:
        sites: candidate sites (from surface_sites); `count` of them (or a `coverage` fraction) are
            picked by farthest-point sampling, else all are used.
        mode: "adsorb" binds on top of the site atoms; "replace" removes the site atoms (e.g. a
            surface halide, or two halides for a bridge site) and puts the binding atoms' centroid
            where they were.
        bond_length: donor-site distance (A). "adsorb": distance to the site atoms (default: the
            covalent-radius sum). "replace": the pose slides along the site normal until the
            closest donor-neighbour distance (to the atoms that bonded the removed ones) equals it;
            None keeps the anchor at the vacated position.
        offset: extra displacement (A) of the anchor along the site normal.
        min_clearance: target minimum distance ratio d/(r_a + r_b) (covalent radii) between each
            molecule and its environment, excluding the intended donor-site contacts. On
            metal/metalloid compounds, contacts between electronegative molecule atoms (O, N, S,
            halogens) and electronegative substrate atoms must also stay above 1.3 x the radii sum.
    The binding atoms point at the site, the body away from the surface; the rotation about the
    site normal (and small tilts) is searched to maximize clearance. Molecule atoms get tag 2 and
    a molecule id; records are kept in atoms.info["p2s_molecules"] (read them with get_molecules).
    """
    if mode not in ("adsorb", "replace"):
        raise ValueError(f"mode must be 'adsorb' or 'replace'; got {mode!r}")
    if not adsorbate.binding_atoms:
        raise ValueError(f"adsorbate {adsorbate.name} has no binding atoms; build it with a BindingMotif")
    sites = list(sites)
    if not sites:
        raise ValueError("no sites to place on")
    rng = random.Random(default_seed(seed))
    atoms = ensure_bookkeeping(atoms.copy())
    cell, pbc = atoms.cell.array, atoms.pbc

    n_target = len(sites)
    if count is not None:
        n_target = int(count)
    elif coverage is not None:
        n_target = int(math.ceil(float(coverage) * len(sites)))
    if n_target > len(sites):
        print(f"[place_adsorbates] requested {n_target} placements but only {len(sites)} sites; using all")
        n_target = len(sites)
    pick = farthest_point_sampling(np.array([s.position for s in sites]), n_target, rng,
                                   cell=cell if any(pbc) else None, pbc=pbc)
    chosen = [sites[k] for k in pick]

    removed: set[int] = set()
    if mode == "replace":
        for s in chosen:
            removed.update(s.indices)
    keep = np.array([i not in removed for i in range(len(atoms))])
    env_idx = np.where(keep)[0]
    symbols = atoms.get_chemical_symbols()
    env_r = rcov([symbols[i] for i in env_idx])
    env_tree = PeriodicTree(atoms.positions[env_idx], env_r, cell, pbc)
    env_pos = {int(o): k for k, o in enumerate(env_idx)}

    ads_sym = adsorbate.atoms.get_chemical_symbols()
    ads_r = rcov(ads_sym)
    local = adsorbate.atoms.get_positions()
    binding = list(adsorbate.binding_atoms)

    # like charges: electronegative adsorbate atoms keep away from electronegative substrate atoms (an O
    # 1.9 A from a surface Br is not caught by covalent-radius clearance alone)
    ids = mol_ids(atoms)
    like = surface_anions(symbols[i] for i in env_idx if ids[i] < 0)
    acceptor = [k for k, s in enumerate(ads_sym) if s in ANION_FORMERS]
    anions = [k for k, i in enumerate(env_idx) if ids[i] < 0 and symbols[i] in like]
    anion_tree = (PeriodicTree(atoms.positions[env_idx[anions]], env_r[anions], cell, pbc)
                  if acceptor and anions else None)

    def contacts(site: Site) -> list[int]:
        """Atoms the donor atoms are meant to touch (env indices)."""
        if mode == "adsorb":
            return [env_pos[i] for i in site.indices if i in env_pos]
        D, d = get_distances(atoms.positions[list(site.indices)], atoms.positions[env_idx], cell=cell, pbc=pbc)
        lim = 1.3 * (rcov([symbols[i] for i in site.indices])[:, None] + env_r[None, :])
        return sorted({int(k) for k in np.where(d < lim)[1]})

    def contact_ratio(coords: np.ndarray, contact: list[int]) -> float:
        """Closest donor-contact distance relative to its target (bond_length, else the
        covalent-radius sum)."""
        _, d = get_distances(coords[binding], atoms.positions[env_idx[contact]], cell=cell, pbc=pbc)
        ref = float(bond_length) if bond_length else ads_r[binding][:, None] + env_r[contact][None, :]
        return float((d / ref).min())

    def score(coords: np.ndarray, excl: dict[int, set[int]], contact: list[int],
              others: Optional[PeriodicTree]) -> float:
        s = env_tree.min_normalized(coords, ads_r, exclude=excl)
        if others is not None:
            s = min(s, others.min_normalized(coords, ads_r))
        if anion_tree is not None:
            s = min(s, anion_tree.min_normalized(coords[acceptor], ads_r[acceptor]) / ANION_CONTACT_SCALE)
        if contact:  # intended contacts are excluded above, but are still checked here
            ratio = contact_ratio(coords, contact)
            if mode == "replace" and ratio < 0.9:
                # donors swing around the vacated site; keep them near bonding distance or beyond
                s = min(s, ratio + (min_clearance - 1.0))
            elif ratio < 0.6:
                s = min(s, ratio)
        return s

    def settle(coords: np.ndarray, normal: np.ndarray, contact: list[int]) -> np.ndarray:
        """Replace mode with a bond_length: move the pose (at most 1.5 A) so the donors sit at
        bond_length from the atoms that held the removed ones. One donor moves straight toward its
        nearest such atom (e.g. a terminal bond to one of the two Pb a bridging halide connected);
        two donors slide along the site normal."""
        if mode != "replace" or not bond_length or not contact:
            return coords
        target = atoms.positions[env_idx[contact]]
        L = float(bond_length)
        if len(binding) == 1:
            D, d = get_distances(coords[binding], target, cell=cell, pbc=pbc)
            k = int(np.argmin(d[0]))
            shift = D[0, k] * (1.0 - L / max(float(d[0, k]), 1e-6))
            n = float(np.linalg.norm(shift))
            return coords + (shift if n <= 1.5 else shift * 1.5 / n)

        def gap(delta: float) -> float:
            _, d = get_distances(coords[binding] + delta * normal, target, cell=cell, pbc=pbc)
            return float(d.min()) - L

        g0 = gap(0.0)
        lo, hi = (-1.5, 0.0) if g0 > 0 else (0.0, 1.5)
        g_lo, g_hi = gap(lo), gap(hi)
        if g_lo * g_hi > 0:  # target out of reach along the normal: closest point of a scan
            deltas = np.linspace(-1.5, 1.5, 31)
            return coords + deltas[int(np.argmin([abs(gap(x)) for x in deltas]))] * normal
        for _ in range(25):
            mid = 0.5 * (lo + hi)
            g_mid = gap(mid)
            if g_mid * g_lo > 0:
                lo, g_lo = mid, g_mid
            else:
                hi = mid
        return coords + 0.5 * (lo + hi) * normal

    def best_pose(site: Site, others: Optional[PeriodicTree]) -> tuple[np.ndarray, float]:
        aligned = local @ rotation_from_u_to_v(Z, site.normal).T
        anchor = _anchor_position(site, aligned, binding, ads_sym, mode, bond_length, offset)
        contact = contacts(site)
        excl = {b: set(contact) for b in binding}
        best = (-np.inf, None, None)
        for theta, axis, tilt in _pose_candidates(site, aligned, binding, coarse_step_deg, tilts, rng):
            c = settle(_coords(aligned, anchor, site.normal, theta, axis, tilt), site.normal, contact)
            s = score(c, excl, contact, others)
            if s > best[0]:
                best = (s, c, (theta, axis, tilt))
        s_best, c_best, (theta, axis, tilt) = best
        if s_best < min_clearance:
            window, step = math.radians(window_deg), math.radians(fine_step_deg)
            for dth in np.arange(-window, window + 1e-9, step):
                c = settle(_coords(aligned, anchor, site.normal, theta + dth, axis, tilt), site.normal, contact)
                s = score(c, excl, contact, others)
                if s > s_best:
                    s_best, c_best = s, c
        return c_best, float(s_best)

    def tree_of(coords_list: list[np.ndarray]) -> Optional[PeriodicTree]:
        if not coords_list:
            return None
        pts = np.vstack(coords_list)
        return PeriodicTree(pts, np.tile(ads_r, len(coords_list)), cell, pbc)

    placed: list[np.ndarray] = []
    scores: list[float] = []
    for site in chosen:
        c, s = best_pose(site, tree_of(placed))
        placed.append(c)
        scores.append(s)
    for _ in range(max_refine):
        low = [k for k, s in enumerate(scores) if s < min_clearance]
        if not low:
            break
        improved = False
        for k in low:
            c, s = best_pose(chosen[k], tree_of([p for j, p in enumerate(placed) if j != k]))
            if s > scores[k] + 1e-3:
                placed[k], scores[k], improved = c, s, True
        if not improved:
            break

    new = atoms[keep]
    molecules = get_molecules(new)
    next_id = int(mol_ids(new).max()) + 1 if len(new) else 0
    next_id = max(next_id, 1 + max((m["mol_id"] for m in molecules), default=-1))
    for site, c, s in zip(chosen, placed, scores):
        ads = Atoms(ads_sym, positions=c)
        ads.set_tags(np.full(len(ads), ROLE_ADSORBATE, dtype=int))
        ads.set_array("mol_id", np.full(len(ads), next_id, dtype=int))
        new.extend(ads)
        molecules.append({"mol_id": next_id, "name": adsorbate.name, "smiles": adsorbate.smiles,
                          "charge": int(adsorbate.charge), "role": "adsorbate", "mode": mode,
                          "site": site.label, "binding_atoms": binding, "clearance": round(s, 3)})
        next_id += 1
    set_molecules(new, molecules)
    restore_vacuum(new)
    low = [s for s in scores if s < min_clearance]
    if low:
        print(f"[place_adsorbates] warning: {len(low)}/{len(scores)} {adsorbate.name} placements below "
              f"clearance {min_clearance} (worst {min(low):.2f})")
    return new


def restore_vacuum(atoms: Atoms) -> None:
    """Keep the builder's vacuum after molecules were added (as PerovKit resets the cell height
    to the adsorbate top plus the original vacuum): slabs grow along c, nanocrystal boxes are
    re-padded."""
    vac = atoms.info.get("p2s_vacuum")
    if not vac:
        return
    kind = atoms.info.get("p2s_kind")
    if kind == "slab":
        cell = atoms.cell.array.copy()
        gap = vacuum_gap(atoms)
        if gap < vac and abs(cell[2, 0]) < 1e-6 and abs(cell[2, 1]) < 1e-6:
            cell[2, 2] += vac - gap
            atoms.set_cell(cell, scale_atoms=False)
    elif kind == "nanocrystal":
        atoms.center(vacuum=float(vac))


def place_adsorbate(atoms: Atoms, adsorbate, site: Site, mode: str = "adsorb", **kwargs) -> Atoms:
    """Place one copy of `adsorbate` on `site` (see place_adsorbates for the keyword arguments)."""
    return place_adsorbates(atoms, adsorbate, [site], mode=mode, **kwargs)


def add_molecule(atoms: Atoms, molecule: Atoms, smiles: str, name: str, role: str = "adsorbate",
                 **record) -> Atoms:
    """Append a molecule you positioned with your own code, with the bookkeeping place_adsorbates and
    add_solvent do: tag 2 (adsorbate) or 3 (solvent, guest), a new mol_id and a molecule record
    whose SMILES lets validation check the molecule's bonds. `molecule` holds all its atoms,
    hydrogens included, at their final Cartesian positions (any order); extra keyword arguments
    (e.g. site="bridge Ti-Ti") go into the record. Returns a new Atoms.

    role: "adsorbate" (bound to a surface; validation checks the binding), "solvent", or "guest"
        (part of the crystal: organic cations of a hybrid perovskite, guests in a framework;
        checked for integrity and overlaps, not for surface binding).

    The molecule must be an RDKit conformer of `smiles` (Adsorbate.from_smiles(smiles).atoms), moved
    rigidly or with torsions turned: its bond lengths and angles are compared with RDKit's and a
    hand-built or reshaped molecule raises ValueError.
    """
    from rdkit import Chem

    if role not in ("adsorbate", "solvent", "guest"):
        raise ValueError("role must be 'adsorbate', 'solvent' or 'guest'")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES {smiles!r}")
    want = sorted(a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms())
    if sorted(molecule.get_chemical_symbols()) != want:
        raise ValueError(f"{name}: atoms {molecule.get_chemical_formula()} do not match SMILES {smiles!r}")
    from paper2structure.tools.geometry_checks import infer_bonds, molecule_geometry

    loose = Atoms(molecule.get_chemical_symbols(), positions=molecule.get_positions())
    off = molecule_geometry(loose.get_chemical_symbols(), loose.positions, sorted(infer_bonds(loose)), smiles)
    if off:
        raise ValueError(f"{name}: not the RDKit conformer of {smiles!r} ({'; '.join(off[:4])}). Build molecules with "
                         "Adsorbate.from_smiles(smiles, seed=SEED).atoms and move them rigidly or turn torsions")
    new = ensure_bookkeeping(atoms.copy())
    molecules = get_molecules(new)
    next_id = max(int(mol_ids(new).max()) + 1 if len(new) else 0,
                  1 + max((m["mol_id"] for m in molecules), default=-1))
    part = Atoms(molecule.get_chemical_symbols(), positions=molecule.get_positions())
    part.set_tags(np.full(len(part), ROLE_ADSORBATE if role == "adsorbate" else ROLE_SOLVENT, dtype=int))
    part.set_array("mol_id", np.full(len(part), next_id, dtype=int))
    new.extend(part)
    molecules.append({"mol_id": next_id, "name": name, "smiles": smiles, "charge": Chem.GetFormalCharge(mol),
                      "role": role, **record})
    set_molecules(new, molecules)
    return new
