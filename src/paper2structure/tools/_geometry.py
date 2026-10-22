"""Geometry helpers shared by the builders. Rotation and sampling follow PerovKit
(https://github.com/sj-cha/PerovKit), generalized to arbitrary (non-orthorhombic) cells."""
from __future__ import annotations

import itertools
import json
import re
import os
import random
from typing import Iterable, Optional, Sequence

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import get_distances
from scipy.spatial import cKDTree

ROLE_FIXED, ROLE_SUBSTRATE, ROLE_ADSORBATE, ROLE_SOLVENT = 0, 1, 2, 3  # OC20-style tags (+ solvent)
MOLECULES_KEY = "p2s_molecules"
NVALCHEMI_MIN_ATOMS = 1500  # below this, ASE is faster than importing torch/warp


def ensure_warp_cache() -> None:
    """Point Warp's kernel cache at a writable directory (the default ~/.cache may not be).
    Must run before the first import of warp/nvalchemiops, which initialize the cache on import."""
    if os.environ.get("WARP_CACHE_PATH"):
        return
    import tempfile

    for base in (os.path.expanduser("~/.cache"), tempfile.gettempdir()):
        path = os.path.join(base, f"paper2structure-warp-{os.getuid()}")
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            continue
        if os.access(path, os.W_OK):
            os.environ["WARP_CACHE_PATH"] = path
            return


def _nvalchemi_pairs(atoms: Atoms, cutoff: float):
    ensure_warp_cache()
    import torch
    from nvalchemiops.torch.neighbors import neighbor_list as nv_neighbor_list

    use_cuda = torch.cuda.is_available() and os.environ.get("P2S_DEVICE", "auto") != "cpu"
    device = "cuda" if use_cuda else "cpu"
    pos = atoms.get_positions(wrap=bool(np.any(atoms.pbc)))
    tpos = torch.tensor(pos, dtype=torch.float64, device=device)
    if np.any(atoms.pbc):
        cell = atoms.cell.array
        out = nv_neighbor_list(tpos, float(cutoff), cell=torch.tensor(cell, dtype=torch.float64, device=device),
                               pbc=torch.tensor(atoms.pbc, dtype=torch.bool, device=device),
                               return_neighbor_list=True)
        pairs, shifts = out[0].cpu().numpy(), out[2].cpu().numpy().astype(float)
        i, j = pairs[0].astype(int), pairs[1].astype(int)
        D = pos[j] - pos[i] + shifts @ cell
    else:
        pairs = nv_neighbor_list(tpos, float(cutoff), return_neighbor_list=True)[0].cpu().numpy()
        i, j = pairs[0].astype(int), pairs[1].astype(int)
        D = pos[j] - pos[i]
    return i, j, D


def neighbor_pairs(atoms: Atoms, cutoff):
    """Ordered pairs (i, j, D) with D the minimum-image vector from i to j, closer than `cutoff`
    (a float) or than r_i + r_j when `cutoff` is a per-atom list of radii.

    Large structures use the NVIDIA ALCHEMI Toolkit-Ops neighbor list (GPU when torch sees one),
    small ones ASE; P2S_NEIGHBOR_BACKEND = "ase" | "nvalchemiops" | "auto" overrides the choice."""
    from ase.neighborlist import neighbor_list

    radii = np.asarray(cutoff, dtype=float) if np.ndim(cutoff) else None
    rc = 2.0 * float(radii.max()) if radii is not None else float(cutoff)
    if len(atoms) < 2 or rc <= 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.zeros((0, 3))
    backend = os.environ.get("P2S_NEIGHBOR_BACKEND", "auto")
    if backend == "nvalchemiops" or (backend == "auto" and len(atoms) >= NVALCHEMI_MIN_ATOMS):
        try:
            i, j, D = _nvalchemi_pairs(atoms, rc)
        except Exception:  # noqa: BLE001 - fall back to ASE unless explicitly requested
            if backend == "nvalchemiops":
                raise
            i, j, D = neighbor_list("ijD", atoms, rc)
    else:
        i, j, D = neighbor_list("ijD", atoms, rc)
    if radii is not None and len(i):
        keep = np.linalg.norm(D, axis=1) < radii[i] + radii[j]
        i, j, D = i[keep], j[keep], D[keep]
    return i, j, D


def default_seed(seed: Optional[int] = None) -> int:
    """The explicit seed, else 0, so a script that passes none is still reproducible."""
    return 0 if seed is None else int(seed)


def rcov(symbols: Iterable[str]) -> np.ndarray:
    """Covalent radii (Cordero 2008, as ase tabulates them), in Angstrom."""
    return np.array([covalent_radii[atomic_numbers[s]] for s in symbols], dtype=float)


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("cannot normalize a zero-length vector")
    return v / n


def rotation_about_axis(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues' rotation matrix for a rotation by `theta` (rad) about `axis`."""
    x, y, z = unit(axis)
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def rotation_from_u_to_v(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotation matrix taking direction `u` onto direction `v`."""
    u, v = unit(u), unit(v)
    c = float(np.dot(u, v))
    if np.isclose(c, 1.0):
        return np.eye(3)
    if np.isclose(c, -1.0):
        tmp = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        return rotation_about_axis(np.cross(u, tmp), np.pi)
    return rotation_about_axis(np.cross(u, v), float(np.arccos(c)))


def signed_angle(u: np.ndarray, v: np.ndarray, axis: np.ndarray) -> float:
    """Angle (rad) rotating `u` onto `v` about `axis`, after projecting both onto the plane
    perpendicular to `axis`."""
    n = unit(axis)
    u = u - np.dot(u, n) * n
    v = v - np.dot(v, n) * n
    return float(np.arctan2(np.dot(np.cross(u, v), n), np.dot(u, v)))


def image_shifts(cell: np.ndarray, pbc: Sequence[bool]) -> np.ndarray:
    """Cartesian translations to the 3^k neighbouring periodic images (k = number of periodic axes)."""
    ranges = [(-1, 0, 1) if p else (0,) for p in pbc]
    return np.array([np.dot(n, cell) for n in itertools.product(*ranges)], dtype=float)


class PeriodicTree:
    """KD-tree over a set of atoms and their neighbouring periodic images, for any cell shape.

    Answers "how close is the nearest atom, relative to the covalent-radius sum" queries used by
    the placement and packing searches. Valid while the queried distances stay below the
    shortest cell width, which holds for overlap-scale cutoffs."""

    def __init__(self, positions: np.ndarray, radii: np.ndarray, cell: np.ndarray, pbc: Sequence[bool]):
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        self.radii = np.asarray(radii, dtype=float)
        self.n = len(positions)
        self.r_max = float(self.radii.max()) if self.n else 0.0
        if self.n == 0:
            self.tree = None
            self.index = np.array([], dtype=int)
            return
        shifts = image_shifts(np.asarray(cell, dtype=float), pbc)
        pts = (positions[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
        self.tree = cKDTree(pts)
        self.index = np.tile(np.arange(self.n), len(shifts))
        self.points = pts

    def min_normalized(self, points: np.ndarray, radii: np.ndarray, cap: float = 3.0,
                       exclude: Optional[dict[int, set[int]]] = None) -> float:
        """Smallest d_ab / (r_a + r_b) between query points a and tree atoms b, capped at `cap`.

        `exclude` maps a query-point index to tree-atom indices whose pairs are skipped
        (intended contacts such as a donor atom and its binding-site atoms)."""
        if self.tree is None or len(points) == 0:
            return cap
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        radii = np.asarray(radii, dtype=float)
        best = cap
        for a, (p, ra) in enumerate(zip(points, radii)):
            hits = self.tree.query_ball_point(p, r=cap * (ra + self.r_max))
            if not hits:
                continue
            hits = np.asarray(hits, dtype=int)
            atoms_b = self.index[hits]
            if exclude and a in exclude:
                keep = ~np.isin(atoms_b, list(exclude[a]))
                hits, atoms_b = hits[keep], atoms_b[keep]
                if len(hits) == 0:
                    continue
            d = np.linalg.norm(self.points[hits] - p, axis=1)
            best = min(best, float(np.min(d / (ra + self.radii[atoms_b]))))
        return best


def farthest_point_sampling(coords: np.ndarray, n_target: int, rng: random.Random,
                            cell: Optional[np.ndarray] = None, pbc: Optional[Sequence[bool]] = None,
                            initial_idx: Optional[int] = None) -> list[int]:
    """Spread `n_target` picks over `coords` (minimum-image distances when a cell is given)."""
    n = len(coords)
    if n_target >= n:
        return list(range(n))
    if n_target <= 0:
        return []

    def dist_from(i: int) -> np.ndarray:
        if cell is None:
            return np.linalg.norm(coords - coords[i], axis=1)
        _, d = get_distances(coords[i:i + 1], coords, cell=cell, pbc=pbc)
        return d[0]

    selected = [rng.randrange(n) if initial_idx is None else int(initial_idx)]
    d = dist_from(selected[0])
    d[selected] = 0.0
    while len(selected) < n_target:
        nxt = int(np.argmax(d))
        selected.append(nxt)
        d = np.minimum(d, dist_from(nxt))
        d[selected] = 0.0
    return selected


_ENCODED = "pct:"  # JSON with the characters ASE's extxyz info strings mangle percent-encoded
_UNSAFE = {"%": "%25", "\\": "%5C", '"': "%22"}
_SAFE = {code[1:]: char for char, code in _UNSAFE.items()}
_BARE_BACKSLASH = re.compile(r'\\(?![\\"/bfnrtu])')


def get_molecules(atoms: Atoms) -> list[dict]:
    """Molecule records (adsorbates, solvent) attached to a structure."""
    raw = atoms.info.get(MOLECULES_KEY)
    if raw is None or len(raw) == 0:  # extxyz reads an empty record list back as an empty array
        return []
    if not isinstance(raw, str):
        return list(raw)
    if raw.startswith(_ENCODED):
        return json.loads(re.sub(r"%(25|5C|22)", lambda m: _SAFE[m.group(1)], raw[len(_ENCODED):]))
    try:
        return json.loads(raw)
    except json.JSONDecodeError:  # plain JSON from older files: the extxyz reader dropped escapes
        return json.loads(_BARE_BACKSLASH.sub(r"\\\\", raw))


def set_molecules(atoms: Atoms, molecules: list[dict]) -> None:
    """Store the records as a JSON string that survives extxyz round trips: ASE's extxyz writer does
    not escape backslashes (stereo bonds in SMILES such as C/C=C\\C) or quotes, so those are
    percent-encoded."""
    text = json.dumps(molecules)
    atoms.info[MOLECULES_KEY] = _ENCODED + "".join(_UNSAFE.get(c, c) for c in text)


def mol_ids(atoms: Atoms) -> np.ndarray:
    """Per-atom molecule id (-1 for the substrate)."""
    if "mol_id" in atoms.arrays:
        return np.asarray(atoms.arrays["mol_id"], dtype=int)
    return np.full(len(atoms), -1, dtype=int)


def ensure_bookkeeping(atoms: Atoms) -> Atoms:
    """Give a structure the per-atom arrays the builders rely on (mol_id; substrate tags)."""
    if "mol_id" not in atoms.arrays:
        atoms.set_array("mol_id", np.full(len(atoms), -1, dtype=int))
    tags = atoms.get_tags()
    if len(atoms) and not np.any(tags):
        atoms.set_tags(np.full(len(atoms), ROLE_SUBSTRATE, dtype=int))
    return atoms


def substrate_mask(atoms: Atoms) -> np.ndarray:
    return mol_ids(atoms) < 0


def register_molecules(atoms: Atoms, parts: Sequence[Atoms], records: Sequence[dict], tag: int) -> Atoms:
    """Append molecules (each an Atoms at its final positions) with the bookkeeping validation
    relies on: tag `tag` (ROLE_ADSORBATE or ROLE_SOLVENT), a new mol_id each, and its record (the
    mol_id is filled in) in atoms.info["p2s_molecules"]. Returns a new Atoms."""
    new = ensure_bookkeeping(atoms.copy())
    molecules = get_molecules(new)
    next_id = max(int(mol_ids(new).max()) + 1 if len(new) else 0, 1 + max((m["mol_id"] for m in molecules), default=-1))
    for part, record in zip(parts, records):
        piece = Atoms(part.get_chemical_symbols(), positions=part.get_positions())
        piece.set_tags(np.full(len(piece), tag, dtype=int))
        piece.set_array("mol_id", np.full(len(piece), next_id, dtype=int))
        new.extend(piece)
        molecules.append({"mol_id": next_id, **record})
        next_id += 1
    set_molecules(new, molecules)
    return new


def require_framework(crystal: Atoms, builder: str) -> None:
    """Builders that cut a crystal (slabs, particles) handle its framework only: they would cut its
    registered molecules apart and drop their records."""
    molecules = get_molecules(crystal)
    if molecules or (mol_ids(crystal) >= 0).any():
        names = sorted({m.get("name", "?") for m in molecules}) or ["unregistered"]
        raise ValueError(f"{builder} cuts a crystal's framework only, and this one holds molecules ({', '.join(names)}). "
                         f"Cut the empty framework (build_bulk without fill_cavities), then add the molecules to the "
                         f"result: fill_cavities(result, smiles, count, name=...) places them in the slab's or "
                         f"particle's pockets; or cut the filled crystal with your own code, keeping molecules "
                         f"whole, and register each with add_molecule(..., role='guest')")


def solvent_mask(atoms: Atoms) -> np.ndarray:
    """Atoms of molecules registered with role "solvent"."""
    ids = [m["mol_id"] for m in get_molecules(atoms) if m.get("role") == "solvent"]
    return np.isin(mol_ids(atoms), ids) if ids else np.zeros(len(atoms), dtype=bool)


def hull_depth(points: np.ndarray, hull_points: Optional[np.ndarray] = None) -> np.ndarray:
    """Distance of each point below the convex hull of `hull_points` (default: the points
    themselves), 0 on or outside it, in 2 or 3 dimensions. Flat or tiny point sets have no
    interior: every point is on the outside."""
    from scipy.spatial import ConvexHull

    hull_points = points if hull_points is None else hull_points
    try:
        eq = ConvexHull(hull_points).equations
    except Exception:  # noqa: BLE001 - degenerate input (too few points, coplanar)
        return np.zeros(len(points))
    # inside a convex hull every facet's signed distance n.x + d is <= 0; the nearest facet is the largest
    return np.maximum(-(points @ eq[:, :-1].T + eq[:, -1]).max(axis=1), 0.0)


KEPT_ARRAYS = ("numbers", "positions", "tags", "mol_id", "momenta", "initial_magmoms", "initial_charges")


def clean_arrays(atoms: Atoms) -> Atoms:
    """Drop per-atom arrays the pipeline does not own (e.g. pymatgen's bulk_wyckoff strings, which
    ASE pads with empty strings when molecules are appended and which then break extxyz)."""
    for key in list(atoms.arrays):
        if key not in KEPT_ARRAYS:
            del atoms.arrays[key]
    return atoms


ANION_FORMERS = frozenset({"N", "O", "F", "S", "Cl", "Br", "I", "Se", "Te"})  # electronegative elements
ANION_CONTACT_SCALE = 1.3  # x covalent sum: an electronegative molecule atom closer to a surface anion crowds it


def cation_elements(symbols: Iterable[str]) -> frozenset[str]:
    """The metals and metalloids of a compound that also contains non-metals (oxides, halides,
    carbonates, hydroxides, ...); empty for metals, alloys and non-metal solids."""
    from pymatgen.core import Element

    elements = set(symbols)
    cations = frozenset(e for e in elements if Element(e).is_metal or Element(e).is_metalloid)
    return cations if cations and cations != elements else frozenset()


def surface_anions(symbols: Iterable[str]) -> frozenset[str]:
    """Electronegative elements (ANION_FORMERS) of a metal/metalloid compound: the substrate atoms
    an electronegative molecule atom should not crowd (like charges). Empty for metals, alloys and
    non-metal solids; never H, so hydrogen bonds to surface OH are fine."""
    elements = set(symbols)
    return frozenset(elements & ANION_FORMERS) if cation_elements(elements) else frozenset()


def coordination_pairs(atoms: Atoms, cutoff_scale: float = 1.2):
    """Bonded pairs (i, j, D) from covalent-radius cutoffs. In compounds, metal-metal contacts
    (Ti-Ti in TiO2) are not bonds; every pair involving a non-metal is (Ti-O, C-O in carbonates,
    O-H in hydroxides, S-S in pyrite)."""
    symbols = np.array(atoms.get_chemical_symbols())
    i, j, D = neighbor_pairs(atoms, cutoff_scale * rcov(symbols))
    cations = cation_elements(symbols)
    if cations and len(i):
        is_cation = np.isin(symbols, list(cations))
        keep = ~(is_cation[i] & is_cation[j])
        i, j, D = i[keep], j[keep], D[keep]
    return i, j, D


def bulk_coordination(bulk: Atoms, cutoff_scale: float = 1.2) -> str:
    """Highest coordination number per element in a periodic bulk cell, as a JSON string for
    atoms.info["p2s_bulk_cn"] (the reference surface_atoms compares against)."""
    i, _, _ = coordination_pairs(bulk, cutoff_scale)
    cn = np.bincount(i, minlength=len(bulk))
    ref: dict[str, int] = {}
    for s, c in zip(bulk.get_chemical_symbols(), cn):
        ref[s] = max(ref.get(s, 0), int(c))
    return json.dumps(ref)
