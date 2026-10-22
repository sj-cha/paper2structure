from __future__ import annotations

import json
import math
import random
from typing import Optional

import numpy as np
from ase import Atoms
from ase.build import bulk as ase_bulk
from ase.cluster import Decahedron, Icosahedron, Octahedron, wulff_construction
from pymatgen.core.surface import get_symmetrically_equivalent_miller_indices
from pymatgen.io.ase import AseAtomsAdaptor

from paper2structure.tools._geometry import (ROLE_SUBSTRATE, bulk_coordination, coordination_pairs,
                                             default_seed, require_framework, unit)

CLUSTER_SHAPES = ("icosahedron", "cuboctahedron", "octahedron", "truncated_octahedron", "decahedron", "wulff")


def _finish(nc: Atoms, vacuum: float, bulk_cn: str) -> Atoms:
    atoms = Atoms(nc.get_chemical_symbols(), positions=nc.get_positions())
    atoms.center(vacuum=float(vacuum))
    atoms.pbc = True
    atoms.set_tags(np.full(len(atoms), ROLE_SUBSTRATE, dtype=int))
    atoms.set_array("mol_id", np.full(len(atoms), -1, dtype=int))
    atoms.info = {"p2s_kind": "nanocrystal", "p2s_bulk_cn": bulk_cn, "p2s_vacuum": float(vacuum)}
    return atoms


def build_cluster(element: str, shape: str, size: int, lattice_constant: float, structure: str = "fcc",
                  surface_energies: Optional[dict] = None, vacuum: float = 15.0) -> Atoms:
    """Elemental metal nanoparticle with a closed-shell shape (ase.cluster), centered in a box.

    Args:
        shape: "icosahedron", "cuboctahedron" or "decahedron" (size = number of shells;
            2/3/4/5 shells give 13/55/147/309 atoms for the first two), "octahedron" or
            "truncated_octahedron" (size = atoms along an edge), "wulff" (size = target number of
            atoms; needs surface_energies such as {(1, 1, 1): 1.0, (1, 0, 0): 1.15}).
        lattice_constant: cubic lattice constant of the bulk metal (A).
        structure: bulk lattice for "wulff" ("fcc", "bcc" or "sc").
        vacuum: vacuum padding on every side (A); images are 2 x vacuum apart.
    """
    n = int(size)
    if shape == "icosahedron":
        nc = Icosahedron(element, noshells=n, latticeconstant=lattice_constant)
    elif shape == "cuboctahedron":
        nc = Octahedron(element, length=2 * n - 1, cutoff=n - 1, latticeconstant=lattice_constant)
    elif shape == "octahedron":
        nc = Octahedron(element, length=n, cutoff=0, latticeconstant=lattice_constant)
    elif shape == "truncated_octahedron":
        nc = Octahedron(element, length=n, cutoff=max(1, (n - 1) // 3), latticeconstant=lattice_constant)
    elif shape == "decahedron":
        nc = Decahedron(element, p=n, q=1, r=0, latticeconstant=lattice_constant)
    elif shape == "wulff":
        if not surface_energies:
            raise ValueError("shape='wulff' needs surface_energies, e.g. {(1, 1, 1): 1.0, (1, 0, 0): 1.15}")
        nc = wulff_construction(element, surfaces=[tuple(k) for k in surface_energies],
                                energies=[float(v) for v in surface_energies.values()], size=n,
                                structure=structure, rounding="closest", latticeconstant=lattice_constant)
    else:
        raise ValueError(f"unknown shape {shape!r}; choose from {CLUSTER_SHAPES}")
    ref = ase_bulk(element, "fcc" if shape != "wulff" else structure, a=lattice_constant, cubic=True)
    return _finish(nc, vacuum, bulk_coordination(ref))


def carve_nanocrystal(bulk: Atoms, diameter: float, facets: Optional[dict] = None,
                      oxidation_states: Optional[dict[str, int]] = None, min_coordination: int = 2,
                      center_element: Optional[str] = None, vacuum: float = 15.0,
                      seed: Optional[int] = None) -> Atoms:
    """Nanocrystal cut out of any bulk crystal, centered in a box with `vacuum` A on every side.

    Args:
        bulk: periodic bulk cell (e.g. from build_bulk).
        diameter: particle size in A: the sphere diameter, or twice the center-to-facet distance
            of a facet family whose relative distance is 1.
        facets: {miller: relative distance} for a faceted particle, e.g. {(1, 0, 0): 1.0} for a
            cube or {(1, 1, 1): 1.0, (1, 0, 0): 1.15} for a truncated octahedron. All
            symmetry-equivalent planes of the bulk are applied. None cuts a sphere.
        oxidation_states: e.g. {"Cs": 1, "Pb": 2, "Br": -1}. When given, the lowest-coordinated
            atoms carrying the excess charge are removed (corners first), each together with the
            terminal atoms bonded only to it (an O takes its H), until the particle is neutral or
            no further removal helps.
        min_coordination: atoms with fewer neighbours than min(min_coordination, their element's
            coordination in the bulk) are pruned repeatedly (dangling atoms); a terminal atom such
            as a hydroxyl H (bulk coordination 1) stays as long as it keeps its neighbour.
        center_element: center the cut on the atom of this element nearest the middle of the
            supercell; this sets which atomic layers terminate the facets.
    """
    require_framework(bulk, "carve_nanocrystal")
    rng = random.Random(default_seed(seed))
    bulk_cn = bulk_coordination(bulk)
    cell = bulk.cell.array
    vol = abs(np.linalg.det(cell))
    widths = [vol / np.linalg.norm(np.cross(cell[(k + 1) % 3], cell[(k + 2) % 3])) for k in range(3)]
    reps = [int(math.ceil((float(diameter) + 8.0) / w)) + 1 for w in widths]
    big = bulk.repeat(reps)
    middle = big.cell.array.sum(axis=0) / 2.0
    cand = [i for i, s in enumerate(big.get_chemical_symbols()) if center_element in (None, s)]
    if not cand:
        raise ValueError(f"no {center_element} atom in the bulk cell")
    c0 = big.positions[min(cand, key=lambda i: np.linalg.norm(big.positions[i] - middle))]
    rel = big.positions - c0
    radius = float(diameter) / 2.0
    if facets:
        structure = AseAtomsAdaptor.get_structure(bulk)
        recip = bulk.cell.reciprocal()
        keep = np.ones(len(big), dtype=bool)
        for hkl, frac in facets.items():
            for eq in get_symmetrically_equivalent_miller_indices(structure, tuple(int(x) for x in hkl)):
                normal = unit(np.dot(np.asarray(eq, dtype=float), recip))
                keep &= rel @ normal <= radius * float(frac) + 1e-6
    else:
        keep = np.linalg.norm(rel, axis=1) <= radius + 1e-6
    nc = big[keep]
    nc.pbc = False
    nc.center(vacuum=6.0)  # tight box: neighbour lists are slow with atoms outside a huge cell

    floor = {e: min(int(min_coordination), c) for e, c in json.loads(bulk_cn).items()}
    for _ in range(20):
        i, _, _ = coordination_pairs(nc)
        cn = np.bincount(i, minlength=len(nc))
        drop = cn < np.array([floor[e] for e in nc.get_chemical_symbols()])
        if not drop.any():
            break
        nc = nc[~drop]

    if oxidation_states:
        missing = set(nc.get_chemical_symbols()) - set(oxidation_states)
        if missing:
            raise ValueError(f"oxidation_states lacks {sorted(missing)}")
        q = np.array([oxidation_states[s] for s in nc.get_chemical_symbols()], dtype=float)
        i, j, _ = coordination_pairs(nc)
        cn = np.bincount(i, minlength=len(nc))
        neighbours = [[] for _ in range(len(nc))]
        for a, b in zip(i, j):
            neighbours[a].append(b)
        alive = np.ones(len(nc), dtype=bool)
        dist = np.round(np.linalg.norm(nc.positions - nc.positions.mean(axis=0), axis=1), 2)
        noise = np.array([rng.random() for _ in range(len(nc))])
        while True:
            total = float(q[alive].sum())
            if abs(total) < 1e-9:
                break
            sign = 1.0 if total > 0 else -1.0
            units = {}  # removable atom -> (it plus the terminal atoms only it holds, their charge)
            for k in np.where(alive & (q * sign > 0))[0]:
                group = [k] + [nb for nb in neighbours[k] if alive[nb] and cn[nb] == 1]
                if q[group].sum() * sign > 0:
                    units[k] = (group, float(q[group].sum()))
            fits = [k for k, (_, charge) in units.items() if abs(charge) <= abs(total) + 1e-9]
            pool = fits or list(units)
            if not pool:
                break
            k = min(pool, key=lambda k: (cn[k], -dist[k], noise[k]))  # least coordinated, outermost first
            group, charge = units[k]
            if not fits and abs(total - charge) >= abs(total):
                break
            for u in group:
                alive[u] = False
                for nb in neighbours[u]:
                    cn[nb] -= 1
        nc = nc[alive]
        charge = float(q[alive].sum())
        if abs(charge) > 1e-9:
            print(f"[carve_nanocrystal] warning: residual charge {charge:+g} after balancing")
    atoms = _finish(nc, vacuum, bulk_cn)
    if oxidation_states:
        atoms.info["p2s_charge"] = float(sum(oxidation_states[s] for s in atoms.get_chemical_symbols()))
        atoms.info["p2s_oxidation_states"] = " ".join(f"{e}:{q:+d}" for e, q in sorted(oxidation_states.items()))
    print(f"[carve_nanocrystal] {atoms.get_chemical_formula()} ({len(atoms)} atoms)")
    return atoms
