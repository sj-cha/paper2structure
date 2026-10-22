from __future__ import annotations

import random
from typing import Optional

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from scipy.spatial.transform import Rotation

from paper2structure.tools._geometry import (ROLE_SOLVENT, PeriodicTree, default_seed, ensure_bookkeeping,
                                             register_molecules, substrate_mask)
from paper2structure.tools.adsorbate_builder import Adsorbate
from paper2structure.tools.structure_export import structure_kind

AVOGADRO = 6.02214076e23
FREE_VOLUME_PROBES = 4000  # random points that estimate the free share of the solvent region


def add_solvent(atoms: Atoms, smiles: str, n_molecules: Optional[int] = None, density: Optional[float] = None,
                thickness: Optional[float] = None, gap: float = 2.0, tolerance: float = 2.0,
                name: Optional[str] = None, seed: Optional[int] = None, max_attempts: int = 500) -> Atoms:
    """Pack rigid copies of a solvent molecule around a structure (random positions/orientations).

    Args:
        smiles: solvent molecule (one conformer, UFF-relaxed).
        n_molecules: number of molecules; or give `density` (g/cm^3, e.g. 0.786 for acetonitrile)
            to fill the free volume of the solvent region at that density.
        thickness: slabs only - solvent layer thickness (A) starting `gap` A above the topmost
            substrate atom, so it surrounds adsorbates; the cell is extended along z when needed,
            keeping `gap` A to the periodic image of the slab bottom. Default: the existing vacuum
            minus the gaps.
        tolerance: minimum distance (A) between any solvent atom and any other atom
            (Packmol's convention).
    Other structures (particles, wires, bulk with pores, an empty cell for a pure liquid) get
    solvent in the whole cell. Space already taken (the substrate, adsorbates, solvent added before,
    plus `gap`) is not counted as free volume in density mode, so mixtures can be built one species
    after another. Inserts as many molecules as fit within `max_attempts` tries each and reports
    the achieved count. Solvent atoms get tag 3 and a molecule id.
    """
    if n_molecules is None and density is None:
        raise ValueError("give n_molecules or density")
    rng = random.Random(default_seed(seed))
    nprng = np.random.default_rng(default_seed(seed))
    atoms = ensure_bookkeeping(atoms.copy())
    mol = Adsorbate.from_smiles(smiles, None, name=name or smiles, seed=seed, optimize=True)
    tpl = mol.atoms.get_positions() - mol.atoms.get_positions().mean(axis=0)
    sym = mol.atoms.get_chemical_symbols()
    mass = float(sum(atomic_masses[atomic_numbers[s]] for s in sym))
    cell = atoms.cell.array.copy()
    if atoms.cell.rank < 3:
        raise ValueError("add_solvent needs a cell with three vectors (the box to fill)")

    if len(atoms) and structure_kind(atoms) == "slab":
        if abs(cell[2, 0]) > 1e-6 or abs(cell[2, 1]) > 1e-6:
            raise ValueError("add_solvent needs the slab's c vector along z (build_slab provides this)")
        sub = substrate_mask(atoms)
        z = atoms.positions[sub if sub.any() else slice(None), 2]
        z0 = float(z.max()) + gap
        available = float(z.min()) + cell[2, 2] - gap - z0
        thick = float(thickness) if thickness is not None else available
        if thick <= 2.0:
            raise ValueError(f"only {available:.1f} A free above the slab; pass thickness= to extend the cell")
        if thick > available:
            cell[2, 2] += thick - available
            atoms.set_cell(cell, scale_atoms=False)
        region_volume = float(np.linalg.norm(np.cross(cell[0], cell[1]))) * thick
        span = np.array([cell[0], cell[1], [0.0, 0.0, thick]])
        origin = np.array([0.0, 0.0, z0])
    else:
        region_volume = abs(float(np.linalg.det(cell)))
        span, origin = cell, np.zeros(3)

    def sample() -> np.ndarray:
        return origin + np.array([rng.random(), rng.random(), rng.random()]) @ span

    env = PeriodicTree(atoms.positions, atoms.cell.array, atoms.pbc)
    if density is not None and env.tree is not None:  # free volume: points farther than `gap` from every atom
        probes = origin + np.random.default_rng(default_seed(seed)).random((FREE_VOLUME_PROBES, 3)) @ span
        region_volume *= float(np.mean(env.tree.query_ball_point(probes, r=gap, return_length=True) == 0))
    target = int(n_molecules) if n_molecules is not None else \
        int(round(float(density) * region_volume * 1e-24 * AVOGADRO / mass))
    if target <= 0:
        raise ValueError(f"no room for {mol.name}: {region_volume:.0f} A^3 of free volume at {density} g/cm^3; "
                         "enlarge the region (thickness=, a larger box) or give n_molecules")

    placed: list[np.ndarray] = []
    trees: dict[str, Optional[PeriodicTree]] = {"placed": None, "recent": None}
    recent: list[np.ndarray] = []

    def tree_of(blocks: list[np.ndarray]) -> PeriodicTree:
        pts = np.vstack(blocks)
        return PeriodicTree(pts, atoms.cell.array, atoms.pbc)

    def clashes(coords: np.ndarray) -> bool:
        for tree in (env, trees["placed"], trees["recent"]):
            if tree is not None and tree.tree is not None:
                if any(tree.tree.query_ball_point(coords, r=tolerance, return_length=True)):
                    return True
        return False

    for _ in range(target):
        for _attempt in range(int(max_attempts)):
            R = Rotation.random(random_state=nprng).as_matrix()
            coords = sample() + tpl @ R.T
            if not clashes(coords):
                placed.append(coords)
                recent.append(coords)
                if len(recent) >= 10:  # fold the recent molecules into the big tree
                    trees["placed"], trees["recent"], recent = tree_of(placed), None, []
                else:
                    trees["recent"] = tree_of(recent)
                break
        else:
            break

    record = {"name": mol.name, "smiles": smiles, "charge": int(mol.charge), "role": "solvent"}
    atoms = register_molecules(atoms, [Atoms(sym, positions=c) for c in placed], [record] * len(placed), ROLE_SOLVENT)
    achieved = len(placed) * mass / AVOGADRO / (region_volume * 1e-24) if region_volume > 0 else 0.0
    msg = (f"[add_solvent] inserted {len(placed)}/{target} {mol.name} molecules "
           f"(~{achieved:.3f} g/cm^3 in {region_volume:.0f} A^3)")
    print(msg if len(placed) == target else msg + " - fewer than requested; lower the density or enlarge the region")
    return atoms
