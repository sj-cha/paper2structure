from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from pymatgen.core import Composition
from pymatgen.core.surface import Slab, SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor

from paper2structure.tools._geometry import (ROLE_FIXED, ROLE_SUBSTRATE, bulk_coordination, clean_arrays,
                                             require_framework)


@dataclass(frozen=True)
class Termination:
    """One way to cut the crystal: which atoms end up in the top and bottom surface layers."""
    index: int
    shift: float
    flipped: bool
    top: str
    bottom: str
    formula: str
    stoichiometric: bool
    symmetric: bool
    n_atoms: int
    thickness: float
    exposed: str = ""  # exposed top-surface atoms per primitive cell with their coordination

    def __str__(self) -> str:
        return (f"termination {self.index}: top layer {self.top}, bottom layer {self.bottom}, "
                f"slab {self.formula} ({self.n_atoms} atoms, {self.thickness:.1f} A), "
                f"{'stoichiometric' if self.stoichiometric else 'non-stoichiometric'}, "
                f"{'symmetric' if self.symmetric else 'asymmetric'}; exposed: {self.exposed or 'n/a'}")


def _exposed(atoms: Atoms, bulk_cn: str) -> str:
    """Exposed top-side atoms and their coordination numbers, e.g. "O(CN 2) x1, Ti(CN 5) x1"."""
    from collections import Counter

    from paper2structure.tools.adsorption_builder import _environment, surface_atoms

    a = atoms.copy()
    a.info = {"p2s_kind": "slab", "p2s_bulk_cn": bulk_cn}
    a.set_array("mol_id", np.full(len(a), -1, dtype=int))
    a.pbc = True
    surf = surface_atoms(a, side="top")
    if not surf:
        return ""
    cn, _, _ = _environment(a, np.arange(len(a)), 1.2)
    counts = Counter(f"{a[i].symbol}(CN {int(cn[i])})" for i in surf)
    return ", ".join(f"{k} x{v}" for k, v in sorted(counts.items()))


def _layer_formula(symbols: Sequence[str]) -> str:
    counts: dict[str, int] = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    return Composition(counts).reduced_formula


def _flip(atoms: Atoms) -> Atoms:
    """Turn the slab upside down with a proper rotation (180 deg about x), keeping the cell
    right-handed and c along +z."""
    atoms = atoms.copy()
    rot = np.array([1.0, -1.0, -1.0])
    a, b, c = np.array(atoms.cell)
    pos = atoms.get_positions() * rot
    # (Ra, -Rb, c) spans the rotated in-plane lattice and stays right-handed
    atoms.set_cell([a * rot, -(b * rot), c], scale_atoms=False)
    atoms.set_positions(pos + c)
    atoms.wrap()
    return atoms


def _candidates(bulk: Atoms, miller, min_thickness: float, symmetrize: bool,
                max_broken_bonds: int, bonds: Optional[dict]) -> list[tuple[Slab, bool]]:
    require_framework(bulk, "build_slab")
    structure = AseAtomsAdaptor.get_structure(bulk)
    # primitive=True without lll_reduce gives the primitive surface cell (lll_reduce keeps e.g. a 2x2
    # cell for fcc(111)); _reduce_in_plane then picks its shortest basis
    gen = SlabGenerator(structure, tuple(int(i) for i in miller), min_slab_size=float(min_thickness),
                        min_vacuum_size=10.0, center_slab=True, primitive=True, lll_reduce=False,
                        reorient_lattice=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slabs = gen.get_slabs(bonds=bonds, max_broken_bonds=max_broken_bonds, symmetrize=symmetrize)
    out: list[tuple[Slab, bool]] = []
    for slab in sorted(slabs, key=lambda s: s.shift):
        out.append((slab, False))
        if not slab.is_symmetric():
            out.append((slab, True))
    if not out:
        raise ValueError(f"pymatgen found no slab for miller index {tuple(miller)} with these settings")
    return out


def _reduce_in_plane(atoms: Atoms) -> Atoms:
    """Shortest (Lagrange-Gauss reduced) in-plane basis, a along x, c along +z, right-handed; so a
    supercell (n1, n2) means n1 x n2 primitive surface cells (e.g. p(2x2))."""
    a, b, c = (np.array(v, dtype=float) for v in atoms.cell.array)
    while True:
        if np.dot(b, b) < np.dot(a, a):
            a, b = b, a
        mu = round(float(np.dot(a, b) / np.dot(a, a)))
        if mu == 0:
            break
        b = b - mu * a
    if np.dot(np.cross(a, b), c) < 0:
        b = -b
    atoms = atoms.copy()
    atoms.set_cell([a, b, c], scale_atoms=False)
    atoms.rotate(-np.degrees(np.arctan2(a[1], a[0])), "z", rotate_cell=True)
    atoms.wrap()
    return atoms


def _slab_atoms(slab: Slab, flipped: bool) -> Atoms:
    atoms = clean_arrays(AseAtomsAdaptor.get_atoms(slab.get_orthogonal_c_slab()))
    atoms.info = {}
    atoms = _reduce_in_plane(atoms)
    if flipped:
        atoms = _flip(atoms)
    atoms.center(axis=2)
    return atoms


def list_terminations(bulk: Atoms, miller: Sequence[int], min_thickness: float = 10.0,
                      symmetrize: bool = False, max_broken_bonds: int = 0,
                      bonds: Optional[dict] = None, layer_tol: float = 1.0) -> list[Termination]:
    """All distinct ways to cut `bulk` along `miller`, described by their top and bottom layers and
    by the exposed top-surface atoms with their coordination numbers (use these to find e.g. the
    termination exposing Ti5c and O2c rather than dangling O).

    Asymmetric slabs appear twice (as cut and flipped), so each face can be the top surface. Pass
    the returned `index` as `termination` to build_slab with the same keyword arguments.
    symmetrize=True asks pymatgen for slabs with identical top and bottom (usually
    non-stoichiometric). `bonds` (e.g. {("Ti", "O"): 2.2}) with `max_broken_bonds` excludes cuts
    that break more than that many of those bonds. `layer_tol` (A) sets how deep the reported
    top/bottom layers reach.
    """
    bulk_formula = Composition(bulk.get_chemical_formula()).reduced_formula
    bulk_cn = bulk_coordination(bulk)
    out = []
    for k, (slab, flipped) in enumerate(_candidates(bulk, miller, min_thickness, symmetrize,
                                                    max_broken_bonds, bonds)):
        atoms = _slab_atoms(slab, flipped)
        atoms.center(vacuum=7.5, axis=2)  # the generator's 10 A of vacuum, so the top really is exposed
        z = atoms.positions[:, 2]
        sym = atoms.get_chemical_symbols()
        top = [s for s, zi in zip(sym, z) if zi >= z.max() - layer_tol]
        bottom = [s for s, zi in zip(sym, z) if zi <= z.min() + layer_tol]
        formula = Composition(atoms.get_chemical_formula()).reduced_formula
        out.append(Termination(index=k, shift=round(float(slab.shift), 4), flipped=flipped,
                               top=_layer_formula(top), bottom=_layer_formula(bottom),
                               formula=formula, stoichiometric=formula == bulk_formula,
                               symmetric=bool(slab.is_symmetric()), n_atoms=len(atoms),
                               thickness=float(z.max() - z.min()), exposed=_exposed(atoms, bulk_cn)))
    return out


def build_slab(bulk: Atoms, miller: Sequence[int], termination: int = 0, min_thickness: float = 10.0,
               vacuum: float = 15.0, supercell: Sequence[int] = (1, 1), fix_bottom: Optional[float] = None,
               symmetrize: bool = False, max_broken_bonds: int = 0, bonds: Optional[dict] = None,
               layer_tol: float = 1.0) -> Atoms:
    """Periodic slab with its surface normal along +z and `vacuum` A of vacuum along z.

    Args:
        bulk: conventional bulk cell (e.g. from build_bulk).
        miller: surface Miller index of the bulk cell.
        termination: index from list_terminations (same keyword arguments).
        min_thickness: minimum slab thickness in A (whole repeat units are kept).
        vacuum: total vacuum thickness along z, in A.
        supercell: in-plane repetition (n1, n2) of the primitive surface cell.
        fix_bottom: thickness (A) of the bottom region whose atoms are fixed (FixAtoms,
            tag 0); None fixes nothing.
    """
    candidates = _candidates(bulk, miller, min_thickness, symmetrize, max_broken_bonds, bonds)
    if not 0 <= termination < len(candidates):
        raise ValueError(f"termination {termination} out of range: {len(candidates)} available "
                         "(see list_terminations)")
    slab, flipped = candidates[termination]
    atoms = _slab_atoms(slab, flipped)
    n1, n2 = (int(supercell[0]), int(supercell[1]))
    atoms = atoms.repeat((n1, n2, 1))
    atoms.center(vacuum=float(vacuum) / 2.0, axis=2)
    atoms.pbc = True
    z = atoms.positions[:, 2]
    tags = np.full(len(atoms), ROLE_SUBSTRATE, dtype=int)
    if fix_bottom:
        fixed = np.where(z < z.min() + float(fix_bottom))[0]
        tags[fixed] = ROLE_FIXED
        atoms.set_constraint(FixAtoms(indices=fixed))
    atoms.set_tags(tags)
    atoms.set_array("mol_id", np.full(len(atoms), -1, dtype=int))
    sym = atoms.get_chemical_symbols()
    top = _layer_formula([s for s, zi in zip(sym, z) if zi >= z.max() - layer_tol])
    # "(1, 1, 1)", not "1 1 1": extxyz would read a space-separated value back as an array
    atoms.info = {"p2s_kind": "slab", "p2s_miller": "(" + ", ".join(str(int(i)) for i in miller) + ")",
                  "p2s_termination": f"{termination} (top layer {top})",
                  "p2s_bulk_cn": bulk_coordination(bulk), "p2s_vacuum": float(vacuum)}
    return atoms
