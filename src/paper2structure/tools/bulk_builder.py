from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
from ase import Atoms
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup

from paper2structure.tools._geometry import ROLE_SUBSTRATE, cation_elements, clean_arrays, coordination_pairs, rcov
from paper2structure.tools._symmetry import orbit, space_group_number, symmetry_operations

SETTING_HINT = ("pymatgen uses the ITA standard setting: origin choice 1 for the space groups with two "
                "origins (e.g. anatase I4_1/amd has Ti at (0, 0, 0) and O at (0, 0, ~0.208)) and "
                "hexagonal axes for rhombohedral groups.")


def _complete_lattice(sg: SpaceGroup, a: float, b, c, alpha, beta, gamma) -> Lattice:
    system = sg.crystal_system
    if system == "cubic":
        b, c, alpha, beta, gamma = a, a, 90.0, 90.0, 90.0
    elif system == "tetragonal":
        b, alpha, beta, gamma = a, 90.0, 90.0, 90.0
    elif system in ("hexagonal", "trigonal"):
        b, alpha, beta, gamma = a, 90.0, 90.0, 120.0
    elif system == "orthorhombic":
        alpha, beta, gamma = 90.0, 90.0, 90.0
    elif system == "monoclinic":
        alpha = 90.0 if alpha is None else alpha
        gamma = 90.0 if gamma is None else gamma
    missing = [n for n, v in zip("b c alpha beta gamma".split(), (b, c, alpha, beta, gamma)) if v is None]
    if missing:
        raise ValueError(f"{sg.symbol} is {system}: lattice parameter(s) {', '.join(missing)} must be given")
    return Lattice.from_parameters(a, b, c, alpha, beta, gamma)


def _check_bonded(atoms: Atoms, scale: float = 1.3) -> None:
    """Every atom of a crystal needs a bonded neighbour (metal-metal contacts in a compound do not
    count). An isolated species means the Wyckoff positions put elements on the wrong sites, e.g.
    Pb and Cs swapped in a perovskite: formula, formula units and overlaps all look fine, but Cs
    ends up octahedrally coordinated and Pb has no Br."""
    if len(atoms) < 2:
        return
    i, _, _ = coordination_pairs(atoms, scale)
    cn = np.bincount(i, minlength=len(atoms))
    lonely = sorted({atoms[k].symbol for k in np.where(cn == 0)[0]})
    if not lonely:
        return
    cations = cation_elements(atoms.get_chemical_symbols())
    details = []
    for el in lonely:
        k = next(k for k in range(len(atoms)) if atoms[k].symbol == el and cn[k] == 0)
        d = atoms.get_distances(k, [j for j in range(len(atoms)) if j != k], mic=True)
        others = [j for j in range(len(atoms)) if j != k]
        partners = [n for n, j in enumerate(others) if not (el in cations and atoms[j].symbol in cations)]
        n = min(partners, key=lambda n: d[n]) if partners else int(np.argmin(d))
        details.append(f"{el} (nearest {atoms[others[n]].symbol} at {d[n]:.2f} A)")
    raise ValueError(f"no bonded neighbour for {', '.join(details)}: the Wyckoff positions probably put elements "
                     "on the wrong sites; check which site each element occupies (e.g. in a perovskite ABX3 "
                     "the X anions must surround B octahedrally). " + SETTING_HINT)


def build_bulk(space_group: str | int, species: Sequence[str], coords: Sequence[Sequence[float]],
               a: float, b: Optional[float] = None, c: Optional[float] = None,
               alpha: Optional[float] = None, beta: Optional[float] = None,
               gamma: Optional[float] = None, formula_units: Optional[int] = None,
               check_bonded: bool = True) -> Atoms:
    """Conventional cell of a crystal from its space group, Wyckoff representatives and lattice.

    Args:
        space_group: Hermann-Mauguin symbol as pymatgen spells it ("Fm-3m", "P4_2/mnm",
            "I4_1/amd", "P6_3mc") or the international number.
        species: element of each Wyckoff representative.
        coords: fractional coordinates of each representative (the asymmetric unit).
        a, b, c, alpha, beta, gamma: lattice parameters in Angstrom/degrees. Parameters fixed by
            the crystal system may be omitted (cubic: only a; tetragonal/hexagonal: a and c).
        formula_units: expected formula units per conventional cell (e.g. 4 for anatase TiO2);
            checked when given.
        check_bonded: require a bonded neighbour for every atom; False for the framework of a
            crystal whose counter-ions are molecules (the halide lattice of a molecular salt).

    Coordinates follow the setting of the symbol: the standard setting pymatgen uses (origin
    choice 1 for groups with two origin choices, hexagonal axes for rhombohedral groups), or a
    non-standard setting ("Pbnm", "Pcab", "Cc2m", "C2cb"), with the lattice in that setting. Raises ValueError when the lattice does not
    fit the space group, when symmetry-generated atoms overlap (usually wrong coordinates or the
    other origin choice), or when the cell holds a different number of formula units.
    """
    numeric = isinstance(space_group, int) or str(space_group).isdigit()
    try:
        sg = SpaceGroup.from_int_number(int(space_group)) if numeric else SpaceGroup(str(space_group))
    except ValueError:  # a setting pymatgen does not know (e.g. "C2cb"): spglib's operations in that setting
        sg = None
    if len(species) != len(coords):
        raise ValueError(f"{len(species)} species but {len(coords)} coordinate triples")
    number = sg.int_number if sg is not None else space_group_number(space_group)
    symbol = sg.symbol if sg is not None else str(space_group)
    lattice = _complete_lattice(SpaceGroup.from_int_number(number), a, b, c, alpha, beta, gamma)
    if sg is not None:
        structure = Structure.from_spacegroup(sg.int_number if numeric else str(space_group), lattice, list(species),
                                              [list(map(float, x)) for x in coords])
    else:
        rotations, translations = symmetry_operations(space_group)
        sites = [(s, p) for s, x in zip(species, coords) for p in orbit(x, rotations, translations)]
        structure = Structure(lattice, [s for s, _ in sites], [p for _, p in sites])

    dm = structure.distance_matrix
    np.fill_diagonal(dm, np.inf)
    r = rcov([str(s.specie) for s in structure])
    ratio = dm / (r[:, None] + r[None, :])
    i, j = np.unravel_index(int(np.argmin(ratio)), ratio.shape)
    if ratio[i, j] < 0.5:
        raise ValueError(f"symmetry-generated atoms overlap: {structure[i].specie}-{structure[j].specie} at "
                         f"{dm[i, j]:.2f} A ({len(structure)} atoms generated). Check the Wyckoff "
                         f"coordinates. {SETTING_HINT}")

    reduced, n_fu = structure.composition.get_reduced_composition_and_factor()
    if formula_units is not None and int(round(n_fu)) != int(formula_units):
        raise ValueError(f"the cell holds {n_fu:g} formula units of {reduced.reduced_formula} "
                         f"({len(structure)} atoms), expected {formula_units}. {SETTING_HINT}")
    if check_bonded:
        _check_bonded(AseAtomsAdaptor.get_atoms(structure))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found = SpacegroupAnalyzer(structure, symprec=0.01).get_space_group_number()
    atoms = clean_arrays(AseAtomsAdaptor.get_atoms(structure))
    atoms.info = {"p2s_kind": "bulk", "p2s_space_group": symbol}
    if found != number:
        note = (f"requested {symbol} (#{number}) but the generated cell has symmetry #{found}; "
                "internal coordinates at special values raise the symmetry")
        atoms.info["p2s_warning"] = note
        print(f"[build_bulk] note: {note}")
    atoms.set_tags(np.full(len(atoms), ROLE_SUBSTRATE, dtype=int))
    atoms.set_array("mol_id", np.full(len(atoms), -1, dtype=int))
    return atoms
