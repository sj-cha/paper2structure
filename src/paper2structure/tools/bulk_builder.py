from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import spglib
from ase import Atoms
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from paper2structure.thresholds import OVERLAP_SCALE  # build_bulk rejects overlaps the gate would fail
from paper2structure.tools._geometry import ROLE_SUBSTRATE, cation_elements, clean_arrays, coordination_pairs, rcov
from paper2structure.tools._symmetry import Setting, hall_operations, orbit, resolve, settings_of

MERGE_TOL = 0.05  # A: symmetry images closer than this are one site (rounded special positions)
LENGTH_TOL = 1e-3  # relative: a given lattice length may differ this much from the one the setting fixes
ANGLE_TOL = 0.1  # deg: ... and an angle this much
SETTING_HINT = ("Coordinates and lattice must be in the setting the symbol names: without a suffix the ITA "
                "standard setting, origin choice 1 for the groups with two origins (e.g. anatase I4_1/amd has Ti at "
                "(0, 0, 0) and O at (0, 0, ~0.208)) and hexagonal axes for rhombohedral groups; write ':2' or ':R' "
                "for the other one (e.g. 'Fd-3m:2', 'R-3m:R').")


def _complete_lattice(setting: Setting, a: float, b, c, alpha, beta, gamma) -> tuple[float, ...]:
    """The six lattice parameters, with those the setting fixes filled in. A given parameter that
    contradicts the setting (b != a in a tetragonal cell, gamma != 90 with unique axis b) raises
    rather than being overwritten."""
    system, values = setting.crystal_system, dict(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
    given = dict(values)
    right = dict(alpha=90.0, beta=90.0, gamma=90.0)
    if system == "cubic":
        rule = dict(b="a", c="a", **right)
    elif system == "tetragonal":
        rule = dict(b="a", **right)
    elif system == "trigonal" and setting.rhombohedral_axes:
        rule = dict(b="a", c="a", beta="alpha", gamma="alpha")
    elif system in ("trigonal", "hexagonal"):
        rule = dict(b="a", alpha=90.0, beta=90.0, gamma=120.0)
    elif system == "orthorhombic":
        rule = right
    elif system == "monoclinic":
        free = {"a": "alpha", "b": "beta", "c": "gamma"}[setting.unique_axis]
        rule = {k: v for k, v in right.items() if k != free}
    else:
        rule = {}
    conflicts = []
    for name, want in rule.items():
        target = values[want] if isinstance(want, str) else want
        if target is None:
            continue
        tol = ANGLE_TOL if name in right else LENGTH_TOL * abs(target)
        if given[name] is not None and abs(float(given[name]) - float(target)) > tol:
            conflicts.append(f"{name} must {'equal ' + want if isinstance(want, str) else f'be {target:g}'} "
                             f"(given {float(given[name]):g})")
        values[name] = float(target)
    where = f"{setting.label} is {system}" + (f" with unique axis {setting.unique_axis}" if setting.unique_axis else "")
    if conflicts:
        raise ValueError(f"{where}: {'; '.join(conflicts)}. {_other_settings(setting, given)}".strip())
    missing = [n for n in ("b", "c", "alpha", "beta", "gamma") if values[n] is None]
    if missing:
        raise ValueError(f"{where}: lattice parameter(s) {', '.join(missing)} must be given")
    return tuple(float(values[n]) for n in ("a", "b", "c", "alpha", "beta", "gamma"))


def _other_settings(setting: Setting, values: dict) -> str:
    """Where the given lattice fits another setting of the same group, name it."""
    system = setting.crystal_system
    if system == "monoclinic":
        oblique = [ax for ax, ang in zip("abc", ("alpha", "beta", "gamma"))
                   if values[ang] is not None and abs(values[ang] - 90.0) > ANGLE_TOL]
        if len(oblique) == 1:
            fits = [s.full for s in settings_of(setting.number) if s.unique_axis == oblique[0]]
            return f"For unique axis {oblique[0]} write the symbol of that setting, e.g. {', '.join(map(repr, fits[:3]))}."
    if system == "trigonal" and setting.choice in ("H", "R"):
        return ("Rhombohedral axes (a = b = c, alpha = beta = gamma): add ':R' to the symbol." if setting.choice == "H"
                else "Hexagonal axes (a = b, c, gamma = 120): leave out ':R'.")
    return ""


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
        space_group: Hermann-Mauguin symbol of the setting the coordinates are in ("Fm-3m",
            "P4_2/mnm", "P6_3mc", "Pbnm", "P2_1/n", "C2cb"), with ':2' for origin choice 2 or ':R'
            for rhombohedral axes ("Fd-3m:2", "R-3m:R"), or the international number (standard
            setting).
        species: element of each Wyckoff representative.
        coords: fractional coordinates of each representative (the asymmetric unit). A coordinate
            a rounding error off a special position (1/3 written as 0.3333) is placed on it.
        a, b, c, alpha, beta, gamma: lattice parameters in Angstrom/degrees. Parameters the setting
            fixes may be omitted (cubic: only a; tetragonal/hexagonal: a and c; rhombohedral axes: a
            and alpha; monoclinic: a, b, c and the angle of the unique axis).
        formula_units: expected formula units per conventional cell (e.g. 4 for anatase TiO2);
            checked when given.
        check_bonded: require a bonded neighbour for every atom; False for the framework of a
            crystal whose counter-ions are molecules (the halide lattice of a molecular salt).

    Without a suffix a symbol means the ITA standard choices: origin choice 1 for groups with two
    origins and hexagonal axes for rhombohedral groups. A non-standard setting ("Pbnm", "Pcab",
    "Cc2m", "P 1 1 2_1/b") takes the lattice in that setting. Raises ValueError when the lattice
    contradicts the setting, when symmetry-generated atoms overlap (usually coordinates in another
    setting), or when the cell holds a different number of formula units. atoms.info records the
    symbol as given ("p2s_space_group") and the exact setting ("p2s_hall_number").
    """
    if len(species) != len(coords):
        raise ValueError(f"{len(species)} species but {len(coords)} coordinate triples")
    setting = resolve(space_group)
    numeric = isinstance(space_group, (int, np.integer)) or str(space_group).strip().isdigit()
    symbol = setting.short if numeric else str(space_group).strip()
    lattice = Lattice.from_parameters(*_complete_lattice(setting, a, b, c, alpha, beta, gamma))
    rotations, translations = hall_operations(setting.hall)
    sites = [(s, p) for s, x in zip(species, coords)
             for p in orbit(x, rotations, translations, MERGE_TOL, lattice.matrix)]
    structure = Structure(lattice, [s for s, _ in sites], [p for _, p in sites])

    dm = structure.distance_matrix
    np.fill_diagonal(dm, np.inf)
    r = rcov([str(s.specie) for s in structure])
    ratio = dm / (r[:, None] + r[None, :])
    i, j = np.unravel_index(int(np.argmin(ratio)), ratio.shape)
    if ratio[i, j] < OVERLAP_SCALE:
        raise ValueError(f"symmetry-generated atoms overlap: {structure[i].specie}-{structure[j].specie} at "
                         f"{dm[i, j]:.2f} A, below {OVERLAP_SCALE} x their covalent-radius sum ({len(structure)} "
                         f"atoms generated in {setting.label}). Check the Wyckoff coordinates: overlapping images "
                         f"usually mean coordinates in another setting, or a fraction such as 1/3 written with too "
                         f"few decimals (0.33). {SETTING_HINT}")

    reduced, n_fu = structure.composition.get_reduced_composition_and_factor()
    if formula_units is not None and int(round(n_fu)) != int(formula_units):
        raise ValueError(f"the cell holds {n_fu:g} formula units of {reduced.reduced_formula} "
                         f"({len(structure)} atoms), expected {formula_units}. {SETTING_HINT}")
    if check_bonded:
        _check_bonded(AseAtomsAdaptor.get_atoms(structure))

    data = spglib.get_symmetry_dataset((lattice.matrix, structure.frac_coords, [s.specie.Z for s in structure]),
                                       symprec=0.01)
    found, n_found = (int(data.number), len(data.rotations)) if data is not None else (1, 1)
    atoms = clean_arrays(AseAtomsAdaptor.get_atoms(structure))
    atoms.info = {"p2s_kind": "bulk", "p2s_space_group": symbol, "p2s_hall_number": setting.hall,
                  "p2s_lattice": [float(x) for x in atoms.cell.array.ravel()]}
    if found != setting.number:
        why = ("the lattice parameters or the coordinates break the requested symmetry" if n_found < len(rotations)
               else "internal coordinates at special values raise the symmetry")
        note = f"requested {symbol} (#{setting.number}) but the generated cell has symmetry #{found}: {why}"
        atoms.info["p2s_warning"] = note
        print(f"[build_bulk] note: {note}")
    atoms.set_tags(np.full(len(atoms), ROLE_SUBSTRATE, dtype=int))
    atoms.set_array("mol_id", np.full(len(atoms), -1, dtype=int))
    return atoms
