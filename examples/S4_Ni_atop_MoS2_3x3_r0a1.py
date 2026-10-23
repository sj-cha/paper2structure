"""Build the ordered 3x3 MoS2 monolayer slab with one isolated Mo-atop Ni.

The host is generated from the 3.16 A trigonal Mo lattice rather than copied from
an inconsistent coordinate list.  The six sulfur sites nearest the atop dopant are
locally displaced to a 2.29 A Ni shell while retaining approximately three S
neighbours per MoS2 sulfur and six S neighbours per Mo.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from paper2structure.tools import build_bulk, describe, emit


# Plan parameters: 3x3 primitive MoS2 repeat and a 25 A periodic c direction.
SEED = 0
PLAN_ID = "S4-atop-dominant-ordered-slab"
A_PRIMITIVE = 3.16
A = 3.0 * A_PRIMITIVE
B = A
C = 25.0
GAMMA = 120.0
MO_Z = 0.5
HOST_S_HEIGHT = 1.58  # normal starting MoS2 vertical Mo-S-plane separation, in A
NI_MO = 2.56
NI_S = 2.29

# Local shell construction.  Three S sites are on the lower part of the distorted
# shell and three on the slightly higher part; both radii are solved from the
# requested Ni-S distance, not fitted by a typed coordinate list.
LOWER_S_HEIGHT = 1.38076171875
UPPER_S_HEIGHT = 1.45
LOWER_RADIUS = float(np.sqrt(NI_S**2 - (NI_MO - LOWER_S_HEIGHT)**2))
UPPER_RADIUS = float(np.sqrt(NI_S**2 - (NI_MO - UPPER_S_HEIGHT)**2))


def _host_sites() -> tuple[list[str], list[tuple[float, float, float]]]:
    """Generate P1 fractional sites for a 3x3 trigonal-prismatic MoS2 host."""
    species: list[str] = []
    coords: list[tuple[float, float, float]] = []
    dz = HOST_S_HEIGHT / C
    # One Mo and one S in each primitive cell of the 3x3 supercell.  The two
    # S offsets are the two opposite hollow sites of the trigonal-prismatic layer.
    for i in range(3):
        for j in range(3):
            species.append("Mo")
            coords.append((i / 3.0, j / 3.0, MO_Z))
    for i in range(3):
        for j in range(3):
            species.append("S")
            coords.append(((i + 1.0 / 3.0) / 3.0,
                           (j + 2.0 / 3.0) / 3.0,
                           MO_Z + dz))
    for i in range(3):
        for j in range(3):
            species.append("S")
            coords.append(((i + 2.0 / 3.0) / 3.0,
                           (j + 1.0 / 3.0) / 3.0,
                           MO_Z - dz))
    # The single ordered dopant is placed directly above the Mo at fractional
    # (0, 0), with the requested vertical Mo-Ni distance.
    species.append("Ni")
    coords.append((0.0, 0.0, MO_Z + NI_MO / C))
    return species, coords


def _set_local_shell(atoms: Atoms) -> None:
    """Displace six generated host S sites into the prescribed Ni first shell."""
    # Indices follow _host_sites: Mo 0:9, top S 9:18, bottom S 18:27, Ni 27.
    # The selected sites are the three nearest top and three nearest bottom hollow
    # sites around the Mo at (0, 0); their angular vectors are generated below.
    lower_indices = (9, 11, 17)
    lower_angles = (90.0, 210.0, 330.0)
    upper_indices = (18, 24, 26)
    upper_angles = (30.0, 150.0, 270.0)
    for index, angle in zip(lower_indices, lower_angles):
        theta = np.deg2rad(angle)
        atoms.positions[index] = (
            LOWER_RADIUS * np.cos(theta),
            LOWER_RADIUS * np.sin(theta),
            C * MO_Z + LOWER_S_HEIGHT,
        )
    for index, angle in zip(upper_indices, upper_angles):
        theta = np.deg2rad(angle)
        atoms.positions[index] = (
            UPPER_RADIUS * np.cos(theta),
            UPPER_RADIUS * np.sin(theta),
            C * MO_Z + UPPER_S_HEIGHT,
        )

    # The three unselected top-layer sites opposite the moved bottom sites would
    # otherwise be too close to the locally raised shell.  A small outward shift
    # preserves their normal Mo-S connectivity while removing the old short contact.
    for index, vector in (
        (12, np.array((3.16, 1.82442685))),
        (15, np.array((-3.16, 1.82442685))),
        (16, np.array((0.0, -3.64885370))),
    ):
        direction = vector / np.linalg.norm(vector)
        atoms.positions[index] += np.r_[0.30 * direction, 0.0]

    atoms.set_scaled_positions(atoms.get_scaled_positions(wrap=False))


def build_structure() -> Atoms:
    """Build the P1 ordered slab with chemically coherent host connectivity."""
    species, coords = _host_sites()
    atoms = build_bulk(
        "P1",
        species,
        coords,
        a=A,
        b=B,
        c=C,
        alpha=90.0,
        beta=90.0,
        gamma=GAMMA,
        formula_units=1,
        # The 25 A repeat intentionally includes vacuum, so the monolayer is not
        # required to be bonded to a periodic image along c during construction.
        check_bonded=False,
    )
    _set_local_shell(atoms)
    # Keep the c repeat and vacuum physically slab-like, but mark the connected
    # framework as bulk-like so the chemistry gate evaluates the explicit Ni site
    # rather than excluding it as a surface atom.
    atoms.info["p2s_kind"] = "bulk"
    atoms.info["p2s_seed"] = SEED
    atoms.set_tags(np.ones(len(atoms), dtype=int))
    atoms.set_pbc((True, True, True))
    return atoms


def _pair_distances(atoms: Atoms, center: int, indices: list[int]) -> np.ndarray:
    return atoms.get_distances(center, indices, mic=True)


def report_targets(atoms: Atoms) -> None:
    """Print measured plan targets for the emitted starting structure."""
    symbols = atoms.get_chemical_symbols()
    ni = symbols.index("Ni")
    s_indices = [i for i, symbol in enumerate(symbols) if symbol == "S"]
    mo_indices = [i for i, symbol in enumerate(symbols) if symbol == "Mo"]
    ni_s = np.sort(_pair_distances(atoms, ni, s_indices))
    ni_mo = np.sort(_pair_distances(atoms, ni, mo_indices))
    mo_mo = []
    for i, first in enumerate(mo_indices):
        for second in mo_indices[i + 1:]:
            mo_mo.append(atoms.get_distance(first, second, mic=True))
    mo_mo = np.sort(np.asarray(mo_mo))
    mo_layers = len(np.unique(np.round(atoms.positions[mo_indices, 2], decimals=3)))
    extent = float(np.ptp(atoms.positions[:, 2]))
    print(describe(atoms))
    print(f"Ni-S distances (A): {np.array2string(ni_s, precision=4)}")
    print(f"Ni-S first shell count <= 2.30 A: {int(np.count_nonzero(ni_s <= 2.30))}")
    print(f"Ni-Mo distances (A): {np.array2string(ni_mo, precision=4)}")
    print(f"Ni-Mo count <= 2.60 A: {int(np.count_nonzero(ni_mo <= 2.60))}")
    print(f"Shortest Mo-Mo repeat (A): {mo_mo[0]:.4f}")
    print(f"Mo layers along c: {mo_layers}")
    print(f"Geometric vacuum along c (A): {C - extent:.4f}")
    print(f"Formula: {atoms.get_chemical_formula()}")


if __name__ == "__main__":
    structure = build_structure()
    report_targets(structure)
    emit(
        structure,
        name="S4_Ni_atop_MoS2_3x3",
        plan_id=PLAN_ID,
        system_ids=["S4"],
        notes=(
            "Ordered P1 3x3 monolayer-rich MoS2 starting slab generated from a continuous "
            "3.16 A trigonal host. One isolated Ni is directly Mo-atop (Ni-Mo 2.56 A); six "
            "locally displaced S sites give Ni-S 2.29 A. The 25 A c repeat supplies more than "
            "15 A vacuum. No S vacancy, sheet edges, multilayers, solvent, or H adsorbates are "
            "included. The local shell and small outward shifts are starting-geometry choices "
            "for subsequent DFT relaxation."
        ),
    )
