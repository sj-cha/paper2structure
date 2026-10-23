"""Build the S3 stoichiometric CoO core-shell nanoparticle surrogate."""
import numpy as np
from ase.neighborlist import neighbor_list
from paper2structure.tools import build_bulk, carve_nanocrystal, describe, emit

SEED = 0
A = 4.258  # paper-reported CoO cubic lattice parameter (E013), Angstrom
DIAMETER = 20.0  # largest stoichiometric sphere that remains below the 570-atom limit
VACUUM = 15.0
CORE_RADIUS = 10.0  # inner core r <= 1.0 nm
PARTICLE_RADIUS = 15.0

# Complete conventional Fm-3m CoO parent: 4 formula units/cell.
bulk = build_bulk(
    space_group="Fm-3m",
    species=["Co", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    a=A,
    formula_units=4,
)

# A spherical, single-phase cut. The oxidation states let the toolkit select a
# neutral stoichiometric termination if the raw cut is charged.
particle = carve_nanocrystal(
    bulk,
    diameter=DIAMETER,
    facets=None,
    oxidation_states={"Co": 2, "O": -2},
    min_coordination=0,
    center_element="Co",
    vacuum=VACUUM,
    seed=SEED,
)

# Explicit radial labels for downstream magnetic/core-shell workflows.
center = np.mean(particle.positions, axis=0)
r = np.linalg.norm(particle.positions - center, axis=1)
particle.new_array("radial_distance", r)
particle.new_array("region", np.where(r <= CORE_RADIUS, "core", "shell"))
particle.new_array("formal_charge", np.array([2 if s == "Co" else -2 for s in particle.symbols], dtype=int))
particle.info["p2s_kind"] = "nanocrystal"
particle.info["core_radius_A"] = CORE_RADIUS
particle.info["particle_radius_A"] = PARTICLE_RADIUS

# Coordination statistics using the bulk Co-O nearest-neighbour distance.
i, j, d = neighbor_list("ijd", particle, cutoff=2.35)
co_counts = {int(k): 0 for k in range(7)}
for idx, symbol in enumerate(particle.symbols):
    if symbol == "Co":
        n = int(np.sum((i == idx) | (j == idx)))
        co_counts[min(n, 6)] = co_counts.get(min(n, 6), 0) + 1
core_n = int(np.sum(particle.get_array("region") == "core"))
shell_n = len(particle) - core_n
print(describe(particle))
print(f"core_atoms={core_n} shell_atoms={shell_n} total_atoms={len(particle)}")
print(f"Co coordination histogram (0..6, capped): {co_counts}")
print(f"Co:O={sum(s == 'Co' for s in particle.symbols)}:{sum(s == 'O' for s in particle.symbols)}")
print(f"radial extent={r.min():.3f}..{r.max():.3f} A; interior Co-O target={A/2:.3f} A")

emit(
    particle,
    name="S3_CoO_core_shell_3nm",
    plan_id="S3_core_shell_nanoparticle",
    notes=(
        "Rock-salt CoO Fm-3m parent, spherical ~2.0 nm stoichiometric cut; "
        "radial_distance and region arrays label core (<=10 A) and shell (>10 A). "
        "Formal oxidation states are stored in formal_charge; no magnetic moments imposed."
    ),
)
