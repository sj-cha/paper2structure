"""Build the two CdSe/ZnS core-shell hypotheses in plans P1 and P2."""
import numpy as np
from ase import Atoms
from paper2structure.tools import (BindingMotif, Ligand, Site, add_molecule,
    build_bulk, carve_nanocrystal, describe, emit, place_ligands)

SEED = 0
CORE_DIAMETER = 28.0       # plan: 2.8 nm CdSe core
OUTER_DIAMETER = 39.0       # plan: 3.9 nm particle
VACUUM = 15.0               # plan: at least 15 A between periodic images
SHELL_RADIUS = OUTER_DIAMETER / 2
CORE_RADIUS = CORE_DIAMETER / 2
LIGAND_COUNT = 80            # reduced-complexity proxy population in both plans
SMILES = "CCCC/C=C/CC(=O)[O-]"


def core_shell():
    """Carve one CdSe lattice and relabel its radial outer region as ZnS."""
    bulk = build_bulk("F-43m", ["Cd", "Se"],
                      [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
                      a=6.08, formula_units=4)
    # The single CdSe lattice is deliberately used for both regions: ZnS is
    # epitaxially registered and therefore coherently strained to CdSe.
    particle = carve_nanocrystal(bulk, OUTER_DIAMETER, facets=None,
                                 oxidation_states=None, min_coordination=1,
                                 vacuum=VACUUM, seed=SEED)
    center = particle.cell.array.sum(axis=0) / 2.0
    radial = np.linalg.norm(particle.positions - center, axis=1)
    symbols = []
    for symbol, radius in zip(particle.get_chemical_symbols(), radial):
        if radius <= CORE_RADIUS:
            symbols.append(symbol)
        else:
            symbols.append("Zn" if symbol == "Cd" else "S")
    particle.set_chemical_symbols(symbols)
    particle.info["p2s_kind"] = "nanocrystal"
    return particle, center, radial


def ligand_and_sites(particle, center, radial, bridge=False):
    ligand = Ligand.from_smiles(
        SMILES, BindingMotif(["O", "O"], anchor="[C](=O)[O-]"),
        name="truncated-oleate-carboxylate-proxy", seed=SEED, optimize=False)
    # Exposed Zn candidates are selected geometrically at the outer two A of
    # the shell. Farthest-point sampling distributes the 80 anchors uniformly.
    zn = [i for i, (s, r) in enumerate(zip(particle.symbols, radial))
          if s == "Zn" and r > SHELL_RADIUS - 2.5]
    if bridge:
        pairs = []
        for a, i in enumerate(zn):
            for j in zn[a + 1:]:
                d = np.linalg.norm(particle.positions[i] - particle.positions[j])
                if 3.0 <= d <= 4.7:
                    pairs.append((i, j))
        # Greedy farthest-point selection of pair midpoints, avoiding reuse.
        chosen = []
        used = set()
        while len(chosen) < LIGAND_COUNT:
            available = [(i, j) for i, j in pairs if i not in used and j not in used]
            if not available:
                raise RuntimeError("fewer than 80 non-overlapping Zn bridge pairs")
            if not chosen:
                pair = max(available, key=lambda q: radial[q[0]] + radial[q[1]])
            else:
                pair = max(available, key=lambda q: min(
                    np.linalg.norm((particle.positions[q[0]] + particle.positions[q[1]]) / 2
                                   - (particle.positions[x[0]] + particle.positions[x[1]]) / 2)
                    for x in chosen))
            chosen.append(pair); used.update(pair)
        sites = []
        for i, j in chosen:
            pos = (particle.positions[i] + particle.positions[j]) / 2
            normal = pos - center; normal /= np.linalg.norm(normal)
            sites.append(Site("bridge", (i, j), ("Zn", "Zn"), pos, normal, (0, 0)))
        return ligand, sites
    chosen = [max(zn, key=lambda i: radial[i])]
    while len(chosen) < LIGAND_COUNT:
        available = [i for i in zn if i not in chosen]
        chosen.append(max(available, key=lambda i: min(
            np.linalg.norm(particle.positions[i] - particle.positions[j]) for j in chosen)))
    sites = [Site("top", (i,), ("Zn",), particle.positions[i],
                   (particle.positions[i] - center) / radial[i], (0,)) for i in chosen]
    return ligand, sites


def build(plan_id, name, bridge):
    particle, center, radial = core_shell()
    ligand, sites = ligand_and_sites(particle, center, radial, bridge=bridge)
    if bridge:
        # The toolkit's bridge optimizer expects slab-style lateral offsets. For
        # this spherical bridge motif, preserve the ligand conformer explicitly
        # and orient its O-O axis along each Zn-Zn pair.
        for m, site in enumerate(sites):
            i, j = site.indices
            pair = particle.positions[j] - particle.positions[i]
            ex = pair / np.linalg.norm(pair)
            ez = site.normal - np.dot(site.normal, ex) * ex
            ez /= np.linalg.norm(ez)
            ey = np.cross(ez, ex)
            rot = np.column_stack((ex, ey, ez))
            donor_half = np.linalg.norm(ligand.atoms.positions[ligand.binding_atoms[1]]
                                        - ligand.atoms.positions[ligand.binding_atoms[0]]) / 2
            half_pair = np.linalg.norm(pair) / 2
            rise = np.sqrt(max(0.5, 2.2**2 - (half_pair - donor_half)**2)) + 2.0
            anchor = site.position + site.normal * rise
            mol = ligand.atoms.copy()
            mol.positions = mol.positions @ rot.T + anchor
            particle = add_molecule(particle, mol, SMILES,
                                    "truncated-oleate-carboxylate-proxy",
                                    role="adsorbate", site="bridge Zn-Zn")
        # add_molecule preserves the substrate cell; enlarge it after the
        # outward bridge placement so the ligand tails retain 15 A vacuum.
        old_center = particle.cell.array.sum(axis=0) / 2.0
        half_extent = np.max(np.abs(particle.positions - old_center), axis=0)
        lengths = 2.0 * (half_extent + VACUUM)
        particle.positions += lengths / 2.0 - old_center
        particle.set_cell(np.diag(lengths))
    else:
        particle = place_ligands(particle, ligand, sites, count=LIGAND_COUNT,
                                 mode="adsorb", bond_length=2.2, min_clearance=1.0,
                                 seed=SEED)
    print(describe(particle))
    notes = ("Coherent ZnS shell is relabelled on the CdSe zinc-blende lattice; "
             "sharp radial CdSe/ZnS interface, 15 A vacuum, C8 carboxylate proxies. "
             + ("Each proxy bridges a distinct Zn-Zn pair." if bridge
                else "Each proxy chelates one surface Zn."))
    emit(particle, name=name, plan_id=plan_id, notes=notes)


if __name__ == "__main__":
    build("P1", "P1_cdse_zns_chelate", bridge=False)
    build("P2", "P2_cdse_zns_bridge", bridge=True)
