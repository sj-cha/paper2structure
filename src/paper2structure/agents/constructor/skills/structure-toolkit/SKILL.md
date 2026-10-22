---
name: structure-toolkit
description: Reference for paper2structure.tools, the helper library for construction scripts (bulk crystals from Wyckoff positions, molecules in crystal cavities, slab terminations, surface sites, adsorbate placement, nanocrystals, solvent, add_molecule, describe, emit), and the bookkeeping conventions every emitted structure follows. Load it before writing the first script.
---

`from paper2structure.tools import ...` works in every script. The functions cover steps that are
easy to get subtly wrong (termination enumeration, site finding, rotational adsorbate placement,
charge balancing, solvent packing). Use them where they fit and plain ASE, pymatgen, RDKit and
numpy for everything else (see the `custom-structures` skill).

## Conventions of an emitted structure

- An `ase.Atoms`. Per-atom `tags` (OC20 style): 0 fixed substrate, 1 substrate, 2 adsorbate, 3 solvent. Untagged structures are treated as all substrate.
- Per-atom array `mol_id`: -1 for the substrate, one id per molecule.
- One record per molecule (name, SMILES, role), stored encoded in `atoms.info["p2s_molecules"]`;
  read them with `get_molecules(atoms)`. place_adsorbates, add_solvent and add_molecule write them;
  validation compares each molecule's bonds with its SMILES, so molecules without a record are
  only checked as substrate.
- Fixed atoms: `ase.constraints.FixAtoms` (written as selective dynamics in the POSCAR), tag 0.
- Slabs: surface normal along +z, vacuum along z. Particles: centered in a box with vacuum on all
  sides. Builders set `atoms.info["p2s_kind"]`; for custom structures `structure_kind()` infers it
  from the vacuum. Vacuum is measured as the exported POSCAR will have it (periodic along all three
  cell vectors, whatever `atoms.pbc` says), so give a slab real vacuum in its cell.
- Oxidation states: the bond-valence check takes them from the plan (nanocrystals) or
  `carve_nanocrystal`, else charge-balanced common states. A slab or bulk model that needs others
  (a reduced or mixed-valence oxide) can pin them with `atoms.info["p2s_oxidation_states"] = "Ti:+3 O:-2"`.
- `emit(atoms, name=..., plan_id=..., notes=...)` writes `structures/<name>.extxyz` (all per-atom
  data), `structures/<name>.vasp` (POSCAR) and `structures/<name>.json`, and records the emitting
  script in `structures/manifest.json`.
- The validation gate (check_structure, and again on your final report): no atoms closer than {{overlap_scale}} x
  the covalent-radius sum; molecules matching their SMILES; every adsorbate bound to the
  substrate (closest contact within {{binding_scale}} x the covalent sum, never beyond {{detached_scale}} x; an electronegative
  donor on an ionic surface binds a cation, not an anion); vacuum; the atom budget; and the plan's
  checks of the paper's claims ("claim C1": lattice, space group, composition, bond lengths and
  angles, coordination, hydrogen bonds, layer counts), for the systems passed to
  `emit(..., system_ids=[...])`.
- `describe(atoms)` prints composition, cell, molecules, substrate coordination, layer or radial
  composition, vacuum, the closest pair and the binding contacts.

## Reference

{{API_REFERENCE}}
