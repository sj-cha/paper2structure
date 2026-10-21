---
name: custom-structures
description: How to build what paper2structure.tools has no builder for - point defects, substitutions, adatoms, interfaces and heterostructures, 2D materials, stepped or reconstructed surfaces, mixtures - with your own ASE/pymatgen/RDKit/numpy code, keeping the bookkeeping, determinism and checks the workflow relies on.
---

## Ground rules

- Positions come from code: crystal symmetry, lattice vectors, geometry (site positions, normals,
  bond lengths) and searches. Never paste coordinate tables recalled from memory. Literature
  parameters such as lattice constants or bond lengths are fine as named constants at the top of
  the script, with a comment saying where they come from.
- Keep the bookkeeping (structure-toolkit skill): substrate tag 1 (fixed atoms 0), molecules via
  place_adsorbates, add_solvent or `add_molecule(atoms, molecule_atoms, smiles, name, role)`.
- Deterministic: `SEED` at the top, `np.random.default_rng(SEED)`, RDKit `randomSeed=SEED`, the
  `seed=SEED` argument of every toolkit function. No time, environment or leftover-file state.
- Self-contained: one file, imports only installed packages, reads no files.

## Recipes

- Vacancy: pick the site by a rule (e.g. the most exposed surface O, `surface_atoms`), then
  `del atoms[i]`; charged defects need a compensating change, or a note.
- Substitution or dopant: `atoms.symbols[i] = "X"`, with the site chosen by a rule and recorded in
  the notes. Choose the concentration from the plan and report the realized one.
- Adatom: take a hollow or bridge site from `surface_sites` and place the atom along its normal at
  the bond length.
- Interface or heterostructure: `pymatgen.analysis.interfaces.CoherentInterfaceBuilder` (with
  `ZSLGenerator`) finds matched supercells; `ase.build.stack` stacks two slabs with matching
  in-plane cells. Report the strain applied to each side and which side keeps its lattice.
- 2D materials: `ase.build.mx2`, `ase.build.graphene`; stack them on a slab with the in-plane cells
  matched by supercells.
- Stepped or vicinal surfaces: a high Miller index in build_slab (e.g. (2,1,1), (5,3,3)).
  Reconstructions: start from the bulk-terminated slab and move or remove atoms by rule.
- Mixtures: call add_solvent once per species with its share of the molecules; ions go in as
  SMILES (`[Na+]`, `[Cl-]`), and the whole cell should stay neutral.

## Checks before emitting

- `print(describe(atoms))`; `composition_profile(atoms)` for layers or radial shells.
- Measure every target of the plan with code (e.g. shell thickness from the radial positions of
  the shell species, coverage from the molecule count per surface site) and print the values.
- Call the check_structure tool on every emitted structure and fix every FAIL.
