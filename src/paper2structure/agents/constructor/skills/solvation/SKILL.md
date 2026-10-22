---
name: solvation
description: How to add explicit solvent, solvent mixtures and ions around a slab or particle with add_solvent, choose the amount (density or molecule count) within the atom budget, and keep the cell neutral.
---

- `add_solvent(atoms, smiles, density=..., thickness=..., seed=SEED)` packs rigid, UFF-relaxed
  copies of one molecule with random positions and orientations. Above a slab it fills a layer
  `thickness` A thick starting `gap` A above the top substrate atom, around any adsorbates,
  extending the cell along z when needed; around a particle (or in a porous bulk, or an empty box
  for a pure liquid or electrolyte) it fills the whole cell. Give `density` in g/cm^3 or
  `n_molecules`; with a density, space already taken by adsorbates or earlier species is not
  counted.
- It reports the count and density it reached. When it falls short, enlarge the region, lower
  `tolerance` slightly (default 2.0 A), or report the shortfall as a deviation.
- Mixtures: call add_solvent once per species into the same region, ions and minority species
  first (random insertion into an already dense liquid falls short), each with its molecule count
  from the composition (mole fractions times the total that fits the region at the mixture density).
- Ions: monatomic or molecular ions are SMILES with formal charges (`[Na+]`, `[Cl-]`,
  `F[P-](F)(F)(F)(F)F`); add matching counter-ions so the whole cell stays neutral.
- Solvent fills the vacuum, so the vacuum check only warns for solvated structures; keep a gap of
  a few A above the solvent layer for slabs, or fill the whole periodic box.
- Atom budget: each solvent molecule costs its full atom count (acetonitrile 6, water 3, DMF 12),
  so compute the count before packing.
