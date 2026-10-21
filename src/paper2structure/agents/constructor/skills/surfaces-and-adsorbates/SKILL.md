---
name: surfaces-and-adsorbates
description: How to build periodic slabs with a chosen termination and place adsorbates on them (adsorb or replace mode, monodentate or bidentate, monatomic species such as a transferred proton) with paper2structure.tools, including a complete example script.
---

## Slab

1. `bulk = build_bulk(space_group, species, coords, a=..., c=..., formula_units=...)` from the plan's
   crystal (Wyckoff representatives in the ITA standard setting, origin choice 1).
2. `for t in list_terminations(bulk, miller, min_thickness=...): print(t)` lists every cut with its
   top and bottom layers and the exposed atoms with their coordination, e.g. `O(CN 2) x1, Ti(CN 5) x1`.
   Pick the termination the plan describes by its exposed atoms, not by its index alone.
   Asymmetric cuts appear twice (as cut and flipped), so either face can be on top.
3. `build_slab(bulk, miller, termination=k, min_thickness=..., vacuum=..., supercell=(n1, n2),
   fix_bottom=...)` with the same keyword arguments as list_terminations. `supercell` repeats the
   primitive surface cell; a rectangular or rotated supercell needs your own transformation
   (`ase.build.make_supercell` with an integer matrix on the slab).

## Sites and placement

- Compute sites on the exact structure you place on: atom indices change when atoms are added or
  removed. `surface_sites(slab, kinds=("top", "bridge", "hollow"), elements=[...])`,
  `unique_sites(...)` for one representative per symmetry-distinct kind.
- Bridge or hollow sites across non-bonded atoms (two surface halides, two cations) need
  `max_pair_distance` larger than their separation.
- `Adsorbate.from_smiles(smiles, BindingMotif([...binding elements...], anchor=SMARTS), name=..., seed=SEED)`:
  one binding element for monodentate binding, two for bidentate (chelating or bridging).
  Formal charges in the SMILES encode the protonation state.
- `place_adsorbates(slab, adsorbate, sites, count=..., mode="adsorb" | "replace", bond_length=..., seed=SEED)`:
  coarse-to-fine rotational search with clearance scoring; farthest-point site selection for
  coverage. In "replace" mode the binding group takes the place of the site atom(s) (e.g. a
  deprotonated acid substituting surface halides; a bidentate adsorbate on a bridge replaces both)
  and settles its donors at `bond_length` from the atoms that held the removed ones.
- Monatomic species (a transferred proton `[H+]`, a halide) are Adsorbates too and go through
  place_adsorbates. Place several adsorbate species one after another.
- For geometries place_adsorbates cannot express (a molecule lying flat across several sites, a
  specific tilt, a molecule bridging two particles), position the atoms with your own code and
  register them with `add_molecule`.

## Example

```python
"""Pt(111) p(4x4) with one CO atop (example)."""
from paper2structure.tools import (Adsorbate, BindingMotif, build_bulk, build_slab, describe, emit,
                                   list_terminations, place_adsorbates, surface_sites, unique_sites)

SEED = 0
A_PT = 3.92  # A, lattice constant

bulk = build_bulk("Fm-3m", ["Pt"], [[0, 0, 0]], a=A_PT, formula_units=4)
for t in list_terminations(bulk, (1, 1, 1), min_thickness=8.0):
    print(t)
slab = build_slab(bulk, (1, 1, 1), termination=0, min_thickness=8.0, vacuum=15.0, supercell=(4, 4), fix_bottom=3.0)
for s in unique_sites(surface_sites(slab, kinds=("top", "bridge", "hollow"), elements=["Pt"])):
    print(s)
co = Adsorbate.from_smiles("[C-]#[O+]", BindingMotif(["C"], anchor="[C-]#[O+]"), name="CO", seed=SEED)
model = place_adsorbates(slab, co, surface_sites(slab, kinds=("top",), elements=["Pt"]), count=1,
                      bond_length=1.85, seed=SEED)
print(describe(model))
emit(model, name="pt111_co_top", plan_id="P1")
```
