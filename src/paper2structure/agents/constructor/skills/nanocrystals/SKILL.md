---
name: nanocrystals
description: How to build finite particles - closed-shell metal clusters, faceted or spherical nanocrystals carved from any crystal with charge balancing, core-shell particles (coherent and incoherent shells), supported particles and adsorbate-capped particles - and how to measure their size, shell thickness and composition.
---

## Single-component particles

- Elemental metals: `build_cluster(element, shape, size, lattice_constant)` gives closed-shell
  icosahedra, cuboctahedra, octahedra, truncated octahedra, decahedra or Wulff shapes.
- Any crystal: `carve_nanocrystal(bulk, diameter, facets={(1,0,0): 1.0, (1,1,1): 1.1},
  oxidation_states={"Cd": 2, "Se": -2}, min_coordination=2, center_element=...)` cuts a sphere
  (no facets) or the polyhedron bounded by the symmetry-equivalent facet planes, prunes dangling
  atoms and removes surface atoms until the formal charges sum to zero.
- Atom counts grow with the cube of the diameter: about 30-70 atoms per nm^3 for inorganic
  crystals, so a 5 nm particle is several thousand atoms before adsorbates. Check the budget first.

## Core-shell particles

Coherent shell (core and shell share a sublattice: metal@metal, CdSe/CdS, CdSe/ZnS, PbS/CdS):
1. Build the whole particle, core plus shell, on one lattice: carve or build it at the final
   diameter from the core crystal, or from a lattice between the two, and state which in the notes.
2. Reassign species by region. Use the distance from the center for spherical particles, or the
   depth below the surface in atomic layers for faceted ones (peel the outermost layers by
   coordination number or by distance to the facet planes). Map each sublattice separately
   (cation to cation, anion to anion, e.g. Cd -> Zn and Se -> S).
3. Report the lattice mismatch the shell carries: it is strained to the core until an MLIP
   relaxation releases it.
4. Measure what you built: core diameter, shell thickness in A and in monolayers along the facet
   normals, and the composition by region (`composition_profile(atoms)` prints radial shells).

Incoherent shell (different structure types, e.g. a metal on an oxide): carve the core from crystal
A, carve a particle of the outer diameter from crystal B, delete the B atoms closer to any core
atom than the sum of covalent radii plus a small gap, merge, then check the interface contacts
(`describe`, check_structure). Say in the notes that the interface is unrelaxed.

Charge neutrality applies to the whole particle for ionic materials: balance by trimming
under-coordinated surface atoms of the excess species, or with anionic adsorbates (each carboxylate or
halide balances one excess cation charge).

## Supported particles

Build the support slab (surfaces-and-adsorbates skill) and the particle separately, orient the
particle facet that touches the support parallel to the surface, and place it at a contact distance
of about the sum of covalent radii above the top layer. Make the in-plane supercell large enough
for at least 10 A between periodic images of the particle and keep the vacuum above it.

## Adsorbates on particles

`surface_sites` and `place_adsorbates` work on particles: site normals point radially outward. Place
adsorbates with `count` or a coverage. For a dense shell of long chains, use a truncated adsorbate (e.g.
acetate for oleate) when the plan allows it, and say so in the deviations.
