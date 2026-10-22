from ..spec import TechniqueSpec


# Rendered structures (polyhedral, ball-and-stick, ORTEP, packing views) from any source:
# a single-crystal refinement, a DFT relaxation, or a schematic. They show motif and
# orientation, not coordinates.
STRUCTURE_MODEL = TechniqueSpec(
    name="Structure model",
    observables=(
        "Polyhedral, ball-and-stick, space-filling, or ORTEP (thermal-ellipsoid) renderings of a crystal structure",
        "Viewing direction, given as an axis triad and/or a labelled direction such as [110] or (001)",
        "Atom colour legend, coordination polyhedra, and their connectivity",
        "Layer or slab stacking, spacer or guest molecules and their orientation, unit-cell outlines",
        "Printed annotations: distances, angles, layer thicknesses, layer counts, tilt or symmetry labels",
    ),
    reading_guidelines=(
        "Read the viewing direction from the label or axis triad first, and give it with every observation; "
        "the same structure looks different along different axes.",
        "Map colours to elements only from the legend; do not assign elements from colour or size alone.",
        "Identify the coordination polyhedra (e.g. PbI6 octahedra) and how they connect: corner-, edge-, or face-sharing.",
        "Count repeat units the figure shows explicitly: layers per slab (n), polyhedra along an edge, molecules per cavity.",
        "Describe polyhedral tilts per viewing direction (none, in-phase, out-of-phase); give the Glazer notation only if printed or stated in the caption.",
        "For layered structures, note whether adjacent slabs are eclipsed or offset (e.g. the (1/2, 1/2) shift of Ruddlesden-Popper phases) and what fills the gap.",
        "For organic or molecular components, note orientation relative to the inorganic framework (e.g. chains perpendicular to the slab, "
        "-NH3 end pointing into the cavity) and any drawn disorder (split sites, multiple orientations).",
        "Record printed distances, angles, and labels as printed_label; do not measure lengths from the drawing.",
        "Note the provenance the caption or context gives: experimental refinement (SCXRD, neutron), DFT-relaxed model, or schematic.",
        "Different colours may mark symmetry-distinct units or layers rather than different elements; check the caption.",
    ),
    inferable_properties=(
        "Framework connectivity and dimensionality (0D, 1D, 2D layers, 3D)",
        "Number of inorganic layers per slab and the stacking sequence or offset between slabs",
        "Octahedral tilt pattern, per axis",
        "Orientation and placement of organic cations, spacers, or guest molecules",
        "Printed metric data: layer thickness, interlayer distance, bond lengths and angles",
        "Which crystallographic axis is the stacking direction (and so the long axis of the cell)",
    ),
    interpretation_guidelines=(
        "State whether the rendered structure is experimental, computed, or schematic; it inherits the confidence of its source.",
        "Use the figure for motif, connectivity, and orientation; take lattice parameters and space groups from the text or tables.",
        "Check printed lengths against the lattice parameters in the text (e.g. layer thickness vs. half the stacking-axis length) and report mismatches.",
        "Tilt patterns read from one projection are partial; combine views along different axes before stating a full tilt system.",
        "A structure the authors draw from the literature or a related compound is not evidence for the sample; say so.",
    ),
    limitations=(
        "Atomic coordinates cannot be read from an image; give motifs and orientations, not positions.",
        "Renderings are 2D projections, often in perspective; overlapping atoms and depth are ambiguous.",
        "Hydrogens and disordered atoms are often omitted or simplified; absence in the drawing is not absence in the structure.",
        "Schematic drawings are not to scale.",
        "A rendering does not establish a space group by itself.",
    ),
)
