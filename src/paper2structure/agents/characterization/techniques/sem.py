from ..spec import TechniqueSpec


SEM = TechniqueSpec(
    name="SEM",
    observables=(
        "Top-view micrographs of surface morphology, grains, or particles",
        "Cross-sectional micrographs of films and multilayer stacks",
        "Secondary-electron (topography) vs backscattered-electron (composition) contrast",
        "EDS spectra, elemental maps, and tables of atomic or weight percentages",
    ),
    reading_guidelines=(
        "Read the scale bar first (value and unit) and use it for all sizes; note the imaging mode (SE/BSE) and tilt if given.",
        "Estimate typical grain or particle size from several representative features and give a range, not a single value.",
        "For cross-sections, measure layer thicknesses against the scale bar; note that tilted views foreshorten vertical lengths.",
        "Note morphology: plate-like, needle, cubic, faceted, porous, or pinholes; and whether the coverage is continuous.",
        "For EDS, read element labels and percentages printed on the figure; record which elements are mapped and whether maps are uniform or segregated.",
    ),
    inferable_properties=(
        "Grain / particle size distribution and morphology (crystal habit)",
        "Film thickness, coverage, and layer stacking in devices",
        "Compositional homogeneity and phase segregation (BSE contrast, EDS maps)",
        "Approximate elemental ratios of heavy elements from EDS",
    ),
    interpretation_guidelines=(
        "Crystal habit is a weak hint about structure: plate-like crystals are consistent with a layered structure; do not treat it as proof.",
        "Compare EDS ratios with the nominal composition and report deviations; treat them as semi-quantitative.",
        "BSE contrast differences or segregated EDS maps indicate secondary phases; identify candidates only with support from caption or context.",
        "SEM rarely constrains atomic structure directly; prefer composition and phase-purity constraints and keep their confidence modest.",
    ),
    limitations=(
        "SEM does not give lattice parameters, space group, or atomic positions.",
        "EDS does not detect H, and C, N, O quantification is unreliable; organic content cannot be quantified from EDS.",
        "EDS is semi-quantitative (typically ±a few at.%) and has a micron-scale interaction volume that can include the substrate.",
        "Surface morphology may not represent the bulk.",
    ),
)
