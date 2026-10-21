from ..spec import TechniqueSpec


TEM = TechniqueSpec(
    name="TEM",
    aliases=(
        "transmission electron microscopy", "HRTEM", "HR-TEM", "STEM", "HAADF-STEM",
        "SAED", "electron diffraction", "cryo-TEM",
    ),
    observables=(
        "Bright-field / dark-field micrographs showing particle or grain shape and size",
        "High-resolution (HRTEM) images with lattice fringes, often with FFT insets",
        "Selected-area electron diffraction (SAED) patterns: spots, rings, or diffuse halos",
        "STEM-HAADF images with atomic-number (Z) contrast",
        "EDS/EELS elemental maps and line scans",
    ),
    reading_guidelines=(
        "Read the scale bar first (value and unit, nm or 1/nm) and use it for every length; do not infer scale from the magnification alone.",
        "For lattice fringes, measure across as many fringes as possible (N spacings over distance L, d = L/N) and state N.",
        "Prefer d-spacings and (hkl) labels printed on the image over your own measurement.",
        "For SAED, measure from the central beam: with a reciprocal-space scale bar, d = 1/r; note whether the pattern is spots (single crystal), rings (polycrystalline), or halos (amorphous).",
        "Record angles between indexed spots or fringe sets; they constrain the zone axis and cell geometry.",
        "Note the zone axis if labelled, and which panel an FFT or SAED inset belongs to.",
        "In STEM-HAADF, brighter columns indicate heavier elements; note the contrast pattern rather than assigning elements from brightness alone.",
    ),
    inferable_properties=(
        "Interplanar d-spacings and their assignment to lattice planes",
        "Zone axis and angles between lattice planes (cell metric, symmetry hints)",
        "Single-crystal vs polycrystalline vs amorphous character",
        "Particle / grain size and shape, layer thickness, stacking of layers",
        "Local defects: stacking faults, dislocations, grain boundaries, intergrowth of phases",
        "Local elemental distribution from EDS/EELS maps",
    ),
    interpretation_guidelines=(
        "Match measured d-spacings to candidate phases using the lattice parameters in the caption or context; report the mismatch.",
        "A single d-spacing rarely identifies a phase uniquely; require multiple spacings or angles for a phase assignment.",
        "Layered materials viewed edge-on show the stacking period as wide fringes; compare it with the expected interlayer spacing.",
        "Hybrid and halide perovskites are beam-sensitive and can degrade to PbI2 (d about 0.70 nm (001), 0.39 nm (100), 0.35 nm (101)) or other products; consider beam damage when fringes match a degradation product.",
        "Distinguish features the authors label from features you identify yourself.",
    ),
    limitations=(
        "TEM images a small, possibly unrepresentative region; do not generalize phase purity to the bulk.",
        "Images are 2D projections; overlapping crystals can produce moiré fringes that are not lattice spacings.",
        "Light atoms (H, often C, N) are not resolved; organic molecular conformation cannot be determined.",
        "d-spacings measured from published images by eye are approximate (typically ±2-5%).",
    ),
)
