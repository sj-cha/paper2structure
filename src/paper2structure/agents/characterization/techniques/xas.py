from ..spec import TechniqueSpec


XAS = TechniqueSpec(
    name="XAS",
    observables=(
        "XANES: normalized absorption vs energy near an absorption edge, with reference spectra",
        "EXAFS: χ(k) in k-space and the Fourier-transform magnitude |χ(R)| in R-space, with fits",
        "Tables of fitted coordination numbers (CN), distances (R), and Debye-Waller factors (σ²)",
    ),
    reading_guidelines=(
        "Identify the element and edge (e.g. Fe K, Ru L3) and which traces are references.",
        "Read the edge position (half-height or first-derivative maximum) and its shift relative to the references.",
        "Note pre-edge features and white-line intensity.",
        "R-space peaks are usually plotted without phase correction and appear about 0.3-0.5 Å shorter than true bond lengths; check the axis label (R or R + ΔR).",
        "Prefer fitted CN, R, and σ² from tables or text over readings from the plotted transform.",
    ),
    inferable_properties=(
        "Average oxidation state of the absorbing element",
        "Local coordination geometry (pre-edge intensity, white-line shape)",
        "First- and second-shell bond distances and coordination numbers",
        "Local disorder; presence or absence of metal-metal paths (e.g. single atoms vs clusters)",
    ),
    interpretation_guidelines=(
        "An edge shift to higher energy relative to references indicates a higher average oxidation state; interpolate between references only if they share the coordination.",
        "For 3d metals, a strong pre-edge peak indicates noncentrosymmetric (e.g. tetrahedral) coordination; a weak one indicates octahedral.",
        "Use fitted distances directly as bond-length constraints; fitted coordination numbers are typically uncertain by about ±20%.",
        "Absence of a metal-metal peak supports isolated metal sites; presence indicates clusters or metallic particles.",
    ),
    limitations=(
        "XAS averages over all absorbing atoms; it cannot separate sites without modelling.",
        "EXAFS gives local structure (a few Å), not lattice parameters or long-range order.",
        "Distances from fits are typically accurate to about 0.01-0.02 Å; unfitted R-space peak positions are not bond lengths.",
    ),
)
