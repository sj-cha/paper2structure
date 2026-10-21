from ..spec import TechniqueSpec


# Worked example of cloning the generic agent; copy this file for TEM, NMR, etc.
XRD = TechniqueSpec(
    name="XRD",
    aliases=("X-ray diffraction", "PXRD", "powder XRD", "GIWAXS", "SCXRD"),
    observables=(
        "Diffraction patterns: intensity vs 2theta, q, or d-spacing",
        "Peak positions, relative intensities, widths (FWHM), and shapes",
        "Reference or simulated patterns (often drop lines / sticks) plotted for comparison",
        "Rietveld or Le Bail fits: observed, calculated, difference curve, and tick marks",
        "2D detector images (GIWAXS) with rings or spots",
    ),
    reading_guidelines=(
        "Check the x-axis quantity first: 2theta (deg), q (A^-1 or nm^-1), or d (A); do not mix them.",
        "Read peak positions at the peak maximum against the nearest ticks; precision is about one tenth of the minor tick spacing at best.",
        "Prefer (hkl) labels and peak positions printed on the figure over axis estimates.",
        "Stacked or offset traces: identify which trace belongs to which sample from the legend or labels, and read each separately.",
        "Distinguish measured patterns from reference sticks, simulated patterns, and fitted curves.",
        "Note markers (asterisks, symbols) that authors use to flag impurity or substrate peaks, and what the legend says they mean.",
        "Note when intensity is on a log or sqrt scale, or when the axis is broken or rescaled in an inset.",
        "Look for weak features near the baseline: small extra peaks, shoulders, and asymmetric peaks.",
    ),
    inferable_properties=(
        "Phase identity and purity (match to reference or simulated patterns)",
        "Lattice parameters and unit-cell metric",
        "Interlayer / stacking spacing from low-angle (00l) reflection series",
        "Preferred orientation (texture) from suppressed or enhanced reflection families",
        "Crystallite size and microstrain from peak broadening (Scherrer / Williamson-Hall)",
    ),
    interpretation_guidelines=(
        "Convert 2theta to d-spacing with Bragg's law, n*lambda = 2d sin(theta); take the wavelength from the caption or context, "
        "assume Cu K-alpha (1.5406 A) only if none is given, and say so.",
        "For layered materials, a periodic series of (00l) peaks at d/l gives the stacking period d.",
        "Strong (00l) dominance in thin films indicates preferred orientation, not necessarily phase purity.",
        "Unindexed extra peaks indicate an impurity or secondary phase; small low-angle peaks may indicate a different layer number n.",
        "Peak shifts between samples indicate lattice expansion or contraction; state the direction.",
    ),
    limitations=(
        "Hydrogen positions and organic molecular conformation are not determined by XRD peak positions alone.",
        "Amorphous phases are not detected beyond a broad background.",
        "Space group assignment needs full indexing; do not assign one from a few peaks.",
        "Peak positions read from a figure by eye are approximate; do not report lattice parameters to more precision than the readings support.",
    ),
)
