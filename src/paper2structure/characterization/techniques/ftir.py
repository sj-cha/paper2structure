from ..spec import TechniqueSpec


FTIR = TechniqueSpec(
    name="FTIR",
    aliases=("FT-IR", "IR", "infrared", "infrared spectroscopy", "ATR-FTIR", "ATR"),
    observables=(
        "Transmittance or absorbance vs wavenumber (cm^-1) spectra",
        "Band positions, relative intensities, widths, and shapes",
        "Comparison of spectra across samples, or against precursors and free ligands",
    ),
    reading_guidelines=(
        "Check whether the y-axis is transmittance (bands are dips) or absorbance (bands are peaks).",
        "Wavenumber axes usually decrease from left to right (e.g. 4000 to 400 cm^-1); read tick values carefully.",
        "Prefer band positions printed on the figure; for axis estimates, give precision from tick spacing (often ±10-20 cm^-1).",
        "For stacked or offset spectra, identify each trace from the legend and read band positions per trace.",
        "Note vertical guide lines or shaded regions authors use to mark band assignments or shifts.",
        "Record broad features (e.g. O-H, N-H stretching envelopes) as ranges rather than single positions.",
    ),
    inferable_properties=(
        "Functional groups and identity of the organic component",
        "Protonation state of amines (NH3+ vs NH2)",
        "Hydrogen bonding between organic cations and the inorganic framework (band shifts and broadening)",
        "Presence of residual solvent, water, or unreacted precursors",
    ),
    interpretation_guidelines=(
        "Typical ranges: O-H stretch 3200-3600 (broad); N-H stretch of NH3+ about 3000-3200; C-H stretch 2850-3100; "
        "C=O 1650-1750; NH3+ bending about 1470-1610; aromatic C=C 1450-1600; C-F 1000-1400 cm^-1.",
        "Shifts of N-H bands relative to the free amine or ammonium salt indicate a changed hydrogen-bonding environment; state the direction and size.",
        "Residual solvent markers: DMF C=O near 1650-1670 cm^-1, DMSO S=O near 1000-1050 cm^-1; their presence matters for the structure model.",
        "Assignments in the fingerprint region (<1500 cm^-1) overlap; give alternatives when ambiguous.",
        "Use the caption and context for the measurement mode (ATR, KBr pellet, thin film), which affects band positions and intensities.",
    ),
    limitations=(
        "FTIR gives no long-range order, lattice parameters, or atomic positions.",
        "Metal-halide framework vibrations lie mostly below the mid-IR range (far IR) and are usually not visible.",
        "Molecular conformation and orientation within the lattice are only weakly constrained.",
    ),
)
