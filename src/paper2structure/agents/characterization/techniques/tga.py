from ..spec import TechniqueSpec


TGA = TechniqueSpec(
    name="TGA",
    observables=(
        "Mass (% of initial) vs temperature, often with the derivative curve (DTG)",
        "Mass-loss steps with onset temperatures and step sizes; residual mass",
        "Coupled evolved-gas analysis (TGA-MS, TGA-FTIR) if shown",
    ),
    reading_guidelines=(
        "Read each step's onset temperature and mass loss (%) against the axis ticks; prefer values printed on the curve.",
        "Use DTG minima to locate overlapping steps.",
        "Record the atmosphere (N2, Ar, air) and heating rate from caption or context.",
        "Read the residual mass at the end of the run.",
    ),
    inferable_properties=(
        "Water, solvent, or guest content (e.g. number of hydrate water molecules per formula unit)",
        "Organic fraction and stoichiometry checks for hybrid materials",
        "Thermal stability and decomposition temperatures",
    ),
    interpretation_guidelines=(
        "Compute the expected mass loss for each candidate formula and compare with the measured step; report the formula that matches and the difference.",
        "For hydrates, n(H2O) ≈ (Δm% / 100) × M(formula) / 18.02, with M the molar mass of the hydrated formula.",
        "In hybrid halide perovskites, early steps are usually loss of the organic halide (and solvent) and later steps sublimation or decomposition of the metal halide; assign with support from the text.",
        "A low-temperature step below about 150 °C usually means adsorbed or lattice solvent, which matters for the structure model.",
    ),
    limitations=(
        "TGA gives composition and stability, not crystal structure.",
        "Overlapping steps make individual assignments uncertain; step temperatures depend on heating rate and atmosphere.",
        "Residual mass depends on the final product (oxide vs metal), which depends on atmosphere.",
    ),
)
