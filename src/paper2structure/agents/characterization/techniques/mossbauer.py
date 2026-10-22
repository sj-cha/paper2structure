from ..spec import TechniqueSpec


MOSSBAUER = TechniqueSpec(
    name="Mossbauer",
    observables=(
        "Transmission (or absorption) vs source velocity (mm/s) for a Mossbauer isotope (usually 57Fe; also 119Sn, 151Eu)",
        "Fitted subspectra: singlets, quadrupole doublets, and magnetic sextets",
        "Tables of isomer shift (δ), quadrupole splitting (ΔEQ), hyperfine field (Bhf), linewidth, and relative area",
    ),
    reading_guidelines=(
        "Identify the isotope, measurement temperature, and isomer-shift reference (usually α-Fe at room temperature).",
        "Read the number and type of fitted subspectra and their labels.",
        "Prefer fitted parameters from tables or text; velocity-axis readings are approximate.",
        "For temperature series, note where sextets collapse into doublets.",
    ),
    inferable_properties=(
        "Oxidation and spin state of the Mossbauer atom",
        "Number of distinct sites and their relative populations",
        "Local site symmetry and coordination distortion (from ΔEQ)",
        "Magnetic ordering and ordering temperature (sextets, hyperfine fields)",
    ),
    interpretation_guidelines=(
        "For 57Fe at room temperature relative to α-Fe: high-spin Fe3+ has δ about 0.2-0.5 mm/s, high-spin Fe2+ about 0.9-1.3 mm/s; assign oxidation states from δ.",
        "Each subspectrum represents a distinct environment; a candidate structure should have the same number of inequivalent Fe sites (or a justified disorder model).",
        "Relative areas approximate site populations if recoil-free fractions are similar; use them to constrain site occupancies.",
        "Large ΔEQ indicates a distorted coordination; a sextet indicates magnetic order at the measurement temperature (Fe3+ oxides typically about 45-55 T).",
    ),
    limitations=(
        "Mossbauer applies only to Mossbauer-active isotopes and gives local information, not lattice parameters.",
        "Different recoil-free fractions between sites bias area-based populations, especially at room temperature.",
        "Overlapping subspectra make fits model-dependent.",
    ),
)
