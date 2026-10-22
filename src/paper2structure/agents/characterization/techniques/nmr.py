from ..spec import TechniqueSpec


NMR = TechniqueSpec(
    name="NMR",
    observables=(
        "Solid-state (MAS) or solution NMR spectra: intensity vs chemical shift (ppm) for a given nucleus",
        "Number, positions, widths, and relative areas of resonances; spinning sidebands",
        "Variable-temperature spectra, 2D correlation spectra, relaxation data",
    ),
    reading_guidelines=(
        "Identify the nucleus (e.g. 1H, 13C, 15N, 207Pb, 133Cs, 27Al, 29Si, 31P, 7Li) and whether it is solid-state or solution NMR.",
        "The chemical-shift axis usually increases from right to left; read tick values carefully.",
        "Distinguish isotropic peaks from spinning sidebands (often marked with asterisks).",
        "Record whether spectra are direct excitation or cross-polarization (CP), and the MAS rate, from caption or context.",
        "Prefer shifts and site assignments printed on the figure or in the text over axis estimates.",
    ),
    inferable_properties=(
        "Number of distinct local environments of a nucleus (related to crystallographically distinct sites)",
        "Local coordination (e.g. 27Al tetrahedral vs octahedral; 29Si Q^n connectivity)",
        "Halide environment of Pb in halide perovskites (207Pb shift depends strongly on the halide)",
        "Incorporation of cations or dopants into the lattice, phase segregation, and cation dynamics",
    ),
    interpretation_guidelines=(
        "The number of resolved resonances constrains how many inequivalent sites a candidate structure may have for that nucleus.",
        "Relative areas give site populations only for quantitative direct-excitation spectra with full relaxation, not for CP spectra.",
        "Typical ranges: 27Al about 50-80 ppm for AlO4 and about 0 ppm for AlO6; 207Pb roughly 1400-1500 ppm for MAPbI3, about 360 ppm for MAPbBr3, about -640 ppm for MAPbCl3.",
        "Peaks that narrow on heating indicate motional averaging (dynamic disorder), so a static ordered model may not represent the room-temperature structure.",
        "Solution NMR confirms molecular identity and purity of organic components, not the solid-state structure.",
    ),
    limitations=(
        "NMR probes local environments; it does not give lattice parameters or long-range order.",
        "Shifts depend on referencing; compare only with values referenced the same way.",
        "Paramagnetic centers and quadrupolar nuclei broaden lines and can hide sites.",
    ),
)
