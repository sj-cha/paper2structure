from ..spec import TechniqueSpec


UV_VIS = TechniqueSpec(
    name="UV-Vis",
    observables=(
        "Absorbance, transmittance, or diffuse reflectance (Kubelka-Munk) vs wavelength or energy",
        "Excitonic absorption peaks and absorption edges",
        "Tauc plots used to estimate band gaps",
    ),
    reading_guidelines=(
        "Check whether the x-axis is wavelength (nm) or energy (eV); convert with E [eV] = 1239.84 / λ [nm].",
        "Read excitonic peak positions at the peak maximum and absorption edges where the curve rises; say which you report.",
        "For Tauc plots, read the y-axis exponent ((αhν)^2 for direct, (αhν)^(1/2) for indirect) and the x-intercept of the drawn linear fit.",
        "For stacked spectra, read each trace separately; prefer band gaps printed on the figure.",
    ),
    inferable_properties=(
        "Optical band gap and exciton energies",
        "Layer number n of layered (Ruddlesden-Popper) halide perovskites from the exciton energy",
        "Phase mixtures (several distinct excitonic peaks in one sample)",
        "Coordination and oxidation state of transition-metal ions from d-d bands",
    ),
    interpretation_guidelines=(
        "For (BA)2(MA)n-1PbnI3n+1, reported gaps are about 2.43 eV (n = 1), 2.17 eV (n = 2), 2.03 eV (n = 3), 1.91 eV (n = 4), and about 1.5-1.6 eV for 3D MAPbI3; use them to identify n.",
        "Several excitonic peaks in one film indicate a mixture of n phases rather than a single phase.",
        "Systematic band-edge shifts with composition indicate alloying; state the direction.",
        "Tauc-plot gaps depend on the chosen exponent and fit range; treat them as approximate.",
    ),
    limitations=(
        "UV-Vis does not give lattice parameters or atomic positions.",
        "Scattering, film thickness, and substrate absorption distort spectra.",
        "Band gaps constrain structure only indirectly (e.g. via layer number or composition).",
    ),
)
