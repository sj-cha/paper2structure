from ..spec import TechniqueSpec


RAMAN = TechniqueSpec(
    name="Raman",
    observables=(
        "Raman spectra: intensity vs Raman shift (cm^-1), including low-frequency lattice modes",
        "Peak positions, widths, relative intensities, and shifts across samples, temperature, or pressure",
        "Raman maps of peak position or intensity",
    ),
    reading_guidelines=(
        "Read the Raman-shift axis and the excitation wavelength (from caption or context).",
        "For stacked spectra, identify each trace from the legend and read peak positions per trace.",
        "Prefer peak positions and assignments printed on the figure; for axis estimates give precision from tick spacing.",
        "Note broad backgrounds (photoluminescence) and whether spectra are baseline-corrected.",
        "Note vertical guide lines or labels authors use to mark modes or shifts.",
    ),
    inferable_properties=(
        "Phase identity by fingerprint comparison (e.g. polymorphs such as anatase vs rutile)",
        "Local symmetry and ordering (number of active modes, mode splitting)",
        "Octahedral tilting and soft modes; phase transitions with temperature or pressure",
        "Strain or composition from systematic peak shifts; disorder from peak broadening",
        "Vibrational modes of organic cations and their environment",
    ),
    interpretation_guidelines=(
        "Compare the number of observed modes with factor-group predictions for candidate space groups; extra modes suggest lower symmetry or a second phase.",
        "Systematic shifts with composition indicate bond-length or force-constant changes; state the direction and size.",
        "Broadened or merged modes indicate structural or compositional disorder.",
        "Laser heating can transform or degrade beam-sensitive samples (e.g. halide perovskites); consider this before assigning new peaks to the material.",
        "In centrosymmetric structures Raman and IR modes are mutually exclusive; coincident modes support a noncentrosymmetric structure.",
    ),
    limitations=(
        "Raman does not give lattice parameters or atomic positions.",
        "Intensities depend on resonance with the excitation and on orientation; they are not direct measures of phase fractions.",
        "Weak scatterers and fluorescing samples may show few usable peaks.",
    ),
)
