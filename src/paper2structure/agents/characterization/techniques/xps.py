from ..spec import TechniqueSpec


XPS = TechniqueSpec(
    name="XPS",
    observables=(
        "Survey spectra and high-resolution core-level spectra: intensity vs binding energy (eV)",
        "Fitted components (envelopes) with labelled chemical states",
        "Valence-band spectra and depth profiles",
    ),
    reading_guidelines=(
        "The binding-energy axis usually increases from right to left; read tick values carefully.",
        "Read component positions and labels printed on the figure; prefer them over axis estimates.",
        "Identify spin-orbit doublets (e.g. Pb 4f7/2 and 4f5/2, about 4.9 eV apart) and treat them as one chemical state.",
        "Record the charge reference used (often adventitious C 1s at about 284.8 eV) from caption or context.",
        "For depth profiles or etching series, read each spectrum separately.",
    ),
    inferable_properties=(
        "Oxidation states and chemical environments of surface elements",
        "Surface composition and stoichiometry",
        "Presence of metallic or reduced species (e.g. Pb0 alongside Pb2+), surface hydroxyls, or oxygen-related defects",
        "Dopant valence and surface segregation",
    ),
    interpretation_guidelines=(
        "Core-level shifts of a few tenths of an eV are within referencing uncertainty; assign chemical states only for clear shifts or separate components.",
        "In lead halide perovskites, a component about 1.5-2 eV below the Pb2+ 4f7/2 peak indicates metallic Pb0, often from X-ray or beam damage.",
        "Fitted component areas depend on the fitting model; treat derived ratios as approximate.",
        "XPS reflects the top few nanometers; extend conclusions to the bulk only with support from bulk techniques.",
    ),
    limitations=(
        "XPS does not detect H or He and gives no lattice parameters or atomic positions.",
        "It is surface-sensitive (information depth about 5-10 nm); surface oxidation or contamination may differ from the bulk.",
        "X-ray exposure and charging can shift or create peaks.",
    ),
)
