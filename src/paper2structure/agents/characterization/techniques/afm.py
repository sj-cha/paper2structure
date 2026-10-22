from ..spec import TechniqueSpec


AFM = TechniqueSpec(
    name="AFM",
    observables=(
        "Height (topography) images with a color scale bar, and phase or amplitude images",
        "Line profiles across steps, grains, or flakes (height vs lateral distance)",
        "Surface roughness values (Ra, Rq/RMS) and step-height histograms",
        "Functional modes: piezoresponse (PFM), conductive (c-AFM), Kelvin probe (KPFM) maps",
    ),
    reading_guidelines=(
        "Read both scales: the lateral scale bar (or image size) and the color bar giving the height range.",
        "Take step heights and thicknesses from line profiles, not from image color; read the profile's height axis and note where the profile is drawn.",
        "Prefer roughness values and thicknesses printed on the figure; otherwise give ranges with the reading precision.",
        "For layered flakes, count terraces and record each step height separately.",
        "Note the imaging mode (tapping/contact, PFM, KPFM) and whether the image is height, phase, or amplitude.",
    ),
    inferable_properties=(
        "Thickness of flakes and films, and the number of layers in exfoliated layered crystals",
        "Step heights matching a unit-cell or layer repeat (constraint on the stacking period)",
        "Surface roughness, grain size, and grain morphology",
        "Ferroelectric domain structure (PFM) and local conductivity or work-function contrast",
    ),
    interpretation_guidelines=(
        "Compare step heights with the expected layer repeat (e.g. half the stacking-axis lattice parameter for many layered structures); a step equal to an integer multiple supports the assignment.",
        "The first step on a substrate is often offset by adsorbates or tip-sample interaction; prefer steps between terraces of the same material.",
        "Lateral sizes are broadened by tip convolution; heights are more reliable than widths.",
        "PFM phase contrast indicates switchable polarization domains, which supports a polar (noncentrosymmetric) structure but does not fix the space group.",
    ),
    limitations=(
        "AFM does not give lattice parameters, space group, or atomic positions (except special atomic-resolution cases).",
        "Heights can be distorted by humidity, adsorbed layers, and tip-sample forces (typically a few angstroms).",
        "AFM probes the surface only; surface reconstruction or degradation may differ from the bulk.",
    ),
)
