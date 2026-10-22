from ..spec import TechniqueSpec


DFT = TechniqueSpec(
    name="DFT",
    observables=(
        "Relaxed or model crystal structures (images, lattice parameters, bond lengths and angles)",
        "Band structures along labelled k-paths, and total or projected density of states (DOS/PDOS)",
        "Relative energies of competing structures, configurations, or polymorphs (often meV per formula unit)",
        "Energy vs strain or pressure curves, formation/adsorption energies, phonon dispersions",
    ),
    reading_guidelines=(
        "Record the functional (e.g. PBE, PBEsol, HSE, with or without SOC or dispersion correction) and the cell/supercell used, from caption or context.",
        "For band structures, read the energy zero (Fermi level or VBM), the gap, and whether VBM and CBM sit at the same k-point (direct) or not (indirect).",
        "Read relative energies with their unit (eV, meV, per atom, per formula unit, per cell) and convert to a common unit before comparing.",
        "Prefer values printed in tables or on the figure over readings from curves.",
        "For figures of relaxed structures, follow the Structure model approach: viewing direction, legend, connectivity, "
        "tilts, and molecular orientation; note which competing configuration each panel shows.",
    ),
    inferable_properties=(
        "Structural motifs of candidate models (organic cation orientations, octahedral tilts, stacking), as descriptors rather than coordinates",
        "Which candidate space group or configuration is computed lowest in energy",
        "Relative stability of competing structures or configurations",
        "Computed lattice parameters and bond lengths",
        "Electronic structure features (band gap character, orbital contributions)",
    ),
    interpretation_guidelines=(
        "Treat DFT results as model predictions, not measurements; label constraints from them as computed.",
        "State whether the cell was fixed to experimental lattice parameters or fully relaxed; only fully relaxed cells give computed lattice parameters.",
        "Typical errors: PBE overestimates lattice parameters by about 1-2%, PBEsol is closer, LDA underestimates; semilocal functionals underestimate band gaps.",
        "Energy differences of a few meV per formula unit are within DFT accuracy and below kT at room temperature (~25 meV); treat such structures as near-degenerate.",
        "For lead and tin halides, spin-orbit coupling strongly lowers the computed gap; compare gaps only at the same level of theory.",
        "If the paper gives deposited coordinates (CIF, CCDC/ICSD number, SI file), cite them as the source for a starting model.",
    ),
    limitations=(
        "DFT structures are usually 0 K, ordered, and in small periodic cells; they miss thermal disorder and dynamic averaging.",
        "Computed band gaps do not constrain the atomic structure by themselves.",
        "Results depend on the functional and dispersion correction; do not treat a single DFT structure as experimentally confirmed.",
        "Atomic coordinates cannot be read from a rendered structure; without deposited data, DFT gives motifs and energies, not positions.",
    ),
)
