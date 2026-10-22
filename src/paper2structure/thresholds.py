"""Default thresholds of the validation gate: the defaults of config.ValidationConfig, of
tools.validate_structure's keyword arguments, and the constants the builders apply on their own
(build_bulk's overlap rejection, the adsorbate placer's anion-contact score). Imports nothing, so the
construction scripts' toolkit can use it without loading the workflow's dependencies."""

OVERLAP_SCALE = 0.6  # fail below this x covalent-radius sum
CLOSE_CONTACT_SCALE = 0.85  # warn below this for molecular contacts
ANION_CONTACT_SCALE = 1.3  # warn below this for molecule O/N/S/halogen - surface anion pairs
BOND_SCALE = 1.2  # bond detection for molecule integrity
BINDING_SCALE = 1.5  # adsorbate-substrate contact beyond this x covalent sum warns (weakly bound)
DETACHED_SCALE = 2.0  # ... and beyond this fails (detached)
MIN_VACUUM = 15.0  # A between periodic images of slabs/nanocrystals
# crystal chemistry of ionic substrates: bond-valence sums of the framework cations at interior sites
BVS_WARN = 0.30  # warn beyond this relative deviation from the oxidation state
BVS_FAIL = 0.40  # ... and fail beyond this
BOND_SHARE_WARN = 0.45  # warn when one bond carries more than this share of its cation's valence
LIKE_CONTACT_WARN = 0.8  # cation-cation / anion-anion pairs below this x covalent sum warn
LIKE_CONTACT_FAIL = 0.7  # ... and fail below this
MOLECULE_BOND_TOL = 0.12  # A: molecule bond lengths must match RDKit conformers of their SMILES
MOLECULE_ANGLE_TOL = 20.0  # degrees: ... and bond angles (RDKit's own spread: 0.09 A, 15 degrees)
