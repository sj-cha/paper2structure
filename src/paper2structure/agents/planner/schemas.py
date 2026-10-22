from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from paper2structure.tools.schemas import ClaimCheck, UncheckedClaim


class LatticeParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    a: float = Field(gt=0)
    b: Optional[float] = Field(None, gt=0)
    c: Optional[float] = Field(None, gt=0)
    alpha: Optional[float] = None
    beta: Optional[float] = None
    gamma: Optional[float] = None
    source: Literal["paper", "literature", "assumption", "user_override"] = Field(
        description="paper: reported in the evidence (cite it); literature: known value not in the paper")
    evidence_ids: list[str] = Field(default_factory=list)


class WyckoffSite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    species: str
    coords: tuple[float, float, float] = Field(description="fractional coordinates of the representative")
    wyckoff: Optional[str] = None


class CrystalMolecule(BaseModel):
    """A molecule of a hybrid or molecular crystal (the organic cations of a layered perovskite,
    guests in a framework), placed into the framework's cavities by the toolkit (fill_cavities)."""
    model_config = ConfigDict(extra="forbid")
    name: str
    smiles: str = Field(description="formal charges encode the protonation state, e.g. 'CCCC[NH3+]'")
    per_cell: int = Field(ge=1, description="molecules per conventional cell")


class BulkSpec(BaseModel):
    """A crystal as space group + Wyckoff representatives (checked by building it), with the
    molecules of a hybrid or molecular crystal listed separately."""
    model_config = ConfigDict(extra="forbid")
    name: str = Field("bulk", description='how the description refers to it, e.g. "core" or "substrate"')
    formula: str = Field(description="of the whole crystal, molecules included, e.g. '(C4H9NH3)2PbI4'")
    space_group: str = Field(description='Hermann-Mauguin symbol of the setting of the sites, e.g. "Fm-3m", "Pbnm", '
                                         '"P2_1/n", "Fd-3m:2" (origin choice 2), "R-3m:R" (rhombohedral axes)')
    sites: list[WyckoffSite] = Field(min_length=1, description="Wyckoff representatives of the framework atoms "
                                                               "(all atoms when the crystal has no molecules)")
    molecules: list[CrystalMolecule] = Field(default_factory=list)
    lattice: LatticeParameters
    formula_units: Optional[int] = Field(None, description="formula units per conventional cell")


class SurfaceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    miller: tuple[int, int, int]
    termination: str = Field(description='exposed species, e.g. "PbBr2-terminated" or "Ti5c and O2c exposed"')
    min_thickness: float = Field(10.0, gt=0, description="A")
    supercell: tuple[int, int] = (2, 2)
    vacuum: float = Field(15.0, gt=0, description="A")
    fix_bottom: Optional[float] = Field(None, description="thickness (A) of the fixed bottom region")


class Facet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    miller: tuple[int, int, int]
    relative_distance: float = Field(1.0, gt=0, description="center-to-facet distance relative to size/2 "
                                                             "(or the relative surface energy for wulff)")


class OxidationState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    element: str
    charge: int


class NanocrystalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    shape: str = Field(description="cuboctahedron, icosahedron, octahedron, truncated_octahedron, decahedron, "
                                   "wulff (elemental metals) or sphere/polyhedron (any crystal)")
    size: float = Field(gt=0, description="shells/edge atoms for magic clusters, diameter in A for carved particles")
    facets: list[Facet] = Field(default_factory=list)
    oxidation_states: list[OxidationState] = Field(default_factory=list, description="for charge balancing")
    vacuum: float = Field(15.0, gt=0)


class AdsorbateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    smiles: str = Field(description="formal charges encode the protonation state; monatomic species "
                                    "(e.g. a transferred proton '[H+]') are adsorbates too")
    binding_atoms: list[str] = Field(min_length=1, max_length=2,
                                     description="elements of the atoms that bind the surface: one entry = "
                                                 "monodentate, two = bidentate (chelating or bridging)")
    anchor_smarts: Optional[str] = None
    mode: Literal["adsorb", "replace"] = "adsorb"
    site_kind: Literal["top", "bridge", "hollow"] = "top"
    site_elements: list[str] = Field(default_factory=list)
    site: str = Field(description='where it binds, e.g. "atop Ti5c" or "replaces a surface Br"')
    count: int = Field(1, ge=1)
    bond_length: Optional[float] = Field(None, gt=0, description="expected donor-site distance (A)")


class SolventSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    smiles: str
    density: Optional[float] = Field(None, gt=0, description="g/cm^3")
    n_molecules: Optional[int] = Field(None, ge=1)
    thickness: Optional[float] = Field(None, gt=0, description="solvent layer thickness above a slab (A)")


class Target(BaseModel):
    """A feature of the model the evidence fixes, which the built structure must show."""
    model_config = ConfigDict(extra="forbid")
    property: str = Field(description='e.g. "core diameter", "shell thickness", "CO coverage", "exposed sites"')
    value: str = Field(description='e.g. "3.0 nm", "2 monolayers ZnS", "0.25 ML", "Ti5c and O2c"')
    tolerance: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class StructurePlan(BaseModel):
    """One structural hypothesis and the model that represents it. Any well-defined atomistic model
    is allowed: the typed specs are helpers for common parts, the description says what to build."""
    model_config = ConfigDict(extra="forbid")
    plan_id: str
    hypothesis: str
    model_type: str = Field(description='short label, e.g. "slab", "nanocrystal", "core-shell nanocrystal", '
                                        '"supported nanoparticle", "interface", "defective slab", "bulk"')
    description: str = Field(description="what to build, component by component, and how the components are "
                                         "arranged; precise enough to build without the paper")
    crystals: list[BulkSpec] = Field(default_factory=list, description="every crystalline component")
    surface: Optional[SurfaceSpec] = None
    nanocrystal: Optional[NanocrystalSpec] = None
    adsorbates: list[AdsorbateSpec] = Field(default_factory=list)
    solvent: list[SolventSpec] = Field(default_factory=list, description="solvent species and ions")
    targets: list[Target] = Field(default_factory=list)
    checks: list[ClaimCheck] = Field(default_factory=list, description="the paper's claims about this model's "
                                                                       "systems as tests the validation gate runs")
    unchecked: list[UncheckedClaim] = Field(default_factory=list, description="listed claims no check can test")
    system_ids: list[str] = Field(default_factory=list, description="the paper's material systems this plan models "
                                                                    "(the summarizer's IDs, e.g. 'S2')")
    evidence_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    construction_notes: str = ""

    def oxidation_states(self) -> dict[str, int]:
        """Formal oxidation states the plan fixes (nanocrystal charge balancing), for the charge check."""
        return {o.element: o.charge for o in self.nanocrystal.oxidation_states} if self.nanocrystal else {}

    @model_validator(mode="after")
    def _unique_crystals(self) -> StructurePlan:
        names = [c.name for c in self.crystals]
        if len(names) != len(set(names)):
            raise ValueError(f"plan {self.plan_id}: crystal names must be unique")
        ids = [c.id for c in self.checks]
        if len(ids) != len(set(ids)):
            raise ValueError(f"plan {self.plan_id}: check IDs must be unique")
        return self


class PlanSet(BaseModel):
    """Agent 3 output: competing structural hypotheses, each with a construction plan."""
    model_config = ConfigDict(extra="forbid")
    plans: list[StructurePlan] = Field(min_length=1)
    rationale: str

    @model_validator(mode="after")
    def _unique(self) -> PlanSet:
        ids = [p.plan_id for p in self.plans]
        if len(ids) != len(set(ids)):
            raise ValueError("plan IDs must be unique")
        return self

    def get(self, plan_id: str) -> StructurePlan:
        return next(p for p in self.plans if p.plan_id == plan_id)
