from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CheckResult(BaseModel):
    name: str
    status: Literal["pass", "warn", "fail", "skip"]  # skip: does not apply to this structure, or could not be assessed
    detail: str = ""


class _ClaimCheck(BaseModel):
    """A claim from the paper, as a deterministic test of the built structure (evidence_checks)."""
    model_config = ConfigDict(extra="forbid")
    id: str = Field(description="unique within the plan, e.g. 'C1'")
    claim: str = Field(description="the claim in a few words, with the paper's value")
    claim_ids: list[str] = Field(default_factory=list, description="IDs of the listed structural claims it tests")
    evidence_ids: list[str] = Field(default_factory=list)
    system_ids: list[str] = Field(default_factory=list, description="systems it applies to; empty: all of the plan's")
    severity: Literal["fail", "warn"] = Field("fail", description="warn only for approximate or low-confidence claims")


class LatticeCheck(_ClaimCheck):
    """Bulk cell lengths (A) and angles (deg); a supercell matches when each length is an integer
    multiple of the target."""
    kind: Literal["lattice"] = "lattice"
    a: Optional[float] = Field(None, gt=0)
    b: Optional[float] = Field(None, gt=0)
    c: Optional[float] = Field(None, gt=0)
    alpha: Optional[float] = None
    beta: Optional[float] = None
    gamma: Optional[float] = None
    tolerance: float = Field(0.01, gt=0, description="relative tolerance on lengths")
    angle_tolerance: float = Field(1.0, gt=0, description="degrees")


class SpaceGroupCheck(_ClaimCheck):
    """Space group of the bulk structure (spglib at symprec)."""
    kind: Literal["space_group"] = "space_group"
    symbol: str = Field(description='Hermann-Mauguin symbol in any setting, e.g. "Pnma", "Pbnm", "P2_1/n", "C2cb", '
                                    'or the international number')
    scope: Literal["all", "framework"] = Field("all", description="framework: without the molecules (for "
                                                                  "disordered or averaged molecules)")
    symprec: float = Field(0.1, gt=0, description="A")


class CompositionCheck(_ClaimCheck):
    """Composition of a bulk structure (atomic fractions)."""
    kind: Literal["composition"] = "composition"
    formula: str = Field(description='e.g. "(C4H9NH3)2PbI4" or "La0.7Bi0.3CrO3"')
    scope: Literal["all", "framework"] = "all"
    tolerance: float = Field(0.005, gt=0, description="largest allowed difference of an atomic fraction")


class BondLengthCheck(_ClaimCheck):
    """Every A-B distance of the first shell (up to `max` + 0.5 A) lies within [min, max] +- tolerance."""
    kind: Literal["bond_length"] = "bond_length"
    elements: tuple[str, str]
    min: float = Field(gt=0, description="A")
    max: float = Field(gt=0, description="A")
    tolerance: float = Field(0.05, ge=0, description="A")


class BondAngleCheck(_ClaimCheck):
    """Every A-B-C angle at vertex B (bonded neighbours A and C) lies within [min, max] +- tolerance,
    e.g. metal-anion-metal bridge angles ("Ti", "O", "Ti"), ("Pb", "I", "Pb") or ("Si", "O", "Si")."""
    kind: Literal["bond_angle"] = "bond_angle"
    elements: tuple[str, str, str] = Field(description="vertex in the middle")
    min: float = Field(ge=0, le=180, description="degrees")
    max: float = Field(ge=0, le=180, description="degrees")
    tolerance: float = Field(3.0, ge=0, description="degrees")
    max_bond: Optional[float] = Field(None, gt=0, description="A; default 1.25 x the covalent-radius sum")


class CoordinationCheck(_ClaimCheck):
    """Number of `neighbor` atoms around every `center` atom of the framework (interior atoms of
    slabs and particles)."""
    kind: Literal["coordination"] = "coordination"
    center: str
    neighbor: str
    number: int = Field(ge=1)
    max_bond: Optional[float] = Field(None, gt=0, description="A; default 1.25 x the covalent-radius sum")


class HydrogenBondCheck(_ClaimCheck):
    """Hydrogen bonds from the molecules' donor atoms (an H on the donor, H...acceptor below
    `max_distance`, D-H...A angle at least `min_angle`): every donor needs `per_donor` of them."""
    kind: Literal["hydrogen_bonds"] = "hydrogen_bonds"
    donor: str = Field("N", description="element bearing the H, e.g. N (ammonium, amines) or O (hydroxyl, water)")
    acceptors: list[str] = Field(min_length=1, description='e.g. ["O"], ["Cl"], ["I"]')
    per_donor: int = Field(1, ge=1, description="hydrogen-bonded H per donor atom, e.g. 3 for an NH3+ group")
    smiles: Optional[str] = Field(None, description="only molecules with this SMILES")
    max_distance: Optional[float] = Field(None, gt=0, description="H...A in A; default the van der Waals sum - 0.1")
    min_angle: float = Field(130.0, ge=90, le=180)


class LayerCheck(_ClaimCheck):
    """Each connected block of the framework holds `count` planes of `element` along `axis`: the
    thickness of the slabs of a layered material (n metal layers per slab of a Ruddlesden-Popper
    phase, an MXene or a layered hydroxide; the layers of a slab model along c)."""
    kind: Literal["layers"] = "layers"
    element: str
    count: int = Field(ge=1)
    axis: Literal["a", "b", "c"] = Field("c", description="cell axis the layers stack along")


ClaimCheck = Annotated[Union[LatticeCheck, SpaceGroupCheck, CompositionCheck, BondLengthCheck, BondAngleCheck,
                             CoordinationCheck, HydrogenBondCheck, LayerCheck], Field(discriminator="kind")]


class UncheckedClaim(BaseModel):
    """A listed claim no check can test, and why."""
    model_config = ConfigDict(extra="forbid")
    claim_ids: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1, description='e.g. "not a property of an atomic model (phase purity)"')

    @model_validator(mode="after")
    def _reason(self) -> UncheckedClaim:
        if not self.reason.strip():
            raise ValueError("give the reason the claim cannot be checked")
        return self


class ValidationReport(BaseModel):
    """Deterministic geometric/chemical checks of one structure. Experimental agreement is not
    assessed here: no observable is simulated."""
    structure_id: str
    passed: bool
    checks: list[CheckResult]
    metrics: dict[str, float | int | str] = Field(default_factory=dict)

    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == "fail"]

    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == "warn"]

    def with_check(self, check: CheckResult) -> ValidationReport:
        """A copy with `check` added (a failing check fails the report)."""
        return self.model_copy(update={"checks": [*self.checks, check],
                                       "passed": self.passed and check.status != "fail"})

    def summary(self, include_skipped: bool = False) -> str:
        """PASS/FAIL with the warning and failing checks; `include_skipped` also names the gate
        checks that could not be assessed (claim checks are reported on their own)."""
        head = "PASS" if self.passed else "FAIL"
        bad = [f"{c.name}: {c.detail}" for c in self.checks if c.status in ("warn", "fail")]
        if include_skipped:
            bad += [f"{c.name} {c.detail}" for c in self.checks if c.status == "skip" and not c.name.startswith("claim ")]
        return head + (" - " + "; ".join(bad) if bad else "")
