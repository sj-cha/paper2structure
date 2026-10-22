from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TargetCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    property: str = Field(description="the plan target's property")
    measured: str = Field(description="the value measured on the built structure, with units")
    satisfied: bool
    method: str = Field("", description="how it was measured")


class BuiltStructure(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="the name passed to emit()")
    plan_id: str
    system_ids: list[str] = Field(default_factory=list, description="the paper's systems it models, as passed to "
                                                                    "emit() (needed when the plan covers several)")
    script: str = Field(description="workspace path of the standalone script that builds it")
    method: str = Field(description="how it was built: which steps used paper2structure.tools and which "
                                    "used your own code")
    target_checks: list[TargetCheck] = Field(default_factory=list)
    deviations: list[str] = Field(default_factory=list, description="every departure from the plan")
    notes: str = ""


class FailedPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plan_id: str
    reason: str


class ConstructionReport(BaseModel):
    """Agent 4 output: which emitted structures realize which plans."""
    model_config = ConfigDict(extra="forbid")
    structures: list[BuiltStructure] = Field(default_factory=list)
    failed: list[FailedPlan] = Field(default_factory=list)
    summary: str = ""


class StructureArtifact(BaseModel):
    """A constructed structure saved in the run directory (paths relative to it)."""
    structure_id: str
    plan_id: str
    name: str
    system_ids: list[str] = Field(default_factory=list)
    revision: int
    attempt: int
    files: dict[str, str]
    source: str = Field(description="the emitted extxyz (full per-atom metadata), relative to the run dir")
    script: str = Field(description="standalone script that regenerates the structure, relative to the run dir")
    image: Optional[str] = None
    method: str = ""
    target_checks: list[TargetCheck] = Field(default_factory=list)
    formula: str
    n_atoms: int
    cell: list[list[float]]
    pbc: tuple[bool, bool, bool]
    n_molecules: int = 0
    description: str = ""
    deviations: list[str] = Field(default_factory=list)
    notes: str = ""
