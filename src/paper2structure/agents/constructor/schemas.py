from __future__ import annotations

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
    """The Constructor's output: which emitted structures realize which plans."""
    model_config = ConfigDict(extra="forbid")
    structures: list[BuiltStructure] = Field(default_factory=list)
    failed: list[FailedPlan] = Field(default_factory=list)
    summary: str = ""


class StructureArtifact(BaseModel):
    """A constructed structure saved in the run directory (paths relative to it)."""
    structure_id: str
    plan_id: str
    name: str
    system_ids: list[str] = Field(default_factory=list, description="the systems it models, as passed to emit()")
    revision: int = Field(description="the plan set it was built from (index into the plan history)")
    files: dict[str, str]
    source: str = Field(description="the emitted extxyz (full per-atom metadata), relative to the run dir")
    script: str = Field(description="standalone script that regenerates the structure, relative to the run dir")
    image: str = Field(description="rendered top and side views (PNG), relative to the run dir")
    method: str = ""
    target_checks: list[TargetCheck] = Field(default_factory=list)
    formula: str
    n_atoms: int
    description: str = ""
    deviations: list[str] = Field(default_factory=list)
    notes: str = Field("", description="the constructor's notes on the structure (emit() notes)")
    construction_summary: str = Field("", description="the summary of the construction run that built it")
