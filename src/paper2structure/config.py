from __future__ import annotations

from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class CampaignConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "gpt-5.6-luna"
    reasoning_effort: Optional[str] = None  # None keeps the provider default
    max_retry: int = 3  # output-validation retries per agent call
    use_images: bool = True  # show rendered structures to the Construction and Revision agents


class InputsConfig(BaseModel):
    """Upstream products. Agents 1-2 are placeholders, so these point at their (fixture) outputs."""
    model_config = ConfigDict(extra="forbid")
    paper_summary: str
    experimental_evidence: str


class PlanningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_plans: int = 3
    max_atoms: int = 5000

    @field_validator("max_plans", "max_atoms")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be positive")
        return v


class ConstructionConfig(BaseModel):
    """The Construction agent is a pydantic-ai-harness Coder working in one workspace per attempt."""
    model_config = ConfigDict(extra="forbid")
    sandbox: Literal["bubblewrap", "none"] = "bubblewrap"  # "none" runs the agent's commands unsandboxed
    network: bool = False  # let sandboxed commands reach the network
    gpu: bool = False  # expose the GPUs to sandboxed commands (network stays off; drops the socket filter CUDA trips)
    hide_paths: list[str] = Field(default_factory=lambda: [".env", "~/.ssh", "~/.aws", "~/.netrc", "~/.config"])
    max_requests: int = 60  # model requests per Construction-agent run
    max_tool_calls: int = 150  # tool calls per Construction-agent run
    gate_retries: int = 2  # final reports the validation gate may send back before passing failures on
    code_timeout: float = 600.0  # seconds for the from-scratch replay of each final script
    memory_limit_gb: Optional[float] = 16.0  # per replay; None disables the cap
    planning_tool: bool = True  # harness Planning: a task list the agent keeps
    reminder_every: int = 10  # model requests between reminders of the script contract; 0 disables
    delegation: bool = False  # Coder's delegate_task (hand a sub-task to a fresh run of the agent)
    memory_dir: Optional[str] = None  # harness Memory notebook kept across runs; None disables


class RevisionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    read_scripts: bool = True  # the Revision agent may read the scripts that built each structure


class ValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    overlap_scale: float = 0.6  # fail below this x covalent-radius sum
    close_contact_scale: float = 0.85  # warn below this for molecular contacts
    anion_contact_scale: float = 1.3  # warn below this for molecule O/N/S/halogen - surface anion pairs
    bond_scale: float = 1.2  # bond detection for molecule integrity
    binding_scale: float = 1.5  # adsorbate-substrate contact beyond this x covalent sum warns (weakly bound)
    detached_scale: float = 2.0  # ... and beyond this fails (detached)
    charge_neutrality: bool = False  # gate on zero net formal charge (substrate oxidation states + SMILES charges)
    min_vacuum: float = 15.0  # A between periodic images of slabs/nanocrystals


class ExportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    formats: tuple[Literal["poscar", "cif", "extxyz"], ...] = ("poscar",)


class OverridesConfig(BaseModel):
    """User-pinned modeling choices the Planner must respect (recorded as user-imposed, not evidence)."""
    model_config = ConfigDict(extra="forbid")
    model_type: Optional[str] = None  # e.g. "slab", "nanocrystal", "core-shell nanocrystal"
    miller: Optional[tuple[int, int, int]] = None
    supercell: Optional[tuple[int, int]] = None
    lattice: Optional[dict[str, float]] = None  # e.g. {"a": 5.95}
    notes: list[str] = Field(default_factory=list)

    def describe(self) -> str:
        parts = []
        if self.model_type:
            parts.append(f"model_type must be {self.model_type!r}")
        if self.miller:
            parts.append(f"slab plans must use Miller index {tuple(self.miller)}")
        if self.supercell:
            parts.append(f"slab plans must use in-plane supercell {tuple(self.supercell)}")
        if self.lattice:
            parts.append("lattice parameters must be " + ", ".join(f"{k}={v}" for k, v in self.lattice.items())
                         + " (source user_override)")
        parts.extend(self.notes)
        return "; ".join(parts) if parts else "none"


class Paper2StructureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    campaign: CampaignConfig = Field(default_factory=CampaignConfig)
    inputs: InputsConfig
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
    construction: ConstructionConfig = Field(default_factory=ConstructionConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    revision: RevisionConfig = Field(default_factory=RevisionConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    overrides: OverridesConfig = Field(default_factory=OverridesConfig)


def load_config(path: str) -> Paper2StructureConfig:
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    return Paper2StructureConfig.model_validate(raw)
