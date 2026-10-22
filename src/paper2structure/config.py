"""Shared configuration: which LLM every agent uses, the technique vocabulary, and the run
configuration read from a campaign YAML.

Settings are read from a ``.env`` file in the repo root (never commit it; it is in .gitignore)::

    OPENAI_API_KEY=sk-...
    P2S_MODEL=openai:gpt-5.6-luna      # optional, any pydantic-ai model name

Special model name ``dry-run`` returns a fake model that answers instantly with canned data.
It costs nothing and is used to test that the plumbing works. Each agent package registers
its canned answers in ``DRY_RUN_OUTPUTS``, keyed by the title of its output schema (the
summarizer and characterization agents have them).

The technique vocabulary is the one place where technique names live. The summarizer writes
canonical names into its evidence, and each characterization agent is keyed by one of them.

The run configuration (``Paper2StructureConfig``) has one Pydantic section per concern; a campaign
YAML fills it (``load_config``) and the CLI (main.py) overrides run-level values.
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Literal, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel

load_dotenv()

# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------

DEFAULT_MODEL = "openai:gpt-5.6-luna"
REASONING_EFFORT = "high"

# Models offered in the web page, with USD per 1M tokens (input / output), checked 2026-10-21.
MODEL_CHOICES = {
    "openai:gpt-5.6-luna": "GPT-5.6 Luna · $0.20/$1.20",
    "openai:gpt-6-luna": "GPT-6 Luna · price ?",
    "openai:gpt-5.4-nano": "GPT-5.4 nano · $0.20/$1.25",
    "openai:gpt-5-nano": "GPT-5 nano · $0.05/$0.40",
    "openai:gpt-5.4-mini": "GPT-5.4 mini · $0.75/$4.50",
    "dry-run": "Dry run · free",
}

DRY_RUN_OUTPUTS: dict[str, dict] = {}


def get_model(name: str | None = None) -> Model | str:
    """A pydantic-ai model name (``openai:`` is the OpenAI Responses API), or the dry-run model.
    A name without a provider prefix (``gpt-5.6-luna``) is an OpenAI model."""
    name = name or os.getenv("P2S_MODEL") or DEFAULT_MODEL
    if name == "dry-run":
        return FunctionModel(_fake_llm, model_name="dry-run")
    if ":" not in name:
        name = f"openai:{name}"
    if name.startswith("openai") and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in a .env file in the repo root (OPENAI_API_KEY=...), or use model 'dry-run'.")
    return name


def _fake_llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    tool = info.output_tools[0]
    title = tool.parameters_json_schema.get("title", "")
    return ModelResponse(parts=[ToolCallPart(tool.name, DRY_RUN_OUTPUTS[title])])


# --------------------------------------------------------------------------------------
# Technique vocabulary
# --------------------------------------------------------------------------------------

TechniqueCategory = Literal[
    "diffraction",  # XRD, neutron diffraction, SAED/electron diffraction
    "microscopy",  # SEM, TEM, HRTEM, AFM, STEM
    "spectroscopy",  # XPS, Raman, FTIR, UV-Vis, XAS, Mossbauer, NMR
    "magnetic",  # SQUID, VSM, magnetization vs T/H
    "electrical",  # resistivity, dielectric, impedance, ferroelectric P-E loops
    "thermal",  # TGA, DSC, DTA
    "computational",  # DFT, MD, simulations reported in the paper
    "other",
]

# Canonical name -> (category, aliases). One characterization agent per canonical name.
TECHNIQUES: dict[str, tuple[TechniqueCategory, tuple[str, ...]]] = {
    "XRD": ("diffraction", (
        "X-ray diffraction", "PXRD", "powder XRD", "SCXRD", "single-crystal XRD", "GIWAXS", "GIXRD",
        "Rietveld refinement", "Rietveld", "Reitveld refinement", "Reitveld", "Le Bail",
    )),
    "TEM": ("microscopy", (
        "transmission electron microscopy", "HRTEM", "STEM", "HAADF-STEM", "SAED", "SAD",
        "electron diffraction", "selected area electron diffraction", "cryo-TEM",
    )),
    "SEM": ("microscopy", ("scanning electron microscopy", "FESEM", "SEM-EDS", "SEM-EDX")),
    "AFM": ("microscopy", ("atomic force microscopy",)),
    "XPS": ("spectroscopy", ("X-ray photoelectron spectroscopy",)),
    "XAS": ("spectroscopy", ("X-ray absorption spectroscopy", "XANES", "EXAFS")),
    "Raman": ("spectroscopy", ("Raman spectroscopy",)),
    "FTIR": ("spectroscopy", ("IR", "infrared", "infrared spectroscopy", "ATR-FTIR", "ATR", "IRRAS", "IRAS", "RAIRS",
                              "PM-IRRAS", "DRIFTS")),
    "UV-Vis": ("spectroscopy", ("UV-Vis absorption", "UV-Vis-NIR", "diffuse reflectance")),
    "NMR": ("spectroscopy", ("nuclear magnetic resonance", "solid-state NMR", "ssNMR", "MAS NMR")),
    "Mossbauer": ("spectroscopy", ("Mossbauer spectroscopy",)),
    "TGA": ("thermal", ("thermogravimetric analysis",)),
    "DFT": ("computational", ("density functional theory", "first-principles calculations")),
    "Structure model": ("other", (
        "crystal structure visualization", "crystal structure model", "crystal structure rendering",
        "crystal structure view", "structure rendering", "structural model", "polyhedral model",
        "polyhedral representation", "ball-and-stick model", "ORTEP", "packing diagram",
    )),
}


def _key(name: str) -> str:
    """Case-, accent-, space- and punctuation-insensitive key: 'FT-IR' == 'ftir', 'Mössbauer' == 'Mossbauer'."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", ascii_name.lower())


_LOOKUP = {_key(alias): canonical for canonical, (_, aliases) in TECHNIQUES.items() for alias in (canonical, *aliases)}
_ISOTOPE = re.compile(r"^\s*\d+\s*[A-Z][a-z]?\b[\s-]*")  # '13C ', '31P-', '57Fe ' before a technique name


def canonical_technique(name: str) -> str | None:
    """Canonical technique name for any spelling or alias ('13C MAS NMR' is NMR), or None if not in
    the vocabulary."""
    return _LOOKUP.get(_key(name)) or _LOOKUP.get(_key(_ISOTOPE.sub("", name)))


def technique_category(name: str) -> TechniqueCategory | None:
    canonical = canonical_technique(name)
    return TECHNIQUES[canonical][0] if canonical else None


def technique_guide() -> str:
    """Instructions block listing canonical names, for agents that tag techniques."""
    lines = "\n".join(f"- {name} (also: {', '.join(aliases)})" for name, (_, aliases) in TECHNIQUES.items())
    return f"""
Use exactly one of these canonical technique names when one fits (aliases in brackets map to it):
{lines}
Otherwise use the name in the paper.
"""


# --------------------------------------------------------------------------------------
# Run configuration (campaign YAML)
# --------------------------------------------------------------------------------------


class CampaignConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Optional[str] = None  # every agent's model; None = $P2S_MODEL or DEFAULT_MODEL; "dry-run" for Agents 1-2 only
    reasoning_effort: Optional[str] = None  # Agents 3-5; None keeps the provider default
    max_retry: int = 3  # output-validation retries per agent call
    use_images: bool = True  # show rendered structures to the Construction and Revision agents


class InputsConfig(BaseModel):
    """The paper: a PDF that Agent 1 reads, or the paper_graph.json of an earlier summarizer run
    (figure paths are relative to its folder). `characterization` may name a folder of saved
    CharacterizationReport files (one JSON per technique, e.g. an earlier run's characterization/)
    to skip Agent 2."""
    model_config = ConfigDict(extra="forbid")
    pdf: Optional[str] = None
    paper_graph: Optional[str] = None
    characterization: Optional[str] = None


class SummarizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    analyze_figures: bool = True  # catalogue the figure images (one vision call per figure)
    max_figures: int = 30  # figure regions cut from the PDF


class CharacterizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    techniques: Optional[list[str]] = None  # only these techniques; None = every technique with evidence
    include_stubs: bool = False  # also run techniques whose spec is not filled in yet
    reasoning_effort: str = REASONING_EFFORT
    output_retries: int = 1


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
    # crystal chemistry of ionic substrates: bond-valence sums of the framework cations at interior sites
    bvs_warn: float = 0.30  # warn beyond this relative deviation from the oxidation state
    bvs_fail: float = 0.40  # ... and fail beyond this
    bond_share_warn: float = 0.45  # warn when one bond carries more than this share of its cation's valence
    like_contact_warn: float = 0.8  # cation-cation / anion-anion pairs below this x covalent sum warn
    like_contact_fail: float = 0.7  # ... and fail below this
    molecule_bond_tol: float = 0.12  # A: molecule bond lengths must match RDKit conformers of their SMILES
    molecule_angle_tol: float = 20.0  # degrees: ... and bond angles (RDKit's own spread: 0.09 A, 15 degrees)


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
    inputs: InputsConfig = Field(default_factory=InputsConfig)
    summarizer: SummarizerConfig = Field(default_factory=SummarizerConfig)
    characterization: CharacterizationConfig = Field(default_factory=CharacterizationConfig)
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
