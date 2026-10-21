"""Shared configuration: which LLM every agent uses, and the technique vocabulary.

Settings are read from a ``.env`` file in the repo root (never commit it; it is in .gitignore)::

    OPENAI_API_KEY=sk-...
    P2S_MODEL=openai:gpt-5.6-luna      # optional, any pydantic-ai model name

Special model name ``dry-run`` returns a fake model that answers instantly with canned data.
It costs nothing and is used to test that the plumbing works. Each agent package registers
its canned answers in ``DRY_RUN_OUTPUTS``, keyed by the title of its output schema.

The technique vocabulary is the one place where technique names live. The summarizer writes
canonical names into its evidence, and each characterization agent is keyed by one of them.
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Literal

from dotenv import load_dotenv
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
    name = name or os.getenv("P2S_MODEL") or DEFAULT_MODEL
    if name == "dry-run":
        return FunctionModel(_fake_llm, model_name="dry-run")
    if name.startswith("openai") and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in a .env file in the repo root (see .env.example), or use model 'dry-run'.")
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
    "spectroscopy",  # XPS, Raman, FTIR, UV-Vis, EDS/EDX, XAS, Mossbauer, NMR
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
        "Rietveld refinement", "Rietveld", "Le Bail",
    )),
    "neutron diffraction": ("diffraction", ("ND", "NPD", "neutron powder diffraction")),
    "TEM": ("microscopy", (
        "transmission electron microscopy", "HRTEM", "STEM", "HAADF-STEM", "SAED", "SAD",
        "electron diffraction", "cryo-TEM",
    )),
    "SEM": ("microscopy", ("scanning electron microscopy", "FESEM", "SEM-EDS", "SEM-EDX")),
    "AFM": ("microscopy", ("atomic force microscopy",)),
    "EDS": ("spectroscopy", ("EDX", "energy-dispersive X-ray spectroscopy", "EDS mapping")),
    "XPS": ("spectroscopy", ("X-ray photoelectron spectroscopy",)),
    "XAS": ("spectroscopy", ("X-ray absorption spectroscopy", "XANES", "EXAFS")),
    "Raman": ("spectroscopy", ("Raman spectroscopy",)),
    "FTIR": ("spectroscopy", ("IR", "infrared", "infrared spectroscopy", "ATR-FTIR", "ATR")),
    "UV-Vis": ("spectroscopy", ("UV-Vis absorption", "UV-Vis-NIR", "diffuse reflectance")),
    "PL": ("spectroscopy", ("photoluminescence", "TRPL")),
    "NMR": ("spectroscopy", ("nuclear magnetic resonance", "solid-state NMR", "ssNMR", "MAS NMR")),
    "Mossbauer": ("spectroscopy", ("Mossbauer spectroscopy",)),
    "magnetization": ("magnetic", ("SQUID", "VSM", "magnetic susceptibility", "M-T", "M-H")),
    "dielectric": ("electrical", ("dielectric spectroscopy", "impedance", "impedance spectroscopy")),
    "resistivity": ("electrical", ("electrical resistivity", "electrical transport")),
    "ferroelectric P-E": ("electrical", ("P-E loop", "P-E hysteresis", "polarization-electric field")),
    "TGA": ("thermal", ("thermogravimetric analysis",)),
    "DSC": ("thermal", ("differential scanning calorimetry",)),
    "DFT": ("computational", ("density functional theory", "first-principles calculations")),
}


def _key(name: str) -> str:
    """Case-, accent-, space- and punctuation-insensitive key: 'FT-IR' == 'ftir', 'Mössbauer' == 'Mossbauer'."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", ascii_name.lower())


_LOOKUP = {_key(alias): canonical for canonical, (_, aliases) in TECHNIQUES.items() for alias in (canonical, *aliases)}


def canonical_technique(name: str) -> str | None:
    """Canonical technique name for any spelling or alias, or None if not in the vocabulary."""
    return _LOOKUP.get(_key(name))


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
