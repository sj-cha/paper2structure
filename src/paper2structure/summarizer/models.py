"""Choosing the LLM.

Settings are read from a ``.env`` file in the repo root (never commit it; it is in .gitignore)::

    OPENAI_API_KEY=sk-...
    P2S_MODEL=openai:gpt-5.4-mini      # optional, any pydantic-ai model name

Special model name ``dry-run`` returns a fake model that answers instantly with canned data.
It costs nothing and is used to test that the plumbing (PDF -> graph -> files) works.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel

DEFAULT_MODEL = "openai:gpt-5.4-mini"

# Models offered in the web page, with USD per 1M tokens (input / output), checked 2026-10-21.
MODEL_CHOICES = {
    "openai:gpt-5.6-luna": "GPT-5.6 Luna · $0.20/$1.20",
    "openai:gpt-6-luna": "GPT-6 Luna · price ?",
    "openai:gpt-5.4-nano": "GPT-5.4 nano · $0.20/$1.25",
    "openai:gpt-5-nano": "GPT-5 nano · $0.05/$0.40",
    "openai:gpt-5.4-mini": "GPT-5.4 mini · $0.75/$4.50",
    "dry-run": "Dry run · free",
}

load_dotenv()


def get_model(name: str | None = None) -> Model | str:
    name = name or os.getenv("P2S_MODEL") or DEFAULT_MODEL
    if name == "dry-run":
        return FunctionModel(_fake_llm, model_name="dry-run")
    if name.startswith("openai:") and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in a .env file in the repo root (see .env.example), or use model 'dry-run'.")
    return name


_CANNED = {
    "SystemsOutput": {
        "metadata": {"title": "Dry-run paper", "authors": ["A. Author"], "doi": None, "year": 2024, "summary": "Canned dry-run answer; no LLM was called."},
        "systems": [
            {"name": "SampleA", "family": "Sample series", "composition_variables": {"x": "0"}, "why_distinct": "x=0"},
            {"name": "SampleB", "family": "Sample series", "composition_variables": {"x": "0.2"}, "why_distinct": "x=0.2"},
        ],
    },
    "TextEvidenceOutput": {
        "synthesis": {"method": "solid-state reaction", "precursors": [], "conditions": ["1250 °C, 5 h"]},
        "structure": {"crystal_system": "orthorhombic", "space_group": "Pnma", "lattice_parameters": [{"quantity": "a", "value": "5.48", "unit": "Å"}]},
        "evidence": [
            {
                "system_ids": ["S1", "S2"],
                "technique": "XRD",
                "technique_category": "diffraction",
                "summary": "Single phase by XRD.",
                "conditions": ["Cu-Kα"],
                "quote": "dry-run quote",
                "page": 1,
            }
        ],
    },
    "FigureOutput": {
        "figure_label": "Fig. 1",
        "panels": [
            {"panel_label": "a", "plot_type": "XRD pattern", "technique": "XRD", "technique_category": "diffraction", "system_ids": ["S1"], "description": "Dry-run panel."}
        ],
        "useful_for_structure": True,
    },
}


def _fake_llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    tool = info.output_tools[0]
    title = tool.parameters_json_schema.get("title", "")
    return ModelResponse(parts=[ToolCallPart(tool.name, _CANNED[title])])
