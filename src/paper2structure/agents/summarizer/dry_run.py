"""Canned answers for the ``dry-run`` model (see ``paper2structure.config.get_model``)."""

from __future__ import annotations

from ...config import DRY_RUN_OUTPUTS

DRY_RUN_OUTPUTS.update({
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
})
