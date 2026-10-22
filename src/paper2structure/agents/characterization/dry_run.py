"""Canned answer for the ``dry-run`` model (see ``paper2structure.config.get_model``)."""

from ...config import DRY_RUN_OUTPUTS

DRY_RUN_OUTPUTS["CharacterizationReport"] = {
    "observations": [
        {"observation_id": "O1", "source_id": "E001", "system_ids": ["S1"], "quantity": "phase", "value": "single phase", "read_from": "stated_in_text"},
    ],
    "findings": [{"claim": "Canned dry-run finding; no LLM was called.", "system_ids": ["S1"], "evidence": ["O1"], "confidence": "low"}],
    "structural_constraints": [
        {"property": "phase", "system_ids": ["S1"], "value": "single phase", "evidence": ["O1"], "confidence": "low"},
    ],
    "caveats": ["Dry run."],
}
