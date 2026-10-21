"""Agent 2 placeholder: returns an evidence snapshot written beforehand (the
`inputs.experimental_evidence` file) instead of running the technique agents on the paper's data.
Re-entered on revision, it returns the same snapshot; the reviewer's feedback travels on to the
planner."""
from __future__ import annotations

from pathlib import Path

from paper2structure.agents.characterization.schemas import ExperimentalEvidence


def analyze_evidence(path: str | Path, feedback: str = "") -> ExperimentalEvidence:
    """The evidence snapshot stored at `path`; `feedback` (from the reviewer) is not used yet."""
    return ExperimentalEvidence.model_validate_json(Path(path).read_text())
