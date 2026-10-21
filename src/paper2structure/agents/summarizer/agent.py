"""Agent 1 placeholder: returns a paper summary written beforehand (the `inputs.paper_summary`
file) instead of reading the paper."""
from __future__ import annotations

from pathlib import Path

from paper2structure.agents.summarizer.schemas import PaperSummary


def summarize_paper(path: str | Path) -> PaperSummary:
    """The paper summary stored at `path`."""
    return PaperSummary.model_validate_json(Path(path).read_text())
