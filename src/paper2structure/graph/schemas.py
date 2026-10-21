from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RunStatus = Literal["accepted", "inconclusive", "revision_budget_exhausted", "failed"]


class ReconstructionResult(BaseModel):
    """What a run returns (also written to result.json); paths are relative to run_dir."""
    run_id: str
    status: RunStatus
    run_dir: str
    accepted_structure_ids: list[str] = Field(default_factory=list)
    structure_files: dict[str, dict[str, str]] = Field(default_factory=dict)
    report: str = ""
    revisions: int = 0
    errors: list[str] = Field(default_factory=list)
