from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RunStatus = Literal["accepted", "inconclusive", "failed"]


class ReconstructionResult(BaseModel):
    """What a run returns (also written to result.json); paths are relative to run_dir."""
    run_id: str
    status: RunStatus
    run_dir: str
    accepted_structure_ids: list[str] = Field(default_factory=list)
    current_structure_ids: list[str] = Field(default_factory=list)  # the latest plan set's structures; the other
    # ones in structure_files were rebuilt, repaired or belong to earlier plan sets
    structure_files: dict[str, dict[str, str]] = Field(default_factory=dict)
    report: str = ""
    revisions: int = 0
    errors: list[str] = Field(default_factory=list)
