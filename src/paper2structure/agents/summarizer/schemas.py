from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class SourceReference(BaseModel):
    """Where in the paper a statement comes from."""
    model_config = ConfigDict(extra="forbid")
    document: str = "main"
    page: Optional[int] = None
    section: Optional[str] = None
    figure: Optional[str] = None
    quote: Optional[str] = None


class PaperSummary(BaseModel):
    """Agent 1 output. Minimal placeholder schema while Agent 1 is a fixture loader."""
    model_config = ConfigDict(extra="forbid")
    title: str
    citation: str = ""
    synthetic: bool = Field(False, description="True for hand-written fixtures, not extracted from a real paper")
    materials: list[str]
    adsorbates: list[str] = Field(default_factory=list)
    objective: str
    methods: list[str] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list)
    author_interpretations: list[str] = Field(default_factory=list)
    reconstruction_target: str
    unresolved: list[str] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)
