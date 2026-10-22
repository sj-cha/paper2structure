import json
import mimetypes
from pathlib import Path
from typing import Optional

from pydantic_ai import BinaryContent, UsageLimits
from pydantic_ai.models import Model

from ...config import REASONING_EFFORT
from ..structured_agent import StructuredAgent
from ..summarizer.schemas import Evidence, MaterialSystem
from . import dry_run  # noqa: F401  (registers the canned answer for the dry-run model)
from .schemas import CharacterizationReport
from .spec import GENERIC, TechniqueSpec, generate_system_prompt


def _compact(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


class CharacterizationAgent(StructuredAgent):
    """The characterization agent for one technique: reads the figures and text records tagged with it. Transient API
    errors are retried and token usage is counted (`usage`) like the other agents."""
    _output_type = CharacterizationReport
    _preflight_probe = "Health check: return a report for technique 'health check' with no observations."

    def __init__(
        self,
        spec: TechniqueSpec = GENERIC,
        model: Model | str | None = None,
        reasoning_effort: str = REASONING_EFFORT,
        output_retries: int = 1,
    ):
        super().__init__(model, reasoning_effort, generate_system_prompt(spec), max_retry=0,
                         output_retries=output_retries)
        self._spec = spec

    def select(self, evidence: list[Evidence]) -> list[Evidence]:
        """Keep only evidence (text statements and figure panels) for this technique."""
        if self._spec is GENERIC:
            return list(evidence)
        return [e for e in evidence if self._spec.matches(e.technique)]

    @staticmethod
    def _figure_records(figure_evidence: list[Evidence]) -> dict[str, dict]:
        """One record per image: a cropped figure can hold panels of several techniques."""
        records: dict[str, dict] = {}
        for e in figure_evidence:
            record = records.setdefault(e.image_path, {
                "figure_id": Path(e.image_path).stem,
                "caption": e.quote or "",
                "context": e.context,
                "panels": [],
            })
            record["panels"].append({
                "panel": e.source_ref,
                "description": e.summary,
                "axes": e.conditions,
                "system_ids": e.system_ids,
            })
        return records

    def _user_prompt(self, evidence: list[Evidence], systems: list[MaterialSystem], base_dir: Optional[Path]) -> list:
        figures = self._figure_records([e for e in evidence if e.source_type == "figure" and e.image_path])
        texts = [
            e.model_dump(include={"evidence_id", "system_ids", "summary", "measurements", "conditions", "quote", "source_ref", "figure_refs"}, exclude_none=True)
            for e in evidence if e.source_type == "text"
        ]
        system_table = [s.model_dump(include={"system_id", "name", "composition_variables", "form"}, exclude_none=True) for s in systems]
        content: list = [
            f"Material systems:\n{_compact(system_table)}",
            f"Figure records:\n{_compact(list(figures.values()))}",
            f"Text records:\n{_compact(texts)}",
        ]

        for image_path, record in figures.items():
            path = base_dir / image_path if base_dir else Path(image_path)
            media_type = mimetypes.guess_type(path.name)[0] or "image/png"
            content.append(f"Image for {record['figure_id']}:")
            content.append(BinaryContent(data=path.read_bytes(), media_type=media_type))

        return content

    async def analyze(
        self,
        evidence: list[Evidence],
        systems: Optional[list[MaterialSystem]] = None,
        base_dir: Optional[str | Path] = None,
        filename: Optional[str | Path] = None,
        filter_by_technique: bool = True,
    ) -> CharacterizationReport:
        """Analyze the figure and text evidence for this technique.

        ``base_dir`` is the summarizer output folder that figure ``image_path``s are relative to.
        """
        if filter_by_technique:
            evidence = self.select(evidence)

        if not evidence:
            report = CharacterizationReport(
                technique=self._spec.name,
                caveats=[f"No figures or text tagged as {self._spec.name}; no analysis performed."],
            )
        else:
            report = await self.arun(
                self._user_prompt(evidence, list(systems or []), Path(base_dir) if base_dir else None),
                usage_limits=UsageLimits(request_limit=1 + self.output_retries),
            )
            report.technique = self._spec.name

        if filename is not None:
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(report.model_dump_json(indent=2) + "\n")

        return report
