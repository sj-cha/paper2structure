import json
import mimetypes
from pathlib import Path
from typing import Optional, TypeVar

from pydantic_ai import Agent, BinaryContent, UsageLimits
from pydantic_ai.models.openai import OpenAIResponsesModel

from .schemas import CharacterizationReport, FigureInput, TextInput
from .spec import GENERIC, TechniqueSpec, generate_system_prompt

InputT = TypeVar("InputT", FigureInput, TextInput)


class CharacterizationAgent:
    def __init__(
        self,
        spec: TechniqueSpec = GENERIC,
        model: str = "gpt-5.6-luna",
        reasoning_effort: str = "high",
        output_retries: int = 1,
    ):
        self._spec = spec
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._output_retries = output_retries
        self._agent = None

    @property
    def spec(self) -> TechniqueSpec:
        return self._spec

    def _ensure(self):
        if self._agent is not None:
            return

        self._agent = Agent(
            model=OpenAIResponsesModel(self._model),
            output_type=CharacterizationReport,
            system_prompt=self._system_prompt(),
            retries={"output": self._output_retries, "tools": 0},
            model_settings={
                "openai_reasoning_effort": self._reasoning_effort,
            },
        )

    def _system_prompt(self) -> str:
        return generate_system_prompt(self._spec)

    def select(self, items: list[InputT]) -> list[InputT]:
        """Keep only figures or text passages tagged with this technique."""
        if self._spec is GENERIC:
            return list(items)
        return [item for item in items if self._spec.matches(item.technique)]

    def _user_prompt(self, figures: list[FigureInput], texts: list[TextInput]) -> list:
        records = [f.model_dump(exclude={"image_path"}, exclude_none=True) for f in figures]
        passages = [t.model_dump(exclude_none=True) for t in texts]
        content: list = [
            f"Figure records:\n{json.dumps(records, indent=2)}",
            f"Text records:\n{json.dumps(passages, indent=2)}",
        ]

        for figure in figures:
            path = Path(figure.image_path)
            media_type = mimetypes.guess_type(path.name)[0] or "image/png"
            content.append(f"Image for {figure.figure_id}:")
            content.append(BinaryContent(data=path.read_bytes(), media_type=media_type))

        return content

    async def analyze(
        self,
        figures: Optional[list[FigureInput]] = None,
        texts: Optional[list[TextInput]] = None,
        filename: Optional[str] = None,
        filter_by_technique: bool = True,
    ) -> CharacterizationReport:
        figures = list(figures or [])
        texts = list(texts or [])
        if filter_by_technique:
            figures = self.select(figures)
            texts = self.select(texts)

        if not figures and not texts:
            report = CharacterizationReport(
                technique=self._spec.name,
                caveats=[f"No figures or text tagged as {self._spec.name}; no analysis performed."],
            )
        else:
            self._ensure()
            result = await self._agent.run(
                self._user_prompt(figures, texts),
                usage_limits=UsageLimits(request_limit=1 + self._output_retries),
            )
            report = result.output

        if filename is not None:
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(report.model_dump_json(indent=2) + "\n")

        return report
