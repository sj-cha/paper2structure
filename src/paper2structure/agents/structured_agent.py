from __future__ import annotations

from functools import cached_property

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel


class StructuredAgent:
    def __init__(self, model: str, effort: str | None, system_prompt: str,
                 system_context: str = ""):
        self.model = model
        self.effort = effort
        self.system_prompt = (f"{system_prompt}\n\n{system_context}" if system_context
                              else system_prompt)
        self.usage = {"input": 0, "output": 0, "calls": 0}

    @cached_property
    def agent(self) -> Agent:
        """The pydantic-ai Agent, built on first use."""
        settings = {"openai_reasoning_effort": self.effort} if self.effort else {}
        return Agent(model=OpenAIResponsesModel(self.model), output_type=self._output_type,
                     system_prompt=self.system_prompt, model_settings=settings)

    def _accumulate(self, result) -> None:
        self.usage["input"] += result.usage.input_tokens
        self.usage["output"] += result.usage.output_tokens
        self.usage["calls"] += 1

    def preflight(self) -> tuple[bool, str]:
        try:
            self.agent.run_sync(self._preflight_probe)
            return True, ""
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"
