from __future__ import annotations

import asyncio
import logging
from functools import cached_property

from pydantic_ai import Agent
from pydantic_ai.exceptions import (ModelConnectionError, ModelRateLimitError, ModelRequestTimeout, ModelServerError,
                                    ModelUnavailableError)
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage

from paper2structure.config import get_model

log = logging.getLogger(__name__)

# provider hiccups worth retrying; anything else (a tool crash, a budget overrun, a validation dead end)
# would only repeat the whole, possibly long, agent run
TRANSIENT_ERRORS = (ModelConnectionError, ModelRateLimitError, ModelRequestTimeout, ModelServerError,
                    ModelUnavailableError)


class StructuredAgent:
    """Shared class for the structured-output LLM agents. Each subclass sets `_output_type` and
    `_preflight_probe`; agents with tools or output validators also set `_deps_type` and
    override `_register`, and agents built from pydantic-ai(-harness) capabilities override
    `_capabilities`."""
    _deps_type = None
    max_attempts = 3
    retry_base_delay = 10.0  # seconds, for transient API errors

    def __init__(self, model: str | Model | None, effort: str | None, system_prompt: str,
                 system_context: str = "", max_retry: int = 3, output_retries: int | None = None):
        self.model = model
        self.effort = effort
        self.max_retry = max_retry
        self.output_retries = output_retries
        self.system_prompt = (f"{system_prompt}\n\n{system_context}" if system_context
                              else system_prompt)
        self.usage = {"input": 0, "cached_input": 0, "output": 0, "calls": 0, "tool_calls": 0}

    @cached_property
    def agent(self) -> Agent:
        """The pydantic-ai Agent, built on first use. `model` may be a model name (see
        paper2structure.config.get_model; None is the default model) or a pydantic-ai Model instance
        (e.g. a FunctionModel in tests)."""
        settings = {"openai_reasoning_effort": self.effort} if self.effort else {}
        model = self.model if isinstance(self.model, Model) else get_model(self.model)
        extra = {"deps_type": self._deps_type} if self._deps_type is not None else {}
        capabilities = self._capabilities()
        if capabilities:
            extra["capabilities"] = capabilities
        retries = (self.max_retry if self.output_retries is None
                   else {"tools": self.max_retry, "output": self.output_retries})
        agent = Agent(model=model, output_type=self._output_type, system_prompt=self.system_prompt,
                      model_settings=settings, retries=retries, **extra)
        self._register(agent)
        return agent

    def _capabilities(self) -> list:
        """Hook for subclasses: capabilities bound to the Agent (e.g. a harness Coder)."""
        return []

    def _register(self, agent: Agent) -> None:
        """Hook for subclasses to add tools and output validators."""

    def _accumulate(self, u: RunUsage) -> None:
        self.usage["input"] += u.input_tokens
        self.usage["cached_input"] += u.cache_read_tokens  # part of input, billed at the cached rate
        self.usage["output"] += u.output_tokens
        self.usage["calls"] += u.requests
        self.usage["tool_calls"] += u.tool_calls

    async def arun(self, prompt, deps=None, usage_limits=None, **run_kwargs):
        """One agent run returning the validated output; transient API errors are retried with
        exponential backoff, everything else is raised. `run_kwargs` go to Agent.run (e.g. the
        run's `workspace`). Token usage is recorded for failed runs too."""
        for attempt in range(1, self.max_attempts + 1):
            usage = RunUsage()
            try:
                result = await self.agent.run(prompt, deps=deps, usage_limits=usage_limits, usage=usage, **run_kwargs)
                break
            except TRANSIENT_ERRORS as e:
                if attempt == self.max_attempts:
                    raise
                delay = self.retry_base_delay * 2 ** (attempt - 1)
                log.warning("%s call failed (attempt %d/%d): %s; retrying in %.0fs",
                            type(self).__name__, attempt, self.max_attempts, e, delay)
                await asyncio.sleep(delay)
            finally:
                self._accumulate(usage)
        return result.output

    def preflight(self, **run_kwargs) -> tuple[bool, str]:
        try:
            self.agent.run_sync(self._preflight_probe, **run_kwargs)
            return True, ""
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"
