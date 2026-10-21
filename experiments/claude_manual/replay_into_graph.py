"""Feed Claude's hand-made answers (step1/2/3 JSON files) through the real pydantic graph.

The graph thinks it is talking to an LLM, but the "LLM" is ``replay`` below, which looks up the
answer Claude wrote for that question. Because the answers go through the same schemas, the same
validators and the same BuildGraph step, the output folder has exactly the format the OpenAI run
produces, so you can compare the two side by side.

    cd experiments/claude_manual
    ../../.venv/bin/python replay_into_graph.py ../../../papers/La1-xBixCrO3_arXiv1403.3526.pdf
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from paper2structure.summarizer.graph import summarize_paper_sync

HERE = Path(__file__).parent
STEP1 = json.loads((HERE / "step1_find_systems.json").read_text())
STEP2 = json.loads((HERE / "step2_text_evidence.json").read_text())
STEP3 = {f.pop("figure_id"): f for f in json.loads((HERE / "step3_figures.json").read_text())}


def _prompt_text(messages: list[ModelMessage]) -> str:
    parts = [p for m in messages for p in getattr(m, "parts", []) if isinstance(p, UserPromptPart)]
    content = parts[0].content
    return content if isinstance(content, str) else next(c for c in content if isinstance(c, str))


def replay(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    tool = info.output_tools[0]
    kind = tool.parameters_json_schema.get("title")
    prompt = _prompt_text(messages)
    if kind == "SystemsOutput":
        answer = STEP1
    elif kind == "TextEvidenceOutput":
        answer = STEP2[re.search(r"TARGET SYSTEM: (S\d+)", prompt).group(1)]
    else:
        answer = STEP3[re.search(r"Figure id: (\S+)", prompt).group(1)]
    return ModelResponse(parts=[ToolCallPart(tool.name, answer)])


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "../../../papers/La1-xBixCrO3_arXiv1403.3526.pdf")
    summarize_paper_sync(pdf, HERE / "output", FunctionModel(replay, model_name="claude-manual"))
