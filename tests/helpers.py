"""Scripted (FunctionModel) agents and fixture data for offline tests. Nothing here comes from a
real publication or a real LLM."""
from __future__ import annotations

import copy
import re
from pathlib import Path

from pydantic_ai.messages import ModelResponse, RetryPromptPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from paper2structure.config import Paper2StructureConfig

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "examples" / "fixtures"

PT_PLAN = {
    "plans": [{
        "plan_id": "P1",
        "hypothesis": "CO binds atop Pt at 0.25 ML (one CO per 2x2 cell)",
        "model_type": "slab",
        "description": "Pt(111) slab, p(2x2), one CO upright on a top site",
        "crystals": [{"name": "bulk", "formula": "Pt", "space_group": "Fm-3m",
                      "sites": [{"species": "Pt", "coords": [0.0, 0.0, 0.0], "wyckoff": "4a"}],
                      "lattice": {"a": 3.924, "source": "paper", "evidence_ids": ["E1"]}, "formula_units": 4}],
        "surface": {"miller": [1, 1, 1], "termination": "Pt", "min_thickness": 6.0, "supercell": [2, 2],
                    "vacuum": 15.0, "fix_bottom": 2.0},
        "adsorbates": [{"name": "CO", "smiles": "[C-]#[O+]", "binding_atoms": ["C"], "anchor_smarts": "[C-]#[O+]",
                        "mode": "adsorb", "site_kind": "top", "site_elements": ["Pt"], "site": "atop Pt",
                        "count": 1, "bond_length": 1.85}],
        "targets": [{"property": "CO coverage", "value": "0.25 ML", "evidence_ids": ["E2"]}],
        "system_ids": ["S1"],
        "evidence_ids": ["E2", "E3"],
        "assumptions": ["CO is upright"],
    }],
    "rationale": "scripted test plan",
}

SCRIPT = '''\
from paper2structure.tools import Adsorbate, BindingMotif, build_bulk, build_slab, describe, emit, place_adsorbates, surface_sites
SEED = 7
bulk = build_bulk("Fm-3m", ["Pt"], [[0, 0, 0]], a=3.924, formula_units=4)
slab = build_slab(bulk, (1, 1, 1), termination=0, min_thickness=6.0, vacuum=15.0, supercell=(2, 2), fix_bottom=2.0)
co = Adsorbate.from_smiles("[C-]#[O+]", BindingMotif(["C"], anchor="[C-]#[O+]"), name="CO", seed=SEED)
model = place_adsorbates(slab, co, surface_sites(slab, kinds=("top",), elements=["Pt"]), count=1, bond_length={bond},
                      seed=SEED)
print(describe(model))
emit(model, name="pt111_co_top", plan_id="P1")
'''
GOOD_SCRIPT = SCRIPT.format(bond=1.85)
BAD_SCRIPT = SCRIPT.format(bond=0.4)  # C pushed into the Pt site: fails the overlap check


def make_config(scenario: str = "pt111_co", **overrides) -> Paper2StructureConfig:
    """A run on a synthetic fixture: its saved summary and characterization reports replace Agents 1-2."""
    raw = {"inputs": {"paper_graph": str(FIXTURES / scenario / "paper_graph.json"),
                      "characterization": str(FIXTURES / scenario / "characterization")},
           "construction": {"code_timeout": 240.0, "memory_limit_gb": None}}
    for key, value in overrides.items():
        raw.setdefault(key, {}).update(value)
    return Paper2StructureConfig.model_validate(raw)


def coding_model(fn) -> FunctionModel:
    """A FunctionModel that also streams: harness capabilities (Coder's file and shell events) run
    the agent with streamed requests. `fn` returns tool calls only."""
    async def stream(messages, info: AgentInfo):
        for k, part in enumerate(fn(messages, info).parts):
            yield {k: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str(),
                                    tool_call_id=part.tool_call_id)}
    return FunctionModel(fn, stream_function=stream)


def _final(info: AgentInfo, payload: dict) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)])


def _user_text(messages) -> str:
    for part in messages[0].parts:
        if isinstance(part, UserPromptPart):
            content = part.content
            return content if isinstance(content, str) else " ".join(c for c in content if isinstance(c, str))
    return ""


def planner_model(plans: list[dict] | None = None) -> FunctionModel:
    """Returns the given plan sets in order (the last one repeats)."""
    seq = [copy.deepcopy(p) for p in (plans or [PT_PLAN])]
    calls = {"n": 0}

    def fn(messages, info: AgentInfo) -> ModelResponse:
        k = min(calls["n"], len(seq) - 1)
        calls["n"] += 1
        return _final(info, seq[k])
    return FunctionModel(fn)


def _emits(script: str) -> list[tuple[str, str]]:
    return re.findall(r'emit\(\w+, name="([^"]+)", plan_id="([^"]+)"', script)


def constructor_model(scripts: list[str]) -> FunctionModel:
    """A scripted coding agent: each construction run writes the next script to build_P1.py, runs it
    with the shell, checks what it emitted and reports it."""
    calls = {"n": 0}
    current = {"script": scripts[0]}

    def fn(messages, info: AgentInfo) -> ModelResponse:
        done = [p.tool_name for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
        last = done[-1] if done else None
        if last is None:
            current["script"] = scripts[min(calls["n"], len(scripts) - 1)]
            calls["n"] += 1
            return ModelResponse(parts=[ToolCallPart("write_file", {"path": "build_P1.py", "content": current["script"]})])
        if last == "write_file":
            return ModelResponse(parts=[ToolCallPart("shell", {"command": "python build_P1.py"})])
        if last == "shell":
            name = _emits(current["script"])[0][0]
            return ModelResponse(parts=[ToolCallPart("check_structure", {"name": name})])
        return _final(info, {"structures": [{"name": n, "plan_id": p, "script": "build_P1.py", "method": "scripted"}
                                            for n, p in _emits(current["script"])],
                             "failed": [], "summary": "scripted"})
    return coding_model(fn)


REPORT = {"summary": "scripted summary", "assessment": "scripted assessment", "next_steps": ["relax with an MLIP"]}


def reported(messages) -> bool:
    return any(isinstance(p, ToolReturnPart) and p.tool_name == "write_report" for m in messages for p in m.parts)


def reviser_model(actions: list[str]) -> FunctionModel:
    """Returns the given actions in order; accepts (or asks to rebuild) every structure it is shown
    and writes the report before a decision that ends the run. When the prompt says no revisions
    (rebuilds) are left, revisit_evidence (rebuild_structures) turns into stop_inconclusive."""
    calls = {"n": 0}

    def fn(messages, info: AgentInfo) -> ModelResponse:
        retried = any(isinstance(p, RetryPromptPart) for m in messages for p in m.parts)
        text = _user_text(messages)
        ids = re.findall(r"### (\S+) \(plan", text)
        action = actions[min(calls["n"], len(actions) - 1)]
        if action == "revisit_evidence" and "Revisions left: 0" in text:
            action = "stop_inconclusive"
        if action == "rebuild_structures" and "Rebuilds left: 0" in text:
            action = "stop_inconclusive"
        if action in ("accept_candidates", "stop_inconclusive") and not reported(messages):
            return ModelResponse(parts=[ToolCallPart("write_report", REPORT)])
        if not retried:
            calls["n"] += 1
        feedback = {"revisit_evidence": "use bridge sites as well",
                    "rebuild_structures": "scripted rebuild: the CO must stand upright"}.get(action, "")
        payload = {"action": action, "reasoning": f"scripted {action}",
                   "verdicts": [{"structure_id": i, "resembles_paper": action == "accept_candidates",
                                 "issues": ["CO tilted"] if action == "rebuild_structures" else []} for i in ids],
                   "accepted_structure_ids": ids if action == "accept_candidates" else [],
                   "rebuild_structure_ids": ids if action == "rebuild_structures" else [],
                   "feedback": feedback}
        return _final(info, payload)
    return FunctionModel(fn)
