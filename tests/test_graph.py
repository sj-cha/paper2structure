"""Routing of the pydantic-graph workflow with scripted agents (no real LLM)."""
import asyncio
import json
import re

from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from helpers import (BAD_SCRIPT, GOOD_SCRIPT, PT_PLAN, REPORT, _final, _user_text, coding_model, constructor_model,
                     make_config, planner_model, reported, reviser_model)
from paper2structure.agents.structured_agent import StructuredAgent
from paper2structure.graph.workflow import build_deps, build_graph, run_workflow

LINEAR = ["SummarizePaper", "AnalyzeEvidence", "PlanStructures", "ConstructStructures", "ValidateStructures",
          "ScientificRevision", "FinalizeResults"]


def _run(tmp_path, scripts, actions, max_revisions=1, max_repairs=2, planner=None, max_rebuilds=2):
    deps = build_deps(make_config(), tmp_path / "run", seed=7, max_revisions=max_revisions, max_repairs=max_repairs,
                      models={"planner": planner or planner_model(), "constructor": constructor_model(scripts),
                              "reviser": reviser_model(actions)}, max_rebuilds=max_rebuilds)
    result = asyncio.run(run_workflow(deps, run_id="test"))
    final = sorted((tmp_path / "run" / "state").glob("*_FinalizeResults.json"))[-1]
    return result, json.loads(final.read_text())


def test_accept_path_writes_structures_and_report(tmp_path):
    result, state = _run(tmp_path, [GOOD_SCRIPT], ["accept_candidates"])
    assert result.status == "accepted" and state["node_history"] == LINEAR
    sid = result.accepted_structure_ids[0]
    run = tmp_path / "run"
    assert (run / result.structure_files[sid]["poscar"]).exists() and state["validation_reports"][sid]["passed"]
    checks = {c["name"] for c in state["validation_reports"][sid]["checks"]}
    assert {"overlaps", "adsorbate_binding", "export_roundtrip"} <= checks  # the gate's report + the round trip
    for name in ("plans.json", "artifacts.json", "validation_reports.json", "revision_history.json",
                 "token_log.json", "result.json", "workflow.mmd", "final_report.md",
                 "inputs/experimental_evidence_v1.json"):
        assert (run / name).exists(), name
    # the fixture's saved summary and reports replaced Agents 1-2
    assert state["summary"]["metadata"]["title"].startswith("[SYNTHETIC FIXTURE]")
    assert set(state["evidence"]["reports"]) == {"FTIR", "LEED", "XPS", "XRD"} and state["evidence_version"] == 1
    report = (run / "final_report.md").read_text()
    assert sid in report and "**Status:** accepted" in report and "## 1. Summary\nscripted summary" in report


def test_revisit_evidence_reruns_agents_2_to_4_but_not_agent_1(tmp_path):
    result, state = _run(tmp_path, [GOOD_SCRIPT], ["revisit_evidence", "accept_candidates"])
    h = state["node_history"]
    assert result.status == "accepted" and result.revisions == 1
    assert h.count("SummarizePaper") == 1
    assert h.count("AnalyzeEvidence") == h.count("PlanStructures") == h.count("ConstructStructures") == 2
    assert h[h.index("ScientificRevision") + 1] == "AnalyzeEvidence"  # the loop re-enters at Agent 2
    assert state["evidence_version"] == 2  # the kept reports are snapshotted again for the new revision
    assert len(state["plan_history"]) == 2 and len(state["artifacts"]) == 2  # earlier versions kept
    assert state["review_feedback"] == "use bridge sites as well"


def test_revision_budget_ends_the_loop(tmp_path):
    result, state = _run(tmp_path, [GOOD_SCRIPT], ["revisit_evidence"], max_revisions=1)
    assert result.status == "inconclusive" and result.revisions == 1
    assert state["node_history"].count("AnalyzeEvidence") == 2


def test_validation_failure_triggers_repair(tmp_path):
    result, state = _run(tmp_path, [BAD_SCRIPT, GOOD_SCRIPT], ["accept_candidates"])
    h = state["node_history"]
    assert h.count("ConstructStructures") == 2 and h.count("PlanStructures") == 1
    first, second = state["artifacts"]
    assert not state["validation_reports"][first["structure_id"]]["passed"]
    assert state["validation_reports"][second["structure_id"]]["passed"]
    assert result.accepted_structure_ids == [second["structure_id"]]


def test_repairs_are_bounded(tmp_path):
    result, state = _run(tmp_path, [BAD_SCRIPT], ["stop_inconclusive"], max_repairs=1)
    assert state["node_history"].count("ConstructStructures") == 2 and result.status == "inconclusive"


def test_agent_failure_finalizes_with_error(tmp_path, monkeypatch):
    monkeypatch.setattr(StructuredAgent, "retry_base_delay", 0.0)

    def down(messages, info):
        raise RuntimeError("model unavailable")

    result, state = _run(tmp_path, [GOOD_SCRIPT], ["accept_candidates"], planner=FunctionModel(down))
    assert result.status == "failed" and "planning failed" in result.errors[0]
    assert state["node_history"] == ["SummarizePaper", "AnalyzeEvidence", "PlanStructures", "FinalizeResults"]


def test_construction_failures_reach_the_planner(tmp_path):
    prompts = []

    def planner(messages, info):
        prompts.append(" ".join(str(p.content) for m in messages for p in m.parts if p.part_kind == "user-prompt"))
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, PT_PLAN)])

    def constructor(messages, info):  # reports the plan as unbuildable without running anything
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {
            "structures": [], "failed": [{"plan_id": "P1", "reason": "bulk generates TiO, not TiO2"}]})])

    deps = build_deps(make_config(), tmp_path / "run", seed=0, max_revisions=1,
                      models={"planner": FunctionModel(planner), "constructor": coding_model(constructor),
                              "reviser": reviser_model(["revisit_evidence"])})
    result = asyncio.run(run_workflow(deps, run_id="test"))
    assert result.status == "inconclusive" and result.revisions == 1 and len(prompts) == 2
    assert "Plans that could not be built" in prompts[1] and "bulk generates TiO" in prompts[1]


def test_graph_edges():
    text = build_graph().render(direction="TB")
    edges = re.findall(r"^\s*(\S+) --> (\S+)\s*$", text, re.M)
    into_agent1 = [src for src, dst in edges if dst == "SummarizePaper"]
    assert into_agent1 == ["start"]  # Agent 1 runs once
    decision = next(dst for src, dst in edges if src == "ScientificRevision")
    assert {dst for src, dst in edges if src == decision} == {"AnalyzeEvidence", "ConstructStructures", "FinalizeResults"}
    validate = next(dst for src, dst in edges if src == "ValidateStructures")
    assert {dst for src, dst in edges if src == validate} == {"ConstructStructures", "ScientificRevision"}


def test_rebuild_goes_back_to_construction_without_replanning(tmp_path):
    result, state = _run(tmp_path, [GOOD_SCRIPT, GOOD_SCRIPT], ["rebuild_structures", "accept_candidates"])
    assert result.status == "accepted" and state["revision"] == 0 and state["rebuilds"] == 1
    assert state["node_history"] == LINEAR[:6] + ["ConstructStructures", "ValidateStructures"] + LINEAR[5:]
    assert len(state["plan_history"]) == 1  # no re-planning
    first, second = state["revision_history"][0]["rebuild_structure_ids"][0], result.accepted_structure_ids[0]
    assert first.endswith("_r0a0") and second.endswith("_r0a1") and state["current_ids"] == [second]
    assert "rebuild" in state["repair_feedback"] and "CO must stand upright" in state["repair_feedback"]
    assert "CO tilted" in state["repair_feedback"]  # the reviewer's per-structure issues
    previous = tmp_path / "run" / "construction" / "r0_a1" / "P1" / "workspace" / "previous" / f"{first}.py"
    assert previous.exists()  # the rebuild starts from the script that built the rejected structure
    report = (tmp_path / "run" / "final_report.md").read_text()
    assert f"rebuild: {first}" in report


def test_rebuilds_are_bounded(tmp_path):
    result, state = _run(tmp_path, [GOOD_SCRIPT], ["rebuild_structures"], max_rebuilds=1)
    assert result.status == "inconclusive" and state["rebuilds"] == 1  # the second request turns into a stop
    assert [d["action"] for d in state["revision_history"]] == ["rebuild_structures", "stop_inconclusive"]


def test_rebuild_keeps_the_structures_the_reviewer_did_not_list(tmp_path):
    two = GOOD_SCRIPT + 'emit(model, name="pt111_co_copy", plan_id="P1")\n'  # the plan's two structures
    calls = []

    def reviewer(messages, info):
        ids = re.findall(r"### (\S+) \(plan", _user_text(messages))
        calls.append(ids)
        if len(calls) == 1:  # rebuild one of the two
            target = next(i for i in ids if i.startswith("pt111_co_top"))
            return _final(info, {"action": "rebuild_structures", "reasoning": "one is wrong",
                                 "rebuild_structure_ids": [target], "feedback": "fix the top-site CO"})
        if not reported(messages):
            return ModelResponse(parts=[ToolCallPart("write_report", REPORT)])
        return _final(info, {"action": "accept_candidates", "reasoning": "fine", "accepted_structure_ids": ids})

    deps = build_deps(make_config(), tmp_path / "run", seed=7,
                      models={"planner": planner_model(), "constructor": constructor_model([two, GOOD_SCRIPT]),
                              "reviser": coding_model(reviewer)})
    result = asyncio.run(run_workflow(deps, run_id="keep"))
    assert result.status == "accepted"
    assert sorted(calls[0]) == ["pt111_co_copy_r0a0", "pt111_co_top_r0a0"]
    assert sorted(result.accepted_structure_ids) == ["pt111_co_copy_r0a0", "pt111_co_top_r0a1"]  # copy kept
