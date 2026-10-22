"""Routing of the pydantic-graph workflow with scripted agents (no real LLM)."""
import asyncio
import copy
import json
import re

from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from helpers import (BAD_SCRIPT, GOOD_SCRIPT, PT_PLAN, REPORT, _final, _user_text, coding_model, constructor_model,
                     make_config, planner_model, reported, reviser_model)
from paper2structure.agents.structured_agent import StructuredAgent
from paper2structure.graph.workflow import build_deps, build_graph, run_workflow

LINEAR = ["SummarizePaper", "AnalyzeEvidence", "PlanStructures", "ConstructStructures", "ValidateStructures",
          "ScientificRevision", "FinalizeResults"]


def _verdicts(messages) -> list[dict]:
    """A verdict for every structure the reviewer is shown (its validator requires one each)."""
    return [{"structure_id": i, "resembles_paper": True} for i in re.findall(r"### (\S+) \(plan", _user_text(messages))]


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
    checks = {c["name"]: c for c in state["validation_reports"][sid]["checks"]}
    assert {"overlaps", "adsorbate_binding", "export_roundtrip"} <= set(checks)  # the gate's report + the round trip
    assert checks["claim C1"]["status"] == "pass" and "Pt-Pt 2.77" in checks["claim C1"]["detail"]  # the plan's claim
    for name in ("plans.json", "artifacts.json", "validation_reports.json", "revision_history.json",
                 "token_log.json", "result.json", "workflow.mmd", "final_report.md",
                 "construction_reports.json", "inputs/experimental_evidence.json"):
        assert (run / name).exists(), name
    # the fixture's saved summary and reports replaced the Summarizer and the characterization agents
    assert state["summary"]["metadata"]["title"].startswith("[SYNTHETIC FIXTURE]")
    assert set(state["evidence"]["reports"]) == {"FTIR", "LEED", "XPS", "XRD"}
    assert result.current_structure_ids == [sid] and json.loads((run / "result.json").read_text())["current_structure_ids"]
    tokens = json.loads((run / "token_log.json").read_text())
    assert {"summarizer", "characterization", "planner", "constructor", "reviser"} <= set(tokens)
    report = (run / "final_report.md").read_text()
    assert sid in report and "**Status:** accepted" in report and "## 1. Summary\nscripted summary" in report
    assert f"| {sid} | C1 | pass |" in report and "unchecked claims: S1:occupancy1" in report


def test_revisit_evidence_reruns_characterization_to_construction_but_not_the_summarizer(tmp_path):
    result, state = _run(tmp_path, [GOOD_SCRIPT], ["revisit_evidence", "accept_candidates"])
    h = state["node_history"]
    assert result.status == "accepted" and result.revisions == 1
    assert h.count("SummarizePaper") == 1
    assert h.count("AnalyzeEvidence") == h.count("PlanStructures") == h.count("ConstructStructures") == 2
    assert h[h.index("ScientificRevision") + 1] == "AnalyzeEvidence"  # the loop re-enters at the characterization step
    run = tmp_path / "run"
    assert sorted(p.name for p in (run / "inputs").iterdir()) == ["experimental_evidence.json", "paper_graph.json"]
    assert len(state["plan_history"]) == 2 and len(state["artifacts"]) == 2  # earlier versions kept
    assert len(json.loads((run / "plans.json").read_text())) == 2 and not (run / "plans").exists()
    assert state["review_feedback"] == "use bridge sites as well"
    first, second = (a["structure_id"] for a in state["artifacts"])
    assert state["candidate_ids"] == [first] and state["current_ids"] == [second]  # the first stays acceptable
    assert result.current_structure_ids == [second]


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


def test_loop_budgets_come_from_the_config(tmp_path):
    deps = build_deps(make_config(budgets={"max_revisions": 0, "max_rebuilds": 3}), tmp_path / "a", seed=0)
    assert (deps.max_revisions, deps.max_repairs, deps.max_rebuilds) == (0, 2, 3)
    assert build_deps(make_config(), tmp_path / "b", seed=0, max_repairs=0).max_repairs == 0  # an explicit override


def test_graph_edges():
    text = build_graph().render(direction="TB")
    edges = re.findall(r"^\s*(\S+) --> (\S+)\s*$", text, re.M)
    into_summarizer = [src for src, dst in edges if dst == "SummarizePaper"]
    assert into_summarizer == ["start"]  # the Summarizer runs once
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
                                 "rebuild_structure_ids": [target], "feedback": "fix the top-site CO",
                                 "verdicts": _verdicts(messages)})
        if not reported(messages):
            return ModelResponse(parts=[ToolCallPart("write_report", REPORT)])
        return _final(info, {"action": "accept_candidates", "reasoning": "fine", "accepted_structure_ids": ids,
                             "verdicts": _verdicts(messages)})

    deps = build_deps(make_config(), tmp_path / "run", seed=7,
                      models={"planner": planner_model(), "constructor": constructor_model([two, GOOD_SCRIPT]),
                              "reviser": coding_model(reviewer)})
    result = asyncio.run(run_workflow(deps, run_id="keep"))
    assert result.status == "accepted"
    assert sorted(calls[0]) == ["pt111_co_copy_r0a0", "pt111_co_top_r0a0"]
    assert sorted(result.accepted_structure_ids) == ["pt111_co_copy_r0a0", "pt111_co_top_r0a1"]  # copy kept


def test_reviewer_runs_its_own_claim_checks(tmp_path):
    results = []

    def reviewer(messages, info):
        sid = re.findall(r"### (\S+) \(plan", _user_text(messages))[0]
        returns = [p.content for m in messages for p in m.parts if p.part_kind == "tool-return"
                   and p.tool_name == "check_claim"]
        if not returns:
            return ModelResponse(parts=[ToolCallPart("check_claim", {"structure_id": sid, "check": {
                "kind": "coordination", "id": "R1", "claim": "Pt is 12-coordinated in the bulk",
                "center": "Pt", "neighbor": "Pt", "number": 12}})])
        results.extend(returns)
        if not reported(messages):
            return ModelResponse(parts=[ToolCallPart("write_report", REPORT)])
        return _final(info, {"action": "accept_candidates", "reasoning": "fine", "accepted_structure_ids": [sid],
                             "verdicts": _verdicts(messages)})

    deps = build_deps(make_config(), tmp_path / "run", seed=7,
                      models={"planner": planner_model(), "constructor": constructor_model([GOOD_SCRIPT]),
                              "reviser": coding_model(reviewer)})
    result = asyncio.run(run_workflow(deps, run_id="claims"))
    assert result.status == "accepted"
    # the thin slab has no Pt deep enough to count as bulk: the reviewer's check says so
    assert results and results[0].startswith("fail: Pt is 12-coordinated") and "no interior framework Pt" in results[0]
    prompt = (tmp_path / "run" / "final_report.md").read_text()
    assert "| C1 | pass |" in prompt
    # the check is kept with its result, for the next review and the report
    state = json.loads(sorted((tmp_path / "run" / "state").glob("*_FinalizeResults.json"))[-1].read_text())
    assert [(c["review"], c["check_id"], c["status"]) for c in state["review_checks"]] == [(0, "R1", "fail")]
    assert "Claims the reviewer checked itself" in prompt and "Pt is 12-coordinated in the bulk" in prompt


def _for_plan(script: str, plan_id: str) -> str:
    return script.replace('name="pt111_co_top", plan_id="P1"', f'name="pt111_co_{plan_id.lower()}", plan_id="{plan_id}"')


def per_plan_constructor(scripts: dict[str, list[str]], runs: dict[str, int]):
    """A scripted coding agent for one plan per run (the plan in its prompt): the plan's k-th run
    writes scripts[plan][k] (the last one repeats), runs it, checks it and reports it."""
    def fn(messages, info):
        pid = re.search(r'"plan_id":"(P\d+)"', _user_text(messages)).group(1)
        done = [p.tool_name for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
        if not done:
            k = runs[pid] = runs.get(pid, 0) + 1
            return ModelResponse(parts=[ToolCallPart("write_file", {"path": f"build_{pid}.py",
                                                                    "content": scripts[pid][min(k, len(scripts[pid])) - 1]})])
        script = next(p.args_as_dict()["content"] for m in messages for p in m.parts
                      if isinstance(p, ToolCallPart) and p.tool_name == "write_file")
        name = re.search(r'emit\(\w+, name="([^"]+)"', script).group(1)
        if done[-1] == "write_file":
            return ModelResponse(parts=[ToolCallPart("shell", {"command": f"python build_{pid}.py"})])
        if done[-1] == "shell":
            return ModelResponse(parts=[ToolCallPart("check_structure", {"name": name})])
        return _final(info, {"structures": [{"name": name, "plan_id": pid, "script": f"build_{pid}.py",
                                             "method": "scripted"}], "failed": [], "summary": f"scripted {pid}"})
    return coding_model(fn)


def test_feedback_repairs_and_rebuilds_stay_with_their_plans(tmp_path):
    """Two plans: each repair run gets only its own plan's validation notes; a rebuild request for
    one plan reopens only that plan, survives the validation repair of the rebuilt structure, and
    the other plan's structure stays current throughout."""
    two = copy.deepcopy(PT_PLAN)
    two["plans"].append({**copy.deepcopy(two["plans"][0]), "plan_id": "P2"})
    runs, reviews = {}, []
    constructor = per_plan_constructor({"P1": [BAD_SCRIPT, GOOD_SCRIPT, BAD_SCRIPT, GOOD_SCRIPT],
                                        "P2": [_for_plan(BAD_SCRIPT, "P2"), _for_plan(GOOD_SCRIPT, "P2")]}, runs)

    def reviewer(messages, info):
        shown = dict(re.findall(r"### (\S+) \(plan (P\d+)", _user_text(messages)))
        if not reviews or reviews[-1] != shown:
            reviews.append(shown)
        if len(reviews) == 1:
            p1 = next(sid for sid, pid in shown.items() if pid == "P1")
            return _final(info, {"action": "rebuild_structures", "reasoning": "P1 is off", "feedback": "fix the P1 CO",
                                 "rebuild_structure_ids": [p1], "verdicts": _verdicts(messages)})
        if not reported(messages):
            return ModelResponse(parts=[ToolCallPart("write_report", REPORT)])
        return _final(info, {"action": "accept_candidates", "reasoning": "fine", "accepted_structure_ids": list(shown),
                             "verdicts": _verdicts(messages)})

    deps = build_deps(make_config(), tmp_path / "run", seed=7, models={
        "planner": planner_model([two]), "constructor": constructor, "reviser": coding_model(reviewer)})
    result = asyncio.run(run_workflow(deps, run_id="plans"))
    run = tmp_path / "run"
    state = json.loads(sorted((run / "state").glob("*_FinalizeResults.json"))[-1].read_text())

    def feedback(attempt, pid):
        return (run / "construction" / f"r0_a{attempt}" / pid / "workspace" / "task" / "feedback.md").read_text()

    assert result.status == "accepted" and runs == {"P1": 4, "P2": 2}  # the rebuild and its repair leave P2 alone
    assert "pt111_co_top" in feedback(1, "P1") and "pt111_co_p2" not in feedback(1, "P1")  # its own notes only
    assert "pt111_co_p2" in feedback(1, "P2") and "pt111_co_top" not in feedback(1, "P2")
    assert "fix the P1 CO" in feedback(2, "P1")
    repair = feedback(3, "P1")  # the rebuilt structure failed validation: the reviewer's request still stands
    assert "fix the P1 CO" in repair and "Deterministic validation failed" in repair
    assert not (run / "construction" / "r0_a2" / "P2").exists() and not (run / "construction" / "r0_a3" / "P2").exists()
    assert sorted(state["current_ids"]) == ["pt111_co_p2_r0a1", "pt111_co_top_r0a3"]
    assert reviews[1] == {"pt111_co_p2_r0a1": "P2", "pt111_co_top_r0a3": "P1"}
    assert sorted(result.accepted_structure_ids) == ["pt111_co_p2_r0a1", "pt111_co_top_r0a3"]


def test_earlier_candidates_stay_acceptable_after_a_revisit(tmp_path):
    """After a revisit the planner sees how its earlier plans fared, the reviewer sees its earlier
    review, and a structure of the replaced plan set that passed validation can still be accepted."""
    planned, reviewed = [], []

    def planner(messages, info):
        planned.append(_user_text(messages))
        return _final(info, PT_PLAN)

    def reviewer(messages, info):
        text = _user_text(messages)
        if text not in reviewed:
            reviewed.append(text)
        if len(reviewed) == 1:
            return _final(info, {"action": "revisit_evidence", "reasoning": "check the bridge sites",
                                 "feedback": "try bridge sites", "verdicts": _verdicts(messages)})
        earlier = re.findall(r"^- (\S+) \(plan P1 of plan set 0", text, re.M)
        if not reported(messages):
            return ModelResponse(parts=[ToolCallPart("write_report", REPORT)])
        return _final(info, {"action": "accept_candidates", "reasoning": "the first was right",
                             "accepted_structure_ids": earlier, "verdicts": _verdicts(messages)})

    deps = build_deps(make_config(), tmp_path / "run", seed=7, models={
        "planner": FunctionModel(planner), "constructor": constructor_model([GOOD_SCRIPT]),
        "reviser": coding_model(reviewer)})
    result = asyncio.run(run_workflow(deps, run_id="candidates"))

    assert result.status == "accepted" and result.accepted_structure_ids == ["pt111_co_top_r0a0"]
    assert result.current_structure_ids == ["pt111_co_top_r1a0"]
    assert "### Plan set 0" in planned[1] and "pt111_co_top_r0a0, validation PASS" in planned[1]
    assert "try bridge sites" in planned[1]
    assert "## Your earlier reviews in this run" in reviewed[1] and "Review 0: revisit_evidence" in reviewed[1]
    report = (tmp_path / "run" / "final_report.md").read_text()
    assert "| candidate | yes |" in report


def test_a_failed_end_drops_the_narrative_of_an_earlier_review(tmp_path):
    calls = []

    def planner(messages, info):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("model unavailable")
        return _final(info, PT_PLAN)

    def reviewer(messages, info):  # writes a report, then sends the run back to planning
        if not reported(messages):
            return ModelResponse(parts=[ToolCallPart("write_report", REPORT)])
        return _final(info, {"action": "revisit_evidence", "reasoning": "again", "feedback": "try bridge sites",
                             "verdicts": _verdicts(messages)})

    deps = build_deps(make_config(), tmp_path / "run", seed=7, models={
        "planner": FunctionModel(planner), "constructor": constructor_model([GOOD_SCRIPT]),
        "reviser": coding_model(reviewer)})
    result = asyncio.run(run_workflow(deps, run_id="failed"))
    report = (tmp_path / "run" / "final_report.md").read_text()
    assert result.status == "failed" and "planning failed" in result.errors[-1]
    assert "scripted summary" not in report and "No reviewer summary" in report
