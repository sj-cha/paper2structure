"""Agents with scripted FunctionModels (no real LLM)."""
import asyncio
import copy
import json

from pydantic_ai.messages import ModelResponse, RetryPromptPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

import pytest

from helpers import FIXTURES, GOOD_SCRIPT, PT_PLAN, REPORT, coding_model, constructor_model, make_config, reported
from paper2structure.agents.characterization.schemas import ExperimentalEvidence
from paper2structure.agents.constructor import LLMConstructor
from paper2structure.agents.constructor.skills import api_reference
from paper2structure.agents.constructor.workspace import sandbox_available
from paper2structure.agents.planner import LLMPlanner, PlanningDeps, check_plan_set
from paper2structure.agents.planner.schemas import PlanSet
from paper2structure.agents.reviser import LLMReviser
from paper2structure.agents.reviser.agent import reviser_prompt
from paper2structure.agents.summarizer import load_graph
from paper2structure.config import OverridesConfig
from paper2structure.graph.state import ArtifactStore, ReconstructionState

SCENARIOS = ("pt111_co", "cspbbr3_phosphonate", "anatase_formate", "cdse_zns_coreshell")


def _inputs(scenario="pt111_co"):
    """A fixture's summary (Agent 1's PaperGraph) and characterization reports (Agent 2)."""
    s = load_graph(FIXTURES / scenario / "paper_graph.json")
    e = ExperimentalEvidence.load(FIXTURES / scenario / "characterization")
    return s, e


def _retried(messages) -> bool:
    return any(isinstance(p, RetryPromptPart) for m in messages for p in m.parts)


def _final(info: AgentInfo, payload: dict) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)])


def test_fixtures_are_marked_synthetic():
    for scenario in SCENARIOS:
        s, e = _inputs(scenario)
        assert s.source_pdf == "synthetic" and s.metadata.title.startswith("[SYNTHETIC FIXTURE]"), scenario
        assert s.metadata.summary.startswith("Synthetic fixture"), scenario
        assert e.reports and all(any(c.startswith("Synthetic fixture") for c in r.caveats)
                                 for r in e.reports.values()), scenario


def test_fixture_reports_cite_the_paper_graph():
    """One report per technique in the summary; every observation comes from an evidence item of
    its own technique, and findings and constraints cite observations of their own report."""
    for scenario in SCENARIOS:
        s, e = _inputs(scenario)
        systems = {x.system_id for x in s.systems}
        technique = {x.evidence_id: x.technique for x in s.evidence}
        assert set(technique.values()) == set(e.reports), scenario
        assert all(x.system_ids and set(x.system_ids) <= systems for x in s.evidence), scenario
        for name, r in e.reports.items():
            observations = {o.observation_id for o in r.observations}
            assert r.technique == name and observations, (scenario, name)
            assert all(technique.get(o.source_id) == name and set(o.system_ids) <= systems
                       for o in r.observations), (scenario, name)
            assert all(set(c.evidence) <= observations and set(c.system_ids) <= systems
                       for c in [*r.findings, *r.structural_constraints]), (scenario, name)


def test_planner_rejects_invented_evidence_and_retries():
    bad = copy.deepcopy(PT_PLAN)
    bad["plans"][0]["evidence_ids"] = ["E99"]
    seen = []

    def fn(messages, info):
        seen.append([str(p.content) for m in messages for p in m.parts if isinstance(p, RetryPromptPart)])
        return _final(info, PT_PLAN if _retried(messages) else bad)

    planner = LLMPlanner(FunctionModel(fn), None, max_plans=3, max_atoms=5000, min_vacuum=15.0,
                         overrides=OverridesConfig())
    plans = asyncio.run(planner.plan(*_inputs(), [], ""))
    assert plans.plans[0].evidence_ids == ["E2", "E3"]
    assert len(seen) == 2 and "E99" in seen[1][0]
    assert "'E5'" in seen[1][0] and "'LEED:O2'" in seen[1][0]  # summary and technique-qualified report IDs


def test_check_plan_set_enforces_overrides_and_chemistry():
    deps = PlanningDeps(evidence_ids={"E1", "E2", "E3"}, max_plans=3,
                        overrides=OverridesConfig(miller=(1, 0, 0), lattice={"a": 3.92}))
    problems = check_plan_set(PlanSet.model_validate(PT_PLAN), deps)
    assert any("Miller" in p for p in problems) and any("lattice a" in p for p in problems)
    bad = copy.deepcopy(PT_PLAN)
    bad["plans"][0]["adsorbates"][0]["smiles"] = "C(("
    bad["plans"][0]["crystals"][0]["lattice"] = {"a": 3.924, "source": "paper"}
    problems = check_plan_set(PlanSet.model_validate(bad), PlanningDeps({"E1", "E2", "E3"}, 3, OverridesConfig()))
    assert any("unparseable SMILES" in p for p in problems)
    assert any("cites no evidence" in p for p in problems)


def test_check_plan_set_builds_every_crystal():
    # the live planner twice mixed anatase origin choices: Ti from choice 2, O from choice 1
    mixed = copy.deepcopy(PT_PLAN)
    mixed["plans"][0]["crystals"] = [{"formula": "TiO2", "space_group": "I4_1/amd", "formula_units": 4,
                                      "sites": [{"species": "Ti", "coords": [0, 0.75, 0.125]},
                                                {"species": "O", "coords": [0, 0, 0.208]}],
                                      "lattice": {"a": 3.784, "c": 9.514, "source": "literature"}}]
    deps = PlanningDeps({"E1", "E2", "E3"}, 3, OverridesConfig())
    problems = check_plan_set(PlanSet.model_validate(mixed), deps)
    assert any("cannot be built" in p and "origin choice 1" in p for p in problems)
    mixed["plans"][0]["crystals"][0]["sites"][0]["coords"] = [0, 0, 0]
    assert check_plan_set(PlanSet.model_validate(mixed), deps) == []
    swapped = copy.deepcopy(PT_PLAN)  # the live planner's CsPbBr3 with Br around Cs instead of Pb
    swapped["plans"][0]["crystals"] = [{"formula": "CsPbBr3", "space_group": "Pm-3m", "formula_units": 1,
                                        "sites": [{"species": "Cs", "coords": [0, 0, 0]},
                                                  {"species": "Pb", "coords": [0.5, 0.5, 0.5]},
                                                  {"species": "Br", "coords": [0.5, 0, 0]}],
                                        "lattice": {"a": 5.87, "source": "paper", "evidence_ids": ["E1"]}}]
    assert any("no bonded neighbour for Pb" in p for p in check_plan_set(PlanSet.model_validate(swapped), deps))


def test_check_plan_set_counts_adsorbate_atoms_against_the_budget():
    # the live planner asked for 120 full oleates (53 atoms each) on a 5000-atom budget
    plan = copy.deepcopy(PT_PLAN)
    plan["plans"][0]["adsorbates"][0].update({"name": "oleate", "smiles": "[O-]C(=O)CCCCCCC/C=C\\CCCCCCCC",
                                              "binding_atoms": ["O", "O"], "anchor_smarts": None, "count": 120})
    problems = check_plan_set(PlanSet.model_validate(plan), PlanningDeps({"E1", "E2", "E3"}, 3, OverridesConfig()))
    assert any("6360 atoms" in p and "truncated adsorbate" in p for p in problems)
    plan["plans"][0]["adsorbates"][0]["count"] = 40
    assert check_plan_set(PlanSet.model_validate(plan), PlanningDeps({"E1", "E2", "E3"}, 3, OverridesConfig())) == []


def test_open_plan_schema_takes_a_core_shell_particle():
    plan = copy.deepcopy(PT_PLAN)
    zb = {"space_group": "F-43m", "formula_units": 4, "lattice": {"a": 6.05, "source": "literature"}}
    plan["plans"][0].update({
        "model_type": "core-shell nanocrystal", "surface": None, "adsorbates": [],
        "description": "zinc-blende CdSe core (3 nm) with a coherent 2-monolayer ZnS shell",
        "crystals": [dict(zb, name="core", formula="CdSe", sites=[{"species": "Cd", "coords": [0, 0, 0]},
                                                                   {"species": "Se", "coords": [0.25, 0.25, 0.25]}]),
                     dict(zb, name="shell", formula="ZnS", lattice={"a": 5.41, "source": "literature"},
                          sites=[{"species": "Zn", "coords": [0, 0, 0]}, {"species": "S", "coords": [0.25, 0.25, 0.25]}])],
        "nanocrystal": {"shape": "sphere", "size": 42.0},
        "targets": [{"property": "shell thickness", "value": "2 monolayers", "evidence_ids": ["E1"]}]})
    deps = PlanningDeps({"E1", "E2", "E3"}, 3, OverridesConfig())
    assert check_plan_set(PlanSet.model_validate(plan), deps) == []
    plan["plans"][0]["targets"][0]["evidence_ids"] = ["E9"]
    assert any("E9" in p for p in check_plan_set(PlanSet.model_validate(plan), deps))


needs_sandbox = pytest.mark.skipif(not sandbox_available()[0], reason="bubblewrap unavailable")


@needs_sandbox
def test_constructor_writes_runs_and_verifies_a_standalone_script(tmp_path):
    seen = {}
    inner = constructor_model([GOOD_SCRIPT]).function

    def spy(messages, info):
        seen["tools"] = {t.name for t in info.function_tools}
        seen["instructions"] = (info.instructions or "") + " ".join(
            str(p.content) for m in messages for p in m.parts if type(p).__name__ == "SystemPromptPart")
        return inner(messages, info)

    agent = LLMConstructor(coding_model(spy), None, make_config())
    report, deps = asyncio.run(agent.construct(PlanSet.model_validate(PT_PLAN), ["P1"], tmp_path, seed=7))
    assert [s.name for s in report.structures] == ["pt111_co_top"]
    assert deps.scripts["pt111_co_top"] == (tmp_path / "workspace" / "build_P1.py").resolve()
    assert deps.verified["pt111_co_top"] == tmp_path / "replay" / "build_P1" / "structures" / "pt111_co_top.extxyz"
    ws = tmp_path / "workspace"
    assert (ws / "task" / "plans.json").exists() and (ws / ".agents" / "skills" / "structure-toolkit" / "SKILL.md").exists()
    # harness Coder + Skills + Planning tools next to the workflow's own checks
    assert {"read_file", "write_file", "edit_file", "shell", "load_capability", "check_structure",
            "view_structure"} <= seen["tools"]
    assert any(t.startswith(("write_plan", "add_task", "read_plan")) for t in seen["tools"])
    assert "structure-toolkit" in seen["instructions"] and "custom-structures" in seen["instructions"]


def _scripted(steps, final_payloads):
    """Run `steps` (tool name, args) in order, then return final_payloads[k] on the k-th final turn."""
    state = {"finals": 0}

    def fn(messages, info):
        done = sum(1 for m in messages for p in m.parts if isinstance(p, ToolReturnPart))
        if done < len(steps):
            return ModelResponse(parts=[ToolCallPart(*steps[done])])
        payload = final_payloads[min(state["finals"], len(final_payloads) - 1)]
        state["finals"] += 1
        return _final(info, payload)
    return fn


FAILED = {"structures": [], "failed": [{"plan_id": "P1", "reason": "gave up"}]}


def _construct(tmp_path, fn):
    seen = {}

    def spy(messages, info):
        seen["messages"] = messages
        return fn(messages, info)

    agent = LLMConstructor(coding_model(spy), None, make_config())
    report, _ = asyncio.run(agent.construct(PlanSet.model_validate(PT_PLAN), ["P1"], tmp_path, seed=7))
    retries = " ".join(str(p.content) for m in seen["messages"] for p in m.parts if isinstance(p, RetryPromptPart))
    return report, retries


def _report(name="pt111_co_top", script="build_P1.py"):
    return {"structures": [{"name": name, "plan_id": "P1", "script": script, "method": "scripted"}], "failed": []}


@needs_sandbox
def test_constructor_cannot_report_structures_it_never_emitted(tmp_path):
    report, retries = _construct(tmp_path, _scripted([], [_report("ghost"), FAILED]))
    assert report.structures == [] and report.failed[0].plan_id == "P1" and "never emitted" in retries


@needs_sandbox
def test_constructor_rejects_structures_emitted_by_inline_code(tmp_path):
    inline = "python - <<'EOF'\n" + GOOD_SCRIPT + "EOF"
    report, retries = _construct(tmp_path, _scripted([("shell", {"command": inline})], [_report(), FAILED]))
    assert report.structures == [] and "inline code" in retries


@needs_sandbox
def test_constructor_rejects_scripts_that_do_not_reproduce(tmp_path):
    noisy = GOOD_SCRIPT.replace("print(describe(model))", "import numpy as np\n"  # unseeded noise
                                "model.positions += np.random.default_rng().normal(0, 0.05, model.positions.shape)")
    steps = [("write_file", {"path": "build_P1.py", "content": noisy}), ("shell", {"command": "python build_P1.py"})]
    report, retries = _construct(tmp_path, _scripted(steps, [_report(), FAILED]))
    assert report.structures == [] and "differs from structures/pt111_co_top.extxyz" in retries


DETACHED_SCRIPT = '''\
import numpy as np
from ase import Atoms
from paper2structure.tools import add_molecule, build_bulk, build_slab, emit
bulk = build_bulk("Fm-3m", ["Pt"], [[0, 0, 0]], a=3.924, formula_units=4)
slab = build_slab(bulk, (1, 1, 1), termination=0, min_thickness=6.0, vacuum=15.0, supercell=(2, 2))
top = slab.positions[np.argmax(slab.positions[:, 2])]
model = add_molecule(slab, Atoms("CO", [top + [0, 0, 6.0], top + [0, 0, 7.13]]), "[C-]#[O+]", "CO")
emit(model, name="pt111_co_floating", plan_id="P1")
'''


@needs_sandbox
def test_validation_gate_sends_reports_back_but_never_stalls_the_run(tmp_path):
    seen = {}
    inner = constructor_model([DETACHED_SCRIPT]).function

    def spy(messages, info):
        seen["retries"] = [str(p.content) for m in messages for p in m.parts if isinstance(p, RetryPromptPart)]
        return inner(messages, info)

    agent = LLMConstructor(coding_model(spy), None, make_config(construction={"gate_retries": 2}))
    report, deps = asyncio.run(agent.construct(PlanSet.model_validate(PT_PLAN), ["P1"], tmp_path, seed=0))
    gate_msgs = [r for r in seen["retries"] if "validation gate" in r]
    assert len(gate_msgs) == 2 and "adsorbate_binding" in gate_msgs[0] and "detached" in gate_msgs[0]
    # after gate_retries rejections the report passes on, with its failing gate report, to ValidateStructures
    assert [s.name for s in report.structures] == ["pt111_co_floating"]
    assert not deps.reports["pt111_co_floating"].passed
    assert "adsorbate_binding" in [c.name for c in deps.reports["pt111_co_floating"].failures()]
    assert len(deps.replays) == 1  # the unchanged script was replayed once across the three report checks


def test_replays_run_in_parallel_and_unchanged_scripts_are_not_rerun(tmp_path, monkeypatch):
    import paper2structure.agents.constructor.agent as constructor
    from pydantic_ai.workspaces import CommandResult

    calls, running, peak = [], [0], [0]

    async def fake_replay(script, out_dir, cc):
        calls.append(script.name)
        running[0] += 1
        peak[0] = max(peak[0], running[0])
        await asyncio.sleep(0.2)
        running[0] -= 1
        return CommandResult(exit_code=0, stdout="", stderr="")

    monkeypatch.setattr(constructor, "replay", fake_replay)
    deps = constructor.ConstructionDeps(workdir=tmp_path, plan_ids=["P1"], plans=PlanSet.model_validate(PT_PLAN),
                                        cfg=make_config())
    scripts = []
    for sub in ("a", "b", "c"):  # same file name in three folders
        script = deps.workspace / sub / "build.py"
        script.parent.mkdir(parents=True)
        script.write_text(f"print('{sub}')\n")
        scripts.append(script.resolve())
    asyncio.run(constructor.replay_scripts(deps, scripts))
    assert sorted(calls) == ["build.py"] * 3 and peak[0] == 3  # all three at once
    assert len({deps.replay_dir(s) for s in scripts}) == 3  # no collision between same-named scripts
    asyncio.run(constructor.replay_scripts(deps, scripts))
    assert len(calls) == 3  # unchanged: reused
    scripts[1].write_text("print('changed')\n")
    asyncio.run(constructor.replay_scripts(deps, scripts))
    assert len(calls) == 4  # only the edited script runs again
    assert deps.plan("P1") is not None and deps.plan("P9") is None


def test_api_reference_covers_public_tools():
    ref = api_reference()
    for name in ("build_bulk", "list_terminations", "build_slab", "build_cluster", "carve_nanocrystal",
                 "Adsorbate.from_smiles", "BindingMotif", "surface_sites", "place_adsorbates", "add_molecule", "add_solvent",
                 "composition_profile", "emit"):
        assert name in ref


def _review(tmp_path, fn, revisions_left, failed=None, rebuilds_left=0):
    s, e = _inputs()
    state = ReconstructionState(run_id="t", summary=s, evidence=e, plan_history=[PlanSet.model_validate(PT_PLAN)],
                                failed_plans=failed or {})
    decision = asyncio.run(LLMReviser(FunctionModel(fn), None).review(state, ArtifactStore(tmp_path), revisions_left,
                                                                      rebuilds_left=rebuilds_left))
    return decision, state


def test_reviser_rebuild_needs_budget_and_known_structures(tmp_path):
    seen = []

    def fn(messages, info):
        seen.append([str(p.content) for m in messages for p in m.parts if isinstance(p, RetryPromptPart)])
        if not _retried(messages):
            return _final(info, {"action": "rebuild_structures", "reasoning": "scripted",
                                 "rebuild_structure_ids": ["ghost"], "feedback": "Pb-I 2.2 A: rebuild at 3.15 A"})
        if not reported(messages):
            return ModelResponse(parts=[ToolCallPart("write_report", REPORT)])
        return _final(info, {"action": "stop_inconclusive", "reasoning": "scripted"})

    decision, state = _review(tmp_path, fn, revisions_left=0, rebuilds_left=0)
    assert decision.action == "stop_inconclusive"
    assert "no rebuild is left" in seen[1][0] and "unknown structure IDs ['ghost']" in seen[1][0]
    assert "Rebuilds left: 0 (rebuild_structures is not available)" in reviser_prompt(state, 0, 0)


def test_reviser_cannot_revisit_without_budget(tmp_path):
    def fn(messages, info):
        if not _retried(messages):
            return _final(info, {"action": "revisit_evidence", "reasoning": "scripted", "feedback": "re-read E3"})
        if not reported(messages):
            return ModelResponse(parts=[ToolCallPart("write_report", REPORT)])
        return _final(info, {"action": "stop_inconclusive", "reasoning": "scripted"})

    decision, _ = _review(tmp_path, fn, revisions_left=0)
    assert decision.action == "stop_inconclusive"
    assert json.loads(decision.model_dump_json())["action"] == "stop_inconclusive"


def test_reviser_must_replan_when_nothing_was_built(tmp_path):
    # the live reviewer once stopped with a revision left after every plan failed to build
    def fn(messages, info):
        action = "revisit_evidence" if _retried(messages) else "stop_inconclusive"
        return _final(info, {"action": action, "reasoning": "scripted", "feedback": "fix the bulk sites"})

    decision, _ = _review(tmp_path, fn, revisions_left=1, failed={"P1": "bulk generates TiO, not TiO2"})
    assert decision.action == "revisit_evidence"


def test_reviser_writes_the_report_before_ending_the_run(tmp_path):
    seen = []

    def fn(messages, info):
        seen.append([str(p.content) for m in messages for p in m.parts if isinstance(p, RetryPromptPart)])
        if _retried(messages) and not reported(messages):
            return ModelResponse(parts=[ToolCallPart("write_report", REPORT)])
        return _final(info, {"action": "stop_inconclusive", "reasoning": "scripted"})

    decision, state = _review(tmp_path, fn, revisions_left=0)
    assert decision.action == "stop_inconclusive" and "call write_report first" in seen[1][0]
    assert state.report.summary == "scripted summary"
    report = (tmp_path / "final_report.md").read_text()
    assert "**Status:** under review" in report and "## 1. Summary\nscripted summary" in report
    # the fixture's PaperGraph (systems) and characterization reports (constraints per technique)
    assert "## 2. Paper and material systems\n- [SYNTHETIC FIXTURE]" in report and "| S1 | Pt(111) |" in report
    assert "## 3. Characterization" in report and "| LEED | CO coverage (required) | S1 |" in report
    assert "## 7. Assessment\nscripted assessment" in report and "- relax with an MLIP" in report
