"""The merged PDF -> summary -> characterization -> planning -> construction path, offline.

Agents 1-2 (summarizer, characterization) answer with the dry-run model's canned outputs; the
planner, constructor and reviser are scripted FunctionModels. The PDF is generated here; nothing in
this module comes from a real publication or a real LLM."""
from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pymupdf
import pytest
from pydantic_ai import BinaryContent
from pydantic_ai.messages import ModelResponse, RetryPromptPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from helpers import GOOD_SCRIPT, PT_PLAN, coding_model, constructor_model, reviser_model
from paper2structure.agents.characterization import CharacterizationReport, ExperimentalEvidence, Observation
from paper2structure.agents.planner import LLMPlanner, PlanningDeps, PlanSet, check_plan_set
from paper2structure.agents.summarizer import Evidence, MaterialSystem, PaperGraph, load_graph
from paper2structure.agents.summarizer.pdf_reader import read_pdf
from paper2structure.agents.summarizer.schemas import PaperMetadata
from paper2structure.agents.structured_agent import StructuredAgent
from paper2structure.config import (DEFAULT_MODEL, DRY_RUN_OUTPUTS, CharacterizationConfig, OverridesConfig,
                                    Paper2StructureConfig, canonical_technique, get_model)
from paper2structure.graph.nodes import characterize
from paper2structure.graph.workflow import build_deps, run_workflow

LINEAR = ["SummarizePaper", "AnalyzeEvidence", "PlanStructures", "ConstructStructures", "ValidateStructures",
          "ScientificRevision", "FinalizeResults"]

# the generated paper
TITLE = "Single-phase SampleA ceramics"
XRD_SENTENCE = "X-ray diffraction (Fig. 1) shows a single orthorhombic phase (space group Pnma)."
CAPTION = "Fig. 1 XRD pattern of SampleA."
AXIS_LABEL = "Intensity"


@pytest.fixture
def paper_pdf(tmp_path) -> Path:
    """A one-page synthetic paper: a title, a sentence about X-ray diffraction citing Fig. 1 and a
    vector 'XRD pattern' (axes frame, a trace with three peaks, an axis label inside it) with its
    caption below."""
    path = tmp_path / "synthetic_paper.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 80), TITLE, fontsize=16)
    page.insert_text((72, 130), XRD_SENTENCE, fontsize=10)
    page.draw_rect(pymupdf.Rect(150, 200, 450, 400), color=(0, 0, 0), width=1)
    trace = [(150 + 3 * i, 390 - (150 if i in (30, 55, 80) else 5 * (i % 3))) for i in range(101)]
    page.draw_polyline(trace, color=(0, 0, 1), width=1)
    page.insert_text((160, 220), AXIS_LABEL, fontsize=8)
    page.insert_text((150, 430), CAPTION, fontsize=10)
    doc.save(path)
    doc.close()
    return path


def _config(**sections) -> Paper2StructureConfig:
    raw = {"construction": {"code_timeout": 240.0, "memory_limit_gb": None}}
    for key, value in sections.items():
        raw.setdefault(key, {}).update(value)
    return Paper2StructureConfig.model_validate(raw)


def _plan(evidence_ids: list[str], system_ids: list[str]) -> dict:
    """PT_PLAN (which GOOD_SCRIPT builds) citing these IDs: the plan all of them, its lattice the
    first, its targets the rest."""
    plans = copy.deepcopy(PT_PLAN)
    for p in plans["plans"]:
        p["evidence_ids"], p["system_ids"] = list(evidence_ids), list(system_ids)
        for crystal in p.get("crystals", []):
            crystal["lattice"]["evidence_ids"] = list(evidence_ids[:1])
        for target in p.get("targets", []):
            target["evidence_ids"] = list(evidence_ids[1:])
    return plans


def _final(info: AgentInfo, payload: dict) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)])


def _user_items(messages) -> list:
    """The user-prompt items (text and images) of an agent run's first request."""
    items = []
    for p in messages[0].parts:
        if isinstance(p, UserPromptPart):
            items += [p.content] if isinstance(p.content, str) else list(p.content)
    return items


def _text(items: list) -> str:
    return "\n".join(x for x in items if isinstance(x, str))


def _recording_dry_run(calls: list) -> FunctionModel:
    """The dry-run model, recording the user-prompt items of every agent run."""
    dry = get_model("dry-run")

    def fn(messages, info):
        if not any(isinstance(p, RetryPromptPart) for m in messages for p in m.parts):
            calls.append(_user_items(messages))
        return dry.function(messages, info)
    return FunctionModel(fn, model_name="dry-run")


def _final_state(run: Path) -> dict:
    return json.loads(sorted((run / "state").glob("*_FinalizeResults.json"))[-1].read_text())


def _paper(evidence: list[Evidence]) -> PaperGraph:
    """A hand-built summary of a synthetic two-system paper."""
    return PaperGraph(source_pdf="synthetic.pdf", model="test",
                      metadata=PaperMetadata(title="Synthetic paper", summary="Not a real publication."),
                      systems=[MaterialSystem(system_id="S1", name="SampleA"),
                               MaterialSystem(system_id="S2", name="SampleB")],
                      evidence=evidence)


def _xrd_text(evidence_id: str = "E001") -> Evidence:
    return Evidence(evidence_id=evidence_id, system_ids=["S1", "S2"], technique="powder XRD",
                    technique_category="other", source_type="text", source_ref="page 1",
                    summary="Single orthorhombic phase for both samples.", quote="single orthorhombic phase")


def _mixed_paper(tmp_path: Path) -> tuple[PaperGraph, Path, Path]:
    """XRD text and figure evidence (implemented spec), NMR (a stub spec) and a technique outside
    the vocabulary; returns the paper, its folder and the figure image (relative path in the paper)."""
    paper_dir = tmp_path / "paper"
    image = paper_dir / "figures" / "p1_f1.png"
    image.parent.mkdir(parents=True)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 16, 16), False)
    pix.clear_with(255)
    pix.save(str(image))
    paper = _paper([
        _xrd_text("E001"),
        Evidence(evidence_id="E002", system_ids=["S1"], technique="XRD", technique_category="diffraction",
                 source_type="figure", source_ref="Fig. 1 (page 1)", summary="XRD pattern: one trace.",
                 quote="Fig. 1 XRD pattern.", image_path="figures/p1_f1.png"),
        Evidence(evidence_id="E003", system_ids=["S1"], technique="solid-state NMR", technique_category="other",
                 source_type="text", source_ref="page 2", summary="A single resonance."),
        Evidence(evidence_id="E004", system_ids=["S2"], technique="positron annihilation",
                 technique_category="other", source_type="text", source_ref="page 2",
                 summary="Vacancy-type defects."),
    ])
    return paper, paper_dir, image


def _xrd_report() -> CharacterizationReport:
    return CharacterizationReport(technique="XRD", observations=[Observation(
        observation_id="O1", source_id="E001", system_ids=["S1", "S2"], quantity="space group",
        value="Pnma", read_from="stated_in_text")])


def test_pdf_to_structures_in_one_graph_run(tmp_path, paper_pdf):
    """Agent 1 reads the generated PDF (dry run: systems S1/S2, the text evidence merged into E001,
    the cut-out figure E002), Agent 2 runs the XRD agent (dry run: observation O1), the planner
    cites 'E001' and 'XRD:O1' for S1, and the scripted constructor and reviser finish the run."""
    prompts = []

    def planner(messages, info):
        prompts.append(_text(_user_items(messages)))
        return _final(info, _plan(["E001", "XRD:O1"], ["S1"]))

    run = tmp_path / "run"
    deps = build_deps(_config(inputs={"pdf": str(paper_pdf)}), run, seed=7,
                      models={"paper": get_model("dry-run"), "planner": FunctionModel(planner),
                              "constructor": constructor_model([GOOD_SCRIPT]),
                              "reviser": reviser_model(["accept_candidates"])})
    result = asyncio.run(run_workflow(deps, run_id="pdf"))
    state = _final_state(run)

    assert result.status == "accepted" and result.errors == [], result.errors
    assert state["node_history"] == LINEAR  # SummarizePaper -> AnalyzeEvidence -> PlanStructures -> ...

    # Agent 1: the PaperGraph of the PDF, with the figure cut out next to it
    paper = load_graph(run / "paper" / "paper_graph.json")
    assert paper.source_pdf == str(paper_pdf) and [s.system_id for s in paper.systems] == ["S1", "S2"]
    text, figure = paper.evidence  # one text item per system, merged; then renumbered without gaps
    assert (text.evidence_id, text.source_type, text.technique, text.system_ids) == ("E001", "text", "XRD", ["S1", "S2"])
    assert (figure.evidence_id, figure.source_type, figure.quote) == ("E002", "figure", CAPTION)
    assert figure.context == [XRD_SENTENCE] and (run / "paper" / figure.image_path).is_file()
    assert Path(state["paper_dir"]) == (run / "paper").resolve()

    # Agent 2: one report per technique, saved, and snapshotted as evidence version 1
    xrd = CharacterizationReport.model_validate_json((run / "characterization" / "XRD.json").read_text())
    assert xrd.technique == "XRD" and [o.observation_id for o in xrd.observations] == ["O1"]
    assert state["evidence_version"] == 1 and set(state["evidence"]["reports"]) == {"XRD"}
    saved = json.loads((run / "inputs" / "experimental_evidence_v1.json").read_text())
    assert set(saved["reports"]) == {"XRD"}

    # Agent 3 got the summary and the reports; its plan names the system it models
    assert len(prompts) == 1 and "Dry-run paper" in prompts[0] and '"observation_id": "O1"' in prompts[0]
    assert state["plan_history"][0]["plans"][0]["system_ids"] == ["S1"]

    # Agent 4 got both in its workspace; the accepted structure passed validation and was exported
    task = run / "construction" / "r0_a0" / "P1" / "workspace" / "task"  # one workspace per plan
    assert load_graph(task / "paper_summary.json").metadata.title == "Dry-run paper"
    assert set(ExperimentalEvidence.model_validate_json((task / "evidence.json").read_text()).reports) == {"XRD"}
    sid = result.accepted_structure_ids[0]
    assert state["validation_reports"][sid]["passed"] and (run / result.structure_files[sid]["poscar"]).exists()

    report = (run / "final_report.md").read_text()
    assert "**Status:** accepted" in report and "Dry-run paper" in report
    assert "| XRD | phase | S1 | single phase | low |" in report


def test_characterize_runs_one_agent_per_implemented_technique(tmp_path):
    paper, paper_dir, image = _mixed_paper(tmp_path)
    calls, out = [], tmp_path / "characterization"
    ev = asyncio.run(characterize(paper, paper_dir, out, _recording_dry_run(calls), CharacterizationConfig()))

    assert set(ev.reports) == {"XRD"} and ev.errors == []
    assert ev.skipped == {"NMR": "spec is a stub", "positron annihilation": "not in the technique vocabulary"}
    assert ev.reports["XRD"].technique == "XRD" and ev.ids == {"XRD:O1"}

    # the XRD agent got only XRD evidence: the text item, and the figure record with its image
    [items] = calls
    prompt = _text(items)
    assert "E001" in prompt and "p1_f1" in prompt and "E003" not in prompt and "E004" not in prompt
    assert [x.data for x in items if isinstance(x, BinaryContent)] == [image.read_bytes()]

    # one JSON per report, which `inputs.characterization` reloads
    assert sorted(p.name for p in out.glob("*.json")) == ["XRD.json"]
    loaded = ExperimentalEvidence.load(out)
    assert loaded.reports == ev.reports and loaded.ids == ev.ids


def test_characterize_stubs_and_technique_filter(tmp_path):
    paper, paper_dir, _ = _mixed_paper(tmp_path)
    dry = get_model("dry-run")

    stubs = asyncio.run(characterize(paper, paper_dir, tmp_path / "stubs", dry,
                                     CharacterizationConfig(include_stubs=True)))
    assert set(stubs.reports) == {"XRD", "NMR"} and stubs.errors == []
    assert stubs.skipped == {"positron annihilation": "not in the technique vocabulary"}
    assert stubs.ids == {"XRD:O1", "NMR:O1"}  # the same observation ID, told apart by its technique
    assert set(ExperimentalEvidence.load(tmp_path / "stubs").reports) == {"XRD", "NMR"}

    only = asyncio.run(characterize(paper, paper_dir, tmp_path / "only", dry,
                                    CharacterizationConfig(techniques=["X-ray diffraction"], include_stubs=True)))
    assert set(only.reports) == {"XRD"}  # an alias names the canonical technique; NMR is not requested
    assert [p.name for p in (tmp_path / "only").glob("*.json")] == ["XRD.json"]


@pytest.mark.parametrize("inputs", [{}, {"pdf": "missing.pdf"}], ids=["no-paper", "missing-pdf"])
def test_summarization_failure_ends_the_run_before_planning(tmp_path, inputs):
    planned = []

    def planner(messages, info):
        planned.append(messages)
        return _final(info, _plan(["E001"], ["S1"]))

    inputs = {k: str(tmp_path / v) for k, v in inputs.items()}
    run = tmp_path / "run"
    deps = build_deps(_config(inputs=inputs), run, seed=0,
                      models={"paper": get_model("dry-run"), "planner": FunctionModel(planner),
                              "constructor": constructor_model([GOOD_SCRIPT]),
                              "reviser": reviser_model(["accept_candidates"])})
    result = asyncio.run(run_workflow(deps, run_id="no-paper"))

    assert result.status == "failed" and not planned
    assert len(result.errors) == 1 and result.errors[0].startswith("summarization failed")
    if not inputs:
        assert "the config needs inputs.pdf or inputs.paper_graph" in result.errors[0]
    assert _final_state(run)["node_history"] == ["SummarizePaper", "FinalizeResults"]
    report = (run / "final_report.md").read_text()
    assert "**Status:** failed" in report and result.errors[0] in report


def test_saved_summary_and_reports_replace_agents_1_and_2(tmp_path):
    """inputs.paper_graph and inputs.characterization (an earlier run's outputs) skip Agents 1-2;
    the planner gets the loaded summary and reports."""
    earlier = tmp_path / "earlier"
    (earlier / "paper").mkdir(parents=True)
    (earlier / "characterization").mkdir()
    (earlier / "paper" / "paper_graph.json").write_text(_paper([_xrd_text()]).model_dump_json())
    (earlier / "characterization" / "XRD.json").write_text(_xrd_report().model_dump_json())

    def unused(messages, info):
        raise AssertionError("Agents 1-2 must not run")

    prompts = []

    def planner(messages, info):
        prompts.append(_text(_user_items(messages)))
        return _final(info, _plan(["E001", "XRD:O1"], ["S1"]))

    def constructor(messages, info):  # reports the plan as unbuildable without running anything
        return _final(info, {"structures": [], "failed": [{"plan_id": "P1", "reason": "scripted failure"}]})

    run = tmp_path / "run"
    cfg = _config(inputs={"paper_graph": str(earlier / "paper" / "paper_graph.json"),
                          "characterization": str(earlier / "characterization")})
    deps = build_deps(cfg, run, seed=0, max_revisions=0,
                      models={"paper": FunctionModel(unused), "planner": FunctionModel(planner),
                              "constructor": coding_model(constructor), "reviser": reviser_model(["stop_inconclusive"])})
    result = asyncio.run(run_workflow(deps, run_id="reuse"))
    state = _final_state(run)

    assert result.status == "inconclusive" and result.errors == [], result.errors
    # the unbuilt plan goes back to construction with its reason, twice (max_repairs), before the review
    repairs = ["ConstructStructures", "ValidateStructures"] * 2
    assert state["node_history"] == LINEAR[:5] + repairs + LINEAR[5:] and state["evidence_version"] == 1
    assert "P1: scripted failure" in state["repair_feedback"] and state["failed_plans"] == {"P1": "scripted failure"}
    assert Path(state["paper_dir"]) == (earlier / "paper").resolve()
    assert not (run / "paper").exists() and not (run / "characterization").exists()
    assert state["evidence"]["reports"]["XRD"] == _xrd_report().model_dump(mode="json")
    assert len(prompts) == 1 and "Synthetic paper" in prompts[0] and '"value": "Pnma"' in prompts[0]


def test_planner_cites_summary_and_technique_qualified_ids():
    paper, evidence = _paper([_xrd_text()]), ExperimentalEvidence(reports={"XRD": _xrd_report()})
    good, bad = _plan(["E001", "XRD:O1"], ["S1"]), _plan(["E001", "XRD:O9", "O1"], ["S1", "S9"])

    # PlanningDeps as LLMPlanner.plan builds them from the summary and the characterization reports
    deps = PlanningDeps(evidence_ids={e.evidence_id for e in paper.evidence} | evidence.ids, max_plans=3,
                        overrides=OverridesConfig(), system_ids={s.system_id for s in paper.systems})
    assert check_plan_set(PlanSet.model_validate(good), deps) == []
    problems = check_plan_set(PlanSet.model_validate(bad), deps)
    assert any("unknown evidence IDs ['O1', 'XRD:O9']" in p for p in problems), problems  # O1 needs its technique
    assert any("unknown system IDs ['S9']" in p for p in problems), problems

    # ... and the planner's output validator sends the same problems back to the model
    retries = []

    def fn(messages, info):
        retried = [str(p.content) for m in messages for p in m.parts if isinstance(p, RetryPromptPart)]
        retries.append(retried)
        return _final(info, good if retried else bad)

    planner = LLMPlanner(FunctionModel(fn), None, max_plans=3, max_atoms=5000, min_vacuum=15.0,
                         overrides=OverridesConfig())
    plans = asyncio.run(planner.plan(paper, evidence, [], ""))
    assert plans.plans[0].evidence_ids == ["E001", "XRD:O1"] and plans.plans[0].system_ids == ["S1"]
    assert len(retries) == 2 and "XRD:O9" in retries[1][0] and "S9" in retries[1][0]


def test_get_model_names(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-key")
    monkeypatch.delenv("P2S_MODEL", raising=False)
    assert get_model("gpt-5.6-luna") == "openai:gpt-5.6-luna"  # a bare name is an OpenAI model
    assert get_model("openai:gpt-5-nano") == "openai:gpt-5-nano"
    assert get_model() == DEFAULT_MODEL
    dry = get_model("dry-run")
    assert isinstance(dry, FunctionModel) and dry.model_name == "dry-run"
    # canned answers for Agents 1-2, registered when their packages are imported
    assert {"SystemsOutput", "TextEvidenceOutput", "FigureOutput", "CharacterizationReport"} <= set(DRY_RUN_OUTPUTS)

    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_model("gpt-5.6-luna")
    monkeypatch.setenv("P2S_MODEL", "dry-run")
    assert isinstance(get_model(), FunctionModel)  # the dry run needs no key


def test_read_pdf_extracts_text_and_vector_figure(tmp_path, paper_pdf):
    doc = read_pdf(paper_pdf, tmp_path / "figures")
    [page] = doc.pages
    assert TITLE in page and XRD_SENTENCE in page and CAPTION in page
    assert AXIS_LABEL not in page  # text inside the plot is not body text
    assert doc.full_text.startswith("[Page 1]")

    [fig] = doc.figures  # the vector drawing, cut out with the caption below it
    assert (fig.figure_id, fig.page, fig.caption, fig.page_captions) == ("p1_f1", 1, CAPTION, [CAPTION])
    assert fig.context == [XRD_SENTENCE]  # the body sentence citing Fig. 1
    assert fig.image_path == tmp_path / "figures" / "p1_f1.png"
    pix = pymupdf.Pixmap(str(fig.image_path))
    assert pix.width > pix.height > 100  # the landscape plot region, not the portrait page

    assert read_pdf(paper_pdf, tmp_path / "none", max_figures=0).figures == []


def _earlier_run(tmp_path: Path, errors: str = "") -> Path:
    """The paper/ and characterization/ folders of an earlier run (optionally with summarizer errors)."""
    earlier = tmp_path / "earlier"
    (earlier / "paper").mkdir(parents=True)
    (earlier / "characterization").mkdir()
    (earlier / "paper" / "paper_graph.json").write_text(_paper([_xrd_text()]).model_dump_json())
    (earlier / "characterization" / "XRD.json").write_text(_xrd_report().model_dump_json())
    if errors:
        (earlier / "paper" / "errors.txt").write_text(errors)
    return earlier


def test_crashing_construction_run_is_recorded_and_retried(tmp_path):
    """A construction run that raises (e.g. out of requests) does not end the run: the plan is
    recorded as not built, retried with the reason up to max_repairs, then reviewed. Summarizer
    errors of the reused summary reach the run's errors."""
    earlier = _earlier_run(tmp_path, errors="Figure analysis failed for p2_f1: scripted\n")
    runs = []

    def planner(messages, info):
        return _final(info, _plan(["E001", "XRD:O1"], ["S1"]))

    def constructor(messages, info):
        runs.append(_text(_user_items(messages)))
        raise RuntimeError("scripted construction crash")

    run = tmp_path / "run"
    cfg = _config(inputs={"paper_graph": str(earlier / "paper" / "paper_graph.json"),
                          "characterization": str(earlier / "characterization")})
    deps = build_deps(cfg, run, seed=0, max_revisions=0, max_repairs=1,
                      models={"planner": FunctionModel(planner), "constructor": coding_model(constructor),
                              "reviser": reviser_model(["stop_inconclusive"])})
    result = asyncio.run(run_workflow(deps, run_id="crash"))
    state = _final_state(run)

    assert result.status == "inconclusive" and len(runs) == 2  # first attempt + one repair
    assert "scripted construction crash" in runs[1]  # the repair attempt is told why
    assert state["node_history"] == LINEAR[:5] + ["ConstructStructures", "ValidateStructures"] + LINEAR[5:]
    assert "RuntimeError: scripted construction crash" in state["failed_plans"]["P1"]
    assert any(e.startswith("summarizer: Figure analysis failed") for e in result.errors)
    assert sum("scripted construction crash" in e for e in result.errors) == 2
    assert (run / "construction" / "r0_a0" / "P1").is_dir() and (run / "construction" / "r0_a1" / "P1").is_dir()


def test_missing_reports_folder_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    for folder in (tmp_path / "empty", tmp_path / "missing"):
        with pytest.raises(FileNotFoundError, match="no characterization reports"):
            ExperimentalEvidence.load(folder)


def test_technique_aliases_seen_in_papers():
    names = {"IRRAS": "FTIR", "DRIFTS": "FTIR", "13C NMR": "NMR", "31P-NMR": "NMR", "1H MAS NMR": "NMR",
             "57Fe Mossbauer": "Mossbauer", "Reitveld refinement": "XRD",
             "selected area electron diffraction": "TEM", "positron annihilation": None}
    assert {n: canonical_technique(n) for n in names} == names


def test_failed_agent_runs_still_count_their_usage():
    class Probe(StructuredAgent):
        _output_type = int

    def fn(messages, info):  # never returns a valid int
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"response": "not an int"})])

    agent = Probe(FunctionModel(fn), None, "probe", max_retry=1)
    with pytest.raises(Exception):
        asyncio.run(agent.arun("go"))
    assert agent.usage["calls"] == 2  # the first try and the one retry, recorded although the run failed
