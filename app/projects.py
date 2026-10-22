"""Project store and the three stages of the web tool (app/server.py).

    1. summarize   Agent 1 on the uploaded PDF                            -> <project>/paper/
    2. analyze     Agent 2, cached per model and technique set, then     -> <project>/characterization/<key>/
                   Agent 3 on the material systems the user chose         -> <project>/plans/<key>.json
    3. generate    Agents 3-5 as one graph run (main.py's workflow):      -> <project>/runs/<time>/
                   construction, validation and review; the plan of
                   stage 2 is handed to the graph while the settings match

A project is a folder under outputs/projects/ holding every stage's files, so closing the page or
restarting the server loses nothing, and "generate again" with new settings only redoes the stages
those settings touch (the summary never, the evidence only for a new model or technique set).
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import queue
import re
import shutil
import threading
import time
import traceback
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ase.io import read, write

from paper2structure.agents.characterization import ExperimentalEvidence, find_spec
from paper2structure.agents.constructor.workspace import sandbox_available
from paper2structure.agents.planner import LLMPlanner
from paper2structure.agents.planner.schemas import PlanSet
from paper2structure.agents.summarizer.graph import summarize_paper
from paper2structure.agents.summarizer.knowledge_graph import build_graph, load_graph
from paper2structure.agents.summarizer.schemas import PaperGraph
from paper2structure.config import (DEFAULT_MODEL, MODEL_CHOICES, CharacterizationConfig, OverridesConfig,
                                    Paper2StructureConfig, canonical_technique, get_model, load_config)
from paper2structure.graph.nodes import characterize
from paper2structure.graph.workflow import build_deps, run_workflow
from structure_info import compute_estimate, structure_info

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("P2S_WEB_ROOT", REPO / "outputs" / "projects"))
BASE_CONFIG = REPO / "config" / "config.yml"
EFFORTS = ("low", "medium", "high")
MODEL_TYPES = ["bulk", "slab", "nanocrystal", "core-shell nanocrystal", "supported nanoparticle", "slab with adsorbates"]
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_PRICE = re.compile(r"\$([\d.]+)/\$([\d.]+)")  # "GPT-5.6 Luna · $0.20/$1.20": USD per 1M input / output tokens
log = logging.getLogger("webtool")  # outside the paper2structure tree: its tracebacks stay out of the job logs


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _clamp(value, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def _hash(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:10]


def _read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


# --------------------------------------------------------------------------------------
# Models and prices
# --------------------------------------------------------------------------------------


def default_model() -> str:
    return os.getenv("P2S_MODEL", DEFAULT_MODEL)


def model_choices() -> list[dict]:
    """The models the page offers, with the prices from paper2structure.config."""
    out = []
    for name, label in MODEL_CHOICES.items():
        title, _, price = label.partition(" · ")
        out.append({"name": name, "title": title, "price": price, "default": name == default_model(),
                    "note": "free; test only: it covers the summary and the evidence, not the structures"
                    if name == "dry-run" else ""})
    return out


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """USD from the price list, or None when the model's price is unknown (cached input billed as input)."""
    m = _PRICE.search(MODEL_CHOICES.get(model, ""))
    if not m:
        return None
    return round((input_tokens * float(m[1]) + output_tokens * float(m[2])) / 1e6, 4)


# --------------------------------------------------------------------------------------
# Settings (panels 2 and 3)
# --------------------------------------------------------------------------------------


def default_max_atoms(method: str, realism: float) -> int:
    """Atom budget from the realism slider: DFT 60-600 atoms, MD 500-10000 (app.js mirrors this)."""
    lo, hi = (60, 600) if method == "DFT" else (500, 10000)
    return int(round((lo + (hi - lo) * realism) / 10) * 10)


@dataclass
class Settings:
    """Everything the user chooses on panels 2 and 3, stored with every plan and run."""
    systems: list[str] = field(default_factory=list)  # the material systems to model (summarizer IDs)
    techniques: list[str] = field(default_factory=list)  # characterize only these; empty = every technique with an agent
    max_plans: int = 1  # planning.max_plans: competing hypotheses, one structure each
    method: str = "DFT"  # what the structures are for: DFT or MD
    realism: float = 0.3  # 0 = computationally cheapest model, 1 = closest to the experiment
    max_atoms: int = 200  # planning.max_atoms
    model_type: Optional[str] = None  # overrides.model_type; None lets the planner decide
    seed: int = 0
    max_revisions: int = 0  # loops the reviewer may take back through planning and construction
    max_repairs: int = 1  # construction retries after a hard validation failure

    @classmethod
    def parse(cls, raw: dict) -> "Settings":
        """From the page's JSON; a missing or null field keeps the default above (0 is a valid value)."""
        s = cls()

        def get(key, default):
            return default if raw.get(key) is None else raw[key]

        s.systems = [str(x) for x in get("systems", [])]
        s.techniques = sorted({canonical_technique(t) or str(t) for t in get("techniques", [])})
        s.max_plans = _clamp(get("max_plans", s.max_plans), 1, 3)
        s.method = "MD" if str(get("method", s.method)).upper() == "MD" else "DFT"
        s.realism = round(min(1.0, max(0.0, float(get("realism", s.realism)))), 2)
        s.max_atoms = _clamp(get("max_atoms", default_max_atoms(s.method, s.realism)), 10, 50000)
        model_type = raw.get("model_type") or None
        s.model_type = model_type if model_type in MODEL_TYPES else None
        s.seed = int(get("seed", s.seed))
        s.max_revisions = _clamp(get("max_revisions", s.max_revisions), 0, 2)
        s.max_repairs = _clamp(get("max_repairs", s.max_repairs), 0, 3)
        return s

    def characterization_key(self, model: str, effort: str) -> str:
        return _hash("characterization", model, effort, self.techniques)

    def plan_key(self, model: str, effort: str) -> str:
        return _hash("plan", model, effort, self.techniques, self.systems, self.max_plans, self.method,
                     self.realism, self.max_atoms, self.model_type)

    def overrides(self, names: dict[str, str]) -> OverridesConfig:
        """The planner's hard constraints: the chosen systems, the simulation method and the realism band."""
        chosen = ", ".join(f"{i} ({names[i]})" if i in names else i for i in self.systems)
        notes = [f"Model only the material system(s) the user chose: {chosen}; every plan's system_ids must be among them"]
        if self.method == "DFT":
            notes.append(f"The structures are for DFT: keep each at or below {self.max_atoms} atoms and prefer small "
                         "periodic cells")
        else:
            notes.append(f"The structures are for MD with a machine-learned potential: cells up to {self.max_atoms} "
                         "atoms are fine")
        if self.realism < 1 / 3:
            notes.append("Prefer the computationally cheapest model consistent with the evidence: the smallest ordered "
                         "cell, no defects, no surfaces unless the evidence requires them")
        elif self.realism < 2 / 3:
            notes.append("Balance realism and cost: an ordered supercell just large enough to represent the reported "
                         "composition approximately")
        else:
            notes.append("Prefer the model closest to the experiment within the atom budget: represent doping and "
                         "partial occupancy in a supercell and include the surfaces, particles or defects the "
                         "evidence supports")
        return OverridesConfig(model_type=self.model_type, notes=notes)


# --------------------------------------------------------------------------------------
# Projects
# --------------------------------------------------------------------------------------


class Project:
    """One uploaded paper and everything made from it, in ROOT/<id>/."""

    def __init__(self, directory: Path):
        self.dir = directory
        self.meta: dict = json.loads((directory / "project.json").read_text())

    @property
    def id(self) -> str:
        return self.dir.name

    @property
    def paper_dir(self) -> Path:
        return self.dir / "paper"

    def save(self) -> None:
        (self.dir / "project.json").write_text(json.dumps(self.meta, indent=2))

    @classmethod
    def create(cls, pdf: bytes, filename: str, model: str, effort: str, analyze_figures: bool,
               max_figures: int) -> "Project":
        stem = re.sub(r"[^A-Za-z0-9]+", "_", Path(filename).stem).strip("_")[:40] or "paper"
        directory = ROOT / f"{stem}_{time.strftime('%Y%m%d-%H%M%S')}"
        directory.mkdir(parents=True)
        (directory / "paper.pdf").write_bytes(pdf)
        meta = {"id": directory.name, "title": Path(filename).stem, "pdf_name": filename, "created": _now(),
                "model": model, "effort": effort, "analyze_figures": analyze_figures, "max_figures": max_figures,
                "summary": None, "current_plan": None, "current_run": None, "last_settings": None}
        (directory / "project.json").write_text(json.dumps(meta, indent=2))
        return cls(directory)

    @classmethod
    def load(cls, project_id: str) -> "Project":
        directory = ROOT / project_id
        if not _SAFE_NAME.match(project_id) or not (directory / "project.json").exists():
            raise KeyError(f"no project {project_id!r}")
        return cls(directory)

    @staticmethod
    def list_all() -> list["Project"]:
        if not ROOT.exists():
            return []
        found = [Project(p.parent) for p in ROOT.glob("*/project.json")]
        return sorted(found, key=lambda p: p.meta.get("created", ""), reverse=True)

    def delete(self) -> None:
        shutil.rmtree(self.dir)

    def graph(self) -> Optional[PaperGraph]:
        path = self.paper_dir / "paper_graph.json"
        return load_graph(path) if path.exists() else None

    def focused_graph(self, system_ids: list[str]) -> PaperGraph:
        """The summary restricted to the chosen systems: the planner can then only model (and the
        reviewer only judge against) those, while series-wide evidence that includes them stays."""
        graph = self.graph()
        keep = set(system_ids)
        systems = [s for s in graph.systems if s.system_id in keep]
        evidence = [e.model_copy(update={"system_ids": [i for i in e.system_ids if i in keep]})
                    for e in graph.evidence if not e.system_ids or keep & set(e.system_ids)]
        return build_graph(graph.source_pdf, graph.model, graph.metadata, systems, evidence)

    def plan_record(self, key: str) -> Optional[dict]:
        return _read_json(self.dir / "plans" / f"{key}.json")

    def characterizations(self) -> list[dict]:
        return sorted((_read_json(p) for p in (self.dir / "characterization").glob("*.json")),
                      key=lambda m: m["created"], reverse=True) if (self.dir / "characterization").exists() else []

    def run_dir(self, name: str) -> Path:
        directory = self.dir / "runs" / name
        if not _SAFE_NAME.match(name) or not directory.is_dir():
            raise KeyError(f"no run {name!r}")
        return directory

    def file(self, rel: str) -> Path:
        """A file inside the project folder (figures, structures, scripts); anything outside is refused."""
        path = (self.dir / rel).resolve()
        if self.dir.resolve() not in path.parents or not path.is_file():
            raise KeyError(f"no file {rel!r}")
        return path


# --------------------------------------------------------------------------------------
# Stage 1: summarize
# --------------------------------------------------------------------------------------


def run_summarize(project: Project, say: Callable[[str], None]) -> dict:
    meta, out = project.meta, project.paper_dir
    model = get_model(meta["model"])  # raises without an API key
    (out / "errors.txt").unlink(missing_ok=True)  # the summarizer only writes it when steps fail
    tokens: dict[str, int] = {}

    def progress(message: str) -> None:
        say(message)
        m = re.search(r"Tokens: (\d+) in / (\d+) out", message)  # the summarizer reports its usage in its last line
        if m:
            tokens.update(input=int(m[1]), output=int(m[2]))

    started = time.time()
    graph = asyncio.run(summarize_paper(project.dir / "paper.pdf", out, model, on_progress=progress,
                                        analyze_figures=meta["analyze_figures"], max_figures=meta["max_figures"]))
    errors = (out / "errors.txt").read_text().splitlines() if (out / "errors.txt").exists() else []
    if not graph.systems:
        raise RuntimeError("the summarizer found no material systems in this paper")
    for stale in ("characterization", "plans"):  # built on the previous summary, if any
        shutil.rmtree(project.dir / stale, ignore_errors=True)
    meta["title"] = graph.metadata.title
    meta["summary"] = {"created": _now(), "seconds": round(time.time() - started), "tokens": tokens,
                       "cost": estimate_cost(meta["model"], tokens.get("input", 0), tokens.get("output", 0)),
                       "errors": [e for e in errors if e.strip()], "systems": len(graph.systems),
                       "evidence": len(graph.evidence)}
    meta["current_plan"] = None
    project.save()
    say(f"Summary done: {len(graph.systems)} material systems, {len(graph.evidence)} evidence items.")
    return {"systems": len(graph.systems), "evidence": len(graph.evidence)}


# --------------------------------------------------------------------------------------
# Stage 2: characterization (cached) + plan
# --------------------------------------------------------------------------------------


def _check_systems(project: Project, settings: Settings) -> dict[str, str]:
    graph = project.graph()
    if graph is None:
        raise RuntimeError("summarize the paper first (panel 1)")
    names = {s.system_id: s.name for s in graph.systems}
    unknown = [i for i in settings.systems if i not in names]
    if not settings.systems or unknown:
        raise RuntimeError(f"choose the material system(s) to model on panel 2 (unknown: {unknown})" if unknown
                           else "choose at least one material system to model on panel 2")
    return names


def ensure_characterization(project: Project, settings: Settings,
                            say: Callable[[str], None]) -> tuple[str, ExperimentalEvidence]:
    """Agent 2's reports for this model, effort and technique set: reused when they exist."""
    model, effort = project.meta["model"], project.meta["effort"]
    key = settings.characterization_key(model, effort)
    folder, meta_path = project.dir / "characterization" / key, project.dir / "characterization" / f"{key}.json"
    if meta_path.exists() and any(folder.glob("*.json")):
        say(f"Evidence already analyzed for this model and technique set ({', '.join(sorted(_read_json(meta_path)['reports']))}): reused.")
        return key, ExperimentalEvidence.load(folder)
    say(f"Analyzing the evidence per technique (Agent 2, {model}, effort {effort})...")
    started = time.time()
    cc = CharacterizationConfig(techniques=settings.techniques or None, reasoning_effort=effort, output_retries=1)
    evidence = asyncio.run(characterize(project.graph(), project.paper_dir, folder, get_model(model), cc))
    if not evidence.reports:
        why = "; ".join(f"{t}: {r}" for t, r in evidence.skipped.items()) or "; ".join(evidence.errors)
        raise RuntimeError(f"no technique could be analyzed ({why})")
    meta_path.write_text(json.dumps({
        "key": key, "model": model, "effort": effort, "techniques": settings.techniques, "created": _now(),
        "seconds": round(time.time() - started), "reports": sorted(evidence.reports), "skipped": evidence.skipped,
        "errors": evidence.errors}, indent=2))
    return key, evidence


def make_plan(project: Project, settings: Settings, evidence: ExperimentalEvidence, char_key: str,
              say: Callable[[str], None]) -> dict:
    """Agent 3 on the chosen systems; the plan set is saved under its settings key."""
    model, effort = project.meta["model"], project.meta["effort"]
    names = _check_systems(project, settings)
    graph = project.focused_graph(settings.systems)
    say(f"Planning structures for {', '.join(names[i] for i in settings.systems)} "
        f"({settings.method}, at most {settings.max_atoms} atoms, {settings.max_plans} hypothesis/es)...")
    planner = LLMPlanner(model, effort, max_plans=settings.max_plans, max_atoms=settings.max_atoms, min_vacuum=15.0,
                         overrides=settings.overrides(names), max_retry=2)
    started = time.time()
    plans = asyncio.run(planner.plan(graph, evidence, [], ""))
    key = settings.plan_key(model, effort)
    record = {"key": key, "characterization": char_key, "model": model, "effort": effort, "created": _now(),
              "seconds": round(time.time() - started), "settings": asdict(settings), "usage": planner.usage,
              "cost": estimate_cost(model, planner.usage["input"], planner.usage["output"]),
              "plans": plans.model_dump(mode="json")}
    path = project.dir / "plans" / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    project.meta["current_plan"], project.meta["last_settings"] = key, asdict(settings)
    project.save()
    say(f"Plan ready: {len(plans.plans)} hypothesis/es: " + "; ".join(f"{p.plan_id} ({p.model_type})" for p in plans.plans))
    return record


def run_analyze(project: Project, settings: Settings, say: Callable[[str], None]) -> dict:
    _check_systems(project, settings)
    char_key, evidence = ensure_characterization(project, settings, say)
    if project.meta["model"] == "dry-run":  # canned answers exist for Agents 1-2 only
        say("Evidence analyzed. Planning needs a real model: choose one in the Model box above and click again.")
        return {"plan": None}
    record = make_plan(project, settings, evidence, char_key, say)
    return {"plan": record["key"]}


# --------------------------------------------------------------------------------------
# Stage 3: construction + review (the graph run of main.py, with the saved plan)
# --------------------------------------------------------------------------------------


class SavedPlanner:
    """Hands the graph the plan the user reviewed on panel 2 instead of planning again; a revision
    (reviewer feedback) goes to the real planner."""

    def __init__(self, plans: PlanSet, planner: LLMPlanner):
        self.saved, self.planner = plans, planner

    async def plan(self, summary, evidence, history, feedback) -> PlanSet:
        if not history and not feedback:
            return self.saved
        return await self.planner.plan(summary, evidence, history, feedback)

    @property
    def usage(self) -> dict:
        return self.planner.usage

    def preflight(self, **kwargs):
        return self.planner.preflight(**kwargs)


def build_config(project: Project, settings: Settings, paper_graph: Path, characterization: Path,
                 names: dict[str, str]) -> Paper2StructureConfig:
    cfg = load_config(str(BASE_CONFIG))
    cfg.campaign.model, cfg.campaign.reasoning_effort, cfg.campaign.max_retry = project.meta["model"], project.meta["effort"], 2
    cfg.inputs.pdf, cfg.inputs.paper_graph, cfg.inputs.characterization = None, str(paper_graph), str(characterization)
    cfg.characterization.techniques = settings.techniques or None
    cfg.characterization.reasoning_effort = project.meta["effort"]
    cfg.planning.max_plans, cfg.planning.max_atoms = settings.max_plans, settings.max_atoms
    cfg.overrides = settings.overrides(names)
    sandboxed = sandbox_available()[0]
    cc = cfg.construction
    cc.sandbox = "bubblewrap" if sandboxed else "none"
    cc.max_requests = int(30 + 30 * settings.realism)  # more realistic models get a larger coding budget
    cc.max_tool_calls = int(80 + 70 * settings.realism)
    cc.code_timeout = 300.0 if settings.max_atoms <= 2000 else 600.0
    if not sandboxed:
        cc.memory_limit_gb = None  # prlimit is Linux-only
    cfg.export.formats = ("poscar", "cif", "extxyz")
    return cfg


def run_generate(project: Project, settings: Settings, say: Callable[[str], None]) -> dict:
    names = _check_systems(project, settings)
    model, effort = project.meta["model"], project.meta["effort"]
    if model == "dry-run":
        raise RuntimeError("the dry-run model only covers the summary and the evidence; choose a real model on panel 1 "
                           "to build structures")
    char_key, evidence = ensure_characterization(project, settings, say)
    record = project.plan_record(settings.plan_key(model, effort))
    if record is None:
        say("The settings changed since the plan on panel 2: planning again with the new limits...")
        record = make_plan(project, settings, evidence, char_key, say)
    else:
        say(f"Reusing the plan of panel 2 ({len(record['plans']['plans'])} hypothesis/es).")
    run_dir = project.dir / "runs" / time.strftime("%Y%m%d-%H%M%S")
    graph_path = run_dir / "inputs" / "paper_graph.json"
    graph_path.parent.mkdir(parents=True)
    graph_path.write_text(project.focused_graph(settings.systems).model_dump_json(indent=2))
    cfg = build_config(project, settings, graph_path, project.dir / "characterization" / char_key, names)
    (run_dir / "settings.json").write_text(json.dumps({
        "settings": asdict(settings), "model": model, "effort": effort, "plan": record["key"],
        "characterization": char_key, "sandbox": cfg.construction.sandbox, "created": _now(),
        "systems": {i: names[i] for i in settings.systems}}, indent=2))
    (run_dir / "config.json").write_text(json.dumps({"config": cfg.model_dump(mode="json")}, indent=2))
    deps = build_deps(cfg, run_dir, seed=settings.seed, max_revisions=settings.max_revisions,
                      max_repairs=settings.max_repairs)
    deps.planner = SavedPlanner(PlanSet.model_validate(record["plans"]), deps.planner)
    say(f"Building and reviewing structures (Agents 4-5, {settings.method}, at most {settings.max_atoms} atoms, "
        f"sandbox {cfg.construction.sandbox}); this takes a few minutes...")
    started = time.time()
    result = asyncio.run(run_workflow(deps))
    project.meta["current_run"], project.meta["last_settings"] = run_dir.name, asdict(settings)
    project.save()
    say(f"Run finished in {round(time.time() - started)} s: status {result.status}, "
        f"{len(result.structure_files)} structure(s), {len(result.accepted_structure_ids)} accepted.")
    return {"run": run_dir.name, "status": result.status}


# --------------------------------------------------------------------------------------
# Views for the page
# --------------------------------------------------------------------------------------


def _reports_view(folder: Path) -> list[dict]:
    out = []
    for path in sorted(folder.glob("*.json")):
        rep = _read_json(path)
        out.append({"technique": rep["technique"], "observations": len(rep["observations"]),
                    "findings": rep["findings"], "constraints": rep["structural_constraints"], "caveats": rep["caveats"]})
    return out


def _summary_view(project: Project, graph: PaperGraph) -> dict:
    techniques: dict[str, dict] = {}
    for e in graph.evidence:
        spec = find_spec(e.technique)
        t = techniques.setdefault(e.technique, {
            "name": e.technique, "count": 0, "figures": 0,
            "agent": "ready" if spec and spec.implemented else ("stub" if spec else "none")})
        t["count"] += 1
        t["figures"] += e.source_type == "figure"
    evidence = []
    for e in graph.evidence:
        d = e.model_dump(mode="json")
        d["image_url"] = f"/api/projects/{project.id}/files/paper/{e.image_path}" if e.image_path else None
        evidence.append(d)
    return {"metadata": graph.metadata.model_dump(mode="json"), "model": graph.model,
            "systems": [s.model_dump(mode="json") for s in graph.systems], "evidence": evidence,
            "techniques": sorted(techniques.values(), key=lambda t: t["name"])}


def _run_summary(project: Project, run: Path) -> dict:
    result, settings = _read_json(run / "result.json"), _read_json(run / "settings.json", {})
    return {"name": run.name, "created": settings.get("created", run.name), "settings": settings.get("settings"),
            "systems": settings.get("systems", {}), "model": settings.get("model"),
            "status": result["status"] if result else "unfinished",
            "n_structures": len(result["structure_files"]) if result else 0,
            "n_accepted": len(result["accepted_structure_ids"]) if result else 0}


def project_state(project: Project, active_job: Optional[str]) -> dict:
    """Everything the page shows for one project (the run details are fetched separately)."""
    graph = project.graph()
    runs = sorted((project.dir / "runs").glob("*/settings.json"), reverse=True) if (project.dir / "runs").exists() else []
    chars = project.characterizations()
    for c in chars:
        c["reports_detail"] = _reports_view(project.dir / "characterization" / c["key"])
    return {"project": project.meta, "active_job": active_job,
            "summary": _summary_view(project, graph) if graph else None,
            "characterizations": chars,
            "plan": project.plan_record(project.meta["current_plan"]) if project.meta.get("current_plan") else None,
            "runs": [_run_summary(project, r.parent) for r in runs]}


def run_detail(project: Project, name: str) -> dict:
    run = project.run_dir(name)
    result = _read_json(run / "result.json")
    settings = _read_json(run / "settings.json", {})
    artifacts = _read_json(run / "artifacts.json", [])
    plans = {p["plan_id"]: p for ps in _read_json(run / "plans.json", []) for p in ps["plans"]}
    validation = _read_json(run / "validation_reports.json", {})
    history = _read_json(run / "revision_history.json", [])
    accepted = set(result["accepted_structure_ids"]) if result else set()
    method = (settings.get("settings") or {}).get("method", "DFT")
    structures = []
    for a in artifacts:
        atoms = read(run / (a["files"].get("extxyz") or a["files"]["poscar"]))
        rep = validation.get(a["structure_id"], {})
        structures.append({
            **a, "accepted": a["structure_id"] in accepted, "info": structure_info(atoms),
            "estimate": compute_estimate(atoms, method), "plan": plans.get(a["plan_id"]),
            "validation": {"passed": rep.get("passed"), "checks": rep.get("checks", [])},
            "downloads": {**{fmt: f"/api/projects/{project.id}/files/runs/{name}/{p}" for fmt, p in a["files"].items()},
                          "script": f"/api/projects/{project.id}/files/runs/{name}/{a['script']}"},
            "image_url": f"/api/projects/{project.id}/files/runs/{name}/{a['image']}" if a.get("image") else None})
    tokens = _read_json(run / "token_log.json", {})
    usage = {k: v for k, v in tokens.items()}
    total_in = sum(u.get("input", 0) for u in usage.values())
    total_out = sum(u.get("output", 0) for u in usage.values())
    return {"name": name, "status": result["status"] if result else "unfinished", "result": result, "settings": settings,
            "structures": structures, "plans": list(plans.values()), "decision": history[-1] if history else None,
            "report": (run / "final_report.md").read_text() if (run / "final_report.md").exists() else "",
            "errors": result["errors"] if result else [],
            "usage": {"agents": usage, "input": total_in, "output": total_out,
                      "cost": estimate_cost(settings.get("model", ""), total_in, total_out)},
            "zip_url": f"/api/projects/{project.id}/runs/{name}/zip"}


def structure_model(project: Project, name: str, structure_id: str) -> dict:
    """The structure as CIF (periodic) or XYZ text for the 3Dmol.js viewer."""
    run = project.run_dir(name)
    artifact = next((a for a in _read_json(run / "artifacts.json", []) if a["structure_id"] == structure_id), None)
    if artifact is None:
        raise KeyError(f"no structure {structure_id!r} in run {name!r}")
    atoms = read(run / (artifact["files"].get("extxyz") or artifact["files"]["poscar"]))
    periodic = bool(atoms.pbc.all())
    buf = io.BytesIO()
    write(buf, atoms, format="cif" if periodic else "xyz")
    return {"format": "cif" if periodic else "xyz", "text": buf.getvalue().decode(), "periodic": periodic,
            "n_atoms": len(atoms), "elements": sorted(set(atoms.get_chemical_symbols()))}


def run_zip(project: Project, name: str) -> bytes:
    run = project.run_dir(name)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in run.rglob("*"):
            if f.is_file() and "construction" not in f.relative_to(run).parts[:1]:  # workspaces are large
                z.write(f, f"{name}/{f.relative_to(run)}")
    return buf.getvalue()


# --------------------------------------------------------------------------------------
# Jobs: one worker thread, the page polls
# --------------------------------------------------------------------------------------


@dataclass
class Job:
    id: str
    project_id: str
    kind: str  # summarize | analyze | generate
    status: str = "queued"  # queued | running | done | failed
    log: list[str] = field(default_factory=list)
    error: Optional[str] = None
    result: Any = None
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


class _JobLog(logging.Handler):
    """Routes the workflow's log lines (the agents log from worker threads too) into the running job."""

    def __init__(self, job: Job):
        super().__init__(logging.INFO)
        self.job = job

    def emit(self, record: logging.LogRecord) -> None:
        try:  # a failing display must never fail the run
            self.job.log.append(record.getMessage())
        except Exception:  # noqa: BLE001
            pass


class JobManager:
    """Runs the jobs one after another on one worker thread: the agents are already concurrent
    inside a stage, and two runs at once would mix their log lines and compete for the API."""

    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        threading.Thread(target=self._worker, name="p2s-jobs", daemon=True).start()

    def submit(self, project_id: str, kind: str, fn: Callable[[Callable[[str], None]], Any]) -> Job:
        with self._lock:
            if self.active(project_id):
                raise RuntimeError("this paper already has a job running; wait for it to finish")
            job = Job(id=uuid.uuid4().hex[:12], project_id=project_id, kind=kind)
            self.jobs[job.id] = job
        self._queue.put((job, fn))
        return job

    def active(self, project_id: str) -> Optional[Job]:
        return next((j for j in self.jobs.values() if j.project_id == project_id and j.status in ("queued", "running")),
                    None)

    def _worker(self) -> None:
        logger = logging.getLogger("paper2structure")
        logger.setLevel(logging.INFO)
        while True:
            job, fn = self._queue.get()
            handler = _JobLog(job)
            logger.addHandler(handler)
            job.status, job.started = "running", time.time()
            try:
                job.result = fn(job.log.append)
                job.status = "done"
            except Exception as e:  # noqa: BLE001 - the page shows any failure
                job.error = f"{type(e).__name__}: {e}"
                job.log.append(f"ERROR {job.error}")
                job.status = "failed"
                log.error("job %s (%s) failed:\n%s", job.id, job.kind, traceback.format_exc())
            finally:
                job.finished = time.time()
                logger.removeHandler(handler)
