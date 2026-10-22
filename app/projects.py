"""Project store, the three stages of the web tool (app/server.py) and its job runner.

    1. summarize     the summarizer agent on the PDF                       -> <project>/paper/
    2. characterize  the characterization agent per technique, cached per model,  -> <project>/characterization/<key>/
                     effort and technique set
    3. generate      the planner agent on the material systems the user chose,  -> <project>/plans/<key>.json
                     the atom budget and DFT/MD brief (cached per settings),
                     then constructor and reviser agents as one graph run,       -> <project>/runs/<time>/
                     validation and review (the saved plan is handed to the
                     graph; the saved summary and evidence are loaded, not
                     recomputed)

A project is a folder under outputs/projects/ holding every stage's files, so closing the page or
restarting the server loses nothing, and a stage only runs again when the settings it depends on
changed. Jobs run on a small pool of worker threads (one job per paper at a time, several papers at
once); each job's log lines are routed to it through a ContextVar.
"""

from __future__ import annotations

import ast
import asyncio
import contextvars
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
from paper2structure.agents.constructor.workspace import replay, sandbox_available
from paper2structure.agents.planner import LLMPlanner
from paper2structure.agents.planner.schemas import PlanSet
from paper2structure.agents.summarizer.graph import summarize_paper
from paper2structure.agents.summarizer.knowledge_graph import build_graph, load_graph
from paper2structure.agents.summarizer.schemas import PaperGraph
from paper2structure.config import (DEFAULT_MODEL, CharacterizationConfig, OverridesConfig, Paper2StructureConfig,
                                    canonical_technique, get_model, load_config)
from paper2structure.graph.nodes import characterize
from paper2structure.graph.workflow import build_deps, make_planner, run_workflow
from paper2structure.tools.geometry_checks import validate_structure
from paper2structure.tools.render import render_png
from paper2structure.tools.structure_export import export_structure
from structure_info import compute_estimate, structure_info

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("P2S_WEB_ROOT", REPO / "outputs" / "projects"))
BASE_CONFIG = REPO / "config" / "config.yml"
WORKERS = max(1, int(os.environ.get("P2S_WORKERS", "3")))  # papers processed at the same time
EFFORTS = ("low", "medium", "high")
MODEL_TYPES = ["bulk", "slab", "nanocrystal", "core-shell nanocrystal", "supported nanoparticle", "slab with adsorbates"]
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
log = logging.getLogger("webtool")  # outside the paper2structure tree: its tracebacks stay out of the job logs

# Models offered in the page: (group, pydantic-ai name, title, USD per 1M input tokens, per 1M output tokens,
# note); prices checked 2026-10-22, None = not published. "google:" names go to the Gemini API (pydantic-ai
# reads GOOGLE_API_KEY, or the older GEMINI_API_KEY); several Gemini models have a free tier. The agents take
# any pydantic-ai model name, so this list lives in the page, not in the library.
MODEL_CATALOG: list[tuple[str, str, str, Optional[float], Optional[float], str]] = [
    ("OpenAI · fast and cheap", "openai:gpt-5.6-luna", "GPT-5.6 Luna", 0.20, 1.20, ""),
    ("OpenAI · fast and cheap", "openai:gpt-6-luna", "GPT-6 Luna", 0.10, 0.50, ""),
    ("OpenAI · fast and cheap", "openai:gpt-5-nano", "GPT-5 nano", 0.05, 0.40, "cheap per token but slow: thinks a lot per call"),
    ("OpenAI · fast and cheap", "openai:gpt-5.4-nano", "GPT-5.4 nano", 0.20, 1.25, ""),
    ("OpenAI · fast and cheap", "openai:gpt-5.4-mini", "GPT-5.4 mini", 0.75, 4.50, ""),
    ("OpenAI · advanced", "openai:gpt-6.1-sol", "GPT-6.1 Sol", 2.00, 10.00, "the newest Sol"),
    ("OpenAI · advanced", "openai:gpt-6-sol", "GPT-6 Sol", 2.00, 10.00, ""),
    ("OpenAI · advanced", "openai:gpt-5.6-sol", "GPT-5.6 Sol", 4.00, 20.00, "promotional price until 2026-11-21"),
    ("OpenAI · advanced", "openai:gpt-5.6-terra", "GPT-5.6 Terra", 2.00, 12.00, ""),
    ("OpenAI · advanced", "openai:gpt-6-astra", "GPT-6 Astra", 10.00, 50.00, "the top tier; expensive"),
    ("Google Gemini · free tier available", "google:gemini-3.8-flash", "Gemini 3.8 Flash", None, None,
     "free tier; paid price not published"),
    ("Google Gemini · free tier available", "google:gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite", None, None, "free tier"),
    ("Google Gemini · free tier available", "google:gemini-2.5-flash", "Gemini 2.5 Flash", 0.30, 2.50, "free tier for accounts that used it before"),
    ("Google Gemini · paid", "google:gemini-2.5-pro", "Gemini 2.5 Pro", 1.25, 10.00, ""),
    ("Test", "dry-run", "Dry run", 0.0, 0.0, "free; canned answers for the summary and the evidence only"),
]
PROVIDER_KEYS = {"openai": ("OPENAI_API_KEY",), "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY")}


def provider_of(name: str) -> str:
    """The provider of a pydantic-ai model name ("openai", "google", ...); a bare name is OpenAI's."""
    return name.split(":", 1)[0] if ":" in name else "openai"


def has_key(provider: str) -> bool:
    keys = PROVIDER_KEYS.get(provider)
    return keys is None or any(os.getenv(k) for k in keys)


def check_model(name: str) -> None:
    """Raise with the variable to set when the model's provider has no API key in the environment; the
    library's get_model only knows OpenAI's, and pydantic-ai would otherwise fail inside the job."""
    if name == "dry-run":
        return
    provider = provider_of(name)
    if not has_key(provider):
        key = PROVIDER_KEYS[provider][0]
        raise RuntimeError(f"{key} is not set. Put it in a .env file in the repo root ({key}=...), "
                           "or use model 'dry-run'.")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _clamp(value, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def _hash(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:10]


def _read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def _write_json(path: Path, obj) -> None:
    """Atomic: handlers read these files while the workers write them, and a crash mid-write must not
    leave a truncated file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------
# Models and prices
# --------------------------------------------------------------------------------------


def default_model() -> str:
    return os.getenv("P2S_MODEL", DEFAULT_MODEL)


def model_choices() -> list[dict]:
    """The models the page offers, grouped, with the prices from paper2structure.config."""
    out = []
    for group, name, title, price_in, price_out, note in MODEL_CATALOG:
        price = "free" if price_in == 0 else f"${price_in:.2f} / ${price_out:.2f}" if price_in else "price not published"
        out.append({"name": name, "title": title, "group": group, "price": price, "note": note,
                    "provider": "none" if name == "dry-run" else provider_of(name), "default": name == default_model()})
    return out


def key_status() -> dict[str, bool]:
    return {provider: has_key(provider) for provider in PROVIDER_KEYS}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """USD from the catalogue prices, or None when the model's price is unknown (cached input billed as input)."""
    for _, name, _, price_in, price_out, _ in MODEL_CATALOG:
        if name == model and price_in is not None:
            return round((input_tokens * price_in + output_tokens * price_out) / 1e6, 4)
    return None


# --------------------------------------------------------------------------------------
# Settings (panels 2 and 3)
# --------------------------------------------------------------------------------------


DFT_MAX_ATOMS = 600  # up to here the brief is DFT-sized periodic cells, beyond it MD on a machine-learned potential


def derived_brief(max_atoms: int) -> tuple[str, float]:
    """The simulation method and the realism band the planner is briefed with, from the one control the
    page has, the atom limit: small limits ask for the cheapest ordered cell, large ones for doping,
    surfaces or particles (app.js mirrors the DFT/MD threshold in its hint)."""
    if max_atoms <= DFT_MAX_ATOMS:
        return "DFT", round(min(1.0, max(0.0, (max_atoms - 60) / 540)), 2)
    return "MD", round(min(1.0, max(0.0, (max_atoms - 500) / 9500)), 2)


@dataclass
class Settings:
    """Everything the user chooses on panels 2 and 3, stored with every characterization, plan and run.
    `method` and `realism` are derived from `max_atoms` (derived_brief) unless given explicitly."""
    systems: list[str] = field(default_factory=list)  # the material systems to model (summarizer IDs)
    techniques: list[str] = field(default_factory=list)  # analyze only these; empty = every technique with an agent
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
        s.max_atoms = _clamp(get("max_atoms", s.max_atoms), 10, 50000)
        if raw.get("method") is None and raw.get("realism") is None:
            s.method, s.realism = derived_brief(s.max_atoms)
        else:  # older callers that still choose them
            s.method = "MD" if str(get("method", s.method)).upper() == "MD" else "DFT"
            s.realism = round(min(1.0, max(0.0, float(get("realism", s.realism)))), 2)
        model_type = raw.get("model_type") or None
        s.model_type = model_type if model_type in MODEL_TYPES else None
        s.seed = int(get("seed", s.seed))
        s.max_revisions = _clamp(get("max_revisions", s.max_revisions), 0, 2)
        s.max_repairs = _clamp(get("max_repairs", s.max_repairs), 0, 3)
        return s

    def plan_key(self, model: str, effort: str, techniques: list[str]) -> str:
        """`techniques` is the resolved list the evidence analysis ran on (see resolve_techniques). The
        method and realism are not part of the key: they follow from max_atoms."""
        return _hash("plan2", model, effort, techniques, self.systems, self.max_plans, self.max_atoms, self.model_type)

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


def available_techniques(graph: PaperGraph) -> list[str]:
    """The techniques found in the paper that have an analysis agent (every one in the vocabulary), sorted."""
    return [t for t in sorted({e.technique for e in graph.evidence}) if find_spec(t)]


def resolve_techniques(graph: PaperGraph, requested: list[str]) -> list[str]:
    """The techniques the evidence analysis runs on: every available one, or the requested ones among
    them. The same list from either spelling keys the same cached analysis."""
    available = available_techniques(graph)
    if not requested:
        return available
    wanted = {canonical_technique(t) or t for t in requested}
    return [t for t in available if t in wanted]


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
        _write_json(self.dir / "project.json", self.meta)

    @classmethod
    def create(cls, pdf: bytes, filename: str, model: str, effort: str, analyze_figures: bool, max_figures: int,
               then_characterize: bool) -> "Project":
        stem = re.sub(r"[^A-Za-z0-9]+", "_", Path(filename).stem).strip("_")[:40] or "paper"
        directory = ROOT / f"{stem}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:4]}"
        directory.mkdir(parents=True)
        (directory / "paper.pdf").write_bytes(pdf)
        meta = {"id": directory.name, "title": Path(filename).stem, "pdf_name": filename, "created": _now(),
                "model": model, "effort": effort, "analyze_figures": analyze_figures, "max_figures": max_figures,
                "then_characterize": then_characterize, "summary": None, "current_plan": None, "current_run": None,
                "last_settings": None}
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
        found = []
        for p in ROOT.glob("*/project.json"):
            try:
                found.append(Project(p.parent))
            except (OSError, ValueError) as e:  # one damaged folder must not hide every other paper
                log.warning("skipping %s: %s", p.parent.name, e)
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

    @staticmethod
    def _plan(record: Optional[dict]) -> Optional[dict]:
        """Records saved before the technique list was stored get it from their settings (the same
        sorted list, so the page's match check agrees with the server's key)."""
        if record is not None:
            record.setdefault("techniques", record["settings"].get("techniques", []))
        return record

    def plan_record(self, key: str) -> Optional[dict]:
        return self._plan(_read_json(self.dir / "plans" / f"{key}.json"))

    def plans(self) -> list[dict]:
        folder = self.dir / "plans"
        return sorted((self._plan(_read_json(p)) for p in folder.glob("*.json")), key=lambda r: r["created"],
                      reverse=True) if folder.exists() else []

    def characterizations(self) -> list[dict]:
        folder = self.dir / "characterization"
        return sorted((_read_json(p) for p in folder.glob("*.json")), key=lambda m: m["created"], reverse=True) \
            if folder.exists() else []

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
    """The summarizer agent into a scratch folder that replaces paper/ only on success, so a rejected summary
    (no material systems) leaves the previous one and the caches built on it untouched."""
    meta = project.meta
    model = get_model(meta["model"])  # raises without the provider's API key
    scratch = project.dir / "paper.new"
    shutil.rmtree(scratch, ignore_errors=True)
    tokens: dict[str, int] = {}

    def progress(message: str) -> None:
        say(message)
        m = re.search(r"Tokens: (\d+) in / (\d+) out", message)  # the summarizer reports its usage in its last line
        if m:
            tokens.update(input=int(m[1]), output=int(m[2]))

    started = time.time()
    try:
        graph = run_async(summarize_paper(project.dir / "paper.pdf", scratch, model, on_progress=progress,
                                          analyze_figures=meta["analyze_figures"], max_figures=meta["max_figures"]))
        if not graph.systems:
            raise RuntimeError("the summarizer found no material systems in this paper")
    except Exception:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    errors = (scratch / "errors.txt").read_text().splitlines() if (scratch / "errors.txt").exists() else []
    for stale in ("characterization", "plans"):  # built on the previous summary, if any
        shutil.rmtree(project.dir / stale, ignore_errors=True)
    shutil.rmtree(project.paper_dir, ignore_errors=True)
    os.replace(scratch, project.paper_dir)
    meta["title"] = graph.metadata.title
    meta["summary"] = {"created": _now(), "seconds": round(time.time() - started), "tokens": tokens,
                       "cost": estimate_cost(meta["model"], tokens.get("input", 0), tokens.get("output", 0)),
                       "errors": [e for e in errors if e.strip()], "systems": len(graph.systems),
                       "evidence": len(graph.evidence)}
    meta["current_plan"] = None
    project.save()
    say(f"Summary done: {len(graph.systems)} material systems, {len(graph.evidence)} evidence items.")
    result = {"systems": len(graph.systems), "evidence": len(graph.evidence)}
    if meta.get("then_characterize"):
        try:
            key, _ = ensure_characterization(project, Settings(), say)
            result["characterization"] = key
        except Exception as e:  # noqa: BLE001 - the summary is saved and usable; only the characterization agent failed
            say(f"Evidence analysis skipped: {e}")
            result["characterization_error"] = str(e)
    return result


# --------------------------------------------------------------------------------------
# Stage 2: characterization (cached per model, effort and technique set)
# --------------------------------------------------------------------------------------


def _check_summary(project: Project) -> PaperGraph:
    graph = project.graph()
    if graph is None:
        raise RuntimeError("summarize the paper first (panel 1)")
    return graph


def ensure_characterization(project: Project, settings: Settings,
                            say: Callable[[str], None]) -> tuple[str, ExperimentalEvidence]:
    """The characterization agent's reports for this model, effort and technique set: reused when they exist."""
    graph = _check_summary(project)
    model, effort = project.meta["model"], project.meta["effort"]
    techniques = resolve_techniques(graph, settings.techniques)
    if not techniques:
        raise RuntimeError("none of the chosen techniques has an analysis agent; available: "
                           + (", ".join(available_techniques(graph)) or "none in this paper"))
    key = _hash("characterization", model, effort, techniques)
    folder, meta_path = project.dir / "characterization" / key, project.dir / "characterization" / f"{key}.json"
    saved = _read_json(meta_path)
    if saved and set(saved.get("reports", [])) >= set(techniques) and any(folder.glob("*.json")):
        say(f"Evidence already analyzed for this model and technique set ({', '.join(techniques)}): reused, no model call.")
        return key, ExperimentalEvidence.load(folder)
    say(("Some techniques failed last time; analyzing again. " if saved else "")
        + f"Analyzing the evidence per technique (characterization agent: {', '.join(techniques)}; {model}, effort {effort})...")
    started = time.time()
    cc = CharacterizationConfig(techniques=techniques, reasoning_effort=effort, output_retries=1)
    evidence = run_async(characterize(graph, project.paper_dir, folder, get_model(model), cc))
    if not evidence.reports:
        raise RuntimeError("no technique could be analyzed: " + ("; ".join(evidence.errors) or "unknown reason"))
    _write_json(meta_path, {
        "key": key, "model": model, "effort": effort, "techniques": techniques, "created": _now(),
        "seconds": round(time.time() - started), "reports": sorted(evidence.reports), "skipped": evidence.skipped,
        "errors": evidence.errors, "complete": not evidence.errors})
    say(f"Evidence analyzed in {round(time.time() - started)} s: " + "; ".join(
        f"{t} {len(r.observations)} observations, {len(r.structural_constraints)} constraints"
        for t, r in evidence.reports.items()))
    return key, evidence


def run_characterize(project: Project, settings: Settings, say: Callable[[str], None]) -> dict:
    key, _ = ensure_characterization(project, settings, say)
    return {"characterization": key}


# --------------------------------------------------------------------------------------
# Stage 3: plan (cached per settings) + construction + review (the graph run of main.py)
# --------------------------------------------------------------------------------------


def _check_systems(project: Project, settings: Settings) -> dict[str, str]:
    graph = _check_summary(project)
    names = {s.system_id: s.name for s in graph.systems}
    unknown = [i for i in settings.systems if i not in names]
    if not settings.systems or unknown:
        raise RuntimeError(f"unknown material system(s) {unknown}; this paper has {sorted(names)}" if unknown
                           else "choose at least one material system to model")
    return names


def make_plan(project: Project, settings: Settings, evidence: ExperimentalEvidence, char_key: str,
              techniques: list[str], say: Callable[[str], None]) -> dict:
    """The planner agent on the chosen systems; the plan set is saved under its settings key."""
    model, effort = project.meta["model"], project.meta["effort"]
    names = _check_systems(project, settings)
    graph = project.focused_graph(settings.systems)
    say(f"Planning structures (planner agent) for {', '.join(names[i] for i in settings.systems)}: "
        f"{settings.method}, at most {settings.max_atoms} atoms, {settings.max_plans} hypothesis/es...")
    planner = make_planner(settings_config(project, settings, names))  # the planner the graph run would build
    started = time.time()
    plans = run_async(planner.plan(graph, evidence, [], ""))
    key = settings.plan_key(model, effort, techniques)
    record = {"key": key, "characterization": char_key, "techniques": techniques, "model": model, "effort": effort,
              "created": _now(), "seconds": round(time.time() - started), "settings": asdict(settings),
              "usage": planner.usage, "cost": estimate_cost(model, planner.usage["input"], planner.usage["output"]),
              "plans": plans.model_dump(mode="json")}
    _write_json(project.dir / "plans" / f"{key}.json", record)
    project.meta["current_plan"] = key
    project.save()
    say(f"Plan ready in {record['seconds']} s: " + "; ".join(f"{p.plan_id} ({p.model_type})" for p in plans.plans))
    return record


class SavedPlanner:
    """Hands the graph the saved plan instead of planning again; a revision (reviewer feedback) goes
    to the real planner."""

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


def settings_config(project: Project, settings: Settings, names: dict[str, str]) -> Paper2StructureConfig:
    """config/config.yml with the paper's model and the page's settings: what panel 3's planner and
    the graph run share."""
    cfg = load_config(str(BASE_CONFIG))
    cfg.campaign.model, cfg.campaign.reasoning_effort, cfg.campaign.max_retry = project.meta["model"], project.meta["effort"], 2
    cfg.campaign.simulation = "DFT" if settings.method == "DFT" else "MLIP"  # MD runs on a machine-learned potential
    cfg.characterization.reasoning_effort = project.meta["effort"]
    cfg.planning.max_plans, cfg.planning.max_atoms = settings.max_plans, settings.max_atoms
    cfg.budgets.max_revisions, cfg.budgets.max_repairs = settings.max_revisions, settings.max_repairs
    cfg.overrides = settings.overrides(names)
    return cfg


def build_config(project: Project, settings: Settings, paper_graph: Path, characterization: Path,
                 names: dict[str, str]) -> Paper2StructureConfig:
    cfg = settings_config(project, settings, names)
    cfg.inputs.pdf, cfg.inputs.paper_graph, cfg.inputs.characterization = None, str(paper_graph), str(characterization)
    sandboxed = sandbox_available()[0]
    cc = cfg.construction
    cc.sandbox = "bubblewrap" if sandboxed else "none"
    # the YAML default (60 requests) at the tractable end, more for realistic models: a slab with adsorbates
    # ran out at 39 requests, a bulk cell needs about 15
    cc.max_requests = int(60 + 40 * settings.realism)
    cc.max_tool_calls = int(150 + 100 * settings.realism)
    cc.code_timeout = 300.0 if settings.max_atoms <= 2000 else 600.0
    if not sandboxed:
        cc.memory_limit_gb = None  # prlimit is Linux-only
    cfg.export.formats = ("poscar", "cif", "extxyz")
    return cfg


def run_generate(project: Project, settings: Settings, say: Callable[[str], None]) -> dict:
    names = _check_systems(project, settings)
    model, effort = project.meta["model"], project.meta["effort"]
    if model == "dry-run":
        raise RuntimeError("the dry-run model only covers the summary and the evidence; choose a real model to plan "
                           "and build structures")
    char_key, evidence = ensure_characterization(project, settings, say)
    techniques = _read_json(project.dir / "characterization" / f"{char_key}.json")["techniques"]
    record = project.plan_record(settings.plan_key(model, effort, techniques))
    if record is None:
        record = make_plan(project, settings, evidence, char_key, techniques, say)
    else:
        say(f"A plan for exactly these settings exists ({len(record['plans']['plans'])} hypothesis/es): reused, no model call.")
    run_dir = project.dir / "runs" / time.strftime("%Y%m%d-%H%M%S")
    graph_path = run_dir / "inputs" / "paper_graph.json"
    graph_path.parent.mkdir(parents=True)
    graph_path.write_text(project.focused_graph(settings.systems).model_dump_json(indent=2))
    cfg = build_config(project, settings, graph_path, project.dir / "characterization" / char_key, names)
    _write_json(run_dir / "settings.json", {
        "settings": asdict(settings), "model": model, "effort": effort, "plan": record["key"],
        "characterization": char_key, "sandbox": cfg.construction.sandbox, "created": _now(),
        "systems": {i: names[i] for i in settings.systems}})
    _write_json(run_dir / "config.json", {"config": cfg.model_dump(mode="json")})
    deps = build_deps(cfg, run_dir, seed=settings.seed)  # the loop budgets are in cfg.budgets
    deps.planner = SavedPlanner(PlanSet.model_validate(record["plans"]), deps.planner)
    say("Starting the graph run: the saved summary, evidence and plan are loaded from disk (summarizer, "
        f"characterization and planner agents are not run again); then construction and review (constructor and reviser agents, {settings.method}, at most {settings.max_atoms} atoms, "
        f"sandbox {cfg.construction.sandbox}). A few minutes...")
    started = time.time()
    result = run_async(run_workflow(deps))
    project.meta["current_run"], project.meta["last_settings"] = run_dir.name, asdict(settings)
    project.save()
    say(f"Run finished in {round(time.time() - started)} s: status {result.status}, "
        f"{len(result.structure_files)} structure(s), {len(result.accepted_structure_ids)} accepted.")
    return {"run": run_dir.name, "status": result.status}


# --------------------------------------------------------------------------------------
# After a run: build more structures from a validated script by changing its parameters
# --------------------------------------------------------------------------------------

_PARAM_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")  # the script contract: named parameters at the top, UPPER_CASE


def _literal(node: ast.expr):
    """The JSON value of a literal node (number, string, bool, or a tuple/list of those), else ValueError."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (bool, int, float, str)):
        return node.value
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)) and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float)) and not isinstance(node.operand.value, bool)):
        return -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_literal(e) for e in node.elts]
    raise ValueError("not a literal")


def _kind(value) -> str:
    return "bool" if isinstance(value, bool) else "number" if isinstance(value, (int, float)) else "list" if isinstance(value, list) else "text"


def script_params(path: Path) -> list[dict]:
    """The named parameters at the top of a build script (module-level ``NAME = literal`` lines, with the
    comment on the line): what the page lets the user change to build more structures."""
    src = path.read_text()
    lines = src.encode().split(b"\n")
    out = []
    for node in ast.parse(src).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and _PARAM_NAME.match(node.targets[0].id) and node.value.lineno == node.value.end_lineno):
            continue
        try:
            value = _literal(node.value)
        except ValueError:
            continue
        raw = lines[node.lineno - 1]
        rest = raw[node.value.end_col_offset:].decode(errors="replace")
        out.append({"name": node.targets[0].id, "value": value, "kind": _kind(value),
                    "comment": rest.split("#", 1)[1].strip() if "#" in rest else "", "line": node.lineno,
                    "col": [node.value.col_offset, node.value.end_col_offset],
                    "tuple": raw[node.value.col_offset:node.value.col_offset + 1] == b"("})
    return out


def _coerce(value, param: dict):
    """A new value in the parameter's kind (ints stay ints when the new value is whole), else ValueError."""
    name, kind, old = param["name"], param["kind"], param["value"]

    def number(v, like):
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{name}: {v!r} is not a number") from None
        if f != f or f in (float("inf"), float("-inf")):
            raise ValueError(f"{name}: must be finite")
        return int(f) if isinstance(like, int) and not isinstance(like, bool) and f == int(f) else f

    if kind == "number":
        return number(value, old)
    if kind == "bool":
        return value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes", "on")
    if kind == "list":
        items = [x.strip() for x in value.split(",") if x.strip()] if isinstance(value, str) else list(value)
        if not items:
            raise ValueError(f"{name}: give at least one value")
        like = old[0] if old else 1.0
        return [number(x, like) if isinstance(like, (int, float)) else str(x) for x in items]
    return str(value)


def _py_literal(value, param: dict) -> str:
    if isinstance(value, list):
        inner = ", ".join(_py_literal(v, param) for v in value)
        return f"({inner},)" if param["tuple"] and len(value) == 1 else f"({inner})" if param["tuple"] else f"[{inner}]"
    return repr(value)


def rewrite_script(src: str, params: list[dict], values: dict) -> tuple[str, dict, dict]:
    """The script with the given parameter values substituted in place (comments and everything else
    kept). Returns (source, every parameter's value, the ones that changed)."""
    lines = src.encode().split(b"\n")
    used, changed = {}, {}
    for p in params:
        if p["name"] not in values or values[p["name"]] is None or values[p["name"]] == "":
            used[p["name"]] = p["value"]
            continue
        new = _coerce(values[p["name"]], p)
        used[p["name"]] = new
        if new != p["value"]:
            changed[p["name"]] = new
            a, b = p["col"]
            raw = lines[p["line"] - 1]
            lines[p["line"] - 1] = raw[:a] + _py_literal(new, p).encode() + raw[b:]
    return b"\n".join(lines).decode(), used, changed


def _run_config(run: Path) -> Paper2StructureConfig:
    """The run's construction, validation, planning and campaign settings on top of today's base config.
    Copied field by field because a run saved by an earlier version of the library may hold keys the
    current config no longer accepts."""
    cfg = load_config(str(BASE_CONFIG))
    saved = (_read_json(run / "config.json") or {}).get("config", {})
    for section in ("campaign", "planning", "validation", "construction"):
        model = getattr(cfg, section)
        for key, value in (saved.get(section) or {}).items():
            if key in type(model).model_fields:
                setattr(model, key, value)
    cc = cfg.construction
    cc.sandbox = "bubblewrap" if sandbox_available()[0] else "none"
    if cc.sandbox == "none":
        cc.memory_limit_gb = None
    return cfg


_UNREPORTED = re.compile(r"^(?P<name>.+)__(?P<attempt>r\d+_a\d+)$")  # a workspace structure's id: <name>__r<rev>_a<attempt>


def _structure_source(run: Path, structure_id: str) -> dict:
    """Where a structure's files and script are: a registered artifact of the run, or a structure the
    constructor agent emitted in a workspace but never reported as finished (the plan was given up
    afterwards). Paths are relative to the run; `script` is None for a structure built inline."""
    artifact = next((a for a in _read_json(run / "artifacts.json", []) if a["structure_id"] == structure_id), None)
    if artifact is not None:
        return {"files": artifact["files"], "script": artifact["script"], "entry": None}
    m = _UNREPORTED.match(structure_id)
    if m:
        for manifest in (run / "construction" / m["attempt"]).glob("*/workspace/structures/manifest.json"):
            for entry in _read_json(manifest, []):
                if entry["name"] != m["name"]:
                    continue
                files = {"extxyz": str((manifest.parent / entry["path"]).relative_to(run))}
                for fmt, suffix in (("poscar", ".vasp"), ("metadata", ".json")):
                    if (manifest.parent / f"{entry['name']}{suffix}").exists():
                        files[fmt] = str((manifest.parent / f"{entry['name']}{suffix}").relative_to(run))
                script = manifest.parents[1] / entry["script"] if entry.get("script") and entry["script"] != "<inline>" else None
                return {"files": files, "script": str(script.relative_to(run)) if script and script.exists() else None,
                        "entry": entry}
    raise KeyError(f"no structure {structure_id!r} in run {run.name!r}")


def _plan_failures(run: Path) -> dict[str, str]:
    """The constructor agent's latest reason for each plan it reported as not built."""
    failures: dict[str, str] = {}
    for report in _read_json(run / "construction_reports.json", []):
        for f in report.get("failed", []):
            failures[f["plan_id"]] = f["reason"]
    return failures


def _unreported_structures(project: Project, run: Path, artifacts: list[dict], method: str) -> list[dict]:
    """Structures emitted in the construction workspaces that never became artifacts: the agent gave the
    plan up (a target or claim it could not meet) after building them. They are still worth looking at,
    downloading and varying, so the page shows them with the same tools. Their geometry checks are
    computed once and cached in runs/<run>/unreported.json."""
    registered = {(a["name"], f"r{a['revision']}_a{a['attempt']}") for a in artifacts}
    failures = _plan_failures(run)
    cache_path = run / "unreported.json"
    cache, dirty, cfg = _read_json(cache_path, {}), False, None
    out = []
    for manifest in sorted(run.glob("construction/r*_a*/*/workspace/structures/manifest.json")):
        attempt = manifest.parents[3].name
        for entry in _read_json(manifest, []):
            path = manifest.parent / entry["path"]
            if (entry["name"], attempt) in registered or not path.exists():
                continue
            sid = f"{entry['name']}__{attempt}"
            atoms = read(path)
            mtime = path.stat().st_mtime
            if sid not in cache or cache[sid].get("mtime") != mtime:
                cfg = cfg or _run_config(run)
                report = validate_structure(atoms, entry["name"], max_atoms=max(cfg.planning.max_atoms, len(atoms)),
                                            **cfg.validation.model_dump())
                cache[sid] = {"mtime": mtime, "validation": {"passed": report.passed, "checks": [c.model_dump() for c in report.checks]}}
                dirty = True
            source = _structure_source(run, sid)
            try:
                params = script_params(run / source["script"]) if source["script"] else []
            except (OSError, SyntaxError, ValueError):
                params = []
            downloads = {fmt: f"/api/projects/{project.id}/files/runs/{run.name}/{p}" for fmt, p in source["files"].items()}
            if source["script"]:
                downloads["script"] = f"/api/projects/{project.id}/files/runs/{run.name}/{source['script']}"
            out.append({"structure_id": sid, "name": entry["name"], "plan_id": entry["plan_id"], "attempt": attempt,
                        "unreported": True, "reason": failures.get(entry["plan_id"], ""), "notes": entry.get("notes", ""),
                        "formula": atoms.get_chemical_formula(), "n_atoms": len(atoms), "accepted": False, "current": None,
                        "info": structure_info(atoms), "estimate": compute_estimate(atoms, method), "plan": None,
                        "validation": cache[sid]["validation"], "params": params, "variants": [], "target_checks": [],
                        "deviations": [], "method": "", "has_script": bool(source["script"]), "downloads": downloads,
                        "image_url": None})
    if dirty:
        _write_json(cache_path, cache)
    return out


def _unreported_count(run: Path) -> int:
    registered = {(a["name"], f"r{a['revision']}_a{a['attempt']}") for a in _read_json(run / "artifacts.json", [])}
    return sum(1 for m in run.glob("construction/r*_a*/*/workspace/structures/manifest.json")
               for e in _read_json(m, []) if (e["name"], m.parents[3].name) not in registered)


def _variant_script(run: Path, structure_id: str) -> Path:
    source = _structure_source(run, structure_id)
    if not source["script"]:
        raise RuntimeError("this structure was built inline (python -c), so there is no script to vary")
    return run / source["script"]


def check_variant(project: Project, run_name: str, structure_id: str, values: dict) -> dict:
    """Validate a variant request before queueing it: the script exists and the values fit its parameters."""
    script = _variant_script(project.run_dir(run_name), structure_id)
    params = script_params(script)
    _, used, changed = rewrite_script(script.read_text(), params, values)
    return {"params": used, "changed": changed}


def run_variant(project: Project, run_name: str, structure_id: str, values: dict, say: Callable[[str], None]) -> dict:
    """Re-run a structure's build script with new parameter values (no agent involved), then validate,
    export and render what it emits, under runs/<run>/variants/<id>/."""
    run = project.run_dir(run_name)
    script = _variant_script(run, structure_id)
    params = script_params(script)
    src, used, changed = rewrite_script(script.read_text(), params, values)
    vid = f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:4]}"
    vdir = run / "variants" / vid
    vdir.mkdir(parents=True)
    (vdir / script.name).write_text(src)
    cfg = _run_config(run)
    say(f"Running {script.name} with " + (", ".join(f"{k} = {v}" for k, v in changed.items()) or "its original parameters")
        + f" (no agent: the saved script, sandbox {cfg.construction.sandbox})...")
    started = time.time()
    result = run_async(replay(vdir / script.name, vdir / "work", cfg.construction))
    structures = []
    for entry in _read_json(vdir / "work" / "structures" / "manifest.json", []):
        atoms = read(vdir / "work" / "structures" / entry["path"])
        # the atom limit was a planning choice; a variant is the user asking for a different size, so only
        # the physical checks apply (an oversized cell is reported, not failed)
        report = validate_structure(atoms, entry["name"], max_atoms=max(cfg.planning.max_atoms, len(atoms)),
                                    **cfg.validation.model_dump())
        files = export_structure(atoms, vdir / "structures", entry["name"], formats=("poscar", "cif", "extxyz"))
        image = vdir / "structures" / f"{entry['name']}.png"
        render_png(atoms, image)
        structures.append({"name": entry["name"], "formula": atoms.get_chemical_formula(), "n_atoms": len(atoms),
                           "over_budget": len(atoms) > cfg.planning.max_atoms, "info": structure_info(atoms),
                           "estimate": compute_estimate(atoms, cfg.campaign.simulation if cfg.campaign.simulation == "DFT" else "MD"),
                           "files": {k: str(Path(v).relative_to(run)) for k, v in files.items()},
                           "image": str(image.relative_to(run)),
                           "validation": {"passed": report.passed, "checks": [c.model_dump() for c in report.checks]}})
    error = None
    if result.exit_code != 0:
        error = f"the script exited with code {result.exit_code}"
    elif not structures:
        error = "the script ran but emitted no structure"
    record = {"id": vid, "structure_id": structure_id, "script": str((vdir / script.name).relative_to(run)),
              "params": used, "changed": changed, "created": _now(), "seconds": round(time.time() - started),
              "exit_code": result.exit_code, "stdout": (result.stdout or "")[-4000:], "stderr": (result.stderr or "")[-4000:],
              "structures": structures, "error": error}
    _write_json(vdir / "variant.json", record)
    say(f"Variant {'failed: ' + error if error else 'built'} in {record['seconds']} s"
        + (": " + "; ".join(f"{s['name']} {s['formula']} ({s['n_atoms']} atoms, geometry "
                            f"{'ok' if s['validation']['passed'] else 'FAILED'})" for s in structures) if structures else ""))
    if error:
        raise RuntimeError(error + (f": {record['stderr'].strip().splitlines()[-1]}" if record["stderr"].strip() else ""))
    return {"run": run_name, "variant": vid, "structures": len(structures)}


def _variants(project: Project, run: Path) -> list[dict]:
    folder = run / "variants"
    out = []
    for path in sorted(folder.glob("*/variant.json"), reverse=True) if folder.exists() else []:
        v = _read_json(path)
        for s in v["structures"]:
            s["downloads"] = {fmt: f"/api/projects/{project.id}/files/runs/{run.name}/{p}" for fmt, p in s["files"].items()}
            s["image_url"] = f"/api/projects/{project.id}/files/runs/{run.name}/{s['image']}"
        v["script_url"] = f"/api/projects/{project.id}/files/runs/{run.name}/{v['script']}"
        out.append(v)
    return out


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
            "agent": "ready" if spec else "none"})
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
            "n_accepted": len(result["accepted_structure_ids"]) if result else 0,
            "n_unreported": _unreported_count(run)}


def project_state(project: Project, active_job: Optional[str]) -> dict:
    """Everything the page shows for one project (the run details are fetched separately)."""
    graph = project.graph()
    runs = sorted((project.dir / "runs").glob("*/settings.json"), reverse=True) if (project.dir / "runs").exists() else []
    chars = project.characterizations()
    for c in chars:
        c["reports_detail"] = _reports_view(project.dir / "characterization" / c["key"])
    plans = [{k: v for k, v in r.items() if k != "plans"} | {"n_plans": len(r["plans"]["plans"])} for r in project.plans()]
    return {"project": project.meta, "active_job": active_job,
            "summary": _summary_view(project, graph) if graph else None,
            "available_techniques": available_techniques(graph) if graph else [],
            "characterizations": chars, "plans": plans,
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
    # the latest plan set's structures, not rebuilt since (None for runs made before result.json listed them)
    current = set(result["current_structure_ids"]) if result and "current_structure_ids" in result else None
    method = (settings.get("settings") or {}).get("method", "DFT")
    variants = _variants(project, run)
    structures = []
    for a in artifacts:
        atoms = read(run / (a["files"].get("extxyz") or a["files"]["poscar"]))
        rep = validation.get(a["structure_id"], {})
        try:
            params = script_params(run / a["script"])
        except (OSError, SyntaxError, ValueError):
            params = []
        structures.append({
            **a, "accepted": a["structure_id"] in accepted, "info": structure_info(atoms),
            "current": None if current is None else a["structure_id"] in current,
            "estimate": compute_estimate(atoms, method), "plan": plans.get(a["plan_id"]),
            "validation": {"passed": rep.get("passed"), "checks": rep.get("checks", [])},
            "params": params, "variants": [v for v in variants if v["structure_id"] == a["structure_id"]],
            "downloads": {**{fmt: f"/api/projects/{project.id}/files/runs/{name}/{p}" for fmt, p in a["files"].items()},
                          "script": f"/api/projects/{project.id}/files/runs/{name}/{a['script']}"},
            "image_url": f"/api/projects/{project.id}/files/runs/{name}/{a['image']}" if a.get("image") else None})
    unreported = _unreported_structures(project, run, artifacts, method)
    for s in unreported:
        s["variants"] = [v for v in variants if v["structure_id"] == s["structure_id"]]
    usage = _read_json(run / "token_log.json", {})
    total_in = sum(u.get("input", 0) for u in usage.values())
    total_out = sum(u.get("output", 0) for u in usage.values())
    plan_record = project.plan_record(settings["plan"]) if settings.get("plan") else None
    return {"name": name, "status": result["status"] if result else "unfinished", "result": result, "settings": settings,
            "structures": structures, "unreported": unreported,
            "failures": [{"plan_id": k, "reason": v} for k, v in _plan_failures(run).items()],
            "plans": list(plans.values()), "decision": history[-1] if history else None,
            "plan_record": {k: v for k, v in plan_record.items() if k != "plans"} if plan_record else None,
            "report": (run / "final_report.md").read_text() if (run / "final_report.md").exists() else "",
            "errors": result["errors"] if result else [],
            "usage": {"agents": usage, "input": total_in, "output": total_out,
                      "cost": estimate_cost(settings.get("model", ""), total_in, total_out)},
            "zip_url": f"/api/projects/{project.id}/runs/{name}/zip"}


OPEN_GAP = 5.0  # A of empty space along a cell vector that counts as vacuum (as the library's structure_kind)


def display_atoms(atoms):
    """The structure as the viewer should show it; the files are left as they are. Periodic wrapping
    folds atoms back into the cell, which splits a particle or slab sitting across the boundary and
    cuts molecules in half (the viewer then draws stray atoms and missing bonds). So: along every
    cell vector that holds vacuum the structure is shifted so the empty space sits at the boundary,
    each molecule (per-atom mol_id) is made whole with the minimum-image rule, and the cell is put
    in its standard orientation so the viewer's box matches the coordinates."""
    import numpy as np
    from paper2structure.tools.structure_export import slab_axis, structure_kind, wire_axis
    shown = atoms.copy()
    if shown.cell.rank < 3:
        return shown
    original = shown.positions.copy()
    kind = structure_kind(shown)  # the library's call on which cell vectors are open (hold vacuum)
    open_axes = {"nanocrystal": [0, 1, 2], "slab": [slab_axis(shown)], "wire": [k for k in range(3) if k != wire_axis(shown)]}.get(kind, [])
    frac = shown.get_scaled_positions(wrap=True)
    for axis in open_axes:
        order = np.sort(frac[:, axis])
        gaps = np.diff(np.append(order, order[0] + 1.0))  # the last gap wraps around
        i = int(np.argmax(gaps))
        start = (order[i] + gaps[i]) % 1.0  # where the occupied range begins
        frac[:, axis] = (frac[:, axis] - start) % 1.0 + gaps[i] / 2  # contiguous and centred
    shown.set_scaled_positions(frac)
    if shown.has("mol_id") and (shown.arrays["mol_id"] >= 0).any():
        # walk each molecule's bond graph, placing every atom next to the bonded atom it was reached
        # from (works for molecules wider than half the cell, where a single minimum image does not)
        from collections import deque
        from ase.neighborlist import natural_cutoffs, neighbor_list
        ids = shown.arrays["mol_id"]
        i_idx, j_idx, vecs = neighbor_list("ijD", shown, natural_cutoffs(shown, mult=1.2))
        shortest: dict[tuple[int, int], np.ndarray] = {}  # one bond per pair: its nearest image, not a periodic contact
        for i, j, d in zip(i_idx, j_idx, vecs):
            if ids[i] == ids[j] and ids[i] >= 0:
                key = (int(i), int(j))
                if key not in shortest or np.linalg.norm(d) < np.linalg.norm(shortest[key]):
                    shortest[key] = d
        bonds: dict[int, list[tuple[int, np.ndarray]]] = {}
        for (i, j), d in shortest.items():
            bonds.setdefault(i, []).append((j, d))
        extent = lambda pos: float((pos.max(axis=0) - pos.min(axis=0)).max())  # noqa: E731
        for mid in np.unique(ids):
            members = np.flatnonzero(ids == mid)
            if mid < 0 or len(members) < 2:
                continue
            anchor = int(members[0])
            placed, queue = {anchor}, deque([anchor])
            while queue:
                i = queue.popleft()
                for j, d in bonds.get(i, []):
                    if j not in placed:
                        shown.positions[j] = shown.positions[i] + d
                        placed.add(j)
                        queue.append(j)
            if extent(shown.positions[members]) > extent(original[members]) + 1e-6:
                # never worse than the file: keep its geometry, moved along with the anchor
                shown.positions[members] = original[members] - original[anchor] + shown.positions[anchor]
    shown.set_cell(shown.cell.cellpar(), scale_atoms=True)  # standard orientation: a along x, b in the xy plane
    return shown


def structure_model(project: Project, name: str, structure_id: str, variant: Optional[str] = None) -> dict:
    """The structure (of the run, or of one of its variants) as XYZ text for the 3Dmol.js viewer, with
    its cell parameters (the viewer draws the box itself), prepared by display_atoms."""
    run = project.run_dir(name)
    if variant:
        if not _SAFE_NAME.match(variant):
            raise KeyError(f"no variant {variant!r}")
        record = _read_json(run / "variants" / variant / "variant.json")
        entry = next((s for s in (record or {}).get("structures", []) if s["name"] == structure_id), None)
        if entry is None:
            raise KeyError(f"no structure {structure_id!r} in variant {variant!r}")
        files = entry["files"]
    else:
        files = _structure_source(run, structure_id)["files"]
    atoms = read(run / (files.get("extxyz") or files["poscar"]))
    shown = display_atoms(atoms)
    buf = io.StringIO()  # ASE's xyz writer wants text (its cif writer wants bytes)
    write(buf, shown, format="xyz")
    cell = None
    if shown.cell.rank == 3 and atoms.pbc.any():
        a, b, c, alpha, beta, gamma = (float(x) for x in shown.cell.cellpar())
        cell = {"a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma}
    return {"format": "xyz", "text": buf.getvalue(), "periodic": bool(atoms.pbc.any()), "cell": cell,
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
# Jobs: a pool of worker threads, one job per paper at a time, the page polls
# --------------------------------------------------------------------------------------

CURRENT_JOB: contextvars.ContextVar[Optional["Job"]] = contextvars.ContextVar("p2s_job", default=None)
JOB_RUNTIME: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Task]] = {}  # the running job's loop and task, for Stop


@dataclass
class Job:
    id: str
    project_id: str
    title: str
    kind: str  # summarize | characterize | generate | variant
    status: str = "queued"  # queued | running | done | failed | cancelled
    log: list[str] = field(default_factory=list)
    error: Optional[str] = None
    result: Any = None
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    cancel_requested: bool = False

    def to_dict(self, full: bool = True) -> dict:
        d = asdict(self)
        if not full:
            d["log"] = self.log[-1:]  # the list views only show the last line
        return d


def run_async(coro):
    """asyncio.run for a stage's coroutine, but on a loop the Stop button can reach: the job's task is
    cancelled from the server thread, which interrupts the agents at their next await and terminates a
    build script's subprocess."""
    job = CURRENT_JOB.get()
    if job is None:
        return asyncio.run(coro)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        task = loop.create_task(coro)
        JOB_RUNTIME[job.id] = (loop, task)
        if job.cancel_requested:
            task.cancel()
        return loop.run_until_complete(task)
    finally:
        JOB_RUNTIME.pop(job.id, None)
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
        finally:
            asyncio.set_event_loop(None)
            loop.close()


class _JobLog(logging.Handler):
    """One handler for every job: a record goes to the job in the ContextVar of the thread or task that
    logged it (asyncio tasks and pydantic-ai's worker threads inherit it), or nowhere."""

    def __init__(self):
        super().__init__(logging.INFO)

    def emit(self, record: logging.LogRecord) -> None:
        job = CURRENT_JOB.get()
        if job is None:
            return
        try:  # a failing display must never fail the run
            job.log.append(record.getMessage())
        except Exception:  # noqa: BLE001
            pass


class JobManager:
    """`workers` threads take jobs from one queue; a paper never has two jobs at once (they would write
    the same folder), different papers run in parallel."""

    def __init__(self, workers: int = WORKERS):
        self.jobs: dict[str, Job] = {}
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.RLock()
        logger = logging.getLogger("paper2structure")
        logger.setLevel(logging.INFO)
        logger.addHandler(_JobLog())
        for i in range(workers):
            threading.Thread(target=self._worker, name=f"p2s-job-{i}", daemon=True).start()

    def submit(self, project: Project, kind: str, fn: Callable[[Callable[[str], None]], Any]) -> Job:
        with self._lock:
            if self.active(project.id):
                raise RuntimeError("this paper already has a job running; wait for it to finish")
            job = Job(id=uuid.uuid4().hex[:12], project_id=project.id, title=project.meta["title"], kind=kind)
            self.jobs[job.id] = job
        self._queue.put((job, fn))
        return job

    def active(self, project_id: str) -> Optional[Job]:
        with self._lock:
            return next((j for j in self.jobs.values()
                         if j.project_id == project_id and j.status in ("queued", "running")), None)

    def recent(self, limit: int = 50) -> list[Job]:
        with self._lock:
            return sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)[:limit]

    def cancel(self, job_id: str) -> Job:
        """Stop a job: a queued one is dropped when a worker reaches it; a running one has its task
        cancelled on its own loop. Files the stage had already written stay."""
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError("no such job")
        if job.status not in ("queued", "running"):
            raise RuntimeError(f"the job is already {job.status}")
        job.cancel_requested = True
        job.log.append("Stop requested: interrupting at the next step...")
        runtime = JOB_RUNTIME.get(job_id)
        if runtime:
            loop, task = runtime
            loop.call_soon_threadsafe(task.cancel)
        return job

    def _worker(self) -> None:
        while True:
            job, fn = self._queue.get()
            if job.cancel_requested:  # stopped while still queued
                job.status, job.finished, job.error = "cancelled", time.time(), "stopped before it started"
                continue
            CURRENT_JOB.set(job)
            job.status, job.started = "running", time.time()
            try:
                job.result = fn(job.log.append)
                job.status = "done"
            except asyncio.CancelledError:
                job.error = "stopped by the user"
                job.log.append("Stopped.")
                job.status = "cancelled"
            except Exception as e:  # noqa: BLE001 - the page shows any failure
                if job.cancel_requested:  # a stage turned the cancellation into its own error
                    job.error, job.status = "stopped by the user", "cancelled"
                else:
                    job.error = f"{type(e).__name__}: {e}"
                    job.log.append(f"ERROR {job.error}")
                    job.status = "failed"
                    log.error("job %s (%s, %s) failed:\n%s", job.id, job.kind, job.project_id, traceback.format_exc())
            finally:
                job.finished = time.time()
                CURRENT_JOB.set(None)
