"""Run the paper-to-structure workflow on one paper in a single graph run:

    PDF -> summary (Agent 1) -> characterization (Agent 2) -> plans (Agent 3)
        -> structures + the scripts that built them (Agent 4) -> review and report (Agent 5)

    python main.py --pdf pdf/acs.chemmater.6b00847.pdf --random_seed 0
    python main.py --pdf paper.pdf --techniques XRD TEM --random_seed 0     # characterize only these
    python main.py --paper_graph runs/x/paper/paper_graph.json --random_seed 0   # reuse a summary
    python main.py --config config/pt111_co.yml --random_seed 0             # synthetic fixture inputs

A campaign YAML (--config) holds every setting; the flags override its run-level values.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from paper2structure.agents.constructor.workspace import sandbox_available  # noqa: E402
from paper2structure.config import load_config  # noqa: E402
from paper2structure.graph.workflow import build_deps, run_workflow  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yml")
    p.add_argument("--pdf", default=None, help="the paper (sets inputs.pdf)")
    p.add_argument("--paper_graph", default=None,
                   help="paper_graph.json of an earlier run: skips the summarizer (sets inputs.paper_graph)")
    p.add_argument("--characterization", default=None,
                   help="folder of saved characterization reports: skips Agent 2 (sets inputs.characterization)")
    p.add_argument("--output_dir", default=None,
                   help="a new or empty folder; default: runs/<pdf or paper name>_<date-time>")
    p.add_argument("--random_seed", type=int, required=True,
                   help="seeds conformer embedding, adsorbate placement and solvent packing")
    p.add_argument("--model", default=None, help="every agent's model, e.g. openai:gpt-5.6-luna (sets campaign.model)")
    p.add_argument("--techniques", nargs="+", default=None,
                   help="characterize only these techniques, e.g. XRD TEM; default: every technique found")
    p.add_argument("--no_figures", action="store_true", help="summarizer skips figure cataloguing")
    p.add_argument("--max_figures", type=int, default=None, help="max figure regions cut from the PDF")
    p.add_argument("--max_revisions", type=int, default=None,
                   help="loops the reviewer may take back through evidence -> planning -> construction "
                        "(sets budgets.max_revisions)")
    p.add_argument("--max_rebuilds", type=int, default=None,
                   help="rounds the reviewer may send structures back to construction under the same plans "
                        "(sets budgets.max_rebuilds)")
    p.add_argument("--max_repairs", type=int, default=None,
                   help="construction retries for structures failing deterministic validation (sets budgets.max_repairs)")
    p.add_argument("--gpu", type=int, default=None,
                   help="GPU index for CUDA-accelerated neighbor lists, in the workflow and in the agent's "
                        "sandboxed scripts (sets CUDA_VISIBLE_DEVICES and construction.gpu)")
    p.add_argument("--skip_preflight", action="store_true", help="skip the per-agent connectivity checks")
    return p


def _resolve(path: str | None, config_path: str) -> str | None:
    """Input paths are relative to the working directory, falling back to the config's folder."""
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute() or p.exists():
        return str(p.resolve())
    alt = Path(config_path).resolve().parent / p
    return str(alt) if alt.exists() else str(p)


def main() -> int:
    args = _build_parser().parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: needs OPENAI_API_KEY in env or .env.", file=sys.stderr)
        return 2
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = load_config(args.config)
    inputs = cfg.inputs
    if args.pdf and args.characterization:
        print("ERROR: --characterization needs the summary its reports were made from: give --paper_graph "
              "instead of --pdf.", file=sys.stderr)
        return 2
    if args.pdf:
        if inputs.characterization:
            print(f"[paper2structure] --pdf: ignoring the config's inputs.characterization ({inputs.characterization}); "
                  "its reports belong to another summary", flush=True)
        inputs.pdf, inputs.paper_graph, inputs.characterization = args.pdf, None, None
    if args.paper_graph:
        inputs.paper_graph, inputs.pdf = args.paper_graph, None
    if args.characterization:
        inputs.characterization = args.characterization
    for field in ("pdf", "paper_graph", "characterization"):
        setattr(inputs, field, _resolve(getattr(inputs, field), args.config))
    if not (inputs.pdf or inputs.paper_graph):
        print("ERROR: give --pdf (or --paper_graph), or set inputs.pdf in the config.", file=sys.stderr)
        return 2
    for field in ("pdf", "paper_graph", "characterization"):
        if getattr(inputs, field) and not Path(getattr(inputs, field)).exists():
            print(f"ERROR: inputs.{field} {getattr(inputs, field)} not found.", file=sys.stderr)
            return 2
    if args.model:
        cfg.campaign.model = args.model
    if args.techniques:
        cfg.characterization.techniques = args.techniques
    if args.no_figures:
        cfg.summarizer.analyze_figures = False
    if args.max_figures is not None:
        cfg.summarizer.max_figures = args.max_figures
    if args.gpu is not None:
        cfg.construction.gpu = True  # the agent's sandboxed scripts may use it too
    budgets = {k: getattr(args, k) for k in ("max_revisions", "max_repairs", "max_rebuilds") if getattr(args, k) is not None}
    if budgets:
        try:
            cfg.budgets = type(cfg.budgets).model_validate({**cfg.budgets.model_dump(), **budgets})
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

    if inputs.pdf and inputs.characterization:
        print("ERROR: inputs.characterization needs the summary its reports were made from: set inputs.paper_graph "
              "instead of inputs.pdf.", file=sys.stderr)
        return 2

    if args.output_dir is None:
        stamp = f"{dt.datetime.now():%Y%m%d-%H%M%S}"
        if inputs.pdf:
            args.output_dir = str(Path("runs") / f"{Path(inputs.pdf).stem}_{stamp}")
        else:  # a summary saved by an earlier run sits in <run>/paper/
            src = Path(inputs.paper_graph).parent
            args.output_dir = str(Path("runs") / f"{src.parent.name if src.name == 'paper' else src.name}_replan_{stamp}")
    elif Path(args.output_dir).is_dir() and any(Path(args.output_dir).iterdir()):
        print(f"ERROR: --output_dir {args.output_dir} is not empty: a run needs a new or empty folder.", file=sys.stderr)
        return 2
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(args.output_dir).resolve() / ".mpl"))

    # Snapshot the config + resolved CLI run-level values
    snapshot = {"config": cfg.model_dump(mode="json"), "cli": vars(args)}
    with open(os.path.join(args.output_dir, "config.json"), "w") as fh:
        json.dump(snapshot, fh, indent=2)

    deps = build_deps(cfg, args.output_dir, seed=args.random_seed)
    cc, b = cfg.construction, cfg.budgets
    print(f"[paper2structure] paper={inputs.pdf or inputs.paper_graph} "
          f"characterization={inputs.characterization or 'run Agent 2'} out={args.output_dir} "
          f"model={cfg.campaign.model or 'default'} seed={args.random_seed} "
          f"techniques={cfg.characterization.techniques or 'all found'} "
          f"budget(revisions={b.max_revisions}, rebuilds={b.max_rebuilds}, repairs={b.max_repairs}, "
          f"construction_requests={cc.max_requests}) sandbox={cc.sandbox}", flush=True)

    if cc.sandbox == "bubblewrap":
        ok, err = sandbox_available()
        if not ok:
            print(f"ERROR: bubblewrap sandbox unavailable: {err} (or set construction.sandbox: none)", file=sys.stderr)
            return 4
    else:
        print("[paper2structure] WARNING: construction.sandbox is 'none': the agent's commands run unsandboxed",
              flush=True)

    if not args.skip_preflight:
        for name, agent in (("planner", deps.planner), ("constructor", deps.constructor),
                            ("reviser", deps.reviser)):
            ok, err = agent.preflight()
            if not ok:
                print(f"ERROR: {name} preflight failed: {err}", file=sys.stderr)
                return 3
            print(f"[paper2structure] {name} preflight OK.", flush=True)

    result = asyncio.run(run_workflow(deps))

    print("\n================ PAPER2STRUCTURE SUMMARY ================", flush=True)
    print(f"status: {result.status}   revisions: {result.revisions}", flush=True)
    shown = [*result.current_structure_ids,
             *(i for i in result.accepted_structure_ids if i not in result.current_structure_ids)]
    for sid in shown:
        mark = "*" if sid in result.accepted_structure_ids else " "
        print(f" {mark} {sid}: {', '.join(result.structure_files[sid].values())}", flush=True)
    superseded = [i for i in result.structure_files if i not in shown]
    if superseded:
        print(f"   superseded (in the report and structures/): {', '.join(superseded)}", flush=True)
    for e in result.errors:
        print(f"error: {e}", flush=True)
    print(f"\nreport: {os.path.join(result.run_dir, result.report)} (* = accepted)", flush=True)
    return 0 if result.status != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
