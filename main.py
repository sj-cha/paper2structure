"""Run the paper-to-structure workflow on one paper.

    python main.py --pdf examples/pdfs/acs.chemmater.6b00847.pdf
    python main.py --pdf paper.pdf --techniques XRD TEM          # rerun only these techniques
    python main.py --pdf paper.pdf --model dry-run               # free plumbing test

The summarizer runs once per paper (its output is reused on later runs unless --resummarize);
the orchestrator (characterization, later hypothesis/constructor/revision) runs every time.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")

from paper2structure.agents.summarizer.graph import summarize_paper_sync
from paper2structure.config import DEFAULT_MODEL, get_model
from paper2structure.graph import run_orchestrator_sync


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", required=True, help="path to the paper PDF")
    p.add_argument("--output_dir", default=None, help="default: outputs/<pdf name>")
    p.add_argument("--model", default=os.getenv("P2S_MODEL") or DEFAULT_MODEL,
                   help="pydantic-ai model name for all agents, or 'dry-run'")
    p.add_argument("--techniques", nargs="+", default=None,
                   help="only (re)run these techniques, e.g. XRD TEM; default: every technique found")
    p.add_argument("--include_stubs", action="store_true",
                   help="also run techniques whose spec is not filled in yet")
    p.add_argument("--resummarize", action="store_true",
                   help="run the summarizer again even if paper_graph.json exists")
    p.add_argument("--no_figures", action="store_true", help="summarizer skips figure cataloguing")
    p.add_argument("--max_figures", type=int, default=30, help="max figure regions cut from the PDF")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    if args.model != "dry-run" and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: needs OPENAI_API_KEY in env or .env (or use --model dry-run).", file=sys.stderr)
        return 2
    if not Path(args.pdf).exists():
        print(f"ERROR: {args.pdf} not found.", file=sys.stderr)
        return 2

    out = Path(args.output_dir) if args.output_dir else Path("outputs") / Path(args.pdf).stem
    out.mkdir(parents=True, exist_ok=True)
    model = get_model(args.model)

    # Snapshot the resolved CLI run-level values
    with open(out / "config.json", "w") as fh:
        json.dump({"cli": vars(args) | {"output_dir": str(out)}}, fh, indent=2)

    print(f"[paper2structure] pdf={args.pdf} out={out} model={args.model} "
          f"techniques={args.techniques or 'all found'} include_stubs={args.include_stubs}", flush=True)

    if args.resummarize or not (out / "paper_graph.json").exists():
        print("[paper2structure] summarizing the paper...", flush=True)
        summarize_paper_sync(args.pdf, out, model, analyze_figures=not args.no_figures,
                             max_figures=args.max_figures)
    else:
        print(f"[paper2structure] reusing {out / 'paper_graph.json'} (pass --resummarize to redo)", flush=True)

    state = run_orchestrator_sync(out, args.techniques, args.include_stubs, model)

    paper = state.paper
    names = {s.system_id: s.name for s in paper.systems}
    print("\n================ PAPER2STRUCTURE SUMMARY ================", flush=True)
    print(f"paper: {paper.metadata.title}", flush=True)
    print(f"systems={len(paper.systems)}  evidence={len(paper.evidence)}  reports={len(state.reports)}", flush=True)
    for technique, report in sorted(state.reports.items()):
        print(f"  {technique:10s} {len(report.observations):3d} observations  "
              f"{len(report.structural_constraints):3d} constraints", flush=True)
    print("structural constraints by system:", flush=True)
    for system_id, name in names.items():
        constraints = [(t, c) for t, r in sorted(state.reports.items())
                       for c in r.structural_constraints if system_id in c.system_ids]
        print(f"  {system_id} {name}: {len(constraints)}", flush=True)
        for technique, c in constraints[:3]:
            print(f"      [{technique}/{c.confidence}] {c.property}: {c.value[:80]}", flush=True)
    for error in state.errors:
        print(f"  ! {error}", flush=True)
    print(f"\nartifacts in {out}/ (paper_graph.json, summary.md, figures/, characterization/)", flush=True)
    return 1 if state.errors else 0


if __name__ == "__main__":
    sys.exit(main())
