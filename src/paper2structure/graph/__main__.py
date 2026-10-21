"""Command line entry point: ``python -m paper2structure.graph <summarizer output folder>``."""

from __future__ import annotations

import argparse

from ..config import get_model
from . import orchestrator_graph, run_orchestrator_sync


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the orchestrator on a summarized paper.")
    parser.add_argument("out_dir", nargs="?", help="Summarizer output folder (contains paper_graph.json)")
    parser.add_argument("--techniques", nargs="+", default=None, help="Only (re)run these techniques, e.g. XRD TEM")
    parser.add_argument("--include-stubs", action="store_true", help="Also run techniques whose spec is a stub")
    parser.add_argument("--model", default=None, help="e.g. openai:gpt-5.6-luna (default), or dry-run")
    parser.add_argument("--show-graph", action="store_true", help="Print the workflow as a Mermaid diagram and exit")
    args = parser.parse_args()

    if args.show_graph:
        print(orchestrator_graph.render(title="orchestrator"))
        return
    if not args.out_dir:
        parser.error("out_dir is required")

    state = run_orchestrator_sync(args.out_dir, args.techniques, args.include_stubs, get_model(args.model))
    print(f"\nReports in {state.characterization_dir}:")
    for technique, report in sorted(state.reports.items()):
        print(f"  {technique}: {len(report.structural_constraints)} structural constraints")
    for error in state.errors:
        print(f"  ! {error}")


if __name__ == "__main__":
    main()
