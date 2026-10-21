"""Command line entry point.

    python -m paper2structure.summarizer.cli paper.pdf                # uses P2S_MODEL / default
    python -m paper2structure.summarizer.cli paper.pdf --model dry-run
    python -m paper2structure.summarizer.cli --show-graph             # print the workflow diagram
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .graph import summarize_paper_sync, summarizer_graph
from .models import get_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a materials paper into a knowledge graph.")
    parser.add_argument("pdf", nargs="?", help="Path to the paper PDF")
    parser.add_argument("--out", default=None, help="Output folder (default: outputs/<pdf name>)")
    parser.add_argument("--model", default=None, help="e.g. openai:gpt-5.4-mini, or dry-run")
    parser.add_argument("--no-figures", action="store_true", help="Skip the (more expensive) figure analysis")
    parser.add_argument("--max-figures", type=int, default=30)
    parser.add_argument("--show-graph", action="store_true", help="Print the workflow as a Mermaid diagram and exit")
    args = parser.parse_args()

    if args.show_graph:
        print(summarizer_graph.render(title="paper_summarizer"))
        return
    if not args.pdf:
        parser.error("pdf is required")

    out = Path(args.out) if args.out else Path("outputs") / Path(args.pdf).stem
    graph = summarize_paper_sync(args.pdf, out, get_model(args.model), analyze_figures=not args.no_figures, max_figures=args.max_figures)
    print(f"\nOpen {out / 'summary.md'} for the readable summary.")
    for tech in sorted({e.technique for e in graph.evidence}):
        print(f"  {tech}: {len(graph.evidence_for(technique=tech))} evidence items")


if __name__ == "__main__":
    main()
