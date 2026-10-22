"""Assemble the PaperGraph and write all output files.

Output folder layout (one folder per paper)::

    <out_dir>/
      paper_graph.json              # everything: metadata, systems, evidence (PaperGraph)
      summary.md                    # human-readable report
      graph.dot                     # graph picture source (Graphviz): paper -> system -> technique
      figures/p1_f1.png ...         # cropped figure images
"""

from __future__ import annotations

import re
import shutil
from collections import Counter
from pathlib import Path

from .schemas import Evidence, MaterialSystem, PaperGraph, PaperMetadata


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_") or "other"


def build_graph(source_pdf: str, model: str, metadata: PaperMetadata, systems: list[MaterialSystem], evidence: list[Evidence]) -> PaperGraph:
    return PaperGraph(source_pdf=source_pdf, model=model, metadata=metadata, systems=systems, evidence=evidence)


def to_dot(graph: PaperGraph) -> str:
    """Graphviz text: paper -> system -> technique, labelled with the number of evidence items
    (evidence nodes are left out to keep the picture readable)."""
    def node(id_: str, label: str, color: str) -> str:
        label = label if len(label) < 60 else label[:57] + "..."
        return f'  "{id_}" [label="{label.replace(chr(34), chr(39))}", fillcolor="{color}"];'

    lines = ["digraph G {", "  rankdir=LR; node [style=filled, shape=box, fontname=Helvetica];",
             node("paper", graph.metadata.title, "#f4d35e")]
    lines += [node(s.system_id, s.name, "#8ecae6") for s in graph.systems]
    lines += [node(f"tech:{_slug(t)}", t, "#90be6d") for t in dict.fromkeys(e.technique for e in graph.evidence)]
    lines += [f'  "paper" -> "{s.system_id}";' for s in graph.systems]
    counts = Counter((sid, e.technique) for e in graph.evidence for sid in e.system_ids)
    lines += [f'  "{sid}" -> "tech:{_slug(t)}" [label="{n}"];' for (sid, t), n in counts.items()]
    lines.append("}")
    return "\n".join(lines)


def to_markdown(graph: PaperGraph) -> str:
    m = graph.metadata
    out = [f"# {m.title}", "", f"**Authors:** {', '.join(m.authors) or '-'}  ", f"**DOI:** {m.doi or '-'}  ", f"**Model:** {graph.model}", "", m.summary, ""]
    out += [f"## Material systems ({len(graph.systems)})", "", "| ID | Name | Variables | Space group | Lattice parameters |", "|---|---|---|---|---|"]
    for s in graph.systems:
        lat = ", ".join(f"{x.quantity}={x.value}{' ' + x.unit if x.unit else ''}" for x in s.structure.lattice_parameters) or "-"
        vars_ = ", ".join(f"{k}={v}" for k, v in s.composition_variables.items()) or "-"
        out.append(f"| {s.system_id} | {s.name} | {vars_} | {s.structure.space_group or '-'} | {lat} |")
    out.append("")
    for tech in dict.fromkeys(e.technique for e in graph.evidence):
        out += [f"## {tech}", ""]
        for ev in (e for e in graph.evidence if e.technique == tech):
            out.append(f"- **[{ev.evidence_id}] {', '.join(ev.system_ids)}** ({ev.source_ref}): {ev.summary}")
            for x in ev.measurements:
                approx = " (read from plot)" if x.approximate else ""
                out.append(f"  - {x.quantity}: {x.value}{' ' + x.unit if x.unit else ''}{approx}")
        out.append("")
    return "\n".join(out)


def save_outputs(graph: PaperGraph, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paper_graph.json").write_text(graph.model_dump_json(indent=2))
    (out_dir / "summary.md").write_text(to_markdown(graph))
    (out_dir / "graph.dot").write_text(to_dot(graph))
    shutil.rmtree(out_dir / "evidence_by_technique", ignore_errors=True)  # written by older versions
    return out_dir / "paper_graph.json"


def load_graph(path: str | Path) -> PaperGraph:
    """Load a saved ``paper_graph.json`` (e.g. from another agent)."""
    return PaperGraph.model_validate_json(Path(path).read_text())
