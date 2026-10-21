"""Build the knowledge graph and write all output files.

Output folder layout (one folder per paper)::

    <out_dir>/
      paper_graph.json              # everything: systems, evidence, nodes, edges (PaperGraph)
      summary.md                    # human-readable report
      graph.dot                     # graph picture source (Graphviz)
      figures/p1_f1.png ...         # cropped figure images
      evidence_by_technique/
        XRD.json, TEM.json, ...     # one "packet" per technique for a specialist agent

Graph shape::

    paper --has_system--> system --has_evidence--> evidence --uses_technique--> technique
                                                   evidence --from_figure-----> figure
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .schemas import Evidence, GraphEdge, GraphNode, MaterialSystem, PaperGraph, PaperMetadata


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_") or "other"


def build_graph(source_pdf: str, model: str, metadata: PaperMetadata, systems: list[MaterialSystem], evidence: list[Evidence]) -> PaperGraph:
    nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []

    nodes["paper"] = GraphNode(id="paper", type="paper", label=metadata.title, data=metadata.model_dump())
    for s in systems:
        nodes[s.system_id] = GraphNode(id=s.system_id, type="system", label=s.name, data=s.model_dump())
        edges.append(GraphEdge(source="paper", target=s.system_id, relation="has_system"))

    for e in evidence:
        nodes[e.evidence_id] = GraphNode(id=e.evidence_id, type="evidence", label=f"{e.technique} ({e.source_ref})", data=e.model_dump())
        for sid in e.system_ids:
            edges.append(GraphEdge(source=sid, target=e.evidence_id, relation="has_evidence"))
        tech_id = f"tech:{_slug(e.technique)}"
        nodes.setdefault(tech_id, GraphNode(id=tech_id, type="technique", label=e.technique, data={"category": e.technique_category}))
        edges.append(GraphEdge(source=e.evidence_id, target=tech_id, relation="uses_technique"))
        if e.image_path:
            fig_id = f"fig:{Path(e.image_path).stem}"
            nodes.setdefault(fig_id, GraphNode(id=fig_id, type="figure", label=e.source_ref.split(" panel")[0], data={"image_path": e.image_path}))
            edges.append(GraphEdge(source=e.evidence_id, target=fig_id, relation="from_figure"))

    return PaperGraph(source_pdf=source_pdf, model=model, metadata=metadata, systems=systems, evidence=evidence, nodes=list(nodes.values()), edges=edges)


def to_dot(graph: PaperGraph) -> str:
    """Graphviz text. Evidence nodes are hidden to keep the picture readable: system -> technique."""
    colors = {"paper": "#f4d35e", "system": "#8ecae6", "technique": "#90be6d"}
    lines = ["digraph G {", "  rankdir=LR; node [style=filled, shape=box, fontname=Helvetica];"]
    for n in graph.nodes:
        if n.type in colors:
            label = n.label if len(n.label) < 60 else n.label[:57] + "..."
            lines.append(f'  "{n.id}" [label="{label.replace(chr(34), chr(39))}", fillcolor="{colors[n.type]}"];')
    seen = set()
    for e in graph.edges:
        if e.relation == "has_system":
            lines.append(f'  "{e.source}" -> "{e.target}";')
    for ev in graph.evidence:
        for sid in ev.system_ids:
            key = (sid, f"tech:{_slug(ev.technique)}")
            if key not in seen:
                seen.add(key)
                count = sum(1 for x in graph.evidence if sid in x.system_ids and x.technique == ev.technique)
                lines.append(f'  "{key[0]}" -> "{key[1]}" [label="{count}"];')
    lines.append("}")
    return "\n".join(lines)


def technique_packets(graph: PaperGraph) -> dict[str, dict]:
    """Group evidence by technique, with the systems it concerns, ready for a specialist agent."""
    packets: dict[str, dict] = {}
    systems_by_id = {s.system_id: s for s in graph.systems}
    for ev in graph.evidence:
        p = packets.setdefault(
            ev.technique,
            {"technique": ev.technique, "category": ev.technique_category, "paper": graph.metadata.model_dump(), "systems": {}, "evidence": []},
        )
        p["evidence"].append(ev.model_dump())
        for sid in ev.system_ids:
            if sid in systems_by_id:
                p["systems"][sid] = systems_by_id[sid].model_dump()
    return packets


def to_markdown(graph: PaperGraph) -> str:
    m = graph.metadata
    out = [f"# {m.title}", "", f"**Authors:** {', '.join(m.authors) or '-'}  ", f"**DOI:** {m.doi or '-'}  ", f"**Model:** {graph.model}", "", m.summary, ""]
    out += [f"## Material systems ({len(graph.systems)})", "", "| ID | Name | Variables | Space group | Lattice parameters |", "|---|---|---|---|---|"]
    for s in graph.systems:
        lat = ", ".join(f"{x.quantity}={x.value}{' ' + x.unit if x.unit else ''}" for x in s.structure.lattice_parameters) or "-"
        vars_ = ", ".join(f"{k}={v}" for k, v in s.composition_variables.items()) or "-"
        out.append(f"| {s.system_id} | {s.name} | {vars_} | {s.structure.space_group or '-'} | {lat} |")
    out.append("")
    for tech, packet in technique_packets(graph).items():
        out += [f"## {tech}", ""]
        for ev in packet["evidence"]:
            out.append(f"- **[{ev['evidence_id']}] {', '.join(ev['system_ids'])}** ({ev['source_ref']}): {ev['summary']}")
            for x in ev["measurements"]:
                approx = " (read from plot)" if x["approximate"] else ""
                out.append(f"  - {x['quantity']}: {x['value']}{' ' + x['unit'] if x['unit'] else ''}{approx}")
        out.append("")
    return "\n".join(out)


def save_outputs(graph: PaperGraph, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    (out_dir / "evidence_by_technique").mkdir(parents=True, exist_ok=True)
    (out_dir / "paper_graph.json").write_text(graph.model_dump_json(indent=2))
    (out_dir / "summary.md").write_text(to_markdown(graph))
    (out_dir / "graph.dot").write_text(to_dot(graph))
    for tech, packet in technique_packets(graph).items():
        (out_dir / "evidence_by_technique" / f"{_slug(tech)}.json").write_text(json.dumps(packet, indent=2))
    return out_dir / "paper_graph.json"


def load_graph(path: str | Path) -> PaperGraph:
    """Load a saved ``paper_graph.json`` (e.g. from another agent)."""
    return PaperGraph.model_validate_json(Path(path).read_text())
