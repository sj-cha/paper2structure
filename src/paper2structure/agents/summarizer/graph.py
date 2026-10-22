"""The summarizer workflow, written as a pydantic-graph.

    ReadPDF -> FindSystems -> ExtractTextEvidence (repeats once per system)
            -> AnalyzeFigure (repeats once per figure) -> BuildGraph -> End

Each step is a class that inherits from ``BaseNode``. Its ``run`` method does the work and
*returns the next step*. The return type hint (e.g. ``-> FindSystems``) is how pydantic-graph
learns the edges of the graph, so the hints are not just documentation: they are required.

``ctx.state`` (``SummarizerState``) is the shared notebook every step reads and writes.
``ctx.deps`` (``SummarizerDeps``) holds fixed settings: which LLM to use, a progress callback,
and the token usage every LLM call adds to.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai import BinaryContent
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage
from pydantic_graph import BaseNode, End, GraphBuilder, GraphRunContext

from .agents import AgentDeps, figure_analyst, system_finder, text_extractor
from .knowledge_graph import build_graph, save_outputs
from .pdf_reader import PaperDocument, read_pdf
from .schemas import Evidence, MaterialSystem, PaperGraph, PaperMetadata


@dataclass
class SummarizerState:
    pdf_path: Path
    out_dir: Path
    document: PaperDocument | None = None
    metadata: PaperMetadata | None = None
    systems: list[MaterialSystem] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def new_evidence_id(self) -> str:
        return f"E{len(self.evidence) + 1:03d}"

    def systems_text(self) -> str:
        return "\n".join(f"- {s.system_id}: {s.name} {s.composition_variables or ''} ({s.form or 'form not stated'})" for s in self.systems)


@dataclass
class SummarizerDeps:
    model: Model | str
    on_progress: Callable[[str], None] = print
    analyze_figures: bool = True
    max_figures: int = 30
    usage: RunUsage = field(default_factory=RunUsage)

    @property
    def model_name(self) -> str:
        return self.model if isinstance(self.model, str) else self.model.model_name


Ctx = GraphRunContext[SummarizerState, SummarizerDeps]


# --------------------------------------------------------------------------------------
# Steps (nodes)
# --------------------------------------------------------------------------------------


@dataclass
class ReadPDF(BaseNode[SummarizerState, SummarizerDeps, PaperGraph]):
    """Extract text and figure images from the PDF. No LLM."""

    async def run(self, ctx: Ctx) -> FindSystems:
        ctx.deps.on_progress("Reading the PDF and cutting out figures...")
        doc = read_pdf(ctx.state.pdf_path, ctx.state.out_dir / "figures", max_figures=ctx.deps.max_figures)
        ctx.state.document = doc
        n_tables = sum(f.kind == "table" for f in doc.figures)
        ctx.deps.on_progress(f"Found {len(doc.pages)} pages, {len(doc.figures) - n_tables} figure regions and "
                             f"{n_tables} tables.")
        return FindSystems()


@dataclass
class FindSystems(BaseNode[SummarizerState, SummarizerDeps, PaperGraph]):
    """LLM call 1: how many material systems were evaluated?"""

    async def run(self, ctx: Ctx) -> ExtractTextEvidence | BuildGraph:
        ctx.deps.on_progress("Identifying the material systems studied...")
        result = await system_finder.run(ctx.state.document.full_text, model=ctx.deps.model, deps=AgentDeps(),
                                         usage=ctx.deps.usage)
        ctx.state.metadata = result.output.metadata
        ctx.state.systems = [
            MaterialSystem(system_id=f"S{i}", **c.model_dump(exclude={"why_distinct"}))
            for i, c in enumerate(result.output.systems, start=1)
        ]
        ctx.deps.on_progress(f"Found {len(ctx.state.systems)} material systems:\n{ctx.state.systems_text()}")
        if not ctx.state.systems:
            ctx.state.errors.append("No material systems found.")
            return BuildGraph()
        return ExtractTextEvidence(index=0)


@dataclass
class ExtractTextEvidence(BaseNode[SummarizerState, SummarizerDeps, PaperGraph]):
    """LLM call 2 (once per system): synthesis, structure and per-technique evidence from the text."""

    index: int = 0

    async def run(self, ctx: Ctx) -> ExtractTextEvidence | AnalyzeFigure | BuildGraph:
        state = ctx.state
        system = state.systems[self.index]
        ctx.deps.on_progress(f"Reading text evidence for {system.system_id} ({system.name}) [{self.index + 1}/{len(state.systems)}]...")
        prompt = (
            f"Material systems in this paper:\n{state.systems_text()}\n\n"
            f"TARGET SYSTEM: {system.system_id} ({system.name})\n\n"
            f"PAPER TEXT:\n{state.document.full_text}"
        )
        try:
            result = await text_extractor.run(prompt, model=ctx.deps.model, deps=AgentDeps({s.system_id for s in state.systems}),
                                              usage=ctx.deps.usage)
            out = result.output
            system.synthesis = out.synthesis
            system.structure = out.structure
            for item in out.evidence:
                state.evidence.append(
                    Evidence(
                        evidence_id=state.new_evidence_id(),
                        source_type="text",
                        source_ref=f"page {item.page}" if item.page else "text",
                        **item.model_dump(exclude={"page"}),
                    )
                )
        except Exception as e:  # keep going: one failed system should not lose the whole paper
            state.errors.append(f"Text extraction failed for {system.system_id}: {e}")
            ctx.deps.on_progress(f"  ! {state.errors[-1]}")

        # Where to go next? This is the "loop": come back to this same step for the next system.
        if self.index + 1 < len(state.systems):
            return ExtractTextEvidence(index=self.index + 1)
        if ctx.deps.analyze_figures and state.document.figures:
            return AnalyzeFigure(index=0)
        return BuildGraph()


@dataclass
class AnalyzeFigure(BaseNode[SummarizerState, SummarizerDeps, PaperGraph]):
    """LLM call 3 (once per figure): send the image to a vision model."""

    index: int = 0

    async def run(self, ctx: Ctx) -> AnalyzeFigure | BuildGraph:
        state = ctx.state
        figures = state.document.figures
        fig = figures[self.index]
        ctx.deps.on_progress(f"Looking at figure {fig.figure_id} (page {fig.page}) [{self.index + 1}/{len(figures)}]...")
        prompt = [
            f"Material systems in this paper:\n{state.systems_text()}\n\n"
            f"Figure id: {fig.figure_id}" + (" (this image is a TABLE)" if fig.kind == "table" else "") + "\n"
            f"Most likely caption for this image: {fig.caption or '(none found)'}\n"
            f"All captions on page {fig.page}: {fig.page_captions}",
            BinaryContent(data=fig.image_path.read_bytes(), media_type="image/png"),
        ]
        try:
            result = await figure_analyst.run(prompt, model=ctx.deps.model, deps=AgentDeps({s.system_id for s in state.systems}),
                                              usage=ctx.deps.usage)
            out = result.output
            for panel in out.panels:
                ref = f"{out.figure_label}" + (f" panel {panel.panel_label}" if panel.panel_label else "") + f" (page {fig.page})"
                state.evidence.append(
                    Evidence(
                        evidence_id=state.new_evidence_id(),
                        system_ids=panel.system_ids,
                        technique=panel.technique,
                        technique_category=panel.technique_category,
                        source_type="figure",
                        source_ref=ref,
                        summary=f"{panel.plot_type}: {panel.description}",
                        conditions=[f"axes: {', '.join(panel.axes)}"] if panel.axes else [],
                        quote=fig.caption or None,
                        context=fig.context,
                        image_path=str(fig.image_path.relative_to(state.out_dir)),
                    )
                )
        except Exception as e:
            state.errors.append(f"Figure analysis failed for {fig.figure_id}: {e}")
            ctx.deps.on_progress(f"  ! {state.errors[-1]}")

        if self.index + 1 < len(figures):
            return AnalyzeFigure(index=self.index + 1)
        return BuildGraph()


@dataclass
class BuildGraph(BaseNode[SummarizerState, SummarizerDeps, PaperGraph]):
    """Merge duplicates, build the knowledge graph and save everything. No LLM."""

    async def run(self, ctx: Ctx) -> End[PaperGraph]:
        state = ctx.state
        ctx.deps.on_progress("Building the knowledge graph and saving files...")
        state.evidence = merge_duplicate_evidence(state.evidence)
        metadata = state.metadata or PaperMetadata(title=state.pdf_path.stem, summary="(not extracted)")
        graph = build_graph(str(state.pdf_path), ctx.deps.model_name, metadata, state.systems, state.evidence)
        path = save_outputs(graph, state.out_dir)
        if state.errors:
            (state.out_dir / "errors.txt").write_text("\n".join(state.errors))
        ctx.deps.on_progress(
            f"Done. {len(graph.systems)} systems, {len(graph.evidence)} evidence items. "
            f"Tokens: {ctx.deps.usage.input_tokens} in / {ctx.deps.usage.output_tokens} out. Saved to {path}"
        )
        return End(graph)


def merge_duplicate_evidence(evidence: list[Evidence]) -> list[Evidence]:
    """Series-wide statements ('all samples were measured with Cu-Kα') come back once per system
    from ExtractTextEvidence. Merge text items with the same technique, quote and summary; items that
    quote the same sentence but say something different about each system stay separate."""
    merged: dict[tuple, Evidence] = {}
    out: list[Evidence] = []
    for ev in evidence:
        if ev.source_type != "text" or not ev.quote:
            out.append(ev)
            continue
        key = (ev.technique.lower(), " ".join(ev.quote.lower().split()), " ".join(ev.summary.lower().split()))
        if key in merged:
            keep = merged[key]
            keep.system_ids = sorted(set(keep.system_ids) | set(ev.system_ids), key=lambda s: int(s[1:]) if s[1:].isdigit() else 0)
            known = {(m.quantity, m.value) for m in keep.measurements}
            keep.measurements += [m for m in ev.measurements if (m.quantity, m.value) not in known]
        else:
            merged[key] = ev
            out.append(ev)
    for i, ev in enumerate(out, start=1):  # renumber so IDs are E001, E002, ... without gaps
        ev.evidence_id = f"E{i:03d}"
    return out


# --------------------------------------------------------------------------------------
# Assemble the graph
# --------------------------------------------------------------------------------------

_builder = GraphBuilder(name="paper_summarizer", state_type=SummarizerState, deps_type=SummarizerDeps, input_type=ReadPDF, output_type=PaperGraph)
_builder.add(
    _builder.edge_from(_builder.start_node).to(ReadPDF),
    _builder.node(ReadPDF),
    _builder.node(FindSystems),
    _builder.node(ExtractTextEvidence),
    _builder.node(AnalyzeFigure),
    _builder.node(BuildGraph),
)
summarizer_graph = _builder.build()


async def summarize_paper(
    pdf_path: str | Path,
    out_dir: str | Path,
    model: Model | str,
    on_progress: Callable[[str], None] = print,
    analyze_figures: bool = True,
    max_figures: int = 30,
    usage: RunUsage | None = None,
) -> PaperGraph:
    """Run the whole workflow on one PDF and return the PaperGraph (also saved in ``out_dir``).
    The token usage of every LLM call is added to ``usage`` when given."""
    state = SummarizerState(pdf_path=Path(pdf_path), out_dir=Path(out_dir))
    deps = SummarizerDeps(model=model, on_progress=on_progress, analyze_figures=analyze_figures, max_figures=max_figures,
                          usage=usage if usage is not None else RunUsage())
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return await summarizer_graph.run(state=state, deps=deps, inputs=ReadPDF())


def summarize_paper_sync(*args, **kwargs) -> PaperGraph:
    return asyncio.run(summarize_paper(*args, **kwargs))
