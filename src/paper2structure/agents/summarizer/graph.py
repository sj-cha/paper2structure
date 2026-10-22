"""The summarizer workflow, written as a pydantic-graph.

    ReadPDF -> FindSystems -> ExtractEvidence (one call per system and per figure, concurrently)
            -> BuildGraph -> End

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
from .pdf_reader import ExtractedFigure, PaperDocument, read_pdf
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

    def systems_text(self) -> str:
        return "\n".join(f"- {s.system_id}: {s.name} {s.composition_variables or ''} ({s.form or 'form not stated'})" for s in self.systems)


@dataclass
class SummarizerDeps:
    model: Model | str
    on_progress: Callable[[str], None] = print
    analyze_figures: bool = True
    max_figures: int = 30
    usage: RunUsage = field(default_factory=RunUsage)


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

    async def run(self, ctx: Ctx) -> ExtractEvidence | BuildGraph:
        ctx.deps.on_progress("Identifying the material systems studied...")
        result = await system_finder.run(ctx.state.document.full_text, model=ctx.deps.model, usage=ctx.deps.usage)
        ctx.state.metadata = result.output.metadata
        ctx.state.systems = [
            MaterialSystem(system_id=f"S{i}", **c.model_dump(exclude={"why_distinct"}))
            for i, c in enumerate(result.output.systems, start=1)
        ]
        ctx.deps.on_progress(f"Found {len(ctx.state.systems)} material systems:\n{ctx.state.systems_text()}")
        if not ctx.state.systems:
            ctx.state.errors.append("No material systems found.")
            return BuildGraph()
        return ExtractEvidence()


MAX_CONCURRENT_CALLS = 8  # LLM calls ExtractEvidence runs at once


def figure_groups(figures: list[ExtractedFigure]) -> list[list[ExtractedFigure]]:
    """The images one vision call catalogues: the crops of one figure (figures on the same page with
    the same caption) together, every table and every uncaptioned crop alone; in reading order."""
    groups: dict[tuple, list[ExtractedFigure]] = {}
    for f in figures:
        key = (f.page, f.caption) if f.caption and f.kind == "figure" else (f.figure_id,)
        groups.setdefault(key, []).append(f)
    return list(groups.values())


@dataclass
class ExtractEvidence(BaseNode[SummarizerState, SummarizerDeps, PaperGraph]):
    """LLM calls 2 and 3, at most MAX_CONCURRENT_CALLS at a time: per system, its synthesis, structure
    and per-technique evidence from the text; per figure (all its crops in one vision call), a
    catalogue of its panels. The first text call runs before the other text calls, which then reuse
    its cached prompt prefix (the paper text). A system or figure that fails is recorded and the rest
    continue; the evidence is kept in input order (text by system, then figures)."""

    async def run(self, ctx: Ctx) -> BuildGraph:
        state, deps = ctx.state, ctx.deps
        limit = asyncio.Semaphore(MAX_CONCURRENT_CALLS)
        groups = figure_groups(state.document.figures) if deps.analyze_figures else []

        async def limited(coro):
            async with limit:
                return await coro

        async def texts() -> list:
            first = await limited(self._text(ctx, 0))
            return [first, *await asyncio.gather(*(limited(self._text(ctx, k)) for k in range(1, len(state.systems))))]

        by_system, by_figure = await asyncio.gather(
            texts(), asyncio.gather(*(limited(self._figure(ctx, k, groups)) for k in range(len(groups)))))
        for evidence, error in [*by_system, *by_figure]:
            state.evidence += evidence
            if error:
                state.errors.append(error)
                deps.on_progress(f"  ! {error}")
        return BuildGraph()

    @staticmethod
    async def _text(ctx: Ctx, k: int) -> tuple[list[Evidence], str]:
        state = ctx.state
        system = state.systems[k]
        ctx.deps.on_progress(f"Reading text evidence for {system.system_id} ({system.name}) [{k + 1}/{len(state.systems)}]...")
        prompt = (  # the paper text first: the calls for the other systems share it as a cached prefix
            f"PAPER TEXT:\n{state.document.full_text}\n\n"
            f"Material systems in this paper:\n{state.systems_text()}\n\n"
            f"TARGET SYSTEM: {system.system_id} ({system.name})"
        )
        try:
            result = await text_extractor.run(prompt, model=ctx.deps.model, deps=AgentDeps({s.system_id for s in state.systems}),
                                              usage=ctx.deps.usage)
        except Exception as e:  # keep going: one failed system should not lose the whole paper
            return [], f"Text extraction failed for {system.system_id}: {e}"
        out = result.output
        system.synthesis = out.synthesis
        system.structure = out.structure
        return [Evidence(evidence_id="",  # BuildGraph numbers the evidence once it is merged
                         source_type="text", source_ref=f"page {item.page}" if item.page else "text",
                         **item.model_dump(exclude={"page"})) for item in out.evidence], ""

    @staticmethod
    async def _figure(ctx: Ctx, k: int, groups: list[list[ExtractedFigure]]) -> tuple[list[Evidence], str]:
        state = ctx.state
        group = groups[k]
        head, ids = group[0], [f.figure_id for f in group]
        ctx.deps.on_progress(f"Looking at figure {'+'.join(ids)} (page {head.page}) [{k + 1}/{len(groups)}]...")
        if len(group) == 1:
            intro = f"Figure id: {head.figure_id}" + (" (this image is a TABLE)" if head.kind == "table" else "")
        else:
            intro = (f"These {len(group)} images are crops of one figure, in this order: {', '.join(ids)}. Give every "
                     "panel the image_id of the image it is in.")
        prompt: list = [f"Material systems in this paper:\n{state.systems_text()}\n\n{intro}\n"
                        f"Most likely caption for this figure: {head.caption or '(none found)'}\n"
                        f"All captions on page {head.page}: {head.page_captions}"]
        for f in group:
            if len(group) > 1:
                prompt.append(f"Image {f.figure_id}:")
            prompt.append(BinaryContent(data=f.image_path.read_bytes(), media_type="image/png"))
        try:
            result = await figure_analyst.run(prompt, model=ctx.deps.model, usage=ctx.deps.usage,
                                              deps=AgentDeps({s.system_id for s in state.systems}, ids))
        except Exception as e:  # keep going: one failed figure should not lose the others
            return [], f"Figure analysis failed for {', '.join(ids)}: {e}"
        out = result.output
        crops = {f.figure_id: f for f in group}
        evidence = []
        for panel in out.panels:
            fig = crops.get(panel.image_id or "", head)
            ref = f"{out.figure_label}" + (f" panel {panel.panel_label}" if panel.panel_label else "") + f" (page {fig.page})"
            evidence.append(Evidence(
                evidence_id="",  # BuildGraph numbers the evidence once it is merged
                system_ids=panel.system_ids, technique=panel.technique, source_type="figure", source_ref=ref,
                summary=f"{panel.plot_type}: {panel.description}",
                conditions=[f"axes: {', '.join(panel.axes)}"] if panel.axes else [],
                quote=fig.caption or None, context=fig.context,
                image_path=str(fig.image_path.relative_to(state.out_dir))))
        return evidence, ""


@dataclass
class BuildGraph(BaseNode[SummarizerState, SummarizerDeps, PaperGraph]):
    """Merge duplicates, number the evidence and save the PaperGraph. No LLM."""

    async def run(self, ctx: Ctx) -> End[PaperGraph]:
        state, model = ctx.state, ctx.deps.model
        ctx.deps.on_progress("Merging the evidence and saving files...")
        state.evidence = merge_duplicate_evidence(state.evidence)
        graph = build_graph(str(state.pdf_path), model if isinstance(model, str) else model.model_name,
                            state.metadata, state.systems, state.evidence)
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
    from ExtractEvidence. Merge text items with the same technique, quote and summary; items that
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
            keep.system_ids = sorted(set(keep.system_ids) | set(ev.system_ids), key=lambda s: (len(s), s))
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
    _builder.node(ExtractEvidence),
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


