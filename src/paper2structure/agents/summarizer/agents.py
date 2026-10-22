"""The three LLM agents used by the summarizer graph.

An ``Agent`` in pydantic-ai = instructions (system prompt) + an output type (a Pydantic model).
When we call ``agent.run(...)`` pydantic-ai:
  1. sends the instructions + our message (text and/or images) to the model,
  2. asks the model to answer in the JSON shape of ``output_type``,
  3. validates the answer; if it is invalid (or our ``output_validator`` raises ``ModelRetry``)
     it tells the model what was wrong and asks again (up to ``retries`` times).

The model is NOT fixed here (``defer_model_check=True``). It is passed in at run time, so the
same agents work with OpenAI, another provider, or the offline fake model used for dry runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai import Agent, ModelRetry, RunContext

from ...config import technique_guide
from . import dry_run  # noqa: F401  (registers canned answers for the dry-run model)
from .schemas import FigureOutput, SystemsOutput, TextEvidenceOutput


@dataclass
class AgentDeps:
    """Extra information the validators need. Passed with ``agent.run(..., deps=...)``."""

    valid_system_ids: set[str] = field(default_factory=set)


TECHNIQUE_GUIDE = technique_guide()

system_finder = Agent(
    None,
    name="system_finder",
    output_type=SystemsOutput,
    defer_model_check=True,
    retries=2,
    instructions=f"""
You are a materials-science research assistant. You will be given the full text of a paper.

Task:
1. Fill in the paper metadata and a short plain-language summary.
2. List EVERY distinct material system that was experimentally (or computationally) evaluated.
   A "system" is one specific material sample: each composition in a doping series is its own
   system (e.g. x = 0, 0.2, 0.25 are three systems), and so are different phases, substrates or
   thicknesses of the same compound if they were characterised separately.
   Do NOT list precursors/reagents (e.g. La2O3 used for synthesis) or impurity phases as systems.
3. Write each name as an explicit formula when the paper gives a series formula
   (La1-xBixCrO3 with x=0.2 -> 'La0.8Bi0.2CrO3') and record the variables.
""",
)

text_extractor = Agent(
    None,
    name="text_extractor",
    output_type=TextEvidenceOutput,
    deps_type=AgentDeps,
    defer_model_check=True,
    retries=2,
    instructions=f"""
You extract structured experimental information about ONE target material system from a paper.
You will get the paper text, the list of all systems with their IDs, and the target system ID.

Extract, using ONLY what the text says (never invent numbers):
- synthesis: method, precursors, temperatures/times/atmospheres.
- structure: crystal system, space group, lattice parameters, site occupancy, phase purity.
- evidence: one item per technique that gives structural information (the canonical names below, or
  others the paper uses) describing what that technique showed for the target system. Include instrument settings in `conditions`, numbers in `measurements`,
  the figures the text cites in `figure_refs`, a short verbatim `quote` and the page number.
  If a statement applies to the whole series, put all relevant system IDs in `system_ids`
  (always including the target ID).
If the text says nothing about the target for a technique, do not create an item for it.
{TECHNIQUE_GUIDE}
""",
)

figure_analyst = Agent(
    None,
    name="figure_analyst",
    output_type=FigureOutput,
    deps_type=AgentDeps,
    defer_model_check=True,
    retries=2,
    instructions=f"""
You catalogue ONE figure image from a materials-science paper. You get the image, its caption,
the other captions on the same page, and the list of material systems with IDs.

Your job is to label the figure so it can be routed to technique specialists, NOT to analyse it.
For each panel:
- identify the panel label, the plot type, the technique, and which systems it shows (use the given IDs);
- give the axis labels with units;
- describe what the panel shows (samples, traces, insets, annotations) in one or two sentences.
Do NOT read values off the figure: no peak positions, spacings, sizes, or trends. The specialists
read the image themselves.
A table is catalogued the same way: one panel per table with plot_type "table", the technique
whose results it lists (Rietveld or diffraction refinement -> XRD, binding energies -> XPS, fitted
shells -> XAS, ...), the systems in its rows or columns, its column headings with units as the axes,
and what it lists in the description, without transcribing its values (its text is in the paper
text, and the specialist reads the image).
If the image is not a scientific figure or table (logo, journal banner), return no panels.
{TECHNIQUE_GUIDE}
""",
)


def _check_ids(ctx: RunContext[AgentDeps], ids_per_item: list[list[str]]) -> None:
    valid = ctx.deps.valid_system_ids
    unknown = sorted({i for ids in ids_per_item for i in ids} - valid)
    if unknown:
        raise ModelRetry(f"Unknown system_ids {unknown}. Only use these IDs: {sorted(valid)}.")


@text_extractor.output_validator
def _validate_text(ctx: RunContext[AgentDeps], output: TextEvidenceOutput) -> TextEvidenceOutput:
    _check_ids(ctx, [e.system_ids for e in output.evidence])
    return output


@figure_analyst.output_validator
def _validate_figure(ctx: RunContext[AgentDeps], output: FigureOutput) -> FigureOutput:
    _check_ids(ctx, [p.system_ids for p in output.panels])
    return output
