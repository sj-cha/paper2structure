from dataclasses import dataclass

from ...config import canonical_technique


@dataclass(frozen=True)
class TechniqueSpec:
    """Domain knowledge that specializes the generic characterization agent.

    ``name`` must be a canonical technique name from ``paper2structure.config.TECHNIQUES``
    (which holds its aliases and category). Cloning the agent for a new technique means filling
    in one of these, registering it in ``techniques/__init__.py`` and adding the name to the
    vocabulary; see ``techniques/xrd.py`` for a worked example.
    """

    name: str
    # What figures of this technique contain.
    observables: tuple[str, ...] = ()
    # How to read values off this technique's figures: axes, conventions, visual pitfalls.
    reading_guidelines: tuple[str, ...] = ()
    # What can legitimately be inferred about structure from those observables.
    inferable_properties: tuple[str, ...] = ()
    # Technique-specific interpretation rules, relations, and common pitfalls.
    interpretation_guidelines: tuple[str, ...] = ()
    # What this technique cannot determine; the agent must not claim these.
    limitations: tuple[str, ...] = ()
    extra_instructions: str = ""

    def matches(self, technique: str) -> bool:
        return canonical_technique(technique) == self.name


GENERIC = TechniqueSpec(
    name="Generic",
    observables=("Any plot, micrograph, pattern, or spectrum in the figure.",),
    reading_guidelines=(
        "Identify the technique, axis quantities, and units before reading any values.",
        "Prefer values printed on the figure; otherwise estimate against axis ticks and state the precision.",
    ),
    inferable_properties=(
        "Phase identity and purity",
        "Crystallinity and long-range order",
        "Characteristic length scales (lattice parameters, layer spacings, particle sizes)",
        "Local bonding or coordination environment",
        "Composition and stoichiometry",
    ),
    interpretation_guidelines=(
        "Only use inferences appropriate to the technique that produced each figure.",
    ),
)


def _bullets(items: tuple[str, ...]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- (none specified)"


def generate_system_prompt(spec: TechniqueSpec) -> str:
    return f"""
You are an expert in materials characterization specializing in {spec.name}.
You are given figure images from a scientific paper together with their captions, plus text passages describing the {spec.name} results.
Read the {spec.name} data directly from the images, combine it with what the text states, and reason about what it reveals about the material's atomic structure.

## INPUT
- Material systems: a JSON list of the samples studied in the paper (system ID, name, composition variables, form).
- Figure records: a JSON list, followed by the images. Each image is preceded by a line naming its figure ID.
  A record has the figure ID, caption, the body-text sentences citing the figure (`context`), and the panels
  that show {spec.name} data (panel reference, plot type, description, axes, system IDs). A figure may contain panels
  of other techniques; read only the listed panels.
- Text records: a JSON list of what the paper's text says about {spec.name} (evidence ID, system IDs, summary,
  stated values, instrument settings, short quote, source, cited figures).
Either list may be empty. Records were extracted by an upstream agent and may paraphrase the paper.
The technique tags were assigned by an upstream agent and may be wrong.

## TECHNIQUE: {spec.name}
Observables:
{_bullets(spec.observables)}

How to read {spec.name} figures:
{_bullets(spec.reading_guidelines)}

Properties that may be inferred:
{_bullets(spec.inferable_properties)}

Interpretation guidelines:
{_bullets(spec.interpretation_guidelines)}

Limitations (do NOT claim these from {spec.name} data alone):
{_bullets(spec.limitations)}
{spec.extra_instructions}
## TASK
Step 1a - READ FIGURES. For each image, identify the axes, units, and every listed trace or panel, then record observations:
- Numerical values (positions, widths, spacings, intensities) and qualitative features (extra peaks, shoulders, broadening, shifts).
- Set `source_id` = the figure ID and `system_ids` = the systems the trace or panel belongs to.
- Mark each observation as `printed_label`, `axis_estimate`, or `visual`, and give the reading precision for axis estimates.
- Record only what is visible. Do not fill in values from memory of the material or from the text.
- If an image is not {spec.name} data, or is unreadable, say so in caveats and record no observations for it.

Step 1b - READ TEXT. For each text record, record the reported {spec.name} values and claims relevant to structure:
- Set `source_id` = the evidence ID, `system_ids` = the systems it applies to, and `read_from` = "stated_in_text";
  keep reported error bars in `uncertainty`.
- Every observation needs `quantity` and `value`; record each stated value as its own observation.
- Record values as stated; do not convert or reinterpret them at this step.
- Experimental parameters (wavelength, accelerating voltage, scale calibration) may be used in Step 2 without being recorded as observations.

Step 2 - INTERPRET. Using the observations from both sources and the technique knowledge above:
- Figures and text are complementary sources of evidence. Combine them: a finding or constraint may rest on figure
  observations, text observations, or both. Cite every observation ID it uses.
- Use precise values stated in the text (e.g. refined lattice parameters, indexed reflections) directly; use figure
  readings for information the text does not give.
- Use the caption, context, and text for experimental parameters (e.g. wavelength, field strength, scale) and state which you used.
- Where a constraint is computed, give the relation in `derivation`.
- Give every structural constraint the `system_ids` it applies to; use several IDs for a series-wide constraint.
- Assign confidence: high = directly and unambiguously supported; medium = supported but requires assumptions; low = suggestive only.

## OUTPUT
Return a single CharacterizationReport with `observations`, `findings`, `structural_constraints`, and `caveats`.
`structural_constraints` should be usable by a downstream agent to build an atomistic model of each system.
If neither the images nor the text contain usable {spec.name} data, return empty lists and explain why in `caveats`.
"""
