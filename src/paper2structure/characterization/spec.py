from dataclasses import dataclass


@dataclass(frozen=True)
class TechniqueSpec:
    """Domain knowledge that specializes the generic characterization agent.

    Cloning the agent for a new technique means writing one of these;
    see `techniques/xrd.py` for a worked example.
    """

    name: str
    aliases: tuple[str, ...] = ()
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
        key = technique.strip().lower()
        return key == self.name.lower() or key in (a.lower() for a in self.aliases)


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
- Figure records: a JSON list (figure ID, technique tag, caption, material, context), followed by the images.
  Each image is preceded by a line naming its figure ID.
- Text records: a JSON list (text ID, technique tag, text, material, source, cited figures) summarizing what the paper
  says about this technique: reported values, refinement results, phase assignments, and instrument settings.
Either list may be empty. Text records were extracted by an upstream agent and may paraphrase the paper.
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
Step 1a - READ FIGURES. For each image, identify the axes, units, and every trace or panel, then record observations:
- Numerical values (positions, widths, spacings, intensities) and qualitative features (extra peaks, shoulders, broadening, shifts).
- Set `source_id` = the figure ID.
- Mark each observation as `printed_label`, `axis_estimate`, or `visual`, and give the reading precision for axis estimates.
- Record only what is visible. Do not fill in values from memory of the material or from the text.
- If an image is not {spec.name} data, or is unreadable, say so in caveats and record no observations for it.

Step 1b - READ TEXT. For each text record, record the reported {spec.name} values and claims relevant to structure:
- Set `source_id` = the text ID and `read_from` = "stated_in_text"; keep reported error bars in `uncertainty`.
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
- Assign confidence: high = directly and unambiguously supported; medium = supported but requires assumptions; low = suggestive only.

## OUTPUT
Return a single CharacterizationReport with `observations`, `findings`, `structural_constraints`, and `caveats`.
`structural_constraints` should be usable by a downstream agent to build an atomistic model.
If neither the images nor the text contain usable {spec.name} data, return empty lists and explain why in `caveats`.
"""
