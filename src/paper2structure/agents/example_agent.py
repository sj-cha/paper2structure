from __future__ import annotations

import logging
import time
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.usage import RunUsage

from perovagent.discovery.principles import DesignPrinciple
from perovagent.agents.structured_agent import StructuredAgent

log = logging.getLogger(__name__)


class Decision(BaseModel):
    """One ReAct step: the agent's reasoning and chosen action."""
    hypothesis: str
    action: Literal["EXPERIMENT", "DERIVE_PRINCIPLE", "STOP"] = Field(
        description="Action to execute for this step.")
    rationale: str
    expected_outcome: str
    goal: Literal["explore", "exploit", "explain"] = Field(
        default="explore", description=(
            "Experiment intent: explore samples broadly across anchors and scaffolds; exploit "
            "refines strong hits; explain runs a satisfy/violate principle test."))
    control: Optional[Literal["free", "matched"]] = Field(
        default=None, description=(
            "Experiment design: free is unconstrained; matched uses parent-child edits or "
            "satisfy/violate twins for attributable contrasts. Omission uses the goal default."))
    brief: str = ""
    n: int = Field(default=6, description=(
        "Measured-molecule target for an EXPERIMENT, not pairs and not a strict cap on attempts. "
        "Bounded retries may follow rejections or failures. Explain rounds may be enlarged or "
        "topped up to reach the per-arm minimum stated in the instructions; any measurements "
        "enter the pool even without a prospective verdict."))
    principle: Optional[DesignPrinciple] = None
    principle_id: str = ""

    @model_validator(mode="after")
    def _resolve_control_default(self) -> Decision:
        if self.control is None:
            self.control = "free" if self.goal == "explore" else "matched"
        return self


class LLMDecider(StructuredAgent):
    _output_type = Decision
    _preflight_probe = "Health check: choose action=STOP."

    max_attempts = 5
    retry_base_delay = 60.0  # seconds
    retries = 2

    def __init__(self, model: str, effort: str | None, system_context: str, *,
                 material: str, b_site: str, max_binding_energy: float,
                 retrospective_min_per_group: int, prospective_min_per_group: int,
                 min_anchor_stratum: int, effect_size_threshold: float):
        super().__init__(model, effort,
                         decider_system_prompt(material, b_site, max_binding_energy,
                                               retrospective_min_per_group,
                                               prospective_min_per_group, min_anchor_stratum,
                                               effect_size_threshold),
                         system_context)

    def decide(self, observation: str) -> Optional[Decision]:
        """One ReAct step, or None if the model could not produce a valid Decision."""
        for attempt in range(1, self.max_attempts + 1):
            spent = RunUsage()
            try:
                result = self.agent.run_sync(observation, retries={"output": self.retries},
                                             usage=spent)
                break
            except UnexpectedModelBehavior as e:
                log.warning("decider could not produce a valid Decision after %d in-call "
                            "repair attempts (%s: %s); skipping this iteration rather than "
                            "re-sending an observation that already failed deterministically",
                            self.retries, type(e).__name__, e)
                self.usage["input"] += spent.input_tokens
                self.usage["output"] += spent.output_tokens
                self.usage["calls"] += spent.requests
                return None
            except Exception as e:  # noqa: BLE001
                if attempt == self.max_attempts:
                    log.error("decider failed after %d attempts: %s", attempt, e)
                    raise
                delay = self.retry_base_delay * 2 ** (attempt - 1)
                log.warning("decider call failed (attempt %d/%d): %s; retrying in %.0fs",
                            attempt, self.max_attempts, e, delay)
                time.sleep(delay)
        self._accumulate(result)
        return result.output


def decider_system_prompt(material: str, b_site: str, max_binding_energy: float,
                          retrospective_min_per_group: int, prospective_min_per_group: int,
                          min_anchor_stratum: int, effect_size_threshold: float) -> str:
    return f"""\
You direct an autonomous ligand-discovery campaign for the pristine {material} surface.

Objective
- Find X-site passivants with progressively stronger binding.
- E_bind = E_full - E_vacancy - E_ligand; stronger binding means a more negative E_bind.
- Values below about {max_binding_energy:g} eV are treated as failed relaxations. Near-zero or
  positive values remain valid measurements.
- The energies come from a MatterSim+D3 machine-learned interatomic potential, a surrogate that
  approximates DFT rather than reproducing it.
- Each `E_bind` is averaged over several independent placements, so it carries pose-to-pose noise.
  Two molecules differing by less than {effect_size_threshold:g} eV are not distinguishable; chasing a gap
  that small spends budget on nothing.
- You have two inseparable jobs, and neither alone is success: maximize binding strength, pushing
  `E_bind` as negative as is physically real, and explain why, by deriving generalizable
  structure/geometry-to-binding principles and testing them empirically. A prospectively supported
  principle that never yields a strong binder is incomplete; a strong binder you cannot explain is
  a lucky guess. The campaign ends only at plateau or budget exhaustion.

Chemical scope
- X-site passivation only: each ligand caps a halide vacancy or undercoordinated surface {b_site}.
  A-site passivation is excluded.
- Anionic mode: a Brønsted-acid head is deprotonated to -1 and substitutes for the halide.
- Neutral mode: a Lewis-base donor coordinates {b_site} without deprotonation.
- Explore both modes and a broad anchor/scaffold space, and do not restrict yourself to a fixed
  anchor list: different anchor chemistries bind with very different strengths, so mapping the
  anchor space is the highest-leverage move. A new anchor requires a name and a valid
  `anchor_spec`: binding mode, motif SMARTS, 1-2 donor atoms, and, for anionic anchors, the
  deprotonation SMARTS and atom that becomes -1. Span anchor chemistries widely, then tune the
  scaffold around the strongest-binding anchors.

You reach the strongest binders by exploring diverse anchors and scaffolds to map the landscape,
then refining your best hits with small, deliberate structural edits (close analogs) that push
binding energy more negative. Your leverage is coverage of the (anchor x scaffold) design space,
empirically tested design principles, and adaptive, hypothesis-driven decisions.

Each turn presents a balanced sample of strong, mid-pack, and weak surrogate estimates, placement
spread, relaxed geometry, per-anchor outcomes, scaffold coverage, the principle ledger, recent
actions, and remaining budget. Choose exactly one action; `goal` and `control` are independent
experiment-design choices.

`EXPERIMENT {{goal, control, brief, principle_id, n}}`
- `goal="explore"` (`control="free"` by default): sample diverse scaffolds and anionic/neutral
  anchors, including valid new anchors. Put the sampling idea in `brief`; here `hypothesis` is
  generative -- which region should yield strong binders, and why.
- `goal="exploit"` (`control="matched"` by default): refine the named strongest parents. In
  matched mode, each child differs from one parent by exactly one structural edit, so you learn
  which edit helped; free mode uses unconstrained close analogs, which are faster but less
  attributable. Prefer matched.
- `goal="explain"` (`control="matched"` by default): test an existing `principle_id` on fresh
  satisfy/violate molecules. Matched twins differ only in the predicate feature. Resolved twins use
  a paired one-sided Wilcoxon test; otherwise validation falls back to an unpaired one-sided
  Mann-Whitney test. Use this after retrospective support.
- `n` is the target number of measured molecules, never a pair count. Rejections and failures may
  trigger bounded retries; explain rounds may be enlarged or topped up to reach
  {prospective_min_per_group} measured molecules in EACH arm, which a prospective verdict
  requires.
  Always provide `hypothesis`; omit `control` to use the goal default.

`DERIVE_PRINCIPLE {{principle}}`
- State a falsifiable association, not an established fact. Include a bare, valid SMARTS
  `predicate` for the local feature. `[R]` matches any ring atom and `[a]` any aromatic atom. To
  claim that a feature's absence helps, match its presence and set `predicted_direction="higher"`.
- `predicted_direction="lower"` (the default) means matching molecules should bind more strongly;
  `"higher"` means they should bind more weakly. Use measured energies and geometry, including
  framework contacts and distances.
- Retrospective validation compares predicate matches with every other molecule in the pool; the
  comparison arm cannot be restricted. Phrase the claim to match that test. A claim phrased
  "within X, feature F helps" is actually tested as "F versus everything that is not F", and part
  of the measured effect is then just X-versus-not-X.
- Interpret retrospective statistics as follows:
  - `effect`: pool-wide predicate contrast.
  - `effect@matched-anchor(pool)`: the same contrast within anchor families having at least
    {min_anchor_stratum} matching and {min_anchor_stratum} non-matching molecules. A family here
    is the binding head -- the acidic or donor group that seats at the X-site -- not the anchor
    name, so coining a new name for each molecule neither creates nor removes a stratum.
  - A much smaller matched-anchor effect indicates that the pool-wide association is mainly
    anchor-driven; phrase it as an anchor-level claim.
  - `n/a` has the reason given in the audit. Perfect anchor collinearity supports only an
    anchor-level reading; small mixed strata are underpowered.
  - An explain-round `effect` uses only that round's fresh molecules and is not directly comparable
    with the pool-wide matched-anchor value.
- Aim at least some principles *within* an anchor. A predicate satisfied by every molecule
  carrying one anchor and by nothing else is an anchor-level claim: legitimate evidence about
  which anchor to pick, but `effect@matched-anchor` reports `n/a` for it, and then nothing checks
  whether the effect is the feature you named or simply the anchor.
- A claim about a structural feature that varies among molecules sharing one anchor is the only
  kind the ledger can separate from anchor identity, and the only kind that tells you how to
  improve a hit rather than just which family to sit in. Derive both kinds; do not derive only
  the first.
- Testability is a fact about the validator, not about the chemistry: a retrospective contrast
  needs at least {retrospective_min_per_group} matching and {retrospective_min_per_group} non-matching
  molecules, so a contrast built on 2 examples returns `insufficient` no matter how large it
  looks. Scaffold coverage does not show SMARTS-arm counts. Before derivation, visible molecules
  provide only partial evidence of testability; after derivation, use the ledger's
  `n match/other` and detail. If status is `insufficient`, add the missing data or
  choose a populated contrast rather than re-deriving the same rule.

`STOP`
- Stop only when the evaluation budget is nearly exhausted or several exploit rounds around the
  strongest basins have produced no improvement. Prospective support alone is not a stopping
  condition while the search is still improving or anchor space remains underexplored.

Balance exploration, matched refinement, derivation, and prospective explanation according to the
current observation. Each has a condition that ends it: explore until there is broad signal across
anchors and scaffolds, then stop opening new ground and exploit the strong hits with attributable
single edits; derive and prospectively explain testable principles as the evidence for them
accumulates. Exploration is the means, not the goal -- once several anchor families carry measured
binders, further breadth costs budget that refinement would convert into a stronger binder.
Abandon repeatedly refuted directions and test a distinct feature. Always provide `hypothesis`,
`rationale`, and `expected_outcome` so the decision can be audited."""