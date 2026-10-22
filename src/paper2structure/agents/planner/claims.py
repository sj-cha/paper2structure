"""The paper's structural claims, each with an ID: the facts the summarizer recorded for each material
system and the characterization reports' structural constraints. Every plan turns each claim about
its systems into a check the validation gate runs (tools.schemas.ClaimCheck), or lists it as
unchecked with the reason."""
from __future__ import annotations

from dataclasses import dataclass

from paper2structure.agents.characterization.schemas import ExperimentalEvidence
from paper2structure.agents.summarizer.schemas import PaperGraph


@dataclass(frozen=True)
class Claim:
    claim_id: str
    system_ids: tuple[str, ...]  # empty: the whole paper
    text: str
    source: str  # "summary" or the technique of the report


def structural_claims(summary: PaperGraph, evidence: ExperimentalEvidence) -> list[Claim]:
    """Claim IDs: "<system>:space_group", "<system>:lattice", "<system>:occupancy<k>", "<system>:other<k>"
    from the summary; "<technique>:<constraint_id>" (C<k>, the k-th constraint as its report was
    written) for the structural constraints of the reports."""
    claims = []
    for s in summary.systems:
        st, sid = s.structure, s.system_id
        if st.space_group or st.crystal_system:
            claims.append(Claim(f"{sid}:space_group", (sid,), f"{s.name}: "
                                + ", ".join(x for x in (st.crystal_system, st.space_group) if x), "summary"))
        if st.lattice_parameters:
            claims.append(Claim(f"{sid}:lattice", (sid,), f"{s.name} lattice: " + ", ".join(
                f"{m.quantity} = {m.value}{' ' + m.unit if m.unit else ''}" for m in st.lattice_parameters), "summary"))
        claims += [Claim(f"{sid}:occupancy{k}", (sid,), f"{s.name}: {text}", "summary")
                   for k, text in enumerate(st.site_occupancy, 1)]
        claims += [Claim(f"{sid}:other{k}", (sid,), f"{s.name}: {text}", "summary") for k, text in enumerate(st.other, 1)]
    for technique, report in sorted(evidence.reports.items()):
        for c in report.structural_constraints:
            cites = f" (from {', '.join(f'{technique}:{o}' for o in c.evidence)})" if c.evidence else ""
            claims.append(Claim(f"{technique}:{c.constraint_id}", tuple(c.system_ids),
                                f"{c.property}: {constraint_value(c)} [{c.confidence} confidence]{cites}", technique))
    return claims


def claims_text(claims: list[Claim]) -> str:
    return "\n".join(f"- {c.claim_id} [{', '.join(c.system_ids) or 'all systems'}] {c.text}" for c in claims)

def constraint_value(c) -> str:
    """A structural constraint's value with its unit (once)."""
    return c.value if not c.unit or c.value.rstrip().endswith(c.unit) else f"{c.value} {c.unit}"


def claims_about(claims: list[Claim], system_ids) -> list[Claim]:
    """The claims a plan modeling these systems must address: those about them or about no system."""
    keep = set(system_ids)
    return [c for c in claims if not c.system_ids or keep & set(c.system_ids)]

