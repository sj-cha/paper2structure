from typing import Optional

from ....config import TECHNIQUES, canonical_technique
from ..spec import GENERIC, TechniqueSpec
from .afm import AFM
from .dft import DFT
from .dielectric import DIELECTRIC
from .dsc import DSC
from .eds import EDS
from .ferroelectric import FERROELECTRIC_PE
from .ftir import FTIR
from .magnetization import MAGNETIZATION
from .mossbauer import MOSSBAUER
from .neutron_diffraction import NEUTRON_DIFFRACTION
from .nmr import NMR
from .pl import PL
from .raman import RAMAN
from .resistivity import RESISTIVITY
from .sem import SEM
from .tem import TEM
from .tga import TGA
from .uv_vis import UV_VIS
from .xas import XAS
from .xps import XPS
from .xrd import XRD

# One spec per canonical technique in paper2structure.config.TECHNIQUES.
REGISTRY: dict[str, TechniqueSpec] = {
    spec.name: spec for spec in (
        XRD, NEUTRON_DIFFRACTION, TEM, SEM, AFM, EDS, XPS, XAS, RAMAN, FTIR, UV_VIS, PL, NMR,
        MOSSBAUER, MAGNETIZATION, DIELECTRIC, RESISTIVITY, FERROELECTRIC_PE, TGA, DSC, DFT,
    )
}
assert set(REGISTRY) == set(TECHNIQUES), f"specs and vocabulary differ: {set(REGISTRY) ^ set(TECHNIQUES)}"


def find_spec(technique: str) -> Optional[TechniqueSpec]:
    """Spec for any spelling or alias of a technique, or None if it is not in the vocabulary."""
    return REGISTRY.get(canonical_technique(technique) or "")


def implemented_specs() -> list[TechniqueSpec]:
    return [spec for spec in REGISTRY.values() if spec.implemented]


__all__ = ["FTIR", "GENERIC", "REGISTRY", "SEM", "TEM", "XRD", "find_spec", "implemented_specs"]
