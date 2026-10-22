from typing import Optional

from ....config import TECHNIQUES, canonical_technique
from ..spec import GENERIC, TechniqueSpec
from .afm import AFM
from .dft import DFT
from .ftir import FTIR
from .mossbauer import MOSSBAUER
from .nmr import NMR
from .raman import RAMAN
from .sem import SEM
from .structure_model import STRUCTURE_MODEL
from .tem import TEM
from .tga import TGA
from .uv_vis import UV_VIS
from .xas import XAS
from .xps import XPS
from .xrd import XRD

# One spec per canonical technique in paper2structure.config.TECHNIQUES.
REGISTRY: dict[str, TechniqueSpec] = {
    spec.name: spec for spec in (
        XRD, TEM, SEM, AFM, XPS, XAS, RAMAN, FTIR,
        UV_VIS, NMR, MOSSBAUER, TGA, DFT, STRUCTURE_MODEL,
    )
}
assert set(REGISTRY) == set(TECHNIQUES), f"specs and vocabulary differ: {set(REGISTRY) ^ set(TECHNIQUES)}"


def find_spec(technique: str) -> Optional[TechniqueSpec]:
    """Spec for any spelling or alias of a technique, or None if it is not in the vocabulary."""
    return REGISTRY.get(canonical_technique(technique) or "")


__all__ = ["GENERIC", "REGISTRY", "find_spec"]
