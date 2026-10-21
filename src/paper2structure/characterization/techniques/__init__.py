from typing import Optional

from ..spec import GENERIC, TechniqueSpec
from .ftir import FTIR
from .sem import SEM
from .tem import TEM
from .xrd import XRD

# Register new technique specs here after cloning xrd.py.
REGISTRY: dict[str, TechniqueSpec] = {
    spec.name: spec for spec in (XRD, TEM, SEM, FTIR)
}


def find_spec(technique: str) -> Optional[TechniqueSpec]:
    for spec in REGISTRY.values():
        if spec.matches(technique):
            return spec
    return None


__all__ = ["FTIR", "GENERIC", "REGISTRY", "SEM", "TEM", "XRD", "find_spec"]
