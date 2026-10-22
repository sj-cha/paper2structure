"""Space groups by Hermann-Mauguin symbol in any ITA setting, from spglib's Hall-symbol database
(pymatgen knows only some non-standard settings: "Cc2m" but not "C2cb")."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import spglib


def _compact(symbol: str) -> str:
    return symbol.replace(" ", "").replace("_", "").strip()


@lru_cache(maxsize=1)
def _table() -> dict[str, int]:
    """Compact symbol -> Hall number. Full symbols name one setting each; a short symbol shared by
    several settings (monoclinic cells, origin choices, rhombohedral axes) maps to the first, the
    standard one (origin choice 1, hexagonal axes). Symbols with the 'e' glide also answer to the
    older 'a'/'b' spelling (C2ce = C2cb, Aea2 = Aba2, Cmce = Cmca)."""
    table: dict[str, int] = {}
    aliases: dict[str, int] = {}
    for hall in range(1, 531):
        t = spglib.get_spacegroup_type(hall)
        names = [t.international_full] + [s for s in t.international.split("=")]
        for name in names:
            table.setdefault(_compact(name), hall)
        table.setdefault(_compact(t.international_short), hall)
        for name in [t.international_short] + names:
            c = _compact(name)
            if "e" in c[1:]:
                for g in "ab":
                    aliases.setdefault(c[0] + c[1:].replace("e", g), hall)
    for k, v in aliases.items():
        table.setdefault(k, v)
    return table


def hall_number(symbol: str | int) -> int:
    """The Hall number of a space group given by its Hermann-Mauguin symbol in any setting
    ("Pnma", "Pbnm", "Cc2m", "C2cb", "P2_1/c", "P 1 1 2_1/b") or its international number
    (standard setting)."""
    if isinstance(symbol, int) or str(symbol).strip().isdigit():
        number = int(symbol)
        if not 1 <= number <= 230:
            raise ValueError(f"space group number {number} is not between 1 and 230")
        return next(h for h in range(1, 531) if spglib.get_spacegroup_type(h).number == number)
    hall = _table().get(_compact(str(symbol)))
    if hall is None:
        raise ValueError(f"unknown space group symbol {symbol!r}")
    return hall


def space_group_number(symbol: str | int) -> int:
    return int(spglib.get_spacegroup_type(hall_number(symbol)).number)


def symmetry_operations(symbol: str | int) -> tuple[np.ndarray, np.ndarray]:
    """Rotations (n, 3, 3) and translations (n, 3) of the space group in the setting of `symbol`,
    acting on fractional coordinates of its conventional cell (centring translations included)."""
    ops = spglib.get_symmetry_from_database(hall_number(symbol))
    return np.asarray(ops["rotations"], dtype=float), np.asarray(ops["translations"], dtype=float)


def orbit(point, rotations: np.ndarray, translations: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """The distinct images (fractional, wrapped into [0, 1)) of one fractional point."""
    images = np.mod(np.einsum("nij,j->ni", rotations, np.asarray(point, float)) + translations, 1.0)
    unique: list[np.ndarray] = []
    for p in images:
        if not any(np.all(np.abs((p - q + 0.5) % 1.0 - 0.5) < tol) for q in unique):
            unique.append(p)
    return np.array(unique)
