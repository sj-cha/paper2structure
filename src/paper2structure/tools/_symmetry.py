"""Space groups by Hermann-Mauguin symbol in any ITA setting, from spglib's Hall-symbol database.
This is the one resolver every tool uses (build_bulk, fill_cavities, the space-group claim check,
the planner), so a symbol means the same setting everywhere."""
from __future__ import annotations

import difflib
import re
from functools import lru_cache
from typing import NamedTuple

import numpy as np
import spglib

# Pre-2002 symbols of the five groups whose double glide plane is now written "e", by setting (ITA
# Table 4.3.2.1). The abc and ba-c settings of Cmme and Ccce share their "e" symbol, so only the old
# symbols tell them apart.
_OLD_E_SYMBOLS = {
    39: {"": "Abm2", "ba-c": "Bma2", "cab": "B2cm", "-cba": "C2mb", "bca": "Cm2a", "a-cb": "Ac2m"},
    41: {"": "Aba2", "ba-c": "Bba2", "cab": "B2cb", "-cba": "C2cb", "bca": "Cc2a", "a-cb": "Ac2a"},
    64: {"": "Cmca", "ba-c": "Ccmb", "cab": "Abma", "-cba": "Acam", "bca": "Bbcm", "a-cb": "Bmab"},
    67: {"": "Cmma", "ba-c": "Cmmb", "cab": "Abmm", "-cba": "Acmm", "bca": "Bmcm", "a-cb": "Bmam"},
    68: {"": "Ccca", "ba-c": "Cccb", "cab": "Abaa", "-cba": "Acaa", "bca": "Bbcb", "a-cb": "Bbab"},
}
_SUFFIX_WORDS = ((re.compile(r"\(?\s*origin\s*(?:choice)?\s*([12])\s*\)?$", re.I), r":\1"),
                 (re.compile(r"\(?\s*hexagonal\s*(?:axes|setting|cell)?\s*\)?$", re.I), ":H"),
                 (re.compile(r"\(?\s*rhombohedral\s*(?:axes|setting|cell)?\s*\)?$", re.I), ":R"))
SPELLING = ("write the ITA number (standard setting), the Hermann-Mauguin symbol of the setting the coordinates are in "
            "('Pnma', 'Pbnm', 'P2_1/n', 'P 1 1 2_1/b', 'C2cb'), and ':2' for origin choice 2 or ':R' for rhombohedral "
            "axes ('Fd-3m:2', 'R-3m:R'); without a suffix a symbol means origin choice 1 and hexagonal axes")


class Setting(NamedTuple):
    """One setting of a space group: spglib's Hall number identifies it exactly."""
    hall: int
    number: int
    choice: str  # spglib's setting choice: '1'/'2' origin, 'H'/'R' axes, 'b1', '-c2', 'ba-c', '2cab', ...
    short: str  # standard short symbol of the group, e.g. "P2_1/c"
    full: str  # full symbol of this setting, e.g. "P 1 2_1/n 1"

    @property
    def crystal_system(self) -> str:
        n = self.number
        return ("triclinic" if n <= 2 else "monoclinic" if n <= 15 else "orthorhombic" if n <= 74 else
                "tetragonal" if n <= 142 else "trigonal" if n <= 167 else "hexagonal" if n <= 194 else "cubic")

    @property
    def unique_axis(self) -> str | None:
        """The monoclinic unique axis ('a', 'b' or 'c') of this setting."""
        if self.crystal_system != "monoclinic":
            return None
        return self.choice.lstrip("-")[:1] or "b"

    @property
    def rhombohedral_axes(self) -> bool:
        return self.choice == "R"

    @property
    def label(self) -> str:
        return f"{self.full} (#{self.number}" + (f", setting {self.choice}" if self.choice else "") + ")"


def _compact(symbol: str) -> str:
    return symbol.replace(" ", "").replace("_", "").strip()


def _normalize(symbol: str) -> str:
    s = str(symbol).strip()
    for pattern, suffix in _SUFFIX_WORDS:
        s = pattern.sub(suffix, s).strip()
    return _compact(s)


def _origin_or_axes(choice: str) -> list[str]:
    """The suffixes a symbol of this setting may carry: the whole choice, its origin digit, H/R."""
    out = [choice] if choice else []
    if choice[:1] in ("1", "2") and len(choice) > 1:
        out.append(choice[0])
    return out


@lru_cache(maxsize=1)
def _table() -> dict[str, int]:
    """Compact symbol -> Hall number. A symbol shared by several settings (monoclinic cells, origin
    choices, rhombohedral axes) names the first, standard one (unique axis b, origin choice 1,
    hexagonal axes); a suffix ':<choice>' names any setting ('Fd-3m:2', 'R-3m:R', 'P2_1/c:c1')."""
    table: dict[str, int] = {}
    types = {h: spglib.get_spacegroup_type(h) for h in range(1, 531)}
    later: list[tuple[str, int]] = []  # aliases, added only where no native name is taken
    for h, t in types.items():
        names = [t.international_full, *t.international.split("="), t.international_short]
        if t.number in _OLD_E_SYMBOLS:  # spglib's own names for these settings are inconsistent ("B b c b")
            names = [t.international_full, t.international_short]
        for name in names:
            table.setdefault(_compact(name), h)
        for name in names:
            for suffix in _origin_or_axes(t.choice):
                table.setdefault(f"{_compact(name)}:{suffix}", h)
        if 3 <= t.number <= 15:  # short monoclinic symbols of every cell choice: P 1 2_1/n 1 -> P2_1/n
            later.append((_compact(" ".join(p for p in t.international_full.split() if p != "1")), h))
        if t.number >= 195:  # the pre-1983 cubic spelling without the bar: Fm3m, Pa3, Ia3d
            later.append((_compact(t.international_short).replace("-3", "3"), h))
        tokens = t.international.split("=")[0].split()
        for k, token in enumerate(tokens[1:]):  # an e glide plane holds both glides lying in it
            if "e" in token:
                for g in "abc".replace("abc"[k], ""):
                    later.append((_compact(" ".join(tokens[:k + 1] + [token.replace("e", g)] + tokens[k + 2:])), h))
    for number, settings in _OLD_E_SYMBOLS.items():  # ITA's own old symbols take precedence
        for h, t in types.items():
            if t.number != number:
                continue
            origin, axes = (t.choice[0], t.choice[1:]) if t.choice[:1] in ("1", "2") else ("", t.choice)
            old = settings[axes]
            if origin in ("", "1"):
                table[old] = h
            if origin:
                table[f"{old}:{origin}"] = h
    for key, h in later:
        table.setdefault(key, h)
        for suffix in _origin_or_axes(types[h].choice):
            table.setdefault(f"{key}:{suffix}", h)
    return table


@lru_cache(maxsize=1)
def _standard_halls() -> dict[int, int]:
    """International number -> Hall number of its standard setting (the first one spglib lists)."""
    out: dict[int, int] = {}
    for h in range(1, 531):
        out.setdefault(int(spglib.get_spacegroup_type(h).number), h)
    return out


@lru_cache(maxsize=None)
def _setting(hall: int) -> Setting:
    t = spglib.get_spacegroup_type(hall)
    return Setting(hall, int(t.number), t.choice, t.international_short, t.international_full)


def resolve(symbol: str | int) -> Setting:
    """The setting a space-group symbol names: a Hermann-Mauguin symbol in any setting ("Pnma",
    "Pbnm", "Cc2m", "C2cb", "P2_1/n", "P 1 1 2_1/b"), with ':1'/':2' for the origin choice and
    ':H'/':R' for hexagonal/rhombohedral axes ("Fd-3m:2", "R-3m:R"), or the international number
    (standard setting). Without a suffix a symbol means origin choice 1 and hexagonal axes."""
    if isinstance(symbol, (int, np.integer)) or str(symbol).strip().isdigit():
        number = int(symbol)
        if not 1 <= number <= 230:
            raise ValueError(f"space group number {number} is not between 1 and 230")
        return _setting(_standard_halls()[number])
    key = _normalize(str(symbol))
    hall = _table().get(key)
    if hall is None:
        close = difflib.get_close_matches(key, _table().keys(), n=4, cutoff=0.75)
        hint = f"; close: {', '.join(sorted({_setting(_table()[c]).label for c in close}))}" if close else ""
        raise ValueError(f"unknown space group symbol {symbol!r}{hint}. Accepted: {SPELLING}")
    return _setting(hall)


def settings_of(number: int) -> list[Setting]:
    """Every setting of the space group with this international number."""
    return [_setting(h) for h in range(1, 531) if spglib.get_spacegroup_type(h).number == number]


def hall_number(symbol: str | int) -> int:
    """The Hall number of the setting `symbol` names (see resolve)."""
    return resolve(symbol).hall


def space_group_number(symbol: str | int) -> int:
    return resolve(symbol).number


@lru_cache(maxsize=None)
def hall_operations(hall: int) -> tuple[np.ndarray, np.ndarray]:
    """Rotations (n, 3, 3) and translations (n, 3) of the setting with this Hall number, acting on
    fractional coordinates of its conventional cell (centring translations included)."""
    ops = spglib.get_symmetry_from_database(int(hall))
    rotations, translations = np.asarray(ops["rotations"], dtype=float), np.asarray(ops["translations"], dtype=float)
    rotations.flags.writeable = translations.flags.writeable = False
    return rotations, translations


def symmetry_operations(symbol: str | int) -> tuple[np.ndarray, np.ndarray]:
    """Operations of the space group in the setting of `symbol` (see resolve and hall_operations)."""
    return hall_operations(hall_number(symbol))


def orbit(point, rotations: np.ndarray, translations: np.ndarray, tol: float = 1e-3,
          cell: np.ndarray | None = None) -> np.ndarray:
    """The distinct images (fractional, wrapped into [0, 1)) of one fractional point. Images closer
    than `tol` (fractional per component, or Angstrom when `cell` is given) are one site and are
    replaced by their mean: a point a rounding error away from a special position (1/3 written as
    0.3333) lands on it."""
    images = np.mod(np.einsum("nij,j->ni", rotations, np.asarray(point, float)) + translations, 1.0)
    diff = images[:, None, :] - images[None, :, :]
    diff -= np.round(diff)
    same = (np.linalg.norm(diff @ np.asarray(cell, float), axis=2) < tol if cell is not None
            else np.all(np.abs(diff) < tol, axis=2))
    owner = np.full(len(images), -1)
    sites = []
    for i in range(len(images)):
        if owner[i] >= 0:
            continue
        members = np.where(same[i] & (owner < 0))[0]
        owner[members] = i
        sites.append(np.mod(images[i] - diff[i, members].mean(axis=0), 1.0))
    return np.array(sites)
