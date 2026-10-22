"""Size and rough compute cost of a structure, for the web page's Structure tab."""

from __future__ import annotations

import math

from ase import Atoms

_NOBLE_CORES = (0, 2, 10, 18, 36, 54, 86)


def valence_electrons(atoms: Atoms) -> int:
    """Electrons outside the noble-gas core, minus filled d/f shells (e.g. Bi 5, not 29)."""
    return int(sum(_valence(int(z)) for z in atoms.numbers))


def _valence(z: int) -> int:
    v = z - max(c for c in _NOBLE_CORES if c < z)
    if 31 <= z <= 36 or 49 <= z <= 54:  # filled 3d / 4d
        v -= 10
    elif 72 <= z <= 80:  # filled 4f
        v -= 14
    elif 81 <= z <= 86:  # filled 4f and 5d
        v -= 24
    return v


def structure_info(atoms: Atoms) -> dict:
    symbols = atoms.get_chemical_symbols()
    info = {
        "formula": atoms.get_chemical_formula("hill"),
        "atoms": len(atoms),
        "species": {s: symbols.count(s) for s in sorted(set(symbols))},
        "valence_electrons": valence_electrons(atoms),
        "periodic": "".join("xyz"[i] for i, p in enumerate(atoms.pbc) if p) or "none",
        "cell_lengths_A": [round(float(x), 2) for x in atoms.cell.lengths()],
    }
    if atoms.pbc.all():
        volume = atoms.get_volume()
        info["volume_A3"] = round(volume, 1)
        info["density_g_cm3"] = round(float(sum(atoms.get_masses())) * 1.66054 / volume, 2)
    return info


def compute_estimate(atoms: Atoms, method: str) -> dict:
    """Order-of-magnitude cost from rules of thumb, good enough to compare structures."""
    n, ne = len(atoms), valence_electrons(atoms)
    if method == "DFT":
        kpts = [max(1, math.ceil(25 / x)) if p else 1 for x, p in zip(atoms.cell.lengths(), atoms.pbc)]
        single = 0.05 * (ne / 100) ** 3 * math.prod(kpts)
        return {
            "k-point grid": "×".join(map(str, kpts)),
            "Single-point (PBE)": _hours(single, "core-h"),
            "Relaxation (~30 steps)": _hours(30 * single, "core-h"),
            "Scaling": "~ electrons³: doubling the cell ≈ 8× the cost",
        }
    steps = 1_000_000  # 1 ns at a 1 fs time step
    return {
        "Classical force field, 1 ns": _hours(n * steps * 2e-6 / 3600, "core-h"),
        "ML potential on 1 GPU, 1 ns": _hours(n * steps * 1e-5 / 3600, "GPU-h"),
        "Scaling": "~ atoms: doubling the cell ≈ 2× the cost",
    }


def _hours(h: float, unit: str) -> str:
    if h < 1:
        return f"< 1 {unit}"
    return f"~{h:,.0f} {unit}" if h < 1000 else f"~{h / 1000:,.0f}k {unit}"
