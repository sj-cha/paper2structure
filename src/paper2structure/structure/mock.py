"""MOCK structure generator, for testing the web page's Structure tab only.

The real generator is being written by someone else. This always returns the same demo crystal
(cubic perovskite SrTiO3), only made bigger or smaller by the settings, so the page can be tried
end to end. To plug in the real one, keep the signature of ``generate_structure`` and return a
``GeneratedStructure``.

- ``method``: "DFT" (small cell, up to ~200 atoms) or "MD" (large cell, thousands of atoms).
- ``realism``: 0 = smallest cell, 1 = largest cell with more thermal noise.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from ase import Atoms
from ase.io import write

from ..summarizer.schemas import MaterialSystem

_NOBLE_CORES = (0, 2, 10, 18, 36, 54, 86)
_TARGET_ATOMS = {"DFT": (5, 200), "MD": (1500, 20000)}  # (at realism 0, at realism 1)


@dataclass
class GeneratedStructure:
    atoms: Atoms
    system_id: str
    system_name: str
    method: str
    realism: float
    supercell: tuple[int, int, int]
    notes: list[str] = field(default_factory=list)

    def info(self) -> dict:
        a = self.atoms
        symbols = a.get_chemical_symbols()
        volume = a.get_volume()
        return {
            "formula": a.get_chemical_formula("hill"),
            "atoms": len(a),
            "species": {s: symbols.count(s) for s in sorted(set(symbols))},
            "valence_electrons": valence_electrons(a),
            "cell_lengths_A": [round(float(x), 2) for x in a.cell.lengths()],
            "volume_A3": round(volume, 1),
            "density_g_cm3": round(float(sum(a.get_masses())) * 1.66054 / volume, 2),
            "supercell": "×".join(map(str, self.supercell)),
        }

    def compute_estimate(self) -> dict:
        """Order-of-magnitude cost from rules of thumb, good enough to compare settings."""
        n, ne = len(self.atoms), valence_electrons(self.atoms)
        if self.method == "DFT":
            kpts = [max(1, math.ceil(25 / x)) for x in self.atoms.cell.lengths()]
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


def generate_structure(system: MaterialSystem, method: str = "DFT", realism: float = 0.3, seed: int = 0) -> GeneratedStructure:
    cell = Atoms("SrTiO3", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)], cell=[3.905] * 3, pbc=True)
    lo, hi = _TARGET_ATOMS[method]
    n = max(1, round(((lo + realism * (hi - lo)) / len(cell)) ** (1 / 3)))
    atoms = cell.repeat(n)
    atoms.rattle(stdev=0.01 + 0.07 * realism, seed=seed)
    notes = [f"Demo crystal (SrTiO3), not {system.name}: the real generator will replace this."]
    return GeneratedStructure(atoms, system.system_id, system.name, method, realism, (n, n, n), notes)


def valence_electrons(atoms: Atoms) -> int:
    return int(sum(z - max(c for c in _NOBLE_CORES if c < z) for z in atoms.numbers))


def export(gen: GeneratedStructure, folder: Path) -> list[Path]:
    """Save the chosen structure in the formats a DFT or MD code needs, plus the settings used."""
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / f"{gen.system_id}_{gen.method}_realism{round(gen.realism * 100):03d}"
    files = [stem.with_suffix(".cif"), stem.with_suffix(".xyz")]
    write(files[0], gen.atoms)
    write(files[1], gen.atoms, format="extxyz")
    if gen.method == "DFT":
        files.append(stem.with_suffix(".POSCAR"))
        write(files[-1], gen.atoms, format="vasp", direct=True, sort=True)
    else:
        files.append(stem.with_suffix(".lammps"))
        write(files[-1], gen.atoms, format="lammps-data", masses=True)
    files.append(stem.with_suffix(".json"))
    settings = {"system": gen.system_name, "method": gen.method, "realism": gen.realism, "mock": True}
    files[-1].write_text(json.dumps({**settings, "info": gen.info(), "compute_estimate": gen.compute_estimate(), "notes": gen.notes}, indent=2))
    return files


def _hours(h: float, unit: str) -> str:
    if h < 1:
        return f"< 1 {unit}"
    return f"~{h:,.0f} {unit}" if h < 1000 else f"~{h / 1000:,.0f}k {unit}"
