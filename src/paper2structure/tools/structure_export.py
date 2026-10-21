from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.geometry import get_distances
from ase.io import read, write

from paper2structure.tools._geometry import (clean_arrays, coordination_pairs, get_molecules, mol_ids, neighbor_pairs,
                                             rcov, substrate_mask)

EXTENSIONS = {"poscar": "vasp", "cif": "cif", "extxyz": "extxyz"}
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _calling_script() -> str:
    """The script that is running, relative to the working directory ("<inline>" for python -c / stdin)."""
    arg = sys.argv[0] if sys.argv else ""
    if arg in ("", "-c", "-") or not os.path.isfile(arg):
        return "<inline>"
    path = os.path.abspath(arg)
    rel = os.path.relpath(path)
    return path if rel.startswith("..") else rel


def emit(atoms: Atoms, name: str, plan_id: str, notes: str = "") -> str:
    """Register a finished structure for the plan `plan_id`: writes structures/<name>.extxyz (all
    per-atom metadata), structures/<name>.vasp (POSCAR) and its .json sidecar under the working
    directory, and records the structure and the emitting script in structures/manifest.json.
    Emitting the same name again replaces it.

    name: short identifier (letters, digits, "_", "-", "."), unique per structure.
    """
    if not _NAME.match(name) or name == "manifest":
        raise ValueError(f"invalid structure name {name!r}: use letters, digits, '_', '-', '.' (max 64)")
    out = Path(os.environ.get("P2S_EMIT_DIR", "structures"))
    out.mkdir(parents=True, exist_ok=True)
    atoms = clean_arrays(atoms.copy())
    script = _calling_script()
    atoms.info.update({"p2s_name": name, "p2s_plan_id": str(plan_id), "p2s_notes": str(notes)})
    path = out / f"{name}.extxyz"
    write(path, atoms, format="extxyz")
    export_structure(atoms, out, name, formats=("poscar",))
    manifest = out / "manifest.json"
    entries = json.loads(manifest.read_text()) if manifest.exists() else []
    entries = [e for e in entries if e["name"] != name]
    entries.append({"name": name, "plan_id": str(plan_id), "path": path.name, "script": script, "notes": str(notes),
                    "formula": atoms.get_chemical_formula(), "n_atoms": len(atoms)})
    manifest.write_text(json.dumps(entries, indent=2))
    print(f"[emit] {name} (plan {plan_id}): {atoms.get_chemical_formula()}, {len(atoms)} atoms -> {path}")
    return str(path)


def read_structure(path: str | os.PathLike) -> Atoms:
    return read(path)


OPEN_GAP = 5.0  # A of empty space along a periodic direction that marks it as vacuum


def _axis_gaps(atoms: Atoms) -> dict[int, float]:
    """Empty space (A) between the structure and its periodic image along each periodic cell vector."""
    cell = atoms.cell.array
    vol = abs(np.linalg.det(cell))
    gaps = {}
    for k in range(3):
        if not atoms.pbc[k]:
            continue
        n = np.cross(cell[(k + 1) % 3], cell[(k + 2) % 3])
        n /= np.linalg.norm(n)
        h = atoms.positions @ n
        gaps[k] = float(vol / np.linalg.norm(np.cross(cell[(k + 1) % 3], cell[(k + 2) % 3])) - (h.max() - h.min()))
    return gaps


def structure_kind(atoms: Atoms) -> str:
    """The builders' label (atoms.info["p2s_kind"]) or, for structures built with custom code, one
    inferred from the vacuum along the periodic directions: "slab" (one open direction), "wire"
    (two), "nanocrystal" (three, or no periodicity) or "bulk"."""
    kind = atoms.info.get("p2s_kind")
    if kind:
        return str(kind)
    gaps = _axis_gaps(atoms)
    n_open = sum(g > OPEN_GAP for g in gaps.values()) + (3 - len(gaps))
    return {0: "bulk", 1: "slab", 2: "wire"}.get(n_open, "nanocrystal")


def vacuum_gap(atoms: Atoms) -> float:
    """Vacuum (A) between the structure and its nearest periodic image across its open directions:
    along the surface normal for slabs, the smallest open direction otherwise."""
    gaps = _axis_gaps(atoms)
    if not gaps:
        return float("inf")
    kind = structure_kind(atoms)
    if kind == "slab":
        return gaps[2] if atoms.info.get("p2s_kind") == "slab" and 2 in gaps else max(gaps.values())
    open_gaps = [g for g in gaps.values() if g > OPEN_GAP] if kind in ("wire", "nanocrystal") else []
    return float(min(open_gaps or gaps.values()))


def composition_profile(atoms: Atoms, layer_tol: float = 0.6, shell_width: float = 3.0) -> str:
    """Where each element sits in the substrate: layer compositions from the top for slabs, radial
    shells from the center for particles (e.g. a core-shell particle reads "0-3 A Cd4Se4 | ...")."""
    sub = np.where(substrate_mask(atoms))[0]
    if len(sub) < 2:
        return ""
    sym = np.array(atoms.get_chemical_symbols())[sub]
    pos = atoms.positions[sub]
    kind = structure_kind(atoms)
    if kind == "slab":
        axis = 2 if atoms.info.get("p2s_kind") == "slab" else max(_axis_gaps(atoms).items(), key=lambda kv: kv[1])[0]
        cell = atoms.cell.array
        n = np.cross(cell[(axis + 1) % 3], cell[(axis + 2) % 3])
        h = pos @ (n / np.linalg.norm(n))
        order = np.argsort(-h)
        layers, current = [], [order[0]]
        for a, b in zip(order[:-1], order[1:]):
            if h[a] - h[b] > layer_tol:
                layers.append(current)
                current = []
            current.append(b)
        layers.append(current)
        shown = [Atoms(list(sym[layer])).get_chemical_formula() for layer in layers[:4]]
        return (f"layers from the top ({len(layers)} in total): " + " | ".join(shown)
                + (f" | ... | bottom {Atoms(list(sym[layers[-1]])).get_chemical_formula()}" if len(layers) > 4 else ""))
    if kind in ("nanocrystal", "wire"):
        r = np.linalg.norm(pos - pos.mean(axis=0), axis=1)
        width = max(shell_width, float(r.max()) / 6 + 1e-9)
        bins = []
        for k in range(int(np.ceil((r.max() + 1e-9) / width))):
            inside = (r >= k * width) & (r < (k + 1) * width)
            if inside.any():
                bins.append(f"{k * width:.0f}-{(k + 1) * width:.0f} A {Atoms(list(sym[inside])).get_chemical_formula()}")
        return "radial composition from the center: " + " | ".join(bins)
    return ""


def binding_contacts(atoms: Atoms, max_ratio: float = 1.5) -> list[str]:
    """For each adsorbed molecule, its binding atoms' nearest substrate atoms, e.g.
    "mol 0 CO: C -> Pt 1.85, Pt 2.70"."""
    out = []
    ids = mol_ids(atoms)
    sub = np.where(substrate_mask(atoms))[0]
    if len(sub) == 0:
        return out
    sym = atoms.get_chemical_symbols()
    r = rcov(sym)
    for m in get_molecules(atoms):
        if m.get("role") != "adsorbate":
            continue
        members = np.where(ids == m["mol_id"])[0]
        if len(members) == 0:
            continue
        binding = m.get("binding_atoms")
        if not binding:  # added by custom code: report the molecule atom closest to the substrate
            _, dm = get_distances(atoms.positions[members], atoms.positions[sub], cell=atoms.cell, pbc=atoms.pbc)
            binding = [int(np.argmin(dm.min(axis=1)))]
        parts = []
        for local in binding:
            if local >= len(members):
                continue
            g = int(members[local])
            _, d = get_distances(atoms.positions[g:g + 1], atoms.positions[sub], cell=atoms.cell, pbc=atoms.pbc)
            d = d[0]
            near = [k for k in np.argsort(d)[:4] if d[k] < max_ratio * (r[g] + r[sub[k]])]
            txt = ", ".join(f"{sym[sub[k]]} {d[k]:.2f}" for k in near) or f"none within {max_ratio}x radii"
            parts.append(f"{sym[g]} -> {txt}")
        out.append(f"mol {m['mol_id']} {m['name']}: " + "; ".join(parts))
    return out


def substrate_coordination(atoms: Atoms) -> str:
    """Per element of the substrate: coordination-number range and its main bonding partner,
    e.g. "Pb CN 5-6 (Br 2.94 A), Cs CN 8-12 (Br 4.15 A)". An element with no partner, or an
    unexpected partner, usually means a wrong bulk."""
    from collections import Counter

    sub = np.where(substrate_mask(atoms))[0]
    if len(sub) < 2:
        return ""
    s = atoms[sub]
    sym = s.get_chemical_symbols()
    i, j, D = coordination_pairs(s)
    cn = np.bincount(i, minlength=len(s))
    d = np.linalg.norm(D, axis=1)
    parts = []
    for el in sorted(set(sym)):
        cns = cn[[k for k, x in enumerate(sym) if x == el]]
        pairs = [(sym[j[m]], d[m]) for m in range(len(i)) if sym[i[m]] == el]
        if not pairs:
            parts.append(f"{el} no bonded neighbours")
            continue
        partner = Counter(p for p, _ in pairs).most_common(1)[0][0]
        dist = float(np.median([x for p, x in pairs if p == partner]))
        span = f"{cns.min()}" if cns.min() == cns.max() else f"{cns.min()}-{cns.max()}"
        parts.append(f"{el} CN {span} ({partner} {dist:.2f} A)")
    return ", ".join(parts)


def describe(atoms: Atoms) -> str:
    """Compact text summary of a structure: composition, cell, molecules, vacuum, closest
    contact and the binding contacts of adsorbed molecules."""
    from collections import Counter

    lines = []
    kind = structure_kind(atoms)
    extra = ""
    if kind == "slab" and "p2s_miller" in atoms.info:
        extra = f" ({atoms.info['p2s_miller']}), termination {atoms.info.get('p2s_termination', '?')}"
    lines.append(f"{atoms.get_chemical_formula()} - {len(atoms)} atoms - {kind}{extra}")
    a, b, c, al, be, ga = atoms.cell.cellpar()
    lines.append(f"cell a={a:.3f} b={b:.3f} c={c:.3f} A, alpha={al:.1f} beta={be:.1f} gamma={ga:.1f}, "
                 f"pbc={''.join('T' if p else 'F' for p in atoms.pbc)}")
    mols = get_molecules(atoms)
    n_sub = int(substrate_mask(atoms).sum())
    counts = Counter((m["name"], m.get("role", "?")) for m in mols)
    mol_txt = ", ".join(f"{n} x {name} ({role})" for (name, role), n in counts.items()) or "none"
    lines.append(f"substrate {n_sub} atoms; molecules: {mol_txt}")
    fixed = sum(len(c.index) for c in atoms.constraints if isinstance(c, FixAtoms))
    if fixed:
        lines.append(f"fixed atoms: {fixed}")
    coordination = substrate_coordination(atoms)
    if coordination:
        lines.append(f"substrate coordination: {coordination}")
    profile = composition_profile(atoms)
    if profile:
        lines.append(profile)
    if any(atoms.pbc):
        lines.append(f"vacuum gap: {vacuum_gap(atoms):.1f} A")
    if len(atoms) > 1:
        r = rcov(atoms.get_chemical_symbols())
        i, j, D = neighbor_pairs(atoms, r)
        d = np.linalg.norm(D, axis=1)
        if len(d):
            ratio = d / (r[i] + r[j])
            k = int(np.argmin(ratio))
            lines.append(f"closest pair: {atoms[i[k]].symbol}{i[k]}-{atoms[j[k]].symbol}{j[k]} {d[k]:.2f} A "
                         f"({ratio[k]:.2f} x covalent-radius sum)")
    contacts = binding_contacts(atoms)
    if contacts:
        lines.append("binding contacts: " + " | ".join(contacts[:12]) + (" | ..." if len(contacts) > 12 else ""))
    for key in ("p2s_warning",):
        if key in atoms.info:
            lines.append(f"note: {atoms.info[key]}")
    return "\n".join(lines)


def grouped_order(atoms: Atoms) -> np.ndarray:
    """Atom order with each element in one block (blocks in order of first appearance, original
    order kept inside a block), as POSCAR expects."""
    sym = atoms.get_chemical_symbols()
    first: dict[str, int] = {}
    for k, s in enumerate(sym):
        first.setdefault(s, k)
    return np.array(sorted(range(len(sym)), key=lambda k: (first[sym[k]], k)), dtype=int)


def export_structure(atoms: Atoms, out_dir: str | os.PathLike, stem: str,
                     formats: Sequence[str] = ("poscar",)) -> dict[str, str]:
    """Write `stem`.vasp (POSCAR, species grouped, FixAtoms as selective dynamics) and any of
    .cif/.extxyz, plus `stem`.json holding the per-atom roles and molecules in file order.
    Returns {format: path}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    unknown = set(formats) - set(EXTENSIONS)
    if unknown:
        raise ValueError(f"unknown formats {sorted(unknown)}; choose from {sorted(EXTENSIONS)}")
    ordered = clean_arrays(atoms[grouped_order(atoms)])
    files: dict[str, str] = {}
    for fmt in formats:
        path = out / f"{stem}.{EXTENSIONS[fmt]}"
        if fmt == "poscar":
            write(path, ordered, format="vasp", direct=True, sort=False)
        else:
            write(path, ordered, format=fmt)
        files[fmt] = str(path)
    fixed = sorted(int(i) for c in ordered.constraints if isinstance(c, FixAtoms) for i in c.index)
    meta = {"stem": stem, "formula": ordered.get_chemical_formula(), "n_atoms": len(ordered),
            "kind": structure_kind(ordered), "species": ordered.get_chemical_symbols(),
            "tags": [int(t) for t in ordered.get_tags()], "mol_id": [int(m) for m in mol_ids(ordered)],
            "fixed": fixed, "molecules": get_molecules(ordered),
            "info": {k: v for k, v in ordered.info.items() if k.startswith("p2s_") and k != "p2s_molecules"}}
    meta_path = out / f"{stem}.json"
    meta_path.write_text(json.dumps(meta, indent=1, default=_json_default))
    files["metadata"] = str(meta_path)
    return files


def _json_default(obj):
    """numpy values that extxyz round trips put into atoms.info."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")
