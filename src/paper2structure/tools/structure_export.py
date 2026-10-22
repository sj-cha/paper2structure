from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.geometry import get_distances
from ase.io import write

from paper2structure.tools._geometry import (clean_arrays, coordination_pairs, get_molecules, hull_depth, mol_ids,
                                             neighbor_pairs, rcov, set_molecules, solvent_mask, substrate_mask)

EXTENSIONS = {"poscar": "vasp", "cif": "cif", "extxyz": "extxyz"}
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")  # structure names and plan IDs: safe as file names
EMIT_DIR = "structures"  # where emit() writes, relative to the working directory
MANIFEST = "manifest.json"  # emit()'s record of every structure and its script, in EMIT_DIR


def structure_path(root: str | os.PathLike, name: str) -> Path:
    """The extxyz emit() writes for `name` when run in `root`."""
    return Path(root) / EMIT_DIR / f"{name}.extxyz"


def _calling_script() -> str:
    """The script that is running, relative to the working directory ("<inline>" for python -c / stdin)."""
    arg = sys.argv[0] if sys.argv else ""
    if arg in ("", "-c", "-") or not os.path.isfile(arg):
        return "<inline>"
    path = os.path.abspath(arg)
    rel = os.path.relpath(path)
    return path if rel.startswith("..") else rel


def emit(atoms: Atoms, name: str, plan_id: str, notes: str = "", system_ids: Optional[Sequence[str]] = None) -> str:
    """Register a finished structure for the plan `plan_id`: writes structures/<name>.extxyz (all
    per-atom metadata), structures/<name>.vasp (POSCAR) and its .json sidecar under the working
    directory, and records the structure and the emitting script in structures/manifest.json.
    Emitting the same name again replaces it.

    name: short identifier (letters, digits, "_", "-", "."), unique per structure.
    system_ids: the paper's systems this structure models (e.g. ["S2"]) when the plan covers
        several; the plan's checks scoped to other systems are then not applied to it.
    """
    if not NAME.match(name) or name == Path(MANIFEST).stem:
        raise ValueError(f"invalid structure name {name!r}: use letters, digits, '_', '-', '.' (max 64)")
    out = Path(EMIT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    atoms = clean_arrays(atoms.copy())
    label = atoms.info.get("p2s_kind")
    if label and label != "bulk" and len(atoms) and _inferred_kind(atoms) != label:
        print(f"[emit] warning: {name} carries the builder label p2s_kind={label!r} but its geometry is that of a "
              f"{_inferred_kind(atoms)}; validation treats it as a {label}. When combining structures start from the "
              "one whose kind the result has (slab + particle for a supported particle), or set atoms.info['p2s_kind'] "
              "to what it is")
    script = _calling_script()
    systems = [str(s) for s in (system_ids or [])]
    atoms.info.update({"p2s_name": name, "p2s_plan_id": str(plan_id), "p2s_notes": str(notes)})
    if systems:
        atoms.info["p2s_system_ids"] = ",".join(systems)
    path = out / f"{name}.extxyz"
    write(path, atoms, format="extxyz")
    export_structure(atoms, out, name, formats=("poscar",))
    manifest = out / MANIFEST
    entries = json.loads(manifest.read_text()) if manifest.exists() else []
    entries = [e for e in entries if e["name"] != name]
    entries.append({"name": name, "plan_id": str(plan_id), "path": path.name, "script": script, "notes": str(notes),
                    "system_ids": systems, "formula": atoms.get_chemical_formula(), "n_atoms": len(atoms)})
    manifest.write_text(json.dumps(entries, indent=2))
    print(f"[emit] {name} (plan {plan_id}): {atoms.get_chemical_formula()}, {len(atoms)} atoms -> {path}")
    return str(path)


OPEN_GAP = 5.0  # A of empty space along a cell vector that marks it as vacuum

# Shape of a structure. Every cell vector is measured as if periodic, whatever atoms.pbc says:
# exported POSCAR and CIF files are periodic along all three, so a slab ASE leaves non-periodic
# along its normal still has exactly the vacuum its POSCAR will have. Positions are read on the
# circle of fractional coordinates, so a structure wrapped across the cell boundary is one piece.


def _structure_atoms(atoms: Atoms) -> np.ndarray:
    """The atoms that make up the structure's shape: all but solvent, which fills the vacuum by
    design (all atoms when there is nothing else)."""
    keep = ~solvent_mask(atoms)
    return np.where(keep)[0] if keep.any() else np.arange(len(atoms))


def _fractional(atoms: Atoms) -> np.ndarray:
    return np.linalg.solve(atoms.cell.complete().T, atoms.positions.T).T


def _widest_arc(f: np.ndarray) -> tuple[float, float]:
    """(middle, width) of the widest empty arc between fractional coordinates on the unit circle."""
    fr = np.sort(np.mod(f, 1.0))
    arcs = np.diff(np.append(fr, fr[0] + 1.0))  # the empty arc after each point; the last one wraps
    i = int(np.argmax(arcs))
    return float(fr[i] + arcs[i] / 2), float(arcs[i])


def _axis_heights(atoms: Atoms, k: int, ref: np.ndarray) -> tuple[float, np.ndarray]:
    """Along cell vector k: the empty space (A) between the `ref` atoms and their periodic image,
    and every atom's height (A, along the normal of the other two vectors) above the bottom of the
    ref atoms. A zero cell vector has an infinite gap."""
    cell = atoms.cell.complete()
    normal = np.cross(cell[(k + 1) % 3], cell[(k + 2) % 3])
    normal /= np.linalg.norm(normal)
    if atoms.cell.lengths()[k] < 1e-6 or len(ref) == 0:
        h = atoms.positions @ normal
        gap = float("inf") if atoms.cell.lengths()[k] < 1e-6 else abs(float(cell[k] @ normal))
        return gap, h - (h[ref].min() if len(ref) else 0.0)
    spacing = abs(float(cell[k] @ normal))
    f = _fractional(atoms)[:, k]
    middle, width = _widest_arc(f[ref])
    return width * spacing, (np.mod(f - middle, 1.0) - width / 2) * spacing


def _axis_gaps(atoms: Atoms, ref: Optional[np.ndarray] = None) -> dict[int, float]:
    """Empty space (A) between the structure (`ref`, default: all but solvent) and its periodic
    image along each cell vector."""
    ref = _structure_atoms(atoms) if ref is None else ref
    return {k: _axis_heights(atoms, k, ref)[0] for k in range(3)}


def _unwrapped(atoms: Atoms, ref: np.ndarray) -> np.ndarray:
    """Cartesian positions with the `ref` atoms in one piece: along each cell vector, coordinates
    are counted from the middle of the widest empty arc."""
    frac = _fractional(atoms)
    for k in range(3):
        if atoms.cell.lengths()[k] >= 1e-6 and len(ref):
            middle, _ = _widest_arc(frac[ref, k])
            frac[:, k] = middle + np.mod(frac[:, k] - middle, 1.0)
    return frac @ atoms.cell.complete()


def _inferred_kind(atoms: Atoms) -> str:
    gaps = _axis_gaps(atoms)
    n_open = sum(1 for k in range(3) if not atoms.pbc[k] or gaps[k] > OPEN_GAP)
    return {0: "bulk", 1: "slab", 2: "wire"}.get(n_open, "nanocrystal")


def structure_kind(atoms: Atoms) -> str:
    """The builders' label (atoms.info["p2s_kind"]) or, for structures built with custom code, one
    inferred from the vacuum around the structure (solvent left out): "slab" (one open direction),
    "wire" (two), "nanocrystal" (three) or "bulk". A cell vector is open when it is not periodic or
    holds more than OPEN_GAP A of vacuum. A build_bulk crystal stays "bulk" while its cell is a
    supercell of the crystal's lattice (molecules added or atoms removed); once the cell changed (a
    supercell given vacuum, ase.build.surface of it) the geometry decides."""
    kind = atoms.info.get("p2s_kind")
    if not kind:
        return _inferred_kind(atoms)
    if kind == "bulk" and not _lattice_supercell(atoms):
        return _inferred_kind(atoms)
    return str(kind)


def _lattice_supercell(atoms: Atoms) -> bool:
    """Whether the cell is an integer supercell of the lattice build_bulk recorded (True when there
    is no record: a custom "bulk" label is trusted)."""
    lattice = atoms.info.get("p2s_lattice")
    if lattice is None:
        return True
    lattice = np.asarray(lattice, dtype=float).reshape(3, 3)
    if atoms.cell.rank < 3 or abs(np.linalg.det(lattice)) < 1e-6:
        return False
    n = atoms.cell.array @ np.linalg.inv(lattice)
    return bool(np.allclose(n, np.round(n), atol=1e-3) and round(abs(np.linalg.det(n))) >= 1)


def slab_axis(atoms: Atoms) -> int:
    """The cell vector across a slab's vacuum: its only non-periodic vector, else the one with the
    most vacuum."""
    nonperiodic = [k for k in range(3) if not atoms.pbc[k]]
    if len(nonperiodic) == 1:
        return nonperiodic[0]
    gaps = _axis_gaps(atoms)
    return max(range(3), key=lambda k: (gaps[k], k))


def wire_axis(atoms: Atoms) -> int:
    """The cell vector along a wire: its only periodic vector without vacuum."""
    gaps = _axis_gaps(atoms)
    closed = [k for k in range(3) if atoms.pbc[k] and gaps[k] <= OPEN_GAP]
    return closed[0] if len(closed) == 1 else min(range(3), key=lambda k: (not atoms.pbc[k], gaps[k]))


def vacuum_gap(atoms: Atoms) -> float:
    """Vacuum (A) between the structure and its nearest periodic image (solvent and adsorbates
    count as filling it): along the surface normal for slabs, across the smaller open direction for
    wires, the smallest over all cell vectors otherwise."""
    gaps = _axis_gaps(atoms, np.arange(len(atoms)))
    kind = structure_kind(atoms)
    if kind == "slab":
        return float(gaps[slab_axis(atoms)])
    if kind == "wire":
        along = wire_axis(atoms)
        return float(min(g for k, g in gaps.items() if k != along))
    return float(min(gaps.values()))


INTERIOR_DEPTH = 4.0  # A: substrate atoms this far below the surface are interior (bulk-like) sites


def substrate_depth(atoms: Atoms, sub: np.ndarray) -> np.ndarray:
    """Distance (A) of every atom below the surface of the substrate atoms `sub`: along the normal
    for slabs, below the convex hull across the axis for wires, below the convex hull for particles,
    infinite for bulk."""
    kind = structure_kind(atoms)
    if kind == "bulk":
        return np.full(len(atoms), np.inf)
    if len(sub) == 0:
        return np.zeros(len(atoms))
    if kind == "slab":
        _, h = _axis_heights(atoms, slab_axis(atoms), sub)
        return np.minimum(h - h[sub].min(), h[sub].max() - h)
    pos = _unwrapped(atoms, sub)
    if kind == "wire":
        pos = pos @ _cross_section(atoms.cell.complete()[wire_axis(atoms)]).T
    return hull_depth(pos, pos[sub])


def _cross_section(axis: np.ndarray) -> np.ndarray:
    """Two orthonormal vectors perpendicular to `axis` (rows)."""
    a = axis / np.linalg.norm(axis)
    u = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    return np.array([u, np.cross(a, u)])


def composition_profile(atoms: Atoms, layer_tol: float = 0.6, shell_width: float = 3.0) -> str:
    """Where each element sits in the substrate: layer compositions from the top for slabs, radial
    shells from the center for particles (e.g. a core-shell particle reads "0-3 A Cd4Se4 | ...")
    and from the axis for wires."""
    sub = np.where(substrate_mask(atoms))[0]
    if len(sub) < 2:
        return ""
    sym = np.array(atoms.get_chemical_symbols())[sub]
    kind = structure_kind(atoms)
    if kind == "slab":
        h = _axis_heights(atoms, slab_axis(atoms), sub)[1][sub]
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
        rel = _unwrapped(atoms, sub)[sub]
        rel = rel - rel.mean(axis=0)
        if kind == "wire":
            rel = rel @ _cross_section(atoms.cell.complete()[wire_axis(atoms)]).T
        r = np.linalg.norm(rel, axis=1)
        width = max(shell_width, float(r.max()) / 6 + 1e-9)
        bins = []
        for k in range(int(np.ceil((r.max() + 1e-9) / width))):
            inside = (r >= k * width) & (r < (k + 1) * width)
            if inside.any():
                bins.append(f"{k * width:.0f}-{(k + 1) * width:.0f} A {Atoms(list(sym[inside])).get_chemical_formula()}")
        return f"radial composition from the {'axis' if kind == 'wire' else 'center'}: " + " | ".join(bins)
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
    if kind != "bulk" and atoms.cell.rank == 3:
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
    order = grouped_order(atoms)
    ordered = clean_arrays(atoms[order])
    _reorder_records(atoms, ordered, order)
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


def _reorder_records(atoms: Atoms, ordered: Atoms, order: np.ndarray) -> None:
    """Molecule records name their binding atoms by position within the molecule; `ordered` is
    `atoms` reordered (order[k] = old index of new atom k), so those positions are translated."""
    molecules = get_molecules(atoms)
    if not any(m.get("binding_atoms") for m in molecules):
        return
    new_index = np.empty(len(order), dtype=int)
    new_index[order] = np.arange(len(order))
    old_ids, new_ids = mol_ids(atoms), mol_ids(ordered)
    records = []
    for m in molecules:
        if m.get("binding_atoms"):
            old = np.where(old_ids == m["mol_id"])[0]
            local = {int(g): k for k, g in enumerate(np.where(new_ids == m["mol_id"])[0])}
            m = {**m, "binding_atoms": [local[int(new_index[old[b]])] for b in m["binding_atoms"] if b < len(old)]}
        records.append(m)
    set_molecules(ordered, records)


def _json_default(obj):
    """numpy values that extxyz round trips put into atoms.info."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")
