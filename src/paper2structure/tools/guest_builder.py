"""Molecules inside a crystal framework, for any framework: organic cations in layered or
three-dimensional halides and oxides, structure-directing cations in zeolites, intercalated
ammonium in clays, guests and solvent in porous frameworks, molecular salts. The placement follows
generic non-covalent chemistry, no material-specific rule: the molecule's anchor (its charged or
binding atom) sits at van der Waals contact with as many framework atoms of opposite character as
possible (anions for a cation or a hydrogen-bond donor, cations for an anion, any atom for a
neutral molecule without donors), the molecule is turned so that the most N-H/O-H/S-H donors point
at acceptors (hydrogen bonds), and nothing clashes. With the crystal's space group, one molecule is
placed per orbit and its symmetry images complete the set, so the filled crystal keeps the
framework's symmetry."""
from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, vdw_radii
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from paper2structure.tools._geometry import (ANION_FORMERS, cation_elements, default_seed, get_molecules,
                                             image_shifts, mol_ids)
from paper2structure.tools._symmetry import orbit, symmetry_operations
from paper2structure.tools.adsorbate_builder import Adsorbate
from paper2structure.tools.adsorption_builder import add_molecule

CLASH_HEAVY = 0.85  # x van der Waals sum: closest allowed contact between heavy atoms
CLASH_H = 0.80  # ... and for other contacts involving hydrogen
CLASH_HBOND = 0.60  # ... and between a donor H and an acceptor: short hydrogen bonds (O-H...O 1.7 A,
                    # N-H...Cl 2.3 A) are 0.65-0.8 x the van der Waals sum
CONTACT_SHELL = 0.6  # A beyond the van der Waals sum that still counts as a contact of the anchor
HBOND_ANGLE = 130.0  # minimum D-H...A angle (deg)
HBOND_SLACK = 0.1  # A below the H...A van der Waals sum
SAME_ATOM = 0.05  # A: a symmetry image this close to an atom of the same molecule is that atom
POSES_PER_SITE = 200  # best poses of a site tried for a clash-free orbit
SHIFTS = np.array([[0, 0, 0], [0.3, 0, 0], [-0.3, 0, 0], [0, 0.3, 0], [0, -0.3, 0], [0, 0, 0.3], [0, 0, -0.3]])


def _vdw(symbols) -> np.ndarray:
    return np.array([vdw_radii[atomic_numbers[s]] for s in symbols], dtype=float)


def _anchor(ads: Adsorbate) -> int:
    """The binding atom of the motif, else the charged heavy atom, else the first heavy atom."""
    if ads.binding_atoms:
        return int(ads.binding_atoms[0])
    charged = [a.GetIdx() for a in ads.mol.GetAtoms() if a.GetFormalCharge() != 0 and a.GetSymbol() != "H"]
    return charged[0] if charged else next(a.GetIdx() for a in ads.mol.GetAtoms() if a.GetSymbol() != "H")


def _donors(ads: Adsorbate) -> list[tuple[int, int]]:
    """(hydrogen, donor) pairs: H bonded to N, O or S."""
    out = []
    for b in ads.mol.GetBonds():
        x, y = b.GetBeginAtom(), b.GetEndAtom()
        for h, d in ((x, y), (y, x)):
            if h.GetSymbol() == "H" and d.GetSymbol() in ("N", "O", "S"):
                out.append((h.GetIdx(), d.GetIdx()))
    return out


class _Env:
    """The atoms already in the cell and their neighbouring periodic images, for contact queries;
    `acceptor` marks the framework's electronegative atoms (hydrogen-bond acceptors)."""

    def __init__(self, atoms: Atoms):
        shifts = image_shifts(atoms.cell.array, atoms.pbc)
        symbols = np.array(atoms.get_chemical_symbols())
        self.points = (atoms.positions[None] + shifts[:, None]).reshape(-1, 3)
        index = np.tile(np.arange(len(atoms)), len(shifts))
        self.symbols = symbols[index]
        self.vdw = _vdw(symbols)[index]
        self.is_h = self.symbols == "H"
        self.acceptor = np.isin(self.symbols, list(ANION_FORMERS)) & (mol_ids(atoms) < 0)[index]
        self.tree = cKDTree(self.points)

    def query(self, xyz: np.ndarray, k: int):
        d, i = self.tree.query(xyz, k=min(k, len(self.points)))
        return d.reshape(*xyz.shape[:-1], -1), i.reshape(*xyz.shape[:-1], -1)


def _clash_free(env: _Env, xyz: np.ndarray, vdw: np.ndarray, is_h: np.ndarray, donor_h: np.ndarray) -> np.ndarray:
    """For poses (P, m, 3): no atom closer than the clash factor x van der Waals sum to the environment."""
    d, i = env.query(xyz, k=8)
    factor = np.where(is_h[None, :, None] | env.is_h[i], CLASH_H, CLASH_HEAVY)
    factor = np.where(donor_h[None, :, None] & env.acceptor[i], CLASH_HBOND, factor)
    return np.all(d >= factor * (vdw[None, :, None] + env.vdw[i]), axis=(1, 2))


def _hbonds(env: _Env, xyz: np.ndarray, donors: list[tuple[int, int]], k: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """For poses (P, m, 3): hydrogen bonds to framework acceptors (one per donor H at most) and
    their summed H...A length, among the k nearest atoms of each H (more when the molecule itself
    is in the environment)."""
    count = np.zeros(len(xyz), dtype=int)
    length = np.zeros(len(xyz))
    r_h = _vdw(["H"])[0]
    for h, dn in donors:
        hx, back = xyz[:, h], xyz[:, dn] - xyz[:, h]  # H and the H->donor vector
        d, i = env.query(hx, k=k)
        va = env.points[i] - hx[:, None, :]
        cos = np.einsum("pj,pkj->pk", back, va) / (np.linalg.norm(back, axis=1)[:, None] * np.linalg.norm(va, axis=2) + 1e-12)
        ok = env.acceptor[i] & (d <= r_h + env.vdw[i] - HBOND_SLACK) & (np.degrees(np.arccos(np.clip(cos, -1, 1))) >= HBOND_ANGLE)
        best = np.where(ok, d, np.inf).min(axis=1)
        hit = np.isfinite(best)
        count += hit
        length += np.where(hit, best, 0.0)
    return count, length


def _anchor_sites(env: _Env, cell: np.ndarray, anchor: str, partner: np.ndarray, grid: float, n_sites: int,
                  ops: Optional[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
    """Cartesian points of the void where the anchor atom touches the most partner atoms (van der
    Waals contact, nothing closer), best first, at least 1 A apart and from each other's symmetry images."""
    n = [max(2, int(math.ceil(np.linalg.norm(v) / grid))) for v in cell]
    frac = np.stack(np.meshgrid(*[np.arange(k) / k for k in n], indexing="ij"), axis=-1).reshape(-1, 3)
    xyz = frac @ cell
    d, i = env.query(xyz, k=12)
    contact = _vdw([anchor])[0] + env.vdw[i]
    factor = np.where(env.is_h[i], CLASH_H, CLASH_HEAVY)
    touching = partner[i] & (d <= contact + CONTACT_SHELL)
    score = np.where(np.all(d >= factor * contact, axis=1), touching.sum(axis=1), -1)
    tight = np.where(touching, d - contact, 0.0).sum(axis=1)
    chosen = []
    taken = np.zeros((0, 3))  # the chosen points and their symmetry images (fractional)
    for k in np.lexsort((tight, -score)):
        if score[k] <= 0 or len(chosen) >= n_sites:
            break
        f = frac[k]
        if len(taken) and np.min(np.linalg.norm(((f - taken + 0.5) % 1.0 - 0.5) @ cell, axis=1)) < 1.0:
            continue
        chosen.append(xyz[k])
        images = f[None] if ops is None else np.mod(np.einsum("nij,j->ni", ops[0], f) + ops[1], 1.0)
        taken = np.vstack([taken, images])
    return chosen


def _images(xyz: np.ndarray, ops: tuple[np.ndarray, np.ndarray], cell: np.ndarray, anchor: int,
            limit: int) -> Optional[list[np.ndarray]]:
    """The distinct symmetry images (Cartesian) of one molecule, each kept whole: it moves with its
    anchor into the cell. An image that coincides with an earlier one atom by atom (a molecule on a
    special position) is the same molecule. None when there are more than `limit`."""
    frac = xyz @ np.linalg.inv(cell)
    unique: list[np.ndarray] = []
    for r, t in zip(*ops):
        img = frac @ r.T + t
        img -= np.floor(img[anchor])
        anchors = np.array([u[anchor] for u in unique]) if unique else np.zeros((0, 3))
        near = [k for k in range(len(unique))  # only an image with its anchor on this one's can coincide
                if np.linalg.norm(((img[anchor] - anchors[k] + 0.5) % 1.0 - 0.5) @ cell) < 1.0]
        if not any(np.all(np.linalg.norm(((img[:, None] - unique[k][None] + 0.5) % 1.0 - 0.5) @ cell, axis=2)
                          .min(axis=1) < SAME_ATOM) for k in near):
            unique.append(img)
            if len(unique) > limit:
                return None
    return [u @ cell for u in unique]


def _images_clash(images: list[np.ndarray], cell: np.ndarray, vdw: np.ndarray, is_h: np.ndarray,
                  hbond: np.ndarray) -> bool:
    """Whether two symmetry images of a molecule clash; `hbond` marks the (donor H, acceptor) atom
    pairs that may form a hydrogen bond between images."""
    inv = np.linalg.inv(cell)
    limit = np.where(is_h[:, None] | is_h[None, :], CLASH_H, CLASH_HEAVY) * (vdw[:, None] + vdw[None, :])
    limit = np.where(hbond, CLASH_HBOND * (vdw[:, None] + vdw[None, :]), limit)
    for a in range(len(images)):
        for b in range(a + 1, len(images)):
            f = (images[a][:, None, :] - images[b][None, :, :]) @ inv
            if np.any(np.linalg.norm((f - np.round(f)) @ cell, axis=2) < limit):
                return True
    return False


def _check_symmetry(atoms: Atoms, ops: tuple[np.ndarray, np.ndarray], symbol: str) -> None:
    """Every operation maps the framework onto itself: the cell is the conventional cell of `symbol`."""
    sym = np.array(atoms.get_chemical_symbols())
    frac, cell = atoms.get_scaled_positions(), atoms.cell.array
    for r, t in zip(*ops):
        img = frac @ r.T + t
        for el in set(sym):
            diff = (img[sym == el][:, None] - frac[sym == el][None] + 0.5) % 1.0 - 0.5
            if np.linalg.norm(diff @ cell, axis=2).min(axis=1).max() > 0.1:
                raise ValueError(f"the crystal is not invariant under the operations of {symbol} in this cell: fill "
                                 "the conventional cell build_bulk returns (in the same setting) before making a "
                                 "supercell, or pass space_group=None")


def fill_cavities(crystal: Atoms, molecule: Union[Adsorbate, str], count: int, name: Optional[str] = None,
                  space_group: Optional[str] = "auto", orientations: int = 400, sites: int = 16, grid: float = 0.4,
                  seed: Optional[int] = None) -> Atoms:
    """Place `count` molecules (per cell) in the cavities of a crystal framework and return the
    crystal with them registered as guests (add_molecule, role "guest").

    molecule: an Adsorbate (its binding atom is the anchor) or a SMILES with formal charges, e.g.
        "CCCC[NH3+]", "C[N+](C)(C)C", "[O-]C=O", "O" (the charged atom is the anchor, else the
        first heavy atom). The molecule stays the rigid RDKit conformer.
    space_group: the framework's space group in the cell's setting; "auto" reads the symbol
        build_bulk stored (atoms.info["p2s_space_group"]); None places the molecules one by one.
        With a space group each placed molecule brings its symmetry images. Molecules no orbit
        fits (e.g. 4 per cell when every orbit holds 8) are placed one by one: the full structure
        then has lower symmetry than the framework, which keeps it.
    orientations, sites, grid: search sizes (random orientations per site, candidate anchor
        sites, grid spacing in A of the site search).

    Call it on the conventional cell, larger molecules first, and make supercells afterwards.
    Raises ValueError when the cavities cannot take `count` molecules without clashes.
    """
    rng = np.random.default_rng(default_seed(seed))
    ads = Adsorbate.from_smiles(molecule, seed=seed) if isinstance(molecule, str) else molecule
    name = name or ads.name
    a0 = _anchor(ads)
    symbols = ads.atoms.get_chemical_symbols()
    vdw, is_h = _vdw(symbols), np.array([s == "H" for s in symbols])
    donors = _donors(ads)
    donor_h = np.isin(np.arange(len(symbols)), [h for h, _ in donors])
    cell = crystal.cell.array
    symbol = crystal.info.get("p2s_space_group") if space_group == "auto" else space_group
    ops = symmetry_operations(symbol) if symbol else None
    if ops is not None:
        _check_symmetry(crystal[mol_ids(crystal) < 0], ops, symbol)
    framework = set(np.array(crystal.get_chemical_symbols())[mol_ids(crystal) < 0])
    if ads.charge < 0:  # an anion's anchor touches cations; a cation's or a donor's touches anions
        partners = cation_elements(framework)
    elif ads.charge > 0 or donors:
        partners = framework & ANION_FORMERS
    else:  # a neutral molecule without donors: van der Waals contact with any framework atom
        partners = framework
    partners = partners or framework
    acceptor = np.isin(symbols, list(ANION_FORMERS))
    hbond = (donor_h[:, None] & acceptor[None, :]) | (acceptor[:, None] & donor_h[None, :])
    rotations = Rotation.concatenate([Rotation.identity(), Rotation.random(orientations - 1, random_state=rng)])
    body = np.stack([r.apply(ads.atoms.positions - ads.atoms.positions[a0]) for r in rotations])

    model, placed, broken = crystal, 0, 0
    while placed < count:
        env = _Env(model)
        left = count - placed
        best = None  # ((symmetric, H-bonds, -H...A length), molecules)
        for p in _anchor_sites(env, cell, symbols[a0], np.isin(env.symbols, list(partners)), grid, sites, ops):
            poses = ((p + SHIFTS)[:, None, None, :] + body[None]).reshape(-1, len(symbols), 3)
            poses = poses[_clash_free(env, poses, vdw, is_h, donor_h)]
            if not len(poses):
                continue
            nb, length = _hbonds(env, poses, donors)
            order = np.lexsort((length, -nb))
            choice = ((0, int(nb[order[0]]), -float(length[order[0]])), [poses[order[0]]])  # alone, no images
            # a molecule's orbit is at least its anchor's orbit over the atoms the anchor could map to
            twins = sum(s == symbols[a0] for s in symbols)
            fits = ops is not None and len(orbit(p @ np.linalg.inv(cell), *ops)) <= left * twins
            if fits:
                for k in order[:POSES_PER_SITE]:
                    images = _images(poses[k], ops, cell, a0, left)
                    if images is not None and not _images_clash(images, cell, vdw, is_h, hbond):
                        choice = ((1, int(nb[k]), -float(length[k])), images)
                        break
            elif ops is None:
                choice = ((1,) + choice[0][1:], choice[1])
            if best is None or choice[0] > best[0]:
                best = choice
        if best is None:
            raise ValueError(f"no clash-free cavity left for {name}: placed {placed} of {count} per cell; check the "
                             "framework (cell, Wyckoff sites) and the number of molecules")
        for xyz in best[1]:
            model = add_molecule(model, Atoms(symbols, positions=xyz), ads.smiles, name, role="guest")
        placed += len(best[1])
        broken += len(best[1]) if ops is not None and best[0][0] == 0 else 0
    note = (f"; {broken} placed without symmetry images (no orbit fits): the full structure has lower symmetry "
            f"than {symbol}, the framework keeps it" if broken else "")
    print(f"[fill_cavities] {placed} x {name} per cell{' in ' + symbol if symbol else ''}; hydrogen bonds per "
          f"molecule: {_hbond_summary(model, name, donors)}{note}")
    return model


def _hbond_summary(model: Atoms, name: str, donors) -> str:
    if not donors:
        return "no donors"
    env, ids = _Env(model), mol_ids(model)
    counts = [int(_hbonds(env, model.positions[ids == m["mol_id"]][None], donors, k=32)[0][0])
              for m in get_molecules(model) if m["name"] == name]
    values, n = np.unique(counts, return_counts=True)
    return ", ".join(f"{v} x{k}" for v, k in zip(values, n))
