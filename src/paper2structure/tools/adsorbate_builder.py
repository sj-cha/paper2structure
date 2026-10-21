"""Adsorbate construction ported from PerovKit's Ligand/BindingMotif (https://github.com/sj-cha/PerovKit),
generalized away from perovskite-specific sites."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

from paper2structure.tools._geometry import default_seed, rotation_about_axis, rotation_from_u_to_v


@dataclass
class BindingMotif:
    """How a molecule anchors to a surface.

    Args:
        atoms: element symbols of the 1 or 2 binding atoms, e.g. ["S"] or ["O", "O"].
        anchor: SMARTS of the binding group, e.g. "C(=O)[O-]" or "P(=O)([O-])O". Binding atoms
            are searched among the atoms of its first match and their bonded neighbours, preferring
            the lowest-coordinated ones.
        indices: explicit binding-atom indices (RDKit order, hydrogens appended after heavy atoms),
            one per entry of `atoms`; when given, `anchor` is not needed.
    """
    atoms: list[str]
    anchor: Optional[str] = None
    indices: Optional[list[int]] = None

    def __post_init__(self):
        if not 1 <= len(self.atoms) <= 2:
            raise NotImplementedError(f"binding motifs need 1 or 2 atoms; got {self.atoms}")
        if self.anchor is None and self.indices is None:
            raise ValueError("BindingMotif: give an anchor SMARTS or explicit indices")
        if self.anchor is not None and Chem.MolFromSmarts(self.anchor) is None:
            raise ValueError(f"BindingMotif: invalid anchor SMARTS {self.anchor!r}")
        if self.indices is not None:
            self.indices = [int(i) for i in self.indices]
            if len(self.indices) != len(self.atoms) or len(set(self.indices)) != len(self.indices):
                raise ValueError("BindingMotif: give one distinct index per binding element")


@dataclass
class Adsorbate:
    """A molecule with 3D coordinates, oriented for placement: the binding atoms' centroid sits at
    the origin, the body points along +z and, for two binding atoms, their axis lies along x.
    Build with Adsorbate.from_smiles. One binding atom means monodentate binding, two mean bidentate
    (chelating or bridging). Monatomic adsorbates such as a transferred proton ("[H+]"), "[O]" or
    "[Cl-]" need no BindingMotif: their only atom binds."""
    atoms: Atoms
    mol: Chem.Mol
    smiles: str
    charge: int
    name: str
    binding_motif: Optional[BindingMotif] = None
    binding_atoms: list[int] = field(default_factory=list)

    def __post_init__(self):
        if [a.GetSymbol() for a in self.mol.GetAtoms()] != self.atoms.get_chemical_symbols():
            raise ValueError("mol and atoms must list the same elements in the same order")
        if self.binding_motif is None and len(self.atoms) == 1:
            self.binding_atoms = [0]
            self.atoms.set_positions(np.zeros((1, 3)))
        elif self.binding_motif is not None:
            self._find_binding_atoms()
            self._orient()

    @classmethod
    def from_smiles(cls, smiles: str, binding_motif: Optional[BindingMotif] = None,
                    name: Optional[str] = None, seed: Optional[int] = None,
                    optimize: bool = False) -> Adsorbate:
        """Embed a molecule from SMILES (formal charges set the protonation state).

        optimize=False sets every rotatable bond to 180 deg (extended chain) and relaxes bonds and
        angles with UFF at fixed torsions; optimize=True relaxes fully with UFF.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"unparseable SMILES {smiles!r}")
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = default_seed(seed)
        if AllChem.EmbedMolecule(mol, params) < 0:
            params.useRandomCoords = True
            if AllChem.EmbedMolecule(mol, params) < 0:
                raise ValueError(f"RDKit could not embed {smiles!r}")
        if mol.GetNumAtoms() > 2:
            try:
                if optimize:
                    AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
                else:
                    _extend_torsions(mol)
            except Exception as e:  # noqa: BLE001 - some charged species lack UFF types
                print(f"[Adsorbate] UFF step skipped for {smiles!r}: {e}")
        atoms = Atoms([a.GetSymbol() for a in mol.GetAtoms()], positions=mol.GetConformer().GetPositions())
        return cls(atoms=atoms, mol=mol, smiles=smiles, charge=Chem.GetFormalCharge(mol),
                   name=name or smiles, binding_motif=binding_motif)

    def clone(self) -> Adsorbate:
        ads = copy.copy(self)
        ads.atoms = self.atoms.copy()
        ads.mol = Chem.Mol(self.mol)
        ads.binding_atoms = list(self.binding_atoms)
        return ads

    @property
    def anchor_pos(self) -> Optional[np.ndarray]:
        if not self.binding_atoms:
            return None
        return self.atoms.get_positions()[self.binding_atoms].mean(axis=0)

    def __str__(self) -> str:
        bind = ",".join(f"{self.atoms[i].symbol}{i}" for i in self.binding_atoms) or "none"
        return f"Adsorbate {self.name}: {self.smiles}, {len(self.atoms)} atoms, charge {self.charge:+d}, binding atoms {bind}"

    def _anchor_region(self) -> list[int]:
        matches = self.mol.GetSubstructMatches(Chem.MolFromSmarts(self.binding_motif.anchor))
        if not matches:
            raise ValueError(f"anchor {self.binding_motif.anchor!r} matches nothing in {self.smiles!r}")
        first = set(matches[0])
        if any(first.isdisjoint(m) for m in matches[1:]):
            print(f"[Adsorbate] anchor {self.binding_motif.anchor!r} matches several groups in {self.name}; "
                  f"using the first match {sorted(first)}")
        region = set(first)
        for i in first:
            region.update(n.GetIdx() for n in self.mol.GetAtomWithIdx(i).GetNeighbors())
        return sorted(region)

    def _find_binding_atoms(self) -> None:
        symbols = self.atoms.get_chemical_symbols()
        elems = list(self.binding_motif.atoms)
        if self.binding_motif.indices is not None:
            idx = list(self.binding_motif.indices)
            for i, e in zip(idx, elems):
                if not 0 <= i < len(symbols) or symbols[i] != e:
                    raise ValueError(f"binding index {i} is not a {e} atom of {self.smiles!r}")
            self.binding_atoms = idx
            return
        region = self._anchor_region()
        cn = np.array([a.GetDegree() for a in self.mol.GetAtoms()], dtype=int)
        pos = self.atoms.get_positions()
        if len(elems) == 1:
            cand = [i for i in region if symbols[i] == elems[0]]
            if not cand:
                raise ValueError(f"no {elems[0]} atom around anchor {self.binding_motif.anchor!r}")
            self.binding_atoms = [min(cand, key=lambda i: (cn[i], i))]
            return
        best = None
        for i in (i for i in region if symbols[i] == elems[0]):
            for j in (j for j in region if symbols[j] == elems[1] and j != i):
                key = (int(cn[i] + cn[j]), float(np.linalg.norm(pos[i] - pos[j])))
                if best is None or key < best[0]:
                    best = (key, (i, j))
        if best is None:
            raise ValueError(f"no distinct {elems} pair around anchor {self.binding_motif.anchor!r}")
        self.binding_atoms = list(best[1])

    def _orient(self, n_angles: int = 720) -> None:
        coords = self.atoms.get_positions()
        b = self.binding_atoms
        coords = coords - coords[b].mean(axis=0)
        others = np.delete(coords, b, axis=0)
        if len(others) == 0:
            self.atoms.set_positions(coords)
            return
        if len(b) == 1:
            axis = others.mean(axis=0)
            if np.linalg.norm(axis) < 1e-3:  # centroid on the binding atom (e.g. NH4+)
                axis = others[int(np.argmax(np.einsum("ij,ij->i", others, others)))]
            coords = coords @ rotation_from_u_to_v(axis, np.array([0.0, 0.0, 1.0])).T
            self.atoms.set_positions(coords)
            return
        ex = np.array([1.0, 0.0, 0.0])
        coords = coords @ rotation_from_u_to_v(coords[b[1]] - coords[b[0]], ex).T
        others = np.delete(coords, b, axis=0)
        thetas = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
        scores = [(others @ rotation_about_axis(ex, t).T)[:, 2].sum() for t in thetas]
        coords = coords @ rotation_about_axis(ex, thetas[int(np.argmax(scores))]).T
        self.atoms.set_positions(coords)


def _extend_torsions(mol: Chem.Mol) -> None:
    """Set rotatable dihedrals to 180 deg and relax bonds/angles with the torsions held."""
    rotatable = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
    conf = mol.GetConformer()
    quads = []
    for i, j in mol.GetSubstructMatches(rotatable):
        ni = [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() != j]
        nj = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != i]
        if ni and nj:
            quad = (ni[0], i, j, nj[0])
            rdMolTransforms.SetDihedralDeg(conf, *quad, 180.0)
            quads.append(quad)
    ff = AllChem.UFFGetMoleculeForceField(mol)
    for a, b, c, d in quads:
        ff.UFFAddTorsionConstraint(a, b, c, d, False, 180.0, 180.0, 1.0e5)
    ff.Minimize(maxIts=200)
