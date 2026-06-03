#!/usr/bin/env python3
from __future__ import annotations

"""
replace_quinol_exactmap.py

Replace ligands in a GROMACS system by preserving the maximum possible number of
old-atom coordinates exactly.

Core idea
---------
1. Start from a seed atom map (usually ring/head atoms).
2. Expand that map automatically using the old/new ITP bond graphs.
3. For every residue to be replaced:
   - compute a global rigid transform from template -> old residue using the
     expanded map;
   - place ALL new atoms with that transform;
   - overwrite mapped new atoms with the EXACT coordinates of the matched old atoms;
   - relax only the unmatched branches/components with loose clash criteria;
   - keep the full residue whole before writing the GRO.

This branch is meant for cases such as UQOL8 -> MUQO8 where the molecules are
not identical but share a large common scaffold. The goal is to preserve headgroup
and tail placement instead of relying on a small ring-only fit.
"""

import argparse
import math
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:
    cKDTree = None


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------


def is_heavy(atomname: str) -> bool:
    return infer_element(atomname) != "H"


def infer_element(atomname: str) -> str:
    s = atomname.strip()
    s = re.sub(r"^[0-9]+", "", s)
    m = re.match(r"([A-Za-z]+)", s)
    if not m:
        return s[:1].upper() if s else "X"
    letters = m.group(1).upper()
    if letters.startswith("CL"):
        return "CL"
    if letters.startswith("BR"):
        return "BR"
    return letters[0]


# -----------------------------------------------------------------------------
# Box + GRO
# -----------------------------------------------------------------------------


class Box:
    def __init__(self, box_floats: Sequence[float]):
        vals = list(box_floats)
        if len(vals) not in (3, 9):
            raise ValueError(f"Unsupported .gro box format with {len(vals)} floats")

        if len(vals) == 3:
            v1 = np.array([vals[0], 0.0, 0.0], float)
            v2 = np.array([0.0, vals[1], 0.0], float)
            v3 = np.array([0.0, 0.0, vals[2]], float)
        else:
            # GROMACS triclinic: v1x v2y v3z v1y v1z v2x v2z v3x v3y
            v1 = np.array([vals[0], vals[3], vals[4]], float)
            v2 = np.array([vals[5], vals[1], vals[6]], float)
            v3 = np.array([vals[7], vals[8], vals[2]], float)

        self.v1, self.v2, self.v3 = v1, v2, v3
        self.B = np.column_stack([v1, v2, v3])
        self.invB = np.linalg.inv(self.B)

    def is_orthorhombic(self, tol: float = 1e-8) -> bool:
        off = self.B.copy()
        np.fill_diagonal(off, 0.0)
        return float(np.max(np.abs(off))) < tol

    def lengths(self) -> Tuple[float, float, float]:
        if self.is_orthorhombic():
            return (float(self.B[0, 0]), float(self.B[1, 1]), float(self.B[2, 2]))
        return (
            float(np.linalg.norm(self.v1)),
            float(np.linalg.norm(self.v2)),
            float(np.linalg.norm(self.v3)),
        )

    def wrap(self, coords: np.ndarray) -> np.ndarray:
        frac = (self.invB @ coords.T).T
        frac -= np.floor(frac)
        return (self.B @ frac.T).T

    def min_image(self, deltas: np.ndarray) -> np.ndarray:
        sh = deltas.shape
        d = deltas.reshape(-1, 3)
        frac = (self.invB @ d.T).T
        frac -= np.round(frac)
        d2 = (self.B @ frac.T).T
        return d2.reshape(sh)

    def translate_whole_into_box(self, coords: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        frac = self.invB @ anchor
        shift_frac = np.floor(frac)
        shift = self.B @ shift_frac
        return coords - shift


@dataclass
class Residue:
    resid: int
    resname: str
    atom_idx: np.ndarray


@dataclass
class Gro:
    title: str
    coords: np.ndarray
    resid: np.ndarray
    resname: List[str]
    atomname: List[str]
    box_line: str
    box: Box
    residues: List[Residue]


@dataclass
class AtomRec:
    nr: int
    atype: str
    resnr: int
    residu: str
    atom: str
    cgnr: int
    charge: float
    mass: float


@dataclass
class Template:
    atom_to_coord: Dict[str, np.ndarray]
    ordered_coords: np.ndarray


@dataclass
class ExpandedMap:
    seed_pairs: List[Tuple[str, str]]
    all_pairs: List[Tuple[str, str]]
    score_lines: List[str]


@dataclass
class BranchInfo:
    anchor_new: int
    root_new: int
    atoms: List[int]


# -----------------------------------------------------------------------------
# IO parsers
# -----------------------------------------------------------------------------


def read_gro(path: str | Path) -> Gro:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        title = f.readline().rstrip("\n")
        natoms = int(f.readline().strip())

        coords = np.zeros((natoms, 3), float)
        resid = np.zeros(natoms, int)
        resname = [""] * natoms
        atomname = [""] * natoms

        residues: List[Residue] = []
        cur_resid = None
        cur_resname = None
        cur_idx: List[int] = []

        for i in range(natoms):
            line = f.readline()
            r = int(line[0:5])
            rn = line[5:10].strip()
            an = line[10:15].strip()
            x = float(line[20:28])
            y = float(line[28:36])
            z = float(line[36:44])

            resid[i] = r
            resname[i] = rn
            atomname[i] = an
            coords[i] = (x, y, z)

            if cur_resid is None:
                cur_resid, cur_resname = r, rn
                cur_idx = [i]
            elif r == cur_resid and rn == cur_resname:
                cur_idx.append(i)
            else:
                residues.append(Residue(int(cur_resid), str(cur_resname), np.array(cur_idx, int)))
                cur_resid, cur_resname = r, rn
                cur_idx = [i]

        if cur_resid is not None:
            residues.append(Residue(int(cur_resid), str(cur_resname), np.array(cur_idx, int)))

        box_line = f.readline().rstrip("\n")
        box = Box([float(x) for x in box_line.split()])

    return Gro(title, coords, resid, resname, atomname, box_line, box, residues)


def write_gro_from_records(
    out_path: str | Path,
    title: str,
    atom_records: List[Tuple[int, str, str, np.ndarray]],
    box_line: str,
):
    out_path = Path(out_path)
    natoms = len(atom_records)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(title + "\n")
        f.write(f"{natoms:d}\n")
        atomnr = 1
        for (rid, rname, aname, xyz) in atom_records:
            rid5 = int(rid) % 100000
            anr5 = int(atomnr) % 100000
            atomnr += 1
            x, y, z = xyz
            f.write(f"{rid5:5d}{rname:<5s}{aname:>5s}{anr5:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")
        f.write(box_line + "\n")


def parse_itp_atoms(itp_path: str | Path) -> List[AtomRec]:
    atoms: List[AtomRec] = []
    in_atoms = False
    with Path(itp_path).open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith("["):
                in_atoms = line.lower().startswith("[ atoms")
                continue
            if not in_atoms:
                continue
            line = line.split(";", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            atoms.append(
                AtomRec(
                    nr=int(parts[0]),
                    atype=parts[1],
                    resnr=int(parts[2]),
                    residu=parts[3],
                    atom=parts[4],
                    cgnr=int(parts[5]),
                    charge=float(parts[6]),
                    mass=float(parts[7]),
                )
            )
    if not atoms:
        raise ValueError(f"Could not parse [ atoms ] from {itp_path}")
    return atoms


def parse_itp_bonds(itp_path: str | Path) -> List[Tuple[int, int]]:
    bonds: List[Tuple[int, int]] = []
    in_bonds = False
    with Path(itp_path).open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith("["):
                in_bonds = line.lower().startswith("[ bonds")
                continue
            if not in_bonds:
                continue
            line = line.split(";", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                i = int(parts[0])
                j = int(parts[1])
            except ValueError:
                continue
            bonds.append((i, j))
    return bonds


def read_atom_map(path: str | Path) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) >= 2:
                out.append((parts[0], parts[1]))
    if len(out) < 3:
        raise ValueError("Atom map too small; provide at least 3 seed pairs.")
    return out


def parse_pdb_xyz(raw: str, scale: float) -> np.ndarray:
    # fixed-column first
    try:
        x = float(raw[30:38]) * scale
        y = float(raw[38:46]) * scale
        z = float(raw[46:54]) * scale
        return np.array([x, y, z], float)
    except Exception:
        parts = raw.split()
        if len(parts) < 9:
            raise ValueError(f"Could not parse PDB coordinates from line: {raw.rstrip()}")
        # PDB fallback: last numeric triplet before occupancy/tempfactor usually at 6:9
        for start in (6, 5, 4):
            try:
                x = float(parts[start]) * scale
                y = float(parts[start + 1]) * scale
                z = float(parts[start + 2]) * scale
                return np.array([x, y, z], float)
            except Exception:
                pass
        raise ValueError(f"Could not parse PDB coordinates from line: {raw.rstrip()}")


def read_template(path: str | Path, units: str, ordered_atomnames: Sequence[str]) -> Template:
    path = Path(path)
    scale = 0.1 if units.upper() == "A" else 1.0
    atom_to_coord: Dict[str, np.ndarray] = {}

    if path.suffix.lower() == ".gro":
        gro = read_gro(path)
        res = gro.residues[0]
        for idx in res.atom_idx:
            atom_to_coord[gro.atomname[idx]] = gro.coords[idx].copy()
    else:
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                if not (raw.startswith("ATOM") or raw.startswith("HETATM")):
                    continue
                an = raw[12:16].strip() if len(raw) >= 16 else raw.split()[2]
                atom_to_coord[an] = parse_pdb_xyz(raw, scale)

    ordered = np.zeros((len(ordered_atomnames), 3), float)
    for i, an in enumerate(ordered_atomnames):
        if an not in atom_to_coord:
            raise ValueError(f"Template missing atom {an} required by NEW ITP order")
        ordered[i] = atom_to_coord[an]
    return Template(atom_to_coord=atom_to_coord, ordered_coords=ordered)


# -----------------------------------------------------------------------------
# Graph helpers
# -----------------------------------------------------------------------------


def build_adj(n_atoms: int, bonds: Sequence[Tuple[int, int]]) -> List[List[int]]:
    adj = [[] for _ in range(n_atoms)]
    for i, j in bonds:
        i0 = i - 1
        j0 = j - 1
        if 0 <= i0 < n_atoms and 0 <= j0 < n_atoms:
            adj[i0].append(j0)
            adj[j0].append(i0)
    return adj


def graph_distance_profile(adj: List[List[int]], seed_idxs: Iterable[int], node: int) -> Tuple[int, ...]:
    dists = []
    for s in seed_idxs:
        if s == node:
            dists.append(0)
            continue
        seen = {s}
        q = deque([(s, 0)])
        found = 999
        while q:
            cur, d = q.popleft()
            if cur == node:
                found = d
                break
            for nb in adj[cur]:
                if nb not in seen:
                    seen.add(nb)
                    q.append((nb, d + 1))
        dists.append(found)
    return tuple(dists)


def mapped_neighbor_signature(
    idx: int,
    adj: List[List[int]],
    mapped_self_to_other: Dict[int, int],
) -> Tuple[int, ...]:
    return tuple(sorted(mapped_self_to_other[n] for n in adj[idx] if n in mapped_self_to_other))


def node_signature(
    idx: int,
    atoms: Sequence[AtomRec],
    adj: List[List[int]],
    seed_idxs: Iterable[int],
) -> Tuple[object, ...]:
    nbs = adj[idx]
    elem = infer_element(atoms[idx].atom)
    heavy_deg = sum(1 for n in nbs if is_heavy(atoms[n].atom))
    hyd_deg = sum(1 for n in nbs if not is_heavy(atoms[n].atom))
    nb_elems = tuple(sorted(Counter(infer_element(atoms[n].atom) for n in nbs).items()))
    return (
        elem,
        is_heavy(atoms[idx].atom),
        len(nbs),
        heavy_deg,
        hyd_deg,
        nb_elems,
        graph_distance_profile(adj, seed_idxs, idx),
    )




def heavy_degree(idx: int, atoms: Sequence[AtomRec], adj: List[List[int]]) -> int:
    return sum(1 for n in adj[idx] if is_heavy(atoms[n].atom))


def carbon_degree(idx: int, atoms: Sequence[AtomRec], adj: List[List[int]]) -> int:
    return sum(1 for n in adj[idx] if infer_element(atoms[n].atom) == "C")


def reachable_carbon_count(
    start: int,
    prev: Optional[int],
    atoms: Sequence[AtomRec],
    adj: List[List[int]],
    forbidden: Set[int],
) -> int:
    if infer_element(atoms[start].atom) != "C":
        return 0
    seen = set(forbidden)
    if prev is not None:
        seen.add(prev)
    if start in seen:
        return 0
    q = deque([start])
    seen.add(start)
    cnt = 0
    while q:
        cur = q.popleft()
        if infer_element(atoms[cur].atom) == "C":
            cnt += 1
        for nb in adj[cur]:
            if nb in seen:
                continue
            if infer_element(atoms[nb].atom) != "C":
                continue
            seen.add(nb)
            q.append(nb)
    return cnt


def choose_tail_anchor_pair(
    old_atoms: Sequence[AtomRec],
    old_adj: List[List[int]],
    new_atoms: Sequence[AtomRec],
    new_adj: List[List[int]],
    mapped_old_to_new: Dict[int, int],
    mapped_new_to_old: Dict[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    best = None
    best_score = (-1, -1, -1)
    for oi, ni in mapped_old_to_new.items():
        old_cands = []
        for onb in old_adj[oi]:
            if onb in mapped_old_to_new:
                continue
            if infer_element(old_atoms[onb].atom) != "C":
                continue
            sz = reachable_carbon_count(onb, oi, old_atoms, old_adj, set(mapped_old_to_new))
            if sz > 0:
                old_cands.append((sz, onb))
        if not old_cands:
            continue
        new_cands = []
        for nnb in new_adj[ni]:
            if nnb in mapped_new_to_old:
                continue
            if infer_element(new_atoms[nnb].atom) != "C":
                continue
            sz = reachable_carbon_count(nnb, ni, new_atoms, new_adj, set(mapped_new_to_old))
            if sz > 0:
                new_cands.append((sz, nnb))
        if not new_cands:
            continue
        old_cands.sort(reverse=True)
        new_cands.sort(reverse=True)
        score = (min(old_cands[0][0], new_cands[0][0]), old_cands[0][0], new_cands[0][0])
        if score > best_score:
            best_score = score
            best = (oi, ni, old_cands[0][1], new_cands[0][1])
    return best


def build_main_carbon_path(
    start: int,
    anchor: int,
    atoms: Sequence[AtomRec],
    adj: List[List[int]],
    mapped_idx: Set[int],
) -> List[int]:
    if infer_element(atoms[start].atom) != "C":
        return []
    path = [start]
    prev = anchor
    cur = start
    while True:
        cands = []
        for nb in adj[cur]:
            if nb == prev or nb in mapped_idx or nb in path:
                continue
            if infer_element(atoms[nb].atom) != "C":
                continue
            downstream = reachable_carbon_count(nb, cur, atoms, adj, mapped_idx.union(set(path)))
            cands.append((downstream, carbon_degree(nb, atoms, adj), heavy_degree(nb, atoms, adj), -nb, nb))
        if not cands:
            break
        cands.sort(reverse=True)
        nxt = cands[0][-1]
        if cands[0][0] <= 0:
            break
        path.append(nxt)
        prev, cur = cur, nxt
    return path


def extend_backbone_and_sidechains(
    old_atoms: Sequence[AtomRec],
    old_adj: List[List[int]],
    new_atoms: Sequence[AtomRec],
    new_adj: List[List[int]],
    mapped_old_to_new: Dict[int, int],
    mapped_new_to_old: Dict[int, int],
) -> int:
    added = 0
    anchor_info = choose_tail_anchor_pair(old_atoms, old_adj, new_atoms, new_adj, mapped_old_to_new, mapped_new_to_old)
    if anchor_info is None:
        return 0
    old_anchor, new_anchor, old_start, new_start = anchor_info
    old_path = build_main_carbon_path(old_start, old_anchor, old_atoms, old_adj, set(mapped_old_to_new))
    new_path = build_main_carbon_path(new_start, new_anchor, new_atoms, new_adj, set(mapped_new_to_old))
    n = min(len(old_path), len(new_path))
    for i in range(n):
        oi = old_path[i]
        ni = new_path[i]
        if oi not in mapped_old_to_new and ni not in mapped_new_to_old:
            mapped_old_to_new[oi] = ni
            mapped_new_to_old[ni] = oi
            added += 1

    backbone_old = [old_anchor] + old_path[:n]
    backbone_new = [new_anchor] + new_path[:n]
    backbone_old_set = set(backbone_old)
    backbone_new_set = set(backbone_new)

    for oi, ni in zip(backbone_old, backbone_new):
        old_side = []
        for onb in old_adj[oi]:
            if onb in mapped_old_to_new or onb in backbone_old_set:
                continue
            if infer_element(old_atoms[onb].atom) != "C":
                continue
            score = (
                reachable_carbon_count(onb, oi, old_atoms, old_adj, set(mapped_old_to_new).union(backbone_old_set)),
                carbon_degree(onb, old_atoms, old_adj),
                heavy_degree(onb, old_atoms, old_adj),
                -onb,
            )
            old_side.append((score, onb))
        new_side = []
        for nnb in new_adj[ni]:
            if nnb in mapped_new_to_old or nnb in backbone_new_set:
                continue
            if infer_element(new_atoms[nnb].atom) != "C":
                continue
            score = (
                reachable_carbon_count(nnb, ni, new_atoms, new_adj, set(mapped_new_to_old).union(backbone_new_set)),
                carbon_degree(nnb, new_atoms, new_adj),
                heavy_degree(nnb, new_atoms, new_adj),
                -nnb,
            )
            new_side.append((score, nnb))
        old_side.sort(reverse=True)
        new_side.sort(reverse=True)
        for (_so, oidx), (_sn, nidx) in zip(old_side, new_side):
            if oidx not in mapped_old_to_new and nidx not in mapped_new_to_old:
                mapped_old_to_new[oidx] = nidx
                mapped_new_to_old[nidx] = oidx
                added += 1
    return added
def expand_seed_map(
    old_atoms: Sequence[AtomRec],
    old_bonds: Sequence[Tuple[int, int]],
    new_atoms: Sequence[AtomRec],
    new_bonds: Sequence[Tuple[int, int]],
    seed_pairs: Sequence[Tuple[str, str]],
) -> ExpandedMap:
    old_by_name = {a.atom: i for i, a in enumerate(old_atoms)}
    new_by_name = {a.atom: i for i, a in enumerate(new_atoms)}
    old_adj = build_adj(len(old_atoms), old_bonds)
    new_adj = build_adj(len(new_atoms), new_bonds)

    seed_old_idx: List[int] = []
    seed_new_idx: List[int] = []
    mapped_old_to_new: Dict[int, int] = {}
    mapped_new_to_old: Dict[int, int] = {}

    for old_name, new_name in seed_pairs:
        if old_name not in old_by_name:
            raise ValueError(f"Seed map references old atom not found in old ITP: {old_name}")
        if new_name not in new_by_name:
            raise ValueError(f"Seed map references new atom not found in new ITP: {new_name}")
        oi = old_by_name[old_name]
        ni = new_by_name[new_name]
        mapped_old_to_new[oi] = ni
        mapped_new_to_old[ni] = oi
        seed_old_idx.append(oi)
        seed_new_idx.append(ni)

    log_lines = [f"seed_pairs={len(seed_pairs)}"]
    changed = True
    while changed:
        changed = False
        # heavy atoms first
        for pass_heavy in (True, False):
            candidates: List[Tuple[int, int]] = []
            for oi, oat in enumerate(old_atoms):
                if oi in mapped_old_to_new:
                    continue
                if is_heavy(oat.atom) != pass_heavy:
                    continue
                old_sig = node_signature(oi, old_atoms, old_adj, seed_old_idx)
                old_mapped_nb = mapped_neighbor_signature(oi, old_adj, mapped_old_to_new)
                if len(old_mapped_nb) == 0:
                    continue
                poss = []
                for ni, nat in enumerate(new_atoms):
                    if ni in mapped_new_to_old:
                        continue
                    if is_heavy(nat.atom) != pass_heavy:
                        continue
                    if node_signature(ni, new_atoms, new_adj, seed_new_idx) != old_sig:
                        continue
                    if mapped_neighbor_signature(ni, new_adj, mapped_new_to_old) != old_mapped_nb:
                        continue
                    poss.append(ni)
                if len(poss) == 1:
                    candidates.append((oi, poss[0]))

            if candidates:
                # enforce one-to-one uniqueness inside this round
                by_new: Dict[int, List[int]] = defaultdict(list)
                for oi, ni in candidates:
                    by_new[ni].append(oi)
                for ni, olds in by_new.items():
                    if len(olds) == 1:
                        oi = olds[0]
                        if oi not in mapped_old_to_new and ni not in mapped_new_to_old:
                            mapped_old_to_new[oi] = ni
                            mapped_new_to_old[ni] = oi
                            changed = True

    # tail-aware expansion: follow the principal carbon backbone and its local side chains
    old_adj = build_adj(len(old_atoms), old_bonds)
    new_adj = build_adj(len(new_atoms), new_bonds)
    tail_added = extend_backbone_and_sidechains(
        old_atoms, old_adj, new_atoms, new_adj, mapped_old_to_new, mapped_new_to_old
    )

    # one more conservative pass after tail propagation
    changed = True
    while changed:
        changed = False
        for pass_heavy in (True, False):
            candidates: List[Tuple[int, int]] = []
            for oi, oat in enumerate(old_atoms):
                if oi in mapped_old_to_new:
                    continue
                if is_heavy(oat.atom) != pass_heavy:
                    continue
                old_sig = node_signature(oi, old_atoms, old_adj, seed_old_idx)
                old_mapped_nb = mapped_neighbor_signature(oi, old_adj, mapped_old_to_new)
                if len(old_mapped_nb) == 0:
                    continue
                poss = []
                for ni, nat in enumerate(new_atoms):
                    if ni in mapped_new_to_old:
                        continue
                    if is_heavy(nat.atom) != pass_heavy:
                        continue
                    if node_signature(ni, new_atoms, new_adj, seed_new_idx) != old_sig:
                        continue
                    if mapped_neighbor_signature(ni, new_adj, mapped_new_to_old) != old_mapped_nb:
                        continue
                    poss.append(ni)
                if len(poss) == 1:
                    candidates.append((oi, poss[0]))
            if candidates:
                by_new: Dict[int, List[int]] = defaultdict(list)
                for oi, ni in candidates:
                    by_new[ni].append(oi)
                for ni, olds in by_new.items():
                    if len(olds) == 1:
                        oi = olds[0]
                        if oi not in mapped_old_to_new and ni not in mapped_new_to_old:
                            mapped_old_to_new[oi] = ni
                            mapped_new_to_old[ni] = oi
                            changed = True

    # hydrogen fallback from already mapped heavy atoms
    for oi, ni in list(mapped_old_to_new.items()):
        if not is_heavy(old_atoms[oi].atom):
            continue
        old_h = [x for x in old_adj[oi] if (x not in mapped_old_to_new and not is_heavy(old_atoms[x].atom))]
        new_h = [x for x in new_adj[ni] if (x not in mapped_new_to_old and not is_heavy(new_atoms[x].atom))]
        old_h = sorted(old_h, key=lambda x: old_atoms[x].atom)
        new_h = sorted(new_h, key=lambda x: new_atoms[x].atom)
        for x_old, x_new in zip(old_h, new_h):
            mapped_old_to_new[x_old] = x_new
            mapped_new_to_old[x_new] = x_old

    all_pairs = sorted(
        [(old_atoms[oi].atom, new_atoms[ni].atom) for oi, ni in mapped_old_to_new.items()],
        key=lambda p: new_by_name[p[1]],
    )

    log_lines.append(f"tail_backbone_added={tail_added}")
    n_heavy = sum(1 for a in new_atoms if is_heavy(a.atom))
    n_mapped_heavy = sum(1 for _, new_name in all_pairs if is_heavy(new_name))
    log_lines.append(f"expanded_pairs={len(all_pairs)}")
    log_lines.append(f"mapped_heavy={n_mapped_heavy}/{n_heavy}")
    return ExpandedMap(seed_pairs=list(seed_pairs), all_pairs=all_pairs, score_lines=log_lines)


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------


def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    P0 = P - Pc
    Q0 = Q - Qc
    H = P0.T @ Q0
    U, _S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    t = Qc - Pc @ R
    return R, t


def rotation_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.eye(3)
    a = a / na
    b = b / nb
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # 180-degree rotation around any perpendicular axis
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, tmp)
        axis /= np.linalg.norm(axis)
        x, y, z = axis
        return np.array([
            [-1 + 2*x*x,     2*x*y,     2*x*z],
            [    2*x*y, -1 + 2*y*y,     2*y*z],
            [    2*x*z,     2*y*z, -1 + 2*z*z],
        ])
    kmat = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s * s))


def rotate_about_axis(points: np.ndarray, p1: np.ndarray, p2: np.ndarray, theta: float) -> np.ndarray:
    axis = p2 - p1
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return points.copy()
    u = axis / n
    c = math.cos(theta)
    s = math.sin(theta)
    v = points - p1
    ux, uy, uz = u
    K = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]], float)
    R = np.eye(3) * c + s * K + (1.0 - c) * np.outer(u, u)
    return v @ R.T + p1


def place_component_from_single_anchor(
    comp_idx: Sequence[int],
    coords: np.ndarray,
    template_coords: np.ndarray,
    anchor_old_exact: np.ndarray,
    anchor_new_idx: int,
    root_new_idx: int,
    desired_direction: Optional[np.ndarray],
) -> np.ndarray:
    out = coords.copy()
    t_anchor = template_coords[anchor_new_idx]
    t_root = template_coords[root_new_idx]
    cur_anchor = out[anchor_new_idx]
    cur_root = out[root_new_idx]

    vec_template = t_root - t_anchor
    if desired_direction is None or np.linalg.norm(desired_direction) < 1e-12:
        vec_target = cur_root - cur_anchor
    else:
        vec_target = desired_direction

    R = rotation_from_vectors(vec_template, vec_target)
    sub = template_coords[np.array(comp_idx)] - t_anchor
    sub = sub @ R.T + anchor_old_exact
    for local, atom_idx in enumerate(comp_idx):
        out[atom_idx] = sub[local]
    return out


def make_residue_whole(coords: np.ndarray, bonds: Sequence[Tuple[int, int]], box: Box, anchor_idx: int) -> np.ndarray:
    out = coords.copy()
    adj = build_adj(len(coords), bonds)
    seen = {anchor_idx}
    q = deque([anchor_idx])
    while q:
        cur = q.popleft()
        for nb in adj[cur]:
            if nb in seen:
                continue
            delta = out[nb] - out[cur]
            delta = box.min_image(delta[np.newaxis, :])[0]
            out[nb] = out[cur] + delta
            seen.add(nb)
            q.append(nb)
    return out


# -----------------------------------------------------------------------------
# Ligand branches and orientation
# -----------------------------------------------------------------------------


def choose_anchor_atoms_for_orientation(
    atoms: Sequence[AtomRec],
    bonds: Sequence[Tuple[int, int]],
) -> Tuple[List[int], List[int], List[int]]:
    adj = build_adj(len(atoms), bonds)

    oxy = [i for i, a in enumerate(atoms) if infer_element(a.atom) == "O"]
    head = sorted(set(oxy))
    if not head:
        head = [i for i, a in enumerate(atoms) if is_heavy(a.atom)][:4]

    heavy_idx = [i for i, a in enumerate(atoms) if is_heavy(a.atom)]
    degrees = {i: sum(1 for n in adj[i] if is_heavy(atoms[n].atom)) for i in heavy_idx}
    ends = [i for i in heavy_idx if degrees[i] <= 1 and i not in head]
    if not ends:
        ends = [heavy_idx[-1]] if heavy_idx else [0]

    # choose tail tip as farthest graph-distance from head set
    head_set = set(head)
    best_tip = ends[0]
    best_dist = -1
    for e in ends:
        q = deque([(e, 0)])
        seen = {e}
        mind = 999
        while q:
            cur, d = q.popleft()
            if cur in head_set:
                mind = d
                break
            for nb in adj[cur]:
                if nb not in seen:
                    seen.add(nb)
                    q.append((nb, d + 1))
        if mind > best_dist:
            best_dist = mind
            best_tip = e

    ringish = sorted(set(head))
    for i, a in enumerate(atoms):
        if infer_element(a.atom) == "C" and sum(1 for n in adj[i] if infer_element(atoms[n].atom) == "O") > 0:
            ringish.append(i)
    ringish = sorted(set(ringish))
    return head, ringish, [best_tip]


def find_unmatched_branches(
    n_new: int,
    bonds: Sequence[Tuple[int, int]],
    mapped_new_idx: Set[int],
) -> List[BranchInfo]:
    adj = build_adj(n_new, bonds)
    seen: Set[int] = set()
    branches: List[BranchInfo] = []
    for anchor in sorted(mapped_new_idx):
        for nb in adj[anchor]:
            if nb in mapped_new_idx or nb in seen:
                continue
            comp = []
            q = deque([nb])
            seen.add(nb)
            while q:
                cur = q.popleft()
                comp.append(cur)
                for x in adj[cur]:
                    if x in mapped_new_idx or x in seen:
                        continue
                    seen.add(x)
                    q.append(x)
            branches.append(BranchInfo(anchor_new=anchor, root_new=nb, atoms=sorted(comp)))
    return branches


def branch_direction_for_leaflet(
    anchor_xyz: np.ndarray,
    leaflet_top: bool,
    midplane_z: Optional[float],
    template_anchor: np.ndarray,
    template_root: np.ndarray,
) -> np.ndarray:
    vec = template_root - template_anchor
    if midplane_z is None:
        return vec
    desired = np.array(vec, float)
    if abs(desired[2]) < 1e-8:
        desired[2] = 1.0 if leaflet_top else -1.0
    if leaflet_top and desired[2] > 0:
        desired[2] *= -1.0
    if (not leaflet_top) and desired[2] < 0:
        desired[2] *= -1.0
    if np.linalg.norm(desired) < 1e-12:
        desired = np.array([0.0, 0.0, -1.0 if leaflet_top else 1.0])
    return desired


def residue_leaflet_from_head(
    head_coords: np.ndarray,
    midplane_z: Optional[float],
) -> bool:
    if midplane_z is None:
        return bool(np.mean(head_coords[:, 2]) >= np.mean(head_coords[:, 2]))
    return bool(np.mean(head_coords[:, 2]) >= midplane_z)


# -----------------------------------------------------------------------------
# Topology update
# -----------------------------------------------------------------------------


def find_old_itp_from_top(top_path: str | Path, old_name: str) -> str:
    top_path = Path(top_path)
    inc_rx = re.compile(r'^\s*#include\s+"([^"]+\.itp)"')
    includes = []
    for line in top_path.read_text(encoding="utf-8").splitlines():
        m = inc_rx.match(line)
        if m:
            includes.append(m.group(1))
    # prefer exact basename, then any path containing the old name
    for inc in includes:
        if Path(inc).stem.upper() == old_name.upper():
            return inc
    for inc in includes:
        if old_name.upper() in Path(inc).stem.upper() or old_name.upper() in inc.upper():
            return inc
    raise ValueError(f"Could not find old ITP include for {old_name} in {top_path}")


def update_topol(
    top_in: str | Path,
    top_out: str | Path,
    old_name: str,
    new_name: str,
    new_itp_include: str,
    n_old_total: int,
    n_replace: int,
) -> None:
    lines = Path(top_in).read_text(encoding="utf-8").splitlines(True)
    inc_rx = re.compile(r'^\s*#include\s+"([^"]+)"')
    has_inc = False
    for ln in lines:
        m = inc_rx.match(ln)
        if m and m.group(1) == new_itp_include:
            has_inc = True
            break

    if not has_inc:
        inserted = False
        for i, ln in enumerate(lines):
            if f'{old_name}.itp' in ln:
                lines.insert(i + 1, f'#include "{new_itp_include}"\n')
                inserted = True
                break
        if not inserted:
            for i, ln in enumerate(lines):
                if ln.strip().lower().startswith("[ system"):
                    lines.insert(i, f'#include "{new_itp_include}"\n')
                    inserted = True
                    break
        if not inserted:
            lines.insert(0, f'#include "{new_itp_include}"\n')

    out_lines = []
    in_mol = False
    for ln in lines:
        s = ln.strip()
        if s.lower().startswith("[ molecules"):
            in_mol = True
            out_lines.append(ln)
            continue
        if in_mol and s.startswith("[") and not s.lower().startswith("[ molecules"):
            in_mol = False
        if in_mol:
            if not s or s.startswith(";"):
                out_lines.append(ln)
                continue
            parts = s.split()
            if len(parts) >= 2 and parts[0] == old_name:
                if n_replace != n_old_total:
                    raise ValueError(
                        f"This version expects replacing ALL old molecules. n_replace={n_replace} n_old_total={n_old_total}"
                    )
                out_lines.append(f"{new_name:<8s}\t{n_replace:>12d}\n")
                continue
        out_lines.append(ln)

    Path(top_out).write_text("".join(out_lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--in-gro", "-f", required=True)
    ap.add_argument("--top", "-p", required=True)
    ap.add_argument("--out-gro", required=True)
    ap.add_argument("--out-top", required=True)
    ap.add_argument("--old-resname", "--old", default="UQOL8")
    ap.add_argument("--new-resname", "--new", required=True)
    ap.add_argument("--new-itp", required=True)
    ap.add_argument("--old-itp", default=None, help="Optional explicit old ITP. If omitted, look it up from topol.top")
    ap.add_argument("--template-pdb", "--new-template", required=True)
    ap.add_argument("--template-units", choices=["A", "nm"], default="A")
    ap.add_argument("--atom-map", required=True, help="Seed map OLDATOM NEWATOM; will be auto-expanded using ITP graphs")
    ap.add_argument("--write-expanded-map", default=None, help="Optional file to write the expanded map actually used")
    ap.add_argument("--midplane-z", type=float, default=None)
    ap.add_argument("--tail-rotate-trials", type=int, default=128)
    ap.add_argument("--min-heavy-dist", type=float, default=0.07)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--new-itp-include", default=None)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(int(args.seed))
    gro = read_gro(args.in_gro)
    box = gro.box

    old_name = args.old_resname.strip()
    new_name = args.new_resname.strip()

    old_res = [r for r in gro.residues if r.resname == old_name]
    if not old_res:
        raise SystemExit(f"No residues named {old_name} found in {args.in_gro}")

    old_itp = args.old_itp if args.old_itp is not None else find_old_itp_from_top(args.top, old_name)
    old_atoms = parse_itp_atoms(old_itp)
    old_bonds = parse_itp_bonds(old_itp)
    new_atoms = parse_itp_atoms(args.new_itp)
    new_bonds = parse_itp_bonds(args.new_itp)

    old_atom_order = [a.atom for a in old_atoms]
    new_atom_order = [a.atom for a in new_atoms]

    seed_map = read_atom_map(args.atom_map)
    expanded = expand_seed_map(old_atoms, old_bonds, new_atoms, new_bonds, seed_map)
    if args.write_expanded_map:
        with Path(args.write_expanded_map).open("w", encoding="utf-8") as f:
            f.write(f"# Expanded map generated from seed: {args.atom_map}\n")
            for line in expanded.score_lines:
                f.write(f"# {line}\n")
            for old_a, new_a in expanded.all_pairs:
                f.write(f"{old_a:<8s} {new_a}\n")

    template = read_template(args.template_pdb, args.template_units, new_atom_order)
    tpl_coords = template.ordered_coords.copy()
    new_adj = build_adj(len(new_atoms), new_bonds)

    # indices for mapped atoms
    old_name_to_idx = {a: i for i, a in enumerate(old_atom_order)}
    new_name_to_idx = {a: i for i, a in enumerate(new_atom_order)}
    mapped_pairs_idx = [(old_name_to_idx[o], new_name_to_idx[n]) for o, n in expanded.all_pairs]
    mapped_new_idx = {ni for _oi, ni in mapped_pairs_idx}

    # orientation helper atoms on NEW ligand
    head_new_idx, ringish_new_idx, tail_tip_new_idx = choose_anchor_atoms_for_orientation(new_atoms, new_bonds)
    tail_tip_idx = tail_tip_new_idx[0]

    # environment heavy atoms (all non-old residues)
    old_mask = np.zeros(len(gro.coords), dtype=bool)
    for r in old_res:
        old_mask[r.atom_idx] = True
    heavy_mask = np.array([is_heavy(a) for a in gro.atomname], dtype=bool)
    env_coords = gro.coords[heavy_mask & (~old_mask)]

    env_tree = None
    if cKDTree is not None and env_coords.size > 0:
        if box.is_orthorhombic():
            env_tree = cKDTree(box.wrap(env_coords), boxsize=list(box.lengths()))
        else:
            imgs = []
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    for k in (-1, 0, 1):
                        shift = i * box.v1 + j * box.v2 + k * box.v3
                        imgs.append(env_coords + shift)
            env_tree = cKDTree(np.vstack(imgs))

    min_dist = float(args.min_heavy_dist)

    def score_candidate(heavy_xyz: np.ndarray) -> float:
        if env_tree is None or heavy_xyz.size == 0:
            return 1.0
        xyz = box.wrap(heavy_xyz) if box.is_orthorhombic() else heavy_xyz
        d, _ = env_tree.query(xyz, k=1)
        return float(np.min(d))

    def clashes(heavy_xyz: np.ndarray) -> bool:
        if env_tree is None or heavy_xyz.size == 0:
            return False
        xyz = box.wrap(heavy_xyz) if box.is_orthorhombic() else heavy_xyz
        hits = env_tree.query_ball_point(xyz, r=min_dist)
        return any(len(h) > 0 for h in hits)

    branches = find_unmatched_branches(len(new_atoms), new_bonds, mapped_new_idx)
    new_heavy_mask = np.array([is_heavy(a) for a in new_atom_order], dtype=bool)

    out_records: List[Tuple[int, str, str, np.ndarray]] = []

    for res in gro.residues:
        if res.resname != old_name:
            for idx in res.atom_idx:
                out_records.append((res.resid, res.resname, gro.atomname[idx], gro.coords[idx].copy()))
            continue

        old_atom_to_coord = {gro.atomname[idx]: gro.coords[idx].copy() for idx in res.atom_idx}
        available_pairs = [(o, n) for (o, n) in expanded.all_pairs if o in old_atom_to_coord]
        if len(available_pairs) < 3:
            raise RuntimeError(
                f"Residue {res.resid} has only {len(available_pairs)} mapped atoms available; need at least 3."
            )

        src = np.array([template.atom_to_coord[new_a] for _old_a, new_a in available_pairs], float)
        tgt = np.array([old_atom_to_coord[old_a] for old_a, _new_a in available_pairs], float)
        R, t = kabsch(src, tgt)
        new_xyz = tpl_coords @ R + t

        # exact overwrite for all mapped atoms available in this residue
        for old_a, new_a in available_pairs:
            new_xyz[new_name_to_idx[new_a]] = old_atom_to_coord[old_a]

        head_coords = new_xyz[np.array(head_new_idx, int)]
        leaflet_top = True if args.midplane_z is None else (float(np.mean(head_coords[:, 2])) >= float(args.midplane_z))

        # Rebuild unmatched branches from the anchor using template-local geometry and leaflet preference.
        for br in branches:
            anchor_exact = new_xyz[br.anchor_new].copy()
            desired = branch_direction_for_leaflet(
                anchor_exact,
                leaflet_top,
                args.midplane_z,
                tpl_coords[br.anchor_new],
                tpl_coords[br.root_new],
            )
            comp_idx = [br.root_new] + [x for x in br.atoms if x != br.root_new]
            new_xyz = place_component_from_single_anchor(
                comp_idx,
                new_xyz,
                tpl_coords,
                anchor_exact,
                br.anchor_new,
                br.root_new,
                desired,
            )

        # small loose clash cleanup: rotate unmatched branches around anchor-root axis if needed
        for br in branches:
            comp_all = np.array([br.root_new] + [x for x in br.atoms if x != br.root_new], int)
            comp_heavy = np.array([x for x in comp_all if new_heavy_mask[x]], int)
            if comp_heavy.size == 0:
                continue
            best_xyz = new_xyz.copy()
            best_score = score_candidate(best_xyz[new_heavy_mask])
            if clashes(best_xyz[comp_heavy]):
                p1 = best_xyz[br.anchor_new].copy()
                p2 = best_xyz[br.root_new].copy()
                for _ in range(int(args.tail_rotate_trials)):
                    theta = float(rng.random() * 2.0 * math.pi)
                    cand = best_xyz.copy()
                    cand[comp_all] = rotate_about_axis(cand[comp_all], p1, p2, theta)
                    sc = score_candidate(cand[new_heavy_mask])
                    if sc > best_score:
                        best_score = sc
                        best_xyz = cand
                    if not clashes(cand[comp_heavy]):
                        best_xyz = cand
                        break
            new_xyz = best_xyz

        # Make residue whole and keep it together in the box using headgroup centroid as anchor.
        anchor_idx = head_new_idx[0] if head_new_idx else 0
        new_xyz = make_residue_whole(new_xyz, new_bonds, box, anchor_idx)
        anchor_centroid = np.mean(new_xyz[np.array(head_new_idx, int)], axis=0) if head_new_idx else new_xyz[anchor_idx]
        new_xyz = box.translate_whole_into_box(new_xyz, anchor_centroid)

        for i, an in enumerate(new_atom_order):
            out_records.append((res.resid, new_name, an, new_xyz[i].copy()))

    write_gro_from_records(
        args.out_gro,
        title=f"{gro.title} | {old_name}->{new_name} exactmap",
        atom_records=out_records,
        box_line=gro.box_line,
    )

    new_inc = args.new_itp_include if args.new_itp_include is not None else args.new_itp
    update_topol(
        args.top,
        args.out_top,
        old_name=old_name,
        new_name=new_name,
        new_itp_include=new_inc,
        n_old_total=len(old_res),
        n_replace=len(old_res),
    )

    print(f"Loaded old ITP: {old_itp}")
    print(f"Expanded seed map {len(seed_map)} -> {len(expanded.all_pairs)} pairs")
    for line in expanded.score_lines:
        print(line)
    print(f"Replaced {len(old_res)} x {old_name} -> {new_name}")
    print(f"Wrote: {args.out_gro}")
    print(f"Wrote: {args.out_top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
