#!/usr/bin/env python3
"""Extract coordinating residues from an XYZ file and center on Zn.

Given an input XYZ file, finds the Zn atom, centers coordinates on Zn,
selects the four closest coordinating atoms among S/N, and writes a new
XYZ that keeps:
  - CYS: SG, CB, CA
  - HIS: imidazole ring (CG, CD2, NE2, CE1, ND1) plus CB/CA

Output XYZ uses the same comment metadata style as other scripts in this repo.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from xyz_plot_helpers import XyzAtom, parse_xyz_atoms

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scrape_pdb.writer import _append_hydrogens_in_place  # noqa: E402


COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "S": 1.05,
    "ZN": 1.22,
}


@dataclass(frozen=True)
class _AtomRef:
    idx: int
    atom: XyzAtom


def _el(atom: XyzAtom) -> str:
    return atom.element.strip().upper()


def _dist(a: XyzAtom, b: XyzAtom) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _build_adjacency(atoms: List[XyzAtom]) -> Dict[int, List[int]]:
    heavy_indices = [i for i, a in enumerate(atoms) if _el(a) != "H"]
    adj: Dict[int, List[int]] = {i: [] for i in heavy_indices}
    for i, idx_i in enumerate(heavy_indices):
        ai = atoms[idx_i]
        ei = _el(ai)
        if ei == "ZN":
            continue
        ri = COVALENT_RADII.get(ei)
        if ri is None:
            continue
        for idx_j in heavy_indices[i + 1 :]:
            aj = atoms[idx_j]
            ej = _el(aj)
            if ej == "ZN":
                continue
            rj = COVALENT_RADII.get(ej)
            if rj is None:
                continue
            d = _dist(ai, aj)
            if 0.6 <= d <= (ri + rj + 0.45):
                adj[idx_i].append(idx_j)
                adj[idx_j].append(idx_i)
    return adj


def _neighbors(adj: Dict[int, List[int]], atoms: List[XyzAtom], idx: int, element: str | None = None) -> List[int]:
    if idx not in adj:
        return []
    if element is None:
        return adj[idx]
    want = element.upper()
    return [j for j in adj[idx] if _el(atoms[j]) == want]


def _n_neighbors(adj: Dict[int, List[int]], atoms: List[XyzAtom], idx: int, element: str) -> int:
    return sum(1 for j in _neighbors(adj, atoms, idx) if _el(atoms[j]) == element.upper())


def _dist_to_zn(atoms: List[XyzAtom], zn: XyzAtom, idx: int) -> float:
    return _dist(atoms[idx], zn)


def _pick_his_stem_from_coord(
    atoms: List[XyzAtom],
    adj: Dict[int, List[int]],
    zn: XyzAtom,
    coord_idx: int,
) -> Tuple[int, int, int, int] | None:
    # Search within the imidazole ring neighborhood for CG.
    max_depth = 4
    depth_map: Dict[int, int] = {coord_idx: 0}
    q: List[int] = [coord_idx]
    while q:
        node = q.pop(0)
        depth = depth_map[node]
        if depth >= max_depth:
            continue
        for nb in _neighbors(adj, atoms, node):
            if _el(atoms[nb]) not in {"C", "N"}:
                continue
            if nb in depth_map:
                continue
            depth_map[nb] = depth + 1
            q.append(nb)

    cg_candidates: List[Tuple[int, float, int]] = []
    for c_idx, depth in depth_map.items():
        if _el(atoms[c_idx]) != "C":
            continue
        # CG must be in ring (has N neighbor) and have a CB neighbor (carbon with no N neighbors).
        if _n_neighbors(adj, atoms, c_idx, "N") == 0:
            continue
        cb_candidates = [
            j
            for j in _neighbors(adj, atoms, c_idx, "C")
            if _n_neighbors(adj, atoms, j, "N") == 0
        ]
        if not cb_candidates:
            continue
        cg_candidates.append((depth, _dist(atoms[c_idx], atoms[coord_idx]), c_idx))

    if not cg_candidates:
        return None

    cg_candidates.sort(key=lambda t: (t[0], t[1]))
    _, _, cg_idx = cg_candidates[0]

    cb_candidates = [
        j
        for j in _neighbors(adj, atoms, cg_idx, "C")
        if _n_neighbors(adj, atoms, j, "N") == 0
    ]
    if not cb_candidates:
        return None
    cb_idx = min(cb_candidates, key=lambda j: _dist(atoms[j], atoms[cg_idx]))

    ca_candidates = [j for j in _neighbors(adj, atoms, cb_idx, "C") if j != cg_idx]
    if not ca_candidates:
        return None
    ca_idx = max(ca_candidates, key=lambda j: _dist_to_zn(atoms, zn, j))
    return (cg_idx, cb_idx, ca_idx, coord_idx)


def _infer_his_ring_from_cg(
    atoms: List[XyzAtom],
    adj: Dict[int, List[int]],
    cg_idx: int,
    cb_idx: int,
) -> Dict[str, int]:
    ring: Dict[str, int] = {}
    ring_neighbors = [
        j
        for j in _neighbors(adj, atoms, cg_idx)
        if _el(atoms[j]) in {"C", "N"} and j != cb_idx
    ]

    nd1_idx = None
    cd2_idx = None
    for j in ring_neighbors:
        if _el(atoms[j]) == "N":
            nd1_idx = j
        elif _el(atoms[j]) == "C":
            cd2_idx = j

    if cd2_idx is None:
        for j in ring_neighbors:
            if _el(atoms[j]) != "C":
                continue
            if any(_el(atoms[k]) == "N" and k != cg_idx for k in _neighbors(adj, atoms, j)):
                cd2_idx = j
                break

    ne2_idx = None
    ce1_idx = None
    if cd2_idx is not None:
        n_neighbors = [k for k in _neighbors(adj, atoms, cd2_idx) if _el(atoms[k]) == "N" and k != cg_idx]
        if n_neighbors:
            ne2_idx = n_neighbors[0]
            c_neighbors = [
                k for k in _neighbors(adj, atoms, ne2_idx) if _el(atoms[k]) == "C" and k != cd2_idx
            ]
            if c_neighbors:
                ce1_idx = c_neighbors[0]
                n_from_ce1 = [
                    k for k in _neighbors(adj, atoms, ce1_idx) if _el(atoms[k]) == "N" and k != ne2_idx
                ]
                if n_from_ce1:
                    nd1_idx = n_from_ce1[0]

    if nd1_idx is not None and ne2_idx is None:
        ce1_candidates = [
            k for k in _neighbors(adj, atoms, nd1_idx) if _el(atoms[k]) == "C" and k != cg_idx
        ]
        if ce1_candidates:
            ce1_idx = ce1_candidates[0]
            ne2_candidates = [
                k for k in _neighbors(adj, atoms, ce1_idx) if _el(atoms[k]) == "N" and k != nd1_idx
            ]
            if ne2_candidates:
                ne2_idx = ne2_candidates[0]
                if cd2_idx is None:
                    cd2_candidates = [
                        k for k in _neighbors(adj, atoms, ne2_idx) if _el(atoms[k]) == "C" and k != ce1_idx
                    ]
                    if cd2_candidates:
                        cd2_idx = cd2_candidates[0]

    if cd2_idx is not None:
        ring["CD2"] = cd2_idx
    if ne2_idx is not None:
        ring["NE2"] = ne2_idx
    if ce1_idx is not None:
        ring["CE1"] = ce1_idx
    if nd1_idx is not None:
        ring["ND1"] = nd1_idx
    return ring


def _infer_coordinating_residues(atoms: List[XyzAtom]) -> List[XyzAtom]:
    zn = next((a for a in atoms if _el(a) == "ZN"), None)
    if zn is None:
        raise SystemExit("No Zn atom found in input XYZ.")

    adj = _build_adjacency(atoms)

    candidates: List[Tuple[float, int]] = []
    for i, a in enumerate(atoms):
        if _el(a) not in {"S", "N"}:
            continue
        candidates.append((_dist(a, zn), i))

    if not candidates:
        raise SystemExit("No S/N coordinating atoms found in input XYZ.")

    candidates.sort(key=lambda t: t[0])
    closest = [idx for _, idx in candidates[:4]]

    inferred_atoms: List[XyzAtom] = []
    inferred_atoms.append(
        XyzAtom(
            element=zn.element,
            x=0.0,
            y=0.0,
            z=0.0,
            meta={"RES": "ZN", "ATOM": "ZN", "RESSEQ": "0", "CHAIN": "A"},
            raw_comment="",
        )
    )
    cys_count = 0
    his_count = 0

    for idx in closest:
        if _el(atoms[idx]) == "S":
            sg_idx = idx
            c_neighbors = _neighbors(adj, atoms, sg_idx, "C")
            if not c_neighbors:
                continue
            cb_idx = min(c_neighbors, key=lambda j: _dist(atoms[j], atoms[sg_idx]))
            ca_candidates = [j for j in _neighbors(adj, atoms, cb_idx, "C") if j != sg_idx]
            if not ca_candidates:
                continue
            ca_idx = max(ca_candidates, key=lambda j: _dist_to_zn(atoms, zn, j))

            cys_count += 1
            resseq = str(cys_count)
            for atom_idx, atom_name in ((sg_idx, "SG"), (cb_idx, "CB"), (ca_idx, "CA")):
                a = atoms[atom_idx]
                meta = {"RES": "CYS", "ATOM": atom_name, "RESSEQ": resseq, "CHAIN": "A"}
                if atom_name == "SG":
                    meta["COORD"] = "1"
                inferred_atoms.append(
                    XyzAtom(
                        element=a.element,
                        x=a.x - zn.x,
                        y=a.y - zn.y,
                        z=a.z - zn.z,
                        meta=meta,
                        raw_comment="",
                    )
                )
            continue

        # HIS inference from N coordination.
        stem = _pick_his_stem_from_coord(atoms, adj, zn, idx)
        if stem is None:
            continue
        cg_idx, cb_idx, ca_idx, coord_idx = stem
        his_count += 1
        resseq = str(100 + his_count)

        ring = _infer_his_ring_from_cg(atoms, adj, cg_idx, cb_idx)
        coord_name = "ND1"
        if coord_idx == ring.get("NE2"):
            coord_name = "NE2"
        elif coord_idx == ring.get("ND1"):
            coord_name = "ND1"

        a_coord = atoms[coord_idx]
        inferred_atoms.append(
            XyzAtom(
                element=a_coord.element,
                x=a_coord.x - zn.x,
                y=a_coord.y - zn.y,
                z=a_coord.z - zn.z,
                meta={
                    "RES": "HIS",
                    "ATOM": coord_name,
                    "RESSEQ": resseq,
                    "CHAIN": "A",
                    "COORD": "1",
                },
                raw_comment="",
            )
        )

        for atom_idx, atom_name in ((cg_idx, "CG"), (cb_idx, "CB"), (ca_idx, "CA")):
            a = atoms[atom_idx]
            inferred_atoms.append(
                XyzAtom(
                    element=a.element,
                    x=a.x - zn.x,
                    y=a.y - zn.y,
                    z=a.z - zn.z,
                    meta={"RES": "HIS", "ATOM": atom_name, "RESSEQ": resseq, "CHAIN": "A"},
                    raw_comment="",
                )
            )

        for atom_name in ("CD2", "NE2", "CE1", "ND1"):
            atom_idx = ring.get(atom_name)
            if atom_idx is None:
                continue
            a = atoms[atom_idx]
            inferred_atoms.append(
                XyzAtom(
                    element=a.element,
                    x=a.x - zn.x,
                    y=a.y - zn.y,
                    z=a.z - zn.z,
                    meta={"RES": "HIS", "ATOM": atom_name, "RESSEQ": resseq, "CHAIN": "A"},
                    raw_comment="",
                )
            )

    if not inferred_atoms:
        raise SystemExit("No coordinating residues could be inferred from input XYZ.")

    return inferred_atoms


def _write_xyz(out_path: Path, atoms: List[XyzAtom], *, source: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"Centered on Zn; coordinators only; source={source.name}\n")
        for a in atoms:
            meta = a.meta or {}
            meta_str = " ".join(
                f"{key}={meta[key]}" for key in ("RES", "CHAIN", "RESSEQ", "ATOM", "COORD") if meta.get(key)
            )
            if meta_str:
                f.write(f"{a.element:2s}  {a.x: .6f}  {a.y: .6f}  {a.z: .6f}  # {meta_str}\n")
            else:
                f.write(f"{a.element:2s}  {a.x: .6f}  {a.y: .6f}  {a.z: .6f}\n")

    _append_hydrogens_in_place(out_path)
    _remove_hydrogens_near_zn(out_path)


def _remove_hydrogens_near_zn(xyz_path: Path, *, cutoff: float = 1.35) -> None:
    """Remove any H atoms within cutoff Å of Zn (coordination H artifacts)."""
    atoms = parse_xyz_atoms(xyz_path)
    if not atoms:
        return
    zn = next((a for a in atoms if _el(a) == "ZN"), None)
    if zn is None:
        return

    kept: List[XyzAtom] = []
    for a in atoms:
        if _el(a) != "H":
            kept.append(a)
            continue
        if _dist(a, zn) > cutoff:
            kept.append(a)

    if len(kept) == len(atoms):
        return

    header = [str(len(kept)), f"Centered on Zn; coordinators only; source={xyz_path.name}"]
    lines = []
    for a in kept:
        meta = a.meta or {}
        meta_str = " ".join(
            f"{key}={meta[key]}" for key in ("RES", "CHAIN", "RESSEQ", "ATOM", "COORD") if meta.get(key)
        )
        if meta_str:
            lines.append(f"{a.element:2s}  {a.x: .6f}  {a.y: .6f}  {a.z: .6f}  # {meta_str}")
        else:
            lines.append(f"{a.element:2s}  {a.x: .6f}  {a.y: .6f}  {a.z: .6f}")

    xyz_path.write_text("\n".join(header + lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Center on Zn and keep coordinating CYS/HIS residues from an XYZ file."
    )
    parser.add_argument("--input", required=True, help="Path to input .xyz file")
    parser.add_argument(
        "--output",
        help="Path to output .xyz file (default: <input>_coord.xyz)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")
    if input_path.suffix.lower() != ".xyz":
        raise SystemExit(f"Input is not an .xyz file: {input_path}")

    atoms = parse_xyz_atoms(input_path)
    if not atoms:
        raise SystemExit(f"No atoms found in: {input_path}")

    inferred_atoms = _infer_coordinating_residues(atoms)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = input_path.with_name(f"{input_path.stem}_coord.xyz")

    _write_xyz(out_path, inferred_atoms, source=input_path)
    print(f"Wrote centered coordinating XYZ: {out_path}")


if __name__ == "__main__":
    main()
