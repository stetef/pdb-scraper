#!/usr/bin/env python3
"""Analyze and plot B-factor statistics for Zn cluster context XYZ files.

Given a base XYZ file (for example: ``1a1h_ZN_homo_d3.00_cluster1.xyz``), this
script looks in the same directory for companion files:
- ``<base>_backbone.xyz``
- ``<base>_nearby_res.xyz``
- ``<base>_10A_pocket.xyz``

It computes mean/std B-factors for:
1. Zn (single value)
2. Each coordinating residue (base XYZ, up-to-CA representation)
3. Nearby residues for each coordinator (from nearby_res XYZ)
4. Pocket (10A pocket excluding coordinating residues and Zn)

And writes a plot with requested styling.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


AA_3 = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}


@dataclass(frozen=True)
class ResidueKey:
    chain: str
    resseq: str
    resname: str


@dataclass(frozen=True)
class XyzAtom:
    element: str
    x: float
    y: float
    z: float
    meta: Dict[str, str]


@dataclass(frozen=True)
class GroupStat:
    label: str
    mean_b: float
    std_b: float
    color: str
    marker: str


def _meta_dict(comment: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for token in comment.split():
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        if k and v:
            out[k.strip()] = v.strip()
    return out


def _safe_float(text: str | None) -> float | None:
    if text is None:
        return None
    raw = str(text).strip()
    if raw in {"", ".", "?"}:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_xyz(path: Path) -> List[XyzAtom]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        return []

    atoms: List[XyzAtom] = []
    for line in lines[2:]:
        stripped = line.strip()
        if not stripped:
            continue
        if "#" in stripped:
            left, right = stripped.split("#", 1)
            comment = right.strip()
        else:
            left, comment = stripped, ""
        parts = left.split()
        if len(parts) < 4:
            continue
        x = _safe_float(parts[1])
        y = _safe_float(parts[2])
        z = _safe_float(parts[3])
        if x is None or y is None or z is None:
            continue
        atoms.append(
            XyzAtom(
                element=parts[0].upper(),
                x=x,
                y=y,
                z=z,
                meta=_meta_dict(comment),
            )
        )
    return atoms


def _residue_key(atom: XyzAtom) -> ResidueKey | None:
    res = (atom.meta.get("RES") or "").upper()
    chain = atom.meta.get("CHAIN") or ""
    resseq = atom.meta.get("RESSEQ") or ""
    if not (res and chain and resseq):
        return None
    return ResidueKey(chain=chain, resseq=resseq, resname=res)


def _distance(a: XyzAtom, b: XyzAtom) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _find_zn(base_atoms: List[XyzAtom]) -> XyzAtom | None:
    for atom in base_atoms:
        if (atom.meta.get("RES") or "").upper() == "ZN":
            return atom
    for atom in base_atoms:
        if atom.element == "ZN":
            return atom
    return None


def _coordinating_residues(base_atoms: List[XyzAtom], zn_atom: XyzAtom) -> List[ResidueKey]:
    min_dist: Dict[ResidueKey, float] = {}
    for atom in base_atoms:
        if atom.element == "H":
            continue
        rkey = _residue_key(atom)
        if rkey is None:
            continue
        if rkey.resname in {"ZN", "HOH"}:
            continue
        d = _distance(atom, zn_atom)
        old = min_dist.get(rkey)
        if old is None or d < old:
            min_dist[rkey] = d

    coordinators = [rk for rk, d in min_dist.items() if d <= 3.0]
    coordinators.sort(key=lambda r: (r.chain, _int_or_big(r.resseq), r.resseq))
    return coordinators


def _int_or_big(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        return 10**9


def _parse_pdb_residue_order(pdb_path: Path) -> Dict[str, List[ResidueKey]]:
    order: Dict[str, List[ResidueKey]] = {}
    seen: Dict[str, set[Tuple[str, str]]] = {}
    with pdb_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.startswith("ATOM"):
                continue
            if len(raw) < 54:
                continue
            resname = raw[17:20].strip().upper()
            chain = raw[21:22].strip()
            resseq = raw[22:26].strip()
            if not chain or not resseq or resname not in AA_3:
                continue
            key = (resseq, resname)
            if chain not in order:
                order[chain] = []
                seen[chain] = set()
            if key in seen[chain]:
                continue
            seen[chain].add(key)
            order[chain].append(ResidueKey(chain=chain, resseq=resseq, resname=resname))
    return order


def _resolve_pdb_for_base(base_xyz: Path) -> Path:
    pdb_id = base_xyz.stem.split("_", 1)[0].lower()
    xyz_dir = base_xyz.parent
    candidates = [
        (xyz_dir / "../results/validated_structures" / f"{pdb_id}.pdb").resolve(),
        (xyz_dir / "../../results/validated_structures" / f"{pdb_id}.pdb").resolve(),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


def _bf_values_for_residue(atoms: Iterable[XyzAtom], residue: ResidueKey) -> List[float]:
    values: List[float] = []
    for atom in atoms:
        if atom.element == "H":
            continue
        rkey = _residue_key(atom)
        if rkey != residue:
            continue
        atom_name = (atom.meta.get("ATOM") or "").upper()
        if atom_name in {"N", "C", "O"}:
            continue
        bf = _safe_float(atom.meta.get("BF"))
        if bf is not None:
            values.append(bf)
    return values


def _bf_values_for_residue_set(atoms: Iterable[XyzAtom], residues: set[ResidueKey]) -> List[float]:
    values: List[float] = []
    for atom in atoms:
        if atom.element == "H":
            continue
        rkey = _residue_key(atom)
        if rkey not in residues:
            continue
        bf = _safe_float(atom.meta.get("BF"))
        if bf is not None:
            values.append(bf)
    return values


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return -1.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), pstdev(values)


def _coord_color(resname: str) -> str:
    if resname == "CYS":
        return "#d4b000"
    if resname == "HIS":
        return "#1f77b4"
    return "#2ca02c"


def _neighbor_residue_set(
    coordinator: ResidueKey,
    coordinators: set[ResidueKey],
    residue_order: Dict[str, List[ResidueKey]],
) -> set[ResidueKey]:
    chain_res = residue_order.get(coordinator.chain, [])
    try:
        idx = chain_res.index(coordinator)
    except ValueError:
        return set()

    out: set[ResidueKey] = set()
    for j in range(max(0, idx - 2), min(len(chain_res), idx + 3)):
        if j == idx:
            continue
        rk = chain_res[j]
        if rk in coordinators:
            continue
        out.add(rk)
    return out


def _plot_stats(stats: List[GroupStat], out_png: Path, title: str) -> None:
    xs = list(range(len(stats)))
    ys = [s.mean_b for s in stats]
    yerr = [s.std_b for s in stats]

    plt.figure(figsize=(9.5, 5.5))
    for x, y, e, s in zip(xs, ys, yerr, stats):
        plt.errorbar(
            [x],
            [y],
            yerr=[e],
            fmt=s.marker,
            color=s.color,
            markersize=8,
            capsize=4,
            linewidth=1.5,
        )

    plt.xticks(xs, [s.label for s in stats], rotation=35, ha="right")
    plt.ylabel("B-factor", fontsize=16)
    plt.title(title, fontsize=16)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", width=2, length=8, labelsize=14)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def run_analysis(base_xyz: Path, out_png: Path | None = None) -> Path:
    base_xyz = base_xyz.resolve()
    stem = base_xyz.stem
    xyz_dir = base_xyz.parent

    backbone_xyz = xyz_dir / f"{stem}_backbone.xyz"
    nearby_xyz = xyz_dir / f"{stem}_nearby_res.xyz"
    pocket_xyz = xyz_dir / f"{stem}_10A_pocket.xyz"

    missing = [p for p in (base_xyz, backbone_xyz, nearby_xyz, pocket_xyz) if not p.exists()]
    if missing:
        names = ", ".join(str(p) for p in missing)
        raise SystemExit(f"Missing required xyz file(s): {names}")

    base_atoms = _parse_xyz(base_xyz)
    nearby_atoms = _parse_xyz(nearby_xyz)
    pocket_atoms = _parse_xyz(pocket_xyz)

    zn_atom = _find_zn(base_atoms)
    if zn_atom is None:
        raise SystemExit(f"Could not locate Zn atom in {base_xyz}")

    zn_b = _safe_float(zn_atom.meta.get("BF"))
    zn_mean = zn_b if zn_b is not None else -1.0

    coordinators = _coordinating_residues(base_atoms, zn_atom)
    if not coordinators:
        raise SystemExit("No coordinating residues detected (distance <= 3.0 A from Zn).")

    pdb_path = _resolve_pdb_for_base(base_xyz)
    if not pdb_path.exists():
        raise SystemExit(f"PDB not found for sequence-neighbor mapping: {pdb_path}")
    residue_order = _parse_pdb_residue_order(pdb_path)

    stats: List[GroupStat] = [GroupStat(label="Zn", mean_b=zn_mean, std_b=0.0, color="black", marker="o")]

    coord_set = set(coordinators)
    for coord in coordinators:
        coord_vals = _bf_values_for_residue(base_atoms, coord)
        c_mean, c_std = _mean_std(coord_vals)
        color = _coord_color(coord.resname)
        stats.append(
            GroupStat(
                label=f"{coord.resname}{coord.resseq}",
                mean_b=c_mean,
                std_b=c_std,
                color=color,
                marker="o",
            )
        )

        nset = _neighbor_residue_set(coord, coord_set, residue_order)
        near_vals = _bf_values_for_residue_set(nearby_atoms, nset)
        n_mean, n_std = _mean_std(near_vals)
        stats.append(
            GroupStat(
                label=f"near{coord.resseq}",
                mean_b=n_mean,
                std_b=n_std,
                color=color,
                marker=">",
            )
        )

    pocket_vals: List[float] = []
    for atom in pocket_atoms:
        if atom.element == "H":
            continue
        rkey = _residue_key(atom)
        if rkey is None:
            continue
        if rkey in coord_set:
            continue
        if rkey.resname == "ZN":
            continue
        bf = _safe_float(atom.meta.get("BF"))
        if bf is not None:
            pocket_vals.append(bf)

    p_mean, p_std = _mean_std(pocket_vals)
    stats.append(GroupStat(label="Pocket", mean_b=p_mean, std_b=p_std, color="gray", marker="s"))

    if out_png is None:
        out_png = xyz_dir / f"{stem}_bfactor_analysis.png"
    _plot_stats(stats, out_png, title=stem)

    print("Group,Mean_B,Std_B")
    for s in stats:
        print(f"{s.label},{s.mean_b:.4f},{s.std_b:.4f}")
    print(f"Saved plot: {out_png}")
    return out_png


def main() -> None:
    parser = argparse.ArgumentParser(description="B-factor analysis for base/backbone/nearby/pocket XYZ files.")
    parser.add_argument("base_xyz", type=Path, help="Base XYZ filename/path, e.g. 1a1h_ZN_homo_d3.00_cluster1.xyz")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path (default: alongside base xyz)")
    args = parser.parse_args()

    run_analysis(args.base_xyz, args.out)


if __name__ == "__main__":
    main()
