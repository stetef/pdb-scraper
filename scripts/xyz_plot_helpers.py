#!/usr/bin/env python3
"""Helper utilities for XYZ validation plots and HTML viewers."""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, deque
import json
import numpy as np
import re
import shutil
import subprocess
import sys
import webbrowser
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Callable, Dict, List, Tuple


PALETTE = [
    "#2A7DBF", "#E9C46A",
    "#89B397", "#B7410E",
    "#6D597A", "#3A3D42",
    "#457B9D", "#4A7C59",
    "#8FA998", "#F4978E",
]


def _apply_hist_rcparams(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.labelsize": 18,
            "axes.titlesize": 22,
            "axes.linewidth": 2,
            "legend.fontsize": 18,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "white",
            "legend.fancybox": False,
            "legend.shadow": False,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "xtick.major.size": 8,
            "ytick.major.size": 8,
            "xtick.major.width": 2,
            "ytick.major.width": 2,
            "xtick.minor.size": 4,
            "ytick.minor.size": 4,
            "xtick.minor.width": 1.5,
            "ytick.minor.width": 1.5,
        }
    )


def _build_reference_gray_cycle(count: int) -> List[str]:
    if count <= 0:
        return []
    if count == 1:
        return ["#000000"]
    max_val = 0xAA
    step = max_val / (count - 1)
    colors: List[str] = []
    for i in range(count):
        val = int(round(i * step))
        val = max(0, min(max_val, val))
        colors.append(f"#{val:02X}{val:02X}{val:02X}")
    return colors


@dataclass(frozen=True)
class XyzAtom:
    element: str
    x: float
    y: float
    z: float
    meta: Dict[str, str]
    raw_comment: str


@dataclass
class MetricAccum:
    values: List[float]
    sources: List[str]
    files_with_lt_expected: int = 0


@dataclass(frozen=True)
class MetricSpec:
    key: str
    compute: Callable[[List[XyzAtom]], List[float]]
    expected_per_file: int
    title: str
    xlabel: str
    out_png_name: str
    color: str
    bins: int | str = "auto"
    unit: str = "Å"


def _output_png_path(out_png_name: str, output_dir: Path, system_label: str | None) -> str:
    base = Path(out_png_name).name
    stem = Path(base).stem
    suffix = Path(base).suffix or ".png"
    if system_label:
        base = f"{stem}_{system_label}{suffix}"
    return str(output_dir / base)


def _accumulate_metric(
    metrics: Dict[str, MetricAccum],
    spec: MetricSpec,
    atoms: List[XyzAtom],
    *,
    source_name: str,
) -> None:
    acc = metrics[spec.key]
    vals = spec.compute(atoms)
    if len(vals) < spec.expected_per_file:
        acc.files_with_lt_expected += 1
    acc.values.extend(vals)
    acc.sources.extend([source_name] * len(vals))


def _print_metric_summary(spec: MetricSpec, acc: MetricAccum, *, total_files: int) -> None:
    if not acc.values:
        return
    arr = np.asarray(acc.values, dtype=float)
    mean = float(arr.mean())
    unit = spec.unit
    unit_suffix = unit if unit in ("°", "%") else f" {unit}" if unit else ""
    print(
        f"{spec.title}: "
        f"{len(acc.values)} values from {total_files} file(s) "
        f"(min={float(arr.min()):.3f}{unit_suffix}, max={float(arr.max()):.3f}{unit_suffix}, mean={mean:.3f}{unit_suffix})"
    )
    if acc.files_with_lt_expected:
        print(
            f"Note: {acc.files_with_lt_expected} file(s) had <{spec.expected_per_file} contributing values; "
            "their histogram contribution is truncated."
        )


def _print_values_summary(label: str, values: List[float], *, unit: str) -> None:
    if not values:
        print(f"{label}: 0 values")
        return
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    unit_suffix = unit if unit in ("°", "%") else f" {unit}" if unit else ""
    print(
        f"{label}: {len(values)} values "
        f"(min={float(arr.min()):.3f}{unit_suffix}, max={float(arr.max()):.3f}{unit_suffix}, mean={mean:.3f}{unit_suffix})"
    )


def _format_reference_label(values: List[float], *, unit: str) -> str:
    if not values:
        return "0 values"
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    unit_suffix = unit if unit in ("°", "%") else f" {unit}" if unit else ""
    if unit == "°":
        mean_str = f"{mean:.1f}"
    elif unit == "Å":
        mean_str = f"{mean:.2f}"
    else:
        mean_str = f"{mean:.3f}"
    return f"{len(values)} values (mean={mean_str}{unit_suffix})"


def _format_reference_label_mean_only(values: List[float], *, unit: str) -> str:
    if not values:
        return "0 values"
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if unit == "Å":
        return f"{mean:.2f} A"
    unit_suffix = unit if unit in ("°", "%") else f" {unit}" if unit else ""
    return f"{mean:.2f}{unit_suffix}"


def _parse_meta_comment(comment: str) -> Dict[str, str]:
    # Comment is expected like: "RES=CYS CHAIN=B RESSEQ=111 ATOM=CA REC=ATOM"
    meta: Dict[str, str] = {}
    for token in comment.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key and value:
            meta[key] = value
    return meta


def parse_xyz_atoms(path: Path) -> List[XyzAtom]:
    atoms: List[XyzAtom] = []
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 2:
        return atoms

    for line in lines[2:]:
        stripped = line.strip()
        if not stripped:
            continue

        # Split data vs comment
        if "#" in stripped:
            left, right = stripped.split("#", 1)
            comment = right.strip()
        else:
            left, comment = stripped, ""

        parts = left.split()
        if len(parts) < 4:
            continue

        element = parts[0]
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue

        meta = _parse_meta_comment(comment) if comment else {}
        atoms.append(XyzAtom(element=element, x=x, y=y, z=z, meta=meta, raw_comment=comment))

    return atoms


def _normalize_residue_letter(residue: str) -> str:
    res = residue.strip().upper()
    if res == "HIS":
        return "H"
    if res == "CYS":
        return "C"
    return res[:1] if res else ""


def _parse_resseq_int(resseq: str) -> int | None:
    try:
        return int(resseq)
    except (TypeError, ValueError):
        return None


def _build_chain_sequence(residue_map: Dict[int, str], *, use_latex: bool) -> str:
    items = sorted(residue_map.items())
    if not items:
        return ""
    parts = [items[0][1]]
    prev_resseq = items[0][0]
    for resseq, letter in items[1:]:
        gap = resseq - prev_resseq
        if gap > 1:
            if use_latex:
                parts.append(f"$x_{{{gap - 1}}}$")
            else:
                parts.append(f"x{gap - 1}")
        parts.append(letter)
        prev_resseq = resseq
    return "".join(parts)


def _format_sequence_distance(by_chain: Dict[str, Dict[int, str]], *, use_latex: bool) -> str:
    chain_ids = sorted(by_chain.keys())
    if not chain_ids:
        return ""
    if len(chain_ids) == 1:
        return _build_chain_sequence(by_chain[chain_ids[0]], use_latex=use_latex)
    chunks: List[str] = []
    for chain_id in chain_ids:
        seq = _build_chain_sequence(by_chain[chain_id], use_latex=use_latex)
        chunks.append(f"({seq})")
    return "".join(chunks)


def _sequence_distance_from_atoms(atoms: List[XyzAtom], *, use_latex: bool) -> str:
    by_chain: Dict[str, Dict[int, str]] = {}
    for atom in atoms:
        if atom.meta.get("ATOM") != "CA":
            continue
        resseq_raw = atom.meta.get("RESSEQ")
        if not resseq_raw:
            continue
        resseq = _parse_resseq_int(resseq_raw)
        if resseq is None:
            continue
        chain = atom.meta.get("CHAIN", "")
        chain_id = chain if chain else "_"
        residue = atom.meta.get("RES", "")
        by_chain.setdefault(chain_id, {}).setdefault(
            resseq, _normalize_residue_letter(residue)
        )
    return _format_sequence_distance(by_chain, use_latex=use_latex)


def _secstruct_letter(value: str | None) -> str:
    sec = (value or "").strip().upper()
    if sec == "HELIX" or sec == "H":
        return "H"
    if sec == "SHEET" or sec == "S":
        return "S"
    if sec == "LOOP" or sec == "L":
        return "L"
    return "?"


def _build_chain_secstruct(residue_map: Dict[int, str]) -> str:
    items = sorted(residue_map.items())
    if not items:
        return ""
    return "".join(letter for _, letter in items)


def _format_secstruct_distance(by_chain: Dict[str, Dict[int, str]]) -> str:
    chain_ids = sorted(by_chain.keys())
    if not chain_ids:
        return ""
    if len(chain_ids) == 1:
        return _build_chain_secstruct(by_chain[chain_ids[0]])
    chunks: List[str] = []
    for chain_id in chain_ids:
        seq = _build_chain_secstruct(by_chain[chain_id])
        chunks.append(f"({seq})")
    return "".join(chunks)


def _sequence_secstruct_from_atoms(atoms: List[XyzAtom]) -> str:
    by_chain: Dict[str, Dict[int, str]] = {}
    for atom in atoms:
        if atom.meta.get("ATOM") != "CA":
            continue
        resseq_raw = atom.meta.get("RESSEQ")
        if not resseq_raw:
            continue
        resseq = _parse_resseq_int(resseq_raw)
        if resseq is None:
            continue
        chain = atom.meta.get("CHAIN", "")
        chain_id = chain if chain else "_"
        sec_value = atom.meta.get("SEC") or atom.meta.get("SECSTRUCT")
        by_chain.setdefault(chain_id, {}).setdefault(
            resseq, _secstruct_letter(sec_value)
        )
    return _format_secstruct_distance(by_chain)


def write_sequence_distance_file(
    xyz_files: List[Path],
    *,
    output_dir: Path,
    system_label: str | None,
) -> None:
    sequence_counts: Counter[str] = Counter()
    secstruct_counts: Dict[str, Counter[str]] = {}
    for p in xyz_files:
        atoms = parse_xyz_atoms(p)
        seq_text = _sequence_distance_from_atoms(atoms, use_latex=False)
        if seq_text:
            sequence_counts[seq_text] += 1
            sec_text = _sequence_secstruct_from_atoms(atoms)
            if sec_text:
                secstruct_counts.setdefault(seq_text, Counter())[sec_text] += 1

    name = "seqence_dist"
    if system_label:
        name = f"{name}_{system_label}"
    out_path = output_dir / f"{name}.txt"
    lines: List[str] = []
    for seq, count in sorted(sequence_counts.items()):
        sec_counts = secstruct_counts.get(seq, Counter())
        if sec_counts:
            sec_parts = [f"{sec}:{n}" for sec, n in sorted(sec_counts.items())]
            lines.append(f"{seq} {count} {', '.join(sec_parts)}")
        else:
            lines.append(f"{seq} {count}")
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _family_token_from_grouped_text(value: str) -> str:
    text = value.strip()
    groups = re.findall(r"\(([^()]*)\)", text)
    if groups:
        pieces = [g.strip() for g in groups if g.strip()]
        if pieces:
            return "-".join(pieces)
    return text.replace(" ", "")


def _extract_pdb_cluster_from_xyz_name(xyz_path: Path) -> Tuple[str, str]:
    stem = xyz_path.stem
    parts = [p for p in stem.split("_") if p]
    pdb_id = parts[0] if parts else stem
    cluster = next((p for p in parts if p.lower().startswith("cluster")), "clusterunknown")
    return pdb_id, cluster


def write_family_struct_examples(
    xyz_files: List[Path],
    *,
    output_dir: Path,
) -> None:
    family_dir = output_dir / "family-structs"
    family_dir.mkdir(parents=True, exist_ok=True)

    first_by_family: Dict[Tuple[str, str], Path] = {}
    for xyz_path in xyz_files:
        atoms = parse_xyz_atoms(xyz_path)
        seq_text = _sequence_distance_from_atoms(atoms, use_latex=False)
        sec_text = _sequence_secstruct_from_atoms(atoms)
        if not seq_text or not sec_text:
            continue
        key = (seq_text, sec_text)
        if key not in first_by_family:
            first_by_family[key] = xyz_path

    copied = 0
    for (seq_text, sec_text), src_path in sorted(first_by_family.items()):
        seq_token = _family_token_from_grouped_text(seq_text)
        sec_token = _family_token_from_grouped_text(sec_text)
        pdb_id, cluster = _extract_pdb_cluster_from_xyz_name(src_path)
        out_name = f"{seq_token}-{sec_token}-{pdb_id}-{cluster}.xyz"
        shutil.copy2(src_path, family_dir / out_name)
        copied += 1

    print(f"Saved {copied} family structure example(s) to {family_dir}")


def count_cys_residues_with_ca_cb_sg(atoms: List[XyzAtom]) -> int:
    """Count CYS residues (by CHAIN+RESSEQ) that have CA, CB, and SG atoms."""
    by_residue: Dict[Tuple[str, str], set[str]] = {}
    for a in atoms:
        if a.meta.get("RES") != "CYS":
            continue
        chain = a.meta.get("CHAIN", "")
        resseq = a.meta.get("RESSEQ")
        atom_name = a.meta.get("ATOM")
        if not resseq or not atom_name:
            continue
        by_residue.setdefault((chain, resseq), set()).add(atom_name)

    return sum(1 for atom_names in by_residue.values() if {"CA", "CB", "SG"}.issubset(atom_names))


def check_dir_for_non_four_cys_residues(xyz_dir: Path) -> List[Path]:
    """Return .xyz files whose count of CYS residues with CA/CB/SG != 4."""
    xyz_files = find_xyz_files(xyz_dir)
    bad: List[Path] = []
    for p in xyz_files:
        atoms = parse_xyz_atoms(p)
        n = count_cys_residues_with_ca_cb_sg(atoms)
        if n != 4:
            bad.append(p)
    return bad


def sg_bond_lengths_to_center(atoms: List[XyzAtom], *, max_sgs: int = 4) -> List[float]:
    """Return bond lengths from center (origin) to up to `max_sgs` nearest CYS SG atoms.

    In this repo's XYZ files, coordinates are translated such that the target/center atom
    is at (0,0,0). Some outlier files contain a 5th CYS SG that is farther away and not
    coordinated; we drop it by taking the `max_sgs` smallest distances.
    """
    sg_atoms = [
        a
        for a in atoms
        if a.meta.get("RES") == "CYS" and a.meta.get("ATOM") == "SG"
    ]
    if not sg_atoms:
        return []
    coords = np.asarray([(a.x, a.y, a.z) for a in sg_atoms], dtype=float)
    dists = np.linalg.norm(coords, axis=1)
    dists_sorted = np.sort(dists)
    return dists_sorted[:max_sgs].astype(float).tolist()


def his_coord_bond_lengths_to_center(atoms: List[XyzAtom], *, max_residues: int = 4) -> List[float]:
    """Return distances from origin to coordinating HIS atom (ND1/NE2), up to `max_residues`.

    For each HIS residue (CHAIN+RESSEQ), choose the nearest coordinating atom
    (ND1 or NE2). Then keep the `max_residues` residues with smallest coordinating
    atom distance-to-origin.
    """
    by_residue: Dict[Tuple[str, str], Dict[str, XyzAtom]] = {}
    for a in atoms:
        if a.meta.get("RES") != "HIS":
            continue
        atom_name = a.meta.get("ATOM")
        if atom_name not in {"ND1", "NE2"}:
            continue
        chain = a.meta.get("CHAIN", "")
        resseq = a.meta.get("RESSEQ")
        if not resseq:
            continue
        by_residue.setdefault((chain, resseq), {})[atom_name] = a

    candidates: List[Tuple[float, float]] = []
    for atom_map in by_residue.values():
        best = None
        best_d2 = None
        for name in ("ND1", "NE2"):
            a = atom_map.get(name)
            if a is None:
                continue
            d2 = float(a.x * a.x + a.y * a.y + a.z * a.z)
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best = a
        if best is None or best_d2 is None:
            continue
        candidates.append((best_d2, float(np.sqrt(best_d2))))

    candidates.sort(key=lambda t: t[0])
    return [d for _, d in candidates[:max_residues]]


def _angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float | None:
    v1_norm = float(np.linalg.norm(v1))
    v2_norm = float(np.linalg.norm(v2))
    if v1_norm == 0.0 or v2_norm == 0.0:
        return None
    cosang = float(np.dot(v1, v2) / (v1_norm * v2_norm))
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def _dihedral_angle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float | None:
    """Return dihedral angle in degrees for p0–p1–p2–p3."""
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = float(np.linalg.norm(b1))
    if b1_norm == 0.0:
        return None

    n1 = np.cross(b0, b1)
    n2 = np.cross(b1, b2)
    n1_norm = float(np.linalg.norm(n1))
    n2_norm = float(np.linalg.norm(n2))
    if n1_norm == 0.0 or n2_norm == 0.0:
        return None

    b1u = b1 / b1_norm
    m1 = np.cross(n1, b1u)

    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))
    return float(np.degrees(np.arctan2(y, x)))


def _his_atoms_by_residue(
    atoms: List[XyzAtom],
    *,
    required_atoms: set[str],
) -> Dict[Tuple[str, str], Dict[str, XyzAtom]]:
    by_residue: Dict[Tuple[str, str], Dict[str, XyzAtom]] = {}
    for a in atoms:
        if a.meta.get("RES") != "HIS":
            continue
        atom_name = a.meta.get("ATOM")
        if atom_name not in required_atoms:
            continue
        chain = a.meta.get("CHAIN", "")
        resseq = a.meta.get("RESSEQ")
        if not resseq:
            continue
        k = (chain, resseq)
        by_residue.setdefault(k, {})[atom_name] = a
    return by_residue


def _select_his_coord_residues(
    atoms: List[XyzAtom],
    *,
    max_residues: int = 4,
    required_atoms: set[str] | None = None,
) -> List[Tuple[str, Dict[str, XyzAtom]]]:
    required = required_atoms or {"ND1", "NE2"}
    required = set(required) | {"ND1", "NE2"}
    by_residue = _his_atoms_by_residue(atoms, required_atoms=required)

    candidates: List[Tuple[float, str, Dict[str, XyzAtom]]] = []
    for atom_map in by_residue.values():
        for name in ("ND1", "NE2"):
            a = atom_map.get(name)
            if a is None:
                continue
            if a.meta.get("COORD") == "1":
                d2 = float(a.x * a.x + a.y * a.y + a.z * a.z)
                candidates.append((d2, name, atom_map))
                break
        else:
            best = None
            best_d2 = None
            best_name = None
            for name in ("ND1", "NE2"):
                a = atom_map.get(name)
                if a is None:
                    continue
                d2 = float(a.x * a.x + a.y * a.y + a.z * a.z)
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best = a
                    best_name = name
            if best is None or best_d2 is None or best_name is None:
                continue
            candidates.append((best_d2, best_name, atom_map))

    candidates.sort(key=lambda t: t[0])
    return [(name, atom_map) for _, name, atom_map in candidates[:max_residues]]


def _select_his_coord_residue_keys(
    atoms: List[XyzAtom],
    *,
    max_residues: int = 4,
) -> List[Tuple[Tuple[str, str], str]]:
    """Return (CHAIN, RESSEQ) and coordinating atom name for nearest HIS residues."""
    by_residue = _his_atoms_by_residue(atoms, required_atoms={"ND1", "NE2"})
    candidates: List[Tuple[float, Tuple[str, str], str]] = []
    for key, atom_map in by_residue.items():
        for name in ("ND1", "NE2"):
            a = atom_map.get(name)
            if a is None:
                continue
            if a.meta.get("COORD") == "1":
                d2 = float(a.x * a.x + a.y * a.y + a.z * a.z)
                candidates.append((d2, key, name))
                break
        else:
            best_d2 = None
            best_name = None
            for name in ("ND1", "NE2"):
                a = atom_map.get(name)
                if a is None:
                    continue
                d2 = float(a.x * a.x + a.y * a.y + a.z * a.z)
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_name = name
            if best_d2 is None or best_name is None:
                continue
            candidates.append((best_d2, key, best_name))

    candidates.sort(key=lambda t: t[0])
    return [(key, name) for _, key, name in candidates[:max_residues]]


def his_coord_distances_by_atom(
    atoms: List[XyzAtom],
    *,
    max_residues: int = 4,
) -> Dict[str, List[float]]:
    """Return Zn→ND1 and Zn→NE2 distances for selected coordinating HIS residues.

    Residues are selected by smallest coordinating-atom distance to origin, then
    distances are grouped by the coordinating atom name (ND1 vs NE2).
    """
    selected = _select_his_coord_residues(atoms, max_residues=max_residues)
    by_atom: Dict[str, List[float]] = {"ND1": [], "NE2": []}
    for coord_name, atom_map in selected:
        coord = atom_map.get(coord_name)
        if coord is None:
            continue
        dist = float(np.sqrt(coord.x * coord.x + coord.y * coord.y + coord.z * coord.z))
        by_atom[coord_name].append(dist)
    return by_atom


def his_coord_cg_angles_by_atom(
    atoms: List[XyzAtom],
    *,
    max_residues: int = 4,
) -> Dict[str, List[float]]:
    """Return angle Zn–N–CG (N=ND1/NE2) grouped by coordinating atom.

    Uses the same residue selection as `his_coord_distances_by_atom`.
    """
    selected = _select_his_coord_residues(
        atoms,
        max_residues=max_residues,
        required_atoms={"ND1", "NE2", "CG"},
    )
    by_atom: Dict[str, List[float]] = {"ND1": [], "NE2": []}
    for coord_name, atom_map in selected:
        coord = atom_map.get(coord_name)
        cg = atom_map.get("CG")
        if coord is None or cg is None:
            continue
        v1 = np.asarray((-coord.x, -coord.y, -coord.z), dtype=float)
        v2 = np.asarray((cg.x - coord.x, cg.y - coord.y, cg.z - coord.z), dtype=float)
        ang = _angle_between_vectors(v1, v2)
        if ang is None:
            continue
        by_atom[coord_name].append(ang)
    return by_atom


def his_atom_distances_by_coord_atom(
    atoms: List[XyzAtom],
    atom_name: str,
    *,
    max_residues: int = 4,
) -> Dict[str, List[float]]:
    """Return Zn→atom distances grouped by coordinating atom (ND1/NE2)."""
    selected = _select_his_coord_residues(
        atoms,
        max_residues=max_residues,
        required_atoms={"ND1", "NE2", atom_name},
    )
    by_atom: Dict[str, List[float]] = {"ND1": [], "NE2": []}
    for coord_name, atom_map in selected:
        target = atom_map.get(atom_name)
        if target is None:
            continue
        dist = float(np.sqrt(target.x * target.x + target.y * target.y + target.z * target.z))
        by_atom[coord_name].append(dist)
    return by_atom


def his_coord_cg_ca_angles_by_atom(
    atoms: List[XyzAtom],
    *,
    max_residues: int = 4,
) -> Dict[str, List[float]]:
    """Return angle ND/NE–CG–CB grouped by coordinating atom."""
    selected = _select_his_coord_residue_keys(atoms, max_residues=max_residues)
    by_residue = _his_atoms_by_residue(atoms, required_atoms={"ND1", "NE2", "CG", "CB"})
    by_atom: Dict[str, List[float]] = {"ND1": [], "NE2": []}
    for key, coord_name in selected:
        atom_map = by_residue.get(key)
        if atom_map is None:
            continue
        coord = atom_map.get(coord_name)
        cg = atom_map.get("CG")
        cb = atom_map.get("CB")
        if coord is None or cg is None or cb is None:
            continue
        v1 = np.asarray((coord.x - cg.x, coord.y - cg.y, coord.z - cg.z), dtype=float)
        v2 = np.asarray((cb.x - cg.x, cb.y - cg.y, cb.z - cg.z), dtype=float)
        ang = _angle_between_vectors(v1, v2)
        if ang is None:
            continue
        by_atom[coord_name].append(ang)
    return by_atom


def his_cg_cb_ca_angles_by_atom(
    atoms: List[XyzAtom],
    *,
    max_residues: int = 4,
) -> Dict[str, List[float]]:
    """Return angle CG–CB–CA grouped by coordinating atom."""
    selected = _select_his_coord_residues(
        atoms,
        max_residues=max_residues,
        required_atoms={"ND1", "NE2", "CG", "CB", "CA"},
    )
    by_atom: Dict[str, List[float]] = {"ND1": [], "NE2": []}
    for coord_name, atom_map in selected:
        cg = atom_map.get("CG")
        cb = atom_map.get("CB")
        ca = atom_map.get("CA")
        if cg is None or cb is None or ca is None:
            continue
        v1 = np.asarray((cg.x - cb.x, cg.y - cb.y, cg.z - cb.z), dtype=float)
        v2 = np.asarray((ca.x - cb.x, ca.y - cb.y, ca.z - cb.z), dtype=float)
        ang = _angle_between_vectors(v1, v2)
        if ang is None:
            continue
        by_atom[coord_name].append(ang)
    return by_atom


def his_coord_zn_cg_ca_dihedral_by_atom(
    atoms: List[XyzAtom],
    *,
    max_residues: int = 4,
) -> Dict[str, List[float]]:
    """Return dihedral Zn–N–CG–CA grouped by coordinating atom."""
    selected = _select_his_coord_residues(
        atoms,
        max_residues=max_residues,
        required_atoms={"ND1", "NE2", "CG", "CA"},
    )
    by_atom: Dict[str, List[float]] = {"ND1": [], "NE2": []}
    p0 = np.asarray((0.0, 0.0, 0.0), dtype=float)
    for coord_name, atom_map in selected:
        coord = atom_map.get(coord_name)
        cg = atom_map.get("CG")
        ca = atom_map.get("CA")
        if coord is None or cg is None or ca is None:
            continue
        p1 = np.asarray((coord.x, coord.y, coord.z), dtype=float)
        p2 = np.asarray((cg.x, cg.y, cg.z), dtype=float)
        p3 = np.asarray((ca.x, ca.y, ca.z), dtype=float)
        ang = _dihedral_angle(p0, p1, p2, p3)
        if ang is None:
            continue
        by_atom[coord_name].append(ang)
    return by_atom


def cys_mean_coord_distance(atoms: List[XyzAtom], *, max_sgs: int = 4) -> float | None:
    """Return mean Zn→SG distance per structure (nearest SGs)."""
    dists = sg_bond_lengths_to_center(atoms, max_sgs=max_sgs)
    if not dists:
        return None
    return float(np.mean(dists))


def his_mean_coord_distance(atoms: List[XyzAtom], *, max_residues: int = 4) -> float | None:
    """Return mean Zn→ND1/NE2 distance per structure (selected residues)."""
    dists = his_coord_bond_lengths_to_center(atoms, max_residues=max_residues)
    if not dists:
        return None
    return float(np.mean(dists))


def ca_to_ca_distances_from_seed(atoms: List[XyzAtom]) -> List[float]:
    """Return pairwise CA distances by walking forward through the CA list.

    Uses the CA order from the file: take the first CA as seed (to others),
    then the second CA (to remaining), then the third CA (to last). This yields
    6 distances for 4 CA atoms.
    """
    ca_atoms = [a for a in atoms if a.meta.get("ATOM") == "CA"]
    if len(ca_atoms) < 2:
        return []
    ca_vecs = [np.asarray((a.x, a.y, a.z), dtype=float) for a in ca_atoms]
    dists: List[float] = []
    # Walk forward to avoid duplicate pairs (i,j) and (j,i).
    for i in range(len(ca_vecs) - 1):
        vi = ca_vecs[i]
        for vj in ca_vecs[i + 1 :]:
            dists.append(float(np.linalg.norm(vj - vi)))
    return dists


def _select_coord_residue_candidates_for_ca(
    atoms: List[XyzAtom],
    *,
    max_residues: int = 4,
) -> List[Tuple[str, str, np.ndarray]]:
    """Select up to `max_residues` coordinating residues and return their CA coords.

    Returns a list of (res_kind, coord_name, ca_vec), where:
      - res_kind is "CYS" or "HIS"
      - coord_name is "SG" for CYS, "ND1" or "NE2" for HIS
    Selection is based on smallest coordinating-atom distance to origin
    across both CYS and HIS residues.
    """
    candidates: List[Tuple[float, str, str, np.ndarray]] = []

    # CYS candidates: require SG + CA.
    cys_by_residue = _cys_atoms_by_residue(atoms, required_atoms={"SG", "CA"})
    for atom_map in cys_by_residue.values():
        sg = atom_map.get("SG")
        ca = atom_map.get("CA")
        if sg is None or ca is None:
            continue
        sg_vec = np.asarray((sg.x, sg.y, sg.z), dtype=float)
        sg_d2 = float(np.dot(sg_vec, sg_vec))
        ca_vec = np.asarray((ca.x, ca.y, ca.z), dtype=float)
        candidates.append((sg_d2, "CYS", "SG", ca_vec))

    # HIS candidates: require ND1/NE2 + CA, choose coordinating atom by COORD or nearest.
    his_by_residue = _his_atoms_by_residue(atoms, required_atoms={"ND1", "NE2", "CA"})
    for atom_map in his_by_residue.values():
        ca = atom_map.get("CA")
        if ca is None:
            continue

        coord_name = None
        coord_atom = None
        for name in ("ND1", "NE2"):
            a = atom_map.get(name)
            if a is None:
                continue
            if a.meta.get("COORD") == "1":
                coord_name = name
                coord_atom = a
                break

        if coord_atom is None:
            best_d2 = None
            for name in ("ND1", "NE2"):
                a = atom_map.get(name)
                if a is None:
                    continue
                d2 = float(a.x * a.x + a.y * a.y + a.z * a.z)
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    coord_name = name
                    coord_atom = a
        if coord_atom is None or coord_name is None:
            continue

        coord_vec = np.asarray((coord_atom.x, coord_atom.y, coord_atom.z), dtype=float)
        coord_d2 = float(np.dot(coord_vec, coord_vec))
        ca_vec = np.asarray((ca.x, ca.y, ca.z), dtype=float)
        candidates.append((coord_d2, "HIS", coord_name, ca_vec))

    candidates.sort(key=lambda t: t[0])
    selected = candidates[:max_residues]
    return [(kind, coord_name, ca_vec) for _, kind, coord_name, ca_vec in selected]


def _select_coord_residue_candidates_for_ca_with_coord(
    atoms: List[XyzAtom],
    *,
    max_residues: int = 4,
) -> List[Tuple[str, str, np.ndarray, float]]:
    """Select coordinating residues and return their CA coords and coord distances.

    Returns a list of (res_kind, coord_name, ca_vec, coord_dist), where coord_dist is
    the Zn-to-coordinating-atom distance (SG for CYS, ND1/NE2 for HIS).
    """
    candidates: List[Tuple[float, str, str, np.ndarray, float]] = []

    cys_by_residue = _cys_atoms_by_residue(atoms, required_atoms={"SG", "CA"})
    for atom_map in cys_by_residue.values():
        sg = atom_map.get("SG")
        ca = atom_map.get("CA")
        if sg is None or ca is None:
            continue
        sg_vec = np.asarray((sg.x, sg.y, sg.z), dtype=float)
        sg_d2 = float(np.dot(sg_vec, sg_vec))
        ca_vec = np.asarray((ca.x, ca.y, ca.z), dtype=float)
        candidates.append((sg_d2, "CYS", "SG", ca_vec, float(np.sqrt(sg_d2))))

    his_by_residue = _his_atoms_by_residue(atoms, required_atoms={"ND1", "NE2", "CA"})
    for atom_map in his_by_residue.values():
        ca = atom_map.get("CA")
        if ca is None:
            continue

        coord_name = None
        coord_atom = None
        for name in ("ND1", "NE2"):
            a = atom_map.get(name)
            if a is None:
                continue
            if a.meta.get("COORD") == "1":
                coord_name = name
                coord_atom = a
                break

        if coord_atom is None:
            best_d2 = None
            for name in ("ND1", "NE2"):
                a = atom_map.get(name)
                if a is None:
                    continue
                d2 = float(a.x * a.x + a.y * a.y + a.z * a.z)
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    coord_name = name
                    coord_atom = a
        if coord_atom is None or coord_name is None:
            continue

        coord_vec = np.asarray((coord_atom.x, coord_atom.y, coord_atom.z), dtype=float)
        coord_d2 = float(np.dot(coord_vec, coord_vec))
        ca_vec = np.asarray((ca.x, ca.y, ca.z), dtype=float)
        candidates.append((coord_d2, "HIS", coord_name, ca_vec, float(np.sqrt(coord_d2))))

    candidates.sort(key=lambda t: t[0])
    selected = candidates[:max_residues]
    return [(kind, coord_name, ca_vec, coord_dist) for _, kind, coord_name, ca_vec, coord_dist in selected]


def _tetrahedron_volume(points: List[np.ndarray]) -> float | None:
    if len(points) < 4:
        return None
    p0, p1, p2, p3 = points[:4]
    mat = np.column_stack((p1 - p0, p2 - p0, p3 - p0))
    return float(abs(np.linalg.det(mat)) / 6.0)


def ca_volume_and_coord_distances(
    atoms: List[XyzAtom],
    *,
    max_residues: int = 4,
) -> Tuple[float | None, List[float]]:
    """Return CA tetrahedron volume and coordinating-atom distances for a structure."""
    selected = _select_coord_residue_candidates_for_ca_with_coord(atoms, max_residues=max_residues)
    if len(selected) < 4:
        return (None, [])
    ca_vecs = [ca_vec for _, _, ca_vec, _ in selected]
    coord_dists = [coord_dist for _, _, _, coord_dist in selected]
    volume = _tetrahedron_volume(ca_vecs)
    return (volume, coord_dists) if volume is not None else (None, [])


def ca_ca_distances_by_category(
    atoms: List[XyzAtom],
    *,
    max_residues: int = 4,
) -> Dict[str, List[float]]:
    """Return CA-CA pair distances grouped by CYS/HIS(ND1/NE2) categories."""
    categories = {
        "CYS-CYS": [],
        "CYS-HIS ND": [],
        "CYS-HIS NE": [],
        "HIS ND-HIS ND": [],
        "HIS ND-HIS NE": [],
        "HIS NE-HIS NE": [],
    }

    selected = _select_coord_residue_candidates_for_ca(atoms, max_residues=max_residues)
    if len(selected) < 2:
        return categories

    def _his_label(coord_name: str) -> str | None:
        if coord_name == "ND1":
            return "ND"
        if coord_name == "NE2":
            return "NE"
        return None

    for i in range(len(selected) - 1):
        kind_i, coord_i, vec_i = selected[i]
        for j in range(i + 1, len(selected)):
            kind_j, coord_j, vec_j = selected[j]
            dist = float(np.linalg.norm(vec_j - vec_i))

            if kind_i == "CYS" and kind_j == "CYS":
                categories["CYS-CYS"].append(dist)
                continue

            if kind_i == "CYS" and kind_j == "HIS":
                his_label = _his_label(coord_j)
                if his_label == "ND":
                    categories["CYS-HIS ND"].append(dist)
                elif his_label == "NE":
                    categories["CYS-HIS NE"].append(dist)
                continue

            if kind_i == "HIS" and kind_j == "CYS":
                his_label = _his_label(coord_i)
                if his_label == "ND":
                    categories["CYS-HIS ND"].append(dist)
                elif his_label == "NE":
                    categories["CYS-HIS NE"].append(dist)
                continue

            if kind_i == "HIS" and kind_j == "HIS":
                his_i = _his_label(coord_i)
                his_j = _his_label(coord_j)
                if his_i == "ND" and his_j == "ND":
                    categories["HIS ND-HIS ND"].append(dist)
                elif his_i == "NE" and his_j == "NE":
                    categories["HIS NE-HIS NE"].append(dist)
                else:
                    categories["HIS ND-HIS NE"].append(dist)

    return categories


def his_coordination_atom_counts(
    atoms: List[XyzAtom],
    *,
    max_atoms: int = 4,
) -> Tuple[int, int]:
    """Count ND1 vs NE2 among the closest coordinating HIS atoms per file."""
    candidates: List[Tuple[float, str]] = []

    # HIS candidates: nearest ND1/NE2 per residue
    his_selected = _select_his_coord_residues(atoms, max_residues=999)
    for coord_name, atom_map in his_selected:
        coord = atom_map.get(coord_name)
        if coord is None:
            continue
        d2 = float(coord.x * coord.x + coord.y * coord.y + coord.z * coord.z)
        candidates.append((d2, coord_name))

    if not candidates:
        return (0, 0)

    candidates.sort(key=lambda t: t[0])
    closest = candidates[:max_atoms]
    nd1_count = sum(1 for _, name in closest if name == "ND1")
    ne2_count = sum(1 for _, name in closest if name == "NE2")
    return (nd1_count, ne2_count)


def his_atom_distances_selected_by_coord(
    atoms: List[XyzAtom],
    atom_name: str,
    *,
    max_residues: int = 4,
) -> List[float]:
    """Return distances from origin to a HIS atom for residues selected by nearest ND1/NE2.

    For each HIS residue (CHAIN+RESSEQ), select the coordinating atom (nearest ND1/NE2).
    Keep the `max_residues` residues with smallest coordinating-atom distance-to-origin,
    then return distances from origin to `atom_name` for those residues (if present).
    """
    by_residue: Dict[Tuple[str, str], Dict[str, XyzAtom]] = {}
    for a in atoms:
        if a.meta.get("RES") != "HIS":
            continue
        an = a.meta.get("ATOM")
        if an not in {"ND1", "NE2", atom_name}:
            continue
        chain = a.meta.get("CHAIN", "")
        resseq = a.meta.get("RESSEQ")
        if not resseq:
            continue
        by_residue.setdefault((chain, resseq), {})[an] = a

    candidates: List[Tuple[float, float]] = []
    for atom_map in by_residue.values():
        best = None
        best_d2 = None
        for name in ("ND1", "NE2"):
            a = atom_map.get(name)
            if a is None:
                continue
            d2 = float(a.x * a.x + a.y * a.y + a.z * a.z)
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best = a
        if best is None or best_d2 is None:
            continue

        target = atom_map.get(atom_name)
        if target is None:
            continue
        dist = float(np.sqrt(target.x * target.x + target.y * target.y + target.z * target.z))
        candidates.append((best_d2, dist))

    candidates.sort(key=lambda t: t[0])
    return [d for _, d in candidates[:max_residues]]


def count_cys_sg_residues(atoms: List[XyzAtom]) -> int:
    """Count coordinating CYS residues by SG presence (CHAIN+RESSEQ)."""
    by_residue: Dict[Tuple[str, str], set[str]] = {}
    for a in atoms:
        if a.meta.get("RES") != "CYS":
            continue
        atom_name = a.meta.get("ATOM")
        if atom_name != "SG":
            continue
        chain = a.meta.get("CHAIN", "")
        resseq = a.meta.get("RESSEQ")
        if not resseq:
            continue
        by_residue.setdefault((chain, resseq), set()).add(atom_name)
    return len(by_residue)


def count_his_coord_residues(atoms: List[XyzAtom]) -> int:
    """Count coordinating HIS residues with ND1/NE2 (CHAIN+RESSEQ)."""
    by_residue: Dict[Tuple[str, str], set[str]] = {}
    for a in atoms:
        if a.meta.get("RES") != "HIS":
            continue
        atom_name = a.meta.get("ATOM")
        if atom_name not in {"ND1", "NE2"}:
            continue
        chain = a.meta.get("CHAIN", "")
        resseq = a.meta.get("RESSEQ")
        if not resseq:
            continue
        by_residue.setdefault((chain, resseq), set()).add(atom_name)
    return len(by_residue)


def _cys_atoms_by_residue(
    atoms: List[XyzAtom],
    *,
    required_atoms: set[str],
) -> Dict[Tuple[str, str], Dict[str, XyzAtom]]:
    by_residue: Dict[Tuple[str, str], Dict[str, XyzAtom]] = {}
    for a in atoms:
        if a.meta.get("RES") != "CYS":
            continue
        atom_name = a.meta.get("ATOM")
        if atom_name not in required_atoms:
            continue
        chain = a.meta.get("CHAIN", "")
        resseq = a.meta.get("RESSEQ")
        if not resseq:
            continue
        k = (chain, resseq)
        by_residue.setdefault(k, {})[atom_name] = a
    return by_residue


def distances_selected_by_sg(
    atoms: List[XyzAtom],
    point_a: str,
    point_b: str,
    *,
    max_residues: int = 4,
    selection_atom: str = "SG",
) -> List[float]:
    """Generic distance collector for nearest-SG CYS residues.

    This consolidates the repeated pattern used by:
      - cb_bond_lengths_to_center_selected_by_sg (CB ↔ origin)
      - ca_bond_lengths_to_center_selected_by_sg (CA ↔ origin)
      - sg_ca_distances_selected_by_sg (SG ↔ CA)
      - cb_ca_distances_selected_by_sg (CB ↔ CA)

    Rules:
      - Residues are keyed by (CHAIN, RESSEQ).
      - Only CYS residues are considered.
      - Residues are selected by smallest distance of `selection_atom` to origin.
      - `point_a` / `point_b` may be an atom name (e.g. "CB") or "origin".
    """

    required: set[str] = {selection_atom}
    if point_a != "origin":
        required.add(point_a)
    if point_b != "origin":
        required.add(point_b)

    by_residue = _cys_atoms_by_residue(atoms, required_atoms=required)

    candidates: List[Tuple[float, float]] = []
    for atom_map in by_residue.values():
        sel = atom_map.get(selection_atom)
        if sel is None:
            continue

        a_atom = None if point_a == "origin" else atom_map.get(point_a)
        b_atom = None if point_b == "origin" else atom_map.get(point_b)
        if point_a != "origin" and a_atom is None:
            continue
        if point_b != "origin" and b_atom is None:
            continue

        ax, ay, az = (0.0, 0.0, 0.0) if a_atom is None else (a_atom.x, a_atom.y, a_atom.z)
        bx, by, bz = (0.0, 0.0, 0.0) if b_atom is None else (b_atom.x, b_atom.y, b_atom.z)
        a_vec = np.asarray((ax, ay, az), dtype=float)
        b_vec = np.asarray((bx, by, bz), dtype=float)
        dist = float(np.linalg.norm(a_vec - b_vec))

        sel_vec = np.asarray((sel.x, sel.y, sel.z), dtype=float)
        sel_d2 = float(np.dot(sel_vec, sel_vec))
        candidates.append((sel_d2, dist))

    candidates.sort(key=lambda t: t[0])
    return [d for _, d in candidates[:max_residues]]


def cb_bond_lengths_to_center_selected_by_sg(atoms: List[XyzAtom], *, max_residues: int = 4) -> List[float]:
    """Return CB-to-center distances for the `max_residues` CYS residues with nearest SG.

    Rationale: some files contain an extra (5th) CYS SG that is farther away and not
    coordinated. For the CB histogram we want the CB distances corresponding to the
    *same* coordinating CYS residues, so we:
      1) group atoms into residues (CHAIN+RESSEQ)
      2) keep residues that have both SG and CB
      3) select the `max_residues` residues with smallest SG distance-to-origin
      4) return their CB distance-to-origin values
    """
    return distances_selected_by_sg(atoms, "CB", "origin", max_residues=max_residues, selection_atom="SG")


def ca_bond_lengths_to_center_selected_by_sg(atoms: List[XyzAtom], *, max_residues: int = 4) -> List[float]:
    """Return CA-to-center distances for the `max_residues` CYS residues with nearest SG.

    Uses the same residue selection rule as the CB histogram: pick the `max_residues`
    residues with smallest SG distance-to-origin, then report their CA distances.
    """
    return distances_selected_by_sg(atoms, "CA", "origin", max_residues=max_residues, selection_atom="SG")


def sg_ca_distances_selected_by_sg(atoms: List[XyzAtom], *, max_residues: int = 4) -> List[float]:
    """Return SG–CA distances for the `max_residues` CYS residues with nearest SG."""
    return distances_selected_by_sg(atoms, "SG", "CA", max_residues=max_residues, selection_atom="SG")


def cb_ca_distances_selected_by_sg(atoms: List[XyzAtom], *, max_residues: int = 4) -> List[float]:
    """Return CB–CA distances for the `max_residues` CYS residues with nearest SG."""
    return distances_selected_by_sg(atoms, "CB", "CA", max_residues=max_residues, selection_atom="SG")


def sg_cb_distances_selected_by_sg(atoms: List[XyzAtom], *, max_residues: int = 4) -> List[float]:
    """Return SG–CB distances for the `max_residues` CYS residues with nearest SG."""
    return distances_selected_by_sg(atoms, "SG", "CB", max_residues=max_residues, selection_atom="SG")


def sg_cb_angle_vs_radial_selected_by_sg(atoms: List[XyzAtom], *, max_residues: int = 4) -> List[float]:
    """Return the angle between (origin→SG) and (SG→CB) for nearest-SG CYS residues.

    For each CYS residue, define:
      - radial vector r = SG - origin = (sg.x, sg.y, sg.z)
      - bond-ish vector b = CB - SG

    Angle is computed as acos( dot(r, b) / (|r||b|) ) in degrees.

    Residue selection matches the other histograms: pick the `max_residues` residues
    with smallest SG distance-to-origin, then report their angles.
    """
    by_residue: Dict[Tuple[str, str], Dict[str, XyzAtom]] = {}
    for a in atoms:
        if a.meta.get("RES") != "CYS":
            continue
        atom_name = a.meta.get("ATOM")
        if atom_name not in {"SG", "CB"}:
            continue
        chain = a.meta.get("CHAIN", "")
        resseq = a.meta.get("RESSEQ")
        if not resseq:
            continue
        k = (chain, resseq)
        by_residue.setdefault(k, {})[atom_name] = a

    candidates: List[Tuple[float, float]] = []
    for atom_map in by_residue.values():
        sg = atom_map.get("SG")
        cb = atom_map.get("CB")
        if sg is None or cb is None:
            continue

        r = np.asarray((sg.x, sg.y, sg.z), dtype=float)
        b = np.asarray((cb.x - sg.x, cb.y - sg.y, cb.z - sg.z), dtype=float)

        r_norm = float(np.linalg.norm(r))
        b_norm = float(np.linalg.norm(b))
        if r_norm == 0.0 or b_norm == 0.0:
            continue

        cosang = float(np.dot(r, b) / (r_norm * b_norm))
        cosang = float(np.clip(cosang, -1.0, 1.0))
        ang_deg = float(np.degrees(np.arccos(cosang)))

        sg_d2 = float(np.dot(r, r))
        candidates.append((sg_d2, ang_deg))

    candidates.sort(key=lambda t: t[0])
    return [ang for _, ang in candidates[:max_residues]]


def sg_cb_ca_angle_selected_by_sg(atoms: List[XyzAtom], *, max_residues: int = 4) -> List[float]:
    """Return the angle between (SG→CB) and (CB→CA) for nearest-SG CYS residues.

    For each CYS residue, define:
      - vector v1 = CB - SG
      - vector v2 = CA - CB

    Angle is computed as acos( dot(v1, v2) / (|v1||v2|) ) in degrees.

    Residue selection matches the other histograms: pick the `max_residues` residues
    with smallest SG distance-to-origin, then report their angles.
    """
    by_residue: Dict[Tuple[str, str], Dict[str, XyzAtom]] = {}
    for a in atoms:
        if a.meta.get("RES") != "CYS":
            continue
        atom_name = a.meta.get("ATOM")
        if atom_name not in {"SG", "CB", "CA"}:
            continue
        chain = a.meta.get("CHAIN", "")
        resseq = a.meta.get("RESSEQ")
        if not resseq:
            continue
        k = (chain, resseq)
        by_residue.setdefault(k, {})[atom_name] = a

    candidates: List[Tuple[float, float]] = []
    for atom_map in by_residue.values():
        sg = atom_map.get("SG")
        cb = atom_map.get("CB")
        ca = atom_map.get("CA")
        if sg is None or cb is None or ca is None:
            continue

        v1 = np.asarray((cb.x - sg.x, cb.y - sg.y, cb.z - sg.z), dtype=float)
        v2 = np.asarray((ca.x - cb.x, ca.y - cb.y, ca.z - cb.z), dtype=float)

        v1_norm = float(np.linalg.norm(v1))
        v2_norm = float(np.linalg.norm(v2))
        if v1_norm == 0.0 or v2_norm == 0.0:
            continue

        cosang = float(np.dot(v1, v2) / (v1_norm * v2_norm))
        cosang = float(np.clip(cosang, -1.0, 1.0))
        ang_deg = float(np.degrees(np.arccos(cosang)))

        sg_pos = np.asarray((sg.x, sg.y, sg.z), dtype=float)
        sg_d2 = float(np.dot(sg_pos, sg_pos))
        candidates.append((sg_d2, ang_deg))

    candidates.sort(key=lambda t: t[0])
    return [ang for _, ang in candidates[:max_residues]]


def sg_cb_ca_angle_selected_by_sg(atoms: List[XyzAtom], *, max_residues: int = 4) -> List[float]:
    """Return the angle between (SG→CB) and (CB→CA) for nearest-SG CYS residues.

    For each CYS residue, define:
      - vector v1 = CB - SG
      - vector v2 = CA - CB

    Angle is computed as acos( dot(v1, v2) / (|v1||v2|) ) in degrees.

    Residue selection matches the other histograms: pick the `max_residues` residues
    with smallest SG distance-to-origin, then report their angles.
    """
    by_residue: Dict[Tuple[str, str], Dict[str, XyzAtom]] = {}
    for a in atoms:
        if a.meta.get("RES") != "CYS":
            continue
        atom_name = a.meta.get("ATOM")
        if atom_name not in {"SG", "CB", "CA"}:
            continue
        chain = a.meta.get("CHAIN", "")
        resseq = a.meta.get("RESSEQ")
        if not resseq:
            continue
        k = (chain, resseq)
        by_residue.setdefault(k, {})[atom_name] = a

    candidates: List[Tuple[float, float]] = []
    for atom_map in by_residue.values():
        sg = atom_map.get("SG")
        cb = atom_map.get("CB")
        ca = atom_map.get("CA")
        if sg is None or cb is None or ca is None:
            continue

        v1 = np.asarray((cb.x - sg.x, cb.y - sg.y, cb.z - sg.z), dtype=float)
        v2 = np.asarray((ca.x - cb.x, ca.y - cb.y, ca.z - cb.z), dtype=float)

        v1_norm = float(np.linalg.norm(v1))
        v2_norm = float(np.linalg.norm(v2))
        if v1_norm == 0.0 or v2_norm == 0.0:
            continue

        cosang = float(np.dot(v1, v2) / (v1_norm * v2_norm))
        cosang = float(np.clip(cosang, -1.0, 1.0))
        ang_deg = float(np.degrees(np.arccos(cosang)))

        sg_pos = np.asarray((sg.x, sg.y, sg.z), dtype=float)
        sg_d2 = float(np.dot(sg_pos, sg_pos))
        candidates.append((sg_d2, ang_deg))

    candidates.sort(key=lambda t: t[0])
    return [ang for _, ang in candidates[:max_residues]]


def dihedral_origin_sg_cb_ca_selected_by_sg(
    atoms: List[XyzAtom], *, max_residues: int = 4
) -> List[float]:
    """Return dihedral angles for Zn(origin)→SG→CB→CA for nearest-SG CYS residues.

    The dihedral is defined by the planes (origin, SG, CB) and (SG, CB, CA).
    Angle is reported in degrees.
    """
    by_residue: Dict[Tuple[str, str], Dict[str, XyzAtom]] = {}
    for a in atoms:
        if a.meta.get("RES") != "CYS":
            continue
        atom_name = a.meta.get("ATOM")
        if atom_name not in {"SG", "CB", "CA"}:
            continue
        chain = a.meta.get("CHAIN", "")
        resseq = a.meta.get("RESSEQ")
        if not resseq:
            continue
        k = (chain, resseq)
        by_residue.setdefault(k, {})[atom_name] = a

    candidates: List[Tuple[float, float]] = []
    for atom_map in by_residue.values():
        sg = atom_map.get("SG")
        cb = atom_map.get("CB")
        ca = atom_map.get("CA")
        if sg is None or cb is None or ca is None:
            continue

        p0 = np.asarray((0.0, 0.0, 0.0), dtype=float)
        p1 = np.asarray((sg.x, sg.y, sg.z), dtype=float)
        p2 = np.asarray((cb.x, cb.y, cb.z), dtype=float)
        p3 = np.asarray((ca.x, ca.y, ca.z), dtype=float)

        b0 = p1 - p0
        b1 = p2 - p1
        b2 = p3 - p2

        b1_norm = float(np.linalg.norm(b1))
        if b1_norm == 0.0:
            continue

        n1 = np.cross(b0, b1)
        n2 = np.cross(b1, b2)
        n1_norm = float(np.linalg.norm(n1))
        n2_norm = float(np.linalg.norm(n2))
        if n1_norm == 0.0 or n2_norm == 0.0:
            continue

        b1u = b1 / b1_norm
        m1 = np.cross(n1, b1u)

        x = float(np.dot(n1, n2))
        y = float(np.dot(m1, n2))
        ang = float(np.degrees(np.arctan2(y, x)))

        sg_d2 = float(np.dot(p1, p1))
        candidates.append((sg_d2, ang))

    candidates.sort(key=lambda t: t[0])
    return [ang for _, ang in candidates[:max_residues]]


def plot_bond_length_histogram(
    distances: List[float],
    sources: List[str],
    title: str,
    xlabel: str,
    out_png_name: str,
    color: str = "blue",
    bins: int | str = "auto",
    unit: str = "Å",
    reference_values: List[float] | None = None,
    reference_label: str | None = None,
    reference_series: List[Tuple[List[float], str, str]] | None = None,
    show_plot: bool = True,
) -> None:
    """Plot a histogram of bond lengths with hover/click bin inspection.

    - Hover a bar to see all filenames that contributed at least one distance to that bin.
    - Click a bar to copy the filenames for that bin to your clipboard.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(
            "matplotlib is required for the histogram plot. Install it with: pip install matplotlib\n"
            f"Import error: {e}"
        )
        return

    _apply_hist_rcparams(plt)

    if not distances:
        print("No bond distances collected; skipping histogram plot.")
        return

    if len(sources) != len(distances):
        raise SystemExit(f"Internal error: sources({len(sources)}) != distances({len(distances)})")

    def _copy_to_clipboard(text: str) -> bool:
        # macOS first.
        if sys.platform == "darwin":
            try:
                subprocess.run(["pbcopy"], input=text, text=True, check=True)
                return True
            except Exception:
                return False
        # Fallback: optional pure-Python helper if installed.
        try:
            import pyperclip  # type: ignore

            pyperclip.copy(text)
            return True
        except Exception:
            return False

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    # Ensure gridlines are rendered behind artists like bars.
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    counts, edges, patches = ax.hist(distances, bins=bins, edgecolor="black", color=color, alpha=1.0, zorder=5)

    if reference_series is not None:
        for ref_values, label, ref_color in reference_series:
            if not ref_values:
                continue
            first = True
            for val in ref_values:
                ax.axvline(
                    val,
                    linestyle="-",
                    linewidth=1.4,
                    color=ref_color,
                    alpha=0.9,
                    zorder=6,
                    label=label if first else None,
                )
                first = False
    elif reference_values:
        label = reference_label or "reference"
        first = True
        for val in reference_values:
            ax.axvline(
                val,
                linestyle="-",
                linewidth=1.4,
                color="#000000",
                alpha=0.9,
                zorder=6,
                label=label if first else None,
            )
            first = False

    # Build bin -> set(files) mapping.
    nbins = max(0, len(edges) - 1)
    files_by_bin: List[set[str]] = [set() for _ in range(nbins)]
    for d, src in zip(distances, sources):
        if nbins == 0:
            continue
        idx = bisect_right(edges, d) - 1
        if idx < 0:
            continue
        if idx >= nbins:
            idx = nbins - 1  # include right edge in the last bin
        files_by_bin[idx].add(src)

    # Hover/click UI: show filenames in a fixed info box; click copies full list.
    info_box = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.95),
    )
    current_bin: Dict[str, int | None] = {"idx": None}

    def _format_bin_info(bin_idx: int) -> str:
        lo = float(edges[bin_idx])
        hi = float(edges[bin_idx + 1])
        file_list = sorted(files_by_bin[bin_idx])
        unit_str = f" {unit}" if unit else ""
        header = f"Bin {bin_idx + 1}/{nbins}: [{lo:.3f}, {hi:.3f}){unit_str}\n"
        header += f"Count={int(counts[bin_idx])}  Files={len(file_list)}\n"
        header += "(click bar to copy filenames)\n\n"
        return header + "\n".join(file_list)

    def _clear_info() -> None:
        if info_box.get_text():
            info_box.set_text("")
            fig.canvas.draw_idle()
        current_bin["idx"] = None

    def _on_move(event):
        if event.inaxes != ax or nbins == 0:
            _clear_info()
            return

        hit_idx: int | None = None
        for i, patch in enumerate(patches):
            contains, _ = patch.contains(event)
            if contains:
                hit_idx = i
                break

        if hit_idx is None:
            _clear_info()
            return

        if current_bin["idx"] != hit_idx:
            current_bin["idx"] = hit_idx
            info_box.set_text(_format_bin_info(hit_idx))
            fig.canvas.draw_idle()

    def _on_click(event):
        if event.inaxes != ax or nbins == 0:
            return
        hit_idx: int | None = None
        for i, patch in enumerate(patches):
            contains, _ = patch.contains(event)
            if contains:
                hit_idx = i
                break
        if hit_idx is None:
            return

        file_list = sorted(files_by_bin[hit_idx])
        text = "\n".join(file_list)
        if _copy_to_clipboard(text):
            print(f"Copied {len(file_list)} filenames to clipboard for bin {hit_idx + 1}/{nbins}.")
        else:
            print("Clipboard copy failed; printing filenames instead:\n" + text)

    fig.canvas.mpl_connect("motion_notify_event", _on_move)
    fig.canvas.mpl_connect("button_press_event", _on_click)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    if unit == "°":
        from matplotlib.ticker import FormatStrFormatter

        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    if reference_values or reference_series:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
            )

    plt.tight_layout(rect=(0.0, 0.0, 0.975, 1.0))

    out_path = Path(out_png_name)
    if out_path.suffix.lower() != ".png":
        out_path = out_path.with_suffix(out_path.suffix + ".png") if out_path.suffix else out_path.with_suffix(".png")
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(out_path, dpi=200)
        print(f"Saved histogram PNG: {out_path}")
    except Exception as e:
        print(f"Failed to save PNG '{out_path}': {e}")

    if show_plot:
        plt.show()


def plot_overlay_histogram(
    series: List[List[float]],
    labels: List[str],
    colors: List[str],
    *,
    title: str,
    xlabel: str,
    out_png_name: str,
    bins: int | str = "auto",
    unit: str = "Å",
    x_range: tuple[float, float] | None = None,
    reference_series: List[List[float]] | None = None,
    reference_sets: List[Tuple[str, List[List[float]]]] | None = None,
    reference_line_styles: List[str] | None = None,
    reference_series_line_styles: List[str] | None = None,
    reference_color_cycle: List[str] | None = None,
    bar_alpha: float = 0.65,
    include_series_mean: bool = True,
    include_reference_mean: bool = True,
    reference_label_suffix: str = "",
    show_plot: bool = True,
) -> None:
    """Plot multiple distributions on a single histogram (no hover UI)."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(
            "matplotlib is required for the histogram plot. Install it with: pip install matplotlib\n"
            f"Import error: {e}"
        )
        return

    _apply_hist_rcparams(plt)

    if not any(series):
        print("No values collected; skipping overlay histogram plot.")
        return

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if not isinstance(bins, int):
        raise SystemExit("overlay histograms require an integer 'bins' count")

    if x_range is not None:
        xmin, xmax = x_range
        if xmin >= xmax:
            raise SystemExit("overlay histogram x_range must satisfy xmin < xmax")

    all_values = [v for s in series for v in s]
    if unit == "count":
        if x_range is not None:
            bin_edges = np.arange(xmin, xmax + 1.0, 1.0)
        else:
            max_val = max(all_values) if all_values else 0
            bin_edges = np.arange(-0.5, max_val + 1.5, 1.0)
    else:
        bin_edges = np.histogram_bin_edges(all_values, bins=bins)
    def _format_mean_label(label: str, values: List[float]) -> str:
        if not values:
            return label
        if unit not in ("°", "Å"):
            return label
        mean = float(np.asarray(values, dtype=float).mean())
        if unit == "°":
            mean_str = f"{mean:.1f}"
        else:
            mean_str = f"{mean:.2f}"
        unit_suffix = unit if unit in ("°", "%") else f" {unit}" if unit else ""
        return f"{label} (mean={mean_str}{unit_suffix})"

    labeled_series = [
        (data, _format_mean_label(label, data) if include_series_mean else label, color)
        for data, label, color in zip(series, labels, colors)
        if data
    ]
    items = list(labeled_series)
    # Draw non-yellow first, then yellow on top to avoid washout.
    items.sort(key=lambda item: item[2] == PALETTE[1])
    for data, label, color in items:
        zorder = 6 if color == PALETTE[1] else 5
        _, _, patches = ax.hist(
            data,
            bins=bin_edges,
            label=label,
            color=color,
            alpha=bar_alpha,
            edgecolor="black",
            zorder=zorder,
        )
        if color == PALETTE[0]:
            for patch in patches:
                patch.set_hatch("//")
                patch.set_edgecolor(PALETTE[4])
        if color == PALETTE[1]:
            for patch in patches:
                patch.set_edgecolor(PALETTE[4])

    ref_sets: List[Tuple[str, List[List[float]]]] = []
    if reference_sets is not None:
        ref_sets = reference_sets
    elif reference_series is not None:
        ref_sets = [("", reference_series)]

    if ref_sets:
        for set_idx, (ref_name, ref_series) in enumerate(ref_sets):
            set_color = None
            if reference_color_cycle:
                set_color = reference_color_cycle[set_idx % len(reference_color_cycle)]
            for series_idx, (ref_values, label, color) in enumerate(zip(ref_series, labels, colors)):
                if not ref_values:
                    continue
                ref_label = _format_mean_label(label, ref_values) if include_reference_mean else label
                if ref_name:
                    display_label = f"{ref_name} {ref_label}{reference_label_suffix}"
                else:
                    display_label = f"{ref_label}{reference_label_suffix}"
                if reference_series_line_styles and series_idx < len(reference_series_line_styles):
                    linestyle = reference_series_line_styles[series_idx]
                elif reference_line_styles and set_idx < len(reference_line_styles):
                    linestyle = reference_line_styles[set_idx]
                else:
                    linestyle = "-"
                line_color = set_color if set_color is not None else color
                for i, v in enumerate(ref_values):
                    ax.axvline(
                        v,
                        color=line_color,
                        linestyle=linestyle,
                        linewidth=1.5,
                        alpha=0.9,
                        zorder=6,
                        label=display_label if i == 0 else None,
                    )

    if x_range is not None:
        ax.set_xlim(xmin, xmax)

    # Label every 3rd bin to keep tick/bin spacing aligned.
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    if x_range is not None:
        bin_centers = bin_centers[(bin_centers >= xmin) & (bin_centers <= xmax)]
    if unit == "count":
        if x_range is not None:
            tick_centers = np.arange(int(np.ceil(xmin + 0.5)), int(np.floor(xmax - 0.5)) + 1)
        else:
            tick_centers = np.arange(int(min(all_values or [0])), int(max(all_values or [0])) + 1)
    else:
        tick_centers = bin_centers[::3]
    ax.set_xticks(tick_centers)
    if unit == "Å":
        from matplotlib.ticker import FormatStrFormatter

        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    elif unit == "°":
        from matplotlib.ticker import FormatStrFormatter

        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    ax.set_xlabel(xlabel)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )
    plt.tight_layout(rect=(0.0, 0.0, 0.975, 1.0))

    out_path = Path(out_png_name)
    if out_path.suffix.lower() != ".png":
        out_path = out_path.with_suffix(out_path.suffix + ".png") if out_path.suffix else out_path.with_suffix(".png")
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(out_path, dpi=200)
        print(f"Saved histogram PNG: {out_path}")
    except Exception as e:
        print(f"Failed to save PNG '{out_path}': {e}")

    if show_plot:
        plt.show()


def plot_coord_atom_count_histogram(
    nd1_counts: List[int],
    ne2_counts: List[int],
    *,
    title: str,
    xlabel: str,
    out_png_name: str,
    max_atoms: int = 4,
    show_plot: bool = True,
) -> None:
    """Plot side-by-side count histogram for ND1 vs NE2 coordination counts per file."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(
            "matplotlib is required for the histogram plot. Install it with: pip install matplotlib\n"
            f"Import error: {e}"
        )
        return

    _apply_hist_rcparams(plt)

    if not nd1_counts and not ne2_counts:
        print("No coordination counts collected; skipping count histogram plot.")
        return

    max_bins = max_atoms + 1
    nd1_hist = np.bincount(np.asarray(nd1_counts, dtype=int), minlength=max_bins)
    ne2_hist = np.bincount(np.asarray(ne2_counts, dtype=int), minlength=max_bins)

    x = np.arange(max_bins)
    width = 0.38

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.bar(x - width / 2, nd1_hist, width, label="ND1", color=PALETTE[0], edgecolor="black", zorder=5)
    ax.bar(x + width / 2, ne2_hist, width, label="NE2", color=PALETTE[1], edgecolor="black", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count (files)")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )
    plt.tight_layout(rect=(0.0, 0.0, 0.975, 1.0))

    out_path = Path(out_png_name)
    if out_path.suffix.lower() != ".png":
        out_path = out_path.with_suffix(out_path.suffix + ".png") if out_path.suffix else out_path.with_suffix(".png")
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(out_path, dpi=200)
        print(f"Saved histogram PNG: {out_path}")
    except Exception as e:
        print(f"Failed to save PNG '{out_path}': {e}")

    if show_plot:
        plt.show()


def plot_ca_volume_vs_coord_distance_scatter(
    volumes: List[float],
    distances: List[float],
    mean_volumes: List[float],
    mean_distances: List[float],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_png_name: str,
    show_plot: bool = True,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(
            "matplotlib is required for the scatter plot. Install it with: pip install matplotlib\n"
            f"Import error: {e}"
        )
        return

    _apply_hist_rcparams(plt)

    if not volumes or not distances:
        print("No volume/coordination data collected; skipping scatter plot.")
        return

    if len(volumes) != len(distances):
        raise SystemExit("Internal error: volume points and distance points are mismatched")
    if len(mean_volumes) != len(mean_distances):
        raise SystemExit("Internal error: mean volume points and distance points are mismatched")

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.scatter(
        volumes,
        distances,
        s=22,
        c="#B0B0B0",
        alpha=0.5,
        edgecolors="none",
        zorder=4,
    )
    ax.scatter(
        mean_volumes,
        mean_distances,
        s=40,
        c="#000000",
        marker=".",
        zorder=6,
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    out_path = Path(out_png_name)
    if out_path.suffix.lower() != ".png":
        out_path = out_path.with_suffix(out_path.suffix + ".png") if out_path.suffix else out_path.with_suffix(".png")
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(out_path, dpi=200)
        print(f"Saved scatter PNG: {out_path}")
    except Exception as e:
        print(f"Failed to save PNG '{out_path}': {e}")

    if show_plot:
        plt.show()


def plot_ca_volume_vs_coord_distance_scatter_by_sequence(
    volumes: List[float],
    distances: List[float],
    mean_volumes: List[float],
    mean_distances: List[float],
    mean_sequences: List[str],
    mean_sequences_plain: List[str],
    sequence_counts: Counter[str],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_png_name: str,
    show_plot: bool = True,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(
            "matplotlib is required for the scatter plot. Install it with: pip install matplotlib\n"
            f"Import error: {e}"
        )
        return

    _apply_hist_rcparams(plt)

    if not volumes or not distances:
        print("No volume/coordination data collected; skipping scatter plot.")
        return

    if len(volumes) != len(distances):
        raise SystemExit("Internal error: volume points and distance points are mismatched")
    if len(mean_volumes) != len(mean_distances):
        raise SystemExit("Internal error: mean volume points and distance points are mismatched")
    if len(mean_volumes) != len(mean_sequences):
        raise SystemExit("Internal error: mean volume points and sequences are mismatched")
    if len(mean_volumes) != len(mean_sequences_plain):
        raise SystemExit("Internal error: mean volume points and plain sequences are mismatched")

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.scatter(
        volumes,
        distances,
        s=22,
        c="#B0B0B0",
        alpha=0.5,
        edgecolors="none",
        zorder=4,
    )

    if mean_volumes and mean_distances:
        unique_sequences = sorted(set(mean_sequences))
        plain_by_latex: Dict[str, str] = {}
        for latex_seq, plain_seq in zip(mean_sequences, mean_sequences_plain):
            plain_by_latex.setdefault(latex_seq, plain_seq)
        cmap = plt.get_cmap("tab20")
        color_by_sequence = {
            seq: cmap(idx % cmap.N) for idx, seq in enumerate(unique_sequences)
        }
        for vol_mean, dist_mean, seq in zip(mean_volumes, mean_distances, mean_sequences):
            ax.scatter(
                [vol_mean],
                [dist_mean],
                s=32,
                c=[color_by_sequence[seq]],
                marker="o",
                edgecolors="none",
                alpha=0.8,
                zorder=6,
            )
        if unique_sequences:
            handles = [
                ax.scatter(
                    [],
                    [],
                    s=32,
                    c=[color_by_sequence[seq]],
                    marker="o",
                    edgecolors="none",
                    alpha=0.8,
                    label=f"{seq} - {sequence_counts.get(plain_by_latex.get(seq, ""), 0)}",
                )
                for seq in unique_sequences
            ]
            ax.legend(
                handles=handles,
                title="Sequence",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=True,
                framealpha=0.9,
                title_fontsize=16,
                fontsize=11,
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    out_path = Path(out_png_name)
    if out_path.suffix.lower() != ".png":
        out_path = out_path.with_suffix(out_path.suffix + ".png") if out_path.suffix else out_path.with_suffix(".png")
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(out_path, dpi=200)
        print(f"Saved scatter PNG: {out_path}")
    except Exception as e:
        print(f"Failed to save PNG '{out_path}': {e}")

    if show_plot:
        plt.show()


def find_xyz_files(directory: Path) -> List[Path]:
    # Skip extended-environment XYZ files written by scripts/build_extended_xyz.py;
    # they share the .xyz extension but are derived data, not source structures.
    return sorted(
        p for p in directory.glob("*.xyz")
        if p.is_file() and not p.stem.endswith("-extended")
    )


def _has_glob(pattern: str) -> bool:
    return any(ch in pattern for ch in ("*", "?", "["))


def resolve_xyz_files(file_arg: str | None, directory: Path) -> List[Path]:
    """Resolve --file into one or more .xyz files.

    - If file_arg is None: returns [first .xyz in directory]
    - If file_arg contains glob wildcards (e.g. "1a71*"): matches within directory unless a parent is provided
    - If file_arg is a concrete existing file path: returns [that file]
    - If file_arg doesn't exist and has no wildcards: treated as a glob pattern within directory
    """
    if file_arg is None:
        xyz_files = find_xyz_files(directory)
        if not xyz_files:
            raise SystemExit(f"No .xyz files found in: {directory}")
        return [xyz_files[0]]

    raw = file_arg
    p = Path(raw)

    # Explicit wildcard pattern.
    if _has_glob(raw):
        search_dir = p.parent if str(p.parent) not in (".", "") else directory
        pattern = p.name
        matches = sorted([m for m in search_dir.glob(pattern) if m.is_file() and m.suffix.lower() == ".xyz"])
        if not matches:
            raise SystemExit(f"No .xyz files matched pattern '{raw}' in {search_dir}")
        return matches

    # Concrete file path.
    if p.exists() and p.is_file():
        if p.suffix.lower() != ".xyz":
            raise SystemExit(f"Not an .xyz file: {p}")
        return [p]

    # Shorthand pattern without wildcards: treat as glob in directory.
    search_dir = directory
    pattern = raw
    matches = sorted([m for m in search_dir.glob(pattern) if m.is_file() and m.suffix.lower() == ".xyz"])
    if matches:
        return matches

    raise SystemExit(f"File not found and no matches for '{raw}' in {search_dir}")


def resolve_reference_xyz(directory: Path) -> Path:
    """Resolve ../../Reference_*.xyz relative to the provided directory."""
    search_dir = (directory / ".." / "..").resolve()
    matches = sorted([p for p in search_dir.glob("Reference_*.xyz") if p.is_file()])
    if not matches:
        raise SystemExit(f"No Reference_*.xyz found in {search_dir}")
    if len(matches) > 1:
        print(f"Multiple Reference_*.xyz files found in {search_dir}; using {matches[0].name}")
    return matches[0]


def _infer_reference_atoms_and_labels(
    atoms: List[XyzAtom],
) -> Tuple[List[XyzAtom], Dict[int, str]]:
    """Infer coordinating CYS/HIS atoms and labels from element-only XYZ.

    Returns inferred atoms (with metadata, Zn-centered) and a mapping of
    original atom indices -> label strings for hover display.
    """
    labels_by_idx: Dict[int, str] = {}
    inferred_atoms: List[XyzAtom] = []

    zn = next((a for a in atoms if a.element.strip().upper() == "ZN"), None)
    if zn is None:
        print("Warning: no Zn atom found in reference XYZ; reference overlays disabled.")
        return inferred_atoms, labels_by_idx

    def _el(a: XyzAtom) -> str:
        return a.element.strip().upper()

    def _dist(a: XyzAtom, b: XyzAtom) -> float:
        dx = a.x - b.x
        dy = a.y - b.y
        dz = a.z - b.z
        return float(np.sqrt(dx * dx + dy * dy + dz * dz))

    covalent_r = {
        "H": 0.31,
        "C": 0.76,
        "N": 0.71,
        "O": 0.66,
        "S": 1.05,
        "ZN": 1.22,
    }

    # Build bond graph among heavy atoms (exclude H, ignore Zn bonds).
    heavy_indices = [i for i, a in enumerate(atoms) if _el(a) not in {"H"}]
    adj: Dict[int, List[int]] = {i: [] for i in heavy_indices}
    for i, idx_i in enumerate(heavy_indices):
        ai = atoms[idx_i]
        ei = _el(ai)
        if ei == "ZN":
            continue
        ri = covalent_r.get(ei)
        if ri is None:
            continue
        for idx_j in heavy_indices[i + 1 :]:
            aj = atoms[idx_j]
            ej = _el(aj)
            if ej == "ZN":
                continue
            rj = covalent_r.get(ej)
            if rj is None:
                continue
            d = _dist(ai, aj)
            if d <= (ri + rj + 0.45) and d >= 0.6:
                adj[idx_i].append(idx_j)
                adj[idx_j].append(idx_i)

    def _neighbors(idx: int, element: str | None = None) -> List[int]:
        if idx not in adj:
            return []
        if element is None:
            return adj[idx]
        return [j for j in adj[idx] if _el(atoms[j]) == element]

    def _n_neighbors(idx: int, element: str) -> int:
        return sum(1 for j in _neighbors(idx) if _el(atoms[j]) == element)

    def _dist_to_zn_idx(idx: int) -> float:
        return _dist(atoms[idx], zn)

    # Pick 4 closest potential coordinating atoms to Zn.
    candidates: List[Tuple[float, int]] = []
    for i, a in enumerate(atoms):
        el = _el(a)
        if el not in {"S", "C", "N"}:
            continue
        candidates.append((_dist(a, zn), i))

    if not candidates:
        print("Warning: no S/C/N atoms found in reference XYZ; reference overlays disabled.")
        return inferred_atoms, labels_by_idx

    candidates.sort(key=lambda t: t[0])
    closest = [idx for _, idx in candidates[:4]]

    inferred_atoms: List[XyzAtom] = []
    cys_count = 0
    his_count = 0

    # CYS inference.
    for idx in closest:
        if _el(atoms[idx]) != "S":
            continue
        sg_idx = idx
        c_neighbors = _neighbors(sg_idx, "C")
        if not c_neighbors:
            print("Warning: could not find C neighbor for CYS SG in reference XYZ.")
            continue
        cb_idx = min(c_neighbors, key=lambda j: _dist(atoms[j], atoms[sg_idx]))
        ca_candidates = [j for j in _neighbors(cb_idx, "C") if j != sg_idx]
        if not ca_candidates:
            print("Warning: could not find CA neighbor for CYS CB in reference XYZ.")
            continue
        ca_idx = max(ca_candidates, key=_dist_to_zn_idx)

        cys_count += 1
        resseq = str(cys_count)
        for atom_idx, atom_name in ((sg_idx, "SG"), (cb_idx, "CB"), (ca_idx, "CA")):
            a = atoms[atom_idx]
            labels_by_idx.setdefault(atom_idx, f"CYS {atom_name}")
            inferred_atoms.append(
                XyzAtom(
                    element=a.element,
                    x=a.x - zn.x,
                    y=a.y - zn.y,
                    z=a.z - zn.z,
                    meta={"RES": "CYS", "ATOM": atom_name, "RESSEQ": resseq, "CHAIN": "A"},
                    raw_comment="",
                )
            )

    # HIS inference.
    def _pick_his_stem_from_coord(coord_idx: int) -> Tuple[int, int, int, int] | None:
        """Return (cg_idx, cb_idx, ca_idx, coord_idx) for a coordinating N/C if possible.

        Uses ring order: CA-CB-CG-CD2-NE2-CE1-ND1-CG.
        CG should be bonded to CB (carbon with no N neighbors) and be in the ring.
        """
        # If coord is a carbon, try to find a bonded N and use that as coord.
        coord_el = _el(atoms[coord_idx])
        if coord_el == "C":
            n_neighbors = _neighbors(coord_idx, "N")
            if n_neighbors:
                coord_idx = n_neighbors[0]

        # Search within the imidazole ring neighborhood for CG.
        max_depth = 4
        depth_map: Dict[int, int] = {coord_idx: 0}
        q = deque([coord_idx])
        while q:
            node = q.popleft()
            depth = depth_map[node]
            if depth >= max_depth:
                continue
            for nb in _neighbors(node):
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
            if _n_neighbors(c_idx, "N") == 0:
                continue
            cb_candidates = [
                j
                for j in _neighbors(c_idx, "C")
                if _n_neighbors(j, "N") == 0
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
            for j in _neighbors(cg_idx, "C")
            if _n_neighbors(j, "N") == 0
        ]
        if not cb_candidates:
            return None
        cb_idx = min(cb_candidates, key=lambda j: _dist(atoms[j], atoms[cg_idx]))

        ca_candidates = [j for j in _neighbors(cb_idx, "C") if j != cg_idx]
        if not ca_candidates:
            return None
        ca_idx = max(ca_candidates, key=_dist_to_zn_idx)
        return (cg_idx, cb_idx, ca_idx, coord_idx)

    def _infer_his_ring_from_cg(cg_idx: int, cb_idx: int) -> Dict[str, int]:
        """Infer imidazole ring atom indices using CG as anchor.

        Ring order: CG-CD2-NE2-CE1-ND1-CG. Returns any atoms found.
        """
        ring: Dict[str, int] = {}
        ring_neighbors = [
            j
            for j in _neighbors(cg_idx)
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
                if any(_el(atoms[k]) == "N" and k != cg_idx for k in _neighbors(j)):
                    cd2_idx = j
                    break

        ne2_idx = None
        ce1_idx = None
        if cd2_idx is not None:
            n_neighbors = [k for k in _neighbors(cd2_idx) if _el(atoms[k]) == "N" and k != cg_idx]
            if n_neighbors:
                ne2_idx = n_neighbors[0]
                c_neighbors = [
                    k for k in _neighbors(ne2_idx) if _el(atoms[k]) == "C" and k != cd2_idx
                ]
                if c_neighbors:
                    ce1_idx = c_neighbors[0]
                    n_from_ce1 = [
                        k for k in _neighbors(ce1_idx) if _el(atoms[k]) == "N" and k != ne2_idx
                    ]
                    if n_from_ce1:
                        nd1_idx = n_from_ce1[0]

        if nd1_idx is not None and ne2_idx is None:
            ce1_candidates = [
                k for k in _neighbors(nd1_idx) if _el(atoms[k]) == "C" and k != cg_idx
            ]
            if ce1_candidates:
                ce1_idx = ce1_candidates[0]
                ne2_candidates = [
                    k for k in _neighbors(ce1_idx) if _el(atoms[k]) == "N" and k != nd1_idx
                ]
                if ne2_candidates:
                    ne2_idx = ne2_candidates[0]
                    if cd2_idx is None:
                        cd2_candidates = [
                            k for k in _neighbors(ne2_idx) if _el(atoms[k]) == "C" and k != ce1_idx
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

    for idx in closest:
        if _el(atoms[idx]) not in {"N", "C"}:
            continue
        coord_idx = idx

        stem = _pick_his_stem_from_coord(coord_idx)
        if stem is None:
            coord_el = _el(atoms[coord_idx])
            print("Warning: could not fully infer HIS sidechain for coordinating atom in reference XYZ.")
            # Still include coordinating atom distance if it is N.
            if coord_el == "N":
                his_count += 1
                resseq = str(100 + his_count)
                a = atoms[coord_idx]
                labels_by_idx.setdefault(coord_idx, "HIS ND1")
                inferred_atoms.append(
                    XyzAtom(
                        element=a.element,
                        x=a.x - zn.x,
                        y=a.y - zn.y,
                        z=a.z - zn.z,
                        meta={
                            "RES": "HIS",
                            "ATOM": "ND1",
                            "RESSEQ": resseq,
                            "CHAIN": "A",
                            "COORD": "1",
                        },
                        raw_comment="",
                    )
                )
            continue

        cg_idx, cb_idx, ca_idx, coord_idx = stem
        his_count += 1
        resseq = str(100 + his_count)

        ring = _infer_his_ring_from_cg(cg_idx, cb_idx)
        coord_name = "ND1"
        if coord_idx == ring.get("NE2"):
            coord_name = "NE2"
        elif coord_idx == ring.get("ND1"):
            coord_name = "ND1"

        # Coordinating atom label: ND1 by default (or inferred NE2).
        a_coord = atoms[coord_idx]
        labels_by_idx.setdefault(coord_idx, f"HIS {coord_name}")
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
            labels_by_idx.setdefault(atom_idx, f"HIS {atom_name}")
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
            labels_by_idx.setdefault(atom_idx, f"HIS {atom_name}")
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
        print("Warning: no reference residues could be inferred; reference overlays disabled.")

    return inferred_atoms, labels_by_idx


def _reference_metrics_from_elements(
    atoms: List[XyzAtom],
    metric_specs: List[MetricSpec],
) -> Dict[str, List[float]]:
    """Infer reference metrics from element-only XYZ (no metadata)."""
    metrics: Dict[str, List[float]] = {spec.key: [] for spec in metric_specs}
    inferred_atoms, _ = _infer_reference_atoms_and_labels(atoms)
    if not inferred_atoms:
        return metrics
    return {spec.key: spec.compute(inferred_atoms) for spec in metric_specs}


def plot_reference_xyz_py3dmol(reference_path: Path, out_html: Path) -> None:
    """Write a py3Dmol HTML for reference XYZ with hover labels for inferred atoms."""
    try:
        import py3Dmol
    except Exception as e:  # pragma: no cover
        print(
            "py3Dmol is required for the reference viewer. Install it with: pip install py3dmol\n"
            f"Import error: {e}"
        )
        return

    xyz_text = reference_path.read_text(encoding="utf-8")
    atoms = parse_xyz_atoms(reference_path)
    inferred_atoms, labels_by_idx = _infer_reference_atoms_and_labels(atoms)

    # Build hover labels per atom index.
    labels: List[str] = []
    for i, a in enumerate(atoms):
        label = labels_by_idx.get(i)
        if label is None:
            label = a.element.strip() if a.element else "atom"
        labels.append(label)

    labels_json = json.dumps(labels)

    view = py3Dmol.view(width=900, height=650)
    view.addModel(xyz_text, "xyz")
    view.setStyle(
        {},
        {
            "stick": {"radius": 0.18, "colorscheme": "Jmol"},
            "sphere": {"radius": 0.33, "colorscheme": "Jmol"},
        },
    )
    view.zoomTo()

    hover_on = (
        "function(atom, viewer) {"
        f"var labels = {labels_json};"
        "var idx = (atom.index !== undefined) ? atom.index : (atom.serial !== undefined ? atom.serial - 1 : -1);"
        "var msg = (idx >= 0 && idx < labels.length) ? labels[idx] : '';"
        "if (!msg) { msg = (atom.elem || 'atom'); }"
        "var el = document.getElementById('hoverinfo'); if (el) { el.textContent = msg; }"
        "}"
    )
    hover_off = "function(atom, viewer) { var el = document.getElementById('hoverinfo'); if (el) { el.textContent = ''; } }"
    view.setHoverable({}, True, hover_on, hover_off)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    try:
        view.write_html(str(out_html))
    except Exception:
        if hasattr(view, "_make_html"):
            out_html.write_text(view._make_html(), encoding="utf-8")
        else:
            raise

    html = out_html.read_text(encoding="utf-8")
    header_html = (
        "<div id=\"titlebar\" style=\"font-family: sans-serif; padding: 10px 12px; border-bottom: 1px solid #ddd;\">"
        f"<div style=\"font-size: 16px; font-weight: 600;\">Reference XYZ: {reference_path.name}</div>"
        "<div id=\"hoverinfo\" style=\"margin-top: 6px; font-size: 13px; color: #333; min-height: 1.2em;\"></div>"
        "</div>"
    )
    if "id=\"hoverinfo\"" not in html:
        if "<body" in html and ">" in html:
            body_open_end = html.find(">", html.find("<body"))
            if body_open_end != -1:
                html = html[: body_open_end + 1] + "\n" + header_html + "\n" + html[body_open_end + 1 :]
    out_html.write_text(html, encoding="utf-8")

    print(f"Wrote reference py3Dmol HTML: {out_html}")


def plot_resseq_paths(
    atoms_by_source: List[Tuple[str, List[XyzAtom]]],
    title: str,
    out_html: Path,
) -> None:
    """
    Write an interactive HTML view of origin→SG→CB→CA paths using py3Dmol.
    """
    try:
        import py3Dmol
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "py3Dmol is required for the HTML trajectory viewer. Install it with: pip install py3dmol\n"
            f"Import error: {e}"
        )

    def _v(xyz: Tuple[float, float, float] | np.ndarray) -> np.ndarray:
        return np.asarray(xyz, dtype=float)

    def _dist2(a: Tuple[float, float, float] | np.ndarray, b: Tuple[float, float, float] | np.ndarray) -> float:
        d = _v(a) - _v(b)
        return float(np.dot(d, d))

    def _unit(a: Tuple[float, float, float] | np.ndarray) -> np.ndarray | None:
        v = _v(a)
        n = float(np.linalg.norm(v))
        if n == 0.0:
            return None
        return v / n

    def _basis_from_two_vectors(
        v1: Tuple[float, float, float],
        v2: Tuple[float, float, float],
    ) -> np.ndarray | None:
        # Build an orthonormal right-handed basis where e1 aligns with v1.
        e1 = _unit(v1)
        if e1 is None:
            return None
        # Make e2 by removing e1 component from v2.
        v2v = _v(v2)
        proj = float(np.dot(v2v, e1))
        v2_ortho = v2v - proj * e1
        e2 = _unit(v2_ortho)
        if e2 is None:
            return None
        e3 = np.cross(e1, e2)
        e3u = _unit(e3)
        if e3u is None:
            return None
        # Columns are basis vectors.
        return np.column_stack((e1, e2, e3u)).astype(float)

    @dataclass(frozen=True)
    class _Residue:
        key: Tuple[str, str, str]
        kind: str
        coord_name: str
        atoms: Dict[str, np.ndarray]

        @property
        def coord(self) -> np.ndarray:
            return self.atoms[self.coord_name]

    his_ring_order = ["CG", "CD2", "NE2", "CE1", "ND1", "CG"]
    his_prefer_coord = ["ND1", "NE2"]
    his_secondary_atoms = ["CB", "CG", "CD2", "NE2", "CE1", "ND1", "CA"]

    # Group atom coordinates by residue per source.
    residues_by_source: Dict[str, Dict[Tuple[str, str], Dict[str, Tuple[float, float, float]]]] = {}
    for source, atoms in atoms_by_source:
        for a in atoms:
            atom_name = a.meta.get("ATOM")
            resseq = a.meta.get("RESSEQ")
            if not atom_name or not resseq:
                continue
            chain = a.meta.get("CHAIN", "")
            src_map = residues_by_source.setdefault(source, {})
            k = (chain, resseq)
            if k not in src_map:
                src_map[k] = {}
            src_map[k][atom_name] = (a.x, a.y, a.z)
            res_name = a.meta.get("RES")
            if res_name and "RES" not in src_map[k]:
                src_map[k]["RES"] = res_name

    def _choose_his_coord_atom(atom_map: Dict[str, Tuple[float, float, float]]) -> str | None:
        candidates = [name for name in his_prefer_coord if name in atom_map]
        if not candidates:
            return None
        best = None
        best_d2 = None
        for name in candidates:
            v = _v(atom_map[name])
            d2 = float(np.dot(v, v))
            if best_d2 is None or d2 < best_d2:
                best = name
                best_d2 = d2
        return best

    def _basis_from_residue(res: _Residue) -> np.ndarray | None:
        seed = res.coord
        for name in his_secondary_atoms:
            if name in res.atoms and name != res.coord_name:
                basis = _basis_from_two_vectors(seed, res.atoms[name])
                if basis is not None:
                    return basis
        return None

    def _score_residue_pair(
        ref: _Residue, cur_atoms_rot: Dict[str, np.ndarray]
    ) -> float:
        score = 0.0
        shared = set(ref.atoms.keys()) & set(cur_atoms_rot.keys())
        for name in shared:
            d = ref.atoms[name] - cur_atoms_rot[name]
            score += float(np.dot(d, d))
        return score

    # Build per-source residue lists (CYS/HIS), selecting the 4 coordinating residues
    # by choosing the smallest distance-to-origin of the coordinating atom per file.
    reslist_by_source: Dict[str, List[_Residue]] = {}
    for source, src_map in residues_by_source.items():
        candidates: List[Tuple[float, str, str, _Residue]] = []
        for (chain, resseq), atom_map in src_map.items():
            res_name = atom_map.get("RES")

            # CYS residues: require SG, keep CB/CA if present.
            if "SG" in atom_map and res_name == "CYS":
                atoms: Dict[str, np.ndarray] = {"SG": _v(atom_map["SG"])}
                if "CB" in atom_map:
                    atoms["CB"] = _v(atom_map["CB"])
                if "CA" in atom_map:
                    atoms["CA"] = _v(atom_map["CA"])
                sg_d2 = float(np.dot(atoms["SG"], atoms["SG"]))
                candidates.append(
                    (
                        sg_d2,
                        chain,
                        resseq,
                        _Residue(
                            key=(source, chain, resseq),
                            kind="CYS",
                            coord_name="SG",
                            atoms=atoms,
                        ),
                    )
                )

            # HIS residues: require a coordinating N (ND1/NE2), keep ring/stem atoms if present.
            if res_name == "HIS":
                coord_name = _choose_his_coord_atom(atom_map)
                if coord_name is None:
                    continue
                atoms: Dict[str, np.ndarray] = {coord_name: _v(atom_map[coord_name])}
                for name in ["CA", "CB", "CG", "CD2", "NE2", "CE1", "ND1"]:
                    if name in atom_map and name not in atoms:
                        atoms[name] = _v(atom_map[name])
                coord_d2 = float(np.dot(atoms[coord_name], atoms[coord_name]))
                candidates.append(
                    (
                        coord_d2,
                        chain,
                        resseq,
                        _Residue(
                            key=(source, chain, resseq),
                            kind="HIS",
                            coord_name=coord_name,
                            atoms=atoms,
                        ),
                    )
                )
        candidates.sort(key=lambda t: (t[0], t[1], t[2]))
        reslist_by_source[source] = [r for _, _, _, r in candidates[:4]]

    sources_in_order = [s for s, _ in atoms_by_source]
    first_source = sources_in_order[0] if sources_in_order else ""
    ref_residues = reslist_by_source.get(first_source, [])
    if not ref_residues:
        raise SystemExit("No coordinating residues found in the first file.")

    # Choose reference seed based on residue composition:
    # - If exactly one CYS or one HIS, use that "odd one out" as the seed.
    # - Otherwise prefer CYS if present; else use the nearest HIS.
    cys_ref = [r for r in ref_residues if r.kind == "CYS"]
    his_ref = [r for r in ref_residues if r.kind == "HIS"]
    if len(cys_ref) == 1:
        ref_seed = cys_ref[0]
    elif len(his_ref) == 1:
        ref_seed = his_ref[0]
    else:
        ref_seed = cys_ref[0] if cys_ref else ref_residues[0]
    ref_residues = [ref_seed] + [r for r in ref_residues if r is not ref_seed]

    def _tint(hex_color: str, factor: float) -> str:
        h = hex_color.lstrip("#")
        if len(h) != 6:
            return hex_color
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        r = max(0, min(255, int(r * factor)))
        g = max(0, min(255, int(g * factor)))
        b = max(0, min(255, int(b * factor)))
        return f"#{r:02X}{g:02X}{b:02X}"

    def _blend_white(hex_color: str, amount: int) -> str:
        h = hex_color.lstrip("#")
        if len(h) != 6:
            return hex_color
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        r = max(0, min(255, r + amount))
        g = max(0, min(255, g + amount))
        b = max(0, min(255, b + amount))
        return f"#{r:02X}{g:02X}{b:02X}"

    # Assign each residue to one of 4 colors based on coordinating-atom proximity.
    # Seed the 4 color groups from the first file, then for residues in later files
    # inherit the color of the nearest previously-seen coordinating atom.
    palette = PALETTE
    mixed_2cys_2his = len([r for r in ref_residues if r.kind == "CYS"]) == 2 and len(
        [r for r in ref_residues if r.kind == "HIS"]
    ) == 2

    # Map residue key -> color index, plus an index of already-assigned coords for nearest-neighbor lookup.
    color_by_residue: Dict[Tuple[str, str, str], int] = {}
    assigned_coord: List[Tuple[np.ndarray, int]] = []

    def _assign_color_from_assigned(coord: np.ndarray) -> int:
        if not assigned_coord:
            return 0
        best_color = assigned_coord[0][1]
        best_d2 = _dist2(coord, assigned_coord[0][0])
        for prev_coord, prev_color in assigned_coord[1:]:
            d2 = _dist2(coord, prev_coord)
            if d2 < best_d2:
                best_d2 = d2
                best_color = prev_color
        return best_color

    # First file: fixed reference. Use the first SG-containing residue as the seed (color 0).
    # Remaining residues keep their encounter order as colors 1..3.
    for i, r in enumerate(ref_residues[:4]):
        ci = i
        color_by_residue[r.key] = ci
        assigned_coord.append((r.coord, ci))

    # For each subsequent file, apply a global rotation that best aligns to the reference.
    # Number of rotation candidates is equal to the number of coordinating CYS residues;
    # if no CYS residues, fall back to coordinating HIS residues.
    ref_seed = ref_residues[0]
    ref_basis = _basis_from_residue(ref_seed)
    if ref_basis is None:
        raise SystemExit("Unable to build a stable reference basis from the first residue (seed).")

    ordered_paths: List[Tuple[_Residue, Dict[str, np.ndarray]]] = []
    # Add reference paths first.
    for r in ref_residues[:4]:
        ordered_paths.append((r, r.atoms))

    for src in sources_in_order[1:]:
        cur_residues = reslist_by_source.get(src, [])[:4]
        if len(cur_residues) < 1:
            continue

        # Ensure we have the same count as reference for scoring/matching.
        n = min(len(ref_residues[:4]), len(cur_residues))
        refN = ref_residues[:n]
        curN = cur_residues[:n]

        best_score = None
        best_rot = None
        best_perm = None

        cys_indices = [i for i, r in enumerate(curN) if r.kind == "CYS"]
        his_indices = [i for i, r in enumerate(curN) if r.kind == "HIS"]
        if len(cys_indices) == 1:
            seed_indices = cys_indices
        elif len(his_indices) == 1:
            seed_indices = his_indices
        elif len(cys_indices) == 2 and len(his_indices) == 2:
            seed_indices = cys_indices
        elif len(cys_indices) == 4 or len(his_indices) == 4:
            seed_indices = list(range(n))
        else:
            seed_indices = cys_indices if cys_indices else his_indices

        for seed_idx in seed_indices:
            seed_cur = curN[seed_idx]
            cur_basis = _basis_from_residue(seed_cur)
            if cur_basis is None:
                continue

            # Rotation mapping current basis -> reference basis.
            rot = ref_basis @ cur_basis.T

            # Evaluate the best residue correspondence under this rotation.
            # Constrain ref[0] to map to the chosen seed_idx; remaining residues can permute.
            indices = list(range(n))
            for perm in permutations(indices, n):
                if perm[0] != seed_idx:
                    continue
                score = 0.0
                for ref_i, cur_i in enumerate(perm):
                    rr = refN[ref_i]
                    cr = curN[cur_i]
                    cur_atoms_rot = {name: rot @ pos for name, pos in cr.atoms.items()}
                    score += _score_residue_pair(rr, cur_atoms_rot)
                if best_score is None or score < best_score:
                    best_score = score
                    best_rot = rot
                    best_perm = perm

        # If we couldn't find a stable rotation, fall back to no rotation.
        if best_rot is None or best_perm is None:
            best_rot = np.eye(3, dtype=float)
            best_perm = tuple(range(n))

        # Apply the chosen rotation and assign colors based on nearest previously-seen SG.
        for cur_i in best_perm:
            cr = curN[cur_i]
            cur_atoms_rot = {name: best_rot @ pos for name, pos in cr.atoms.items()}
            coord_r = cur_atoms_rot[cr.coord_name]
            ci = _assign_color_from_assigned(coord_r)
            color_by_residue[cr.key] = ci
            assigned_coord.append((coord_r, ci))
            ordered_paths.append((cr, cur_atoms_rot))

    def _pt(v: np.ndarray) -> Dict[str, float]:
        vv = _v(v)
        return {"x": float(vv[0]), "y": float(vv[1]), "z": float(vv[2])}

    view = py3Dmol.view(width=950, height=720)
    # No molecule model needed; we only draw geometric primitives.

    # Draw origin.
    view.addSphere({"center": {"x": 0.0, "y": 0.0, "z": 0.0}, "radius": 0.16, "color": "black"})

    for res, atoms_rot in ordered_paths:
        ci = color_by_residue.get(res.key, 0)
        color = palette[ci % 4]
        line_opacity = None
        sphere_opacity = None
        if mixed_2cys_2his:
            if res.kind == "CYS":
                color = _tint(color, 0.8)
            else:
                color = _blend_white(color, 70)
                line_opacity = 0.95
                sphere_opacity = 0.95

        coord = atoms_rot.get(res.coord_name)
        if coord is not None:
            line_spec = {"start": {"x": 0.0, "y": 0.0, "z": 0.0}, "end": _pt(coord), "color": color, "linewidth": 10}
            if line_opacity is not None:
                line_spec["opacity"] = line_opacity
            view.addLine(line_spec)
            sphere_spec = {"center": _pt(coord), "radius": 0.14, "color": color}
            if sphere_opacity is not None:
                sphere_spec["opacity"] = sphere_opacity
            view.addSphere(sphere_spec)

        if res.kind == "CYS":
            sg = atoms_rot.get("SG")
            cb = atoms_rot.get("CB")
            ca = atoms_rot.get("CA")
            if sg is not None and cb is not None:
                line_spec = {"start": _pt(sg), "end": _pt(cb), "color": color, "linewidth": 10}
                if line_opacity is not None:
                    line_spec["opacity"] = line_opacity
                view.addLine(line_spec)
                sphere_spec = {"center": _pt(cb), "radius": 0.1, "color": color}
                if sphere_opacity is not None:
                    sphere_spec["opacity"] = sphere_opacity
                view.addSphere(sphere_spec)
            if cb is not None and ca is not None:
                line_spec = {"start": _pt(cb), "end": _pt(ca), "color": color, "linewidth": 10}
                if line_opacity is not None:
                    line_spec["opacity"] = line_opacity
                view.addLine(line_spec)
                sphere_spec = {"center": _pt(ca), "radius": 0.1, "color": color}
                if sphere_opacity is not None:
                    sphere_spec["opacity"] = sphere_opacity
                view.addSphere(sphere_spec)

        if res.kind == "HIS":
            # Stem: CA->CB->CG if present.
            ca = atoms_rot.get("CA")
            cb = atoms_rot.get("CB")
            cg = atoms_rot.get("CG")
            if ca is not None and cb is not None:
                line_spec = {"start": _pt(ca), "end": _pt(cb), "color": color, "linewidth": 10}
                if line_opacity is not None:
                    line_spec["opacity"] = line_opacity
                view.addLine(line_spec)
                sphere_spec = {"center": _pt(ca), "radius": 0.1, "color": color}
                if sphere_opacity is not None:
                    sphere_spec["opacity"] = sphere_opacity
                view.addSphere(sphere_spec)
                sphere_spec = {"center": _pt(cb), "radius": 0.1, "color": color}
                if sphere_opacity is not None:
                    sphere_spec["opacity"] = sphere_opacity
                view.addSphere(sphere_spec)
            if cb is not None and cg is not None:
                line_spec = {"start": _pt(cb), "end": _pt(cg), "color": color, "linewidth": 10}
                if line_opacity is not None:
                    line_spec["opacity"] = line_opacity
                view.addLine(line_spec)
                sphere_spec = {"center": _pt(cg), "radius": 0.1, "color": color}
                if sphere_opacity is not None:
                    sphere_spec["opacity"] = sphere_opacity
                view.addSphere(sphere_spec)

            # Ring: CG-CD2-NE2-CE1-ND1-CG (draw segments when atoms exist).
            ring_points = [atoms_rot.get(name) for name in his_ring_order]
            for p1, p2 in zip(ring_points, ring_points[1:]):
                if p1 is None or p2 is None:
                    continue
                line_spec = {"start": _pt(p1), "end": _pt(p2), "color": color, "linewidth": 10}
                if line_opacity is not None:
                    line_spec["opacity"] = line_opacity
                view.addLine(line_spec)
                sphere_spec = {"center": _pt(p2), "radius": 0.1, "color": color}
                if sphere_opacity is not None:
                    sphere_spec["opacity"] = sphere_opacity
                view.addSphere(sphere_spec)

    view.zoomTo()

    out_html.parent.mkdir(parents=True, exist_ok=True)
    try:
        view.write_html(str(out_html))
    except Exception:
        # Fallback for older py3Dmol versions
        if hasattr(view, "_make_html"):
            out_html.write_text(view._make_html(), encoding="utf-8")
        else:
            raise

    # Patch in a top-of-page title + short instructions.
    html = out_html.read_text(encoding="utf-8")
    header_html = (
        "<div id=\"titlebar\" style=\"font-family: sans-serif; padding: 10px 12px; border-bottom: 1px solid #ddd;\">"
        f"<div style=\"font-size: 16px; font-weight: 600;\">{title}</div>"
        "<div style=\"margin-top: 6px; font-size: 13px; color: #333;\">Rotate/zoom/pan in the browser. Colors track nearest SG groups.</div>"
        "</div>"
    )
    if "id=\"titlebar\"" not in html:
        if "<body" in html and ">" in html:
            body_open_end = html.find(">", html.find("<body"))
            if body_open_end != -1:
                html = html[: body_open_end + 1] + "\n" + header_html + "\n" + html[body_open_end + 1 :]
    out_html.write_text(html, encoding="utf-8")

    print(f"Wrote RESSEQ path viewer HTML: {out_html}")


def plot_all_atoms_py3dmol(xyz_text: str, atoms: List[XyzAtom], title: str, out_html: Path) -> None:
    """Write an interactive py3Dmol view (HTML) with hover labels for atom comments."""
    try:
        import py3Dmol
    except Exception as e:  # pragma: no cover
        print(
            "py3Dmol is required for the molecule viewer. Install it with: pip install py3dmol\n"
            f"Import error: {e}"
        )
        return

    # Map atom index -> original per-line comment string.
    comments = [a.raw_comment for a in atoms]
    comments_json = json.dumps(comments)

    view = py3Dmol.view(width=900, height=650)
    view.addModel(xyz_text, "xyz")

    # Show inferred bonds and element-based coloring.
    view.setStyle(
        {},
        {
            "stick": {"radius": 0.18, "colorscheme": "Jmol"},
            "sphere": {"radius": 0.33, "colorscheme": "Jmol"},
        },
    )
    view.zoomTo()

    # Hover: update a fixed info bar (lighter/faster than creating 3D labels).
    hover_on = (
        "function(atom, viewer) {"
        f"var comments = {comments_json};"
        "var idx = (atom.index !== undefined) ? atom.index : (atom.serial !== undefined ? atom.serial - 1 : -1);"
        "var msg = (idx >= 0 && idx < comments.length) ? comments[idx] : '';"
        "if (!msg) { msg = (atom.elem || 'atom'); }"
        "var el = document.getElementById('hoverinfo'); if (el) { el.textContent = msg; }"
        "}"
    )
    hover_off = "function(atom, viewer) { var el = document.getElementById('hoverinfo'); if (el) { el.textContent = ''; } }"
    view.setHoverable({}, True, hover_on, hover_off)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    try:
        view.write_html(str(out_html))
    except Exception:
        # Fallback for older py3Dmol versions
        if hasattr(view, "_make_html"):
            out_html.write_text(view._make_html(), encoding="utf-8")
        else:
            raise

    # Patch in a top-of-page title + hover info bar.
    html = out_html.read_text(encoding="utf-8")
    header_html = (
        "<div id=\"titlebar\" style=\"font-family: sans-serif; padding: 10px 12px; border-bottom: 1px solid #ddd;\">"
        f"<div style=\"font-size: 16px; font-weight: 600;\">{title}</div>"
        "<div id=\"hoverinfo\" style=\"margin-top: 6px; font-size: 13px; color: #333; min-height: 1.2em;\"></div>"
        "</div>"
    )
    if "id=\"hoverinfo\"" not in html:
        if "<body" in html and ">" in html:
            # Insert immediately after the opening <body ...> tag.
            body_open_end = html.find(">", html.find("<body"))
            if body_open_end != -1:
                html = html[: body_open_end + 1] + "\n" + header_html + "\n" + html[body_open_end + 1 :]
    out_html.write_text(html, encoding="utf-8")

    print(f"Wrote py3Dmol viewer HTML: {out_html}")
    print("Open it in a browser and hover atoms to see metadata in the top bar.")


def _build_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(
            key="sg_center",
            compute=lambda atoms: sg_bond_lengths_to_center(atoms, max_sgs=4),
            expected_per_file=4,
            title="CYS Zn → S bond lengths",
            xlabel="Distance from Zn to Sγ (Å)",
            out_png_name="Figures/cys_zn_sg_distances_histogram.png",
            color=PALETTE[3],
            bins=20,
            unit="Å",
        ),
        MetricSpec(
            key="cb_center",
            compute=lambda atoms: cb_bond_lengths_to_center_selected_by_sg(atoms, max_residues=4),
            expected_per_file=4,
            title="CYS Zn → Cβ distances",
            xlabel="Distance from Zn to Cβ (Å)",
            out_png_name="Figures/cys_zn_cb_distances_histogram.png",
            color=PALETTE[3],
            bins=15,
            unit="Å",
        ),
        MetricSpec(
            key="ca_center",
            compute=lambda atoms: ca_bond_lengths_to_center_selected_by_sg(atoms, max_residues=4),
            expected_per_file=4,
            title="CYS Zn → Cα distances",
            xlabel="Distance from Zn to Cα (Å)",
            out_png_name="Figures/cys_zn_ca_distances_histogram.png",
            color=PALETTE[3],
            bins=15,
            unit="Å",
        ),
        MetricSpec(
            key="sg_ca",
            compute=lambda atoms: sg_ca_distances_selected_by_sg(atoms, max_residues=4),
            expected_per_file=4,
            title="CYS S → Cα distances",
            xlabel="Distance from Sγ to Cα (Å)",
            out_png_name="Figures/cys_sg_ca_distances_histogram.png",
            color=PALETTE[3],
            bins=15,
            unit="Å",
        ),
        MetricSpec(
            key="cb_ca",
            compute=lambda atoms: cb_ca_distances_selected_by_sg(atoms, max_residues=4),
            expected_per_file=4,
            title="CYS Cβ → Cα distances",
            xlabel="Distance from Cβ to Cα (Å)",
            out_png_name="Figures/cys_cb_ca_distances_histogram.png",
            color=PALETTE[3],
            bins=15,
            unit="Å",
        ),
        MetricSpec(
            key="sg_cb",
            compute=lambda atoms: sg_cb_distances_selected_by_sg(atoms, max_residues=4),
            expected_per_file=4,
            title="CYS S → Cβ distances",
            xlabel="Distance from Sγ to Cβ (Å)",
            out_png_name="Figures/cys_sg_cb_distances_histogram.png",
            color=PALETTE[3],
            bins=15,
            unit="Å",
        ),
        MetricSpec(
            key="sgcb_angle",
            compute=lambda atoms: sg_cb_angle_vs_radial_selected_by_sg(atoms, max_residues=4),
            expected_per_file=4,
            title="CYS Zn→S→Cβ angle",
            xlabel="Angle Zn→Sγ→Cβ (°)",
            out_png_name="Figures/cys_zn_sg_cb_angle_histogram.png",
            color=PALETTE[3],
            bins=18,
            unit="°",
        ),
        MetricSpec(
            key="sg_cb_ca_angle",
            compute=lambda atoms: sg_cb_ca_angle_selected_by_sg(atoms, max_residues=4),
            expected_per_file=4,
            title="CYS S→Cβ→Cα angle",
            xlabel="Angle Sγ→Cβ→Cα (°)",
            out_png_name="Figures/cys_sg_cb_ca_angle_histogram.png",
            color=PALETTE[3],
            bins=18,
            unit="°",
        ),
        MetricSpec(
            key="zn_sg_cb_ca_dihedral",
            compute=lambda atoms: dihedral_origin_sg_cb_ca_selected_by_sg(atoms, max_residues=4),
            expected_per_file=4,
            title="CYS Zn→S→Cβ→Cα dihedral",
            xlabel="Dihedral angle (Zn→SG→Cβ→Cα) (°)",
            out_png_name="Figures/cys_zn_sg_cb_ca_dihedral_histogram.png",
            color=PALETTE[3],
            bins=25,
            unit="°",
        ),
        MetricSpec(
            key="cys_sg_center_mean",
            compute=lambda atoms: [cys_mean_coord_distance(atoms, max_sgs=4)]
            if cys_mean_coord_distance(atoms, max_sgs=4) is not None
            else [],
            expected_per_file=1,
            title="CYS Zn → S average distance",
            xlabel="Mean distance from Zn to Sγ (Å)",
            out_png_name="Figures/cys_zn_sg_mean_distance_histogram.png",
            color=PALETTE[3],
            bins=12,
            unit="Å",
        ),
        MetricSpec(
            key="his_coord_center_mean",
            compute=lambda atoms: [his_mean_coord_distance(atoms, max_residues=4)]
            if his_mean_coord_distance(atoms, max_residues=4) is not None
            else [],
            expected_per_file=1,
            title="HIS Zn → Nδ/Nε average distance",
            xlabel="Mean distance from Zn to Nδ/Nε (Å)",
            out_png_name="Figures/his_zn_coord_mean_distance_histogram.png",
            color=PALETTE[2],
            bins=17,
            unit="Å",
        ),
    ]


def _generate_histograms(
    metrics: Dict[str, MetricAccum],
    metric_specs: List[MetricSpec],
    *,
    output_dir: Path,
    system_label: str | None,
    show_plots: bool,
    reference_metrics: Dict[str, List[float]] | None = None,
    reference_label: str | None = None,
    reference_entries: List[Tuple[str, Dict[str, List[float]]]] | None = None,
) -> None:
    for spec in metric_specs:
        acc = metrics[spec.key]
        if not acc.values:
            continue
        ref_vals = reference_metrics.get(spec.key) if reference_metrics else None
        label = reference_label
        reference_series: List[Tuple[List[float], str, str]] | None = None
        if reference_entries:
            ref_colors = _build_reference_gray_cycle(len(reference_entries))
            reference_series = []
            for idx, (ref_label, ref_metrics) in enumerate(reference_entries):
                ref_values = ref_metrics.get(spec.key, [])
                color = ref_colors[idx] if idx < len(ref_colors) else "#000000"
                reference_series.append((ref_values, ref_label, color))
        elif reference_metrics:
            if spec.key in {"cys_sg_center_mean", "his_coord_center_mean"}:
                label = _format_reference_label_mean_only(ref_vals or [], unit=spec.unit)
            else:
                label = _format_reference_label(ref_vals or [], unit=spec.unit)
        plot_bond_length_histogram(
            acc.values,
            acc.sources,
            title=spec.title,
            xlabel=spec.xlabel,
            out_png_name=_output_png_path(spec.out_png_name, output_dir, system_label),
            bins=spec.bins,
            color=spec.color,
            unit=spec.unit,
            reference_values=ref_vals,
            reference_label=label,
            reference_series=reference_series,
            show_plot=show_plots,
        )


def _print_metric_summaries(
    metrics: Dict[str, MetricAccum],
    metric_specs: List[MetricSpec],
    *,
    total_files: int,
) -> None:
    for spec in metric_specs:
        _print_metric_summary(spec, metrics[spec.key], total_files=total_files)


def run_check_mode(
    xyz_dir: Path,
    *,
    open_html: bool,
    output_dir: Path | None = None,
    system_label: str | None = None,
    show_plots: bool = True,
    reference_path: Path | None = None,
    reference_entries: List[Tuple[str, Path]] | None = None,
    generate_outlier_html: bool = True,
) -> None:
    output_dir = output_dir or Path("Figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_xyz = find_xyz_files(xyz_dir)
    if not all_xyz:
        print(f"No .xyz files found in: {xyz_dir}")
        return

    outliers_with_3 = 0
    outliers_with_5 = 0
    max_open = 5
    num_open = 0

    atoms_by_path: Dict[Path, List[XyzAtom]] = {}
    outliers: List[Path] = []

    metric_specs = _build_metric_specs()
    cys_metric_keys = {
        "sg_center",
        "cb_center",
        "ca_center",
        "sg_ca",
        "cb_ca",
        "sg_cb",
        "sgcb_angle",
        "sg_cb_ca_angle",
        "zn_sg_cb_ca_dihedral",
        "cys_sg_center_mean",
    }
    his_metric_keys = {"his_coord_center_mean"}
    metrics: Dict[str, MetricAccum] = {spec.key: MetricAccum(values=[], sources=[]) for spec in metric_specs}

    his_coord_nd1_dists: List[float] = []
    his_coord_ne2_dists: List[float] = []
    his_coord_nd1_angles: List[float] = []
    his_coord_ne2_angles: List[float] = []
    his_ca_nd1_dists: List[float] = []
    his_ca_ne2_dists: List[float] = []
    his_cb_nd1_dists: List[float] = []
    his_cb_ne2_dists: List[float] = []
    his_cg_nd1_dists: List[float] = []
    his_cg_ne2_dists: List[float] = []
    his_cg_cb_nd1_angles: List[float] = []
    his_cg_cb_ne2_angles: List[float] = []
    his_cg_cb_ca_nd1_angles: List[float] = []
    his_cg_cb_ca_ne2_angles: List[float] = []
    his_zn_cg_ca_nd1_dihedrals: List[float] = []
    his_zn_cg_ca_ne2_dihedrals: List[float] = []
    his_coord_nd1_counts: List[int] = []
    his_coord_ne2_counts: List[int] = []
    ca_ca_category_dists: Dict[str, List[float]] = {
        "CYS-CYS": [],
        "CYS-HIS ND": [],
        "CYS-HIS NE": [],
        "HIS ND-HIS ND": [],
        "HIS ND-HIS NE": [],
        "HIS NE-HIS NE": [],
    }
    ca_volume_points: List[float] = []
    ca_coord_distance_points: List[float] = []
    ca_volume_means: List[float] = []
    ca_coord_distance_means: List[float] = []
    ca_sequence_labels: List[str] = []
    ca_sequence_labels_plain: List[str] = []
    sequence_counts: Counter[str] = Counter()

    reference_metrics: Dict[str, List[float]] | None = None
    reference_label: str | None = None
    reference_entries_data: List[Dict[str, object]] = []
    ref_his_coord_nd1_dists: List[float] = []
    ref_his_coord_ne2_dists: List[float] = []
    ref_his_ca_nd1_dists: List[float] = []
    ref_his_ca_ne2_dists: List[float] = []
    ref_his_cb_nd1_dists: List[float] = []
    ref_his_cb_ne2_dists: List[float] = []
    ref_his_cg_nd1_dists: List[float] = []
    ref_his_cg_ne2_dists: List[float] = []
    ref_his_coord_nd1_angles: List[float] = []
    ref_his_coord_ne2_angles: List[float] = []
    ref_his_cg_cb_nd1_angles: List[float] = []
    ref_his_cg_cb_ne2_angles: List[float] = []
    ref_his_cg_cb_ca_nd1_angles: List[float] = []
    ref_his_cg_cb_ca_ne2_angles: List[float] = []
    ref_his_zn_cg_ca_nd1_dihedrals: List[float] = []
    ref_his_zn_cg_ca_ne2_dihedrals: List[float] = []
    ref_his_coord_nd1_counts: List[int] = []
    ref_his_coord_ne2_counts: List[int] = []

    def _build_reference_sets(*series_keys: str) -> List[Tuple[str, List[List[float]]]]:
        sets: List[Tuple[str, List[List[float]]]] = []
        for entry in reference_entries_data:
            label = str(entry.get("label", "Reference"))
            series = [entry.get(k, []) for k in series_keys]
            if any(series):
                sets.append((label, series))
        return sets

    def _fill_missing_reference_metrics(
        primary: Dict[str, List[float]],
        fallback: Dict[str, List[float]],
    ) -> Dict[str, List[float]]:
        merged = dict(primary)
        for key, values in fallback.items():
            if not merged.get(key):
                merged[key] = values
        return merged

    if reference_entries:
        for label, ref_path in reference_entries:
            print(f"Loading reference XYZ ({label}): {ref_path}")
            ref_atoms = parse_xyz_atoms(ref_path)
            if not ref_atoms:
                print(f"Warning: reference XYZ appears empty or unreadable: {ref_path}")
                continue
            if any(a.meta for a in ref_atoms):
                ref_metrics = {spec.key: spec.compute(ref_atoms) for spec in metric_specs}
                ref_atoms_for_his = ref_atoms
            else:
                print("Reference XYZ has no metadata; using element-based nearest-4-to-Zn inference.")
                ref_metrics = _reference_metrics_from_elements(ref_atoms, metric_specs)
                ref_atoms_for_his, _ = _infer_reference_atoms_and_labels(ref_atoms)

            fallback_metrics = _reference_metrics_from_elements(ref_atoms, metric_specs)
            ref_metrics = _fill_missing_reference_metrics(ref_metrics, fallback_metrics)

            plot_reference_xyz_py3dmol(ref_path, ref_path.with_suffix(".html"))
            print(f"Reference metrics ({label}):")
            for spec in metric_specs:
                _print_values_summary(
                    f"  - {spec.title}",
                    ref_metrics.get(spec.key, []),
                    unit=spec.unit,
                )

            ref_entry: Dict[str, object] = {
                "label": label,
                "metrics": ref_metrics,
                "his_coord_nd1_dists": [],
                "his_coord_ne2_dists": [],
                "his_ca_nd1_dists": [],
                "his_ca_ne2_dists": [],
                "his_cb_nd1_dists": [],
                "his_cb_ne2_dists": [],
                "his_cg_nd1_dists": [],
                "his_cg_ne2_dists": [],
                "his_coord_nd1_angles": [],
                "his_coord_ne2_angles": [],
                "his_cg_cb_nd1_angles": [],
                "his_cg_cb_ne2_angles": [],
                "his_cg_cb_ca_nd1_angles": [],
                "his_cg_cb_ca_ne2_angles": [],
                "his_zn_cg_ca_nd1_dihedrals": [],
                "his_zn_cg_ca_ne2_dihedrals": [],
                "his_coord_nd1_counts": [],
                "his_coord_ne2_counts": [],
            }

            if ref_atoms_for_his:
                coord_by_atom = his_coord_distances_by_atom(ref_atoms_for_his, max_residues=4)
                ref_entry["his_coord_nd1_dists"] = coord_by_atom.get("ND1", [])
                ref_entry["his_coord_ne2_dists"] = coord_by_atom.get("NE2", [])

                ca_by_atom = his_atom_distances_by_coord_atom(ref_atoms_for_his, "CA", max_residues=4)
                ref_entry["his_ca_nd1_dists"] = ca_by_atom.get("ND1", [])
                ref_entry["his_ca_ne2_dists"] = ca_by_atom.get("NE2", [])

                cb_by_atom = his_atom_distances_by_coord_atom(ref_atoms_for_his, "CB", max_residues=4)
                ref_entry["his_cb_nd1_dists"] = cb_by_atom.get("ND1", [])
                ref_entry["his_cb_ne2_dists"] = cb_by_atom.get("NE2", [])

                cg_by_atom = his_atom_distances_by_coord_atom(ref_atoms_for_his, "CG", max_residues=4)
                ref_entry["his_cg_nd1_dists"] = cg_by_atom.get("ND1", [])
                ref_entry["his_cg_ne2_dists"] = cg_by_atom.get("NE2", [])

                angle_by_atom = his_coord_cg_angles_by_atom(ref_atoms_for_his, max_residues=4)
                ref_entry["his_coord_nd1_angles"] = angle_by_atom.get("ND1", [])
                ref_entry["his_coord_ne2_angles"] = angle_by_atom.get("NE2", [])

                cg_cb_angles = his_coord_cg_ca_angles_by_atom(ref_atoms_for_his, max_residues=4)
                ref_entry["his_cg_cb_nd1_angles"] = cg_cb_angles.get("ND1", [])
                ref_entry["his_cg_cb_ne2_angles"] = cg_cb_angles.get("NE2", [])

                cg_cb_ca_angles = his_cg_cb_ca_angles_by_atom(ref_atoms_for_his, max_residues=4)
                ref_entry["his_cg_cb_ca_nd1_angles"] = cg_cb_ca_angles.get("ND1", [])
                ref_entry["his_cg_cb_ca_ne2_angles"] = cg_cb_ca_angles.get("NE2", [])

                dihedrals = his_coord_zn_cg_ca_dihedral_by_atom(ref_atoms_for_his, max_residues=4)
                ref_entry["his_zn_cg_ca_nd1_dihedrals"] = dihedrals.get("ND1", [])
                ref_entry["his_zn_cg_ca_ne2_dihedrals"] = dihedrals.get("NE2", [])

                ref_nd1_count, ref_ne2_count = his_coordination_atom_counts(ref_atoms_for_his, max_atoms=4)
                ref_entry["his_coord_nd1_counts"] = [ref_nd1_count]
                ref_entry["his_coord_ne2_counts"] = [ref_ne2_count]

            reference_entries_data.append(ref_entry)
    elif reference_path is not None:
        print(f"Loading reference XYZ: {reference_path}")
        ref_atoms = parse_xyz_atoms(reference_path)
        if not ref_atoms:
            print(f"Warning: reference XYZ appears empty or unreadable: {reference_path}")
        if any(a.meta for a in ref_atoms):
            reference_metrics = {spec.key: spec.compute(ref_atoms) for spec in metric_specs}
            ref_atoms_for_his = ref_atoms
        else:
            print("Reference XYZ has no metadata; using element-based nearest-4-to-Zn inference.")
            reference_metrics = _reference_metrics_from_elements(ref_atoms, metric_specs)
            ref_atoms_for_his, _ = _infer_reference_atoms_and_labels(ref_atoms)
        fallback_metrics = _reference_metrics_from_elements(ref_atoms, metric_specs)
        reference_metrics = _fill_missing_reference_metrics(reference_metrics, fallback_metrics)
        reference_label = "Reference"
        plot_reference_xyz_py3dmol(reference_path, reference_path.with_suffix(".html"))
        print("Reference metrics:")
        for spec in metric_specs:
            _print_values_summary(
                f"  - {spec.title}",
                reference_metrics.get(spec.key, []),
                unit=spec.unit,
            )

        if ref_atoms_for_his:
            coord_by_atom = his_coord_distances_by_atom(ref_atoms_for_his, max_residues=4)
            ref_his_coord_nd1_dists = coord_by_atom.get("ND1", [])
            ref_his_coord_ne2_dists = coord_by_atom.get("NE2", [])

            ca_by_atom = his_atom_distances_by_coord_atom(ref_atoms_for_his, "CA", max_residues=4)
            ref_his_ca_nd1_dists = ca_by_atom.get("ND1", [])
            ref_his_ca_ne2_dists = ca_by_atom.get("NE2", [])

            cb_by_atom = his_atom_distances_by_coord_atom(ref_atoms_for_his, "CB", max_residues=4)
            ref_his_cb_nd1_dists = cb_by_atom.get("ND1", [])
            ref_his_cb_ne2_dists = cb_by_atom.get("NE2", [])

            cg_by_atom = his_atom_distances_by_coord_atom(ref_atoms_for_his, "CG", max_residues=4)
            ref_his_cg_nd1_dists = cg_by_atom.get("ND1", [])
            ref_his_cg_ne2_dists = cg_by_atom.get("NE2", [])

            angle_by_atom = his_coord_cg_angles_by_atom(ref_atoms_for_his, max_residues=4)
            ref_his_coord_nd1_angles = angle_by_atom.get("ND1", [])
            ref_his_coord_ne2_angles = angle_by_atom.get("NE2", [])

            cg_cb_angles = his_coord_cg_ca_angles_by_atom(ref_atoms_for_his, max_residues=4)
            ref_his_cg_cb_nd1_angles = cg_cb_angles.get("ND1", [])
            ref_his_cg_cb_ne2_angles = cg_cb_angles.get("NE2", [])

            cg_cb_ca_angles = his_cg_cb_ca_angles_by_atom(ref_atoms_for_his, max_residues=4)
            ref_his_cg_cb_ca_nd1_angles = cg_cb_ca_angles.get("ND1", [])
            ref_his_cg_cb_ca_ne2_angles = cg_cb_ca_angles.get("NE2", [])

            dihedrals = his_coord_zn_cg_ca_dihedral_by_atom(ref_atoms_for_his, max_residues=4)
            ref_his_zn_cg_ca_nd1_dihedrals = dihedrals.get("ND1", [])
            ref_his_zn_cg_ca_ne2_dihedrals = dihedrals.get("NE2", [])

            ref_nd1_count, ref_ne2_count = his_coordination_atom_counts(ref_atoms_for_his, max_atoms=4)
            ref_his_coord_nd1_counts = [ref_nd1_count]
            ref_his_coord_ne2_counts = [ref_ne2_count]

    for p in all_xyz:
        atoms = parse_xyz_atoms(p)
        atoms_by_path[p] = atoms

        seq_plain = _sequence_distance_from_atoms(atoms, use_latex=False)
        if seq_plain:
            sequence_counts[seq_plain] += 1

        cys_count = count_cys_sg_residues(atoms)
        his_count = count_his_coord_residues(atoms)

        for spec in metric_specs:
            if spec.key in cys_metric_keys and cys_count == 0 and his_count >= 4:
                continue
            if spec.key in his_metric_keys and his_count == 0:
                continue
            _accumulate_metric(metrics, spec, atoms, source_name=p.name)

        coord_by_atom = his_coord_distances_by_atom(atoms, max_residues=4)
        his_coord_nd1_dists.extend(coord_by_atom.get("ND1", []))
        his_coord_ne2_dists.extend(coord_by_atom.get("NE2", []))

        ca_by_atom = his_atom_distances_by_coord_atom(atoms, "CA", max_residues=4)
        his_ca_nd1_dists.extend(ca_by_atom.get("ND1", []))
        his_ca_ne2_dists.extend(ca_by_atom.get("NE2", []))

        cb_by_atom = his_atom_distances_by_coord_atom(atoms, "CB", max_residues=4)
        his_cb_nd1_dists.extend(cb_by_atom.get("ND1", []))
        his_cb_ne2_dists.extend(cb_by_atom.get("NE2", []))

        cg_by_atom = his_atom_distances_by_coord_atom(atoms, "CG", max_residues=4)
        his_cg_nd1_dists.extend(cg_by_atom.get("ND1", []))
        his_cg_ne2_dists.extend(cg_by_atom.get("NE2", []))

        angle_by_atom = his_coord_cg_angles_by_atom(atoms, max_residues=4)
        his_coord_nd1_angles.extend(angle_by_atom.get("ND1", []))
        his_coord_ne2_angles.extend(angle_by_atom.get("NE2", []))

        cg_cb_angle_by_atom = his_coord_cg_ca_angles_by_atom(atoms, max_residues=4)
        his_cg_cb_nd1_angles.extend(cg_cb_angle_by_atom.get("ND1", []))
        his_cg_cb_ne2_angles.extend(cg_cb_angle_by_atom.get("NE2", []))

        cg_cb_ca_angle_by_atom = his_cg_cb_ca_angles_by_atom(atoms, max_residues=4)
        his_cg_cb_ca_nd1_angles.extend(cg_cb_ca_angle_by_atom.get("ND1", []))
        his_cg_cb_ca_ne2_angles.extend(cg_cb_ca_angle_by_atom.get("NE2", []))

        dihedral_by_atom = his_coord_zn_cg_ca_dihedral_by_atom(atoms, max_residues=4)
        his_zn_cg_ca_nd1_dihedrals.extend(dihedral_by_atom.get("ND1", []))
        his_zn_cg_ca_ne2_dihedrals.extend(dihedral_by_atom.get("NE2", []))

        if his_count > 0:
            nd1_count, ne2_count = his_coordination_atom_counts(atoms, max_atoms=4)
            his_coord_nd1_counts.append(nd1_count)
            his_coord_ne2_counts.append(ne2_count)

        ca_ca_by_cat = ca_ca_distances_by_category(atoms, max_residues=4)
        for key, values in ca_ca_by_cat.items():
            if key in ca_ca_category_dists:
                ca_ca_category_dists[key].extend(values)

        volume, coord_dists = ca_volume_and_coord_distances(atoms, max_residues=4)
        if volume is not None and len(coord_dists) == 4:
            ca_volume_points.extend([volume] * len(coord_dists))
            ca_coord_distance_points.extend(coord_dists)
            ca_volume_means.append(volume)
            mean_coord = float(np.mean(coord_dists))
            ca_coord_distance_means.append(mean_coord)
            seq_text = _sequence_distance_from_atoms(atoms, use_latex=True)
            if seq_text:
                ca_sequence_labels.append(seq_text)
                ca_sequence_labels_plain.append(seq_plain)

        n = count_cys_residues_with_ca_cb_sg(atoms)
        if n != 4:
            outliers.append(p)
            if n == 3:
                outliers_with_3 += 1
            elif n == 5:
                outliers_with_5 += 1

    if not outliers:
        print("No outlier xyz files found (all have exactly 4 CYS(CA+CB+SG) residues).")
        print(f"Outliers: [0/{len(all_xyz)}]")
    else:
        print("Files with CYS(CA+CB+SG) residue count != 4:")
        for p in outliers:
            atoms = atoms_by_path.get(p, [])
            n = count_cys_residues_with_ca_cb_sg(atoms)
            print(f"  - {p.name} (count={n})")
            if generate_outlier_html:
                xyz_text = p.read_text(encoding="utf-8")
                plot_all_atoms_py3dmol(
                    xyz_text,
                    atoms,
                    title=p.name,
                    out_html=p.with_suffix(".html"),
                )

                if open_html and num_open < max_open:
                    out_html = p.with_suffix(".html")
                    webbrowser.open(out_html.resolve().as_uri())
                    num_open += 1

    print(f"Outliers: [{len(outliers)}/{len(all_xyz)}]")
    print(f"Outliers with 3 residues: {outliers_with_3}")
    print(f"Outliers with 5 residues: {outliers_with_5}")
    _print_metric_summaries(metrics, metric_specs, total_files=len(all_xyz))
    reference_metrics_entries: List[Tuple[str, Dict[str, List[float]]]] = []
    if reference_entries_data:
        for entry in reference_entries_data:
            metrics_entry = entry.get("metrics")
            if isinstance(metrics_entry, dict):
                reference_metrics_entries.append(
                    (str(entry.get("label", "Reference")), metrics_entry)
                )
    _generate_histograms(
        metrics,
        metric_specs,
        output_dir=output_dir,
        system_label=system_label,
        show_plots=show_plots,
        reference_metrics=reference_metrics,
        reference_label=reference_label,
        reference_entries=reference_metrics_entries if reference_metrics_entries else None,
    )

    use_multi_refs = bool(reference_entries_data)
    ref_set_count = len(reference_entries_data) if reference_entries_data else (1 if reference_path is not None else 0)
    reference_color_cycle = _build_reference_gray_cycle(ref_set_count) if ref_set_count else None

    if his_coord_nd1_dists or his_coord_ne2_dists:
        _print_values_summary(
            "HIS ND1 coordinating-atom distances",
            his_coord_nd1_dists,
            unit="Å",
        )
        _print_values_summary(
            "HIS NE2 coordinating-atom distances",
            his_coord_ne2_dists,
            unit="Å",
        )
        plot_overlay_histogram(
            [his_coord_ne2_dists, his_coord_nd1_dists],
            ["Nε", "Nδ"],
            [PALETTE[1], PALETTE[0]],
            title="HIS Zn → Nδ/Nε distances",
            xlabel="Distance from Zn to Nδ/Nε (Å)",
            out_png_name=_output_png_path(
                "Figures/his_zn_coord_distances_histogram.png",
                output_dir,
                system_label,
            ),
            bins=18,
            unit="Å",
            bar_alpha=0.5,
            include_series_mean=False,
            include_reference_mean=False,
            reference_series_line_styles=["-", "--"],
            reference_color_cycle=reference_color_cycle,
            reference_sets=
            _build_reference_sets("his_coord_ne2_dists", "his_coord_nd1_dists")
            if use_multi_refs
            else None,
            reference_series=
            [ref_his_coord_ne2_dists, ref_his_coord_nd1_dists]
            if ref_his_coord_ne2_dists or ref_his_coord_nd1_dists
            else None,
            show_plot=show_plots,
        )

        plot_overlay_histogram(
            [his_coord_ne2_dists, his_coord_nd1_dists],
            ["Nε", "Nδ"],
            [PALETTE[1], PALETTE[0]],
            title="HIS Zn → Nδ/Nε distances (zoomed)",
            xlabel="Distance from Zn to Nδ/Nε (Å)",
            out_png_name=_output_png_path(
                "Figures/his_zn_coord_distances_histogram_2.0_2.25.png",
                output_dir,
                system_label,
            ),
            bins=30,
            unit="Å",
            x_range=(2.0, 2.25),
            bar_alpha=0.5,
            include_series_mean=False,
            include_reference_mean=False,
            reference_series_line_styles=["-", "--"],
            reference_color_cycle=reference_color_cycle,
            reference_sets=
            _build_reference_sets("his_coord_ne2_dists", "his_coord_nd1_dists")
            if use_multi_refs
            else None,
            reference_series=
            [ref_his_coord_ne2_dists, ref_his_coord_nd1_dists]
            if ref_his_coord_ne2_dists or ref_his_coord_nd1_dists
            else None,
            show_plot=show_plots,
        )

    if his_ca_nd1_dists or his_ca_ne2_dists:
        plot_overlay_histogram(
            [his_ca_ne2_dists, his_ca_nd1_dists],
            ["Nε", "Nδ"],
            [PALETTE[1], PALETTE[0]],
            title="HIS Zn → Cα distances",
            xlabel="Distance from Zn to Cα (Å)",
            out_png_name=_output_png_path(
                "Figures/his_zn_ca_distances_histogram.png",
                output_dir,
                system_label,
            ),
            bins=15,
            unit="Å",
            bar_alpha=0.5,
            include_series_mean=False,
            include_reference_mean=False,
            reference_series_line_styles=["-", "--"],
            reference_color_cycle=reference_color_cycle,
            reference_sets=
            _build_reference_sets("his_ca_ne2_dists", "his_ca_nd1_dists")
            if use_multi_refs
            else None,
            reference_series=
            [ref_his_ca_ne2_dists, ref_his_ca_nd1_dists]
            if ref_his_ca_ne2_dists or ref_his_ca_nd1_dists
            else None,
            show_plot=show_plots,
        )

    if his_cb_nd1_dists or his_cb_ne2_dists:
        plot_overlay_histogram(
            [his_cb_ne2_dists, his_cb_nd1_dists],
            ["Nε", "Nδ"],
            [PALETTE[1], PALETTE[0]],
            title="HIS Zn → Cβ distances",
            xlabel="Distance from Zn to Cβ (Å)",
            out_png_name=_output_png_path(
                "Figures/his_zn_cb_distances_histogram.png",
                output_dir,
                system_label,
            ),
            bins=15,
            unit="Å",
            bar_alpha=0.5,
            include_series_mean=False,
            include_reference_mean=False,
            reference_series_line_styles=["-", "--"],
            reference_color_cycle=reference_color_cycle,
            reference_sets=
            _build_reference_sets("his_cb_ne2_dists", "his_cb_nd1_dists")
            if use_multi_refs
            else None,
            reference_series=
            [ref_his_cb_ne2_dists, ref_his_cb_nd1_dists]
            if ref_his_cb_ne2_dists or ref_his_cb_nd1_dists
            else None,
            show_plot=show_plots,
        )

    if his_cg_nd1_dists or his_cg_ne2_dists:
        plot_overlay_histogram(
            [his_cg_ne2_dists, his_cg_nd1_dists],
            ["Nε", "Nδ"],
            [PALETTE[1], PALETTE[0]],
            title="HIS Zn → Cγ distances",
            xlabel="Distance from Zn to Cγ (Å)",
            out_png_name=_output_png_path(
                "Figures/his_zn_cg_distances_histogram.png",
                output_dir,
                system_label,
            ),
            bins=15,
            unit="Å",
            bar_alpha=0.5,
            include_series_mean=False,
            include_reference_mean=False,
            reference_series_line_styles=["-", "--"],
            reference_color_cycle=reference_color_cycle,
            reference_sets=
            _build_reference_sets("his_cg_ne2_dists", "his_cg_nd1_dists")
            if use_multi_refs
            else None,
            reference_series=
            [ref_his_cg_ne2_dists, ref_his_cg_nd1_dists]
            if ref_his_cg_ne2_dists or ref_his_cg_nd1_dists
            else None,
            show_plot=show_plots,
        )

    if his_coord_nd1_angles or his_coord_ne2_angles:
        _print_values_summary(
            "HIS ND1 Zn–N–CG angles",
            his_coord_nd1_angles,
            unit="°",
        )
        _print_values_summary(
            "HIS NE2 Zn–N–CG angles",
            his_coord_ne2_angles,
            unit="°",
        )
        plot_overlay_histogram(
            [his_coord_ne2_angles, his_coord_nd1_angles],
            ["Nε", "Nδ"],
            [PALETTE[1], PALETTE[0]],
            title="HIS Zn–N–Cγ angle",
            xlabel="Angle Zn–N–Cγ (°)",
            out_png_name=_output_png_path(
                "Figures/his_zn_coord_cg_angle_histogram.png",
                output_dir,
                system_label,
            ),
            bins=18,
            unit="°",
            bar_alpha=0.5,
            include_series_mean=False,
            include_reference_mean=False,
            reference_series_line_styles=["-", "--"],
            reference_color_cycle=reference_color_cycle,
            reference_sets=
            _build_reference_sets("his_coord_ne2_angles", "his_coord_nd1_angles")
            if use_multi_refs
            else None,
            reference_series=
            [ref_his_coord_ne2_angles, ref_his_coord_nd1_angles]
            if ref_his_coord_ne2_angles or ref_his_coord_nd1_angles
            else None,
            show_plot=show_plots,
        )

    if his_cg_cb_nd1_angles or his_cg_cb_ne2_angles:
        plot_overlay_histogram(
            [his_cg_cb_ne2_angles, his_cg_cb_nd1_angles],
            ["Nε", "Nδ"],
            [PALETTE[1], PALETTE[0]],
            title="HIS N–Cγ–Cβ angle",
            xlabel="Angle NN–Cγ–Cβ (°)",
            out_png_name=_output_png_path(
                "Figures/his_coord_cg_cb_angle_histogram.png",
                output_dir,
                system_label,
            ),
            bins=18,
            unit="°",
            bar_alpha=0.5,
            include_series_mean=False,
            include_reference_mean=False,
            reference_series_line_styles=["-", "--"],
            reference_color_cycle=reference_color_cycle,
            reference_sets=
            _build_reference_sets("his_cg_cb_ne2_angles", "his_cg_cb_nd1_angles")
            if use_multi_refs
            else None,
            reference_series=
            [ref_his_cg_cb_ne2_angles, ref_his_cg_cb_nd1_angles]
            if ref_his_cg_cb_ne2_angles or ref_his_cg_cb_nd1_angles
            else None,
            show_plot=show_plots,
        )

    if his_cg_cb_ca_nd1_angles or his_cg_cb_ca_ne2_angles:
        plot_overlay_histogram(
            [his_cg_cb_ca_ne2_angles, his_cg_cb_ca_nd1_angles],
            ["Nε", "Nδ"],
            [PALETTE[1], PALETTE[0]],
            title="HIS Cγ–Cβ–Cα angle",
            xlabel="Angle Cγ–Cβ–Cα (°)",
            out_png_name=_output_png_path(
                "Figures/his_coord_cg_cb_ca_angle_histogram.png",
                output_dir,
                system_label,
            ),
            bins=18,
            unit="°",
            bar_alpha=0.5,
            include_series_mean=False,
            include_reference_mean=False,
            reference_series_line_styles=["-", "--"],
            reference_color_cycle=reference_color_cycle,
            reference_sets=
            _build_reference_sets("his_cg_cb_ca_ne2_angles", "his_cg_cb_ca_nd1_angles")
            if use_multi_refs
            else None,
            reference_series=
            [ref_his_cg_cb_ca_ne2_angles, ref_his_cg_cb_ca_nd1_angles]
            if ref_his_cg_cb_ca_ne2_angles or ref_his_cg_cb_ca_nd1_angles
            else None,
            show_plot=show_plots,
        )

    if his_zn_cg_ca_nd1_dihedrals or his_zn_cg_ca_ne2_dihedrals:
        plot_overlay_histogram(
            [his_zn_cg_ca_ne2_dihedrals, his_zn_cg_ca_nd1_dihedrals],
            ["Nε", "Nδ"],
            [PALETTE[1], PALETTE[0]],
            title="HIS Zn–N–Cγ–Cα dihedral",
            xlabel="Dihedral angle Zn–N–Cγ–Cα (°)",
            out_png_name=_output_png_path(
                "Figures/his_zn_coord_cg_ca_dihedral_histogram.png",
                output_dir,
                system_label,
            ),
            bins=25,
            unit="°",
            bar_alpha=0.5,
            include_series_mean=False,
            include_reference_mean=False,
            reference_series_line_styles=["-", "--"],
            reference_color_cycle=reference_color_cycle,
            reference_sets=
            _build_reference_sets("his_zn_cg_ca_ne2_dihedrals", "his_zn_cg_ca_nd1_dihedrals")
            if use_multi_refs
            else None,
            reference_series=
            [ref_his_zn_cg_ca_ne2_dihedrals, ref_his_zn_cg_ca_nd1_dihedrals]
            if ref_his_zn_cg_ca_ne2_dihedrals or ref_his_zn_cg_ca_nd1_dihedrals
            else None,
            show_plot=show_plots,
        )

    if his_coord_nd1_counts or his_coord_ne2_counts:
        plot_overlay_histogram(
            [his_coord_ne2_counts, his_coord_nd1_counts],
            ["Nε", "Nδ"],
            [PALETTE[1], PALETTE[0]],
            title="HIS Coordinating atom counts",
            xlabel="Number per structure",
            out_png_name=_output_png_path(
                "Figures/his_coord_atom_counts_histogram.png",
                output_dir,
                system_label,
            ),
            bins=5,
            unit="count",
            x_range=(-0.5, 4.5),
            bar_alpha=0.5,
            include_series_mean=False,
            include_reference_mean=False,
            reference_series_line_styles=["-", "--"],
            reference_color_cycle=reference_color_cycle,
            reference_sets=
            _build_reference_sets("his_coord_ne2_counts", "his_coord_nd1_counts")
            if use_multi_refs
            else None,
            reference_series=
            [ref_his_coord_ne2_counts, ref_his_coord_nd1_counts]
            if ref_his_coord_ne2_counts or ref_his_coord_nd1_counts
            else None,
            show_plot=show_plots,
        )

    if any(ca_ca_category_dists.values()):
        ca_ca_labels = [
            "CYS-CYS",
            "CYS-HIS Nδ",
            "CYS-HIS Nε",
            "HIS Nδ-HIS Nδ",
            "HIS Nδ-HIS Nε",
            "HIS Nε-HIS Nε",
        ]
        ca_ca_series = [
            ca_ca_category_dists["CYS-CYS"],
            ca_ca_category_dists["CYS-HIS ND"],
            ca_ca_category_dists["CYS-HIS NE"],
            ca_ca_category_dists["HIS ND-HIS ND"],
            ca_ca_category_dists["HIS ND-HIS NE"],
            ca_ca_category_dists["HIS NE-HIS NE"],
        ]
        ca_ca_colors = [PALETTE[3], PALETTE[0], PALETTE[1], PALETTE[6], PALETTE[2], PALETTE[4]]
        plot_overlay_histogram(
            ca_ca_series,
            ca_ca_labels,
            ca_ca_colors,
            title="CA-CA pairwise distances by coordination type",
            xlabel="Distance between CA atoms (Å)",
            out_png_name=_output_png_path(
                "Figures/ca_ca_pairwise_distances_histogram.png",
                output_dir,
                system_label,
            ),
            bins=20,
            unit="Å",
            bar_alpha=0.5,
            include_series_mean=False,
            include_reference_mean=False,
            show_plot=show_plots,
        )
    if ca_volume_points and ca_coord_distance_points:
        plot_ca_volume_vs_coord_distance_scatter(
            ca_volume_points,
            ca_coord_distance_points,
            ca_volume_means,
            ca_coord_distance_means,
            title="CA volume vs fist shell distance",
            xlabel="CA tetrahedron volume (Å^3)",
            ylabel="Zn → coordinating atom distance (Å)",
            out_png_name=_output_png_path(
                "Figures/ca_volume_vs_coord_distance_scatter.png",
                output_dir,
                system_label,
            ),
            show_plot=show_plots,
        )
        if ca_volume_means and ca_coord_distance_means and ca_sequence_labels:
            plot_ca_volume_vs_coord_distance_scatter_by_sequence(
                ca_volume_points,
                ca_coord_distance_points,
                ca_volume_means,
                ca_coord_distance_means,
                ca_sequence_labels,
                ca_sequence_labels_plain,
                sequence_counts,
                title="CA volume vs fist shell distance (by sequence)",
                xlabel="CA tetrahedron volume (Å^3)",
                ylabel="Zn → coordinating atom distance (Å)",
                out_png_name=_output_png_path(
                    "Figures/ca_volume_vs_coord_distance_scatter_by_sequence.png",
                    output_dir,
                    system_label,
                ),
                show_plot=show_plots,
            )
    if open_html and generate_outlier_html:
        print(f"Opened {min(num_open, max_open)}/{len(outliers)} HTML files (max {max_open}).")


def run_plot_mode(xyz_dir: Path, *, file_arg: str | None, open_html: bool) -> None:
    selected_paths = resolve_xyz_files(file_arg, xyz_dir)
    if len(selected_paths) == 1:
        print(f"Selected file: {selected_paths[0]}")
    else:
        print(f"Selected {len(selected_paths)} files:")
        for p in selected_paths:
            print(f"  - {p}")

    atoms_by_source: List[Tuple[str, List[XyzAtom]]] = []
    for p in selected_paths:
        atoms_by_source.append((p.name, parse_xyz_atoms(p)))

    # Only show the alignment/trajectory figure when we actually have multiple inputs
    # (e.g. a wildcard pattern). If the user gave one concrete file path, just emit
    # the py3Dmol HTML viewer.
    single_concrete_file = False
    if len(selected_paths) == 1 and file_arg is not None and not _has_glob(file_arg):
        p = Path(file_arg)
        single_concrete_file = p.exists() and p.is_file() and p.suffix.lower() == ".xyz"

    if not single_concrete_file and len(selected_paths) > 1:
        resseq_out_html = Path("Figures/resseq_paths_py3dmol.html")
        plot_resseq_paths(
            atoms_by_source,
            title="origin → SG → CB → CA per RESSEQ",
            out_html=resseq_out_html,
        )
    else:
        resseq_out_html = None

    # Generate one py3Dmol HTML per file.
    for p in selected_paths:
        atoms = parse_xyz_atoms(p)
        xyz_text = p.read_text(encoding="utf-8")
        plot_all_atoms_py3dmol(
            xyz_text,
            atoms,
            title=f"{p.name}",
            out_html=p.with_suffix(".html"),
        )

    if open_html:
        # If we built the multi-file RESSEQ-path viewer, prefer opening that.
        if resseq_out_html is not None:
            webbrowser.open(resseq_out_html.resolve().as_uri())
        else:
            out_html = selected_paths[0].with_suffix(".html")
            webbrowser.open(out_html.resolve().as_uri())


def run_directory_mode(
    xyz_dir: Path,
    *,
    open_html: bool,
    output_dir: Path,
    system_label: str | None,
    show_plots: bool,
    reference_path: Path | None = None,
    reference_entries: List[Tuple[str, Path]] | None = None,
) -> None:
    all_xyz = find_xyz_files(xyz_dir)
    if not all_xyz:
        print(f"No .xyz files found in: {xyz_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    write_sequence_distance_file(
        all_xyz,
        output_dir=output_dir,
        system_label=system_label,
    )
    write_family_struct_examples(
        all_xyz,
        output_dir=output_dir,
    )

    run_check_mode(
        xyz_dir,
        open_html=False,
        output_dir=output_dir,
        system_label=system_label,
        show_plots=show_plots,
        reference_path=reference_path,
        reference_entries=reference_entries,
        generate_outlier_html=False,
    )

    atoms_by_source: List[Tuple[str, List[XyzAtom]]] = []
    for p in all_xyz:
        atoms_by_source.append((p.name, parse_xyz_atoms(p)))

    out_name = "3dmodel.html"
    if system_label:
        out_name = f"3dmodel_{system_label}.html"
    resseq_out_html = output_dir / out_name
    plot_resseq_paths(
        atoms_by_source,
        title="origin → SG → CB → CA per RESSEQ",
        out_html=resseq_out_html,
    )

    if open_html:
        webbrowser.open(resseq_out_html.resolve().as_uri())
