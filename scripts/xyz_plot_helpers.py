#!/usr/bin/env python3
"""Helper utilities for XYZ validation plots and HTML viewers."""

from __future__ import annotations

from bisect import bisect_right
import json
import numpy as np
import subprocess
import sys
import webbrowser
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Callable, Dict, List, Tuple


PALETTE = ["#3A3D42", "#457B9D", "#2A9D8F", "#E63946", 
           "#6D597A", "#F4A261", "#F4978E", "#B7410E",
           "#4A7C59"]


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
    summary_label: str
    title: str
    xlabel: str
    out_png_name: str
    color: str
    bins: int | str = "auto"
    unit: str = "Å"


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
        f"{spec.summary_label}: "
        f"{len(acc.values)} values from {total_files} file(s) "
        f"(min={float(arr.min()):.3f}{unit_suffix}, max={float(arr.max()):.3f}{unit_suffix}, mean={mean:.3f}{unit_suffix})"
    )
    if acc.files_with_lt_expected:
        print(
            f"Note: {acc.files_with_lt_expected} file(s) had <{spec.expected_per_file} contributing values; "
            "their histogram contribution is truncated."
        )


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

    fig, ax = plt.subplots()
    # Ensure gridlines are rendered behind artists like bars.
    ax.set_axisbelow(True)
    ax.grid(axis="both", zorder=0, alpha=0.5)
    counts, edges, patches = ax.hist(distances, bins=bins, edgecolor="black", color=color, alpha=0.8, zorder=5)

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
    plt.tight_layout()

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

    plt.show()


def find_xyz_files(directory: Path) -> List[Path]:
    return sorted([p for p in directory.glob("*.xyz") if p.is_file()])


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
            summary_label="SG-to-center distances collected for histogram",
            title="CYS Histogram: Zn → S bond lengths",
            xlabel="Distance from Zn to CYS SG (Å)",
            out_png_name="Figures/cys_zn_sg_distances_histogram.png",
            color=PALETTE[5],
            bins=20,
            unit="Å",
        ),
        MetricSpec(
            key="cb_center",
            compute=lambda atoms: cb_bond_lengths_to_center_selected_by_sg(atoms, max_residues=4),
            expected_per_file=4,
            summary_label="CB-to-center distances collected for histogram",
            title="CYS Histogram: Zn → Cβ distances",
            xlabel="Distance from Zn to CYS CB (Å)",
            out_png_name="Figures/cys_zn_cb_distances_histogram.png",
            color=PALETTE[1],
            bins=15,
            unit="Å",
        ),
        MetricSpec(
            key="ca_center",
            compute=lambda atoms: ca_bond_lengths_to_center_selected_by_sg(atoms, max_residues=4),
            expected_per_file=4,
            summary_label="CA-to-center distances collected for histogram",
            title="CYS Histogram: Zn → Cα distances",
            xlabel="Distance from Zn to CYS CA (Å)",
            out_png_name="Figures/cys_zn_ca_distances_histogram.png",
            color=PALETTE[2],
            bins=15,
            unit="Å",
        ),
        MetricSpec(
            key="sg_ca",
            compute=lambda atoms: sg_ca_distances_selected_by_sg(atoms, max_residues=4),
            expected_per_file=4,
            summary_label="SG–CA distances collected for histogram",
            title="CYS Histogram: S → Cα distances",
            xlabel="Distance from CYS SG to CYS CA (Å)",
            out_png_name="Figures/cys_sg_ca_distances_histogram.png",
            color=PALETTE[3],
            bins=15,
            unit="Å",
        ),
        MetricSpec(
            key="cb_ca",
            compute=lambda atoms: cb_ca_distances_selected_by_sg(atoms, max_residues=4),
            expected_per_file=4,
            summary_label="CB–CA distances collected for histogram",
            title="CYS Histogram: Cβ → Cα distances",
            xlabel="Distance from CYS CB to CYS CA (Å)",
            out_png_name="Figures/cys_cb_ca_distances_histogram.png",
            color=PALETTE[0],
            bins=15,
            unit="Å",
        ),
        MetricSpec(
            key="sg_cb",
            compute=lambda atoms: sg_cb_distances_selected_by_sg(atoms, max_residues=4),
            expected_per_file=4,
            summary_label="SG–CB distances collected for histogram",
            title="CYS Histogram: S → Cβ distances",
            xlabel="Distance from CYS SG to CYS CB (Å)",
            out_png_name="Figures/cys_sg_cb_distances_histogram.png",
            color=PALETTE[7],
            bins=15,
            unit="Å",
        ),
        MetricSpec(
            key="sgcb_angle",
            compute=lambda atoms: sg_cb_angle_vs_radial_selected_by_sg(atoms, max_residues=4),
            expected_per_file=4,
            summary_label="Angles collected for histogram (angle between origin→SG and SG→CB)",
            title="CYS Histogram: Zn→S→Cβ angle",
            xlabel="Angle between (Zn→SG) and (SG→CB) (°)",
            out_png_name="Figures/cys_zn_sg_cb_angle_histogram.png",
            color=PALETTE[4],
            bins=18,
            unit="°",
        ),
        MetricSpec(
            key="sg_cb_ca_angle",
            compute=lambda atoms: sg_cb_ca_angle_selected_by_sg(atoms, max_residues=4),
            expected_per_file=4,
            summary_label="Angles collected for histogram (angle between SG→CB and CB→CA)",
            title="CYS Histogram: S→Cβ→Cα angle",
            xlabel="Angle between (SG→CB) and (CB→CA) (°)",
            out_png_name="Figures/cys_sg_cb_ca_angle_histogram.png",
            color=PALETTE[8],
            bins=18,
            unit="°",
        ),
        MetricSpec(
            key="zn_sg_cb_ca_dihedral",
            compute=lambda atoms: dihedral_origin_sg_cb_ca_selected_by_sg(atoms, max_residues=4),
            expected_per_file=4,
            summary_label="Dihedral angles collected for histogram (origin→SG→CB→CA)",
            title="CYS Histogram: Zn→S→Cβ→Cα dihedral",
            xlabel="Dihedral angle (Zn→SG→CB→CA) (°)",
            out_png_name="Figures/cys_zn_sg_cb_ca_dihedral_histogram.png",
            color=PALETTE[6],
            bins=25,
            unit="°",
        ),
        MetricSpec(
            key="his_coord_center",
            compute=lambda atoms: his_coord_bond_lengths_to_center(atoms, max_residues=4),
            expected_per_file=4,
            summary_label="HIS coordinating-atom distances collected for histogram",
            title="HIS Histogram: Zn → coordinating atom distances",
            xlabel="Distance from Zn to HIS ND1/NE2 (Å)",
            out_png_name="Figures/his_zn_coord_distances_histogram.png",
            color=PALETTE[8],
            bins=18,
            unit="Å",
        ),
        MetricSpec(
            key="his_ca_center",
            compute=lambda atoms: his_atom_distances_selected_by_coord(atoms, "CA", max_residues=4),
            expected_per_file=4,
            summary_label="HIS CA-to-center distances collected for histogram",
            title="HIS Histogram: Zn → Cα distances",
            xlabel="Distance from Zn to HIS CA (Å)",
            out_png_name="Figures/his_zn_ca_distances_histogram.png",
            color=PALETTE[2],
            bins=15,
            unit="Å",
        ),
        MetricSpec(
            key="his_cb_center",
            compute=lambda atoms: his_atom_distances_selected_by_coord(atoms, "CB", max_residues=4),
            expected_per_file=4,
            summary_label="HIS CB-to-center distances collected for histogram",
            title="HIS Histogram: Zn → Cβ distances",
            xlabel="Distance from Zn to HIS CB (Å)",
            out_png_name="Figures/his_zn_cb_distances_histogram.png",
            color=PALETTE[1],
            bins=15,
            unit="Å",
        ),
        MetricSpec(
            key="his_cg_center",
            compute=lambda atoms: his_atom_distances_selected_by_coord(atoms, "CG", max_residues=4),
            expected_per_file=4,
            summary_label="HIS CG-to-center distances collected for histogram",
            title="HIS Histogram: Zn → Cγ distances",
            xlabel="Distance from Zn to HIS CG (Å)",
            out_png_name="Figures/his_zn_cg_distances_histogram.png",
            color=PALETTE[5],
            bins=15,
            unit="Å",
        ),
    ]


def _generate_histograms(
    metrics: Dict[str, MetricAccum],
    metric_specs: List[MetricSpec],
) -> None:
    for spec in metric_specs:
        acc = metrics[spec.key]
        if not acc.values:
            continue
        plot_bond_length_histogram(
            acc.values,
            acc.sources,
            title=spec.title,
            xlabel=spec.xlabel,
            out_png_name=spec.out_png_name,
            bins=spec.bins,
            color=spec.color,
            unit=spec.unit,
        )


def _print_metric_summaries(
    metrics: Dict[str, MetricAccum],
    metric_specs: List[MetricSpec],
    *,
    total_files: int,
) -> None:
    for spec in metric_specs:
        _print_metric_summary(spec, metrics[spec.key], total_files=total_files)


def run_check_mode(xyz_dir: Path, *, open_html: bool) -> None:
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
    }
    his_metric_keys = {"his_coord_center", "his_ca_center", "his_cb_center", "his_cg_center"}
    metrics: Dict[str, MetricAccum] = {spec.key: MetricAccum(values=[], sources=[]) for spec in metric_specs}

    for p in all_xyz:
        atoms = parse_xyz_atoms(p)
        atoms_by_path[p] = atoms

        cys_count = count_cys_sg_residues(atoms)
        his_count = count_his_coord_residues(atoms)

        for spec in metric_specs:
            if spec.key in cys_metric_keys and cys_count == 0 and his_count >= 4:
                continue
            if spec.key in his_metric_keys and his_count == 0:
                continue
            _accumulate_metric(metrics, spec, atoms, source_name=p.name)

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
    _generate_histograms(metrics, metric_specs)
    if open_html:
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
