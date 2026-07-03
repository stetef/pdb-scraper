#!/usr/bin/env python3
"""Compute per-structure stats from XYZ files and (optionally) PDB files.

For each XYZ file in --xyz-dir computes:

  Geometry (from XYZ coordinates, always):
    volume_A3             Cα tetrahedron volume
    q_tetra_coord         Errington-Debenedetti q using coordinating atoms (SG)
    q_tetra_ca            Same metric using Cα positions
    cys_dihedral_N_deg    Zn→SG→Cβ→Cα dihedral for residue N (N=1..4, sorted by RESSEQ)
    cys_dihedral_mean_deg Mean of the 4 dihedrals

  From XYZ header:
    resolution_A          RESOLUTION_A field in header line

  From PDB (requires --pdb-dir, skipped gracefully otherwise):
    r_work                R VALUE (WORKING SET) from REMARK 3
    r_free                FREE R VALUE from REMARK 3
    zn_bfactor            B-factor of the Zn HETATM record
    coord_cys_N_bfactor_avg  mean(CA, CB, SG) B-factors for coordinating CYS N

Output: --out-csv  (default: <xyz-dir>/structure_stats.csv)

Skips if output already exists unless --force.

Usage
-----
  uv run python scripts/structure-clustering-analysis/00a_compute_structure_stats.py \\
      --xyz-dir  data/test-4cys/initial_xyz_files \\
      --pdb-dir  data/large-cys-his-datasets/4cys-large/results/validated_structures

  uv run python scripts/structure-clustering-analysis/00a_compute_structure_stats.py \\
      --xyz-dir  data/test-4cys/initial_xyz_files \\
      --pdb-dir  data/large-cys-his-datasets/4cys-large/results/validated_structures \\
      --force
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import parse_structure, Structure


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _q_tetra(points: np.ndarray, center: np.ndarray) -> Optional[float]:
    """Errington-Debenedetti tetrahedral order parameter.

    q = 1 − (3/8) · Σ_{i<j} (cos θ_ij + 1/3)²
    where θ_ij is the angle center–pointi–pointj (at center).
    Returns None if any point coincides with center.
    """
    if len(points) != 4:
        return None
    unit_vecs = []
    for p in points:
        v = p - center
        n = float(np.linalg.norm(v))
        if n == 0.0:
            return None
        unit_vecs.append(v / n)
    total = 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            cos_t = float(np.dot(unit_vecs[i], unit_vecs[j]))
            cos_t = max(-1.0, min(1.0, cos_t))
            total += (cos_t + 1.0 / 3.0) ** 2
    return 1.0 - (3.0 / 8.0) * total


def _dihedral_deg(p0: np.ndarray, p1: np.ndarray,
                  p2: np.ndarray, p3: np.ndarray) -> Optional[float]:
    """Dihedral angle p0–p1–p2–p3 in degrees (−180 to +180)."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1n = float(np.linalg.norm(b1))
    n2n = float(np.linalg.norm(b2))
    n3n = float(np.linalg.norm(b3))
    if n1n < 1e-10 or n2n < 1e-10 or n3n < 1e-10:
        return None
    b2u = b2 / n2n
    v = b1 - np.dot(b1, b2u) * b2u
    w = b3 - np.dot(b3, b2u) * b2u
    nv, nw = np.linalg.norm(v), np.linalg.norm(w)
    if nv < 1e-10 or nw < 1e-10:
        return None
    cos_t = float(np.dot(v, w) / (nv * nw))
    cos_t = max(-1.0, min(1.0, cos_t))
    angle = math.degrees(math.acos(cos_t))
    if np.dot(np.cross(v, w), b2u) < 0:
        angle = -angle
    return angle


def compute_geometry(s: Structure) -> dict:
    """Compute all geometry stats from a Structure."""
    zn = s.zn

    # Cα tetrahedron volume
    ca = s.ca  # (4, 3), sorted by RESSEQ via parse_structure
    v1 = ca[1] - ca[0]
    v2 = ca[2] - ca[0]
    v3 = ca[3] - ca[0]
    volume = float(abs(float(np.dot(v1, np.cross(v2, v3)))) / 6.0)

    # q_tetra (coordinating SG atoms, then Cα)
    q_coord = _q_tetra(s.s, zn)
    q_ca    = _q_tetra(s.ca, zn)

    # Zn→SG→Cβ→Cα dihedral per residue
    dihedrals: list[Optional[float]] = []
    for r in range(4):
        d = _dihedral_deg(zn, s.s[r], s.cb[r], s.ca[r])
        dihedrals.append(d)

    valid_d = [d for d in dihedrals if d is not None]
    dihedral_mean = float(np.mean(valid_d)) if valid_d else None

    row: dict = {
        "volume_A3":            f"{volume:.4f}",
        "q_tetra_coord":        f"{q_coord:.4f}" if q_coord is not None else "",
        "q_tetra_ca":           f"{q_ca:.4f}"    if q_ca    is not None else "",
        "cys_dihedral_mean_deg": f"{dihedral_mean:.2f}" if dihedral_mean is not None else "",
    }
    for i, d in enumerate(dihedrals, start=1):
        row[f"cys_dihedral_{i}_deg"] = f"{d:.2f}" if d is not None else ""

    return row


# ---------------------------------------------------------------------------
# XYZ header / metadata parsing
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(r"(\w+)=([^\s]+)")

_SEC_ABBREV: dict[str, str] = {"LOOP": "L", "HELIX": "H", "SHEET": "S"}


def _parse_sec(comment: str) -> str:
    """Return SEC field value (LOOP, HELIX, SHEET) from comment, upper-cased."""
    for part in comment.split():
        if part.startswith("SEC="):
            return part[4:].strip().upper()
    return ""


def _parse_xyz_header(path: Path) -> dict:
    """Extract key=value tokens from the XYZ comment line (line 2)."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    if len(lines) < 2:
        return {}
    return dict(_HEADER_RE.findall(lines[1]))


def _parse_xyz_residues(path: Path) -> dict:
    """Return RESSEQ-sorted list of (chain, resseq) for Zn and coordinating CYS,
    plus SEC annotation for each CYS from its CA atom.

    Returns {"zn": (chain, resseq), "cys": [(chain, resseq), ...],
             "cys_sec": {(chain, resseq): sec_string}}
    Only reads heavy atoms (skips H lines).
    """
    result: dict = {"zn": None, "cys": [], "cys_sec": {}}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return result

    seen_cys: set[tuple[str, int]] = set()
    for line in lines[2:]:
        if "#" not in line:
            continue
        comment = line.split("#", 1)[1]
        meta = dict(_HEADER_RE.findall(comment))
        atom = meta.get("ATOM", "").upper()
        if atom in ("H", ""):
            continue
        res   = meta.get("RES", "").upper()
        chain = meta.get("CHAIN", "")
        try:
            resseq = int(meta.get("RESSEQ", ""))
        except (ValueError, TypeError):
            continue
        if res == "ZN" and atom == "ZN":
            result["zn"] = (chain, resseq)
        elif res == "CYS":
            key = (chain, resseq)
            if key not in seen_cys:
                seen_cys.add(key)
                result["cys"].append(key)
            if atom == "CA":
                result["cys_sec"][key] = _parse_sec(comment)

    result["cys"].sort()  # sort by (chain, resseq) to match parse_structure ordering
    return result


def compute_family_string(
    cys_keys: list[tuple[str, int]],
    cys_sec: dict[tuple[str, int], str],
) -> str:
    """Build family string from sorted (chain, resseq) list + SEC annotation map.

    Single chain example:  Cx2Cx2Cx8C-LHLL
    Two-chain example:     Cx2C-Cx8C-LH-LL
    """
    if len(cys_keys) != 4:
        return ""

    # Group by chain preserving sort order
    chain_groups: dict[str, list[tuple[int, str]]] = {}
    chain_order: list[str] = []
    for chain, resseq in cys_keys:
        if chain not in chain_groups:
            chain_groups[chain] = []
            chain_order.append(chain)
        sec = cys_sec.get((chain, resseq), "")
        chain_groups[chain].append((resseq, sec))

    c_parts: list[str] = []
    s_parts: list[str] = []
    for chain in chain_order:
        residues = chain_groups[chain]
        c_str = "C"
        for i in range(len(residues) - 1):
            spacing = residues[i + 1][0] - residues[i][0] - 1
            c_str += f"x{spacing}C"
        s_str = "".join(_SEC_ABBREV.get(sec, "?") for _, sec in residues)
        c_parts.append(c_str)
        s_parts.append(s_str)

    return "-".join(c_parts) + "-" + "-".join(s_parts)


# ---------------------------------------------------------------------------
# PDB parsing
# ---------------------------------------------------------------------------

_RWORK_RE = re.compile(r"R VALUE\s+\(WORKING SET\)\s*:\s*([\d.]+)")
_RFREE_RE  = re.compile(r"FREE R VALUE\s*:\s*([\d.]+)")


def _parse_pdb_stats(
    pdb_path: Path,
    zn_chain: str,
    zn_resseq: int,
    cys_list: list[tuple[str, int]],
) -> dict:
    """Extract R-factors and B-factors from a PDB file.

    Returns dict with r_work, r_free, zn_bfactor, coord_cys_N_bfactor_avg.
    Empty strings for any field not found.
    """
    result: dict = {
        "r_work": "", "r_free": "", "zn_bfactor": "",
    }
    for i in range(1, 5):
        result[f"coord_cys_{i}_bfactor_avg"] = ""

    try:
        text = pdb_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return result

    # R-factors from REMARK 3 (take the last match to handle duplicates)
    r_work_val = r_free_val = None
    for line in text.splitlines():
        if line.startswith("REMARK   3"):
            m = _RWORK_RE.search(line)
            if m and "FREE" not in line:
                r_work_val = m.group(1)
            m = _RFREE_RE.search(line)
            if m and "SIZE" not in line and "COUNT" not in line and "TEST" not in line:
                r_free_val = m.group(1)

    if r_work_val:
        result["r_work"] = r_work_val
    if r_free_val:
        result["r_free"] = r_free_val

    # Atom-level B-factors
    cys_bf: dict[tuple[str, int], dict[str, float]] = {}
    zn_bf: Optional[float] = None

    for line in text.splitlines():
        rec = line[:6].strip()
        if rec not in ("ATOM", "HETATM"):
            continue
        if len(line) < 66:
            continue
        try:
            atom_name = line[12:16].strip()
            chain     = line[21]
            resseq    = int(line[22:26])
            bf_val    = float(line[60:66])
        except (ValueError, IndexError):
            continue

        # Zn
        if rec == "HETATM" and atom_name.upper() in ("ZN", "ZN2+") \
                and chain == zn_chain and resseq == zn_resseq:
            zn_bf = bf_val

        # CYS sidechain + Cα atoms
        if rec == "ATOM" and atom_name in ("CA", "CB", "SG"):
            key = (chain, resseq)
            if key in set(cys_list):
                cys_bf.setdefault(key, {})[atom_name] = bf_val

    if zn_bf is not None:
        result["zn_bfactor"] = f"{zn_bf:.3f}"

    for i, key in enumerate(cys_list[:4], start=1):
        atoms = cys_bf.get(key, {})
        vals = [v for v in atoms.values()]
        if vals:
            result[f"coord_cys_{i}_bfactor_avg"] = f"{sum(vals)/len(vals):.3f}"

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_stats(
    xyz_dir: Path,
    pdb_dir: Optional[Path],
    out_csv: Path,
    force: bool = False,
) -> None:
    if out_csv.exists() and not force:
        print(f"Stats CSV already exists: {out_csv}  (use --force to recalculate)")
        return

    xyz_files = sorted(xyz_dir.glob("*.xyz"))
    if not xyz_files:
        raise SystemExit(f"No XYZ files found in {xyz_dir}")

    print(f"Computing stats for {len(xyz_files)} XYZ files …")

    FIELDNAMES = [
        "id", "xyz_path",
        "resolution_A",
        "family",
        "volume_A3",
        "q_tetra_coord", "q_tetra_ca",
        "cys_dihedral_mean_deg",
        "cys_dihedral_1_deg", "cys_dihedral_2_deg",
        "cys_dihedral_3_deg", "cys_dihedral_4_deg",
        "r_work", "r_free",
        "zn_bfactor",
        "coord_cys_1_bfactor_avg", "coord_cys_2_bfactor_avg",
        "coord_cys_3_bfactor_avg", "coord_cys_4_bfactor_avg",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_geom_fail = n_pdb_miss = 0

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()

        for xyz_path in tqdm.tqdm(xyz_files, desc="stats", leave=True):
            sid = xyz_path.stem
            row: dict = {k: "" for k in FIELDNAMES}
            row["id"]       = sid
            row["xyz_path"] = str(xyz_path.resolve())

            # --- XYZ header ---
            header = _parse_xyz_header(xyz_path)
            row["resolution_A"] = header.get("RESOLUTION_A", "")

            pdb_id = header.get("PDB", sid[:4]).lower()

            # --- Geometry ---
            s = parse_structure(xyz_path)
            if s is None:
                n_geom_fail += 1
                writer.writerow(row)
                continue
            row.update(compute_geometry(s))

            # --- Family string (always computed from XYZ CA atom SEC tags) ---
            res_meta = _parse_xyz_residues(xyz_path)
            row["family"] = compute_family_string(res_meta["cys"], res_meta["cys_sec"])

            # --- PDB stats ---
            if pdb_dir is not None:
                pdb_path = pdb_dir / f"{pdb_id}.pdb"
                if pdb_path.is_file():
                    zn_info  = res_meta["zn"]
                    cys_list = res_meta["cys"]
                    if zn_info is not None:
                        pdb_stats = _parse_pdb_stats(
                            pdb_path, zn_info[0], zn_info[1], cys_list
                        )
                        row.update(pdb_stats)
                else:
                    n_pdb_miss += 1

            n_ok += 1
            writer.writerow(row)

    print(f"\nWrote {n_ok} rows → {out_csv}")
    if n_geom_fail:
        print(f"  {n_geom_fail} XYZ files skipped (parse failed)")
    if n_pdb_miss:
        print(f"  {n_pdb_miss} structures missing PDB file (B-factors / R-factors left empty)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute per-structure stats from XYZ + PDB files."
    )
    parser.add_argument("--xyz-dir", type=Path, required=True,
                        help="Directory containing .xyz files.")
    parser.add_argument("--pdb-dir", type=Path, default=None,
                        help="Directory containing <pdbid>.pdb files for B-factor and R-factor extraction.")
    parser.add_argument("--out-csv", type=Path, default=None,
                        help="Output CSV path (default: <xyz-dir>/structure_stats.csv).")
    parser.add_argument("--force", action="store_true",
                        help="Recalculate even if output already exists.")
    args = parser.parse_args()

    xyz_dir = args.xyz_dir.expanduser().resolve()
    if not xyz_dir.is_dir():
        raise SystemExit(f"--xyz-dir not found: {xyz_dir}")

    pdb_dir = args.pdb_dir.expanduser().resolve() if args.pdb_dir else None
    if pdb_dir is not None and not pdb_dir.is_dir():
        raise SystemExit(f"--pdb-dir not found: {pdb_dir}")

    out_csv = (args.out_csv or xyz_dir / "structure_stats.csv").expanduser().resolve()

    compute_stats(xyz_dir, pdb_dir, out_csv, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
