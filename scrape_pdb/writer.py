#!/usr/bin/env python3
"""XYZ, CSV, cache writers."""

from pathlib import Path
import csv
import json
from typing import Optional
import os
import tempfile

from .models import Atom
from .constants import CLUSTERS_CSV_FIELDS, ALTLOC_REPORT_FIELDS
from .config import PipelineConfig

import logging

logger = logging.getLogger("pipeline.writer")


def _filter_backbone_atoms(atoms: list[Atom]) -> list[Atom]:
    """Remove HIS/CYS backbone atoms (ATOM=N/C/O) while keeping CA and others."""
    filtered: list[Atom] = []
    for a in atoms:
        if a.resname in {"HIS", "CYS"} and a.atom_name in {"N", "C", "O"}:
            continue
        filtered.append(a)
    return filtered


def _normalize_element_symbol(symbol: str) -> str:
    if not symbol:
        return symbol
    if len(symbol) == 1:
        return symbol.upper()
    return symbol[0].upper() + symbol[1:].lower()


def _write_normalized_xyz(input_path: Path) -> Path:
    lines = input_path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError("XYZ file is too short")

    header = lines[:2]
    body = lines[2:]

    normalized_body = []
    for line in body:
        stripped = line.strip()
        if not stripped:
            normalized_body.append(line)
            continue
        parts = stripped.split()
        if len(parts) < 4:
            normalized_body.append(line)
            continue
        parts[0] = _normalize_element_symbol(parts[0])
        normalized_body.append(" ".join(parts))

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xyz")
    tmp_path = Path(tmp.name)
    tmp.close()
    tmp_path.write_text("\n".join(header + normalized_body) + "\n")
    return tmp_path


def _assign_implicit_h_counts(mol) -> None:
    typical_valence = {
        1: 1,
        6: 4,
        7: 3,
        8: 2,
        9: 1,
        15: 3,
        16: 2,
        17: 1,
        35: 1,
        53: 1,
    }

    try:
        from openbabel import openbabel as ob
    except Exception:
        return

    for atom in ob.OBMolAtomIter(mol):
        atomic_num = atom.GetAtomicNum()
        if atomic_num in typical_valence:
            explicit_valence = atom.GetExplicitValence()
            needed = max(0, typical_valence[atomic_num] - explicit_valence)
            atom.SetImplicitHCount(needed)


def _extract_atom_comments(input_path: Path) -> list[str]:
    lines = input_path.read_text().splitlines()
    if len(lines) < 3:
        return []
    comments: list[str] = []
    for line in lines[2:]:
        stripped = line.strip()
        if not stripped:
            comments.append("")
            continue
        parts = stripped.split()
        if len(parts) <= 4:
            comments.append("")
            continue
        comment = " ".join(parts[4:])
        comments.append(comment)
    return comments


def _restore_atom_comments(output_path: Path, comments: list[str], header_comment: str) -> None:
    lines = output_path.read_text().splitlines()
    if len(lines) < 3:
        return

    header = [lines[0], header_comment]
    body = lines[2:]
    restored_body: list[str] = []

    for idx, line in enumerate(body):
        stripped = line.strip()
        if not stripped:
            restored_body.append(line)
            continue
        if idx < len(comments) and comments[idx]:
            restored_body.append(f"{stripped}  {comments[idx]}")
        else:
            restored_body.append(stripped)

    output_path.write_text("\n".join(header + restored_body) + "\n")


def _add_hydrogens_inplace(xyz_path: Path) -> None:
    try:
        from openbabel import openbabel as ob
    except Exception as exc:
        raise SystemExit("Open Babel is required to add hydrogens. Install openbabel-python.") from exc

    header_lines = xyz_path.read_text().splitlines()
    header_comment = header_lines[1] if len(header_lines) > 1 else ""
    comments = _extract_atom_comments(xyz_path)
    temp_path = None
    try:
        temp_path = _write_normalized_xyz(xyz_path)
        conv = ob.OBConversion()
        conv.SetInFormat("xyz")
        conv.SetOutFormat("xyz")

        mol = ob.OBMol()
        if not conv.ReadFile(mol, str(temp_path)):
            raise SystemExit(f"Error: Could not read {xyz_path}")

        mol.ConnectTheDots()
        mol.PerceiveBondOrders()
        _assign_implicit_h_counts(mol)
        mol.AddHydrogens()

        if not conv.WriteFile(mol, str(xyz_path)):
            raise SystemExit(f"Error: Could not write {xyz_path}")

        _restore_atom_comments(xyz_path, comments, header_comment)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def write_xyz(path: str,
              pdb_id: str,
              cluster_index: int,
              target: str,
              cutoff: float,
              origin_kind: str,
              centroid_pt: tuple[float, float, float],
              atoms: list[Atom],
              origin_atom: Atom,
              resolution_angs: Optional[float] = None,
              extra_comment: str = "") -> None:
    """Write XYZ with origin translated to origin_atom."""
    atoms = _filter_backbone_atoms(atoms)
    ox, oy, oz = origin_atom.coord
    translated = []
    for a in atoms:
        translated.append((a.element, a.x - ox, a.y - oy, a.z - oz, a))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{len(translated)}\n")
        res_str = f"{resolution_angs:.2f}" if isinstance(resolution_angs, (int, float)) else "NA"
        comment = (
            f"PDB={pdb_id} CLUSTER={cluster_index} TARGET={target} CUTOFF={cutoff:.3f} "
            f"ORIGIN={origin_kind} CENTROID=({centroid_pt[0]:.3f},{centroid_pt[1]:.3f},{centroid_pt[2]:.3f}) "
            f"RESOLUTION_A={res_str}"
        )
        if extra_comment:
            comment += " " + extra_comment.strip()
        f.write(comment + "\n")
        for elm, x, y, z, a in translated:
            resseq_icode = f"{a.resseq}{a.icode}".strip()
            meta = f"RES={a.resname} CHAIN={a.chain} RESSEQ={resseq_icode} ATOM={a.atom_name} REC={a.record}"
            f.write(f"{elm:2s}  {x: .6f}  {y: .6f}  {z: .6f}  # {meta}\n")

    _add_hydrogens_inplace(Path(path))

def ensure_csv_headers(config: PipelineConfig):
    # Ensure output directory exists and clusters CSV has header row
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    clusters_path = config.clusters_csv
    if not clusters_path.is_file():
        with open(clusters_path, "w", newline="") as csvfile:
            w = csv.writer(csvfile)
            w.writerow(CLUSTERS_CSV_FIELDS)

def write_clusters_csv_row(row: list, config: PipelineConfig):
    ensure_csv_headers(config)
    clusters_path = config.clusters_csv
    with open(clusters_path, "a", newline="") as csvfile:
        w = csv.writer(csvfile)
        w.writerow(row)

def write_altloc_report_header(config: PipelineConfig):
    altloc_path = config.altloc_report
    Path(altloc_path.parent).mkdir(parents=True, exist_ok=True)
    if not altloc_path.is_file():
        with open(altloc_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(ALTLOC_REPORT_FIELDS)

def append_altloc_rows(rows: list[list], config: PipelineConfig):
    if not rows:
        return
    altloc_path = config.altloc_report
    with open(altloc_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)

def append_cache(runs: list[dict], config: PipelineConfig):
    try:
        if os.path.isfile(config.cache):
            with open(config.cache, "r") as f:
                prev = json.load(f)
        else:
            prev = {}
        prev_runs = prev.get("runs", [])
        prev_runs.extend(runs)
        with open(config.cache, "w") as f:
            json.dump({"runs": prev_runs}, f, indent=2)
        logger.info(f"[+] Updated cache: {config.cache}")
    except Exception as e:
        logger.info(f"[!] Failed to update cache: {e}")

