#!/usr/bin/env python3
"""XYZ, CSV, cache writers."""

from pathlib import Path
import csv
import json
from typing import Optional
import os
import tempfile

from openbabel import openbabel as ob

from .models import Atom
from .constants import CLUSTERS_CSV_FIELDS, ALTLOC_REPORT_FIELDS
from .config import PipelineConfig

import logging

logger = logging.getLogger("pipeline.writer")


def _should_drop_backbone_atom(atom: Atom) -> bool:
    """Return True if HIS/CYS backbone atom (N/C/O) should be dropped."""
    res = (atom.resname or "").upper()
    if res not in {"HIS", "CYS"}:
        return False
    atom_name = (atom.atom_name or "").strip().upper()
    return atom_name in {"N", "C", "O"}


def _filter_backbone_atoms(atoms: list[Atom]) -> list[Atom]:
    """Remove HIS/CYS backbone atoms (N/C/O)."""
    return [a for a in atoms if not _should_drop_backbone_atom(a)]


def _normalize_element_symbol(symbol: str) -> str:
    """Normalize element symbols to proper case (e.g., "ZN" -> "Zn")."""
    if not symbol:
        return symbol
    if len(symbol) == 1:
        return symbol.upper()
    return symbol[0].upper() + symbol[1:].lower()


def _write_normalized_xyz(input_path: Path) -> Path:
    """Write a temporary XYZ with normalized element symbols for Open Babel."""
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


def _assign_implicit_h_counts(mol: ob.OBMol) -> None:
    """Approximate implicit H counts based on typical valence for common elements."""
    typical_valence = {
        1: 1,   # H
        6: 4,   # C
        7: 3,   # N
        8: 2,   # O
        9: 1,   # F
        15: 3,  # P
        16: 2,  # S
        17: 1,  # Cl
        35: 1,  # Br
        53: 1,  # I
    }

    for atom in ob.OBMolAtomIter(mol):
        atomic_num = atom.GetAtomicNum()
        if atomic_num in typical_valence:
            explicit_valence = atom.GetExplicitValence()
            needed = max(0, typical_valence[atomic_num] - explicit_valence)
            atom.SetImplicitHCount(needed)


def _parse_comment_fields(comment: str) -> dict[str, str]:
    """Parse key=value fields from the comment portion."""
    fields: dict[str, str] = {}
    for part in comment.split():
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key.strip()] = value.strip()
    return fields


def _extract_atom_meta_from_xyz(lines: list[str], atom_count: int) -> list[dict[str, str]]:
    """Extract per-atom metadata from XYZ trailing comments (first atom_count atoms)."""
    meta: list[dict[str, str]] = []
    for line in lines[2:2 + atom_count]:
        if "#" not in line:
            meta.append({})
            continue
        _, comment = line.split("#", 1)
        meta.append(_parse_comment_fields(comment))
    return meta


def _apply_residue_h_overrides(mol: ob.OBMol, meta: list[dict[str, str]]) -> None:
    """Override implicit H counts for specific residue/atom cases (HIS/CYS)."""
    metal_atomic_nums = {12, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 42, 44, 47, 48, 50, 52, 53}

    for idx, atom in enumerate(ob.OBMolAtomIter(mol)):
        if idx >= len(meta):
            break
        fields = meta[idx]
        res = (fields.get("RES") or "").upper()
        atom_name = (fields.get("ATOM") or "").upper()

        if res == "HIS" and atom.GetAtomicNum() == 7 and atom_name in {"ND1", "NE2"}:
            bonded_to_metal = any(
                nbr.GetAtomicNum() in metal_atomic_nums
                for nbr in ob.OBAtomAtomIter(atom)
            )
            atom.SetImplicitHCount(0 if bonded_to_metal else 1)
            continue

        if res == "CYS" and atom.GetAtomicNum() == 16 and atom_name == "SG":
            bonded_to_metal = any(
                nbr.GetAtomicNum() in metal_atomic_nums
                for nbr in ob.OBAtomAtomIter(atom)
            )
            if bonded_to_metal:
                atom.SetImplicitHCount(0)


def _append_hydrogens_in_place(xyz_path: Path) -> None:
    """Append Open Babel-added hydrogens to an XYZ file and update atom count."""
    temp_path = None
    try:
        lines = xyz_path.read_text().splitlines()
        if len(lines) < 2:
            return

        try:
            original_count = int(lines[0].strip())
        except ValueError:
            return

        meta = _extract_atom_meta_from_xyz(lines, original_count)

        temp_path = _write_normalized_xyz(xyz_path)

        conv = ob.OBConversion()
        conv.SetInFormat("xyz")
        conv.SetOutFormat("xyz")

        mol = ob.OBMol()
        if not conv.ReadFile(mol, str(temp_path)):
            logger.warning("[!] Open Babel could not read %s", xyz_path)
            return

        mol.ConnectTheDots()
        mol.PerceiveBondOrders()
        _assign_implicit_h_counts(mol)
        _apply_residue_h_overrides(mol, meta)
        mol.AddHydrogens()

        h_lines = []
        for atom in ob.OBMolAtomIter(mol):
            if atom.GetAtomicNum() != 1:
                continue
            x, y, z = atom.GetX(), atom.GetY(), atom.GetZ()
            h_lines.append(f"H  {x: .6f}  {y: .6f}  {z: .6f}")

        if not h_lines:
            return

        lines[0] = str(original_count + len(h_lines))
        xyz_path.write_text("\n".join(lines + h_lines) + "\n")
    except Exception as e:
        logger.warning("[!] Failed to append hydrogens for %s: %s", xyz_path, e)
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
    ox, oy, oz = origin_atom.coord
    translated = []
    filtered_atoms = _filter_backbone_atoms(atoms)
    for a in filtered_atoms:
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

    _append_hydrogens_in_place(Path(path))

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

