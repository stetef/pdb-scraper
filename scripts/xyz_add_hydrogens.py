#!/usr/bin/env python3
"""Hydrogen addition helpers for XYZ files using Open Babel.

This mirrors the hydrogen logic used in `scrape_pdb/writer.py` so scripts can
reuse consistent protonation behavior from the command line layer.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from openbabel import openbabel as ob


def _normalize_element_symbol(symbol: str) -> str:
    if not symbol:
        return symbol
    if len(symbol) == 1:
        return symbol.upper()
    return symbol[0].upper() + symbol[1:].lower()


def _write_normalized_xyz(input_path: Path) -> Path:
    lines = input_path.read_text(encoding="utf-8").splitlines()
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
    tmp_path.write_text("\n".join(header + normalized_body) + "\n", encoding="utf-8")
    return tmp_path


def _assign_implicit_h_counts(mol: ob.OBMol) -> None:
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

    for atom in ob.OBMolAtomIter(mol):
        atomic_num = atom.GetAtomicNum()
        if atomic_num in typical_valence:
            explicit_valence = atom.GetExplicitValence()
            needed = max(0, typical_valence[atomic_num] - explicit_valence)
            atom.SetImplicitHCount(needed)


def _parse_comment_fields(comment: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in comment.split():
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key.strip()] = value.strip()
    return fields


def _extract_atom_meta_from_xyz(lines: list[str], atom_count: int) -> list[dict[str, str]]:
    meta: list[dict[str, str]] = []
    for line in lines[2 : 2 + atom_count]:
        if "#" not in line:
            meta.append({})
            continue
        _, comment = line.split("#", 1)
        meta.append(_parse_comment_fields(comment))
    return meta


def _apply_residue_h_overrides(mol: ob.OBMol, meta: list[dict[str, str]]) -> None:
    metal_atomic_nums = {12, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 42, 44, 47, 48, 50, 52, 53}

    for idx, atom in enumerate(ob.OBMolAtomIter(mol)):
        if idx >= len(meta):
            break
        fields = meta[idx]
        res = (fields.get("RES") or "").upper()
        atom_name = (fields.get("ATOM") or "").upper()

        if res == "HIS" and atom.GetAtomicNum() == 7 and atom_name in {"ND1", "NE2"}:
            bonded_to_metal = any(nbr.GetAtomicNum() in metal_atomic_nums for nbr in ob.OBAtomAtomIter(atom))
            coord_flag = fields.get("COORD") == "1"
            atom.SetImplicitHCount(0 if bonded_to_metal or coord_flag else 1)
            continue

        if res == "CYS" and atom.GetAtomicNum() == 16 and atom_name == "SG":
            bonded_to_metal = any(nbr.GetAtomicNum() in metal_atomic_nums for nbr in ob.OBAtomAtomIter(atom))
            coord_flag = fields.get("COORD") == "1"
            if bonded_to_metal or coord_flag:
                atom.SetImplicitHCount(0)


def add_hydrogens_in_place(xyz_path: Path) -> int:
    """Append Open Babel-added hydrogens to XYZ and return number of H atoms added."""
    temp_path = None
    try:
        lines = xyz_path.read_text(encoding="utf-8").splitlines()
        if len(lines) < 2:
            return 0

        try:
            original_count = int(lines[0].strip())
        except ValueError:
            return 0

        meta = _extract_atom_meta_from_xyz(lines, original_count)
        temp_path = _write_normalized_xyz(xyz_path)

        conv = ob.OBConversion()
        conv.SetInFormat("xyz")
        conv.SetOutFormat("xyz")

        mol = ob.OBMol()
        if not conv.ReadFile(mol, str(temp_path)):
            return 0

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
            return 0

        lines[0] = str(original_count + len(h_lines))
        xyz_path.write_text("\n".join(lines + h_lines) + "\n", encoding="utf-8")
        return len(h_lines)
    except Exception:
        return 0
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)
