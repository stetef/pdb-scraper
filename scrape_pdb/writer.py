#!/usr/bin/env python3
"""XYZ, CSV, cache writers."""

from pathlib import Path
import csv
import json
from typing import Optional
import os
import tempfile
from functools import lru_cache
from collections import Counter
import subprocess
import sys
import re

from openbabel import openbabel as ob
from openmm.app import ForceField

from .models import Atom
from .constants import CLUSTERS_CSV_FIELDS, ALTLOC_REPORT_FIELDS
from .config import PipelineConfig

import logging

logger = logging.getLogger("pipeline.writer")


# Canonical aliases for mapping PDB residue names to AMBER template names.
_AMBER_TEMPLATE_ALIASES: dict[str, tuple[str, ...]] = {
    "HIS": ("HIE", "HID", "HIP", "HIS"),
    "CYS": ("CYS", "CYX", "CYM"),
}

_COVALENT_RADII_ANG: dict[str, float] = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "S": 1.05,
    "P": 1.07,
    "ZN": 1.22,
}

# Matches the element table used by scripts/check_multiplicity.py
_ATOMIC_NUMBERS: dict[str, int] = {
    "H": 1, "HE": 2,
    "LI": 3, "BE": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "NE": 10,
    "NA": 11, "MG": 12, "AL": 13, "SI": 14, "P": 15, "S": 16, "CL": 17, "AR": 18,
    "K": 19, "CA": 20, "SC": 21, "TI": 22, "V": 23, "CR": 24, "MN": 25,
    "FE": 26, "CO": 27, "NI": 28, "CU": 29, "ZN": 30,
    "SE": 34, "BR": 35, "MO": 42, "CD": 48, "I": 53, "HG": 80,
}

_ELEMENT_FORMAL_CHARGE_FALLBACKS: dict[str, int] = {
    "ZN": 2,
}


def _should_drop_backbone_atom(atom: Atom) -> bool:
    """Return True if HIS/CYS backbone atom (N/C/O) should be dropped."""
    res = (atom.resname or "").upper()
    if res not in {"HIS", "CYS"}:
        return False
    atom_name = (atom.atom_name or "").strip().upper()
    return atom_name in {"N", "C", "O"}


def _residue_identity(atom: Atom) -> tuple[str, str]:
    """Residue identity keyed by chain and integer RESSEQ when parsable."""
    return atom.chain, str(atom.resseq).strip()


def _is_origin_zn(atom: Atom, origin_atom: Atom) -> bool:
    """Return True when atom is the translated Zn origin atom for this cluster."""
    return atom.serial == origin_atom.serial and (atom.element or "").strip().upper() == "ZN"


def _covalent_radius(element: str) -> float:
    """Return covalent radius for bond-heuristic checks."""
    return _COVALENT_RADII_ANG.get((element or "").strip().upper(), 0.77)


def _are_probably_bonded(a: Atom, b: Atom) -> bool:
    """Heuristic covalent-bond check from interatomic distance and covalent radii."""
    if a.serial == b.serial:
        return False

    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    d2 = dx * dx + dy * dy + dz * dz
    if d2 < 0.25:
        return False

    max_dist = _covalent_radius(a.element) + _covalent_radius(b.element) + 0.45
    return d2 <= max_dist * max_dist


def _is_bonded_to_any_qm_atom(atom: Atom, qm_atoms: list[Atom]) -> bool:
    """Return True when atom is directly bonded to any heavy atom in the QM XYZ set."""
    for qm_atom in qm_atoms:
        if _are_probably_bonded(atom, qm_atom):
            return True
    return False


def _parse_formal_charge_value(charge_text: str) -> Optional[int]:
    """Parse PDB-style formal charge text (e.g., '2+', '+2', '1-', '-1')."""
    text = (charge_text or "").strip()
    if not text:
        return None

    if text[-1] in {"+", "-"} and text[:-1].isdigit():
        sign = 1 if text[-1] == "+" else -1
        return sign * int(text[:-1])
    if text[0] in {"+", "-"} and text[1:].isdigit():
        return int(text)
    if text in {"+", "-"}:
        return 1 if text == "+" else -1
    if text.isdigit():
        return int(text)
    return None


def _fallback_charge(atom: Atom) -> float:
    """Fallback integer formal charge for atoms missing AMBER partial charges."""
    parsed = _parse_formal_charge_value(atom.charge)
    if parsed is not None:
        return float(parsed)
    return float(_ELEMENT_FORMAL_CHARGE_FALLBACKS.get((atom.element or "").strip().upper(), 0))


def _estimate_cluster_charge(atoms: list[Atom]) -> tuple[float, int]:
    """Estimate cluster charge from AMBER partial charges with sensible fallbacks."""
    charges = build_amber_charge_lookup()
    total = 0.0
    for atom in atoms:
        q = _lookup_atom_charge(atom, charges)
        if q is None:
            q = _fallback_charge(atom)
        total += q

    rounded = int(round(total))
    if abs(total - rounded) > 0.1:
        logger.warning(
            "[!] Cluster charge %.3f differs from rounded integer %d by > 0.1 e.",
            total,
            rounded,
        )
    return total, rounded


def _read_xyz_elements(xyz_path: Path) -> list[str]:
    """Read element symbols from XYZ atom lines."""
    lines = xyz_path.read_text().splitlines()
    if len(lines) < 2:
        return []
    try:
        atom_count = int(lines[0].strip())
    except ValueError:
        return []

    elements: list[str] = []
    for line in lines[2:2 + atom_count]:
        parts = line.split()
        if parts:
            elements.append(parts[0])
    return elements


def _infer_multiplicity_from_elements(elements: list[str], charge: int) -> int:
    """Infer lowest multiplicity using the same electron-parity logic as check_multiplicity.py."""
    counts = Counter((e or "").strip().upper() for e in elements if e)

    unknown = [el for el in counts if el not in _ATOMIC_NUMBERS]
    if unknown:
        logger.warning("[!] Unknown element(s) for multiplicity estimate: %s", sorted(unknown))

    neutral_electrons = sum(_ATOMIC_NUMBERS.get(el, 0) * n for el, n in counts.items())
    total_electrons = neutral_electrons - charge
    return 1 if (total_electrons % 2 == 0) else 2


def _infer_multiplicity_via_script(xyz_path: Path, charge: int) -> Optional[int]:
    """Call scripts/check_multiplicity.py and parse its reported lowest multiplicity."""
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "check_multiplicity.py"
    if not script_path.is_file():
        return None

    try:
        proc = subprocess.run(
            [sys.executable, str(script_path), str(xyz_path), str(charge)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            logger.warning("[!] check_multiplicity.py failed for %s: %s", xyz_path, proc.stderr.strip())
            return None

        match = re.search(r"Lowest multiplicity:\s*(\d+)", proc.stdout)
        if not match:
            logger.warning("[!] Could not parse multiplicity from check_multiplicity.py output for %s", xyz_path)
            return None
        return int(match.group(1))
    except Exception as exc:
        logger.warning("[!] Failed invoking check_multiplicity.py for %s: %s", xyz_path, exc)
        return None


def _upsert_comment_field(comment: str, key: str, value: str) -> str:
    """Insert or replace KEY=value token in the XYZ comment line."""
    key_eq = f"{key}="
    parts = comment.split()
    replaced = False
    out: list[str] = []
    for part in parts:
        if part.startswith(key_eq):
            out.append(f"{key}={value}")
            replaced = True
        else:
            out.append(part)
    if not replaced:
        out.append(f"{key}={value}")
    return " ".join(out)


def _set_xyz_comment_field(xyz_path: Path, key: str, value: str) -> None:
    """Update a comment KEY=value field in-place in an XYZ file."""
    lines = xyz_path.read_text().splitlines()
    if len(lines) < 2:
        return
    lines[1] = _upsert_comment_field(lines[1], key, value)
    xyz_path.write_text("\n".join(lines) + "\n")


def _estimate_cluster_charge_from_xyz(xyz_path: Path) -> Optional[tuple[float, int]]:
    """Estimate total partial charge from final XYZ (including Open Babel-added H atoms)."""
    temp_path = None
    try:
        temp_path = _write_normalized_xyz(xyz_path)

        conv = ob.OBConversion()
        conv.SetInFormat("xyz")
        mol = ob.OBMol()
        if not conv.ReadFile(mol, str(temp_path)):
            return None

        mol.ConnectTheDots()
        mol.PerceiveBondOrders()

        model_names = ("gasteiger", "eem", "qeq", "qtpie")
        model_used = None
        for model_name in model_names:
            model = ob.OBChargeModel.FindType(model_name)
            if model is None:
                continue
            if model.ComputeCharges(mol):
                model_used = model_name
                break

        if model_used is None:
            logger.warning("[!] Open Babel failed to compute partial charges for %s", xyz_path)
            return None

        total = 0.0
        for atom in ob.OBMolAtomIter(mol):
            total += float(atom.GetPartialCharge())

        rounded = int(round(total))
        logger.info("[+] XYZ charge estimated with Open Babel model: %s", model_used)
        if abs(total - rounded) > 0.1:
            logger.warning(
                "[!] Cluster charge %.3f differs from rounded integer %d by > 0.1 e.",
                total,
                rounded,
            )
        return total, rounded
    except Exception as exc:
        logger.warning("[!] Failed XYZ-based charge estimate for %s: %s", xyz_path, exc)
        return None
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _coord_targets_from_keys(
    coord_residue_keys: Optional[set[tuple[str, str, str]]],
) -> set[tuple[str, str]]:
    """Convert coordinating residue keys into (chain, resseq_str) pairs."""
    out: set[tuple[str, str]] = set()
    if not coord_residue_keys:
        return out
    for _res, chain, resseq in coord_residue_keys:
        out.add((chain, str(resseq).strip()))
    return out


def _neighbor_targets_from_coord_keys(
    coord_residue_keys: Optional[set[tuple[str, str, str]]],
    *,
    offsets: tuple[int, ...],
) -> set[tuple[str, int]]:
    """Build (chain, resseq_int+offset) targets from coordinating residue keys."""
    out: set[tuple[str, int]] = set()
    if not coord_residue_keys:
        return out
    for _resname, chain, resseq in coord_residue_keys:
        r_int = _parse_resseq_int(resseq)
        if r_int is None:
            continue
        for off in offsets:
            out.add((chain, r_int + off))
    return out


def _filter_backbone_atoms(
    atoms: list[Atom],
    *,
    keep_residue_targets: Optional[set[tuple[str, str]]] = None,
) -> list[Atom]:
    """Remove HIS/CYS backbone atoms (N/C/O), except for explicitly kept residues."""
    keep_residue_targets = keep_residue_targets or set()
    out: list[Atom] = []
    for atom in atoms:
        if _residue_identity(atom) in keep_residue_targets:
            out.append(atom)
            continue
        if _should_drop_backbone_atom(atom):
            continue
        out.append(atom)
    return out


def _augment_xyz_atoms_with_neighbors(
    selected_atoms: list[Atom],
    source_atoms: list[Atom],
    coord_residue_keys: Optional[set[tuple[str, str, str]]],
) -> tuple[list[Atom], set[tuple[str, str]]]:
    """Add residues at RESSEQ±1 for coordinating residues into XYZ atom set."""
    coord_targets = _coord_targets_from_keys(coord_residue_keys)
    neighbor_targets_int = _neighbor_targets_from_coord_keys(coord_residue_keys, offsets=(-1, 1))
    neighbor_targets = {(chain, str(resseq)) for chain, resseq in neighbor_targets_int}

    keep_targets = set(coord_targets)
    keep_targets.update(neighbor_targets)

    serial_to_atom: dict[int, Atom] = {}
    ordered: list[Atom] = []

    for atom in selected_atoms:
        if atom.serial in serial_to_atom:
            continue
        serial_to_atom[atom.serial] = atom
        ordered.append(atom)

    for atom in source_atoms:
        if _residue_identity(atom) not in neighbor_targets:
            continue
        if atom.serial in serial_to_atom:
            continue
        serial_to_atom[atom.serial] = atom
        ordered.append(atom)

    return ordered, keep_targets


@lru_cache(maxsize=1)
def build_amber_charge_lookup() -> dict[tuple[str, str], float]:
    """
    Build (template_name, atom_name) -> partial charge from OpenMM amber19-all.xml.

    Values are in elementary charge units.
    """
    ff = ForceField("amber19-all.xml")
    charges: dict[tuple[str, str], float] = {}

    for template_name, template in getattr(ff, "_templates", {}).items():
        tname = str(template_name).upper()
        for atom in getattr(template, "atoms", []):
            atom_name = str(getattr(atom, "name", "")).upper()
            params = getattr(atom, "parameters", {}) or {}
            charge = params.get("charge")
            if not atom_name or charge is None:
                continue
            charges[(tname, atom_name)] = float(charge)

    return charges


def _template_candidates_for_residue(resname: str) -> list[str]:
    """Return likely AMBER template names for a residue name."""
    base = (resname or "").strip().upper()
    if not base:
        return []

    aliases = _AMBER_TEMPLATE_ALIASES.get(base, (base,))
    out: list[str] = []
    for alias in aliases:
        for candidate in (alias, f"N{alias}", f"C{alias}"):
            if candidate not in out:
                out.append(candidate)
    return out


def _lookup_atom_charge(atom: Atom, charges: dict[tuple[str, str], float]) -> Optional[float]:
    """Return AMBER charge for an atom using residue-template lookup."""
    resname = (atom.resname or "").strip().upper()
    atom_name = (atom.atom_name or "").strip().upper()

    candidates = _template_candidates_for_residue(resname)
    for template_name in candidates:
        key = (template_name, atom_name)
        if key in charges:
            return charges[key]

    logger.warning(
        "[!] Missing AMBER charge for %s/%s; skipping this point charge entry.",
        resname,
        atom_name,
    )
    return None


def _parse_resseq_int(resseq: str) -> Optional[int]:
    try:
        return int(str(resseq).strip())
    except Exception:
        return None


def _select_pc_neighbor_atoms(
    atoms: list[Atom],
    coord_residue_keys: set[tuple[str, str, str]],
    *,
    offsets: tuple[int, ...] = (-1, 1),
) -> tuple[list[Atom], set[tuple[str, int]]]:
    """
    Select atoms from sequence-neighbor residues (RESSEQ ± 1) of coordinating residues.

    Deduplicates by serial and limits matches to the same chain.
    """
    neighbor_targets: set[tuple[str, int]] = set()
    coord_targets: set[tuple[str, int]] = set()
    for _resname, chain, resseq in coord_residue_keys:
        r_int = _parse_resseq_int(resseq)
        if r_int is None:
            continue
        coord_targets.add((chain, r_int))
        for off in offsets:
            neighbor_targets.add((chain, r_int + off))

    if not neighbor_targets:
        return [], neighbor_targets

    seen_serials: set[int] = set()
    out: list[Atom] = []
    for atom in atoms:
        r_int = _parse_resseq_int(atom.resseq)
        if r_int is None:
            continue
        residue_id = (atom.chain, r_int)
        if residue_id not in neighbor_targets:
            continue
        # Never include coordinating residues, even if they are +/-1 neighbors
        if residue_id in coord_targets:
            continue
        if atom.serial in seen_serials:
            continue
        seen_serials.add(atom.serial)
        out.append(atom)

    return out, neighbor_targets


def _select_pc_radius_atoms(
    atoms: list[Atom],
    origin_atom: Atom,
    coord_residue_keys: set[tuple[str, str, str]],
    neighbor_targets: set[tuple[str, int]],
    radius_ang: float = 10.0,
) -> list[Atom]:
    """Select atoms within radius from origin, excluding coordinating and +/-1 neighbor residues."""
    ox, oy, oz = origin_atom.coord
    r2 = radius_ang * radius_ang

    coord_targets = {(chain, _parse_resseq_int(resseq)) for _res, chain, resseq in coord_residue_keys}
    coord_targets = {(c, r) for (c, r) in coord_targets if r is not None}

    out: list[Atom] = []
    for atom in atoms:
        r_int = _parse_resseq_int(atom.resseq)
        if r_int is None:
            continue
        residue_id = (atom.chain, r_int)

        if residue_id in coord_targets:
            continue
        if residue_id in neighbor_targets:
            continue

        dx = atom.x - ox
        dy = atom.y - oy
        dz = atom.z - oz
        if dx * dx + dy * dy + dz * dz <= r2:
            out.append(atom)

    return out


def _write_backbone_point_charges(
    pc_path: Path,
    atoms: list[Atom],
    origin_atom: Atom,
    coord_residue_keys: Optional[set[tuple[str, str, str]]] = None,
    source_atoms: Optional[list[Atom]] = None,
    exclude_serials: Optional[set[int]] = None,
    qm_atoms: Optional[list[Atom]] = None,
) -> None:
    """Write .pc file for sequence-neighbor residues (RESSEQ +/- 1) of coordinating residues."""
    if not coord_residue_keys:
        lines = ["0"]
        pc_path.parent.mkdir(parents=True, exist_ok=True)
        pc_path.write_text("\n".join(lines) + "\n")
        return

    atom_pool = source_atoms if source_atoms is not None else atoms
    neighbor_atoms, _neighbor_targets = _select_pc_neighbor_atoms(
        atom_pool,
        coord_residue_keys,
        offsets=(-1, 1),
    )

    selected_atoms: list[Atom] = []
    seen_serials: set[int] = set()
    excluded = exclude_serials or set()
    for atom in neighbor_atoms:
        if atom.serial in seen_serials:
            continue
        if atom.serial in excluded:
            continue
        seen_serials.add(atom.serial)
        selected_atoms.append(atom)
    charges = build_amber_charge_lookup()
    ox, oy, oz = origin_atom.coord

    payload_lines: list[str] = []
    for atom in selected_atoms:
        q = _lookup_atom_charge(atom, charges)
        if q is None:
            continue
        x = atom.x - ox
        y = atom.y - oy
        z = atom.z - oz
        payload_lines.append(f"{q:.6f} {x:.6f} {y:.6f} {z:.6f}")

    lines = [str(len(payload_lines)), *payload_lines]

    pc_path.parent.mkdir(parents=True, exist_ok=True)
    pc_path.write_text("\n".join(lines) + "\n")


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
            # Only the Zn-bound imidazole N should be deprotonated.
            atom.SetImplicitHCount(0 if bonded_to_metal else 1)
            continue

        if res == "HIS" and atom_name in {"CD2", "CE1"} and atom.GetAtomicNum() == 6:
            # Ensure aromatic C-H population on the non-coordinating imidazole positions.
            atom.SetImplicitHCount(1)
            continue

        if res == "HIS" and atom_name == "CG" and atom.GetAtomicNum() == 6:
            atom.SetImplicitHCount(0)
            continue

        coord_value = (fields.get("COORD") or "").strip().upper()
        is_coord = coord_value in {"1", "TRUE", "YES"}

        if res == "CYS" and atom.GetAtomicNum() == 16 and atom_name == "SG":
            bonded_to_metal = any(
                nbr.GetAtomicNum() in metal_atomic_nums
                for nbr in ob.OBAtomAtomIter(atom)
            )
            if bonded_to_metal or is_coord:
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

        def _coord_flag_from_fields(fields: dict[str, str]) -> Optional[bool]:
            value = (fields.get("COORD") or "").strip().upper()
            if value in {"1", "TRUE", "YES"}:
                return True
            if value in {"0", "FALSE", "NO"}:
                return False
            return None

        heavy_atoms_for_fallback: list[tuple[float, float, float, bool, str]] = []
        for idx, atom in enumerate(ob.OBMolAtomIter(mol)):
            if idx >= original_count:
                break
            if atom.GetAtomicNum() == 1:
                continue
            fields = meta[idx] if idx < len(meta) else {}
            coord_flag = _coord_flag_from_fields(fields)
            bonded_atom_name = (fields.get("ATOM") or "UNK").strip().upper() or "UNK"
            heavy_atoms_for_fallback.append(
                (atom.GetX(), atom.GetY(), atom.GetZ(), bool(coord_flag), bonded_atom_name)
            )

        def _closest_heavy_info(x: float, y: float, z: float) -> tuple[bool, str]:
            best_dist2 = None
            best_flag = False
            best_atom_name = "UNK"
            for hx, hy, hz, hflag, hname in heavy_atoms_for_fallback:
                dx = hx - x
                dy = hy - y
                dz = hz - z
                d2 = dx * dx + dy * dy + dz * dz
                if best_dist2 is None or d2 < best_dist2:
                    best_dist2 = d2
                    best_flag = hflag
                    best_atom_name = hname
            return best_flag, best_atom_name

        h_lines = []
        for atom in ob.OBMolAtomIter(mol):
            if atom.GetAtomicNum() != 1:
                continue
            x, y, z = atom.GetX(), atom.GetY(), atom.GetZ()
            coord_flag: Optional[bool] = None
            bonded_atom_name = "UNK"
            for nbr in ob.OBAtomAtomIter(atom):
                if nbr.GetAtomicNum() == 1:
                    continue
                n_idx = nbr.GetIdx() - 1
                if 0 <= n_idx < len(meta):
                    parent_meta = meta[n_idx]
                    coord_flag = _coord_flag_from_fields(parent_meta)
                    bonded_atom_name = (parent_meta.get("ATOM") or "UNK").strip().upper() or "UNK"
                if coord_flag is None:
                    coord_flag, closest_name = _closest_heavy_info(nbr.GetX(), nbr.GetY(), nbr.GetZ())
                    if bonded_atom_name == "UNK":
                        bonded_atom_name = closest_name
                break

            if coord_flag is None:
                coord_flag, closest_name = _closest_heavy_info(x, y, z)
                if bonded_atom_name == "UNK":
                    bonded_atom_name = closest_name

            coord_text = "TRUE" if coord_flag else "FALSE"
            h_lines.append(
                f"H  {x: .6f}  {y: .6f}  {z: .6f}  # ATOM=H COORD={coord_text} BONDEDATOM={bonded_atom_name}"
            )

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
              extra_comment: str = "",
              coord_residue_keys: Optional[set[tuple[str, str, str]]] = None,
              pc_source_atoms: Optional[list[Atom]] = None) -> None:
    """Write compact and extended XYZ files, plus point charges for compact file."""
    ox, oy, oz = origin_atom.coord

    source_atoms = pc_source_atoms if pc_source_atoms is not None else atoms

    coord_targets = _coord_targets_from_keys(coord_residue_keys)
    neighbor_targets_int = _neighbor_targets_from_coord_keys(coord_residue_keys, offsets=(-1, 1))
    neighbor_targets = {(chain, str(resseq)) for chain, resseq in neighbor_targets_int}
    neighbor_targets -= coord_targets

    def _collect_atoms_for_targets(targets: set[tuple[str, str]]) -> list[Atom]:
        seen: set[int] = set()
        out: list[Atom] = []
        for atom in source_atoms:
            residue_id = _residue_identity(atom)
            include = residue_id in targets or atom.serial == origin_atom.serial
            if not include:
                continue
            if atom.serial in seen:
                continue
            seen.add(atom.serial)
            out.append(atom)
        return out

    compact_atoms_raw = _collect_atoms_for_targets(coord_targets)
    compact_atoms: list[Atom] = []
    for atom in compact_atoms_raw:
        residue_id = _residue_identity(atom)
        atom_name = (atom.atom_name or "").strip().upper()
        if residue_id in coord_targets and atom_name in {"N", "C", "O"}:
            continue
        compact_atoms.append(atom)

    extended_targets = set(coord_targets)
    extended_targets.update(neighbor_targets)
    extended_atoms = _collect_atoms_for_targets(extended_targets)

    def _write_single_xyz_file(
        xyz_path: Path,
        file_atoms: list[Atom],
        neighbor_targets_for_meta: set[tuple[str, str]],
    ) -> None:
        translated: list[tuple[str, float, float, float, Atom]] = []
        for atom in file_atoms:
            translated.append((atom.element, atom.x - ox, atom.y - oy, atom.z - oz, atom))

        os.makedirs(os.path.dirname(str(xyz_path)), exist_ok=True)
        with open(xyz_path, "w") as f:
            f.write(f"{len(translated)}\n")
            res_str = f"{resolution_angs:.2f}" if isinstance(resolution_angs, (int, float)) else "NA"
            comment = (
                f"PDB={pdb_id} CLUSTER={cluster_index} TARGET={target} CUTOFF={cutoff:.3f} "
                f"ORIGIN={origin_kind} CENTROID=({centroid_pt[0]:.3f},{centroid_pt[1]:.3f},{centroid_pt[2]:.3f}) "
                f"RESOLUTION_A={res_str} CHARGE_UNROUNDED=NA CHARGE_ROUNDED=NA"
            )
            if extra_comment:
                comment += " " + extra_comment.strip()
            f.write(comment + "\n")

            for elm, x, y, z, atom in translated:
                resseq_icode = f"{atom.resseq}{atom.icode}".strip()
                meta = (
                    f"RES={atom.resname} CHAIN={atom.chain} RESSEQ={resseq_icode} "
                    f"ATOM={atom.atom_name} REC={atom.record}"
                )
                residue_id = _residue_identity(atom)
                if residue_id in coord_targets or _is_origin_zn(atom, origin_atom):
                    meta += " COORD=TRUE"
                elif residue_id in neighbor_targets_for_meta:
                    meta += " COORD=FALSE"
                f.write(f"{elm:2s}  {x: .6f}  {y: .6f}  {z: .6f}  # {meta}\n")

        _append_hydrogens_in_place(xyz_path)

        charge_from_xyz = _estimate_cluster_charge_from_xyz(xyz_path)
        if charge_from_xyz is None:
            charge_unrounded, charge_rounded = _estimate_cluster_charge(file_atoms)
        else:
            charge_unrounded, charge_rounded = charge_from_xyz
        _set_xyz_comment_field(xyz_path, "CHARGE_UNROUNDED", f"{charge_unrounded:.3f}")
        _set_xyz_comment_field(xyz_path, "CHARGE_ROUNDED", str(charge_rounded))

        multiplicity = _infer_multiplicity_via_script(xyz_path, charge_rounded)
        if multiplicity is None:
            elements = _read_xyz_elements(xyz_path)
            multiplicity = _infer_multiplicity_from_elements(elements, charge_rounded)
        _set_xyz_comment_field(xyz_path, "MULTIPLICITY", str(multiplicity))

    small_xyz_path = Path(path)
    extended_xyz_path = small_xyz_path.with_name(f"{small_xyz_path.stem}-extended{small_xyz_path.suffix}")

    _write_single_xyz_file(small_xyz_path, compact_atoms, set())
    _write_single_xyz_file(extended_xyz_path, extended_atoms, neighbor_targets)

    xyz_serials = {a.serial for a in compact_atoms}

    pc_path = small_xyz_path.with_suffix(".pc")
    _write_backbone_point_charges(
        pc_path,
        atoms,
        origin_atom,
        coord_residue_keys=coord_residue_keys,
        source_atoms=pc_source_atoms,
        exclude_serials=xyz_serials,
        qm_atoms=compact_atoms,
    )

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

