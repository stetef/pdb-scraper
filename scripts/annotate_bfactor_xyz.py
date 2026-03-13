#!/usr/bin/env python3
"""Annotate XYZ atom metadata with B-factors from PDB/mmCIF sources.

For each non-hydrogen atom line in an XYZ file, this script reads metadata from the
end-of-line comment (for example: ``RES=... CHAIN=... RESSEQ=... ATOM=...``),
matches it to a structure atom record, and appends or replaces ``BF=<value>``.

Matching priority:
1. Local PDB in ``../results/validated_structures/<pdb_id>.pdb`` (relative to XYZ dir)
2. Local CIF in ``../results/validated_structures/<pdb_id>.cif``
3. Download CIF into that directory and parse it
4. If still unresolved, use ``BF=-1``
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from xyz_add_hydrogens import add_hydrogens_in_place

RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
METAL_RESNAMES = {"ZN", "MG", "CA", "FE", "MN", "CU", "CO", "NI"}


@dataclass(frozen=True)
class _AtomRecord:
    record_type: str  # ATOM or HETATM
    resname: str
    chain: str
    resseq: str
    atom_name: str
    altloc: str
    x: float
    y: float
    z: float
    element: str
    bfactor: float
    occupancy: float


def _parse_meta_tokens(comment: str) -> List[str]:
    return [token for token in comment.strip().split() if token]


def _meta_dict(tokens: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key and value:
            out[key] = value
    return out


def _normalize_resseq(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    try:
        return str(int(text))
    except ValueError:
        return text


def _safe_float(text: str | None, default: float) -> float:
    if text is None:
        return default
    raw = str(text).strip()
    if raw in {"", ".", "?"}:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _update_comment_with_bf(comment: str, bf_value: float) -> str:
    tokens = _parse_meta_tokens(comment)
    tokens = [t for t in tokens if not t.startswith("BF=")]
    if bf_value < 0:
        tokens.append("BF=-1")
    else:
        tokens.append(f"BF={bf_value:.2f}")
    return " ".join(tokens).strip()


def _is_hydrogen_xyz_line(line_left: str) -> bool:
    parts = line_left.split()
    if not parts:
        return False
    return parts[0].strip().upper() == "H"


def _parse_pdb_atoms(pdb_path: Path) -> List[_AtomRecord]:
    atoms: List[_AtomRecord] = []
    with pdb_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not (raw.startswith("ATOM") or raw.startswith("HETATM")):
                continue
            line = raw.rstrip("\n")
            if len(line) < 80:
                line = line + (" " * (80 - len(line)))

            record_type = line[0:6].strip().upper()
            atom_name = line[12:16].strip().upper()
            altloc = line[16:17].strip().upper()
            resname = line[17:20].strip().upper()
            chain = line[21:22].strip()
            resseq = _normalize_resseq(line[22:26])
            x = _safe_float(line[30:38], default=0.0)
            y = _safe_float(line[38:46], default=0.0)
            z = _safe_float(line[46:54], default=0.0)
            element = line[76:78].strip().upper()
            if not element:
                element = atom_name[:1].upper()
            occupancy = _safe_float(line[54:60], default=0.0)
            bfactor = _safe_float(line[60:66], default=-1.0)

            atoms.append(
                _AtomRecord(
                    record_type=record_type,
                    resname=resname,
                    chain=chain,
                    resseq=resseq,
                    atom_name=atom_name,
                    altloc=altloc,
                    x=x,
                    y=y,
                    z=z,
                    element=element,
                    bfactor=bfactor,
                    occupancy=occupancy,
                )
            )
    return atoms


def _get_cif_col(cif_dict: dict, key: str) -> List[str]:
    values = cif_dict.get(key, [])
    if isinstance(values, str):
        return [values]
    return list(values)


def _parse_cif_atoms(cif_path: Path) -> List[_AtomRecord]:
    cif_dict = MMCIF2Dict(str(cif_path))

    group = _get_cif_col(cif_dict, "_atom_site.group_PDB")
    label_atom = _get_cif_col(cif_dict, "_atom_site.label_atom_id")
    label_comp = _get_cif_col(cif_dict, "_atom_site.label_comp_id")
    auth_atom = _get_cif_col(cif_dict, "_atom_site.auth_atom_id")
    auth_comp = _get_cif_col(cif_dict, "_atom_site.auth_comp_id")
    auth_asym = _get_cif_col(cif_dict, "_atom_site.auth_asym_id")
    auth_seq = _get_cif_col(cif_dict, "_atom_site.auth_seq_id")
    label_alt = _get_cif_col(cif_dict, "_atom_site.label_alt_id")
    label_asym = _get_cif_col(cif_dict, "_atom_site.label_asym_id")
    label_seq = _get_cif_col(cif_dict, "_atom_site.label_seq_id")
    type_symbol = _get_cif_col(cif_dict, "_atom_site.type_symbol")
    cartn_x = _get_cif_col(cif_dict, "_atom_site.Cartn_x")
    cartn_y = _get_cif_col(cif_dict, "_atom_site.Cartn_y")
    cartn_z = _get_cif_col(cif_dict, "_atom_site.Cartn_z")
    b_iso = _get_cif_col(cif_dict, "_atom_site.B_iso_or_equiv")
    occupancy = _get_cif_col(cif_dict, "_atom_site.occupancy")

    count = len(group)
    atoms: List[_AtomRecord] = []
    for idx in range(count):
        record_type = (group[idx] if idx < len(group) else "ATOM").strip().upper()

        atom_name = ""
        if idx < len(auth_atom) and auth_atom[idx] not in {".", "?", ""}:
            atom_name = auth_atom[idx]
        elif idx < len(label_atom):
            atom_name = label_atom[idx]

        resname = ""
        if idx < len(auth_comp) and auth_comp[idx] not in {".", "?", ""}:
            resname = auth_comp[idx]
        elif idx < len(label_comp):
            resname = label_comp[idx]

        chain = ""
        if idx < len(auth_asym) and auth_asym[idx] not in {".", "?", ""}:
            chain = auth_asym[idx]
        elif idx < len(label_asym):
            chain = label_asym[idx]

        resseq_raw = ""
        if idx < len(auth_seq) and auth_seq[idx] not in {".", "?", ""}:
            resseq_raw = auth_seq[idx]
        elif idx < len(label_seq):
            resseq_raw = label_seq[idx]

        altloc = ""
        if idx < len(label_alt) and label_alt[idx] not in {".", "?", ""}:
            altloc = label_alt[idx].strip().upper()

        element = ""
        if idx < len(type_symbol) and type_symbol[idx] not in {".", "?", ""}:
            element = type_symbol[idx].strip().upper()
        if not element:
            element = atom_name[:1].upper()

        x = _safe_float(cartn_x[idx] if idx < len(cartn_x) else None, default=0.0)
        y = _safe_float(cartn_y[idx] if idx < len(cartn_y) else None, default=0.0)
        z = _safe_float(cartn_z[idx] if idx < len(cartn_z) else None, default=0.0)
        bfac = _safe_float(b_iso[idx] if idx < len(b_iso) else None, default=-1.0)
        occ = _safe_float(occupancy[idx] if idx < len(occupancy) else None, default=0.0)

        atoms.append(
            _AtomRecord(
                record_type=record_type,
                resname=resname.strip().upper(),
                chain=chain.strip(),
                resseq=_normalize_resseq(resseq_raw),
                atom_name=atom_name.strip().upper(),
                altloc=altloc,
                x=x,
                y=y,
                z=z,
                element=element,
                bfactor=bfac,
                occupancy=occ,
            )
        )

    return atoms


def _extract_cluster_id_from_xyz_name(xyz_path: Path) -> str:
    match = re.search(r"_cluster(\d+)$", xyz_path.stem)
    return match.group(1) if match else ""


def _preferred_altloc_for_xyz(xyz_path: Path) -> str:
    xyz_dir = xyz_path.parent
    output_dir = xyz_dir.parent
    pdb_id = _extract_pdb_id_from_xyz_name(xyz_path)
    cluster_id = _extract_cluster_id_from_xyz_name(xyz_path)

    clusters_summary = output_dir / "clusters_summary.csv"
    if clusters_summary.exists():
        try:
            with clusters_summary.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    xyz_rel = (row.get("XYZ_PATH", "") or "").strip()
                    if not xyz_rel:
                        continue
                    if Path(xyz_rel).name != xyz_path.name:
                        continue
                    altloc = (row.get("ALTLOC", "") or "").strip().upper()
                    if altloc:
                        return altloc
        except Exception:
            pass

    altloc_report = output_dir / "altloc_report.csv"
    if altloc_report.exists() and pdb_id and cluster_id:
        try:
            found: set[str] = set()
            with altloc_report.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if (row.get("pdb_id", "") or "").strip().lower() != pdb_id:
                        continue
                    if (row.get("cluster_id", "") or "").strip() != cluster_id:
                        continue
                    altloc = (row.get("altloc", "") or "").strip().upper()
                    if altloc:
                        found.add(altloc)
            if len(found) == 1:
                return next(iter(found))
        except Exception:
            pass

    return ""


def _atom_selection_score(atom: _AtomRecord, preferred_altloc: str) -> Tuple[int, int, float]:
    altloc = atom.altloc.strip().upper()
    if preferred_altloc:
        # Prefer requested altloc, then no-altloc atoms, then others.
        altloc_rank = 2 if altloc == preferred_altloc else (1 if not altloc else 0)
    else:
        # Without a preferred altloc, prefer no-altloc atoms, then occupancy.
        altloc_rank = 1 if not altloc else 0
    has_b = 1 if atom.bfactor >= 0 else 0
    return (altloc_rank, has_b, atom.occupancy)


def _pick_best_atoms(
    atoms: Iterable[_AtomRecord],
    preferred_altloc: str,
) -> Dict[Tuple[str, str, str, str, str], _AtomRecord]:
    best: Dict[Tuple[str, str, str, str, str], _AtomRecord] = {}
    for atom in atoms:
        key = (atom.record_type, atom.resname, atom.chain, atom.resseq, atom.atom_name)
        current = best.get(key)
        if current is None or _atom_selection_score(atom, preferred_altloc) > _atom_selection_score(
            current, preferred_altloc
        ):
            best[key] = atom
    return best


def _extract_pdb_id_from_xyz_name(xyz_path: Path) -> str:
    token = xyz_path.stem.split("_", 1)[0].strip()
    return token.lower()


def _candidate_structure_dirs(xyz_path: Path) -> List[Path]:
    xyz_dir = xyz_path.parent
    return [
        (xyz_dir / "../results/validated_structures").resolve(),
        (xyz_dir / "../../results/validated_structures").resolve(),
    ]


def _resolve_structure_files(xyz_path: Path) -> Tuple[Path, Path]:
    pdb_id = _extract_pdb_id_from_xyz_name(xyz_path)
    dirs = _candidate_structure_dirs(xyz_path)

    existing_dir = None
    for cand in dirs:
        if cand.exists():
            existing_dir = cand
            break
    target_dir = existing_dir or dirs[0]

    pdb_path = target_dir / f"{pdb_id}.pdb"
    cif_path = target_dir / f"{pdb_id}.cif"
    return pdb_path, cif_path


def _seqres_by_chain(pdb_path: Path) -> Dict[str, List[str]]:
    seqres: Dict[str, List[str]] = {}
    try:
        with pdb_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("SEQRES"):
                    continue
                chain = line[11:12].strip()
                if not chain:
                    continue
                residues = [tok.strip().upper() for tok in line[19:].split() if tok.strip()]
                if not residues:
                    continue
                seqres.setdefault(chain, []).extend(residues)
    except Exception:
        return {}
    return seqres


def _residue_sort_key(resseq: str) -> Tuple[int, str]:
    text = (resseq or "").strip()
    m = re.match(r"^(-?\d+)([A-Za-z]?)$", text)
    if m:
        return (int(m.group(1)), m.group(2))
    try:
        return (int(text), "")
    except Exception:
        return (10**9, text)


def _ordered_residues_by_chain(structure_atoms: Iterable[_AtomRecord]) -> Dict[str, List[Tuple[str, str, str, str]]]:
    seen: Set[Tuple[str, str, str, str]] = set()
    by_chain: Dict[str, List[Tuple[str, str, str, str]]] = {}
    for atom in structure_atoms:
        if atom.record_type != "ATOM":
            continue
        key = (atom.record_type, atom.resname, atom.chain, atom.resseq)
        if key in seen:
            continue
        seen.add(key)
        by_chain.setdefault(atom.chain, []).append(key)

    for chain in by_chain:
        by_chain[chain].sort(key=lambda k: _residue_sort_key(k[3]))
    return by_chain


def _map_observed_residues_to_seqres(
    ordered_residues: Dict[str, List[Tuple[str, str, str, str]]],
    seqres: Dict[str, List[str]],
) -> Dict[Tuple[str, str, str, str], int]:
    mapping: Dict[Tuple[str, str, str, str], int] = {}
    for chain, residues in ordered_residues.items():
        chain_seq = seqres.get(chain, [])
        if not chain_seq:
            continue
        pos = 0
        for residue_key in residues:
            _, resname, _, _ = residue_key
            found = -1
            for idx in range(pos, len(chain_seq)):
                if chain_seq[idx] == resname:
                    found = idx
                    break
            if found >= 0:
                mapping[residue_key] = found
                pos = found + 1
    return mapping


def _format_xyz_atom_line(
    atom: _AtomRecord,
    cx: float,
    cy: float,
    cz: float,
    bf_lookup: Dict[Tuple[str, str, str, str, str], float],
) -> str:
    bf = bf_lookup.get((atom.record_type, atom.resname, atom.chain, atom.resseq, atom.atom_name), -1.0)
    bf_token = "BF=-1" if bf < 0 else f"BF={bf:.2f}"
    comment = (
        f"RES={atom.resname} CHAIN={atom.chain} RESSEQ={atom.resseq} "
        f"ATOM={atom.atom_name} REC={atom.record_type} {bf_token}"
    )
    x = atom.x - cx
    y = atom.y - cy
    z = atom.z - cz
    return f"{atom.element:<2} {x:11.6f} {y:11.6f} {z:11.6f}  # {comment}"


def _write_variant_xyz(
    out_path: Path,
    header: List[str],
    atom_lines: List[str],
) -> None:
    first_line = str(len(atom_lines))
    output_lines = [first_line] + (header[1:] if len(header) > 1 else []) + atom_lines
    out_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    add_hydrogens_in_place(out_path)


def _coordinating_residue_keys_from_xyz(heavy_body: List[str]) -> Set[Tuple[str, str, str, str]]:
    residue_keys: Set[Tuple[str, str, str, str]] = set()
    for raw in heavy_body:
        stripped = raw.strip()
        if not stripped or "#" not in stripped:
            continue
        _, right = stripped.split("#", 1)
        meta = _meta_dict(_parse_meta_tokens(right.strip()))
        res = meta.get("RES", "").strip().upper()
        chain = meta.get("CHAIN", "").strip()
        resseq = _normalize_resseq(meta.get("RESSEQ", ""))
        rec = meta.get("REC", "").strip().upper() or "ATOM"
        atom_name = meta.get("ATOM", "").strip().upper()
        if not (res and chain and resseq and atom_name):
            continue
        if rec == "ATOM" and res not in METAL_RESNAMES:
            residue_keys.add((rec, res, chain, resseq))
    return residue_keys


def _nearby_residue_keys(
    structure_atoms: List[_AtomRecord],
    seed_residues: Set[Tuple[str, str, str, str]],
    pdb_path: Path | None,
) -> Set[Tuple[str, str, str, str]]:
    ordered = _ordered_residues_by_chain(structure_atoms)
    seq_map = _map_observed_residues_to_seqres(ordered, _seqres_by_chain(pdb_path)) if pdb_path else {}
    out: Set[Tuple[str, str, str, str]] = set()

    # Build quick index by chain for observed-order fallback.
    index_by_chain: Dict[str, Dict[Tuple[str, str, str, str], int]] = {}
    for chain, residues in ordered.items():
        index_by_chain[chain] = {r: i for i, r in enumerate(residues)}

    for seed in seed_residues:
        _, _, chain, _ = seed
        chain_residues = ordered.get(chain, [])
        if not chain_residues:
            continue

        if seed in seq_map:
            seed_idx = seq_map[seed]
            for residue in chain_residues:
                mapped = seq_map.get(residue)
                if mapped is None:
                    continue
                if abs(mapped - seed_idx) <= 2:
                    out.add(residue)
            continue

        # Fallback to observed residue order when SEQRES mapping is unavailable.
        obs_idx = index_by_chain.get(chain, {}).get(seed)
        if obs_idx is None:
            continue
        lo = max(0, obs_idx - 2)
        hi = min(len(chain_residues), obs_idx + 3)
        out.update(chain_residues[lo:hi])

    return out


def _atoms_within_radius(structure_atoms: Iterable[_AtomRecord], centroid: Tuple[float, float, float], radius: float) -> List[_AtomRecord]:
    cx, cy, cz = centroid
    r2 = radius * radius
    out: List[_AtomRecord] = []
    for atom in structure_atoms:
        if atom.element.upper() == "H":
            continue
        dx = atom.x - cx
        dy = atom.y - cy
        dz = atom.z - cz
        if (dx * dx + dy * dy + dz * dz) <= r2:
            out.append(atom)
    return out


def _download_cif(pdb_id: str, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = RCSB_CIF_URL.format(pdb_id=pdb_id.upper())
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pdb-scraper-bfactor-annotator/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            if getattr(resp, "status", 200) != 200:
                return False
            data = resp.read()
        out_path.write_bytes(data)
        return True
    except Exception:
        return False


def _resolve_structure_atoms_for_xyz(xyz_path: Path) -> List[_AtomRecord]:
    pdb_path, cif_path = _resolve_structure_files(xyz_path)

    if pdb_path.exists():
        return _parse_pdb_atoms(pdb_path)

    if cif_path.exists():
        return _parse_cif_atoms(cif_path)

    pdb_id = _extract_pdb_id_from_xyz_name(xyz_path)
    downloaded = _download_cif(pdb_id, cif_path)
    if downloaded and cif_path.exists():
        return _parse_cif_atoms(cif_path)

    return []


def _extract_centroid(header_line: str) -> Tuple[float, float, float] | None:
    match = re.search(r"CENTROID=\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)", header_line)
    if not match:
        return None
    return (float(match.group(1)), float(match.group(2)), float(match.group(3)))


def _append_backbone_xyz(
    xyz_path: Path,
    header: List[str],
    body: List[str],
    structure_atoms: List[_AtomRecord],
    bf_lookup: Dict[Tuple[str, str, str, str, str], float],
    pdb_path: Path | None,
) -> None:
    if not header:
        return

    centroid = _extract_centroid(header[1] if len(header) > 1 else "")
    if centroid is None:
        # Without centroid there is no reliable way to transform PDB/CIF coordinates.
        return

    cx, cy, cz = centroid
    # Start backbone output from heavy atoms only; hydrogens are rebuilt afterward.
    heavy_body = [line for line in body if not _is_hydrogen_xyz_line(line.split("#", 1)[0].strip())]
    residue_keys: set[Tuple[str, str, str, str]] = set()
    existing_atoms: set[Tuple[str, str, str, str, str]] = set()

    for raw in heavy_body:
        stripped = raw.strip()
        if not stripped:
            continue
        if "#" not in stripped:
            continue
        _, right = stripped.split("#", 1)
        meta = _meta_dict(_parse_meta_tokens(right.strip()))
        res = meta.get("RES", "").strip().upper()
        chain = meta.get("CHAIN", "").strip()
        resseq = _normalize_resseq(meta.get("RESSEQ", ""))
        rec = meta.get("REC", "").strip().upper() or "ATOM"
        atom_name = meta.get("ATOM", "").strip().upper()
        if not (res and chain and resseq and atom_name):
            continue
        residue_keys.add((rec, res, chain, resseq))
        existing_atoms.add((rec, res, chain, resseq, atom_name))

    # Pick highest-occupancy representative for every structure atom key.
    best_structure: Dict[Tuple[str, str, str, str, str], _AtomRecord] = {}
    for atom in structure_atoms:
        key = (atom.record_type, atom.resname, atom.chain, atom.resseq, atom.atom_name)
        current = best_structure.get(key)
        if current is None or atom.occupancy > current.occupancy:
            best_structure[key] = atom

    appended_lines: List[str] = []
    for rec, res, chain, resseq in sorted(residue_keys):
        if res in METAL_RESNAMES:
            continue
        for atom_name in ("N", "C", "O"):
            if (rec, res, chain, resseq, atom_name) in existing_atoms:
                continue
            hit = best_structure.get((rec, res, chain, resseq, atom_name))
            if hit is None and rec != "ATOM":
                hit = best_structure.get(("ATOM", res, chain, resseq, atom_name))
            if hit is None:
                hit = best_structure.get(("HETATM", res, chain, resseq, atom_name))
            if hit is None:
                continue
            bf = bf_lookup.get((hit.record_type, hit.resname, hit.chain, hit.resseq, hit.atom_name), -1.0)
            x = hit.x - cx
            y = hit.y - cy
            z = hit.z - cz
            bf_token = "BF=-1" if bf < 0 else f"BF={bf:.2f}"
            comment = (
                f"RES={hit.resname} CHAIN={hit.chain} RESSEQ={hit.resseq} "
                f"ATOM={hit.atom_name} REC={hit.record_type} {bf_token}"
            )
            line = f"{hit.element:<2} {x:11.6f} {y:11.6f} {z:11.6f}  # {comment}"
            appended_lines.append(line)

    out_path = xyz_path.with_name(f"{xyz_path.stem}_backbone{xyz_path.suffix}")
    backbone_lines = heavy_body + appended_lines
    _write_variant_xyz(out_path, header, backbone_lines)

    # Nearby-residue variant: +/-2 residues around each coordinating residue.
    coord_residues = _coordinating_residue_keys_from_xyz(heavy_body)
    near_keys = _nearby_residue_keys(structure_atoms, coord_residues, pdb_path)
    near_atoms = [
        atom
        for atom in structure_atoms
        if atom.record_type == "ATOM"
        and (atom.record_type, atom.resname, atom.chain, atom.resseq) in near_keys
        and atom.element.upper() != "H"
    ]
    # Keep metal centers from the original cluster context.
    for raw in heavy_body:
        stripped = raw.strip()
        if not stripped:
            continue
        if "#" not in stripped:
            continue
        _, right = stripped.split("#", 1)
        meta = _meta_dict(_parse_meta_tokens(right.strip()))
        if meta.get("RES", "").strip().upper() in METAL_RESNAMES:
            left = stripped.split("#", 1)[0].strip()
            near_line = f"{left}  # {right.strip()}"
            near_atoms.append(
                _AtomRecord(
                    record_type=meta.get("REC", "HETATM").strip().upper() or "HETATM",
                    resname=meta.get("RES", "").strip().upper(),
                    chain=meta.get("CHAIN", "").strip(),
                    resseq=_normalize_resseq(meta.get("RESSEQ", "")),
                    atom_name=meta.get("ATOM", "").strip().upper(),
                    altloc="",
                    x=float(left.split()[1]) + cx,
                    y=float(left.split()[2]) + cy,
                    z=float(left.split()[3]) + cz,
                    element=left.split()[0].strip().upper(),
                    bfactor=_safe_float(meta.get("BF", "-1"), -1.0),
                    occupancy=1.0,
                )
            )

    # Deduplicate by atom identity.
    uniq_near: Dict[Tuple[str, str, str, str, str], _AtomRecord] = {}
    for atom in near_atoms:
        key = (atom.record_type, atom.resname, atom.chain, atom.resseq, atom.atom_name)
        if key not in uniq_near:
            uniq_near[key] = atom
    near_lines = [_format_xyz_atom_line(atom, cx, cy, cz, bf_lookup) for atom in uniq_near.values()]
    out_near = xyz_path.with_name(f"{xyz_path.stem}_nearby_res{xyz_path.suffix}")
    _write_variant_xyz(out_near, header, near_lines)

    # 10 A pocket variant around centroid (Zn-centered frame).
    pocket_atoms = _atoms_within_radius(structure_atoms, centroid, radius=10.0)
    uniq_pocket: Dict[Tuple[str, str, str, str, str], _AtomRecord] = {}
    for atom in pocket_atoms:
        key = (atom.record_type, atom.resname, atom.chain, atom.resseq, atom.atom_name)
        if key not in uniq_pocket:
            uniq_pocket[key] = atom
    pocket_lines = [_format_xyz_atom_line(atom, cx, cy, cz, bf_lookup) for atom in uniq_pocket.values()]
    out_pocket = xyz_path.with_name(f"{xyz_path.stem}_10A_pocket{xyz_path.suffix}")
    _write_variant_xyz(out_pocket, header, pocket_lines)


def _lookup_bfactor(
    bf_lookup: Dict[Tuple[str, str, str, str, str], float],
    meta: Dict[str, str],
) -> float:
    record_type = meta.get("REC", "").strip().upper()
    resname = meta.get("RES", "").strip().upper()
    chain = meta.get("CHAIN", "").strip()
    resseq = _normalize_resseq(meta.get("RESSEQ", ""))
    atom_name = meta.get("ATOM", "").strip().upper()

    if not (resname and chain and resseq and atom_name):
        return -1.0

    if record_type:
        exact = bf_lookup.get((record_type, resname, chain, resseq, atom_name))
        if exact is not None:
            return exact

    # Fallback when REC is missing or differs between sources.
    for rt in ("ATOM", "HETATM"):
        value = bf_lookup.get((rt, resname, chain, resseq, atom_name))
        if value is not None:
            return value

    return -1.0


def annotate_xyz_bfactors(xyz_path: Path) -> Tuple[bool, int, int]:
    if not xyz_path.exists():
        return False, 0, 0

    lines = xyz_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        return False, 0, 0

    structure_atoms = _resolve_structure_atoms_for_xyz(xyz_path)
    pdb_path, _ = _resolve_structure_files(xyz_path)
    pdb_for_seq = pdb_path if pdb_path.exists() else None
    preferred_altloc = _preferred_altloc_for_xyz(xyz_path)
    selected_atoms = _pick_best_atoms(structure_atoms, preferred_altloc)
    bf_lookup = {key: atom.bfactor for key, atom in selected_atoms.items()}

    header = lines[:2]
    body = lines[2:]
    updated_body: List[str] = []

    updated_atoms = 0
    total_non_h = 0

    for raw in body:
        stripped = raw.strip()
        if not stripped:
            updated_body.append(raw)
            continue

        if "#" in stripped:
            left, right = stripped.split("#", 1)
            left = left.strip()
            comment = right.strip()
        else:
            left = stripped
            comment = ""

        if _is_hydrogen_xyz_line(left):
            updated_body.append(stripped)
            continue

        total_non_h += 1
        meta = _meta_dict(_parse_meta_tokens(comment))
        bf_value = _lookup_bfactor(bf_lookup, meta)
        comment = _update_comment_with_bf(comment, bf_value)
        updated_atoms += 1

        if comment:
            updated_body.append(f"{left}  # {comment}")
        else:
            updated_body.append(left)

    xyz_path.write_text("\n".join(header + updated_body) + "\n", encoding="utf-8")
    _append_backbone_xyz(
        xyz_path,
        header,
        updated_body,
        list(selected_atoms.values()),
        bf_lookup,
        pdb_for_seq,
    )
    return True, updated_atoms, total_non_h


def _iter_xyz_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    skip_suffixes = ("_backbone", "_nearby_res", "_10A_pocket")
    return sorted(
        p
        for p in path.rglob("*.xyz")
        if p.is_file() and not any(p.stem.endswith(suffix) for suffix in skip_suffixes)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate XYZ non-H atom metadata with BF=<B-factor> using matching PDB/CIF records."
        )
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to an .xyz file or a directory to process recursively.",
    )

    args = parser.parse_args()
    target = args.path

    xyz_files = _iter_xyz_files(target)
    if not xyz_files:
        raise SystemExit(f"No .xyz files found for: {target}")

    failures = 0
    total_files = 0
    total_atoms = 0

    for xyz_path in xyz_files:
        ok, updated_atoms, _ = annotate_xyz_bfactors(xyz_path)
        total_files += 1
        total_atoms += updated_atoms
        if not ok:
            failures += 1
            print(f"Failed: {xyz_path}", file=sys.stderr)

    print(f"Processed {total_files} file(s); annotated {total_atoms} non-H atom line(s).")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
