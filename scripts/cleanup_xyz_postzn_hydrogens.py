#!/usr/bin/env python3
"""Clean duplicate H atoms in XYZ files caused by post-processing protonation.

Behavior:
1) For each .xyz file, detect whether there are hydrogen atoms *before* Zn that
   look like PDB-derived atoms (metadata includes RES/CHAIN/RESSEQ).
2) If such pre-Zn hydrogens exist, remove auto-added hydrogen lines *after* Zn
   that look like writer-generated entries (e.g. '# ATOM=H ... BONDEDATOM=...').
3) Rewrite atom count in line 1.

Default mode is preview-only: cleaned files are written to a separate directory.
Use --apply-in-place to modify source files directly.
"""

from __future__ import annotations

import argparse
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm


@dataclass
class CleanupResult:
    source: Path
    output: Path | None
    status: str
    removed_count: int = 0
    removed_loose_count: int = 0
    added_count: int = 0


@dataclass
class AtomRecord:
    element: str
    x: float
    y: float
    z: float
    comment: str
    fields: dict[str, str]


_COVALENT_RADII_ANG: dict[str, float] = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "S": 1.05,
    "P": 1.07,
    "ZN": 1.22,
}


def _collect_xyz_files(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.xyz" if recursive else "*.xyz"
    return sorted(path for path in input_dir.glob(pattern) if path.is_file())


def _split_atom_line(atom_line: str) -> tuple[str, str]:
    if "#" not in atom_line:
        return atom_line, ""
    left, right = atom_line.split("#", 1)
    return left, right.strip()


def _parse_comment_fields(comment: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in comment.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key.strip().upper()] = value.strip()
    return fields


def _parse_atom_records(atom_lines: list[str]) -> list[AtomRecord]:
    records: list[AtomRecord] = []
    for line in atom_lines:
        left, comment = _split_atom_line(line)
        parts = left.split()
        if len(parts) < 4:
            continue
        try:
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
        except ValueError:
            continue
        records.append(
            AtomRecord(
                element=parts[0].strip().upper(),
                x=x,
                y=y,
                z=z,
                comment=comment,
                fields=_parse_comment_fields(comment),
            )
        )
    return records


def _norm(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float] | None:
    n = _norm(v)
    if n < 1.0e-10:
        return None
    return (v[0] / n, v[1] / n, v[2] / n)


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return _norm(_sub(a, b))


def _residue_key(fields: dict[str, str]) -> tuple[str, str, str] | None:
    res = (fields.get("RES") or "").strip().upper()
    chain = (fields.get("CHAIN") or "").strip()
    resseq = (fields.get("RESSEQ") or "").strip()
    if not res or not chain or not resseq:
        return None
    return (res, chain, resseq)


def _element_from_atom_line(atom_line: str) -> str:
    left, _ = _split_atom_line(atom_line)
    parts = left.split()
    if not parts:
        return ""
    return parts[0].strip().upper()


def _find_first_zn_index(atom_lines: list[str]) -> int | None:
    for idx, line in enumerate(atom_lines):
        if _element_from_atom_line(line) == "ZN":
            return idx
    return None


def _has_prezn_pdb_hydrogens(atom_lines: list[str], zn_index: int) -> bool:
    for line in atom_lines[:zn_index]:
        if _element_from_atom_line(line) != "H":
            continue
        _, comment = _split_atom_line(line)
        comment_u = comment.upper()
        if "RES=" in comment_u and "CHAIN=" in comment_u and "RESSEQ=" in comment_u:
            return True
    return False


def _is_postzn_autogen_hydrogen(atom_line: str) -> bool:
    if _element_from_atom_line(atom_line) != "H":
        return False
    _, comment = _split_atom_line(atom_line)
    comment_u = comment.upper()
    if "ATOM=H" not in comment_u:
        return False
    if "BONDEDATOM=" not in comment_u:
        return False
    # Keep any explicit residue-tagged H records.
    if "RESSEQ=" in comment_u or "CHAIN=" in comment_u or "RES=" in comment_u:
        return False
    return True


def _clean_atom_lines(atom_lines: list[str]) -> tuple[list[str], int, str]:
    zn_index = _find_first_zn_index(atom_lines)
    if zn_index is None:
        return atom_lines, 0, "no_zn"

    if not _has_prezn_pdb_hydrogens(atom_lines, zn_index):
        return atom_lines, 0, "no_prezn_pdb_h"

    keep: list[str] = []
    removed = 0
    for idx, line in enumerate(atom_lines):
        if idx > zn_index and _is_postzn_autogen_hydrogen(line):
            removed += 1
            continue
        keep.append(line)

    if removed == 0:
        return atom_lines, 0, "no_postzn_autogen_h"
    return keep, removed, "cleaned"


def _covalent_radius(element: str) -> float:
    return _COVALENT_RADII_ANG.get((element or "").strip().upper(), 0.77)


def _remove_loose_hydrogens(atom_lines: list[str]) -> tuple[list[str], int]:
    """Delete H atoms too far from any non-H atom to be plausibly bonded.

    Bond plausibility is estimated from covalent radii:
      threshold = r(H) + r(X) + 0.25
    """
    records = _parse_atom_records(atom_lines)
    if not records:
        return atom_lines, 0

    non_h_positions: list[tuple[float, float, float, str]] = [
        (r.x, r.y, r.z, r.element) for r in records if r.element != "H"
    ]
    if not non_h_positions:
        return atom_lines, 0

    kept_lines: list[str] = []
    removed = 0

    for line in atom_lines:
        el = _element_from_atom_line(line)
        if el != "H":
            kept_lines.append(line)
            continue

        left, _ = _split_atom_line(line)
        parts = left.split()
        if len(parts) < 4:
            kept_lines.append(line)
            continue
        try:
            hx, hy, hz = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            kept_lines.append(line)
            continue

        has_neighbor = False
        for x, y, z, elem in non_h_positions:
            threshold = _covalent_radius("H") + _covalent_radius(elem) + 0.25
            if _distance((hx, hy, hz), (x, y, z)) <= threshold:
                has_neighbor = True
                break

        if has_neighbor:
            kept_lines.append(line)
        else:
            removed += 1

    return kept_lines, removed


def _build_manual_ca_cap_hydrogens(atom_lines: list[str]) -> list[str]:
    """Add missing hydrogens for capped CA carbons by simple tetrahedral geometry.

    A residue is treated as "capped" when it has ATOM=CA and ATOM=CB but no
    residue-level backbone N/C atoms. For each such CA, ensure total CA-bound H
    count is 3 by appending missing H atoms.
    """
    records = _parse_atom_records(atom_lines)
    if not records:
        return []

    by_res: dict[tuple[str, str, str], list[AtomRecord]] = {}
    for rec in records:
        key = _residue_key(rec.fields)
        if key is None:
            continue
        by_res.setdefault(key, []).append(rec)

    bond_len = 1.09
    new_lines: list[str] = []

    for res_key, residue_atoms in by_res.items():
        name_map = {
            (a.fields.get("ATOM") or "").strip().upper(): a
            for a in residue_atoms
            if (a.fields.get("ATOM") or "").strip()
        }

        ca_rec = name_map.get("CA")
        cb_rec = name_map.get("CB")
        if ca_rec is None or cb_rec is None:
            continue

        has_backbone_n = "N" in name_map
        has_backbone_c = "C" in name_map
        if has_backbone_n or has_backbone_c:
            # Not a capped CA in this extracted representation.
            continue

        ca_pos = (ca_rec.x, ca_rec.y, ca_rec.z)
        cb_pos = (cb_rec.x, cb_rec.y, cb_rec.z)
        e1 = _normalize(_sub(cb_pos, ca_pos))
        if e1 is None:
            continue

        # Existing CA-attached H atoms:
        existing_h_dirs: list[tuple[float, float, float]] = []
        for atom_name, rec in name_map.items():
            if rec.element != "H":
                continue
            bonded_atom = (rec.fields.get("BONDEDATOM") or "").strip().upper()
            if bonded_atom == "CA" or atom_name.startswith("HA"):
                dvec = _normalize(_sub((rec.x, rec.y, rec.z), ca_pos))
                if dvec is not None:
                    existing_h_dirs.append(dvec)

        existing_h_count = len(existing_h_dirs)
        missing = max(0, 3 - existing_h_count)
        if missing == 0:
            continue

        # Build two perpendicular vectors to e1.
        trial_axis = (1.0, 0.0, 0.0)
        if abs(_dot(e1, trial_axis)) > 0.9:
            trial_axis = (0.0, 1.0, 0.0)
        n1 = _normalize(_cross(e1, trial_axis))
        if n1 is None:
            continue
        n2 = _normalize(_cross(e1, n1))
        if n2 is None:
            continue

        # Tetrahedral directions opposite CB.
        candidate_dirs: list[tuple[float, float, float]] = []
        perp_mag = 2.0 * math.sqrt(2.0) / 3.0
        for k in range(3):
            phi = 2.0 * math.pi * k / 3.0
            vec = (
                (-1.0 / 3.0) * e1[0] + perp_mag * (math.cos(phi) * n1[0] + math.sin(phi) * n2[0]),
                (-1.0 / 3.0) * e1[1] + perp_mag * (math.cos(phi) * n1[1] + math.sin(phi) * n2[1]),
                (-1.0 / 3.0) * e1[2] + perp_mag * (math.cos(phi) * n1[2] + math.sin(phi) * n2[2]),
            )
            nvec = _normalize(vec)
            if nvec is not None:
                candidate_dirs.append(nvec)

        # Prefer directions farthest from already-present CA-H vectors.
        scored: list[tuple[float, tuple[float, float, float]]] = []
        for cdir in candidate_dirs:
            if existing_h_dirs:
                max_dot = max(_dot(cdir, edir) for edir in existing_h_dirs)
            else:
                max_dot = -1.0
            scored.append((max_dot, cdir))
        scored.sort(key=lambda t: t[0])

        chosen: list[tuple[float, float, float]] = []
        for _, cdir in scored:
            too_close = False
            for edir in existing_h_dirs:
                if _dot(cdir, edir) > 0.95:
                    too_close = True
                    break
            if too_close:
                continue
            chosen.append(cdir)
            if len(chosen) >= missing:
                break

        # Fallback if filtering was too strict.
        if len(chosen) < missing:
            for _, cdir in scored:
                if cdir in chosen:
                    continue
                chosen.append(cdir)
                if len(chosen) >= missing:
                    break

        coord_val = (ca_rec.fields.get("COORD") or "").strip().upper()
        coord_text = "TRUE" if coord_val in {"1", "TRUE", "YES"} else "FALSE"
        res, chain, resseq = res_key

        for cdir in chosen[:missing]:
            hx = ca_pos[0] + bond_len * cdir[0]
            hy = ca_pos[1] + bond_len * cdir[1]
            hz = ca_pos[2] + bond_len * cdir[2]
            new_lines.append(
                f"H  {hx: .6f}  {hy: .6f}  {hz: .6f}  # "
                f"RES={res} CHAIN={chain} RESSEQ={resseq} ATOM=H COORD={coord_text} BONDEDATOM=CA"
            )

    return new_lines


def _rewrite_xyz_content(lines: list[str], cleaned_atoms: list[str], original_atom_count: int) -> str:
    header_line = lines[1] if len(lines) > 1 else ""
    trailing = lines[2 + original_atom_count :]
    out_lines = [str(len(cleaned_atoms)), header_line, *cleaned_atoms, *trailing]
    return "\n".join(out_lines) + "\n"


def _process_file(
    xyz_path: Path,
    input_root: Path,
    preview_root: Path | None,
    apply_in_place: bool,
) -> CleanupResult:
    lines = xyz_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return CleanupResult(source=xyz_path, output=None, status="invalid_short")

    try:
        atom_count = int(lines[0].strip())
    except ValueError:
        return CleanupResult(source=xyz_path, output=None, status="invalid_count")

    atom_lines = lines[2 : 2 + atom_count]
    if len(atom_lines) < atom_count:
        return CleanupResult(source=xyz_path, output=None, status="invalid_truncated")

    cleaned_atoms, removed_count, status = _clean_atom_lines(atom_lines)
    if status != "cleaned":
        return CleanupResult(source=xyz_path, output=None, status=status)

    # Step 1: remove likely orphan H atoms before adding targeted CA cap hydrogens.
    no_loose_atoms, removed_loose_count = _remove_loose_hydrogens(cleaned_atoms)

    # Step 2: add only missing CA-cap hydrogens.
    added_h_lines = _build_manual_ca_cap_hydrogens(no_loose_atoms)
    final_atoms = no_loose_atoms + added_h_lines
    out_text = _rewrite_xyz_content(lines, final_atoms, atom_count)

    if apply_in_place:
        out_path = xyz_path
    else:
        assert preview_root is not None
        rel = xyz_path.relative_to(input_root)
        out_path = preview_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(out_text, encoding="utf-8")
    return CleanupResult(
        source=xyz_path,
        output=out_path,
        status="cleaned",
        removed_count=removed_count,
        removed_loose_count=removed_loose_count,
        added_count=len(added_h_lines),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Remove post-Zn auto-added H atoms when residue-annotated pre-Zn H atoms already exist."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing .xyz files")
    parser.add_argument("-r", "--recursive", action="store_true", help="Process subdirectories recursively")
    parser.add_argument(
        "--apply-in-place",
        action="store_true",
        help="Modify source XYZ files directly (default: preview mode writes to temp dir)",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="Preview output directory (default: auto temp dir)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap for number of XYZ files to process (useful for testing)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        print(f"Error: not a directory: {input_dir}")
        return 2

    xyz_files = _collect_xyz_files(input_dir, recursive=args.recursive)
    if args.max_files is not None:
        xyz_files = xyz_files[: max(0, args.max_files)]

    if not xyz_files:
        print(f"No .xyz files found in {input_dir}")
        return 0

    preview_root: Path | None = None
    if not args.apply_in_place:
        if args.preview_dir is not None:
            preview_root = args.preview_dir.expanduser().resolve()
            preview_root.mkdir(parents=True, exist_ok=True)
        else:
            preview_root = Path(tempfile.mkdtemp(prefix="xyz_h_cleanup_preview_"))

    cleaned = 0
    removed_total = 0
    removed_loose_total = 0
    added_total = 0
    skipped = 0
    changed_paths: list[tuple[Path, Path, int, int, int]] = []

    for xyz_path in tqdm(xyz_files, desc="Cleaning XYZ", unit="file"):
        result = _process_file(
            xyz_path=xyz_path,
            input_root=input_dir,
            preview_root=preview_root,
            apply_in_place=args.apply_in_place,
        )

        if result.status == "cleaned" and result.output is not None:
            cleaned += 1
            removed_total += result.removed_count
            removed_loose_total += result.removed_loose_count
            added_total += result.added_count
            changed_paths.append(
                (
                    result.source,
                    result.output,
                    result.removed_count,
                    result.removed_loose_count,
                    result.added_count,
                )
            )
        else:
            skipped += 1

    mode = "in-place" if args.apply_in_place else "preview"
    print(f"Mode: {mode}")
    if preview_root is not None:
        print(f"Preview directory: {preview_root}")

    print(
        f"Done. Processed {len(xyz_files)} file(s): "
        f"cleaned {cleaned}, skipped {skipped}, removed {removed_total} H line(s), "
        f"removed loose {removed_loose_total} H line(s), "
        f"added {added_total} CA-only H line(s)."
    )

    if changed_paths:
        print("\nChanged files:")
        for src, out, removed, removed_loose, added in changed_paths:
            print(
                f"- {src} -> {out} "
                f"(removed {removed} post-Zn H; removed loose {removed_loose} H; "
                f"added {added} CA-only H)"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
