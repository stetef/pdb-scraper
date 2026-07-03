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
    removed_bare_h_count: int = 0
    removed_loose_count: int = 0
    removed_sulfur_h_count: int = 0
    removed_cap_h_count: int = 0
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
    return sorted(
        p for p in input_dir.glob(pattern)
        if p.is_file() and "-extended" not in p.stem
    )


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


def _remove_bare_hydrogens(atom_lines: list[str]) -> tuple[list[str], int]:
    """Remove H atom lines lacking metadata comments (double-placed writer artifacts)."""
    out: list[str] = []
    removed = 0
    for line in atom_lines:
        if _element_from_atom_line(line) != "H":
            out.append(line)
            continue

        _, comment = _split_atom_line(line)
        if comment.strip():
            out.append(line)
        else:
            removed += 1

    return out, removed


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


def _remove_sulfur_bonded_hydrogens(atom_lines: list[str]) -> tuple[list[str], int]:
    """Remove H atoms bound to sulfur (thiol protons)."""
    records = _parse_atom_records(atom_lines)
    if not records:
        return atom_lines, 0

    s_positions: list[tuple[float, float, float]] = [
        (r.x, r.y, r.z) for r in records if r.element == "S"
    ]
    if not s_positions:
        return atom_lines, 0

    out: list[str] = []
    removed = 0
    for line in atom_lines:
        if _element_from_atom_line(line) != "H":
            out.append(line)
            continue

        left, comment = _split_atom_line(line)
        fields = _parse_comment_fields(comment)
        bonded_atom = (fields.get("BONDEDATOM") or "").strip().upper()
        if bonded_atom == "S":
            removed += 1
            continue

        parts = left.split()
        if len(parts) < 4:
            out.append(line)
            continue
        try:
            hx, hy, hz = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            out.append(line)
            continue

        # Also remove if geometrically bonded to sulfur.
        near_s = False
        threshold = _covalent_radius("H") + _covalent_radius("S") + 0.25
        for sx, sy, sz in s_positions:
            if _distance((hx, hy, hz), (sx, sy, sz)) <= threshold:
                near_s = True
                break

        if near_s:
            removed += 1
        else:
            out.append(line)

    return out, removed


def _is_coord_true(fields: dict[str, str]) -> bool:
    val = (fields.get("COORD") or "").strip().upper()
    return val in {"1", "TRUE", "YES"}


def _remove_uncoordinated_residues(atom_lines: list[str]) -> tuple[list[str], int]:
    """Remove residue-tagged atoms from residues without any Zn coordination flag."""
    records = _parse_atom_records(atom_lines)
    if not records:
        return atom_lines, 0

    residue_has_coord: dict[tuple[str, str, str], bool] = {}
    for rec in records:
        key = _residue_key(rec.fields)
        if key is None:
            continue
        residue_has_coord[key] = residue_has_coord.get(key, False) or _is_coord_true(rec.fields)

    # If there is no coordination metadata at all, do not prune aggressively.
    if not residue_has_coord:
        return atom_lines, 0

    out: list[str] = []
    removed = 0
    for line in atom_lines:
        el = _element_from_atom_line(line)
        if el == "ZN":
            out.append(line)
            continue

        _, comment = _split_atom_line(line)
        fields = _parse_comment_fields(comment)
        key = _residue_key(fields)
        if key is None:
            out.append(line)
            continue

        if residue_has_coord.get(key, False):
            out.append(line)
        else:
            removed += 1

    return out, removed


def _enforce_cap_ca_cb_hydrogens(atom_lines: list[str]) -> tuple[list[str], int, int]:
    """Enforce exact CA/CB hydrogen counts for capped CA-CB residues.

    A residue is treated as "capped" when it has ATOM=CA and ATOM=CB but no
    residue-level backbone N/C atoms.
    - CA is enforced to 3 H
    - CB is enforced to 2 H

    Existing CA/CB-attached hydrogens in those capped residues are removed and
    rebuilt to guarantee exact counts.
    """
    records = _parse_atom_records(atom_lines)
    if not records:
        return atom_lines, 0, 0

    by_res: dict[tuple[str, str, str], list[AtomRecord]] = {}
    for rec in records:
        key = _residue_key(rec.fields)
        if key is None:
            continue
        by_res.setdefault(key, []).append(rec)

    capped_keys: set[tuple[str, str, str]] = set()
    for res_key, residue_atoms in by_res.items():
        name_map = {
            (a.fields.get("ATOM") or "").strip().upper(): a
            for a in residue_atoms
            if (a.fields.get("ATOM") or "").strip()
        }
        if "CA" not in name_map or "CB" not in name_map:
            continue
        if "N" in name_map or "C" in name_map:
            continue
        capped_keys.add(res_key)

    # Remove existing CA/CB hydrogens for capped residues so we can rebuild exact counts.
    filtered_lines: list[str] = []
    removed_existing = 0
    for line in atom_lines:
        if _element_from_atom_line(line) != "H":
            filtered_lines.append(line)
            continue

        _, comment = _split_atom_line(line)
        fields = _parse_comment_fields(comment)
        key = _residue_key(fields)
        if key not in capped_keys:
            filtered_lines.append(line)
            continue

        atom_name = (fields.get("ATOM") or "").strip().upper()
        bonded_atom = (fields.get("BONDEDATOM") or "").strip().upper()
        if bonded_atom in {"CA", "CB"} or atom_name.startswith("HA") or atom_name.startswith("HB"):
            removed_existing += 1
            continue

        filtered_lines.append(line)

    # Re-parse after removal for clean geometry source.
    records = _parse_atom_records(filtered_lines)
    by_res = {}
    for rec in records:
        key = _residue_key(rec.fields)
        if key is None:
            continue
        by_res.setdefault(key, []).append(rec)

    bond_len = 1.09
    new_lines: list[str] = []

    for res_key, residue_atoms in by_res.items():
        if res_key not in capped_keys:
            continue

        name_map = {
            (a.fields.get("ATOM") or "").strip().upper(): a
            for a in residue_atoms
            if (a.fields.get("ATOM") or "").strip()
        }

        ca_rec = name_map.get("CA")
        cb_rec = name_map.get("CB")
        if ca_rec is None or cb_rec is None:
            continue

        ca_pos = (ca_rec.x, ca_rec.y, ca_rec.z)
        cb_pos = (cb_rec.x, cb_rec.y, cb_rec.z)
        e_ca_to_cb = _normalize(_sub(cb_pos, ca_pos))
        e_cb_to_ca = _normalize(_sub(ca_pos, cb_pos))
        if e_ca_to_cb is None or e_cb_to_ca is None:
            continue

        def _tetra_dirs_opposite(main_axis: tuple[float, float, float], count: int) -> list[tuple[float, float, float]]:
            trial_axis = (1.0, 0.0, 0.0)
            if abs(_dot(main_axis, trial_axis)) > 0.9:
                trial_axis = (0.0, 1.0, 0.0)
            n1 = _normalize(_cross(main_axis, trial_axis))
            if n1 is None:
                return []
            n2 = _normalize(_cross(main_axis, n1))
            if n2 is None:
                return []

            dirs: list[tuple[float, float, float]] = []
            perp_mag = 2.0 * math.sqrt(2.0) / 3.0
            for k in range(3):
                phi = 2.0 * math.pi * k / 3.0
                vec = (
                    (-1.0 / 3.0) * main_axis[0] + perp_mag * (math.cos(phi) * n1[0] + math.sin(phi) * n2[0]),
                    (-1.0 / 3.0) * main_axis[1] + perp_mag * (math.cos(phi) * n1[1] + math.sin(phi) * n2[1]),
                    (-1.0 / 3.0) * main_axis[2] + perp_mag * (math.cos(phi) * n1[2] + math.sin(phi) * n2[2]),
                )
                nvec = _normalize(vec)
                if nvec is not None:
                    dirs.append(nvec)
            return dirs[:count]

        ca_dirs = _tetra_dirs_opposite(e_ca_to_cb, count=3)
        cb_dirs = _tetra_dirs_opposite(e_cb_to_ca, count=2)
        if len(ca_dirs) < 3 or len(cb_dirs) < 2:
            continue

        coord_ca = "TRUE" if _is_coord_true(ca_rec.fields) else "FALSE"
        coord_cb = "TRUE" if _is_coord_true(cb_rec.fields) else "FALSE"
        res, chain, resseq = res_key

        for cdir in ca_dirs:
            hx = ca_pos[0] + bond_len * cdir[0]
            hy = ca_pos[1] + bond_len * cdir[1]
            hz = ca_pos[2] + bond_len * cdir[2]
            new_lines.append(
                f"H  {hx: .6f}  {hy: .6f}  {hz: .6f}  # "
                f"RES={res} CHAIN={chain} RESSEQ={resseq} ATOM=H COORD={coord_ca} BONDEDATOM=CA"
            )

        for cdir in cb_dirs:
            hx = cb_pos[0] + bond_len * cdir[0]
            hy = cb_pos[1] + bond_len * cdir[1]
            hz = cb_pos[2] + bond_len * cdir[2]
            new_lines.append(
                f"H  {hx: .6f}  {hy: .6f}  {hz: .6f}  # "
                f"RES={res} CHAIN={chain} RESSEQ={resseq} ATOM=H COORD={coord_cb} BONDEDATOM=CB"
            )

    return filtered_lines + new_lines, removed_existing, len(new_lines)


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

    # Step 1: remove H records without metadata comments.
    no_bare_h_atoms, removed_bare_h_count = _remove_bare_hydrogens(cleaned_atoms)

    # Step 2: remove sulfur-bound hydrogens.
    no_sulfur_h_atoms, removed_sulfur_h_count = _remove_sulfur_bonded_hydrogens(no_bare_h_atoms)

    # Step 3: remove likely orphan H atoms.
    no_loose_atoms, removed_loose_count = _remove_loose_hydrogens(no_sulfur_h_atoms)

    # Step 4: enforce exact CA/CB hydrogen counts on capped residues.
    final_atoms, removed_cap_h_count, added_count = _enforce_cap_ca_cb_hydrogens(no_loose_atoms)
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
        removed_bare_h_count=removed_bare_h_count,
        removed_loose_count=removed_loose_count,
        removed_sulfur_h_count=removed_sulfur_h_count,
        removed_cap_h_count=removed_cap_h_count,
        added_count=added_count,
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
    removed_bare_total = 0
    removed_loose_total = 0
    removed_sulfur_total = 0
    removed_cap_h_total = 0
    added_total = 0
    skipped = 0
    changed_paths: list[tuple[Path, Path, int, int, int, int, int, int]] = []

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
            removed_bare_total += result.removed_bare_h_count
            removed_loose_total += result.removed_loose_count
            removed_sulfur_total += result.removed_sulfur_h_count
            removed_cap_h_total += result.removed_cap_h_count
            added_total += result.added_count
            changed_paths.append(
                (
                    result.source,
                    result.output,
                    result.removed_count,
                    result.removed_bare_h_count,
                    result.removed_loose_count,
                    result.removed_sulfur_h_count,
                    result.removed_cap_h_count,
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
        f"cleaned {cleaned}, skipped {skipped}, removed {removed_total} post-Zn H line(s), "
        f"removed bare-commentless {removed_bare_total} H line(s), "
        f"removed S-bonded {removed_sulfur_total} H line(s), "
        f"removed loose {removed_loose_total} H line(s), "
        f"removed {removed_cap_h_total} existing capped CA/CB H line(s), "
        f"added {added_total} rebuilt capped CA/CB H line(s)."
    )

    if changed_paths:
        print("\nChanged files:")
        for src, out, removed, removed_bare, removed_loose, removed_sulfur, removed_cap, added in changed_paths:
            print(
                f"- {src} -> {out} "
                f"(removed {removed} post-Zn H; removed bare {removed_bare} H; "
                f"removed S-bonded {removed_sulfur} H; "
                f"removed loose {removed_loose} H; "
                f"removed capped CA/CB H {removed_cap}; added capped CA/CB H {added})"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
