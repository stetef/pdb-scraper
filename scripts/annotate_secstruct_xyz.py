#!/usr/bin/env python3
"""Annotate XYZ CA atoms with HELIX/SHEET/LOOP secondary structure from a PDB."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class _SecStructRange:
    chain: str
    start: int
    end: int


def _parse_int(value: str) -> int | None:
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        return None


def parse_pdb_secondary_structure(pdb_path: Path) -> Dict[str, List[_SecStructRange]]:
    helix_ranges: List[_SecStructRange] = []
    sheet_ranges: List[_SecStructRange] = []

    with pdb_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("HELIX"):
                # PDB v3.x fixed columns
                start_resseq = _parse_int(line[21:25])
                end_resseq = _parse_int(line[33:37])
                chain = line[19:20].strip()
                if start_resseq is None or end_resseq is None or not chain:
                    continue
                start = min(start_resseq, end_resseq)
                end = max(start_resseq, end_resseq)
                helix_ranges.append(_SecStructRange(chain=chain, start=start, end=end))
            elif line.startswith("SHEET"):
                start_resseq = _parse_int(line[22:26])
                end_resseq = _parse_int(line[33:37])
                chain = line[21:22].strip()
                if start_resseq is None or end_resseq is None or not chain:
                    continue
                start = min(start_resseq, end_resseq)
                end = max(start_resseq, end_resseq)
                sheet_ranges.append(_SecStructRange(chain=chain, start=start, end=end))

    return {"HELIX": helix_ranges, "SHEET": sheet_ranges}


def determine_secondary_structure(
    secstruct: Dict[str, List[_SecStructRange]],
    chain: str,
    resseq: int,
) -> str:
    for entry in secstruct.get("HELIX", []):
        if entry.chain == chain and entry.start <= resseq <= entry.end:
            return "HELIX"
    for entry in secstruct.get("SHEET", []):
        if entry.chain == chain and entry.start <= resseq <= entry.end:
            return "SHEET"
    return "LOOP"


def _parse_meta_tokens(comment: str) -> List[str]:
    return [token for token in comment.strip().split() if token]


def _meta_dict(tokens: List[str]) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key and value:
            meta[key] = value
    return meta


def _update_comment_with_secstruct(comment: str, secstruct: str) -> str:
    tokens = _parse_meta_tokens(comment)
    tokens = [t for t in tokens if not t.startswith("SEC=") and not t.startswith("SECSTRUCT=")]
    tokens.append(f"SEC={secstruct}")
    return " ".join(tokens).strip()


def _resolve_pdb_for_xyz(xyz_path: Path) -> Path:
    pdb_id = xyz_path.stem.split("_", 1)[0]
    base_dir = xyz_path.parents[2]
    return base_dir / "results" / "validated_structures" / f"{pdb_id}.pdb"


def annotate_xyz_with_secstruct(xyz_path: Path) -> bool:
    if not xyz_path.exists():
        raise SystemExit(f"XYZ file not found: {xyz_path}")

    pdb_path = _resolve_pdb_for_xyz(xyz_path)
    if not pdb_path.exists():
        print(f"Warning: PDB not found for {xyz_path.name}: {pdb_path}", file=sys.stderr)
        return False

    secstruct = parse_pdb_secondary_structure(pdb_path)
    lines = xyz_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise SystemExit(f"XYZ file too short: {xyz_path}")

    header = lines[:2]
    body = lines[2:]
    updated_body: List[str] = []

    for line in body:
        stripped = line.strip()
        if not stripped:
            updated_body.append(line)
            continue

        if "#" in stripped:
            left, right = stripped.split("#", 1)
            comment = right.strip()
        else:
            left, comment = stripped, ""

        parts = left.split()
        if len(parts) < 4:
            updated_body.append(line)
            continue

        meta = _meta_dict(_parse_meta_tokens(comment))
        if meta.get("ATOM") == "CA" and "RESSEQ" in meta and "CHAIN" in meta:
            resseq = _parse_int(meta.get("RESSEQ", ""))
            chain = meta.get("CHAIN", "").strip()
            if resseq is not None and chain:
                sec = determine_secondary_structure(secstruct, chain, resseq)
                comment = _update_comment_with_secstruct(comment, sec)

        if comment:
            updated_body.append(f"{left.strip()}  # {comment}")
        else:
            updated_body.append(left.strip())

    xyz_path.write_text("\n".join(header + updated_body) + "\n", encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate XYZ CA atoms with HELIX/SHEET/LOOP metadata from a PDB file."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to an input .xyz file or a directory containing .xyz files",
    )
    args = parser.parse_args()

    target = args.path
    if target.is_dir():
        xyz_files = sorted(target.glob("*.xyz"))
        if not xyz_files:
            raise SystemExit(f"No .xyz files found in {target}")
        failures = 0
        for xyz_path in xyz_files:
            if not annotate_xyz_with_secstruct(xyz_path):
                failures += 1
        if failures:
            raise SystemExit(1)
    else:
        ok = annotate_xyz_with_secstruct(target)
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
