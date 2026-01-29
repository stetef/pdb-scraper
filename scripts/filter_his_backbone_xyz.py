#!/usr/bin/env python3
"""Filter HIS/CYS backbone atoms (N, C, O) from XYZ files.

Removes lines where RES=HIS or RES=CYS and ATOM is exactly N, C, or O.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List


def parse_comment_fields(comment: str) -> dict[str, str]:
    """Parse key=value fields from the comment portion."""
    fields: dict[str, str] = {}
    for part in comment.split():
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key.strip()] = value.strip()
    return fields


def should_drop_line(line: str) -> bool:
    """Return True if line is HIS/CYS with ATOM=N/C/O (exact)."""
    if "#" not in line:
        return False
    _, comment = line.split("#", 1)
    fields = parse_comment_fields(comment)
    if fields.get("RES") not in {"HIS", "CYS"}:
        return False
    atom = fields.get("ATOM")
    return atom in {"N", "C", "O"}


def filter_xyz_lines(lines: List[str]) -> List[str]:
    """Filter XYZ lines and update atom count when applicable."""
    if not lines:
        return lines

    # Detect XYZ header with atom count on first line.
    header_count = None
    try:
        header_count = int(lines[0].strip())
    except ValueError:
        header_count = None

    if header_count is None or len(lines) < 2:
        return [line for line in lines if not should_drop_line(line)]

    header = lines[:2]
    body = lines[2:]
    filtered_body = [line for line in body if not should_drop_line(line)]
    header[0] = f"{len(filtered_body)}\n"
    return header + filtered_body


def iter_xyz_files(input_dir: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*.xyz" if recursive else "*.xyz"
    return input_dir.glob(pattern)


def process_file(input_path: Path, output_path: Path) -> None:
    lines = input_path.read_text().splitlines(keepends=True)
    filtered = filter_xyz_lines(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(filtered))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Filter HIS/CYS backbone atoms (ATOM=N/C/O) from XYZ files in a directory."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing .xyz files")
    parser.add_argument("output_dir", type=Path, help="Directory to write filtered .xyz files")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for .xyz files recursively",
    )

    args = parser.parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    for xyz_path in iter_xyz_files(input_dir, args.recursive):
        rel_path = xyz_path.relative_to(input_dir)
        out_path = output_dir / rel_path
        process_file(xyz_path, out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
