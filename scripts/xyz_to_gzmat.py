#!/usr/bin/env python3
"""Convert all XYZ files in a directory to Gaussian-style Z-matrix files.

For each ``.xyz`` file in the given directory, this script writes a matching
``.gzmat`` file using ``chemcoord``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# chemcoord 2.1.x can fail with pandas string inference enabled.
if hasattr(pd.options, "future") and hasattr(pd.options.future, "infer_string"):
    pd.options.future.infer_string = False

import chemcoord as cc


def _collect_xyz_files(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.xyz" if recursive else "*.xyz"
    return sorted(path for path in input_dir.glob(pattern) if path.is_file())


def _convert_xyz_to_gzmat(xyz_path: Path) -> Path:
    cart = cc.Cartesian.read_xyz(xyz_path)
    zmat = cart.get_zmat()
    output_path = xyz_path.with_suffix(".gzmat")
    zmat_text = zmat.to_zmat(implicit_index=False)
    output_path.write_text(f"{zmat_text}\n", encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert all .xyz files in a directory to .gzmat with chemcoord."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing .xyz files")
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Also process .xyz files in subdirectories",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        print(f"Error: not a directory: {input_dir}", file=sys.stderr)
        return 2

    xyz_files = _collect_xyz_files(input_dir, recursive=args.recursive)
    if not xyz_files:
        print(f"No .xyz files found in {input_dir}")
        return 0

    success_count = 0
    fail_count = 0
    for xyz_path in tqdm(xyz_files, desc="Converting XYZ", unit="file"):
        try:
            output_path = _convert_xyz_to_gzmat(xyz_path)
            # print(f"Converted: {xyz_path} -> {output_path}")
            success_count += 1
        except Exception as exc:
            print(f"Failed: {xyz_path}: {exc}", file=sys.stderr)
            fail_count += 1

    print(f"Done. Converted {success_count} file(s), failed {fail_count} file(s).")
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
