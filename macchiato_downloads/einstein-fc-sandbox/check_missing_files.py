#!/usr/bin/env python3
"""Report which subdirs of param-downloads are missing expected files."""

from __future__ import annotations

import argparse
from pathlib import Path

EXACT_FILES = ["feff.inp", "list.dat", "paths.dat", "xmu.dat"]
GLOB_FILES = ["*.hess", "*.xyz"]


def missing_files(subdir: Path) -> list[str]:
    missing = []
    for name in EXACT_FILES:
        if not (subdir / name).is_file():
            missing.append(name)
    for pattern in GLOB_FILES:
        if not any(subdir.glob(pattern)):
            missing.append(pattern)
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=Path(__file__).parent / "param-downloads",
        type=Path,
        help="Directory containing the subdirs to check.",
    )
    args = parser.parse_args()

    root: Path = args.root
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    incomplete = [(d, missing_files(d)) for d in subdirs]
    incomplete = [(d, m) for d, m in incomplete if m]

    print(f"Checked {len(subdirs)} subdir(s) in {root}")
    print(f"Expecting: {', '.join(EXACT_FILES + GLOB_FILES)}\n")

    if not incomplete:
        print("All subdirs have every expected file.")
        return

    print(f"{len(incomplete)} subdir(s) missing files:\n")
    width = max(len(d.name) for d, _ in incomplete)
    for d, miss in incomplete:
        print(f"  {d.name:<{width}}  missing: {', '.join(miss)}")


if __name__ == "__main__":
    main()
