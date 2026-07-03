#!/usr/bin/env python3
"""Append CHARGE=-2 MULTIPLICITY=1 to line 2 (the header/comment line) of each XYZ file.

Usage:
  uv run python scripts/structure-clustering-analysis/add_charge_multiplicity.py <xyz_dir>

Default directory if omitted:
  data/test-4cys-weighted/validation/approach1/sampled-xyz-files-for-val
"""
import sys
from pathlib import Path

xyz_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "data/test-4cys-weighted/validation/approach1/sampled-xyz-files-for-val"
)

if not xyz_dir.is_dir():
    raise SystemExit(f"Directory not found: {xyz_dir}")

updated = 0
for p in sorted(xyz_dir.glob("*.xyz")):
    lines = p.read_text(encoding="utf-8").splitlines()
    if len(lines) >= 2:
        lines[1] = lines[1].rstrip() + " CHARGE=-2 MULTIPLICITY=1"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Updated {p.name}")
        updated += 1

print(f"\nDone. Updated {updated} file(s).")
