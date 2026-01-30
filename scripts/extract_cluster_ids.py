#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


def read_unique_pdb_ids(csv_path: Path) -> list[str]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "PDB" not in reader.fieldnames:
            raise ValueError(f"Missing 'PDB' column in {csv_path}")

        seen: set[str] = set()
        ordered: list[str] = []
        for row in reader:
            pdb_id = (row.get("PDB") or "").strip()
            if not pdb_id or pdb_id in seen:
                continue
            seen.add(pdb_id)
            ordered.append(pdb_id)

    return ordered


def write_ids(ids: list[str], output_path: Path) -> None:
    output_path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"

    if not data_dir.is_dir():
        raise SystemExit(f"data directory not found: {data_dir}")

    for subdir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        clusters_csv = subdir / "output" / "clusters_summary.csv"
        if not clusters_csv.is_file():
            continue

        ids = read_unique_pdb_ids(clusters_csv)
        output_path = repo_root / f"{subdir.name}-ids.txt"
        write_ids(ids, output_path)

        print(f"Wrote {len(ids)} IDs -> {output_path}")


if __name__ == "__main__":
    main()
