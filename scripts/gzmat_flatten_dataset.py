#!/usr/bin/env python3
"""Build numeric datasets from Gaussian-style Z-matrix (.gzmat) files.

This script collects .gzmat files, converts all tokens to numeric values,
flattens each structure into a one-row vector, and writes CSV tables suitable
for downstream clustering workflows.

Outputs:
- <prefix>.raw_flat.csv: direct flattened numeric tokens per structure
- <prefix>.cluster_features.csv: engineered geometry features
- <prefix>.cluster_features_zscore.csv: z-score normalized cluster features
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


ELEMENT_TO_Z: dict[str, int] = {
    "H": 1,
    "HE": 2,
    "LI": 3,
    "BE": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "NE": 10,
    "NA": 11,
    "MG": 12,
    "AL": 13,
    "SI": 14,
    "P": 15,
    "S": 16,
    "CL": 17,
    "AR": 18,
    "K": 19,
    "CA": 20,
    "SC": 21,
    "TI": 22,
    "V": 23,
    "CR": 24,
    "MN": 25,
    "FE": 26,
    "CO": 27,
    "NI": 28,
    "CU": 29,
    "ZN": 30,
    "GA": 31,
    "GE": 32,
    "AS": 33,
    "SE": 34,
    "BR": 35,
    "KR": 36,
    "RB": 37,
    "SR": 38,
    "Y": 39,
    "ZR": 40,
    "NB": 41,
    "MO": 42,
    "TC": 43,
    "RU": 44,
    "RH": 45,
    "PD": 46,
    "AG": 47,
    "CD": 48,
    "IN": 49,
    "SN": 50,
    "SB": 51,
    "TE": 52,
    "I": 53,
    "XE": 54,
    "CS": 55,
    "BA": 56,
    "LA": 57,
    "CE": 58,
    "PR": 59,
    "ND": 60,
    "PM": 61,
    "SM": 62,
    "EU": 63,
    "GD": 64,
    "TB": 65,
    "DY": 66,
    "HO": 67,
    "ER": 68,
    "TM": 69,
    "YB": 70,
    "LU": 71,
    "HF": 72,
    "TA": 73,
    "W": 74,
    "RE": 75,
    "OS": 76,
    "IR": 77,
    "PT": 78,
    "AU": 79,
    "HG": 80,
    "TL": 81,
    "PB": 82,
    "BI": 83,
    "PO": 84,
    "AT": 85,
    "RN": 86,
    "FR": 87,
    "RA": 88,
    "AC": 89,
    "TH": 90,
    "PA": 91,
    "U": 92,
    "NP": 93,
    "PU": 94,
    "AM": 95,
    "CM": 96,
    "BK": 97,
    "CF": 98,
    "ES": 99,
    "FM": 100,
    "MD": 101,
    "NO": 102,
    "LR": 103,
    "RF": 104,
    "DB": 105,
    "SG": 106,
    "BH": 107,
    "HS": 108,
    "MT": 109,
    "DS": 110,
    "RG": 111,
    "CN": 112,
    "NH": 113,
    "FL": 114,
    "MC": 115,
    "LV": 116,
    "TS": 117,
    "OG": 118,
}


FRAME_TOKEN_TO_NUM: dict[str, float] = {
    "origin": 0.0,
    "e_z": -1.0,
    "e_x": -2.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Flatten .gzmat files into one-row numeric vectors and write clustering-ready tables."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing .gzmat files")
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Also process .gzmat files in subdirectories",
    )
    parser.add_argument(
        "--pattern",
        default="*.gzmat",
        help="Glob pattern for input files (default: *.gzmat)",
    )
    parser.add_argument(
        "--include-extended",
        action="store_true",
        help="Include files whose stem contains '-extended' (default: excluded).",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help=(
            "Output file prefix path. Default: <input_dir>/gzmat_dataset "
            "(files will be <prefix>.<suffix>.csv)."
        ),
    )
    parser.add_argument(
        "--strict-length",
        action="store_true",
        help=(
            "Require equal vector lengths across all files. "
            "Default behavior pads ragged rows with NaN."
        ),
    )
    return parser.parse_args()


def collect_gzmat_files(input_dir: Path, recursive: bool, pattern: str) -> list[Path]:
    if recursive:
        paths = sorted(path for path in input_dir.rglob(pattern) if path.is_file())
    else:
        paths = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    return paths


def _token_to_numeric(token: str) -> float:
    lower = token.lower()
    if lower in FRAME_TOKEN_TO_NUM:
        return FRAME_TOKEN_TO_NUM[lower]

    symbol_key = token.upper()
    if symbol_key in ELEMENT_TO_Z:
        return float(ELEMENT_TO_Z[symbol_key])

    try:
        return float(token)
    except ValueError as exc:
        raise ValueError(f"Unrecognized token '{token}'") from exc


def parse_gzmat_tokens(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(line.split())
    return rows


def flatten_raw_numeric(rows: list[list[str]]) -> list[float]:
    flat: list[float] = []
    for row in rows:
        for tok in row:
            flat.append(_token_to_numeric(tok))
    return flat


def build_cluster_features(rows: list[list[str]]) -> list[float]:
    """Create angle-aware geometry features.

    For each expected 8-column row:
      [idx, elem, ref1, dist, ref2, angle_deg, ref3, dihedral_deg]

    We keep: atomic number, distance, sin/cos(angle), sin/cos(dihedral).
    This avoids angle wrap discontinuities near +/-180.
    """

    features: list[float] = []
    for row in rows:
        if len(row) != 8:
            raise ValueError(
                "Expected 8 columns per .gzmat row for clustering features, "
                f"got {len(row)}: {' '.join(row)}"
            )

        atom_z = _token_to_numeric(row[1])
        dist = _token_to_numeric(row[3])
        angle_deg = _token_to_numeric(row[5])
        dih_deg = _token_to_numeric(row[7])

        angle_rad = math.radians(angle_deg)
        dih_rad = math.radians(dih_deg)

        features.extend(
            [
                atom_z,
                dist,
                math.sin(angle_rad),
                math.cos(angle_rad),
                math.sin(dih_rad),
                math.cos(dih_rad),
            ]
        )

    return features


def _pad_rows(rows: list[list[float]], fill: float = math.nan) -> list[list[float]]:
    max_len = max(len(row) for row in rows)
    return [row + [fill] * (max_len - len(row)) for row in rows]


def _ensure_equal_lengths(vectors: list[list[float]], label: str, allow_ragged: bool) -> list[list[float]]:
    lengths = sorted({len(vec) for vec in vectors})
    if len(lengths) == 1:
        return vectors
    if allow_ragged:
        return _pad_rows(vectors)
    raise ValueError(
        f"{label} vectors have different lengths: {lengths}. "
        "Either remove --strict-length (default pads with NaN), or filter inputs further."
    )


def _zscore_table(table: list[list[float]]) -> list[list[float]]:
    arr = np.asarray(table, dtype=float)
    col_mean = np.nanmean(arr, axis=0)
    col_std = np.nanstd(arr, axis=0)
    col_std[col_std == 0.0] = 1.0

    z = (arr - col_mean) / col_std
    # For padded ragged columns (NaN), set to 0 after scaling.
    z = np.nan_to_num(z, nan=0.0)
    return z.tolist()


def _write_table(path: Path, ids: list[str], data: list[list[float]], col_prefix: str) -> None:
    if not data:
        raise ValueError("Cannot write empty table")

    width = len(data[0])
    header = ["id"] + [f"{col_prefix}{i}" for i in range(width)]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for sid, row in zip(ids, data):
            writer.writerow([sid, *row])


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Error: not a directory: {input_dir}")

    gzmat_files = collect_gzmat_files(input_dir, args.recursive, args.pattern)
    if not args.include_extended:
        gzmat_files = [p for p in gzmat_files if "-extended" not in p.stem]
    if not gzmat_files:
        raise SystemExit(f"No .gzmat files found in {input_dir}")

    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = input_dir / "gzmat_dataset"
    output_prefix = output_prefix.expanduser().resolve()

    ids: list[str] = []
    raw_vectors: list[list[float]] = []
    cluster_vectors: list[list[float]] = []

    for path in gzmat_files:
        rows = parse_gzmat_tokens(path)
        if not rows:
            continue

        sid = path.stem
        ids.append(sid)
        raw_vectors.append(flatten_raw_numeric(rows))
        cluster_vectors.append(build_cluster_features(rows))

    if not ids:
        raise SystemExit("No non-empty .gzmat files parsed.")

    allow_ragged = not args.strict_length
    raw_vectors = _ensure_equal_lengths(raw_vectors, "Raw flattened", allow_ragged)
    cluster_vectors = _ensure_equal_lengths(cluster_vectors, "Cluster feature", allow_ragged)
    cluster_z = _zscore_table(cluster_vectors)

    raw_path = output_prefix.with_suffix(".raw_flat.csv")
    cluster_path = output_prefix.with_suffix(".cluster_features.csv")
    cluster_z_path = output_prefix.with_suffix(".cluster_features_zscore.csv")

    _write_table(raw_path, ids, raw_vectors, col_prefix="raw_")
    _write_table(cluster_path, ids, cluster_vectors, col_prefix="feat_")
    _write_table(cluster_z_path, ids, cluster_z, col_prefix="zfeat_")

    print(f"Parsed {len(ids)} .gzmat files")
    print(f"Wrote raw flattened table: {raw_path}")
    print(f"Wrote cluster feature table: {cluster_path}")
    print(f"Wrote z-score feature table: {cluster_z_path}")
    print(
        "Feature encoding: atom_Z, bond_distance, sin(angle), cos(angle), sin(dihedral), cos(dihedral) per row."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
