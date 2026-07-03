#!/usr/bin/env python3
"""Approach 3 — Permutation Invariant Vector (PIV).

Fully invariant to rotation, translation, and residue permutation by construction.
Independent of any reference structure.

Feature: 78-D vector of sorted pairwise distances grouped by atom-pair type.
Blocks (sorted ascending within each):
  Zn–S (4), Zn–Cβ (4), Zn–Cα (4)
  S–S (6), S–Cβ (16), S–Cα (16)
  Cβ–Cβ (6), Cβ–Cα (16), Cα–Cα (6)
Total: 78

Near-constant covalent-bond columns (S–Cβ, Cβ–Cα) are handled by the variance
floor in z-scoring; no explicit exclusion needed.

Outputs to --out-dir:
  labels.csv, medoids.csv, k_sweep.csv, per_cluster_intra.csv

Usage
-----
  uv run python scripts/structure-clustering-analysis/04_approach3_piv.py \\
      --xyz-dir data/test-4cys/initial_xyz_files \\
      --out-dir data/test-4cys/approach3 \\
      --k-min 2 --k-max 8
"""
from __future__ import annotations

import argparse
import logging
import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np
import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    EQUAL_WEIGHTS, SHELL_WEIGHTS, DISTANCE_WEIGHTS,
    Structure, parse_structure,
    sweep_k, save_outputs, build_cluster_distribution_plots,
)


# ---------------------------------------------------------------------------
# PIV featurization
# ---------------------------------------------------------------------------

def piv(s: Structure) -> np.ndarray:
    """Compute 78-D PIV for one structure."""
    zn, sv, cb, ca = s.zn, s.s, s.cb, s.ca
    n = 4

    def d(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    return np.concatenate([
        np.sort([d(zn, sv[i])  for i in range(n)]),                           # 4
        np.sort([d(zn, cb[i])  for i in range(n)]),                           # 4
        np.sort([d(zn, ca[i])  for i in range(n)]),                           # 4
        np.sort([d(sv[i], sv[j]) for i, j in combinations(range(n), 2)]),     # 6
        np.sort([d(sv[i], cb[j]) for i, j in product(range(n), repeat=2)]),   # 16
        np.sort([d(sv[i], ca[j]) for i, j in product(range(n), repeat=2)]),   # 16
        np.sort([d(cb[i], cb[j]) for i, j in combinations(range(n), 2)]),     # 6
        np.sort([d(cb[i], ca[j]) for i, j in product(range(n), repeat=2)]),   # 16
        np.sort([d(ca[i], ca[j]) for i, j in combinations(range(n), 2)]),     # 6
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    xyz_dir: Path,
    out_dir: Path,
    k_values: list[int],
    w_type: dict | str = EQUAL_WEIGHTS,
    allow_reflection: bool = True,
    stats_csv: Path | None = None,
) -> dict | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "approach3_run.log"
    logging.basicConfig(filename=str(log), level=logging.INFO,
                        format="%(asctime)s %(message)s", filemode="w")

    xyz_files = sorted(xyz_dir.glob("*.xyz"))
    print(f"Parsing {len(xyz_files)} XYZ files …")
    structures = [s for p in tqdm.tqdm(xyz_files, desc="parsing", leave=False)
                  if (s := parse_structure(p)) is not None]
    n_failed = len(xyz_files) - len(structures)
    print(f"Parsed {len(structures)}/{len(xyz_files)} ({n_failed} failed)")

    if len(structures) < 4:
        raise SystemExit("Too few structures to cluster.")

    print("Computing PIV features …")
    X = np.array([piv(s) for s in tqdm.tqdm(structures, desc="PIV", leave=False)])
    print(f"Feature matrix: {X.shape}")

    k_values = [k for k in k_values if 2 <= k < len(structures)]
    if not k_values:
        raise SystemExit("No valid k values for this dataset size.")

    best, table = sweep_k(structures, X, k_values, w_type, allow_reflection, desc="A3 k sweep")
    if best is None:
        print("No valid clustering results.")
        return None

    print(f"Best k={best['k']}  intra={best['intra']:.4f}  "
          f"inter={best['inter']:.4f}  ratio={best['ratio']:.4f}  ch_score={best['ch_score']:.4f}")

    # Save standard outputs (includes embeddings.csv + tsne_kmeans.png)
    save_outputs(out_dir, [s.id for s in structures], table, best)
    # Distribution plots (optional; requires --stats-csv)
    build_cluster_distribution_plots(
        out_dir, [s.id for s in structures], best["labels"], stats_csv
    )
    logging.info("Done. Best k=%d ratio=%.4f ch_score=%.4f", best["k"], best["ratio"], best["ch_score"])
    return best


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Approach 3: PIV (sorted pairwise distances) clustering."
    )
    parser.add_argument("--xyz-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--k-min", type=int, default=10)
    parser.add_argument("--k-max", type=int, default=40)
    parser.add_argument("--k-step", type=int, default=2)
    parser.add_argument("--no-reflection", action="store_true")
    parser.add_argument("--stats-csv", type=Path, default=None,
                        help="Per-structure metadata CSV (id column required) for "
                             "distribution histograms and stats summary.")
    parser.add_argument("--weight-scheme", choices=["equal", "shell", "distance"],
                        default="distance",
                        help="RMSD atom weighting: equal, shell (S=1, Cb/Ca=0.5), "
                             "or distance (weight = 1/avg_Zn_distance per atom pair; default).")
    args = parser.parse_args()

    xyz_dir = args.xyz_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not xyz_dir.is_dir():
        raise SystemExit(f"--xyz-dir not found: {xyz_dir}")

    _w_map = {"equal": EQUAL_WEIGHTS, "shell": SHELL_WEIGHTS, "distance": DISTANCE_WEIGHTS}
    stats_csv = args.stats_csv.expanduser().resolve() if args.stats_csv else None
    k_values = list(range(args.k_min, args.k_max + 1, args.k_step))
    run(xyz_dir, out_dir, k_values,
        w_type=_w_map[args.weight_scheme],
        allow_reflection=not args.no_reflection, stats_csv=stats_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
