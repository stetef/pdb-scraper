#!/usr/bin/env python3
"""Pipeline orchestrator for Zn(Cys)₄ structure clustering.

Chains the five pipeline scripts in order, skipping steps whose key output
already exists (idempotent).  Pass --force to re-run everything.

Pipeline
--------
  01  cleanup      H cleanup on raw XYZ files
  02  approach1    Aligned Cartesian featurization + clustering (Approach 1)
  03  approach2    Z-matrix featurization + clustering (Approach 2; optional)
  04  approach3    PIV featurization + clustering (Approach 3)
  05  validate     RMSD metrics, plots, PCA→XYZ, cross-approach comparison

Approach 2 uses pure numpy (no chemcoord dependency).

Usage
-----
  uv run python scripts/structure-clustering-analysis/00_orchestrate.py \\
      --base-dir data/large-cys-his-datasets/4cys-large \\
      [--approaches 1 3]      # which approaches to run (default: 1 3)
      [--k-min 10 --k-max 40] # k sweep range
      [--force]               # re-run all steps

  # Test on small dataset:
  uv run python scripts/structure-clustering-analysis/00_orchestrate.py \\
      --base-dir data/test-4cys \\
      --xyz-subdir initial_xyz_files \\
      --approaches 1 3 \\
      --k-min 2 --k-max 8

Directory layout under --base-dir
----------------------------------
  {xyz_subdir}/          raw input XYZ files  (default: output/xyz_files)
  cleaned/               01 cleanup output
  approach1/             02 output (labels, medoids, k_sweep, aligned_xyz/)
  approach2/             03 output (optional)
  approach3/             04 output
  validation/approach1/  05 output
  validation/approach3/  05 output
  validation/comparison/ 05 comparison output
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).parent
_PYTHON  = sys.executable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(
    label: str,
    cmd: list[Path | str],
    *,
    skip_if: Path | None = None,
    force: bool = False,
) -> bool:
    """Run cmd.  Skip if skip_if exists and not force.  Return True if ran."""
    if not force and skip_if is not None:
        if skip_if.is_file() or (skip_if.is_dir() and any(skip_if.iterdir())):
            print(f"  skip  {label}")
            return False
    print(f"  run   {label}")
    subprocess.run([str(c) for c in cmd], check=True)
    return True


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_cleanup(base: Path, xyz_in: Path, force: bool) -> Path:
    out = base / "cleaned"
    _run(
        "01 cleanup_xyz_hydrogens",
        [_PYTHON, _SCRIPTS / "01_cleanup_xyz_hydrogens.py",
         xyz_in, "--preview-dir", out],
        skip_if=out,
        force=force,
    )
    return out


def step_compute_stats(
    xyz_dir: Path,
    pdb_dir: Path | None,
    force: bool,
) -> Path:
    out_csv = xyz_dir / "structure_stats.csv"
    _run(
        "00a compute_structure_stats",
        [_PYTHON, _SCRIPTS / "00a_compute_structure_stats.py",
         "--xyz-dir", xyz_dir]
        + (["--pdb-dir", pdb_dir] if pdb_dir is not None else [])
        + (["--force"] if force else []),
        skip_if=out_csv,
        force=force,
    )
    return out_csv


def step_approach1(base: Path, xyz_dir: Path, k_min: int, k_max: int, k_step: int,
                   force: bool, stats_csv: Path | None = None,
                   weight_scheme: str = "distance") -> Path:
    out = base / "approach1"
    cmd = [_PYTHON, _SCRIPTS / "02_approach1_cartesian.py",
           "--xyz-dir", xyz_dir,
           "--out-dir", out,
           "--k-min", str(k_min), "--k-max", str(k_max), "--k-step", str(k_step),
           "--weight-scheme", weight_scheme]
    if stats_csv is not None:
        cmd += ["--stats-csv", stats_csv]
    _run("02 approach1_cartesian", cmd, skip_if=out / "labels.csv", force=force)
    return out


def step_approach2(base: Path, aligned_dir: Path, k_min: int, k_max: int, k_step: int,
                   force: bool, stats_csv: Path | None = None,
                   weight_scheme: str = "distance") -> Path:
    """Run Approach 2 (pure numpy; no chemcoord dependency)."""
    out = base / "approach2"
    cmd = [_PYTHON, _SCRIPTS / "03_approach2_zmatrix.py",
           "--aligned-xyz-dir", aligned_dir,
           "--out-dir", out,
           "--k-min", str(k_min), "--k-max", str(k_max), "--k-step", str(k_step),
           "--weight-scheme", weight_scheme]
    if stats_csv is not None:
        cmd += ["--stats-csv", stats_csv]
    _run("03 approach2_zmatrix", cmd, skip_if=out / "labels.csv", force=force)
    return out


def step_approach3(base: Path, xyz_dir: Path, k_min: int, k_max: int, k_step: int,
                   force: bool, stats_csv: Path | None = None,
                   weight_scheme: str = "distance") -> Path:
    out = base / "approach3"
    cmd = [_PYTHON, _SCRIPTS / "04_approach3_piv.py",
           "--xyz-dir", xyz_dir,
           "--out-dir", out,
           "--k-min", str(k_min), "--k-max", str(k_max), "--k-step", str(k_step),
           "--weight-scheme", weight_scheme]
    if stats_csv is not None:
        cmd += ["--stats-csv", stats_csv]
    _run("04 approach3_piv", cmd, skip_if=out / "labels.csv", force=force)
    return out


def step_validate(
    base: Path,
    approach_dir: Path,
    xyz_dir: Path,
    is_approach1: bool,
    force: bool,
    weight_scheme: str = "distance",
    sampled_val_dir: Path | None = None,
) -> Path:
    out = base / "validation" / approach_dir.name
    cmd = [_PYTHON, _SCRIPTS / "05_validate_clusters.py",
           "--approach-dir", approach_dir,
           "--xyz-dir", xyz_dir,
           "--out-dir", out,
           "--weight-scheme", weight_scheme]
    if is_approach1:
        cmd.append("--approach1")
    if sampled_val_dir is not None and sampled_val_dir.is_dir():
        cmd += ["--sampled-val-dir", sampled_val_dir]
    _run(
        f"05 validate  ({approach_dir.name})",
        cmd,
        skip_if=out / "k_sweep_plot.png",
        force=force,
    )
    return out


def step_compare(base: Path, approach_dirs: list[Path], force: bool) -> None:
    out = base / "validation" / "comparison"
    _run(
        "05 validate --compare-dirs",
        [_PYTHON, _SCRIPTS / "05_validate_clusters.py",
         "--compare-dirs"] + [str(d) for d in approach_dirs] + ["--out-dir", out],
        skip_if=out / "comparison_table.csv",
        force=force,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Orchestrate the Zn(Cys)₄ structure clustering pipeline."
    )
    parser.add_argument("--base-dir", type=Path, required=True,
                        help="Base directory for all outputs.")
    parser.add_argument("--xyz-subdir", type=str, default="output/xyz_files",
                        help="Subdirectory under --base-dir containing raw XYZ files "
                             "(default: output/xyz_files).")
    parser.add_argument("--approaches", type=int, nargs="+", default=[1, 3],
                        choices=[1, 2, 3],
                        help="Which approaches to run (default: 1 3).")
    parser.add_argument("--k-min",  type=int, default=10)
    parser.add_argument("--k-max",  type=int, default=40)
    parser.add_argument("--k-step", type=int, default=2)
    parser.add_argument("--skip-cleanup", action="store_true",
                        help="Skip step 01 and use --xyz-subdir as-is.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all steps even if outputs exist.")
    parser.add_argument("--weight-scheme", choices=["equal", "shell", "distance"],
                        default="distance",
                        help="RMSD atom weighting for clustering evaluation: equal (default), "
                             "shell (S=1, Cb/Ca=0.5), or distance (weight = 1/avg_Zn_distance).")
    parser.add_argument("--pdb-dir", type=Path, default=None,
                        help="Directory of <pdbid>.pdb files for B-factor and R-factor extraction. "
                             "When provided, structure_stats.csv is generated automatically.")
    parser.add_argument("--stats-csv", type=Path, default=None,
                        help="Explicit per-structure metadata CSV.  If omitted and --pdb-dir is "
                             "given, the auto-generated <xyz-dir>/structure_stats.csv is used.")
    parser.add_argument("--sampled-val-dir", type=Path, default=None,
                        help="Directory of sampled XYZ files for validation overlays "
                             "(passed to 05_validate_clusters.py --sampled-val-dir).")
    args = parser.parse_args()

    base = args.base_dir.expanduser().resolve()
    xyz_raw = base / args.xyz_subdir
    if not xyz_raw.is_dir():
        raise SystemExit(f"XYZ input directory not found: {xyz_raw}")

    k_min, k_max, k_step = args.k_min, args.k_max, args.k_step
    approaches = sorted(set(args.approaches))
    weight_scheme = args.weight_scheme
    pdb_dir         = args.pdb_dir.expanduser().resolve() if args.pdb_dir else None
    stats_csv       = args.stats_csv.expanduser().resolve() if args.stats_csv else None
    sampled_val_dir = args.sampled_val_dir.expanduser().resolve() if args.sampled_val_dir else None

    print(f"\n=== Pipeline: base={base} approaches={approaches} k={k_min}..{k_max} ===\n")

    # Step 01: cleanup
    if args.skip_cleanup:
        xyz_dir = xyz_raw
        print(f"  skip  01 cleanup (using {xyz_dir})")
    else:
        xyz_dir = step_cleanup(base, xyz_raw, args.force)

    # Step 00a: structure stats (auto-generated when --pdb-dir given)
    if stats_csv is None and pdb_dir is not None:
        stats_csv = step_compute_stats(xyz_dir, pdb_dir, args.force)
    elif pdb_dir is None and stats_csv is None:
        print("  skip  00a compute_structure_stats (no --pdb-dir or --stats-csv provided)")

    # Track completed approach output dirs for comparison step
    completed: list[Path] = []

    # Step 02: Approach 1
    if 1 in approaches:
        a1_dir = step_approach1(base, xyz_dir, k_min, k_max, k_step, args.force,
                                stats_csv=stats_csv, weight_scheme=weight_scheme)
        completed.append(a1_dir)
        aligned_dir = a1_dir / "aligned_xyz"
    else:
        a1_dir = base / "approach1"
        aligned_dir = a1_dir / "aligned_xyz"

    # Step 03: Approach 2 (uses aligned structures from approach 1)
    if 2 in approaches:
        if not aligned_dir.is_dir():
            print(f"  skip  03 approach2 (aligned_xyz not found; run approach 1 first)")
        else:
            a2_dir = step_approach2(base, aligned_dir, k_min, k_max, k_step, args.force,
                                    stats_csv=stats_csv, weight_scheme=weight_scheme)
            if a2_dir is not None:
                completed.append(a2_dir)

    # Step 04: Approach 3
    if 3 in approaches:
        a3_dir = step_approach3(base, xyz_dir, k_min, k_max, k_step, args.force,
                                stats_csv=stats_csv, weight_scheme=weight_scheme)
        completed.append(a3_dir)

    # Step 05: Validate each approach
    for app_dir in completed:
        is_a1 = (app_dir.name == "approach1") and (app_dir / "aligned_xyz").is_dir()
        val_xyz = app_dir / "aligned_xyz" if is_a1 else xyz_dir
        # Only pass sampled_val_dir for approach 1 (it lives under validation/approach1/)
        svd = sampled_val_dir if (is_a1 and sampled_val_dir is not None) else None
        step_validate(base, app_dir, val_xyz, is_approach1=is_a1, force=args.force,
                      weight_scheme=weight_scheme, sampled_val_dir=svd)

    # Step 05: Cross-approach comparison
    if len(completed) > 1:
        step_compare(base, completed, args.force)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
