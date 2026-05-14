#!/usr/bin/env python3
"""Summarize origin distances for selected atoms from XYZ files.

Given a parent directory, this script inspects each immediate subdirectory at:
  <subdir>/output/xyz_files

It reads all .xyz files in that directory whose names do not contain "extended"
(case-insensitive). For each file it:
  - keeps SG and CA atoms directly
  - groups ND1/NE2 by RESSEQ and keeps only the closer one to the origin

It then aggregates distances across all files/subdirectories and prints count,
mean, and standard deviation for SG, ND1, NE2, and CA, plus CA split by the
coordinating atom type for that RESSEQ (SG vs ND1 vs NE2).
"""

from __future__ import annotations

import argparse
import importlib
import math
import statistics
from collections import defaultdict
from pathlib import Path


TARGET_ATOMS = {"SG", "ND1", "NE2", "CA"}

CONFIG_ORDER = ("4cys", "3cys1his", "2cys2his", "1cys3his", "4his")

HISTOGRAM_KEYS = (
    "SG",
    "ND1",
    "NE2",
    "CA",
    "CA_for_SG",
    "CA_for_ND1",
    "CA_for_NE2",
)

HISTOGRAM_TITLES = {
    "SG": "S$_\gamma$ (CYS)",
    "ND1": "N$_\delta$ (HIS)",
    "NE2": "N$_\epsilon$ (HIS)",
    "CA": "CA distance from origin",
    "CA_for_SG": "CA distance (coordinator: SG)",
    "CA_for_ND1": "CA distance (coordinator: ND1)",
    "CA_for_NE2": "CA distance (coordinator: NE2)",
}

HISTOGRAM_COLORS = {
    "SG": "#B8860B",  # dark yellow
    "ND1": "#A8D8F0",  # pastel blue
    "NE2": "#A8D8F0",  # pastel blue
    "CA": "#BCC6D0",
    "CA_for_SG": "#B8860B",  # match SG family
    "CA_for_ND1": "#A8D8F0",  # match ND1 family
    "CA_for_NE2": "#A8D8F0",  # match NE2 family
}


def _parse_meta_comment(comment: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    for token in comment.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key and value:
            meta[key] = value
    return meta


def _distance_from_origin(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


def _collect_distances_from_xyz(xyz_path: Path) -> dict[str, list[float]]:
    distances: dict[str, list[float]] = {
        "SG": [],
        "ND1": [],
        "NE2": [],
        "CA": [],
        "CA_for_SG": [],
        "CA_for_ND1": [],
        "CA_for_NE2": [],
    }
    atoms_by_resseq: dict[str, list[tuple[str, float]]] = defaultdict(list)

    lines = xyz_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        return distances

    for line in lines[2:]:
        stripped = line.strip()
        if not stripped:
            continue

        if "#" in stripped:
            left, right = stripped.split("#", 1)
            comment = right.strip()
        else:
            left, comment = stripped, ""

        parts = left.split()
        if len(parts) < 4:
            continue

        try:
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
        except ValueError:
            continue

        meta = _parse_meta_comment(comment) if comment else {}
        atom_name = meta.get("ATOM", "").strip().upper()
        if atom_name not in TARGET_ATOMS:
            continue

        distance = _distance_from_origin(x, y, z)

        if atom_name in {"SG", "CA"}:
            distances[atom_name].append(distance)
        resseq = meta.get("RESSEQ", "").strip()
        if not resseq:
            continue

        atoms_by_resseq[resseq].append((atom_name, distance))

    for grouped_atoms in atoms_by_resseq.values():
        coordinators = [item for item in grouped_atoms if item[0] in {"SG", "ND1", "NE2"}]
        if not coordinators:
            continue

        # Choose one coordinator per RESSEQ by closest distance.
        # Tie-breaker preference is SG, then ND1, then NE2.
        chosen_atom, chosen_distance = min(
            coordinators,
            key=lambda item: (item[1], {"SG": 0, "ND1": 1, "NE2": 2}[item[0]]),
        )
        distances[chosen_atom].append(chosen_distance)

        for atom_name, ca_distance in grouped_atoms:
            if atom_name != "CA":
                continue
            if chosen_atom == "SG":
                distances["CA_for_SG"].append(ca_distance)
            elif chosen_atom == "ND1":
                distances["CA_for_ND1"].append(ca_distance)
            elif chosen_atom == "NE2":
                distances["CA_for_NE2"].append(ca_distance)

    return distances


def _config_label(cys_count: int, his_count: int) -> str:
    parts: list[str] = []
    if cys_count:
        parts.append(f"{cys_count}cys")
    if his_count:
        parts.append(f"{his_count}his")
    return "".join(parts) if parts else "none"


def _xyz_config_label(xyz_path: Path) -> str | None:
    cys_resseqs: set[str] = set()
    his_resseqs: set[str] = set()

    lines = xyz_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        return None

    for line in lines[2:]:
        stripped = line.strip()
        if not stripped or "#" not in stripped:
            continue
        _, comment = stripped.split("#", 1)
        meta = _parse_meta_comment(comment)
        res = meta.get("RES", "").strip().upper()
        resseq = meta.get("RESSEQ", "").strip()
        if not resseq:
            continue
        if res == "CYS":
            cys_resseqs.add(resseq)
        elif res == "HIS":
            his_resseqs.add(resseq)

    if not cys_resseqs and not his_resseqs:
        return None
    return _config_label(len(cys_resseqs), len(his_resseqs))


def _count_files_per_config(parent_dir: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"pdb": 0, "xyz": 0})

    for subdir in sorted(p for p in parent_dir.iterdir() if p.is_dir()):
        xyz_dir = subdir / "output" / "xyz_files"
        pdb_dir = subdir / "results" / "validated_structures"
        if not xyz_dir.is_dir():
            continue

        xyz_files = sorted(
            path
            for path in xyz_dir.glob("*.xyz")
            if "extended" not in path.name.lower()
        )
        if not xyz_files:
            continue

        subdir_label: str | None = None
        for xyz_file in xyz_files:
            label = _xyz_config_label(xyz_file)
            if label is None:
                continue
            counts[label]["xyz"] += 1
            if subdir_label is None:
                subdir_label = label

        if subdir_label and pdb_dir.is_dir():
            counts[subdir_label]["pdb"] += sum(1 for _ in pdb_dir.glob("*.pdb"))

    return counts


def _write_config_bar_chart(
    counts: dict[str, dict[str, int]],
    output_dir: Path,
) -> None:
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Bar chart output requested, but matplotlib is not installed. "
            "Install it or run without --histogram-dir."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_vals = [counts.get(cfg, {}).get("pdb", 0) for cfg in CONFIG_ORDER]
    xyz_vals = [counts.get(cfg, {}).get("xyz", 0) for cfg in CONFIG_ORDER]

    width = 0.4
    positions = list(range(len(CONFIG_ORDER)))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_pdb = ax.bar(
        [p - width / 2 for p in positions],
        pdb_vals,
        width,
        label="PDB",
        color="#FFCBA4",  # peach
        edgecolor="white",
        linewidth=0.8,
    )
    bars_xyz = ax.bar(
        [p + width / 2 for p in positions],
        xyz_vals,
        width,
        label="XYZ",
        color="#800000",  # maroon
        edgecolor="white",
        linewidth=0.8,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(CONFIG_ORDER, fontsize=18)
    ax.set_ylabel("Count", fontsize=20)
    ax.tick_params(axis="both", labelsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=18)

    for bars, vals in ((bars_pdb, pdb_vals), (bars_xyz, xyz_vals)):
        for bar, val in zip(bars, vals):
            if not val:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val,
                str(val),
                ha="center",
                va="bottom",
                fontsize=14,
            )

    fig.tight_layout()
    fig.savefig(output_dir / "config_counts_bar.png", dpi=200)
    plt.close(fig)


def _format_stats(values: list[float]) -> str:
    if not values:
        return "count=0 mean=nan stddev=nan"

    mean_value = statistics.mean(values)
    stddev_value = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"count={len(values)} mean={mean_value:.6f} stddev={stddev_value:.6f}"


def _write_histograms(
    aggregated: dict[str, list[float]],
    output_dir: Path,
    bins: int,
) -> None:
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Histogram output requested, but matplotlib is not installed. "
            "Install it or run without --histogram-dir."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    x_label = r"Bond Distance ($\AA$)"

    for key in HISTOGRAM_KEYS:
        values = aggregated[key]
        if not values:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(
            values,
            bins=bins,
            color=HISTOGRAM_COLORS[key],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.95,
        )
        ax.set_title(HISTOGRAM_TITLES[key], fontsize=16)
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel("Count", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(output_dir / f"hist_{key}.png", dpi=200)
        plt.close(fig)

    combined_keys = ("ND1", "NE2", "SG")
    fig, axes = plt.subplots(1, 3, figsize=(7, 2), sharey=False)
    plt.subplots_adjust(wspace=0.15)
    for ax, key in zip(axes, combined_keys):
        values = aggregated[key]
        if key == "SG":
            bins = 30
        if values:
            ax.hist(
                values,
                bins=bins,
                color=HISTOGRAM_COLORS[key],
                edgecolor="white",
                linewidth=0.8,
                alpha=0.95,
            )
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(HISTOGRAM_TITLES[key], fontsize=16)
        if key == "NE2":
            ax.set_xlabel(x_label, fontsize=14)
        ax.set_xlim(1.98, 2.62)
        ax.tick_params(axis="both", labelsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Count", fontsize=14)
    # fig.tight_layout()
    fig.savefig(output_dir / "hist_SG_ND1_NE2_combined.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def summarize_parent(parent_dir: Path, histogram_dir: Path | None, bins: int) -> int:
    if not parent_dir.exists() or not parent_dir.is_dir():
        raise SystemExit(f"Parent directory does not exist or is not a directory: {parent_dir}")

    aggregated: dict[str, list[float]] = {
        "SG": [],
        "ND1": [],
        "NE2": [],
        "CA": [],
        "CA_for_SG": [],
        "CA_for_ND1": [],
        "CA_for_NE2": [],
    }

    subdirs_processed = 0
    xyz_files_processed = 0

    for subdir in sorted(p for p in parent_dir.iterdir() if p.is_dir()):
        xyz_dir = subdir / "output" / "xyz_files"
        if not xyz_dir.is_dir():
            continue

        xyz_files = sorted(
            path
            for path in xyz_dir.glob("*.xyz")
            if "extended" not in path.name.lower()
        )
        if not xyz_files:
            continue

        subdirs_processed += 1

        for xyz_file in xyz_files:
            xyz_files_processed += 1
            per_file = _collect_distances_from_xyz(xyz_file)
            for atom_name, values in per_file.items():
                aggregated[atom_name].extend(values)

    print(f"Parent directory: {parent_dir}")
    print(f"Subdirectories processed: {subdirs_processed}")
    print(f"XYZ files processed: {xyz_files_processed}")
    print()
    print("Distance summary from origin")
    print("----------------------------")
    for atom_name in ("SG", "ND1", "NE2", "CA"):
        print(f"{atom_name:>4}: {_format_stats(aggregated[atom_name])}")

    print()
    print("CA distance summary by coordinating atom")
    print("----------------------------------------")
    print(f"CA->SG  : {_format_stats(aggregated['CA_for_SG'])}")
    print(f"CA->ND1 : {_format_stats(aggregated['CA_for_ND1'])}")
    print(f"CA->NE2 : {_format_stats(aggregated['CA_for_NE2'])}")

    config_counts = _count_files_per_config(parent_dir)
    print()
    print("File counts by Cys-His configuration")
    print("------------------------------------")
    print(f"{'config':>10}  {'pdb':>6}  {'xyz':>6}")
    for cfg in CONFIG_ORDER:
        entry = config_counts.get(cfg, {"pdb": 0, "xyz": 0})
        print(f"{cfg:>10}  {entry['pdb']:>6}  {entry['xyz']:>6}")
    extras = sorted(set(config_counts) - set(CONFIG_ORDER))
    for cfg in extras:
        entry = config_counts[cfg]
        print(f"{cfg:>10}  {entry['pdb']:>6}  {entry['xyz']:>6}  (not in default chart)")

    if histogram_dir is not None:
        _write_histograms(aggregated=aggregated, output_dir=histogram_dir, bins=bins)
        _write_config_bar_chart(counts=config_counts, output_dir=histogram_dir)
        print()
        print(f"Saved histogram PNG files to: {histogram_dir}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize SG/ND1/NE2/CA distances from origin across "
            "<subdir>/output/xyz_files/*.xyz files."
        )
    )
    parser.add_argument(
        "parent_dir",
        type=Path,
        help="Parent directory containing subdirectories to scan",
    )
    parser.add_argument(
        "--histogram-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory for histogram images. "
            "If omitted, no histograms are written."
        ),
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of bins to use for histograms (default: 40)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bins < 1:
        raise SystemExit("--bins must be >= 1")
    return summarize_parent(args.parent_dir, histogram_dir=args.histogram_dir, bins=args.bins)


if __name__ == "__main__":
    raise SystemExit(main())
