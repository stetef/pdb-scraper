#!/usr/bin/env python3
"""Plot per-cluster and cross-cluster distributions from cluster-label CSV files.

This script expects a CSV with at least:
- cluster label column (default: "cluster")
- optional per-row cluster color (default: "cluster_color")

It generates:
1) One figure per cluster with a single row of histograms/subplots for:
   - volume_A3
   - q_tetra_coord
   - q_tetra_ca
   - r_free
   - r_work
   - all 4 cys dihedral columns pooled into one histogram
   - zn_bfactor
   - all 4 coordinator residue average B-factors pooled into one histogram
   - family distribution (bar chart)

2) One figure per metric with all clusters overlaid on the same histogram
   (density=True, stacked=False) for numeric metrics.
   Family is plotted as a grouped normalized bar chart across clusters.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


NumberExtractor = Callable[[dict[str, str]], list[float]]
TextExtractor = Callable[[dict[str, str]], list[str]]


@dataclass(frozen=True)
class NumericMetric:
    key: str
    title: str
    xlabel: str
    extractor: NumberExtractor
    bins: int | str = 12
    overlay_bins: int = 12


@dataclass(frozen=True)
class CategoricalMetric:
    key: str
    title: str
    extractor: TextExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-cluster and all-cluster histograms from cluster-label CSVs."
    )
    parser.add_argument("csv_path", type=Path, help="Input CSV path with cluster labels and stats.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <csv_dir>/cluster_distribution_plots).",
    )
    parser.add_argument(
        "--cluster-col",
        default="cluster",
        help="Column containing cluster labels (default: cluster).",
    )
    parser.add_argument(
        "--cluster-color-col",
        default="cluster_color",
        help="Column containing per-row cluster color hex (default: cluster_color).",
    )
    parser.add_argument(
        "--family-col",
        default="family",
        help="Column containing family string labels (default: family).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI (default: 220).",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.25,
        help="Global font scale multiplier (default: 1.25).",
    )
    return parser.parse_args()


def configure_plot_style(font_scale: float) -> bool:
    latex_available = shutil.which("latex") is not None

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.titlesize": 14 * font_scale + 6,
            "axes.labelsize": 13 * font_scale + 6,
            "xtick.labelsize": 11 * font_scale + 6,
            "ytick.labelsize": 11 * font_scale + 6,
            "legend.fontsize": 10 * font_scale + 6,
        }
    )

    plt.rcParams["text.usetex"] = bool(latex_available)
    return latex_available


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def _first_float(row: dict[str, str], names: list[str]) -> float | None:
    for name in names:
        if name not in row:
            continue
        value = _safe_float(row.get(name))
        if value is not None:
            return value
    return None


def _collect_float_values(row: dict[str, str], names: list[str]) -> list[float]:
    values: list[float] = []
    for name in names:
        if name not in row:
            continue
        value = _safe_float(row.get(name))
        if value is not None:
            values.append(value)
    return values


def _clean_family(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text if text else None


def _cluster_sort_key(raw: str) -> tuple[int, str]:
    text = raw.strip()
    try:
        return (0, f"{int(text):09d}")
    except ValueError:
        return (1, text)


def _build_cluster_colors(rows: list[dict[str, str]], cluster_col: str, cluster_color_col: str) -> dict[str, str]:
    colors: dict[str, str] = {}
    for row in rows:
        cluster = (row.get(cluster_col) or "").strip()
        if not cluster:
            continue
        raw_color = (row.get(cluster_color_col) or "").strip()
        if not raw_color:
            continue
        try:
            colors.setdefault(cluster, mcolors.to_hex(raw_color, keep_alpha=False))
        except ValueError:
            continue

    missing = sorted({(row.get(cluster_col) or "").strip() for row in rows if (row.get(cluster_col) or "").strip()} - set(colors))
    if missing:
        cmap = plt.get_cmap("tab20", max(1, len(missing)))
        for idx, cluster in enumerate(missing):
            colors[cluster] = mcolors.to_hex(cmap(idx), keep_alpha=False)
    return colors


def _find_bfactor_columns(fieldnames: list[str]) -> list[str]:
    # Accept both CYS/HIS coordinator patterns when present.
    pattern = re.compile(r"^coord_(?:cys|his)_[1-4]_bfactor_avg$", flags=re.IGNORECASE)
    matched = [name for name in fieldnames if pattern.match(name)]
    if matched:
        return sorted(matched)

    fallback = [name for name in fieldnames if "bfactor" in name.lower() and "coord" in name.lower()]
    return sorted(fallback)


def _find_dihedral_columns(fieldnames: list[str]) -> list[str]:
    pattern = re.compile(r"^cys_dihedral_[1-4]_deg$", flags=re.IGNORECASE)
    matched = [name for name in fieldnames if pattern.match(name)]
    if matched:
        return sorted(matched)

    fallback = [name for name in fieldnames if "dihedral" in name.lower() and "deg" in name.lower()]
    return sorted(fallback)


def _metric_specs(fieldnames: list[str], family_col: str) -> tuple[list[NumericMetric], CategoricalMetric]:
    dihedral_cols = _find_dihedral_columns(fieldnames)
    bfactor_cols = _find_bfactor_columns(fieldnames)

    numeric_metrics = [
        NumericMetric(
            key="volume_A3",
            title=r"Volume",
            xlabel=r"$\mathrm{volume}\ (\AA^3)$",
            extractor=lambda row: (
                [v] if (v := _first_float(row, ["volume_A3", "volume", "volume_a3"])) is not None else []
            ),
            bins=12,
            overlay_bins=12,
        ),
        NumericMetric(
            key="q_tetra_coord",
            title=r"$q_{\mathrm{tetra}}$ coord",
            xlabel=r"$q_{\mathrm{tetra,coord}}$",
            extractor=lambda row: [v] if (v := _first_float(row, ["q_tetra_coord"])) is not None else [],
            bins=10,
            overlay_bins=10,
        ),
        NumericMetric(
            key="q_tetra_ca",
            title=r"$q_{\mathrm{tetra}}$ CA",
            xlabel=r"$q_{\mathrm{tetra,CA}}$",
            extractor=lambda row: [v] if (v := _first_float(row, ["q_tetra_ca"])) is not None else [],
            bins=10,
            overlay_bins=10,
        ),
        NumericMetric(
            key="r_free",
            title=r"$R_{\mathrm{free}}$",
            xlabel=r"$R_{\mathrm{free}}$",
            extractor=lambda row: [v] if (v := _first_float(row, ["r_free", "rfree"])) is not None else [],
            bins=9,
            overlay_bins=9,
        ),
        NumericMetric(
            key="r_work",
            title=r"$R_{\mathrm{work}}$",
            xlabel=r"$R_{\mathrm{work}}$",
            extractor=lambda row: [v] if (v := _first_float(row, ["r_work", "rwork"])) is not None else [],
            bins=9,
            overlay_bins=9,
        ),
        NumericMetric(
            key="all_dihedrals_deg",
            title=r"All Cys Dihedrals",
            xlabel=r"dihedral\ (deg)",
            extractor=lambda row, cols=dihedral_cols: _collect_float_values(row, cols),
            bins=12,
            overlay_bins=12,
        ),
        NumericMetric(
            key="zn_bfactor",
            title=r"Zn B-factor",
            xlabel=r"$B_{\mathrm{Zn}}$",
            extractor=lambda row: [v] if (v := _first_float(row, ["zn_bfactor", "zn_b_factor"])) is not None else [],
            bins=10,
            overlay_bins=10,
        ),
        NumericMetric(
            key="all_coord_res_bfactor_avg",
            title=r"Coordinator Residue B-factors",
            xlabel=r"$\langle B\rangle_{\mathrm{coord\ residue}}$",
            extractor=lambda row, cols=bfactor_cols: _collect_float_values(row, cols),
            bins=12,
            overlay_bins=12,
        ),
    ]

    family_metric = CategoricalMetric(
        key="family",
        title="Family",
        extractor=lambda row, col=family_col: ([f] if (f := _clean_family(row.get(col))) is not None else []),
    )

    return numeric_metrics, family_metric


def _extract_cluster_rows(rows: list[dict[str, str]], cluster_col: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        cluster = (row.get(cluster_col) or "").strip()
        if cluster:
            grouped[cluster].append(row)
    return dict(grouped)


def _compute_shared_bin_edges(
    numeric_metrics: list[NumericMetric], rows: list[dict[str, str]]
) -> dict[str, np.ndarray]:
    """Build common histogram bin edges per metric so cluster panels are comparable."""
    shared_edges: dict[str, np.ndarray] = {}

    for metric in numeric_metrics:
        values: list[float] = []
        for row in rows:
            values.extend(metric.extractor(row))
        if not values:
            continue

        arr = np.asarray(values, dtype=float)
        data_min = float(np.min(arr))
        data_max = float(np.max(arr))

        if data_min == data_max:
            # Keep a visible bin width for degenerate (constant) distributions.
            span = max(0.5, abs(data_min) * 0.05)
            shared_edges[metric.key] = np.array([data_min - span, data_max + span], dtype=float)
            continue

        shared_edges[metric.key] = np.histogram_bin_edges(arr, bins=metric.bins)

    return shared_edges


def _plot_cluster_row(
    cluster: str,
    rows: list[dict[str, str]],
    color: str,
    numeric_metrics: list[NumericMetric],
    shared_bin_edges: dict[str, np.ndarray],
    family_metric: CategoricalMetric,
    out_path: Path,
    dpi: int,
) -> None:
    n_panels = len(numeric_metrics) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6.0 * n_panels, 6.8), constrained_layout=True)

    if n_panels == 1:
        axes = [axes]

    for idx, metric in enumerate(numeric_metrics):
        ax = axes[idx]
        values: list[float] = []
        for row in rows:
            values.extend(metric.extractor(row))

        if values:
            ax.hist(
                values,
                bins=shared_bin_edges.get(metric.key, metric.bins),
                color=color,
                edgecolor="white",
                linewidth=1.1,
                alpha=0.95,
            )
            ax.text(
                0.5,
                0.94,
                rf"$n={len(values)}$",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=19,
            )
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

        ax.set_title(metric.title, fontsize=32)
        ax.set_xlabel(metric.xlabel)
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fam_ax = axes[-1]
    families: list[str] = []
    for row in rows:
        families.extend(family_metric.extractor(row))

    if families:
        fam_counter = Counter(families)
        ordered = fam_counter.most_common()
        labels = [item[0] for item in ordered]
        counts = [item[1] for item in ordered]
        x = np.arange(len(labels), dtype=float)
        fam_ax.bar(x, counts, color=color, edgecolor="white", linewidth=1.1)
        fam_ax.set_xticks([])
        fam_ax.set_ylabel("Count")

        top_family = labels[0]
        top_count = counts[0]
        fam_ax.text(
            0.5,
            0.97,
            f"top: {top_family} (n={top_count})",
            transform=fam_ax.transAxes,
            ha="center",
            va="top",
            fontsize=18,
        )
    else:
        fam_ax.text(0.5, 0.5, "no family data", ha="center", va="center", transform=fam_ax.transAxes)

    fam_ax.set_title(family_metric.title, fontsize=32)
    fam_ax.set_xlabel("Family category")
    fam_ax.set_ylabel("")
    fam_ax.set_yticks([])
    fam_ax.spines["top"].set_visible(False)
    fam_ax.spines["right"].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_overlay_numeric_metric(
    metric: NumericMetric,
    grouped_rows: dict[str, list[dict[str, str]]],
    clusters_sorted: list[str],
    color_map: dict[str, str],
    out_path: Path,
    dpi: int,
) -> None:
    values_by_cluster: list[list[float]] = []
    labels: list[str] = []
    colors: list[str] = []

    for cluster in clusters_sorted:
        vals: list[float] = []
        for row in grouped_rows[cluster]:
            vals.extend(metric.extractor(row))
        if vals:
            values_by_cluster.append(vals)
            labels.append(f"cluster {cluster}")
            colors.append(color_map[cluster])

    if not values_by_cluster:
        return

    fig, ax = plt.subplots(figsize=(15.5, 7.2), constrained_layout=True)
    ax.hist(
        values_by_cluster,
        bins=metric.overlay_bins,
        density=True,
        stacked=False,
        histtype="bar",
        label=labels,
        color=colors,
        edgecolor="none",
        linewidth=0.0,
        alpha=0.9,
    )

    ax.set_title(rf"All clusters: {metric.title}")
    ax.set_xlabel(metric.xlabel)
    ax.set_ylabel("Density")
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_overlay_family(
    family_metric: CategoricalMetric,
    grouped_rows: dict[str, list[dict[str, str]]],
    clusters_sorted: list[str],
    color_map: dict[str, str],
    out_path: Path,
    dpi: int,
) -> None:
    family_counts_per_cluster: dict[str, Counter[str]] = {}
    all_families_counter: Counter[str] = Counter()

    for cluster in clusters_sorted:
        fam_values: list[str] = []
        for row in grouped_rows[cluster]:
            fam_values.extend(family_metric.extractor(row))
        counter = Counter(fam_values)
        family_counts_per_cluster[cluster] = counter
        all_families_counter.update(counter)

    if not all_families_counter:
        return

    top_families = [fam for fam, _ in all_families_counter.most_common(10)]
    x = np.arange(len(top_families), dtype=float)
    width = 0.8 / max(1, len(clusters_sorted))

    fig, ax = plt.subplots(figsize=(14.5, 7.5), constrained_layout=True)

    for idx, cluster in enumerate(clusters_sorted):
        counter = family_counts_per_cluster[cluster]
        total = sum(counter.values())
        if total <= 0:
            y = np.zeros(len(top_families), dtype=float)
        else:
            y = np.array([counter.get(fam, 0) / total for fam in top_families], dtype=float)

        offset = (idx - (len(clusters_sorted) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            y,
            width=width,
            color=color_map[cluster],
            edgecolor="white",
            linewidth=0.9,
            label=f"cluster {cluster}",
            alpha=0.95,
        )

    ax.set_title(r"All clusters: Family distribution (normalized)")
    ax.set_xlabel("Family")
    ax.set_ylabel("Normalized frequency")
    ax.set_yticks([])
    ax.set_xticks(x)
    ax.set_xticklabels(top_families, rotation=35, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=True, ncols=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path: Path = args.csv_path.expanduser().resolve()

    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else csv_path.parent / "cluster_distribution_plots"
    per_cluster_dir = out_dir / "per_cluster_rows"
    overlays_dir = out_dir / "all_cluster_overlays"

    configure_plot_style(font_scale=max(0.5, args.font_scale))

    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows:
        raise SystemExit(f"No rows found in CSV: {csv_path}")
    if args.cluster_col not in fieldnames:
        raise SystemExit(
            f"Cluster column '{args.cluster_col}' not found. Available columns: {', '.join(fieldnames)}"
        )

    numeric_metrics, family_metric = _metric_specs(fieldnames, args.family_col)
    grouped_rows = _extract_cluster_rows(rows, args.cluster_col)
    if not grouped_rows:
        raise SystemExit(f"No non-empty cluster labels found in column '{args.cluster_col}'.")

    shared_bin_edges = _compute_shared_bin_edges(numeric_metrics, rows)

    clusters_sorted = sorted(grouped_rows.keys(), key=_cluster_sort_key)
    color_map = _build_cluster_colors(rows, args.cluster_col, args.cluster_color_col)

    for cluster in clusters_sorted:
        out_path = per_cluster_dir / f"cluster_{cluster}_metrics_row.png"
        _plot_cluster_row(
            cluster=cluster,
            rows=grouped_rows[cluster],
            color=color_map[cluster],
            numeric_metrics=numeric_metrics,
            shared_bin_edges=shared_bin_edges,
            family_metric=family_metric,
            out_path=out_path,
            dpi=args.dpi,
        )

    for metric in numeric_metrics:
        out_path = overlays_dir / f"{metric.key}_all_clusters_overlay.png"
        _plot_overlay_numeric_metric(
            metric=metric,
            grouped_rows=grouped_rows,
            clusters_sorted=clusters_sorted,
            color_map=color_map,
            out_path=out_path,
            dpi=args.dpi,
        )

    family_out = overlays_dir / "family_all_clusters_overlay.png"
    _plot_overlay_family(
        family_metric=family_metric,
        grouped_rows=grouped_rows,
        clusters_sorted=clusters_sorted,
        color_map=color_map,
        out_path=family_out,
        dpi=args.dpi,
    )

    print(f"Wrote per-cluster rows to: {per_cluster_dir}")
    print(f"Wrote all-cluster overlays to: {overlays_dir}")


if __name__ == "__main__":
    main()
