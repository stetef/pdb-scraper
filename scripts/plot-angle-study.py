#!/usr/bin/env python3
"""Generate 3D EXAFS plots for angle studies.

This script expects child directories named like:
    <prefix>-alpha-<value>-beta-<value>-gamma-<value>
where each angle value can be numeric or prefixed with "neg" (e.g. neg10 -> -10).

For each varied angle (alpha, beta, gamma), the script:
1) Selects all directories where that angle is non-zero.
2) Adds all-zero baseline directory/directories (alpha=beta=gamma=0).
3) Sorts entries by the varied angle value.
4) Plots chi(R) spectra in 3D with:
   x-axis: varied angle
   y-axis: R
   z-axis: |chi(R)|
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ANGLE_DIR_PATTERN = re.compile(
    r"^(?P<prefix>.+)-alpha-(?P<alpha>[^-]+)-beta-(?P<beta>[^-]+)-gamma-(?P<gamma>[^-]+)$",
    flags=re.IGNORECASE,
)
ANGLE_NAMES = ("alpha", "beta", "gamma")


@dataclass(frozen=True)
class AngleEntry:
    directory: Path
    prefix: str
    alpha: float
    beta: float
    gamma: float
    chi_path: Path


def _apply_plot_rcparams(plt_module) -> None:
    plt_module.rcParams.update(
        {
            "font.family": "serif",
            "axes.labelsize": 10,
            "axes.titlesize": 16,
            "legend.fontsize": 10,
            "axes.linewidth": 1.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _style_3d_axes(ax, *, angle_axis_ratio: float) -> None:
    # Shrink the angle-axis visual footprint so angle slices sit closer together.
    ax.set_box_aspect((max(angle_axis_ratio, 0.05), 1.35, 1.0))
    ax.grid(False)

    ax.tick_params(width=2, length=7, direction="out", labelsize=7)

    # Use fully white panes/background to avoid default gray 3D shading.
    ax.set_facecolor("white")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        pane.set_edgecolor((1.0, 1.0, 1.0, 1.0))



def _parse_angle_token(token: str) -> float:
    t = token.strip().lower()
    if not t:
        raise ValueError("empty angle token")

    if t.startswith("neg"):
        raw = t[3:]
        if not raw:
            raise ValueError(f"invalid neg angle token: {token}")
        return -float(raw)

    return float(t)


def _nearly_zero(value: float, tol: float = 1.0e-12) -> bool:
    return abs(value) <= tol


def _find_chi_file(child_dir: Path, patterns: list[str]) -> Path:
    matches: set[Path] = set()
    for pattern in patterns:
        matches.update(child_dir.glob(pattern))

    sorted_matches = sorted(matches)
    if not sorted_matches:
        raise FileNotFoundError(f"No chi(R) file matching {patterns} in {child_dir}")
    if len(sorted_matches) > 1:
        print(f"[warn] Multiple chi(R) files in {child_dir}; using {sorted_matches[0].name}")
    return sorted_matches[0]


def _load_r_chir_mag(path: Path) -> tuple[np.ndarray, np.ndarray]:
    r_vals: list[float] = []
    chir_mag_vals: list[float] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split()
            try:
                r_val = float(parts[0])
                chir_mag = float(parts[1])
            except (ValueError, IndexError):
                continue

            r_vals.append(r_val)
            chir_mag_vals.append(chir_mag)

    if not r_vals:
        raise ValueError(f"No numeric chi(R) data found in {path}")

    return np.asarray(r_vals), np.asarray(chir_mag_vals)


def _iter_angle_entries(parent_dir: Path, chi_patterns: list[str]) -> list[AngleEntry]:
    entries: list[AngleEntry] = []

    for child_dir in sorted(p for p in parent_dir.iterdir() if p.is_dir()):
        match = ANGLE_DIR_PATTERN.match(child_dir.name)
        if match is None:
            continue

        try:
            alpha = _parse_angle_token(match.group("alpha"))
            beta = _parse_angle_token(match.group("beta"))
            gamma = _parse_angle_token(match.group("gamma"))
        except ValueError as exc:
            print(f"[skip] {child_dir.name}: {exc}")
            continue

        try:
            chi_path = _find_chi_file(child_dir, chi_patterns)
        except FileNotFoundError as exc:
            print(f"[skip] {exc}")
            continue

        entries.append(
            AngleEntry(
                directory=child_dir,
                prefix=match.group("prefix"),
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                chi_path=chi_path,
            )
        )

    return entries


def _is_all_zero(entry: AngleEntry) -> bool:
    return _nearly_zero(entry.alpha) and _nearly_zero(entry.beta) and _nearly_zero(entry.gamma)


def _axis_value(entry: AngleEntry, axis_name: str) -> float:
    if axis_name == "alpha":
        return entry.alpha
    if axis_name == "beta":
        return entry.beta
    if axis_name == "gamma":
        return entry.gamma
    raise ValueError(f"Unknown axis: {axis_name}")


def _entry_label(entry: AngleEntry) -> str:
    return f"{entry.directory.name}"


def _make_group(entries: list[AngleEntry], varied_axis: str) -> list[AngleEntry]:
    baseline = [e for e in entries if _is_all_zero(e)]
    varied = [e for e in entries if not _nearly_zero(_axis_value(e, varied_axis))]

    by_dir: dict[Path, AngleEntry] = {e.directory: e for e in varied}
    for e in baseline:
        by_dir[e.directory] = e

    grouped = list(by_dir.values())
    grouped.sort(key=lambda e: (_axis_value(e, varied_axis), _entry_label(e)))
    return grouped


def _plot_group_3d(
    entries: list[AngleEntry],
    *,
    varied_axis: str,
    out_path: Path,
    elev: float,
    azim: float,
    angle_axis_ratio: float,
    r_shift_step: float,
) -> None:
    fig = plt.figure(figsize=(10.5, 7.5), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    _style_3d_axes(ax, angle_axis_ratio=angle_axis_ratio)

    line_color = "black"
    r_min = float("inf")
    r_max = float("-inf")
    chi_min = float("inf")
    chi_max = float("-inf")

    for idx, entry in enumerate(entries):
        x_val = _axis_value(entry, varied_axis)
        r_vals, chir_mag_vals = _load_r_chir_mag(entry.chi_path)
        shifted_r_vals = r_vals + (idx * r_shift_step)
        r_min = min(r_min, float(np.min(shifted_r_vals)))
        r_max = max(r_max, float(np.max(shifted_r_vals)))
        chi_min = min(chi_min, float(np.min(chir_mag_vals)))
        chi_max = max(chi_max, float(np.max(chir_mag_vals)))

        x_vals = np.full_like(r_vals, fill_value=x_val, dtype=float)

        ax.plot(x_vals, shifted_r_vals, chir_mag_vals, color=line_color, linewidth=0.7, alpha=0.92)

    x_values = [_axis_value(e, varied_axis) for e in entries]
    ax.set_xlim(min(x_values), max(x_values))
    ax.set_xlabel(f"$\{varied_axis}$\n(deg)")
    ax.set_ylabel(r"R ($\AA$)")
    ax.set_zlabel(r"$|\chi(R)|$")
    ax.set_title(f"3D EXAFS vs $\{varied_axis}$ angle")
    ax.view_init(elev=elev, azim=azim)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_all_groups_grid(
    grouped: dict[str, list[AngleEntry]],
    *,
    out_path: Path,
    elev: float,
    azim: float,
    angle_axis_ratio: float,
    r_shift_step: float,
) -> None:
    fig = plt.figure(figsize=(18, 6.2), facecolor="white")

    for idx, varied_axis in enumerate(ANGLE_NAMES, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        _style_3d_axes(ax, angle_axis_ratio=angle_axis_ratio)
        entries = grouped[varied_axis]
        line_color = "black"
        r_min = float("inf")
        r_max = float("-inf")
        chi_min = float("inf")
        chi_max = float("-inf")

        for jdx, entry in enumerate(entries):
            x_val = _axis_value(entry, varied_axis)
            r_vals, chir_mag_vals = _load_r_chir_mag(entry.chi_path)
            shifted_r_vals = r_vals + (jdx * r_shift_step)
            r_min = min(r_min, float(np.min(shifted_r_vals)))
            r_max = max(r_max, float(np.max(shifted_r_vals)))
            chi_min = min(chi_min, float(np.min(chir_mag_vals)))
            chi_max = max(chi_max, float(np.max(chir_mag_vals)))
            x_vals = np.full_like(r_vals, fill_value=x_val, dtype=float)

            ax.plot(x_vals, shifted_r_vals, chir_mag_vals, color=line_color, linewidth=0.7, alpha=0.9)

        x_values = [_axis_value(e, varied_axis) for e in entries]
        ax.set_xlim(min(x_values), max(x_values))
        ax.set_xlabel(f"$\{varied_axis}$ (deg)")
        ax.set_ylabel(r"R ($\AA$)")
        ax.set_zlabel(r"$|\chi(R)|$")
        ax.set_title(f"$\{varied_axis}$", y=0.85)
        ax.view_init(elev=elev, azim=azim)

    fig.suptitle("Angle Study EXAFS (3D)", fontsize=18, y=0.8)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build alpha/beta/gamma non-zero + baseline groups from angle-study directories "
            "and generate 3D chi(R) figures."
        )
    )
    parser.add_argument("parent_dir", type=Path, help="Directory containing angle-study child folders")
    parser.add_argument(
        "--chi-pattern",
        action="append",
        default=["chi-R*.dat", "chi_R*.dat"],
        help="Glob pattern for chi(R) files inside each child folder (repeatable).",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=30.0,
        help="3D camera elevation angle (default: 30)",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=15.0,
        help="3D camera azimuth angle (default: 15)",
    )
    parser.add_argument(
        "--angle-axis-ratio",
        type=float,
        default=0.35,
        help=(
            "Relative 3D axis length for the angle axis (x). "
            "Smaller values visually compress angle spacing (default: 0.35)."
        ),
    )
    parser.add_argument(
        "--r-shift-step",
        type=float,
        default=0.07,
        help=(
            "Per-spectrum right shift applied on the R axis in plotting order. "
            "A value of 0.1 shifts each next spectrum by +0.1 in R (default: 0.5)."
        ),
    )

    args = parser.parse_args()
    parent_dir = args.parent_dir

    if not parent_dir.is_dir():
        raise SystemExit(f"Parent directory not found: {parent_dir}")
    if args.r_shift_step < 0:
        raise SystemExit("--r-shift-step must be >= 0")

    _apply_plot_rcparams(plt)
    entries = _iter_angle_entries(parent_dir, args.chi_pattern)
    if not entries:
        raise SystemExit("No valid angle-study child directories were found.")

    grouped: dict[str, list[AngleEntry]] = {}
    for axis_name in ANGLE_NAMES:
        group_entries = _make_group(entries, axis_name)
        if not group_entries:
            print(f"[skip] No entries for {axis_name} grouping")
            continue
        grouped[axis_name] = group_entries

    if not grouped:
        raise SystemExit("No group had plottable entries.")

    safe_dir_name = parent_dir.name.replace(" ", "")
    for axis_name, group_entries in grouped.items():
        out_path = parent_dir / f"EXAFS-angle-study-3d-{axis_name}-{safe_dir_name}.png"
        _plot_group_3d(
            group_entries,
            varied_axis=axis_name,
            out_path=out_path,
            elev=args.elev,
            azim=args.azim,
            angle_axis_ratio=args.angle_axis_ratio,
            r_shift_step=args.r_shift_step,
        )
        print(f"[ok] Wrote figure: {out_path}")
        print(f"[ok] {axis_name} group size: {len(group_entries)}")

    if all(axis in grouped for axis in ANGLE_NAMES):
        grid_out_path = parent_dir / f"EXAFS-angle-study-3d-grid-{safe_dir_name}.png"
        _plot_all_groups_grid(
            grouped,
            out_path=grid_out_path,
            elev=args.elev,
            azim=args.azim,
            angle_axis_ratio=args.angle_axis_ratio,
            r_shift_step=args.r_shift_step,
        )
        print(f"[ok] Wrote figure: {grid_out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
