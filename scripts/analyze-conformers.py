#!/usr/bin/env python3
"""Process dw.dat files and make plots of the spectra and disorder contributions.

First, center XYZ files on Zn and annotate matched atoms with dw sigma^2 values.
Given a parent directory, this script iterates over each direct child directory and:
1) Reads one dw*.dat file from that child directory.
2) Reads each .xyz file in that child directory (excluding *_w_dw.xyz outputs).
3) Centers all coordinates by translating the Zn atom to the origin.
4) Matches centered XYZ atoms to dw rows by (element, x, y, z) within tolerance.
5) Writes <stem>_w_dw.xyz with trailing comments like: # dw=<sig2_tot>
Then generate figures figures in the parent directory:
    - dw vs distance for first-shell (S) and C$\alpha$ (C)
    - dw vs pocket volume for first-shell (S) and C$\alpha$ (C)
    - EXAFS spectra (chi(R) magnitude)
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


@dataclass(frozen=True)
class XyzAtom:
    symbol: str
    x: float
    y: float
    z: float
    trailing_comment: str


@dataclass(frozen=True)
class DwAtom:
    symbol: str
    x: float
    y: float
    z: float
    sig2_tot: str


DW_PATTERN = re.compile(r"(?:^|\s)dw=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
FOUR_ALNUM_PATTERN = re.compile(r"^[A-Za-z0-9]{4}$")
CLUSTER_TOKEN_PATTERN = re.compile(r"^cluster\d+$", re.IGNORECASE)
CHLS_ONLY_TOKEN_PATTERN = re.compile(r"^[CHLSchls]{4}$")
PDB_ID_PATTERN = re.compile(r"^[0-9][A-Za-z0-9]{3}$")
PLACEHOLDER_ID_PATTERN = re.compile(r"^[Xx]{4}$")


@dataclass(frozen=True)
class DwTaggedAtom:
    symbol: str
    x: float
    y: float
    z: float
    dw: float


@dataclass(frozen=True)
class FileMetrics:
    path: Path
    prefix: str
    suffix: str
    volume: float
    aggregate_raw_mean_dw_first: float
    aggregate_raw_mean_dw_ca: float
    aggregate_variance_first_from_origin: float
    aggregate_variance_ca_from_origin: float
    aggregate_mean_dw_first: float
    aggregate_mean_dw_ca: float
    aggregate_first_distance_dw: list[tuple[float, float]]
    aggregate_ca_distance_dw: list[tuple[float, float]]
    by_atom_type: dict[str, "CoordinationTypeMetrics"]


@dataclass(frozen=True)
class CoordinationTypeMetrics:
    atom_type: str
    raw_mean_dw_first: float
    raw_mean_dw_ca: float
    variance_first_from_origin: float
    variance_ca_from_origin: float
    mean_dw_first: float
    mean_dw_ca: float
    first_distance_dw: list[tuple[float, float]]
    ca_distance_dw: list[tuple[float, float]]


@dataclass(frozen=True)
class DwDistanceSummary:
    prefix: str
    suffix: str
    mean_distance: float
    std_distance: float
    n_points: int
    distances: list[float]
    source_dir: Path


@dataclass(frozen=True)
class DwFactorSummary:
    prefix: str
    suffix: str
    mean_dw: float
    std_dw: float
    n_points: int
    dw_values: list[float]
    source_dir: Path


def _apply_plot_rcparams(plt_module) -> None:
    plt_module.rcParams.update(
        {
            "font.family": "serif",
            "axes.labelsize": 18,
            "axes.titlesize": 22,
            "axes.linewidth": 2,
            "legend.fontsize": 18,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "white",
            "legend.fancybox": False,
            "legend.shadow": False,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "xtick.major.size": 8,
            "ytick.major.size": 8,
            "xtick.major.width": 2,
            "ytick.major.width": 2,
            "xtick.minor.size": 4,
            "ytick.minor.size": 4,
            "xtick.minor.width": 1.5,
            "ytick.minor.width": 1.5,
        }
    )


def _iter_data_files(base_dir: Path, patterns: Iterable[str]) -> list[Path]:
    files: set[Path] = set()
    for pattern in patterns:
        files.update(base_dir.rglob(pattern))
    return sorted(files)


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
        raise ValueError(f"No numeric data found in {path}")

    return np.asarray(r_vals), np.asarray(chir_mag_vals)


def _load_k_chi(path: Path) -> tuple[np.ndarray, np.ndarray]:
    k_vals: list[float] = []
    chi_vals: list[float] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split()
            if len(parts) < 2:
                continue

            try:
                # FEFF-style exafs-k files often have columns:
                # omega e k mu mu0 chi [@#]
                # while generated average files are 2-column: k chi.
                if len(parts) >= 6:
                    k_vals.append(float(parts[2]))
                    chi_vals.append(float(parts[5]))
                else:
                    k_vals.append(float(parts[0]))
                    chi_vals.append(float(parts[1]))
            except ValueError:
                continue

    if not k_vals:
        raise ValueError(f"No numeric chi(k) data found in {path}")

    k_arr = np.asarray(k_vals, dtype=float)
    chi_arr = np.asarray(chi_vals, dtype=float)

    order = np.argsort(k_arr)
    k_sorted = k_arr[order]
    chi_sorted = chi_arr[order]

    # Keep one value per k for robust interpolation.
    uniq_k, uniq_idx = np.unique(k_sorted, return_index=True)
    uniq_chi = chi_sorted[uniq_idx]
    return uniq_k, uniq_chi


def _build_avg_exafs_k_chi_r(parent_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    exafs_k_files: list[Path] = []
    for child_dir in iter_child_dirs(parent_dir):
        exafs_k_files.extend(sorted(child_dir.glob("exafs-k-*.dat")))

    if not exafs_k_files:
        return None

    loaded_k_chi: list[tuple[np.ndarray, np.ndarray, Path]] = []
    for path in exafs_k_files:
        try:
            k_vals, chi_vals = _load_k_chi(path)
        except ValueError as exc:
            print(f"[skip] exafs-k file {path}: {exc}")
            continue
        loaded_k_chi.append((k_vals, chi_vals, path))

    if not loaded_k_chi:
        return None

    # Regrid all chi(k) traces to a regular spacing to keep point count manageable
    # for downstream Larch FFT defaults.
    k_mins = [float(np.min(k_vals)) for k_vals, _, _ in loaded_k_chi]
    k_maxs = [float(np.max(k_vals)) for k_vals, _, _ in loaded_k_chi]
    common_k_min = max(k_mins)
    common_k_max = min(k_maxs)

    # Use an explicit transform window and resample to a compact grid.
    fft_kmin = 0.0
    fft_kmax = 13.0
    fft_dk = 3.0
    fft_window = "kaiser"
    common_k_min = max(common_k_min, fft_kmin)
    common_k_max = min(common_k_max, fft_kmax)
    if not np.isfinite(common_k_min) or not np.isfinite(common_k_max) or common_k_max <= common_k_min:
        print("[skip] no overlapping k-range found across exafs-k files")
        return None

    # Interpolate onto a fixed-size grid to reduce point count in downstream FFT input.
    target_n_points = 512
    target_k = np.linspace(common_k_min, common_k_max, num=target_n_points, dtype=float)
    if target_k.size < 4:
        print("[skip] overlapping exafs-k range too small after regridding")
        return None

    aligned = [
        np.interp(target_k, k_vals, chi_vals)
        for k_vals, chi_vals, _ in loaded_k_chi
    ]
    avg_chi_k = np.mean(np.vstack(aligned), axis=0)

    dir_suffix = parent_dir.name.replace(" ", "")
    avg_k_path = parent_dir / f"avg_exafs_k-{dir_suffix}.dat"
    np.savetxt(
        avg_k_path,
        np.column_stack([target_k, avg_chi_k]),
        header="k chi",
    )

    converter_script = Path("/Users/stetef/Documents/SLAC/pdb-scraper/scripts/chi_k_to_chi_r.py")
    if not converter_script.exists():
        print(f"[skip] chi(k)->chi(R) converter not found: {converter_script}")
        return None

    try:
        subprocess.run(
            [
                sys.executable,
                str(converter_script),
                str(avg_k_path),
                "--kmin",
                str(fft_kmin),
                "--kmax",
                str(fft_kmax),
                "--dk",
                str(fft_dk),
                "--window",
                fft_window,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        print(f"[skip] failed to convert averaged exafs-k data: {details}")
        return None

    avg_r_path = avg_k_path.with_name(f"{avg_k_path.stem}_chi_R.dat")
    if not avg_r_path.exists():
        print(f"[skip] converted chi(R) file not found: {avg_r_path}")
        return None

    try:
        return _load_r_chir_mag(avg_r_path)
    except ValueError as exc:
        print(f"[skip] averaged converted chi(R) file {avg_r_path}: {exc}")
        return None


def _normalized_stem(path: Path) -> str:
    stem = path.stem
    for lead in ("chi-R-", "chi_R", "chi-R", "chiR-"):
        if stem.startswith(lead):
            stem = stem[len(lead) :]
            break
    return stem


def _extract_id_token(value: str) -> str | None:
    for token in re.findall(r"[A-Za-z0-9]+", value):
        if not FOUR_ALNUM_PATTERN.fullmatch(token):
            continue
        if CHLS_ONLY_TOKEN_PATTERN.fullmatch(token):
            continue
        if PLACEHOLDER_ID_PATTERN.fullmatch(token):
            continue
        return token.upper()
    return None


def _is_pdb_id_token(token: str) -> bool:
    if not PDB_ID_PATTERN.fullmatch(token):
        return False
    if CHLS_ONLY_TOKEN_PATTERN.fullmatch(token):
        return False
    if PLACEHOLDER_ID_PATTERN.fullmatch(token):
        return False
    return True


def _split_known_variant_suffix(base: str) -> tuple[str, str]:
    for suffix in ("h-only-w-n", "h-only", "w-n"):
        tag = f"-{suffix}"
        if base.lower().endswith(tag):
            return base[: -len(tag)], base[-len(suffix) :]
    return base, ""


def _exp_legend_label(path: Path) -> str:
    label = path.stem
    match = re.match(r"^(.+?)(?:[_-]exp)(?:$|[_-].*)", label, flags=re.IGNORECASE)
    if match is not None:
        trimmed = match.group(1).strip("_-")
        if trimmed:
            return trimmed
    return label


def _resolve_exp_data_path(exp_path: Path, parent_dir: Path) -> Path:
    if exp_path.exists():
        return exp_path
    if not exp_path.is_absolute():
        candidate = parent_dir / exp_path
        if candidate.exists():
            return candidate
    return exp_path


def parse_w_dw_xyz(path: Path) -> list[DwTaggedAtom]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"XYZ too short: {path}")

    atoms: list[DwTaggedAtom] = []
    for line in lines[2:]:
        stripped = line.strip()
        if not stripped or "#" not in stripped:
            continue
        left, right = stripped.split("#", 1)
        dw_match = DW_PATTERN.search(right)
        if dw_match is None:
            continue

        parts = left.split()
        if len(parts) < 4:
            continue

        symbol = parts[0].strip()
        try:
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            dw = float(dw_match.group(1))
        except ValueError:
            continue

        atoms.append(DwTaggedAtom(symbol=symbol, x=x, y=y, z=z, dw=dw))

    return atoms


def tetrahedron_volume(points: list[np.ndarray]) -> float:
    if len(points) != 4:
        raise ValueError("Tetrahedron volume requires exactly 4 points")
    p0, p1, p2, p3 = points
    mat = np.column_stack((p1 - p0, p2 - p0, p3 - p0))
    return float(abs(np.linalg.det(mat)) / 6.0)


def split_prefix_suffix(stem: str) -> tuple[str, str]:
    base = stem
    if base.endswith("_w_dw"):
        base = base[: -len("_w_dw")]

    tokens = [token for token in base.split("-") if token and not PLACEHOLDER_ID_PATTERN.fullmatch(token)]
    if not tokens:
        return base, ""

    id_idx = next(
        (
            idx
            for idx, token in enumerate(tokens)
            if _is_pdb_id_token(token)
        ),
        None,
    )
    if id_idx is not None:
        end_idx = id_idx
        if id_idx + 1 < len(tokens) and CLUSTER_TOKEN_PATTERN.fullmatch(tokens[id_idx + 1]):
            end_idx = id_idx + 1

        prefix = "-".join(tokens[: end_idx + 1])
        suffix = "-".join(tokens[end_idx + 1 :]).strip()
        return prefix, suffix

    no_placeholder_base = "-".join(tokens)
    return _split_known_variant_suffix(no_placeholder_base)


def suffix_sort_key(suffix: str) -> tuple[int, str]:
    normalized = suffix.strip().lower()
    if normalized.startswith("h-only"):
        return (0, "")
    if normalized == "":
        return (1, "")
    return (2, normalized)


def _include_dw_suffix(suffix: str) -> bool:
    normalized = suffix.strip().lower()
    return normalized == ""


def format_prefix_label(prefix: str) -> str:
    parts = [part for part in prefix.split("-") if part]
    if len(parts) > 2:
        return "-".join(parts[:-2])
    return prefix


def format_prefix_label_with_pdb(prefix: str) -> str:
    parts = [part for part in prefix.split("-") if part]
    if len(parts) >= 3:
        return "-".join(parts[:3])
    return prefix


def _assign_ca_to_first_shell(
    first_shell_atoms: list[DwTaggedAtom], ca_atoms: list[DwTaggedAtom]
) -> list[DwTaggedAtom]:
    if len(first_shell_atoms) > len(ca_atoms):
        raise ValueError(
            "missing dw notation for C atoms "
            f"(need at least {len(first_shell_atoms)} C atoms with dw=<value>, found {len(ca_atoms)})"
        )

    best_perm: tuple[int, ...] | None = None
    best_total = float("inf")
    for perm in itertools.permutations(range(len(ca_atoms)), len(first_shell_atoms)):
        total = 0.0
        for idx, atom in enumerate(first_shell_atoms):
            ca = ca_atoms[perm[idx]]
            total += _dist2((atom.x, atom.y, atom.z), (ca.x, ca.y, ca.z))
        if total < best_total:
            best_total = total
            best_perm = perm

    if best_perm is None:
        raise ValueError("unable to match C atoms to first-shell atoms")

    return [ca_atoms[i] for i in best_perm]


def metrics_from_file(path: Path) -> FileMetrics:
    atoms = parse_w_dw_xyz(path)
    ca_atoms = [a for a in atoms if a.symbol.strip().upper() == "C"]
    first_shell_atoms = [a for a in atoms if a.symbol.strip().upper() in {"S", "N"}]
    first_shell_type_labels = _match_first_shell_n_to_labels(path, first_shell_atoms)

    if len(first_shell_atoms) != 4:
        raise ValueError(
            "missing dw notation for first-shell S/N atoms "
            f"(expected 4 S/N atoms with dw=<value>, found {len(first_shell_atoms)})"
        )
    if len(ca_atoms) < 4:
        raise ValueError(
            "missing dw notation for C atoms "
            f"(expected at least 4 C atoms with dw=<value>, found {len(ca_atoms)})"
        )

    paired_ca = _assign_ca_to_first_shell(first_shell_atoms, ca_atoms)

    type_buckets: dict[str, dict[str, list[DwTaggedAtom]]] = {}
    for first_atom, ca_atom, atom_type in zip(first_shell_atoms, paired_ca, first_shell_type_labels):
        bucket = type_buckets.setdefault(atom_type, {"first": [], "ca": []})
        bucket["first"].append(first_atom)
        bucket["ca"].append(ca_atom)

    c_points = [np.asarray((a.x, a.y, a.z), dtype=float) for a in ca_atoms]
    volume = tetrahedron_volume(c_points)

    first_distances_from_origin = [float(np.linalg.norm((a.x, a.y, a.z))) for a in first_shell_atoms]
    ca_distances_from_origin = [float(np.linalg.norm((a.x, a.y, a.z))) for a in paired_ca]
    aggregate_first_distance_dw = list(zip(first_distances_from_origin, [a.dw for a in first_shell_atoms]))
    aggregate_ca_distance_dw = list(zip(ca_distances_from_origin, [a.dw for a in paired_ca]))
    aggregate_variance_first_from_origin = float(np.var(first_distances_from_origin))
    aggregate_variance_ca_from_origin = float(np.var(ca_distances_from_origin))
    aggregate_raw_mean_dw_first = float(np.mean([a.dw for a in first_shell_atoms]))
    aggregate_raw_mean_dw_ca = float(np.mean([a.dw for a in paired_ca]))
    aggregate_mean_dw_first = aggregate_raw_mean_dw_first + aggregate_variance_first_from_origin
    aggregate_mean_dw_ca = aggregate_raw_mean_dw_ca + aggregate_variance_ca_from_origin

    by_atom_type: dict[str, CoordinationTypeMetrics] = {}
    for atom_type, atoms_for_type in type_buckets.items():
        first_for_type = atoms_for_type["first"]
        ca_for_type = atoms_for_type["ca"]
        first_distances = [float(np.linalg.norm((a.x, a.y, a.z))) for a in first_for_type]
        ca_distances = [float(np.linalg.norm((a.x, a.y, a.z))) for a in ca_for_type]
        raw_mean_dw_first = float(np.mean([a.dw for a in first_for_type]))
        raw_mean_dw_ca = float(np.mean([a.dw for a in ca_for_type]))
        variance_first_from_origin = float(np.var(first_distances))
        variance_ca_from_origin = float(np.var(ca_distances))
        by_atom_type[atom_type] = CoordinationTypeMetrics(
            atom_type=atom_type,
            raw_mean_dw_first=raw_mean_dw_first,
            raw_mean_dw_ca=raw_mean_dw_ca,
            variance_first_from_origin=variance_first_from_origin,
            variance_ca_from_origin=variance_ca_from_origin,
            mean_dw_first=raw_mean_dw_first + variance_first_from_origin,
            mean_dw_ca=raw_mean_dw_ca + variance_ca_from_origin,
            first_distance_dw=list(zip(first_distances, [a.dw for a in first_for_type])),
            ca_distance_dw=list(zip(ca_distances, [a.dw for a in ca_for_type])),
        )

    prefix, suffix = split_prefix_suffix(path.stem)
    return FileMetrics(
        path=path,
        prefix=prefix,
        suffix=suffix,
        volume=volume,
        aggregate_raw_mean_dw_first=aggregate_raw_mean_dw_first,
        aggregate_raw_mean_dw_ca=aggregate_raw_mean_dw_ca,
        aggregate_variance_first_from_origin=aggregate_variance_first_from_origin,
        aggregate_variance_ca_from_origin=aggregate_variance_ca_from_origin,
        aggregate_mean_dw_first=aggregate_mean_dw_first,
        aggregate_mean_dw_ca=aggregate_mean_dw_ca,
        aggregate_first_distance_dw=aggregate_first_distance_dw,
        aggregate_ca_distance_dw=aggregate_ca_distance_dw,
        by_atom_type=by_atom_type,
    )


def collect_w_dw_xyz(parent_dir: Path) -> list[Path]:
    return sorted(parent_dir.rglob("*_w_dw.xyz"))


def make_plot(
    metrics: list[FileMetrics],
    *,
    out_path: Path,
    variant_separation: float,
    group_by_atom_type: bool,
    variance_ymin: float | None,
    variance_ymax: float | None,
    total_ymin: float | None,
    total_ymax: float | None,
    atom_type_separation: float,
) -> None:
    _apply_plot_rcparams(plt)
    fig = plt.figure(figsize=(12.5, 14.5))
    gs = fig.add_gridspec(
        3,
        1,
        width_ratios=[1.0],
        height_ratios=[1.25, 1.25, 1.6],
        wspace=0.0,
        hspace=0.16,
    )
    ax_raw_s = fig.add_subplot(gs[0, 0])
    ax_var_s = fig.add_subplot(gs[1, 0], sharex=ax_raw_s)
    ax_total_s = fig.add_subplot(gs[2, 0], sharex=ax_raw_s)

    filtered = [m for m in metrics if _include_dw_suffix(m.suffix)]
    by_prefix: dict[str, list[FileMetrics]] = {}
    for m in filtered:
        by_prefix.setdefault(m.prefix, []).append(m)

    def _metric_value(metric: FileMetrics, species: str, component: str, atom_type: str | None) -> float:
        if atom_type is not None and atom_type in metric.by_atom_type:
            group = metric.by_atom_type[atom_type]
            if component == "raw":
                return group.raw_mean_dw_first if species == "first" else group.raw_mean_dw_ca
            if component == "variance":
                return group.variance_first_from_origin if species == "first" else group.variance_ca_from_origin
            return group.mean_dw_first if species == "first" else group.mean_dw_ca

        if component == "raw":
            return metric.aggregate_raw_mean_dw_first if species == "first" else metric.aggregate_raw_mean_dw_ca
        if component == "variance":
            return (
                metric.aggregate_variance_first_from_origin
                if species == "first"
                else metric.aggregate_variance_ca_from_origin
            )
        return metric.aggregate_mean_dw_first if species == "first" else metric.aggregate_mean_dw_ca

    def _types_for_metric(metric: FileMetrics) -> list[str | None]:
        if not group_by_atom_type:
            return [None]
        ordered_types = []
        if "S" in metric.by_atom_type:
            ordered_types.append("S")
        if "N" in metric.by_atom_type:
            ordered_types.append("N")
        extras = sorted(t for t in metric.by_atom_type if t not in {"S", "N"})
        return ordered_types + extras

    def _type_color(atom_type: str | None, species: str) -> str:
        if atom_type == "N":
            return "#1f77b4"
        if atom_type == "S":
            return "#D4A017"
        return "#D4A017" if species == "first" else "#333333"

    def _type_shift(atom_type: str | None) -> float:
        if not group_by_atom_type:
            return 0.0
        if atom_type == "S":
            return -1.0 * atom_type_separation
        if atom_type == "N":
            return 0.0
        return 0.0

    def _draw_species_points(ax: plt.Axes, species: str, component: str) -> None:
        for items in by_prefix.values():
            ordered = sorted(items, key=lambda it: suffix_sort_key(it.suffix))
            n_items = len(ordered)
            for idx, m in enumerate(ordered):
                variant_shift = (idx - (n_items - 1) / 2.0) * variant_separation
                for atom_type in _types_for_metric(m):
                    base_x = m.volume + variant_shift + _type_shift(atom_type)
                    y_value = _metric_value(m, species, component, atom_type)
                    ax.scatter(
                        base_x,
                        y_value,
                        color=_type_color(atom_type, species),
                        marker="o",
                        edgecolors="none",
                        s=145,
                        alpha=0.7,
                        zorder=4,
                    )

    all_axes = [ax_raw_s, ax_var_s, ax_total_s]

    for ax in all_axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.2)
        for items in by_prefix.values():
            for m in items:
                ax.axvline(m.volume, color="#B0B0B0", linewidth=0.9, zorder=2)

    _draw_species_points(ax_raw_s, "first", "raw")
    _draw_species_points(ax_var_s, "first", "variance")
    _draw_species_points(ax_total_s, "first", "total")

    ordered_metric_lines: list[str] = []
    ordered_metrics = sorted(filtered, key=lambda m: (m.volume, m.prefix, suffix_sort_key(m.suffix)))
    for idx, metric in enumerate(ordered_metrics, start=1):
        suffix_label = metric.suffix if metric.suffix else "single point"
        ordered_metric_lines.append(
            f"{idx}. {metric.prefix} [{suffix_label}]\n"
            f"  V={metric.volume:.6f} $\AA^3$"
        )

    consistency_lines: list[str] = []
    for prefix, items in sorted(by_prefix.items()):
        volumes = [m.volume for m in items]
        ref = volumes[0]
        consistent = all(np.isclose(v, ref, rtol=0.0, atol=1e-12) for v in volumes[1:])
        status = "OK" if consistent else "MISMATCH"
        spread = max(volumes) - min(volumes)
        consistency_lines.append(f"{prefix}: {status} (spread={spread:.3e} " + r"$\AA^3$)")

    sequence_text = (
        "Order by tetrahedron volume\n"
        "(smallest to largest):\n"
        + "\n".join(ordered_metric_lines)
        # + "\n\nPrefix consistency check:\n"
        # + "\n".join(consistency_lines)
    )
    fig.text(
        0.75,
        0.95,
        sequence_text,
        ha="left",
        va="top",
        fontsize=18,
        # family="monospace",
    )

    def _collect_first_shell_values(component: str) -> list[float]:
        values: list[float] = []
        for m in filtered:
            for atom_type in _types_for_metric(m):
                values.append(_metric_value(m, "first", component, atom_type))
        return values

    def _auto_bottom_from_top(values: list[float], top: float) -> float | None:
        visible = [v for v in values if v <= top]
        if not visible:
            return None
        low = min(visible)
        high = max(visible)
        if high > low:
            pad = 0.08 * (high - low)
        else:
            scale = max(abs(low), abs(high), 1e-6)
            pad = 0.08 * scale
        return max(0.0, low - pad)

    variance_values = _collect_first_shell_values("variance")
    if variance_ymax is not None and variance_ymin is None:
        auto_bottom = _auto_bottom_from_top(variance_values, variance_ymax)
        if auto_bottom is None:
            ax_var_s.set_ylim(top=variance_ymax)
        else:
            ax_var_s.set_ylim(bottom=auto_bottom, top=variance_ymax)
    elif variance_ymax is not None:
        ax_var_s.set_ylim(bottom=variance_ymin, top=variance_ymax)
    elif variance_ymin is not None:
        ax_var_s.set_ylim(bottom=variance_ymin)

    total_values = _collect_first_shell_values("total")
    if total_ymax is not None and total_ymin is None:
        auto_bottom = _auto_bottom_from_top(total_values, total_ymax)
        if auto_bottom is None:
            ax_total_s.set_ylim(top=total_ymax)
        else:
            ax_total_s.set_ylim(bottom=auto_bottom, top=total_ymax)
    elif total_ymax is not None:
        ax_total_s.set_ylim(bottom=total_ymin, top=total_ymax)
    elif total_ymin is not None:
        ax_total_s.set_ylim(bottom=total_ymin)

    ax_var_s.tick_params(axis="x", which="both", labelbottom=False)
    ax_raw_s.tick_params(axis="x", which="both", labelbottom=False)

    ax_raw_s.set_title("First shell (S/N)")
    ax_raw_s.set_ylabel(r"DW from xyz ($\overline{\sigma^2_{FEFF}}$)")
    ax_var_s.set_ylabel(r"Variance ($\sigma^2_{static}$)")
    ax_total_s.set_ylabel(r"Total ($\overline{\sigma^2}$)")
    ax_total_s.set_xlabel(r"C$\alpha$-tetrahedron volume ($\AA^3$)")

    if group_by_atom_type:
        has_s = any("S" in m.by_atom_type for m in filtered)
        has_n = any("N" in m.by_atom_type for m in filtered)
        legend_s: list[Line2D] = []
        if has_s:
            legend_s.extend(
                [
                    Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#D4A017", markeredgecolor="none", markersize=11, alpha=0.7, label="S, single point"),
                ]
            )
        if has_n:
            legend_s.extend(
                [
                    Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#1f77b4", markeredgecolor="none", markersize=11, alpha=0.7, label="N, single point"),
                ]
            )
    else:
        legend_s = [
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#D4A017", markeredgecolor="none", markersize=11, alpha=0.7, label="First shell, single point"),
        ]

    if filtered:
        ax_total_s.legend(handles=legend_s, loc="best", frameon=True)

    fig.subplots_adjust(left=0.11, right=0.73, bottom=0.07, top=0.94, wspace=0.0, hspace=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_distance_plot(
    metrics: list[FileMetrics],
    *,
    out_path: Path,
    variant_separation: float,
    group_by_atom_type: bool,
    atom_type_separation: float,
) -> None:
    _apply_plot_rcparams(plt)
    fig, ax_s = plt.subplots(1, 1, figsize=(9.2, 5.8), sharey=False)

    filtered = [m for m in metrics if _include_dw_suffix(m.suffix)]
    by_prefix: dict[str, list[FileMetrics]] = {}
    for m in filtered:
        by_prefix.setdefault(m.prefix, []).append(m)

    ax_s.spines["top"].set_visible(False)
    ax_s.spines["right"].set_visible(False)
    ax_s.grid(alpha=0.2)

    def _types_for_metric(metric: FileMetrics) -> list[str | None]:
        if not group_by_atom_type:
            return [None]
        ordered_types = []
        if "S" in metric.by_atom_type:
            ordered_types.append("S")
        if "N" in metric.by_atom_type:
            ordered_types.append("N")
        extras = sorted(t for t in metric.by_atom_type if t not in {"S", "N"})
        return ordered_types + extras

    def _type_color(atom_type: str | None, species: str) -> str:
        if atom_type == "N":
            return "#1f77b4"
        if atom_type == "S":
            return "#D4A017"
        return "#D4A017" if species == "first" else "#333333"

    def _type_shift(atom_type: str | None) -> float:
        if not group_by_atom_type:
            return 0.0
        if atom_type == "S":
            return -1.0 * atom_type_separation
        if atom_type == "N":
            return 0.0
        return 0.0

    def _distance_pairs_for_type(
        metric: FileMetrics, atom_type: str | None
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        if atom_type is None:
            return metric.aggregate_first_distance_dw, metric.aggregate_ca_distance_dw

        group = metric.by_atom_type.get(atom_type)
        if group is None:
            return [], []
        return group.first_distance_dw, group.ca_distance_dw

    for items in by_prefix.values():
        ordered = sorted(items, key=lambda it: suffix_sort_key(it.suffix))
        n_items = len(ordered)
        for idx, m in enumerate(ordered):
            variant_shift = (idx - (n_items - 1) / 2.0) * variant_separation

            for atom_type in _types_for_metric(m):
                x_shift = variant_shift + _type_shift(atom_type)
                first_distance_dw, ca_distance_dw = _distance_pairs_for_type(m, atom_type)

                for distance, dw in first_distance_dw:
                    ax_s.scatter(
                        distance + x_shift,
                        dw,
                        color=_type_color(atom_type, "first"),
                        marker="o",
                        edgecolors="none",
                        s=145,
                        alpha=0.7,
                        zorder=4,
                    )

    ax_s.set_title("First shell (S/N)")
    ax_s.set_xlabel(r"Distance from origin ($\AA$)")
    ax_s.set_ylabel(r"Debye-Waller factor ($\sigma^2_{FEFF}$)")

    if group_by_atom_type:
        has_s = any("S" in m.by_atom_type for m in filtered)
        has_n = any("N" in m.by_atom_type for m in filtered)
        legend_s: list[Line2D] = []
        if has_s:
            legend_s.extend(
                [
                    Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#D4A017", markeredgecolor="none", markersize=11, alpha=0.7, label="S, single point"),
                ]
            )
        if has_n:
            legend_s.extend(
                [
                    Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#1f77b4", markeredgecolor="none", markersize=11, alpha=0.7, label="N, single point"),
                ]
            )
    else:
        legend_s = [
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#D4A017", markeredgecolor="none", markersize=11, alpha=0.7, label="First shell, single point"),
        ]

    if filtered:
        ax_s.legend(handles=legend_s, loc="best", frameon=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _canonical_symbol(symbol: str) -> str:
    s = symbol.strip()
    if not s:
        return s
    return s[0].upper() + s[1:].lower()


def parse_xyz(path: Path) -> tuple[list[str], list[XyzAtom]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"XYZ file too short: {path}")

    header = lines[:2]
    atoms: list[XyzAtom] = []
    for line in lines[2:]:
        stripped = line.strip()
        if not stripped:
            continue

        if "#" in stripped:
            left, right = stripped.split("#", 1)
            trailing_comment = right.strip()
        else:
            left, trailing_comment = stripped, ""

        parts = left.split()
        if len(parts) < 4:
            continue

        symbol = _canonical_symbol(parts[0])
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue

        atoms.append(XyzAtom(symbol=symbol, x=x, y=y, z=z, trailing_comment=trailing_comment))

    return header, atoms


def parse_dw(path: Path) -> list[DwAtom]:
    rows: list[DwAtom] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        parts = stripped.split()
        if len(parts) < 8:
            continue
        if parts[0].lower() == "group" and parts[1].lower() == "symbol":
            continue

        # Expected columns:
        # group symbol x y z distance sig2_tot atom_index
        symbol = _canonical_symbol(parts[1])
        try:
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            continue
        sig2_tot = parts[6]
        rows.append(DwAtom(symbol=symbol, x=x, y=y, z=z, sig2_tot=sig2_tot))

    if not rows:
        raise ValueError(f"No parseable DW rows in: {path}")

    return rows


def _parse_first_shell_distances_by_symbol(path: Path) -> dict[str, list[float]]:
    distances_by_symbol: dict[str, list[float]] = {"S": [], "N": []}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        parts = stripped.split()
        if len(parts) < 8:
            continue
        if parts[0].lower() == "group" and parts[1].lower() == "symbol":
            continue

        group = parts[0].strip().lower()
        if group != "nearest":
            continue

        symbol = parts[1].strip().upper()
        if symbol.startswith("N"):
            symbol_key = "N"
        elif symbol.startswith("S"):
            symbol_key = "S"
        else:
            continue

        try:
            distance = float(parts[5])
        except ValueError:
            continue
        distances_by_symbol[symbol_key].append(distance)

    return distances_by_symbol


def _parse_first_shell_dw_factors_by_symbol(path: Path) -> dict[str, list[float]]:
    dw_values_by_symbol: dict[str, list[float]] = {"S": [], "N": []}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        parts = stripped.split()
        if len(parts) < 8:
            continue
        if parts[0].lower() == "group" and parts[1].lower() == "symbol":
            continue

        group = parts[0].strip().lower()
        if group != "nearest":
            continue

        symbol = parts[1].strip().upper()
        if symbol.startswith("N"):
            symbol_key = "N"
        elif symbol.startswith("S"):
            symbol_key = "S"
        else:
            continue

        try:
            dw = float(parts[6])
        except ValueError:
            continue
        dw_values_by_symbol[symbol_key].append(dw)

    return dw_values_by_symbol


def center_on_zn(atoms: list[XyzAtom]) -> list[XyzAtom]:
    zn_atom = next((atom for atom in atoms if atom.symbol.upper() == "ZN"), None)
    if zn_atom is None:
        raise ValueError("No Zn atom found in XYZ")

    cx, cy, cz = zn_atom.x, zn_atom.y, zn_atom.z
    centered: list[XyzAtom] = []
    for atom in atoms:
        centered.append(
            XyzAtom(
                symbol=atom.symbol,
                x=atom.x - cx,
                y=atom.y - cy,
                z=atom.z - cz,
                trailing_comment=atom.trailing_comment,
            )
        )
    return centered


def _dist2(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return dx * dx + dy * dy + dz * dz


def _match_first_shell_n_to_labels(
    _path: Path, first_shell_atoms: list[DwTaggedAtom], tolerance: float = 5.0e-3
) -> list[str]:
    _ = tolerance
    labels: list[str] = []
    for atom in first_shell_atoms:
        symbol = atom.symbol.strip().upper()
        if symbol.startswith("N"):
            labels.append("N")
        elif symbol.startswith("S"):
            labels.append("S")
        else:
            labels.append(symbol)
    return labels


def match_dw(
    centered_atoms: list[XyzAtom], dw_atoms: list[DwAtom], tolerance: float
) -> list[str | None]:
    remaining: dict[str, list[DwAtom]] = {}
    for dw in dw_atoms:
        remaining.setdefault(dw.symbol, []).append(dw)

    annotations: list[str | None] = []
    tol2 = tolerance * tolerance

    for atom in centered_atoms:
        candidates = remaining.get(atom.symbol, [])
        best_i = -1
        best_d2 = float("inf")
        for i, cand in enumerate(candidates):
            d2 = _dist2((atom.x, atom.y, atom.z), (cand.x, cand.y, cand.z))
            if d2 < best_d2:
                best_d2 = d2
                best_i = i

        if best_i >= 0 and best_d2 <= tol2:
            chosen = candidates.pop(best_i)
            annotations.append(chosen.sig2_tot)
        else:
            annotations.append(None)

    return annotations


def format_xyz_lines(header: list[str], centered_atoms: list[XyzAtom], dw_values: list[str | None]) -> str:
    out_lines = [str(len(centered_atoms)), header[1]]

    for atom, dw_val in zip(centered_atoms, dw_values):
        base = f"{atom.symbol:<2} {atom.x:16.8f} {atom.y:16.8f} {atom.z:16.8f}"
        comments: list[str] = []
        if atom.trailing_comment:
            comments.append(atom.trailing_comment)
        if dw_val is not None:
            comments.append(f"dw={dw_val}")

        if comments:
            out_lines.append(f"{base} # {' '.join(comments)}")
        else:
            out_lines.append(base)

    return "\n".join(out_lines) + "\n"


def find_dw_file(child_dir: Path) -> Path:
    matches = sorted(child_dir.glob("dw*.dat"))
    if not matches:
        raise FileNotFoundError(f"No dw*.dat found in {child_dir}")
    if len(matches) > 1:
        print(f"[warn] Multiple dw files in {child_dir}; using {matches[0].name}")
    return matches[0]


def process_child_dir(child_dir: Path, tolerance: float) -> tuple[int, int]:
    dw_path = find_dw_file(child_dir)
    dw_atoms = parse_dw(dw_path)

    xyz_files = sorted(
        p for p in child_dir.glob("*.xyz") if not p.name.endswith("_w_dw.xyz")
    )
    if not xyz_files:
        print(f"[skip] No .xyz files in {child_dir}")
        return 0, 0

    written = 0
    matched_total = 0
    for xyz_path in xyz_files:
        header, atoms = parse_xyz(xyz_path)
        centered = center_on_zn(atoms)
        dw_values = match_dw(centered, dw_atoms, tolerance=tolerance)
        matched = sum(1 for v in dw_values if v is not None)

        out_path = xyz_path.with_name(f"{xyz_path.stem}_w_dw.xyz")
        out_text = format_xyz_lines(header, centered, dw_values)
        out_path.write_text(out_text, encoding="utf-8")

        written += 1
        matched_total += matched
        print(
            f"[ok] {xyz_path.name} -> {out_path.name} "
            f"(matched {matched}/{len(centered)} atoms using {dw_path.name})"
        )

    return written, matched_total


def iter_child_dirs(parent_dir: Path) -> list[Path]:
    return sorted(p for p in parent_dir.iterdir() if p.is_dir())


def generate_dw_figures(
    parent_dir: Path,
    *,
    variant_separation: float,
    group_by_atom_type: bool,
    variance_ymin: float | None,
    variance_ymax: float | None,
    total_ymin: float | None,
    total_ymax: float | None,
    atom_type_separation: float,
) -> tuple[Path, Path, int]:
    cropped = any(v is not None for v in (variance_ymin, variance_ymax, total_ymin, total_ymax))
    name_prefix = "cropped-" if cropped else ""
    dir_suffix = parent_dir.name.replace(" ", "")
    out_path = parent_dir / f"{name_prefix}dw_vs_tetra_volume-{dir_suffix}.png"
    out_distance_path = parent_dir / f"dw_vs_distance-{dir_suffix}.png"

    paths = collect_w_dw_xyz(parent_dir)
    if not paths:
        raise ValueError(f"No *_w_dw.xyz files found under: {parent_dir}")

    metrics = []
    skipped = 0
    for path in paths:
        try:
            metrics.append(metrics_from_file(path))
        except ValueError as exc:
            skipped += 1
            print(f"[skip] {path}: {exc}")

    if not metrics:
        raise ValueError("No valid *_w_dw.xyz files for plotting.")

    make_plot(
        metrics,
        out_path=out_path,
        variant_separation=variant_separation,
        group_by_atom_type=group_by_atom_type,
        variance_ymin=variance_ymin,
        variance_ymax=variance_ymax,
        total_ymin=total_ymin,
        total_ymax=total_ymax,
        atom_type_separation=atom_type_separation,
    )
    make_distance_plot(
        metrics,
        out_path=out_distance_path,
        variant_separation=variant_separation,
        group_by_atom_type=group_by_atom_type,
        atom_type_separation=atom_type_separation,
    )
    return out_path, out_distance_path, skipped


def generate_chi_figure(
    parent_dir: Path,
    patterns: list[str],
    exp_data_paths: list[Path] | None = None,
) -> Path:
    data_files = _iter_data_files(parent_dir, patterns)
    if not data_files:
        raise ValueError(f"No chi_R*.dat files found under {parent_dir}")

    loaded_exp_spectra: list[tuple[str, np.ndarray, np.ndarray]] = []
    for exp_path in exp_data_paths or []:
        resolved_exp_path = _resolve_exp_data_path(exp_path, parent_dir)
        if not resolved_exp_path.exists():
            print(f"[skip] experimental chi(R) file not found: {exp_path}")
            continue

        try:
            r_vals, chir_mag_vals = _load_r_chir_mag(resolved_exp_path)
        except ValueError as exc:
            print(f"[skip] experimental chi(R) file {resolved_exp_path}: {exc}")
            continue

        loaded_exp_spectra.append((_exp_legend_label(resolved_exp_path), r_vals, chir_mag_vals))

    avg_exafs_k_chi_r = _build_avg_exafs_k_chi_r(parent_dir)

    _apply_plot_rcparams(plt)
    fig, (all_ax, avg_ax) = plt.subplots(1, 2, figsize=(13.5, 5.0), sharey=False)

    loaded_spectra: list[tuple[np.ndarray, np.ndarray, Path]] = []
    for data_path in sorted(data_files):
        try:
            r_vals, chir_mag_vals = _load_r_chir_mag(data_path)
        except ValueError as exc:
            print(f"[skip] {data_path}: {exc}")
            continue
        loaded_spectra.append((r_vals, chir_mag_vals, data_path))

    if not loaded_spectra:
        raise ValueError("No valid chi(R) spectra could be loaded.")

    for ax in (all_ax, avg_ax):
        ax.set_axisbelow(True)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel(r"R ($\AA$)")
        ax.set_ylabel(r"$|\chi(R)|$")
        ax.set_xlim(0.0, 6.0)

    for r_vals, chir_mag_vals, _ in loaded_spectra:
        all_ax.plot(r_vals, chir_mag_vals, color="#808080", linewidth=0.7, alpha=0.8)
    all_ax.set_title("All spectra")

    # avg_r = loaded_spectra[0][0]
    # aligned = [np.interp(avg_r, r_vals, chir_mag_vals) for r_vals, chir_mag_vals, _ in loaded_spectra]
    # avg_chir = np.mean(np.vstack(aligned), axis=0)
    # avg_ax.plot(avg_r, avg_chir * 0.9, color="black", linewidth=1.8, label="Model average * 0.9")

    if avg_exafs_k_chi_r is not None:
        avg_k_r, avg_k_chir_mag = avg_exafs_k_chi_r
        avg_ax.plot(
            avg_k_r,
            avg_k_chir_mag,
            color="#2E8B57",
            linewidth=2.1,
            alpha=0.95,
            label="Avg exafs-k -> chi(R)",
        )

    for i, (label, r_vals, chir_mag_vals) in enumerate(loaded_exp_spectra):
        if i % 2 == 1:
            color = "salmon"
        else:
            color = "#3780C8"
        avg_ax.plot(r_vals, chir_mag_vals, color=color, linewidth=2.1, alpha=0.95, label=f"{label}")

    if loaded_exp_spectra:
        avg_ax.legend(loc="best", fontsize=12)

    avg_ax.set_title("Average spectrum")

    fig.suptitle(parent_dir.name, fontsize=20)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    default_name = f"EXAFS-{parent_dir.name}".replace(" ", "")
    out_path = parent_dir / f"{default_name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return out_path


def generate_first_shell_distance_figure(parent_dir: Path) -> tuple[Path, int]:
    by_symbol: dict[str, list[tuple[str, list[float]]]] = {"S": [], "N": []}
    skipped = 0

    for child_dir in iter_child_dirs(parent_dir):
        try:
            dw_path = find_dw_file(child_dir)
        except FileNotFoundError:
            skipped += 1
            continue

        distances_by_symbol = _parse_first_shell_distances_by_symbol(dw_path)
        if not distances_by_symbol["S"] and not distances_by_symbol["N"]:
            skipped += 1
            print(f"[skip] {dw_path}: no first-shell S/N (group=nearest) distances found")
            continue

        label = child_dir.name
        for symbol in ("S", "N"):
            if distances_by_symbol[symbol]:
                by_symbol[symbol].append((label, distances_by_symbol[symbol]))

    if not by_symbol["S"] and not by_symbol["N"]:
        raise ValueError("No first-shell distance summaries could be built from dw*.dat files.")

    _apply_plot_rcparams(plt)
    fig, (s_ax, n_ax) = plt.subplots(1, 2, figsize=(13.6, 10.8), sharey=False)

    def _plot_symbol_panel(ax: plt.Axes, symbol: str, color: str, title: str) -> None:
        entries = by_symbol[symbol]
        x_vals = list(range(len(entries)))
        for x, (_, values) in zip(x_vals, entries):
            ax.scatter(
                [x + 0.05] * len(values),
                values,
                marker="o",
                color=color,
                edgecolors="none",
                s=58,
                alpha=0.5,
                zorder=1,
            )
            ax.errorbar(
                x,
                float(np.mean(values)),
                yerr=float(np.std(values)),
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.8,
                capsize=4,
                markersize=8,
                alpha=0.9,
                zorder=4,
            )

        ax.set_title(title)
        ax.set_xticks(x_vals)
        ax.set_xticklabels([label for label, _ in entries], rotation=65, ha="right")
        ax.set_xlabel("Filename")
        ax.set_ylabel(r"First-shell distance ($\AA$)")
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if x_vals:
            ax.set_xlim(-0.5, len(x_vals) - 0.5)

    _plot_symbol_panel(s_ax, "S", "#D4A017", "S distances")
    _plot_symbol_panel(n_ax, "N", "#1f77b4", "N distances")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    dir_suffix = parent_dir.name.replace(" ", "")
    out_path = parent_dir / f"dw_first_shell_distance_grid-{dir_suffix}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return out_path, skipped


def generate_first_shell_dw_figure(parent_dir: Path) -> tuple[Path, int]:
    by_symbol: dict[str, list[tuple[str, list[float]]]] = {"S": [], "N": []}
    skipped = 0

    for child_dir in iter_child_dirs(parent_dir):
        try:
            dw_path = find_dw_file(child_dir)
        except FileNotFoundError:
            skipped += 1
            continue

        dw_values_by_symbol = _parse_first_shell_dw_factors_by_symbol(dw_path)
        if not dw_values_by_symbol["S"] and not dw_values_by_symbol["N"]:
            skipped += 1
            print(f"[skip] {dw_path}: no first-shell S/N (group=nearest) DW factors found")
            continue

        label = child_dir.name
        for symbol in ("S", "N"):
            if dw_values_by_symbol[symbol]:
                by_symbol[symbol].append((label, dw_values_by_symbol[symbol]))

    if not by_symbol["S"] and not by_symbol["N"]:
        raise ValueError("No first-shell DW-factor summaries could be built from dw*.dat files.")

    _apply_plot_rcparams(plt)
    fig, (s_ax, n_ax) = plt.subplots(1, 2, figsize=(13.6, 10.8), sharey=False)

    def _plot_symbol_panel(ax: plt.Axes, symbol: str, color: str, title: str) -> None:
        entries = by_symbol[symbol]
        x_vals = list(range(len(entries)))
        for x, (_, values) in zip(x_vals, entries):
            ax.scatter(
                [x + 0.05] * len(values),
                values,
                marker="o",
                color=color,
                edgecolors="none",
                s=58,
                alpha=0.5,
                zorder=1,
            )
            ax.errorbar(
                x,
                float(np.mean(values)),
                yerr=float(np.std(values)),
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.8,
                capsize=4,
                markersize=8,
                alpha=0.9,
                zorder=4,
            )

        ax.set_title(title)
        ax.set_xticks(x_vals)
        ax.set_xticklabels([label for label, _ in entries], rotation=65, ha="right")
        ax.set_xlabel("Filename")
        ax.set_ylabel(r"First-shell DW factor ($\sigma^2_{FEFF}$)")
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if x_vals:
            ax.set_xlim(-0.5, len(x_vals) - 0.5)

    _plot_symbol_panel(s_ax, "S", "#D4A017", "S DW factors")
    _plot_symbol_panel(n_ax, "N", "#1f77b4", "N DW factors")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    dir_suffix = parent_dir.name.replace(" ", "")
    out_path = parent_dir / f"dw_first_shell_dw_grid-{dir_suffix}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return out_path, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "For each direct child directory, center XYZ files on Zn and annotate "
            "atoms matched to dw*.dat rows with dw=<sig2_tot>, then generate all figures."
        )
    )
    parser.add_argument("parent_dir", type=Path, help="Parent directory containing child dirs")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=2.0e-3,
        help="Coordinate match tolerance in angstrom (default: 0.002)",
    )
    parser.add_argument(
        "--variant-separation",
        type=float,
        default=0.08,
        help="Horizontal separation between suffix variants in DW figures (default: 0.08)",
    )
    parser.add_argument(
        "--atom-type-separation",
        type=float,
        default=0.03,
        help="Horizontal separation between S and N grouped points (default: 0.03)",
    )
    parser.add_argument(
        "--group-by-atom-type",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Group first-shell and matched Cα metrics by coordinating atom type (S vs N).",
    )
    parser.add_argument(
        "--variance-ymin",
        type=float,
        default=None,
        help="Optional lower y-limit for first-shell variance panel (left column only).",
    )
    parser.add_argument(
        "--variance-ymax",
        type=float,
        default=None,
        help="Optional upper y-limit for first-shell variance panel (left column only).",
    )
    parser.add_argument(
        "--total-ymin",
        type=float,
        default=None,
        help="Optional lower y-limit for first-shell total-disorder panel (left column only).",
    )
    parser.add_argument(
        "--total-ymax",
        type=float,
        default=None,
        help="Optional upper y-limit for first-shell total-disorder panel (left column only).",
    )
    parser.add_argument(
        "--chi-pattern",
        action="append",
        default=["chi_R*.dat", "chi-R*.dat"],
        help="Glob pattern to match chi(R) files for EXAFS figure (repeatable).",
    )
    parser.add_argument(
        "--exp-data",
        action="append",
        type=Path,
        default=[],
        help=(
            "Path to an experimental chi(R) data file to overlay on the EXAFS average-spectrum subplot. "
            "Repeatable."
        ),
    )

    args = parser.parse_args()
    parent_dir: Path = args.parent_dir
    tolerance: float = args.tolerance

    if not parent_dir.is_dir():
        raise SystemExit(f"Parent directory not found: {parent_dir}")
    if tolerance <= 0:
        raise SystemExit("--tolerance must be > 0")
    if args.variant_separation < 0:
        raise SystemExit("--variant-separation must be >= 0")
    if args.atom_type_separation < 0:
        raise SystemExit("--atom-type-separation must be >= 0")
    if args.variance_ymin is not None and args.variance_ymax is not None and args.variance_ymin >= args.variance_ymax:
        raise SystemExit("--variance-ymin must be < --variance-ymax")
    if args.variance_ymax is not None and args.variance_ymax <= 0:
        raise SystemExit("--variance-ymax must be > 0")
    if args.total_ymin is not None and args.total_ymax is not None and args.total_ymin >= args.total_ymax:
        raise SystemExit("--total-ymin must be < --total-ymax")
    if args.total_ymax is not None and args.total_ymax <= 0:
        raise SystemExit("--total-ymax must be > 0")

    child_dirs = iter_child_dirs(parent_dir)
    if not child_dirs:
        raise SystemExit(f"No child directories found in {parent_dir}")

    processed_dirs = 0
    written_files = 0
    matched_atoms = 0

    for child_dir in child_dirs:
        try:
            written, matched = process_child_dir(child_dir, tolerance=tolerance)
        except FileNotFoundError:
            print(f"[skip] {child_dir}: missing dw*.dat")
            continue
        except ValueError as exc:
            print(f"[skip] {child_dir}: {exc}")
            continue

        if written > 0:
            processed_dirs += 1
            written_files += written
            matched_atoms += matched

    print(
        f"Done. Processed {processed_dirs} child dirs, wrote {written_files} file(s), "
        f"matched {matched_atoms} atom annotation(s)."
    )

    try:
        dw_volume_path, dw_distance_path, dw_skipped = generate_dw_figures(
            parent_dir,
            variant_separation=args.variant_separation,
            group_by_atom_type=args.group_by_atom_type,
            variance_ymin=args.variance_ymin,
            variance_ymax=args.variance_ymax,
            total_ymin=args.total_ymin,
            total_ymax=args.total_ymax,
            atom_type_separation=args.atom_type_separation,
        )
        print(f"[ok] Wrote figure: {dw_volume_path}")
        print(f"[ok] Wrote figure: {dw_distance_path}")
        print(f"[ok] DW figure metrics skipped {dw_skipped} file(s).")
    except ValueError as exc:
        print(f"[skip] DW figures: {exc}")

    try:
        chi_path = generate_chi_figure(
            parent_dir,
            patterns=args.chi_pattern,
            exp_data_paths=args.exp_data,
        )
        print(f"[ok] Wrote figure: {chi_path}")
    except ValueError as exc:
        print(f"[skip] chi(R) figure: {exc}")

    try:
        first_shell_path, distance_skipped = generate_first_shell_distance_figure(parent_dir)
        print(f"[ok] Wrote figure: {first_shell_path}")
        print(f"[ok] First-shell distance figure skipped {distance_skipped} folder(s).")
    except ValueError as exc:
        print(f"[skip] first-shell distance figure: {exc}")

    try:
        first_shell_dw_path, dw_factor_skipped = generate_first_shell_dw_figure(parent_dir)
        print(f"[ok] Wrote figure: {first_shell_dw_path}")
        print(f"[ok] First-shell DW-factor figure skipped {dw_factor_skipped} folder(s).")
    except ValueError as exc:
        print(f"[skip] first-shell DW-factor figure: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
