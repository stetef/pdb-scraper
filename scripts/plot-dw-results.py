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
METAL_TOKEN_SET = {"zn", "fe", "cu", "co", "ni", "mn", "mg"}


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


##############################
# File Naming and Matching
# Canonical parsing for prefix/suffix/variant labels used by all outputs.
##############################


@dataclass(frozen=True)
class NameParts:
    prefix: str
    suffix: str
    variant_label: str
    pdb_id: str | None


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


def _normalized_stem(path: Path) -> str:
    stem = path.stem
    for lead in ("chi-R-", "chi_R", "chi-R", "chiR-"):
        if stem.startswith(lead):
            stem = stem[len(lead) :]
            break
    return stem


def _group_data_files(paths: list[Path]) -> list[tuple[str, list[Path], list[str]]]:
    normalized_by_path: dict[Path, str] = {path: _normalized_stem(path) for path in paths}
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        stem = normalized_by_path[path]
        prefix, _ = split_prefix_suffix(stem)
        grouped.setdefault(prefix, []).append(path)

    groups: list[tuple[str, list[Path], list[str]]] = []
    for prefix in sorted(grouped):
        paths_for_group = sorted(grouped[prefix], key=lambda p: p.as_posix())
        display_labels: list[str] = []
        for path in paths_for_group:
            stem = normalized_by_path[path]
            _, suffix = split_prefix_suffix(stem)
            display_labels.append(_legend_label_from_suffix(suffix))
        groups.append((prefix, paths_for_group, display_labels))

    return groups


# ---------------------------------------------------------------------------
# Central data registry — built once, consumed by all plot generators
# ---------------------------------------------------------------------------

@dataclass
class ChiEntry:
    """One chi(R) data file with its resolved prefix/suffix/label."""
    path: Path
    prefix: str
    suffix: str
    label: str  # display label for legend
    r_vals: "np.ndarray"
    chir_mag_vals: "np.ndarray"
    is_experiment: bool = False


@dataclass
class SourceRegistry:
    """All parsed data, grouped by (prefix, suffix), built once in main()."""
    # chi(R) entries (model + experiment)
    chi_entries: list[ChiEntry]

    # FileMetrics from *_w_dw.xyz (used by DW vs volume / distance plots)
    file_metrics: list[FileMetrics]
    file_metrics_skipped: int

    # First-shell distance summaries (from dw*.dat in child dirs + --exp-data)
    distance_summaries: list[DwDistanceSummary]
    distance_skipped: int

    # First-shell DW-factor summaries (from dw*.dat in child dirs + --exp-data)
    dw_factor_summaries: list[DwFactorSummary]
    dw_factor_skipped: int


def _prefix_from_exp_path(path: Path) -> str:
    """Derive grouping prefix from the containing directory name."""
    return parse_name_parts(path.parent.name).prefix


def _name_parts_for_chi_model_path(path: Path, parent_dir: Path) -> NameParts:
    """Resolve model chi naming from the containing directory.

    This keeps EXAFS matching identical to first-shell/DW matching even when
    filenames include extra tokens or omit identifiers.
    """
    if path.parent == parent_dir:
        return parse_name_parts(_normalized_stem(path))
    return parse_name_parts(path.parent.name)


def build_registry(
    parent_dir: Path,
    chi_patterns: list[str],
    exp_data_paths: list[Path],
) -> SourceRegistry:
    """Scan *parent_dir* once and build SourceRegistry consumed by all generators."""

    # ------------------------------------------------------------------ chi(R)
    chi_entries: list[ChiEntry] = []

    # Model chi(R) files
    model_chi_paths = _iter_data_files(parent_dir, chi_patterns)
    seen_model_variant_keys: set[tuple[str, str]] = set()
    for path in model_chi_paths:
        parts = _name_parts_for_chi_model_path(path, parent_dir)
        variant_key = _normalize_variant_key(parts.variant_label)
        dedupe_key = (parts.prefix, variant_key)
        if dedupe_key in seen_model_variant_keys:
            print(f"[skip] duplicate chi(R) model variant for {parts.prefix}: {path.name}")
            continue
        try:
            r_vals, chir_mag_vals = _load_r_chir_mag(path)
        except ValueError as exc:
            print(f"[skip] chi(R) model file {path}: {exc}")
            continue
        seen_model_variant_keys.add(dedupe_key)
        chi_entries.append(ChiEntry(
            path=path, prefix=parts.prefix, suffix=parts.suffix, label=parts.variant_label,
            r_vals=r_vals, chir_mag_vals=chir_mag_vals, is_experiment=False,
        ))

    # Experimental chi(R) files — match to subplot prefix via split_prefix_suffix
    for raw_path in exp_data_paths or []:
        resolved = _resolve_exp_data_path(raw_path, parent_dir)
        if not resolved.exists():
            print(f"[skip] experimental chi(R) file not found: {raw_path}")
            continue
        prefix = _prefix_from_exp_path(resolved)
        label = _exp_legend_label(resolved)
        try:
            r_vals, chir_mag_vals = _load_r_chir_mag(resolved)
        except ValueError as exc:
            print(f"[skip] experimental chi(R) file {resolved}: {exc}")
            continue
        chi_entries.append(ChiEntry(
            path=resolved, prefix=prefix, suffix="", label=f"exp {label}",
            r_vals=r_vals, chir_mag_vals=chir_mag_vals, is_experiment=True,
        ))

    # -------------------------------------------------------- FileMetrics (*_w_dw.xyz)
    w_dw_paths = collect_w_dw_xyz(parent_dir)
    file_metrics: list[FileMetrics] = []
    file_metrics_skipped = 0
    for path in w_dw_paths:
        try:
            file_metrics.append(metrics_from_file(path))
        except ValueError as exc:
            file_metrics_skipped += 1
            print(f"[skip] {path}: {exc}")

    # ------------------------------------------------- First-shell summaries
    distance_summaries: list[DwDistanceSummary] = []
    dw_factor_summaries: list[DwFactorSummary] = []
    distance_skipped = 0
    dw_factor_skipped = 0

    for child_dir in iter_child_dirs(parent_dir):
        try:
            dw_path = find_dw_file(child_dir)
        except FileNotFoundError:
            distance_skipped += 1
            dw_factor_skipped += 1
            continue

        prefix, suffix = split_prefix_suffix(child_dir.name)

        distances = _parse_first_shell_distances(dw_path)
        if not distances:
            distance_skipped += 1
            print(f"[skip] {dw_path}: no first-shell (group=nearest) distances found")
        else:
            distance_summaries.append(DwDistanceSummary(
                prefix=prefix, suffix=suffix,
                mean_distance=float(np.mean(distances)),
                std_distance=float(np.std(distances)),
                n_points=len(distances),
                distances=list(distances),
                source_dir=child_dir,
            ))

        dw_values = _parse_first_shell_dw_factors(dw_path)
        if not dw_values:
            dw_factor_skipped += 1
            print(f"[skip] {dw_path}: no first-shell (group=nearest) DW factors found")
        else:
            dw_factor_summaries.append(DwFactorSummary(
                prefix=prefix, suffix=suffix,
                mean_dw=float(np.mean(dw_values)),
                std_dw=float(np.std(dw_values)),
                n_points=len(dw_values),
                dw_values=list(dw_values),
                source_dir=child_dir,
            ))

    # ------------------------------- Experimental dw*.dat files from --exp-data
    # If an --exp-data path points at a dw*.dat-style file rather than a chi(R)
    # file, include it in the first-shell summaries too, matched by prefix.
    for raw_path in exp_data_paths or []:
        resolved = _resolve_exp_data_path(raw_path, parent_dir)
        if not resolved.exists():
            continue
        # Only treat as dw*.dat if it starts with "dw"
        if not resolved.name.lower().startswith("dw"):
            continue
        exp_parts = parse_name_parts(resolved.parent.name)
        distances = _parse_first_shell_distances(resolved)
        if distances:
            distance_summaries.append(DwDistanceSummary(
                prefix=exp_parts.prefix, suffix=exp_parts.suffix,
                mean_distance=float(np.mean(distances)),
                std_distance=float(np.std(distances)),
                n_points=len(distances),
                distances=list(distances),
                source_dir=resolved.parent,
            ))
        dw_values_exp = _parse_first_shell_dw_factors(resolved)
        if dw_values_exp:
            dw_factor_summaries.append(DwFactorSummary(
                prefix=exp_parts.prefix, suffix=exp_parts.suffix,
                mean_dw=float(np.mean(dw_values_exp)),
                std_dw=float(np.std(dw_values_exp)),
                n_points=len(dw_values_exp),
                dw_values=list(dw_values_exp),
                source_dir=resolved.parent,
            ))

    return SourceRegistry(
        chi_entries=chi_entries,
        file_metrics=file_metrics,
        file_metrics_skipped=file_metrics_skipped,
        distance_summaries=distance_summaries,
        distance_skipped=distance_skipped,
        dw_factor_summaries=dw_factor_summaries,
        dw_factor_skipped=dw_factor_skipped,
    )


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
    normalized = base.replace("_", "-")
    for suffix in ("h-only-w-n", "h-only", "w-n", "ca-fixed"):
        tag = f"-{suffix}"
        if normalized.lower().endswith(tag):
            root = normalized[: -len(tag)].strip("-_")
            root_tokens = [t for t in re.findall(r"[A-Za-z0-9]+", root) if t]
            return "-".join(root_tokens), suffix.replace("-", "_")
    root_tokens = [t for t in re.findall(r"[A-Za-z0-9]+", normalized) if t]
    return "-".join(root_tokens), ""


def parse_name_parts(stem: str) -> NameParts:
    base = stem
    if base.endswith("_w_dw"):
        base = base[: -len("_w_dw")]

    tokens = [
        token
        for token in re.findall(r"[A-Za-z0-9]+", base)
        if token and not PLACEHOLDER_ID_PATTERN.fullmatch(token)
    ]
    if not tokens:
        return NameParts(prefix=base, suffix="", variant_label="CA_fixed", pdb_id=None)

    id_idx = next((idx for idx, token in enumerate(tokens) if _is_pdb_id_token(token)), None)
    if id_idx is not None:
        end_idx = id_idx
        if id_idx + 1 < len(tokens) and CLUSTER_TOKEN_PATTERN.fullmatch(tokens[id_idx + 1]):
            end_idx = id_idx + 1
        if end_idx + 1 < len(tokens) and tokens[end_idx + 1].lower() in METAL_TOKEN_SET:
            end_idx = end_idx + 1

        prefix = "-".join(tokens[: end_idx + 1])
        suffix = "_".join(tokens[end_idx + 1 :]).strip("_")
        variant_label = _legend_label_from_suffix(suffix)
        return NameParts(prefix=prefix, suffix=suffix, variant_label=variant_label, pdb_id=_extract_id_token(prefix))

    prefix, suffix = _split_known_variant_suffix("-".join(tokens))
    return NameParts(
        prefix=prefix,
        suffix=suffix,
        variant_label=_legend_label_from_suffix(suffix),
        pdb_id=_extract_id_token(prefix),
    )


def _normalize_variant_key(value: str) -> str:
    return value.strip().replace("_", "-").lower()


def _legend_label_from_suffix(suffix: str) -> str:
    key = _normalize_variant_key(suffix)
    if key in {"", "ca-fixed"}:
        return "CA_fixed"
    if key == "h-only":
        return "H-only"
    if key == "h-only-w-n":
        return "H-only-w-n"
    return suffix


def _subplot_title_from_prefix(prefix: str) -> str:
    pdb_id = parse_name_parts(prefix).pdb_id
    if pdb_id is not None:
        return pdb_id
    return prefix


def _exp_legend_label(path: Path) -> str:
    label = path.stem
    match = re.match(r"^(.+?)(?:[_-]exp)(?:$|[_-].*)", label, flags=re.IGNORECASE)
    if match is not None:
        trimmed = match.group(1).strip("_-")
        if trimmed:
            return trimmed
    return label


def _exp_match_key_from_filename(path: Path) -> str | None:
    stem = path.stem
    match = re.match(r"^(.+?)(?:[_-]exp)(?:$|[_-].*)", stem, flags=re.IGNORECASE)
    if match is not None:
        key = match.group(1).strip("_-")
        if key:
            return key.upper()

    key = _extract_id_token(stem)
    if key is not None:
        return key
    return None


def _model_prefix_match_keys(prefix: str) -> set[str]:
    keys: set[str] = set()
    id_token = _extract_id_token(prefix)
    if id_token is not None:
        keys.add(id_token)

    tokens = [t for t in re.findall(r"[A-Za-z0-9]+", prefix) if t]
    if any(t.lower() == "cons" for t in tokens):
        keys.add("CONS")

    return keys


def _resolve_exp_data_path(exp_path: Path, parent_dir: Path) -> Path:
    if exp_path.exists():
        return exp_path
    if not exp_path.is_absolute():
        candidate = parent_dir / exp_path
        if candidate.exists():
            return candidate
    return exp_path


def _label_color(label: str, fallback_index: int) -> str:
    normalized = _normalize_variant_key(label)
    if normalized == "h-only":
        return "#D3D3D3"
    if normalized in {"", "ca-fixed"}:
        return "#A7C7E7"
    fallback_colors = ["black", "#C27BA0", "#76A5AF", "#F6B26B"]
    return fallback_colors[fallback_index % len(fallback_colors)]


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
    parts = parse_name_parts(stem)
    return parts.prefix, parts.suffix


def suffix_sort_key(suffix: str) -> tuple[int, str]:
    normalized = _normalize_variant_key(suffix)
    if normalized.startswith("h-only"):
        return (0, "")
    if normalized == "ca-fixed":
        return (1, "")
    if normalized == "":
        return (1, "")
    return (2, normalized)


def _include_dw_suffix(suffix: str) -> bool:
    normalized = _normalize_variant_key(suffix)
    return normalized in {"", "ca-fixed", "h-only", "w-n", "h-only-w-n"}


def _is_h_only_variant(suffix: str) -> bool:
    return _normalize_variant_key(suffix).startswith("h-only")


def _build_suffix_marker_map(suffixes: Iterable[str]) -> dict[str, str]:
    marker_cycle = ["o", "s", "^", "v", "D", "P", "X", "<", ">", "*", "h", "8", "p"]
    unique_keys = sorted({_normalize_variant_key(suffix) for suffix in suffixes}, key=suffix_sort_key)
    return {key: marker_cycle[idx % len(marker_cycle)] for idx, key in enumerate(unique_keys)}


def _marker_for_suffix(suffix: str, marker_map: dict[str, str]) -> str:
    return marker_map.get(_normalize_variant_key(suffix), "o")


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
    if not first_shell_atoms:
        raise ValueError(
            "missing dw notation for first-shell S/N atoms "
            "(expected at least 1 S/N atom with dw=<value>, found 0)"
        )
    first_shell_type_labels = _match_first_shell_n_to_labels(path, first_shell_atoms)

    paired_ca: list[DwTaggedAtom] = []
    if len(ca_atoms) >= len(first_shell_atoms):
        paired_ca = _assign_ca_to_first_shell(first_shell_atoms, ca_atoms)

    type_buckets: dict[str, dict[str, list[DwTaggedAtom]]] = {}
    for first_atom, atom_type in zip(first_shell_atoms, first_shell_type_labels):
        bucket = type_buckets.setdefault(atom_type, {"first": [], "ca": []})
        bucket["first"].append(first_atom)
    for ca_atom, atom_type in zip(paired_ca, first_shell_type_labels):
        bucket = type_buckets.setdefault(atom_type, {"first": [], "ca": []})
        bucket["ca"].append(ca_atom)

    if len(ca_atoms) >= 4:
        c_points = [np.asarray((a.x, a.y, a.z), dtype=float) for a in ca_atoms[:4]]
        volume = tetrahedron_volume(c_points)
    else:
        volume = float("nan")

    first_distances_from_origin = [float(np.linalg.norm((a.x, a.y, a.z))) for a in first_shell_atoms]
    ca_distances_from_origin = [float(np.linalg.norm((a.x, a.y, a.z))) for a in paired_ca]
    aggregate_first_distance_dw = list(zip(first_distances_from_origin, [a.dw for a in first_shell_atoms]))
    aggregate_ca_distance_dw = list(zip(ca_distances_from_origin, [a.dw for a in paired_ca]))
    aggregate_variance_first_from_origin = float(np.var(first_distances_from_origin))
    aggregate_variance_ca_from_origin = float(np.var(ca_distances_from_origin)) if ca_distances_from_origin else float("nan")
    aggregate_raw_mean_dw_first = float(np.mean([a.dw for a in first_shell_atoms]))
    aggregate_raw_mean_dw_ca = float(np.mean([a.dw for a in paired_ca])) if paired_ca else float("nan")
    aggregate_mean_dw_first = aggregate_raw_mean_dw_first + aggregate_variance_first_from_origin
    aggregate_mean_dw_ca = (
        aggregate_raw_mean_dw_ca + aggregate_variance_ca_from_origin
        if np.isfinite(aggregate_raw_mean_dw_ca) and np.isfinite(aggregate_variance_ca_from_origin)
        else float("nan")
    )

    by_atom_type: dict[str, CoordinationTypeMetrics] = {}
    for atom_type, atoms_for_type in type_buckets.items():
        first_for_type = atoms_for_type["first"]
        ca_for_type = atoms_for_type["ca"]
        first_distances = [float(np.linalg.norm((a.x, a.y, a.z))) for a in first_for_type]
        ca_distances = [float(np.linalg.norm((a.x, a.y, a.z))) for a in ca_for_type]
        raw_mean_dw_first = float(np.mean([a.dw for a in first_for_type]))
        raw_mean_dw_ca = float(np.mean([a.dw for a in ca_for_type])) if ca_for_type else float("nan")
        variance_first_from_origin = float(np.var(first_distances))
        variance_ca_from_origin = float(np.var(ca_distances)) if ca_distances else float("nan")
        by_atom_type[atom_type] = CoordinationTypeMetrics(
            atom_type=atom_type,
            raw_mean_dw_first=raw_mean_dw_first,
            raw_mean_dw_ca=raw_mean_dw_ca,
            variance_first_from_origin=variance_first_from_origin,
            variance_ca_from_origin=variance_ca_from_origin,
            mean_dw_first=raw_mean_dw_first + variance_first_from_origin,
            mean_dw_ca=(
                raw_mean_dw_ca + variance_ca_from_origin
                if np.isfinite(raw_mean_dw_ca) and np.isfinite(variance_ca_from_origin)
                else float("nan")
            ),
            first_distance_dw=list(zip(first_distances, [a.dw for a in first_for_type])),
            ca_distance_dw=list(zip(ca_distances, [a.dw for a in ca_for_type])),
        )

    parts = parse_name_parts(path.parent.name)
    return FileMetrics(
        path=path,
        prefix=parts.prefix,
        suffix=parts.suffix,
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
    atom_type_separation: float,
) -> None:
    _apply_plot_rcparams(plt)
    filtered = list(metrics)
    plottable = [m for m in filtered if np.isfinite(m.volume)]
    if not plottable:
        raise ValueError("No metrics with finite tetrahedron volume available for plotting.")

    dynamic_height = max(14.5, 7.5 + 0.42 * len(plottable))
    fig = plt.figure(figsize=(12.5, dynamic_height))
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

    suffix_marker_map = _build_suffix_marker_map(m.suffix for m in filtered)
    by_prefix: dict[str, list[FileMetrics]] = {}
    for m in plottable:
        by_prefix.setdefault(m.prefix, []).append(m)

    def _metric_value(metric: FileMetrics, species: str, component: str, atom_type: str | None) -> float:
        if atom_type is None:
            if component == "raw":
                return metric.aggregate_raw_mean_dw_first if species == "first" else metric.aggregate_raw_mean_dw_ca
            if component == "variance":
                return (
                    metric.aggregate_variance_first_from_origin
                    if species == "first"
                    else metric.aggregate_variance_ca_from_origin
                )
            return metric.aggregate_mean_dw_first if species == "first" else metric.aggregate_mean_dw_ca

        if atom_type == "N":
            groups = [metric.by_atom_type[key] for key in ("N", "ND", "NE") if key in metric.by_atom_type]
            if not groups:
                return float("nan")

            if species == "first":
                distances = [distance for group in groups for distance, _ in group.first_distance_dw]
                dw_values = [dw for group in groups for _, dw in group.first_distance_dw]
            else:
                distances = [distance for group in groups for distance, _ in group.ca_distance_dw]
                dw_values = [dw for group in groups for _, dw in group.ca_distance_dw]

            if not distances or not dw_values:
                return float("nan")

            if component == "raw":
                return float(np.mean(dw_values))
            if component == "variance":
                return float(np.var(distances))
            return float(np.mean(dw_values) + np.var(distances))

        group = metric.by_atom_type.get(atom_type)
        if group is None:
            return float("nan")
        if component == "raw":
            return group.raw_mean_dw_first if species == "first" else group.raw_mean_dw_ca
        if component == "variance":
            return group.variance_first_from_origin if species == "first" else group.variance_ca_from_origin
        return group.mean_dw_first if species == "first" else group.mean_dw_ca

    def _types_for_metric(metric: FileMetrics) -> list[str | None]:
        if not group_by_atom_type:
            return [None]
        # Combine ND and NE into N
        ordered_types = []
        if "S" in metric.by_atom_type:
            ordered_types.append("S")
        if any(t in metric.by_atom_type for t in ["ND", "NE", "N"]):
            ordered_types.append("N")
        extras = sorted(t for t in metric.by_atom_type if t not in {"S", "ND", "NE", "N"})
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
                    marker = _marker_for_suffix(m.suffix, suffix_marker_map)
                    y_value = _metric_value(m, species, component, atom_type)
                    if not np.isfinite(y_value):
                        continue
                    ax.scatter(
                        base_x,
                        y_value,
                        color=_type_color(atom_type, species),
                        marker=marker,
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
    ordered_metrics = sorted(plottable, key=lambda m: (m.volume, m.prefix, suffix_sort_key(m.suffix)))
    for idx, metric in enumerate(ordered_metrics, start=1):
        suffix_label = metric.suffix if metric.suffix else r"C$\alpha$ fixed"
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
    line_count = sequence_text.count("\n") + 1
    text_fontsize = max(10, 18 - max(0, line_count - 26) // 3)
    fig.text(
        0.75,
        0.95,
        sequence_text,
        ha="left",
        va="top",
        fontsize=text_fontsize,
        # family="monospace",
    )

    ax_var_s.tick_params(axis="x", which="both", labelbottom=False)
    ax_raw_s.tick_params(axis="x", which="both", labelbottom=False)

    ax_raw_s.set_title("First shell (S/N)")
    ax_raw_s.set_ylabel(r"DW from xyz ($\overline{\sigma^2_{FEFF}}$)")
    ax_var_s.set_ylabel(r"Variance ($\sigma^2_{static}$)")
    ax_total_s.set_ylabel(r"Total ($\overline{\sigma^2}$)")
    ax_total_s.set_xlabel(r"C$\alpha$-tetrahedron volume ($\AA^3$)")

    legend_handles: list[Line2D] = []
    if group_by_atom_type:
        has_s = any("S" in m.by_atom_type for m in filtered)
        has_n = any(("ND" in m.by_atom_type) or ("NE" in m.by_atom_type) or ("N" in m.by_atom_type) for m in filtered)
        if has_s:
            legend_handles.append(
                Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#D4A017", markeredgecolor="none", markersize=11, alpha=0.7, label="S")
            )
        if has_n:
            legend_handles.append(
                Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#1f77b4", markeredgecolor="none", markersize=11, alpha=0.7, label="N")
            )
    else:
        legend_handles.append(
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#D4A017", markeredgecolor="none", markersize=11, alpha=0.7, label="First shell")
        )

    suffix_keys = sorted({_normalize_variant_key(m.suffix) for m in filtered}, key=suffix_sort_key)
    for suffix_key in suffix_keys:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=_marker_for_suffix(suffix_key, suffix_marker_map),
                linestyle="",
                markerfacecolor="#666666",
                markeredgecolor="none",
                markersize=11,
                alpha=0.9,
                label=f"suffix: {_legend_label_from_suffix(suffix_key)}",
            )
        )

    if filtered:
        ax_total_s.legend(handles=legend_handles, loc="best", frameon=True)

    fig.subplots_adjust(left=0.11, right=0.73, bottom=0.05, top=0.94, wspace=0.0, hspace=0.2)
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
    fig, ax_s = plt.subplots(1, 1, figsize=(9.2, 6.8), sharey=False)

    filtered = list(metrics)
    suffix_marker_map = _build_suffix_marker_map(m.suffix for m in filtered)
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
        if any(t in metric.by_atom_type for t in ["ND", "NE", "N"]):
            ordered_types.append("N")
        extras = sorted(t for t in metric.by_atom_type if t not in {"S", "ND", "NE", "N"})
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

        if atom_type == "N":
            first_distance_dw: list[tuple[float, float]] = []
            ca_distance_dw: list[tuple[float, float]] = []
            for key in ("N", "ND", "NE"):
                group = metric.by_atom_type.get(key)
                if group is None:
                    continue
                first_distance_dw.extend(group.first_distance_dw)
                ca_distance_dw.extend(group.ca_distance_dw)
            return first_distance_dw, ca_distance_dw

        group = metric.by_atom_type.get(atom_type)
        if group is None:
            return [], []
        return group.first_distance_dw, group.ca_distance_dw

    for items in by_prefix.values():
        ordered = sorted(items, key=lambda it: suffix_sort_key(it.suffix))
        n_items = len(ordered)
        for idx, m in enumerate(ordered):
            variant_shift = (idx - (n_items - 1) / 2.0) * variant_separation
            marker = _marker_for_suffix(m.suffix, suffix_marker_map)

            for atom_type in _types_for_metric(m):
                x_shift = variant_shift + _type_shift(atom_type)
                first_distance_dw, ca_distance_dw = _distance_pairs_for_type(m, atom_type)

                for distance, dw in first_distance_dw:
                    ax_s.scatter(
                        distance + x_shift,
                        dw,
                        color=_type_color(atom_type, "first"),
                        marker=marker,
                        edgecolors="none",
                        s=145,
                        alpha=0.7,
                        zorder=4,
                    )

    ax_s.set_title("First shell (S/N)")
    ax_s.set_xlabel(r"Distance from origin ($\AA$)")
    ax_s.set_ylabel(r"Debye-Waller factor ($\sigma^2_{FEFF}$)")

    legend_handles: list[Line2D] = []
    if group_by_atom_type:
        has_s = any("S" in m.by_atom_type for m in filtered)
        has_n = any(("ND" in m.by_atom_type) or ("NE" in m.by_atom_type) or ("N" in m.by_atom_type) for m in filtered)
        if has_s:
            legend_handles.append(
                Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#D4A017", markeredgecolor="none", markersize=11, alpha=0.7, label="S")
            )
        if has_n:
            legend_handles.append(
                Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#1f77b4", markeredgecolor="none", markersize=11, alpha=0.7, label="N")
            )
    else:
        legend_handles.append(
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#D4A017", markeredgecolor="none", markersize=11, alpha=0.7, label="First shell")
        )

    suffix_keys = sorted({_normalize_variant_key(m.suffix) for m in filtered}, key=suffix_sort_key)
    for suffix_key in suffix_keys:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=_marker_for_suffix(suffix_key, suffix_marker_map),
                linestyle="",
                markerfacecolor="#666666",
                markeredgecolor="none",
                markersize=11,
                alpha=0.9,
                label=f"suffix: {_legend_label_from_suffix(suffix_key)}",
            )
        )

    if filtered:
        ax_s.legend(handles=legend_handles, loc="best", frameon=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


##############################
# DW Annotation I/O
# Parse XYZ and DW rows, match atoms, and write annotated *_w_dw.xyz files.
##############################


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
    ca_kept = 0
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
        if group == "ca":
            if ca_kept >= 4:
                continue
            ca_kept += 1
        elif group != "nearest":
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


def _parse_first_shell_distances(path: Path) -> list[float]:
    distances: list[float] = []
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

        try:
            distance = float(parts[5])
        except ValueError:
            continue
        distances.append(distance)

    return distances


def _parse_first_shell_dw_factors(path: Path) -> list[float]:
    dw_values: list[float] = []
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

        try:
            dw = float(parts[6])
        except ValueError:
            continue
        dw_values.append(dw)

    return dw_values


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


def _bond_threshold(symbol_a: str, symbol_b: str) -> float | None:
    pair = frozenset({symbol_a.upper(), symbol_b.upper()})
    if pair == frozenset({"C", "C"}):
        return 1.75
    if pair == frozenset({"C", "N"}):
        return 1.62
    if pair == frozenset({"C", "S"}):
        return 1.90
    if pair == frozenset({"C", "H"}):
        return 1.20
    if pair == frozenset({"N", "H"}):
        return 1.15
    return None


def _build_adjacency(atoms: list[XyzAtom]) -> list[set[int]]:
    adjacency = [set() for _ in atoms]
    for i, atom_i in enumerate(atoms):
        for j in range(i + 1, len(atoms)):
            atom_j = atoms[j]
            threshold = _bond_threshold(atom_i.symbol, atom_j.symbol)
            if threshold is None:
                continue
            if _dist2((atom_i.x, atom_i.y, atom_i.z), (atom_j.x, atom_j.y, atom_j.z)) <= threshold * threshold:
                adjacency[i].add(j)
                adjacency[j].add(i)
    return adjacency


def _classify_ring_nitrogens(centered_atoms: list[XyzAtom]) -> dict[int, str]:
    adjacency = _build_adjacency(centered_atoms)
    n_indices = [i for i, atom in enumerate(centered_atoms) if atom.symbol.upper() == "N"]
    labels: dict[int, str] = {idx: "N" for idx in n_indices}

    c_neighbors_by_n: dict[int, set[int]] = {
        n_idx: {j for j in adjacency[n_idx] if centered_atoms[j].symbol.upper() == "C"}
        for n_idx in n_indices
    }

    for n_idx in n_indices:
        carbon_neighbors = c_neighbors_by_n[n_idx]
        if len(carbon_neighbors) < 2:
            continue

        partner: int | None = None
        best_score: tuple[int, float] | None = None
        for other in n_indices:
            if other == n_idx:
                continue
            shared = carbon_neighbors & c_neighbors_by_n[other]
            if not shared:
                continue
            distance_score = _dist2(
                (centered_atoms[n_idx].x, centered_atoms[n_idx].y, centered_atoms[n_idx].z),
                (centered_atoms[other].x, centered_atoms[other].y, centered_atoms[other].z),
            )
            score = (len(shared), -distance_score)
            if best_score is None or score > best_score:
                best_score = score
                partner = other

        if partner is None:
            continue

        partner_neighbors = c_neighbors_by_n[partner]
        shared = carbon_neighbors & partner_neighbors
        unique = list(carbon_neighbors - shared)
        probe_carbons = unique if unique else list(carbon_neighbors)
        ring_carbons = carbon_neighbors | partner_neighbors

        is_nd = False
        for carbon_idx in probe_carbons:
            external_carbon_neighbors = [
                nbr
                for nbr in adjacency[carbon_idx]
                if centered_atoms[nbr].symbol.upper() == "C" and nbr not in ring_carbons
            ]
            if external_carbon_neighbors:
                is_nd = True
                break

        labels[n_idx] = "ND" if is_nd else "NE"

    return labels


def _match_first_shell_n_to_labels(
    path: Path, first_shell_atoms: list[DwTaggedAtom], tolerance: float = 5.0e-3
) -> list[str]:
    default_labels = ["N" if atom.symbol.strip().upper() == "N" else atom.symbol.strip().upper() for atom in first_shell_atoms]
    source_xyz = path.with_name(path.stem.removesuffix("_w_dw") + ".xyz")
    if not source_xyz.exists():
        return default_labels

    try:
        _, original_atoms = parse_xyz(source_xyz)
        centered_atoms = center_on_zn(original_atoms)
    except ValueError:
        return default_labels

    labels_by_n_index = _classify_ring_nitrogens(centered_atoms)
    available_n_indices = [i for i, a in enumerate(centered_atoms) if a.symbol.upper() == "N"]
    tol2 = tolerance * tolerance

    output_labels: list[str] = []
    for atom in first_shell_atoms:
        atom_symbol = atom.symbol.strip().upper()
        if atom_symbol != "N":
            output_labels.append(atom_symbol)
            continue

        best_i = -1
        best_d2 = float("inf")
        for i, n_idx in enumerate(available_n_indices):
            n_atom = centered_atoms[n_idx]
            d2 = _dist2((atom.x, atom.y, atom.z), (n_atom.x, n_atom.y, n_atom.z))
            if d2 < best_d2:
                best_d2 = d2
                best_i = i

        if best_i >= 0 and best_d2 <= tol2:
            matched_n_idx = available_n_indices.pop(best_i)
            output_labels.append(labels_by_n_index.get(matched_n_idx, "N"))
        else:
            output_labels.append("N")

    return output_labels


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


##############################
# Figure Builders
# Create all output figures using shared naming/matching rules.
##############################


def generate_dw_figures(
    parent_dir: Path,
    registry: "SourceRegistry",
    *,
    variant_separation: float,
    group_by_atom_type: bool,
    atom_type_separation: float,
) -> tuple[Path, Path, int]:
    dir_suffix = parent_dir.name.replace(" ", "")
    out_path = parent_dir / f"dw_vs_tetra_volume-{dir_suffix}.png"
    out_distance_path = parent_dir / f"dw_vs_distance-{dir_suffix}.png"

    metrics = registry.file_metrics
    skipped = registry.file_metrics_skipped

    if not metrics:
        raise ValueError("No valid *_w_dw.xyz files for plotting.")

    make_plot(
        metrics,
        out_path=out_path,
        variant_separation=variant_separation,
        group_by_atom_type=group_by_atom_type,
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
    registry: "SourceRegistry",
    patterns: list[str],
    exp_data_paths: list[Path] | None = None,
) -> Path:
    all_entries = registry.chi_entries
    model_entries = [e for e in all_entries if not e.is_experiment]
    exp_entries = [e for e in all_entries if e.is_experiment]

    if not model_entries and not exp_entries:
        raise ValueError(f"No chi_R*.dat files found under {parent_dir}")

    # Build prefix → model entries map (preserving suffix sort order)
    by_prefix: dict[str, list[ChiEntry]] = {}
    for entry in model_entries:
        by_prefix.setdefault(entry.prefix, []).append(entry)
    for prefix in by_prefix:
        by_prefix[prefix].sort(key=lambda e: (suffix_sort_key(e.suffix), e.path.as_posix()))

    # Build prefix → exp entries map
    exp_by_prefix: dict[str, list[ChiEntry]] = {}
    for entry in exp_entries:
        exp_by_prefix.setdefault(entry.prefix, []).append(entry)

    sorted_prefixes = sorted(by_prefix)

    # Collect ordered variant labels across all model entries
    all_variant_labels = {e.label for e in model_entries}
    ordered_variant_labels = sorted(
        all_variant_labels,
        key=lambda label: suffix_sort_key(_normalize_variant_key(label).replace("-", "_")),
    )
    has_exp_panel = bool(exp_entries)
    num_groups = len(sorted_prefixes)
    total_subplots = num_groups + len(ordered_variant_labels) + (1 if has_exp_panel else 0)
    ncols = 3
    nrows = math.ceil(total_subplots / ncols)
    max_group_size = max((len(v) for v in by_prefix.values()), default=0)
    max_variant_stack = max(
        (sum(1 for e in model_entries if e.label == variant) for variant in ordered_variant_labels),
        default=0,
    )
    base_width = 7.5 if max_group_size > 3 or max_variant_stack > 3 else 6.0
    fig_width = base_width * ncols
    fig_height = 4.5 * nrows

    _apply_plot_rcparams(plt)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))
    if hasattr(axes, "flat"):
        axes_list = list(axes.flat)
    else:
        axes_list = [axes]

    variant_series: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    variant_color_map: dict[str, str] = {}
    for idx, variant_label in enumerate(ordered_variant_labels):
        variant_color_map[variant_label] = _label_color(variant_label, idx)

    for idx_group, prefix in enumerate(sorted_prefixes):
        ax = axes_list[idx_group]
        ax.set_axisbelow(True)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for entry in by_prefix[prefix]:
            color = variant_color_map.get(entry.label, _label_color(entry.label, 0))
            ax.plot(entry.r_vals, entry.chir_mag_vals, label=entry.label, linewidth=3.0, color=color)
            variant_series.setdefault(entry.label, []).append((entry.r_vals, entry.chir_mag_vals))

        seen_exp_paths: set[Path] = set()
        for exp_entry in exp_by_prefix.get(prefix, []):
            if exp_entry.path in seen_exp_paths:
                continue
            seen_exp_paths.add(exp_entry.path)
            ax.plot(
                exp_entry.r_vals,
                exp_entry.chir_mag_vals,
                label=exp_entry.label,
                linewidth=2.5,
                color="salmon",
                linestyle="-",
                alpha=0.9,
            )

        subplot_title = _subplot_title_from_prefix(prefix) if prefix else (by_prefix[prefix][0].path.stem if by_prefix[prefix] else prefix)
        ax.set_title(subplot_title)
        ax.set_xlabel(r"R ($\AA$)")
        ax.set_ylabel(r"$|\chi(R)|$")
        ax.set_xlim(0.0, 6.0)

        group_size = len(by_prefix[prefix])
        if group_size > 3:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=14)
        else:
            ax.legend(loc="upper right", fontsize=14)

    next_axis = num_groups
    for variant_label in ordered_variant_labels:
        agg_ax = axes_list[next_axis]
        next_axis += 1
        agg_ax.set_axisbelow(True)
        agg_ax.grid(False)
        agg_ax.spines["top"].set_visible(False)
        agg_ax.spines["right"].set_visible(False)
        color = variant_color_map.get(variant_label, _label_color(variant_label, 0))
        for r_vals, chir_mag_vals in variant_series.get(variant_label, []):
            agg_ax.plot(r_vals, chir_mag_vals, linewidth=3.0, color=color, alpha=0.7)
        agg_ax.set_title(variant_label)
        agg_ax.set_xlabel(r"R ($\AA$)")
        agg_ax.set_ylabel(r"$|\chi(R)|$")
        agg_ax.set_xlim(0.0, 6.0)

    if has_exp_panel:
        exp_ax = axes_list[next_axis]
        next_axis += 1
        exp_ax.set_axisbelow(True)
        exp_ax.grid(False)
        exp_ax.spines["top"].set_visible(False)
        exp_ax.spines["right"].set_visible(False)
        seen_exp_labels: set[str] = set()
        for exp_entry in exp_entries:
            label_to_use = exp_entry.label if exp_entry.label not in seen_exp_labels else None
            if label_to_use:
                seen_exp_labels.add(label_to_use)
            exp_ax.plot(
                exp_entry.r_vals,
                exp_entry.chir_mag_vals,
                linewidth=2.8,
                color="salmon",
                alpha=0.9,
                label=label_to_use,
            )
        exp_ax.set_title("Experimental")
        exp_ax.set_xlabel(r"R ($\AA$)")
        exp_ax.set_ylabel(r"$|\chi(R)|$")
        exp_ax.set_xlim(0.0, 6.0)
        if seen_exp_labels:
            exp_ax.legend(loc="best", fontsize=14)

    for ax in axes_list[total_subplots:]:
        ax.remove()

    fig.suptitle(parent_dir.name, fontsize=20)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    default_name = f"EXAFS-{parent_dir.name}".replace(" ", "")
    out_path = parent_dir / f"{default_name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return out_path


def _generate_summary_grid(
    parent_dir: Path,
    summaries: list[DwDistanceSummary] | list[DwFactorSummary],
    skipped: int,
    *,
    mean_getter,
    std_getter,
    values_getter,
    y_label: str,
    out_name_prefix: str,
) -> tuple[Path, int]:
    if not summaries:
        raise ValueError(f"No summaries found for {out_name_prefix}.")

    _apply_plot_rcparams(plt)
    by_prefix: dict[str, list] = {}
    for summary in summaries:
        by_prefix.setdefault(summary.prefix, []).append(summary)

    sorted_prefixes = sorted(by_prefix)
    n_plots = len(sorted_prefixes)
    ncols = 3
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.0 * ncols, 4.6 * nrows), sharey=False)
    if hasattr(axes, "flat"):
        axes_list = list(axes.flat)
    else:
        axes_list = [axes]

    for idx, prefix in enumerate(sorted_prefixes):
        ax = axes_list[idx]
        items = sorted(by_prefix[prefix], key=lambda it: suffix_sort_key(it.suffix))

        x_vals = list(range(len(items)))
        x_labels = [_legend_label_from_suffix(it.suffix) for it in items]

        for x, item in zip(x_vals, items):
            marker = ">" if _is_h_only_variant(item.suffix) else "o"
            raw_values = values_getter(item)
            if raw_values:
                ax.scatter(
                    [x + 0.05] * len(raw_values),
                    raw_values,
                    marker=marker,
                    color="#D3D3D3",
                    edgecolors="none",
                    s=58,
                    alpha=0.85,
                    zorder=1,
                )
            ax.errorbar(
                x,
                mean_getter(item),
                yerr=std_getter(item),
                fmt=marker,
                color="#333333",
                ecolor="#333333",
                elinewidth=1.8,
                capsize=4,
                markersize=8,
                alpha=0.9,
                zorder=4,
            )

        ax.set_title(format_prefix_label_with_pdb(prefix))
        ax.set_xticks(x_vals)
        ax.set_xticklabels(x_labels, rotation=25, ha="right")
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if x_vals:
            ax.set_xlim(-0.5, len(x_vals) - 0.5)

    for ax in axes_list[n_plots:]:
        ax.remove()

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    dir_suffix = parent_dir.name.replace(" ", "")
    out_path = parent_dir / f"{out_name_prefix}-{dir_suffix}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return out_path, skipped


def generate_first_shell_distance_figure(parent_dir: Path, registry: "SourceRegistry") -> tuple[Path, int]:
    summaries = registry.distance_summaries
    skipped = registry.distance_skipped
    return _generate_summary_grid(
        parent_dir,
        summaries,
        skipped,
        mean_getter=lambda item: item.mean_distance,
        std_getter=lambda item: item.std_distance,
        values_getter=lambda item: item.distances,
        y_label=r"First-shell distance ($\AA$)",
        out_name_prefix="dw_first_shell_distance_grid",
    )


def generate_first_shell_dw_figure(parent_dir: Path, registry: "SourceRegistry") -> tuple[Path, int]:
    summaries = registry.dw_factor_summaries
    skipped = registry.dw_factor_skipped
    return _generate_summary_grid(
        parent_dir,
        summaries,
        skipped,
        mean_getter=lambda item: item.mean_dw,
        std_getter=lambda item: item.std_dw,
        values_getter=lambda item: item.dw_values,
        y_label=r"First-shell DW factor ($\sigma^2_{FEFF}$)",
        out_name_prefix="dw_first_shell_dw_grid",
    )


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
            "Path to an experimental chi(R) data file to overlay on matching EXAFS subplots. "
            "Repeatable; the first 4 filename characters are used as the matching ID."
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

    # Build the data registry once — all four plot generators read from it.
    registry = build_registry(
        parent_dir,
        chi_patterns=args.chi_pattern,
        exp_data_paths=args.exp_data,
    )

    try:
        dw_volume_path, dw_distance_path, dw_skipped = generate_dw_figures(
            parent_dir,
            registry,
            variant_separation=args.variant_separation,
            group_by_atom_type=args.group_by_atom_type,
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
            registry,
            patterns=args.chi_pattern,
            exp_data_paths=args.exp_data,
        )
        print(f"[ok] Wrote figure: {chi_path}")
    except ValueError as exc:
        print(f"[skip] chi(R) figure: {exc}")

    try:
        first_shell_path, distance_skipped = generate_first_shell_distance_figure(parent_dir, registry)
        print(f"[ok] Wrote figure: {first_shell_path}")
        print(f"[ok] First-shell distance figure skipped {distance_skipped} folder(s).")
    except ValueError as exc:
        print(f"[skip] first-shell distance figure: {exc}")

    try:
        first_shell_dw_path, dw_factor_skipped = generate_first_shell_dw_figure(parent_dir, registry)
        print(f"[ok] Wrote figure: {first_shell_dw_path}")
        print(f"[ok] First-shell DW-factor figure skipped {dw_factor_skipped} folder(s).")
    except ValueError as exc:
        print(f"[skip] first-shell DW-factor figure: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())