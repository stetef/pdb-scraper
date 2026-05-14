#!/usr/bin/env python3
"""Focused study of CA tetrahedron volume against coordination geometry.

For each .xyz file in --dir:
  1) Compute CA tetrahedron volume + 4 Zn->coord-atom distances.
  2) Compute CYS Zn->SG->CB->CA dihedral angles (one per coordinating CYS).
  3) Build a "family" label of the form "<sequence>-<secstruct>" (e.g. "CCx5Cx2C-LLHH").

Outputs:
  - <out-dir>/volume_vs_coord_distance_<label>.png            full dataset scatter
  - <out-dir>/volume_vs_cys_dihedral_<label>.png              full dataset scatter
  - <out-dir>/volume_vs_ca_q_tetra_<label>.png                CA volume vs q_tetra(CA)
  - <out-dir>/volume_vs_s_q_tetra_<label>.png                 CA volume vs q_tetra(S)
  - <out-dir>/s_q_tetra_vs_ca_q_tetra_<label>.png             q_tetra(S) vs q_tetra(CA)
  - <out-dir>/volume_vs_coord_distance_extremes_<label>.png   broken-axis scatter, top-n only
  - <out-dir>/volume_vs_cys_dihedral_extremes_<label>.png     broken-axis scatter, top-n only
  - <out-dir>/volume_extremes_<label>.csv                     5 smallest + 5 largest
  - terminal: 5 smallest + 5 largest volumes with family labels

q_tetra is the Errington-Debenedetti tetrahedral order parameter. Both
q_tetra(S) and q_tetra(CA) are computed about the origin (Zn). q = 1 means
perfect tetrahedron, q -> 0 means random.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np

from xyz_plot_helpers import (
    _apply_hist_rcparams,
    _cys_atoms_by_residue,
    _family_token_from_grouped_text,
    _his_atoms_by_residue,
    _sequence_distance_from_atoms,
    _sequence_secstruct_from_atoms,
    ca_volume_and_coord_distances,
    dihedral_origin_sg_cb_ca_selected_by_sg,
    find_xyz_files,
    parse_xyz_atoms,
)


@dataclass
class FileData:
    path: Path
    volume: float
    family: str
    coord_dists: List[float] = field(default_factory=list)
    cys_dihedrals: List[float] = field(default_factory=list)
    q_tetra_coord: float | None = None
    q_tetra_ca: float | None = None


def _normalize_system_label(label: str) -> str:
    return "-".join(label.strip().split())


def _q_tetra(points: List[np.ndarray], center: np.ndarray) -> float | None:
    if len(points) != 4:
        return None
    unit_vecs: List[np.ndarray] = []
    for p in points:
        v = p - center
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            return None
        unit_vecs.append(v / norm)
    total = 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            cos_t = float(np.dot(unit_vecs[i], unit_vecs[j]))
            cos_t = max(-1.0, min(1.0, cos_t))
            total += (cos_t + 1.0 / 3.0) ** 2
    return 1.0 - (3.0 / 8.0) * total


def _select_coord_and_ca_vectors(
    atoms,
    *,
    max_residues: int = 4,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    candidates: List[Tuple[float, np.ndarray, np.ndarray]] = []

    cys_by_res = _cys_atoms_by_residue(atoms, required_atoms={"SG", "CA"})
    for atom_map in cys_by_res.values():
        sg = atom_map.get("SG")
        ca = atom_map.get("CA")
        if sg is None or ca is None:
            continue
        sg_vec = np.asarray((sg.x, sg.y, sg.z), dtype=float)
        ca_vec = np.asarray((ca.x, ca.y, ca.z), dtype=float)
        candidates.append((float(np.dot(sg_vec, sg_vec)), sg_vec, ca_vec))

    his_by_res = _his_atoms_by_residue(atoms, required_atoms={"ND1", "NE2", "CA"})
    for atom_map in his_by_res.values():
        ca = atom_map.get("CA")
        if ca is None:
            continue
        coord_atom = None
        for name in ("ND1", "NE2"):
            a = atom_map.get(name)
            if a is not None and a.meta.get("COORD") == "1":
                coord_atom = a
                break
        if coord_atom is None:
            best_d2 = None
            for name in ("ND1", "NE2"):
                a = atom_map.get(name)
                if a is None:
                    continue
                d2 = float(a.x * a.x + a.y * a.y + a.z * a.z)
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    coord_atom = a
        if coord_atom is None:
            continue
        coord_vec = np.asarray((coord_atom.x, coord_atom.y, coord_atom.z), dtype=float)
        ca_vec = np.asarray((ca.x, ca.y, ca.z), dtype=float)
        candidates.append((float(np.dot(coord_vec, coord_vec)), coord_vec, ca_vec))

    candidates.sort(key=lambda t: t[0])
    selected = candidates[:max_residues]
    return [c for _, c, _ in selected], [ca for _, _, ca in selected]


def _family_label(atoms) -> str:
    seq = _sequence_distance_from_atoms(atoms, use_latex=False)
    sec = _sequence_secstruct_from_atoms(atoms)
    if not seq and not sec:
        return ""
    seq_token = _family_token_from_grouped_text(seq) if seq else ""
    sec_token = _family_token_from_grouped_text(sec) if sec else ""
    if seq_token and sec_token:
        return f"{seq_token}-{sec_token}"
    return seq_token or sec_token


def _flatten(
    files: List[FileData],
    *,
    y_attr: str,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    x_residue: List[float] = []
    y_residue: List[float] = []
    x_mean: List[float] = []
    y_mean: List[float] = []
    for fd in files:
        ys = getattr(fd, y_attr)
        if not ys:
            continue
        x_residue.extend([fd.volume] * len(ys))
        y_residue.extend(ys)
        x_mean.append(fd.volume)
        y_mean.append(float(np.mean(ys)))
    return x_residue, y_residue, x_mean, y_mean


def _scatter(
    x_per_residue: List[float],
    y_per_residue: List[float],
    x_means: List[float],
    y_means: List[float],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    show_plot: bool,
) -> None:
    import matplotlib.pyplot as plt

    _apply_hist_rcparams(plt)

    if not x_per_residue and not x_means:
        print(f"No data for '{title}'; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if x_per_residue:
        ax.scatter(
            x_per_residue,
            y_per_residue,
            s=22,
            c="#B0B0B0",
            alpha=0.5,
            edgecolors="none",
            zorder=4,
        )
    if x_means:
        ax.scatter(
            x_means,
            y_means,
            s=40,
            c="#000000",
            marker=".",
            zorder=6,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _scatter_means_colored(
    x_per_residue: List[float],
    y_per_residue: List[float],
    x_means: List[float],
    y_means: List[float],
    c_means: List[float],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
    out_path: Path,
    show_plot: bool,
    vlines: List[float] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    _apply_hist_rcparams(plt)

    if not x_per_residue and not x_means:
        print(f"No data for '{title}'; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if x_per_residue:
        ax.scatter(
            x_per_residue,
            y_per_residue,
            s=22,
            c="#B0B0B0",
            alpha=0.5,
            edgecolors="none",
            zorder=4,
        )

    sc = None
    if x_means:
        sc = ax.scatter(
            x_means,
            y_means,
            c=c_means,
            cmap="viridis",
            s=60,
            alpha=0.6,
            edgecolors="none",
            zorder=6,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(cbar_label)

    if vlines:
        for v in vlines:
            ax.axvline(v, linestyle="--", color="black", linewidth=1.2, zorder=7)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _scatter_colored(
    x: List[float],
    y: List[float],
    c: List[float],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
    out_path: Path,
    show_plot: bool,
    hline: float | None = None,
    vline: float | None = None,
) -> None:
    import matplotlib.pyplot as plt

    _apply_hist_rcparams(plt)

    if not x:
        print(f"No data for '{title}'; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    sc = ax.scatter(
        x,
        y,
        c=c,
        cmap="viridis",
        s=80,
        alpha=0.45,
        edgecolors="none",
        zorder=6,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(cbar_label)

    if hline is not None:
        ax.axhline(hline, linestyle="--", color="black", linewidth=1.2, zorder=7)
    if vline is not None:
        ax.axvline(vline, linestyle="--", color="black", linewidth=1.2, zorder=7)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _broken_axis_scatter(
    smallest: List[FileData],
    largest: List[FileData],
    *,
    y_attr: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    show_plot: bool,
) -> None:
    import matplotlib.pyplot as plt

    _apply_hist_rcparams(plt)

    x_left, y_left, xm_left, ym_left = _flatten(smallest, y_attr=y_attr)
    x_right, y_right, xm_right, ym_right = _flatten(largest, y_attr=y_attr)

    if not (x_left or x_right):
        print(f"No data for '{title}'; skipping plot.")
        return

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, sharey=True, figsize=(10.5, 6.0), gridspec_kw={"wspace": 0.06}
    )

    for ax in (ax_l, ax_r):
        ax.set_axisbelow(True)
        ax.grid(False)
        ax.spines["top"].set_visible(False)

    ax_l.spines["right"].set_visible(False)
    ax_r.spines["left"].set_visible(False)
    ax_r.tick_params(axis="y", which="both", left=False, labelleft=False)

    for ax, x_res, y_res, x_m, y_m in (
        (ax_l, x_left, y_left, xm_left, ym_left),
        (ax_r, x_right, y_right, xm_right, ym_right),
    ):
        if x_res:
            ax.scatter(
                x_res, y_res, s=22, c="#B0B0B0", alpha=0.5, edgecolors="none", zorder=4
            )
        if x_m:
            ax.scatter(x_m, y_m, s=40, c="#000000", marker=".", zorder=6)

    def _padded_xlim(values: List[float]) -> Tuple[float, float]:
        lo, hi = min(values), max(values)
        span = hi - lo
        pad = span * 0.1 if span > 0 else max(0.05, abs(lo) * 0.05)
        return lo - pad, hi + pad

    if x_left:
        ax_l.set_xlim(*_padded_xlim(x_left))
    if x_right:
        ax_r.set_xlim(*_padded_xlim(x_right))

    d = 0.015
    kwargs = dict(transform=ax_l.transAxes, color="k", clip_on=False, linewidth=1.2, zorder=10)
    ax_l.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax_l.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax_r.transAxes)
    ax_r.plot((-d, +d), (-d, +d), **kwargs)
    ax_r.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    fig.suptitle(title)
    fig.supxlabel(xlabel)
    ax_l.set_ylabel(ylabel)
    fig.tight_layout()

    from matplotlib.patches import Rectangle

    pos_l = ax_l.get_position()
    pos_r = ax_r.get_position()
    gap = Rectangle(
        (pos_l.x1, pos_l.y0),
        pos_r.x0 - pos_l.x1,
        pos_l.y1 - pos_l.y0,
        transform=fig.transFigure,
        facecolor="#E8E8E8",
        edgecolor="none",
        zorder=0,
    )
    fig.add_artist(gap)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot CA volume vs coord distance and vs CYS dihedral; report volume extremes."
    )
    parser.add_argument(
        "--dir",
        default="data/output/xyz_files",
        help="Directory containing .xyz files (default: data/output/xyz_files).",
    )
    parser.add_argument(
        "--out-dir",
        default="Figures",
        help="Output directory for PNG/CSV (default: Figures).",
    )
    parser.add_argument(
        "--system-label",
        help="Optional label appended to output filenames.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="How many smallest / largest volumes to print (default: 5).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open matplotlib windows (save PNGs only).",
    )
    parser.add_argument(
        "--q-s-threshold",
        type=float,
        default=0.74,
        help="Files with q_tetra(S) below this threshold are flagged in plots and CSV (default: 0.74).",
    )
    args = parser.parse_args()

    xyz_dir = Path(args.dir)
    if not xyz_dir.exists():
        raise SystemExit(f"Directory not found: {xyz_dir}")

    xyz_files = find_xyz_files(xyz_dir)
    if not xyz_files:
        raise SystemExit(f"No .xyz files in {xyz_dir}")

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = _normalize_system_label(args.system_label or xyz_dir.name)

    files: List[FileData] = []
    for p in xyz_files:
        atoms = parse_xyz_atoms(p)
        volume, coord_dists = ca_volume_and_coord_distances(atoms, max_residues=4)
        if volume is None or len(coord_dists) != 4:
            continue
        coord_vecs, ca_vecs = _select_coord_and_ca_vectors(atoms, max_residues=4)
        origin = np.zeros(3, dtype=float)
        q_coord = _q_tetra(coord_vecs, origin) if len(coord_vecs) == 4 else None
        q_ca = _q_tetra(ca_vecs, origin) if len(ca_vecs) == 4 else None
        files.append(
            FileData(
                path=p,
                volume=volume,
                family=_family_label(atoms),
                coord_dists=list(coord_dists),
                cys_dihedrals=dihedral_origin_sg_cb_ca_selected_by_sg(atoms, max_residues=4),
                q_tetra_coord=q_coord,
                q_tetra_ca=q_ca,
            )
        )

    if not files:
        raise SystemExit("No xyz files yielded a valid CA tetrahedron volume.")

    show = not args.no_show

    vol_label = r"CA tetrahedron volume (Å$^3$)"
    q_ca_label = r"$q_{CA}$"
    q_s_label = r"$q_{S}$"

    x_res, y_res, x_m, y_m = _flatten(files, y_attr="coord_dists")
    _scatter(
        x_res, y_res, x_m, y_m,
        title="CA volume vs first shell distance",
        xlabel=vol_label,
        ylabel=r"Zn $\rightarrow$ coordinating atom distance (Å)",
        out_path=output_dir / f"volume_vs_coord_distance_{label}.png",
        show_plot=show,
    )

    files_with_qca = [fd for fd in files if fd.q_tetra_ca is not None and fd.coord_dists]
    xq_m = [fd.volume for fd in files_with_qca]
    yq_m = [float(np.mean(fd.coord_dists)) for fd in files_with_qca]
    cq_m = [fd.q_tetra_ca for fd in files_with_qca]
    _scatter_means_colored(
        x_res, y_res, xq_m, yq_m, cq_m,
        title=f"CA volume vs first shell distance (mean color = {q_ca_label})",
        xlabel=vol_label,
        ylabel=r"Zn $\rightarrow$ coordinating atom distance (Å)",
        cbar_label=q_ca_label,
        out_path=output_dir / f"volume_vs_coord_distance_qca_{label}.png",
        show_plot=show,
        vlines=[0.7, 45.0],
    )

    x_res, y_res, x_m, y_m = _flatten(files, y_attr="cys_dihedrals")
    _scatter(
        x_res, y_res, x_m, y_m,
        title=r"CA volume vs CYS Zn--SG--CB--CA dihedral",
        xlabel=vol_label,
        ylabel=r"Dihedral Zn--SG--CB--CA ($^\circ$)",
        out_path=output_dir / f"volume_vs_cys_dihedral_{label}.png",
        show_plot=show,
    )

    triplets = [
        (fd.volume, fd.q_tetra_coord, fd.q_tetra_ca)
        for fd in files
        if fd.q_tetra_coord is not None and fd.q_tetra_ca is not None
    ]
    if triplets:
        vols = [v for v, _, _ in triplets]
        qss = [s for _, s, _ in triplets]
        qcs = [c for _, _, c in triplets]

        _scatter_colored(
            vols, qcs, qss,
            title=f"CA volume vs {q_ca_label} (color = {q_s_label})",
            xlabel=vol_label,
            ylabel=q_ca_label,
            cbar_label=q_s_label,
            out_path=output_dir / f"volume_vs_ca_q_tetra_{label}.png",
            show_plot=show,
        )

        _scatter_colored(
            vols, qss, qcs,
            title=f"CA volume vs {q_s_label} (color = {q_ca_label})",
            xlabel=vol_label,
            ylabel=q_s_label,
            cbar_label=q_ca_label,
            out_path=output_dir / f"volume_vs_s_q_tetra_{label}.png",
            show_plot=show,
            hline=args.q_s_threshold,
        )

        _scatter_colored(
            qss, qcs, vols,
            title=f"{q_s_label} vs {q_ca_label} (color = {vol_label})",
            xlabel=q_s_label,
            ylabel=q_ca_label,
            cbar_label=vol_label,
            out_path=output_dir / f"s_q_tetra_vs_ca_q_tetra_{label}.png",
            show_plot=show,
            vline=args.q_s_threshold,
        )

    files.sort(key=lambda fd: fd.volume)
    n = max(1, min(args.top_n, len(files)))
    smallest = files[:n]
    largest = files[-n:]

    _broken_axis_scatter(
        smallest, largest,
        y_attr="coord_dists",
        title=f"CA volume vs first shell distance ({n} smallest / {n} largest)",
        xlabel=vol_label,
        ylabel=r"Zn $\rightarrow$ coordinating atom distance (Å)",
        out_path=output_dir / f"volume_vs_coord_distance_extremes_{label}.png",
        show_plot=show,
    )

    _broken_axis_scatter(
        smallest, largest,
        y_attr="cys_dihedrals",
        title=f"CA volume vs CYS Zn--SG--CB--CA dihedral ({n} smallest / {n} largest)",
        xlabel=vol_label,
        ylabel=r"Dihedral Zn--SG--CB--CA ($^\circ$)",
        out_path=output_dir / f"volume_vs_cys_dihedral_extremes_{label}.png",
        show_plot=show,
    )

    largest_desc = list(reversed(largest))
    low_q_s = sorted(
        (fd for fd in files if fd.q_tetra_coord is not None and fd.q_tetra_coord < args.q_s_threshold),
        key=lambda fd: fd.q_tetra_coord,
    )
    extremes = [("smallest CA volume", i, fd) for i, fd in enumerate(smallest, start=1)]
    extremes += [("largest CA volume", i, fd) for i, fd in enumerate(largest_desc, start=1)]
    extremes += [("smallest q_S", i, fd) for i, fd in enumerate(low_q_s, start=1)]
    max_dih = max((len(fd.cys_dihedrals) for _, _, fd in extremes), default=0)

    csv_path = output_dir / f"volume_extremes_{label}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["group", "rank", "volume_A3", "family"]
        header.extend(f"cys_dihedral_{i + 1}_deg" for i in range(max_dih))
        header.extend(["cys_dihedral_mean_deg", "q_tetra_coord", "q_tetra_ca", "xyz_path"])
        writer.writerow(header)
        for group, rank, fd in extremes:
            dihedrals = list(fd.cys_dihedrals)
            row = [group, rank, f"{fd.volume:.4f}", fd.family]
            for i in range(max_dih):
                row.append(f"{dihedrals[i]:.2f}" if i < len(dihedrals) else "")
            mean_str = f"{float(np.mean(dihedrals)):.2f}" if dihedrals else ""
            q_coord_str = f"{fd.q_tetra_coord:.4f}" if fd.q_tetra_coord is not None else ""
            q_ca_str = f"{fd.q_tetra_ca:.4f}" if fd.q_tetra_ca is not None else ""
            row.extend([mean_str, q_coord_str, q_ca_str, str(fd.path)])
            writer.writerow(row)
    print(
        f"Saved: {csv_path}  ({len(extremes)} rows: "
        f"{n} smallest CA volume + {n} largest CA volume + "
        f"{len(low_q_s)} with q_S < {args.q_s_threshold})"
    )

    def _fmt(fd: FileData) -> str:
        parts = [str(fd.path), f"{fd.volume:.2f} A^3", fd.family]
        if fd.q_tetra_coord is not None:
            parts.append(f"q_coord={fd.q_tetra_coord:.2f}")
        if fd.q_tetra_ca is not None:
            parts.append(f"q_ca={fd.q_tetra_ca:.2f}")
        return ", ".join(parts)

    print(f"\n{n} smallest CA tetrahedron volumes:")
    for fd in smallest:
        print(f"  {_fmt(fd)}")

    print(f"\n{n} largest CA tetrahedron volumes:")
    for fd in largest_desc:
        print(f"  {_fmt(fd)}")

    print(f"\n{len(low_q_s)} files with q_S < {args.q_s_threshold} (sorted ascending):")
    for fd in low_q_s:
        print(f"  {_fmt(fd)}")


if __name__ == "__main__":
    """
    Example:
    ```bash
    uv run python ./scripts/xyz-volume-study.py \\
        --dir data/large-cys-his-datasets/4cys-large/output/xyz_files \\
        --out-dir Figures \\
        --system-label 4cys-large
    ```
    """
    main()
