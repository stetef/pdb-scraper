#!/usr/bin/env python3
"""Plot sig^2 vs the effective force constant of each FEFF scattering
path, with the subplot layout matching figs/top_paths_sig2_vs_reff_
bycomp_nlegs_lt5.png.

For each top intra-residue path (same seed/MIN_POINTS gating as
top_paths_sig2_vs_reff.py's bycomp_nlegs_lt5 figure), one (FC_path,
sig2_path) point is plotted. The FC is the n=-2 moment-derived effective
force constant of the *whole path*, not a per-bond decomposition: dmdw
emits a single FC per scattering path (Zn-anchored sequence of legs),
and that's what we read.

The dmdw lookup uses the same paths.dat-driven leg sequence as
top_paths_sig2_vs_reff, so subplot titles (Zn-ND1(HisND), Zn-SG(Cys)-
CB(Cys), ...) and grid order match the existing reff figure exactly.

Usage:
    python top_paths_sig2_vs_fc.py <dir> [<dir> ...] [--out fig.png]

Each <dir> needs feff.inp, paths.dat, dmdw.out, xmu*.dat and a non-
_clean/_trj .xyz file (same expectations as top_paths_sig2_vs_reff.py).
"""

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
})

# Reuse all the path-identification / atom-naming machinery from the
# reff plotter so subplot labels and the seed/MIN_POINTS gating match.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from robust_fit import plot_slope_ci, theilsen_linfit  # noqa: E402
from top_paths_sig2_vs_reff import _fmt_p  # noqa: E402
from top_paths_sig2_vs_reff import (  # noqa: E402
    MIN_POINTS, SPINE_DEFAULT_COLOR, SPINE_HIGHLIGHT_COLOR,
    STUDY_COLORS, STUDY_ORDER,
    TOP_N_NLEGS2, TOP_N_NLEGS3, TOP_N_NLEGS4,
    _atom_name, _classify_nlegs4, _find_xmu, _find_xyz,
    _format_subplot_title, _path_scatterers,
    _wrap_path_label, align_xyz_to_feff, build_adjacency, distance,
    identify_residue_for_path, match_paths_dat_to_feff,
    name_atoms_and_residues, parse_feff_atoms, parse_xmu_paths, parse_xyz,
    residue_index_of, structure_composition, study_of_workdir,
)
MIN_POINTS = 5

# Single dark-green color for every (FC_path, sig2_path) point — the
# subplots are now per-path, not per-bond, so there's nothing to color-
# code within a subplot.
PATH_COLOR = "#1f6f3a"


def parse_dmdw_paths(dmdw_path):
    """Return dict keyed by tuple of dmdw 1-based path indices, each value
    a dict with fc_n2 / path_len / reduced_mass. Inlined from the prior
    einstein_fc_bonds.py so this script is self-contained."""
    text = Path(dmdw_path).read_text()
    chunks = re.split(r"Path Indices:\s*", text)
    results = {}
    for chunk in chunks[1:]:
        nl = chunk.find("\n")
        indices = tuple(int(x) for x in chunk[:nl].split())
        m = re.search(
            r"^\s*-2\s+\S+\s+\S+\s+\S+\s+(\S+)", chunk, re.MULTILINE
        )
        fc_n2 = float(m.group(1)) if m else None
        mpl = re.search(
            r"Path Length \(Ang\),\s*s\^2.*?:\s+(\S+)", chunk
        )
        path_len = float(mpl.group(1)) if mpl else None
        mmass = re.search(r"Path Red\. Mass \(AMU\):\s+(\S+)", chunk)
        red_mass = float(mmass.group(1)) if mmass else None
        results[indices] = {
            "fc_n2": fc_n2,
            "path_len": path_len,
            "reduced_mass": red_mass,
        }
    return results


def _label_for_path(p, scatterers, feff_atoms, names):
    """Return the subplot label for a path. Byte-for-byte identical to
    top_paths_sig2_vs_reff._row_for_path's labelling so subplot grids line
    up across the two scripts."""
    nlegs = p["nlegs"]
    if nlegs == 2:
        (i,) = scatterers
        return f"Zn-{_atom_name(i, feff_atoms, names)}"
    if nlegs == 3:
        i_path, j_path = scatterers
        if distance(feff_atoms[0], feff_atoms[i_path]) <= distance(
            feff_atoms[0], feff_atoms[j_path]
        ):
            apex, far = i_path, j_path
        else:
            apex, far = j_path, i_path
        return (
            f"Zn-{_atom_name(apex, feff_atoms, names)}"
            f"-{_atom_name(far, feff_atoms, names)}"
        )
    if nlegs == 4:
        a, b, c = scatterers
        topology = _classify_nlegs4(scatterers)
        if topology == "rattle":
            return (
                f"Zn-{_atom_name(a, feff_atoms, names)}"
                f"-{_atom_name(b, feff_atoms, names)}"
                f"-{_atom_name(a, feff_atoms, names)}"
            )
        if topology == "double_back":
            return (
                f"Zn-{_atom_name(a, feff_atoms, names)}"
                f"-Zn-{_atom_name(a, feff_atoms, names)}"
            )
        if topology == "diff_back":
            if distance(feff_atoms[0], feff_atoms[a]) <= distance(
                feff_atoms[0], feff_atoms[c]
            ):
                close_a, far_c = a, c
            else:
                close_a, far_c = c, a
            return (
                f"Zn-{_atom_name(close_a, feff_atoms, names)}-Zn"
                f"-{_atom_name(far_c, feff_atoms, names)}"
            )
        # linear Zn-A-B-C
        if distance(feff_atoms[0], feff_atoms[a]) <= distance(
            feff_atoms[0], feff_atoms[c]
        ):
            ordered = (a, b, c)
        else:
            ordered = (c, b, a)
        return (
            f"Zn-{_atom_name(ordered[0], feff_atoms, names)}"
            f"-{_atom_name(ordered[1], feff_atoms, names)}"
            f"-{_atom_name(ordered[2], feff_atoms, names)}"
        )
    return None


def _fc_for_path(p, path_legs, fc_paths):
    """Return the path-level n=-2 effective FC for xmu path `p` by
    looking up the matching dmdw entry. dmdw keys are 1-based and
    include the absorber as the leading index, so we shift the
    paths.dat-derived scatterer sequence (0-based, 0=Zn) by +1 and
    prepend 1. Returns None if no exact match in dmdw.out."""
    legs = path_legs.get(p["file"]) if path_legs is not None else None
    if not legs:
        return None
    scatterers = legs[:-1]  # drop the trailing return-to-Zn (index 0)
    key = (1,) + tuple(s + 1 for s in scatterers)
    info = fc_paths.get(key)
    return info["fc_n2"] if info else None


def _row_for_path(
    p, feff_atoms, names, residues, adj, workdir, path_legs, fc_paths
):
    """Build one row per nlegs<5 path with its path-level FC.

    Same drop filters as top_paths_sig2_vs_reff._row_for_path: H scatterers
    are excluded for chemistry-clarity; nlegs>=5 is out of scope here."""
    scatterers = _path_scatterers(p, path_legs)
    if scatterers is None:
        return None
    if any(a != 0 and feff_atoms[a]["element"] == "H" for a in scatterers):
        return None

    nlegs = p["nlegs"]
    if nlegs > 4:
        return None

    label = _label_for_path(p, scatterers, feff_atoms, names)
    if label is None:
        return None

    fc = _fc_for_path(p, path_legs, fc_paths)
    if fc is None:
        return None

    res_ids = [
        residue_index_of(a, residues, feff_atoms, adj) if a != 0 else None
        for a in scatterers
    ]
    non_none = [r for r in res_ids if r is not None]
    cross = (
        len(non_none) != len([a for a in scatterers if a != 0])
        or len(set(non_none)) > 1
    )

    row = {
        "label": label,
        "nlegs": nlegs,
        "reff": p["reff"],
        "sig2_tot": p["sig2_tot"],
        "cw_amp": p["cw_amp"],
        "file": p["file"],
        "workdir": workdir.name,
        "cross_residue": cross,
        "fc": fc,
    }
    res = identify_residue_for_path(p, feff_atoms, residues, path_legs, adj)
    if res is not None:
        row["res_tag"] = res.get("atom_tag")
    return row


def process_workdir(workdir):
    workdir = Path(workdir)
    feff_inp = workdir / "feff.inp"
    paths_dat = workdir / "paths.dat"
    dmdw_out = workdir / "dmdw.out"
    for f in (feff_inp, paths_dat, dmdw_out):
        if not f.is_file():
            raise FileNotFoundError(f)
    xmu_dat = _find_xmu(workdir)
    xyz_path = _find_xyz(workdir)

    feff_atoms = parse_feff_atoms(feff_inp)
    xyz_atoms = parse_xyz(xyz_path)
    align_xyz_to_feff(feff_atoms, xyz_atoms)
    adj = build_adjacency(feff_atoms)
    names, residues = name_atoms_and_residues(feff_atoms, adj)
    composition = structure_composition(residues)

    path_legs = match_paths_dat_to_feff(paths_dat, feff_atoms)
    fc_paths = parse_dmdw_paths(dmdw_out)

    paths = parse_xmu_paths(xmu_dat)
    short_paths = sorted(
        (p for p in paths if p["nlegs"] < 5), key=lambda p: -p["cw_amp"]
    )
    all_short_rows = []
    for p in short_paths:
        r = _row_for_path(
            p, feff_atoms, names, residues, adj, workdir, path_legs, fc_paths
        )
        if r is None:
            continue
        r["composition"] = composition
        all_short_rows.append(r)

    seed_rows = []
    for nlegs_val, cap in (
        (2, TOP_N_NLEGS2), (3, TOP_N_NLEGS3), (4, TOP_N_NLEGS4),
    ):
        cls = [
            r for r in all_short_rows
            if r["nlegs"] == nlegs_val and not r.get("cross_residue")
        ]
        seed_rows.extend(cls[:cap])

    print(
        f"# {workdir.name} [{composition}]: "
        f"nlegs<5 rows={len(all_short_rows)}, seed={len(seed_rows)}"
    )
    return all_short_rows, seed_rows


def plot_paths_bycomp(
    all_rows, seed_rows, out_path,
    point_fn, x_label, y_label, suptitle=None, bold_title=None,
):
    """Subplots match the bycomp_lt5 layout: one subplot per path label
    nominated by seed_rows (>= MIN_POINTS occurrences in the seed pool),
    sorted by nlegs ascending then mean cw_amp descending. Each subplot
    plots one (x, y) point per path-instance.

    `point_fn(row)` returns the (x, y) for one path — callers pick which
    axis is sig2/reff and which is fc."""
    by_label = defaultdict(list)
    for r in all_rows:
        by_label[r["label"]].append(r)
    seed_by_label = defaultdict(list)
    for r in seed_rows:
        seed_by_label[r["label"]].append(r)

    labels = sorted(
        (L for L in seed_by_label if len(seed_by_label[L]) >= MIN_POINTS),
        key=lambda L: (
            seed_by_label[L][0]["nlegs"],
            -float(np.mean([r["cw_amp"] for r in seed_by_label[L]])),
        ),
    )
    if not labels:
        print(
            f"\nno label has >= {MIN_POINTS} seed points; skipping {out_path}"
        )
        return

    n = len(labels)
    ncols = min(4, max(1, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.6 * ncols, 2.9 * nrows), squeeze=False
    )
    for ax in axes.flat:
        ax.set_visible(False)

    global_seen_studies = set()
    for k, label in enumerate(labels):
        ax = axes[k // ncols][k % ncols]
        ax.set_visible(True)
        rows = by_label[label]
        nlegs_val = rows[0]["nlegs"]

        pts = [point_fn(r) for r in rows]
        xs = [t[0] for t in pts]
        ys = [t[1] for t in pts]

        fit = theilsen_linfit(xs, ys)
        if fit is not None:
            plot_slope_ci(ax, fit, np.asarray(xs), color="lightgray", alpha=0.35)
            x_fit = np.array([min(xs), max(xs)])
            ax.plot(
                x_fit, fit["slope"] * x_fit + fit["intercept"], "-",
                color="black", alpha=0.6, lw=1.2,
            )

        xs_arr = np.asarray(xs)
        ys_arr = np.asarray(ys)
        studies = [study_of_workdir(r["workdir"]) for r in rows]
        for study in STUDY_ORDER:
            mask = np.array([s == study for s in studies])
            if not mask.any():
                continue
            global_seen_studies.add(study)
            ax.scatter(
                xs_arr[mask], ys_arr[mask],
                color=STUDY_COLORS[study],
                s=34, alpha=0.4,
                edgecolors="white", linewidths=0.3,
                label=study,
            )

        annot = f"nlegs={nlegs_val}\nN={len(rows)}"
        if fit is not None:
            annot += f"\n$R^2$={fit['r2']:.2f}"
            annot += f"\n$\\tau$={fit['tau']:.2f} (p={_fmt_p(fit['tau_p'])})"
        ax.text(
            0.97, 0.97, annot,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8,
        )
        # Highlight subplots whose linear fit is tight (R^2 > 0.9): the
        # paths whose y-quantity is genuinely explained by the x-quantity.
        tight = fit is not None and (
            fit["r2"] > 0.9
            or (abs(fit["tau"]) > 0.8 and fit["tau_p"] < 0.005)
        )
        for spine in ax.spines.values():
            spine.set_linewidth(2.3 if tight else 0.7)
            spine.set_color(
                SPINE_HIGHLIGHT_COLOR if tight else SPINE_DEFAULT_COLOR
            )

        ax.set_title(
            _wrap_path_label(_format_subplot_title(label)), fontsize=9
        )
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.grid(alpha=0.3)

    if bold_title:
        fig.text(
            0.5, 0.875, bold_title, ha="center", va="top",
            fontsize=18, fontweight="bold",
        )
    if suptitle:
        fig.suptitle(
            (
                f"{suptitle}\n"
                r"Highlighted panels: $R^2 > 0.9$ or "
                r"$|\tau| > 0.8$ ($p < 0.005$)"
            ),
            fontsize=11, y=0.85,
        )

    legend_studies = [s for s in STUDY_ORDER if s in global_seen_studies]
    if legend_studies:
        legend_handles = [
            plt.Line2D(
                [0], [0], marker="o", linestyle="",
                color=STUDY_COLORS[s], markeredgecolor="white",
                markeredgewidth=0.3, markersize=7, alpha=0.7, label=s,
            )
            for s in legend_studies
        ]
        fig.legend(
            handles=legend_handles, loc="upper center",
            bbox_to_anchor=(0.5, 0.822), ncol=len(legend_handles),
            fontsize=11, framealpha=0.85, handlelength=1.0,
            handletextpad=0.4, columnspacing=1.5, borderpad=0.3, frameon=False,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("workdirs", nargs="+", type=Path)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("top_paths_sig2_vs_fc.png"),
    )
    args = ap.parse_args()

    # Silently skip directories without dmdw.out so the user can glob
    # `param-downloads/*` even when only some structures have a DMDW run.
    # Anything that's not a directory is also dropped (e.g. stray files).
    workdirs = [wd for wd in args.workdirs if wd.is_dir()]
    runnable = [wd for wd in workdirs if (wd / "dmdw.out").is_file()]
    skipped = [wd for wd in workdirs if not (wd / "dmdw.out").is_file()]
    if skipped:
        print(f"# skipping {len(skipped)} dir(s) without dmdw.out:")
        for wd in skipped:
            print(f"#   {wd.name}")
    if not runnable:
        raise SystemExit("no workdirs with dmdw.out among the given paths")

    all_rows = []
    seed_rows = []
    for wd in runnable:
        rows, seeds = process_workdir(wd)
        all_rows.extend(rows)
        seed_rows.extend(seeds)

    base_dir = args.out.parent
    suf = args.out.suffix
    figs_dir = base_dir / "figs"
    figs_dir.mkdir(exist_ok=True)
    fc_vs_reff_out = (
        figs_dir / f"top_paths_fc_vs_reff_bycomp_nlegs_lt5{suf}"
    )

    intra_lt5 = [r for r in all_rows if not r.get("cross_residue")]

    plot_paths_bycomp(
        intra_lt5, seed_rows, fc_vs_reff_out,
        point_fn=lambda r: (r["reff"], r["fc"]),
        x_label=r"$R_{\mathrm{eff}}$ (Å)",
        y_label=r"Eff. FC, $n{=}{-2}$ (N/m)",
        suptitle=(
            r"Top intra-residue paths ($n_{\mathrm{legs}} < 5$; "
            f"{TOP_N_NLEGS2}/{TOP_N_NLEGS3}/{TOP_N_NLEGS4} per "
            r"$n_{\mathrm{legs}} = 2/3/4$)"
        ),
        bold_title="Effective force constant vs. effective path length",
    )


if __name__ == "__main__":
    main()
