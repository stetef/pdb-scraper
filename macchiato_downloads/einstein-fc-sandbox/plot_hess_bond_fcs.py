#!/usr/bin/env python3
"""Plot per-bond stretch force constant vs bond length, one subplot per
(atom_tag, bond_label) in the CSV produced by hess_bond_fcs.py.

Splitting on (atom_tag, bond_label) keeps HisND ND1-CG and HisNE ND1-CG
in separate subplots -- the atomic identifiers happen to coincide but
the chemistry is different (in HisND, ND1 is the Zn-bonded ring N; in
HisNE, ND1 is the far ring N).

Each subplot is a scatter of (bond_length, fc_n_per_m) for every row in
the group across all workdirs. Points are colored by composition
(4cys / 2cys2his / etc.) using the palette already used elsewhere in
this sandbox so figures read together. A linear regression is drawn per
subplot; subplots whose fit is tight (R^2 > 0.9) get bold spines.

Usage:
    python plot_hess_bond_fcs.py [--csv hess_bond_fcs.csv] [--out fig.png]
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robust_fit import plot_slope_ci, theilsen_linfit  # noqa: E402
from top_paths_sig2_vs_reff import (  # noqa: E402
    SPINE_DEFAULT_COLOR, SPINE_HIGHLIGHT_COLOR,
    STUDY_COLORS, STUDY_ORDER, study_of_workdir,
)


def _bond_sort_key(row):
    """Sort subplots so Zn-X bonds come first, then bonds outward toward
    CA, with His ring bonds in the middle. Falls back to alphabetical if
    we don't know the bond. Sort key is a tuple (atom_tag, position)."""
    atom_tag = row["atom_tag"]
    bond = row["bond_label"]
    canonical = {
        "Cys": ["Zn-SG", "SG-CB", "CB-CA"],
        "HisND": [
            "Zn-ND1", "ND1-CG", "ND1-CE1", "CE1-NE2",
            "NE2-CD2", "CD2-CG", "CG-CB", "CB-CA",
        ],
        "HisNE": [
            "Zn-NE2", "NE2-CE1", "NE2-CD2", "CE1-ND1",
            "ND1-CG", "CD2-CG", "CG-CB", "CB-CA",
        ],
    }
    order = canonical.get(atom_tag, [])
    pos = order.index(bond) if bond in order else len(order)
    tag_rank = {"Cys": 0, "HisND": 1, "HisNE": 2}.get(atom_tag, 3)
    return (tag_rank, pos, bond)


def _fmt_p(p):
    """Compact p-value formatter for in-figure annotation."""
    if p < 1e-3:
        return f"{p:.0e}"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.2f}"


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["bond_length"] = float(r["bond_length"])
            r["fc_n_per_m"] = float(r["fc_n_per_m"])
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).with_name("hess_bond_fcs.csv"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "figs" / "hess_bond_fcs_vs_length.png"
        ),
    )
    args = ap.parse_args()

    rows = load_csv(args.csv)
    if not rows:
        raise SystemExit(f"no rows in {args.csv}")

    # Group by (atom_tag, bond_label) so HisND/HisNE bonds with the
    # same atomic-name string don't collide.
    by_group = defaultdict(list)
    for r in rows:
        by_group[(r["atom_tag"], r["bond_label"])].append(r)

    groups = sorted(
        by_group.keys(),
        key=lambda key: _bond_sort_key(by_group[key][0]),
    )
    n = len(groups)
    ncols = min(4, max(1, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.6 * ncols, 2.9 * nrows), squeeze=False
    )
    for ax in axes.flat:
        ax.set_visible(False)

    for k, key in enumerate(groups):
        ax = axes[k // ncols][k % ncols]
        ax.set_visible(True)
        atom_tag, bond_label = key
        group = by_group[key]
        xs = np.array([r["bond_length"] for r in group])
        ys = np.array([r["fc_n_per_m"] for r in group])

        # Per-study scatter so color carries which sweep (charge / angle /
        # family) the workdir came from. Legend is drawn once at the figure
        # level below the suptitle.
        studies = [study_of_workdir(r["workdir"]) for r in group]
        for study in STUDY_ORDER:
            mask = [s == study for s in studies]
            if not any(mask):
                continue
            ax.scatter(
                xs[mask], ys[mask],
                color=STUDY_COLORS[study],
                s=34, alpha=0.4,
                edgecolors="white", linewidths=0.3,
            )

        fit = theilsen_linfit(xs, ys)
        if fit is not None:
            plot_slope_ci(ax, fit, xs, color="lightgray", alpha=0.35)
            x_fit = np.array([xs.min(), xs.max()])
            ax.plot(
                x_fit, fit["slope"] * x_fit + fit["intercept"], "-",
                color="black", alpha=0.6, lw=1.2,
            )

        ax.set_title(f"{atom_tag}: {bond_label}", fontsize=9)

        annot = f"N={len(group)}"
        r2 = fit["r2"] if fit is not None else None
        if fit is not None:
            annot += f"\n$R^2$={fit['r2']:.2f}"
            annot += f"\n$\\tau$={fit['tau']:.2f} (p={_fmt_p(fit['tau_p'])})"
        ax.text(
            0.97, 0.97, annot,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8,
        )
        bold = fit is not None and (
            fit["r2"] > 0.9
            or (abs(fit["tau"]) > 0.8 and fit["tau_p"] < 0.005)
        )
        for spine in ax.spines.values():
            spine.set_linewidth(2.3 if bold else 0.7)
            spine.set_color(
                SPINE_HIGHLIGHT_COLOR if bold else SPINE_DEFAULT_COLOR
            )

        ax.set_xlabel("bond length (Å)", fontsize=9)
        ax.set_ylabel("FC (N/m)", fontsize=9)
        ax.grid(alpha=0.3)

    fig.text(
        0.5, 0.87, "FC v.s. Bond Length",
        ha="center", va="top", fontsize=18, fontweight="bold",
    )
    fig.suptitle(
        (
            "Per-bond stretch FC vs bond length (from ORCA Hessian)\n"
            r"Bold spines: $R^2 > 0.9$ or ($|\tau| > 0.8$ and $p < 0.005$)"
            "\n"
            r"$R^2 = 1 - \sum_i (y_i - \hat y_i)^2 \,/\, \sum_i (y_i - \bar y)^2$"
            "      "
            r"$\tau = (n_{\rm conc} - n_{\rm disc}) / \binom{n}{2}$"
        ),
        fontsize=11, y=0.85, verticalalignment="top",
    )

    from matplotlib.lines import Line2D
    seen_studies = {study_of_workdir(r["workdir"]) for r in rows}
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="",
            markerfacecolor=STUDY_COLORS[s],
            markeredgecolor="none", markersize=8, label=s,
        )
        for s in STUDY_ORDER if s in seen_studies
    ]
    if handles:
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, 0.802),
            frameon=False,
            fontsize=10,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.84])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
