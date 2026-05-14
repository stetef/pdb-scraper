"""
Plot sig2_tot vs distance for S and N atoms in the "nearest" section
of dw*.dat files, colored by atom type.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = "/Users/stetef/Documents/SLAC/pdb-scraper/macchiato_downloads/dw-parameterization"

COLOR_S = "#C8A000"   # dark yellow
COLOR_N = "#AEC6E8"   # pastel blue

distances = {"S": [], "N": []}
sig2_tots = {"S": [], "N": []}

for entry in sorted(os.listdir(BASE_DIR)):
    if "unbound" in entry:
        continue
    dirpath = os.path.join(BASE_DIR, entry)
    if not os.path.isdir(dirpath):
        continue
    dat_files = glob.glob(os.path.join(dirpath, "dw*.dat"))
    if not dat_files:
        continue
    dat_file = dat_files[0]
    with open(dat_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 7 or parts[0] != "nearest":
                continue
            symbol = parts[1]
            if symbol not in ("S", "N"):
                continue
            try:
                distance = float(parts[5])
                sig2_tot = float(parts[6])
            except ValueError:
                continue
            distances[symbol].append(distance)
            sig2_tots[symbol].append(sig2_tot)

fig, ax = plt.subplots(figsize=(7, 5))

for symbol, color, label in [("N", COLOR_N, "N"), ("S", COLOR_S, "S")]:
    x = np.array(distances[symbol])
    y = np.array(sig2_tots[symbol])

    ax.scatter(
        x, y,
        color=color,
        edgecolors="grey",
        linewidths=0.5,
        s=60,
        label=label,
        alpha=0.85,
        zorder=3,
    )

    coeffs = np.polyfit(x, y, 2)
    x_fit = np.linspace(x.min(), x.max(), 300)
    y_fit = np.polyval(coeffs, x_fit)
    ax.plot(x_fit, y_fit, color=color, linewidth=2, zorder=4)

ax.set_xlabel(r"Distance ($\AA$)", fontsize=13)
ax.set_ylabel(r"$\sigma^2_\mathrm{tot}$ ($\AA^2$)", fontsize=13)
ax.set_title(r"$\sigma^2_\mathrm{tot}$ vs Distance for Nearest S and N Atoms", fontsize=13)
ax.legend(title="Atom", fontsize=11, title_fontsize=11)
ax.grid(True, alpha=0.3)

out_path = os.path.join(BASE_DIR, "sig2_vs_distance_SN.png")
fig.tight_layout()
fig.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
plt.show()
