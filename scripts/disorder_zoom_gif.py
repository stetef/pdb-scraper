#!/usr/bin/env python3
"""Generate a GIF that conveys structural disorder around a metal center.

The animation has four phases that loop:
    1. Thermal vibration: every atom wiggles with smooth multi-mode noise while
       the camera slowly orbits, giving a phonon-like impression of disorder.
    2. Freeze: vibration amplitude decays to zero (camera keeps drifting).
    3. Zoom: camera tracks in on the metal center and the N closest
       coordinators while the structure stays frozen.
    4. Labels + pause: dotted lines from the metal to each coordinator and a
       3D-floating distance panel above the structure (drawn at high zorder
       so it stays visible). Held for a long pause before the GIF loops.

Style follows scripts/view_xyz_pc_py3dmol.py (white background, Jmol colors,
sticks + spheres). py3Dmol renders to WebGL in a browser, so the GIF
rasterization here uses matplotlib's 3D backend with shaded ``plot_surface``
spheres for a lit/3D look.

Example:
  uv run python scripts/disorder_zoom_gif.py \
    --xyz data/large-cys-his-datasets/4cys-large/output/xyz_files/1a71_ZN_homo_d2.60_cluster2.xyz \
    --output data/large-cys-his-datasets/4cys-large/output/gifs/1a71_disorder.gif

Override the metal / coordinator selection if your structure differs:
  uv run python scripts/disorder_zoom_gif.py \
    --xyz path.xyz --output out.gif --target FE --coord-elem N --n-coord 6
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from PIL import Image

# Use a serif (Times-like) font for any text drawn in the GIF.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif", "serif"]

# Jmol-style element colors (subset that covers common biological atoms).
JMOL_COLORS: dict[str, str] = {
    "H": "#FFFFFF",
    "C": "#909090",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "S": "#FFFF30",
    "P": "#FF8000",
    "ZN": "#7D80B0",
    "FE": "#E06633",
    "CU": "#C88033",
    "MG": "#8AFF00",
    "MN": "#9C7AC7",
    "CA": "#3DFF00",
    "NA": "#AB5CF2",
    "K": "#8F40D4",
    "CL": "#1FF01F",
}
DEFAULT_COLOR = "#B0B0B0"

# Covalent radii (Å) — used for bond detection.
COVALENT_RADII: dict[str, float] = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "P": 1.07,
    "ZN": 1.22, "FE": 1.32, "CU": 1.32, "MG": 1.41, "MN": 1.39, "CA": 1.76,
    "NA": 1.66, "K": 2.03, "CL": 1.02,
}
DEFAULT_COV_RADIUS = 0.75

# Van der Waals radii (Å) — used for sphere drawing (matches Jmol/py3Dmol).
VDW_RADII: dict[str, float] = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
    "ZN": 1.39, "FE": 1.94, "CU": 1.40, "MG": 1.73, "MN": 1.97, "CA": 2.31,
    "NA": 2.27, "K": 2.75, "CL": 1.75,
}
DEFAULT_VDW_RADIUS = 1.60


@dataclass
class Atom:
    element: str
    pos: np.ndarray
    meta: dict[str, str] = field(default_factory=dict)


def parse_xyz(path: Path) -> list[Atom]:
    lines = path.read_text(encoding="utf-8").splitlines()
    n = int(lines[0].strip())
    atoms: list[Atom] = []
    for line in lines[2 : 2 + n]:
        head, _, comment = line.partition("#")
        parts = head.split()
        if len(parts) < 4:
            continue
        elem = parts[0].upper()
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        meta: dict[str, str] = {}
        for tok in comment.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                meta[k.strip().upper()] = v.strip()
        atoms.append(Atom(element=elem, pos=np.array([x, y, z], dtype=float), meta=meta))
    return atoms


def find_target_and_coordinators(
    atoms: list[Atom], target_elem: str, coord_elem: str, n_coord: int
) -> tuple[int, list[int]]:
    target_elem = target_elem.upper()
    coord_elem = coord_elem.upper()
    target_idx = next((i for i, a in enumerate(atoms) if a.element == target_elem), None)
    if target_idx is None:
        raise ValueError(f"No {target_elem} atom in XYZ")
    target_pos = atoms[target_idx].pos
    candidates = sorted(
        (
            (i, float(np.linalg.norm(a.pos - target_pos)))
            for i, a in enumerate(atoms)
            if a.element == coord_elem
        ),
        key=lambda t: t[1],
    )
    if len(candidates) < n_coord:
        raise ValueError(
            f"Found only {len(candidates)} {coord_elem} atoms (needed {n_coord})"
        )
    return target_idx, [i for i, _ in candidates[:n_coord]]


def detect_bonds(atoms: list[Atom], factor: float = 1.25) -> list[tuple[int, int]]:
    bonds: list[tuple[int, int]] = []
    n = len(atoms)
    for i in range(n):
        ri = COVALENT_RADII.get(atoms[i].element, DEFAULT_COV_RADIUS)
        for j in range(i + 1, n):
            rj = COVALENT_RADII.get(atoms[j].element, DEFAULT_COV_RADIUS)
            if np.linalg.norm(atoms[i].pos - atoms[j].pos) < (ri + rj) * factor:
                bonds.append((i, j))
    return bonds


def smooth_thermal_displacements(
    n_atoms: int, n_frames: int, sigma: float, rng: np.random.Generator,
    n_modes: int = 4,
) -> np.ndarray:
    """Sum of a few sinusoidal modes per atom — looks like a soft phonon bath."""
    disp = np.zeros((n_frames, n_atoms, 3))
    if sigma <= 0 or n_frames == 0:
        return disp
    t = np.arange(n_frames)
    for _ in range(n_modes):
        directions = rng.normal(size=(n_atoms, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-9
        freq = rng.uniform(0.04, 0.16)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=(n_atoms,))
        amp = sigma / np.sqrt(n_modes)
        wave = np.sin(2.0 * np.pi * freq * t[None, :] + phase[:, None])
        disp += amp * np.einsum("af,ad->fad", wave, directions)
    return disp


def make_unit_sphere(mesh: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 2.0 * np.pi, mesh * 2)
    v = np.linspace(0.0, np.pi, mesh)
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones_like(u), np.cos(v))
    return sx, sy, sz


def draw_lit_sphere(
    ax,
    center: np.ndarray,
    radius: float,
    color: str,
    unit_sphere: tuple[np.ndarray, np.ndarray, np.ndarray],
    light_source: LightSource,
    zorder: int = 3,
) -> None:
    sx, sy, sz = unit_sphere
    x = center[0] + radius * sx
    y = center[1] + radius * sy
    z = center[2] + radius * sz
    ax.plot_surface(
        x, y, z,
        color=color,
        rstride=1, cstride=1,
        linewidth=0,
        antialiased=True,
        shade=True,
        lightsource=light_source,
        zorder=zorder,
    )


def render_frame(
    positions: np.ndarray,
    elements: list[str],
    bonds: list[tuple[int, int]],
    distance_lines: list[tuple[int, int, float, str]] | None,
    azim: float,
    elev: float,
    center: np.ndarray,
    half_extent: float,
    sphere_scale: float,
    sphere_mesh: int,
    width: int,
    height: int,
    dpi: int,
    label_offset: float = 0.35,
) -> np.ndarray:
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    light = LightSource(azdeg=315, altdeg=55)
    unit_heavy = make_unit_sphere(sphere_mesh)
    unit_light = make_unit_sphere(max(sphere_mesh - 4, 6))

    for i, j in bonds:
        pi, pj = positions[i], positions[j]
        ax.plot(
            [pi[0], pj[0]], [pi[1], pj[1]], [pi[2], pj[2]],
            color="#404040", linewidth=2.2, solid_capstyle="round", zorder=1,
        )

    for k, elem in enumerate(elements):
        radius = VDW_RADII.get(elem, DEFAULT_VDW_RADIUS) * sphere_scale
        color = JMOL_COLORS.get(elem, DEFAULT_COLOR)
        unit = unit_light if elem == "H" else unit_heavy
        draw_lit_sphere(
            ax, positions[k], radius, color,
            unit_sphere=unit, light_source=light, zorder=3,
        )

    if distance_lines:
        for a, b, d, color in distance_lines:
            p1, p2 = positions[a], positions[b]
            ax.plot(
                [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                linestyle=(0, (1, 2)), color=color, linewidth=2.2, zorder=20,
            )
            mid = 0.5 * (p1 + p2)
            ax.text(
                mid[0], mid[1], mid[2] + label_offset,
                f"{d:.2f} Å",
                color=color, fontsize=18, fontweight="bold",
                family="serif",
                ha="center", va="center",
                zorder=10000,
                bbox=dict(
                    facecolor="white", edgecolor=color,
                    boxstyle="round,pad=0.25", alpha=0.95, linewidth=1.0,
                ),
            )

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(center[0] - half_extent, center[0] + half_extent)
    ax.set_ylim(center[1] - half_extent, center[1] + half_extent)
    ax.set_zlim(center[2] - half_extent, center[2] + half_extent)
    try:
        ax.set_box_aspect((1, 1, 1))
    except (AttributeError, ValueError):
        pass
    ax.set_axis_off()

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


# Color cycle for the four (or N) distance lines + matching panel labels.
_LINE_PALETTE = [
    "#D81B60", "#1E88E5", "#43A047", "#FB8C00",
    "#8E24AA", "#00ACC1", "#C0CA33", "#5D4037",
]


def build_frames(
    atoms: list[Atom],
    target_idx: int,
    coord_idx: list[int],
    args: argparse.Namespace,
) -> list[np.ndarray]:
    base_pos = np.array([a.pos for a in atoms])
    elements = [a.element for a in atoms]
    bonds_all = detect_bonds(atoms)
    metal_coord_pairs = {tuple(sorted((target_idx, c))) for c in coord_idx}
    bonds_no_metal = [b for b in bonds_all if tuple(sorted(b)) not in metal_coord_pairs]
    rng = np.random.default_rng(args.seed)

    centroid = base_pos.mean(axis=0)
    full_extent = float(np.max(np.linalg.norm(base_pos - centroid, axis=1)))
    half_full = max(full_extent * 1.15, 1.5)

    metal = base_pos[target_idx]
    site_extent = float(np.max(np.linalg.norm(base_pos[coord_idx] - metal, axis=1)))
    half_site = max(site_extent * args.zoom_factor, 1.2)

    distance_lines: list[tuple[int, int, float, str]] = []
    for i, c in enumerate(coord_idx):
        d = float(np.linalg.norm(base_pos[c] - metal))
        color = _LINE_PALETTE[i % len(_LINE_PALETTE)]
        distance_lines.append((target_idx, c, d, color))

    n_vib = args.n_vibration_frames
    n_freeze = args.n_freeze_frames
    n_zoom = args.n_zoom_frames

    azim_start, elev_start = -60.0, 18.0
    azim_after_vib = azim_start + 0.6 * n_vib
    azim_after_freeze = azim_after_vib + 0.6 * n_freeze
    azim_end = azim_after_freeze + 85.0
    elev_end = 8.0

    frames: list[np.ndarray] = []

    # Phase 1: full-amplitude thermal vibration with slow orbit.
    disp1 = smooth_thermal_displacements(
        n_atoms=len(atoms), n_frames=n_vib,
        sigma=args.vibration_amplitude, rng=rng,
    )
    for f in range(n_vib):
        positions = base_pos + disp1[f]
        azim = azim_start + (azim_after_vib - azim_start) * f / max(n_vib - 1, 1)
        frames.append(
            render_frame(
                positions=positions, elements=elements,
                bonds=bonds_all, distance_lines=None,
                azim=azim, elev=elev_start,
                center=centroid, half_extent=half_full,
                sphere_scale=args.sphere_scale, sphere_mesh=args.sphere_mesh,
                width=args.width, height=args.height, dpi=args.dpi,
            )
        )

    # Phase 2: vibration damps to zero, camera keeps drifting.
    disp2 = smooth_thermal_displacements(
        n_atoms=len(atoms), n_frames=n_freeze,
        sigma=args.vibration_amplitude, rng=rng,
    )
    for f in range(n_freeze):
        t = (f + 1) / max(n_freeze, 1)
        damp = (1.0 - t) ** 2
        positions = base_pos + disp2[f] * damp
        azim = azim_after_vib + (azim_after_freeze - azim_after_vib) * t
        frames.append(
            render_frame(
                positions=positions, elements=elements,
                bonds=bonds_all, distance_lines=None,
                azim=azim, elev=elev_start,
                center=centroid, half_extent=half_full,
                sphere_scale=args.sphere_scale, sphere_mesh=args.sphere_mesh,
                width=args.width, height=args.height, dpi=args.dpi,
            )
        )

    # Phase 3: camera zooms onto the first shell (frozen).
    for f in range(n_zoom):
        t = (f + 1) / max(n_zoom, 1)
        ease = 0.5 - 0.5 * np.cos(np.pi * t)
        center = centroid * (1.0 - ease) + metal * ease
        half_extent = half_full * (1.0 - ease) + half_site * ease
        azim = azim_after_freeze + (azim_end - azim_after_freeze) * ease
        elev = elev_start + (elev_end - elev_start) * ease
        frames.append(
            render_frame(
                positions=base_pos, elements=elements,
                bonds=bonds_all, distance_lines=None,
                azim=azim, elev=elev,
                center=center, half_extent=half_extent,
                sphere_scale=args.sphere_scale, sphere_mesh=args.sphere_mesh,
                width=args.width, height=args.height, dpi=args.dpi,
            )
        )

    # Phase 4: labelled hold — emit ONE frame; the pause is handled by giving
    # this frame a long per-frame duration in the GIF (much smaller file size
    # and more reliable than duplicating an identical frame hundreds of times).
    label_frame = render_frame(
        positions=base_pos, elements=elements,
        bonds=bonds_no_metal,
        distance_lines=distance_lines,
        azim=azim_end, elev=elev_end,
        center=metal, half_extent=half_site,
        sphere_scale=args.sphere_scale, sphere_mesh=args.sphere_mesh,
        width=args.width, height=args.height, dpi=args.dpi,
    )
    frames.append(label_frame)
    return frames


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--xyz", required=True, help="Input XYZ file")
    p.add_argument("--output", required=True, help="Output GIF path")
    p.add_argument("--target", default="ZN",
                   help="Element symbol of the central metal (default: ZN)")
    p.add_argument("--coord-elem", default="S",
                   help="Element symbol of coordinating atoms (default: S)")
    p.add_argument("--n-coord", type=int, default=4,
                   help="Number of nearest coordinators to label (default: 4)")
    p.add_argument("--vibration-amplitude", type=float, default=0.07,
                   help="RMS thermal displacement in Å (default: 0.07)")
    p.add_argument("--n-vibration-frames", type=int, default=26,
                   help="Frames in the vibration phase (default: 26)")
    p.add_argument("--n-freeze-frames", type=int, default=8,
                   help="Frames in the vibration → freeze damping phase (default: 8)")
    p.add_argument("--n-zoom-frames", type=int, default=52,
                   help="Frames in the camera zoom phase (default: 52)")
    p.add_argument("--pause-seconds", type=float, default=3.0,
                   help="Seconds to hold the labelled view before looping (default: 3)")
    p.add_argument("--zoom-factor", type=float, default=1.05,
                   help="Half-extent multiplier on Zn–coord distance during the "
                        "label phase; lower → tighter zoom (default: 1.05)")
    p.add_argument("--fps", type=int, default=18, help="GIF frame rate")
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--sphere-scale", type=float, default=0.30,
                   help="VdW radius multiplier for sphere drawing (default: 0.30)")
    p.add_argument("--sphere-mesh", type=int, default=30,
                   help="Sphere mesh resolution (default: 22; raise for smoother)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def save_gif(
    frames: list[np.ndarray], path: Path, fps: int, pause_seconds: float
) -> None:
    """Write a looping GIF with per-frame durations in milliseconds.

    All frames except the last use 1/fps. The last frame (the labelled view)
    is held for ``pause_seconds`` so the viewer can read the distances.
    PIL's ``Image.save`` is used directly because ``imageio.mimsave`` does
    not reliably honor sub-second per-frame durations with the GIF writer.
    """
    if not frames:
        raise ValueError("No frames to save")
    movement_ms = max(int(round(1000.0 / max(fps, 1))), 20)
    pause_ms = max(int(round(pause_seconds * 1000.0)), movement_ms)
    durations_ms = [movement_ms] * (len(frames) - 1) + [pause_ms]

    pil_frames = [Image.fromarray(f.astype(np.uint8)) for f in frames]
    pil_frames[0].save(
        str(path),
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )


def main() -> None:
    args = parse_args()
    xyz_path = Path(args.xyz)
    if not xyz_path.exists():
        raise SystemExit(f"XYZ file not found: {xyz_path}")
    atoms = parse_xyz(xyz_path)
    target_idx, coord_idx = find_target_and_coordinators(
        atoms, args.target, args.coord_elem, args.n_coord,
    )
    print(
        f"Target: {atoms[target_idx].element} (idx {target_idx}) | "
        f"Coordinators: {[atoms[i].element + str(i) for i in coord_idx]}"
    )

    frames = build_frames(atoms, target_idx, coord_idx, args)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_gif(frames, out_path, fps=args.fps, pause_seconds=args.pause_seconds)
    print(
        f"Wrote GIF: {out_path}  ({len(frames)} frames @ {args.fps} fps, "
        f"final-frame pause {args.pause_seconds:.1f} s)"
    )


if __name__ == "__main__":
    main()
