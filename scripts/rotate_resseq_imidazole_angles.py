#!/usr/bin/env python3
"""Rotate a single annotated XYZ residue (typically HIS) by alpha/beta/gamma angles.

Conventions implemented (per user-confirmed defaults):
- Alpha: rotation around Zn->coordinating-N axis.
- Beta: opening/cone angle between Zn->N and N->ring-bisector vectors.
- Zero reference: the input geometry itself (no pre-canonicalization).

Only atoms in --resseq are moved, with coordinating N fixed in place. All other
atoms remain unchanged.

The script also writes a py3Dmol HTML visualization that overlays:
- Original structure (transparent gray)
- Rotated structure (colored)
- A small local-frame coordinate cube at the coordinating N
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import re

import numpy as np
import py3Dmol


COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "S": 1.05,
    "ZN": 1.22,
}


@dataclass
class AtomRecord:
    element: str
    x: float
    y: float
    z: float
    comment: str
    meta: Dict[str, str]

    @property
    def coord(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    def set_coord(self, v: np.ndarray) -> None:
        self.x = float(v[0])
        self.y = float(v[1])
        self.z = float(v[2])


def _parse_comment_meta(comment: str) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    for token in comment.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            meta[key] = value
    return meta


def parse_annotated_xyz(path: Path) -> Tuple[int, str, List[AtomRecord]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise SystemExit(f"XYZ file too short: {path}")

    try:
        natoms = int(lines[0].strip())
    except ValueError as exc:
        raise SystemExit(f"First line is not an atom count in {path}") from exc

    title = lines[1]
    atoms: List[AtomRecord] = []

    for raw in lines[2:]:
        stripped = raw.strip()
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
        meta = _parse_comment_meta(comment)
        atoms.append(AtomRecord(parts[0], x, y, z, comment, meta))

    if len(atoms) != natoms:
        print(
            f"Warning: header says {natoms} atoms, parsed {len(atoms)} atoms from {path.name}. Continuing.")
    return natoms, title, atoms


def write_annotated_xyz(path: Path, title: str, atoms: Sequence[AtomRecord]) -> None:
    out_lines = [str(len(atoms)), title]
    for atom in atoms:
        base = f"{atom.element:>2s} {atom.x:16.8f} {atom.y:16.8f} {atom.z:16.8f}"
        if atom.comment:
            out_lines.append(f"{base}   # {atom.comment}")
        else:
            out_lines.append(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def _el(atom: AtomRecord) -> str:
    return atom.element.strip().upper()


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _unit(v: np.ndarray, *, name: str = "vector") -> np.ndarray:
    n = _norm(v)
    if n < 1e-12:
        raise SystemExit(f"Cannot normalize near-zero {name}.")
    return v / n


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    aa = _unit(a, name="a")
    bb = _unit(b, name="b")
    c = float(np.clip(np.dot(aa, bb), -1.0, 1.0))
    return math.degrees(math.acos(c))


def _rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis_u = _unit(axis, name="rotation axis")
    theta = math.radians(angle_deg)
    kx, ky, kz = axis_u
    K = np.array(
        [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]],
        dtype=float,
    )
    I = np.eye(3)
    return I + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def _rotate_points_about_pivot(
    points: np.ndarray,
    pivot: np.ndarray,
    axis: np.ndarray,
    angle_deg: float,
) -> np.ndarray:
    R = _rotation_matrix(axis, angle_deg)
    shifted = points - pivot
    return (shifted @ R.T) + pivot


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _build_residue_heavy_adjacency(res_atoms: List[int], atoms: Sequence[AtomRecord]) -> Dict[int, List[int]]:
    heavy = [idx for idx in res_atoms if _el(atoms[idx]) != "H"]
    adj: Dict[int, List[int]] = {idx: [] for idx in heavy}
    for i, idx_i in enumerate(heavy):
        ai = atoms[idx_i]
        ei = _el(ai)
        ri = COVALENT_RADII.get(ei)
        if ri is None:
            continue
        for idx_j in heavy[i + 1 :]:
            aj = atoms[idx_j]
            ej = _el(aj)
            rj = COVALENT_RADII.get(ej)
            if rj is None:
                continue
            d = _dist(ai.coord, aj.coord)
            if 0.6 <= d <= (ri + rj + 0.45):
                adj[idx_i].append(idx_j)
                adj[idx_j].append(idx_i)
    return adj


def _find_5_cycles_containing_node(adj: Dict[int, List[int]], node: int) -> List[Tuple[int, ...]]:
    cycles: set[Tuple[int, ...]] = set()

    def dfs(start: int, current: int, path: List[int]) -> None:
        if len(path) == 5:
            if start in adj.get(current, []):
                cyc = tuple(path)
                min_idx = min(range(5), key=lambda i: cyc[i])
                ordered = cyc[min_idx:] + cyc[:min_idx]
                reversed_order = tuple(reversed(ordered))
                cycles.add(min(ordered, reversed_order))
            return
        for nxt in adj.get(current, []):
            if nxt in path:
                continue
            dfs(start, nxt, path + [nxt])

    dfs(node, node, [node])
    return sorted(cycles)


def _select_imidazole_ring(
    res_atoms: Sequence[int],
    atoms: Sequence[AtomRecord],
    coordinating_n_idx: int,
) -> Tuple[int, ...]:
    ring_indices: List[int] = []
    for idx in res_atoms:
        atom = atoms[idx]
        if _el(atom) == "H":
            continue
        atom_name = atom.meta.get("ATOM", "").upper()
        if _el(atom) == "C" and atom_name in {"CA", "CB"}:
            continue
        ring_indices.append(idx)

    if coordinating_n_idx not in ring_indices:
        raise SystemExit("Coordinating N is not part of the inferred ring atom set.")

    if len(ring_indices) < 5:
        raise SystemExit(
            f"Inferred ring has only {len(ring_indices)} atoms; expected ~5 after filtering RESSEQ atoms."
        )

    if len(ring_indices) > 5:
        coord_n = atoms[coordinating_n_idx].coord
        others = [idx for idx in ring_indices if idx != coordinating_n_idx]
        others_sorted = sorted(others, key=lambda i: _dist(atoms[i].coord, coord_n))
        ring_indices = [coordinating_n_idx] + others_sorted[:4]

    return tuple(ring_indices)


def _cycle_neighbors(cycle: Sequence[int], center_idx: int) -> Tuple[int, int]:
    if center_idx not in cycle:
        raise SystemExit("Center atom is not in cycle.")
    i = cycle.index(center_idx)
    left = cycle[(i - 1) % len(cycle)]
    right = cycle[(i + 1) % len(cycle)]
    return left, right


def _shortest_path_len_on_cycle(cycle: Sequence[int], a: int, b: int) -> int:
    n = len(cycle)
    ia = cycle.index(a)
    ib = cycle.index(b)
    d = abs(ia - ib)
    return min(d, n - d)


def _two_farthest_from_center_on_cycle(cycle: Sequence[int], center_idx: int) -> Tuple[int, int]:
    others = [idx for idx in cycle if idx != center_idx]
    pairs = [(_shortest_path_len_on_cycle(cycle, center_idx, idx), idx) for idx in others]
    max_d = max(d for d, _ in pairs)
    candidates = [idx for d, idx in pairs if d == max_d]
    if len(candidates) < 2:
        raise SystemExit("Unable to find two farthest ring atoms from coordinating N.")
    if len(candidates) > 2:
        candidates = candidates[:2]
    return candidates[0], candidates[1]


def _closest_two_and_opposite_two(
    ring_indices: Sequence[int],
    center_idx: int,
    atoms: Sequence[AtomRecord],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    if center_idx not in ring_indices:
        raise SystemExit("Center atom is not in inferred ring indices.")

    others = [idx for idx in ring_indices if idx != center_idx]
    if len(others) != 4:
        raise SystemExit(f"Expected 4 non-center ring atoms, got {len(others)}.")

    center = atoms[center_idx].coord
    ranked = sorted(others, key=lambda i: _dist(atoms[i].coord, center))
    adjacent = (ranked[0], ranked[1])
    opposite = (ranked[2], ranked[3])
    return adjacent, opposite


def _line_root_near_zero(
    zn: np.ndarray,
    pivot: np.ndarray,
    axis: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    def g(theta_deg: float) -> float:
        pts = np.vstack([a, b])
        rot = _rotate_points_about_pivot(pts, pivot, axis, theta_deg)
        da2 = _dist(rot[0], zn) ** 2
        db2 = _dist(rot[1], zn) ** 2
        return da2 - db2

    sample = np.linspace(-180.0, 180.0, 1441)
    vals = np.array([g(t) for t in sample])

    signs = np.sign(vals)
    for i in range(len(sample) - 1):
        if signs[i] == 0:
            return float(sample[i])
        if signs[i] * signs[i + 1] < 0:
            a_t = float(sample[i])
            b_t = float(sample[i + 1])
            a_v = float(vals[i])
            b_v = float(vals[i + 1])
            return a_t - a_v * (b_t - a_t) / (b_v - a_v)

    j = int(np.argmin(np.abs(vals)))
    return float(sample[j])


def _ring_normal(ring_coords: np.ndarray) -> np.ndarray:
    ctr = ring_coords.mean(axis=0)
    X = ring_coords - ctr
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    n = vt[-1]
    return _unit(n, name="ring normal")


def _build_local_frame(
    zn: np.ndarray,
    coord_n: np.ndarray,
    ring_coords: np.ndarray,
    adj_a: np.ndarray,
    adj_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # y-hat is Zn->N, axis passing through coordinating N.
    y_hat = _unit(coord_n - zn, name="y_hat (Zn->N)")

    # z-hat starts from ring normal, then orthogonalized to y-hat.
    z_raw = _ring_normal(ring_coords)
    z_perp = z_raw - np.dot(z_raw, y_hat) * y_hat

    if _norm(z_perp) < 1e-8:
        # Fallback: use midpoint direction in ring plane.
        v = 0.5 * (adj_a + adj_b) - coord_n
        v_perp = v - np.dot(v, y_hat) * y_hat
        if _norm(v_perp) < 1e-8:
            trial = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(trial, y_hat)) > 0.9:
                trial = np.array([0.0, 1.0, 0.0])
            v_perp = trial - np.dot(trial, y_hat) * y_hat
        z_perp = np.cross(y_hat, v_perp)

    z_hat = _unit(z_perp, name="z_hat (ring normal)")
    x_hat = _unit(np.cross(z_hat, y_hat), name="x_hat (ring-plane, perp y)")
    return x_hat, y_hat, z_hat


def _project_perp(v: np.ndarray, axis_u: np.ndarray) -> np.ndarray:
    return v - np.dot(v, axis_u) * axis_u


def _axis_from_opening(u: np.ndarray, helper: np.ndarray) -> np.ndarray:
    axis = np.cross(u, helper)
    if _norm(axis) < 1e-8:
        # fallback to any axis perpendicular to u
        trial = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(trial, u)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        axis = np.cross(u, trial)
    return _unit(axis, name="beta axis")


def _to_xyz_text(title: str, atoms: Sequence[AtomRecord]) -> str:
    lines = [str(len(atoms)), title]
    for atom in atoms:
        lines.append(f"{atom.element} {atom.x:.8f} {atom.y:.8f} {atom.z:.8f}")
    return "\n".join(lines) + "\n"


def _format_angle_for_path(value: float) -> str:
    return f"{value:+.0f}".replace("+", "").replace("-", "neg").replace(".", "pt")


def _parse_angle_list(text: str, label: str) -> List[float]:
    cleaned = text.strip()
    if not cleaned:
        raise SystemExit(f"--{label} cannot be empty.")
    cleaned = cleaned.replace("[", " ").replace("]", " ")
    parts = [p for p in re.split(r"[\s,]+", cleaned) if p]
    try:
        values = [float(p) for p in parts]
    except ValueError as exc:
        raise SystemExit(f"Invalid --{label} values: {text}") from exc
    if not values:
        raise SystemExit(f"--{label} did not contain any numeric values.")
    return values


def _broadcast_angles(
    alpha_vals: List[float],
    beta_vals: List[float],
    gamma_vals: List[float],
) -> Tuple[List[float], List[float], List[float], int]:
    n = max(len(alpha_vals), len(beta_vals), len(gamma_vals))

    def expand(values: List[float], name: str) -> List[float]:
        if len(values) == 1:
            return values * n
        if len(values) == n:
            return values
        raise SystemExit(
            f"--{name} has length {len(values)} but must be length 1 or {n} for broadcasting."
        )

    return expand(alpha_vals, "alpha"), expand(beta_vals, "beta"), expand(gamma_vals, "gamma"), n


def _is_any_nonzero(values: Sequence[float], tol: float = 1e-12) -> bool:
    return any(abs(v) > tol for v in values)


def _add_line(
    view: py3Dmol.view,
    start: np.ndarray,
    end: np.ndarray,
    color: str,
    radius: float = 0.06,
) -> None:
    view.addLine(
        {
            "start": {"x": float(start[0]), "y": float(start[1]), "z": float(start[2])},
            "end": {"x": float(end[0]), "y": float(end[1]), "z": float(end[2])},
            "dashed": False,
            "radius": radius,
            "color": color,
        }
    )


def _add_arrow(
    view: py3Dmol.view,
    start: np.ndarray,
    direction: np.ndarray,
    color: str,
    length: float = 1.0,
    radius: float = 0.06,
    radius_ratio: float = 1.8,
    mid: float = 0.78,
) -> None:
    d = _unit(direction, name="arrow direction") * length
    end = start + d
    view.addArrow(
        {
            "start": {"x": float(start[0]), "y": float(start[1]), "z": float(start[2])},
            "end": {"x": float(end[0]), "y": float(end[1]), "z": float(end[2])},
            "color": color,
            "radius": radius,
            "radiusRatio": radius_ratio,
            "mid": mid,
        }
    )


def _add_coordinate_axes(
    view: py3Dmol.view,
    origin: np.ndarray,
    x_hat: np.ndarray,
    y_hat: np.ndarray,
    z_hat: np.ndarray,
    size: float = 1.0,
) -> None:
    _add_arrow(view, origin, x_hat, "#E63946", length=size)  # x
    _add_arrow(view, origin, y_hat, "#2A9D8F", length=size)  # y
    _add_arrow(view, origin, z_hat, "#1D3557", length=size)  # z


def build_combined_visualization_html(
    original_atoms: Sequence[AtomRecord],
    rotated_atom_sets: Sequence[Sequence[AtomRecord]],
    coord_n_idx: int,
    x_hat: np.ndarray,
    y_hat: np.ndarray,
    z_hat: np.ndarray,
    alpha_list: Sequence[float],
    beta_list: Sequence[float],
    gamma_list: Sequence[float],
    out_html: Path,
) -> None:
    view = py3Dmol.view(width=1000, height=760)

    orig_xyz = _to_xyz_text("original", original_atoms)
    view.addModel(orig_xyz, "xyz")
    for idx, atoms in enumerate(rotated_atom_sets, start=1):
        view.addModel(_to_xyz_text(f"rotated_{idx}", atoms), "xyz")

    model_count = 1 + len(rotated_atom_sets)
    if model_count <= 1:
        opacities = [0.6]
    else:
        opacities = [0.6 + 0.4 * (i / (model_count - 1)) for i in range(model_count)]

    view.setStyle(
        {"model": 0},
        {
            "stick": {"radius": 0.09, "color": "#B0B0B0", "opacity": opacities[0]},
            "sphere": {"scale": 0.16, "color": "#B0B0B0", "opacity": opacities[0]},
        },
    )
    for model_idx in range(1, model_count):
        opacity = opacities[model_idx]
        view.setStyle(
            {"model": model_idx},
            {
                "stick": {"radius": 0.11, "colorscheme": "Jmol", "opacity": opacity},
                "sphere": {"scale": 0.18, "colorscheme": "Jmol", "opacity": opacity},
            },
        )

    n = rotated_atom_sets[-1][coord_n_idx].coord if rotated_atom_sets else original_atoms[coord_n_idx].coord
    _add_coordinate_axes(view, n, x_hat, y_hat, z_hat, size=1.0)

    view.zoomTo()
    view.setBackgroundColor("white")

    html = view._make_html()
    style = (
        "<style>body{margin:0;}"
        ".legend{position:fixed;left:12px;top:12px;background:rgba(255,255,255,0.92);"
        "padding:8px 10px;border:1px solid #ddd;border-radius:6px;font-family:Arial,sans-serif;font-size:12px;z-index:10;}"
        "</style>"
    )
    legend = (
        "<div class='legend'>"
        f"<div><b>α={list(alpha_list)}</b></div>"
        f"<div><b>β={list(beta_list)}</b></div>"
        f"<div><b>γ={list(gamma_list)}</b></div>"
        "</div>"
    )
    if "<head>" in html:
        html = html.replace("<head>", f"<head>{style}", 1)
    else:
        html = style + html
    if "<body>" in html:
        html = html.replace("<body>", f"<body>{legend}", 1)
    else:
        html = legend + html

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")


def _measured_angles(
    zn: np.ndarray,
    n: np.ndarray,
    far_midpoint: np.ndarray,
    ring_coords: np.ndarray,
    adj_a: np.ndarray,
    adj_b: np.ndarray,
) -> Dict[str, float]:
    u = _unit(zn - n, name="Zn->N")
    v = _unit(far_midpoint - n, name="N->bisector")
    beta = _angle_deg(u, v)

    # Alpha proxy: azimuth of ring normal about Zn->N axis.
    ring_n = _ring_normal(ring_coords)
    v_perp = _project_perp(v, u)
    if _norm(v_perp) < 1e-8:
        v_perp = _project_perp((adj_a + adj_b) * 0.5 - n, u)
    if _norm(v_perp) < 1e-8:
        trial = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(trial, u)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        v_perp = _project_perp(trial, u)
    e1 = _unit(v_perp, name="alpha e1")
    e2 = _unit(np.cross(u, e1), name="alpha e2")
    alpha = math.degrees(math.atan2(np.dot(ring_n, e2), np.dot(ring_n, e1)))

    # Gamma proxy: signed asymmetry of adjacent atoms w.r.t Zn distance.
    da = _dist(adj_a, zn)
    db = _dist(adj_b, zn)
    gamma = (da - db) * 20.0

    return {"alpha": alpha, "beta": beta, "gamma": gamma}


def _normalize_angle_cli_tokens(argv: Sequence[str]) -> List[str]:
    """Allow forms like: --alpha -10,-5,0,5,10

    argparse can treat a next token beginning with '-' as another option.
    Convert to --alpha=-10,-5,0,5,10 for alpha/beta/gamma when needed.
    """
    angle_flags = {"--alpha", "--beta", "--gamma"}
    out: List[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in angle_flags and i + 1 < len(argv):
            nxt = argv[i + 1]
            if nxt.startswith("-") and not nxt.startswith("--"):
                out.append(f"{token}={nxt}")
                i += 2
                continue
        out.append(token)
        i += 1
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rotate a single RESSEQ residue in an annotated XYZ using alpha/beta/gamma conventions."
    )
    parser.add_argument("xyz", help="Input annotated XYZ (axyz-like) file")
    parser.add_argument("--resseq", type=int, required=True, help="Target residue sequence number")
    parser.add_argument(
        "--alpha",
        required=True,
        help="Alpha angle(s) in degrees. Accepts scalar or list: '15' or '0,15,30'.",
    )
    parser.add_argument(
        "--beta",
        required=True,
        help="Beta angle(s) in degrees. Accepts scalar or list: '0' or '0,5,10'.",
    )
    parser.add_argument(
        "--gamma",
        required=True,
        help="Gamma angle(s) in degrees. Accepts scalar or list: '0' or '0,15,30'.",
    )
    parser.add_argument("--output-xyz", help="Output transformed XYZ path")
    parser.add_argument("--output-html", help="Output py3Dmol HTML path")
    parser.add_argument(
        "--report-json",
        help="Optional path to write a JSON report containing measured original and applied angles",
    )
    return parser.parse_args(argv)


def main() -> None:
    normalized_argv = _normalize_angle_cli_tokens(sys.argv[1:])
    args = parse_args(normalized_argv)
    in_path = Path(args.xyz)
    _, title, atoms = parse_annotated_xyz(in_path)

    if not atoms:
        raise SystemExit("No atoms parsed from input XYZ.")

    alpha_vals = _parse_angle_list(args.alpha, "alpha")
    beta_vals = _parse_angle_list(args.beta, "beta")
    gamma_vals = _parse_angle_list(args.gamma, "gamma")
    alpha_vals, beta_vals, gamma_vals, n_sets = _broadcast_angles(alpha_vals, beta_vals, gamma_vals)

    if args.output_html:
        out_html = Path(args.output_html)
    else:
        changed = []
        if _is_any_nonzero(alpha_vals):
            changed.append("alpha")
        if _is_any_nonzero(beta_vals):
            changed.append("beta")
        if _is_any_nonzero(gamma_vals):
            changed.append("gamma")
        changed_tag = "-".join(changed) if changed else "none"
        out_html = in_path.with_name(f"{in_path.stem}-resseq{args.resseq}-{changed_tag}-angles.html")

    base_xyz = Path(args.output_xyz) if args.output_xyz else None

    res_atoms = [
        i for i, a in enumerate(atoms) if a.meta.get("RESSEQ") == str(args.resseq)
    ]
    if not res_atoms:
        raise SystemExit(f"No atoms with RESSEQ={args.resseq} found.")

    zn_idx = next((i for i, a in enumerate(atoms) if _el(a) == "ZN"), None)
    if zn_idx is None:
        raise SystemExit("No Zn atom found.")

    zn = atoms[zn_idx].coord
    residue_n = [i for i in res_atoms if _el(atoms[i]) == "N"]
    if not residue_n:
        raise SystemExit(f"No N atoms in RESSEQ={args.resseq}; cannot identify coordinating N.")

    coord_n_idx = min(residue_n, key=lambda i: _dist(atoms[i].coord, zn))
    coord_n = atoms[coord_n_idx].coord

    ring_cycle = _select_imidazole_ring(res_atoms, atoms, coord_n_idx)
    adj_pair, (far_a_idx, far_b_idx) = _closest_two_and_opposite_two(ring_cycle, coord_n_idx, atoms)

    original_atoms = [AtomRecord(a.element, a.x, a.y, a.z, a.comment, dict(a.meta)) for a in atoms]
    measured_original = {"alpha": 0.0, "beta": 0.0, "gamma": 0.0}

    # --- Input geometry is the zero state ---
    ring_coords_zero = np.vstack([atoms[idx].coord for idx in ring_cycle])
    x_hat, y_hat, z_hat = _build_local_frame(
        zn,
        coord_n,
        ring_coords_zero,
        atoms[adj_pair[0]].coord,
        atoms[adj_pair[1]].coord,
    )

    movable_idx = [i for i in res_atoms if i != coord_n_idx]
    rotated_sets: List[List[AtomRecord]] = []
    out_xyz_files: List[str] = []

    for i in range(n_sets):
        alpha_i = alpha_vals[i]
        beta_i = beta_vals[i]
        gamma_i = gamma_vals[i]

        working_atoms = [AtomRecord(a.element, a.x, a.y, a.z, a.comment, dict(a.meta)) for a in original_atoms]
        movable_pts = np.vstack([working_atoms[idx].coord for idx in movable_idx])

        movable_pts = _rotate_points_about_pivot(movable_pts, coord_n, y_hat, alpha_i)
        movable_pts = _rotate_points_about_pivot(movable_pts, coord_n, x_hat, beta_i)
        movable_pts = _rotate_points_about_pivot(movable_pts, coord_n, z_hat, gamma_i)
        for k, idx in enumerate(movable_idx):
            working_atoms[idx].set_coord(movable_pts[k])

        angle_suffix_i = (
            f"-alpha-{_format_angle_for_path(alpha_i)}"
            f"-beta-{_format_angle_for_path(beta_i)}"
            f"-gamma-{_format_angle_for_path(gamma_i)}"
        )
        if base_xyz is None:
            out_xyz = in_path.with_name(f"{in_path.stem}-resseq{args.resseq}{angle_suffix_i}.xyz")
        else:
            stem = base_xyz.stem
            suffix = base_xyz.suffix if base_xyz.suffix else ".xyz"
            out_xyz = base_xyz.with_name(f"{stem}{angle_suffix_i}{suffix}")

        transformed_title = (
            f"{title} | rotated RESSEQ={args.resseq} alpha={alpha_i:.3f} beta={beta_i:.3f} gamma={gamma_i:.3f}"
        )
        write_annotated_xyz(out_xyz, transformed_title, working_atoms)
        out_xyz_files.append(str(out_xyz))
        rotated_sets.append(working_atoms)

    build_combined_visualization_html(
        original_atoms=original_atoms,
        rotated_atom_sets=rotated_sets,
        coord_n_idx=coord_n_idx,
        x_hat=x_hat,
        y_hat=y_hat,
        z_hat=z_hat,
        alpha_list=alpha_vals,
        beta_list=beta_vals,
        gamma_list=gamma_vals,
        out_html=out_html,
    )

    report = {
        "input_file": str(in_path),
        "output_xyz_files": out_xyz_files,
        "output_html": str(out_html),
        "resseq": args.resseq,
        "coordinating_n_index": coord_n_idx,
        "ring_cycle_indices": list(ring_cycle),
        "adjacent_pair_indices": [adj_pair[0], adj_pair[1]],
        "farthest_pair_indices": [far_a_idx, far_b_idx],
        "canonicalization": {
            "align_angle_deg": 0.0,
            "equalize_angle_deg": 0.0,
        },
        "requested_angles_deg": {
            "alpha": alpha_vals,
            "beta": beta_vals,
            "gamma": gamma_vals,
        },
        "measured_original_deg": measured_original,
    }

    print(json.dumps(report, indent=2))

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
