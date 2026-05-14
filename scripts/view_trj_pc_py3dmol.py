#!/usr/bin/env python3
"""Build an interactive py3Dmol HTML for a trajectory XYZ plus its .pc charges.

This script renders the same compact style as scripts/view_xyz_pc_py3dmol.py
(trajectory + .pc points, no protein cartoon). The XYZ is animated frame-by-frame,
and the final frame is repeated a configurable number of times to create a short
end pause before looping again.

Usage examples:
    # Auto-find one *_trj.xyz and one .pc in the directory
  uv run python scripts/view_trj_pc_py3dmol.py \
    --input-dir path/to/xyz_dir

    # Auto-find files and choose custom output
  uv run python scripts/view_trj_pc_py3dmol.py \
    --input-dir path/to/xyz_dir \
        --output path/to/xyz_dir/custom.viewer.html \
    --pause-frames 12
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import py3Dmol


@dataclass
class PcPoint:
    charge: float
    x: float
    y: float
    z: float


@dataclass
class PdbAtomPoint:
    atom_name: str
    resname: str
    element: str
    x: float
    y: float
    z: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a py3Dmol HTML viewer for *_trj.xyz + .pc with trajectory animation "
            "and a short pause on the last frame."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing exactly one *_trj.xyz and one .pc file",
    )
    parser.add_argument(
        "--output",
        help="Output HTML path (default: <trj stem>.viewer.html in --input-dir)",
    )
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--height", type=int, default=760)
    parser.add_argument(
        "--pause-frames",
        type=int,
        default=10,
        help="Number of extra copies of the last frame to append for end pause",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=120,
        help="Animation frame interval in milliseconds",
    )
    return parser.parse_args()


def resolve_single_file(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise SystemExit(f"No {label} found in {directory} matching '{pattern}'")
    if len(matches) > 1:
        names = "\n  - ".join(str(p) for p in matches)
        raise SystemExit(
            f"Found multiple {label} files in {directory}; expected exactly one and will not continue:\n"
            f"  - {names}"
        )
    return matches[0]


def resolve_traj_path(input_dir: Path) -> Path:
    return resolve_single_file(input_dir, "*_trj.xyz", "traj")


def resolve_pc_path(input_dir: Path) -> Path:
    return resolve_single_file(input_dir, "*.pc", "pc")


def resolve_start_xyz_path(input_dir: Path) -> Path:
    return resolve_single_file(input_dir, "*_start.xyz", "start-xyz")


def resolve_pdb_path(input_dir: Path) -> Path:
    return resolve_single_file(input_dir, "*.pdb", "pdb")


def resolve_output_path(input_dir: Path, traj_path: Path, output_arg: str | None) -> Path:
    _ = traj_path
    if output_arg:
        return Path(output_arg)
    return input_dir / f"{input_dir.name}_traj.viewer.html"


def parse_pc(path: Path) -> list[PcPoint]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Empty .pc file: {path}")

    try:
        declared = int(lines[0])
    except ValueError as exc:
        raise ValueError(f"First line of .pc must be integer count: {path}") from exc

    points: list[PcPoint] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"Expected 4 columns in .pc line: '{line}'")
        q, x, y, z = map(float, parts)
        points.append(PcPoint(q, x, y, z))

    if declared != len(points):
        raise ValueError(
            f"Point-count mismatch in {path}: header={declared}, rows={len(points)}"
        )

    return points


def split_xyz_frames(xyz_text: str) -> list[str]:
    """Split concatenated XYZ trajectory text into per-frame XYZ blocks."""
    lines = xyz_text.splitlines()
    frames: list[str] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue

        try:
            atom_count = int(lines[i].strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid XYZ trajectory format at line {i + 1}: expected atom count"
            ) from exc

        end = i + atom_count + 2
        if end > len(lines):
            raise ValueError(
                f"Truncated XYZ frame starting at line {i + 1}: expected {atom_count} atoms"
            )

        frame_lines = lines[i:end]
        frames.append("\n".join(frame_lines) + "\n")
        i = end

    if not frames:
        raise ValueError("No XYZ frames found in trajectory file")

    return frames


def parse_centroid_from_start_xyz(start_xyz_text: str) -> tuple[float, float, float] | None:
    lines = start_xyz_text.splitlines()
    if len(lines) < 2:
        return None
    match = re.search(r"CENTROID=\(([^)]+)\)", lines[1])
    if not match:
        return None
    parts = [p.strip() for p in match.group(1).split(",")]
    if len(parts) != 3:
        return None
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return None


def parse_xyz_atoms(frame_text: str) -> list[tuple[str, float, float, float]]:
    lines = frame_text.splitlines()
    if len(lines) < 2:
        return []
    atoms: list[tuple[str, float, float, float]] = []
    for line in lines[2:]:
        raw = line.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) < 4:
            continue
        try:
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
        except ValueError:
            continue
        atoms.append((parts[0].upper(), x, y, z))
    return atoms


def find_metal_in_xyz_frame(frame_text: str, target_element: str = "ZN") -> tuple[float, float, float] | None:
    target = target_element.upper()
    atoms = parse_xyz_atoms(frame_text)
    for element, x, y, z in atoms:
        if element == target:
            return (x, y, z)
    return None


def parse_pdb_atoms(pdb_text: str) -> list[PdbAtomPoint]:
    atoms: list[PdbAtomPoint] = []
    for line in pdb_text.splitlines():
        record = line[0:6].strip().upper() if len(line) >= 6 else ""
        if record not in {"ATOM", "HETATM"}:
            continue
        if len(line) < 54:
            continue

        atom_name = line[12:16].strip().upper()
        resname = line[17:20].strip().upper() if len(line) >= 20 else ""
        altloc = line[16:17].strip().upper() if len(line) >= 17 else ""
        if altloc not in {"", "A"}:
            continue

        element = line[76:78].strip().title() if len(line) >= 78 else ""
        if not element:
            element = _infer_element_from_atom_name(atom_name)
        if not element:
            continue

        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue

        atoms.append(
            PdbAtomPoint(
                atom_name=atom_name,
                resname=resname,
                element=element,
                x=x,
                y=y,
                z=z,
            )
        )

    return atoms


def find_pdb_metal_near_centroid(
    pdb_text: str,
    centroid: tuple[float, float, float] | None,
    target_element: str = "ZN",
) -> tuple[float, float, float] | None:
    atoms = parse_pdb_atoms(pdb_text)
    target = target_element.upper()
    metal_atoms = [a for a in atoms if a.element.upper() == target]
    if not metal_atoms:
        return None

    if centroid is None:
        a = metal_atoms[0]
        return (a.x, a.y, a.z)

    cx, cy, cz = centroid
    best_atom = min(
        metal_atoms,
        key=lambda a: (a.x - cx) ** 2 + (a.y - cy) ** 2 + (a.z - cz) ** 2,
    )
    return (best_atom.x, best_atom.y, best_atom.z)


def translate_pdb_text(pdb_text: str, dx: float, dy: float, dz: float) -> str:
    out: list[str] = []
    for line in pdb_text.splitlines():
        record = line[0:6].strip().upper() if len(line) >= 6 else ""
        if record in {"ATOM", "HETATM"} and len(line) >= 54:
            try:
                x = float(line[30:38]) + dx
                y = float(line[38:46]) + dy
                z = float(line[46:54]) + dz
                line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
            except ValueError:
                pass
        out.append(line)
    return "\n".join(out) + "\n"


def build_translated_pdb_overlay(
    input_dir: Path,
    first_traj_frame: str,
) -> str | None:
    try:
        start_xyz_path = resolve_start_xyz_path(input_dir)
        pdb_path = resolve_pdb_path(input_dir)
    except SystemExit:
        return None

    start_xyz_text = start_xyz_path.read_text(encoding="utf-8")
    centroid = parse_centroid_from_start_xyz(start_xyz_text)

    traj_zn = find_metal_in_xyz_frame(first_traj_frame, target_element="ZN")
    if traj_zn is None:
        return None

    pdb_text = pdb_path.read_text(encoding="utf-8", errors="replace")
    pdb_zn = find_pdb_metal_near_centroid(pdb_text, centroid, target_element="ZN")
    if pdb_zn is None:
        return None

    dx = traj_zn[0] - pdb_zn[0]
    dy = traj_zn[1] - pdb_zn[1]
    dz = traj_zn[2] - pdb_zn[2]
    return translate_pdb_text(pdb_text, dx, dy, dz)


def infer_element_from_charge(q: float) -> str | None:
    refs = {
        "N": -0.4157,
        "C": 0.5973,
        "O": -0.5679,
    }
    best_name = None
    best_diff = 1e9
    for name, ref_q in refs.items():
        d = abs(q - ref_q)
        if d < best_diff:
            best_diff = d
            best_name = name
    if best_diff <= 0.08:
        return best_name
    return None


def _infer_element_from_atom_name(atom_name: str) -> str:
    name = (atom_name or "").strip()
    if not name:
        return ""
    if len(name) >= 2 and name[0].isalpha() and name[1].islower():
        return name[:2].title()
    for ch in name:
        if ch.isalpha():
            return ch.upper()
    return ""


def infer_pc_points(points: list[PcPoint], pdb_text: str | None) -> list[tuple[str, str, str]]:
    """
    Infer (element, atom_name) for each .pc point.

    Priority:
    1) nearest translated atom in aligned PDB
    2) fallback by AMBER charge proximity
    """
    inferred: list[tuple[str, str, str]] = []
    pdb_points = parse_pdb_atoms(pdb_text) if pdb_text else []

    for p in points:
        element = None
        atom_name = ""
        resname = ""
        if pdb_points:
            best_idx = None
            best_d2 = 1e18
            for idx, b in enumerate(pdb_points):
                dx = p.x - b.x
                dy = p.y - b.y
                dz = p.z - b.z
                d2 = dx * dx + dy * dy + dz * dz
                if d2 < best_d2:
                    best_d2 = d2
                    best_idx = idx
            # 0.30 A tolerance handles translation/rounding drift while staying specific.
            if best_idx is not None and best_d2 <= 0.09:
                best = pdb_points[best_idx]
                element = best.element
                atom_name = best.atom_name
                resname = best.resname

        if element is None:
            element = infer_element_from_charge(p.charge)
            atom_name = ""
            resname = ""

        inferred.append((element if element is not None else "UNK", atom_name, resname))

    return inferred


def build_pc_xyz(points: list[PcPoint], elements: list[str]) -> str:
    body: list[str] = []
    for p, element in zip(points, elements):
        symbol = (element or "").strip().title() or "He"
        body.append(f"{symbol} {p.x:.6f} {p.y:.6f} {p.z:.6f}")
    return f"{len(points)}\nPOINT_CHARGES\n" + "\n".join(body) + "\n"


def build_html(
    traj_frames: list[str],
    points: list[PcPoint],
    output_path: Path,
    width: int,
    height: int,
    pause_frames: int,
    interval_ms: int,
    pdb_text: str | None,
) -> None:
    frame_blocks = list(traj_frames)
    if pause_frames > 0:
        frame_blocks.extend([traj_frames[-1]] * pause_frames)
    traj_xyz = "".join(frame_blocks)

    view = py3Dmol.view(width=width, height=height)

    # model 0: animated trajectory frames
    view.addModelsAsFrames(traj_xyz, "xyz")
    view.setStyle(
        {"model": 0},
        {
            "stick": {"radius": 0.16, "colorscheme": "Jmol"},
            "sphere": {"scale": 0.25, "colorscheme": "Jmol"},
        },
    )

    next_model = 1
    if pdb_text:
        view.addModel(pdb_text, "pdb")
        view.setStyle(
            {"model": 1},
            {
                "cartoon": {"color": "lightgray", "opacity": 0.25},
                "line": {"color": "gray", "opacity": 0.22},
            },
        )
        next_model = 2

    inferred_pc = infer_pc_points(points, pdb_text)
    pc_elements = [x[0] for x in inferred_pc]
    pc_atom_names = [x[1] for x in inferred_pc]
    pc_resnames = [x[2] for x in inferred_pc]
    pc_xyz = build_pc_xyz(points, pc_elements)
    view.addModel(pc_xyz, "xyz")
    pc_model_id = next_model

    view.setStyle(
        {"model": pc_model_id},
        {
            "sphere": {
                "radius": 0.36,
                "colorscheme": "Jmol",
                "opacity": 0.40,
            }
        },
    )

    point_labels = [
        (
            f"elem={pc_elements[i]}"
            + (f" RES={pc_resnames[i]}" if pc_resnames[i] else "")
            + (f" atom={pc_atom_names[i]}" if pc_atom_names[i] else "")
            + f"  q={p.charge:+.4f} e  x={p.x:.3f}  y={p.y:.3f}  z={p.z:.3f}"
        )
        for i, p in enumerate(points)
    ]

    labels_json = json.dumps(point_labels)
    hover_on_pc = (
        "function(atom, viewer) {"
        f"var labels={labels_json};"
        "var idx = (atom.index !== undefined) ? atom.index : (atom.serial !== undefined ? atom.serial - 1 : -1);"
        "var msg = (idx >= 0 && idx < labels.length) ? labels[idx] : 'point-charge';"
        "var el = document.getElementById('hoverinfo'); if (el) { el.textContent = msg; }"
        "}"
    )
    hover_off = (
        "function(atom, viewer) {"
        "var el = document.getElementById('hoverinfo'); if (el) { el.textContent = ''; }"
        "}"
    )
    view.setHoverable({"model": pc_model_id}, True, hover_on_pc, hover_off)

    view.setHoverable(
        {"model": 0},
        True,
        "function(atom, viewer) { var el = document.getElementById('hoverinfo'); if (el) { el.textContent = (atom.elem || 'atom') + ' (' + atom.x.toFixed(3) + ', ' + atom.y.toFixed(3) + ', ' + atom.z.toFixed(3) + ')'; } }",
        hover_off,
    )

    view.addSphere(
        {
            "center": {"x": 0.0, "y": 0.0, "z": 0.0},
            "radius": 0.18,
            "color": "black",
            "opacity": 1.0,
        }
    )

    view.zoomTo()
    view.setBackgroundColor("white")
    view.animate({"loop": "forward", "interval": interval_ms})

    html = view._make_html()
    title = output_path.stem
    header_html = (
        '<div id="hoverwrap">'
        f'<div id="title">{title}</div>'
        '<div id="legend">'
        'Trajectory XYZ: sticks/spheres | .pc points: element-colored (Jmol), alpha=0.40'
        '</div>'
        '<div id="hoverinfo">Hover over .pc points to inspect q, x, y, z</div>'
        '</div>'
    )
    style = (
        "<style>"
        "body{margin:0;padding-top:70px;font-family:Helvetica,Arial,sans-serif;background:#f5f5f3;}"
        "#hoverwrap{position:fixed;top:0;left:0;right:0;height:70px;"
        "background:#1e2a2f;color:#eef3f5;z-index:9999;padding:8px 14px;box-sizing:border-box;}"
        "#title{font-size:14px;font-weight:700;line-height:18px;}"
        "#legend{font-size:12px;opacity:0.92;line-height:16px;}"
        "#hoverinfo{font-size:12px;color:#f8dfb6;line-height:16px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}"
        "</style>"
    )

    if "<head>" in html:
        html = html.replace("<head>", "<head>" + style, 1)
    else:
        html = style + html

    if "<body>" in html:
        html = html.replace("<body>", "<body>" + header_html, 1)
    else:
        html = header_html + html

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    traj_path = resolve_traj_path(input_dir)
    pc_path = resolve_pc_path(input_dir)
    output_path = resolve_output_path(input_dir, traj_path, args.output)

    if not traj_path.exists():
        raise SystemExit(f"Trajectory XYZ file not found: {traj_path}")
    if not pc_path.exists():
        raise SystemExit(f"PC file not found: {pc_path}")

    traj_text = traj_path.read_text(encoding="utf-8")
    traj_frames = split_xyz_frames(traj_text)
    points = parse_pc(pc_path)
    pdb_text = build_translated_pdb_overlay(input_dir, traj_frames[0])

    build_html(
        traj_frames=traj_frames,
        points=points,
        output_path=output_path,
        width=args.width,
        height=args.height,
        pause_frames=max(args.pause_frames, 0),
        interval_ms=max(args.interval_ms, 1),
        pdb_text=pdb_text,
    )

    print(f"Trajectory file:      {traj_path}")
    print(f"Point-charge file:    {pc_path}")
    print(f"Frames in trajectory: {len(traj_frames)}")
    print(f"Pause frames added:   {max(args.pause_frames, 0)}")
    print(f"Protein overlay:      {'enabled' if pdb_text else 'disabled'}")
    print(f"Wrote viewer HTML:    {output_path}")


if __name__ == "__main__":
    main()
