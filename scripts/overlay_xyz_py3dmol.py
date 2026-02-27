#!/usr/bin/env python3
"""Overlay all .xyz structures in a directory into a single py3Dmol HTML view.

Usage:
  python scripts/overlay_xyz_py3dmol.py \
    --input macchiato_downloads \
    --output macchiato_downloads/overlay.html

    python scripts/overlay_xyz_py3dmol.py \
        --input-file path/to/a.xyz --label "Structure A" \
        --input-file path/to/b.xyz --label "Structure B" \
        --output overlay.html

Notes:
- Requires py3Dmol (pip install py3Dmol)
- Hover labels show the XYZ filename on atoms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import py3Dmol


DEFAULT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def read_xyz_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_ca_coords_from_xyz(xyz_text: str) -> list[tuple[float, float, float]]:
    coords: list[tuple[float, float, float]] = []
    for line in xyz_text.splitlines():
        if "ATOM=CA" not in line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
        except ValueError:
            continue
        coords.append((x, y, z))
    return coords


def build_viewer(
    xyz_files: list[Path],
    hover_labels: list[str],
    width: int,
    height: int,
) -> py3Dmol.view:
    view = py3Dmol.view(width=width, height=height)
    ca_coords: list[tuple[float, float, float]] = []

    for idx, xyz_path in enumerate(xyz_files):
        xyz_text = read_xyz_text(xyz_path)
        color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]

        model = view.addModel(xyz_text, "xyz")
        model.setStyle(
            {},
            {
                "stick": {"radius": 0.09, "colorscheme": color},
                "sphere": {"scale": 0.12, "colorscheme": color},
            },
        )

        if "original" in xyz_path.name.lower():
            ca_coords.extend(parse_ca_coords_from_xyz(xyz_text))

    if ca_coords:
        for x, y, z in ca_coords:
            view.addSphere(
                {
                    "center": {"x": x, "y": y, "z": z},
                    "radius": 0.23,
                    "color": "black",
                    "opacity": 1.0,
                }
            )

    names_json = json.dumps(hover_labels)
    view.setHoverable(
        {},
        True,
        f"function(atom,viewer,event,container){{var names={names_json}; var bar=document.getElementById('hover-bar'); if(bar){{bar.textContent=names[atom.model] || '';}}}}",
        "function(atom,viewer,event,container){var bar=document.getElementById('hover-bar'); if(bar){bar.textContent='';}}",
    )

    view.zoomTo()
    view.setBackgroundColor("white")
    return view


def wrap_html_with_hover_bar(html: str) -> str:
    bar_html = "<div id=\"hover-bar\">Hover over a structure to see its label</div>"
    style_block = (
        "<style>"
        "body{margin:0;padding-top:44px;font-family:Arial,Helvetica,sans-serif;}"
        "#hover-bar{position:fixed;top:0;left:0;right:0;height:44px;"
        "background:#111;color:#fff;display:flex;align-items:center;"
        "padding:0 16px;box-sizing:border-box;font-size:14px;z-index:9999;}"
        "</style>"
    )

    if "<head>" in html:
        html = html.replace("<head>", f"<head>{style_block}", 1)
    else:
        html = f"{style_block}\n{html}"

    if "<body>" in html:
        html = html.replace("<body>", f"<body>\n{bar_html}", 1)
    else:
        html = f"{bar_html}\n{html}"

    return html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay all XYZ files in a directory into one py3Dmol HTML view."
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Directory containing .xyz files",
    )
    parser.add_argument(
        "--input-file",
        "-f",
        action="append",
        default=[],
        help="Path to an individual .xyz file (can be provided multiple times)",
    )
    parser.add_argument(
        "--label",
        "-l",
        action="append",
        default=[],
        help="Hover label for each --input-file (must match count when provided)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output HTML path",
    )
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=700)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    xyz_files: list[Path] = []
    hover_labels: list[str] = []

    if args.input_file:
        xyz_files = [Path(p) for p in args.input_file]
        for xyz_path in xyz_files:
            if not xyz_path.exists() or not xyz_path.is_file():
                raise SystemExit(f"Input file not found: {xyz_path}")

        if args.label and len(args.label) != len(xyz_files):
            raise SystemExit(
                "When using --label, provide exactly one label for each --input-file."
            )

        hover_labels = args.label if args.label else [p.name for p in xyz_files]
    elif args.input:
        input_dir = Path(args.input)
        if not input_dir.exists() or not input_dir.is_dir():
            raise SystemExit(f"Input directory not found: {input_dir}")

        xyz_files = list(input_dir.glob("*.xyz"))
        subdirs = [p for p in input_dir.iterdir() if p.is_dir()]
        for subdir in subdirs:
            xyz_files.extend(subdir.glob("*.xyz"))
        xyz_files = sorted(set(xyz_files))
        hover_labels = [p.name for p in xyz_files]
    else:
        raise SystemExit("Provide either --input directory or one/more --input-file values.")

    if not xyz_files:
        if args.input:
            raise SystemExit(f"No .xyz files found in {args.input}")
        raise SystemExit("No .xyz input files were provided.")

    view = build_viewer(
        xyz_files,
        hover_labels=hover_labels,
        width=args.width,
        height=args.height,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = wrap_html_with_hover_bar(view._make_html())
    output_path.write_text(html, encoding="utf-8")

    print(f"Wrote overlay HTML to: {output_path}")


if __name__ == "__main__":
    main()
