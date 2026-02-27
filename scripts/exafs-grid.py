#!/usr/bin/env python3
"""Build a single HTML grid for EXAFS PNGs under macchiato_downloads."""

from __future__ import annotations

import argparse
from pathlib import Path

from generate_combined_figures import png_to_base64


ORDERED_GROUPS = [
    ("4cys", "4Cys (0His)"),
    ("3cys1his", "3Cys1His"),
    ("2cys2his", "2Cys2His"),
    ("1cys3his", "1Cys3His"),
    ("4his", "0Cys4His"),
]


def _assign_group(path: Path) -> str:
    name = path.as_posix().lower()
    for key, _label in ORDERED_GROUPS:
        if key in name:
            return key
    return "other"


def _collect_pngs(root: Path, pattern: str) -> list[Path]:
    return [p for p in root.rglob(pattern) if p.is_file()]


def _build_ordered_list(paths: list[Path]) -> list[Path]:
    grouped: dict[str, list[Path]] = {key: [] for key, _ in ORDERED_GROUPS}
    other: list[Path] = []
    for path in paths:
        key = _assign_group(path)
        if key in grouped:
            grouped[key].append(path)
        else:
            other.append(path)

    ordered: list[Path] = []
    for key, _label in ORDERED_GROUPS:
        ordered.extend(sorted(grouped[key]))
    ordered.extend(sorted(other))
    return ordered


def _render_html(png_files: list[Path], *, title: str, columns: int) -> str:
    html_parts = [
        "<!DOCTYPE html>\n",
        "<html lang=\"en\">\n",
        "<head>\n",
        "    <meta charset=\"UTF-8\">\n",
        "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n",
        f"    <title>{title}</title>\n",
        "    <style>\n",
        "        body {\n",
        "            font-family: Arial, sans-serif;\n",
        "            margin: 20px;\n",
        "            background-color: #f5f5f5;\n",
        "        }\n",
        "        h1 {\n",
        "            text-align: center;\n",
        "            color: #333;\n",
        "            margin-bottom: 30px;\n",
        "        }\n",
        "        .grid-container {\n",
        f"            grid-template-columns: repeat({columns}, 1fr);\n",
        "            display: grid;\n",
        "            gap: 20px;\n",
        "            max-width: 1800px;\n",
        "            margin: 0 auto;\n",
        "        }\n",
        "        .figure-item {\n",
        "            background: white;\n",
        "            padding: 15px;\n",
        "            border-radius: 8px;\n",
        "            box-shadow: 0 2px 4px rgba(0,0,0,0.1);\n",
        "            display: flex;\n",
        "            flex-direction: column;\n",
        "            align-items: center;\n",
        "        }\n",
        "        .figure-item img {\n",
        "            width: 100%;\n",
        "            height: auto;\n",
        "            border-radius: 4px;\n",
        "            cursor: pointer;\n",
        "            transition: transform 0.2s;\n",
        "        }\n",
        "        .figure-item img:hover {\n",
        "            transform: scale(1.02);\n",
        "        }\n",
        "        .modal {\n",
        "            display: none;\n",
        "            position: fixed;\n",
        "            z-index: 1000;\n",
        "            left: 0;\n",
        "            top: 0;\n",
        "            width: 100%;\n",
        "            height: 100%;\n",
        "            background-color: rgba(0,0,0,0.9);\n",
        "            cursor: pointer;\n",
        "        }\n",
        "        .modal img {\n",
        "            position: absolute;\n",
        "            top: 50%;\n",
        "            left: 50%;\n",
        "            transform: translate(-50%, -50%);\n",
        "            max-width: 95%;\n",
        "            max-height: 95%;\n",
        "        }\n",
        "        .close {\n",
        "            position: absolute;\n",
        "            top: 20px;\n",
        "            right: 40px;\n",
        "            color: white;\n",
        "            font-size: 40px;\n",
        "            font-weight: bold;\n",
        "            cursor: pointer;\n",
        "        }\n",
        "    </style>\n",
        "</head>\n",
        "<body>\n",
        f"    <h1>{title}</h1>\n",
        "    <div class=\"grid-container\">\n",
    ]

    for png_file in png_files:
        base64_data = png_to_base64(png_file)
        caption = png_file.name
        html_parts.append(
            "        <div class=\"figure-item\">\n"
            f"            <img src=\"data:image/png;base64,{base64_data}\" alt=\"{caption}\" onclick=\"openModal(this.src)\">\n"
            "        </div>\n\n"
        )

    html_parts.extend(
        [
            "    </div>\n\n",
            "    <div id=\"imageModal\" class=\"modal\" onclick=\"closeModal()\">\n",
            "        <span class=\"close\">&times;</span>\n",
            "        <img id=\"modalImg\">\n",
            "    </div>\n\n",
            "    <script>\n",
            "        function openModal(src) {\n",
            "            document.getElementById('imageModal').style.display = 'block';\n",
            "            document.getElementById('modalImg').src = src;\n",
            "        }\n\n",
            "        function closeModal() {\n",
            "            document.getElementById('imageModal').style.display = 'none';\n",
            "        }\n\n",
            "        document.addEventListener('keydown', function(event) {\n",
            "            if (event.key === 'Escape') {\n",
            "                closeModal();\n",
            "            }\n",
            "        });\n",
            "    </script>\n",
            "</body>\n",
            "</html>\n",
        ]
    )

    return "".join(html_parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a single HTML grid for EXAFS*.png under macchiato_downloads.",
    )
    parser.add_argument(
        "--root",
        default="macchiato_downloads",
        help="Root directory to search (default: macchiato_downloads).",
    )
    parser.add_argument(
        "--pattern",
        default="EXAFS*.png",
        help="Filename glob to match (default: EXAFS*.png).",
    )
    parser.add_argument(
        "--out-dir",
        default="figures",
        help="Output directory for the HTML (default: figures).",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=3,
        help="Number of grid columns (default: 3).",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root directory not found: {root}")

    png_files = _collect_pngs(root, args.pattern)
    if not png_files:
        raise SystemExit(f"No PNG files matched {args.pattern} under {root}")

    ordered = _build_ordered_list(png_files)
    title = "EXAFS Grid (4Cys to 4His)"
    html_content = _render_html(ordered, title=title, columns=args.columns)

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "combined_exafs_grid.html"
    output_path.write_text(html_content)

    print(f"Wrote {output_path} with {len(ordered)} image(s).")


if __name__ == "__main__":
    main()
