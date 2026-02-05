#!/usr/bin/env python3
"""Generate a self-contained HTML file with all PNG figures from a directory."""

import argparse
import base64
from pathlib import Path


def png_to_base64(png_path: Path) -> str:
    """Convert a PNG file to base64 string."""
    with open(png_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def format_caption(filename: str) -> str:
    """Convert filename to a readable caption."""
    # Remove .png extension and replace underscores with spaces
    caption = filename.replace('.png', '').replace('_', ' ')
    caption = caption.replace('zn', 'Zn').replace('cb', 'Cβ')
    caption = caption.replace('ca', 'Cα').replace('sg', 'S')
    # Capitalize words
    return caption


def _detect_grid_columns(png_files: list[Path]) -> int:
    return 3


def _detect_system_label(figures_dir: Path, png_files: list[Path]) -> str:
    name_hint = figures_dir.as_posix().lower()
    for key, label in (
        ("4cys", "Zn-4Cys"),
        ("4his", "Zn-4His"),
        ("3cys1his", "Zn-3Cys-1His"),
        ("2cys2his", "Zn-2Cys-2His"),
        ("1cys3his", "Zn-1Cys-3His"),
    ):
        if key in name_hint:
            return label

    names = [p.name.lower() for p in png_files]
    has_cys = any("cys" in n for n in names)
    has_his = any("his" in n for n in names)
    if has_cys and has_his:
        return "Zn-2Cys-2His"
    if has_his and not has_cys:
        return "Zn-4His"
    return "Zn-4Cys"


def generate_html(
    output_dir: Path,
    *,
    system_label: str | None = None,
    delete_pngs: bool = False,
) -> Path | None:
    """Generate self-contained HTML with embedded images."""
    
    # Get all PNG files
    output_dir.mkdir(parents=True, exist_ok=True)
    png_files = sorted(output_dir.glob('*.png'))
    
    if not png_files:
        print(f"No PNG files found in {output_dir}")
        return
    
    print(f"Found {len(png_files)} PNG files:")
    for png in png_files:
        print(f"  - {png.name}")
    
    grid_columns = _detect_grid_columns(png_files)
    system_label = system_label or _detect_system_label(output_dir, png_files)
    output_path = output_dir / f"combined_figures_grid_{system_label}.html"

    # Generate HTML header
    html_parts = ['''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDB Analysis Figures - ''' + system_label + '''</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
        .grid-container {
            display: grid;
            grid-template-columns: repeat(''' + str(grid_columns) + ''', 1fr);
            gap: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }
        .figure-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .figure-item img {
            width: 100%;
            height: auto;
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .figure-item img:hover {
            transform: scale(1.02);
        }
        /* Modal for full-size viewing */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
            cursor: pointer;
        }
        .modal img {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            max-width: 95%;
            max-height: 95%;
        }
        .close {
            position: absolute;
            top: 20px;
            right: 40px;
            color: white;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <h1>Validation of ''' + system_label + ''' Systems on PDB</h1>
    
    <div class="grid-container">
''']
    
    # Add each image as embedded base64
    for png_file in png_files:
        print(f"Encoding {png_file.name}...")
        base64_data = png_to_base64(png_file)
        caption = format_caption(png_file.name)
        
        html_parts.append(f'''        <div class="figure-item">
            <img src="data:image/png;base64,{base64_data}" alt="{caption}" onclick="openModal(this.src)">
        </div>
        
''')
    
    # Add HTML footer
    html_parts.append('''    </div>

    <!-- Modal for full-size image viewing -->
    <div id="imageModal" class="modal" onclick="closeModal()">
        <span class="close">&times;</span>
        <img id="modalImg">
    </div>

    <script>
        function openModal(src) {
            document.getElementById('imageModal').style.display = 'block';
            document.getElementById('modalImg').src = src;
        }

        function closeModal() {
            document.getElementById('imageModal').style.display = 'none';
        }

        // Close modal with Escape key
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                closeModal();
            }
        });
    </script>
</body>
</html>
''')
    
    # Write the file
    html_content = ''.join(html_parts)
    output_path.write_text(html_content)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Generated {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Images embedded: {len(png_files)}")
    if delete_pngs:
        removed = 0
        for png in png_files:
            try:
                png.unlink()
                removed += 1
            except OSError as exc:
                print(f"Warning: failed to delete {png}: {exc}")
        print(f"  PNG files deleted: {removed}")
    print(f"\nYou can now share this single HTML file with collaborators!")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a self-contained HTML for PNGs in a directory.")
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        default="Figures",
        help="Directory containing PNG files (default: Figures).",
    )
    parser.add_argument(
        "--system-label",
        dest="system_label",
        help="Override system label used in title and output filename.",
    )
    parser.add_argument(
        "--delete-pngs",
        action="store_true",
        help="Delete PNG files after embedding them in the HTML.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    generate_html(
        Path(args.out_dir),
        system_label=args.system_label,
        delete_pngs=args.delete_pngs,
    )
