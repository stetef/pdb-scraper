#!/usr/bin/env python3
"""
Generate a self-contained HTML file with all PNG figures from the Figures directory.
Images are embedded as base64, creating a single shareable file.

Usage:
    python scripts/generate_combined_figures.py
"""

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


def generate_html(figures_dir: Path, output_path: Path):
    """Generate self-contained HTML with embedded images."""
    
    # Get all PNG files
    png_files = sorted(figures_dir.glob('*.png'))
    
    if not png_files:
        print(f"No PNG files found in {figures_dir}")
        return
    
    print(f"Found {len(png_files)} PNG files:")
    for png in png_files:
        print(f"  - {png.name}")
    
    # Generate HTML header
    html_parts = ['''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDB Analysis Figures - 3x3 Grid</title>
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
            grid-template-columns: repeat(3, 1fr);
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
        .figure-caption {
            margin-top: 10px;
            font-size: 14px;
            color: #666;
            text-align: center;
            font-weight: 500;
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
    <h1>Validation of Zn-4Cys Systems on PDB</h1>
    
    <div class="grid-container">
''']
    
    # Add each image as embedded base64
    for png_file in png_files:
        print(f"Encoding {png_file.name}...")
        base64_data = png_to_base64(png_file)
        caption = format_caption(png_file.name)
        
        html_parts.append(f'''        <div class="figure-item">
            <img src="data:image/png;base64,{base64_data}" alt="{caption}" onclick="openModal(this.src)">
            <div class="figure-caption">{caption}</div>
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
    print(f"\nYou can now share this single HTML file with collaborators!")


if __name__ == '__main__':
    # Set up paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    figures_dir = project_root / 'Figures'
    output_path = figures_dir / 'combined_figures_grid.html'
    
    # Generate the HTML
    generate_html(figures_dir, output_path)
