#!/usr/bin/env python3
"""Plot CYS CA (black) and CYS SG (dark yellow) atoms from an XYZ file.

Reads all .xyz files under the provided directory (default: data/output/xyz_files),
selects the first file (lexicographic sort), filters atoms whose metadata comment
contains RES=CYS and ATOM=CA or ATOM=SG, and plots their coordinates.

Alternatively, you can provide a specific XYZ file to plot.

Also creates a second plot of *all* atoms using py3Dmol, showing inferred bonds
and element-based atom coloring, with hover text showing the atom's comment
string.

XYZ format expected (as written by this repo):
  - Line 1: atom count (ignored)
  - Line 2: file comment (ignored)
  - Remaining lines: "ELM  X  Y  Z  # RES=... CHAIN=... RESSEQ=... ATOM=... REC=..."
"""

from __future__ import annotations

import argparse
from pathlib import Path

from xyz_plot_helpers import run_check_mode, run_plot_mode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot CYS CA (black) and CYS SG (dark yellow) from an XYZ file."
    )
    parser.add_argument(
        "--dir",
        default="data/output/xyz_files",
        help="Directory containing .xyz files (default: data/output/xyz_files). Used as the base for --file patterns.",
    )
    parser.add_argument(
        "--file",
        help="Path/pattern for .xyz file(s) to plot. Supports wildcards like '1a71*'",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only scan --dir for files where CYS(CA+CB+SG) residue count != 4 and write .html for those outliers",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated py3Dmol HTML in your default browser",
    )
    args = parser.parse_args()

    xyz_dir = Path(args.dir)
    if not xyz_dir.exists():
        raise SystemExit(f"Directory not found: {xyz_dir}")

    # Check mode: ONLY write outlier HTMLs (no other plots).
    if args.check:
        if args.file is not None:
            raise SystemExit("--check cannot be used with --file; it scans the entire --dir")
        run_check_mode(xyz_dir, open_html=args.open)
        return
    run_plot_mode(xyz_dir, file_arg=args.file, open_html=args.open)


if __name__ == "__main__":
    """
    To combine a few files and see the "aligned" residue figure, do:
    ```bash
    uv run python ./scripts/xyz-val-plots.py --dir data/output/xyz_files --file "1*" --open
    ```

    To auto open html file for a specific molecule, do:
    ```bash
    uv run python ./scripts/xyz-val-plots.py --file data/output/xyz_files/1bto_ZN_homo_d3.00_cluster8.xyz --open

    1a71_ZN_homo_d3.00_cluster4.xyz
    1bto_ZN_homo_d3.00_1axe_ZN_homo_d3.00_cluster2.xyzcluster8.xyz
    
    ```

    To check a directory for S!=4, do
    ```bash
    uv run python ./scripts/xyz-val-plots.py --dir data/output/xyz_files --check
    ```
    Add the `--open` flag to above if you want to open a max of 5 html files for those outliers.
    """
    main()
