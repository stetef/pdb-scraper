#!/usr/bin/env python3
"""Generate validation histograms and a py3Dmol overlay for an XYZ directory.

Reads all .xyz files under the provided directory (default: data/output/xyz_files),
generates histograms, embeds them into a combined HTML, deletes the PNGs, and
writes a directory-wide py3Dmol overlay HTML.

XYZ format expected (as written by this repo):
    - Line 1: atom count (ignored)
    - Line 2: file comment (ignored)
    - Remaining lines: "ELM  X  Y  Z  # RES=... CHAIN=... RESSEQ=... ATOM=... REC=..."
"""

from __future__ import annotations

import argparse
from pathlib import Path

from generate_combined_figures import generate_html
from xyz_plot_helpers import resolve_reference_xyz, run_directory_mode


def _normalize_system_label(label: str) -> str:
    return "-".join(label.strip().split())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate histograms and py3Dmol HTMLs for an XYZ directory."
    )
    parser.add_argument(
        "--dir",
        default="data/output/xyz_files",
        help="Directory containing .xyz files (default: data/output/xyz_files). Used as the base for --file patterns.",
    )
    parser.add_argument(
        "--out-dir",
        default="Figures",
        help="Output directory for HTML files (default: Figures).",
    )
    parser.add_argument(
        "--system-label",
        help="System label appended to output filenames.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        default=True,
        help="Open the 3D model HTML in your default browser (default: true)",
    )
    parser.add_argument(
        "--no-open",
        action="store_false",
        dest="open",
        help="Do not open the directory index HTML",
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Overlay Reference_*.xyz metrics from ../../ relative to --dir as dashed lines in histograms (legacy single reference)",
    )
    parser.add_argument(
        "--reference-set",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        help=(
            "Add a labeled reference XYZ (repeatable). Example: "
            "--reference-set WT data/Reference_WT.xyz --reference-set Mut data/Reference_Mut.xyz"
        ),
    )
    args = parser.parse_args()

    xyz_dir = Path(args.dir)
    if not xyz_dir.exists():
        raise SystemExit(f"Directory not found: {xyz_dir}")

    output_dir = Path(args.out_dir)
    system_label = args.system_label or xyz_dir.name
    system_label = _normalize_system_label(system_label)

    reference_entries = []
    if args.reference_set:
        reference_entries.extend(
            [(label, Path(path)) for label, path in args.reference_set]
        )
    if args.reference and reference_entries:
        reference_entries.append(("Reference", resolve_reference_xyz(xyz_dir)))

    if reference_entries:
        run_directory_mode(
            xyz_dir,
            open_html=args.open,
            output_dir=output_dir,
            system_label=system_label,
            show_plots=args.open,
            reference_entries=reference_entries,
        )
    elif args.reference:
        reference_path = resolve_reference_xyz(xyz_dir)
        run_directory_mode(
            xyz_dir,
            open_html=args.open,
            output_dir=output_dir,
            system_label=system_label,
            show_plots=args.open,
            reference_path=reference_path,
        )
    else:
        run_directory_mode(
            xyz_dir,
            open_html=args.open,
            output_dir=output_dir,
            system_label=system_label,
            show_plots=args.open,
        )

    generate_html(output_dir, system_label=system_label, delete_pngs=True)


if __name__ == "__main__":
    """
    Generate histograms and a py3Dmol overlay for a directory:
    ```bash
    uv run python ./scripts/xyz-val-plots.py --dir data/output/xyz_files --out-dir Figures --system-label Zn-4Cys
    ```

    Disable auto-open:
    ```bash
    uv run python ./scripts/xyz-val-plots.py --dir data/output/xyz_files --out-dir Figures --system-label Zn-4Cys --no-open
    ```
    """
    main()
