#!/usr/bin/env python3
"""Build an interactive py3Dmol HTML for one XYZ structure plus its .pc charges.

Features:
- Renders a compact XYZ + .pc viewer (current behavior).
- Renders an extended XYZ-in-protein viewer (no .pc points).
- Optionally overlays a translated validated PDB for visual cross-check.

Usage examples:
  uv run python scripts/view_xyz_pc_py3dmol.py \
        --xyz data/extended-backbone/output/xyz_files/1a1g_cluster2_Zn.xyz \
        --pc data/extended-backbone/output/xyz_files/1a1g_cluster2_Zn.pc \
        --output data/extended-backbone/output/xyz_files/1a1g_cluster2_Zn.viewer.html

    # Auto-find .pc, -extended.xyz, and validated PDB
  uv run python scripts/view_xyz_pc_py3dmol.py \
    --xyz data/extended-backbone/output/xyz_files/1a1g_cluster2_Zn.xyz \
    --output data/extended-backbone/output/xyz_files/1a1g_cluster2_Zn.viewer.html
"""

from __future__ import annotations

import argparse
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
    element: str
    x: float
    y: float
    z: float


@dataclass
class XyzOriginRef:
    atom_name: str
    resname: str
    chain: str
    resseq: str
    icode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate two py3Dmol HTML viewers: "
            "(1) compact xyz + pc, and (2) extended xyz inside protein (no pc)."
        )
    )
    parser.add_argument("--xyz", required=True, help="Input XYZ file path")
    parser.add_argument(
        "--extended-xyz",
        help=(
            "Extended XYZ file path (default: same stem as --xyz with '-extended' suffix)"
        ),
    )
    parser.add_argument(
        "--pc",
        help="Input point-charge .pc file path (default: same stem as --xyz with .pc)",
    )
    parser.add_argument(
        "--pdb",
        default="auto",
        help=(
            "Validated PDB path, 'auto', or 'none' (default: auto). "
            "Auto tries ../../results/validated_structures/<PDB>.pdb from the xyz directory."
        ),
    )
    parser.add_argument("--output", required=True, help="Output HTML path")
    parser.add_argument(
        "--output-extended",
        help=(
            "Output HTML for extended xyz view "
            "(default: --output with '-extended' before .html)"
        ),
    )
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--height", type=int, default=760)
    parser.add_argument(
        "--only-active-site",
        action="store_true",
        help=(
            "Render only COORD=TRUE atoms in full Jmol color. COORD=FALSE atoms "
            "and point charges become a grayed-out background."
        ),
    )
    parser.add_argument(
        "--show-protein",
        action="store_true",
        help=(
            "Color the protein cartoon (overlaid PDB) instead of leaving it gray. "
            "Nucleic-acid chains and waters stay gray."
        ),
    )
    return parser.parse_args()


def read_xyz_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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


def split_xyz_by_coord_flag(xyz_text: str) -> tuple[str, str]:
    """Split an xyz block into (active_site_xyz, neighbor_xyz) by COORD=TRUE/FALSE.

    Atoms without an explicit COORD=FALSE tag are treated as active site.
    Each returned xyz is a self-contained xyz string with its own atom count.
    """
    lines = xyz_text.splitlines()
    if len(lines) < 2:
        return ("", "")
    try:
        declared = int(lines[0].strip())
    except ValueError:
        return ("", "")
    comment = lines[1]

    active: list[str] = []
    neighbor: list[str] = []
    for line in lines[2 : 2 + declared]:
        if "COORD=FALSE" in line.upper():
            neighbor.append(line)
        else:
            active.append(line)

    def _assemble(rows: list[str], tag: str) -> str:
        if not rows:
            return ""
        return f"{len(rows)}\n{comment} SUBSET={tag}\n" + "\n".join(rows) + "\n"

    return (_assemble(active, "active"), _assemble(neighbor, "neighbor"))


def parse_xyz_origin_ref(xyz_text: str) -> XyzOriginRef | None:
    """Find the origin atom metadata from the xyz line at/near (0,0,0)."""
    for line in xyz_text.splitlines()[2:]:
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

        if abs(x) > 1e-6 or abs(y) > 1e-6 or abs(z) > 1e-6:
            continue

        if "#" not in raw:
            continue
        _, comment = raw.split("#", 1)
        fields: dict[str, str] = {}
        for token in comment.split():
            if "=" in token:
                k, v = token.split("=", 1)
                fields[k.strip().upper()] = v.strip()

        atom_name = fields.get("ATOM", "")
        resname = fields.get("RES", "")
        chain = fields.get("CHAIN", "")
        resseq_icode = fields.get("RESSEQ", "")
        if not atom_name or not resname or not chain or not resseq_icode:
            continue

        match = re.match(r"^(-?\d+)([A-Za-z]?)$", resseq_icode)
        if not match:
            continue

        resseq = match.group(1)
        icode = match.group(2)
        return XyzOriginRef(
            atom_name=atom_name.upper(),
            resname=resname.upper(),
            chain=chain,
            resseq=resseq,
            icode=icode,
        )

    return None


def parse_pdb_id_from_xyz_comment(xyz_text: str) -> str | None:
    lines = xyz_text.splitlines()
    if len(lines) < 2:
        return None
    m = re.search(r"\bPDB=([A-Za-z0-9]+)\b", lines[1])
    if not m:
        return None
    return m.group(1)


def infer_validated_pdb_path(xyz_path: Path, pdb_id: str) -> Path | None:
    # Common layout: <root>/data/<run>/output/xyz_files/*.xyz and
    # validated PDBs in <root>/data/<run>/results/validated_structures/
    candidates = [
        xyz_path.parent.parent.parent / "results" / "validated_structures" / f"{pdb_id}.pdb",
        xyz_path.parent.parent / "results" / "validated_structures" / f"{pdb_id}.pdb",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_and_translate_pdb(pdb_path: Path, origin_ref: XyzOriginRef | None) -> str:
    """Load PDB text and translate so origin_ref atom is at (0,0,0) if possible."""
    lines = pdb_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if origin_ref is None:
        return "\n".join(lines) + "\n"

    ox = oy = oz = None
    for line in lines:
        rec = line[0:6].strip().upper() if len(line) >= 6 else ""
        if rec not in {"ATOM", "HETATM"}:
            continue

        atom_name = line[12:16].strip().upper() if len(line) >= 16 else ""
        altloc = line[16:17].strip().upper() if len(line) >= 17 else ""
        resname = line[17:20].strip().upper() if len(line) >= 20 else ""
        chain = line[21:22].strip() if len(line) >= 22 else ""
        resseq = line[22:26].strip() if len(line) >= 26 else ""
        icode = line[26:27].strip() if len(line) >= 27 else ""

        if altloc not in {"", "A"}:
            continue
        if atom_name != origin_ref.atom_name:
            continue
        if resname != origin_ref.resname:
            continue
        if chain != origin_ref.chain:
            continue
        if resseq != origin_ref.resseq:
            continue
        if icode != origin_ref.icode:
            continue

        try:
            ox = float(line[30:38])
            oy = float(line[38:46])
            oz = float(line[46:54])
        except ValueError:
            continue
        break

    if ox is None:
        return "\n".join(lines) + "\n"

    translated: list[str] = []
    for line in lines:
        rec = line[0:6].strip().upper() if len(line) >= 6 else ""
        if rec in {"ATOM", "HETATM"} and len(line) >= 54:
            try:
                x = float(line[30:38]) - ox
                y = float(line[38:46]) - oy
                z = float(line[46:54]) - oz
                line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
            except ValueError:
                pass
        translated.append(line)

    return "\n".join(translated) + "\n"


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


def extract_translated_pdb_points(pdb_text: str) -> list[PdbAtomPoint]:
    """Extract translated PDB atom coordinates and element symbols."""
    out: list[PdbAtomPoint] = []
    for line in pdb_text.splitlines():
        rec = line[0:6].strip().upper() if len(line) >= 6 else ""
        if rec not in {"ATOM", "HETATM"}:
            continue
        if len(line) < 54:
            continue

        atom_name = line[12:16].strip().upper()
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

        out.append(PdbAtomPoint(atom_name=atom_name, element=element, x=x, y=y, z=z))
    return out


def infer_element_from_charge(q: float) -> str | None:
    """Fallback element inference from common backbone AMBER charges."""
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


def infer_pc_points(points: list[PcPoint], pdb_text: str | None) -> list[tuple[str, str]]:
    """
    Infer (element, atom_name) for each .pc point.

    Priority:
    1) nearest translated atom in validated PDB
    2) fallback by AMBER charge proximity
    """
    inferred: list[tuple[str, str]] = []
    pdb_points = extract_translated_pdb_points(pdb_text) if pdb_text else []

    for p in points:
        element = None
        atom_name = ""
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

        if element is None:
            element = infer_element_from_charge(p.charge)
            atom_name = ""

        inferred.append((element if element is not None else "UNK", atom_name))

    return inferred


def build_pc_xyz(points: list[PcPoint], elements: list[str]) -> str:
    body = []
    for p, element in zip(points, elements):
        symbol = (element or "").strip().title() or "He"
        body.append(f"{symbol} {p.x:.6f} {p.y:.6f} {p.z:.6f}")
    return f"{len(points)}\nPOINT_CHARGES\n" + "\n".join(body) + "\n"


NUCLEIC_RESN = ["DA", "DC", "DG", "DT", "DU", "A", "C", "G", "T", "U"]


def build_html(
    xyz_text: str,
    points: list[PcPoint] | None,
    output_path: Path,
    width: int,
    height: int,
    pdb_text: str | None,
    only_active_site: bool = False,
    show_protein: bool = False,
) -> None:
    view = py3Dmol.view(width=width, height=height)
    inferred_pc = infer_pc_points(points, pdb_text) if points else []
    pc_elements = [x[0] for x in inferred_pc]
    pc_atom_names = [x[1] for x in inferred_pc]

    # model 0: xyz (active site if only_active_site, else everything)
    active_xyz, neighbor_xyz = (
        split_xyz_by_coord_flag(xyz_text) if only_active_site else ("", "")
    )
    view.addModel(active_xyz or xyz_text, "xyz")
    view.setStyle(
        {"model": 0},
        {
            "stick": {"radius": 0.16, "colorscheme": "Jmol"},
            "sphere": {"scale": 0.25, "colorscheme": "Jmol"},
        },
    )

    next_model = 1
    if neighbor_xyz:
        view.addModel(neighbor_xyz, "xyz")
        view.setStyle(
            {"model": next_model},
            {
                "stick": {"radius": 0.10, "color": "lightgray", "opacity": 0.30},
                "sphere": {"scale": 0.12, "color": "lightgray", "opacity": 0.30},
            },
        )
        next_model += 1
    if pdb_text:
        view.addModel(pdb_text, "pdb")
        if show_protein:
            view.setStyle(
                {"model": next_model},
                {
                    "cartoon": {"color": "lightblue", "opacity": 0.75},
                    "line": {"color": "gray", "opacity": 0.22},
                },
            )
            view.setStyle(
                {"model": next_model, "resn": NUCLEIC_RESN},
                {
                    "cartoon": {"color": "lightgray", "opacity": 0.30},
                    "line": {"color": "gray", "opacity": 0.22},
                },
            )
        else:
            view.setStyle(
                {"model": next_model},
                {
                    "cartoon": {"color": "lightgray", "opacity": 0.25},
                    "line": {"color": "gray", "opacity": 0.22},
                },
            )
        next_model += 1

    if points:
        pc_model_id = next_model
        pc_xyz = build_pc_xyz(points, pc_elements)
        view.addModel(pc_xyz, "xyz")

        if only_active_site:
            pc_style = {
                "sphere": {
                    "radius": 0.22,
                    "color": "lightgray",
                    "opacity": 0.18,
                }
            }
        else:
            pc_style = {
                "sphere": {
                    "radius": 0.36,
                    "colorscheme": "Jmol",
                    "opacity": 0.40,
                }
            }
        # Base style for all charge points.
        view.setStyle({"model": pc_model_id}, pc_style)

        point_labels = [
            (
                f"elem={pc_elements[i]}"
                + (f" atom={pc_atom_names[i]}" if pc_atom_names[i] else "")
                + f"  q={p.charge:+.4f} e  x={p.x:.3f}  y={p.y:.3f}  z={p.z:.3f}"
            )
            for i, p in enumerate(points)
        ]

        import json

        labels_json = json.dumps(point_labels)
        hover_on = (
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
        view.setHoverable({"model": pc_model_id}, True, hover_on, hover_off)
    else:
        hover_off = (
            "function(atom, viewer) {"
            "var el = document.getElementById('hoverinfo'); if (el) { el.textContent = ''; }"
            "}"
        )

    # Keep xyz hover useful too.
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

    html = view._make_html()
    title = output_path.stem
    if only_active_site and show_protein:
        legend_text = (
            "Active site (COORD=TRUE): Jmol sticks/spheres | "
            "Protein cartoon: lightblue | Nucleic acids + neighbors + .pc: gray"
        )
    elif only_active_site:
        legend_text = (
            "Active site (COORD=TRUE): Jmol sticks/spheres | "
            "Neighbor residues + .pc points: grayed-out background"
        )
    elif show_protein:
        legend_text = (
            "XYZ structure: sticks/spheres | Protein cartoon: lightblue | "
            "Nucleic acids: gray | .pc points: element-colored (Jmol), alpha=0.40"
        )
    else:
        legend_text = (
            "XYZ structure: sticks/spheres | .pc points: element-colored (Jmol), alpha=0.40"
        )
    header_html = (
        "<div id=\"hoverwrap\">"
        "<div id=\"title\">" + title + "</div>"
        "<div id=\"legend\">" + legend_text + "</div>"
        "<div id=\"hoverinfo\">Hover over .pc points to inspect q, x, y, z</div>"
        "</div>"
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


def resolve_pc_path(xyz_path: Path, pc_arg: str | None) -> Path:
    if pc_arg:
        return Path(pc_arg)
    return xyz_path.with_suffix(".pc")


def resolve_extended_xyz_path(xyz_path: Path, extended_xyz_arg: str | None) -> Path:
    if extended_xyz_arg:
        return Path(extended_xyz_arg)
    return xyz_path.with_name(f"{xyz_path.stem}-extended{xyz_path.suffix}")


def resolve_extended_output_path(output_path: Path, output_extended_arg: str | None) -> Path:
    if output_extended_arg:
        return Path(output_extended_arg)
    return output_path.with_name(f"{output_path.stem}-extended{output_path.suffix}")


def resolve_pdb_text(xyz_path: Path, xyz_text: str, pdb_arg: str) -> str | None:
    if pdb_arg.lower() == "none":
        return None

    origin_ref = parse_xyz_origin_ref(xyz_text)

    if pdb_arg.lower() == "auto":
        pdb_id = parse_pdb_id_from_xyz_comment(xyz_text)
        if not pdb_id:
            return None
        candidate = infer_validated_pdb_path(xyz_path, pdb_id)
        if candidate is None:
            return None
        return load_and_translate_pdb(candidate, origin_ref)

    pdb_path = Path(pdb_arg)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    return load_and_translate_pdb(pdb_path, origin_ref)


def main() -> None:
    args = parse_args()

    compact_xyz_path = Path(args.xyz)
    if not compact_xyz_path.exists():
        raise SystemExit(f"XYZ file not found: {compact_xyz_path}")

    extended_xyz_path = resolve_extended_xyz_path(compact_xyz_path, args.extended_xyz)
    has_extended = extended_xyz_path.exists()

    pc_path = resolve_pc_path(compact_xyz_path, args.pc)
    if not pc_path.exists():
        raise SystemExit(f"PC file not found: {pc_path}")

    compact_xyz_text = read_xyz_text(compact_xyz_path)
    points = parse_pc(pc_path)
    compact_pdb_text = resolve_pdb_text(compact_xyz_path, compact_xyz_text, args.pdb)

    out_path = Path(args.output)

    build_html(
        xyz_text=compact_xyz_text,
        points=points,
        output_path=out_path,
        width=args.width,
        height=args.height,
        pdb_text=compact_pdb_text,
        only_active_site=args.only_active_site,
        show_protein=args.show_protein,
    )
    print(f"Wrote compact viewer HTML:  {out_path}")

    if has_extended:
        extended_xyz_text = read_xyz_text(extended_xyz_path)
        extended_pdb_text = resolve_pdb_text(extended_xyz_path, extended_xyz_text, args.pdb)
        out_extended_path = resolve_extended_output_path(out_path, args.output_extended)
        build_html(
            xyz_text=extended_xyz_text,
            points=None,
            output_path=out_extended_path,
            width=args.width,
            height=args.height,
            pdb_text=extended_pdb_text,
            only_active_site=args.only_active_site,
        show_protein=args.show_protein,
        )
        print(f"Wrote extended viewer HTML: {out_extended_path}")
    else:
        print(f"Skipped extended viewer (no file at {extended_xyz_path})")


if __name__ == "__main__":
    main()
