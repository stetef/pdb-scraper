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
from bisect import bisect_right
import json
import math
import subprocess
import sys
import webbrowser
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class XyzAtom:
    element: str
    x: float
    y: float
    z: float
    meta: Dict[str, str]
    raw_comment: str


def _parse_meta_comment(comment: str) -> Dict[str, str]:
    # Comment is expected like: "RES=CYS CHAIN=B RESSEQ=111 ATOM=CA REC=ATOM"
    meta: Dict[str, str] = {}
    for token in comment.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key and value:
            meta[key] = value
    return meta


def parse_xyz_atoms(path: Path) -> List[XyzAtom]:
    atoms: List[XyzAtom] = []
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 2:
        return atoms

    for line in lines[2:]:
        stripped = line.strip()
        if not stripped:
            continue

        # Split data vs comment
        if "#" in stripped:
            left, right = stripped.split("#", 1)
            comment = right.strip()
        else:
            left, comment = stripped, ""

        parts = left.split()
        if len(parts) < 4:
            continue

        element = parts[0]
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue

        meta = _parse_meta_comment(comment) if comment else {}
        atoms.append(XyzAtom(element=element, x=x, y=y, z=z, meta=meta, raw_comment=comment))

    return atoms


def count_cys_residues_with_ca_cb_sg(atoms: List[XyzAtom]) -> int:
    """Count CYS residues (by CHAIN+RESSEQ) that have CA, CB, and SG atoms."""
    by_residue: Dict[Tuple[str, str], set[str]] = {}
    for a in atoms:
        if a.meta.get("RES") != "CYS":
            continue
        chain = a.meta.get("CHAIN", "")
        resseq = a.meta.get("RESSEQ")
        atom_name = a.meta.get("ATOM")
        if not resseq or not atom_name:
            continue
        by_residue.setdefault((chain, resseq), set()).add(atom_name)

    return sum(1 for atom_names in by_residue.values() if {"CA", "CB", "SG"}.issubset(atom_names))


def check_dir_for_non_four_cys_residues(xyz_dir: Path) -> List[Path]:
    """Return .xyz files whose count of CYS residues with CA/CB/SG != 4."""
    xyz_files = find_xyz_files(xyz_dir)
    bad: List[Path] = []
    for p in xyz_files:
        atoms = parse_xyz_atoms(p)
        n = count_cys_residues_with_ca_cb_sg(atoms)
        if n != 4:
            bad.append(p)
    return bad


def sg_bond_lengths_to_center(atoms: List[XyzAtom], *, max_sgs: int = 4) -> List[float]:
    """Return bond lengths from center (origin) to up to `max_sgs` nearest CYS SG atoms.

    In this repo's XYZ files, coordinates are translated such that the target/center atom
    is at (0,0,0). Some outlier files contain a 5th CYS SG that is farther away and not
    coordinated; we drop it by taking the `max_sgs` smallest distances.
    """
    sg_atoms = [
        a
        for a in atoms
        if a.meta.get("RES") == "CYS" and a.meta.get("ATOM") == "SG"
    ]
    dists = [math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z) for a in sg_atoms]
    dists.sort()
    return dists[:max_sgs]


def plot_sg_bond_length_histogram(distances: List[float], sources: List[str], title: str) -> None:
    """Plot a histogram of SG-to-center bond lengths with hover/click bin inspection.

    - Hover a bar to see all filenames that contributed at least one SG distance to that bin.
    - Click a bar to copy the filenames for that bin to your clipboard.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(
            "matplotlib is required for the histogram plot. Install it with: pip install matplotlib\n"
            f"Import error: {e}"
        )
        return

    if not distances:
        print("No SG bond distances collected; skipping histogram plot.")
        return

    if len(sources) != len(distances):
        raise SystemExit(f"Internal error: sources({len(sources)}) != distances({len(distances)})")

    def _copy_to_clipboard(text: str) -> bool:
        # macOS first.
        if sys.platform == "darwin":
            try:
                subprocess.run(["pbcopy"], input=text, text=True, check=True)
                return True
            except Exception:
                return False
        # Fallback: optional pure-Python helper if installed.
        try:
            import pyperclip  # type: ignore

            pyperclip.copy(text)
            return True
        except Exception:
            return False

    fig, ax = plt.subplots()
    counts, edges, patches = ax.hist(distances, bins="auto", edgecolor="black", alpha=0.8)

    # Build bin -> set(files) mapping.
    nbins = max(0, len(edges) - 1)
    files_by_bin: List[set[str]] = [set() for _ in range(nbins)]
    for d, src in zip(distances, sources):
        if nbins == 0:
            continue
        idx = bisect_right(edges, d) - 1
        if idx < 0:
            continue
        if idx >= nbins:
            idx = nbins - 1  # include right edge in the last bin
        files_by_bin[idx].add(src)

    # Hover/click UI: show filenames in a fixed info box; click copies full list.
    info_box = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.95),
    )
    current_bin: Dict[str, int | None] = {"idx": None}

    def _format_bin_info(bin_idx: int) -> str:
        lo = float(edges[bin_idx])
        hi = float(edges[bin_idx + 1])
        file_list = sorted(files_by_bin[bin_idx])
        header = f"Bin {bin_idx + 1}/{nbins}: [{lo:.3f}, {hi:.3f}) Å\n"
        header += f"Count={int(counts[bin_idx])}  Files={len(file_list)}\n"
        header += "(click bar to copy filenames)\n\n"
        return header + "\n".join(file_list)

    def _clear_info() -> None:
        if info_box.get_text():
            info_box.set_text("")
            fig.canvas.draw_idle()
        current_bin["idx"] = None

    def _on_move(event):
        if event.inaxes != ax or nbins == 0:
            _clear_info()
            return

        hit_idx: int | None = None
        for i, patch in enumerate(patches):
            contains, _ = patch.contains(event)
            if contains:
                hit_idx = i
                break

        if hit_idx is None:
            _clear_info()
            return

        if current_bin["idx"] != hit_idx:
            current_bin["idx"] = hit_idx
            info_box.set_text(_format_bin_info(hit_idx))
            fig.canvas.draw_idle()

    def _on_click(event):
        if event.inaxes != ax or nbins == 0:
            return
        hit_idx: int | None = None
        for i, patch in enumerate(patches):
            contains, _ = patch.contains(event)
            if contains:
                hit_idx = i
                break
        if hit_idx is None:
            return

        file_list = sorted(files_by_bin[hit_idx])
        text = "\n".join(file_list)
        if _copy_to_clipboard(text):
            print(f"Copied {len(file_list)} filenames to clipboard for bin {hit_idx + 1}/{nbins}.")
        else:
            print("Clipboard copy failed; printing filenames instead:\n" + text)

    fig.canvas.mpl_connect("motion_notify_event", _on_move)
    fig.canvas.mpl_connect("button_press_event", _on_click)

    plt.title(title)
    plt.xlabel("Distance from center (origin) to CYS SG (Å)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def find_xyz_files(directory: Path) -> List[Path]:
    return sorted([p for p in directory.glob("*.xyz") if p.is_file()])


def _has_glob(pattern: str) -> bool:
    return any(ch in pattern for ch in ("*", "?", "["))


def resolve_xyz_files(file_arg: str | None, directory: Path) -> List[Path]:
    """Resolve --file into one or more .xyz files.

    - If file_arg is None: returns [first .xyz in directory]
    - If file_arg contains glob wildcards (e.g. "1a71*"): matches within directory unless a parent is provided
    - If file_arg is a concrete existing file path: returns [that file]
    - If file_arg doesn't exist and has no wildcards: treated as a glob pattern within directory
    """
    if file_arg is None:
        xyz_files = find_xyz_files(directory)
        if not xyz_files:
            raise SystemExit(f"No .xyz files found in: {directory}")
        return [xyz_files[0]]

    raw = file_arg
    p = Path(raw)

    # Explicit wildcard pattern.
    if _has_glob(raw):
        search_dir = p.parent if str(p.parent) not in (".", "") else directory
        pattern = p.name
        matches = sorted([m for m in search_dir.glob(pattern) if m.is_file() and m.suffix.lower() == ".xyz"])
        if not matches:
            raise SystemExit(f"No .xyz files matched pattern '{raw}' in {search_dir}")
        return matches

    # Concrete file path.
    if p.exists() and p.is_file():
        if p.suffix.lower() != ".xyz":
            raise SystemExit(f"Not an .xyz file: {p}")
        return [p]

    # Shorthand pattern without wildcards: treat as glob in directory.
    search_dir = directory
    pattern = raw
    matches = sorted([m for m in search_dir.glob(pattern) if m.is_file() and m.suffix.lower() == ".xyz"])
    if matches:
        return matches

    raise SystemExit(f"File not found and no matches for '{raw}' in {search_dir}")


def plot_resseq_paths(
    atoms_by_source: List[Tuple[str, List[XyzAtom]]],
    title: str,
    out_html: Path,
) -> None:
    """
    Write an interactive HTML view of origin→SG→CB→CA paths using py3Dmol.
    """
    try:
        import py3Dmol
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "py3Dmol is required for the HTML trajectory viewer. Install it with: pip install py3dmol\n"
            f"Import error: {e}"
        )

    def _dist2(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return dx * dx + dy * dy + dz * dz

    def _dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def _cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    def _norm(a: Tuple[float, float, float]) -> float:
        return (_dot(a, a)) ** 0.5

    def _unit(a: Tuple[float, float, float]) -> Tuple[float, float, float] | None:
        n = _norm(a)
        if n == 0.0:
            return None
        return (a[0] / n, a[1] / n, a[2] / n)

    def _sub(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def _matmul(a: Tuple[Tuple[float, float, float], ...], b: Tuple[Tuple[float, float, float], ...]) -> Tuple[Tuple[float, float, float], ...]:
        # 3x3 * 3x3
        return (
            (
                a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
                a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
                a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
            ),
            (
                a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
                a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
                a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
            ),
            (
                a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
                a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
                a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
            ),
        )

    def _transpose(m: Tuple[Tuple[float, float, float], ...]) -> Tuple[Tuple[float, float, float], ...]:
        return (
            (m[0][0], m[1][0], m[2][0]),
            (m[0][1], m[1][1], m[2][1]),
            (m[0][2], m[1][2], m[2][2]),
        )

    def _apply_rot(m: Tuple[Tuple[float, float, float], ...], v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return (
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        )

    def _basis_from_two_vectors(
        v1: Tuple[float, float, float],
        v2: Tuple[float, float, float],
    ) -> Tuple[Tuple[float, float, float], ...] | None:
        # Build an orthonormal right-handed basis where e1 aligns with v1.
        e1 = _unit(v1)
        if e1 is None:
            return None
        # Make e2 by removing e1 component from v2.
        proj = _dot(v2, e1)
        v2_ortho = (v2[0] - proj * e1[0], v2[1] - proj * e1[1], v2[2] - proj * e1[2])
        e2 = _unit(v2_ortho)
        if e2 is None:
            return None
        e3 = _cross(e1, e2)
        e3u = _unit(e3)
        if e3u is None:
            return None
        # Columns are basis vectors.
        return (
            (e1[0], e2[0], e3u[0]),
            (e1[1], e2[1], e3u[1]),
            (e1[2], e2[2], e3u[2]),
        )

    @dataclass(frozen=True)
    class _Residue:
        key: Tuple[str, str, str]
        sg: Tuple[float, float, float]
        cb: Tuple[float, float, float]
        ca: Tuple[float, float, float]

    # Group atom coordinates by residue per source.
    residues_by_source: Dict[str, Dict[Tuple[str, str], Dict[str, Tuple[float, float, float]]]] = {}
    for source, atoms in atoms_by_source:
        for a in atoms:
            atom_name = a.meta.get("ATOM")
            resseq = a.meta.get("RESSEQ")
            if not atom_name or not resseq:
                continue
            chain = a.meta.get("CHAIN", "")
            src_map = residues_by_source.setdefault(source, {})
            k = (chain, resseq)
            if k not in src_map:
                src_map[k] = {}
            src_map[k][atom_name] = (a.x, a.y, a.z)

    # Build per-source residue lists (SG/CB/CA only), selecting the 4 coordinating residues
    # by choosing the smallest SG distance-to-origin per file.
    reslist_by_source: Dict[str, List[_Residue]] = {}
    for source, src_map in residues_by_source.items():
        candidates: List[Tuple[float, str, str, _Residue]] = []
        for (chain, resseq), atom_map in src_map.items():
            if "SG" in atom_map and "CB" in atom_map and "CA" in atom_map:
                sg = atom_map["SG"]
                sg_d2 = _dot(sg, sg)  # origin is (0,0,0) in these XYZ files
                candidates.append(
                    (
                        sg_d2,
                        chain,
                        resseq,
                        _Residue(key=(source, chain, resseq), sg=sg, cb=atom_map["CB"], ca=atom_map["CA"]),
                    )
                )
        candidates.sort(key=lambda t: (t[0], t[1], t[2]))
        reslist_by_source[source] = [r for _, _, _, r in candidates[:4]]

    sources_in_order = [s for s, _ in atoms_by_source]
    first_source = sources_in_order[0] if sources_in_order else ""
    ref_residues = reslist_by_source.get(first_source, [])
    if not ref_residues:
        raise SystemExit("No residues found with SG, CB, and CA atoms in the first file.")

    # Assign each residue to one of 4 colors based on SG proximity.
    # Seed the 4 color groups from the first file, then for residues in later files
    # inherit the color of the nearest previously-seen SG.
    palette = ["#3A3D42", "#457B9D", "#2A9D8F", "#E63946"]

    # Map residue key -> color index, plus an index of already-assigned SGs for nearest-neighbor lookup.
    color_by_residue: Dict[Tuple[str, str, str], int] = {}
    assigned_sg: List[Tuple[Tuple[float, float, float], int]] = []

    def _assign_color_from_assigned(sg: Tuple[float, float, float]) -> int:
        if not assigned_sg:
            return 0
        best_color = assigned_sg[0][1]
        best_d2 = _dist2(sg, assigned_sg[0][0])
        for prev_sg, prev_color in assigned_sg[1:]:
            d2 = _dist2(sg, prev_sg)
            if d2 < best_d2:
                best_d2 = d2
                best_color = prev_color
        return best_color

    # First file: fixed reference. Use the first SG-containing residue as the seed (color 0).
    # Remaining residues keep their encounter order as colors 1..3.
    for i, r in enumerate(ref_residues[:4]):
        ci = i
        color_by_residue[r.key] = ci
        assigned_sg.append((r.sg, ci))

    # For each subsequent file, apply a global rotation that best aligns to the reference.
    # We try 4 candidate rotations by choosing which of the 4 residues maps to the reference seed residue.
    ref_seed = ref_residues[0]
    ref_basis = _basis_from_two_vectors(ref_seed.sg, ref_seed.cb)
    if ref_basis is None:
        # Fallback: try CA if CB is collinear.
        ref_basis = _basis_from_two_vectors(ref_seed.sg, ref_seed.ca)
    if ref_basis is None:
        raise SystemExit("Unable to build a stable reference basis from the first residue (seed).")

    ordered_paths: List[Tuple[Tuple[str, str, str], Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]] = []
    # Add reference paths first.
    for r in ref_residues[:4]:
        ordered_paths.append((r.key, r.sg, r.cb, r.ca))

    for src in sources_in_order[1:]:
        cur_residues = reslist_by_source.get(src, [])[:4]
        if len(cur_residues) < 1:
            continue

        # Ensure we have the same count as reference for scoring/matching.
        n = min(len(ref_residues[:4]), len(cur_residues))
        refN = ref_residues[:n]
        curN = cur_residues[:n]

        best_score = None
        best_rot = None
        best_perm = None

        for seed_idx in range(n):
            seed_cur = curN[seed_idx]
            cur_basis = _basis_from_two_vectors(seed_cur.sg, seed_cur.cb)
            if cur_basis is None:
                cur_basis = _basis_from_two_vectors(seed_cur.sg, seed_cur.ca)
            if cur_basis is None:
                continue

            # Rotation mapping current basis -> reference basis.
            rot = _matmul(ref_basis, _transpose(cur_basis))

            # Evaluate the best residue correspondence under this rotation.
            # Constrain ref[0] to map to the chosen seed_idx; remaining residues can permute.
            indices = list(range(n))
            for perm in permutations(indices, n):
                if perm[0] != seed_idx:
                    continue
                score = 0.0
                for ref_i, cur_i in enumerate(perm):
                    rr = refN[ref_i]
                    cr = curN[cur_i]
                    sg_r = _apply_rot(rot, cr.sg)
                    cb_r = _apply_rot(rot, cr.cb)
                    ca_r = _apply_rot(rot, cr.ca)
                    score += _dist2(sg_r, rr.sg)
                    score += _dist2(cb_r, rr.cb)
                    score += _dist2(ca_r, rr.ca)
                if best_score is None or score < best_score:
                    best_score = score
                    best_rot = rot
                    best_perm = perm

        # If we couldn't find a stable rotation, fall back to no rotation.
        if best_rot is None or best_perm is None:
            best_rot = (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
            best_perm = tuple(range(n))

        # Apply the chosen rotation and assign colors based on nearest previously-seen SG.
        for cur_i in best_perm:
            cr = curN[cur_i]
            sg_r = _apply_rot(best_rot, cr.sg)
            cb_r = _apply_rot(best_rot, cr.cb)
            ca_r = _apply_rot(best_rot, cr.ca)
            ci = _assign_color_from_assigned(sg_r)
            color_by_residue[cr.key] = ci
            assigned_sg.append((sg_r, ci))
            ordered_paths.append((cr.key, sg_r, cb_r, ca_r))

    def _pt(v: Tuple[float, float, float]) -> Dict[str, float]:
        return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}

    view = py3Dmol.view(width=950, height=720)
    # No molecule model needed; we only draw geometric primitives.

    # Draw origin.
    view.addSphere({"center": {"x": 0.0, "y": 0.0, "z": 0.0}, "radius": 0.16, "color": "black"})

    for key, sg, cb, ca in ordered_paths:
        ci = color_by_residue.get(key, 0)
        color = palette[ci % 4]

        # Lines: origin->SG, SG->CB, CB->CA.
        view.addLine({"start": {"x": 0.0, "y": 0.0, "z": 0.0}, "end": _pt(sg), "color": color, "linewidth": 10})
        view.addLine({"start": _pt(sg), "end": _pt(cb), "color": color, "linewidth": 10})
        view.addLine({"start": _pt(cb), "end": _pt(ca), "color": color, "linewidth": 10})

        # Points at SG/CB/CA.
        view.addSphere({"center": _pt(sg), "radius": 0.14, "color": color})
        view.addSphere({"center": _pt(cb), "radius": 0.1, "color": color})
        view.addSphere({"center": _pt(ca), "radius": 0.1, "color": color})

    view.zoomTo()

    out_html.parent.mkdir(parents=True, exist_ok=True)
    try:
        view.write_html(str(out_html))
    except Exception:
        # Fallback for older py3Dmol versions
        if hasattr(view, "_make_html"):
            out_html.write_text(view._make_html(), encoding="utf-8")
        else:
            raise

    # Patch in a top-of-page title + short instructions.
    html = out_html.read_text(encoding="utf-8")
    header_html = (
        "<div id=\"titlebar\" style=\"font-family: sans-serif; padding: 10px 12px; border-bottom: 1px solid #ddd;\">"
        f"<div style=\"font-size: 16px; font-weight: 600;\">{title}</div>"
        "<div style=\"margin-top: 6px; font-size: 13px; color: #333;\">Rotate/zoom/pan in the browser. Colors track nearest SG groups.</div>"
        "</div>"
    )
    if "id=\"titlebar\"" not in html:
        if "<body" in html and ">" in html:
            body_open_end = html.find(">", html.find("<body"))
            if body_open_end != -1:
                html = html[: body_open_end + 1] + "\n" + header_html + "\n" + html[body_open_end + 1 :]
    out_html.write_text(html, encoding="utf-8")

    print(f"Wrote RESSEQ path viewer HTML: {out_html}")


def plot_all_atoms_py3dmol(xyz_text: str, atoms: List[XyzAtom], title: str, out_html: Path) -> None:
    """Write an interactive py3Dmol view (HTML) with hover labels for atom comments."""
    try:
        import py3Dmol
    except Exception as e:  # pragma: no cover
        print(
            "py3Dmol is required for the molecule viewer. Install it with: pip install py3dmol\n"
            f"Import error: {e}"
        )
        return

    # Map atom index -> original per-line comment string.
    comments = [a.raw_comment for a in atoms]
    comments_json = json.dumps(comments)

    view = py3Dmol.view(width=900, height=650)
    view.addModel(xyz_text, "xyz")

    # Show inferred bonds and element-based coloring.
    view.setStyle(
        {},
        {
            "stick": {"radius": 0.18, "colorscheme": "Jmol"},
            "sphere": {"radius": 0.33, "colorscheme": "Jmol"},
        },
    )
    view.zoomTo()

    # Hover: update a fixed info bar (lighter/faster than creating 3D labels).
    hover_on = (
        "function(atom, viewer) {"
        f"var comments = {comments_json};"
        "var idx = (atom.index !== undefined) ? atom.index : (atom.serial !== undefined ? atom.serial - 1 : -1);"
        "var msg = (idx >= 0 && idx < comments.length) ? comments[idx] : '';"
        "if (!msg) { msg = (atom.elem || 'atom'); }"
        "var el = document.getElementById('hoverinfo'); if (el) { el.textContent = msg; }"
        "}"
    )
    hover_off = "function(atom, viewer) { var el = document.getElementById('hoverinfo'); if (el) { el.textContent = ''; } }"
    view.setHoverable({}, True, hover_on, hover_off)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    try:
        view.write_html(str(out_html))
    except Exception:
        # Fallback for older py3Dmol versions
        if hasattr(view, "_make_html"):
            out_html.write_text(view._make_html(), encoding="utf-8")
        else:
            raise

    # Patch in a top-of-page title + hover info bar.
    html = out_html.read_text(encoding="utf-8")
    header_html = (
        "<div id=\"titlebar\" style=\"font-family: sans-serif; padding: 10px 12px; border-bottom: 1px solid #ddd;\">"
        f"<div style=\"font-size: 16px; font-weight: 600;\">{title}</div>"
        "<div id=\"hoverinfo\" style=\"margin-top: 6px; font-size: 13px; color: #333; min-height: 1.2em;\"></div>"
        "</div>"
    )
    if "id=\"hoverinfo\"" not in html:
        if "<body" in html and ">" in html:
            # Insert immediately after the opening <body ...> tag.
            body_open_end = html.find(">", html.find("<body"))
            if body_open_end != -1:
                html = html[: body_open_end + 1] + "\n" + header_html + "\n" + html[body_open_end + 1 :]
    out_html.write_text(html, encoding="utf-8")

    print(f"Wrote py3Dmol viewer HTML: {out_html}")
    print("Open it in a browser and hover atoms to see metadata in the top bar.")


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
        help="Only scan --dir for files where CYS(CA+CB+SG) residue count != 4 and write .py3dmol.html for those outliers",
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

        all_xyz = find_xyz_files(xyz_dir)
        if not all_xyz:
            print(f"No .xyz files found in: {xyz_dir}")
            return

        outliers_with_3 = 0
        outliers_with_5 = 0
        max_open = 5
        num_open = 0

        # Parse each file once. Histogram includes ALL files; outlier HTML only for outliers.
        atoms_by_path: Dict[Path, List[XyzAtom]] = {}
        sg_bond_lengths_all: List[float] = []
        sg_bond_sources_all: List[str] = []
        files_with_lt4_sg = 0
        outliers: List[Path] = []

        for p in all_xyz:
            atoms = parse_xyz_atoms(p)
            atoms_by_path[p] = atoms

            # For histogram: take ONLY the 4 nearest SG distances (drops any 5th farther SG).
            sg_dists = sg_bond_lengths_to_center(atoms, max_sgs=4)
            if len(sg_dists) < 4:
                files_with_lt4_sg += 1
            sg_bond_lengths_all.extend(sg_dists)
            sg_bond_sources_all.extend([p.name] * len(sg_dists))

            n = count_cys_residues_with_ca_cb_sg(atoms)
            if n != 4:
                outliers.append(p)
                if n == 3:
                    outliers_with_3 += 1
                elif n == 5:
                    outliers_with_5 += 1

        if not outliers:
            print("No outlier xyz files found (all have exactly 4 CYS(CA+CB+SG) residues).")
            print(f"Outliers: [0/{len(all_xyz)}]")
        else:
            print("Files with CYS(CA+CB+SG) residue count != 4:")
            for p in outliers:
                atoms = atoms_by_path.get(p, [])
                n = count_cys_residues_with_ca_cb_sg(atoms)
                print(f"  - {p.name} (count={n})")
                xyz_text = p.read_text(encoding="utf-8")
                plot_all_atoms_py3dmol(
                    xyz_text,
                    atoms,
                    title=p.name,
                    out_html=p.with_suffix(".py3dmol.html"),
                )

                if args.open and num_open < max_open:
                    out_html = p.with_suffix(".py3dmol.html")
                    webbrowser.open(out_html.resolve().as_uri())
                    num_open += 1

        print(f"Outliers: [{len(outliers)}/{len(all_xyz)}]")
        print(f"Outliers with 3 residues: {outliers_with_3}")
        print(f"Outliers with 5 residues: {outliers_with_5}")
        if sg_bond_lengths_all:
            mean = sum(sg_bond_lengths_all) / len(sg_bond_lengths_all)
            print(
                "SG-to-center distances collected for histogram: "
                f"{len(sg_bond_lengths_all)} values from {len(all_xyz)} file(s) "
                f"(min={min(sg_bond_lengths_all):.3f} Å, max={max(sg_bond_lengths_all):.3f} Å, mean={mean:.3f} Å)"
            )
            if files_with_lt4_sg:
                print(
                    f"Note: {files_with_lt4_sg} file(s) had <4 CYS SG atoms; "
                    "their histogram contribution is truncated."
                )

        plot_sg_bond_length_histogram(
            sg_bond_lengths_all,
            sg_bond_sources_all,
            title="Histogram: center → (4 nearest) CYS SG bond lengths\n(all files; outliers auto-trimmed)",
        )
        if args.open:
            print(f"Opened {min(num_open, max_open)}/{len(outliers)} HTML files (max {max_open}).")

        return

    selected_paths = resolve_xyz_files(args.file, xyz_dir)
    if len(selected_paths) == 1:
        print(f"Selected file: {selected_paths[0]}")
    else:
        print(f"Selected {len(selected_paths)} files:")
        for p in selected_paths:
            print(f"  - {p}")

    atoms_by_source: List[Tuple[str, List[XyzAtom]]] = []
    for p in selected_paths:
        atoms_by_source.append((p.name, parse_xyz_atoms(p)))

    # Only show the alignment/trajectory figure when we actually have multiple inputs
    # (e.g. a wildcard pattern). If the user gave one concrete file path, just emit
    # the py3Dmol HTML viewer.
    single_concrete_file = False
    if len(selected_paths) == 1 and args.file is not None and not _has_glob(args.file):
        p = Path(args.file)
        single_concrete_file = p.exists() and p.is_file() and p.suffix.lower() == ".xyz"

    if not single_concrete_file and len(selected_paths) > 1:
        resseq_out_html = xyz_dir / "resseq_paths.py3dmol.html"
        plot_resseq_paths(
            atoms_by_source,
            title="origin → SG → CB → CA per RESSEQ",
            out_html=resseq_out_html,
        )
    else:
        resseq_out_html = None

    # Generate one py3Dmol HTML per file.
    for p in selected_paths:
        atoms = parse_xyz_atoms(p)
        xyz_text = p.read_text(encoding="utf-8")
        plot_all_atoms_py3dmol(
            xyz_text,
            atoms,
            title=f"{p.name}",
            out_html=p.with_suffix(".py3dmol.html"),
        )

    if args.open:
        # If we built the multi-file RESSEQ-path viewer, prefer opening that.
        if resseq_out_html is not None:
            webbrowser.open(resseq_out_html.resolve().as_uri())
        else:
            out_html = selected_paths[0].with_suffix(".py3dmol.html")
            webbrowser.open(out_html.resolve().as_uri())


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
    Add the `--open` flag to above if you want to oen a max of 5 html files for those outliers.
    """
    main()
