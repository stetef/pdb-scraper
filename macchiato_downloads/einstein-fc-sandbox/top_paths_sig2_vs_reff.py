#!/usr/bin/env python3
"""For each FEFF result dir, parse the xmu file, take the top-30 paths
overall (by curved-wave amplitude ratio, all nlegs in the same pool)
and plot:

    sig2_tot vs reff      every surviving path        (green fit)
    sig2_tot vs angle     surviving nlegs=3 paths     (blue fit)
    sig2_tot vs dihedral  surviving nlegs=4 linear    (blue fit)

Color convention: reff axis is green, angle/dihedral axis is blue.

Labels:
    nlegs=2/3/4 paths get chemical labels (Zn-SG(Cys),
        Zn-ND1(HisND)-CG(HisND), Zn-A-B-A rattle, etc.) so paths from
        different workdirs in the same chemical environment merge.
    nlegs >= 5 paths are labeled by their full atom sequence
        (e.g. Zn-ND1-CE1-NE2-CD2-CG-ND1) so each unique chemical path
        gets its own subplot — a HisND ring traversal lands separately
        from other 7-leg paths in the same residue.
    Top-N paths whose non-Zn legs include an H atom are dropped (kept
        out for chemistry-clarity, same convention as the prior heavy-
        only matchers): a structure contributes < TOP_N rows in that
        case rather than backfilling.
    Same-residue paths are plotted as circles, cross-residue paths as
    triangles within the same subplot. The linear fit and amp stats
    span both marker types.

Atom naming follows the reordered-by-distance ATOMS block in feff.inp
matched to the full .xyz geometry (after translating xyz so Zn is at
the origin). For histidine/cysteine ligands:
    stem:  CA - CB - CG (and for Cys: CA - CB - SG)
    ring:  CG - CD2 - NE2 - CE1 - ND1 - CG       (His)

Path identification: paths.dat (read via scripts/identify_path_atoms.py)
gives the exact xyz of each leg endpoint, so every scatterer's
ATOMS-block index is determined directly — no reff-based topology
guessing. The leg sequence then drives:
    nlegs=3: triangle Zn-A-B-Zn. The plotted angle is the interior
             angle at the closer-to-Zn atom A, i.e. Zn-A-B (forward-
             scattering geometry has this near 180 deg).
    nlegs=4: topology read off the leg pattern (rattle = A-B-A,
             double_back = A-Zn-A, diff_back = A-Zn-B, linear = A-B-C).
             Only `linear` (4 distinct vertices) has a defined dihedral.

Subplots are kept only if a label has >= 5 points (circles + triangles
combined) across workdirs.

Usage:
    python top_paths_sig2_vs_reff.py <dir> [<dir> ...] [--out fig.png]

Each <dir> is a flat directory containing:
    feff.inp
    xmu*.dat                (path table; e.g. xmu-zn-1his-eps-charge-3.dat)
    <stem>.xyz              (no "_clean" / "_trj" suffix; ORCA-optimized geom)

Output PNGs (written into figs/ alongside --out's parent):
    figs/
        <prefix>_vs_reff[_bycomp]<ext>          sig2 vs reff
        <prefix>_vs_angle_nlegs3[_bycomp]<ext>  sig2 vs angle (nlegs=3 only)
        <prefix>_vs_dihedral_nlegs4[_bycomp]<ext> sig2 vs dihedral (nlegs=4 linear)
        <stem>_byresidue_reff[_nlegs_lt5]<ext>  all within-residue paths
    where stem is --out's stem ("top_paths_sig2_vs_reff") and prefix is
    that stem with the trailing "_vs_reff" stripped ("top_paths_sig2").

Per-subplot annotation: total point count and triangle (cross-residue)
count, cw amp avg + Q1/Q3, and the linear-fit (m, b, R²).
"""

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
})

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robust_fit import plot_slope_ci, theilsen_linfit  # noqa: E402


def _fmt_p(p):
    """Compact p-value formatter for in-figure annotation."""
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-3:
        return f"{p:.0e}"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.2f}"

# Reuse paths.dat parsing from scripts/identify_path_atoms.py so a single
# script call here drives both the leg-level atom identification and the
# downstream plotting.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from identify_path_atoms import parse_paths_dat as _parse_paths_dat  # noqa: E402


# ---- knobs you'll most often want to tweak --------------------------------
TOP_N = 10          # how many highest-cw_amp paths to keep per workdir
MIN_POINTS = 7      # min points (circles + triangles) for a subplot to render
# Per-nlegs seed caps for the nlegs<5 backfilled bycomp plot. Each
# structure contributes up to this many highest-cw_amp *intra-residue*
# paths of the given nlegs class to the subplot-grid seed pool.
TOP_N_NLEGS2 = 8
TOP_N_NLEGS3 = 3
TOP_N_NLEGS4 = 1
REFF_FIT_COLOR = "black"       # fits drawn in black across all panels
ANGLE_FIT_COLOR = "black"

COMPOSITION_COLORS = {
    "1res":     "#404040",       # dark gray
    "4cys":     "#1f77b4",       # blue (tab10[0])
    "3cys1his": "tab:purple",
    "2cys2his": "#ff7f0e",       # orange (tab10[1])
    "1cys3his": "tab:red",
    "4his":     "tab:green",
}
COMPOSITION_ORDER = ["1res", "4cys", "3cys1his", "2cys2his", "1cys3his", "4his"]

# Color by which study each workdir belongs to (charge sweep, geometric
# angle sweep, or the cross-family survey). Substring match on the
# workdir directory name picks the study.
STUDY_COLORS = {
    "charge study": "#1f77b4",   # blue
    "angle study":  "#ff7f0e",   # orange
    "family study": "#2ca02c",   # green
}
STUDY_ORDER = ["charge study", "angle study", "family study"]

# Lighter-gray axis spine for non-highlighted subplots so the bold-spine
# highlights pop more.
SPINE_DEFAULT_COLOR = "0.65"
SPINE_HIGHLIGHT_COLOR = "black"


def study_of_workdir(workdir_name):
    """Classify a workdir name into one of three studies by substring.
    'charge' in name → charge study (charge sweep).
    'alpha'  in name → angle  study (Zn-coord-angle sweep).
    everything else  → family study (cross-protein-family survey).
    """
    name = str(workdir_name).lower()
    if "charge" in name:
        return "charge study"
    if "alpha" in name:
        return "angle study"
    return "family study"
# ---------------------------------------------------------------------------


BOND_CUTOFFS = {
    frozenset(["Zn", "N"]): 2.50,
    frozenset(["Zn", "O"]): 2.50,
    frozenset(["Zn", "S"]): 2.80,
    frozenset(["Zn", "C"]): 2.30,
    frozenset(["C", "C"]): 1.75,
    frozenset(["C", "N"]): 1.65,
    frozenset(["C", "O"]): 1.65,
    frozenset(["C", "H"]): 1.25,
    frozenset(["C", "S"]): 1.90,
    frozenset(["N", "H"]): 1.20,
    frozenset(["N", "N"]): 1.55,
    frozenset(["O", "H"]): 1.20,
    frozenset(["S", "H"]): 1.45,
}


def distance(a, b):
    return math.sqrt(
        (a["x"] - b["x"]) ** 2
        + (a["y"] - b["y"]) ** 2
        + (a["z"] - b["z"]) ** 2
    )


def is_bonded(a, b):
    key = frozenset([a["element"], b["element"]])
    cutoff = BOND_CUTOFFS.get(key)
    if cutoff is None:
        return False
    d = distance(a, b)
    return 0.1 < d <= cutoff


def angle_at_apex(apex, p1, p2):
    """Interior angle at apex between rays apex->p1 and apex->p2 (deg)."""
    v1 = np.array([p1["x"] - apex["x"], p1["y"] - apex["y"], p1["z"] - apex["z"]])
    v2 = np.array([p2["x"] - apex["x"], p2["y"] - apex["y"], p2["z"] - apex["z"]])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos_a))


def dihedral_unsigned(p1, p2, p3, p4):
    """Unsigned dihedral angle for points p1-p2-p3-p4 (deg, [0, 180]).

    Defined as the angle between the plane containing (p1, p2, p3) and
    the plane containing (p2, p3, p4), measured around the p2-p3 bond.
    Path-direction-invariant (reversing p1..p4 order gives the same
    value), so safe for FEFF paths where Zn-A-B-C-Zn and Zn-C-B-A-Zn
    are degenerate."""
    b1 = np.array([p2["x"] - p1["x"], p2["y"] - p1["y"], p2["z"] - p1["z"]])
    b2 = np.array([p3["x"] - p2["x"], p3["y"] - p2["y"], p3["z"] - p2["z"]])
    b3 = np.array([p4["x"] - p3["x"], p4["y"] - p3["y"], p4["z"] - p3["z"]])
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    nn1, nn2 = np.linalg.norm(n1), np.linalg.norm(n2)
    if nn1 < 1e-9 or nn2 < 1e-9:
        return float("nan")
    cos_d = float(np.clip(np.dot(n1, n2) / (nn1 * nn2), -1.0, 1.0))
    return math.degrees(math.acos(cos_d))


def parse_feff_atoms(feff_path):
    """Parse ATOMS + POTENTIALS from feff.inp. Returns list of dicts with
    keys: x, y, z, pot, element, dist_from_zn (from 5th ATOMS column)."""
    atoms = []
    potentials = {}
    state = None
    for raw in Path(feff_path).read_text().splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        first = stripped.split()[0]
        if first.isalpha() and first.isupper():
            state = {"ATOMS": "atoms", "POTENTIALS": "potentials"}.get(first)
            continue
        if state == "atoms":
            parts = stripped.split()
            if len(parts) >= 4:
                atoms.append(
                    {
                        "x": float(parts[0]),
                        "y": float(parts[1]),
                        "z": float(parts[2]),
                        "pot": int(parts[3]),
                        "dist_from_zn": float(parts[4]) if len(parts) > 4 else None,
                    }
                )
        elif state == "potentials":
            parts = stripped.split()
            if len(parts) >= 3:
                try:
                    potentials[int(parts[0])] = parts[2]
                except ValueError:
                    pass
    for atom in atoms:
        atom["element"] = potentials[atom["pot"]]
    return atoms


def parse_xyz(xyz_path):
    """Parse a standard .xyz file. Returns list of dicts: element, x, y, z."""
    lines = Path(xyz_path).read_text().splitlines()
    n = int(lines[0].strip())
    out = []
    for raw in lines[2 : 2 + n]:
        parts = raw.split()
        out.append(
            {
                "element": parts[0].capitalize(),
                "x": float(parts[1]),
                "y": float(parts[2]),
                "z": float(parts[3]),
            }
        )
    return out


def align_xyz_to_feff(feff_atoms, xyz_atoms, tol=0.05):
    """Translate xyz so its Zn sits at the origin, then verify each feff
    atom matches an xyz atom (same element, same coords within tol).
    Mutates feff_atoms to add 'xyz_index' pointing back to the xyz entry.
    Returns the (possibly element-corrected) feff_atoms."""
    zn_xyz = [a for a in xyz_atoms if a["element"] == "Zn"]
    if len(zn_xyz) != 1:
        raise ValueError(f"expected exactly one Zn in xyz, found {len(zn_xyz)}")
    z = zn_xyz[0]
    shifted = [
        {**a, "x": a["x"] - z["x"], "y": a["y"] - z["y"], "z": a["z"] - z["z"]}
        for a in xyz_atoms
    ]
    used = set()
    for fi, fa in enumerate(feff_atoms):
        best_j, best_d = None, float("inf")
        for j, xa in enumerate(shifted):
            if j in used:
                continue
            d = distance(fa, xa)
            if d < best_d:
                best_d = d
                best_j = j
        if best_j is None or best_d > tol:
            fa["xyz_index"] = None
            fa["xyz_d"] = best_d
            continue
        used.add(best_j)
        fa["xyz_index"] = best_j
        fa["xyz_d"] = best_d
        xyz_el = shifted[best_j]["element"]
        if fa["element"] != xyz_el:
            # trust xyz as source of truth (per user's ask)
            fa["element"] = xyz_el
    return feff_atoms


def build_adjacency(atoms):
    n = len(atoms)
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if is_bonded(atoms[i], atoms[j]):
                adj[i].add(j)
                adj[j].add(i)
    return adj


def _fmt(base, tag):
    return f"{base}({tag})" if tag else base


def _find_residues(atoms, adj, zn_idx):
    """Connected components of heavy non-Zn atoms. Each component is a
    chemical residue. Returns list of dicts with keys: atoms (set of
    atom indices), first_shell (the component atom bonded to Zn, or
    None)."""
    heavy = {
        i for i, a in enumerate(atoms) if a["element"] not in ("H", "Zn")
    }
    visited = set()
    residues = []
    for start in sorted(heavy):
        if start in visited:
            continue
        comp = set()
        stack = [start]
        while stack:
            i = stack.pop()
            if i in visited:
                continue
            visited.add(i)
            comp.add(i)
            for j in adj[i]:
                if j in heavy and j not in visited:
                    stack.append(j)
        zn_bonded = [i for i in comp if zn_idx in adj[i]]
        first_shell = (
            min(zn_bonded, key=lambda i: distance(atoms[zn_idx], atoms[i]))
            if zn_bonded
            else None
        )
        residues.append({"atoms": comp, "first_shell": first_shell})
    return residues


def _classify(residue, atoms):
    elements = [atoms[i]["element"] for i in residue["atoms"]]
    if "S" in elements:
        return "Cys"
    if elements.count("N") >= 2 and elements.count("C") >= 3:
        return "His"
    return "Other"


def name_atoms_and_residues(atoms, adj):
    """Label Zn, His, and Cys atoms across all coordinated residues.
    Returns (names, residues) where names is list of strings (one per
    atom) and residues is list of dicts with type and tag fields added."""
    n = len(atoms)
    names = [None] * n
    zn_idx = next(
        (i for i, a in enumerate(atoms) if a["element"] == "Zn"), None
    )
    if zn_idx is None:
        return names, []
    names[zn_idx] = "Zn"

    residues = _find_residues(atoms, adj, zn_idx)
    # Order residues by first-shell distance (closest first)
    residues.sort(
        key=lambda r: (
            distance(atoms[zn_idx], atoms[r["first_shell"]])
            if r["first_shell"] is not None
            else float("inf")
        )
    )
    counters = {"His": 0, "Cys": 0, "Other": 0}
    for r in residues:
        r["type"] = _classify(r, atoms)
        counters[r["type"]] += 1
        r["tag"] = f"{r['type']}{counters[r['type']]}"

    for r in residues:
        if r["type"] == "His":
            _name_his(r, atoms, adj, names)
        elif r["type"] == "Cys":
            _name_cys(r, atoms, adj, names)
        # "Other" → leave atoms with element-symbol fallback labels
    return names, residues


def _name_his(residue, atoms, adj, names):
    """Name a histidine residue, auto-detecting eps vs delta coord. Sets
    residue['atom_tag'] to 'HisND' (delta coord) or 'HisNE' (eps coord)
    so paths from same-coord-type residues merge in plots."""
    comp = residue["atoms"]
    zn_n = residue["first_shell"]
    if zn_n is None or atoms[zn_n]["element"] != "N":
        return

    ring_ns = [i for i in comp if atoms[i]["element"] == "N"]
    if len(ring_ns) != 2:
        return
    other_n = next(n for n in ring_ns if n != zn_n)

    ring_cs = [i for i in comp if atoms[i]["element"] == "C"]
    ce1_cands = [
        c for c in ring_cs if zn_n in adj[c] and other_n in adj[c]
    ]
    if not ce1_cands:
        return
    ce1 = ce1_cands[0]

    other_ring_cs = [
        c
        for c in ring_cs
        if c != ce1 and (zn_n in adj[c] or other_n in adj[c])
    ]
    if len(other_ring_cs) != 2:
        return

    ring_set = {zn_n, other_n, ce1, *other_ring_cs}
    cg = None
    for c in other_ring_cs:
        for j in adj[c]:
            if (
                atoms[j]["element"] == "C"
                and j in comp
                and j not in ring_set
            ):
                cg = c
                break
        if cg is not None:
            break
    if cg is None:
        return
    cd2 = next(c for c in other_ring_cs if c != cg)

    # Detection: if Zn-N is directly bonded to CG → delta (Zn-ND1).
    # Otherwise eps (Zn-NE2). The atom_tag — "HisND" or "HisNE" — is
    # what shows up in path labels; same-coord residues merge.
    if zn_n in adj[cg]:
        residue["atom_tag"] = "HisND"
        zn_n_base, other_n_base = "ND1", "NE2"
    else:
        residue["atom_tag"] = "HisNE"
        zn_n_base, other_n_base = "NE2", "ND1"
    tag = residue["atom_tag"]
    names[zn_n] = _fmt(zn_n_base, tag)
    names[other_n] = _fmt(other_n_base, tag)
    names[ce1] = _fmt("CE1", tag)
    names[cd2] = _fmt("CD2", tag)
    names[cg] = _fmt("CG", tag)

    cb_cands = [
        j
        for j in adj[cg]
        if atoms[j]["element"] == "C" and j in comp and j not in ring_set
    ]
    if not cb_cands:
        return
    cb = cb_cands[0]
    names[cb] = _fmt("CB", tag)

    ca_cands = [
        j
        for j in adj[cb]
        if atoms[j]["element"] == "C" and j in comp and j != cg
    ]
    ca = ca_cands[0] if ca_cands else None
    if ca is not None:
        names[ca] = _fmt("CA", tag)

    _label_hs(zn_n, zn_n_base, tag, atoms, adj, names)
    _label_hs(other_n, other_n_base, tag, atoms, adj, names)
    _label_hs(ce1, "CE1", tag, atoms, adj, names)
    _label_hs(cd2, "CD2", tag, atoms, adj, names)
    _label_hs(cg, "CG", tag, atoms, adj, names)
    _label_hs(cb, "CB", tag, atoms, adj, names)
    if ca is not None:
        _label_hs(ca, "CA", tag, atoms, adj, names)


def _name_cys(residue, atoms, adj, names):
    comp = residue["atoms"]
    residue["atom_tag"] = "Cys"
    tag = residue["atom_tag"]
    sg = residue["first_shell"]
    if sg is None or atoms[sg]["element"] != "S":
        return
    names[sg] = _fmt("SG", tag)

    cb_cands = [
        j for j in adj[sg] if atoms[j]["element"] == "C" and j in comp
    ]
    if not cb_cands:
        return
    cb = cb_cands[0]
    names[cb] = _fmt("CB", tag)

    ca_cands = [
        j
        for j in adj[cb]
        if atoms[j]["element"] == "C" and j in comp and j != sg
    ]
    ca = ca_cands[0] if ca_cands else None
    if ca is not None:
        names[ca] = _fmt("CA", tag)

    _label_hs(sg, "SG", tag, atoms, adj, names)
    _label_hs(cb, "CB", tag, atoms, adj, names)
    if ca is not None:
        _label_hs(ca, "CA", tag, atoms, adj, names)


def _label_hs(heavy_idx, heavy_base, tag, atoms, adj, names):
    if heavy_idx is None:
        return
    hs = sorted(j for j in adj[heavy_idx] if atoms[j]["element"] == "H")
    if not hs:
        return
    suffix = heavy_base[1:]  # "CE1" -> "E1", "CB" -> "B"
    if len(hs) == 1:
        names[hs[0]] = _fmt("H" + suffix, tag)
    else:
        for k, h in enumerate(hs, start=1):
            names[h] = _fmt("H" + suffix + str(k), tag)


def parse_xmu_paths(xmu_path):
    """Parse the path table at the bottom of xmu.dat. Returns list of dicts."""
    lines = Path(xmu_path).read_text().splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if "cw amp ratio" in line and "nlegs" in line and "reff" in line:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"no path table header found in {xmu_path}")
    paths = []
    for raw in lines[header_idx + 1 :]:
        stripped = raw.strip()
        if not stripped:
            continue
        parts = stripped.lstrip("#").split()
        if len(parts) < 6:
            continue
        try:
            path = {
                "file": int(parts[0]),
                "sig2_tot": float(parts[1]),
                "cw_amp": float(parts[2]),
                "deg": float(parts[3]),
                "nlegs": int(parts[4]),
                "reff": float(parts[5]),
            }
        except ValueError:
            continue
        paths.append(path)
    return paths


def atom_label(atom, name):
    return name if name else atom["element"]


def match_paths_dat_to_feff(paths_dat_path, feff_atoms, tol=2e-3):
    """Map every path in paths.dat to its leg-by-leg feff_atoms indices.

    paths.dat lists each scattering path with the explicit xyz of each
    leg's endpoint, so we can identify the participating atoms exactly
    rather than guessing topology from reff. The final leg of each
    path lands at the absorber (xyz = origin) and is encoded as index 0.

    Returns dict[int, list[int]] keyed by FEFF path index.
    """
    feff_paths = _parse_paths_dat(Path(paths_dat_path))
    lookup = {}
    for fp in feff_paths:
        leg_indices = []
        ok = True
        for leg in fp.legs:
            if (
                leg.ipot == 0
                and abs(leg.x) < tol
                and abs(leg.y) < tol
                and abs(leg.z) < tol
            ):
                leg_indices.append(0)
                continue
            best_i, best_d = None, float("inf")
            for i in range(1, len(feff_atoms)):
                a = feff_atoms[i]
                d = max(
                    abs(a["x"] - leg.x),
                    abs(a["y"] - leg.y),
                    abs(a["z"] - leg.z),
                )
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i is None or best_d > tol:
                ok = False
                break
            leg_indices.append(best_i)
        if ok:
            lookup[fp.index] = leg_indices
    return lookup


def _path_scatterers(p, path_legs):
    """Return the list of feff_atoms indices for the non-final legs of
    path `p`, in path order. The final leg (return to absorber) is
    dropped. Returns None if the path isn't in path_legs."""
    legs = path_legs.get(p["file"]) if path_legs is not None else None
    if not legs:
        return None
    return legs[:-1]


def _classify_nlegs4(scatterers):
    """Classify a 3-element scatterer sequence (excluding the final Zn
    return) into one of the four nlegs=4 topologies. The scatterer list
    comes straight from paths.dat in path order, so this is exact."""
    a, b, c = scatterers
    if b == 0:
        return "double_back" if a == c else "diff_back"
    if a == c:
        return "rattle"
    return "linear"


def _canonical_seq(seq, feff_atoms, names):
    """Return the lexicographically smaller of seq and reversed(seq) by
    atom-name string."""
    fwd = [
        names[i] or f"{feff_atoms[i]['element']}{i+1}" for i in seq
    ]
    rev_idx = list(reversed(seq))
    rev = [
        names[i] or f"{feff_atoms[i]['element']}{i+1}" for i in rev_idx
    ]
    return seq if fwd <= rev else rev_idx


def structure_composition(residues):
    """Classify a structure by its labeled-residue (His/Cys) makeup.
    Returns one of "1res", "4cys", "3cys1his", "2cys2his", "1cys3his",
    "4his", or "other"."""
    labeled = [r for r in residues if r.get("atom_tag")]
    n_total = len(labeled)
    if n_total == 1:
        return "1res"
    if n_total != 4:
        return "other"
    n_cys = sum(1 for r in labeled if r["atom_tag"] == "Cys")
    n_his = n_total - n_cys
    return {
        (4, 0): "4cys",
        (3, 1): "3cys1his",
        (2, 2): "2cys2his",
        (1, 3): "1cys3his",
        (0, 4): "4his",
    }.get((n_cys, n_his), "other")


def identify_residue_for_path(p, feff_atoms, residues, path_legs, adj=None):
    """Return the single residue dict containing every scatterer of path
    `p`, or None if the path spans residues / has Zn as an intermediate /
    isn't represented in path_legs. Atom membership comes straight from
    paths.dat via path_legs, so nlegs >= 5 identification is exact."""
    scatterers = _path_scatterers(p, path_legs)
    if not scatterers:
        return None
    res = None
    for atom_idx in scatterers:
        if atom_idx == 0:
            return None  # Zn appearing mid-path => double_back / diff_back
        ri = residue_index_of(atom_idx, residues, feff_atoms, adj)
        if ri is None:
            return None
        if res is None:
            res = ri
        elif ri != res:
            return None
    return residues[res] if res is not None else None


def residue_index_of(atom_idx, residues, atoms=None, adj=None):
    """Index of residue containing atom_idx. For an H atom (which isn't
    part of the heavy-atom-only residue components), we map it to the
    residue of its bonded heavy atom."""
    for k, r in enumerate(residues):
        if atom_idx in r["atoms"]:
            return k
    if atoms is not None and adj is not None and atoms[atom_idx]["element"] == "H":
        for j in adj[atom_idx]:
            if atoms[j]["element"] not in ("H", "Zn"):
                for k, r in enumerate(residues):
                    if j in r["atoms"]:
                        return k
    return None


def _find_xmu(workdir):
    candidates = [workdir / "xmu.dat"] + sorted(workdir.glob("xmu*.dat"))
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"no xmu*.dat in {workdir}")


def _find_xyz(workdir):
    xyzs = [
        p
        for p in workdir.glob("*.xyz")
        if not p.stem.endswith("_clean") and not p.stem.endswith("_trj")
    ]
    if not xyzs:
        raise FileNotFoundError(f"no non-_clean/_trj .xyz file in {workdir}")
    return sorted(xyzs, key=lambda p: len(p.stem))[0]


def _atom_name(idx, feff_atoms, names):
    a = feff_atoms[idx]
    return names[idx] or f"{a['element']}{idx + 1}"


def _format_subplot_title(label):
    """Reformat a path label like 'Zn-ND1(HisNE)-CB(HisNE)' to
    'HisNE: Zn-ND1-CB' so subplot titles match the bond-plot style.
    When the label spans multiple residues (cross-residue paths) or
    has no residue tag, it's returned unchanged."""
    tags = set(re.findall(r"\(([^)]+)\)", label))
    if len(tags) != 1:
        return label
    tag = next(iter(tags))
    stripped = re.sub(r"\([^)]+\)", "", label)
    return f"{tag}: {stripped}" if stripped else tag


def _wrap_path_label(label, max_len=22):
    """Break a path label like Zn-ND1-CE1-NE2-CD2-CG-ND1 onto multiple
    lines at hyphen boundaries when it exceeds max_len, so subplot
    titles don't overflow into the next subplot. The trailing hyphen
    stays on each non-final line so the path reads continuously."""
    if len(label) <= max_len:
        return label
    parts = label.split("-")
    lines = []
    cur = ""
    for p in parts:
        candidate = f"{cur}-{p}" if cur else p
        if cur and len(candidate) > max_len:
            lines.append(cur + "-")
            cur = p
        else:
            cur = candidate
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _row_for_path(p, feff_atoms, names, residues, adj, workdir, path_legs):
    """Build a row dict for one path using the leg sequence from paths.dat.

    The row always has reff/sig2_tot/cw_amp/label/cross_residue and may
    have angle (nlegs=3) or dihedral (nlegs=4 linear topology only). For
    nlegs >= 5 the label is the full path sequence (e.g.
    Zn-ND1-CE1-NE2-CD2-CG-ND1) so each unique path lands in its own
    subplot. cross_residue is True when the path visits atoms in more
    than one chemical residue; rows with this flag get rendered as
    triangles (vs. circles for same-residue).

    Returns None when the path isn't represented in paths.dat or any of
    its non-Zn legs is an H atom (kept out for chemistry-clarity reasons,
    matching the previous heavy-only behavior). Callers filter Nones."""
    scatterers = _path_scatterers(p, path_legs)
    if scatterers is None:
        return None
    # Drop paths that traverse H — same filter as the prior heavy-only
    # matchers, kept for plot comparability.
    if any(a != 0 and feff_atoms[a]["element"] == "H" for a in scatterers):
        return None

    nlegs = p["nlegs"]
    res_ids = [
        residue_index_of(a, residues, feff_atoms, adj) if a != 0 else None
        for a in scatterers
    ]
    non_none = [r for r in res_ids if r is not None]
    cross = (
        len(non_none) != len([a for a in scatterers if a != 0])
        or len(set(non_none)) > 1
    )

    base = {
        "nlegs": nlegs,
        "reff": p["reff"],
        "sig2_tot": p["sig2_tot"],
        "cw_amp": p["cw_amp"],
        "file": p["file"],
        "workdir": workdir.name,
        "cross_residue": cross,
    }

    def attach_fs_dist(row):
        """Annotate row with fs_dist (Zn → first-shell atom of the path's
        residue) when the path is contained in a single labeled residue."""
        if row.get("cross_residue"):
            return row
        res = identify_residue_for_path(p, feff_atoms, residues, path_legs, adj)
        if res is None:
            return row
        fs = res.get("first_shell")
        if fs is None:
            return row
        row["fs_dist"] = distance(feff_atoms[0], feff_atoms[fs])
        row["res_tag"] = res.get("atom_tag")
        return row

    if nlegs == 2:
        (i,) = scatterers
        label = f"Zn-{_atom_name(i, feff_atoms, names)}"
        return attach_fs_dist({**base, "label": label})

    if nlegs == 3:
        i_path, j_path = scatterers
        # Convention from prior version: angle at the closer-to-Zn atom.
        if distance(feff_atoms[0], feff_atoms[i_path]) <= distance(
            feff_atoms[0], feff_atoms[j_path]
        ):
            apex, far = i_path, j_path
        else:
            apex, far = j_path, i_path
        label = (
            f"Zn-{_atom_name(apex, feff_atoms, names)}"
            f"-{_atom_name(far, feff_atoms, names)}"
        )
        angle = angle_at_apex(feff_atoms[apex], feff_atoms[0], feff_atoms[far])
        row = {**base, "label": label, "angle": angle}
        return attach_fs_dist(row)

    if nlegs == 4:
        a, b, c = scatterers
        topology = _classify_nlegs4(scatterers)
        if topology == "rattle":
            # Zn-A-B-A-Zn (a == c by definition).
            label = (
                f"Zn-{_atom_name(a, feff_atoms, names)}"
                f"-{_atom_name(b, feff_atoms, names)}"
                f"-{_atom_name(a, feff_atoms, names)}"
            )
            dihedral = None
        elif topology == "double_back":
            label = f"Zn-{_atom_name(a, feff_atoms, names)}-Zn-{_atom_name(a, feff_atoms, names)}"
            dihedral = None
        elif topology == "diff_back":
            # Order endpoints by Zn-distance for stable label merging.
            if distance(feff_atoms[0], feff_atoms[a]) <= distance(
                feff_atoms[0], feff_atoms[c]
            ):
                close_a, far_c = a, c
            else:
                close_a, far_c = c, a
            label = (
                f"Zn-{_atom_name(close_a, feff_atoms, names)}-Zn"
                f"-{_atom_name(far_c, feff_atoms, names)}"
            )
            dihedral = None
        else:  # linear: Zn-A-B-C-Zn
            if distance(feff_atoms[0], feff_atoms[a]) <= distance(
                feff_atoms[0], feff_atoms[c]
            ):
                ordered = (a, b, c)
            else:
                ordered = (c, b, a)
            label = (
                f"Zn-{_atom_name(ordered[0], feff_atoms, names)}"
                f"-{_atom_name(ordered[1], feff_atoms, names)}"
                f"-{_atom_name(ordered[2], feff_atoms, names)}"
            )
            dihedral = dihedral_unsigned(
                feff_atoms[0],
                feff_atoms[ordered[0]],
                feff_atoms[ordered[1]],
                feff_atoms[ordered[2]],
            )
        row = {**base, "label": label, "topology": topology}
        if dihedral is not None:
            row["dihedral"] = dihedral
        return attach_fs_dist(row)

    # nlegs >= 5: paths.dat already gave us the exact atom sequence
    # visited from Zn back to Zn. Drop paths that span multiple residues
    # or land on unlabeled atoms (matches prior heavy-only behavior).
    # Otherwise use the full scatterer sequence for the label, with
    # forward/reverse canonicalization so the same physical path merges
    # across workdirs (FEFF treats Zn-A-B-C-Zn and Zn-C-B-A-Zn as
    # degenerate). This includes paths the old reff-based matcher
    # silently relabeled to a Zn-fs-...-fs-Zn template they didn't
    # actually traverse.
    res = identify_residue_for_path(p, feff_atoms, residues, path_legs, adj)
    if res is None or res.get("first_shell") is None or res.get("atom_tag") is None:
        return None
    canonical = _canonical_seq(scatterers, feff_atoms, names)
    label = "Zn-" + "-".join(_atom_name(i, feff_atoms, names) for i in canonical)
    fs = res["first_shell"]
    row = {**base, "label": label}
    row["fs_dist"] = distance(feff_atoms[0], feff_atoms[fs])
    row["res_tag"] = res.get("atom_tag")
    return row


def process_workdir(workdir):
    workdir = Path(workdir)
    feff_inp = workdir / "feff.inp"
    if not feff_inp.is_file():
        raise FileNotFoundError(f"no feff.inp in {workdir}")
    paths_dat = workdir / "paths.dat"
    if not paths_dat.is_file():
        raise FileNotFoundError(f"no paths.dat in {workdir}")
    xmu_dat = _find_xmu(workdir)
    xyz_path = _find_xyz(workdir)

    feff_atoms = parse_feff_atoms(feff_inp)
    xyz_atoms = parse_xyz(xyz_path)
    align_xyz_to_feff(feff_atoms, xyz_atoms)
    adj = build_adjacency(feff_atoms)
    names, residues = name_atoms_and_residues(feff_atoms, adj)

    composition = structure_composition(residues)
    # paths.dat gives us the exact leg sequence (and thus atom sequence)
    # for every path FEFF considered, indexed by the same path number used
    # in xmu.dat. This replaces the old reff-based topology guessing.
    path_legs = match_paths_dat_to_feff(paths_dat, feff_atoms)

    paths = parse_xmu_paths(xmu_dat)
    # Top TOP_N by cw_amp across all nlegs. Paths whose non-Zn legs
    # include an H atom are dropped (kept out for chemistry-clarity, same
    # as the prior heavy-only behavior), so a structure may contribute
    # fewer than TOP_N rows.
    top_n = sorted(paths, key=lambda p: -p["cw_amp"])[:TOP_N]
    rows = [
        _row_for_path(p, feff_atoms, names, residues, adj, workdir, path_legs)
        for p in top_n
    ]
    rows = [r for r in rows if r is not None]
    for r in rows:
        r["composition"] = composition

    # nlegs<5 pool: every nlegs<5 path that we can build a row for,
    # sorted by cw_amp. The top-N head is the "seed" set that picks which
    # path labels show up in the bycomp-nlegs<5 subplot grid; the full
    # list is what gets plotted in those subplots, so each subplot shows
    # every instance of that path across all structures.
    short_paths = sorted(
        (p for p in paths if p["nlegs"] < 5), key=lambda p: -p["cw_amp"]
    )
    all_short_rows = []
    for p in short_paths:
        r = _row_for_path(
            p, feff_atoms, names, residues, adj, workdir, path_legs
        )
        if r is None:
            continue
        all_short_rows.append(r)
    for r in all_short_rows:
        r["composition"] = composition
    # Per-nlegs intra-residue seed for the nlegs<5 bycomp subplot grid.
    # all_short_rows is already cw_amp-sorted so each per-class slice
    # gives that class's top contributors.
    short_rows = []
    for nlegs_val, cap in (
        (2, TOP_N_NLEGS2), (3, TOP_N_NLEGS3), (4, TOP_N_NLEGS4),
    ):
        cls = [
            r for r in all_short_rows
            if r["nlegs"] == nlegs_val and not r.get("cross_residue")
        ]
        short_rows.extend(cls[:cap])

    # Within-residue rows: classify EVERY path in the xmu file (not just
    # top-N) by which single residue it sits in. Skips paths that span
    # residues or fail identification.
    zn = feff_atoms[0]
    by_residue_rows = []
    for p in paths:
        r = identify_residue_for_path(p, feff_atoms, residues, path_legs, adj)
        if r is None or r.get("atom_tag") is None:
            continue
        fs = r.get("first_shell")
        if fs is None:
            continue
        by_residue_rows.append(
            {
                "workdir": workdir.name,
                "composition": composition,
                "res_tag": r["atom_tag"],
                "reff": p["reff"],
                "sig2_tot": p["sig2_tot"],
                "cw_amp": p["cw_amp"],
                "fs_dist": distance(zn, feff_atoms[fs]),
                "nlegs": p["nlegs"],
                "file": p["file"],
            }
        )

    parts = []
    for r in residues:
        fs = r["first_shell"]
        if fs is None:
            parts.append(f"{r['tag']}(orphan)")
            continue
        fs_name = names[fs] or feff_atoms[fs]["element"]
        # strip the "(tag)" suffix from the bonded-atom label since we
        # already know the residue tag
        base = fs_name.split("(")[0]
        d = distance(feff_atoms[0], feff_atoms[fs])
        parts.append(f"{r['tag']}[Zn-{base}]@{d:.3f}Å")
    print(
        f"# {workdir.name} [{composition}]: {', '.join(parts)} "
        f"(within-residue paths: {len(by_residue_rows)})"
    )
    return {
        "rows": rows,
        "by_residue": by_residue_rows,
        "short_rows": short_rows,
        "all_short_rows": all_short_rows,
    }


def print_summary(all_rows):
    """Per-structure summary of the top-N paths: count by nlegs class
    and how many crossed residues."""
    if not all_rows:
        return
    by_wd = defaultdict(list)
    for r in all_rows:
        by_wd[r["workdir"]].append(r)
    print("\n# top-N path summary per structure")
    print(
        f"{'Structure':<40} {'total':>5} {'cross':>5} "
        f"{'nlegs=2':>7} {'nlegs=3':>7} {'nlegs=4':>7} {'nlegs>=5':>8}"
    )
    print("-" * 92)
    for wd, rows in sorted(by_wd.items()):
        total = len(rows)
        cross = sum(1 for r in rows if r.get("cross_residue"))
        n2 = sum(1 for r in rows if r["nlegs"] == 2)
        n3 = sum(1 for r in rows if r["nlegs"] == 3)
        n4 = sum(1 for r in rows if r["nlegs"] == 4)
        n5p = sum(1 for r in rows if r["nlegs"] >= 5)
        print(
            f"{wd:<40} {total:>5} {cross:>5} "
            f"{n2:>7} {n3:>7} {n4:>7} {n5p:>8}"
        )


def plot_sig2_vs_x(
    all_rows, x_key, x_label, fit_color, out_path, suptitle=None,
    color_by_study=False, seed_rows=None, bold_title=None,
):
    """If seed_rows is given, those rows decide which path labels get a
    subplot and what the displayed n=… count is (e.g. count of top-N
    contributors). all_rows is what actually gets plotted and is what
    drives the amp-stats and the linear fit, so each subplot can show
    every instance of a path even when only a subset seeded it."""
    by_label = defaultdict(list)
    for r in all_rows:
        # rows missing the x-axis quantity (e.g. dihedral undefined for
        # non-linear nlegs=4 topologies) are dropped from this plot
        if r.get(x_key) is None:
            continue
        by_label[r["label"]].append(r)
    if seed_rows is not None:
        seed_by_label = defaultdict(list)
        for r in seed_rows:
            if r.get(x_key) is None:
                continue
            seed_by_label[r["label"]].append(r)
        # Only paths that the seed pool nominated; MIN_POINTS gate
        # applied against the seed count, not the all-instance count.
        # Sort: nlegs ascending, then avg cw_amp descending — so the
        # subplot grid reads as [all nlegs=2 by amp][all nlegs=3 by
        # amp][all nlegs=4 by amp] in row-major order.
        labels = sorted(
            (L for L in seed_by_label if len(seed_by_label[L]) >= MIN_POINTS),
            key=lambda L: (
                seed_by_label[L][0]["nlegs"],
                -float(np.mean([r["cw_amp"] for r in seed_by_label[L]])),
            ),
        )
    else:
        seed_by_label = by_label
        # Sort: nlegs ascending, then mean cw_amp descending — keeps the
        # subplot grid grouped by scatter order (all nlegs=2 first, etc.)
        # with the highest-amp paths leading each group.
        labels = sorted(
            (L for L, rows in by_label.items() if len(rows) >= MIN_POINTS),
            key=lambda L: (
                by_label[L][0]["nlegs"],
                -float(np.mean([r["cw_amp"] for r in by_label[L]])),
            ),
        )
    if not labels:
        print(
            f"\nno {x_key} label has >= {MIN_POINTS} points; "
            f"skipping {out_path}"
        )
        return

    n = len(labels)
    ncols = min(4, max(1, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), squeeze=False
    )
    for ax in axes.flat:
        ax.set_visible(False)

    for k, label in enumerate(labels):
        ax = axes[k // ncols][k % ncols]
        ax.set_visible(True)
        rows = by_label[label]
        xs = [r[x_key] for r in rows]
        sigs = [r["sig2_tot"] for r in rows]
        amps = [r["cw_amp"] for r in rows]
        # Same-residue rows render as circles, cross-residue as triangles.
        # The fit / amp stats span both groups (whole subplot).
        cross_xs = [r[x_key] for r in rows if r.get("cross_residue")]
        if color_by_study:
            # One scatter per (study, cross-residue) group so each marker
            # carries both pieces of info.
            groups = defaultdict(list)
            for r in rows:
                study = study_of_workdir(r.get("workdir", ""))
                key = (study, bool(r.get("cross_residue")))
                groups[key].append(r)
            for (study, is_cross), grp in groups.items():
                color = STUDY_COLORS.get(study, "dimgray")
                marker = "^" if is_cross else "o"
                ax.scatter(
                    [r[x_key] for r in grp],
                    [r["sig2_tot"] for r in grp],
                    color=color, marker=marker,
                    alpha=0.4, s=34, edgecolors="white", linewidths=0.4,
                )
        else:
            same_xs = [r[x_key] for r in rows if not r.get("cross_residue")]
            same_sigs = [r["sig2_tot"] for r in rows if not r.get("cross_residue")]
            cross_sigs = [r["sig2_tot"] for r in rows if r.get("cross_residue")]
            if same_xs:
                ax.scatter(
                    same_xs, same_sigs, marker="o", color="#3a3a3a",
                    alpha=0.4, s=34, edgecolors="white", linewidths=0.4,
                )
            if cross_xs:
                ax.scatter(
                    cross_xs, cross_sigs, marker="^", color="#3a3a3a",
                    alpha=0.4, s=34, edgecolors="white", linewidths=0.4,
                )
        fit = theilsen_linfit(xs, sigs)
        if fit is not None:
            r2 = fit["r2"]
            tau = fit["tau"]
            tau_p = fit["tau_p"]
            plot_slope_ci(ax, fit, np.asarray(xs), color="lightgray", alpha=0.35)
            x_fit = np.array([min(xs), max(xs)])
            ax.plot(
                x_fit, fit["slope"] * x_fit + fit["intercept"], "-",
                color=fit_color, alpha=0.6, lw=2,
            )
        else:
            r2 = float("nan")
            tau = float("nan")
            tau_p = float("nan")
        amp_avg = float(np.mean(amps))
        amp_q1, amp_q3 = (float(x) for x in np.percentile(amps, [25, 75]))
        # All rows in a subplot share a label, so they share an nlegs.
        nlegs_val = rows[0]["nlegs"]
        # Highlight (bold annotation + thick spines) when the fit is tight:
        # R² > 0.9 or a strong rank correlation (|τ| > 0.8 with p < 0.005).
        tau_strong = (
            np.isfinite(tau) and np.isfinite(tau_p)
            and abs(tau) > 0.8 and tau_p < 0.005
        )
        highlight = r2 > 0.9 or tau_strong
        ann_weight = "bold" if highlight else "normal"
        ax.text(
            0.03,
            0.97,
            f"nlegs={nlegs_val}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            fontweight=ann_weight,
        )
        ax.text(
            0.03,
            0.97 - 0.09,
            (
                f"N={len(rows)}\n"
                # f"amp avg={amp_avg:.1f}\n"
                # f"  Q1={amp_q1:.1f}, Q3={amp_q3:.1f}\n"
                f"$R^2$={r2:.3f}\n"
                f"$\\tau$={tau:.2f} (p={_fmt_p(tau_p)})"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(2.3 if highlight else 0.7)
            spine.set_color(
                SPINE_HIGHLIGHT_COLOR if highlight else SPINE_DEFAULT_COLOR
            )
        ax.set_title(
            _wrap_path_label(_format_subplot_title(label)), fontsize=9
        )
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(r"$\sigma^2_{tot}$ (Å$^2$)", fontsize=9)
        ax.grid(alpha=0.3)

    if bold_title:
        fig.text(
            0.5, 0.865, bold_title, ha="center", va="top",
            fontsize=18, fontweight="bold",
        )
    if suptitle:
        fig.suptitle(
            (
                f"{suptitle}\n"
                r"Highlighted panels: $R^2 > 0.9$ or "
                r"$|\tau| > 0.8$ ($p < 0.005$)"
            ),
            fontsize=11, y=0.84, verticalalignment="top",
        )

    if color_by_study:
        from matplotlib.lines import Line2D
        seen = {study_of_workdir(r.get("workdir", "")) for r in all_rows}
        handles = [
            Line2D(
                [0], [0], marker="o", linestyle="",
                markerfacecolor=STUDY_COLORS[s],
                markeredgecolor="none", markersize=8, label=s,
            )
            for s in STUDY_ORDER if s in seen
        ]
        if handles:
            fig.legend(
                handles=handles,
                loc="upper center",
                ncol=len(handles),
                bbox_to_anchor=(0.5, 0.81),
                frameon=False,
                fontsize=11,
            )

    top_pad = 0.84 if color_by_study else 0.88
    fig.tight_layout(rect=[0, 0, 1, top_pad])
    fig.savefig(
        out_path, dpi=150,
        bbox_inches="tight" if color_by_study else None,
    )
    print(f"\nwrote {out_path}")


def plot_sig2_by_residue(
    rows, x_key, x_label, out_path,
    suptitle=None, split_by_composition=False,
):
    """Subplots split by residue tag (HisND, HisNE, Cys), each showing
    sig2 vs `x_key` for every path classified into that residue.

    Default (`split_by_composition=False`): a single row of subplots,
    points from every composition overlaid (one color each).

    With `split_by_composition=True`: one row per composition (1res /
    4cys / 3cys1his / 2cys2his / 1cys3his / 4his), 3 columns for
    residue tag — each row uses the composition's color so a row of
    plots tells you that composition's behavior across residue types.
    Columns share x and y axes so columns are directly comparable."""
    if not rows:
        print(f"\nno within-residue rows; skipping {out_path}")
        return

    by_tag = defaultdict(list)
    for r in rows:
        by_tag[r["res_tag"]].append(r)
    tag_order = ["HisND", "HisNE", "Cys"]
    tags = [t for t in tag_order if t in by_tag]
    if not tags:
        print(f"\nno HisND/HisNE/Cys paths; skipping {out_path}")
        return

    if split_by_composition:
        compositions = sorted(
            {r["composition"] for r in rows},
            key=lambda c: (
                COMPOSITION_ORDER.index(c) if c in COMPOSITION_ORDER else 99
            ),
        )
        nrows = len(compositions)
        ncols = len(tags)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(4.5 * ncols, 2.8 * nrows),
            squeeze=False,
            sharex="col", sharey="col",
        )
        for i, comp in enumerate(compositions):
            color = COMPOSITION_COLORS.get(comp, "gray")
            for j, tag in enumerate(tags):
                ax = axes[i][j]
                pts = [r for r in by_tag[tag] if r["composition"] == comp]
                if pts:
                    xs = [r[x_key] for r in pts]
                    ys = [r["sig2_tot"] for r in pts]
                    ax.scatter(
                        xs, ys, color=color, alpha=0.12,
                        s=14, edgecolors="none",
                    )
                if i == 0:
                    ax.set_title(tag, fontsize=11)
                if j == 0:
                    ax.set_ylabel(
                        f"{comp}\n" + r"$\sigma^2_{tot}$ (Å$^2$)",
                        fontsize=10,
                    )
                if i == nrows - 1:
                    ax.set_xlabel(x_label, fontsize=10)
                ax.grid(alpha=0.3)
                ax.text(
                    0.97, 0.97, f"n={len(pts)}",
                    transform=ax.transAxes,
                    ha="right", va="top", fontsize=8,
                )
        if suptitle:
            fig.suptitle(suptitle, fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
        else:
            fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        print(f"\nwrote {out_path}")
        return

    # Default layout: single row, all compositions overlaid per subplot.
    n = len(tags)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.5 * ncols, 3.6 * nrows), squeeze=False,
    )
    for ax in axes.flat:
        ax.set_visible(False)

    seen_compositions = set()
    for k, tag in enumerate(tags):
        ax = axes[k // ncols][k % ncols]
        ax.set_visible(True)
        groups = defaultdict(list)
        for r in by_tag[tag]:
            groups[r["composition"]].append(r)
        for comp in COMPOSITION_ORDER:
            if comp not in groups:
                continue
            seen_compositions.add(comp)
            xs = [r[x_key] for r in groups[comp]]
            ys = [r["sig2_tot"] for r in groups[comp]]
            ax.scatter(
                xs, ys,
                color=COMPOSITION_COLORS[comp],
                alpha=0.12, s=14, edgecolors="none",
            )
        ax.set_title(f"{tag}  (n={len(by_tag[tag])})", fontsize=11)
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(r"$\sigma^2_{tot}$ (Å$^2$)", fontsize=10)
        ax.grid(alpha=0.3)

    from matplotlib.lines import Line2D
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="",
            markerfacecolor=COMPOSITION_COLORS[c],
            markeredgecolor="none", markersize=8, label=c,
        )
        for c in COMPOSITION_ORDER if c in seen_compositions
    ]
    if handles:
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, 0.93 if suptitle else 0.99),
            frameon=False,
        )

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
        # Reserve top for suptitle + legend.
        fig.tight_layout(rect=[0, 0, 1, 0.90])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("workdirs", nargs="+", type=Path)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("top_paths_sig2_vs_reff.png"),
    )
    args = ap.parse_args()

    all_rows = []
    by_residue_rows = []
    topn_lt5_rows = []
    all_lt5_rows = []
    for wd in args.workdirs:
        out = process_workdir(wd)
        all_rows.extend(out["rows"])
        by_residue_rows.extend(out["by_residue"])
        topn_lt5_rows.extend(out["short_rows"])
        all_lt5_rows.extend(out["all_short_rows"])

    print_summary(all_rows)

    # All figures land in a single figs/ dir alongside --out's parent.
    base_dir = args.out.parent
    suf = args.out.suffix
    stem = args.out.stem  # e.g. "top_paths_sig2_vs_reff"
    # Strip the trailing axis suffix so we can name reff/angle/dihedral
    # variants consistently as "<prefix>_vs_<axis>_<class>".
    prefix = stem[:-len("_vs_reff")] if stem.endswith("_vs_reff") else stem

    figs_dir = base_dir / "figs"
    figs_dir.mkdir(exist_ok=True)

    out_n4_dih = figs_dir / f"{prefix}_vs_dihedral_nlegs4{suf}"
    out_reff_bycomp = figs_dir / f"{prefix}_vs_reff_bycomp{suf}"
    out_n4_dih_bycomp = figs_dir / f"{prefix}_vs_dihedral_nlegs4_bycomp{suf}"
    out_reff_bycomp_lt5 = (
        figs_dir / f"{prefix}_vs_reff_bycomp_nlegs_lt5{suf}"
    )

    out_byres_reff = figs_dir / f"{stem}_byresidue_reff{suf}"

    # Color convention: reff axis -> REFF_FIT_COLOR, angle/dihedral -> ANGLE_FIT_COLOR.
    n4_rows = [r for r in all_rows if r["nlegs"] == 4]
    if n4_rows:
        plot_sig2_vs_x(
            n4_rows, "dihedral", r"Dihedral Zn$-$A$-$B$-$C (deg)",
            ANGLE_FIT_COLOR, out_n4_dih,
            suptitle=r"$n_{\mathrm{legs}}=4$ (linear) paths",
            bold_title="Debye–Waller factor vs. dihedral angle",
        )

    # Same set, but points colored by which study each workdir belongs to
    # (charge sweep, angle sweep, or family survey).
    plot_sig2_vs_x(
        all_rows, "reff", r"$R_{\mathrm{eff}}$ (Å)",
        REFF_FIT_COLOR, out_reff_bycomp,
        suptitle=f"Top-{TOP_N} scattering paths per structure",
        color_by_study=True,
        bold_title="Debye–Waller factor vs. effective path length",
    )
    if n4_rows:
        plot_sig2_vs_x(
            n4_rows, "dihedral", r"Dihedral Zn$-$A$-$B$-$C (deg)",
            ANGLE_FIT_COLOR, out_n4_dih_bycomp,
            suptitle=r"$n_{\mathrm{legs}}=4$ (linear) paths",
            color_by_study=True,
            bold_title="Debye–Waller factor vs. dihedral angle",
        )

    # Per-nlegs top-N intra-residue seed (TOP_N_NLEGS{2,3,4} caps in
    # process_workdir) decides which path labels get a subplot. We
    # plot every instance of those paths from the full nlegs<5 set so
    # amp stats and the fit see all the data; n=… in the upper-left
    # counts only the seed contributors. The data pool is filtered to
    # intra-residue here too — the bycomp story is about how within-
    # residue chemistry varies, so triangles should be 0.
    intra_lt5 = [r for r in all_lt5_rows if not r.get("cross_residue")]
    plot_sig2_vs_x(
        intra_lt5, "reff", r"$R_{\mathrm{eff}}$ (Å)",
        REFF_FIT_COLOR, out_reff_bycomp_lt5,
        suptitle=(
            r"Top intra-residue paths ($n_{\mathrm{legs}} < 5$; "
            f"{TOP_N_NLEGS2}/{TOP_N_NLEGS3}/{TOP_N_NLEGS4} per "
            r"$n_{\mathrm{legs}} = 2/3/4$)"
        ),
        color_by_study=True,
        seed_rows=topn_lt5_rows,
        bold_title="Debye–Waller factor vs. effective path length",
    )

    # Within-residue plots: every path classified into a single residue
    # (HisND / HisNE / Cys), colored by structure composition.
    # Split into one row per composition so each row tells you that
    # coordination's behavior across residue types.
    plot_sig2_by_residue(
        by_residue_rows, "reff", "reff (Å)", out_byres_reff,
        suptitle=r"all within-residue paths: $\sigma^2$ vs reff",
        split_by_composition=True,
    )


if __name__ == "__main__":
    main()
