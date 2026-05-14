#!/usr/bin/env python3
"""Extract per-bond stretch force constants from an ORCA .hess file.

For each Zn-coordinated workdir, parses the cartesian Hessian, projects
it onto each intra-residue sidechain bond direction, and emits one row
per bond to a CSV. Sidechain bonds are limited to the chain that runs
from the Zn-bonded heavy atom up to the residue's CA -- backbone past
CA and inter-residue contacts are excluded.

Bond inventory (per residue type):
    Cys    -> Zn-SG, SG-CB, CB-CA                            (3 bonds)
    HisND  -> Zn-ND1, ND1-CG, ND1-CE1, CE1-NE2, NE2-CD2,
              CD2-CG, CG-CB, CB-CA                           (8 bonds)
    HisNE  -> Zn-NE2, NE2-CE1, NE2-CD2, CE1-ND1, ND1-CG,
              CD2-CG, CG-CB, CB-CA                           (8 bonds)

Why this exists: dmdw.out emits one effective FC per scattering path
and every dmdw path starts at the absorber. So Zn-X first-shell
stiffnesses are recoverable cleanly, but stiffnesses for non-Zn bonds
(SG-CB, CB-CA, ring bonds) are not -- the closest dmdw proxy is a
3-atom path through Zn whose FC mixes legs. The cartesian Hessian
carries every pair-coupling, so projecting it onto bond unit vectors
recovers true per-bond FCs.

Usage:
    python hess_bond_fcs.py <workdir> [<workdir> ...] [--out fcs.csv]
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from top_paths_sig2_vs_reff import (  # noqa: E402
    align_xyz_to_feff, build_adjacency, distance, name_atoms_and_residues,
    parse_feff_atoms, parse_xyz, structure_composition, _find_xyz,
)
from top_paths_sig2_vs_fc import parse_dmdw_paths  # noqa: E402

# 1 Hartree/Bohr^2 in N/m. Hartree = 4.359744722e-18 J;
# Bohr = 5.29177210903e-11 m. Result ~= 1556.8930.
HARTREE_PER_BOHR2_TO_NM = 4.359744722207185e-18 / (5.29177210903e-11) ** 2
BOHR_TO_ANG = 0.529177210903

_CYS_BONDS = [
    ("Zn", "SG"), ("SG", "CB"), ("CB", "CA"),
]
_HIS_ND_BONDS = [
    ("Zn", "ND1"),
    ("ND1", "CG"), ("ND1", "CE1"),
    ("CE1", "NE2"), ("NE2", "CD2"), ("CD2", "CG"),
    ("CG", "CB"), ("CB", "CA"),
]
_HIS_NE_BONDS = [
    ("Zn", "NE2"),
    ("NE2", "CE1"), ("NE2", "CD2"),
    ("CE1", "ND1"), ("ND1", "CG"), ("CD2", "CG"),
    ("CG", "CB"), ("CB", "CA"),
]
_BONDS_BY_TAG = {
    "Cys": _CYS_BONDS,
    "HisND": _HIS_ND_BONDS,
    "HisNE": _HIS_NE_BONDS,
}


def parse_orca_hess(path):
    """Parse an ORCA .hess. Returns {atoms, hessian}.
    atoms: list of {element, mass, x, y, z} with coords in Bohr.
    hessian: numpy array (3N, 3N) in Hartree/Bohr^2."""
    text = Path(path).read_text()

    m = re.search(
        r"^\$atoms\s*\n(\d+)\n(.*?)(?=\n\$|\Z)",
        text, re.DOTALL | re.MULTILINE,
    )
    if not m:
        raise ValueError(f"no $atoms block in {path}")
    n_atoms = int(m.group(1))
    atom_lines = [ln for ln in m.group(2).splitlines() if ln.strip()][:n_atoms]
    if len(atom_lines) < n_atoms:
        raise ValueError(
            f"$atoms truncated in {path}: expected {n_atoms} got "
            f"{len(atom_lines)}"
        )
    atoms = []
    for ln in atom_lines:
        parts = ln.split()
        atoms.append({
            "element": parts[0].capitalize(),
            "mass": float(parts[1]),
            "x": float(parts[2]),
            "y": float(parts[3]),
            "z": float(parts[4]),
        })

    m = re.search(
        r"^\$hessian\s*\n(\d+)\n(.*?)(?=\n\$|\Z)",
        text, re.DOTALL | re.MULTILINE,
    )
    if not m:
        raise ValueError(f"no $hessian block in {path}")
    dim = int(m.group(1))
    if dim != 3 * n_atoms:
        raise ValueError(
            f"hessian dim {dim} != 3*natoms {3 * n_atoms} in {path}"
        )
    H = np.zeros((dim, dim))

    body = m.group(2).splitlines()
    i = 0
    while i < len(body):
        line = body[i].strip()
        if not line:
            i += 1
            continue
        # ORCA's banded format: header is a row of column indices, then
        # `dim` data rows of (row_label, val_for_each_col).
        try:
            cols = [int(c) for c in line.split()]
        except ValueError:
            i += 1
            continue
        i += 1
        for _ in range(dim):
            if i >= len(body):
                break
            parts = body[i].split()
            i += 1
            if not parts:
                continue
            row = int(parts[0])
            for k, c in enumerate(cols):
                H[row, c] = float(parts[1 + k])

    return {"atoms": atoms, "hessian": H}


def hess_to_feff_index(hess_atoms_centered, feff_atoms, tol=0.05):
    """Per-feff-atom matching index into the .hess atom list (Å, both
    Zn-centered). Greedy nearest-element-match within tol; returns None
    for unmatched atoms."""
    used = set()
    out = []
    for fa in feff_atoms:
        best_i, best_d = None, float("inf")
        for hi, ha in enumerate(hess_atoms_centered):
            if hi in used or ha["element"] != fa["element"]:
                continue
            dx = ha["x"] - fa["x"]
            dy = ha["y"] - fa["y"]
            dz = ha["z"] - fa["z"]
            d = (dx * dx + dy * dy + dz * dz) ** 0.5
            if d < best_d:
                best_d, best_i = d, hi
        if best_i is None or best_d > tol:
            out.append(None)
        else:
            out.append(best_i)
            used.add(best_i)
    return out


def bond_stretch_fc(H, hess_coords, hi, hj):
    """Wilson B-matrix internal-coordinate FC for the i-j bond stretch:

        k = (1/4) u^T (Hii - Hij - Hji + Hjj) u

    The 1/4 comes from |B|^2 = 2 for the bond-stretch B-vector
    B = (-u, +u): chain rule gives B H B^T = k_q (B B^T)^2 = 4 k_q.
    Sanity: for a diatomic where Hii = Hjj = k uu^T and
    Hij = -k uu^T, the bracket evaluates to 4k, so k_q = k.

    Units pass through: Hartree/Bohr^2 in -> Hartree/Bohr^2 out. Caller
    multiplies by HARTREE_PER_BOHR2_TO_NM once for the whole hessian."""
    diff = hess_coords[hj] - hess_coords[hi]
    norm = float(np.linalg.norm(diff))
    if norm == 0:
        raise ValueError(f"zero-length bond between hess indices {hi}, {hj}")
    u = diff / norm
    Hii = H[3 * hi:3 * hi + 3, 3 * hi:3 * hi + 3]
    Hij = H[3 * hi:3 * hi + 3, 3 * hj:3 * hj + 3]
    Hji = H[3 * hj:3 * hj + 3, 3 * hi:3 * hi + 3]
    Hjj = H[3 * hj:3 * hj + 3, 3 * hj:3 * hj + 3]
    K = Hii - Hij - Hji + Hjj
    return float(u @ K @ u) / 4.0


def _split_name(s):
    """'SG(Cys)' -> ('SG', 'Cys'); 'Zn' -> ('Zn', None); None -> (None, None)."""
    if s is None:
        return None, None
    if "(" in s:
        i = s.index("(")
        return s[:i], s[i + 1:s.index(")")]
    return s, None


def _residue_atom_index(residue, base, names, zn_idx):
    """Find the feff index of the atom in `residue` whose base name
    matches `base`. 'Zn' returns the absorber index. Returns None if the
    atom wasn't named (e.g. CA missing because the residue is truncated
    before the backbone)."""
    if base == "Zn":
        return zn_idx
    for i in residue["atoms"]:
        b, _ = _split_name(names[i])
        if b == base:
            return i
    return None


def process_workdir(wd, writer, sanity_check=True):
    wd = Path(wd)
    feff_inp = wd / "feff.inp"
    if not feff_inp.is_file():
        return 0, f"# {wd.name}: missing feff.inp -- skipped"
    hess_path = next(iter(wd.glob("*.hess")), None)
    if hess_path is None:
        return 0, f"# {wd.name}: no .hess -- skipped"

    feff_atoms = parse_feff_atoms(feff_inp)
    xyz_atoms = parse_xyz(_find_xyz(wd))
    align_xyz_to_feff(feff_atoms, xyz_atoms)
    adj = build_adjacency(feff_atoms)
    names, residues = name_atoms_and_residues(feff_atoms, adj)
    composition = structure_composition(residues)
    zn_idx = next(
        i for i, a in enumerate(feff_atoms) if a["element"] == "Zn"
    )

    hess = parse_orca_hess(hess_path)
    hess_zn = next(
        (a for a in hess["atoms"] if a["element"] == "Zn"), None
    )
    if hess_zn is None:
        return 0, f"# {wd.name}: no Zn in {hess_path.name} -- skipped"

    # Center hess atoms on their Zn and convert Bohr -> Å so they're in
    # the same frame feff_atoms uses (Zn at origin, Å).
    hess_atoms_aligned = [
        {
            "element": a["element"],
            "x": (a["x"] - hess_zn["x"]) * BOHR_TO_ANG,
            "y": (a["y"] - hess_zn["y"]) * BOHR_TO_ANG,
            "z": (a["z"] - hess_zn["z"]) * BOHR_TO_ANG,
        }
        for a in hess["atoms"]
    ]
    hess_coords_ang = np.array(
        [[a["x"], a["y"], a["z"]] for a in hess_atoms_aligned]
    )

    feff_to_hess = hess_to_feff_index(hess_atoms_aligned, feff_atoms)
    # FEFF inputs for tiny clusters (e.g. 1-residue caps) include padding
    # atoms placed far from origin to satisfy FEFF's minimum-atom
    # requirement; align_xyz_to_feff marks them with xyz_index=None.
    # Only require the real (xyz-matched) feff atoms to map into the hess.
    n_unmapped = sum(
        1
        for fa, hi in zip(feff_atoms, feff_to_hess)
        if hi is None and fa.get("xyz_index") is not None
    )
    if n_unmapped:
        return 0, (
            f"# {wd.name}: {n_unmapped} feff atoms had no .hess match "
            f"within tol -- skipped"
        )

    H_nm = hess["hessian"] * HARTREE_PER_BOHR2_TO_NM

    fc_paths = None
    if sanity_check and (wd / "dmdw.out").is_file():
        fc_paths = parse_dmdw_paths(wd / "dmdw.out")
    sanity_lines = []

    n_bonds = 0
    for residue in residues:
        atom_tag = residue.get("atom_tag")
        bonds = _BONDS_BY_TAG.get(atom_tag)
        if not bonds:
            continue
        for base_a, base_b in bonds:
            ia = _residue_atom_index(residue, base_a, names, zn_idx)
            ib = _residue_atom_index(residue, base_b, names, zn_idx)
            if ia is None or ib is None:
                continue
            hi = feff_to_hess[ia]
            hj = feff_to_hess[ib]
            if hi is None or hj is None:
                continue
            fc = bond_stretch_fc(H_nm, hess_coords_ang, hi, hj)
            bond_length = distance(feff_atoms[ia], feff_atoms[ib])
            bond_label = f"{base_a}-{base_b}"
            writer.writerow([
                wd.name, composition, residue["tag"], atom_tag,
                bond_label, f"{bond_length:.6f}", f"{fc:.4f}",
            ])
            n_bonds += 1

            if (
                fc_paths is not None
                and base_a == "Zn"
                and ia == zn_idx
                and ib != zn_idx
            ):
                key = (1, ib + 1)
                info = fc_paths.get(key)
                if info is not None and info["fc_n2"] is not None:
                    sanity_lines.append(
                        f"#   {residue['tag']} {bond_label}: "
                        f"hess={fc:.1f} N/m  dmdw(n=-2)={info['fc_n2']:.1f} N/m"
                    )

    status = f"# {wd.name} [{composition}]: {n_bonds} bonds"
    if sanity_lines:
        status += "\n" + "\n".join(sanity_lines)
    return n_bonds, status


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("workdirs", nargs="+", type=Path)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("hess_bond_fcs.csv"),
    )
    args = ap.parse_args()

    workdirs = [wd for wd in args.workdirs if wd.is_dir()]
    runnable = [wd for wd in workdirs if next(iter(wd.glob("*.hess")), None)]
    skipped = [wd for wd in workdirs if not next(iter(wd.glob("*.hess")), None)]
    if skipped:
        print(f"# skipping {len(skipped)} dir(s) without .hess:")
        for wd in skipped:
            print(f"#   {wd.name}")
    if not runnable:
        raise SystemExit("no workdirs with .hess among the given paths")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "workdir", "composition", "residue_tag", "atom_tag",
            "bond_label", "bond_length", "fc_n_per_m",
        ])
        total = 0
        for wd in runnable:
            n, status = process_workdir(wd, w)
            print(status)
            total += n
    print(f"\nwrote {args.out} ({total} bond rows)")


if __name__ == "__main__":
    main()
