#!/usr/bin/env python3
"""Generate extended-environment XYZ files for the volume-study extremes.

For every xyz file referenced in `volume_extremes_<label>.csv`:

  1. Parse the comment lines to find the Zn origin atom (chain, resseq, atom name).
  2. Open the corresponding validated PDB file.
  3. Translate every ATOM/HETATM record so the Zn is at the origin.
  4. Keep atoms within `--cutoff` Å (default 10).
  5. Write `<xyz_stem>-extended.xyz` next to the original XYZ, preserving the
     same comment-style metadata (RES, CHAIN, RESSEQ, ATOM, REC).

Usage
-----
    uv run python scripts/build_extended_xyz.py \\
        --csv data/large-cys-his-datasets/4cys-large/figures/volume_extremes_4cys-large-dataset.csv \\
        --pdb-dir data/large-cys-his-datasets/4cys-large/results/validated_structures
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

ATOM_RE = re.compile(r"\bATOM=(\w+)\b")
RES_RE = re.compile(r"\bRES=(\w+)\b")
CHAIN_RE = re.compile(r"\bCHAIN=(\S+)\b")
RESSEQ_RE = re.compile(r"\bRESSEQ=(\S+)\b")
PDB_RE = re.compile(r"\bPDB=([A-Za-z0-9]+)\b")
RESSEQ_SPLIT = re.compile(r"^(-?\d+)([A-Za-z]?)$")


@dataclass
class OriginRef:
    atom_name: str
    resname: str
    chain: str
    resseq: str
    icode: str


@dataclass
class PdbAtom:
    rec: str
    serial: int
    atom_name: str
    altloc: str
    resname: str
    chain: str
    resseq: str
    icode: str
    x: float
    y: float
    z: float
    element: str


def parse_origin_ref(xyz_text: str) -> OriginRef | None:
    """Find the atom positioned at (0,0,0) and pull its residue identity."""
    for line in xyz_text.splitlines()[2:]:
        if "#" not in line:
            continue
        coord_part, comment = line.split("#", 1)
        parts = coord_part.split()
        if len(parts) < 4:
            continue
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue
        if abs(x) > 1e-4 or abs(y) > 1e-4 or abs(z) > 1e-4:
            continue
        atom_m = ATOM_RE.search(comment)
        res_m = RES_RE.search(comment)
        chain_m = CHAIN_RE.search(comment)
        resseq_m = RESSEQ_RE.search(comment)
        if not (atom_m and res_m and chain_m and resseq_m):
            continue
        rs = RESSEQ_SPLIT.match(resseq_m.group(1))
        if rs is None:
            continue
        return OriginRef(
            atom_name=atom_m.group(1).upper(),
            resname=res_m.group(1).upper(),
            chain=chain_m.group(1),
            resseq=rs.group(1),
            icode=rs.group(2),
        )
    return None


def parse_pdb_id_from_xyz(xyz_text: str) -> str | None:
    lines = xyz_text.splitlines()
    if len(lines) < 2:
        return None
    m = PDB_RE.search(lines[1])
    return m.group(1) if m else None


def parse_pdb(pdb_path: Path) -> list[PdbAtom]:
    atoms: list[PdbAtom] = []
    for line in pdb_path.read_text(encoding="utf-8", errors="replace").splitlines():
        rec = line[0:6].strip().upper() if len(line) >= 6 else ""
        if rec not in {"ATOM", "HETATM"}:
            continue
        if len(line) < 54:
            continue
        try:
            atoms.append(
                PdbAtom(
                    rec=rec,
                    serial=int((line[6:11].strip() or "0")),
                    atom_name=line[12:16].strip(),
                    altloc=line[16:17].strip(),
                    resname=line[17:20].strip(),
                    chain=line[21:22].strip(),
                    resseq=line[22:26].strip(),
                    icode=line[26:27].strip(),
                    x=float(line[30:38]),
                    y=float(line[38:46]),
                    z=float(line[46:54]),
                    element=line[76:78].strip().upper() if len(line) >= 78 else "",
                )
            )
        except ValueError:
            continue
    return atoms


def infer_element(atom_name: str) -> str:
    name = atom_name.strip()
    if not name:
        return ""
    two_letter = {"FE", "ZN", "CL", "MG", "MN", "CU", "NI", "CO", "CA", "NA", "BR", "SE", "CD", "HG"}
    if name[:2].upper() in two_letter:
        return name[:2].upper()
    for ch in name:
        if ch.isalpha():
            return ch.upper()
    return ""


def find_origin_atom(atoms: list[PdbAtom], ref: OriginRef) -> PdbAtom | None:
    for a in atoms:
        if a.altloc and a.altloc.upper() not in {"", "A"}:
            continue
        if a.atom_name.upper() != ref.atom_name:
            continue
        if a.resname.upper() != ref.resname:
            continue
        if a.chain != ref.chain:
            continue
        if a.resseq != ref.resseq:
            continue
        if a.icode != ref.icode:
            continue
        return a
    return None


def write_extended_xyz(
    out_path: Path,
    *,
    pdb_id: str,
    cutoff: float,
    origin_atom: PdbAtom,
    atoms: list[PdbAtom],
) -> int:
    cutoff2 = cutoff * cutoff
    selected: list[tuple[str, float, float, float, PdbAtom]] = []
    for a in atoms:
        if a.altloc and a.altloc.upper() not in {"", "A"}:
            continue
        elem = a.element if a.element else infer_element(a.atom_name)
        if not elem:
            continue
        dx = a.x - origin_atom.x
        dy = a.y - origin_atom.y
        dz = a.z - origin_atom.z
        d2 = dx * dx + dy * dy + dz * dz
        if d2 > cutoff2:
            continue
        selected.append((elem, dx, dy, dz, a))

    selected.sort(key=lambda t: t[1] * t[1] + t[2] * t[2] + t[3] * t[3])

    header = (
        f"PDB={pdb_id} EXTENDED CUTOFF_A={cutoff:.2f} "
        f"ORIGIN_ATOM={origin_atom.atom_name} ORIGIN_RES={origin_atom.resname} "
        f"ORIGIN_CHAIN={origin_atom.chain} ORIGIN_RESSEQ={origin_atom.resseq}{origin_atom.icode}"
    )
    lines = [str(len(selected)), header]
    for elem, dx, dy, dz, a in selected:
        meta = (
            f"# RES={a.resname} CHAIN={a.chain} RESSEQ={a.resseq}{a.icode} "
            f"ATOM={a.atom_name} REC={a.rec}"
        )
        lines.append(f"{elem:<3s} {dx:11.6f} {dy:11.6f} {dz:11.6f}  {meta}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(selected)


def resolve_pdb_path(pdb_dir: Path, pdb_id: str) -> Path | None:
    for variant in (pdb_id, pdb_id.lower(), pdb_id.upper()):
        candidate = pdb_dir / f"{variant}.pdb"
        if candidate.exists():
            return candidate
    return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, type=Path,
                   help="volume_extremes_<label>.csv listing the xyz_path column.")
    p.add_argument("--pdb-dir", required=True, type=Path,
                   help="Directory containing validated <pdb_id>.pdb files.")
    p.add_argument("--cutoff", type=float, default=10.0,
                   help="Distance cutoff in angstroms (default: 10).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing -extended.xyz files.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")
    if not args.pdb_dir.is_dir():
        raise SystemExit(f"PDB dir not found: {args.pdb_dir}")

    with args.csv.open() as f:
        rows = list(csv.DictReader(f))

    n_done = n_skipped = n_failed = 0
    for r in rows:
        xyz_path = Path(r["xyz_path"]).expanduser()
        out_path = xyz_path.with_name(f"{xyz_path.stem}-extended{xyz_path.suffix}")

        if not xyz_path.exists():
            print(f"  SKIP missing xyz: {xyz_path}")
            n_failed += 1
            continue
        if out_path.exists() and not args.force:
            n_skipped += 1
            continue

        xyz_text = xyz_path.read_text(encoding="utf-8")
        pdb_id = parse_pdb_id_from_xyz(xyz_text) or xyz_path.stem[:4]
        pdb_path = resolve_pdb_path(args.pdb_dir, pdb_id)
        if pdb_path is None:
            print(f"  FAIL no PDB for {xyz_path.name} (looked for {pdb_id}.pdb in {args.pdb_dir})")
            n_failed += 1
            continue

        ref = parse_origin_ref(xyz_text)
        if ref is None:
            print(f"  FAIL no origin in {xyz_path.name}")
            n_failed += 1
            continue

        atoms = parse_pdb(pdb_path)
        origin = find_origin_atom(atoms, ref)
        if origin is None:
            print(f"  FAIL origin atom not found in {pdb_path.name} "
                  f"({ref.resname} {ref.chain} {ref.resseq}{ref.icode} {ref.atom_name})")
            n_failed += 1
            continue

        n = write_extended_xyz(
            out_path, pdb_id=pdb_id, cutoff=args.cutoff, origin_atom=origin, atoms=atoms,
        )
        print(f"  ✓ {out_path.name}: {n} atoms ≤ {args.cutoff} Å")
        n_done += 1

    print(f"\nWrote: {n_done}    Skipped (existing): {n_skipped}    Failed: {n_failed}")
    if n_skipped:
        print("Use --force to overwrite existing -extended.xyz files.")


if __name__ == "__main__":
    main()
