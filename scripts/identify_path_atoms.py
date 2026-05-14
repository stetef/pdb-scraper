#!/usr/bin/env python3
"""Map FEFF paths to the atoms that compose them.

For a FEFF run directory containing ``feff.inp``, ``paths.dat``, and
``xmu.dat``, this script identifies which atoms from the ATOMS block of
``feff.inp`` participate in each path. Matching is done by xyz coordinates:
``paths.dat`` lists each leg's xyz, and those coordinates correspond directly
to rows in the ``feff.inp`` ATOMS block. The POTENTIALS block of ``feff.inp``
is used to translate the ``ipot`` index into an element symbol.

Output is a TSV with one row per path summarizing the legs as
``symbol#atom_idx`` tokens (atom_idx = 1-based row number in the ATOMS block,
with the absorber being index 0).
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path


COORD_TOL = 1e-3  # Angstrom — paths.dat coords are 6 decimals, ATOMS 10 decimals


@dataclass(frozen=True)
class Potential:
    ipot: int
    z: int
    symbol: str


@dataclass(frozen=True)
class FeffAtom:
    index: int  # 0 for absorber, 1..N for ATOMS block rows in order
    x: float
    y: float
    z: float
    ipot: int
    distance: float


@dataclass
class PathLeg:
    x: float
    y: float
    z: float
    ipot: int
    label: str
    rleg: float


@dataclass
class FeffPath:
    index: int
    nleg: int
    degeneracy: float
    reff: float
    legs: list[PathLeg]


def parse_feff_inp(path: Path) -> tuple[dict[int, Potential], list[FeffAtom]]:
    """Return (potentials_by_ipot, atoms_in_order)."""
    potentials: dict[int, Potential] = {}
    atoms: list[FeffAtom] = []
    section: str | None = None
    atom_counter = 0  # 0 = absorber

    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        upper = stripped.upper()
        if upper.startswith("ATOMS"):
            section = "ATOMS"
            continue
        if upper.startswith("POTENTIALS"):
            section = "POTENTIALS"
            continue
        # Any other all-caps directive ends the current block.
        first_token = stripped.split()[0]
        if first_token.isupper() and not first_token.lstrip("-").replace(".", "").isdigit():
            section = None
            continue

        if section == "POTENTIALS":
            parts = stripped.split()
            if len(parts) < 3:
                continue
            try:
                ipot = int(parts[0])
                z = int(parts[1])
            except ValueError:
                continue
            symbol = parts[2]
            potentials[ipot] = Potential(ipot=ipot, z=z, symbol=symbol)
        elif section == "ATOMS":
            parts = stripped.split()
            if len(parts) < 5:
                continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                ipot = int(parts[3])
                distance = float(parts[4])
            except ValueError:
                continue
            atoms.append(
                FeffAtom(
                    index=atom_counter,
                    x=x,
                    y=y,
                    z=z,
                    ipot=ipot,
                    distance=distance,
                )
            )
            atom_counter += 1

    if not potentials:
        raise ValueError(f"No POTENTIALS block found in {path}")
    if not atoms:
        raise ValueError(f"No ATOMS block found in {path}")
    return potentials, atoms


def parse_paths_dat(path: Path) -> list[FeffPath]:
    paths: list[FeffPath] = []
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if "index, nleg, degeneracy" in line:
            parts = line.split()
            try:
                idx = int(parts[0])
                nleg = int(parts[1])
                deg = float(parts[2])
                reff = float(parts[-1])
            except (ValueError, IndexError):
                i += 1
                continue
            # Next line is the column header; the nleg lines after are legs.
            legs: list[PathLeg] = []
            j = i + 2
            while j < len(lines) and len(legs) < nleg:
                leg_parts = lines[j].split()
                if len(leg_parts) >= 5:
                    try:
                        lx = float(leg_parts[0])
                        ly = float(leg_parts[1])
                        lz = float(leg_parts[2])
                        lipot = int(leg_parts[3])
                    except ValueError:
                        j += 1
                        continue
                    # label is single-quoted, e.g.  'S     '
                    line_text = lines[j]
                    label = ""
                    if "'" in line_text:
                        first = line_text.find("'")
                        second = line_text.find("'", first + 1)
                        if second > first:
                            label = line_text[first + 1 : second].strip()
                    # rleg is the first numeric token after the label
                    after = line_text[second + 1 :].split() if "'" in line_text else []
                    rleg = float(after[0]) if after else 0.0
                    legs.append(
                        PathLeg(x=lx, y=ly, z=lz, ipot=lipot, label=label, rleg=rleg)
                    )
                j += 1
            paths.append(
                FeffPath(index=idx, nleg=nleg, degeneracy=deg, reff=reff, legs=legs)
            )
            i = j
            continue
        i += 1
    return paths


def parse_xmu_path_table(path: Path) -> dict[int, dict[str, float]]:
    """Extract the per-path summary table embedded in xmu.dat header.

    The table looks like ``# file sig2_tot cw_amp_ratio deg nlegs reff inp_sig2``.
    """
    rows: dict[int, dict[str, float]] = {}
    in_table = False
    for raw in path.read_text().splitlines():
        if not raw.startswith("#"):
            continue
        text = raw.lstrip("#").strip()
        if "file" in text and "reff" in text and "sig2" in text:
            in_table = True
            continue
        if not in_table:
            continue
        parts = text.split()
        if len(parts) < 6:
            # Reached the end of the path summary table.
            in_table = False
            continue
        try:
            file_idx = int(parts[0])
            sig2_tot = float(parts[1])
            amp = float(parts[2])
            deg = float(parts[3])
            nlegs = int(parts[4])
            reff = float(parts[5])
        except ValueError:
            continue
        inp_sig2 = float(parts[6]) if len(parts) >= 7 else 0.0
        rows[file_idx] = {
            "sig2_tot": sig2_tot,
            "amp_ratio": amp,
            "deg": deg,
            "nlegs": nlegs,
            "reff": reff,
            "inp_sig2": inp_sig2,
        }
    return rows


def find_atom_by_xyz(
    atoms: list[FeffAtom], x: float, y: float, z: float, tol: float = COORD_TOL
) -> FeffAtom | None:
    for atom in atoms:
        if (
            abs(atom.x - x) <= tol
            and abs(atom.y - y) <= tol
            and abs(atom.z - z) <= tol
        ):
            return atom
    return None


def describe_leg(
    leg: PathLeg,
    atoms: list[FeffAtom],
    potentials: dict[int, Potential],
    tol: float,
) -> str:
    if leg.ipot == 0:
        return "Zn#0"
    atom = find_atom_by_xyz(atoms, leg.x, leg.y, leg.z, tol=tol)
    symbol = potentials.get(leg.ipot, Potential(leg.ipot, 0, leg.label or "?")).symbol
    if atom is None:
        return f"{symbol}#?({leg.x:.3f},{leg.y:.3f},{leg.z:.3f})"
    return f"{symbol}#{atom.index}"


def build_rows(
    feff_paths: list[FeffPath],
    atoms: list[FeffAtom],
    potentials: dict[int, Potential],
    xmu_rows: dict[int, dict[str, float]],
    tol: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fp in feff_paths:
        atom_tokens = [describe_leg(leg, atoms, potentials, tol) for leg in fp.legs]
        # Unique scattering atoms (drop the absorber).
        unique_scatterers = []
        seen = set()
        for tok in atom_tokens:
            if tok == "Zn#0":
                continue
            if tok in seen:
                continue
            seen.add(tok)
            unique_scatterers.append(tok)
        xmu = xmu_rows.get(fp.index, {})
        rows.append(
            {
                "path": fp.index,
                "nleg": fp.nleg,
                "deg": fp.degeneracy,
                "reff": fp.reff,
                "amp_ratio": xmu.get("amp_ratio", ""),
                "sig2_tot": xmu.get("sig2_tot", ""),
                "scatterers": ",".join(unique_scatterers),
                "leg_sequence": " -> ".join(atom_tokens + [atom_tokens[0]])
                if atom_tokens
                else "",
            }
        )
    return rows


def write_output(rows: list[dict[str, object]], out_path: Path) -> None:
    fieldnames = [
        "path",
        "nleg",
        "deg",
        "reff",
        "amp_ratio",
        "sig2_tot",
        "scatterers",
        "leg_sequence",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_atoms_legend(
    atoms: list[FeffAtom],
    potentials: dict[int, Potential],
    out_path: Path,
) -> None:
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["atom_idx", "symbol", "ipot", "x", "y", "z", "distance"])
        for atom in atoms:
            symbol = potentials.get(atom.ipot, Potential(atom.ipot, 0, "?")).symbol
            writer.writerow(
                [
                    atom.index,
                    symbol,
                    atom.ipot,
                    f"{atom.x:.6f}",
                    f"{atom.y:.6f}",
                    f"{atom.z:.6f}",
                    f"{atom.distance:.6f}",
                ]
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "feff_dir",
        type=Path,
        help="Directory containing feff.inp, paths.dat, and xmu.dat.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output TSV path (default: <feff_dir>/path_atoms.tsv).",
    )
    parser.add_argument(
        "--legend",
        type=Path,
        default=None,
        help=(
            "Optional output TSV mapping atom_idx -> (symbol, ipot, xyz, distance) "
            "for the ATOMS block (default: <feff_dir>/atoms_legend.tsv)."
        ),
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=COORD_TOL,
        help=f"xyz match tolerance in Angstrom (default: {COORD_TOL}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Print only the first N paths to stdout (file output is unaffected).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    feff_dir: Path = args.feff_dir
    feff_inp = feff_dir / "feff.inp"
    paths_dat = feff_dir / "paths.dat"
    xmu_dat = feff_dir / "xmu.dat"
    for required in (feff_inp, paths_dat):
        if not required.exists():
            raise SystemExit(f"Missing required file: {required}")

    potentials, atoms = parse_feff_inp(feff_inp)
    feff_paths = parse_paths_dat(paths_dat)
    xmu_rows = parse_xmu_path_table(xmu_dat) if xmu_dat.exists() else {}

    rows = build_rows(feff_paths, atoms, potentials, xmu_rows, tol=args.tol)

    out_path = args.out or (feff_dir / "path_atoms.tsv")
    write_output(rows, out_path)

    legend_path = args.legend or (feff_dir / "atoms_legend.tsv")
    write_atoms_legend(atoms, potentials, legend_path)

    print(f"Wrote {out_path} ({len(rows)} paths)")
    print(f"Wrote {legend_path} ({len(atoms) + 1} atoms incl. absorber)")

    preview = rows if args.limit is None else rows[: args.limit]
    if preview:
        print()
        print(
            f"{'path':>5}  {'nleg':>4}  {'deg':>5}  {'reff':>7}  {'amp':>6}  scatterers / leg sequence"
        )
        for row in preview:
            amp = row["amp_ratio"]
            amp_s = f"{amp:6.2f}" if isinstance(amp, float) else f"{'':>6}"
            print(
                f"{row['path']:>5}  {row['nleg']:>4}  {row['deg']:>5.2f}  "
                f"{row['reff']:>7.4f}  {amp_s}  {row['scatterers']}  |  {row['leg_sequence']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
