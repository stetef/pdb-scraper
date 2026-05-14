#!/usr/bin/env python3
"""
check_multiplicity.py
Usage: python check_multiplicity.py <xyz_file_or_directory>

Reads one XYZ file or all .xyz files in a directory, extracts
rounded_charge from each comment line, computes the lowest
spin multiplicity (singlet/doublet), and updates
MULTIPLICITY=? in-place.
"""

import argparse
import re
from collections import Counter
from pathlib import Path

# Atomic numbers for common bio/inorganic elements
ATOMIC_NUMBERS = {
    'H': 1, 'He': 2,
    'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
    'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Se': 34, 'Br': 35, 'Mo': 42, 'Cd': 48, 'I': 53, 'Hg': 80,
}

def parse_xyz(filename):
    with open(filename) as f:
        lines = f.readlines()
    n_atoms = int(lines[0].strip())
    comment = lines[1].strip()
    elements = []
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        if parts:
            elements.append(parts[0])
    return n_atoms, comment, elements, lines


def extract_rounded_charge(comment):
    match = re.search(
        r"(?:rounded_charge|charge_rounded|charge)\s*[:=]\s*([+-]?\d+(?:\.\d+)?)",
        comment,
        re.IGNORECASE,
    )
    if not match:
        raise ValueError("rounded_charge/charge_rounded not found in comment line")

    charge_value = float(match.group(1))
    rounded_charge = int(round(charge_value))
    if abs(charge_value - rounded_charge) > 1e-6:
        print(
            f"WARNING: rounded_charge value '{charge_value}' is not an integer; "
            f"using {rounded_charge}"
        )
    return rounded_charge


def compute_multiplicity(elements, charge):
    counts = Counter(elements)

    unknown = [el for el in counts if el not in ATOMIC_NUMBERS]
    if unknown:
        print(f"WARNING: Unknown elements (not in table): {unknown}")

    total_electrons_neutral = sum(ATOMIC_NUMBERS.get(el, 0) * n for el, n in counts.items())
    total_electrons = total_electrons_neutral - charge
    multiplicity = 1 if total_electrons % 2 == 0 else 2
    return multiplicity, total_electrons_neutral, total_electrons, counts


def update_comment_multiplicity(comment, multiplicity):
    pattern = re.compile(r"(MULTIPLICITY\s*=\s*)(\?|\d+)", re.IGNORECASE)
    if pattern.search(comment):
        return pattern.sub(rf"\g<1>{multiplicity}", comment, count=1)
    return f"{comment} MULTIPLICITY={multiplicity}"


def process_xyz_file(path):
    n_atoms, comment, elements, lines = parse_xyz(path)
    charge = extract_rounded_charge(comment)
    multiplicity, total_electrons_neutral, total_electrons, counts = compute_multiplicity(elements, charge)

    updated_comment = update_comment_multiplicity(comment, multiplicity)
    changed = updated_comment != comment

    if changed:
        lines[1] = updated_comment + "\n"
        with open(path, "w") as f:
            f.writelines(lines)

    print(f"\nFile:              {path}")
    print(f"Comment line:      {comment or '(empty)'}")
    print(f"Updated comment:   {updated_comment}")
    print(f"Total atoms:       {n_atoms}")
    print(f"\nAtom counts:")
    for el, n in sorted(counts.items()):
        print(f"  {el:4s}: {n}")
    print(f"\nCharge (rounded_charge):   {charge:+d}")
    print(f"Total electrons (neutral): {total_electrons_neutral}")
    print(f"Total electrons:           {total_electrons}")
    print(f"Lowest multiplicity:       {multiplicity}")

    return changed


def collect_xyz_files(path):
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".xyz")
    return []

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Update MULTIPLICITY in XYZ comment lines using rounded_charge from "
            "the same comment line."
        )
    )
    parser.add_argument(
        "path",
        help="Path to one .xyz file or a directory containing .xyz files",
    )
    args = parser.parse_args()

    target = Path(args.path)
    xyz_files = collect_xyz_files(target)

    if not xyz_files:
        print(f"No .xyz files found at: {target}")
        raise SystemExit(1)

    changed = 0
    failed = 0
    for xyz_file in xyz_files:
        try:
            if process_xyz_file(xyz_file):
                changed += 1
        except Exception as exc:
            failed += 1
            print(f"\nERROR processing {xyz_file}: {exc}")

    print("\nSummary")
    print(f"  Files found:     {len(xyz_files)}")
    print(f"  Files updated:   {changed}")
    print(f"  Files failed:    {failed}")
    if failed:
        raise SystemExit(2)

if __name__ == "__main__":
    main()