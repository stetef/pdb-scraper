#!/usr/bin/env python3
"""
check_multiplicity.py
Usage: python check_multiplicity.py structure.xyz [charge]

Reads an xyz file, counts atoms and electrons, and reports the
expected spin multiplicity for a given charge (default: 0).
Assumes lowest spin state (singlet or doublet).
"""

import sys
from collections import Counter

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
    return n_atoms, comment, elements

def main():
    if len(sys.argv) < 2:
        print("Usage: python xyz_multiplicity.py structure.xyz [charge]")
        sys.exit(1)

    filename = sys.argv[1]
    charge = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    n_atoms, comment, elements = parse_xyz(filename)
    counts = Counter(elements)

    unknown = [el for el in counts if el not in ATOMIC_NUMBERS]
    if unknown:
        print(f"WARNING: Unknown elements (not in table): {unknown}")

    total_electrons_neutral = sum(ATOMIC_NUMBERS.get(el, 0) * n for el, n in counts.items())
    total_electrons = total_electrons_neutral - charge

    remainder = total_electrons % 2
    if remainder == 0:
        multiplicity = 1
        spin_label = "singlet (closed shell)"
    else:
        multiplicity = 2
        spin_label = "doublet (open shell radical)"

    print(f"\nFile:              {filename}")
    print(f"Comment line:      {comment or '(empty)'}")
    print(f"Total atoms:       {n_atoms}")
    print(f"\nAtom counts:")
    for el, n in sorted(counts.items()):
        print(f"  {el:4s}: {n}")
    print(f"\nTotal electrons (neutral): {total_electrons_neutral}")
    print(f"Charge:                    {charge:+d}")
    print(f"Total electrons:           {total_electrons}")
    print(f"\nLowest multiplicity:       {multiplicity}  ({spin_label})")
    print(f"\nFor ORCA input: *xyzfile {charge} {multiplicity} your_file.xyz")

    # Also show what other charges would give a singlet/doublet
    print(f"\nNote: singlet (mult=1) requires even electron count.")
    for c in range(-3, 4):
        ne = total_electrons_neutral - c
        if ne % 2 == 0:
            print(f"  Charge {c:+d} -> {ne} electrons -> singlet")

if __name__ == "__main__":
    main()