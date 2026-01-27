#!/usr/bin/env env python
"""
Add hydrogens to XYZ files using Open Babel.

Usage:
    python add_hydrogens.py <path>
    
Where <path> can be either:
    - A single XYZ file
    - A directory containing XYZ files
"""

import argparse
import sys
from pathlib import Path
import tempfile
from openbabel import openbabel as ob


def _normalize_element_symbol(symbol: str) -> str:
    """
    Normalize element symbols to proper case (e.g., "ZN" -> "Zn").
    """
    if not symbol:
        return symbol
    if len(symbol) == 1:
        return symbol.upper()
    return symbol[0].upper() + symbol[1:].lower()


def _write_normalized_xyz(input_path: Path) -> Path:
    """
    Write a temporary XYZ with normalized element symbols so Open Babel
    can interpret elements like Zn correctly.
    """
    lines = input_path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError("XYZ file is too short")

    header = lines[:2]
    body = lines[2:]

    normalized_body = []
    for line in body:
        stripped = line.strip()
        if not stripped:
            normalized_body.append(line)
            continue
        parts = stripped.split()
        if len(parts) < 4:
            normalized_body.append(line)
            continue
        parts[0] = _normalize_element_symbol(parts[0])
        normalized_body.append(" ".join(parts))

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xyz")
    tmp_path = Path(tmp.name)
    tmp.close()
    tmp_path.write_text("\n".join(header + normalized_body) + "\n")
    return tmp_path


def _assign_implicit_h_counts(mol: ob.OBMol) -> None:
    """
    Approximate implicit H counts based on typical valence for common elements.
    XYZ lacks bonding/valence info, so this helps Open Babel add H atoms.
    """
    typical_valence = {
        1: 1,   # H
        6: 4,   # C
        7: 3,   # N
        8: 2,   # O
        9: 1,   # F
        15: 3,  # P
        16: 2,  # S
        17: 1,  # Cl
        35: 1,  # Br
        53: 1,  # I
    }

    for atom in ob.OBMolAtomIter(mol):
        atomic_num = atom.GetAtomicNum()
        if atomic_num in typical_valence:
            explicit_valence = atom.GetExplicitValence()
            needed = max(0, typical_valence[atomic_num] - explicit_valence)
            atom.SetImplicitHCount(needed)


def _extract_atom_comments(input_path: Path) -> list[str]:
    """
    Extract per-atom trailing comments from the input XYZ.
    Returns a list aligned with the original atom order.
    """
    lines = input_path.read_text().splitlines()
    if len(lines) < 3:
        return []
    comments: list[str] = []
    for line in lines[2:]:
        stripped = line.strip()
        if not stripped:
            comments.append("")
            continue
        parts = stripped.split()
        if len(parts) <= 4:
            comments.append("")
            continue
        comment = " ".join(parts[4:])
        comments.append(comment)
    return comments


def _restore_atom_comments(output_path: Path, comments: list[str]) -> None:
    """
    Append original trailing comments to the first N atoms in the output XYZ.
    """
    lines = output_path.read_text().splitlines()
    if len(lines) < 3:
        return

    header = lines[:2]
    body = lines[2:]
    restored_body: list[str] = []

    for idx, line in enumerate(body):
        stripped = line.strip()
        if not stripped:
            restored_body.append(line)
            continue
        if idx < len(comments) and comments[idx]:
            restored_body.append(f"{stripped}  {comments[idx]}")
        else:
            restored_body.append(stripped)

    output_path.write_text("\n".join(header + restored_body) + "\n")


def add_hydrogens_to_file(input_path: Path, output_path: Path = None) -> bool:
    """
    Add hydrogens to a single XYZ file.
    
    Args:
        input_path: Path to input XYZ file
        output_path: Path for output file (default: input_with_H.xyz)
    
    Returns:
        True if successful, False otherwise
    """
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_with_H.xyz"
    
    temp_path = None
    try:
        comments = _extract_atom_comments(input_path)
        temp_path = _write_normalized_xyz(input_path)
        # Setup conversion
        conv = ob.OBConversion()
        conv.SetInFormat("xyz")
        conv.SetOutFormat("xyz")
        
        # Read molecule
        mol = ob.OBMol()
        if not conv.ReadFile(mol, str(temp_path)):
            print(f"Error: Could not read {input_path}", file=sys.stderr)
            return False
        
        # XYZ has no bonding; let Open Babel perceive connectivity
        mol.ConnectTheDots()
        mol.PerceiveBondOrders()
        _assign_implicit_h_counts(mol)

        # Add hydrogens
        mol.AddHydrogens()
        
        # Write output
        if not conv.WriteFile(mol, str(output_path)):
            print(f"Error: Could not write {output_path}", file=sys.stderr)
            return False

        _restore_atom_comments(output_path, comments)
        
        print(f"✓ Processed: {input_path.name} → {output_path.name}")
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}", file=sys.stderr)
        return False
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def process_path(path: Path) -> None:
    """
    Process either a single file or all XYZ files in a directory.
    
    Args:
        path: Path to file or directory
    """
    if path.is_file():
        if path.suffix.lower() != '.xyz':
            print(f"Error: {path} is not an XYZ file", file=sys.stderr)
            sys.exit(1)
        
        success = add_hydrogens_to_file(path)
        sys.exit(0 if success else 1)
    
    elif path.is_dir():
        xyz_files = list(path.glob("*.xyz"))
        
        if not xyz_files:
            print(f"No XYZ files found in {path}", file=sys.stderr)
            sys.exit(1)
        
        print(f"Found {len(xyz_files)} XYZ file(s) in {path}\n")
        
        successes = 0
        for xyz_file in sorted(xyz_files):
            if add_hydrogens_to_file(xyz_file):
                successes += 1
        
        print(f"\nSuccessfully processed {successes}/{len(xyz_files)} files")
        sys.exit(0 if successes == len(xyz_files) else 1)
    
    else:
        print(f"Error: {path} does not exist", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Add hydrogens to XYZ files using Open Babel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python add_hydrogens.py structure.xyz
  python add_hydrogens.py ./xyz_files/
        """
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to XYZ file or directory containing XYZ files"
    )
    
    args = parser.parse_args()
    process_path(args.path)


if __name__ == "__main__":
    main()