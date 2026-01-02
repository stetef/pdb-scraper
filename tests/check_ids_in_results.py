"""
Helper script to check if specific PDB IDs appear in the search results.

This script compares PDB IDs from a reference file against the search results
and provides a detailed report showing which IDs were found and which weren't.
"""

from pathlib import Path


def check_ids_in_search_results(reference_file: str, search_results_file: str):
    """
    Check if PDB IDs from reference file are present in search results.
    
    Args:
        reference_file: Path to file containing PDB IDs to check
        search_results_file: Path to search results file
    """
    # Read reference IDs
    ref_path = Path(reference_file)
    if not ref_path.exists():
        print(f"ERROR: Reference file not found: {reference_file}")
        return
    
    with open(ref_path, 'r') as f:
        reference_ids = [line.strip().upper() for line in f if line.strip()]
    
    # Read search result IDs
    results_path = Path(search_results_file)
    if not results_path.exists():
        print(f"ERROR: Search results file not found: {search_results_file}")
        return
    
    with open(results_path, 'r') as f:
        search_ids = set(line.strip().upper() for line in f if line.strip())
    
    print(f"\n{'='*70}")
    print(f"PDB ID COMPARISON REPORT")
    print(f"{'='*70}")
    print(f"\nReference file: {reference_file}")
    print(f"Search results file: {search_results_file}")
    print(f"\nTotal reference IDs: {len(reference_ids)}")
    print(f"Total search results: {len(search_ids)}")
    print(f"\n{'-'*70}")
    
    # Check each reference ID
    found_ids = []
    missing_ids = []
    
    for pdb_id in reference_ids:
        if pdb_id in search_ids:
            found_ids.append(pdb_id)
            print(f"✓ FOUND:   {pdb_id}")
        else:
            missing_ids.append(pdb_id)
            print(f"✗ MISSING: {pdb_id}")
    
    print(f"\n{'-'*70}")
    print(f"Summary:")
    print(f"  Found:   {len(found_ids)}/{len(reference_ids)} ({100*len(found_ids)/len(reference_ids):.1f}%)")
    print(f"  Missing: {len(missing_ids)}/{len(reference_ids)} ({100*len(missing_ids)/len(reference_ids):.1f}%)")
    
    if missing_ids:
        print(f"\n{'-'*70}")
        print(f"Missing IDs: {', '.join(missing_ids)}")
        print(f"\nPossible reasons why IDs might be missing:")
        print(f"  - Structure doesn't meet resolution cutoff (≤ 2.0 Å)")
        print(f"  - Structure doesn't contain ZN")
        print(f"  - Structure not determined by X-RAY DIFFRACTION")
        print(f"  - Structure is not a Protein")
        print(f"  - Structure doesn't have 4 CYS coordinating residues")
    
    print(f"{'='*70}\n")
    
    return found_ids, missing_ids


if __name__ == "__main__":
    # Check example-ids.txt against search results
    repo_root = Path(__file__).resolve().parent.parent
    
    reference = repo_root / "example-ids.txt"
    results = repo_root / "data" / "search_results_ids.txt"
    
    check_ids_in_search_results(str(reference), str(results))
