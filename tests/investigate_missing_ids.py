"""
Utility to investigate why specific PDB IDs might not appear in search results.

This script queries the RCSB PDB API to retrieve metadata for specific PDB IDs
and checks them against the search parameters to identify why they might be excluded.
"""

import requests
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scrape_pdb.config import load_config


def get_pdb_metadata(pdb_id: str) -> dict:
    """Fetch metadata for a specific PDB ID from RCSB using REST API."""
    pdb_id = pdb_id.upper()
    
    # Use REST API to get entry data
    base_url = "https://data.rcsb.org/rest/v1/core/entry/"
    url = base_url + pdb_id
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"Error fetching entry data for {pdb_id}: HTTP {response.status_code}")
            return {"error": f"HTTP {response.status_code}"}
        
        entry_data = response.json()
        
        # Get non-polymer entity IDs
        non_polymer_ids = entry_data.get('rcsb_entry_container_identifiers', {}).get('non_polymer_entity_ids', [])
        
        # Fetch each non-polymer entity to get component IDs (metal ions)
        metal_ions = []
        for entity_id in non_polymer_ids:
            entity_url = f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id}/{entity_id}"
            entity_resp = requests.get(entity_url, timeout=10)
            if entity_resp.status_code == 200:
                entity_data = entity_resp.json()
                comp_id = entity_data.get('pdbx_entity_nonpoly', {}).get('comp_id')
                if comp_id:
                    metal_ions.append(comp_id)
        
        # Add metal ions to entry data
        entry_data['metal_ions'] = metal_ions
        
        # Get polymer entity IDs and types
        polymer_ids = entry_data.get('rcsb_entry_container_identifiers', {}).get('polymer_entity_ids', [])
        polymer_types = []
        for entity_id in polymer_ids:
            polymer_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"
            polymer_resp = requests.get(polymer_url, timeout=10)
            if polymer_resp.status_code == 200:
                polymer_data = polymer_resp.json()
                ptype = polymer_data.get('entity_poly', {}).get('type')
                if ptype:
                    polymer_types.append(ptype)
        
        entry_data['polymer_types'] = polymer_types
        
        return entry_data
        
    except Exception as e:
        print(f"Exception fetching data for {pdb_id}: {e}")
        return {"error": str(e)}


def check_pdb_against_criteria(pdb_id: str, config_path: str = None):
    """
    Check a specific PDB ID against search criteria and explain why it might be excluded.
    
    Args:
        pdb_id: PDB ID to check
        config_path: Path to config file (defaults to examples/config.search.yaml)
    """
    if config_path is None:
        repo_root = Path(__file__).resolve().parent.parent
        config_path = repo_root / "examples" / "config.search.yaml"
    
    # Load config
    config = load_config(str(config_path))
    params = config.search_parameters
    
    print(f"\n{'='*70}")
    print(f"CHECKING PDB ID: {pdb_id.upper()}")
    print(f"{'='*70}")
    
    # Fetch metadata
    print(f"\nFetching metadata from RCSB...")
    data = get_pdb_metadata(pdb_id)
    
    if data is None:
        print(f"ERROR: Could not fetch metadata for {pdb_id} - returned None")
        print(f"{'='*70}\n")
        return
    
    if 'error' in data:
        print(f"ERROR: Could not fetch metadata for {pdb_id}")
        print(f"Error: {data['error']}")
        print(f"{'='*70}\n")
        return
    
    # Extract relevant fields from REST API response
    title = data.get('struct', {}).get('title', 'N/A')
    resolution_raw = data.get('rcsb_entry_info', {}).get('resolution_combined')
    # Resolution might be a list, take first value if so
    if isinstance(resolution_raw, list):
        resolution = resolution_raw[0] if resolution_raw else None
    else:
        resolution = resolution_raw
    exptl_methods = data.get('exptl', [])
    method = exptl_methods[0].get('method') if exptl_methods else 'N/A'
    
    # Get polymer types (added by get_pdb_metadata)
    polymer_types = data.get('polymer_types', [])
    
    # Get metal ions (added by get_pdb_metadata)
    metal_ions = data.get('metal_ions', [])
    
    print(f"\nStructure Information:")
    print(f"  Title: {title}")
    print(f"  Resolution: {resolution} Å" if resolution else "  Resolution: N/A")
    print(f"  Experimental Method: {method}")
    print(f"  Polymer Types: {', '.join(polymer_types) if polymer_types else 'N/A'}")
    print(f"  Metal Ions: {', '.join(metal_ions) if metal_ions else 'None detected'}")
    
    print(f"\n{'-'*70}")
    print(f"Search Criteria from Config:")
    print(f"  Metal Ion: {params.metal_ion}")
    print(f"  Resolution Cutoff: ≤ {params.resolution_cutoff} Å")
    print(f"  Experimental Method: {params.experimental_method}")
    print(f"  Polymer Type: {params.polymer_type}")
    
    print(f"\n{'-'*70}")
    print(f"Criteria Check:")
    
    checks = []
    
    # Check metal ion
    has_metal = params.metal_ion.upper() in metal_ions
    status = "✓ PASS" if has_metal else "✗ FAIL"
    checks.append(("Metal Ion", status, has_metal))
    print(f"  {status}: Metal ion {params.metal_ion} {'found' if has_metal else 'NOT FOUND'}")
    
    # Check resolution
    if resolution is not None:
        passes_resolution = resolution <= params.resolution_cutoff
        status = "✓ PASS" if passes_resolution else "✗ FAIL"
        checks.append(("Resolution", status, passes_resolution))
        print(f"  {status}: Resolution {resolution} Å ({'≤' if passes_resolution else '>'} {params.resolution_cutoff} Å)")
    else:
        checks.append(("Resolution", "? N/A", None))
        print(f"  ? N/A: Resolution data not available")
    
    # Check experimental method
    if params.experimental_method.upper() != "ALL":
        passes_method = method == params.experimental_method.upper()
        status = "✓ PASS" if passes_method else "✗ FAIL"
        checks.append(("Experimental Method", status, passes_method))
        print(f"  {status}: Method is {method} (required: {params.experimental_method})")
    else:
        checks.append(("Experimental Method", "✓ PASS", True))
        print(f"  ✓ PASS: Any experimental method accepted")
    
    # Check polymer type
    if params.polymer_type.upper() != "ALL":
        has_correct_polymer = any(pt and params.polymer_type.upper() in pt.upper() 
                                  for pt in polymer_types)
        status = "✓ PASS" if has_correct_polymer else "✗ FAIL"
        checks.append(("Polymer Type", status, has_correct_polymer))
        print(f"  {status}: Polymer type ({'found' if has_correct_polymer else 'NOT FOUND'}: {params.polymer_type})")
    else:
        checks.append(("Polymer Type", "✓ PASS", True))
        print(f"  ✓ PASS: Any polymer type accepted")
    
    # Summary
    failures = [name for name, _, passed in checks if passed is False]
    
    print(f"\n{'-'*70}")
    if failures:
        print(f"VERDICT: Structure would be EXCLUDED from search")
        print(f"Reasons: Failed {len(failures)} criteria - {', '.join(failures)}")
    else:
        print(f"VERDICT: Structure SHOULD be included in search")
        print(f"Note: Actual inclusion also depends on post-search filters:")
        print(f"  - Coordinating residues: {params.coordinating_residues}")
        print(f"  - Coordination count: {params.coordination_count}")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    reference_file = repo_root / "example-ids.txt"
    
    # Read IDs from example-ids.txt
    if reference_file.exists():
        with open(reference_file, 'r') as f:
            pdb_ids = [line.strip() for line in f if line.strip()]
        
        print(f"Checking {len(pdb_ids)} PDB IDs from {reference_file.name}...")
        
        for pdb_id in pdb_ids:
            check_pdb_against_criteria(pdb_id)
    else:
        print(f"ERROR: {reference_file} not found")
