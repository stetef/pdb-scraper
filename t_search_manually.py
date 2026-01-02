#!/usr/bin/env python3
"""Test script to validate PDB search results."""

import tempfile
import json
from pathlib import Path
from datetime import datetime

# Mock config for testing without importing your actual module
class MockSearchParams:
    def __init__(self):
        self.metal_ion = "ZN"
        self.coordinating_residues = ["CYS"]
        self.coordination_count = 4
        self.resolution_cutoff = 2.0
        self.experimental_method = "X-RAY DIFFRACTION"
        self.polymer_type = "Protein"
    
    def model_dump(self):
        return {
            "metal_ion": self.metal_ion,
            "coordinating_residues": self.coordinating_residues,
            "coordination_count": self.coordination_count,
            "resolution_cutoff": self.resolution_cutoff,
            "experimental_method": self.experimental_method,
            "polymer_type": self.polymer_type
        }

class MockConfig:
    def __init__(self):
        self.search_parameters = MockSearchParams()


def test_search_with_output():
    """Test PDB search and write results to a file for manual validation."""
    import requests
    
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    
    # Create mock config
    config = MockConfig()
    params = config.search_parameters
    
    query_nodes = []
    
    # 1. Metal Ion
    if params.metal_ion:
        query_nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_nonpolymer_entity_instance_container_identifiers.comp_id",
                "operator": "exact_match",
                "value": params.metal_ion.upper()
            }
        })
    
    # 2. Resolution
    if params.resolution_cutoff:
        query_nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less_or_equal",
                "value": params.resolution_cutoff
            }
        })
    
    # 3. Experimental Method
    if params.experimental_method and params.experimental_method.upper() != "ALL":
        query_nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "exptl.method",
                "operator": "exact_match",
                "value": params.experimental_method.upper()
            }
        })
    
    # 4. Polymer Type - FIXED: uppercase the value
    if params.polymer_type and params.polymer_type.upper() != "ALL":
        query_nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "entity_poly.rcsb_entity_polymer_type",
                "operator": "exact_match",
                "value": params.polymer_type.upper()  # Changed to uppercase
            }
        })
    
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": query_nodes
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True}
    }
    
    print("=" * 80)
    print("TESTING PDB SEARCH")
    print("=" * 80)
    print(f"\nSearch Parameters:")
    for key, value in params.model_dump().items():
        print(f"  {key}: {value}")
    
    print(f"\nNumber of query nodes: {len(query_nodes)}")
    print("\nGenerated Query:")
    print(json.dumps(query, indent=2))
    
    print("\n" + "=" * 80)
    print("Executing search...")
    print("=" * 80)
    
    response = requests.post(url, json=query)
    
    if response.status_code == 200:
        results = response.json()
        pdb_ids = [entry["identifier"] for entry in results.get("result_set", [])]
        
        print(f"\n✓ Search successful!")
        print(f"✓ Found {len(pdb_ids)} PDB IDs")
        
        # Write to temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(tempfile.gettempdir()) / f"pdb_search_results_{timestamp}.txt"
        
        with open(output_file, 'w') as f:
            f.write(f"PDB Search Results\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"=" * 80 + "\n\n")
            f.write(f"Search Parameters:\n")
            for key, value in params.model_dump().items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nTotal Results: {len(pdb_ids)}\n")
            f.write(f"=" * 80 + "\n\n")
            f.write("PDB IDs:\n")
            for pdb_id in pdb_ids:
                f.write(f"{pdb_id}\n")
        
        print(f"\n✓ Results written to: {output_file}")
        print(f"\nFirst 10 PDB IDs:")
        for pdb_id in pdb_ids[:10]:
            print(f"  - {pdb_id}")
        
        if len(pdb_ids) > 10:
            print(f"  ... and {len(pdb_ids) - 10} more")
        
        print(f"\nValidation URLs for first few entries:")
        for pdb_id in pdb_ids[:3]:
            print(f"  https://www.rcsb.org/structure/{pdb_id}")
        
        return pdb_ids
    else:
        print(f"\n✗ Search failed!")
        print(f"✗ Status code: {response.status_code}")
        print(f"✗ Response: {response.text}")
        return []


if __name__ == "__main__":
    test_search_with_output()

# %%
