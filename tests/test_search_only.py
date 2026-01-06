"""
Test that runs only the search part of the pipeline and saves IDs to a file.

This test loads config.search.yaml, executes the RCSB search,
saves all returned PDB IDs to a text file, and exits without downloading
or processing any structures.
"""

import sys
from pathlib import Path

# Add parent directory to path to import scrape_pdb modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scrape_pdb.config import load_config
from scrape_pdb.search import search_pdb
from scrape_pdb.logger import setup_logger
import logging


def test_search_only_save_ids():
    """
    Run search using config.search.yaml and save all returned IDs to a text file.
    """
    # Setup paths
    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / "examples" / "config.search.yaml"
    output_file = repo_root / "data" / "search_results_ids.txt"
    
    # Ensure config file exists
    assert config_path.exists(), f"Config file not found: {config_path}"
    
    # Setup logger
    logger = setup_logger(
        name="test_search",
        log_file=None,
        level=logging.INFO,
        console=True
    )
    
    logger.info(f"Loading configuration from: {config_path}")
    
    # Load configuration
    config = load_config(str(config_path))
    
    logger.info(f"Search parameters:")
    logger.info(f"  Metal ion: {config.search_parameters.metal_ion}")
    logger.info(f"  Resolution cutoff: {config.search_parameters.resolution_cutoff}")
    logger.info(f"  Experimental method: {config.search_parameters.experimental_method}")
    logger.info(f"  Polymer type: {config.search_parameters.polymer_type}")
    
    # Execute the search
    logger.info("Executing RCSB PDB search...")
    pdb_ids = search_pdb(config)
    
    logger.info(f"Search completed. Found {len(pdb_ids)} PDB IDs.")
    
    # Save IDs to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for pdb_id in pdb_ids:
            f.write(f"{pdb_id}\n")
    
    logger.info(f"Saved {len(pdb_ids)} PDB IDs to: {output_file}")
    
    # Display first few IDs
    if pdb_ids:
        logger.info(f"First 10 IDs: {', '.join(pdb_ids[:10])}")
    
    # Verify we got results
    assert len(pdb_ids) > 0, "Search returned no results"
    
    print(f"\n{'='*60}")
    print(f"SUCCESS: Search completed!")
    print(f"Total IDs found: {len(pdb_ids)}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}\n")
    
    return pdb_ids


if __name__ == "__main__":
    # Run the test when executed directly
    test_search_only_save_ids()
