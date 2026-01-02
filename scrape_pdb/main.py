#!/usr/bin/env python3
"""Main orchestration for PDB Scraper pipeline."""

import logging
import sys
from pathlib import Path
import argparse

from .config import load_config, print_config_summary, create_example_config
from .logger import setup_logger, PipelineLogger
from .downloader import resolve_input_sources
from .checkpoint import CheckpointManager
from tqdm import tqdm
from .parser import process_pdb
from .writer import append_cache


def run_pipeline(config_path: str, verbose: bool = False) -> int:
    """
    Run the complete PDB Scraper pipeline.
    
    Args:
        config_path: Path to JSON configuration file
        verbose: Enable verbose output to console
    
    Returns:
        Exit code (0 = success, 1 = error)
    """
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Setup logging and assign the configured logger to the module-level
        # `logger` so subsequent calls use the same configured logger.
        log_level = logging.DEBUG if verbose else getattr(logging, config.log_level)
        logger = setup_logger(
            name="pipeline",
            log_file=config.log_file,
            level=log_level,
            console=verbose
        )
        
        logger.info("PDB Scraper Started")
        logger.info(f"Config loaded from: {config_path}")
        
        # Print configuration
        print_config_summary(config)
        
        # Create output directories
        config.output_dir.mkdir(parents=True, exist_ok=True)
        config.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize checkpoint manager
        checkpoint = CheckpointManager(str(config.output.checkpoint_file))

        # Resolve input sources (may perform search and download)
        logger.info(f"Resolving input sources ({config.input_mode})...")
        sources = resolve_input_sources(config, checkpoint=checkpoint)
        
        if not sources:
            logger.error("No input sources resolved. Check configuration.")
            return 1
        
        logger.info(f"Found {len(sources)} PDB file(s) to process")
        
        # Process each PDB
        all_written: list[str] = []
        cache_runs: list[dict] = []
        failed_pdbs: list[str] = []
        
        processed_since_cleanup = 0
        for idx, source_path in enumerate(tqdm(sources, desc="Processing PDBs"), 1):
            pdb_id = Path(source_path).stem
            
            logger.info("")
            logger.info(f"[{idx}/{len(sources)}] Processing: {pdb_id}")
            
            try:
                # Update checkpoint: mark in progress
                checkpoint.update_status(pdb_id, "in_progress")

                with PipelineLogger(logger, pdb_id) as pdb_log:
                    written = process_pdb(
                        pdb_path=source_path,
                        config=config
                    )

                    pdb_log.log_clusters(len(written))

                    # Cache this run
                    run = {
                        "pdb_path": str(source_path),
                        "pdb_id": pdb_id,
                        "target": config.target,
                        "cutoff": config.cutoff,
                        "selection_radius": config.selection_radius,
                        "clusters": [{"xyz_path": p} for p in written],
                    }
                    cache_runs.append(run)
                    all_written.extend(written)

                    # Update checkpoint status
                    if written:
                        checkpoint.update_status(pdb_id, "matched")
                    else:
                        checkpoint.update_status(pdb_id, "rejected", rejection_reason="no_clusters")

            except Exception as e:
                logger.error(f"Failed to process {pdb_id}: {e}", exc_info=True)
                failed_pdbs.append(pdb_id)
                checkpoint.update_status(pdb_id, "error", error_message=str(e))
                continue

            # Batch cleanup of download directory to limit disk usage
            processed_since_cleanup += 1
            if processed_since_cleanup >= config.processing.batch_size:
                try:
                    dl_dir = config.download_dir
                    logger.info(f"Batch complete ({processed_since_cleanup}). Cleaning download directory: {dl_dir}")
                    for p in dl_dir.glob("*"):
                        try:
                            if p.is_file():
                                p.unlink()
                            elif p.is_dir():
                                import shutil
                                shutil.rmtree(p)
                        except Exception:
                            logger.debug(f"Failed removing {p}")
                except Exception as e:
                    logger.debug(f"Cleanup failed: {e}")
                processed_since_cleanup = 0
        
        # Update cache
        if cache_runs:
            logger.info("Updating cache file...")
            append_cache(cache_runs, config)
        
        # Final summary
        logger.info("")
        logger.info("="*60)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*60)
        logger.info(f"Total PDBs processed: {len(sources)}")
        logger.info(f"Successful: {len(sources) - len(failed_pdbs)}")
        logger.info(f"Failed: {len(failed_pdbs)}")
        logger.info(f"Total XYZ files written: {len(all_written)}")
        logger.info("")
        logger.info("Output files:")
        logger.info(f"  - XYZ files: {config.output_dir}/")
        logger.info(f"  - Summary CSV: {config.clusters_csv}")
        logger.info(f"  - AltLoc report: {config.altloc_report}")
        logger.info(f"  - Cache: {config.cache}")
        logger.info(f"  - Log: {config.log_file}")
        logger.info("="*60)
        
        if failed_pdbs:
            logger.warning(f"Failed PDBs: {', '.join(failed_pdbs)}")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Scraper: PDB → XYZ extractor (Spec-compliant v1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example config.json:
{
  "cutoff": 5.0,
  "target": "NI",
  "metals_excluded": ["NA", "K"],
  "must_have": "S,N",
  "include_waters": true,
  "input_mode": "ids",
  "input_data": ["1ubq", "2qmt"],
  "download_dir": "PDB",
  "output_dir": "pdb_env_outputs",
  "cache": "cache.json",
  "log_file": "pdb_env_outputs/pipeline.log",
  "log_level": "INFO"
}

Input modes:
  - "ids": download PDB IDs from RCSB
  - "paths": use local file paths
  - "folder": scan folder for *.pdb files
  - "list_file": read PDB IDs from text file
  - "mixed": mix of IDs and local paths

Usage:
  python -m scrape-pdb config.json
  python -m scrape-pdb config.json --verbose
  python -m scrape-pdb --example
        """
    )
    
    parser.add_argument(
        "config",
        nargs='?',
        help="Path to JSON configuration file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output to console"
    )
    
    parser.add_argument(
        "--example",
        action="store_true",
        help="Create an example configuration file"
    )
    
    args = parser.parse_args()
    
    # Handle example config creation
    if args.example:
        create_example_config("example_config.json")
        return 0
    
    # Require config file
    if not args.config:
        parser.print_help()
        return 1
    
    # Run pipeline
    exit_code = run_pipeline(args.config, verbose=args.verbose)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()