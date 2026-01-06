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
        storage_dir = config.output.matched_structures_dir if config.output.save_matching_structures else config.output.kept_structures_dir
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize checkpoint manager
        checkpoint = CheckpointManager(str(config.output.checkpoint_file))

        # Track how many PDBs we've kept (successfully validated)
        max_to_keep = config.processing.max_downloads or float('inf')
        kept_count = checkpoint.get_kept_count()
        
        logger.info(f"Already kept {kept_count} structures from previous runs")
        logger.info(f"Target: keep up to {max_to_keep} total structures")
        
        # Process in batches until we have enough kept structures
        all_written: list[str] = []
        cache_runs: list[dict] = []
        failed_pdbs: list[str] = []
        
        while kept_count < max_to_keep:
            # Download next batch
            logger.info("")
            logger.info(f"=== Downloading next batch (kept so far: {kept_count}/{max_to_keep}) ===")
            
            batch_limit = config.processing.batch_size
            if max_to_keep != float('inf'):
                remaining_needed = max_to_keep - kept_count
                if remaining_needed <= 0:
                    logger.info("No remaining structures needed; stopping downloads.")
                    break
                batch_limit = min(config.processing.batch_size, remaining_needed)

            # Get next batch of sources
            batch_sources = resolve_input_sources(
                config, 
                checkpoint=checkpoint,
                limit=batch_limit,
                skip_kept=True
            )
            
            if not batch_sources:
                logger.info("No more structures to download.")
                break
            
            logger.info(f"Downloaded {len(batch_sources)} structure(s) in this batch")
            
            # Process each PDB in the batch
            import shutil
            for idx, source_path in enumerate(batch_sources, 1):
                pdb_id = Path(source_path).stem
                
                logger.info("")
                logger.info(f"[Batch {idx}/{len(batch_sources)}] Processing: {pdb_id}")
                
                try:
                    # Update checkpoint: mark in progress
                    checkpoint.update_status(pdb_id, "in_progress")

                    with PipelineLogger(logger, pdb_id) as pdb_log:
                        written = process_pdb(
                            pdb_path=source_path,
                            config=config
                        )

                        pdb_log.log_clusters(len(written))

                        # Update checkpoint status and handle file
                        if written:
                            # Validation passed
                            checkpoint.update_status(pdb_id, "matched")
                            kept_count += 1
                            
                            # Handle PDB file based on save_matching_structures setting
                            if config.output.save_matching_structures:
                                # Keep PDB: move to matched directory
                                matched_path = storage_dir / Path(source_path).name
                                shutil.move(str(source_path), str(matched_path))
                                logger.info(f"✓ Kept {pdb_id} → {matched_path}")
                                pdb_path_for_cache = str(matched_path)
                            else:
                                # Don't keep PDB: delete it
                                pdb_path_for_cache = str(source_path)
                                Path(source_path).unlink(missing_ok=True)
                                logger.info(f"✓ Validated {pdb_id}, deleted PDB (save_matching_structures=false)")
                            
                            # Cache this run
                            run = {
                                "pdb_path": pdb_path_for_cache,
                                "pdb_id": pdb_id,
                                "target": config.target,
                                "cutoff": config.cutoff,
                                "selection_radius": config.selection_radius,
                                "clusters": [{"xyz_path": p} for p in written],
                            }
                            cache_runs.append(run)
                            all_written.extend(written)
                            
                            # Check if we've reached our target
                            if kept_count >= max_to_keep:
                                logger.info(f"✓ Reached target of {max_to_keep} kept structures!")
                                break
                        else:
                            # Validation failed - delete this PDB
                            checkpoint.update_status(pdb_id, "rejected", rejection_reason="no_clusters")
                            Path(source_path).unlink(missing_ok=True)
                            logger.info(f"✗ Deleted {pdb_id} (no clusters found)")

                except Exception as e:
                    logger.error(f"Failed to process {pdb_id}: {e}", exc_info=True)
                    failed_pdbs.append(pdb_id)
                    checkpoint.update_status(pdb_id, "error", error_message=str(e))
                    # Delete failed PDB
                    Path(source_path).unlink(missing_ok=True)
                    logger.info(f"✗ Deleted {pdb_id} (processing error)")
                    continue
            
            # Clean up any remaining files in download directory
            logger.info(f"Cleaning download directory: {config.download_dir}")
            for p in config.download_dir.glob("*"):
                try:
                    if p.is_file():
                        p.unlink()
                    elif p.is_dir():
                        shutil.rmtree(p)
                except Exception:
                    logger.debug(f"Failed removing {p}")
            
            # Check if we've reached our target
            if kept_count >= max_to_keep:
                break
        
        # Update cache
        if cache_runs:
            logger.info("Updating cache file...")
            append_cache(cache_runs, config)
        
        # Final summary
        logger.info("")
        logger.info("="*60)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*60)
        logger.info(f"Total PDBs validated: {kept_count}")
        logger.info(f"Total XYZ files written: {len(all_written)}")
        logger.info(f"Failed: {len(failed_pdbs)}")
        logger.info("")
        logger.info("Output files:")
        if config.output.save_matching_structures:
            logger.info(f"  - Kept PDB files: {storage_dir}/")
        else:
            logger.info(f"  - PDB files: Not saved (save_matching_structures=false)")
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