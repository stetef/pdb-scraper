#!/usr/bin/env python3
"""Main orchestration for PDB Scraper pipeline."""

import logging
import sys
import time
from pathlib import Path
import argparse
from collections import Counter

from .config import load_config, print_config_summary, create_example_config
from .logger import setup_logger, PipelineLogger
from .downloader import resolve_input_sources, fetch_pdb
from .checkpoint import CheckpointManager
from tqdm import tqdm
from .parser import process_pdb
from .writer import append_cache
from .search import search_pdb
from .writer import ensure_csv_headers, write_altloc_report_header


# Search-mode download resilience: retries for a batch in which every download
# failed (exponential backoff from DOWNLOAD_BACKOFF_S), and how many such batches
# in a row abort the run.
DOWNLOAD_RETRIES = 3
DOWNLOAD_BACKOFF_S = 30.0
MAX_FAILED_BATCHES = 5


def _resolve_path(path) -> str:
    """Best-effort absolute, symlink-resolved string form of a path."""
    try:
        return str(Path(path).resolve())
    except Exception:
        return str(Path(path).absolute())


def _is_within_dir(path, directory) -> bool:
    """True iff ``path`` sits strictly inside ``directory`` (both resolved)."""
    try:
        rp = Path(path).resolve()
        rd = Path(directory).resolve()
    except Exception:
        return False
    return rd in rp.parents


def _pipeline_may_remove(path, download_dir, created: set) -> bool:
    """Whether the pipeline may delete or overwrite ``path`` (P1.10; 04 §2).

    Safe by construction: only files the pipeline itself downloaded this run
    (``path`` in ``created``) **and** that live inside its own ``download_dir``
    may be removed. A caller-supplied input path (never in ``created``, usually
    outside ``download_dir``) is left untouched even if it happens to sit in the
    download directory.
    """
    return _resolve_path(path) in created and _is_within_dir(path, download_dir)


def _log_running_rejection_stats(
    logger: logging.Logger,
    run_rejection_counts: Counter[str],
    ligand_summary: list[dict] | None,
    rejected_cn: Counter[int],
    rejected_coord: Counter[str],
    *,
    heading: str,
    max_rejections: int = 15,
    max_seen_resnames: int = 10,
    max_cn: int = 15,
    max_coord: int = 15,
) -> None:
    if not run_rejection_counts and not ligand_summary and not rejected_cn and not rejected_coord:
        return

    logger.info(heading)

    if run_rejection_counts:
        logger.info("Rejection stats (counts are filter-events):")
        for k, v in run_rejection_counts.most_common(max_rejections):
            logger.info(f"  - {k}: {v}")

    if rejected_cn:
        logger.info("Coordination Numbers (CN) of REJECTED clusters:")
        for cn, count in rejected_cn.most_common(max_cn):
            logger.info(f"  - CN={cn}: {count}")
    
    if rejected_coord:
        logger.info("Coord strings of REJECTED clusters:")
        for coord, count in rejected_coord.most_common(max_coord):
            logger.info(f"  - {coord}: {count}")

        logger.info("Selected coord strings (running totals):")
        for coord in ("2N2S", "3N1S", "4N"):
            logger.info(f"  - {coord}: {rejected_coord.get(coord, 0)}")

    if ligand_summary:
        logger.info("Ligand-requirements diagnostics (helps catch residue-name variants):")
        for i, item in enumerate(ligand_summary, start=1):
            expected = ",".join(item["expected_resnames"]) or "<none>"
            atoms = ",".join(item["atom_names"]) or "<any>"
            miss = item["missing"]
            mismatch = item["naming_mismatch"]
            logger.info(f"  - Req {i}: resnames=[{expected}] atom_names=[{atoms}] count={item['count']}")
            logger.info(f"    missing={miss} naming_mismatch={mismatch}")
            top = item["seen_resnames_for_atom_names"].most_common(max_seen_resnames)
            if top:
                logger.info("    top_seen_resnames_for_atom_names=" + ", ".join(f"{r}:{c}" for r, c in top))


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

        # Ensure output CSV files exist even if no clusters are produced.
        ensure_csv_headers(config)
        write_altloc_report_header(config)
        
        # Initialize checkpoint manager
        checkpoint = CheckpointManager(str(config.output.checkpoint_file))

        # Track how many PDBs we've kept (successfully validated)
        max_to_keep = config.processing.max_downloads or float('inf')
        kept_count = checkpoint.get_kept_count()

        # Aggregate rejection stats for this run (helps tune ligand requirement resnames).
        run_rejection_counts: Counter[str] = Counter()
        rejected_cn: Counter[int] = Counter()
        rejected_coord: Counter[str] = Counter()
        ligand_summary = None
        if config.validation.ligand_requirements:
            ligand_summary = [
                {
                    "expected_resnames": list(getattr(req, "resnames", None) or ([getattr(req, "resname", None)] if getattr(req, "resname", None) else [])),
                    "atom_names": list(getattr(req, "atom_names", None) or []),
                    "count": int(getattr(req, "count", getattr(req, "min_count", 1))),
                    "missing": 0,
                    "naming_mismatch": 0,
                    "seen_resnames_for_atom_names": Counter(),
                }
                for req in config.validation.ligand_requirements
            ]
        
        logger.info(f"Already kept {kept_count} structures from previous runs")
        logger.info(f"Target: keep up to {max_to_keep} total structures")
        
        # Process in batches until we have enough kept structures
        all_written: list[str] = []
        cache_runs: list[dict] = []
        failed_pdbs: list[str] = []
        # Files the pipeline downloaded this run — the only ones it may delete or
        # overwrite (P1.10; 04 §2). Caller-supplied inputs never land here.
        created_paths: set[str] = set()
        # Caller-supplied inputs left untouched (reported in the summary).
        untouched_inputs: list[str] = []
        # Sources already processed this run — so a kept caller input is not re-fed
        # every batch now that it is no longer deleted (P1.10).
        processed_sources: set[str] = set()
        
        # In search mode, cache the full candidate list once per run.
        cached_search_ids: list[str] | None = None

        batch_num = 0
        failed_batches_in_a_row = 0
        aborted_on_outage = False

        while kept_count < max_to_keep:
            batch_num += 1
            kept_before_batch = kept_count
            # Download next batch
            logger.info("")
            logger.info(f"=== Downloading batch {batch_num} (kept so far: {kept_count}/{max_to_keep}) ===")
            
            batch_limit = config.processing.batch_size
            if max_to_keep != float('inf'):
                remaining_needed = max_to_keep - kept_count
                if remaining_needed <= 0:
                    logger.info("No remaining structures needed; stopping downloads.")
                    break
                batch_limit = min(config.processing.batch_size, remaining_needed)

            # Get next batch of sources
            if config.input_mode == "search":
                if cached_search_ids is None:
                    logger.info("Executing search (cached for this run)...")
                    try:
                        ids = search_pdb(config)
                    except Exception as e:
                        logger.error(f"Search failed: {e}")
                        break

                    # Normalize + preserve order + de-duplicate
                    cached_search_ids = []
                    seen: set[str] = set()
                    for pdb_id in ids:
                        pid = (pdb_id or "").strip().lower()
                        if not pid or pid in seen:
                            continue
                        seen.add(pid)
                        cached_search_ids.append(pid)

                    logger.info(f"Search returned {len(cached_search_ids)} unique PDB IDs.")

                pending_ids = checkpoint.get_pending_ids(cached_search_ids)
                if not pending_ids:
                    batch_sources = []
                else:
                    next_ids = pending_ids[:batch_limit]
                    batch_sources = []
                    # A batch where *every* download fails is an outage (network /
                    # RCSB), not the end of the search: back off and retry it, then
                    # move on instead of mistaking the empty batch for "no more".
                    for attempt in range(DOWNLOAD_RETRIES + 1):
                        failed_ids = []
                        for pdb_id in next_ids:
                            path = fetch_pdb(pdb_id, str(config.download_dir))
                            if path:
                                created_paths.add(_resolve_path(path))
                                batch_sources.append(path)
                            else:
                                failed_ids.append(pdb_id)
                        if batch_sources or attempt == DOWNLOAD_RETRIES:
                            break
                        wait_s = DOWNLOAD_BACKOFF_S * 2 ** attempt
                        logger.warning(
                            f"All {len(next_ids)} downloads in batch {batch_num} failed; "
                            f"retrying in {wait_s:.0f} s (attempt {attempt + 1}/{DOWNLOAD_RETRIES})"
                        )
                        time.sleep(wait_s)
                    for pdb_id in failed_ids:
                        checkpoint.update_status(pdb_id, "download_failed", error_message="download_failed")
                    if not batch_sources:
                        failed_batches_in_a_row += 1
                        if failed_batches_in_a_row >= MAX_FAILED_BATCHES:
                            logger.error(
                                f"Aborting: {failed_batches_in_a_row} batches in a row failed to download "
                                "(network or RCSB outage?). Re-run the same config to resume; delete "
                                "'download_failed' rows from the checkpoint to retry those IDs."
                            )
                            aborted_on_outage = True
                            break
                        continue
                    failed_batches_in_a_row = 0
            else:
                batch_sources = resolve_input_sources(
                    config,
                    checkpoint=checkpoint,
                    limit=batch_limit,
                    skip_kept=True,
                    created=created_paths,
                )
            
            # Never process the same source twice in one run. Previously the loop
            # relied on inputs being *deleted* to shrink the batch; now that caller
            # inputs are kept (P1.10), file-input modes would otherwise re-feed the
            # same rejected file forever, so we track processed sources explicitly.
            batch_sources = [s for s in batch_sources if _resolve_path(s) not in processed_sources]

            if not batch_sources:
                logger.info("No more structures to download.")
                break

            logger.info(f"Downloaded {len(batch_sources)} structure(s) in this batch")

            # Process each PDB in the batch
            import shutil
            for idx, source_path in enumerate(batch_sources, 1):
                pdb_id = Path(source_path).stem
                processed_sources.add(_resolve_path(source_path))
                
                logger.info("")
                logger.info(f"[Batch {idx}/{len(batch_sources)}] Processing: {pdb_id}")
                
                try:
                    # Update checkpoint: mark in progress
                    checkpoint.update_status(pdb_id, "in_progress")

                    with PipelineLogger(logger, pdb_id) as pdb_log:
                        result = process_pdb(
                            pdb_path=source_path,
                            config=config
                        )

                        if isinstance(result, tuple) and len(result) == 2:
                            written, pdb_stats = result
                        else:
                            written, pdb_stats = result, None

                        # Aggregate stats (best-effort; doesn't affect pipeline outcomes)
                        try:
                            if isinstance(pdb_stats, dict):
                                for k, v in (pdb_stats.get("rejections") or {}).items():
                                    run_rejection_counts[str(k)] += int(v)
                                for cn, count in (pdb_stats.get("rejected_cn_distribution") or {}).items():
                                    rejected_cn[int(cn)] += int(count)
                                for coord, count in (pdb_stats.get("rejected_coord_distribution") or {}).items():
                                    rejected_coord[str(coord)] += int(count)
                                if ligand_summary and pdb_stats.get("ligand_requirements"):
                                    per_req = pdb_stats["ligand_requirements"].get("per_req") or []
                                    for i, pr in enumerate(per_req):
                                        if i >= len(ligand_summary):
                                            break
                                        ligand_summary[i]["missing"] += int(pr.get("missing", 0))
                                        ligand_summary[i]["naming_mismatch"] += int(pr.get("naming_mismatch", 0))
                                        ligand_summary[i]["seen_resnames_for_atom_names"].update(pr.get("seen_resnames_for_atom_names", {}))
                        except Exception:
                            pass

                        pdb_log.log_clusters(len(written))

                        # Update checkpoint status and handle file
                        if written:
                            # Validation passed
                            checkpoint.update_status(pdb_id, "matched")
                            kept_count += 1
                            
                            # Handle PDB file based on save_matching_structures setting
                            if config.output.save_matching_structures:
                                # Keep PDB in the matched directory. Move it only if
                                # the pipeline downloaded it this run; a caller-supplied
                                # input is copied so the original is preserved (P1.10).
                                matched_path = storage_dir / Path(source_path).name
                                if _pipeline_may_remove(source_path, config.download_dir, created_paths):
                                    shutil.move(str(source_path), str(matched_path))
                                    created_paths.discard(_resolve_path(source_path))
                                else:
                                    shutil.copy2(str(source_path), str(matched_path))
                                    untouched_inputs.append(str(source_path))
                                logger.info(f"✓ Kept {pdb_id} → {matched_path}")
                                pdb_path_for_cache = str(matched_path)
                            else:
                                # Don't keep the PDB: delete it only if the pipeline
                                # created it; leave a caller input untouched.
                                pdb_path_for_cache = str(source_path)
                                if _pipeline_may_remove(source_path, config.download_dir, created_paths):
                                    Path(source_path).unlink(missing_ok=True)
                                    created_paths.discard(_resolve_path(source_path))
                                    logger.info(f"✓ Validated {pdb_id}, deleted downloaded PDB (save_matching_structures=false)")
                                else:
                                    untouched_inputs.append(str(source_path))
                                    logger.info(f"✓ Validated {pdb_id}, left caller input in place: {source_path}")
                            
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
                            # Validation failed. Delete the PDB only if the pipeline
                            # downloaded it; a rejected caller input is left untouched
                            # and reported (P1.10; 04 §2 — the P1.9 hazard).
                            checkpoint.update_status(pdb_id, "rejected", rejection_reason="no_clusters")
                            if _pipeline_may_remove(source_path, config.download_dir, created_paths):
                                Path(source_path).unlink(missing_ok=True)
                                created_paths.discard(_resolve_path(source_path))
                                logger.info(f"✗ Deleted {pdb_id} (no clusters found)")
                            else:
                                untouched_inputs.append(str(source_path))
                                logger.info(f"✗ Rejected {pdb_id} (no clusters); left caller input in place: {source_path}")

                except Exception as e:
                    logger.error(f"Failed to process {pdb_id}: {e}", exc_info=True)
                    failed_pdbs.append(pdb_id)
                    checkpoint.update_status(pdb_id, "error", error_message=str(e))
                    # Delete the PDB only if the pipeline downloaded it.
                    if _pipeline_may_remove(source_path, config.download_dir, created_paths):
                        Path(source_path).unlink(missing_ok=True)
                        created_paths.discard(_resolve_path(source_path))
                        logger.info(f"✗ Deleted {pdb_id} (processing error)")
                    else:
                        untouched_inputs.append(str(source_path))
                        logger.info(f"✗ Error on {pdb_id}; left caller input in place: {source_path}")
                    continue
            
            # Clean up only the files the pipeline itself downloaded this run that
            # are still in the download directory — never caller files that happen
            # to sit there (P1.10; 04 §2). This replaces the old "wipe download_dir"
            # glob, which could delete a user's own PDB folder.
            logger.info(f"Cleaning downloaded files from: {config.download_dir}")
            for created in list(created_paths):
                p = Path(created)
                if not _is_within_dir(p, config.download_dir):
                    continue
                try:
                    if p.is_file():
                        p.unlink()
                        created_paths.discard(created)
                except Exception:
                    logger.debug(f"Failed removing {p}")

            # Log running stats at the end of each batch (useful for diagnosing low-yield motifs)
            batch_kept = kept_count - kept_before_batch
            logger.info("")
            logger.info(f"=== Batch {batch_num} complete: processed={len(batch_sources)} kept={batch_kept} total_kept={kept_count}/{max_to_keep} ===")
            _log_running_rejection_stats(
                logger,
                run_rejection_counts,
                ligand_summary,
                rejected_cn,
                rejected_coord,
                heading="Running stats (this run so far):",
            )
            
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

        _log_running_rejection_stats(
            logger,
            run_rejection_counts,
            ligand_summary,
            rejected_cn,
            rejected_coord,
            heading="Final stats (this run):",
            max_rejections=999999,
            max_seen_resnames=20,
            max_cn=30,
            max_coord=30,
        )
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
        if untouched_inputs:
            logger.info("")
            logger.info(f"Caller-supplied inputs left untouched ({len(untouched_inputs)}):")
            for src in untouched_inputs:
                logger.info(f"  - {src}")
        logger.info("="*60)

        if failed_pdbs:
            logger.warning(f"Failed PDBs: {', '.join(failed_pdbs)}")
            return 1
        if aborted_on_outage:
            logger.error("Run aborted on a download outage before the search was exhausted.")
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