#!/usr/bin/env python3
"""PDB download functions"""

from typing import Optional
import os
from tqdm import tqdm

from .constants import USER_AGENT
from .config import PipelineConfig
from .search import search_pdb
from .checkpoint import CheckpointManager
import urllib.request
import gzip
import shutil
import glob
import re
import logging


logger = logging.getLogger("pipeline.downloader")


def fetch_pdb(pdb_id: str, dest_dir: str, timeout: int = 30) -> Optional[str]:
    """Download {id}.pdb (fallback to mmCIF) into dest_dir and return local path."""
    pdb_id = pdb_id.strip().lower()
    if not pdb_id or len(pdb_id) != 4 or not pdb_id.isalnum():
        logger.info(f"[!] Skipping invalid PDB id: {pdb_id}")
        return None
    os.makedirs(dest_dir, exist_ok=True)

    out_path = os.path.join(dest_dir, f"{pdb_id}.pdb")
    
    # Try plain .pdb first
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                with open(out_path, "wb") as out:
                    out.write(resp.read())
                logger.info(f"[+] Downloaded {pdb_id} → {out_path}")
                return out_path
    except Exception as e:
        logger.debug(f"Could not fetch {pdb_id}.pdb: {e}")

    # Fallback: .pdb.gz
    url_gz = f"https://files.rcsb.org/download/{pdb_id}.pdb.gz"
    gz_path = os.path.join(dest_dir, f"{pdb_id}.pdb.gz")
    try:
        req = urllib.request.Request(url_gz, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                with open(gz_path, "wb") as out:
                    out.write(resp.read())
                with gzip.open(gz_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(gz_path)
                logger.info(f"[+] Downloaded {pdb_id} (gz) → {out_path}")
                return out_path
    except Exception as e:
        logger.debug(f"Could not fetch {pdb_id}.pdb.gz: {e}")
    
    # Fallback 2: Try mmCIF format and convert (for newer/extended PDB IDs)
    # The middle two characters determine the subdirectory
    middle_chars = pdb_id[1:3]
    url_cif = f"https://files.rcsb.org/pub/pdb/data/structures/divided/mmCIF/{middle_chars}/{pdb_id}.cif.gz"
    cif_gz_path = os.path.join(dest_dir, f"{pdb_id}.cif.gz")
    cif_path = os.path.join(dest_dir, f"{pdb_id}.cif")
    
    try:
        req = urllib.request.Request(url_cif, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                with open(cif_gz_path, "wb") as out:
                    out.write(resp.read())
                with gzip.open(cif_gz_path, 'rb') as f_in, open(cif_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(cif_gz_path)
                logger.info(f"[+] Downloaded {pdb_id} (mmCIF) → {cif_path}")
                logger.warning(f"[!] Note: {pdb_id} downloaded as CIF format, not PDB format")
                # Return the CIF file - your parser will need to handle this
                return cif_path
    except Exception as e:
        logger.debug(f"Could not fetch {pdb_id}.cif.gz: {e}")
    
    logger.warning(f"[!] Could not download {pdb_id} in any format")
    return None

def batch_download_from_list(listfile: str, outdir: str) -> list[str]:
    with open(listfile) as f:
        contents = f.read()
    tokens = re.split(r"[\s,]+", contents.strip())
    paths = []
    # for token in tokens:
    for token in tqdm(tokens):
        if not token:
            continue
        p = fetch_pdb(token, outdir)
        if p:
            paths.append(p)
    return paths

def resolve_input_sources(config: PipelineConfig, checkpoint: Optional[CheckpointManager] = None, limit: Optional[int] = None, skip_kept: bool = False) -> list[str]:
    """Resolve input sources based on `PipelineConfig` attributes.

    If a `CheckpointManager` is provided, already-completed PDB IDs
    will be skipped to allow restart/resume behavior.
    
    Args:
        config: Pipeline configuration
        checkpoint: Optional checkpoint manager for tracking processed IDs
        limit: Maximum number of structures to download in this call
        skip_kept: If True, skip structures that have already been kept (matched status)
    """
    mode = config.input_mode
    data = config.input_data
    download_dir = str(config.download_dir)

    sources: list[str] = []
    config_max = getattr(getattr(config, "processing", None), "max_downloads", None)
    max_dl = limit if limit is not None else config_max
    downloaded = 0
    skip_statuses = {"matched", "rejected", "error", "download_failed"}

    def norm_id(pdb_id: str) -> str:
        return (pdb_id or "").strip().lower()

    def record_download_failure(pdb_id: str) -> None:
        pdb_id = norm_id(pdb_id)
        if not pdb_id:
            return
        if checkpoint:
            checkpoint.update_status(pdb_id, "download_failed", error_message="download_failed")
        logger.info(f"[i] Marking {pdb_id} as download_failed")

    def should_process(pdb_id: str) -> bool:
        pdb_id = norm_id(pdb_id)
        if not pdb_id:
            return False
        if not checkpoint:
            return True
        status = checkpoint.get_status(pdb_id)
        # Always skip if already processed (matched, rejected, or error)
        if status in skip_statuses:
            logger.debug(f"[i] Skipping {pdb_id}, already {status} in checkpoint")
            return False
        return True

    if mode == "ids":
        # List of PDB IDs to download
        for pdb_id in data:
            pdb_id = norm_id(pdb_id)
            if not should_process(pdb_id):
                continue
            if max_dl is not None and downloaded >= max_dl:
                logger.info(f"Reached max_downloads limit ({max_dl}). Stopping further downloads.")
                break
            path = fetch_pdb(pdb_id, download_dir)
            if path:
                sources.append(path)
                downloaded += 1
            else:
                record_download_failure(pdb_id)
    
    elif mode == "paths":
        # List of local file paths
        for path in data:
            if os.path.isfile(path):
                sources.append(path)
            else:
                logger.info(f"[!] File not found, skipping: {path}")
    
    elif mode == "folder":
        # Single folder path containing *.pdb files
        folder = data[0] if data else "."
        if not os.path.isdir(folder):
            logger.info(f"[!] Not a directory: {folder}")
            return []
        sources = sorted(glob.glob(os.path.join(folder, "*.pdb")))
        logger.info(f"[i] Found {len(sources)} PDB files in {folder}")
    
    elif mode == "list_file":
        # Text file containing PDB IDs
        list_file = data[0] if data else None
        if not list_file or not os.path.isfile(list_file):
            logger.info(f"[!] List file not found: {list_file}")
            return []
        # batch_download_from_list downloads all listed IDs
        tokens = []
        with open(list_file) as f:
            tokens = [t for t in re.split(r"[\s,]+", f.read().strip()) if t]
        for t in tokens:
            t = norm_id(t)
            if not should_process(t):
                continue
            if max_dl is not None and downloaded >= max_dl:
                logger.info(f"Reached max_downloads limit ({max_dl}). Stopping further downloads.")
                break
            p = fetch_pdb(t, download_dir)
            if p:
                sources.append(p)
                downloaded += 1
            else:
                record_download_failure(t)
    
    elif mode == "search":
        # Perform RCSB search based on config, then download resulting IDs
        try:
            ids = search_pdb(config)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

        # Normalize + preserve order + de-duplicate
        normalized_ids: list[str] = []
        seen: set[str] = set()
        for pdb_id in ids:
            pdb_id = norm_id(pdb_id)
            if not pdb_id or pdb_id in seen:
                continue
            seen.add(pdb_id)
            normalized_ids.append(pdb_id)

        filtered_ids = normalized_ids
        if checkpoint and skip_kept:
            filtered_ids = checkpoint.get_pending_ids(normalized_ids)
            logger.info(f"[i] {len(filtered_ids)} pending IDs remain after checkpoint filter (from {len(normalized_ids)})")

        for pdb_id in filtered_ids:
            if not should_process(pdb_id):
                continue
            if max_dl is not None and downloaded >= max_dl:
                logger.info(f"Reached max_downloads limit ({max_dl}). Stopping further downloads.")
                break
            p = fetch_pdb(pdb_id, download_dir)
            if p:
                sources.append(p)
                downloaded += 1
            else:
                record_download_failure(pdb_id)
    elif mode == "mixed":
        # Mix of IDs and paths
        for item in data:
            if os.path.isfile(item):
                sources.append(item)
            elif len(item) == 4 and item.isalnum():
                item = norm_id(item)
                if max_dl is not None and downloaded >= max_dl:
                    logger.info(f"Reached max_downloads limit ({max_dl}). Stopping further downloads.")
                    break
                path = fetch_pdb(item, download_dir)
                if path:
                    sources.append(path)
                    downloaded += 1
                else:
                    record_download_failure(item)
            else:
                logger.info(f"[!] Skipping invalid item: {item}")
    
    else:
        logger.info(f"[!] Unknown input_mode: {mode}")
        return []
    
    return sources

    