#!/usr/bin/env python3
"""PDB download functions"""

from typing import Optional
import os

from .constants import USER_AGENT
from .config import PipelineConfig
from pathlib import Path
import urllib.request
import gzip
import shutil
import glob
import re
import logging

logger = logging.getLogger(__name__)


def fetch_pdb(pdb_id: str, dest_dir: str) -> Optional[str]:
    """Download {id}.pdb (fallback to .pdb.gz) into dest_dir and return local path."""
    pdb_id = pdb_id.strip().lower()
    if not pdb_id or len(pdb_id) != 4 or not pdb_id.isalnum():
        logger.info(f"[!] Skipping invalid PDB id: {pdb_id}")
        return None
    os.makedirs(dest_dir, exist_ok=True)

    # Try plain .pdb first
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    out_path = os.path.join(dest_dir, f"{pdb_id}.pdb")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as resp, open(out_path, "wb") as out:
            out.write(resp.read())
        logger.info(f"[+] Downloaded {pdb_id} → {out_path}")
        return out_path
    except Exception:
        pass

    # Fallback: .pdb.gz
    url_gz = f"https://files.rcsb.org/download/{pdb_id}.pdb.gz"
    gz_path = os.path.join(dest_dir, f"{pdb_id}.pdb.gz")
    try:
        req = urllib.request.Request(url_gz, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as resp, open(gz_path, "wb") as out:
            out.write(resp.read())
        with gzip.open(gz_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(gz_path)
        logger.info(f"[+] Downloaded {pdb_id} (gz) → {out_path}")
        return out_path
    except Exception as e:
        logger.info(f"[!] Failed to fetch {pdb_id}: {e}")
        return None

def batch_download_from_list(listfile: str, outdir: str) -> list[str]:
    with open(listfile) as f:
        contents = f.read()
    tokens = re.split(r"[\s,]+", contents.strip())
    paths = []
    for token in tokens:
        if not token:
            continue
        p = fetch_pdb(token, outdir)
        if p:
            paths.append(p)
    return paths

def resolve_input_sources(config: dict) -> list[str]:
    """Resolve input sources based on config mode and data."""
    mode = config.get("input_mode", "ids")
    data = config.get("input_data", [])
    download_dir = config.get("download_dir", "PDB")
    
    sources = []
    
    if mode == "ids":
        # List of PDB IDs to download
        for pdb_id in data:
            path = fetch_pdb(pdb_id, download_dir)
            if path:
                sources.append(path)
    
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
        sources = batch_download_from_list(list_file, download_dir)
    
    elif mode == "mixed":
        # Mix of IDs and paths
        for item in data:
            if os.path.isfile(item):
                sources.append(item)
            elif len(item) == 4 and item.isalnum():
                path = fetch_pdb(item, download_dir)
                if path:
                    sources.append(path)
            else:
                logger.info(f"[!] Skipping invalid item: {item}")
    
    else:
        logger.info(f"[!] Unknown input_mode: {mode}")
        return []
    
    return sources