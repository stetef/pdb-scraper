#!/usr/bin/env python3
"""PDB download functions"""

from typing import Optional
import os
from tqdm import tqdm

from .constants import USER_AGENT
from .config import PipelineConfig
from pathlib import Path
import urllib.request
import gzip
import shutil
import glob
import re
import logging


logger = logging.getLogger("pipeline.downloader")


def fetch_pdb(pdb_id: str, dest_dir: str) -> Optional[str]:
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
        with urllib.request.urlopen(req, timeout=30) as resp:
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
        with urllib.request.urlopen(req, timeout=30) as resp:
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
        with urllib.request.urlopen(req, timeout=30) as resp:
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

def resolve_input_sources(config: PipelineConfig) -> list[str]:
    """Resolve input sources based on `PipelineConfig` attributes."""
    mode = config.input_mode
    data = config.input_data
    download_dir = str(config.download_dir)
    
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