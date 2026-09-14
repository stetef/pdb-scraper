#!/usr/bin/env python3
"""Dependency-free PDB download (stdlib ``urllib`` only).

Split out of :mod:`scrape_pdb.downloader` for card **P1.9** (04 §5, D-10):
``downloader`` is CLI machinery and imports ``tqdm`` plus ``.search`` (which
imports ``requests``) at module level, so the library path (``scrape_pdb.api``)
must not go through it. This module imports nothing outside the standard
library and :mod:`scrape_pdb.constants`, which is what lets a core-only install
(``biopython``/``numpy``/``pydantic``) do ``import scrape_pdb.api``.

``downloader`` re-exports :func:`fetch_pdb` so the CLI and its tests keep
working unchanged.
"""

from typing import Optional
import gzip
import logging
import os
import shutil
import urllib.request

from .constants import USER_AGENT

# Same logger name as before the split, so log output is unchanged.
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
