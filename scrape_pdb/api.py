#!/usr/bin/env python3
"""Public library API for EARL (and other library callers).

The whole point of this module: call one function with one PDB id and one
absorber element, get structured site objects back. No YAML config, no writing
to the current directory, no global state. See ``04-pdb-scraper-library-spec.md``.

Typical use::

    from pathlib import Path
    from scrape_pdb.api import fetch_entry, extract_sites, ExtractConfig

    work = Path("/scratch/run123")
    pdb = fetch_entry("5XP6", workdir=work)
    result = extract_sites(pdb, element="ZN", config=ExtractConfig(workdir=work))
    for site in result.sites:
        print(site.cluster_id, site.target_atom_index, site.geometry["label"])
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional

from .constants import ALL_METALS
from .downloader import fetch_pdb
from .extract import extract_sites_from_file
from .site import ExtractConfig, ExtractResult, FetchError, SiteCandidate

__all__ = [
    "fetch_entry",
    "extract_sites",
    "ExtractConfig",
    "ExtractResult",
    "SiteCandidate",
    "FetchError",
    "search_entries",
    "search_by_name",
    "dedup_entries",
]

_LOGGER = logging.getLogger("pipeline.api")

# Classic 4-character PDB id (first char is a digit 1-9). Extended 8-char ids
# are a v2 concern (04 §2).
_PDB_ID_RE = re.compile(r"^[1-9][A-Za-z0-9]{3}$")

# RCSB PDB entries are small; a 50 MB cap rejects anything pathological while
# comfortably clearing the largest real assemblies.
_SIZE_CAP_BYTES = 50 * 1024 * 1024


def fetch_entry(
    pdb_id: str,
    *,
    workdir: Path | str,
    timeout_s: float = 30,
    max_retries: int = 3,
    size_cap_bytes: int = _SIZE_CAP_BYTES,
) -> Path:
    """Download one RCSB entry into ``workdir`` and return the local file path.

    Wraps the existing downloader but adds what a web backend needs: strict id
    validation, a size cap, an explicit timeout, and bounded retries with
    exponential backoff. Raises :class:`FetchError` (with a ``kind`` tag) on any
    failure instead of returning ``None``.

    The returned path may be ``.pdb`` or ``.cif`` (RCSB mmCIF fallback); the
    extractor handles both.
    """
    pid = (pdb_id or "").strip()
    if not _PDB_ID_RE.match(pid):
        raise FetchError(f"{pdb_id!r} is not a valid 4-character PDB id", kind="invalid_id")

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    delay = 1.0
    for attempt in range(1, max_retries + 1):
        path: Optional[str] = None
        try:
            path = fetch_pdb(pid, str(workdir), timeout=int(timeout_s))
        except Exception as e:  # network/urllib errors: retry
            _LOGGER.debug("fetch_pdb attempt %d for %s failed: %s", attempt, pid, e)

        if path:
            p = Path(path)
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            if size > size_cap_bytes:
                p.unlink(missing_ok=True)
                raise FetchError(
                    f"{pid} is {size} bytes, exceeding the {size_cap_bytes} byte cap",
                    kind="too_large",
                )
            return p

        if attempt < max_retries:
            time.sleep(delay)
            delay *= 2

    raise FetchError(f"could not download {pid} from RCSB", kind="not_found")


def extract_sites(
    pdb_path: Path | str,
    *,
    element: str,
    config: ExtractConfig,
    logger: Optional[logging.Logger] = None,
) -> ExtractResult:
    """Extract every site of ``element`` from a local PDB/mmCIF file.

    Pure function of (file, config): writes nothing outside ``config.workdir``,
    deletes no inputs, appends to no shared files, mutates no global logger.
    Returns an :class:`ExtractResult` with ``sites``, ``rejections``,
    ``warnings`` and ``stats``.
    """
    return extract_sites_from_file(pdb_path, element, config, logger=logger)


# ---------------------------------------------------------------------------
# Search-by-name flow — Part 4 of the TODO ("later; design only"). The surface
# is declared here so callers can see the intended shape, but it is not built:
# do not implement until the PDB-ID flow has been used in EARL (04 §4).
# ---------------------------------------------------------------------------

def search_entries(query, *, timeout_s: float = 30) -> list[str]:  # pragma: no cover
    raise NotImplementedError(
        "search_entries is a v2 feature (search-by-name); use fetch_entry with an explicit id."
    )


def search_by_name(name: str, element: str, *, limit: int = 200, timeout_s: float = 30) -> list:  # pragma: no cover
    raise NotImplementedError(
        "search_by_name is a v2 feature; not built until the PDB-ID flow is in use (04 §4)."
    )


def dedup_entries(hits: list, *, identity: int = 50) -> list:  # pragma: no cover
    raise NotImplementedError(
        "dedup_entries is a v2 feature (RCSB sequence-cluster dedup)."
    )
