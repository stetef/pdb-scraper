#!/usr/bin/env python3
"""
PDB Scraper Pipeline - Metal cluster extraction from PDB files.

This package implements a complete pipeline for:
- Downloading PDB files from RCSB
- Parsing PDB format with full altloc handling
- Detecting metal clusters using union-of-spheres
- Classifying coordination geometry
- Writing XYZ files and CSV summaries
"""

__version__ = "0.2.1"
__author__ = "Samantha Tetef"

# Core data models
from .models import (
    Atom,
    MustHaveSpec,
    parse_must_have,
    GeometryResult,
    ClusterInfo,
    coord_string_from_atoms
)

# Configuration
from .config import PipelineConfig, load_config

# Main pipeline (CLI orchestration) — imported lazily: ``scrape_pdb.main``
# pulls in tqdm/requests/pyyaml, which are `cli` extras, not core dependencies
# (P1.9, 04 §5, D-10). ``from scrape_pdb import run_pipeline`` still works and
# still raises ImportError there if the `cli` extra is not installed.

# Library API (EARL structure source) — see scrape_pdb.api / 04-pdb-scraper-library-spec.md
from .site import SiteCandidate, ExtractConfig, ExtractResult, FetchError
from .api import fetch_entry, extract_sites

# Constants
from .constants import (
    ALL_METALS,
    USER_AGENT,
    DEFAULT_CUTOFF,
    DEFAULT_TARGET,
    WATER_RESIDUES
)

__all__ = [
    # Models
    "Atom",
    "MustHaveSpec",
    "parse_must_have",
    "GeometryResult",
    "ClusterInfo",
    "coord_string_from_atoms",
    
    # Config
    "PipelineConfig",
    "load_config",
    
    # Main
    "run_pipeline",

    # Library API
    "fetch_entry",
    "extract_sites",
    "SiteCandidate",
    "ExtractConfig",
    "ExtractResult",
    "FetchError",

    # Constants
    "ALL_METALS",
    "USER_AGENT",
    "DEFAULT_CUTOFF",
    "DEFAULT_TARGET",
    "WATER_RESIDUES",
]


# --- lazy CLI-only attributes (PEP 562) ------------------------------------
# Keeping these out of the eager import list is what lets a core-only install
# (biopython/numpy/pydantic) do ``import scrape_pdb.api`` without tqdm,
# requests, pyyaml, matplotlib, py3dmol or xraylarch present.
_LAZY_ATTRS = {
    "run_pipeline": ("scrape_pdb.main", "run_pipeline"),
}


def __getattr__(name: str):
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_name, attr = target
    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS))
