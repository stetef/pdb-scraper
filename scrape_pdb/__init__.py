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

__version__ = "0.2.0"
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

# Main pipeline (CLI orchestration)
from .main import run_pipeline

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