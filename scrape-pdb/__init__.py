#!/usr/bin/env python3
"""
PDB Phase 1 Pipeline - Metal cluster extraction from PDB files.

This package implements a complete pipeline for:
- Downloading PDB files from RCSB
- Parsing PDB format with full altloc handling
- Detecting metal clusters using union-of-spheres
- Classifying coordination geometry
- Writing XYZ files and CSV summaries
"""

__version__ = "1.0.0"
__author__ = "PDB Phase 1 Team"

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

# Main pipeline
from .main import run_pipeline

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
    
    # Constants
    "ALL_METALS",
    "USER_AGENT",
    "DEFAULT_CUTOFF",
    "DEFAULT_TARGET",
    "WATER_RESIDUES",
]