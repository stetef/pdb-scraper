#!/usr/bin/env python3
"""Configuration loading and validation for PDB Phase 1 pipeline."""

import json
import logging
from pathlib import Path
from typing import Dict, Set
from dataclasses import dataclass

from .models import parse_must_have, MustHaveSpec


logger = logging.getLogger("pipeline.config")


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    # Core parameters
    cutoff: float
    target: str
    metals_excluded: Set[str]
    must_have: MustHaveSpec
    include_waters: bool
    
    # Input/output
    input_mode: str
    input_data: list
    download_dir: Path
    output_dir: Path
    phase1_cache: Path
    
    # Logging
    log_file: Path
    log_level: str
    
    @property
    def altloc_report(self) -> Path:
        """Path to altloc report CSV."""
        return self.output_dir / "altloc_report.csv"
    
    @property
    def clusters_csv(self) -> Path:
        """Path to clusters summary CSV."""
        return self.output_dir / "clusters_summary.csv"


def load_config(config_path: str) -> PipelineConfig:
    """
    Load and validate configuration from JSON file.
    
    Args:
        config_path: Path to JSON config file
    
    Returns:
        PipelineConfig instance
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_file, 'r') as f:
            raw_config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")
    
    # Set defaults
    defaults = {
        "cutoff": 5.0,
        "target": "NI",
        "metals_excluded": [],
        "must_have": "",
        "include_waters": True,
        "input_mode": "ids",
        "input_data": [],
        "download_dir": "data/PDB-downloads",
        "output_dir": "data/output",
        "phase1_cache": "cache.json",
        "log_file": "data/pipeline.log",
        "log_level": "INFO"
    }
    
    # Merge with defaults
    config = {**defaults, **raw_config}
    
    # Validate input_mode
    valid_modes = {"ids", "paths", "folder", "list_file", "mixed"}
    if config["input_mode"] not in valid_modes:
        raise ValueError(
            f"Invalid input_mode: {config['input_mode']}. "
            f"Must be one of: {valid_modes}"
        )
    
    # Validate input_data
    if not config["input_data"]:
        raise ValueError("input_data cannot be empty")
    
    # Validate cutoff
    if config["cutoff"] <= 0:
        raise ValueError("cutoff must be positive")
    
    # Validate log_level
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if config["log_level"].upper() not in valid_levels:
        logger.warning(
            f"Invalid log_level: {config['log_level']}. Using INFO."
        )
        config["log_level"] = "INFO"
    
    # Parse must_have
    must_have = parse_must_have(config["must_have"])
    
    # Convert to PipelineConfig
    return PipelineConfig(
        cutoff=float(config["cutoff"]),
        target=str(config["target"]).upper(),
        metals_excluded=set(m.upper() for m in config["metals_excluded"]),
        must_have=must_have,
        include_waters=bool(config["include_waters"]),
        input_mode=config["input_mode"],
        input_data=config["input_data"],
        download_dir=Path(config["download_dir"]),
        output_dir=Path(config["output_dir"]),
        phase1_cache=Path(config["phase1_cache"]),
        log_file=Path(config["log_file"]),
        log_level=config["log_level"].upper()
    )


def print_config_summary(config: PipelineConfig) -> None:
    """Log configuration summary to logger."""
    logger.info("="*60)
    logger.info("PIPELINE CONFIGURATION")
    logger.info("="*60)
    logger.info(f"Cutoff distance: {config.cutoff} Å")
    logger.info(f"Target HETATM: {config.target}")
    
    if config.metals_excluded:
        logger.info(f"Metals excluded: {', '.join(sorted(config.metals_excluded))}")
    else:
        logger.info("Metals excluded: none")
    
    logger.info(f"Must-have filter: {config.must_have}")
    logger.info(f"Include waters: {config.include_waters}")
    logger.info(f"Input mode: {config.input_mode}")
    logger.info(f"Download directory: {config.download_dir}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Log file: {config.log_file}")
    logger.info("="*60)


def create_example_config(output_path: str = "example_config.json") -> None:
    """
    Create an example configuration file.
    
    Args:
        output_path: Where to write the example config
    """
    example = {
        "cutoff": 5.0,
        "target": "NI",
        "metals_excluded": [],
        "must_have": "",
        "include_waters": True,
        "input_mode": "list_file",
        "input_data": ["data/ids.txt"],
        "download_dir": "data/PDB-downloads",
        "output_dir": "data/output",
        "phase1_cache": "data/cache.json",
        "log_file": "data/pipeline.log",
        "log_level": "INFO",
        "_comments": {
            "input_mode_options": ["ids", "paths", "folder", "list_file", "mixed"],
            "must_have_examples": ["S", "S,N", "ALL:S,O", "S>=2,N>=1"],
            "log_level_options": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(example, f, indent=2)
    
    print(f"Example config written to: {output_path}")