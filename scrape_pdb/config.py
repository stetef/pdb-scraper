#!/usr/bin/env python3
"""Configuration loading and validation for PDB Scraper pipeline."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from .models import MustHaveSpec, parse_must_have

logger = logging.getLogger("pipeline.config")

class SearchParameters(BaseModel):
    metal_ion: str
    resolution_cutoff: Optional[float] = None
    experimental_method: str = "ALL"
    polymer_type: Optional[str] = None  # None means all polymer types

class ProcessingConfig(BaseModel):
    batch_size: int = 100
    parallel_workers: int = 4
    rate_limit_delay: float = 0.1
    temp_directory: Path = Path("./temp_cif")
    cutoff: float = 3.0  # Metal-to-metal clustering distance
    selection_radius: Optional[float] = None  # Atom selection radius (defaults to cutoff if not specified)
    target: str = "ZN"
    metals_excluded: List[str] = Field(default_factory=list)
    # Maximum number of files to download during a run (None = unlimited)
    max_downloads: Optional[int] = None
    must_have: MustHaveSpec = Field(default_factory=MustHaveSpec)
    include_waters: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("must_have", mode="before")
    @classmethod
    def validate_must_have(cls, v: Any) -> MustHaveSpec:
        if isinstance(v, str):
            return parse_must_have(v)
        if isinstance(v, MustHaveSpec):
            return v
        if isinstance(v, dict):
            return MustHaveSpec(**v)
        return MustHaveSpec()

    @field_validator("temp_directory", mode="before")
    @classmethod
    def to_path(cls, v: Any) -> Path:
        return Path(v)

class OutputConfig(BaseModel):
    base_output_dir: Optional[Path] = None
    results_database: Path = Path("./results/results.csv")
    checkpoint_file: Path = Path("./results/checkpoint.db")
    log_file: Path = Path("./results/pipeline.log")
    save_matching_structures: bool = False
    matched_structures_dir: Path = Path("./results/matched_cifs")
    kept_structures_dir: Path = Path("./data/kept_structures")  # Directory for validated PDB files
    output_dir: Path = Path("./results")
    altloc_report_file: Optional[Path] = None

    @field_validator("base_output_dir", "results_database", "checkpoint_file", "log_file", "matched_structures_dir", "kept_structures_dir", "output_dir", "altloc_report_file", mode="before")
    @classmethod
    def to_path(cls, v: Any) -> Optional[Path]:
        if v is None: return None
        return Path(v)


class LigandRequirement(BaseModel):
    """Residue/atom-name based requirements on coordinating neighbors."""

    # Backwards compatible single residue name
    resname: Optional[str] = None
    # Optional list of residue names accepted for this requirement
    resnames: Optional[List[str]] = None
    atom_names: Optional[List[str]] = None
    count: int = 1
    # Backwards-compatibility shim (deprecated)
    min_count: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def migrate_min_count(cls, v: Any) -> Any:
        if isinstance(v, dict):
            if "count" not in v and "min_count" in v:
                v = dict(v)
                v["count"] = v.get("min_count")
        return v

    @field_validator("resname", mode="before")
    @classmethod
    def normalize_resname(cls, v: Any) -> str:
        if v is None:
            return None
        return str(v).strip().upper()

    @field_validator("resnames", mode="before")
    @classmethod
    def normalize_resnames(cls, v: Any) -> Optional[List[str]]:
        if v is None:
            return None
        if isinstance(v, str):
            v = [v]
        try:
            out = [str(s).strip().upper() for s in v if str(s).strip()]
        except Exception:
            return None
        return out or None

    @field_validator("atom_names", mode="before")
    @classmethod
    def normalize_atom_names(cls, v: Any) -> Optional[List[str]]:
        if v is None:
            return None
        if isinstance(v, str):
            v = [v]
        try:
            out = [str(s).strip().upper() for s in v if str(s).strip()]
        except Exception:
            return None
        return out or None

    @field_validator("count", mode="before")
    @classmethod
    def normalize_count(cls, v: Any) -> int:
        try:
            n = int(v)
        except Exception:
            n = 1
        return max(0, n)

    @model_validator(mode="after")
    def fill_resnames(self):
        # If only resname is provided, populate resnames.
        if not self.resnames:
            if self.resname:
                self.resnames = [self.resname]
        else:
            # Ensure resname is set consistently for older code/printing.
            if not self.resname and self.resnames:
                self.resname = self.resnames[0]
        return self


class ValidationConfig(BaseModel):
    coordination_distance_max: float = 2.8
    coordination_distance_min: float = 2.0
    # Exact coord-string filter (legacy): e.g. {"1N3S"}
    coord: Optional[Set[str]] = None
    # Residue-aware filter on coordinating neighbors: list of requirements that must all be met.
    ligand_requirements: Optional[List[LigandRequirement]] = None

    @field_validator("coord", mode="before")
    @classmethod
    def to_coord_set(cls, v: Any) -> Optional[Set[str]]:
        if v is None:
            return None
        if isinstance(v, str):
            v = [v]
        try:
            vals = [s.strip().upper() for s in v if s]
        except Exception:
            return None
        return set(vals) if vals else None

    @field_validator("ligand_requirements", mode="before")
    @classmethod
    def to_ligand_requirements(cls, v: Any) -> Optional[List[LigandRequirement]]:
        if v is None:
            return None
        if isinstance(v, dict):
            v = [v]
        if not isinstance(v, list):
            return None
        out: List[LigandRequirement] = []
        for item in v:
            try:
                out.append(LigandRequirement(**item) if isinstance(item, dict) else LigandRequirement(resname=str(item)))
            except Exception:
                continue
        return out or None

class PipelineConfig(BaseModel):
    search_parameters: SearchParameters
    processing: ProcessingConfig
    output: OutputConfig
    validation: ValidationConfig
    
    # Legacy/Manual input support (for main.py compatibility)
    input_mode: str = "ids"
    input_data: List[str] = Field(default_factory=list)
    cache_file: Path = Path("cache.json")
    log_level: str = "INFO"

    @field_validator("cache_file", mode="before")
    @classmethod
    def to_path(cls, v: Any) -> Path:
        return Path(v)

    # Compatibility properties for existing code (parser.py, etc.)
    @property
    def cutoff(self) -> float:
        return self.processing.cutoff
    
    @property
    def selection_radius(self) -> float:
        """Atom selection radius around metal centers (defaults to cutoff if not specified)."""
        return self.processing.selection_radius if self.processing.selection_radius is not None else self.processing.cutoff
    
    @property
    def target(self) -> str:
        return self.processing.target
        
    @property
    def metals_excluded(self) -> Set[str]:
        return set(m.upper() for m in self.processing.metals_excluded)
        
    @property
    def must_have(self) -> MustHaveSpec:
        return self.processing.must_have
        
    @property
    def include_waters(self) -> bool:
        return self.processing.include_waters

    @property
    def output_dir(self) -> Path:
        return self.output.output_dir
        
    @property
    def download_dir(self) -> Path:
        return self.processing.temp_directory
        
    @property
    def log_file(self) -> Path:
        return self.output.log_file
        
    @property
    def clusters_csv(self) -> Path:
        # Map to results_database if it is a CSV, otherwise default
        if self.output.results_database.suffix == '.csv':
            return self.output.results_database
        return self.output.output_dir / "clusters_summary.csv"
    
    @property
    def altloc_report(self) -> Path:
        if self.output.altloc_report_file:
            return self.output.altloc_report_file
        return self.output.output_dir / "altloc_report.csv"
        
    @property
    def cache(self) -> Path:
        return self.cache_file

    @property
    def coord_filters(self) -> Optional[Set[str]]:
        return self.validation.coord

def load_config(config_path: str) -> PipelineConfig:
    """Load and validate configuration from YAML file."""
    path_obj = Path(config_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    config_data = apply_base_output_dir_defaults(config_data)
    return PipelineConfig(**config_data)


def apply_base_output_dir_defaults(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """If base_output_dir is set, fill default output paths derived from it."""
    if not isinstance(config_data, dict):
        return config_data

    output_section = config_data.get("output")
    base_dir = output_section.get("base_output_dir") if isinstance(output_section, dict) else None
    if not base_dir:
        return config_data

    base = Path(base_dir)

    processing = config_data.setdefault("processing", {})
    if isinstance(processing, dict):
        processing.setdefault("temp_directory", str(base / "PDB-downloads"))

    output = config_data.setdefault("output", {})
    if isinstance(output, dict):
        output.setdefault("results_database", str(base / "results" / "summary.db"))
        output.setdefault("checkpoint_file", str(base / "results" / "checkpoint.db"))
        output.setdefault("log_file", str(base / "pipeline.log"))
        output.setdefault("matched_structures_dir", str(base / "results" / "validated_structures"))
        output.setdefault("kept_structures_dir", str(base / "results" / "kept_structures"))
        output.setdefault("output_dir", str(base / "output"))

    return config_data

def print_config_summary(config: PipelineConfig) -> None:
    """Log configuration summary to logger."""
    logger.info("="*60)
    logger.info("PIPELINE CONFIGURATION")
    logger.info("="*60)
    logger.info(f"Metal clustering cutoff: {config.cutoff} Å")
    logger.info(f"Atom selection radius: {config.selection_radius} Å")
    logger.info(f"Target HETATM: {config.target}")
    logger.info(f"Search Metal: {config.search_parameters.metal_ion}")
    
    if config.metals_excluded:
        logger.info(f"Metals excluded: {', '.join(sorted(config.metals_excluded))}")
    else:
        logger.info("Metals excluded: none")
    
    logger.info(f"Must-have filter: {config.must_have}")
    logger.info(f"Include waters: {config.include_waters}")
    logger.info(f"Input mode: {config.input_mode}")
    logger.info(f"Temp/Download directory: {config.download_dir}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Log file: {config.log_file}")
    logger.info("="*60)

def create_example_config(output_path: str = "config.yaml") -> None:
    """Create an example configuration file."""
    example = {
        "search_parameters": {
            "metal_ion": "ZN",
            "resolution_cutoff": 2.0,
            "experimental_method": "X-RAY DIFFRACTION",
            "polymer_type": "Protein"
        },
        "processing": {
            "batch_size": 50,
            "parallel_workers": 4,
            "rate_limit_delay": 0.2,
            "temp_directory": "./temp_cif",
            "cutoff": 3.0,
            "target": "ZN",
            "metals_excluded": [],
            "must_have": "",
            "include_waters": True
        },
        "output": {
            "results_database": "./results/clusters.csv",
            "checkpoint_file": "./results/checkpoint.db",
            "log_file": "./results/pipeline.log",
            "save_matching_structures": False,
            "matched_structures_dir": "./results/matched_cifs",
            "output_dir": "./results"
        },
        "validation": {
            "coordination_distance_max": 2.8,
            "coordination_distance_min": 2.0
        },
        "input_mode": "ids",
        "input_data": [],
        "log_level": "INFO"
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(example, f, sort_keys=False)
    
    print(f"Example config written to: {output_path}")