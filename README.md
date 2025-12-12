# PDB Phase 1 Pipeline - Refactored

A modular, production-ready pipeline for extracting metal-centered clusters from PDB files.

## Features

- **Modular Architecture**: Separated concerns into focused modules
- **Comprehensive Logging**: All output goes to log files with structured messages
- **Context Managers**: Proper resource handling and error management
- **Type Hints**: Full type annotations for better code quality
- **Configuration-Driven**: JSON-based configuration with validation
- **Error Recovery**: Graceful handling of failures with detailed logging

## Project Structure

```
pdb_phase1/
├── __init__.py              # Package initialization
├── __main__.py              # Entry point for `python -m pdb_phase1`
├── main.py                  # Pipeline orchestration
├── config.py                # Configuration loading & validation
├── constants.py             # Global contstants
├── models.py                # Data classes (Atom, MustHaveSpec, etc.)
├── parser.py                # PDB parsing functions
├── geometry.py              # Geometry classification & metrics
├── cluster.py               # Cluster detection & classification
├── altloc.py                # AltLoc handling logic
├── writer.py                # XYZ, CSV, cache writers
├── downloader.py            # PDB download functions
├── utils.py                 # General utilities
└── logger.py                # Logging setup
```

## Installation

```bash
# Clone or copy the pdb_phase1 package
cd your_project/
# Ensure the pdb_phase1 directory is in your Python path
```

## Quick Start

### 1. Create a configuration file

```bash
python -m pdb_phase1 --example
```

This creates `example_config.json`:

```json
{
  "cutoff": 5.0,
  "target": "NI",
  "metals_excluded": ["NA", "K"],
  "must_have": "S,N",
  "include_waters": true,
  "input_mode": "ids",
  "input_data": ["1ubq", "2qmt", "3hhp"],
  "download_dir": "PDB",
  "output_dir": "pdb_env_outputs",
  "phase1_cache": "phase1_cache.json",
  "log_file": "pdb_env_outputs/pipeline.log",
  "log_level": "INFO"
}
```

### 2. Run the pipeline

```bash
# Basic usage
python -m pdb_phase1 config.json

# With verbose console output
python -m pdb_phase1 config.json --verbose

# Or use the main module directly
python -m pdb_phase1.main config.json
```

## Configuration Reference

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cutoff` | float | 5.0 | Distance cutoff in Ångströms |
| `target` | string | "NI" | Target HETATM atom name |
| `metals_excluded` | list | [] | Metal symbols to exclude |
| `must_have` | string | "" | Required elements filter |
| `include_waters` | bool | true | Include water molecules |

### Must-Have Filter Examples

- `"S"` - Require at least one sulfur
- `"S,N"` - Require sulfur OR nitrogen (ANY mode)
- `"ALL:S,O"` - Require both sulfur AND oxygen
- `"S>=2,N>=1"` - Require at least 2 sulfurs and 1 nitrogen

### Input Modes

| Mode | Description | `input_data` Format |
|------|-------------|---------------------|
| `ids` | Download from RCSB | List of PDB IDs: `["1ubq", "2qmt"]` |
| `paths` | Local files | List of file paths: `["path/to/1ubq.pdb"]` |
| `folder` | Scan directory | Single folder path: `["./pdbs/"]` |
| `list_file` | Read IDs from file | Path to text file: `["pdb_list.txt"]` |
| `mixed` | Mix of IDs and paths | Combined list |

### Logging Levels

- `DEBUG` - Detailed diagnostic information
- `INFO` - General informational messages (default)
- `WARNING` - Warning messages
- `ERROR` - Error messages
- `CRITICAL` - Critical failures

## Output Files

### 1. XYZ Files
Located in `output_dir/`, one per cluster:
```
1ubq_NI_homo_d5.000_cluster1.xyz
1ubq_NI_multi_homo_d5.000_cluster2_altlocA.xyz
```

### 2. Clusters Summary CSV
`output_dir/clusters_summary.csv` contains:
- PDB ID, cluster info, center details
- Coordination number (CN), geometry, COORD string
- Geometric metrics (RMS_θ, σ_d, Δ_d, RMS_plane)
- Flags (planar, axial, distorted, Jahn-Teller)

### 3. AltLoc Report CSV
`output_dir/altloc_report.csv` tracks all metals with alternate locations.

### 4. Cache File
`phase1_cache.json` stores run manifests for reproducibility.

### 5. Log File
`output_dir/pipeline.log` contains all pipeline activity.

## Module Reference

### logger.py
```python
from pdb_phase1.logger import setup_logger, PipelineLogger

# Setup logging
logger = setup_logger(
    name="my_logger",
    log_file=Path("output.log"),
    level=logging.INFO
)

# Use context manager
with PipelineLogger(logger, "1ubq") as pdb_log:
    pdb_log.log_clusters(5)
    pdb_log.log_metals(10, 3)
```

### models.py
```python
from pdb_phase1.models import Atom, MustHaveSpec, parse_must_have

# Parse must-have specification
spec = parse_must_have("S>=2,N>=1")
print(spec.passes(neighbor_atoms))  # True/False
```

### config.py
```python
from pdb_phase1.config import load_config, print_config_summary

config = load_config("config.json")
print_config_summary(config)
```

### utils.py
```python
from pdb_phase1.utils import dist, centroid, connected_components

# Calculate distance
d = dist((0,0,0), (1,1,1))

# Find connected components
points = [(0,0,0), (1,0,0), (5,0,0)]
comps = connected_components(points, cutoff=2.0)
```

## Advanced Usage

### Custom Processing Script

```python
from pdb_phase1 import load_config, run_pipeline
from pdb_phase1.parser import load_pdb_atoms_all
from pdb_phase1.logger import setup_logger
import logging

# Setup custom logging
logger = setup_logger(
    name="custom",
    log_file=Path("custom.log"),
    level=logging.DEBUG
)

# Load config
config = load_config("config.json")

# Run pipeline
exit_code = run_pipeline("config.json", verbose=True)
```

### Programmatic Use

```python
from pdb_phase1.parser import process_pdb
from pdb_phase1.config import PipelineConfig
from pdb_phase1.models import parse_must_have

# Create config programmatically
config = PipelineConfig(
    cutoff=5.0,
    target="FE",
    metals_excluded=set(),
    must_have=parse_must_have("S>=2"),
    include_waters=True,
    # ... other params
)

# Process single PDB
written_files = process_pdb(
    pdb_path="1ubq.pdb",
    config=config,
    logger=logger
)
```

## Best Practices

1. **Always use configuration files** - Easier to reproduce and share
2. **Check log files** - All issues are logged with context
3. **Use verbose mode during development** - See real-time progress
4. **Validate configs** - The loader will catch issues early
5. **Handle exit codes** - Non-zero means something failed

## Error Handling

The pipeline uses several error handling strategies:

1. **Graceful PDB failures** - Failed PDBs are logged but don't stop the pipeline
2. **Configuration validation** - Invalid configs raise clear errors before processing
3. **Resource cleanup** - Context managers ensure proper cleanup
4. **Detailed logging** - Stack traces in log files for debugging

## Migration from Old Script

If you have existing code using the old monolithic script:

**Old:**
```python
python pdb_phase1.py  # Interactive prompts
```

**New:**
```bash
# Create config once
python -m pdb_phase1 --example

# Edit config.json with your settings

# Run pipeline
python -m pdb_phase1 config.json
```

## Troubleshooting

### Problem: "Config file not found"
**Solution:** Provide the correct path to your JSON config file.

### Problem: "No input sources resolved"
**Solution:** Check `input_data` in your config matches `input_mode`.

### Problem: Pipeline hangs
**Solution:** Use `--verbose` to see real-time progress. Check network connection for downloads.

### Problem: Missing dependencies
**Solution:** Install required packages:
```bash
pip install numpy  # Optional, for geometry calculations
```

## Performance

- **Parsing**: O(n) single-pass per PDB
- **Cluster detection**: O(m²) where m = number of metals
- **Memory**: Proportional to largest PDB file
- **Parallelization**: Process multiple PDBs by running multiple pipeline instances

## Testing

```bash
# Run with example config
python -m pdb_phase1 --example
python -m pdb_phase1 example_config.json --verbose

# Check outputs
ls pdb_env_outputs/
cat pdb_env_outputs/pipeline.log
```

## License

[Your license here]

## Citation

If you use this pipeline in your research, please cite:
[Your citation here]