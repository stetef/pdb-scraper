# PDB Scraper Pipeline

A modular, production-ready pipeline for extracting metal-centered clusters from PDB files.

**Current capabilities:**
  1. Download pdb or cif files for each PDB ID from list
  2. Parse .pdb or .cif files
  - Extarct clusters
  - Extract alternate locations
  3. Generate a separate xyz file for each target atom site

**Future capabilities:**
  1. Search pdb for candidate IDs rather than from a predetermined list
  2. Prepare xyz files for DFT relxation
  - Center target atom at origin
  - Add hydrogens
  - Remove floating structures
  3. Make pipeline modular enough to be able to scrape and parse other databases, like PubChem, to extract xyz files for DFT calculations

## Features

- **Modular Architecture**: Separated concerns into focused modules
- **Comprehensive Logging**: All output goes to log files with structured messages
- **Context Managers**: Proper resource handling and error management
- **Type Hints**: Full type annotations for better code quality
- **Configuration-Driven**: JSON-based configuration with validation
- **Error Recovery**: Graceful handling of failures with detailed logging

## Project Structure

```
scrape-pdb/
├── __init__.py              # Package initialization
├── __main__.py              # Entry point for `python -m scrape-pdb`
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

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

#### 1. Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

#### 2. Install dependencies

```bash
# From project root directory
uv sync
```

This will create a virtual environment and install all required dependencies from `pyproject.toml`.

### Usage

#### Run the pipeline

```bash
# Using uv run (recommended)
uv run python -m scrape-pdb config.json

# With verbose output
uv run python -m scrape-pdb config.json --verbose

# Generate example config
uv run python -m scrape-pdb --example
```

#### Alternative: Manual activation

```bash
# Activate the virtual environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Run directly
python -m scrape-pdb config.json
```

### Development

```bash
# Add new dependencies
uv add package-name

# Update dependencies
uv sync --upgrade

# Run tests (if you have them)
uv run pytest
```

## Quick Start

### 1. Create a configuration file

```bash
uv run python -m scrape-pdb --example
```

This creates `example_config.json`:

```json
{
  "cutoff": 5.0,
  "target": "NI",
  "metals_excluded": [],
  "must_have": "",
  "include_waters": true,
  "input_mode": "list_file",
  "input_data": ["data/ids.txt"],
  "download_dir": "data/PDB-downloads",
  "output_dir": "data/output",
  "cache": "data/cache.json",
  "log_file": "data/pipeline.log",
  "log_level": "INFO",
}
```

### 2. Run the pipeline

```bash
# Basic usage
uv run python -m scrape-pdb config.json

# With verbose console output
uv run python -m scrape-pdb config.json --verbose

# Or use the main module directly
uv run python -m scrape-pdb.main config.json
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
Located in `output_dir/xyz_files/`, one per cluster. For example:
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
`cache.json` stores run manifests for reproducibility.

### 5. Log File
`output_dir/pipeline.log` contains all pipeline activity.


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

## Troubleshooting

### Problem: "Config file not found"
**Solution:** Provide the correct path to your JSON config file.

### Problem: "No input sources resolved"
**Solution:** Check `input_data` in your config matches `input_mode`.

### Problem: Pipeline hangs
**Solution:** Use `--verbose` to see real-time progress. Check network connection for downloads.

### Problem: Missing dependencies
**Solution:** resync uv packages:
```bash
uv sync --upgrade
```

## Performance

- **Parsing**: O(n) single-pass per PDB
- **Cluster detection**: O(m²) where m = number of metals
- **Memory**: Proportional to largest PDB file
- **Parallelization**: Process multiple PDBs by running multiple pipeline instances

## Citation

If you use this pipeline in your research, please cite:
[Your citation here]