# PDB Scraper Pipeline

A modular, production-ready pipeline for searching, downloading, and extracting metal-centered clusters from PDB files.

**Current capabilities:**
  1. Download pdb or cif files for each PDB ID from list
  2. Parse .pdb or .cif files
  - Extarct clusters
  - Extract alternate locations
  3. Generate a separate xyz file for each target atom site
  4. Search the PDB for candidate IDs based on configuration, then manage downloads with checkpointing and periodic cleanup (new)

**Future capabilities:**
  1. Prepare xyz files for DFT relxation
  - Center target atom at origin
  - Add hydrogens
  - Remove floating structures
  2. Make pipeline modular enough to be able to scrape and parse other databases, like PubChem, to extract xyz files for DFT calculations

## Features

- **Modular Architecture**: Separated concerns into focused modules
- **Comprehensive Logging**: All output goes to log files with structured messages
- **Context Managers**: Proper resource handling and error management
- **Type Hints**: Full type annotations for better code quality
- **Configuration-Driven**: JSON or YAML configuration with validation
- **Error Recovery**: Graceful handling of failures with detailed logging
- **Search Mode (new)**: Use config to query RCSB PDB and process discovered IDs
- **Checkpointing & Cleanup (new)**: Resume safely and periodically clean temp files based on batch size

## Project Structure

```
scrape_pdb/
├── __init__.py              # Package initialization
├── __main__.py              # Entry point for `python -m scrape_pdb`
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
uv run python -m scrape_pdb config.json

# With verbose output
uv run python -m scrape_pdb config.json --verbose

# Generate example config
uv run python -m scrape_pdb --example
```

### Search Mode (Config-Driven)

Enable `input_mode: "search"` in your config to query RCSB PDB based on your criteria, download matching IDs, and process them with checkpointing and periodic cleanup to manage memory. See `search-spec.md` for the full specification.

Highlights:
- Query by metal ion, resolution cutoff, experimental method, polymer type
- Optional pre-filtering by metadata (e.g., number of metal ions, residue presence)
- Respect checkpoint to skip previously processed IDs
- Rate-limited downloads to `processing.temp_directory` with parallel workers
- Periodic cleanup of temp downloads after each `processing.batch_size`
- Results stored and IDs marked as `matched`, `rejected`, or `error`

Example:
```bash
uv run python -m scrape_pdb config.yaml --verbose
```

#### Alternative: Manual activation

```bash
# Activate the virtual environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Run directly
python -m scrape_pdb config.json
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

#### Testing
- Tests live under `tests/`.
- Run the full suite:
```bash
uv run pytest tests/
```
- Run a specific local pipeline test:
```bash
uv run pytest tests/test_run_and_compare_local.py -q -s
```

#### Dev modules (uv)
- Development-only dependencies are managed with `uv` and recorded in `pyproject.toml`.
- Add dev deps:
```bash
uv add --dev <package>
uv sync
```

## Quick Start

### 1. Create a configuration file

```bash
uv run python -m scrape_pdb --example
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
uv run python -m scrape_pdb config.json

# With verbose console output
uv run python -m scrape_pdb config.json --verbose

# Or use the main module directly
uv run python -m scrape_pdb.main config.json
```

### Search Mode Configuration Example (YAML)

You can provide a YAML config to enable search mode (see `search-spec.md`).

```yaml
search_parameters:
  metal_ion: "ZN"
  coordinating_residues: ["CYS"]
  coordination_count: 4
  resolution_cutoff: 2.0
  experimental_method: "X-RAY DIFFRACTION"
  polymer_type: "Protein"

processing:
  batch_size: 100
  parallel_workers: 4
  rate_limit_delay: 0.1
  temp_directory: "./data/PDB-downloads"

output:
  results_database: "./data/results/zn_cys4.db"
  checkpoint_file: "./data/results/checkpoint.db"
  log_file: "./data/pipeline.log"
  save_matching_structures: false
  matched_structures_dir: "./data/results/matched_cifs"

validation:
  coordination_distance_max: 2.8
  coordination_distance_min: 2.0
  geometric_criteria: {}
  coord: "4S"

# Enable search mode (the pipeline will perform the query and process IDs)
input_mode: "search"
```

Run it:
```bash
uv run python -m scrape_pdb examples/config.search.yaml
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

### Validation Parameters

- `coord` (string or list): Allowed COORD strings (e.g., `4S`, `1N3S`). Only clusters/XYZ with a matching COORD are retained; others are skipped (altloc XYZ files are removed if filtered out). Omit or set to `null` to accept all.

### Input Modes

| Mode | Description | `input_data` Format |
|------|-------------|---------------------|
| `ids` | Download from RCSB | List of PDB IDs: `["1ubq", "2qmt"]` |
| `paths` | Local files | List of file paths: `["path/to/1ubq.pdb"]` |
| `folder` | Scan directory | Single folder path: `["./pdbs/"]` |
| `list_file` | Read IDs from file | Path to text file: `["pdb_list.txt"]` |
| `mixed` | Mix of IDs and paths | Combined list |
| `search` | Discover IDs via RCSB query | Use search parameters in config; downloads managed with checkpointing and cleanup (new) |

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

## Search & Processing Architecture (Overview)

For a detailed specification of search behavior and processing phases, see `search-spec.md`.
- Phase 1: RCSB search (metadata pre-filtering planned), checkpoint init
- Phase 2: Stream processing with status tracking, download → parse → analyze → store → cleanup
- Parallel workers with global rate limiting (planned); periodic batch checkpointing
- Phase 3: Results aggregation, reporting, and optional visualization export

See the implementation details and roadmap in [search-spec.md](search-spec.md) under "Implementation Status".

## Implementation Status

- **Implemented:** RCSB search mode; coord-based filtering of clusters/XYZ retention (`validation.coord`); checkpoint skip of matched/rejected with retry of error/in_progress; download fallbacks (.pdb → .pdb.gz → .cif.gz) into `processing.temp_directory`; batch cleanup after `processing.batch_size`; serial processing; outputs (clusters summary CSV, altloc report CSV, cache.json); optional `processing.max_downloads` for tests.
- **Planned:** Parallel workers with coordinated rate limiting, metadata pre-filtering before parse, matched CIF retention/export, additional output formats/visualizations.
- Details: see [search-spec.md](search-spec.md) "Implementation Status" section.

## Examples

- Example search-mode config: see [examples/config.search.yaml](examples/config.search.yaml).
- Customize it by adjusting:
  - **search_parameters**: set `metal_ion`, `resolution_cutoff`, `experimental_method`, `polymer_type`, and optional residue criteria.
  - **processing**: tune `batch_size`, `parallel_workers`, `rate_limit_delay`, `temp_directory`, analysis `cutoff`, and `target` metal. 
  - **output**: choose `results_database`, `checkpoint_file`, `log_file`, `matched_structures_dir`, and `output_dir`.
  - **validation**: set coordination distance thresholds and optional geometric criteria.

Run the example:
```bash
uv run python -m scrape_pdb examples/config.search.yaml --verbose
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