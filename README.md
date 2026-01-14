# PDB Scraper Pipeline

A modular, memory-efficient pipeline for searching, downloading, and extracting metal-centered clusters from PDB files.

**Capabilities:**
  1. SEARCH - search the pdb for specific target ligands
  2. DOWNLOAD - download pdb or cif files from id list file or SEARCH
  3. VALIDATE - parse and validate structures fall within validation specification
  4. GENERATE - for each validated structure:
    
  * Extract clusters
  * Extract alternate locations (altloc)
  * Write cropped xyz file for each cluster, centered at each validated target atom site
  5. PREPARE - prepare files for DFT calculations (in progress)

## Project Structure

```
scrape_pdb/
├── __init__.py              # Package initialization
├── __main__.py              # Entry point for `python -m scrape_pdb`
├── altloc.py                # AltLoc handling logic
├── checkpoint.py            # Manage download status using SQLite
├── cluster.py               # Cluster detection & classification
├── config.py                # Configuration loading & validation
├── constants.py             # Global contstants
├── downloader.py            # PDB download functions
├── geometry.py              # Geometry classification & metrics
├── logger.py                # Logging setup
├── main.py                  # Pipeline orchestration
├── models.py                # Data classes (Atom, MustHaveSpec, etc.)
├── parser.py                # PDB parsing functions
├── search.py                # Search PDB using config parameters
├── utils.py                 # General utilities
└── writer.py                # XYZ, CSV, cache writers
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
uv run python -m scrape_pdb example_configs/config.search.yaml --verbose

# Generate example config
uv run python -m scrape_pdb --example
```

### Search Mode (Config-Driven)

Enable `input_mode: "search"` in your config to query RCSB PDB based on your criteria, download matching IDs, and process them with checkpointing and periodic cleanup to manage memory.

Highlights:
- Query by metal ion, resolution cutoff, experimental method, polymer type
- Respect checkpoint to skip previously processed IDs
- Periodic cleanup of temp downloads after each `processing.batch_size`
- Optional saving of validated pdb files
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
python -m scrape_pdb example_configs/config.search.yaml --verbose
```

### Development

```bash
# Add new dependencies
uv add package-name

# Update dependencies
uv sync --upgrade

# Run tests (if you have them)
uv run pytest

# resync uv packages:
uv sync --upgrade
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

### Search Mode Configuration Example (YAML)

You can provide a YAML config to enable search mode (see `example_configs/config.search.yaml`).

```yaml
search_parameters:
  metal_ion: "ZN"
  resolution_cutoff: 2.0  #angstroms
  experimental_method: "X-RAY DIFFRACTION"
  polymer_type: null  # null := all; or use "polypeptide(L)"

processing:
  batch_size: 10
  parallel_workers: 4
  rate_limit_delay: 0.1
  temp_directory: "./data/PDB-downloads"
  max_downloads: 30
  cutoff: 3.0
  selection_radius: 6.0
  target: "ZN"
  metals_excluded: []
  must_have: "S>=4"
  include_waters: true

output:
  results_database: "./data/results/summary.db"
  checkpoint_file: "./data/results/checkpoint.db"
  log_file: "./data/pipeline.log"
  save_matching_structures: true
  matched_structures_dir: "./data/results/validated_structures"
  output_dir: "./data/output"

validation:
  coordination_distance_max: 2.8
  coordination_distance_min: 2.0
  coord: "4S"

# Enable search mode
input_mode: "search"
log_level: "INFO"
```

Run it:
```bash
uv run python -m scrape_pdb examples/config.search.yaml
```

## Configuration File Parameters

<!-- Update below here HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH -->

### Search Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metal_ion` | string | "Zn" |  |
| `resolution_cutoff` | float | 2.0 | Minimum allowed resolution of experiment, in Ångströms |
| `experimental_method` | string | [] | The experimental method for obtaining structure, either "X-RAY DIFFRACTION" or "SOLUTION NMR"
|
| `polymer_type` | string | "null" | Possible filter by polymer type, can be ... |

### Processing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | float | . | . |
| `parallel_workers` | int | . | . |
| `rate_limit_delay` | float | . | . |
| `temp_directory` | string | . | . |
| `max_downloads` | int | . | . |
| `cutoff` | float | . | . |
| `selection_radius` | float | . | . |
| `target` | string | . | . |
| `metals_excluded` | list | [] | . |
| `must_have` | string | . | . |
| `include_waters` | bool | . | . |

#### Must-Have Filter

The validation stage checks that the `must_have` specification is inside the coordination sphere determined by the `selection_radius` parameter. For example, if `selection_radius` is 6 Ångströms, and `must_have` is `S=4`, the validation checks that there are exactly 4 sulfur atoms within 6 Ångströms of the target metal.


Here are some examples:
- `"S"` - Require at least one sulfur
- `"S,N"` - Require sulfur OR nitrogen (ANY mode)
- `"ALL:S,O"` - Require both sulfur AND oxygen
- `"S>=2,N>=1"` - Require at least 2 sulfurs and 1 nitrogen

### Validation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `coordination_distance_max` | float | . | . |
| `coordination_distance_max` | float | . | . |
| `coord` | string or list | `null` | Allowed COORD strings (e.g., `4S`, `1N3S`). Only clusters/XYZ with a matching COORD are retained; others are skipped (altloc XYZ files are removed if filtered out). Omit or set to `null` to accept all. |

### Input Modes

| Mode | Description | `input_data` Format |
|------|-------------|---------------------|
| `ids` | Download from RCSB | List of PDB IDs: `["1ubq", "2qmt"]` |
| `paths` | Local files | List of file paths: `["path/to/1ubq.pdb"]` |
| `folder` | Scan directory | Single folder path: `["./pdbs/"]` |
| `list_file` | Read IDs from file | Path to text file: `["pdb_list.txt"]` |
| `mixed` | Mix of IDs and paths | Combined list |
| `search` | Discover IDs via RCSB query | Use search parameters in config |


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

### 6. Checkpoint Database
`checkpoint_file` keeps track of all PDB IDs processed and their status, i.e., whether they have been download, validated, or rejected.

### 7. Kept PDB files (Optional)
`matched_structures_dir/` has all PDB files that have passed validation, i.e., produced cluster xyz files.

