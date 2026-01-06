# PDB Metalloprotein Pipeline Specification

## Overview
A memory-efficient, fault-tolerant pipeline for searching, downloading, and analyzing metalloprotein structures from the RCSB PDB based on configurable criteria.

---

## Configuration File Schema (Current Implementation)

```yaml
search_parameters:
  metal_ion: string              # e.g., "ZN", "FE", "MG"
  resolution_cutoff: float       # maximum resolution in Angstroms
  experimental_method: string    # e.g., "X-RAY DIFFRACTION", "ALL"
  polymer_type: string           # e.g., "Protein", "DNA", "ALL"

processing:
  batch_size: int                # structures processed before cleanup
  parallel_workers: int          # planned; current implementation processes serially
  rate_limit_delay: float        # planned global rate limit; currently not enforced
  temp_directory: string         # path for temporary file storage (download dir)
  cutoff: float                  # analysis distance cutoff for clustering (Å)
  target: string                 # target metal atom name (e.g., "ZN")
  metals_excluded: list          # metals to exclude from analysis
  must_have: string              # element/residue presence specification for validation
  include_waters: bool           # include water molecules in analysis
  max_downloads: int | null      # optional cap on downloads per run (dev/testing)
  
output:
  results_database: string       # path to CSV file (current default)
  checkpoint_file: string        # path to progress tracking file
  log_file: string              # path to log file
  save_matching_structures: bool # planned; currently not enforced during cleanup
  matched_structures_dir: string # planned directory for matched CIFs
  output_dir: string             # base directory for CSV/XYZ outputs
  altloc_report_file: string     # optional explicit path for altloc report

validation:
  coordination_distance_max: float    # max distance for coordination bond (Angstroms)
  coordination_distance_min: float    # min distance for coordination bond (Angstroms)
  geometric_criteria: dict            # optional additional geometry checks
  coord: string | list                # optional allowed COORD strings (e.g., "4S", "1N3S"); others are discarded
```

---

## Pipeline Architecture

### Phase 1: Search & ID Resolution

**Input**: Configuration parameters  
**Output**: List of candidate PDB IDs

1. **Initial Search**
  - Query RCSB PDB Search API v2 with:
    - Metal ion (exact match on `rcsb_nonpolymer_entity_instance_container_identifiers.comp_id`)
    - Resolution cutoff (≤ on `rcsb_entry_info.resolution_combined`)
    - Experimental method (exact match on `exptl.method`, if not `ALL`)
    - Polymer type (exact match on `entity_poly.rcsb_entity_polymer_type`, if not `ALL`)
  - Retrieve list of PDB IDs (`return_all_hits: true`)

  Note: Enable `input_mode: "search"` in the configuration to run the query automatically. Returned
  IDs are downloaded into `processing.temp_directory` and processed like manually provided IDs.

2. **Pre-filtering**
   - Current implementation does not perform a separate metadata pre-filter step; filtering occurs
     during parsing/validation (e.g., `must_have`, distance `cutoff`, target metal).

3. **Progress Initialization**
   - Check checkpoint file for previously processed IDs
   - Create/update progress tracking database
   - Mark remaining IDs as "pending"

Behavioral notes (Current):
- If `input_mode` is `search`, IDs returned by the search are filtered against the checkpoint so
  previously `matched` or `rejected` IDs are skipped. `error` and `in_progress` entries are retried.
- The download step stores files into `processing.temp_directory`. After each `processing.batch_size`
  processed structures, the pipeline cleans that directory by removing all files (matched CIFs are
  not preserved yet).
- Use `processing.max_downloads` to limit downloads per run (useful in tests/dev).

---

### Phase 2: Stream Processing with Checkpointing

**Input**: Filtered PDB ID list  
**Output**: Results database with coordination analysis

#### Per-Structure Workflow

For each PDB ID (serial processing in current implementation):

1. **Status Check**
   - Query checkpoint: skip if already processed
   - Mark as "in_progress" in checkpoint

2. **Download**
  - Fetch `.pdb` (fallback to `.pdb.gz`, then `.cif.gz`) to temporary directory
  - Global rate limiting is planned; `rate_limit_delay` is not enforced yet
  - Download errors are logged and marked `error` (no built-in retry loop)

3. **Parse**
   - Extract metal ion coordinates
   - Extract potential coordinating residue coordinates
   - Parse resolution, experimental method, metadata

4. **Analyze Coordination**
  - Calculate distances between metal and candidate residues
  - Filter by coordination distance thresholds
  - Apply configuration criteria (`cutoff`, `target`, `must_have`, etc.)
  - Validate residue types and counts during analysis (post-parse), not during search

5. **Store Results**
   - **If match**: 
     - Persist coordination data in summary CSVs and cache
     - Mark as "matched" in checkpoint
   - **If no match**:
     - Log rejection reason (e.g., "wrong_count", "no_metal", "wrong_residues")
     - Mark as "rejected" in checkpoint
   - **If error**:
     - Log error details
     - Mark as "error" in checkpoint

6. **Cleanup**
  - Batch cleanup: after `batch_size` processed structures, delete all files in the download directory
  - Matched CIF retention is planned but not currently implemented

7. **Batch Checkpoint/Cleanup**
  - Checkpoint updates occur per-structure
  - Download directory cleanup occurs after each batch

#### Parallelization Strategy

- Use worker pool with configurable worker count
- Each worker processes one structure at a time
- Shared queue for pending IDs
- Thread-safe checkpoint and results database access
- Coordinate rate limiting across workers (global semaphore/token bucket)

---

### Phase 3: Results Aggregation & Reporting

**Input**: Results database  
**Output**: Summary statistics and filtered dataset

1. **Statistics Generation**
   - Total structures processed
   - Number matched vs rejected vs errors
   - Breakdown of rejection reasons
   - Processing time and throughput

2. **Results Export (Current)**
  - Generate summary report CSVs (clusters summary, altloc report)
  - Write cache manifest (`cache.json`) for reproducibility
  - Export matched CIFs to a folder is planned (not implemented yet)

---

## Data Structures

### Checkpoint Database Schema

```
checkpoints table:
  - pdb_id: string (primary key)
  - status: enum (pending, in_progress, matched, rejected, error)
  - timestamp: datetime
  - rejection_reason: string (nullable)
  - error_message: string (nullable)
```

### Results Database Schema

```
structures table:
  - pdb_id: string (primary key)
  - resolution: float
  - experimental_method: string
  - metal_ion: string
  - metal_count: int
  - deposited_date: date

coordination_sites table:
  - site_id: int (primary key, auto-increment)
  - pdb_id: string (foreign key)
  - metal_atom_id: string
  - residue_types: json/text (count by type, e.g., {"CYS": 4})
  - geometry_metrics: json/text (distances, angles, etc.)
  - validation_passed: boolean
```

---

## Error Handling & Fault Tolerance (Current)

### Retry Logic
- Network errors: logged and marked `error`; no built-in retry loop
- Parse errors: logged and marked `error`
- API rate limiting/backoff: planned; not currently enforced

### Resume Capability
- On restart, entries with status `matched` or `rejected` are skipped.
- `in_progress` and `error` entries are retried (no automatic reset to `pending`).

### Graceful Shutdown
- Handle SIGINT/SIGTERM signals
- Complete current batch processing
- Flush all buffers to disk
- Report partial progress

---

## Progress Tracking & Logging

### Real-time Progress Display
- Use progress bar (tqdm or similar) showing:
  - Structures processed / total
  - Current processing rate (structures/minute)
  - Estimated time remaining
  - Current match rate (%)

### Logging Levels
- **INFO**: Batch completions, matches found, summary statistics
- **WARNING**: Download retries, unusual structures
- **ERROR**: Failed downloads, parse errors, validation failures
- **DEBUG**: Per-structure details, coordination geometry data

### Periodic Status Reports
- Every N structures or M minutes, log:
  - Total processed
  - Matches found
  - Error rate
  - Processing throughput

---

## Performance Targets

- **Memory**: <100 MB peak (excluding parallel worker overhead)
- **Disk**: <50 MB temporary storage (configurable batch size)
- **Throughput**: 10-50 structures/minute (network-dependent)
- **Fault tolerance**: <1% data loss on unexpected termination

---

## Implementation Status

**Implemented**
- Config-driven search mode (`input_mode: "search"`) using RCSB Search API v2 with metal ion, resolution cutoff, experimental method (optional), and polymer type (optional).
- Checkpoint integration: skip IDs with status `matched` or `rejected`; retry `error` and `in_progress` entries.
- Download fallbacks: `.pdb` → `.pdb.gz` → `.cif.gz` stored in `processing.temp_directory`.
- Serial processing in a single process (parallelization planned).
- Post-parse validation and analysis using `cutoff`, `target`, `must_have`, and `include_waters`.
- Batch cleanup: after `processing.batch_size` processed structures, remove all files in the download directory.
- Outputs: clusters summary CSV, altloc report CSV, and `cache.json` manifest.
- Optional `processing.max_downloads` to cap downloads for tests/dev.

**Planned**
- Parallel worker pool (`processing.parallel_workers`) with coordinated global rate limiting (`processing.rate_limit_delay`).
- Optional metadata pre-filtering prior to parse.
- Preservation/export of matched CIFs to `output.matched_structures_dir` when `output.save_matching_structures` is true.
- Additional result stores/formats and visualization exports.

## Configuration Example Reference (YAML)

```yaml
search_parameters:
  metal_ion: "ZN"
  resolution_cutoff: 2.0
  experimental_method: "X-RAY DIFFRACTION"
  polymer_type: "Protein"

processing:
  batch_size: 100
  parallel_workers: 4       # planned; current run is serial
  rate_limit_delay: 0.1     # planned global rate limit
  temp_directory: "./data/PDB-downloads"
  cutoff: 3.0
  target: "ZN"
  metals_excluded: []
  must_have: ""
  include_waters: true
  max_downloads: 10

output:
  results_database: "./data/results/zn_cys4.csv"
  checkpoint_file: "./data/results/checkpoint.db"
  log_file: "./data/pipeline.log"
  save_matching_structures: false
  matched_structures_dir: "./data/results/matched_cifs"
  output_dir: "./data/output"
  altloc_report_file: "./data/output/altloc_report.csv"

validation:
  coordination_distance_max: 2.8
  coordination_distance_min: 2.0
  geometric_criteria: {}
  coord: "4S"
```