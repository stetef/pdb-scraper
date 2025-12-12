# Complete File Organization Guide

## Quick Answer to Your Questions

### 1. Must-Have Elements Filter (Step 11) → `models.py`
- ✅ `class MustHaveSpec` (already in models.py)
- ✅ `def parse_must_have(text: str) -> MustHaveSpec` (already in models.py)

### 2. All Dataclasses → `models.py`
- ✅ `class Atom`
- ✅ `class MustHaveSpec`
- ✅ `class GeometryResult`
- ✅ `class ClusterInfo`

### 3. Constants (`USER_AGENT`, `ALL_METALS`) → `constants.py` (NEW FILE)
- `USER_AGENT`
- `ALL_METALS`
- `DEFAULT_*` values
- `PDB_RECORD_COLUMNS`
- `WATER_RESIDUES`
- `GEOMETRY_THRESHOLDS`
- CSV field names

### 4. Complete `models.py` Contents
See the updated artifact above - it now includes:
- All 4 dataclasses
- MustHaveSpec logic
- parse_must_have() function
- Helper functions like coord_string_from_atoms()

---

## Complete Module Breakdown

### 📦 Package Structure

```
pdb_phase1/
├── __init__.py              # Package exports
├── __main__.py              # Entry point: python -m pdb_phase1
├── constants.py             # ⭐ NEW - All constants
├── models.py                # ✅ COMPLETE - All data models
├── utils.py                 # ✅ COMPLETE - General utilities
├── config.py                # ✅ COMPLETE - Config management
├── logger.py                # ✅ COMPLETE - Logging setup
├── main.py                  # ✅ COMPLETE - Pipeline orchestration
├── parser.py                # ⏳ TODO - Extract from original
├── geometry.py              # ⏳ TODO - Extract from original
├── cluster.py               # ⏳ TODO - Extract from original
├── altloc.py                # ⏳ TODO - Extract from original
├── writer.py                # ⏳ TODO - Extract from original
└── downloader.py            # ⏳ TODO - Extract from original
```

---

## Detailed File Contents

### ✅ `constants.py` (NEW - COMPLETE)
```python
# Network
USER_AGENT = "pdb-env-tool/1.0 (+https://rcsb.org)"

# Metals
ALL_METALS = {"LI", "NA", "K", ..., "FE", "CO", "NI", ...}

# Defaults
DEFAULT_CUTOFF = 5.0
DEFAULT_TARGET = "NI"
DEFAULT_DOWNLOAD_DIR = "PDB"
DEFAULT_OUTPUT_DIR = "pdb_env_outputs"

# PDB Format
PDB_RECORD_COLUMNS = {...}
WATER_RESIDUES = {"HOH", "WAT"}

# Thresholds
GEOMETRY_THRESHOLDS = {...}

# CSV fields
CLUSTERS_CSV_FIELDS = [...]
ALTLOC_REPORT_FIELDS = [...]
```

**Import in other files:**
```python
from .constants import ALL_METALS, USER_AGENT, WATER_RESIDUES
```

---

### ✅ `models.py` (COMPLETE)

**Contains:**
1. **Atom dataclass** - PDB atom representation
2. **MustHaveSpec class** - Element filter logic (Step 11)
3. **parse_must_have()** - Parse filter strings
4. **GeometryResult dataclass** - Geometry classification results
5. **ClusterInfo dataclass** - Cluster metadata
6. **coord_string_from_atoms()** - Helper function

**What it does NOT contain:**
- ❌ Parsing logic (goes in parser.py)
- ❌ Geometry calculations (goes in geometry.py)
- ❌ File I/O (goes in writer.py)

---

### ✅ `utils.py` (COMPLETE)

**Contains:**
```python
def dist(a, b) -> float
def dist_squared(a, b) -> float
def centroid(points) -> Tuple[float, float, float]
def connected_components(points, cutoff) -> List[List[int]]
def coord_string(elements) -> str
def format_other_metals(metals) -> str
def safe_float(value, default) -> float
def safe_int(value, default) -> int
```

Pure utility functions with no side effects.

---

### ✅ `config.py` (COMPLETE)

**Contains:**
```python
class PipelineConfig - dataclass for all config
def load_config(path) -> PipelineConfig
def print_config_summary(config) -> None
def create_example_config(output_path) -> None
```

Handles JSON loading and validation.

---

### ✅ `logger.py` (COMPLETE)

**Contains:**
```python
def setup_logger(name, log_file, level) -> logging.Logger
class PipelineLogger - context manager for PDB processing
```

All logging infrastructure.

---

### ✅ `main.py` (COMPLETE)

**Contains:**
```python
def run_pipeline(config_path, verbose) -> int
def main() - CLI entry point
```

High-level orchestration only.

---

### ⏳ `parser.py` (TODO - Extract from original)

**Should contain:**
```python
def _slice(line, a, b) -> str
def parse_pdb_atom_line(line) -> Optional[Atom]
def load_pdb_atoms_all(path, first_model_only) -> List[Atom]
def collapse_altloc(atoms, policy) -> List[Atom]
def parse_pdb_resolution(path) -> Optional[float]
def group_by_id(atoms) -> Dict
def altloc_set_for_atom_id(G, key) -> Set[str]
def iter_altloc_metal_records(pdb_path, metal_set) -> Iterable[Atom]
def process_pdb(pdb_path, config, logger) -> List[str]  # MAIN FUNCTION
```

**Import at top:**
```python
from .models import Atom
from .constants import PDB_RECORD_COLUMNS, ALL_METALS
from .utils import dist, connected_components
import logging

logger = logging.getLogger(__name__)
```

---

### ⏳ `geometry.py` (TODO - Extract from original)

**Should contain:**
```python
def _angle(a, o, b) -> float
def _planarity_metrics(points) -> Tuple[float, float]
def classify_geometry(center, neighbors) -> GeometryResult
```

**Import at top:**
```python
from .models import Atom, GeometryResult, coord_string_from_atoms
from .constants import GEOMETRY_THRESHOLDS
from .utils import dist
import math
import logging

logger = logging.getLogger(__name__)

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False
    logger.warning("NumPy not available - planarity metrics disabled")
```

---

### ⏳ `cluster.py` (TODO - Extract from original)

**Should contain:**
```python
def determine_cluster_type(metals, target_upper, target_element) -> str
def select_neighbors_union(atoms, centers, cutoff) -> List[Atom]
def select_neighbors_from(center, atoms, cutoff) -> List[Atom]
def apply_water_toggle(atoms, include_waters) -> List[Atom]
```

**Import at top:**
```python
from .models import Atom
from .constants import WATER_RESIDUES
from .utils import dist
import logging

logger = logging.getLogger(__name__)
```

---

### ⏳ `altloc.py` (TODO - Extract from original)

**Should contain:**
```python
def build_altloc_files_for_center(
    pdb_id, center, selected_atoms_raw, raw_groups,
    cutoff, out_dir, base_name_common, origin_kind,
    centroid_pt, resolution_angs, cluster_index,
    cluster_type, target_upper, include_waters,
    must_have, csv_common, metals_in_comp
) -> List[str]
```

This implements Step 6 (Situations 1-3).

**Import at top:**
```python
from .models import Atom, MustHaveSpec
from .writer import write_xyz, write_clusters_csv_row
from .cluster import select_neighbors_from, apply_water_toggle
from .geometry import classify_geometry
from .utils import dist
import logging

logger = logging.getLogger(__name__)
```

---

### ⏳ `writer.py` (TODO - Extract from original)

**Should contain:**
```python
def write_xyz(path, pdb_id, cluster_index, target, cutoff,
              origin_kind, centroid_pt, atoms, origin_atom,
              resolution_angs, extra_comment) -> None

def ensure_csv_headers(csv_path, fields) -> None
def write_clusters_csv_row(row, csv_path) -> None
def write_altloc_report_header(csv_path) -> None
def append_altloc_rows(rows, csv_path) -> None
def append_cache(runs, cache_path) -> None
```

**Import at top:**
```python
from .models import Atom
from .constants import CLUSTERS_CSV_FIELDS, ALTLOC_REPORT_FIELDS
from pathlib import Path
import csv
import json
import logging

logger = logging.getLogger(__name__)
```

**Key change:** Pass paths as parameters instead of using globals.

---

### ⏳ `downloader.py` (TODO - Extract from original)

**Should contain:**
```python
def fetch_pdb(pdb_id, dest_dir) -> Optional[str]
def batch_download_from_list(listfile, outdir) -> List[str]
def resolve_input_sources(config) -> List[str]
```

**Import at top:**
```python
from .constants import USER_AGENT
from .config import PipelineConfig
from pathlib import Path
import urllib.request
import gzip
import shutil
import glob
import re
import logging

logger = logging.getLogger(__name__)
```

---

## Migration Checklist

### Step 1: Create New Files ✅
- [x] constants.py - DONE
- [x] Update models.py - DONE
- [x] Update __init__.py - DONE

### Step 2: Extract from Original Script
- [ ] parser.py - Extract parsing functions
- [ ] geometry.py - Extract geometry classification
- [ ] cluster.py - Extract cluster detection
- [ ] altloc.py - Extract altloc handling
- [ ] writer.py - Extract file writers
- [ ] downloader.py - Extract download functions

### Step 3: Update Imports
For each new file, replace:
```python
# OLD (in original script)
OUTPUT_DIR = "pdb_env_outputs"
print(f"Processing {pdb_id}")

# NEW (in new modules)
from .config import PipelineConfig
logger.info(f"Processing {pdb_id}")
# Use config.output_dir instead of OUTPUT_DIR
```

### Step 4: Update Function Signatures
```python
# OLD
def process_pdb(pdb_path, cutoff, target_atom_name, 
                include_waters_by_default, metals_excluded, must_have):

# NEW
def process_pdb(pdb_path: str, config: PipelineConfig, 
                logger: logging.Logger) -> List[str]:
```

---

## Example: Extracting `parser.py`

Here's how to extract the parsing functions:

**1. Copy these functions from original script:**
- `_slice()`
- `parse_pdb_atom_line()`
- `load_pdb_atoms_all()`
- `collapse_altloc()`
- `parse_pdb_resolution()`
- `group_by_id()`
- `altloc_set_for_atom_id()`
- `iter_altloc_metal_records()`
- `process_pdb()` ← THE BIG ONE

**2. Add imports at top:**
```python
from typing import Optional, List, Dict, Set, Iterable, Tuple
from collections import defaultdict
import logging

from .models import Atom
from .constants import ALL_METALS, PDB_RECORD_COLUMNS, WATER_RESIDUES
from .utils import dist, connected_components, centroid
from .config import PipelineConfig

logger = logging.getLogger(__name__)
```

**3. Update `process_pdb()` signature:**
```python
def process_pdb(
    pdb_path: str,
    config: PipelineConfig,
    logger: logging.Logger
) -> List[str]:
    """
    Process a single PDB file and extract metal clusters.
    
    Args:
        pdb_path: Path to PDB file
        config: Pipeline configuration
        logger: Logger instance
    
    Returns:
        List of paths to written XYZ files
    """
    # Replace print() with logger.info()
    # Use config.cutoff instead of cutoff parameter
    # Use config.target instead of target_atom_name
    # Use config.metals_excluded instead of metals_excluded
    # etc.
```

**4. Replace all `print()` statements:**
```python
# OLD
print(f"[=] Processing {base_id} …")

# NEW
logger.info(f"Processing {base_id}")
```

**5. Use config instead of parameters:**
```python
# OLD
metal_set = set(ALL_METALS) - metals_excluded

# NEW
metal_set = set(ALL_METALS) - config.metals_excluded
```

---

## Testing Each Module

After extracting each module, test it:

```python
# Test parser.py
from pdb_phase1.parser import load_pdb_atoms_all
atoms = load_pdb_atoms_all("test.pdb")
print(f"Loaded {len(atoms)} atoms")

# Test geometry.py
from pdb_phase1.geometry import classify_geometry
result = classify_geometry(center, neighbors)
print(f"Geometry: {result.label}, CN: {result.coordination_number}")

# Test constants.py
from pdb_phase1.constants import ALL_METALS
print(f"Total metals defined: {len(ALL_METALS)}")
```

---

## Summary

| Question | Answer |
|----------|--------|
| Where does MustHaveSpec go? | ✅ `models.py` (already there) |
| Where do dataclasses go? | ✅ `models.py` (all 4 already there) |
| Where do USER_AGENT and ALL_METALS go? | ✅ `constants.py` (new file created) |
| What's in models.py? | ✅ Atom, MustHaveSpec, GeometryResult, ClusterInfo + helpers |

Everything is now clearly organized and documented!