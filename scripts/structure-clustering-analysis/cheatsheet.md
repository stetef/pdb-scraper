# Structure Clustering Analysis — Quick Reference

All commands are run from the repo root with `uv run python`.
Scripts live in `scripts/structure-clustering-analysis/`.

---

## One-command orchestrator (recommended)

```bash
# Full dataset — runs cleanup → approaches 1 and 3 → validation → comparison
uv run python scripts/structure-clustering-analysis/00_orchestrate.py \
    --base-dir data/large-cys-his-datasets/4cys-large

# Include approach 2 (pure numpy; no extra dependencies)
uv run python scripts/structure-clustering-analysis/00_orchestrate.py \
    --base-dir data/large-cys-his-datasets/4cys-large \
    --approaches 1 2 3

# Small test dataset (k-min/max adjusted for 22 structures)
uv run python scripts/structure-clustering-analysis/00_orchestrate.py \
    --base-dir data/test-4cys \
    --xyz-subdir initial_xyz_files \
    --skip-cleanup \
    --approaches 1 3 \
    --k-min 2 --k-max 8

# Skip cleanup if xyz_files_cleaned already exists
uv run python scripts/structure-clustering-analysis/00_orchestrate.py \
    --base-dir data/large-cys-his-datasets/4cys-large \
    --xyz-subdir output/xyz_files_cleaned \
    --skip-cleanup

# Force re-run everything
uv run python scripts/structure-clustering-analysis/00_orchestrate.py \
    --base-dir data/large-cys-his-datasets/4cys-large \
    --force
```

---

## Step-by-step

### 01 — H cleanup

```bash
uv run python scripts/structure-clustering-analysis/01_cleanup_xyz_hydrogens.py \
    data/large-cys-his-datasets/4cys-large/output/xyz_files \
    --preview-dir data/large-cys-his-datasets/4cys-large/cleaned
```

### 02 — Approach 1: aligned Cartesian

```bash
uv run python scripts/structure-clustering-analysis/02_approach1_cartesian.py \
    --xyz-dir  data/large-cys-his-datasets/4cys-large/cleaned \
    --out-dir  data/large-cys-his-datasets/4cys-large/approach1 \
    --k-min 10 --k-max 40
```

Outputs: `labels.csv`, `medoids.csv`, `k_sweep.csv`, `aligned_xyz/`, `r0_id.txt`

### 03 — Approach 2: Z-matrix (pure numpy; no extra dependencies)

```bash
# From approach 1 aligned structures (recommended — avoids recomputing R₀)
uv run python scripts/structure-clustering-analysis/03_approach2_zmatrix.py \
    --aligned-xyz-dir data/large-cys-his-datasets/4cys-large/approach1/aligned_xyz \
    --out-dir         data/large-cys-his-datasets/4cys-large/approach2 \
    --k-min 10 --k-max 40

# Standalone (re-computes R₀ from raw XYZ)
uv run python scripts/structure-clustering-analysis/03_approach2_zmatrix.py \
    --xyz-dir  data/large-cys-his-datasets/4cys-large/cleaned \
    --out-dir  data/large-cys-his-datasets/4cys-large/approach2
```

### 04 — Approach 3: PIV

```bash
uv run python scripts/structure-clustering-analysis/04_approach3_piv.py \
    --xyz-dir  data/large-cys-his-datasets/4cys-large/cleaned \
    --out-dir  data/large-cys-his-datasets/4cys-large/approach3 \
    --k-min 10 --k-max 40
```

### 05 — Validate one approach

```bash
BASE=data/large-cys-his-datasets/4cys-large

# Approach 1 — RMSD metrics + PCA→XYZ reconstruction
uv run python scripts/structure-clustering-analysis/05_validate_clusters.py \
    --approach-dir $BASE/approach1 \
    --xyz-dir      $BASE/approach1/aligned_xyz \
    --approach1 \
    --out-dir      $BASE/validation/approach1

# Approach 3 — RMSD metrics
uv run python scripts/structure-clustering-analysis/05_validate_clusters.py \
    --approach-dir $BASE/approach3 \
    --xyz-dir      $BASE/cleaned \
    --out-dir      $BASE/validation/approach3
```

### 05 — Cross-approach comparison

```bash
uv run python scripts/structure-clustering-analysis/05_validate_clusters.py \
    --compare-dirs \
        data/large-cys-his-datasets/4cys-large/approach1 \
        data/large-cys-his-datasets/4cys-large/approach3 \
    --out-dir data/large-cys-his-datasets/4cys-large/validation/comparison
```

---

## Standard output format (all approach scripts)

```
{out_dir}/
  labels.csv              structure_id, cluster
  medoids.csv             cluster_id, medoid_id
  k_sweep.csv             k, intra, inter, ratio, ch_score
  per_cluster_intra.csv   cluster_id, mean_intra_rmsd
  *_run.log               detailed run log

approach1 additionally:
  aligned_xyz/            one 13-atom XYZ per structure (R₀ residue order)
  r0_id.txt               stem of the R₀ reference structure
```

**k selection criterion:** best k is the one that maximizes the CH-analogue score:
`ch_score = (inter² / (k−1)) / (intra² / (N−k))`.
This penalizes overly fine-grained k; the raw `ratio = inter/intra` is also recorded but
trends upward with k and is not used for selection.

---

## Key parameters

| Parameter | Default | Where |
|---|---|---|
| `--k-min` | 10 | 02, 03, 04 |
| `--k-max` | 40 | 02, 03, 04 |
| `--k-step` | 2 | 02, 03, 04 |
| `--convergence-tol` | 0.005 | 02 (R₀ iteration) |
| `--max-ref-iter` | 10 | 02 (R₀ iteration) |
| `--no-reflection` | off | 02, 03, 04 (off = merge enantiomers) |
| `--approaches` | 1 3 | 00 |
| `--force` | off | 00 |
