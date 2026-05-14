# Clustering-plots pipeline

This pipeline takes a directory of XYZ files and produces:

- clustered labels/stats CSVs
- per-cluster and all-cluster distribution figures
- interactive clustering HTML reports (online + fully offline)

```text
                            input XYZ files
       data/.../output/xyz_files/*.xyz  (compact, Zn-centered)
                                      |
                                      v
        scripts/cleanup_xyz_postzn_hydrogens.py  (H cleanup + CA-cap H repair)
                                      |
                                      v
                 cleaned XYZ files (preview copy OR in-place)
                                      |
                                      v
                scripts/xyz_to_gzmat.py  (XYZ -> .gzmat)
                                      |
                                      v
                data/.../output/xyz_files/*.gzmat
                                      |
                                      v
          scripts/gzmat_flatten_dataset.py  (flatten + features + z-score)
                                      |
                                      v
      gzmat_dataset.cluster_features_zscore.csv   (main clustering input)
                                      |
                                      v
      scripts/cluster_gzmat_features.py  (PCA/UMAP/t-SNE + KMeans/DBSCAN)
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
      <method>_labels_with_stats.csv         kmeans_cluster_stats_summary.csv
      embeddings.csv, tsne_<method>.png      (or <method>_cluster_stats_summary.csv)
                     |                                 |
                     +----------------+----------------+
                                      |
                                      v
  scripts/plot_cluster_label_distributions.py  (distribution figure generation)
                                      |
                                      v
      cluster_distribution_plots/per_cluster_rows/*.png
      cluster_distribution_plots/all_cluster_overlays/*.png
                                      |
                                      v
   scripts/build_cluster_distribution_report.py  (compose HTML deliverables)
                                      |
                                      v
        report_cluster_distribution.html (CDN)
        report_cluster_distribution_offline.html (self-contained)
```

## What each script does

### scripts/cleanup_xyz_postzn_hydrogens.py (run first)

Cleans duplicate hydrogen artifacts when structures already contain PDB-derived H atoms before Zn.

Behavior summary:

1. Detects pre-Zn PDB-style H atoms (metadata tags like RES/CHAIN/RESSEQ).
2. Removes post-Zn auto-generated H atoms that look writer-added.
3. Removes loose/orphan H atoms not plausibly bonded by covalent-radius thresholds.
4. Adds missing H atoms to capped CA carbons (CA/CB present, no backbone N/C) so CA has three attached H atoms.
5. Rewrites XYZ atom count.

Important mode detail:

- Default mode is preview: writes cleaned files to a separate directory (temp unless --preview-dir is provided).
- --apply-in-place modifies source XYZ files directly.

### scripts/xyz_to_gzmat.py

Converts every .xyz file in a directory to a matching Gaussian-style .gzmat using chemcoord.

- Output: sibling files with the same stem and .gzmat extension.
- Supports recursive traversal via -r/--recursive.

### scripts/gzmat_flatten_dataset.py

Builds clustering-ready numeric tables from .gzmat files.

Outputs:

- <prefix>.raw_flat.csv
- <prefix>.cluster_features.csv
- <prefix>.cluster_features_zscore.csv

Feature encoding used for clustering table:

- atom_Z
- bond_distance
- sin(angle), cos(angle)
- sin(dihedral), cos(dihedral)

Defaults/notes:

- Excludes stems containing -extended unless --include-extended is set.
- Default output prefix is <input_dir>/gzmat_dataset.
- Ragged vectors are padded with NaN by default; use --strict-length to enforce equal lengths.

### scripts/cluster_gzmat_features.py

Runs PCA + optional UMAP + optional t-SNE, then clusters with KMeans (default) or DBSCAN.

Primary input:

- CSV from gzmat_flatten_dataset.py, usually *.cluster_features_zscore.csv

Key outputs in --out-dir (or inferred default):

- <method>_labels_with_stats.csv
- <method>_cluster_stats_summary.csv
- embeddings.csv
- pca_<method>.png
- umap_<method>.png (if computed)
- tsne_<method>.png (if computed)
- dbscan_sweep.csv (DBSCAN mode)

Important defaults:

- method defaults to kmeans
- --kmeans-k defaults to 10
- UMAP is skipped by default (use --use-umap to enable)
- t-SNE is enabled by default
- If --compute-xyz-stats-dir is not provided, stats dir is inferred from feature CSV when possible

### scripts/plot_cluster_label_distributions.py

Consumes cluster labels/stats CSV (for example kmeans_labels_with_stats.csv) and writes figure panels used in the report.

Outputs under <out-dir> (default: <csv_dir>/cluster_distribution_plots):

- per_cluster_rows/cluster_<id>_metrics_row.png
- all_cluster_overlays/*_all_clusters_overlay.png

Metrics include volume, q_tetra_coord, q_tetra_ca, r_free, r_work, pooled Cys dihedrals, Zn B-factor, pooled coordinator-residue B-factors, and family distributions.

### scripts/build_cluster_distribution_report.py (final report step)

Composes interactive report HTML from clustering artifacts.

Default expected inputs in --clustering-dir:

- embeddings.csv
- kmeans_cluster_stats_summary.csv
- tsne_kmeans.png
- cluster_distribution_plots/per_cluster_rows/*.png
- cluster_distribution_plots/all_cluster_overlays/*.png
- optional: kmeans_labels_with_stats.csv (for id->cluster/color mapping)

Outputs:

- report_cluster_distribution.html (CDN Plotly)
- report_cluster_distribution_offline.html (Plotly + images + data inlined)

Offline report note:

- The offline HTML is self-contained for viewing (no CSV sidecar required).

## Typical end-to-end run (4cys-large example)

```bash
# 1) Cleanup duplicate/loose Hs in preview mode first (safe validation run)
uv run python scripts/cleanup_xyz_postzn_hydrogens.py \
    data/large-cys-his-datasets/4cys-large/output/xyz_files \
    --preview-dir data/large-cys-his-datasets/4cys-large/output/xyz_files_h_cleanup_preview

# Optional: apply directly once validated
uv run python scripts/cleanup_xyz_postzn_hydrogens.py \
    data/large-cys-his-datasets/4cys-large/output/xyz_files \
    --apply-in-place

# 2) Convert XYZ -> gzmat
uv run python scripts/xyz_to_gzmat.py \
    data/large-cys-his-datasets/4cys-large/output/xyz_files

# 3) Build flattened + feature + z-score datasets
uv run python scripts/gzmat_flatten_dataset.py \
    data/large-cys-his-datasets/4cys-large/output/xyz_files

# 4) Cluster z-scored features and compute embeddings/stats
uv run python scripts/cluster_gzmat_features.py \
    data/large-cys-his-datasets/4cys-large/output/xyz_files/gzmat_dataset.cluster_features_zscore.csv \
    --out-dir data/large-cys-his-datasets/4cys-large/clustering-default-check \
    --compute-xyz-stats-dir data/large-cys-his-datasets/4cys-large/output/xyz_files

# 5) Generate per-cluster and overlay distribution figures
uv run python scripts/plot_cluster_label_distributions.py \
    data/large-cys-his-datasets/4cys-large/clustering-default-check/kmeans_labels_with_stats.csv

# 6) Build online + offline interactive HTML reports
uv run python scripts/build_cluster_distribution_report.py \
    --clustering-dir data/large-cys-his-datasets/4cys-large/clustering-default-check \
    --title "4Cys Large Cluster Distribution Report"
```

## Hand-off contracts between scripts

| Producer | Consumer | What is exchanged |
|---|---|---|
| cleanup_xyz_postzn_hydrogens.py | xyz_to_gzmat.py | cleaned .xyz files (preview copy or in-place updates) |
| xyz_to_gzmat.py | gzmat_flatten_dataset.py | .gzmat files |
| gzmat_flatten_dataset.py | cluster_gzmat_features.py | gzmat_dataset.cluster_features_zscore.csv |
| cluster_gzmat_features.py | plot_cluster_label_distributions.py | <method>_labels_with_stats.csv (commonly kmeans_labels_with_stats.csv) |
| plot_cluster_label_distributions.py | build_cluster_distribution_report.py | cluster_distribution_plots/per_cluster_rows/*.png and all_cluster_overlays/*.png |
| cluster_gzmat_features.py | build_cluster_distribution_report.py | embeddings.csv + <method>_cluster_stats_summary.csv + tsne_<method>.png |

## Method naming compatibility note

build_cluster_distribution_report.py defaults are kmeans-specific filenames:

- kmeans_cluster_stats_summary.csv
- tsne_kmeans.png
- kmeans_labels_with_stats.csv

If you cluster with DBSCAN, either:

1. pass explicit override paths via --summary-csv, --labels-csv, --tsne-png, or
2. rename/copy DBSCAN outputs to the expected names before report build.
