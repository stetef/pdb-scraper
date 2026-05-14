# kmeans_labels_with_stats.csv Data Dictionary

Source CSV: `kmeans_labels_with_stats.csv`

Purpose of this file:
- Describe what each column contains.
- Specify practical data types (including nullability).
- Capture interpretation notes so another AI/statistical workflow can choose appropriate tests.

## Row-Level Meaning

Each row corresponds to one XYZ structure instance (one clustered conformer/structure record) with:
- Cluster assignment metadata.
- Geometry/order metrics.
- PDB fit-quality metrics.
- B-factor summaries.
- Family annotation string.

## Column Dictionary

| Column | Suggested Type | Nullable | Entry Pattern / Example | Meaning |
|---|---|---:|---|---|
| `id` (1st) | string | No | `13mt_ZN_homo_d2.60_cluster1` | Structure identifier (left-most ID column). |
| `cluster` | int (categorical) | No | `8`, `13`, `6` | Cluster label assigned by clustering workflow. Treat as categorical, not ordinal. |
| `cluster_color` | string (hex color) | Yes | `#c49c94`, `#17becf` | Display color associated with the cluster. Plotting metadata only. |
| `has_stats` | int/bool flag | Likely No | `1` | Indicates whether merged stats are present for the row. |
| `id` (2nd) | string | No | `13mt_ZN_homo_d2.60_cluster1` | Duplicate ID column from merge/join step; usually same as first `id`. |
| `volume_A3` | float | Yes | `21.2364` | Tetrahedral/coordination volume in Angstrom^3. |
| `family` | string (categorical) | Yes | `Cx2Cx31Cx1C-SLLL`, `Cx2Cx17Cx2C-LLHH` | Family label. Encodes sequence-like motif + secondary-structure-like token (split by `-`). |
| `cys_dihedral_mean_deg` | float | Yes | `-6.71`, `60.63` | Mean of Cys dihedral angles (degrees) for that structure. |
| `q_tetra_coord` | float | Yes | `0.9977`, `0.9855` | Tetrahedral order parameter for coordinating atoms. |
| `q_tetra_ca` | float | Yes | `0.3758`, `0.5519` | Tetrahedral order parameter using CA reference geometry. |
| `r_work` | float | Yes | `0.1730`, `0.2320` | Crystallographic refinement metric (working set). External to clustering feature generation. |
| `r_free` | float | Yes | `0.1900`, `0.2650` | Crystallographic validation metric (free set). External to clustering feature generation. |
| `zn_bfactor` | float | Yes | `69.670`, `18.220` | B-factor for Zn atom/site summary per structure. |
| `xyz_path` | string (absolute path) | Yes | `/Users/.../13mt_ZN_homo_d2.60_cluster1.xyz` | File path to source XYZ structure. |
| `cys_dihedral_1_deg` | float | Yes | `-178.80`, `149.85` | Per-residue Cys dihedral angle #1 (degrees). |
| `coord_cys_1_bfactor_avg` | float | Yes | `73.360`, `64.493` | Average B-factor for coordinating Cys residue #1. |
| `coord_his_1_bfactor_avg` | float | Yes | empty in 4Cys sample | Average B-factor for coordinating His residue #1. Often null in pure-4Cys systems. |
| `cys_dihedral_2_deg` | float | Yes | `151.10`, `-174.04` | Per-residue Cys dihedral angle #2 (degrees). |
| `coord_cys_2_bfactor_avg` | float | Yes | `58.713`, `79.920` | Average B-factor for coordinating Cys residue #2. |
| `coord_his_2_bfactor_avg` | float | Yes | empty in 4Cys sample | Average B-factor for coordinating His residue #2. |
| `cys_dihedral_3_deg` | float | Yes | `111.32`, `113.77` | Per-residue Cys dihedral angle #3 (degrees). |
| `coord_cys_3_bfactor_avg` | float | Yes | `66.497`, `72.490` | Average B-factor for coordinating Cys residue #3. |
| `coord_his_3_bfactor_avg` | float | Yes | empty in 4Cys sample | Average B-factor for coordinating His residue #3. |
| `cys_dihedral_4_deg` | float | Yes | `-110.48`, `-109.91` | Per-residue Cys dihedral angle #4 (degrees). |
| `coord_cys_4_bfactor_avg` | float | Yes | `75.350`, `80.657` | Average B-factor for coordinating Cys residue #4. |
| `coord_his_4_bfactor_avg` | float | Yes | empty in 4Cys sample | Average B-factor for coordinating His residue #4. |

## Family String Interpretation

Observed pattern resembles:
- `<motif_token>-<secondary_token>`

Examples:
- `Cx2Cx31Cx1C-SLLL`
- `Cx2Cx17Cx2C-LLHH`

Practical interpretation guidance:
- Treat full `family` string as a nominal category for contingency/statistical testing.
- Optionally split into two categorical features:
  - `family_motif` = left side of `-` (motif-like sequence/spacing encoding).
  - `family_secstruct` = right side of `-` (secondary-structure-like token).

This allows both:
- Full-string tests (high specificity).
- Component-level tests (more interpretable, fewer categories).

## Statistical Role Notes (for downstream AI)

- `cluster`: categorical outcome/grouping variable.
- Structural metrics likely connected to clustering basis:
  - `volume_A3`, `q_tetra_coord`, `q_tetra_ca`, `cys_dihedral_*`, `cys_dihedral_mean_deg`.
- External/"new" metadata relative to clustering basis:
  - `r_work`, `r_free`, `family`.
- Partially external but structure-linked quality/thermal terms:
  - `zn_bfactor`, `coord_*_bfactor_avg`.

## Data Hygiene Notes

- CSV contains two `id` columns; enforce unique names when loading (e.g., `id_left`, `id_right`) to avoid silent overwrite.
- Many `coord_his_*_bfactor_avg` entries can be null for 4Cys families; use missing-aware tests/imputation strategy.
- Dihedrals are angular/circular values (degrees). Circular statistics are preferable to linear assumptions.
- `cluster_color` is visualization metadata; generally exclude from inferential models.
