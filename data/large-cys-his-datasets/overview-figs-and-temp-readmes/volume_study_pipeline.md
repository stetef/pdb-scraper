# Volume-study report pipeline

Three scripts work together to take a directory of cluster XYZ files and produce
the interactive HTML report at
`data/large-cys-his-datasets/4cys-large/figures/report_<label>.html`.

```
                            ┌────────────────────┐
   data/.../xyz_files/*.xyz │  full XYZ dataset  │
                            └─────────┬──────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │  scripts/xyz-volume-study.py │
                       │  (analysis)                  │
                       └──────────────┬───────────────┘
                                      │
                ┌─────────────────────┼─────────────────────┐
                ▼                     ▼                     ▼
   data/.../figures/                   .../figures/         (terminal print
   volume_vs_*_<label>.png             volume_extremes_     of the 5+5+11
   (7 PNG plots, full dataset)         <label>.csv          extreme rows)
                │                       │
                │                       ├──────────────────────────────────┐
                │                       │                                  │
                │                       ▼                                  │
                │           ┌─────────────────────────────────┐            │
                │           │  scripts/build_extended_xyz.py  │            │
                │           │  (one-time data prep)           │            │
                │           └────────────────┬────────────────┘            │
                │                            │                             │
                │                            ▼                             │
                │           data/.../xyz_files/<stem>-extended.xyz         │
                │                            │                             │
                ▼                            ▼                             ▼
              ┌──────────────────────────────────────────────────────────────┐
              │            scripts/build_volume_report.py                    │
              │            (compose self-contained HTML)                     │
              └────────────────────────────┬─────────────────────────────────┘
                                           │
                                           ▼
                          report_<label>.html  (the deliverable)
```

## What each script does

### `scripts/xyz-volume-study.py` — analysis (run first)

Reads every `*.xyz` cluster file in a directory and, per structure, computes:

- **C𝛼 tetrahedron volume** (convex hull of the four coordinating residues' α-carbons)
- **Zn→coord-atom distances** (one per Cys SG / His ND1/NE2)
- **Cys Zn–S–C𝛽–C𝛼 dihedrals** (one per coordinating Cys)
- **q_tetra(S)** and **q_tetra(C𝛼)** — Errington–Debenedetti tetrahedral order
  parameters about Zn for the sulfur and α-carbon sub-tetrahedra
- A "family" label encoding the sequence spacing + secondary structure (e.g. `Cx2Cx82Cx2C-LLLL`)

**Outputs (in `<out-dir>/`):**

- `volume_vs_coord_distance_<label>.png` — full-dataset scatter, used as background context in the report
- `volume_vs_cys_dihedral_<label>.png` — (currently not embedded in the report)
- `volume_vs_ca_q_tetra_<label>.png`, `volume_vs_s_q_tetra_<label>.png`, `s_q_tetra_vs_ca_q_tetra_<label>.png`
- `volume_vs_coord_distance_extremes_<label>.png`, `volume_vs_cys_dihedral_extremes_<label>.png` — broken-axis versions of the extremes (also superseded by the report's interactive Plotly plots)
- **`volume_extremes_<label>.csv`** — 21 rows: 5 smallest CA volume + 5 largest CA volume + 11 lowest q_S, with columns `group, rank, volume_A3, family, cys_dihedral_{1..4}_deg, cys_dihedral_mean_deg, q_tetra_coord, q_tetra_ca, xyz_path`. **This CSV is the hand-off to the next two scripts.**

### `scripts/build_extended_xyz.py` — extended-environment XYZ prep (run once)

Reads `volume_extremes_<label>.csv`. For each `xyz_path`:

1. Parses the XYZ comment to find the Zn origin (chain, resseq, atom name).
2. Opens the matching validated PDB at `<results-dir>/<pdb_id>.pdb`.
3. Translates every ATOM/HETATM so the Zn is at (0, 0, 0).
4. Keeps atoms within `--cutoff` Å (default 10) and writes them to `<original_xyz_stem>-extended.xyz` next to the compact XYZ.

This is what gives the report's "Extended (10 Å)" toggle the larger atom set
showing neighboring residues, waters, and other ligands.

```bash
uv run python scripts/build_extended_xyz.py \
    --csv     data/large-cys-his-datasets/4cys-large/figures/volume_extremes_4cys-large-dataset.csv \
    --pdb-dir data/large-cys-his-datasets/4cys-large/results/validated_structures
```

Re-run with `--force` to regenerate after changing the cutoff.

### `scripts/build_volume_report.py` — compose the HTML report

Reads:
- `volume_extremes_<label>.csv` (from `xyz-volume-study.py`)
- For each row: the compact `<xyz_path>` and the `<stem>-extended.xyz` (from `build_extended_xyz.py`) if present
- Two of the full-dataset PNGs: `volume_vs_coord_distance_<label>.png` and `volume_vs_s_q_tetra_<label>.png`

Emits one self-contained HTML file with:
- A silver-gradient header with dataset stats and section summaries
- **Section 1 — C𝛼 volume vs first-shell coord distance:** full-dataset PNG + interactive Plotly broken-axis scatter (smallest left, largest right) + hero card with 3Dmol viewer + 10 mini-cards
- **Section 2 — Low q_S structures:** full-dataset PNG + Plotly scatter + hero + 11 mini-cards, with the formula rendered via KaTeX
- Hero cards include an RCSB link (first 4 chars of file stem → `https://www.rcsb.org/structure/<ID>`) and a Compact / Extended (10 Å) viewer toggle
- Plotly + 3Dmol + KaTeX assets either via CDN (default) or fully inlined (`--offline`)

```bash
# CDN-linked (1.2 MB; needs internet to view)
uv run python scripts/build_volume_report.py \
    --figures-dir data/large-cys-his-datasets/4cys-large/figures \
    --label 4cys-large-dataset \
    --system-name "4Cys Zn-binding structures" \
    --pdb-count 1500 --xyz-count 2698

# Self-contained (~8 MB; works on any machine offline)
uv run python scripts/build_volume_report.py \
    --figures-dir data/large-cys-his-datasets/4cys-large/figures \
    --label 4cys-large-dataset \
    --system-name "4Cys Zn-binding structures" \
    --pdb-count 1500 --xyz-count 2698 \
    --offline \
    --output ..../report_4cys-large-dataset_offline.html
```

## Typical end-to-end run for this study

```bash
# 1. (already done) compute geometry + emit PNGs + extremes CSV
uv run python scripts/xyz-volume-study.py \
    --dir    data/large-cys-his-datasets/4cys-large/output/xyz_files \
    --out-dir data/large-cys-his-datasets/4cys-large/figures \
    --system-label  4cys-large-dataset \
    --no-show

# 2. (one-time) materialize the 10 Å neighborhood XYZ for each extreme
uv run python scripts/build_extended_xyz.py \
    --csv     data/large-cys-his-datasets/4cys-large/figures/volume_extremes_4cys-large-dataset.csv \
    --pdb-dir data/large-cys-his-datasets/4cys-large/results/validated_structures

# 3. compose the HTML (re-run any time you tweak --offline, stats, etc.)
uv run python scripts/build_volume_report.py \
    --figures-dir data/large-cys-his-datasets/4cys-large/figures \
    --label 4cys-large-dataset \
    --system-name "4Cys Zn-binding structures" \
    --pdb-count 1500 --xyz-count 2698
```

## Hand-off contracts between scripts

| Producer | Consumer | What's exchanged |
|---|---|---|
| `xyz-volume-study.py` | `build_extended_xyz.py` | `volume_extremes_<label>.csv` (specifically the `xyz_path` column) |
| `xyz-volume-study.py` | `build_volume_report.py` | `volume_extremes_<label>.csv` (all columns) + `volume_vs_coord_distance_*.png` + `volume_vs_s_q_tetra_*.png` |
| `build_extended_xyz.py` | `build_volume_report.py` | `<stem>-extended.xyz` files (sibling of each `xyz_path`) |

The CSV is the single source of truth for which structures appear in the
report; changing the extremes (e.g., top-N count, q_S threshold) only requires
re-running `xyz-volume-study.py` and then steps 2 and 3.
