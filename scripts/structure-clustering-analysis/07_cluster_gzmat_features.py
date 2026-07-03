#!/usr/bin/env python3
"""Cluster flattened gzmat features with PCA/UMAP/t-SNE + DBSCAN/KMeans.

Inputs:
- CSV produced by scripts/gzmat_flatten_dataset.py, usually
  *.cluster_features_zscore.csv

Outputs (to --out-dir):
- <method>_labels_with_stats.csv
- embeddings.csv
- dbscan_sweep.csv (DBSCAN only)
- pca_<method>.png
- umap_<method>.png (if UMAP available and not skipped)
- tsne_<method>.png (if requested)
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from xyz_plot_helpers import (
    _cys_atoms_by_residue,
    _family_token_from_grouped_text,
    _his_atoms_by_residue,
    _sequence_distance_from_atoms,
    _sequence_secstruct_from_atoms,
    ca_volume_and_coord_distances,
    dihedral_origin_sg_cb_ca_selected_by_sg,
    find_xyz_files,
    parse_xyz_atoms,
)

try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "scikit-learn is required. Install with `uv add scikit-learn` "
        "or run with `uv run --with scikit-learn ...`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PCA/UMAP embedding and DBSCAN or KMeans clustering for gzmat feature tables."
    )
    parser.add_argument(
        "features_csv",
        type=Path,
        help="Input feature CSV (typically *.cluster_features_zscore.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help=(
            "Output directory for labels/plots. If omitted and the input CSV is under "
            "data/large-cys-his-datasets/<system>/output/xyz_files, defaults to "
            "data/large-cys-his-datasets/<system>/clustering; otherwise uses "
            "data/conformer-analysis."
        ),
    )
    parser.add_argument(
        "--method",
        choices=("dbscan", "kmeans"),
        default="kmeans",
        help="Clustering method (default: kmeans).",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        help=(
            "Fixed number of PCA components for clustering space. "
            "If omitted, --pca-variance is used to choose components automatically."
        ),
    )
    parser.add_argument(
        "--pca-variance",
        type=float,
        default=0.95,
        help=(
            "Target cumulative explained variance for automatic PCA dimension selection "
            "(default: 0.95)."
        ),
    )
    parser.add_argument(
        "--dbscan-space",
        choices=("pca", "umap"),
        default="pca",
        help="Embedding space used by DBSCAN (default: pca). Ignored by KMeans.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        help="DBSCAN eps. If omitted, a sweep is run and best eps is selected.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="DBSCAN min_samples (default: 10).",
    )
    parser.add_argument(
        "--eps-start",
        type=float,
        default=0.6,
        help="Sweep start eps if --eps is omitted (default: 0.6).",
    )
    parser.add_argument(
        "--eps-stop",
        type=float,
        default=3.0,
        help="Sweep stop eps if --eps is omitted (default: 3.0).",
    )
    parser.add_argument(
        "--eps-step",
        type=float,
        default=0.1,
        help="Sweep step for eps if --eps is omitted (default: 0.1).",
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=30,
        help="UMAP n_neighbors (default: 30).",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.05,
        help="UMAP min_dist (default: 0.05).",
    )
    parser.add_argument(
        "--umap-input",
        choices=("pca", "raw"),
        default="pca",
        help=(
            "Input space for UMAP embedding: pca (default) uses PCA components, "
            "raw uses original feature columns."
        ),
    )
    parser.add_argument(
        "--skip-umap",
        action="store_true",
        help="Skip UMAP embedding and UMAP plots.",
    )
    parser.add_argument(
        "--use-umap",
        dest="skip_umap",
        action="store_false",
        help="Enable UMAP embedding and UMAP plots (default: disabled).",
    )
    parser.add_argument(
        "--use-tsne",
        action="store_true",
        help="Compute and plot a 2D t-SNE projection.",
    )
    parser.add_argument(
        "--tsne-input",
        choices=("pca", "raw"),
        default="pca",
        help=(
            "Input space for t-SNE projection: pca (default) uses PCA components, "
            "raw uses original feature columns."
        ),
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (default: 30.0).",
    )
    parser.add_argument(
        "--tsne-learning-rate",
        default="auto",
        help="t-SNE learning rate (default: auto).",
    )
    parser.add_argument(
        "--kmeans-k",
        type=int,
        default=10,
        help="Number of KMeans clusters when --method kmeans (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for PCA/UMAP/KMeans determinism (default: 42).",
    )
    parser.add_argument(
        "--stats-csv",
        type=Path,
        help=(
            "Optional CSV with per-structure stats (e.g., volume_extremes csv). "
            "When provided, this stats table is merged with cluster labels."
        ),
    )
    parser.add_argument(
        "--compute-xyz-stats-dir",
        type=Path,
        help=(
            "Optional directory of .xyz files. If provided, compute per-structure "
            "volume/family/dihedral/q stats for all structures and use those stats "
            "for merge + cluster summaries. If omitted, this is inferred from "
            "the feature CSV path when possible."
        ),
    )
    parser.set_defaults(skip_umap=True, use_tsne=True)
    return parser.parse_args()


def configure_plot_style() -> bool:
    """Use LaTeX text rendering when available, otherwise Times-like serif."""
    latex_available = shutil.which("latex") is not None

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        }
    )

    # Enable full LaTeX rendering only when the system toolchain is present.
    plt.rcParams["text.usetex"] = bool(latex_available)
    return latex_available


def load_feature_csv(path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    ids: list[str] = []
    rows: list[list[float]] = []

    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty CSV: {path}") from exc

        if len(header) < 2 or header[0] != "id":
            raise ValueError("Expected first column to be 'id'")

        feature_names = header[1:]
        for line in reader:
            if not line:
                continue
            ids.append(line[0])
            rows.append([float(x) for x in line[1:]])

    if not rows:
        raise ValueError(f"No data rows in: {path}")

    arr = np.asarray(rows, dtype=float)
    return ids, arr, feature_names


def _dbscan_summary(labels: np.ndarray) -> tuple[int, int, float, float]:
    n_total = labels.size
    noise_mask = labels == -1
    n_noise = int(noise_mask.sum())
    noise_frac = float(n_noise / n_total)

    cluster_ids = sorted(set(int(x) for x in labels if x != -1))
    n_clusters = len(cluster_ids)

    if n_clusters == 0:
        return n_clusters, n_noise, noise_frac, 1.0

    counts = [int((labels == cid).sum()) for cid in cluster_ids]
    largest_cluster_frac = max(counts) / float(n_total)
    return n_clusters, n_noise, noise_frac, largest_cluster_frac


def _eps_values(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("eps-step must be > 0")
    vals = np.arange(start, stop + 1e-9, step)
    return [round(float(x), 6) for x in vals]


def _run_dbscan(x: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    return model.fit_predict(x)


def _run_kmeans(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    if k < 2:
        raise ValueError("kmeans-k must be >= 2")
    model = KMeans(n_clusters=k, n_init=20, random_state=seed)
    return model.fit_predict(x)


def choose_eps(
    x: np.ndarray,
    min_samples: int,
    eps_values: Sequence[float],
) -> tuple[float, np.ndarray, list[tuple[float, int, int, float, float]]]:
    sweep_rows: list[tuple[float, int, int, float, float]] = []
    best_key: tuple[int, float, float] | None = None
    best_eps: float | None = None
    best_labels: np.ndarray | None = None

    for eps in eps_values:
        labels = _run_dbscan(x, eps=eps, min_samples=min_samples)
        n_clusters, n_noise, noise_frac, largest_cluster_frac = _dbscan_summary(labels)
        sweep_rows.append((eps, n_clusters, n_noise, noise_frac, largest_cluster_frac))

        # Prefer more clusters, then lower noise, then lower cluster domination.
        key = (n_clusters, -noise_frac, -largest_cluster_frac)
        if best_key is None or key > best_key:
            best_key = key
            best_eps = eps
            best_labels = labels

    assert best_eps is not None
    assert best_labels is not None
    return best_eps, best_labels, sweep_rows


def _cluster_color_map(labels: np.ndarray) -> dict[int, str]:
    cluster_ids = sorted(set(int(x) for x in labels if x != -1))
    cmap = plt.get_cmap("tab20", max(1, len(cluster_ids)))
    color_map: dict[int, str] = {-1: "#BDBDBD"}
    for idx, cid in enumerate(cluster_ids):
        color_map[cid] = mcolors.to_hex(cmap(idx), keep_alpha=False)
    return color_map


def plot_embedding(
    path: Path,
    emb: np.ndarray,
    labels: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    color_map = _cluster_color_map(labels)

    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    for cid in sorted(set(int(x) for x in labels)):
        mask = labels == cid
        cluster_name = "noise" if cid == -1 else f"cluster {cid}"
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=20,
            alpha=0.9,
            c=[color_map[cid]],
            label=cluster_name,
            linewidths=0,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_embeddings_csv(
    path: Path,
    ids: list[str],
    pca_2d: np.ndarray,
    pca_nd: np.ndarray,
    umap_2d: np.ndarray | None,
    tsne_2d: np.ndarray | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        header = ["id", "pca1", "pca2"]
        header.extend(f"pc{i + 1}" for i in range(pca_nd.shape[1]))
        if umap_2d is not None:
            header.extend(["umap1", "umap2"])
        if tsne_2d is not None:
            header.extend(["tsne1", "tsne2"])
        writer.writerow(header)

        for i, sid in enumerate(ids):
            row: list[float | str] = [sid, float(pca_2d[i, 0]), float(pca_2d[i, 1])]
            row.extend(float(x) for x in pca_nd[i, :])
            if umap_2d is not None:
                row.extend([float(umap_2d[i, 0]), float(umap_2d[i, 1])])
            if tsne_2d is not None:
                row.extend([float(tsne_2d[i, 0]), float(tsne_2d[i, 1])])
            writer.writerow(row)


def _as_float_or_nan(value: str | None) -> float:
    if value is None:
        return float("nan")
    txt = value.strip()
    if not txt:
        return float("nan")
    try:
        return float(txt)
    except ValueError:
        return float("nan")


def _id_from_stats_row(row: dict[str, str]) -> str | None:
    sid = row.get("id", "").strip()
    if sid:
        return sid
    xyz_path = row.get("xyz_path", "").strip()
    if not xyz_path:
        return None
    return Path(xyz_path).stem


def _split_resseq_icode(resseq: str) -> tuple[str, str]:
    txt = (resseq or "").strip()
    m = re.match(r"^(-?\d+)([A-Za-z]?)$", txt)
    if not m:
        return txt, ""
    return m.group(1), m.group(2)


def _xyz_to_validated_pdb_path(xyz_path: Path) -> Path | None:
    pdb_id = xyz_path.stem.split("_", 1)[0].lower()
    if not pdb_id:
        return None
    # .../<system>/output/xyz_files/<file>.xyz -> .../<system>/results/validated_structures/<pdbid>.pdb
    try:
        root = xyz_path.parent.parent.parent
    except Exception:
        return None
    return root / "results" / "validated_structures" / f"{pdb_id}.pdb"


def _parse_float_from_remark_line(line: str) -> float | None:
    m = re.search(r":\s*([-+]?\d*\.\d+|[-+]?\d+)", line)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _parse_pdb_metrics(
    pdb_path: Path,
) -> tuple[float | None, float | None, dict[tuple[str, str, str], list[float]], list[tuple[str, str, str, float]]]:
    """Return (r_work, r_free, atom_map, zn_atoms).

    atom_map key: (chain, resseq_with_icode, atom_name)
    zn_atoms entries: (chain, resseq_with_icode, atom_name, bfactor)
    """
    r_work: float | None = None
    r_free: float | None = None
    atom_map: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    zn_atoms: list[tuple[str, str, str, float]] = []

    with pdb_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("REMARK   3"):
                u = line.upper()
                if "FREE R VALUE" in u and r_free is None:
                    r_free = _parse_float_from_remark_line(line)
                elif "R VALUE" in u and "WORKING" in u and r_work is None:
                    r_work = _parse_float_from_remark_line(line)

            if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
                continue
            if len(line) < 66:
                continue

            atom_name = line[12:16].strip().upper()
            chain = line[21].strip()
            resseq = line[22:26].strip()
            icode = line[26].strip()
            resseq_ic = f"{resseq}{icode}" if icode else resseq

            try:
                bfac = float(line[60:66].strip())
            except ValueError:
                continue

            key = (chain, resseq_ic, atom_name)
            atom_map[key].append(bfac)

            element = line[76:78].strip().upper() if len(line) >= 78 else ""
            if element == "ZN" or atom_name == "ZN":
                zn_atoms.append((chain, resseq_ic, atom_name, bfac))

    return r_work, r_free, dict(atom_map), zn_atoms


def _select_coord_residue_keys(atoms, *, max_residues: int = 4) -> list[tuple[tuple[str, str], str]]:
    candidates: list[tuple[float, tuple[str, str], str]] = []

    cys_by_res = _cys_atoms_by_residue(atoms, required_atoms={"SG"})
    for key, atom_map in cys_by_res.items():
        sg = atom_map.get("SG")
        if sg is None:
            continue
        d2 = float(sg.x * sg.x + sg.y * sg.y + sg.z * sg.z)
        candidates.append((d2, key, "CYS"))

    his_by_res = _his_atoms_by_residue(atoms, required_atoms={"ND1", "NE2"})
    for key, atom_map in his_by_res.items():
        coord_atom = None
        for name in ("ND1", "NE2"):
            a = atom_map.get(name)
            if a is not None and a.meta.get("COORD") == "1":
                coord_atom = a
                break
        if coord_atom is None:
            best_d2 = None
            for name in ("ND1", "NE2"):
                a = atom_map.get(name)
                if a is None:
                    continue
                d2 = float(a.x * a.x + a.y * a.y + a.z * a.z)
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    coord_atom = a
        if coord_atom is None:
            continue
        d2 = float(coord_atom.x * coord_atom.x + coord_atom.y * coord_atom.y + coord_atom.z * coord_atom.z)
        candidates.append((d2, key, "HIS"))

    candidates.sort(key=lambda t: t[0])
    selected: list[tuple[tuple[str, str], str]] = []
    seen: set[tuple[str, str]] = set()
    for _, key, resname in candidates:
        if key in seen:
            continue
        seen.add(key)
        selected.append((key, resname))
        if len(selected) >= max_residues:
            break
    return selected


def _per_residue_nonh_atom_names(atoms) -> dict[tuple[str, str], set[str]]:
    by_res: dict[tuple[str, str], set[str]] = defaultdict(set)
    for a in atoms:
        if (a.element or "").upper() == "H":
            continue
        atom_name = (a.meta.get("ATOM") or "").strip().upper()
        chain = (a.meta.get("CHAIN") or "").strip()
        resseq = (a.meta.get("RESSEQ") or "").strip()
        if not atom_name or not resseq:
            continue
        by_res[(chain, resseq)].add(atom_name)
    return dict(by_res)


def _find_zn_bfactor(atoms, pdb_atom_map: dict[tuple[str, str, str], list[float]], zn_atoms) -> float | None:
    zn_xyz = None
    for a in atoms:
        if (a.element or "").upper() == "ZN":
            zn_xyz = a
            break

    if zn_xyz is not None:
        chain = (zn_xyz.meta.get("CHAIN") or "").strip()
        resseq = (zn_xyz.meta.get("RESSEQ") or "").strip()
        atom_name = (zn_xyz.meta.get("ATOM") or "ZN").strip().upper() or "ZN"
        if resseq:
            key = (chain, resseq, atom_name)
            vals = pdb_atom_map.get(key)
            if vals:
                return float(np.mean(vals))
            # fallback with normalized resseq/icode shape
            resseq_n, icode = _split_resseq_icode(resseq)
            key2 = (chain, f"{resseq_n}{icode}" if icode else resseq_n, atom_name)
            vals2 = pdb_atom_map.get(key2)
            if vals2:
                return float(np.mean(vals2))

    if zn_atoms:
        return float(np.mean([t[3] for t in zn_atoms]))
    return None


def _q_tetra(points: list[np.ndarray], center: np.ndarray) -> float | None:
    if len(points) != 4:
        return None
    unit_vecs: list[np.ndarray] = []
    for p in points:
        v = p - center
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            return None
        unit_vecs.append(v / norm)
    total = 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            cos_t = float(np.dot(unit_vecs[i], unit_vecs[j]))
            cos_t = max(-1.0, min(1.0, cos_t))
            total += (cos_t + 1.0 / 3.0) ** 2
    return 1.0 - (3.0 / 8.0) * total


def _select_coord_and_ca_vectors(atoms, *, max_residues: int = 4) -> tuple[list[np.ndarray], list[np.ndarray]]:
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []

    cys_by_res = _cys_atoms_by_residue(atoms, required_atoms={"SG", "CA"})
    for atom_map in cys_by_res.values():
        sg = atom_map.get("SG")
        ca = atom_map.get("CA")
        if sg is None or ca is None:
            continue
        sg_vec = np.asarray((sg.x, sg.y, sg.z), dtype=float)
        ca_vec = np.asarray((ca.x, ca.y, ca.z), dtype=float)
        candidates.append((float(np.dot(sg_vec, sg_vec)), sg_vec, ca_vec))

    his_by_res = _his_atoms_by_residue(atoms, required_atoms={"ND1", "NE2", "CA"})
    for atom_map in his_by_res.values():
        ca = atom_map.get("CA")
        if ca is None:
            continue
        coord_atom = None
        for name in ("ND1", "NE2"):
            a = atom_map.get(name)
            if a is not None and a.meta.get("COORD") == "1":
                coord_atom = a
                break
        if coord_atom is None:
            best_d2 = None
            for name in ("ND1", "NE2"):
                a = atom_map.get(name)
                if a is None:
                    continue
                d2 = float(a.x * a.x + a.y * a.y + a.z * a.z)
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    coord_atom = a
        if coord_atom is None:
            continue
        coord_vec = np.asarray((coord_atom.x, coord_atom.y, coord_atom.z), dtype=float)
        ca_vec = np.asarray((ca.x, ca.y, ca.z), dtype=float)
        candidates.append((float(np.dot(coord_vec, coord_vec)), coord_vec, ca_vec))

    candidates.sort(key=lambda t: t[0])
    selected = candidates[:max_residues]
    return [c for _, c, _ in selected], [ca for _, _, ca in selected]


def _family_label(atoms) -> str:
    seq = _sequence_distance_from_atoms(atoms, use_latex=False)
    sec = _sequence_secstruct_from_atoms(atoms)
    if not seq and not sec:
        return ""
    seq_token = _family_token_from_grouped_text(seq) if seq else ""
    sec_token = _family_token_from_grouped_text(sec) if sec else ""
    if seq_token and sec_token:
        return f"{seq_token}-{sec_token}"
    return seq_token or sec_token


def compute_xyz_stats_rows(xyz_dir: Path) -> tuple[int, int, list[dict[str, str]]]:
    xyz_files = find_xyz_files(xyz_dir)
    xyz_files = [p for p in xyz_files if "-extended" not in p.stem]
    if not xyz_files:
        raise ValueError(f"No .xyz files found in {xyz_dir}")

    computed = 0
    rows: list[dict[str, str]] = []
    pdb_cache: dict[Path, tuple[float | None, float | None, dict[tuple[str, str, str], list[float]], list[tuple[str, str, str, float]]]] = {}
    for p in xyz_files:
        atoms = parse_xyz_atoms(p)
        volume, coord_dists = ca_volume_and_coord_distances(atoms, max_residues=4)
        if volume is None or len(coord_dists) != 4:
            continue

        coord_vecs, ca_vecs = _select_coord_and_ca_vectors(atoms, max_residues=4)
        origin = np.zeros(3, dtype=float)
        q_coord = _q_tetra(coord_vecs, origin) if len(coord_vecs) == 4 else None
        q_ca = _q_tetra(ca_vecs, origin) if len(ca_vecs) == 4 else None
        dihedrals = dihedral_origin_sg_cb_ca_selected_by_sg(atoms, max_residues=4)

        r_work: float | None = None
        r_free: float | None = None
        zn_bfac: float | None = None
        residue_bfac: list[float | None] = [None, None, None, None]
        coord_residue_types: list[str | None] = [None, None, None, None]

        pdb_path = _xyz_to_validated_pdb_path(p)
        if pdb_path is not None and pdb_path.is_file():
            if pdb_path not in pdb_cache:
                pdb_cache[pdb_path] = _parse_pdb_metrics(pdb_path)
            r_work, r_free, pdb_atom_map, zn_atoms = pdb_cache[pdb_path]

            zn_bfac = _find_zn_bfactor(atoms, pdb_atom_map, zn_atoms)

            residue_atom_names = _per_residue_nonh_atom_names(atoms)
            coord_res_keys = _select_coord_residue_keys(atoms, max_residues=4)
            for i, ((chain, resseq), resname) in enumerate(coord_res_keys[:4]):
                coord_residue_types[i] = resname
                atom_names = residue_atom_names.get((chain, resseq), set())
                if not atom_names:
                    continue
                vals: list[float] = []
                resseq_n, icode = _split_resseq_icode(resseq)
                resseq_ic = f"{resseq_n}{icode}" if icode else resseq_n
                for an in atom_names:
                    for key in ((chain, resseq, an), (chain, resseq_ic, an)):
                        b_list = pdb_atom_map.get(key)
                        if b_list:
                            vals.extend(b_list)
                            break
                if vals:
                    residue_bfac[i] = float(np.mean(np.asarray(vals, dtype=float)))

        row: dict[str, str] = {
            "id": p.stem,
            "volume_A3": f"{volume:.4f}",
            "family": _family_label(atoms),
            "cys_dihedral_mean_deg": f"{float(np.mean(dihedrals)):.2f}" if dihedrals else "",
            "q_tetra_coord": f"{q_coord:.4f}" if q_coord is not None else "",
            "q_tetra_ca": f"{q_ca:.4f}" if q_ca is not None else "",
            "r_work": f"{r_work:.4f}" if r_work is not None else "",
            "r_free": f"{r_free:.4f}" if r_free is not None else "",
            "zn_bfactor": f"{zn_bfac:.3f}" if zn_bfac is not None else "",
            "xyz_path": str(p),
        }
        for i in range(4):
            row[f"cys_dihedral_{i + 1}_deg"] = f"{dihedrals[i]:.2f}" if i < len(dihedrals) else ""
            rb = residue_bfac[i]
            # Type-specific columns requested for coordinating residue B-factors.
            row[f"coord_cys_{i + 1}_bfactor_avg"] = ""
            row[f"coord_his_{i + 1}_bfactor_avg"] = ""
            rtype = coord_residue_types[i]
            if rb is not None and rtype == "CYS":
                row[f"coord_cys_{i + 1}_bfactor_avg"] = f"{rb:.3f}"
            elif rb is not None and rtype == "HIS":
                row[f"coord_his_{i + 1}_bfactor_avg"] = f"{rb:.3f}"
        rows.append(row)
        computed += 1

    return len(xyz_files), computed, rows


def write_labels_with_stats_csv(
    ids: list[str],
    labels: np.ndarray,
    cluster_colors: dict[int, str],
    stats_csv: Path | None,
    stats_rows: list[dict[str, str]] | None,
    merged_out: Path,
) -> tuple[int, int, list[str]]:
    stats_map: dict[str, dict[str, str]] = {}
    stats_fields: list[str] = []

    if stats_rows is not None:
        for row in stats_rows:
            sid = _id_from_stats_row(row)
            if sid is None:
                continue
            if sid not in stats_map:
                stats_map[sid] = row
        if stats_rows:
            stats_fields = list(stats_rows[0].keys())
    elif stats_csv is not None:
        if not stats_csv.is_file():
            raise FileNotFoundError(f"Stats CSV not found: {stats_csv}")
        with stats_csv.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise ValueError(f"Stats CSV has no header: {stats_csv}")
            for row in reader:
                sid = _id_from_stats_row(row)
                if sid is None:
                    continue
                if sid not in stats_map:
                    stats_map[sid] = row
            stats_fields = list(reader.fieldnames)

    merged_out.parent.mkdir(parents=True, exist_ok=True)
    matches = 0
    with merged_out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "cluster", "cluster_color", "has_stats", *stats_fields])
        for sid, lab in zip(ids, labels):
            cluster = int(lab)
            cluster_color = cluster_colors.get(cluster, "#BDBDBD")
            stats_row = stats_map.get(sid)
            has_stats = 1 if stats_row is not None else 0
            if has_stats:
                matches += 1
            payload = [stats_row.get(col, "") if stats_row is not None else "" for col in stats_fields]
            writer.writerow([sid, cluster, cluster_color, has_stats, *payload])

    return len(ids), matches, stats_fields


def write_cluster_stats_summary(merged_csv: Path, out_csv: Path) -> None:
    groups: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"volume_A3": [], "q_tetra_ca": [], "q_tetra_coord": []}
    )
    cluster_counts: dict[str, int] = defaultdict(int)
    cluster_with_stats: dict[str, int] = defaultdict(int)

    with merged_csv.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Merged CSV has no header: {merged_csv}")

        for row in reader:
            cluster = row.get("cluster", "")
            if not cluster:
                continue
            cluster_counts[cluster] += 1
            if row.get("has_stats", "0") == "1":
                cluster_with_stats[cluster] += 1

            for key in ("volume_A3", "q_tetra_ca", "q_tetra_coord"):
                val = _as_float_or_nan(row.get(key))
                if not np.isnan(val):
                    groups[cluster][key].append(val)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        header = [
            "cluster",
            "n_total",
            "n_with_stats",
        ]
        for metric in ("volume_A3", "q_tetra_ca", "q_tetra_coord"):
            header.extend(
                [
                    f"{metric}_n",
                    f"{metric}_mean",
                    f"{metric}_std",
                    f"{metric}_min",
                    f"{metric}_q25",
                    f"{metric}_median",
                    f"{metric}_q75",
                    f"{metric}_max",
                ]
            )
        writer.writerow(header)

        def _stats(values: list[float]) -> list[float | str]:
            if not values:
                return [0, "", "", "", "", "", "", ""]
            arr = np.asarray(values, dtype=float)
            return [
                int(arr.size),
                float(np.mean(arr)),
                float(np.std(arr)),
                float(np.min(arr)),
                float(np.quantile(arr, 0.25)),
                float(np.median(arr)),
                float(np.quantile(arr, 0.75)),
                float(np.max(arr)),
            ]

        for cluster in sorted(cluster_counts, key=lambda x: int(x)):
            row: list[float | int | str] = [
                cluster,
                cluster_counts[cluster],
                cluster_with_stats[cluster],
            ]
            for metric in ("volume_A3", "q_tetra_ca", "q_tetra_coord"):
                row.extend(_stats(groups[cluster][metric]))
            writer.writerow(row)


def write_sweep_csv(path: Path, rows: list[tuple[float, int, int, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["eps", "n_clusters", "n_noise", "noise_fraction", "largest_cluster_fraction"])
        for row in rows:
            writer.writerow(row)


def _cluster_source_text(args: argparse.Namespace, pca_components: int) -> str:
    if args.method == "kmeans":
        return f"Color by KMeans labels learned in PCA({pca_components}) space, k={args.kmeans_k}"
    return (
        "Color by DBSCAN labels learned in "
        f"{args.dbscan_space.upper()} space, eps={args.eps if args.eps is not None else 'auto'}, "
        f"min_samples={args.min_samples}"
    )


def _pca_meta_text(args: argparse.Namespace, pca_components: int, pca_explained: float) -> str:
    if args.pca_components is not None:
        return (
            f"PCA fixed components={pca_components}, cumulative explained variance={pca_explained:.3f}"
        )
    return (
        f"PCA variance threshold={args.pca_variance:.2f}, "
        f"components kept={pca_components}, cumulative explained variance={pca_explained:.3f}"
    )


def infer_default_out_dir(features_csv: Path) -> Path:
    """Infer output directory from input feature path when possible.

    Pattern handled:
      .../data/large-cys-his-datasets/<system>/output/xyz_files/<file>.csv
    -> .../data/large-cys-his-datasets/<system>/clustering
    """
    parts = list(features_csv.parts)
    marker = ("data", "large-cys-his-datasets")

    for idx in range(len(parts) - len(marker)):
        if tuple(parts[idx : idx + len(marker)]) != marker:
            continue
        system_idx = idx + len(marker)
        if system_idx + 2 >= len(parts):
            continue
        if parts[system_idx + 1] != "output" or parts[system_idx + 2] != "xyz_files":
            continue
        base = Path(*parts[: system_idx + 1])
        return base / "clustering"

    return Path("data/conformer-analysis").resolve()


def infer_default_xyz_stats_dir(features_csv: Path) -> Path | None:
        """Infer xyz directory from feature path.

        Pattern handled:
            .../data/large-cys-his-datasets/<system>/output/xyz_files/<file>.csv
        -> .../data/large-cys-his-datasets/<system>/output/xyz_files
        """
        parent = features_csv.parent
        if parent.name == "xyz_files" and parent.is_dir():
                return parent
        return None


def main() -> int:
    args = parse_args()
    latex_enabled = configure_plot_style()

    features_csv = args.features_csv.expanduser().resolve()
    if not features_csv.is_file():
        raise SystemExit(f"Input feature CSV not found: {features_csv}")

    if args.out_dir is not None:
        out_dir = args.out_dir.expanduser().resolve()
    else:
        out_dir = infer_default_out_dir(features_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids, x_raw, _feature_names = load_feature_csv(features_csv)

    max_pca = min(x_raw.shape[0], x_raw.shape[1])
    if args.pca_components is not None:
        pca_components_req = max(2, min(args.pca_components, max_pca))
        pca_model = PCA(n_components=pca_components_req, random_state=args.seed)
    else:
        if not (0.0 < args.pca_variance <= 1.0):
            raise SystemExit("--pca-variance must be in (0, 1].")
        pca_model = PCA(n_components=args.pca_variance, svd_solver="full", random_state=args.seed)
    x_pca = pca_model.fit_transform(x_raw)
    pca_components = int(x_pca.shape[1])
    pca_explained = float(np.sum(pca_model.explained_variance_ratio_))
    x_pca2 = x_pca[:, :2]

    x_umap2: np.ndarray | None = None
    umap_error: str | None = None
    if not args.skip_umap:
        try:
            import umap

            umap_input = x_pca if args.umap_input == "pca" else x_raw

            um = umap.UMAP(
                n_components=2,
                n_neighbors=args.umap_neighbors,
                min_dist=args.umap_min_dist,
                metric="euclidean",
                random_state=args.seed,
            )
            x_umap2 = um.fit_transform(umap_input)
        except Exception as exc:
            umap_error = str(exc)

    x_tsne2: np.ndarray | None = None
    tsne_error: str | None = None
    if args.use_tsne:
        try:
            tsne_input = x_pca if args.tsne_input == "pca" else x_raw
            learning_rate: float | str
            if str(args.tsne_learning_rate).lower() == "auto":
                learning_rate = "auto"
            else:
                learning_rate = float(args.tsne_learning_rate)

            tsne = TSNE(
                n_components=2,
                perplexity=args.tsne_perplexity,
                learning_rate=learning_rate,
                init="pca",
                random_state=args.seed,
            )
            x_tsne2 = tsne.fit_transform(tsne_input)
        except Exception as exc:
            tsne_error = str(exc)

    if args.method == "kmeans":
        x_cluster = x_pca
        labels = _run_kmeans(x_cluster, k=args.kmeans_k, seed=args.seed)
        n_clusters, n_noise, noise_frac, largest_cluster_frac = _dbscan_summary(labels)
        chosen_eps = None
        sweep_rows: list[tuple[float, int, int, float, float]] = []
    else:
        if args.dbscan_space == "umap":
            if x_umap2 is None:
                raise SystemExit(
                    "DBSCAN space was set to umap, but UMAP embedding was unavailable. "
                    f"UMAP error: {umap_error or 'not installed'}"
                )
            x_cluster = x_umap2
        else:
            x_cluster = x_pca

        if args.eps is not None:
            labels = _run_dbscan(x_cluster, eps=args.eps, min_samples=args.min_samples)
            n_clusters, n_noise, noise_frac, largest_cluster_frac = _dbscan_summary(labels)
            chosen_eps = args.eps
            sweep_rows = [(args.eps, n_clusters, n_noise, noise_frac, largest_cluster_frac)]
        else:
            eps_values = _eps_values(args.eps_start, args.eps_stop, args.eps_step)
            chosen_eps, labels, sweep_rows = choose_eps(
                x_cluster,
                min_samples=args.min_samples,
                eps_values=eps_values,
            )

        n_clusters, n_noise, noise_frac, largest_cluster_frac = _dbscan_summary(labels)

    pca_meta = _pca_meta_text(args, pca_components, pca_explained)
    cluster_source = _cluster_source_text(args, pca_components)
    cluster_colors = _cluster_color_map(labels)

    labels_stats_csv = out_dir / f"{args.method}_labels_with_stats.csv"
    cluster_stats_csv = out_dir / f"{args.method}_cluster_stats_summary.csv"
    emb_csv = out_dir / "embeddings.csv"
    sweep_csv = out_dir / "dbscan_sweep.csv"
    pca_png = out_dir / f"pca_{args.method}.png"
    umap_png = out_dir / f"umap_{args.method}.png"
    tsne_png = out_dir / f"tsne_{args.method}.png"

    merged_rows = 0
    merged_matches = 0
    stats_source: Path | None = None
    computed_stats_rows: list[dict[str, str]] | None = None
    total_xyz = 0
    computed_xyz = 0
    xyz_stats_dir = (
        args.compute_xyz_stats_dir.expanduser().resolve()
        if args.compute_xyz_stats_dir is not None
        else infer_default_xyz_stats_dir(features_csv)
    )
    if xyz_stats_dir is not None:
        total_xyz, computed_xyz, computed_stats_rows = compute_xyz_stats_rows(xyz_stats_dir)
    elif args.stats_csv is not None:
        stats_source = args.stats_csv.expanduser().resolve()

    merged_rows, merged_matches, _ = write_labels_with_stats_csv(
        ids=ids,
        labels=labels,
        cluster_colors=cluster_colors,
        stats_csv=stats_source,
        stats_rows=computed_stats_rows,
        merged_out=labels_stats_csv,
    )
    write_cluster_stats_summary(labels_stats_csv, cluster_stats_csv)
    write_embeddings_csv(emb_csv, ids, x_pca2, x_pca, x_umap2, x_tsne2)
    if args.method == "dbscan":
        write_sweep_csv(sweep_csv, sweep_rows)
    plot_embedding(
        pca_png,
        x_pca2,
        labels,
        title=(
            f"{args.method.upper()} labels on PCA projection\n"
            f"{pca_meta}\n"
            f"{cluster_source}"
        ),
        xlabel="PC1 score (PCA space)",
        ylabel="PC2 score (PCA space)",
    )

    if x_umap2 is not None:
        plot_embedding(
            umap_png,
            x_umap2,
            labels,
            title=(
                f"{args.method.upper()} labels on UMAP projection\n"
                f"{pca_meta}\n"
                f"{cluster_source}"
            ),
            xlabel=f"UMAP-1 (input={args.umap_input.upper()})",
            ylabel=f"UMAP-2 (input={args.umap_input.upper()})",
        )

    if x_tsne2 is not None:
        plot_embedding(
            tsne_png,
            x_tsne2,
            labels,
            title=(
                f"{args.method.upper()} labels on t-SNE projection\n"
                f"{pca_meta}\n"
                f"{cluster_source}; t-SNE input={args.tsne_input.upper()}, "
                f"perplexity={args.tsne_perplexity:g}"
            ),
            xlabel=f"t-SNE-1 (input={args.tsne_input.upper()})",
            ylabel=f"t-SNE-2 (input={args.tsne_input.upper()})",
        )

    print(f"Loaded {len(ids)} rows x {x_raw.shape[1]} features from: {features_csv}")
    print(f"Plot text rendering: {'LaTeX' if latex_enabled else 'serif fallback (Times-like)'}")
    print(f"PCA components used for clustering/PCA outputs: {pca_components}")
    print(f"PCA cumulative explained variance: {pca_explained:.4f}")
    if x_umap2 is not None:
        print("UMAP embedding: computed")
        print(f"UMAP input space: {args.umap_input}")
    else:
        print(f"UMAP embedding: skipped/unavailable ({umap_error or 'skip requested'})")
    if x_tsne2 is not None:
        print("t-SNE embedding: computed")
        print(f"t-SNE input space: {args.tsne_input}")
    else:
        if args.use_tsne:
            print(f"t-SNE embedding: failed ({tsne_error})")
        else:
            print("t-SNE embedding: not requested")
    print(f"Method: {args.method}")
    if args.method == "dbscan":
        print(f"DBSCAN space: {args.dbscan_space}")
        print(f"Chosen eps: {chosen_eps:.3f}")
        print(f"min_samples: {args.min_samples}")
    else:
        print("KMeans space: pca")
        print(f"k: {args.kmeans_k}")
    print(f"Clusters (excluding noise): {n_clusters}")
    if args.method == "dbscan":
        print(f"Noise points: {n_noise} ({noise_frac:.2%})")
        print(f"Largest-cluster fraction: {largest_cluster_frac:.2%}")
    if xyz_stats_dir is not None:
        print(
            f"Computed per-structure xyz stats in-memory "
            f"(valid {computed_xyz}/{total_xyz} non-extended xyz files)"
        )
    print(
        f"Wrote labels+stats: {labels_stats_csv} "
        f"(matched {merged_matches}/{merged_rows} rows by structure id)"
    )
    print(f"Wrote cluster stats summary: {cluster_stats_csv}")
    print(f"Wrote embeddings: {emb_csv}")
    if args.method == "dbscan":
        print(f"Wrote sweep summary: {sweep_csv}")
    print(f"Wrote PCA plot: {pca_png}")
    if x_umap2 is not None:
        print(f"Wrote UMAP plot: {umap_png}")
    if x_tsne2 is not None:
        print(f"Wrote t-SNE plot: {tsne_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
