"""Shared utilities for Zn(Cys)₄ featurization and clustering.

Implements the data model, parsing, Kabsch superposition, matching-minimized
structural RMSD, and the clustering/evaluation pipeline shared across all three
featurization approaches (scripts 02–04).

Public API
----------
Structure, EQUAL_WEIGHTS, SHELL_WEIGHTS, DISTANCE_WEIGHTS
parse_structure(path) -> Structure | None
weighted_kabsch(P, Q, w, allow_reflection) -> (R, t, rmsd)
structural_rmsd(A, B, w_type, allow_reflection) -> (rmsd, best_perm, (R, t))
cluster_pipeline(X, k, ...) -> (labels, centroids_pca, pca, (means, stds))
evaluate_clustering(structures, labels, X_pca, centroids, w_type) -> dict  # returns intra/inter/ratio/ch_score
sweep_k(structures, X, k_values, w_type, ...) -> (best_result, all_results)
save_outputs(out_dir, id_list, results_table, best_result)
save_embeddings_and_tsne(out_dir, id_list, X_pca, labels) -> np.ndarray
build_cluster_distribution_plots(out_dir, id_list, labels, stats_csv)
write_structure_xyz(structure, path)
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from itertools import combinations, permutations, product
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
except ImportError:
    _MPL_OK = False

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
except ImportError:
    raise SystemExit("scikit-learn required: uv add scikit-learn")

import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EQUAL_WEIGHTS: dict[str, float] = {"Zn": 1.0, "S": 1.0, "Cb": 1.0, "Ca": 1.0}
SHELL_WEIGHTS: dict[str, float] = {"Zn": 1.0, "S": 1.0, "Cb": 0.5, "Ca": 0.5}
# Sentinel: weights = 1 / avg_distance_from_Zn per atom pair; Zn itself gets 1.
DISTANCE_WEIGHTS: str = "distance"

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Structure:
    id: str
    zn: np.ndarray   # (3,)
    s:  np.ndarray   # (4, 3) — residue-indexed
    cb: np.ndarray   # (4, 3) — aligned with s
    ca: np.ndarray   # (4, 3) — aligned with s
    # Per-residue metadata (original PDB values, preserved through alignment).
    resseqs:   list = field(default_factory=lambda: [1, 2, 3, 4])
    chains:    list = field(default_factory=lambda: ["A", "A", "A", "A"])
    res_names: list = field(default_factory=lambda: ["CYS", "CYS", "CYS", "CYS"])
    # ZN metadata
    zn_resseq: int = 0
    zn_chain:  str = "A"
    zn_res:    str = "ZN"

    def heavy(self) -> np.ndarray:
        """(13, 3) fixed layout [Zn | S0..3 | Cb0..3 | Ca0..3]."""
        return np.vstack([self.zn[None], self.s, self.cb, self.ca])

    def w_vec(self, w_type: dict) -> np.ndarray:
        """(13,) weight vector from {Zn, S, Cb, Ca} dict."""
        return np.array(
            [w_type["Zn"]] + [w_type["S"]] * 4 + [w_type["Cb"]] * 4 + [w_type["Ca"]] * 4,
            dtype=float,
        )

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_atom_tag(comment: str) -> Optional[str]:
    """Return normalised heavy-atom type from 'ATOM=...' EOL tag."""
    for part in comment.split():
        if part.startswith("ATOM="):
            tag = part[5:].strip().upper()
            if tag == "ZN":   return "ZN"
            if tag in ("SG", "S"):  return "S"
            if tag == "CB":   return "CB"
            if tag == "CA":   return "CA"
            return None   # H or unknown
    return None


def _parse_resseq(comment: str) -> Optional[int]:
    for part in comment.split():
        if part.startswith("RESSEQ="):
            try:
                return int(part[7:])
            except ValueError:
                pass
    return None


def _parse_chain(comment: str) -> str:
    for part in comment.split():
        if part.startswith("CHAIN="):
            return part[6:].strip()
    return ""


def _parse_res(comment: str) -> str:
    for part in comment.split():
        if part.startswith("RES="):
            return part[4:].strip()
    return ""


def parse_structure(path: Path) -> Optional[Structure]:
    """Parse XYZ file with ATOM= EOL tags.  Returns None on failure."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    if len(lines) < 2:
        return None

    zn_coords: list[np.ndarray] = []
    zn_comments: list[str] = []
    residues: dict[tuple[str, int], dict] = {}
    orphan: dict[str, list] = {"S": [], "CB": [], "CA": []}

    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue
        halves = line.split("#", 1)
        coords_part = halves[0].split()
        comment = halves[1] if len(halves) > 1 else ""

        atom_type = _parse_atom_tag(comment)
        if atom_type is None:
            continue
        if len(coords_part) < 4:
            continue

        try:
            xyz = np.array([float(coords_part[1]), float(coords_part[2]), float(coords_part[3])])
        except ValueError:
            continue

        if atom_type == "ZN":
            zn_coords.append(xyz)
            zn_comments.append(comment)
            continue

        resseq = _parse_resseq(comment)
        if resseq is not None:
            chain = _parse_chain(comment)
            res_name = _parse_res(comment)
            entry = residues.setdefault((chain, resseq), {})
            entry[atom_type] = xyz
            if "_RES" not in entry:
                entry["_RES"] = res_name or "CYS"
        else:
            orphan[atom_type].append(xyz)

    if len(zn_coords) != 1:
        return None
    zn = zn_coords[0]
    zn_comment = zn_comments[0]

    # Build complete (S, CB, CA) triples, prefer RESSEQ grouping
    complete: list[tuple[int, dict]] = [
        (k, v) for k, v in residues.items()
        if "S" in v and "CB" in v and "CA" in v
    ]

    # Fallback: nearest-neighbour grouping for orphan atoms
    if len(complete) < 4 and len(orphan["S"]) == 4 == len(orphan["CB"]) == len(orphan["CA"]):
        sl, cbl, cal = orphan["S"], orphan["CB"], orphan["CA"]
        used_cb: set[int] = set()
        groups = []
        for sv in sl:
            cb_idx = min((j for j in range(4) if j not in used_cb),
                         key=lambda j: np.linalg.norm(sv - cbl[j]))
            used_cb.add(cb_idx)
            ca_idx = min(range(4), key=lambda j: np.linalg.norm(cbl[cb_idx] - cal[j]))
            groups.append({"S": sv, "CB": cbl[cb_idx], "CA": cal[ca_idx]})
        complete = list(enumerate(groups))

    if len(complete) != 4:
        return None

    complete.sort(key=lambda x: x[0])
    s  = np.array([v["S"]  for _, v in complete])
    cb = np.array([v["CB"] for _, v in complete])
    ca = np.array([v["CA"] for _, v in complete])

    resseqs   = [k[1] for k, _ in complete]
    chains    = [k[0] for k, _ in complete]
    res_names = [v.get("_RES", "CYS") for _, v in complete]

    zn_resseq_val = _parse_resseq(zn_comment) or 0
    zn_chain_val  = _parse_chain(zn_comment) or "A"
    zn_res_val    = _parse_res(zn_comment) or "ZN"

    return Structure(
        id=path.stem, zn=zn, s=s, cb=cb, ca=ca,
        resseqs=resseqs, chains=chains, res_names=res_names,
        zn_resseq=zn_resseq_val, zn_chain=zn_chain_val, zn_res=zn_res_val,
    )


def write_structure_xyz(structure: Structure, path: Path) -> None:
    """Write 13-atom heavy-atom XYZ preserving original RES/CHAIN/RESSEQ metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("13\n")
        fh.write(f"id={structure.id}\n")
        zn = structure.zn
        fh.write(
            f"ZN  {zn[0]:.6f}  {zn[1]:.6f}  {zn[2]:.6f}"
            f"  # RES={structure.zn_res} CHAIN={structure.zn_chain}"
            f" RESSEQ={structure.zn_resseq} ATOM=ZN\n"
        )
        for r in range(4):
            s  = structure.s[r]
            cb = structure.cb[r]
            ca = structure.ca[r]
            res  = structure.res_names[r]
            ch   = structure.chains[r]
            rsq  = structure.resseqs[r]
            fh.write(
                f"S   {s[0]:.6f}  {s[1]:.6f}  {s[2]:.6f}"
                f"  # RES={res} CHAIN={ch} RESSEQ={rsq} ATOM=SG\n"
            )
            fh.write(
                f"C   {cb[0]:.6f}  {cb[1]:.6f}  {cb[2]:.6f}"
                f"  # RES={res} CHAIN={ch} RESSEQ={rsq} ATOM=CB\n"
            )
            fh.write(
                f"C   {ca[0]:.6f}  {ca[1]:.6f}  {ca[2]:.6f}"
                f"  # RES={res} CHAIN={ch} RESSEQ={rsq} ATOM=CA\n"
            )

# ---------------------------------------------------------------------------
# Weighted Kabsch superposition
# ---------------------------------------------------------------------------

def weighted_kabsch(
    P: np.ndarray,
    Q: np.ndarray,
    w: np.ndarray,
    allow_reflection: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Weighted Kabsch: find R, t such that P @ R.T + t ≈ Q (minimises weighted RMSD).

    Returns (R, t, rmsd).  If allow_reflection=True, keeps an improper rotation
    when it gives lower RMSD (merges enantiomers).
    """
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    cp = (w[:, None] * P).sum(0)
    cq = (w[:, None] * Q).sum(0)
    Pc = P - cp
    Qc = Q - cq

    H = (w[:, None] * Pc).T @ Qc
    U, _, Vt = np.linalg.svd(H)

    R_raw = Vt.T @ U.T
    det = np.linalg.det(R_raw)

    # Proper rotation (standard Kabsch diag(1,1,d) correction)
    Vt_corr = Vt.copy()
    if det < 0:
        Vt_corr[-1] *= -1
    R_proper = Vt_corr.T @ U.T

    def _rmsd(R: np.ndarray) -> float:
        return float(np.sqrt((w * ((Pc @ R.T - Qc) ** 2).sum(1)).sum()))

    if allow_reflection and det < 0:
        rmsd_p = _rmsd(R_proper)
        rmsd_r = _rmsd(R_raw)
        R = R_raw if rmsd_r < rmsd_p else R_proper
    else:
        R = R_proper

    t = cq - R @ cp
    return R, t, _rmsd(R)

# ---------------------------------------------------------------------------
# Matching-minimized structural RMSD (metric)
# ---------------------------------------------------------------------------

_ALL_PERMS: list[list[int]] = [list(p) for p in permutations(range(4))]


def _heavy_perm(B: Structure, perm: list[int]) -> np.ndarray:
    return np.vstack([B.zn[None], B.s[perm], B.cb[perm], B.ca[perm]])


def _zn_distance_weights(A: Structure, B: Structure, perm: list[int]) -> np.ndarray:
    """(13,) weight vector = 1 / avg_Zn_distance for a given residue permutation of B.

    For each atom slot i, D_i = (dist(A_atom_i, A.zn) + dist(B_atom_perm[i], B.zn)) / 2,
    weight = 1/D_i.  Zn itself uses a sentinel distance of 1.0 Å → weight = 1.0.
    """
    d_A = np.array([
        1.0,  # Zn sentinel
        *[np.linalg.norm(A.s[i]  - A.zn) for i in range(4)],
        *[np.linalg.norm(A.cb[i] - A.zn) for i in range(4)],
        *[np.linalg.norm(A.ca[i] - A.zn) for i in range(4)],
    ])
    d_B = np.array([
        1.0,  # Zn sentinel
        *[np.linalg.norm(B.s[perm[i]]  - B.zn) for i in range(4)],
        *[np.linalg.norm(B.cb[perm[i]] - B.zn) for i in range(4)],
        *[np.linalg.norm(B.ca[perm[i]] - B.zn) for i in range(4)],
    ])
    return 1.0 / ((d_A + d_B) / 2.0)


def structural_rmsd(
    A: Structure,
    B: Structure,
    w_type: dict | str,
    allow_reflection: bool = True,
) -> tuple[float, list[int], tuple]:
    """Matching-minimized RMSD over all 4! residue permutations of B.

    Returns (rmsd, best_perm, (R, t)).
    When w_type == DISTANCE_WEIGHTS, per-atom weights are 1/avg_Zn_distance and
    are recomputed for each permutation (since permuting B changes which residue
    occupies each slot).
    """
    A_heavy = A.heavy()
    best_rmsd = math.inf
    best_perm = list(range(4))
    best_Rt: tuple = (np.eye(3), np.zeros(3))
    _static_w = None if w_type == DISTANCE_WEIGHTS else A.w_vec(w_type)

    for perm in _ALL_PERMS:
        w = _zn_distance_weights(A, B, perm) if w_type == DISTANCE_WEIGHTS else _static_w
        R, t, rmsd = weighted_kabsch(_heavy_perm(B, perm), A_heavy, w, allow_reflection)
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_perm = perm
            best_Rt = (R, t)

    return best_rmsd, best_perm, best_Rt

# ---------------------------------------------------------------------------
# Clustering pipeline
# ---------------------------------------------------------------------------

def cluster_pipeline(
    X: np.ndarray,
    k: int,
    variance: float = 0.95,
    var_floor: float = 1e-3,
    random_state: int = 0,
    n_init: int = 10,
    clustering_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, object, tuple]:
    """Z-score → PCA (95% var) → k-means.

    Returns (labels, centroids_pca, pca_model, (means, stds)).
    centroids_pca[c] is the k-means centroid for cluster c in PCA space.
    """
    Xu = X[:, clustering_mask] if clustering_mask is not None else X
    means = Xu.mean(0)
    stds  = np.maximum(Xu.std(0), var_floor)
    Xs = (Xu - means) / stds

    pca_full = PCA(random_state=random_state)
    pca_full.fit(Xs)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = min(max(1, int(np.searchsorted(cumvar, variance)) + 1), Xs.shape[1])

    pca = PCA(n_components=n_comp, random_state=random_state)
    X_pca = pca.fit_transform(Xs)

    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    labels = km.fit_predict(X_pca)

    return labels, km.cluster_centers_, pca, (means, stds), pca_full.explained_variance_ratio_

# ---------------------------------------------------------------------------
# Evaluation metric
# ---------------------------------------------------------------------------

def evaluate_clustering(
    structures: list[Structure],
    labels: np.ndarray,
    X_pca: np.ndarray,
    centroids_pca: np.ndarray,
    w_type: dict | str,
    allow_reflection: bool = True,
) -> dict:
    """Intra/inter structural-RMSD evaluation.

    Returns dict with keys: k, intra, inter, ratio, ch_score, medoid_ids, per_cluster_intra.
    ch_score = (inter²/(k−1)) / (intra²/(N−k))  — Calinski-Harabasz analogue for RMSD.
    """
    unique = sorted(set(int(l) for l in labels))

    # Medoid = member nearest centroid in PCA space
    medoid_idx: dict[int, int] = {}
    for c in unique:
        mask = labels == c
        idxs = np.where(mask)[0]
        centroid = centroids_pca[c] if c < len(centroids_pca) else X_pca[mask].mean(0)
        medoid_idx[c] = int(idxs[np.argmin(np.linalg.norm(X_pca[mask] - centroid, axis=1))])

    # Intra: mean structural_rmsd(medoid, member) per cluster
    per_cluster_intra: list[float] = []
    for c in unique:
        idxs = np.where(labels == c)[0]
        med = structures[medoid_idx[c]]
        rmsds = [structural_rmsd(med, structures[i], w_type, allow_reflection)[0]
                 for i in idxs if i != medoid_idx[c]]
        per_cluster_intra.append(float(np.mean(rmsds)) if rmsds else 0.0)

    intra = float(np.mean(per_cluster_intra)) if per_cluster_intra else 0.0

    # Inter: mean structural_rmsd between every medoid pair
    meds = [structures[medoid_idx[c]] for c in unique]
    inter_vals = [structural_rmsd(meds[i], meds[j], w_type, allow_reflection)[0]
                  for i, j in combinations(range(len(meds)), 2)]
    inter = float(np.mean(inter_vals)) if inter_vals else 0.0
    ratio = inter / intra if intra > 0 else 0.0

    # Calinski-Harabasz-analogue score (RMSD-based):
    #   ch_score = (inter² / (k-1)) / (intra² / (N-k))
    # Penalizes large k via (k-1) in denominator; peaks at a genuinely good k.
    N = len(structures)
    k = len(unique)
    ch_denom = intra ** 2 * (k - 1)
    ch_score = ((inter ** 2 * (N - k)) / ch_denom) if (ch_denom > 0 and N > k) else 0.0

    return {
        "k": k,
        "intra": intra,
        "inter": inter,
        "ratio": ratio,
        "ch_score": ch_score,
        "medoid_ids": {c: structures[medoid_idx[c]].id for c in unique},
        "per_cluster_intra": per_cluster_intra,
    }

# ---------------------------------------------------------------------------
# k sweep
# ---------------------------------------------------------------------------

def sweep_k(
    structures: list[Structure],
    X: np.ndarray,
    k_values: list[int],
    w_type: dict | str,
    allow_reflection: bool = True,
    clustering_mask: Optional[np.ndarray] = None,
    desc: str = "k sweep",
) -> tuple[Optional[dict], list[dict]]:
    """Run cluster_pipeline + evaluate_clustering for each valid k.

    Returns (best_result, all_results).  Each result dict includes labels, X_pca,
    pca, scale in addition to the evaluation fields.
    """
    valid_ks = [k for k in k_values if 2 <= k < len(structures)]
    if not valid_ks:
        return None, []

    results: list[dict] = []
    best_ch = -math.inf
    best: Optional[dict] = None

    for k in tqdm.tqdm(valid_ks, desc=desc, leave=True):
        labels, centroids, pca, scale, evr_full = cluster_pipeline(X, k, clustering_mask=clustering_mask)
        means, stds = scale
        Xu = X[:, clustering_mask] if clustering_mask is not None else X
        X_pca = pca.transform((Xu - means) / stds)

        ev = evaluate_clustering(structures, labels, X_pca, centroids, w_type, allow_reflection)
        ev.update({"labels": labels, "X_pca": X_pca, "pca": pca, "scale": scale, "evr_full": evr_full})
        results.append(ev)

        if ev["ch_score"] > best_ch:
            best_ch = ev["ch_score"]
            best = ev

    return best, results

# ---------------------------------------------------------------------------
# Standard output persistence
# ---------------------------------------------------------------------------

def save_outputs(
    out_dir: Path,
    id_list: list[str],
    k_sweep_table: list[dict],
    best: dict,
) -> None:
    """Write labels.csv, medoids.csv, k_sweep.csv, per_cluster_intra.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # k sweep table
    with (out_dir / "k_sweep.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["k", "intra", "inter", "ratio", "ch_score"])
        w.writeheader()
        for r in k_sweep_table:
            w.writerow({"k": r["k"], "intra": f"{r['intra']:.6f}",
                        "inter": f"{r['inter']:.6f}", "ratio": f"{r['ratio']:.6f}",
                        "ch_score": f"{r['ch_score']:.6f}"})

    # Labels
    labels = best["labels"]
    with (out_dir / "labels.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["structure_id", "cluster"])
        for sid, lbl in zip(id_list, labels):
            w.writerow([sid, int(lbl)])

    # Medoids
    with (out_dir / "medoids.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["cluster_id", "medoid_id"])
        for cid, mid in best["medoid_ids"].items():
            w.writerow([int(cid), mid])

    # Per-cluster intra
    with (out_dir / "per_cluster_intra.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["cluster_id", "mean_intra_rmsd"])
        for cid, val in enumerate(best["per_cluster_intra"]):
            w.writerow([cid, f"{val:.6f}"])

    # PCA scree data (full EVR from best-k run; same feature matrix for all k)
    evr_full = best.get("evr_full")
    if evr_full is not None:
        n_retained = best["pca"].n_components_
        cumvar = np.cumsum(evr_full)
        with (out_dir / "pca_scree.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["component", "explained_variance_ratio", "cumulative_variance_ratio", "retained"])
            for i, (evr, cum) in enumerate(zip(evr_full, cumvar)):
                w.writerow([i + 1, f"{evr:.6f}", f"{cum:.6f}", i < n_retained])

    # t-SNE embeddings
    save_embeddings_and_tsne(out_dir, id_list, best["X_pca"], best["labels"])


# ---------------------------------------------------------------------------
# t-SNE embeddings
# ---------------------------------------------------------------------------

def save_embeddings_and_tsne(
    out_dir: Path,
    id_list: list[str],
    X_pca: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Run t-SNE on PCA embeddings; write embeddings.csv and tsne_kmeans.png.

    embeddings.csv columns: id, pc1…pcN, tsne1, tsne2
    tsne_kmeans.png: scatter colored by cluster with tab20.
    Returns the (N, 2) t-SNE coordinate array.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  t-SNE skipped (scikit-learn TSNE not available)")
        return np.zeros((len(id_list), 2))

    N = len(id_list)
    perplexity = float(min(30, max(5, N // 4)))
    print(f"  Running t-SNE (n={N}, perplexity={perplexity:.0f}) …")
    X_tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0).fit_transform(X_pca)

    n_pca = X_pca.shape[1]
    emb_path = out_dir / "embeddings.csv"
    with emb_path.open("w", newline="", encoding="utf-8") as fh:
        cw = csv.writer(fh)
        cw.writerow(["id"] + [f"pc{i+1}" for i in range(n_pca)] + ["tsne1", "tsne2"])
        for sid, pca_row, (t1, t2) in zip(id_list, X_pca, X_tsne):
            cw.writerow([sid] + [f"{v:.8f}" for v in pca_row] + [f"{t1:.8f}", f"{t2:.8f}"])
    print(f"  Embeddings → {emb_path}")

    if not _MPL_OK:
        return X_tsne

    unique_labels = sorted(set(int(l) for l in labels))
    cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, c in enumerate(unique_labels):
        mask = labels == c
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=[cmap(i % 20)], s=15, alpha=0.7, label=str(c))

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE of PCA space  (k={len(unique_labels)})")
    ncol = max(1, len(unique_labels) // 20)
    ax.legend(title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=7, ncol=ncol, markerscale=1.5)
    fig.tight_layout()
    png_path = out_dir / "tsne_kmeans.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  t-SNE plot → {png_path}")

    return X_tsne


# ---------------------------------------------------------------------------
# Cluster distribution plots (requires per-structure stats CSV)
# ---------------------------------------------------------------------------

# Numeric metrics to extract from the stats CSV and plot.
# Each entry: (column_name, display_label) — labels use matplotlib mathtext.
_NUMERIC_PLOT_METRICS: list[tuple[str, str]] = [
    ("volume_A3",              r"Volume ($\AA^3$)"),
    ("q_tetra_coord",          r"$q_\mathrm{tetra}$ (coord)"),
    ("q_tetra_ca",             r"$q_\mathrm{tetra}$ ($C_\alpha$)"),
    ("r_work",                 r"$R_\mathrm{work}$"),
    ("r_free",                 r"$R_\mathrm{free}$"),
    ("zn_bfactor",             r"Zn $B$-factor"),
    ("cys_dihedral_mean_deg",  r"Dihedral mean ($^\circ$)"),
    ("all_coord_res_bfactor_avg", r"Coord-res $\bar{B}$"),
]

# Summary CSV column name for "all_dihedrals_deg" uses cys_dihedral_mean_deg as source.
_DIHEDRAL_COL = "cys_dihedral_mean_deg"
_DIHEDRAL_SUMMARY_KEY = "all_dihedrals_deg"

# Font sizes used in cluster distribution plots.
_FS_LABEL  = 13
_FS_TICK   = 11
_FS_TITLE  = 14
_FS_SUP    = 15
_FS_ANNOT  = 9


def _tab20_hex(idx: int) -> str:
    """Return tab20 color at position idx % 20 as '#rrggbb'."""
    if not _MPL_OK:
        return "#888888"
    rgba = plt.get_cmap("tab20")(idx % 20)
    return "#{:02x}{:02x}{:02x}".format(
        int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
    )


def _safe_float(v: str | None) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _coord_res_bfactor_avg(row: dict) -> Optional[float]:
    """Average of per-residue coordinating-residue B-factors."""
    vals = []
    for i in range(1, 5):
        v = _safe_float(row.get(f"coord_cys_{i}_bfactor_avg"))
        if v is not None:
            vals.append(v)
        v = _safe_float(row.get(f"coord_his_{i}_bfactor_avg"))
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else None


def _merge_labels_stats(
    id_list: list[str],
    labels: np.ndarray,
    stats_csv: Path,
) -> tuple[list[dict], dict[int, str]]:
    """Read stats CSV, join with labels; return (rows, color_by_cluster).

    Each row has: id, cluster (int), cluster_color (hex), has_stats (0/1),
    plus all columns from the stats CSV, plus all_coord_res_bfactor_avg.
    """
    stats_by_id: dict[str, dict] = {}
    with stats_csv.open(newline="", encoding="utf-8") as fh:
        for srow in csv.DictReader(fh):
            sid = (srow.get("id") or "").strip()
            if sid:
                stats_by_id[sid] = srow

    unique_clusters = sorted(set(int(l) for l in labels))
    color_by_cluster: dict[int, str] = {c: _tab20_hex(i) for i, c in enumerate(unique_clusters)}

    rows: list[dict] = []
    for sid, label in zip(id_list, labels):
        c = int(label)
        srow = stats_by_id.get(sid, {})
        has_stats = 1 if srow else 0

        merged: dict = {
            "id": sid,
            "cluster": c,
            "cluster_color": color_by_cluster[c],
            "has_stats": has_stats,
        }
        merged.update(srow)

        if has_stats:
            avg = _coord_res_bfactor_avg(srow)
            merged["all_coord_res_bfactor_avg"] = f"{avg:.4f}" if avg is not None else ""

        rows.append(merged)

    return rows, color_by_cluster


def _write_labels_with_stats(out_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    path = out_dir / "kmeans_labels_with_stats.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Labels+stats → {path}")


def _compute_stats_summary(rows: list[dict], unique_clusters: list[int]) -> list[dict]:
    """Per-cluster aggregated statistics for numeric metrics."""
    rows_by_cluster: dict[int, list[dict]] = {c: [] for c in unique_clusters}
    for row in rows:
        rows_by_cluster[int(row["cluster"])].append(row)

    summary_cols = ["cluster", "n_total", "n_with_stats"]
    numeric_metrics = [col for col, _ in _NUMERIC_PLOT_METRICS]
    for m in numeric_metrics:
        key = _DIHEDRAL_SUMMARY_KEY if m == _DIHEDRAL_COL else m
        summary_cols += [f"{key}_n", f"{key}_mean", f"{key}_std",
                         f"{key}_min", f"{key}_q25", f"{key}_median",
                         f"{key}_q75", f"{key}_max"]

    summary: list[dict] = []
    for c in unique_clusters:
        cluster_rows = rows_by_cluster[c]
        n_total = len(cluster_rows)
        n_with_stats = sum(int(r.get("has_stats", 0) or 0) for r in cluster_rows)
        srow: dict = {"cluster": c, "n_total": n_total, "n_with_stats": n_with_stats}

        for col, _ in _NUMERIC_PLOT_METRICS:
            key = _DIHEDRAL_SUMMARY_KEY if col == _DIHEDRAL_COL else col
            vals = [v for r in cluster_rows if (v := _safe_float(r.get(col))) is not None]
            if vals:
                arr = np.array(vals)
                srow[f"{key}_n"] = len(arr)
                srow[f"{key}_mean"] = f"{arr.mean():.6f}"
                srow[f"{key}_std"]  = f"{arr.std():.6f}"
                srow[f"{key}_min"]  = f"{arr.min():.6f}"
                srow[f"{key}_q25"]  = f"{np.percentile(arr, 25):.6f}"
                srow[f"{key}_median"] = f"{np.median(arr):.6f}"
                srow[f"{key}_q75"]  = f"{np.percentile(arr, 75):.6f}"
                srow[f"{key}_max"]  = f"{arr.max():.6f}"
            else:
                for suffix in ("n", "mean", "std", "min", "q25", "median", "q75", "max"):
                    srow[f"{key}_{suffix}"] = ""

        summary.append(srow)
    return summary


def _write_stats_summary(out_dir: Path, summary: list[dict]) -> None:
    if not summary:
        return
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in summary:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    path = out_dir / "kmeans_cluster_stats_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)
    print(f"  Cluster stats summary → {path}")


def _plot_per_cluster_rows(
    rows: list[dict],
    unique_clusters: list[int],
    color_by_cluster: dict[int, str],
    out_dir: Path,
) -> None:
    if not _MPL_OK:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    from collections import Counter

    # Pre-compute full-dataset values for background histograms.
    all_vals: dict[str, list[float]] = {col: [] for col, _ in _NUMERIC_PLOT_METRICS}
    all_family: list[str] = []
    for row in rows:
        for col, _ in _NUMERIC_PLOT_METRICS:
            v = _safe_float(row.get(col))
            if v is not None:
                all_vals[col].append(v)
        f = (row.get("family") or "").strip()
        if f:
            all_family.append(f)

    N_all = len(rows)
    all_families_sorted = sorted(set(all_family))
    has_family = bool(all_families_sorted)
    metrics_with_label = list(_NUMERIC_PLOT_METRICS) + (
        [("family", "Family")] if has_family else []
    )
    n_metrics = len(metrics_with_label)
    ncols = min(4, n_metrics)
    nrows_fig = math.ceil(n_metrics / ncols)

    for c in unique_clusters:
        cluster_rows = [r for r in rows if int(r["cluster"]) == c]
        color = color_by_cluster[c]

        fig, axes = plt.subplots(nrows_fig, ncols,
                                 figsize=(ncols * 4.5, nrows_fig * 3.5),
                                 squeeze=False)
        fig.suptitle(f"Cluster {c}  (n={len(cluster_rows)})", fontsize=_FS_SUP, fontweight="bold")

        for idx, (col, label) in enumerate(metrics_with_label):
            ax = axes[idx // ncols][idx % ncols]

            if col == "family":
                c_fam = [r.get("family", "").strip() for r in cluster_rows
                         if r.get("family", "").strip()]
                counts = Counter(c_fam)
                if counts:
                    # Show top-20 families; sort by count descending.
                    fams = sorted(counts, key=lambda x: counts[x], reverse=True)[:20]
                    xs = range(len(fams))
                    ax.bar(xs, [counts[f] for f in fams], color=color, alpha=0.85)
                    ax.set_xticks(list(xs))
                    ax.set_xticklabels(fams, rotation=90, ha="center", fontsize=7)
                    # Annotate dominant family in top-right corner.
                    ax.text(0.98, 0.97, fams[0], transform=ax.transAxes,
                            ha="right", va="top", fontsize=_FS_ANNOT,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                      edgecolor="#cccccc", alpha=0.85))
                ax.set_ylabel("count", fontsize=_FS_LABEL)
                ax.set_title("Family", fontsize=_FS_TITLE)
                ax.tick_params(axis="y", labelsize=_FS_TICK)
            else:
                cluster_vals = [v for r in cluster_rows
                                if (v := _safe_float(r.get(col))) is not None]
                overall = all_vals.get(col, [])
                N_col = len(overall)
                if N_col > 0:
                    bins = min(30, max(5, N_col // 10))
                    lo, hi = float(np.min(overall)), float(np.max(overall))
                    if lo == hi:
                        lo, hi = lo - 0.5, hi + 0.5
                    bin_edges = np.linspace(lo, hi, bins + 1)
                    # Both histograms normalised to fraction of N_col so axes match.
                    ax.hist(overall, bins=bin_edges, color="#cccccc", alpha=0.65,
                            weights=np.ones(N_col) / N_col, label="all")
                    if cluster_vals:
                        ax.hist(cluster_vals, bins=bin_edges, color=color, alpha=0.80,
                                weights=np.ones(len(cluster_vals)) / N_col, label=f"c{c}")
                ax.set_xlabel(label, fontsize=_FS_LABEL)
                ax.set_ylabel("fraction of total", fontsize=_FS_LABEL)
                ax.set_title(label, fontsize=_FS_TITLE)
                ax.tick_params(labelsize=_FS_TICK)

        for idx in range(n_metrics, nrows_fig * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.tight_layout()
        fig.savefig(out_dir / f"cluster_{c}_metrics_row.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    print(f"  Per-cluster row plots → {out_dir}")


def _plot_overlay_metrics(
    rows: list[dict],
    unique_clusters: list[int],
    color_by_cluster: dict[int, str],
    out_dir: Path,
) -> None:
    """Side-by-side per-cluster subplots for each metric (one file per metric)."""
    if not _MPL_OK:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    from collections import Counter

    rows_by_cluster: dict[int, list[dict]] = {c: [] for c in unique_clusters}
    for row in rows:
        rows_by_cluster[int(row["cluster"])].append(row)

    n_c = len(unique_clusters)
    # Layout: up to 4 columns, wrap to multiple rows.
    ncols_fig = min(4, n_c)
    nrows_fig = math.ceil(n_c / ncols_fig)

    def _make_axes_grid(n_clusters: int, subplot_w: float = 4.0, subplot_h: float = 3.5):
        """Return (fig, axes_flat) with shared x/y axes."""
        nc = min(4, n_clusters)
        nr = math.ceil(n_clusters / nc)
        fig, axes = plt.subplots(
            nr, nc,
            figsize=(nc * subplot_w, nr * subplot_h),
            sharey=True, sharex=True,
            squeeze=False,
        )
        return fig, axes, nr, nc

    # ── Numeric side-by-side overlays ──────────────────────────────────────
    for col, label in _NUMERIC_PLOT_METRICS:
        all_vals = [v for row in rows if (v := _safe_float(row.get(col))) is not None]
        if not all_vals:
            continue
        N_all = len(all_vals)
        lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        bins = min(30, max(5, N_all // 10))
        bin_edges = np.linspace(lo, hi, bins + 1)

        fig, axes, nr, nc = _make_axes_grid(n_c)
        fig.suptitle(f"{label} — all clusters", fontsize=_FS_SUP, fontweight="bold")

        for i, c in enumerate(unique_clusters):
            ax = axes[i // nc][i % nc]
            cvals = [v for r in rows_by_cluster[c]
                     if (v := _safe_float(r.get(col))) is not None]
            # Gray = all data, both normalised to fraction of N_all.
            ax.hist(all_vals, bins=bin_edges, color="#cccccc", alpha=0.65,
                    weights=np.ones(N_all) / N_all, label="all")
            if cvals:
                ax.hist(cvals, bins=bin_edges, color=color_by_cluster[c], alpha=0.80,
                        weights=np.ones(len(cvals)) / N_all, label=f"c{c}")
            ax.set_title(f"Cluster {c}  (n={len(rows_by_cluster[c])})",
                         fontsize=_FS_TITLE)
            ax.set_xlabel(label, fontsize=_FS_LABEL)
            if i % nc == 0:
                ax.set_ylabel("fraction of total", fontsize=_FS_LABEL)
            ax.tick_params(labelsize=_FS_TICK)

        # Hide empty panels.
        for idx in range(n_c, nr * nc):
            axes[idx // nc][idx % nc].set_visible(False)

        fig.tight_layout()
        metric_key = _DIHEDRAL_SUMMARY_KEY if col == _DIHEDRAL_COL else col
        fig.savefig(out_dir / f"{metric_key}_all_clusters_overlay.png",
                    dpi=130, bbox_inches="tight")
        plt.close(fig)

    # ── Family side-by-side overlays ────────────────────────────────────────
    _MAX_FAM_OVERLAY = 30  # show top-N families in the overlay figure
    all_family_pairs = [(int(row["cluster"]), (row.get("family") or "").strip())
                        for row in rows if (row.get("family") or "").strip()]
    if all_family_pairs:
        # Keep only the top families across all clusters.
        top_fam_overall = [f for f, _ in Counter(fam for _, fam in all_family_pairs)
                           .most_common(_MAX_FAM_OVERLAY)]
        all_families = sorted(top_fam_overall)
        n_fam = len(all_families)
        xs = np.arange(n_fam)
        sp_w = max(4.0, n_fam * 0.45 + 1.5)

        fig, axes, nr, nc = _make_axes_grid(n_c, subplot_w=sp_w, subplot_h=4.0)
        fig.suptitle("Family — all clusters", fontsize=_FS_SUP, fontweight="bold")

        for i, c in enumerate(unique_clusters):
            ax = axes[i // nc][i % nc]
            c_fams = [fam for cl, fam in all_family_pairs if cl == c]
            counts = Counter(c_fams)
            ax.bar(xs, [counts.get(f, 0) for f in all_families],
                   color=color_by_cluster[c], alpha=0.85)
            ax.set_xticks(list(xs))
            ax.set_xticklabels(all_families, rotation=90, ha="center", fontsize=7)
            ax.set_title(f"Cluster {c}  (n={len(rows_by_cluster[c])})",
                         fontsize=_FS_TITLE)
            if i % nc == 0:
                ax.set_ylabel("count", fontsize=_FS_LABEL)
            ax.tick_params(axis="y", labelsize=_FS_TICK)

        for idx in range(n_c, nr * nc):
            axes[idx // nc][idx % nc].set_visible(False)

        fig.tight_layout()
        fig.savefig(out_dir / "family_all_clusters_overlay.png",
                    dpi=130, bbox_inches="tight")
        plt.close(fig)

    print(f"  Overlay plots → {out_dir}")


def build_cluster_distribution_plots(
    out_dir: Path,
    id_list: list[str],
    labels: np.ndarray,
    stats_csv: Optional[Path],
) -> None:
    """Generate kmeans_labels_with_stats.csv, stats summary, and histogram PNGs.

    Requires stats_csv (per-structure metadata CSV with an 'id' column).
    Silently skips if stats_csv is None or does not exist.
    """
    if stats_csv is None or not stats_csv.is_file():
        print(f"  Distribution plots skipped (no stats CSV)")
        return

    print(f"\nBuilding cluster distribution plots from {stats_csv.name} …")
    rows, color_by_cluster = _merge_labels_stats(id_list, labels, stats_csv)
    unique_clusters = sorted(set(int(l) for l in labels))

    _write_labels_with_stats(out_dir, rows)

    summary = _compute_stats_summary(rows, unique_clusters)
    _write_stats_summary(out_dir, summary)

    plots_dir = out_dir / "cluster_distribution_plots"
    # Clear stale PNGs so old cluster counts don't bleed into the new report.
    for _subdir in [plots_dir / "per_cluster_rows", plots_dir / "all_cluster_overlays"]:
        if _subdir.is_dir():
            for _f in _subdir.glob("*.png"):
                _f.unlink()
    _plot_per_cluster_rows(rows, unique_clusters, color_by_cluster,
                           plots_dir / "per_cluster_rows")
    _plot_overlay_metrics(rows, unique_clusters, color_by_cluster,
                          plots_dir / "all_cluster_overlays")
    print("  Distribution plots done.")
