#!/usr/bin/env python3
"""Build an interactive clustering report (online + offline HTML).

The report layout is designed to mirror the style of scripts/build_volume_report.py:

- Left: clickable t-SNE scatter.
- Right: rows of cluster distribution plots, with a toggle between:
  - per-cluster metric rows
  - per-metric all-cluster overlays
- Center hero card: selected point metadata + cluster-level metrics.

The cluster-level metrics shown in the hero card come from
`kmeans_cluster_stats_summary.csv`.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import csv
import html as html_lib
import json
import re
from pathlib import Path

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def _fetch(url: str, cache_dir: Path) -> bytes:
    """Fetch a URL and cache by SHA1 prefix."""
    import hashlib
    import urllib.request

    cache_dir.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    suffix = Path(url.split("?")[0]).suffix or ".bin"
    cached = cache_dir / f"{name}{suffix}"
    if cached.exists():
        return cached.read_bytes()

    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "build_cluster_distribution_report/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    cached.write_bytes(data)
    return data


def png_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _cluster_sort_key(text: str) -> tuple[int, str]:
    s = text.strip()
    try:
        return (0, f"{int(s):09d}")
    except ValueError:
        return (1, s)


def _float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def read_summary_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out: dict[str, dict[str, str]] = {}
    for row in rows:
        cluster = (row.get("cluster") or "").strip()
        if not cluster:
            continue
        out[cluster] = row
    return out


def read_label_metadata(path: Path) -> tuple[dict[str, str], dict[str, str]]:
  """Read id->cluster and cluster->color maps from kmeans label CSV."""
  id_to_cluster: dict[str, str] = {}
  color_votes: dict[str, Counter[str]] = {}

  with path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
      sid = (row.get("id") or "").strip()
      cluster = (row.get("cluster") or "").strip()
      if sid and cluster:
        id_to_cluster[sid] = cluster

      color = (row.get("cluster_color") or "").strip().lower()
      if cluster and color and re.match(r"^#[0-9a-f]{6}$", color):
        color_votes.setdefault(cluster, Counter())[color] += 1

  cluster_to_color: dict[str, str] = {}
  for cluster, counter in color_votes.items():
    cluster_to_color[cluster] = counter.most_common(1)[0][0]

  return id_to_cluster, cluster_to_color


_CLUSTER_SUFFIX_RE = re.compile(r"_cluster(\d+)$", flags=re.IGNORECASE)


def read_embedding_points(path: Path, id_to_cluster: dict[str, str] | None = None) -> list[dict[str, str | float]]:
  points: list[dict[str, str | float]] = []
  id_to_cluster = id_to_cluster or {}
  with path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
      sid = (row.get("id") or "").strip()
      if not sid:
        continue

      cluster = id_to_cluster.get(sid, "")
      if not cluster:
        m = _CLUSTER_SUFFIX_RE.search(sid)
        if not m:
          continue
        cluster = m.group(1)

      x = _float_or_none(row.get("tsne1"))
      y = _float_or_none(row.get("tsne2"))
      if x is None or y is None:
        continue
      points.append({"id": sid, "cluster": cluster, "x": x, "y": y})
  return points


def discover_row_images(per_cluster_dir: Path) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    pat = re.compile(r"cluster_(\d+)_metrics_row\.png$", flags=re.IGNORECASE)
    for p in sorted(per_cluster_dir.glob("*.png")):
        m = pat.match(p.name)
        if not m:
            continue
        cluster = m.group(1)
        items.append({"cluster": cluster, "path": str(p), "name": p.name})
    items.sort(key=lambda d: _cluster_sort_key(d["cluster"]))
    return items


def discover_overlay_images(overlay_dir: Path) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    suf = "_all_clusters_overlay.png"
    for p in sorted(overlay_dir.glob("*.png")):
        if not p.name.endswith(suf):
            continue
        metric = p.name[: -len(suf)]
        title = metric.replace("_", " ")
        items.append({"metric": metric, "title": title, "path": str(p), "name": p.name})
    return items


def _fmt(v: str | None, digits: int = 4) -> str:
    f = _float_or_none(v)
    if f is None:
        return "n/a"
    return f"{f:.{digits}f}"


def build_html(
    *,
    title: str,
    summary_rows: dict[str, dict[str, str]],
    points: list[dict[str, str | float]],
    color_by_cluster: dict[str, str],
    tsne_b64: str,
    row_images: list[dict[str, str]],
    overlay_images: list[dict[str, str]],
    image_b64: dict[str, str],
    inline_plotly_js: str | None,
) -> str:
    clusters_sorted = sorted(summary_rows.keys(), key=_cluster_sort_key)

    # Palette fallback for clusters not present in labels CSV.
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    ]
    all_clusters = sorted({str(p["cluster"]) for p in points} | set(clusters_sorted), key=_cluster_sort_key)
    for i, c in enumerate(all_clusters):
        color_by_cluster.setdefault(c, palette[i % len(palette)])

    rows_minicards = []
    for c in clusters_sorted:
        r = summary_rows[c]
        rows_minicards.append(
            (
                c,
                {
                    "cluster": c,
                    "n_total": r.get("n_total", ""),
                    "n_with_stats": r.get("n_with_stats", ""),
                    "volume_mean": _fmt(r.get("volume_A3_mean"), 3),
                    "q_coord_mean": _fmt(r.get("q_tetra_coord_mean"), 4),
                    "q_ca_mean": _fmt(r.get("q_tetra_ca_mean"), 4),
                    "r_work_mean": _fmt(r.get("r_work_mean"), 4),
                    "r_free_mean": _fmt(r.get("r_free_mean"), 4),
                    "zn_bfactor_mean": _fmt(r.get("zn_bfactor_mean"), 3),
                    "dihedral_mean": _fmt(r.get("all_dihedrals_deg_mean"), 2),
                    "coord_res_bf_mean": _fmt(r.get("all_coord_res_bfactor_avg_mean"), 3),
                },
            )
        )

    state = {
        "summary": {c: d for c, d in rows_minicards},
        "clusters": clusters_sorted,
        "points": points,
        "colors": color_by_cluster,
        "initialCluster": clusters_sorted[0] if clusters_sorted else None,
        "tsneImage": f"data:image/png;base64,{tsne_b64}",
    }
    state_json = json.dumps(state)

    row_cards_html = "\n".join(
        (
            f'<div class="cluster-row-card" data-cluster="{html_lib.escape(it["cluster"])}">'
            f'<div class="cluster-row-title">Cluster {html_lib.escape(it["cluster"])}</div>'
            f'<img src="data:image/png;base64,{image_b64[it["path"]]}" alt="{html_lib.escape(it["name"])}">'
            f"</div>"
        )
        for it in row_images
    )

    overlay_cards_html = "\n".join(
        (
            f'<div class="metric-row-card">'
            f'<div class="cluster-row-title">{html_lib.escape(it["title"])}</div>'
            f'<img src="data:image/png;base64,{image_b64[it["path"]]}" alt="{html_lib.escape(it["name"])}">'
            f"</div>"
        )
        for it in overlay_images
    )

    if inline_plotly_js is None:
        plotly_html = f'<script src="{PLOTLY_CDN}"></script>'
    else:
        plotly_html = f"<script>{inline_plotly_js}</script>"

    css = """
    :root {
      --bg: #f3f1ed;
      --card: #ffffff;
      --ink: #1e1d1a;
      --muted: #6c6a63;
      --border: #d9d4ca;
      --accent: #0d6b60;
      --accent-soft: #e4f1ef;
      --hero-accent: #0d6b60;
      --hero-soft: rgba(13, 107, 96, 0.12);
      --hero-text: #111111;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at 20% -10%, #e7ece7 0%, var(--bg) 40%, #ebe7df 100%);
      color: var(--ink);
      font-family: "STIX Two Text", "Iowan Old Style", "Times New Roman", serif;
    }
    header {
      padding: 30px 28px 18px 28px;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(135deg, #f8faf7 0%, #ece8df 100%);
    }
    header h1 {
      margin: 0;
      font-size: 30px;
      letter-spacing: -0.01em;
    }
    header p {
      margin: 8px 0 0 0;
      color: var(--muted);
      font-size: 14px;
    }
    .wrap {
      max-width: 1700px;
      margin: 0 auto;
      padding: 18px 22px 24px 22px;
    }
    .top-layout {
      display: grid;
      grid-template-columns: minmax(400px, 540px) 1fr;
      gap: 16px;
      align-items: start;
    }
    .panel {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .panel-head {
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      background: #fcfbf8;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }
    #tsne-plot {
      width: 100%;
      height: 620px;
    }
    .toggle-bar {
      display: inline-flex;
      margin: 12px;
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }
    .toggle-bar button {
      border: none;
      padding: 9px 14px;
      font-size: 13px;
      cursor: pointer;
      background: white;
      color: #3d3b36;
      border-right: 1px solid var(--border);
    }
    .toggle-bar button:last-child { border-right: none; }
    .toggle-bar button.active {
      background: var(--accent);
      color: white;
      font-weight: 600;
    }
    .rows-scroll {
      max-height: 620px;
      overflow: auto;
      padding: 0 12px 12px 12px;
      display: grid;
      gap: 10px;
    }
    .cluster-row-card,
    .metric-row-card {
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }
    .cluster-row-title {
      padding: 8px 10px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: #4f4c46;
      background: #faf8f3;
      border-bottom: 1px solid var(--border);
    }
    .cluster-row-card img,
    .metric-row-card img {
      width: 100%;
      height: auto;
      display: block;
    }
    .hero {
      margin: 0 0 16px 0;
      border: 2px solid var(--hero-accent);
      border-radius: 12px;
      background: linear-gradient(180deg, #ffffff 0%, #f8fbfa 100%);
      padding: 14px;
      box-shadow: 0 0 0 1px rgba(0,0,0,0.03), 0 4px 16px rgba(0,0,0,0.05);
    }
    .hero-head {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 10px;
      margin-bottom: 10px;
    }
    .hero-title {
      margin: 0;
      font-size: 20px;
    }
    .hero-sub {
      color: var(--muted);
      font-size: 13px;
      margin-top: 4px;
    }
    .hero-right {
      text-align: right;
      font-size: 13px;
      color: #43413c;
      line-height: 1.4;
      min-width: 220px;
    }
    .hero-right .cluster-line {
      font-weight: 700;
      color: var(--hero-accent);
    }
    .hero-right .count-line {
      color: #5c5952;
    }
    .hero-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(140px, 1fr));
      gap: 8px;
    }
    .metric {
      border: 1px solid #d7e3e0;
      border-radius: 8px;
      padding: 8px;
      background: var(--hero-soft);
    }
    .metric .k {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--hero-text);
      opacity: 0.82;
    }
    .metric .v {
      margin-top: 2px;
      font-size: 17px;
      font-weight: 600;
      color: var(--hero-text);
    }
    @media (max-width: 1250px) {
      .top-layout { grid-template-columns: 1fr; }
      #tsne-plot { height: 500px; }
      .rows-scroll { max-height: 560px; }
      .hero-grid { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
    }
    """

    js = """
const STATE = """ + state_json + """;

function showRows(mode) {
  const a = document.getElementById("rows-per-cluster");
  const b = document.getElementById("rows-per-metric");
  const ba = document.getElementById("btn-cluster");
  const bb = document.getElementById("btn-metric");
  const clusterMode = mode === "cluster";
  a.style.display = clusterMode ? "grid" : "none";
  b.style.display = clusterMode ? "none" : "grid";
  ba.classList.toggle("active", clusterMode);
  bb.classList.toggle("active", !clusterMode);
}

function metricHtml(label, value) {
  return '<div class="metric"><div class="k">' + label + '</div><div class="v">' + value + '</div></div>';
}

function hexToRgb(hex) {
  const m = /^#?([a-f\\d]{2})([a-f\\d]{2})([a-f\\d]{2})$/i.exec(hex || "");
  if (!m) return null;
  return {
    r: parseInt(m[1], 16),
    g: parseInt(m[2], 16),
    b: parseInt(m[3], 16),
  };
}

function updateHero(point) {
  const cluster = String(point.cluster);
  const m = STATE.summary[cluster];
  document.getElementById("hero-title").textContent = point.id;
  document.getElementById("hero-sub").textContent = "Selected from t-SNE";

  const nTotal = m ? (m.n_total || "n/a") : "n/a";
  const nStats = m ? (m.n_with_stats || "n/a") : "n/a";
  document.getElementById("hero-cluster-line").textContent = "Cluster " + cluster;
  document.getElementById("hero-count-line").textContent = "N total: " + nTotal + " | N stats: " + nStats;

  const accent = STATE.colors[cluster] || "#0d6b60";
  const rgb = hexToRgb(accent);
  document.documentElement.style.setProperty("--hero-accent", accent);
  if (rgb) {
    document.documentElement.style.setProperty("--hero-soft", "rgba(" + rgb.r + ", " + rgb.g + ", " + rgb.b + ", 0.14)");
  }
  document.documentElement.style.setProperty("--hero-text", "#111111");

  if (!m) {
    document.getElementById("hero-grid").innerHTML = "";
    return;
  }

  const html = [
    metricHtml("volume mean (A^3)", m.volume_mean),
    metricHtml("q tetra coord mean", m.q_coord_mean),
    metricHtml("q tetra CA mean", m.q_ca_mean),
    metricHtml("r work mean", m.r_work_mean),
    metricHtml("r free mean", m.r_free_mean),
    metricHtml("Zn B-factor mean", m.zn_bfactor_mean),
    metricHtml("dihedral mean (deg)", m.dihedral_mean),
    metricHtml("coord-res B mean", m.coord_res_bf_mean),
  ].join("");
  document.getElementById("hero-grid").innerHTML = html;
}

function buildTsnePlot() {
  const byCluster = {};
  STATE.points.forEach(p => {
    const c = String(p.cluster);
    if (!byCluster[c]) byCluster[c] = [];
    byCluster[c].push(p);
  });

  const traces = Object.keys(byCluster).sort((a, b) => Number(a) - Number(b)).map(c => {
    const pts = byCluster[c];
    return {
      x: pts.map(p => p.x),
      y: pts.map(p => p.y),
      customdata: pts,
      mode: "markers",
      type: "scattergl",
      name: "cluster " + c,
      marker: {
        size: 7,
        color: STATE.colors[c] || "#444",
        opacity: 0.72,
        line: { width: 0 }
      },
      hovertemplate:
        "<b>%{customdata.id}</b><br>cluster %{customdata.cluster}<br>" +
        "t-SNE: (%{x:.2f}, %{y:.2f})<extra></extra>",
    };
  });

  const xs = STATE.points.map(p => p.x);
  const ys = STATE.points.map(p => p.y);
  const xmin = Math.min.apply(null, xs);
  const xmax = Math.max.apply(null, xs);
  const ymin = Math.min.apply(null, ys);
  const ymax = Math.max.apply(null, ys);
  const padx = (xmax - xmin) * 0.06;
  const pady = (ymax - ymin) * 0.06;

  const layout = {
    margin: { l: 48, r: 14, t: 24, b: 42 },
    xaxis: { title: "t-SNE 1", range: [xmin - padx, xmax + padx], zeroline: false },
    yaxis: { title: "t-SNE 2", range: [ymin - pady, ymax + pady], zeroline: false },
    legend: { orientation: "h", yanchor: "bottom", y: 1.02, x: 0 },
    hovermode: "closest",
    plot_bgcolor: "white",
    paper_bgcolor: "white"
  };

  Plotly.newPlot("tsne-plot", traces, layout, { displayModeBar: false, responsive: true });

  const plot = document.getElementById("tsne-plot");
  plot.on("plotly_click", ev => {
    if (!ev || !ev.points || !ev.points.length) return;
    const p = ev.points[0].customdata;
    if (!p) return;
    updateHero(p);
  });

  if (STATE.points.length) updateHero(STATE.points[0]);
}

window.addEventListener("DOMContentLoaded", () => {
  document.getElementById("btn-cluster").addEventListener("click", () => showRows("cluster"));
  document.getElementById("btn-metric").addEventListener("click", () => showRows("metric"));
  showRows("cluster");
  buildTsnePlot();
});
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html_lib.escape(title)}</title>
  {plotly_html}
  <style>{css}</style>
</head>
<body>
  <header>
    <h1>{html_lib.escape(title)}</h1>
    <p>Clickable t-SNE at left. Right panel toggles between per-cluster rows and per-metric overlays. Hero metrics are sourced from kmeans_cluster_stats_summary.csv.</p>
  </header>
  <main class="wrap">
    <section class="hero">
      <div class="hero-head">
        <div>
          <h2 id="hero-title" class="hero-title">No point selected</h2>
          <div id="hero-sub" class="hero-sub">Click a point in the t-SNE panel.</div>
        </div>
        <div class="hero-right">
          <div id="hero-cluster-line" class="cluster-line">Cluster -</div>
          <div id="hero-count-line" class="count-line">N total: - | N stats: -</div>
        </div>
      </div>
      <div id="hero-grid" class="hero-grid"></div>
    </section>

    <section class="top-layout">
      <div class="panel">
        <div class="panel-head">t-SNE (click points)</div>
        <div id="tsne-plot"></div>
      </div>
      <div class="panel">
        <div class="panel-head">Cluster Distribution Panels</div>
        <div class="toggle-bar">
          <button id="btn-cluster" class="active">Per Cluster Rows</button>
          <button id="btn-metric">Per Metric Overlays</button>
        </div>
        <div id="rows-per-cluster" class="rows-scroll">{row_cards_html}</div>
        <div id="rows-per-metric" class="rows-scroll" style="display:none;">{overlay_cards_html}</div>
      </div>
    </section>
  </main>
  <script>{js}</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--clustering-dir",
        type=Path,
        default=Path("data/large-cys-his-datasets/4cys-large/clustering"),
        help="Directory containing embeddings.csv, kmeans_cluster_stats_summary.csv, tsne_kmeans.png, and cluster_distribution_plots/.",
    )
    p.add_argument(
        "--title",
        default="4Cys Large Cluster Distribution Report",
        help="Report title displayed in the HTML.",
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Path to kmeans_cluster_stats_summary.csv (default: <clustering-dir>/kmeans_cluster_stats_summary.csv).",
    )
    p.add_argument(
        "--embeddings-csv",
        type=Path,
        default=None,
        help="Path to embeddings.csv with tsne1/tsne2/id columns (default: <clustering-dir>/embeddings.csv).",
    )
    p.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Path to kmeans_labels_with_stats.csv for id->cluster and cluster_color mapping (default: <clustering-dir>/kmeans_labels_with_stats.csv).",
    )
    p.add_argument(
        "--tsne-png",
        type=Path,
        default=None,
        help="Path to tsne_kmeans.png (default: <clustering-dir>/tsne_kmeans.png).",
    )
    p.add_argument(
        "--out-online",
        type=Path,
        default=None,
        help="Output path for CDN-backed HTML (default: <clustering-dir>/report_cluster_distribution.html).",
    )
    p.add_argument(
        "--out-offline",
        type=Path,
        default=None,
        help="Output path for fully offline HTML (default: <clustering-dir>/report_cluster_distribution_offline.html).",
    )
    p.add_argument(
        "--vendor-cache",
        type=Path,
        default=Path.home() / ".cache" / "build_cluster_distribution_report",
        help="Cache directory used for downloading Plotly for offline builds.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    clustering_dir = args.clustering_dir

    summary_csv = args.summary_csv or (clustering_dir / "kmeans_cluster_stats_summary.csv")
    embeddings_csv = args.embeddings_csv or (clustering_dir / "embeddings.csv")
    labels_csv = args.labels_csv or (clustering_dir / "kmeans_labels_with_stats.csv")
    tsne_png = args.tsne_png or (clustering_dir / "tsne_kmeans.png")

    plots_root = clustering_dir / "cluster_distribution_plots"
    per_cluster_dir = plots_root / "per_cluster_rows"
    overlay_dir = plots_root / "all_cluster_overlays"

    out_online = args.out_online or (clustering_dir / "report_cluster_distribution.html")
    out_offline = args.out_offline or (clustering_dir / "report_cluster_distribution_offline.html")

    required = [summary_csv, embeddings_csv, tsne_png, per_cluster_dir, overlay_dir]
    missing = [p for p in required if not p.exists()]
    if missing:
        missing_text = "\n".join(f"- {p}" for p in missing)
        raise SystemExit(f"Missing required inputs:\n{missing_text}")

    id_to_cluster: dict[str, str] = {}
    color_by_cluster: dict[str, str] = {}
    if labels_csv.exists():
        id_to_cluster, color_by_cluster = read_label_metadata(labels_csv)
    else:
        print(f"Warning: labels CSV not found, falling back to cluster suffix colors: {labels_csv}")

    summary_rows = read_summary_csv(summary_csv)
    points = read_embedding_points(embeddings_csv, id_to_cluster=id_to_cluster)
    if not summary_rows:
        raise SystemExit(f"No cluster rows found in {summary_csv}")
    if not points:
        raise SystemExit(f"No t-SNE points could be parsed from {embeddings_csv}")

    row_images = discover_row_images(per_cluster_dir)
    overlay_images = discover_overlay_images(overlay_dir)
    if not row_images:
        raise SystemExit(f"No per-cluster row images found in {per_cluster_dir}")
    if not overlay_images:
        raise SystemExit(f"No overlay images found in {overlay_dir}")

    image_paths = [tsne_png] + [Path(i["path"]) for i in row_images + overlay_images]
    image_b64 = {str(p): png_to_b64(p) for p in image_paths}
    tsne_b64 = image_b64[str(tsne_png)]

    online_html = build_html(
        title=args.title,
        summary_rows=summary_rows,
        points=points,
        color_by_cluster=dict(color_by_cluster),
        tsne_b64=tsne_b64,
        row_images=row_images,
        overlay_images=overlay_images,
        image_b64=image_b64,
        inline_plotly_js=None,
    )

    print("Building offline HTML (inlining Plotly.js)...")
    inline_plotly_js = _fetch(PLOTLY_CDN, args.vendor_cache).decode("utf-8")
    offline_html = build_html(
        title=args.title,
        summary_rows=summary_rows,
        points=points,
        color_by_cluster=dict(color_by_cluster),
        tsne_b64=tsne_b64,
        row_images=row_images,
        overlay_images=overlay_images,
        image_b64=image_b64,
        inline_plotly_js=inline_plotly_js,
    )

    out_online.parent.mkdir(parents=True, exist_ok=True)
    out_offline.parent.mkdir(parents=True, exist_ok=True)

    out_online.write_text(online_html, encoding="utf-8")
    out_offline.write_text(offline_html, encoding="utf-8")

    s1 = out_online.stat().st_size / (1024 * 1024)
    s2 = out_offline.stat().st_size / (1024 * 1024)
    print(f"✓ Wrote {out_online} ({s1:.2f} MB)")
    print(f"✓ Wrote {out_offline} ({s2:.2f} MB)")


if __name__ == "__main__":
    main()
