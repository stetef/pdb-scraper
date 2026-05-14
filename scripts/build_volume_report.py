#!/usr/bin/env python3
"""Build a self-contained HTML report for the volume-study output.

Reads `volume_extremes_<label>.csv` plus the full-dataset PNGs from a figures
directory and emits a single HTML file with:

  - The two full-dataset scatter PNGs (base64-embedded) at the top of the
    Cys section.
  - Two interactive Plotly scatters of the 5 smallest + 5 largest CA-volume
    extremes. Each structure's per-atom (Zn->X distance, Cys dihedral) values
    are drawn as light-gray reference points; only the structure-level
    AVERAGE point is colored and clickable.
  - A centered "selected structure" hero card with a 3Dmol.js viewer that
    rebuilds itself on every selection. The list of all extreme structures
    sits below as compact text-only mini-cards.
  - The same flow for low-q_tetra(S) structures: full-dataset PNG, an
    interactive Plotly scatter (half-width, centered, and sticky alongside
    the card list on wide screens), a hero card, and the 11 mini-cards.

Usage
-----
    uv run python scripts/build_volume_report.py \
        --figures-dir data/large-cys-his-datasets/4cys-large/figures \
        --label 4cys-large-dataset

Output goes to `<figures-dir>/report_<label>.html` by default. Plotly.js and
3Dmol.js are loaded from CDN, so the resulting HTML opens in any modern
browser with internet access.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html as html_lib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
THREEDMOL_CDN = "https://3Dmol.org/build/3Dmol-min.js"
KATEX_CSS = "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css"
KATEX_JS = "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"
KATEX_AUTORENDER_JS = "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"

# Viridis-derived palette to match matplotlib plots.
ACCENT = "#c2185b"           # rose accent (highlight); reads well over viridis
SMALLEST_COLOR = "#440154"   # viridis @ 0.0 — deep purple
LARGEST_COLOR = "#fde725"    # viridis @ 1.0 — yellow
QS_COLOR = "#21918c"         # viridis @ 0.5 — teal
GRAY = "#cdcdcd"
DEFAULT_SIZE = 11
HIGHLIGHT_SIZE = 18
GRAY_SIZE = 6


@dataclass
class StructureRow:
    group: str
    rank: int
    volume: float
    family: str
    dihedrals: list[float]
    dihedral_mean: float
    q_tetra_coord: float
    q_tetra_ca: float
    xyz_path: Path
    key: str
    coord_distances: list[float] = field(default_factory=list)

    @property
    def coord_distance_mean(self) -> float:
        return sum(self.coord_distances) / len(self.coord_distances) if self.coord_distances else 0.0


def read_extremes_csv(path: Path) -> list[StructureRow]:
    rows: list[StructureRow] = []
    skipped_extended = 0
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            xyz_path = Path(r["xyz_path"]).expanduser()
            # Defensive: if xyz-volume-study.py was re-run after extended XYZ
            # files were created, they may have leaked into the CSV. Drop them.
            if xyz_path.stem.endswith("-extended"):
                skipped_extended += 1
                continue
            rows.append(
                StructureRow(
                    group=r["group"].strip(),
                    rank=int(r["rank"]),
                    volume=float(r["volume_A3"]),
                    family=r["family"],
                    dihedrals=[float(r[f"cys_dihedral_{i}_deg"]) for i in (1, 2, 3, 4)],
                    dihedral_mean=float(r["cys_dihedral_mean_deg"]),
                    q_tetra_coord=float(r["q_tetra_coord"]),
                    q_tetra_ca=float(r["q_tetra_ca"]),
                    xyz_path=xyz_path,
                    key=xyz_path.stem,
                )
            )
    if skipped_extended:
        print(
            f"Note: dropped {skipped_extended} '*-extended.xyz' rows from {path.name}; "
            f"re-run xyz-volume-study.py to regenerate the CSV cleanly."
        )
    return rows


def read_xyz(path: Path) -> str:
    return path.read_text(encoding="utf-8")


_ATOM_TAG_RE = re.compile(r"\bATOM=(\w+)\b")
_RESSEQ_TAG_RE = re.compile(r"\bRESSEQ=(\S+)\b")
_CHAIN_TAG_RE = re.compile(r"\bCHAIN=(\S+)\b")
_RES_TAG_RE = re.compile(r"\bRES=(\w+)\b")
_COORD_TAG_RE = re.compile(r"\bCOORD=(\w+)\b")


def coord_distances_from_xyz(xyz_text: str) -> list[float]:
    """Distances from origin (Zn) to each coordinating atom.

    Mirrors the selection logic used in scripts/xyz-volume-study.py: for each
    Cys residue keep SG; for each His residue prefer the atom flagged
    `COORD=1`/`COORD=TRUE`, otherwise pick the closer of ND1/NE2.
    """
    @dataclass
    class _Atom:
        atom: str
        x: float
        y: float
        z: float
        coord_flag: str

    by_res: dict[tuple[str, str, str], dict[str, _Atom]] = {}
    for line in xyz_text.splitlines()[2:]:
        if "#" not in line:
            continue
        coord_part, comment = line.split("#", 1)
        atom_match = _ATOM_TAG_RE.search(comment)
        res_match = _RES_TAG_RE.search(comment)
        chain_match = _CHAIN_TAG_RE.search(comment)
        resseq_match = _RESSEQ_TAG_RE.search(comment)
        if not (atom_match and res_match and chain_match and resseq_match):
            continue
        atom_name = atom_match.group(1).upper()
        if atom_name not in ("SG", "ND1", "NE2"):
            continue
        try:
            parts = coord_part.split()
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
        except (IndexError, ValueError):
            continue
        coord_flag = (_COORD_TAG_RE.search(comment).group(1) if _COORD_TAG_RE.search(comment) else "")
        key = (chain_match.group(1), res_match.group(1).upper(), resseq_match.group(1))
        by_res.setdefault(key, {})[atom_name] = _Atom(atom_name, x, y, z, coord_flag.upper())

    chosen: list[tuple[float, _Atom]] = []
    for atoms in by_res.values():
        sg = atoms.get("SG")
        if sg is not None:
            d2 = sg.x * sg.x + sg.y * sg.y + sg.z * sg.z
            chosen.append((d2, sg))
            continue
        flagged = None
        for name in ("ND1", "NE2"):
            a = atoms.get(name)
            if a is not None and a.coord_flag in ("1", "TRUE"):
                flagged = a
                break
        if flagged is not None:
            d2 = flagged.x * flagged.x + flagged.y * flagged.y + flagged.z * flagged.z
            chosen.append((d2, flagged))
            continue
        best = None
        best_d2 = math.inf
        for name in ("ND1", "NE2"):
            a = atoms.get(name)
            if a is None:
                continue
            d2 = a.x * a.x + a.y * a.y + a.z * a.z
            if d2 < best_d2:
                best_d2 = d2
                best = a
        if best is not None:
            chosen.append((best_d2, best))

    chosen.sort(key=lambda t: t[0])
    chosen = chosen[:4]
    return [math.sqrt(d2) for d2, _ in chosen]


def png_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


# --------------------------------------------------------------------------
# Offline-mode asset embedding
# --------------------------------------------------------------------------

def _fetch(url: str, cache_dir: Path) -> bytes:
    """Fetch a URL, caching to cache_dir (keyed by URL hash). Returns bytes."""
    import hashlib
    import urllib.request

    cache_dir.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    suffix = Path(url.split("?")[0]).suffix or ".bin"
    cached = cache_dir / f"{name}{suffix}"
    if cached.exists():
        return cached.read_bytes()
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "build_volume_report/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    cached.write_bytes(data)
    return data


_FONT_MIME = {".woff2": "font/woff2", ".woff": "font/woff", ".ttf": "font/ttf", ".otf": "font/otf"}


def _inline_katex_css(css_url: str, cache_dir: Path) -> str:
    """Fetch KaTeX CSS and replace every url(...) font reference with a data: URI."""
    from urllib.parse import urljoin

    css_text = _fetch(css_url, cache_dir).decode("utf-8")

    def repl(m: "re.Match[str]") -> str:
        raw = m.group(1).strip().strip('"').strip("'")
        if raw.startswith("data:"):
            return m.group(0)
        font_url = raw if raw.startswith("http") else urljoin(css_url, raw)
        ext = Path(font_url.split("?")[0]).suffix.lower()
        mime = _FONT_MIME.get(ext, "application/octet-stream")
        try:
            blob = _fetch(font_url, cache_dir)
        except Exception as exc:
            print(f"  warning: failed to fetch font {font_url}: {exc}")
            return m.group(0)
        b64 = base64.b64encode(blob).decode("ascii")
        return f'url("data:{mime};base64,{b64}")'

    return re.sub(r"url\(([^)]+)\)", repl, css_text)


def build_mini_card_html(row: StructureRow, section: str) -> str:
    """A compact, clickable text-only card. The hero card shows the 3D viewer."""
    return f"""
    <div class="mini-card" data-key="{html_lib.escape(row.key)}" data-section="{section}">
      <div class="mini-rank">#{row.rank}</div>
      <div class="mini-name">{html_lib.escape(row.key)}</div>
      <div class="mini-stats">
        <span><b>V</b> {row.volume:.3f} &Aring;<sup>3</sup></span>
        <span><b>q<sub>S</sub></b> {row.q_tetra_coord:.3f}</span>
        <span><b>q<sub>&alpha;</sub></b> {row.q_tetra_ca:.3f}</span>
      </div>
    </div>
    """


def row_to_dict(row: StructureRow) -> dict:
    return {
        "key": row.key,
        "rank": row.rank,
        "group": row.group,
        "volume": row.volume,
        "family": row.family,
        "dihedrals": row.dihedrals,
        "dihedral_mean": row.dihedral_mean,
        "q_tetra_coord": row.q_tetra_coord,
        "q_tetra_ca": row.q_tetra_ca,
        "coord_distances": row.coord_distances,
        "coord_distance_mean": row.coord_distance_mean,
    }


def build_html(
    *,
    cys_smallest: list[StructureRow],
    cys_largest: list[StructureRow],
    qs_rows: list[StructureRow],
    xyz_by_key: dict[str, str],
    xyz_extended_by_key: dict[str, str],
    png_b64: dict[str, str],
    label: str,
    system_name: str,
    pdb_count: int | None,
    xyz_count: int | None,
    inline_assets: dict[str, str] | None = None,
) -> str:
    cys_extremes = cys_smallest + cys_largest

    coord_atom_points: list[dict] = []
    coord_avg_points: list[dict] = []
    for row in cys_extremes:
        color = LARGEST_COLOR if "largest" in row.group else SMALLEST_COLOR
        for i, d in enumerate(row.coord_distances):
            coord_atom_points.append({"key": row.key, "volume": row.volume, "y": d, "atom_idx": i + 1})
        coord_avg_points.append({
            "key": row.key, "volume": row.volume, "y": row.coord_distance_mean,
            "color": color, "group": row.group, "rank": row.rank,
        })

    qs_points = [
        {
            "key": row.key, "volume": row.volume, "y": row.q_tetra_coord,
            "q_tetra_s": row.q_tetra_coord, "q_tetra_ca": row.q_tetra_ca,
            "color": QS_COLOR, "rank": row.rank,
        }
        for row in qs_rows
    ]

    cys_rows_map = {row.key: row_to_dict(row) for row in cys_extremes}
    qs_rows_map = {row.key: row_to_dict(row) for row in qs_rows}

    cys_smallest_cards = "".join(build_mini_card_html(r, "cys") for r in cys_smallest)
    cys_largest_cards = "".join(build_mini_card_html(r, "cys") for r in cys_largest)
    qs_cards = "".join(build_mini_card_html(r, "qs") for r in qs_rows)

    js_state = {
        "coordAtomPoints": coord_atom_points,
        "coordAvgPoints": coord_avg_points,
        "qsPoints": qs_points,
        "xyzExtendedByKey": xyz_extended_by_key,
        "cysRowsMap": cys_rows_map,
        "qsRowsMap": qs_rows_map,
        "xyzByKey": xyz_by_key,
        "accent": ACCENT,
        "smallestColor": SMALLEST_COLOR,
        "largestColor": LARGEST_COLOR,
        "qsColor": QS_COLOR,
        "grayColor": GRAY,
        "defaultSize": DEFAULT_SIZE,
        "highlightSize": HIGHLIGHT_SIZE,
        "graySize": GRAY_SIZE,
        "initialCysKey": cys_smallest[0].key if cys_smallest else (cys_largest[0].key if cys_largest else None),
        "initialQsKey": qs_rows[0].key if qs_rows else None,
    }
    js_state_json = json.dumps(js_state)

    title = f"Volume study report — {label}"

    css = """
    :root {
      --bg: #f5f5f3;
      --card-bg: #ffffff;
      --muted: #777;
      --border: #e3e3df;
      --accent: """ + ACCENT + """;
      --smallest: """ + SMALLEST_COLOR + """;
      --largest: """ + LARGEST_COLOR + """;
      --qs: """ + QS_COLOR + """;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; padding: 0;
      font-family: 'Computer Modern Serif', 'CMU Serif', 'STIX Two Text',
                   'Times New Roman', Times, 'Liberation Serif', serif;
      background: var(--bg);
      color: #1f1f1f;
      font-size: 15px;
      line-height: 1.55;
    }
    .mono {
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    }
    header {
      background:
        linear-gradient(135deg,
          #f7f7f8 0%,
          #e3e4e6 18%,
          #cdced0 40%,
          #b9babd 55%,
          #d3d4d6 75%,
          #ededef 100%);
      color: #1a1a1a;
      padding: 44px 36px 32px 36px;
      border-bottom: 1px solid #a6a8ac;
      box-shadow: inset 0 -1px 0 rgba(255,255,255,0.6),
                  inset 0 1px 0 rgba(255,255,255,0.7);
    }
    .header-inner { max-width: 1500px; margin: 0 auto; }
    header h1 {
      margin: 0 0 6px 0;
      font-size: 32px;
      font-weight: 700;
      letter-spacing: -0.005em;
      color: #1a1a1a;
    }
    .header-stats {
      display: flex; gap: 22px;
      margin: 0 0 22px 0;
      color: #3a3a3a;
      font-size: 15px;
    }
    .header-stats .stat {
      display: inline-flex; align-items: baseline; gap: 6px;
    }
    .header-stats .stat-num {
      font-weight: 700; font-size: 20px; color: #111;
    }
    .header-stats .stat-sep {
      color: #888; font-weight: 300;
    }
    .header-summary {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 14px;
    }
    .header-card {
      background: rgba(255,255,255,0.65);
      border-radius: 6px;
      padding: 14px 18px;
      border: 1px solid rgba(120,120,125,0.32);
      border-left: 4px solid #555;
      box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .header-card.cys { border-left-color: """ + SMALLEST_COLOR + """; }
    .header-card.qs { border-left-color: """ + QS_COLOR + """; }
    .header-card h3 {
      margin: 0 0 6px 0;
      font-size: 15px;
      font-weight: 700;
      letter-spacing: 0;
      color: #1a1a1a;
    }
    .header-card p {
      margin: 0;
      font-size: 14px;
      color: #2c2c2c;
      line-height: 1.45;
    }
    .header-card .swatch-row {
      display: flex; gap: 6px; margin-top: 8px;
    }
    .header-card .swatch {
      display: inline-block; width: 14px; height: 14px;
      border-radius: 2px;
      border: 1px solid rgba(0,0,0,0.18);
    }
    .header-tip {
      max-width: 1500px; margin: 14px auto 0 auto;
      font-size: 13px; color: #4a4a4a; font-style: italic;
    }
    section {
      max-width: 1500px;
      margin: 64px auto;
      padding: 0 24px;
    }
    section + section { margin-top: 96px; }
    section h2 {
      font-size: 26px;
      font-weight: 700;
      margin: 0 0 18px 0;
      border-bottom: 2px solid #b8b8b8;
      padding-bottom: 12px;
      letter-spacing: -0.005em;
      color: #1a1a1a;
    }
    section h2 .section-num {
      display: inline-block;
      width: 30px; height: 30px;
      line-height: 30px;
      text-align: center;
      background: var(--border);
      color: #444;
      border-radius: 50%;
      font-size: 14px;
      font-weight: 700;
      margin-right: 12px;
      vertical-align: 3px;
      font-family: ui-monospace, "SF Mono", Menlo, monospace;
    }
    section.cys h2 .section-num { background: """ + SMALLEST_COLOR + """; color: white; }
    section.qs h2 .section-num { background: """ + QS_COLOR + """; color: white; }
    .info-panel {
      background: #fafaf8;
      border: 1px solid var(--border);
      border-left: 3px solid """ + QS_COLOR + """;
      border-radius: 4px;
      padding: 14px 18px;
      margin-bottom: 18px;
      font-size: 14px;
      color: #2a2a2a;
    }
    section.cys .info-panel { border-left-color: """ + SMALLEST_COLOR + """; }
    .info-panel p { margin: 0 0 8px 0; }
    .info-panel p:last-child { margin-bottom: 0; }
    .info-panel .formula {
      display: block;
      text-align: center;
      margin: 12px 0;
      font-size: 16px;
      overflow-x: auto;
    }
    .row { display: grid; gap: 14px; margin-bottom: 14px; }
    .row.two-col { grid-template-columns: 1fr 1fr; }
    .panel {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
    }
    .panel img { width: 100%; height: auto; display: block; border-radius: 4px; }
    .panel .panel-title {
      font-size: 11px; color: var(--muted); margin-bottom: 4px;
      text-transform: uppercase; letter-spacing: 0.04em;
    }
    .plot-small { width: 100%; height: 300px; }

    /* Hero card: the prominent "selected structure" view. */
    .hero {
      background: var(--card-bg);
      border: 2px solid var(--accent);
      border-radius: 10px;
      padding: 14px 16px;
      margin: 14px auto;
      max-width: 1100px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .hero-empty {
      color: var(--muted);
      font-size: 13px;
      text-align: center;
      padding: 30px 0;
    }
    .hero[data-populated="true"] .hero-empty { display: none; }
    .hero[data-populated="false"] .hero-content { display: none; }
    .hero-content {
      display: grid;
      grid-template-columns: minmax(360px, 1fr) minmax(260px, 360px);
      gap: 18px;
      align-items: start;
    }
    .hero-viewer {
      width: 100%; height: 380px;
      background: white; border: 1px solid var(--border); border-radius: 6px;
      overflow: hidden; position: relative;
    }
    .hero-meta {
      font-size: 13px; line-height: 1.6;
    }
    .hero-meta h3 {
      margin: 0 0 8px 0;
      font-family: ui-monospace, "SF Mono", Menlo, monospace;
      font-size: 14px; color: var(--accent);
      word-break: break-all;
    }
    .hero-meta .meta-row { margin-bottom: 4px; }
    .hero-meta b { color: #333; font-weight: 600; }
    .hero-meta .muted { color: var(--muted); font-size: 12px; }
    .meta-pdb-link {
      margin: 0 0 10px 0;
      font-size: 13px;
    }
    .meta-pdb-link a {
      color: var(--accent);
      text-decoration: none;
      font-weight: 600;
      letter-spacing: 0.02em;
    }
    .meta-pdb-link a:hover { text-decoration: underline; }
    .hero-toggle {
      display: inline-flex;
      border: 1px solid #b9babd;
      border-radius: 4px;
      overflow: hidden;
      margin: 0 0 12px 0;
      font-family: inherit;
    }
    .hero-toggle button {
      background: white;
      border: none;
      padding: 5px 14px;
      font-size: 12px;
      cursor: pointer;
      color: #444;
      font-family: inherit;
      transition: background 0.12s ease, color 0.12s ease;
    }
    .hero-toggle button:not(:last-child) { border-right: 1px solid #b9babd; }
    .hero-toggle button:hover:not(.active):not(:disabled) { background: #f0f0ee; }
    .hero-toggle button.active {
      background: var(--accent);
      color: white;
      font-weight: 600;
    }
    .hero-toggle button:disabled {
      color: #b0b0b0;
      cursor: not-allowed;
      background: #f7f7f6;
    }
    .hero-tag {
      display: inline-block;
      font-size: 11px;
      padding: 2px 8px;
      border-radius: 12px;
      color: white;
      background: var(--smallest);
      margin-right: 8px;
    }
    .hero-tag.largest { background: var(--largest); }
    .hero-tag.qs { background: var(--qs); }

    /* Mini-cards: text-only list. */
    .mini-cards {
      display: grid;
      gap: 8px;
    }
    .mini-cards.two-col { grid-template-columns: 1fr 1fr; }
    .mini-cards-col-title {
      font-size: 11px; font-weight: 600; color: var(--muted);
      text-transform: uppercase; letter-spacing: 0.04em;
      margin-bottom: 6px;
    }
    .mini-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-left: 4px solid var(--smallest);
      border-radius: 6px;
      padding: 8px 12px;
      cursor: pointer;
      transition: background 0.12s ease, border-color 0.12s ease;
      display: grid;
      grid-template-columns: 36px 1fr auto;
      align-items: center;
      gap: 10px;
      opacity: 0.6;
    }
    .mini-card:hover { background: #fafaf8; opacity: 0.95; }
    .mini-card.active {
      opacity: 1;
      border-left-color: var(--accent);
      border-color: var(--accent);
      background: #fff7f7;
    }
    /* color the left bar by group */
    .mini-card[data-group-color="largest"] { border-left-color: var(--largest); }
    .mini-card[data-group-color="qs"] { border-left-color: var(--qs); }
    .mini-card.active { border-left-color: var(--accent); }
    .mini-rank {
      font-size: 11px; font-weight: 700; color: var(--muted);
      font-family: ui-monospace, "SF Mono", Menlo, monospace;
    }
    .mini-card.active .mini-rank { color: var(--accent); }
    .mini-name {
      font-family: ui-monospace, "SF Mono", Menlo, monospace;
      font-size: 12px;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .mini-stats {
      display: flex; gap: 10px;
      font-size: 11px;
      color: #555;
      white-space: nowrap;
    }
    .mini-stats b { color: #333; }
    .mini-stats sub { font-size: 9px; }

    /* Two-column section layout: sticky plot column on the left, cards on the right. */
    .split-layout {
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
    }
    .split-layout .plot-col .panel { padding: 8px; }
    .split-layout .plot-col .plot-half { width: 100%; height: 320px; }
    .split-layout .plot-col img { max-height: 320px; object-fit: contain; margin: 0 auto; }
    @media (min-width: 1180px) {
      .split-layout {
        grid-template-columns: minmax(420px, 520px) 1fr;
        align-items: start;
      }
      .split-layout .plot-col {
        position: sticky;
        top: 12px;
        align-self: start;
      }
    }
    @media (max-width: 1179px) {
      .split-layout .plot-col {
        max-width: 720px;
        margin: 0 auto;
      }
    }
    .group-divider {
      font-size: 11px; font-weight: 700; color: var(--muted);
      text-transform: uppercase; letter-spacing: 0.05em;
      margin: 14px 0 6px 0;
      padding-bottom: 4px;
      border-bottom: 1px solid var(--border);
    }
    .group-divider:first-child { margin-top: 0; }
    .group-divider .swatch {
      display: inline-block; width: 10px; height: 10px;
      border-radius: 2px; margin-right: 6px; vertical-align: middle;
    }
    .group-divider .swatch.smallest { background: var(--smallest); }
    .group-divider .swatch.largest { background: var(--largest); }
    .group-divider .swatch.qs { background: var(--qs); }

    .legend-hint {
      font-size: 12px; color: var(--muted); margin: -2px 0 10px 0;
    }
    """

    js = """
const STATE = """ + js_state_json + """;

// ---------- Plot construction ----------

function buildScatterPlot(divId, opts) {
    // Background gray trace (per-atom decorative) - non-interactive.
    const grayTrace = opts.atomPoints && opts.atomPoints.length ? {
        x: opts.atomPoints.map(p => p[opts.xKey]),
        y: opts.atomPoints.map(p => p.y),
        mode: "markers",
        type: "scatter",
        marker: { color: STATE.grayColor, size: STATE.graySize, line: { width: 0 }, opacity: 0.55 },
        hoverinfo: "skip",
        name: "per-atom",
        showlegend: false,
    } : null;

    // Foreground colored trace (per-structure averages) - clickable.
    const avgTrace = {
        x: opts.avgPoints.map(p => p[opts.xKey]),
        y: opts.avgPoints.map(p => p.y),
        mode: "markers",
        type: "scatter",
        marker: {
            color: opts.avgPoints.map(p => p.color),
            size: opts.avgPoints.map(_ => STATE.defaultSize),
            line: { color: "#222", width: 0.6 },
        },
        customdata: opts.avgPoints.map(p => p),
        hovertemplate: opts.hovertemplate,
        name: "per-structure mean",
        showlegend: false,
    };

    const traces = grayTrace ? [grayTrace, avgTrace] : [avgTrace];
    const avgTraceIdx = grayTrace ? 1 : 0;

    const layout = {
        margin: { l: 56, r: 16, t: 36, b: 46 },
        xaxis: {
            title: { text: opts.xTitle, font: { size: 12, family: "Times, serif" } },
            tickfont: { family: "Times, serif" },
            type: opts.xType || "linear", zeroline: false,
        },
        yaxis: {
            title: { text: opts.yTitle, font: { size: 12, family: "Times, serif" } },
            tickfont: { family: "Times, serif" },
            type: opts.yType || "linear", zeroline: false,
        },
        title: { text: opts.title, font: { size: 14, family: "Times, serif" } },
        font: { family: "Times, serif" },
        plot_bgcolor: "white",
        paper_bgcolor: "white",
        showlegend: false,
        hovermode: "closest",
    };

    const promise = Plotly.newPlot(divId, traces, layout, { displayModeBar: false, responsive: true });
    const div = document.getElementById(divId);
    div._meta = {
        avgTraces: [{
            traceIdx: avgTraceIdx,
            keys: opts.avgPoints.map(p => p.key),
            baseColors: opts.avgPoints.map(p => p.color),
        }],
    };
    const clickableTraceIdxs = new Set([avgTraceIdx]);
    div.on("plotly_click", function(e) {
        if (!e || !e.points || !e.points.length) return;
        const pt = e.points[0];
        if (!clickableTraceIdxs.has(pt.curveNumber)) return;
        const cd = pt.customdata;
        if (cd && cd.key) activate(opts.section, cd.key);
    });
    return promise;
}

function highlightPlot(divId, activeKey) {
    const div = document.getElementById(divId);
    if (!div || !div._meta) return;
    div._meta.avgTraces.forEach(t => {
        const colors = t.keys.map((k, i) => k === activeKey ? STATE.accent : t.baseColors[i]);
        const sizes = t.keys.map(k => k === activeKey ? STATE.highlightSize : STATE.defaultSize);
        Plotly.restyle(divId, { "marker.color": [colors], "marker.size": [sizes] }, [t.traceIdx]);
    });
}

// Specialized broken-axis scatter for the volume-vs-coord-distance plot.
// Smallest-volume structures sit on the left x-axis, largest on the right,
// with a gray strip filling the visual break between them.
function buildBrokenAxisCoordPlot(divId) {
    const isLargest = p => p.group && p.group.toLowerCase().indexOf("largest") >= 0;
    const leftAvg = STATE.coordAvgPoints.filter(p => !isLargest(p));
    const rightAvg = STATE.coordAvgPoints.filter(p => isLargest(p));
    const leftKeys = new Set(leftAvg.map(p => p.key));
    const rightKeys = new Set(rightAvg.map(p => p.key));
    const leftAtom = STATE.coordAtomPoints.filter(p => leftKeys.has(p.key));
    const rightAtom = STATE.coordAtomPoints.filter(p => rightKeys.has(p.key));

    function pad(min, max) {
        const d = max - min;
        return d > 0 ? d * 0.18 : Math.max(0.1, min * 0.1);
    }
    const lMin = Math.min.apply(null, leftAvg.map(p => p.volume));
    const lMax = Math.max.apply(null, leftAvg.map(p => p.volume));
    const rMin = Math.min.apply(null, rightAvg.map(p => p.volume));
    const rMax = Math.max.apply(null, rightAvg.map(p => p.volume));
    const lPad = pad(lMin, lMax), rPad = pad(rMin, rMax);

    const grayMarker = { color: STATE.grayColor, size: STATE.graySize, line: { width: 0 }, opacity: 0.55 };
    const traces = [
        { x: leftAtom.map(p=>p.volume),  y: leftAtom.map(p=>p.y),
          mode: "markers", type: "scatter", xaxis: "x", yaxis: "y",
          marker: grayMarker, hoverinfo: "skip", showlegend: false },
        { x: rightAtom.map(p=>p.volume), y: rightAtom.map(p=>p.y),
          mode: "markers", type: "scatter", xaxis: "x2", yaxis: "y",
          marker: grayMarker, hoverinfo: "skip", showlegend: false },
        { x: leftAvg.map(p=>p.volume), y: leftAvg.map(p=>p.y),
          customdata: leftAvg,
          mode: "markers", type: "scatter", xaxis: "x", yaxis: "y",
          marker: { color: leftAvg.map(p=>p.color),
                    size: leftAvg.map(_=>STATE.defaultSize),
                    line: { color: "#222", width: 0.6 } },
          hovertemplate:
              "<b>%{customdata.key}</b><br>" +
              "volume = %{x:.3f} \\u00c5\\u00b3<br>" +
              "mean d(Zn-X) = %{y:.3f} \\u00c5<br>" +
              "%{customdata.group} #%{customdata.rank}" +
              "<extra></extra>",
          showlegend: false },
        { x: rightAvg.map(p=>p.volume), y: rightAvg.map(p=>p.y),
          customdata: rightAvg,
          mode: "markers", type: "scatter", xaxis: "x2", yaxis: "y",
          marker: { color: rightAvg.map(p=>p.color),
                    size: rightAvg.map(_=>STATE.defaultSize),
                    line: { color: "#222", width: 0.6 } },
          hovertemplate:
              "<b>%{customdata.key}</b><br>" +
              "volume = %{x:.3f} \\u00c5\\u00b3<br>" +
              "mean d(Zn-X) = %{y:.3f} \\u00c5<br>" +
              "%{customdata.group} #%{customdata.rank}" +
              "<extra></extra>",
          showlegend: false },
    ];

    const leftDomain = [0.0, 0.46];
    const rightDomain = [0.54, 1.0];
    const layout = {
        margin: { l: 56, r: 16, t: 36, b: 50 },
        xaxis: {
            domain: leftDomain,
            range: [Math.max(0, lMin - lPad), lMax + lPad],
            zeroline: false,
            title: { text: "C<sub>\\u03b1</sub> volume (\\u00c5\\u00b3) \\u2014 smallest", standoff: 6, font: { size: 12, family: "Times, serif" } },
            tickfont: { family: "Times, serif" },
        },
        xaxis2: {
            domain: rightDomain,
            range: [rMin - rPad, rMax + rPad],
            anchor: "y", zeroline: false,
            title: { text: "C<sub>\\u03b1</sub> volume (\\u00c5\\u00b3) \\u2014 largest", standoff: 6, font: { size: 12, family: "Times, serif" } },
            tickfont: { family: "Times, serif" },
        },
        yaxis: {
            title: { text: "Zn\\u2013X distance (\\u00c5)", font: { size: 12, family: "Times, serif" } },
            tickfont: { family: "Times, serif" },
            zeroline: false,
        },
        title: { text: "Volume vs Zn\\u2013X distance \\u2014 click a colored point", font: { size: 14, family: "Times, serif" } },
        font: { family: "Times, serif" },
        plot_bgcolor: "white",
        paper_bgcolor: "white",
        showlegend: false,
        hovermode: "closest",
        shapes: [
            { type: "rect", xref: "paper", yref: "paper",
              x0: leftDomain[1], x1: rightDomain[0], y0: 0, y1: 1,
              fillcolor: "#dddddd", line: { width: 0 }, layer: "below" },
        ],
        annotations: [
            { xref: "paper", yref: "paper", x: 0.5, y: 0.5,
              showarrow: false, text: "\\u2afd",
              font: { size: 22, color: "#888", family: "Helvetica, Arial, sans-serif" } },
        ],
    };

    const promise = Plotly.newPlot(divId, traces, layout, { displayModeBar: false, responsive: true });
    const div = document.getElementById(divId);
    div._meta = {
        avgTraces: [
            { traceIdx: 2, keys: leftAvg.map(p=>p.key),  baseColors: leftAvg.map(p=>p.color) },
            { traceIdx: 3, keys: rightAvg.map(p=>p.key), baseColors: rightAvg.map(p=>p.color) },
        ],
    };
    const clickable = new Set([2, 3]);
    div.on("plotly_click", function(e) {
        if (!e || !e.points || !e.points.length) return;
        const pt = e.points[0];
        if (!clickable.has(pt.curveNumber)) return;
        const cd = pt.customdata;
        if (cd && cd.key) activate("cys", cd.key);
    });
    return promise;
}

// ---------- Section wiring ----------

const SECTIONS = {
    cys: {
        plotIds: ["plot-extremes-coord"],
        cardSelector: "#cys-mini-cards .mini-card",
        rowsMap: () => STATE.cysRowsMap,
        heroId: "hero-cys",
    },
    qs: {
        plotIds: ["plot-extremes-qs"],
        cardSelector: "#qs-mini-cards .mini-card",
        rowsMap: () => STATE.qsRowsMap,
        heroId: "hero-qs",
    },
};

function activate(sectionName, key) {
    const section = SECTIONS[sectionName];
    if (!section) return;
    section.plotIds.forEach(pid => highlightPlot(pid, key));
    document.querySelectorAll(section.cardSelector).forEach(card => {
        card.classList.toggle("active", card.dataset.key === key);
    });
    updateHero(sectionName, key);
}

// ---------- Hero card ----------

function fmt(num, digits) { return Number(num).toFixed(digits); }

function buildHeroMetaHtml(sectionName, row) {
    let tagClass = "";
    let tagLabel = "";
    if (sectionName === "cys") {
        if (row.group.toLowerCase().indexOf("largest") >= 0) {
            tagClass = "largest"; tagLabel = "largest C\\u03b1 volume #" + row.rank;
        } else {
            tagClass = ""; tagLabel = "smallest C\\u03b1 volume #" + row.rank;
        }
    } else {
        tagClass = "qs"; tagLabel = "low q_S #" + row.rank;
    }
    const dihedrals = row.dihedrals.map(d => fmt(d, 1) + "\\u00b0").join(", ");
    const coordStr = (row.coord_distances && row.coord_distances.length)
        ? row.coord_distances.map(d => fmt(d, 3) + " \\u00c5").join(", ")
        : "(unavailable)";
    const pdbId = (row.key.slice(0, 4) || "").toUpperCase();
    const pdbLink = pdbId
        ? "<div class=\\"meta-pdb-link\\"><a href=\\"https://www.rcsb.org/structure/" + pdbId +
          "\\" target=\\"_blank\\" rel=\\"noopener noreferrer\\">RCSB: " + pdbId + " \\u2197</a></div>"
        : "";
    const hasExtended = !!STATE.xyzExtendedByKey[row.key];
    const currentMode = HERO_MODE[sectionName] || "compact";
    const extDisabled = hasExtended ? "" : " disabled";
    const extTitle = hasExtended ? "" : ' title="Extended xyz not generated for this structure"';
    const toggleHtml =
        "<div class=\\"hero-toggle\\" data-section=\\"" + sectionName + "\\">" +
            "<button data-mode=\\"compact\\" class=\\"" + (currentMode === "compact" ? "active" : "") + "\\">" +
                "Compact" +
            "</button>" +
            "<button data-mode=\\"extended\\"" + extDisabled + extTitle +
                " class=\\"" + (currentMode === "extended" && hasExtended ? "active" : "") + "\\">" +
                "Extended (10 \\u00c5)" +
            "</button>" +
        "</div>";
    return (
        "<h3 class=\\"mono\\">" + row.key + "</h3>" +
        pdbLink +
        toggleHtml +
        "<div class=\\"meta-row\\"><span class=\\"hero-tag " + tagClass + "\\">" + tagLabel + "</span>" +
            "<span class=\\"muted mono\\">" + row.family + "</span></div>" +
        "<div class=\\"meta-row\\"><b>C<sub>\\u03b1</sub> volume:</b> " + fmt(row.volume, 4) + " \\u00c5\\u00b3</div>" +
        "<div class=\\"meta-row\\"><b>Zn\\u2013X distances:</b> " + coordStr +
            " <span class=\\"muted\\">(mean " + fmt(row.coord_distance_mean, 3) + " \\u00c5)</span></div>" +
        "<div class=\\"meta-row\\"><b>Cys Zn\\u2013S\\u2013C<sub>\\u03b2</sub>\\u2013C<sub>\\u03b1</sub> dihedrals:</b> " + dihedrals +
            " <span class=\\"muted\\">(mean " + fmt(row.dihedral_mean, 2) + "\\u00b0)</span></div>" +
        "<div class=\\"meta-row\\"><b>q<sub>S</sub>:</b> " + fmt(row.q_tetra_coord, 4) +
            " &nbsp; <b>q<sub>\\u03b1</sub>:</b> " + fmt(row.q_tetra_ca, 4) + "</div>"
    );
}

// Hero state per section: { mode: "compact" | "extended", key: string | null }
const HERO_MODE = { cys: "compact", qs: "compact" };
const HERO_KEY = { cys: null, qs: null };

function renderHeroViewer(sectionName) {
    const section = SECTIONS[sectionName];
    const key = HERO_KEY[sectionName];
    if (!section || !key) return;
    const viewerEl = document.getElementById(section.heroId + "-viewer");
    const mode = HERO_MODE[sectionName];
    let xyzText = mode === "extended" ? STATE.xyzExtendedByKey[key] : null;
    let usingExtended = false;
    if (xyzText) {
        usingExtended = true;
    } else {
        xyzText = STATE.xyzByKey[key];
    }
    viewerEl.innerHTML = "";
    if (!xyzText) return;
    const v = $3Dmol.createViewer(viewerEl, { backgroundColor: "white" });
    v.addModel(xyzText, "xyz");
    if (usingExtended) {
        // Larger atom set: thinner sticks, smaller spheres so the broader
        // environment is legible. Highlight the Zn at the origin.
        v.setStyle({}, {
            stick: { radius: 0.10, colorscheme: "Jmol" },
            sphere: { scale: 0.18, colorscheme: "Jmol" },
        });
    } else {
        v.setStyle({}, {
            stick: { radius: 0.16, colorscheme: "Jmol" },
            sphere: { scale: 0.25, colorscheme: "Jmol" },
        });
    }
    v.addSphere({ center: { x: 0, y: 0, z: 0 }, radius: 0.22, color: "black", opacity: 1.0 });
    v.zoomTo();
    v.render();
}

function updateHero(sectionName, key) {
    const section = SECTIONS[sectionName];
    if (!section) return;
    const row = section.rowsMap()[key];
    if (!row) return;
    HERO_KEY[sectionName] = key;
    // If we previously selected "extended" but this structure doesn't have one,
    // silently fall back to compact.
    if (HERO_MODE[sectionName] === "extended" && !STATE.xyzExtendedByKey[key]) {
        HERO_MODE[sectionName] = "compact";
    }
    const heroEl = document.getElementById(section.heroId);
    const metaEl = document.getElementById(section.heroId + "-meta");
    metaEl.innerHTML = buildHeroMetaHtml(sectionName, row);
    wireHeroToggle(metaEl);
    renderHeroViewer(sectionName);
    heroEl.dataset.populated = "true";
}

function setHeroMode(sectionName, mode) {
    const key = HERO_KEY[sectionName];
    if (!key) return;
    if (mode === "extended" && !STATE.xyzExtendedByKey[key]) return;
    HERO_MODE[sectionName] = mode;
    // Reflect in the toggle UI without rebuilding the whole meta block.
    const section = SECTIONS[sectionName];
    const metaEl = document.getElementById(section.heroId + "-meta");
    metaEl.querySelectorAll(".hero-toggle button").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.mode === mode);
    });
    renderHeroViewer(sectionName);
}

function wireHeroToggle(metaEl) {
    const wrap = metaEl.querySelector(".hero-toggle");
    if (!wrap) return;
    const sectionName = wrap.dataset.section;
    wrap.querySelectorAll("button").forEach(btn => {
        btn.addEventListener("click", () => {
            if (btn.disabled) return;
            setHeroMode(sectionName, btn.dataset.mode);
        });
    });
}

// ---------- Initialization ----------

function initPlots() {
    const p1 = buildBrokenAxisCoordPlot("plot-extremes-coord");
    const p2 = buildScatterPlot("plot-extremes-qs", {
        section: "qs",
        atomPoints: null,
        avgPoints: STATE.qsPoints,
        xKey: "volume",
        xTitle: "C<sub>\\u03b1</sub> volume (\\u00c5\\u00b3)",
        yTitle: "q<sub>S</sub>",
        title: "Volume vs q<sub>S</sub> \\u2014 click a point",
        hovertemplate:
            "<b>%{customdata.key}</b><br>" +
            "volume = %{x:.3f} \\u00c5\\u00b3<br>" +
            "q<sub>S</sub> = %{y:.4f}<br>" +
            "q<sub>\\u03b1</sub> = %{customdata.q_tetra_ca:.4f}" +
            "<extra></extra>",
    });
    return Promise.all([p1, p2]);
}

function tagMiniCardColors() {
    document.querySelectorAll("#cys-mini-cards .mini-card").forEach(card => {
        const key = card.dataset.key;
        const row = STATE.cysRowsMap[key];
        if (row && row.group.toLowerCase().indexOf("largest") >= 0) {
            card.dataset.groupColor = "largest";
        }
    });
    document.querySelectorAll("#qs-mini-cards .mini-card").forEach(card => {
        card.dataset.groupColor = "qs";
    });
}

function wireCardClicks() {
    document.querySelectorAll(".mini-card").forEach(card => {
        card.addEventListener("click", () => {
            const sectionName = card.dataset.section;
            activate(sectionName, card.dataset.key);
        });
    });
}

function renderMath() {
    if (typeof renderMathInElement === "function") {
        renderMathInElement(document.body, {
            delimiters: [
                { left: "$$", right: "$$", display: true },
                { left: "$", right: "$", display: false },
            ],
            throwOnError: false,
        });
    } else {
        // KaTeX still loading; retry shortly.
        setTimeout(renderMath, 60);
    }
}

window.addEventListener("DOMContentLoaded", () => {
    tagMiniCardColors();
    wireCardClicks();
    renderMath();
    initPlots().then(() => {
        if (STATE.initialCysKey) activate("cys", STATE.initialCysKey);
        if (STATE.initialQsKey) activate("qs", STATE.initialQsKey);
    });
});
"""

    stats_html = ""
    if pdb_count is not None or xyz_count is not None:
        parts = []
        if pdb_count is not None:
            parts.append(f'<span class="stat"><span class="stat-num">{pdb_count:,}</span> PDB entries</span>')
        if xyz_count is not None:
            parts.append(f'<span class="stat"><span class="stat-num">{xyz_count:,}</span> XYZ clusters</span>')
        sep = '<span class="stat-sep">/</span>'
        stats_html = f'<div class="header-stats">{sep.join(parts)}</div>'

    if inline_assets:
        external_assets_html = (
            f"<style>{inline_assets['katex_css']}</style>\n"
            f"<script>{inline_assets['plotly_js']}</script>\n"
            f"<script>{inline_assets['threedmol_js']}</script>\n"
            f"<script>{inline_assets['katex_js']}</script>\n"
            f"<script>{inline_assets['katex_autorender_js']}</script>"
        )
    else:
        external_assets_html = (
            f'<link rel="stylesheet" href="{KATEX_CSS}">\n'
            f'  <script src="{PLOTLY_CDN}"></script>\n'
            f'  <script src="{THREEDMOL_CDN}"></script>\n'
            f'  <script defer src="{KATEX_JS}"></script>\n'
            f'  <script defer src="{KATEX_AUTORENDER_JS}"></script>'
        )

    head = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html_lib.escape(system_name)}</title>
  {external_assets_html}
  <style>{css}</style>
</head>
<body>
  <header>
    <div class="header-inner">
      <h1>{html_lib.escape(system_name)}</h1>
      {stats_html}
      <div class="header-summary">
        <div class="header-card cys">
          <h3>1 &middot; C<sub>&alpha;</sub> volume vs first-shell distance</h3>
          <p>5 smallest + 5 largest tetrahedron volumes. Mean Zn&ndash;X distance per structure is clickable.</p>
          <div class="swatch-row">
            <span class="swatch" style="background:{SMALLEST_COLOR}" title="smallest"></span>
            <span class="swatch" style="background:{LARGEST_COLOR}" title="largest"></span>
          </div>
        </div>
        <div class="header-card qs">
          <h3>2 &middot; Low $q_S$ outliers</h3>
          <p>Structures with the most distorted sulfur sub-tetrahedron about Zn ($q_S$ &lt; 0.74).</p>
          <div class="swatch-row">
            <span class="swatch" style="background:{QS_COLOR}" title="low q_S"></span>
          </div>
        </div>
      </div>
      <div class="header-tip">Click a colored point or card to load it into the hero viewer. Click-drag to rotate the 3D viewer; scroll to zoom.</div>
    </div>
  </header>
"""

    cys_section = f"""
  <section id="volume-cys" class="cys">
    <h2><span class="section-num">1</span>$C_\\alpha$ volume vs first-shell coordination distance</h2>
    <div class="info-panel">
      <p>The $C_\\alpha$ tetrahedron volume measures the size of the four-residue scaffold framing the Zn site.
         It is the volume of the convex hull of the four coordinating residues' $\\alpha$&ndash;carbon atoms.</p>
      <p>Below: gray markers are per-atom Zn&ndash;X distances; colored markers are the per-structure mean
         (clickable). The x-axis has a break to separate the smallest and largest volume clusters.</p>
    </div>
    <div class="split-layout">
      <div class="plot-col">
        <div class="panel">
          <div class="panel-title">Volume vs coord-atom distance &mdash; full dataset</div>
          <img src="data:image/png;base64,{png_b64['volume_vs_coord_distance']}" alt="volume vs coord distance">
        </div>
        <div class="panel" style="margin-top: 10px;">
          <div id="plot-extremes-coord" class="plot-half"></div>
        </div>
      </div>
      <div class="cards-col">
        <div class="hero" id="hero-cys" data-populated="false" style="margin: 0 0 14px 0;">
          <div class="hero-empty">Click a colored point or a card to view a structure.</div>
          <div class="hero-content">
            <div class="hero-viewer" id="hero-cys-viewer"></div>
            <div class="hero-meta" id="hero-cys-meta"></div>
          </div>
        </div>
        <div id="cys-mini-cards" class="mini-cards" style="grid-template-columns: 1fr;">
          <div class="group-divider"><span class="swatch smallest"></span>5 smallest $C_\\alpha$ volume</div>
          {cys_smallest_cards}
          <div class="group-divider"><span class="swatch largest"></span>5 largest $C_\\alpha$ volume</div>
          {cys_largest_cards}
        </div>
      </div>
    </div>
  </section>
"""

    qs_section = f"""
  <section id="volume-qs" class="qs">
    <h2><span class="section-num">2</span>Low $q_S$ structures</h2>
    <div class="info-panel">
      <p>$q_S$ is the Errington&ndash;Debenedetti tetrahedral order parameter, computed about the Zn origin
         using the four coordinating sulfur atoms (subscript $S$ = <em>sulfur sub-tetrahedron</em>):</p>
      <span class="formula">$$q_{{tetra}} \\;=\\; 1 - \\frac{{3}}{{8}} \\sum_{{j=1}}^{{3}} \\sum_{{k=j+1}}^{{4}} \\left( \\cos\\psi_{{jk}} + \\frac{{1}}{{3}} \\right)^{{2}}$$</span>
      <p>$q_S = 1$ for a perfect tetrahedron, $q_S = 0$ for a random arrangement.
         The structures listed below have $q_S \\lesssim 0.74$, i.e. the most distorted sulfur sub-tetrahedra in the dataset.</p>
    </div>
    <div class="split-layout">
      <div class="plot-col">
        <div class="panel">
          <div class="panel-title">Volume vs $q_S$ &mdash; full dataset</div>
          <img src="data:image/png;base64,{png_b64['volume_vs_s_q_tetra']}" alt="volume vs q_S">
        </div>
        <div class="panel" style="margin-top: 10px;">
          <div id="plot-extremes-qs" class="plot-half"></div>
        </div>
      </div>
      <div class="cards-col">
        <div class="hero" id="hero-qs" data-populated="false" style="margin: 0 0 14px 0;">
          <div class="hero-empty">Click a point or card to view a structure.</div>
          <div class="hero-content">
            <div class="hero-viewer" id="hero-qs-viewer"></div>
            <div class="hero-meta" id="hero-qs-meta"></div>
          </div>
        </div>
        <div id="qs-mini-cards" class="mini-cards" style="grid-template-columns: 1fr;">
          <div class="group-divider"><span class="swatch qs"></span>Lowest $q_S$ (most distorted)</div>
          {qs_cards}
        </div>
      </div>
    </div>
  </section>
"""

    return head + cys_section + qs_section + f"""
  <script>{js}</script>
</body>
</html>
"""


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--figures-dir", required=True, type=Path,
                   help="Directory containing the volume_extremes csv and the full-dataset PNGs.")
    p.add_argument("--label", required=True,
                   help="Dataset label (e.g. '4cys-large-dataset'). Used in filenames.")
    p.add_argument("--system-name", default=None,
                   help="Display title for the dataset (e.g. '4Cys Zn-binding structures').")
    p.add_argument("--pdb-count", type=int, default=None,
                   help="Optional: number of unique PDB entries to show in the header stats.")
    p.add_argument("--xyz-count", type=int, default=None,
                   help="Optional: number of XYZ cluster files to show in the header stats.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output HTML path (default: <figures-dir>/report_<label>.html).")
    p.add_argument("--offline", action="store_true",
                   help="Inline Plotly.js, 3Dmol.js, KaTeX (CSS+JS+fonts) into the HTML "
                        "so it works on any machine without internet. Adds ~5-7 MB.")
    p.add_argument("--vendor-cache", type=Path,
                   default=Path.home() / ".cache" / "build_volume_report",
                   help="Cache directory for downloaded JS/CSS/font assets used in --offline mode.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    fig_dir: Path = args.figures_dir
    label: str = args.label
    csv_path = fig_dir / f"volume_extremes_{label}.csv"
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    if args.output is not None:
        out_path: Path = args.output
    else:
        # Default to '_offline' suffix when --offline is used so the two builds
        # don't clobber each other.
        suffix = "_offline" if args.offline else ""
        out_path = fig_dir / f"report_{label}{suffix}.html"

    rows = read_extremes_csv(csv_path)
    cys_smallest = sorted([r for r in rows if r.group == "smallest CA volume"], key=lambda r: r.rank)
    cys_largest = sorted([r for r in rows if r.group == "largest CA volume"], key=lambda r: r.rank)
    qs_rows = sorted([r for r in rows if r.group == "smallest q_S"], key=lambda r: r.rank)

    print(f"Loaded {len(rows)} rows: "
          f"{len(cys_smallest)} smallest CA volume, "
          f"{len(cys_largest)} largest CA volume, "
          f"{len(qs_rows)} smallest q_S")

    xyz_by_key: dict[str, str] = {}
    xyz_extended_by_key: dict[str, str] = {}
    for row in cys_smallest + cys_largest + qs_rows:
        if not row.xyz_path.exists():
            raise SystemExit(f"XYZ file not found: {row.xyz_path}")
        text = read_xyz(row.xyz_path)
        xyz_by_key[row.key] = text
        row.coord_distances = coord_distances_from_xyz(text)

        ext_path = row.xyz_path.with_name(f"{row.xyz_path.stem}-extended{row.xyz_path.suffix}")
        if ext_path.exists():
            xyz_extended_by_key[row.key] = read_xyz(ext_path)

    n_extended = len(xyz_extended_by_key)
    print(f"Extended-environment xyz files found: {n_extended}/{len(xyz_by_key)}")
    if n_extended < len(xyz_by_key):
        print("(Run scripts/build_extended_xyz.py to generate the missing ones.)")

    pngs = {
        "volume_vs_coord_distance": fig_dir / f"volume_vs_coord_distance_qca_{label}.png",
        "volume_vs_s_q_tetra":      fig_dir / f"volume_vs_s_q_tetra_{label}.png",
    }
    for path in pngs.values():
        if not path.exists():
            raise SystemExit(f"PNG not found: {path}")
    png_b64 = {name: png_to_base64(path) for name, path in pngs.items()}

    system_name = args.system_name or label

    inline_assets: dict[str, str] | None = None
    if args.offline:
        print(f"Offline mode: fetching/inlining JS+CSS+fonts (cache: {args.vendor_cache})...")
        inline_assets = {
            "katex_css": _inline_katex_css(KATEX_CSS, args.vendor_cache),
            "plotly_js": _fetch(PLOTLY_CDN, args.vendor_cache).decode("utf-8"),
            "threedmol_js": _fetch(THREEDMOL_CDN, args.vendor_cache).decode("utf-8"),
            "katex_js": _fetch(KATEX_JS, args.vendor_cache).decode("utf-8"),
            "katex_autorender_js": _fetch(KATEX_AUTORENDER_JS, args.vendor_cache).decode("utf-8"),
        }

    html_text = build_html(
        cys_smallest=cys_smallest,
        cys_largest=cys_largest,
        qs_rows=qs_rows,
        xyz_by_key=xyz_by_key,
        xyz_extended_by_key=xyz_extended_by_key,
        png_b64=png_b64,
        label=label,
        system_name=system_name,
        pdb_count=args.pdb_count,
        xyz_count=args.xyz_count,
        inline_assets=inline_assets,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"✓ Wrote {out_path}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
