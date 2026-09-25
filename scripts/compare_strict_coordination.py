#!/usr/bin/env python3
"""Legacy-vs-strict coordination-classification regression harness.

Runs the pdb-scraper pipeline twice per dataset over LOCAL PDB files (no
downloading, ``input_mode: folder``): once as-is (legacy) and once with
``validation.strict_coordination.enabled: true``. For every Zn site it records
whether the legacy run and the strict run kept it, the per-site ``[strict]``
rejection reason, and (where available) an independent geometry-QC verdict and
the original data-clone result, then writes a per-dataset comparison CSV.

Design / provenance
-------------------
* A "site" is one metal connected-component center, identified in the pipeline's
  ``clusters_summary.csv`` by ``(PDB, CLUSTER)`` where ``CLUSTER`` is the
  per-structure ``cluster_counter`` used in the ``<pdb>_cluster<N>_Zn.xyz`` file
  names. The Zn chain/resseq come from the CSV ``CHAIN``/``RESSEQ`` columns.
* "Kept" == a row is present in that run's ``clusters_summary.csv`` (rejected
  sites get no row).
* ``strict_reasons`` are parsed from the strict run's ``pipeline.log``: strict
  rejections log a line ``[strict] <pdb> <chain> <resseq> rejected: <reasons>``.
* The independent QC report (``qc_report-*.csv``) gives an OK/REJECT verdict per
  ``<pdb>_cluster<N>_Zn.xyz`` file; the original data-clone ``clusters_summary``
  is the legacy baseline the legacy run must reproduce.

Safety
------
Input PDB dirs and the QC / original CSVs are treated READ-ONLY. All generated
files (configs, pipeline outputs, comparison CSVs, symlinked spot-check inputs)
go only under ``--output-root``. The spec YAML holds no absolute paths; dataset
locations resolve against ``--data-root`` (machine defaults for both live in this
script, not the committed spec).

Usage
-----
    uv run python scripts/compare_strict_coordination.py            # all datasets, both modes
    uv run python scripts/compare_strict_coordination.py --datasets 4his-large
    uv run python scripts/compare_strict_coordination.py --mode legacy   # legacy only
    uv run python scripts/compare_strict_coordination.py \
        --data-root /path/to/zn-cys-his --output-root /path/to/out
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPEC = REPO_ROOT / "scripts" / "regression_configs" / "datasets.yaml"

# Machine-specific locations live here as CLI defaults only (overridable with
# --data-root / --output-root), so the committed spec YAML holds no absolute
# paths. --data-root is the parent of the read-only data clones; dataset paths in
# the spec are relative to it.
DATA_ROOT_DEFAULT = Path("/Users/stetef/Documents/SLAC/zn-cys-his")
OUTPUT_ROOT_DEFAULT = Path(tempfile.gettempdir()) / "strict-coord-regression"


def resolve_dataset_paths(ds: dict, data_root: Path) -> None:
    """Expand the spec's rel_* keys into absolute paths under ``data_root``.

    Injects ``pdb_files_dir`` / ``qc_report`` / ``original_clusters_csv`` so the
    rest of the harness can stay path-source agnostic.
    """
    ds["pdb_files_dir"] = str(data_root / ds["rel_pdb_files_dir"])
    if ds.get("rel_qc_report"):
        ds["qc_report"] = str(data_root / ds["rel_qc_report"])
    if ds.get("rel_original_clusters_csv"):
        ds["original_clusters_csv"] = str(data_root / ds["rel_original_clusters_csv"])

# Strict rejections log as: ``[strict] <pdb> ZN<chain><resseq> rejected: <reasons>``
# e.g. ``[strict] 1axe ZNA402 rejected: donor_distance: CYSA111 SG 2.00 A ...``.
# The Zn token is "ZN" + a 1-char chain id + the numeric resseq, no separators.
STRICT_LINE_RE = re.compile(
    r"\[strict\]\s+(?P<pdb>\S+)\s+ZN(?P<chain>\S)(?P<resseq>-?\d+[A-Za-z]?)\s+rejected:\s*(?P<reason>.*)",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Spec + config construction
# --------------------------------------------------------------------------- #
def load_spec(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_input_dir(ds: dict, out_dir: Path) -> str:
    """Return the folder the pipeline should scan.

    For a spot-check dataset with ``limit`` set, symlink the first ``limit``
    sorted ``*.pdb`` files into ``out_dir/input_links`` (read-only originals are
    never touched) and return that dir. Otherwise return the real dir.
    """
    src = Path(ds["pdb_files_dir"])
    limit = ds.get("limit")
    if not limit:
        return str(src)
    link_dir = out_dir / "input_links"
    link_dir.mkdir(parents=True, exist_ok=True)
    pdbs = sorted(src.glob("*.pdb"))[: int(limit)]
    for p in pdbs:
        dst = link_dir / p.name
        if not dst.exists():
            dst.symlink_to(p.resolve())
    return str(link_dir)


def build_config(ds: dict, mode: str, spec: dict, run_dir: Path, input_dir: str) -> Path:
    """Write a PipelineConfig YAML for one dataset+mode and return its path."""
    proc = ds["processing"]
    validation: dict = {
        "coordination_distance_min": spec.get("coordination_distance_min", 2.0),
        "coordination_distance_max": spec.get("coordination_distance_max", 2.8),
    }
    if ds.get("ligand_requirements"):
        validation["ligand_requirements"] = ds["ligand_requirements"]
    if mode == "strict":
        strict = dict(spec["strict_coordination"])
        strict["enabled"] = True
        validation["strict_coordination"] = strict
    else:
        # Explicitly disabled so a stray global default can't leak into legacy.
        validation["strict_coordination"] = {"enabled": False}

    cfg = {
        "search_parameters": {
            "metal_ion": "ZN",
            "resolution_cutoff": None,       # inputs are already curated PDBs
            "experimental_method": "ALL",
            "polymer_type": None,
        },
        "processing": {
            "batch_size": 1000,
            "parallel_workers": 1,
            "rate_limit_delay": 0.0,
            "temp_directory": str(run_dir / "downloads"),
            "cutoff": proc["cutoff"],
            "selection_radius": proc.get("selection_radius"),
            "target": proc.get("target", "ZN"),
            "metals_excluded": [],
            "max_downloads": None,
            "must_have": proc.get("must_have", ""),
            "include_waters": proc.get("include_waters", False),
        },
        "output": {
            "results_database": str(run_dir / "output" / "clusters_summary.csv"),
            "checkpoint_file": str(run_dir / "results" / "checkpoint.db"),
            "log_file": str(run_dir / "pipeline.log"),
            "save_matching_structures": False,
            "matched_structures_dir": str(run_dir / "results" / "matched"),
            "kept_structures_dir": str(run_dir / "results" / "kept"),
            "output_dir": str(run_dir / "output"),
        },
        "validation": validation,
        "input_mode": "folder",
        "input_data": [input_dir],
        "log_level": "INFO",
    }
    # Wipe any prior output: the pipeline APPENDS to clusters_summary.csv and
    # persists a checkpoint, so a reused dir would mix runs. run_dir is always a
    # fresh per-mode subdir under output_root (never an input dir).
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path


def run_pipeline_subprocess(cfg_path: Path) -> int:
    """Run one pipeline config in a clean subprocess (isolates logging/state)."""
    proc = subprocess.run(
        ["uv", "run", "python", "-m", "scrape_pdb", str(cfg_path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        # rc==1 also happens when a subset of PDBs error; not necessarily fatal.
        sys.stderr.write(
            f"[warn] pipeline rc={proc.returncode} for {cfg_path}\n"
            f"       stderr tail: {proc.stderr.strip()[-500:]}\n"
        )
    return proc.returncode


# --------------------------------------------------------------------------- #
# Output parsing
# --------------------------------------------------------------------------- #
@dataclass
class Site:
    pdb: str
    cluster_n: str
    chain: str = ""
    resseq: str = ""
    cn: str = ""
    coord: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.pdb.lower(), str(self.cluster_n))

    @property
    def xyz_name(self) -> str:
        return f"{self.pdb.lower()}_cluster{self.cluster_n}_Zn.xyz"


def parse_clusters_csv(path: Path) -> dict[tuple[str, str], Site]:
    """Map (pdb, cluster_n) -> Site for every kept row in a clusters_summary.csv."""
    sites: dict[tuple[str, str], Site] = {}
    if not path.exists():
        return sites
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb = (row.get("PDB") or "").strip()
            cluster_n = (row.get("CLUSTER") or "").strip()
            if not pdb or not cluster_n:
                continue
            s = Site(
                pdb=pdb,
                cluster_n=cluster_n,
                chain=(row.get("CHAIN") or "").strip(),
                resseq=(row.get("RESSEQ") or "").strip(),
                cn=(row.get("CN") or "").strip(),
                coord=(row.get("COORD") or "").strip(),
            )
            # A polynuclear cluster can yield several center rows sharing CLUSTER;
            # keep the first (mononuclear in these datasets) but do not overwrite.
            sites.setdefault(s.key, s)
    return sites


def parse_strict_log(path: Path) -> dict[tuple[str, str, str], str]:
    """Map (pdb, chain, resseq) -> strict rejection reason from pipeline.log."""
    reasons: dict[tuple[str, str, str], str] = {}
    if not path.exists():
        return reasons
    with open(path, errors="replace") as f:
        for line in f:
            m = STRICT_LINE_RE.search(line)
            if not m:
                continue
            key = (
                m.group("pdb").strip().lower(),
                m.group("chain").strip(),
                m.group("resseq").strip(),
            )
            reasons[key] = m.group("reason").strip()
    return reasons


def parse_qc(path: Optional[str]) -> dict[str, tuple[str, str]]:
    """Map xyz filename -> (status, reasons) from an independent QC report."""
    out: dict[str, tuple[str, str]] = {}
    if not path or not Path(path).exists():
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            fname = (row.get("file") or "").strip()
            if fname:
                out[fname.lower()] = (
                    (row.get("status") or "").strip().upper(),
                    (row.get("reasons") or "").strip(),
                )
    return out


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
@dataclass
class DatasetResult:
    name: str
    rows: list[dict] = field(default_factory=list)
    legacy_kept: int = 0
    strict_kept: int = 0
    original_kept: int = 0
    has_qc: bool = False
    has_original: bool = False
    expected_strict: Optional[int] = None
    expected_total: Optional[int] = None
    strict_qc_disagreements: list[dict] = field(default_factory=list)
    legacy_original_mismatches: list[dict] = field(default_factory=list)
    strict_reason_counts: dict[str, int] = field(default_factory=dict)


def compare_dataset(
    ds: dict,
    legacy_sites: dict,
    strict_sites: dict,
    strict_reasons: dict,
    qc: dict,
    original_sites: dict,
) -> DatasetResult:
    res = DatasetResult(
        name=ds["name"],
        has_qc=bool(qc),
        has_original=bool(original_sites),
        expected_strict=ds.get("expected_strict_kept"),
        expected_total=ds.get("expected_total_sites"),
    )
    res.legacy_kept = len(legacy_sites)
    res.strict_kept = len(strict_sites)
    res.original_kept = len(original_sites)

    all_keys = set(legacy_sites) | set(strict_sites) | set(original_sites)
    # Also fold in QC-only files (parsed from filename back to a key).
    for fname in qc:
        m = re.match(r"(?P<pdb>[0-9a-z]+)_cluster(?P<n>\d+)_zn\.xyz", fname)
        if m:
            all_keys.add((m.group("pdb"), m.group("n")))

    for key in sorted(all_keys):
        pdb, cluster_n = key
        site = legacy_sites.get(key) or strict_sites.get(key) or original_sites.get(key)
        chain = site.chain if site else ""
        resseq = site.resseq if site else ""
        legacy_kept = key in legacy_sites
        strict_kept = key in strict_sites
        original_kept = key in original_sites
        xyz_name = f"{pdb}_cluster{cluster_n}_Zn.xyz".lower()
        qc_status, qc_reason = qc.get(xyz_name, ("", ""))

        reason = ""
        if not strict_kept:
            reason = strict_reasons.get((pdb, chain, resseq), "")
            # Fall back to a pdb-only match if chain/resseq formatting differs.
            if not reason:
                cand = [v for (rp, rc, rr), v in strict_reasons.items() if rp == pdb]
                if len(cand) == 1:
                    reason = cand[0]

        legacy_vs_original = ""
        if res.has_original:
            legacy_vs_original = "OK" if legacy_kept == original_kept else "MISMATCH"
        strict_vs_qc = ""
        if qc_status:
            strict_vs_qc = "AGREE" if (strict_kept == (qc_status == "OK")) else "DISAGREE"

        row = {
            "dataset": ds["name"],
            "pdb_id": pdb,
            "zn_chain": chain,
            "zn_resseq": resseq,
            "cluster_n": cluster_n,
            "legacy_cn": site.cn if site else "",
            "legacy_coord": site.coord if site else "",
            "legacy_kept": int(legacy_kept),
            "strict_kept": int(strict_kept),
            "qc_status": qc_status,
            "strict_reasons": reason,
            "qc_reasons": qc_reason,
            "legacy_vs_original": legacy_vs_original,
            "strict_vs_qc": strict_vs_qc,
        }
        res.rows.append(row)
        if strict_vs_qc == "DISAGREE":
            res.strict_qc_disagreements.append(row)
        if legacy_vs_original == "MISMATCH":
            res.legacy_original_mismatches.append(row)
        if reason:
            # Tally leading reason tokens for a quick histogram.
            for tag in re.findall(
                r"carbon_contact|donor_distance|his_geometry|extra_ligand|coordination_number",
                reason,
            ):
                res.strict_reason_counts[tag] = res.strict_reason_counts.get(tag, 0) + 1
    return res


def write_comparison_csv(res: DatasetResult, out_path: Path) -> None:
    cols = [
        "dataset", "pdb_id", "zn_chain", "zn_resseq", "cluster_n",
        "legacy_cn", "legacy_coord", "legacy_kept", "strict_kept",
        "qc_status", "strict_reasons", "qc_reasons",
        "legacy_vs_original", "strict_vs_qc",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in res.rows:
            w.writerow(row)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_report(res: DatasetResult, csv_path: Path) -> None:
    print("\n" + "=" * 72)
    print(f"DATASET: {res.name}")
    print("=" * 72)
    exp_s = f" (expected {res.expected_strict})" if res.expected_strict is not None else ""
    exp_t = f"/{res.expected_total}" if res.expected_total is not None else ""
    print(f"  legacy kept sites : {res.legacy_kept}{exp_t}")
    print(f"  strict kept sites : {res.strict_kept}{exp_s}{exp_t}")
    print(f"  strict dropped    : {res.legacy_kept - res.strict_kept}")
    if res.strict_reason_counts:
        hist = ", ".join(f"{k}={v}" for k, v in sorted(res.strict_reason_counts.items()))
        print(f"  strict reasons    : {hist}")

    if res.has_original:
        if res.legacy_original_mismatches:
            print(f"  LEGACY != ORIGINAL: {len(res.legacy_original_mismatches)} mismatch(es) — FLAG:")
            for r in res.legacy_original_mismatches:
                print(f"     - {r['pdb_id']} cluster{r['cluster_n']} "
                      f"(legacy_kept={r['legacy_kept']}, original_kept differs)")
        else:
            print(f"  legacy vs original: MATCH ({res.original_kept} original kept sites reproduced)")

    if res.has_qc:
        n_dis = len(res.strict_qc_disagreements)
        print(f"  strict vs QC      : {n_dis} disagreement(s)")
        for r in res.strict_qc_disagreements:
            side = "strict kept but QC REJECT" if r["strict_kept"] else "strict dropped but QC OK"
            print(f"     - {r['pdb_id']} cluster{r['cluster_n']} [{side}]")
            print(f"         strict: {r['strict_reasons'] or '(no reason line)'}")
            print(f"         qc    : {r['qc_reasons'] or '(none)'}")
    print(f"  comparison CSV    : {csv_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC, help="dataset spec YAML")
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT,
                    help="parent dir the spec's rel_* dataset paths resolve against")
    ap.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT,
                    help="where all generated configs/outputs/CSVs are written")
    ap.add_argument("--datasets", nargs="*", help="subset of dataset names to run")
    ap.add_argument("--mode", choices=["both", "legacy", "strict"], default="both")
    args = ap.parse_args()

    spec = load_spec(args.spec)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    for ds in spec["datasets"]:
        resolve_dataset_paths(ds, args.data_root)

    from scrape_pdb.config import ValidationConfig  # noqa: E402
    if "strict_coordination" not in ValidationConfig.model_fields:
        sys.stderr.write(
            "[warn] ValidationConfig has no 'strict_coordination' field — strict "
            "mode is not implemented in this worktree; strict runs will equal legacy.\n"
        )

    datasets = spec["datasets"]
    if args.datasets:
        wanted = set(args.datasets)
        datasets = [d for d in datasets if d["name"] in wanted]
        if not datasets:
            sys.stderr.write(f"[error] no datasets match {sorted(wanted)}\n")
            return 2

    run_legacy = args.mode in ("both", "legacy")
    run_strict = args.mode in ("both", "strict")

    results: list[DatasetResult] = []
    for ds in datasets:
        ds_root = output_root / ds["name"]
        ds_root.mkdir(parents=True, exist_ok=True)
        input_dir = resolve_input_dir(ds, ds_root)
        n_pdb = len(list(Path(input_dir).glob("*.pdb")))
        print(f"\n### {ds['name']}: {n_pdb} PDB file(s) from {input_dir}")

        legacy_sites: dict = {}
        strict_sites: dict = {}
        strict_reasons: dict = {}

        if run_legacy:
            rd = ds_root / "legacy"
            cfg = build_config(ds, "legacy", spec, rd, input_dir)
            print(f"  running LEGACY  -> {rd}")
            run_pipeline_subprocess(cfg)
            legacy_sites = parse_clusters_csv(rd / "output" / "clusters_summary.csv")

        if run_strict:
            rd = ds_root / "strict"
            cfg = build_config(ds, "strict", spec, rd, input_dir)
            print(f"  running STRICT  -> {rd}")
            run_pipeline_subprocess(cfg)
            strict_sites = parse_clusters_csv(rd / "output" / "clusters_summary.csv")
            strict_reasons = parse_strict_log(rd / "pipeline.log")

        qc = parse_qc(ds.get("qc_report"))
        original_sites = parse_clusters_csv(Path(ds["original_clusters_csv"])) if ds.get("original_clusters_csv") else {}

        res = compare_dataset(ds, legacy_sites, strict_sites, strict_reasons, qc, original_sites)
        csv_path = ds_root / f"compare_{ds['name']}.csv"
        write_comparison_csv(res, csv_path)
        print_report(res, csv_path)
        results.append(res)

    print("\n" + "#" * 72)
    print("SUMMARY")
    print("#" * 72)
    for res in results:
        exp = ""
        if res.expected_strict is not None:
            ok = "✓" if res.strict_kept == res.expected_strict else "✗"
            exp = f"  (expected {res.expected_strict} {ok})"
        print(f"  {res.name:16s} legacy={res.legacy_kept:3d}  strict={res.strict_kept:3d}"
              f"  dropped={res.legacy_kept - res.strict_kept:3d}{exp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
