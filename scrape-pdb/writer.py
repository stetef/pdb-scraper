#!/usr/bin/env python3
"""XYZ, CSV, cache writers."""

from pathlib import Path
import csv
import json
from typing import Optional
import os

from .models import Atom
from .constants import CLUSTERS_CSV_FIELDS, ALTLOC_REPORT_FIELDS
from .config import PipelineConfig

import logging

logger = logging.getLogger("pipeline.writer")


def write_xyz(path: str,
              pdb_id: str,
              cluster_index: int,
              target: str,
              cutoff: float,
              origin_kind: str,
              centroid_pt: tuple[float, float, float],
              atoms: list[Atom],
              origin_atom: Atom,
              resolution_angs: Optional[float] = None,
              extra_comment: str = "") -> None:
    """Write XYZ with origin translated to origin_atom."""
    ox, oy, oz = origin_atom.coord
    translated = []
    for a in atoms:
        translated.append((a.element, a.x - ox, a.y - oy, a.z - oz, a))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{len(translated)}\n")
        res_str = f"{resolution_angs:.2f}" if isinstance(resolution_angs, (int, float)) else "NA"
        comment = (
            f"PDB={pdb_id} CLUSTER={cluster_index} TARGET={target} CUTOFF={cutoff:.3f} "
            f"ORIGIN={origin_kind} CENTROID=({centroid_pt[0]:.3f},{centroid_pt[1]:.3f},{centroid_pt[2]:.3f}) "
            f"RESOLUTION_A={res_str}"
        )
        if extra_comment:
            comment += " " + extra_comment.strip()
        f.write(comment + "\n")
        for elm, x, y, z, a in translated:
            resseq_icode = f"{a.resseq}{a.icode}".strip()
            meta = f"RES={a.resname} CHAIN={a.chain} RESSEQ={resseq_icode} ATOM={a.atom_name} REC={a.record}"
            f.write(f"{elm:2s}  {x: .6f}  {y: .6f}  {z: .6f}  # {meta}\n")

def ensure_csv_headers(config: PipelineConfig):
    # Ensure output directory exists and clusters CSV has header row
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    clusters_path = config.clusters_csv
    if not clusters_path.is_file():
        with open(clusters_path, "w", newline="") as csvfile:
            w = csv.writer(csvfile)
            w.writerow(CLUSTERS_CSV_FIELDS)

def write_clusters_csv_row(row: list, config: PipelineConfig):
    ensure_csv_headers(config)
    clusters_path = config.clusters_csv
    with open(clusters_path, "a", newline="") as csvfile:
        w = csv.writer(csvfile)
        w.writerow(row)

def write_altloc_report_header(config: PipelineConfig):
    altloc_path = config.altloc_report
    Path(altloc_path.parent).mkdir(parents=True, exist_ok=True)
    if not altloc_path.is_file():
        with open(altloc_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(ALTLOC_REPORT_FIELDS)

def append_altloc_rows(rows: list[list], config: PipelineConfig):
    if not rows:
        return
    altloc_path = config.altloc_report
    with open(altloc_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)

def append_cache(runs: list[dict], config: PipelineConfig):
    try:
        if os.path.isfile(config.phase1_cache):
            with open(config.phase1_cache, "r") as f:
                prev = json.load(f)
        else:
            prev = {}
        prev_runs = prev.get("runs", [])
        prev_runs.extend(runs)
        with open(config.phase1_cache, "w") as f:
            json.dump({"runs": prev_runs}, f, indent=2)
        logger.info(f"[+] Updated cache: {config.phase1_cache}")
    except Exception as e:
        logger.info(f"[!] Failed to update cache: {e}")

