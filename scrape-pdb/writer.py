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

logger = logging.getLogger(__name__)


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
    os.makedirs(config.output_dir, exist_ok=True)
    if not os.path.isfile(CLUSTERS_CSV):
        with open(CLUSTERS_CSV, "w", newline="") as csvfile:
            w = csv.writer(csvfile)
            w.writerow([
                "PDB","CLUSTER","CLUSTER_TYPE","CENTER_IDX","CENTER_ELEM","CENTER_ATOMNAME","CHAIN","RESSEQ","ICODE","RESNAME","ALTLOC","OCC",
                "CN","GEOM","COORD","RMS_theta","MAX_angle_dev","sigma_d","delta_d","RMS_plane","h_max",
                "planar","axial","distorted","JT","OTHER_METALS","ALTLOC_CASE","ALTLOC_LABEL","XYZ_PATH","RESOLUTION_A"
            ])

def write_clusters_csv_row(row: list):
    ensure_csv_headers()
    with open(CLUSTERS_CSV, "a", newline="") as csvfile:
        w = csv.writer(csvfile)
        w.writerow(row)

def write_altloc_report_header():
    os.makedirs(os.path.dirname(ALTLOC_REPORT), exist_ok=True)
    if not os.path.isfile(ALTLOC_REPORT):
        with open(ALTLOC_REPORT, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["pdb_id","record","serial","atom_name","element","altloc","resname","chain","resseq","x","y","z","cluster_id","cluster_type"])

def append_altloc_rows(rows: list[list]):
    if not rows:
        return
    with open(ALTLOC_REPORT, "a", newline="") as csvfile:
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

