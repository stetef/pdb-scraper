#!/usr/bin/env python3
"""Geometry classification & metrics."""

from .models import Atom, GeometryResult, coord_string_from_atoms
from .constants import GEOMETRY_THRESHOLDS
from .utils import dist
import math
import logging

logger = logging.getLogger("pipeline.geometry")

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False
    logger.warning("NumPy not available - planarity metrics disabled")


def _angle(a: tuple[float,float,float], o: tuple[float,float,float], b: tuple[float,float,float]) -> float:
    """Angle A-O-B in degrees."""
    ax, ay, az = a[0]-o[0], a[1]-o[1], a[2]-o[2]
    bx, by, bz = b[0]-o[0], b[1]-o[1], b[2]-o[2]
    da = math.sqrt(ax*ax+ay*ay+az*az)
    db = math.sqrt(bx*bx+by*by+bz*bz)
    if da == 0 or db == 0:
        return 0.0
    dot = (ax*bx + ay*by + az*bz) / (da*db)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))

def _planarity_metrics(points: list[tuple[float,float,float]]) -> tuple[float,float]:
    """Return RMS height from best plane and max height. If numpy missing, return (NA, NA) as (-1,-1)."""
    if not HAVE_NUMPY:
        return (-1.0, -1.0)
    if len(points) < 3:
        return (0.0, 0.0)
    P = np.array(points, dtype=float)
    C = P.mean(axis=0)
    X = P - C
    # SVD: principal axes; normal is smallest singular vector
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    normal = Vt[-1, :]
    heights = X @ normal
    rms = float(np.sqrt(np.mean(heights**2)))
    hmax = float(np.max(np.abs(heights)))
    return (rms, hmax)

def classify_geometry(center: Atom, neighbors: list[Atom]) -> tuple[str, dict[str, float], dict[str, str]]:
    """
    Return (geom_label, metrics, flags)
    metrics: {"CN":..., "RMS_theta":..., "MAX_angle_dev":..., "sigma_d":..., "delta_d":..., "RMS_plane":..., "h_max":...}
    flags: {"planar":"Yes/No","axial":"Yes/No","distorted":"Yes/No","JT":"Yes/No"}
    """
    CN = len(neighbors)
    # Distances
    dists = [dist(center.coord, n.coord) for n in neighbors]
    sigma_d = (float(np.std(dists)) if HAVE_NUMPY and len(dists)>1 else (0.0 if len(dists)<=1 else 0.0))
    delta_d = (max(dists)-min(dists)) if dists else 0.0

    # Angles
    angles = []
    for i in range(CN):
        for j in range(i+1, CN):
            angles.append(_angle(neighbors[i].coord, center.coord, neighbors[j].coord))
    RMS_theta = float(np.sqrt(np.mean([(a-109.47)**2 for a in angles]))) if HAVE_NUMPY and angles else 0.0  # baseline vs tetra
    MAX_angle_dev = max([abs(a-109.47) for a in angles]) if angles else 0.0

    # Planarity
    RMS_plane, h_max = _planarity_metrics([n.coord for n in neighbors])

    # Heuristic flags
    planar_flag = "Yes" if (RMS_plane >= 0 and RMS_plane <= 0.35) else ("No" if RMS_plane >= 0 else "NA")
    axial_flag = "Yes" if angles and max(angles) >= 160.0 else ("No" if angles else "NA")

    # Candidate by CN
    geom = "HCN" if CN > 6 else "UNK"
    distorted = False
    JT = "No"

    # Threshold helpers
    def is_good_angle(dev): return dev <= 12.0
    def is_border_angle(dev): return 12.0 < dev <= 20.0

    def pick(label, baseline=109.47):
        nonlocal geom, distorted
        # Recompute RMS vs a baseline if provided
        if HAVE_NUMPY and angles:
            rms = float(np.sqrt(np.mean([(a-baseline)**2 for a in angles])))
        else:
            rms = RMS_theta
        geom = label
        if rms > 20.0 or sigma_d > 0.15 or (RMS_plane > 0.35 and label in ("SQP","TP")):
            geom = label + "_d"
            distorted = True
        return rms

    if CN == 2:
        # linear vs bent
        if angles and max(angles) >= 160.0:
            pick("LIN", baseline=180.0)
        else:
            pick("BEN", baseline=120.0)
    elif CN == 3:
        # TP vs T-shaped
        if planar_flag == "Yes":
            # Equiangular?
            if HAVE_NUMPY and angles:
                rms_tp = float(np.sqrt(np.mean([(a-120.0)**2 for a in angles])))
                rms_ts = float(np.sqrt(np.mean([(a-90.0 if i==0 else 180.0)-angles[i] for i in range(len(angles))]))) if False else 999.0
                if rms_tp <= rms_ts:
                    pick("TP", baseline=120.0)
                else:
                    pick("TS", baseline=90.0)
            else:
                pick("TP", baseline=120.0)
        else:
            pick("TS", baseline=90.0)
    elif CN == 4:
        # TD vs SQP (rare: seesaw)
        if planar_flag == "Yes":
            pick("SQP", baseline=90.0)
        else:
            # default tetra baseline
            pick("TD", baseline=109.47)
    elif CN == 5:
        # TBP vs SPY
        if axial_flag == "Yes":
            # Choose based on planarity of basal 4?
            if planar_flag == "Yes":
                pick("SPY", baseline=90.0)
            else:
                pick("TBP", baseline=109.47)  # mixed angles
        else:
            pick("TBP", baseline=109.47)
    elif CN == 6:
        # OH (allow JT: two long trans bonds)
        pick("OH", baseline=90.0)
        # Simple JT heuristic: delta_d >= 0.2 suggests elongation
        if delta_d >= 0.2:
            JT = "Yes"

    flags = {
        "planar": planar_flag,
        "axial": axial_flag,
        "distorted": "Yes" if distorted else "No",
        "JT": JT
    }
    metrics = {
        "CN": CN,
        "RMS_theta": RMS_theta,
        "MAX_angle_dev": MAX_angle_dev,
        "sigma_d": sigma_d,
        "delta_d": delta_d,
        "RMS_plane": RMS_plane,
        "h_max": h_max
    }
    return geom, metrics, flags

def coord_string(neighbors: list[Atom]) -> str:
    """COORD string like 4N1O from neighbor elements (uppercase)."""
    from collections import Counter
    cnt = Counter([n.element.upper() for n in neighbors if n.element])
    parts = []
    for elm in sorted(cnt.keys()):
        parts.append(f"{cnt[elm]}{elm.title() if len(elm)==2 else elm.upper()}")
    return "".join(parts)