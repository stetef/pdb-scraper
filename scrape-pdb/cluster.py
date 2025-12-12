#!/usr/bin/env python3
"""Cluster detection & classification."""

from typing import Optional

from .models import Atom
from .constants import WATER_RESIDUES
from .utils import dist
import logging

logger = logging.getLogger(__name__)


def determine_cluster_type(metals_in_comp: list[Atom], target_upper: str, target_element: Optional[str]) -> str:
    if len(metals_in_comp) == 1:
        return "homo"
    if target_element is None:
        # fallback to atom_name check
        same = all(m.atom_name.upper() == target_upper for m in metals_in_comp)
    else:
        same = all(m.element.upper() == target_element for m in metals_in_comp)
    return "multi_homo" if same else "multi_hetero"

def select_neighbors_union(atoms: list[Atom], centers: list[Atom], cutoff: float) -> list[Atom]:
    ccoords = [c.coord for c in centers]
    def in_union(a: Atom) -> bool:
        for cx, cy, cz in ccoords:
            dx = a.x - cx; dy = a.y - cy; dz = a.z - cz
            if dx*dx + dy*dy + dz*dz <= cutoff*cutoff:
                return True
        return False
    return [a for a in atoms if in_union(a)]

def select_neighbors_from(center: Atom, atoms: list[Atom], cutoff: float) -> list[Atom]:
    return [a for a in atoms if dist(a.coord, center.coord) <= cutoff]

def apply_water_toggle(atoms: list[Atom], include_waters: bool) -> list[Atom]:
    if include_waters:
        return [a for a in atoms if a.element.upper() != "H"]
    else:
        return [a for a in atoms if a.element.upper() != "H" and a.resname.upper() not in ("HOH","WAT")]