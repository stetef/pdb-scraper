#!/usr/bin/env python3
"""Validation helpers for coordination-site filtering."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional, Sequence

from .models import Atom


def passes_ligand_requirements(
    coord_neighbors: Iterable[Atom],
    ligand_requirements: Optional[Sequence[object]],
) -> bool:
    """Return True if the coordinating neighbors satisfy all ligand requirements.

    A requirement is expected to have attributes:
      - resname: str (e.g. "CYS")
      - atom_names: Optional[list[str]] (e.g. ["SG"])
      - min_count: int

    Matching is done on Atom.resname (uppercased) and Atom.atom_name (stripped+uppercased).
    """

    if not ligand_requirements:
        return True

    # Pre-count all (resname, atom_name) pairs to make checks fast.
    pairs = [
        (a.resname.strip().upper(), a.atom_name.strip().upper())
        for a in coord_neighbors
    ]
    pair_counts = Counter(pairs)

    # Also count by residue only (for requirements that don't specify atom_names)
    res_counts = Counter(r for (r, _) in pairs)

    for req in ligand_requirements:
        resname = getattr(req, "resname", None)
        if not resname:
            return False
        resname_u = str(resname).strip().upper()

        atom_names = getattr(req, "atom_names", None)
        try:
            min_count = int(getattr(req, "min_count", 1))
        except Exception:
            min_count = 1

        if atom_names:
            # Any of the atom_names count toward the requirement.
            atom_names_u = [str(x).strip().upper() for x in atom_names if str(x).strip()]
            count = sum(pair_counts.get((resname_u, an), 0) for an in atom_names_u)
        else:
            count = res_counts.get(resname_u, 0)

        if count < max(0, min_count):
            return False

    return True
