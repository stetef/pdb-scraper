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
        # Accept either `resnames` (preferred) or legacy single `resname`
        resnames = getattr(req, "resnames", None)
        if resnames is None:
            single = getattr(req, "resname", None)
            resnames_u = [str(single).strip().upper()] if single else []
        else:
            resnames_u = [str(r).strip().upper() for r in resnames if str(r).strip()]

        if not resnames_u:
            return False

        atom_names = getattr(req, "atom_names", None)
        try:
            min_count = int(getattr(req, "min_count", 1))
        except Exception:
            min_count = 1

        if atom_names:
            # Any of the atom_names count toward the requirement.
            atom_names_u = [str(x).strip().upper() for x in atom_names if str(x).strip()]
            count = 0
            for rn in resnames_u:
                count += sum(pair_counts.get((rn, an), 0) for an in atom_names_u)
        else:
            count = sum(res_counts.get(rn, 0) for rn in resnames_u)

        if count < max(0, min_count):
            return False

    return True


def diagnose_ligand_requirements(
    coord_neighbors: Iterable[Atom],
    ligand_requirements: Optional[Sequence[object]],
) -> list[dict]:
    """Return per-requirement diagnostics for why a ligand requirement may fail.

    Intended for reporting statistics (e.g., common residue-name variants like HID/HIE/HIP).
    """

    if not ligand_requirements:
        return []

    pairs = [
        (a.resname.strip().upper(), a.atom_name.strip().upper())
        for a in coord_neighbors
    ]
    pair_counts = Counter(pairs)
    res_counts = Counter(r for (r, _) in pairs)

    out: list[dict] = []
    for idx, req in enumerate(ligand_requirements):
        resnames = getattr(req, "resnames", None)
        if resnames is None:
            single = getattr(req, "resname", None)
            resnames_u = [str(single).strip().upper()] if single else []
        else:
            resnames_u = [str(r).strip().upper() for r in resnames if str(r).strip()]

        atom_names = getattr(req, "atom_names", None)
        atom_names_u = [str(x).strip().upper() for x in (atom_names or []) if str(x).strip()]

        try:
            min_count = int(getattr(req, "min_count", 1))
        except Exception:
            min_count = 1
        min_count = max(0, min_count)

        if atom_names_u:
            # Count matches by atom name regardless of residue name (to detect naming mismatches)
            resname_counts_for_atom_names = Counter(
                r for (r, an) in pairs if an in set(atom_names_u)
            )
            count_any_resname = sum(resname_counts_for_atom_names.values())
            count_allowed = 0
            for rn in resnames_u:
                count_allowed += sum(pair_counts.get((rn, an), 0) for an in atom_names_u)
        else:
            # No atom-name constraint: work at residue level
            resname_counts_for_atom_names = Counter(res_counts)
            count_any_resname = sum(resname_counts_for_atom_names.values())
            count_allowed = sum(res_counts.get(rn, 0) for rn in resnames_u)

        out.append(
            {
                "req_index": idx,
                "expected_resnames": resnames_u,
                "atom_names": atom_names_u or None,
                "min_count": min_count,
                "count_allowed": int(count_allowed),
                "count_any_resname": int(count_any_resname),
                "resname_counts_for_atom_names": dict(resname_counts_for_atom_names),
            }
        )

    return out
