#!/usr/bin/env python3
"""AltLoc handling logic."""

from typing import Optional
from pathlib import Path
import os

from .models import Atom, MustHaveSpec
from .writer import write_xyz
from .cluster import apply_water_toggle, select_coordinating_neighbors
from .geometry import coord_string
from .validation import passes_ligand_requirements
from .utils import dist

import logging

logger = logging.getLogger("pipeline.altloc")


def group_by_id(atoms: list[Atom]) -> dict[tuple[str,str,str,str], dict[str, Atom]]:
    """
    Group atoms by identity (chain, resseq, icode, atom_name) and altloc label.
    Returns: { (chain, resseq, icode, atom_name): {altloc_label_or_blank: Atom, ...}, ... }
    """
    from collections import defaultdict
    G: dict[tuple[str,str,str,str], dict[str, Atom]] = defaultdict(dict)
    for a in atoms:
        key = (a.chain, a.resseq, a.icode, a.atom_name)
        lab = a.altloc or ""
        G[key][lab] = a
    return G

def altloc_set_for_atom_id(G: dict[tuple[str,str,str,str], dict[str, Atom]], key: tuple[str,str,str,str]) -> set[str]:
    if key not in G:
        return set()
    labs = set(G[key].keys())
    if "" in labs:
        labs.remove("")
    return labs

def build_altloc_files_for_center(
        pdb_id: str,
        center: Atom,
        selected_atoms_raw: list[Atom],
    all_atoms: Optional[list[Atom]],
        raw_groups: dict[tuple[str,str,str,str], dict[str,Atom]],
        cutoff: float,
        out_dir: Path | str,
        base_name_common: str,
        origin_kind: str,
        centroid_pt: tuple[float,float,float],
        resolution_angs: Optional[float],
        cluster_index: int,
        cluster_type: str,
        target_upper: str,
        include_waters: bool,
        must_have: MustHaveSpec,
        coord_distance_min: float,
        coord_distance_max: float,
        coord_filters: Optional[set[str]],
        ligand_requirements: Optional[list[object]] = None,
        coord_residue_keys: Optional[set[tuple[str, str, str]]] = None) -> list[str]:
    """
    Implements Step 6 Situations 1–3 around the chosen center.
    Returns list of written XYZ file paths.
    """
    written: list[str] = []

    def keep_atom_in_cluster(a: Atom, origin: Atom) -> bool:
        if a.serial == origin.serial:
            return True
        if coord_residue_keys and (a.resname.upper(), a.chain, a.resseq) in coord_residue_keys:
            return True
        return dist(a.coord, origin.coord) <= cutoff

    center_key = (center.chain, center.resseq, center.icode, center.atom_name)
    center_labels = altloc_set_for_atom_id(raw_groups, center_key)  # excludes blank

    # Build neighbor identity set from selected_atoms_raw (to limit scope)
    neighbor_keys = set((a.chain, a.resseq, a.icode, a.atom_name) for a in selected_atoms_raw if a.serial != center.serial)

    # Determine neighbor labels present
    neighbor_label_union: set[str] = set()
    for key in neighbor_keys:
        labs = altloc_set_for_atom_id(raw_groups, key)
        neighbor_label_union |= labs
    # Remove blanks in union; blanks handled separately
    neighbor_label_union = set(l for l in neighbor_label_union if l)

    # Situation 3: center unlabeled; neighbors have altlocs
    if not center_labels and neighbor_label_union:
        # For each neighbor label ℓ, build one file with center + neighbors at ℓ + blanks
        for lab in sorted(neighbor_label_union):
            # Build atom set for this label
            chosen_atoms: list[Atom] = []
            for key in neighbor_keys:
                # pick neighbor at this label, else blank, else skip
                a = raw_groups.get(key, {}).get(lab) or raw_groups.get(key, {}).get("")
                if a:
                    chosen_atoms.append(a)
            # Include center as-is (unlabeled)
            chosen_atoms.append(center)

            # Filter by cutoff from center (to avoid drift)
            chosen_atoms = [a for a in chosen_atoms if keep_atom_in_cluster(a, center)]

            # Apply H/water toggle
            chosen_atoms = apply_water_toggle(chosen_atoms, include_waters=include_waters)

            # Must-have filter on all cluster atoms (exclude center for counts)
            if not must_have.passes([a for a in chosen_atoms if a.serial != center.serial]):
                continue
            
            # Coord + ligand filter on coordinating neighbors only
            if coord_filters or ligand_requirements:
                coord_neigh = select_coordinating_neighbors(center, chosen_atoms, coord_distance_min, coord_distance_max)
                coord_neigh = apply_water_toggle(coord_neigh, include_waters=include_waters)
                if ligand_requirements and not passes_ligand_requirements(coord_neigh, ligand_requirements):
                    continue
                if coord_filters:
                    coord_str = coord_string(coord_neigh)
                    if coord_str not in coord_filters:
                        continue

            alt_tag = f"altlocLIG{lab}"
            xyz_name = f"{base_name_common}_{alt_tag}.xyz"
            xyz_path = str(Path(out_dir) / "xyz_files" / xyz_name)
            extra = f"CLUSTER_TYPE={cluster_type} ALTLOC_CASE=3 ALTLOC_LABEL={lab}"
            write_xyz(
                xyz_path,
                pdb_id,
                cluster_index,
                target_upper,
                cutoff,
                origin_kind,
                centroid_pt,
                chosen_atoms,
                center,
                resolution_angs,
                extra_comment=extra,
                coord_residue_keys=coord_residue_keys,
                pc_source_atoms=(all_atoms if all_atoms is not None else chosen_atoms),
            )
            written.append(xyz_path)
        return written

    # If center has labels (A/B/..):
    if center_labels:
        # Check if ALL neighbors also have exactly those labels (situation 1) or some unlabeled (situation 2)
        all_neighbors_have_labels = True
        for key in neighbor_keys:
            labs = altloc_set_for_atom_id(raw_groups, key)
            if not labs or not center_labels.issubset(labs):
                all_neighbors_have_labels = False
                break

        if all_neighbors_have_labels:
            # Situation 1: write one file per label; use only atoms with that label
            for lab in sorted(center_labels):
                chosen_atoms: list[Atom] = []
                for key in neighbor_keys:
                    a = raw_groups.get(key, {}).get(lab)
                    if a:
                        chosen_atoms.append(a)
                # Include center at lab
                c_lab = raw_groups.get(center_key, {}).get(lab)
                if c_lab:
                    chosen_atoms.append(c_lab)
                    origin = c_lab
                else:
                    origin = center
                # cutoff from origin
                chosen_atoms = [a for a in chosen_atoms if keep_atom_in_cluster(a, origin)]
                # Apply toggles
                chosen_atoms = apply_water_toggle(chosen_atoms, include_waters=include_waters)
                # Must-have on all cluster atoms (exclude center)
                if not must_have.passes([a for a in chosen_atoms if a.serial != origin.serial]):
                    continue
                # Coord + ligand filter on coordinating neighbors only
                if coord_filters or ligand_requirements:
                    coord_neigh = select_coordinating_neighbors(origin, chosen_atoms, coord_distance_min, coord_distance_max)
                    coord_neigh = apply_water_toggle(coord_neigh, include_waters=include_waters)
                    if ligand_requirements and not passes_ligand_requirements(coord_neigh, ligand_requirements):
                        continue
                    if coord_filters:
                        coord_str = coord_string(coord_neigh)
                        if coord_str not in coord_filters:
                            continue
                alt_tag = f"altloc{lab}"
                xyz_name = f"{base_name_common}_{alt_tag}.xyz"
                xyz_path = str(Path(out_dir) / "xyz_files" / xyz_name)
                extra = f"CLUSTER_TYPE={cluster_type} ALTLOC_CASE=1 ALTLOC_LABEL={lab}"
                write_xyz(
                    xyz_path,
                    pdb_id,
                    cluster_index,
                    target_upper,
                    cutoff,
                    origin_kind,
                    centroid_pt,
                    chosen_atoms,
                    origin,
                    resolution_angs,
                    extra_comment=extra,
                    coord_residue_keys=coord_residue_keys,
                    pc_source_atoms=(all_atoms if all_atoms is not None else chosen_atoms),
                )
                written.append(xyz_path)
            return written
        else:
            # Situation 2: neighbors may be unlabeled; build one per center label, include unlabeled neighbors in each
            for lab in sorted(center_labels):
                chosen_atoms: list[Atom] = []
                for key in neighbor_keys:
                    a = raw_groups.get(key, {}).get(lab) or raw_groups.get(key, {}).get("")
                    if a:
                        chosen_atoms.append(a)
                c_lab = raw_groups.get(center_key, {}).get(lab)
                if c_lab:
                    chosen_atoms.append(c_lab)
                    origin = c_lab
                else:
                    origin = center
                chosen_atoms = [a for a in chosen_atoms if keep_atom_in_cluster(a, origin)]
                chosen_atoms = apply_water_toggle(chosen_atoms, include_waters=include_waters)
                # Must-have on all cluster atoms
                if not must_have.passes([a for a in chosen_atoms if a.serial != origin.serial]):
                    continue
                # Coord + ligand filter on coordinating neighbors only
                if coord_filters or ligand_requirements:
                    coord_neigh = select_coordinating_neighbors(origin, chosen_atoms, coord_distance_min, coord_distance_max)
                    coord_neigh = apply_water_toggle(coord_neigh, include_waters=include_waters)
                    if ligand_requirements and not passes_ligand_requirements(coord_neigh, ligand_requirements):
                        continue
                    if coord_filters:
                        coord_str = coord_string(coord_neigh)
                        if coord_str not in coord_filters:
                            continue
                alt_tag = f"altloc{center.element.title()}{lab}"
                xyz_name = f"{base_name_common}_{alt_tag}.xyz"
                xyz_path = str(Path(out_dir) / "xyz_files" / xyz_name)
                extra = f"CLUSTER_TYPE={cluster_type} ALTLOC_CASE=2 ALTLOC_LABEL={lab}"
                write_xyz(
                    xyz_path,
                    pdb_id,
                    cluster_index,
                    target_upper,
                    cutoff,
                    origin_kind,
                    centroid_pt,
                    chosen_atoms,
                    origin,
                    resolution_angs,
                    extra_comment=extra,
                    coord_residue_keys=coord_residue_keys,
                    pc_source_atoms=(all_atoms if all_atoms is not None else chosen_atoms),
                )
                written.append(xyz_path)
            return written

    # No altloc scenario → write a single base file (no tag) handled by caller; nothing to do here
    return written
