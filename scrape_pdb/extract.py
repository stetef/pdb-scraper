#!/usr/bin/env python3
"""Pure site extraction for the EARL library API.

``extract_sites_from_file`` is a side-effect-free function of (file, element,
config): it reads a PDB/mmCIF file and returns :class:`SiteCandidate` objects in
memory. It reuses the exact scientific primitives the CLI uses (clustering,
neighbour selection, geometry classification) but writes nothing to disk,
deletes nothing, appends to no shared CSV/cache, and mutates no global logger.

Key difference from the CLI's ``parser.process_pdb``: a polynuclear cluster
yields **one SiteCandidate per absorber-element atom**, each centred on its own
absorber (design doc 04 §3.1), which is what powers EARL's multi-site averaging.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Optional

from .altloc import altloc_set_for_atom_id, group_by_id
from .cluster import (
    apply_water_toggle,
    determine_cluster_type,
    expand_selection_by_residue_keys,
    residue_key,
    select_coordinating_neighbors,
    select_neighbors_union,
)
from .constants import ALL_METALS, WATER_RESIDUES
from .geometry import classify_geometry, coord_string
from .models import Atom, MustHaveSpec, parse_must_have
from .parser import (
    _extract_pdb_model_blocks,
    collapse_altloc,
    detect_file_format,
    load_atoms_auto,
    parse_pdb_resolution,
)
from .site import ExtractConfig, ExtractResult, SiteCandidate
from .utils import centroid, connected_components, dist
from .validation import passes_ligand_requirements
from .writer import append_hydrogens_to_text, render_xyz_text

_LOGGER = logging.getLogger("pipeline.extract")


def _pretty_symbol(elem: str) -> str:
    """"ZN" -> "Zn", "N" -> "N"."""
    e = (elem or "").strip()
    if len(e) == 2:
        return e[0].upper() + e[1].lower()
    return e.upper()


def _residue_label(a: Atom) -> str:
    """Human-readable coordinating-atom label, e.g. "CYS A 12 SG"."""
    resseq = f"{a.resseq}{a.icode}".strip()
    return f"{a.resname} {a.chain} {resseq} {a.atom_name}".replace("  ", " ").strip()


def _coordination_summary(origin: Atom, coord_neigh: list[Atom]) -> list[dict]:
    """First-shell summary grouped by element with distance range and residues."""
    by_elem: dict[str, list[tuple[float, Atom]]] = {}
    for a in coord_neigh:
        e = (a.element or "").upper()
        by_elem.setdefault(e, []).append((dist(a.coord, origin.coord), a))

    out: list[dict] = []
    for e in sorted(by_elem):
        items = by_elem[e]
        dists = [d for d, _ in items]
        out.append({
            "element": _pretty_symbol(e),
            "count": len(items),
            "rmin": round(min(dists), 3),
            "rmax": round(max(dists), 3),
            "is_water": any(a.resname.upper() in WATER_RESIDUES for _, a in items),
            "residues": sorted({_residue_label(a) for _, a in items}),
        })
    return out


def _altloc_groups(
    center: Atom,
    selected_atoms_raw: list[Atom],
    raw_groups: dict,
    cutoff: float,
    include_waters: bool,
    coord_residue_keys: Optional[set],
    altloc_split: bool,
) -> list[tuple[list[Atom], Atom, Optional[int], Optional[str]]]:
    """Enumerate altloc variants of the crop around ``center``.

    Returns a list of ``(atoms, origin, altloc_case, altloc_label)``. When there
    are no altlocs (or ``altloc_split`` is False) a single group is returned with
    ``altloc_case=None``. Pure re-implementation of the three altloc situations
    the CLI writes, minus filtering/file-writing (the library shows all sites).
    """

    def keep(a: Atom, origin: Atom) -> bool:
        if a.serial == origin.serial:
            return True
        if coord_residue_keys and (a.resname.upper(), a.chain, a.resseq) in coord_residue_keys:
            return True
        return dist(a.coord, origin.coord) <= cutoff

    if not altloc_split:
        return [(apply_water_toggle(selected_atoms_raw, include_waters), center, None, None)]

    center_key = (center.chain, center.resseq, center.icode, center.atom_name)
    center_labels = altloc_set_for_atom_id(raw_groups, center_key)

    neighbor_keys = {
        (a.chain, a.resseq, a.icode, a.atom_name)
        for a in selected_atoms_raw
        if a.serial != center.serial
    }
    neighbor_label_union: set[str] = set()
    for key in neighbor_keys:
        neighbor_label_union |= altloc_set_for_atom_id(raw_groups, key)
    neighbor_label_union = {lbl for lbl in neighbor_label_union if lbl}

    groups: list[tuple[list[Atom], Atom, Optional[int], Optional[str]]] = []

    # Situation 3: center unlabeled, neighbours labelled.
    if not center_labels and neighbor_label_union:
        for lab in sorted(neighbor_label_union):
            chosen: list[Atom] = []
            for key in neighbor_keys:
                a = raw_groups.get(key, {}).get(lab) or raw_groups.get(key, {}).get("")
                if a:
                    chosen.append(a)
            chosen.append(center)
            chosen = [a for a in chosen if keep(a, center)]
            chosen = apply_water_toggle(chosen, include_waters)
            groups.append((chosen, center, 3, lab))
        return groups

    # Situations 1 & 2: center labelled.
    if center_labels:
        all_labeled = True
        for key in neighbor_keys:
            labs = altloc_set_for_atom_id(raw_groups, key)
            if not labs or not center_labels.issubset(labs):
                all_labeled = False
                break
        for lab in sorted(center_labels):
            chosen = []
            for key in neighbor_keys:
                if all_labeled:
                    a = raw_groups.get(key, {}).get(lab)
                else:
                    a = raw_groups.get(key, {}).get(lab) or raw_groups.get(key, {}).get("")
                if a:
                    chosen.append(a)
            c_lab = raw_groups.get(center_key, {}).get(lab)
            origin = c_lab if c_lab else center
            if c_lab:
                chosen.append(c_lab)
            chosen = [a for a in chosen if keep(a, origin)]
            chosen = apply_water_toggle(chosen, include_waters)
            groups.append((chosen, origin, 1 if all_labeled else 2, lab))
        return groups

    # No altlocs anywhere.
    return [(apply_water_toggle(selected_atoms_raw, include_waters), center, None, None)]


def _extract_from_atoms(
    *,
    raw_atoms: list[Atom],
    resolution: Optional[float],
    element: str,
    config: ExtractConfig,
    pdb_id: str,
    model: Optional[int],
    must_have: MustHaveSpec,
    logger: logging.Logger,
) -> tuple[list[SiteCandidate], Counter]:
    """Extract every absorber site from one already-parsed structure/model."""
    rejections: Counter = Counter()
    sites: list[SiteCandidate] = []

    atoms = collapse_altloc(raw_atoms, policy="prefer_blank_or_A")
    raw_groups = group_by_id(raw_atoms)

    metal_set = set(ALL_METALS) - {m.upper() for m in config.metals_excluded}
    metals_all = [a for a in atoms if a.record == "HETATM" and a.element.upper() in metal_set]
    if not metals_all:
        rejections["no_metals"] += 1
        return sites, rejections

    comps = connected_components([m.coord for m in metals_all], config.cutoff)

    coord_filters = config.coord
    cmin = config.coordination_distance_min
    cmax = config.coordination_distance_max
    sel_radius = config.selection_radius

    cluster_counter = 0
    found_target = False

    for comp in comps:
        comp_metals = [metals_all[i] for i in comp]
        target_atoms = [m for m in comp_metals if m.atom_name.upper() == element]
        if not target_atoms:
            continue
        found_target = True
        cluster_counter += 1
        cluster_id = f"cluster{cluster_counter}"
        comp_type = determine_cluster_type(comp_metals, element, element)

        centers = comp_metals
        c_centroid = centroid([c.coord for c in centers])

        selected_union = select_neighbors_union(atoms, centers, sel_radius)

        coord_residue_keys: set = set()
        for t in target_atoms:
            for a in select_coordinating_neighbors(t, selected_union, cmin, cmax):
                coord_residue_keys.add(residue_key(a))
        selection_residue_keys = {residue_key(a) for a in selected_union} | coord_residue_keys
        selected_expanded = expand_selection_by_residue_keys(atoms, selected_union, selection_residue_keys)

        sibling_serials = [t.serial for t in target_atoms]

        for t in target_atoms:
            siblings = tuple(s for s in sibling_serials if s != t.serial)
            groups = _altloc_groups(
                t, selected_expanded, raw_groups, sel_radius,
                config.include_waters, selection_residue_keys, config.altloc_split,
            )

            for grp_atoms, origin, altloc_case, altloc_label in groups:
                coord_neigh = select_coordinating_neighbors(origin, grp_atoms, cmin, cmax)
                coord_neigh = apply_water_toggle(coord_neigh, include_waters=config.include_waters)

                # Optional filters (OFF by default = show all sites).
                neigh_all = [a for a in grp_atoms if a.serial != origin.serial]
                if not must_have.passes(neigh_all):
                    rejections["must_have"] += 1
                    continue
                if config.ligand_requirements and not passes_ligand_requirements(coord_neigh, config.ligand_requirements):
                    rejections["ligand_requirements"] += 1
                    continue
                coord_str = coord_string(coord_neigh)
                if coord_filters and coord_str not in coord_filters:
                    rejections["coord_string"] += 1
                    continue

                geom, metrics, flags = classify_geometry(origin, coord_neigh)

                extra_bits = [f"CLUSTER_TYPE={comp_type}"]
                if altloc_case is not None:
                    extra_bits.append(f"ALTLOC_CASE={altloc_case} ALTLOC_LABEL={altloc_label}")
                origin_kind = "centroid" if len(centers) > 1 else "single_center"

                xyz_text = render_xyz_text(
                    grp_atoms, origin,
                    pdb_id=pdb_id, cluster_index=cluster_counter, target=element,
                    cutoff=sel_radius, origin_kind="absorber", centroid_pt=c_centroid,
                    resolution_angs=resolution, extra_comment=" ".join(extra_bits),
                    drop_backbone=config.drop_backbone, titlecase=True, absorber_first=True,
                )
                hydrogens_added = False
                if config.add_hydrogens:
                    xyz_text, hydrogens_added = append_hydrogens_to_text(xyz_text)

                geometry = {
                    "label": geom,
                    "CN": int(metrics["CN"]),
                    "RMS_theta": round(metrics["RMS_theta"], 3),
                    "sigma_d": round(metrics["sigma_d"], 4),
                    "distorted": flags["distorted"] == "Yes",
                    "planar": flags["planar"] == "Yes",
                    "axial": flags["axial"] == "Yes",
                    "jahn_teller": flags["JT"] == "Yes",
                }
                coordination = _coordination_summary(origin, coord_neigh)
                ligand_residues = sorted({
                    _residue_label(a) for a in coord_neigh
                    if a.resname.upper() not in WATER_RESIDUES
                })

                provenance = {
                    "pdb_id": pdb_id,
                    "model": model,
                    "cluster_id": cluster_id,
                    "cluster_type": comp_type,
                    "altloc_case": altloc_case,
                    "altloc_label": altloc_label,
                    "resolution_A": resolution,
                    "coord_string": coord_str,
                    "origin_kind_in_cluster": origin_kind,
                    # modelling choices applied (surfaced for expert review):
                    "drop_backbone": config.drop_backbone,
                    "include_waters": config.include_waters,
                    "cutoff": config.cutoff,
                    "selection_radius": sel_radius,
                    "coordination_window": [cmin, cmax],
                }

                sites.append(SiteCandidate(
                    source="pdb",
                    pdb_id=pdb_id,
                    model=model,
                    cluster_id=cluster_id,
                    cluster_type=comp_type,
                    target_atom_index=t.serial,
                    sibling_target_indices=siblings,
                    altloc_case=altloc_case,
                    altloc_label=altloc_label,
                    resolution_A=resolution,
                    geometry=geometry,
                    coordination=coordination,
                    ligand_residues=ligand_residues,
                    xyz_text=xyz_text,
                    hydrogens_added=hydrogens_added,
                    provenance=provenance,
                ))

    if not found_target:
        rejections["no_target_component"] += 1

    return sites, rejections


def extract_sites_from_file(
    pdb_path: Path | str,
    element: str,
    config: ExtractConfig,
    logger: Optional[logging.Logger] = None,
) -> ExtractResult:
    """Extract every site of ``element`` from a PDB/mmCIF file. Pure function."""
    logger = logger or _LOGGER
    element_up = element.strip().upper()
    if element_up not in ALL_METALS:
        raise ValueError(f"element {element!r} is not a recognised metal")

    pdb_path = str(pdb_path)
    base_id = os.path.splitext(os.path.basename(pdb_path))[0]
    must_have = parse_must_have(config.must_have or "")

    result = ExtractResult(stats={"pdb_id": base_id, "element": element_up})

    # Multi-MODEL (NMR) splitting, mirroring parser.process_pdb.
    model_inputs: list[tuple[Optional[int], list[Atom], Optional[float]]] = []
    tmpdir: Optional[tempfile.TemporaryDirectory] = None
    try:
        if detect_file_format(pdb_path) == "pdb":
            try:
                header_lines, model_blocks = _extract_pdb_model_blocks(pdb_path)
            except Exception:
                model_blocks = []
        else:
            model_blocks = []

        if len(model_blocks) > 1 and not config.first_model_only:
            tmpdir = tempfile.TemporaryDirectory(prefix=f"{base_id}_models_", dir=str(config.workdir))
            resolution = parse_pdb_resolution(pdb_path)
            for mi, model_lines in enumerate(model_blocks, start=1):
                mpath = Path(tmpdir.name) / f"{base_id}_model{mi}.pdb"
                mpath.write_text("".join(header_lines + model_lines + ["END\n"]))
                model_inputs.append((mi, load_atoms_auto(str(mpath), first_model_only=False), resolution))
        else:
            atoms = load_atoms_auto(pdb_path, first_model_only=config.first_model_only)
            model_inputs.append((None, atoms, parse_pdb_resolution(pdb_path)))

        n_with_atoms = 0
        for model, raw_atoms, resolution in model_inputs:
            if not raw_atoms:
                result.rejections["no_atoms"] += 1
                continue
            if config.resolution_cutoff is not None:
                if resolution is None:
                    result.rejections["resolution_na"] += 1
                    continue
                if resolution > config.resolution_cutoff:
                    result.rejections["resolution_above_cutoff"] += 1
                    continue
            n_with_atoms += 1
            sites, rej = _extract_from_atoms(
                raw_atoms=raw_atoms, resolution=resolution, element=element_up,
                config=config, pdb_id=base_id, model=model, must_have=must_have, logger=logger,
            )
            result.sites.extend(sites)
            result.rejections.update(rej)

        result.stats["n_models"] = len(model_inputs)
        result.stats["n_sites"] = len(result.sites)
        result.stats["n_clusters"] = len({(s.model, s.cluster_id) for s in result.sites})
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()

    return result
