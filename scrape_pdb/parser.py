#!/usr/bin/env python3
"""PDB parsing functions."""

from typing import Optional, Iterable
from collections import Counter
import re
import os
from pathlib import Path

from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from .altloc import group_by_id, build_altloc_files_for_center
from .models import Atom
from .constants import ALL_METALS
from .cluster import determine_cluster_type, select_neighbors_union, select_neighbors_from, apply_water_toggle, select_coordinating_neighbors
from .utils import _slice, connected_components, centroid
from .config import PipelineConfig
from .writer import ensure_csv_headers, write_altloc_report_header, write_clusters_csv_row, append_altloc_rows, write_xyz
from .geometry import classify_geometry, coord_string
from .validation import passes_ligand_requirements, diagnose_ligand_requirements

import logging


logger = logging.getLogger("pipeline.parser")


def detect_file_format(path: str) -> str:
    """Detect if file is PDB or CIF format based on extension and content."""
    ext = os.path.splitext(path)[1].lower()
    
    if ext in ('.cif', '.mmcif'):
        return 'cif'
    elif ext == '.pdb':
        return 'pdb'
    else:
        # Fallback: peek at first line
        try:
            with open(path, 'r') as f:
                first_line = f.readline().strip()
                if first_line.startswith('data_'):
                    return 'cif'
                else:
                    return 'pdb'
        except Exception:
            return 'pdb'  # default assumption


def parse_cif_atom_line(cif_dict: dict, idx: int) -> Optional[Atom]:
    """Parse a single atom from mmCIF data structure."""
    # mmCIF uses different column names
    # You'll need to map CIF columns to your Atom model
    try:
        group_PDB = cif_dict.get('_atom_site.group_PDB', [])[idx]  # ATOM or HETATM
        serial = int(cif_dict.get('_atom_site.id', [])[idx])
        atom_name = cif_dict.get('_atom_site.label_atom_id', [])[idx]
        altloc = cif_dict.get('_atom_site.label_alt_id', [''])[idx]
        if altloc == '.':
            altloc = ''
        resname = cif_dict.get('_atom_site.label_comp_id', [])[idx]
        chain = cif_dict.get('_atom_site.label_asym_id', [])[idx]
        resseq = cif_dict.get('_atom_site.label_seq_id', [])[idx]
        icode = cif_dict.get('_atom_site.pdbx_PDB_ins_code', [''])[idx]
        if icode == '?':
            icode = ''
        
        x = float(cif_dict.get('_atom_site.Cartn_x', [])[idx])
        y = float(cif_dict.get('_atom_site.Cartn_y', [])[idx])
        z = float(cif_dict.get('_atom_site.Cartn_z', [])[idx])
        
        occ_str = cif_dict.get('_atom_site.occupancy', ['1.0'])[idx]
        occ = float(occ_str) if occ_str not in ('.', '?') else 1.0
        
        bfac_str = cif_dict.get('_atom_site.B_iso_or_equiv', ['0.0'])[idx]
        bfac = float(bfac_str) if bfac_str not in ('.', '?') else 0.0
        
        element = cif_dict.get('_atom_site.type_symbol', [])[idx].upper()
        charge = cif_dict.get('_atom_site.pdbx_formal_charge', [''])[idx]
        
        # Create a fake "line" for compatibility
        line = f"{group_PDB:<6}{serial:>5} {atom_name:<4}{altloc:1}{resname:>3} {chain:1}{resseq:>4}{icode:1}   {x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}          {element:>2}{charge:>2}"
        
        return Atom(group_PDB, serial, atom_name, altloc, resname, chain, resseq, icode,
                   x, y, z, occ, bfac, element, charge, line)
    except (IndexError, ValueError, KeyError) as e:
        logger.debug(f"Failed to parse CIF atom at index {idx}: {e}")
        return None


def load_cif_atoms_all(path: str, first_model_only: bool = False) -> list[Atom]:
    """Load all atoms from mmCIF file."""
    try:
        # Use CIF parser from BioPython
        cif_dict = MMCIF2Dict(path)
        
        atoms: list[Atom] = []
        atom_count = len(cif_dict.get('_atom_site.id', []))
        
        for idx in range(atom_count):
            atom = parse_cif_atom_line(cif_dict, idx)
            if atom:
                atoms.append(atom)
        
        return atoms
        
    except ImportError:
        logger.error("BioPython not installed. Install with: pip install biopython")
        return []
    except Exception as e:
        logger.error(f"Failed to parse CIF file {path}: {e}")
        return []


def load_atoms_auto(path: str, first_model_only: bool = False) -> list[Atom]:
    """Auto-detect format and load atoms."""
    fmt = detect_file_format(path)
    
    if fmt == 'cif':
        logger.info(f"Detected CIF format for {path}")
        return load_cif_atoms_all(path, first_model_only)
    else:
        return load_pdb_atoms_all(path, first_model_only)


def parse_pdb_atom_line(line: str) -> Optional[Atom]:
    if len(line) < 6:
        return None
    rec = line[0:6].strip().upper()
    if rec not in ("ATOM", "HETATM"):
        return None
    if len(line) < 80:
        line = line.rstrip("\n")
        line = line + (" " * (80 - len(line)))
    serial_s   = _slice(line, 6, 11).strip()
    atom_name  = _slice(line,12, 16).strip()
    altloc     = _slice(line,16, 17).strip()
    resname    = _slice(line,17, 20).strip()
    chain      = _slice(line,21, 22).strip()
    resseq     = _slice(line,22, 26).strip()
    icode      = _slice(line,26, 27).strip()
    x_s        = _slice(line,30, 38).strip()
    y_s        = _slice(line,38, 46).strip()
    z_s        = _slice(line,46, 54).strip()
    occ_s      = _slice(line,54, 60).strip()  # occupancy often 55–60, but some files shift; use 54:60
    bfac_s     = _slice(line,60, 66).strip()
    element    = _slice(line,76, 78).strip()
    charge     = _slice(line,78, 80).strip()

    try:
        serial = int(serial_s)
    except Exception:
        serial = 0
    try:
        x = float(x_s); y = float(y_s); z = float(z_s)
    except Exception:
        return None
    try:
        occ = float(occ_s) if occ_s else 1.0
    except Exception:
        occ = 1.0
    try:
        bfac = float(bfac_s) if bfac_s else 0.0
    except Exception:
        bfac = 0.0

    if not element:
        nm = atom_name.strip()
        # Infer per PDB rule: if starts with letter, take first 1–2 letters, else skip digits
        if len(nm) >= 2 and nm[0].isalpha() and nm[1].islower():
            element = nm[:2].title()
        else:
            element = (nm[0] if nm and nm[0].isalpha() else nm[:1]).upper()
    else:
        element = element.upper()

    return Atom(rec, serial, atom_name, altloc, resname, chain, resseq, icode,
                x, y, z, occ, bfac, element, charge, line.rstrip("\n"))


def load_pdb_atoms_all(path: str, first_model_only: bool = False) -> list[Atom]:
    """Load all ATOM/HETATM lines from file, no altloc collapsing (raw list)."""
    atoms: list[Atom] = []
    try:
        with open(path, "r") as f:
            in_model = True
            current_model = None
            for line in f:
                rec6 = (line[0:6].strip().upper() if len(line) >= 6 else "")
                if rec6 == "MODEL":
                    try:
                        current_model = int(line[10:14].strip())
                    except Exception:
                        current_model = 1
                    in_model = (current_model == 1) if first_model_only else True
                elif rec6 == "ENDMDL" and first_model_only:
                    break
                elif rec6 in ("ATOM", "HETATM"):
                    if not in_model:
                        continue
                    a = parse_pdb_atom_line(line)
                    if a:
                        atoms.append(a)
    except FileNotFoundError:
        logger.info(f"[!] File not found: {path}")
        return []
    return atoms

def collapse_altloc(atoms: list[Atom], policy: str = "prefer_blank_or_A") -> list[Atom]:
    """Collapse altloc groups by (chain, resseq, icode, atom_name)."""
    from collections import defaultdict
    groups: dict[tuple[str,str,str,str], list[Atom]] = defaultdict(list)
    for a in atoms:
        key = (a.chain, a.resseq, a.icode, a.atom_name)
        groups[key].append(a)
    out: list[Atom] = []
    for key, alts in groups.items():
        if len(alts) == 1:
            out.append(alts[0])
            continue
        if policy == "prefer_blank_or_A":
            chosen = None
            for a in alts:
                if a.altloc in ("", "A"):
                    chosen = a
                    break
            if chosen is None:
                chosen = max(alts, key=lambda x: x.occ)
            out.append(chosen)
        elif policy == "best_occupancy":
            out.append(max(alts, key=lambda x: x.occ))
        else:
            out.append(alts[0])
    return out

def parse_pdb_resolution(path: str) -> Optional[float]:
    """Parse REMARK 2 RESOLUTION line before first ATOM/HETATM."""
    try:
        with open(path, "r") as f:
            for line in f:
                if len(line) >= 6 and line[0:6].strip().upper() in ("ATOM", "HETATM"):
                    break
                if line.startswith("REMARK   2") and "RESOLUTION." in line.upper():
                    if "NOT APPLICABLE" in line.upper():
                        return None
                    m = re.search(r"RESOLUTION\.\s*([0-9]+(?:\.[0-9]+)?)\s+ANGSTROM", line, re.IGNORECASE)
                    if m:
                        return float(m.group(1))
        return None
    except Exception:
        return None

def iter_altloc_metal_records(pdb_path: str, metal_set: set[str]) -> Iterable[Atom]:
    """Yield HETATM atoms from raw file that have a non-blank altloc and are metals (by element)."""
    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            altc = _slice(line,16,17).strip()
            if altc in ("", "A"):
                continue
            a = parse_pdb_atom_line(line)
            if not a:
                continue
            if a.element.upper() in metal_set:
                yield a

# ------------------------------
# Main per-PDB processor
# ------------------------------

def process_pdb(
    pdb_path: str,
    config: PipelineConfig
) -> tuple[list[str], dict]:
    """
    Process a single PDB file.
    
    Args:
        pdb_path: Path to PDB file
        config: Pipeline configuration
        logger: Logger instance
    
    Returns:
        (List of written XYZ file paths, stats dict)
    """
    base_id = os.path.splitext(os.path.basename(pdb_path))[0]
    logger.info(f"[=] Processing {base_id} …")

    stats: dict = {
        "pdb_id": base_id,
        "rejections": Counter(),
        "ligand_requirements": None,
        "rejected_cn_distribution": Counter(),
        "rejected_coord_distribution": Counter(),
    }

    raw_atoms = load_atoms_auto(pdb_path, first_model_only=False)
    if not raw_atoms:
        logger.info(f"[!] No atoms parsed in {pdb_path}")
        stats["rejections"]["no_atoms"] += 1
        return [], stats
    atoms = collapse_altloc(raw_atoms, policy="prefer_blank_or_A")

    # Resolution
    resolution_angs = parse_pdb_resolution(pdb_path)
    
    # Filter out structures with NA resolution if resolution_cutoff is specified
    if config.search_parameters and config.search_parameters.resolution_cutoff is not None:
        if resolution_angs is None:
            logger.info(f"  [i] Skipping {base_id}: resolution is NA but resolution_cutoff={config.search_parameters.resolution_cutoff} Å is specified")
            stats["rejections"]["resolution_na"] += 1
            return [], stats
        if resolution_angs > config.search_parameters.resolution_cutoff:
            logger.info(f"  [i] Skipping {base_id}: resolution {resolution_angs:.2f} Å exceeds cutoff {config.search_parameters.resolution_cutoff} Å")
            stats["rejections"]["resolution_above_cutoff"] += 1
            return [], stats

    # Metals universe (after exclusions)
    # metal_set = set(ALL_METALS) - config.metals_excluded
    metal_set = set(ALL_METALS) - {m.upper() for m in config.metals_excluded}

    # Build metal list (HETATM only)
    metals_all = [a for a in atoms if a.record == "HETATM" and a.element.upper() in metal_set]
    if not metals_all:
        logger.info("  [i] No metals found (after exclusions).")
        stats["rejections"]["no_metals"] += 1
    target_upper = config.target.upper()
    # Determine target element from first target occurrence if present
    targ_elems = [a.element.upper() for a in atoms if a.record == "HETATM" and a.atom_name.upper() == target_upper]
    target_element = (targ_elems[0] if targ_elems else None)

    # Build metal graph components (union-of-spheres on ALL metals, not just target)
    points = [m.coord for m in metals_all]
    comps = connected_components(points, config.cutoff)
    logger.info(f"  [i] Metals: {len(metals_all)}; components within {config.cutoff:.2f} Å: {len(comps)}")

    # Prepare outputs
    os.makedirs(config.output_dir, exist_ok=True)
    write_altloc_report_header(config)
    ensure_csv_headers(config)

    # Group raw atoms by id for altloc logic
    raw_groups = group_by_id(raw_atoms)

    written_paths: list[str] = []
    cluster_counter = 0

    # Prepare ligand-requirements aggregation (if configured)
    if config.validation.ligand_requirements:
        stats["ligand_requirements"] = {
            "per_req": [
                {
                    "missing": 0,
                    "naming_mismatch": 0,
                    "seen_resnames_for_atom_names": Counter(),
                }
                for _ in config.validation.ligand_requirements
            ]
        }

    found_target_component = False

    for comp in comps:
        comp_metals = [metals_all[i] for i in comp]
        # Component must contain at least one target atom by atom_name
        if not any(m.atom_name.upper() == target_upper for m in comp_metals):
            continue
        found_target_component = True

        cluster_counter += 1
        comp_type = determine_cluster_type(comp_metals, target_upper, target_element)
        centers = comp_metals  # metals defining union-of-spheres
        c_centroid = centroid([c.coord for c in centers])

        # Build selection universe from atoms (non-H; waters default included, soft exclusion later)
        # For union: neighbors are atoms within selection_radius of ANY center metal
        selected_union = select_neighbors_union(atoms, centers, config.selection_radius)

        # Create a base name common (without altloc tag)
        base_name_common = f"{base_id}_{target_upper}_{comp_type}_d{config.cutoff:.2f}_cluster{cluster_counter}"

        # ALTLOC splitting around the chosen "center": pick the first target occurrence in this comp
        center_candidates = [m for m in comp_metals if m.atom_name.upper() == target_upper]
        if not center_candidates:
            center_for_alt = comp_metals[0]
        else:
            center_for_alt = center_candidates[0]

        # Attempt Step 6 generation
        alt_written = build_altloc_files_for_center(
            pdb_id=base_id,
            center=center_for_alt,
            selected_atoms_raw=selected_union,
            raw_groups=raw_groups,
            cutoff=config.selection_radius,
            out_dir=config.output_dir,
            base_name_common=base_name_common,
            origin_kind=("centroid" if len(centers)>1 else "single_center"),
            centroid_pt=c_centroid,
            resolution_angs=resolution_angs,
            cluster_index=cluster_counter,
            cluster_type=comp_type,
            target_upper=target_upper,
            include_waters=config.include_waters,
            must_have=config.must_have,
            coord_distance_min=config.validation.coordination_distance_min,
            coord_distance_max=config.validation.coordination_distance_max,
            coord_filters=config.coord_filters,
            ligand_requirements=config.validation.ligand_requirements,
        )

        if alt_written:
            # For each altloc file written, we still need to append CSV rows (geometry etc.)
            for path in alt_written:
                # Geometry & CSV per Step 10 rules
                # Re-read the just-written XYZ back? Instead, compute geometry directly here:
                # For multi_homo: one row per center; else one per cluster (target metal only)
                # Build neighbor sets for geometry per-center
                if comp_type == "multi_homo":
                    for idx_c, c in enumerate(centers, start=1):
                        # Get all neighbors within selection_radius for must_have filter
                        neigh_all = select_neighbors_from(c, selected_union, config.selection_radius)
                        neigh_all = apply_water_toggle(neigh_all, include_waters=config.include_waters)
                        # Get coordinating neighbors for geometry and coord filter
                        coord_neigh = select_coordinating_neighbors(c, selected_union,
                                                                    config.validation.coordination_distance_min,
                                                                    config.validation.coordination_distance_max)
                        coord_neigh = apply_water_toggle(coord_neigh, include_waters=config.include_waters)
                        cn_diag = len(coord_neigh)
                        coord_diag = coord_string(coord_neigh)
                        # Must-have filter on all cluster atoms
                        if not config.must_have.passes(neigh_all):
                            stats["rejections"]["must_have"] += 1
                            stats["rejected_cn_distribution"][cn_diag] += 1
                            stats["rejected_coord_distribution"][coord_diag] += 1
                            continue
                        if not passes_ligand_requirements(coord_neigh, config.validation.ligand_requirements):
                            stats["rejections"]["ligand_requirements"] += 1
                            stats["rejected_cn_distribution"][cn_diag] += 1
                            stats["rejected_coord_distribution"][coord_diag] += 1
                            if stats.get("ligand_requirements") and config.validation.ligand_requirements:
                                diag = diagnose_ligand_requirements(coord_neigh, config.validation.ligand_requirements)
                                for d in diag:
                                    i = d["req_index"]
                                    if d["count_allowed"] < d["min_count"]:
                                        stats["ligand_requirements"]["per_req"][i]["missing"] += 1
                                        if d["count_any_resname"] >= d["min_count"]:
                                            stats["ligand_requirements"]["per_req"][i]["naming_mismatch"] += 1
                                    stats["ligand_requirements"]["per_req"][i]["seen_resnames_for_atom_names"].update(
                                        {k: int(v) for k, v in d["resname_counts_for_atom_names"].items()}
                                    )
                            try:
                                Path(path).unlink(missing_ok=True)
                            except Exception:
                                pass
                            continue
                        geom, metrics, flags = classify_geometry(c, coord_neigh)
                        coord = coord_string(coord_neigh)
                        if config.coord_filters and coord not in config.coord_filters:
                            stats["rejections"]["coord_string"] += 1
                            try:
                                Path(path).unlink(missing_ok=True)
                            except Exception:
                                pass
                            continue
                        other_metals = ""
                        write_clusters_csv_row([
                            base_id, cluster_counter, comp_type, idx_c,
                            c.element.upper(), c.atom_name.upper(), c.chain, c.resseq, c.icode, c.resname, c.altloc, c.occ,
                            metrics["CN"], geom, coord, metrics["RMS_theta"], metrics["MAX_angle_dev"],
                            metrics["sigma_d"], metrics["delta_d"], metrics["RMS_plane"], metrics["h_max"],
                            flags["planar"], flags["axial"], flags["distorted"], flags["JT"],
                            other_metals,  # OTHER_METALS
                            "ALTLOC", "", path, (f"{resolution_angs:.2f}" if isinstance(resolution_angs,(int,float)) else "NA")
                        ], config)
                else:
                    # homo or multi_hetero → one row per cluster focused on the target metal
                    c = center_for_alt
                    # Get all neighbors within selection_radius for must_have filter
                    neigh_all = select_neighbors_from(c, selected_union, config.selection_radius)
                    neigh_all = apply_water_toggle(neigh_all, include_waters=config.include_waters)
                    
                    # Always compute coordination info for diagnostics before filtering
                    coord_neigh_diag = select_coordinating_neighbors(c, selected_union,
                                                                config.validation.coordination_distance_min,
                                                                config.validation.coordination_distance_max)
                    coord_neigh_diag = apply_water_toggle(coord_neigh_diag, include_waters=config.include_waters)
                    cn_diag = len(coord_neigh_diag)
                    coord_diag = coord_string(coord_neigh_diag)
                    
                    if not config.must_have.passes(neigh_all):
                        stats["rejections"]["must_have"] += 1
                        stats["rejected_cn_distribution"][cn_diag] += 1
                        stats["rejected_coord_distribution"][coord_diag] += 1
                        continue
                    # Get coordinating neighbors for geometry and coord filter
                    coord_neigh = select_coordinating_neighbors(c, selected_union,
                                                                config.validation.coordination_distance_min,
                                                                config.validation.coordination_distance_max)
                    if not passes_ligand_requirements(coord_neigh, config.validation.ligand_requirements):
                        stats["rejections"]["ligand_requirements"] += 1
                        stats["rejected_cn_distribution"][cn_diag] += 1
                        stats["rejected_coord_distribution"][coord_diag] += 1
                        stats["rejected_cn_distribution"][cn_diag] += 1
                        stats["rejected_coord_distribution"][coord_diag] += 1
                        if stats.get("ligand_requirements") and config.validation.ligand_requirements:
                            diag = diagnose_ligand_requirements(coord_neigh, config.validation.ligand_requirements)
                            for d in diag:
                                i = d["req_index"]
                                if d["count_allowed"] < d["min_count"]:
                                    stats["ligand_requirements"]["per_req"][i]["missing"] += 1
                                    if d["count_any_resname"] >= d["min_count"]:
                                        stats["ligand_requirements"]["per_req"][i]["naming_mismatch"] += 1
                                stats["ligand_requirements"]["per_req"][i]["seen_resnames_for_atom_names"].update(
                                    {k: int(v) for k, v in d["resname_counts_for_atom_names"].items()}
                                )
                        try:
                            Path(path).unlink(missing_ok=True)
                        except Exception:
                            pass
                        continue
                    geom, metrics, flags = classify_geometry(c, coord_neigh)
                    coord = coord_string(coord_neigh)
                    if config.coord_filters and coord not in config.coord_filters:
                        stats["rejections"]["coord_string"] += 1
                        try:
                            Path(path).unlink(missing_ok=True)
                        except Exception:
                            pass
                        continue
                    # OTHER_METALS for multi_hetero
                    other = ""
                    if comp_type == "multi_hetero":
                        metals_other = [m.element.upper() for m in centers if m.serial != c.serial]
                        cc = Counter(metals_other)
                        other = ", ".join(f"{k}:{v}" for k,v in sorted(cc.items()))
                    write_clusters_csv_row([
                        base_id, cluster_counter, comp_type, 1,
                        c.element.upper(), c.atom_name.upper(), c.chain, c.resseq, c.icode, c.resname, c.altloc, c.occ,
                        metrics["CN"], geom, coord, metrics["RMS_theta"], metrics["MAX_angle_dev"],
                        metrics["sigma_d"], metrics["delta_d"], metrics["RMS_plane"], metrics["h_max"],
                        flags["planar"], flags["axial"], flags["distorted"], flags["JT"],
                        other, "ALTLOC", "", path, (f"{resolution_angs:.2f}" if isinstance(resolution_angs,(int,float)) else "NA")
                    ], config)
            # Only keep paths that survived coord filtering
            for p in alt_written:
                if Path(p).exists():
                    written_paths.append(p)
        else:
            # No altloc case → write a single base file
            selected = apply_water_toggle(selected_union, include_waters=config.include_waters)
            # Must-have: for homo/multi_hetero apply to target metal; for multi_homo per-center rows below
            # For writing XYZ we use origin = first target center
            origin_atom = center_for_alt
            xyz_name = f"{base_name_common}.xyz"
            xyz_path = str(Path(config.output_dir) / "xyz_files" / xyz_name)
            extra = f"CLUSTER_TYPE={comp_type}"

            # CSV rows
            if comp_type == "multi_homo":
                wrote_xyz = False
                for idx_c, c in enumerate(centers, start=1):
                    # Get all neighbors within selection_radius for must_have filter
                    neigh_all = select_neighbors_from(c, selected, config.selection_radius)
                    # Get coordinating neighbors for geometry and coord filter
                    coord_neigh = select_coordinating_neighbors(c, selected,
                                                                config.validation.coordination_distance_min,
                                                                config.validation.coordination_distance_max)
                    cn_diag = len(coord_neigh)
                    coord_diag = coord_string(coord_neigh)
                    
                    if not config.must_have.passes(neigh_all):
                        stats["rejections"]["must_have"] += 1
                        stats["rejected_cn_distribution"][cn_diag] += 1
                        stats["rejected_coord_distribution"][coord_diag] += 1
                        continue
                    if not passes_ligand_requirements(coord_neigh, config.validation.ligand_requirements):
                        stats["rejections"]["ligand_requirements"] += 1
                        stats["rejected_cn_distribution"][cn_diag] += 1
                        stats["rejected_coord_distribution"][coord_diag] += 1
                        if stats.get("ligand_requirements") and config.validation.ligand_requirements:
                            diag = diagnose_ligand_requirements(coord_neigh, config.validation.ligand_requirements)
                            for d in diag:
                                i = d["req_index"]
                                if d["count_allowed"] < d["min_count"]:
                                    stats["ligand_requirements"]["per_req"][i]["missing"] += 1
                                    if d["count_any_resname"] >= d["min_count"]:
                                        stats["ligand_requirements"]["per_req"][i]["naming_mismatch"] += 1
                                stats["ligand_requirements"]["per_req"][i]["seen_resnames_for_atom_names"].update(
                                    {k: int(v) for k, v in d["resname_counts_for_atom_names"].items()}
                                )
                        continue
                    geom, metrics, flags = classify_geometry(c, coord_neigh)
                    coord = coord_string(coord_neigh)
                    if config.coord_filters and coord not in config.coord_filters:
                        stats["rejections"]["coord_string"] += 1
                        continue
                    if not wrote_xyz:
                        write_xyz(xyz_path, base_id, cluster_counter, target_upper, config.selection_radius,
                                  ("centroid" if len(centers)>1 else "single_center"), c_centroid,
                                  selected, origin_atom, resolution_angs, extra_comment=extra)
                        written_paths.append(xyz_path)
                        wrote_xyz = True
                    write_clusters_csv_row([
                        base_id, cluster_counter, comp_type, idx_c,
                        c.element.upper(), c.atom_name.upper(), c.chain, c.resseq, c.icode, c.resname, c.altloc, c.occ,
                        metrics["CN"], geom, coord, metrics["RMS_theta"], metrics["MAX_angle_dev"],
                        metrics["sigma_d"], metrics["delta_d"], metrics["RMS_plane"], metrics["h_max"],
                        flags["planar"], flags["axial"], flags["distorted"], flags["JT"],
                        "", "", "", xyz_path, (f"{resolution_angs:.2f}" if isinstance(resolution_angs,(int,float)) else "NA")
                    ], config)
            else:
                c = center_for_alt
                # Get all neighbors within selection_radius for must_have filter
                neigh_all = select_neighbors_from(c, selected, config.selection_radius)
                # Get coordinating neighbors for geometry and coord filter
                coord_neigh = select_coordinating_neighbors(c, selected,
                                                            config.validation.coordination_distance_min,
                                                            config.validation.coordination_distance_max)
                cn_diag = len(coord_neigh)
                coord_diag = coord_string(coord_neigh)
                
                if not config.must_have.passes(neigh_all):
                    stats["rejections"]["must_have"] += 1
                    stats["rejected_cn_distribution"][cn_diag] += 1
                    stats["rejected_coord_distribution"][coord_diag] += 1
                    continue
                if not passes_ligand_requirements(coord_neigh, config.validation.ligand_requirements):
                    stats["rejections"]["ligand_requirements"] += 1
                    stats["rejected_cn_distribution"][cn_diag] += 1
                    stats["rejected_coord_distribution"][coord_diag] += 1
                    if stats.get("ligand_requirements") and config.validation.ligand_requirements:
                        diag = diagnose_ligand_requirements(coord_neigh, config.validation.ligand_requirements)
                        for d in diag:
                            i = d["req_index"]
                            if d["count_allowed"] < d["min_count"]:
                                stats["ligand_requirements"]["per_req"][i]["missing"] += 1
                                if d["count_any_resname"] >= d["min_count"]:
                                    stats["ligand_requirements"]["per_req"][i]["naming_mismatch"] += 1
                            stats["ligand_requirements"]["per_req"][i]["seen_resnames_for_atom_names"].update(
                                {k: int(v) for k, v in d["resname_counts_for_atom_names"].items()}
                            )
                    continue
                geom, metrics, flags = classify_geometry(c, coord_neigh)
                coord = coord_string(coord_neigh)
                if config.coord_filters and coord not in config.coord_filters:
                    stats["rejections"]["coord_string"] += 1
                    continue
                write_xyz(xyz_path, base_id, cluster_counter, target_upper, config.selection_radius,
                          ("centroid" if len(centers)>1 else "single_center"), c_centroid,
                          selected, origin_atom, resolution_angs, extra_comment=extra)
                written_paths.append(xyz_path)
                other = ""
                if comp_type == "multi_hetero":
                    metals_other = [m.element.upper() for m in centers if m.serial != c.serial]
                    cc = Counter(metals_other)
                    other = ", ".join(f"{k}:{v}" for k,v in sorted(cc.items()))
                write_clusters_csv_row([
                    base_id, cluster_counter, comp_type, 1,
                    c.element.upper(), c.atom_name.upper(), c.chain, c.resseq, c.icode, c.resname, c.altloc, c.occ,
                    metrics["CN"], geom, coord, metrics["RMS_theta"], metrics["MAX_angle_dev"],
                    metrics["sigma_d"], metrics["delta_d"], metrics["RMS_plane"], metrics["h_max"],
                    flags["planar"], flags["axial"], flags["distorted"], flags["JT"],
                    other, "", "", xyz_path, (f"{resolution_angs:.2f}" if isinstance(resolution_angs,(int,float)) else "NA")
                ], config)

        # AltLoc metals report within vicinity
        alt_rows: list[list] = []
        # Build predicate: within union-of-spheres around centers
        ccoords = [c.coord for c in centers]
        def in_union_xyz(x: float, y: float, z: float) -> bool:
            for cx, cy, cz in ccoords:
                dx = x - cx; dy = y - cy; dz = z - cz
                if dx*dx + dy*dy + dz*dz <= config.selection_radius*config.selection_radius:
                    return True
            return False
        for a in iter_altloc_metal_records(pdb_path, metal_set):
            if in_union_xyz(a.x, a.y, a.z):
                alt_rows.append([base_id, a.record, a.serial, a.atom_name, a.element, a.altloc,
                                 a.resname, a.chain, a.resseq, a.x, a.y, a.z, cluster_counter, comp_type])
        append_altloc_rows(alt_rows, config)

    if not found_target_component:
        stats["rejections"]["no_target_component"] += 1

    # Convert Counters to plain dicts for JSON-serializable consumption in caller.
    stats["rejections"] = dict(stats["rejections"])
    if stats.get("ligand_requirements"):
        for per in stats["ligand_requirements"]["per_req"]:
            per["seen_resnames_for_atom_names"] = dict(per["seen_resnames_for_atom_names"])

    return written_paths, stats