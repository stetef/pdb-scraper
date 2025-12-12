#!/usr/bin/env python3
"""Constants for PDB Phase 1 pipeline."""

from typing import Set

# HTTP User-Agent for RCSB downloads
USER_AGENT = "pdb-env-tool/1.0 (+https://rcsb.org)"

# Comprehensive set of all metals that can appear in PDB files
# Includes: alkali, alkaline earth, transition, post-transition, 
# lanthanides, and actinides
ALL_METALS: Set[str] = {
    # Alkali metals (Group 1)
    "LI", "NA", "K", "RB", "CS",
    
    # Alkaline earth metals (Group 2)
    "MG", "CA", "SR", "BA",
    
    # Transition metals (Groups 3-12)
    "SC", "Y", "TI", "ZR", "HF",
    "V", "NB", "TA",
    "CR", "MO", "W",
    "MN", "TC", "RE",
    "FE", "CO", "NI", "CU", "ZN",
    "RU", "RH", "PD", "AG",
    "OS", "IR", "PT", "AU",
    "CD", "HG",
    
    # Post-transition metals
    "AL", "GA", "IN", "SN", "SB", "PB",
    
    # Lanthanides
    "LA", "CE", "PR", "ND", "SM", "EU", 
    "GD", "TB", "DY", "HO", "ER", "TM", "YB", "LU",
    
    # Actinides (common in structures)
    "U", "TH"
}

# Default file/directory names
DEFAULT_DOWNLOAD_DIR = "PDB"
DEFAULT_OUTPUT_DIR = "pdb_env_outputs"
DEFAULT_CACHE_FILE = "phase1_cache.json"
DEFAULT_LOG_FILE = "pdb_env_outputs/pipeline.log"

# Default processing parameters
DEFAULT_CUTOFF = 5.0
DEFAULT_TARGET = "NI"
DEFAULT_INCLUDE_WATERS = True

# PDB format specifications
PDB_RECORD_COLUMNS = {
    "record": (0, 6),
    "serial": (6, 11),
    "atom_name": (12, 16),
    "altloc": (16, 17),
    "resname": (17, 20),
    "chain": (21, 22),
    "resseq": (22, 26),
    "icode": (26, 27),
    "x": (30, 38),
    "y": (38, 46),
    "z": (46, 54),
    "occupancy": (54, 60),
    "bfactor": (60, 66),
    "element": (76, 78),
    "charge": (78, 80),
}

# Water residue names
WATER_RESIDUES = {"HOH", "WAT"}

# CSV field names
CLUSTERS_CSV_FIELDS = [
    "PDB", "CLUSTER", "CLUSTER_TYPE", "CENTER_IDX", 
    "CENTER_ELEM", "CENTER_ATOMNAME", "CHAIN", "RESSEQ", 
    "ICODE", "RESNAME", "ALTLOC", "OCC",
    "CN", "GEOM", "COORD", 
    "RMS_theta", "MAX_angle_dev", "sigma_d", "delta_d", 
    "RMS_plane", "h_max",
    "planar", "axial", "distorted", "JT",
    "OTHER_METALS", "ALTLOC_CASE", "ALTLOC_LABEL", 
    "XYZ_PATH", "RESOLUTION_A"
]

ALTLOC_REPORT_FIELDS = [
    "pdb_id", "record", "serial", "atom_name", "element", 
    "altloc", "resname", "chain", "resseq", 
    "x", "y", "z", "cluster_id", "cluster_type"
]

# Geometry classification thresholds
GEOMETRY_THRESHOLDS = {
    "angle_good": 12.0,      # degrees
    "angle_border": 20.0,    # degrees
    "planarity_good": 0.35,  # Angstroms RMS
    "axial_min": 160.0,      # degrees
    "jt_delta_d": 0.2,       # Angstroms
    "distortion_rms": 20.0,  # degrees
    "distortion_sigma": 0.15 # Angstroms
}