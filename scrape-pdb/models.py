#!/usr/bin/env python3
"""Data models for PDB Phase 1 pipeline."""

import re
from dataclasses import dataclass
from typing import Tuple, List, Dict, Set, Optional
from collections import Counter


# ============================================================================
# ATOM DATA MODEL
# ============================================================================

@dataclass
class Atom:
    """Represents a single atom from PDB file."""
    record: str      # "ATOM" or "HETATM"
    serial: int
    atom_name: str
    altloc: str
    resname: str
    chain: str
    resseq: str
    icode: str
    x: float
    y: float
    z: float
    occ: float
    bfac: float
    element: str
    charge: str
    raw_line: str

    @property
    def coord(self) -> Tuple[float, float, float]:
        """Get coordinates as tuple."""
        return (self.x, self.y, self.z)
    
    @property
    def residue_id(self) -> Tuple[str, str, str]:
        """Get unique residue identifier (chain, resseq, icode)."""
        return (self.chain, self.resseq, self.icode)
    
    @property
    def atom_id(self) -> Tuple[str, str, str, str]:
        """Get unique atom identifier for altloc grouping."""
        return (self.chain, self.resseq, self.icode, self.atom_name)


# ============================================================================
# MUST-HAVE ELEMENTS FILTER (Step 11)
# ============================================================================

class MustHaveSpec:
    """
    Specification for required elements in clusters.
    
    Implements Step 11 of the specification: optional filter to require
    presence of specific elements in coordination sphere.
    """
    
    def __init__(
        self, 
        mode: str = "ANY", 
        elems: Optional[Set[str]] = None, 
        min_counts: Optional[Dict[str, int]] = None
    ):
        """
        Initialize must-have specification.
        
        Args:
            mode: "ANY" (at least one required) or "ALL" (all required)
            elems: Set of element symbols
            min_counts: Dict of element -> minimum count
        """
        self.mode = mode
        self.elems = set(e.upper() for e in (elems or []))
        self.min_counts = {k.upper(): int(v) for k, v in (min_counts or {}).items()}
    
    def passes(self, neighbors: List[Atom]) -> bool:
        """
        Check if neighbor list satisfies requirements.
        
        Args:
            neighbors: List of neighbor atoms to check
        
        Returns:
            True if requirements satisfied, False otherwise
        """
        if not self.elems and not self.min_counts:
            return True
        
        # Count elements in neighbors
        cnt = Counter([n.element.upper() for n in neighbors])
        
        # Check min counts (ALL semantics - must have at least N of each)
        for elm, min_val in self.min_counts.items():
            if cnt.get(elm, 0) < min_val:
                return False
        
        # Check element presence
        if not self.elems:
            return True
        
        if self.mode == "ALL":
            # Must have at least 1 of ALL specified elements
            return all(cnt.get(e, 0) >= 1 for e in self.elems)
        else:  # ANY mode
            # Must have at least 1 of ANY specified element
            return any(cnt.get(e, 0) >= 1 for e in self.elems)
    
    def __str__(self) -> str:
        """String representation for logging."""
        parts = []
        if self.min_counts:
            parts.extend(f"{e}>={c}" for e, c in sorted(self.min_counts.items()))
        if self.elems:
            prefix = "ALL:" if self.mode == "ALL" and not self.min_counts else ""
            parts.append(prefix + ",".join(sorted(self.elems)))
        return " ".join(parts) if parts else "none"


def parse_must_have(text: str) -> MustHaveSpec:
    """
    Parse must-have specification from string.
    
    Supports multiple formats:
    - "S" -> ANY of S
    - "S,N" -> ANY of S or N  
    - "ALL:S,O" -> ALL of S and O
    - "S>=2,N>=1" -> at least 2 S and 1 N
    
    Args:
        text: Specification string
    
    Returns:
        MustHaveSpec instance
    
    Examples:
        >>> spec = parse_must_have("S,N")
        >>> spec.mode
        'ANY'
        >>> spec.elems
        {'S', 'N'}
        
        >>> spec = parse_must_have("S>=2,N>=1")
        >>> spec.min_counts
        {'S': 2, 'N': 1}
    """
    text = (text or "").strip()
    if not text:
        return MustHaveSpec()
    
    mode = "ANY"
    min_counts: Dict[str, int] = {}
    elems: Set[str] = set()
    
    # Check for ALL prefix
    if text.upper().startswith("ALL:"):
        mode = "ALL"
        text = text[4:]
    
    # Split on commas/spaces
    parts = [p.strip() for p in re.split(r"[,\s]+", text) if p.strip()]
    
    # Parse each part
    for part in parts:
        if ">=" in part:
            # Min count specification (e.g., "S>=2")
            match = re.match(r"^([A-Za-z]{1,2})\s*>=\s*([0-9]+)$", part)
            if match:
                elm = match.group(1).upper()
                count = int(match.group(2))
                min_counts[elm] = count
        elif re.match(r"^[A-Za-z]{1,2}$", part):
            # Simple element symbol (e.g., "S", "FE")
            elems.add(part.upper())
    
    return MustHaveSpec(mode=mode, elems=elems, min_counts=min_counts)


# ============================================================================
# GEOMETRY RESULTS
# ============================================================================

@dataclass
class GeometryResult:
    """
    Results from geometry classification (Step 10).
    
    Contains the classified geometry label, computed metrics,
    and boolean flags for special properties.
    """
    label: str                    # TD, OH, SQP, TBP, etc.
    metrics: Dict[str, float]     # CN, RMS_theta, sigma_d, etc.
    flags: Dict[str, str]         # planar, axial, distorted, JT
    coord_string: str             # e.g., "4N1O"
    
    @property
    def coordination_number(self) -> int:
        """Get coordination number."""
        return int(self.metrics.get("CN", 0))
    
    @property
    def is_distorted(self) -> bool:
        """Check if geometry is distorted."""
        return self.flags.get("distorted", "No") == "Yes"
    
    @property
    def is_jahn_teller(self) -> bool:
        """Check if Jahn-Teller distortion present (octahedral only)."""
        return self.flags.get("JT", "No") == "Yes"
    
    @property
    def is_planar(self) -> bool:
        """Check if coordination is planar."""
        return self.flags.get("planar", "No") == "Yes"
    
    @property
    def has_axial(self) -> bool:
        """Check if has axial ligands."""
        return self.flags.get("axial", "No") == "Yes"


# ============================================================================
# CLUSTER INFORMATION
# ============================================================================

@dataclass
class ClusterInfo:
    """
    Information about a detected metal cluster.
    
    Represents a connected component of metals within cutoff distance,
    classified as homo, multi_homo, or multi_hetero.
    """
    pdb_id: str
    cluster_index: int
    cluster_type: str          # homo, multi_homo, multi_hetero
    centers: List[Atom]        # Metal atoms in cluster
    cutoff: float
    resolution: Optional[float]
    
    @property
    def center_count(self) -> int:
        """Number of metal centers in cluster."""
        return len(self.centers)
    
    @property
    def centroid(self) -> Tuple[float, float, float]:
        """Calculate centroid of all metal centers."""
        if not self.centers:
            return (0.0, 0.0, 0.0)
        coords = [c.coord for c in self.centers]
        n = len(coords)
        return (
            sum(c[0] for c in coords) / n,
            sum(c[1] for c in coords) / n,
            sum(c[2] for c in coords) / n
        )
    
    @property
    def is_mononuclear(self) -> bool:
        """Check if cluster contains single metal."""
        return len(self.centers) == 1
    
    @property
    def is_polynuclear(self) -> bool:
        """Check if cluster contains multiple metals."""
        return len(self.centers) > 1


# ============================================================================
# HELPER FUNCTIONS FOR MODELS
# ============================================================================

def coord_string_from_atoms(atoms: List[Atom]) -> str:
    """
    Generate coordination string from list of atoms.
    
    Example: [N, N, N, N, O] -> "4N1O"
    
    Args:
        atoms: List of coordinating atoms
    
    Returns:
        Coordination string sorted alphabetically
    """
    cnt = Counter(a.element.upper() for a in atoms if a.element)
    parts = []
    for elm in sorted(cnt.keys()):
        # Proper case: Fe, Cl (2-letter) or N, O (1-letter)
        formatted = elm.title() if len(elm) == 2 else elm.upper()
        parts.append(f"{cnt[elm]}{formatted}")
    return "".join(parts)