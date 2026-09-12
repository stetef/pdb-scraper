#!/usr/bin/env python3
"""Library data model for the EARL structure-source API.

These are the objects the EARL app (and any other library caller) consumes.
They are deliberately kept separate from the CLI's ``PipelineConfig`` /
file-writing world so the extraction path can be a *pure* function of
(file, config) with no filesystem side effects.

See design doc ``04-pdb-scraper-library-spec.md`` §1–§2.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


class FetchError(Exception):
    """Raised by :func:`scrape_pdb.api.fetch_entry` for any download failure
    (bad id, not found, too large, timeout, network error).

    ``kind`` is a stable, machine-readable tag from the EARL error taxonomy
    (``not_found`` / ``too_large`` / ``timeout`` / ``invalid_id`` / ``network``)
    so the app can map it to a typed UI message without string-matching.
    """

    def __init__(self, message: str, *, kind: str = "network") -> None:
        super().__init__(message)
        self.kind = kind


@dataclass
class ExtractConfig:
    """The subset of the CLI's ``ProcessingConfig`` + ``ValidationConfig`` that
    affects *extraction*, plus the library-only knobs.

    Search / output / checkpoint / parallelism fields are intentionally absent:
    the library never searches, never writes shared files, and holds all state
    per-call.

    Defaults implement the "user typed a single PDB ID" policy agreed for EARL
    (design decision, 2026-09-11): **show every site of the element** — no
    ``must_have`` / coordination / ligand filtering — and let the user pick.
    Filters remain available for callers that want the old bulk-harvest
    behaviour.
    """

    # Where any transient files (e.g. per-model splits, optional H-addition)
    # may be written. Everything the library creates lives under here and is
    # cleaned up; inputs are never touched.
    workdir: Path

    # --- clustering / selection geometry ---
    # Metal-to-metal distance that groups atoms into one polynuclear cluster.
    # Must exceed the metal-metal separation you want treated as "one site":
    # e.g. the two Zn in 5XP6 sit 3.95 Å apart, so the default 5.0 groups them.
    cutoff: float = 5.0
    # Radius around each metal centre within which neighbour atoms are pulled in.
    selection_radius: float = 6.0

    # --- element / exclusions ---
    # Metals to ignore entirely when clustering (e.g. exclude structural Fe4S4
    # while targeting Ni). The absorber element itself is passed to
    # ``extract_sites`` separately and is never excluded.
    metals_excluded: tuple[str, ...] = ()

    # --- coordination window (first shell) ---
    coordination_distance_min: float = 2.0
    coordination_distance_max: float = 2.8

    # --- optional filters (OFF by default = "show all sites") ---
    # ``must_have`` accepts the same mini-language as the CLI (e.g. "S,N" or
    # "ALL:N,O" or "S>=2"); None/"" means no requirement.
    must_have: Optional[str] = None
    # Legacy exact coord-string whitelist, e.g. {"1N3S"}; None means no filter.
    coord: Optional[frozenset[str]] = None
    # Residue-aware requirements (list of dicts as accepted by the CLI's
    # ``LigandRequirement``); None means no requirement.
    ligand_requirements: Optional[list[dict]] = None

    # --- water / backbone / hydrogen conventions ---
    # Keep crystallographic waters in the site (they are real ~2.0–2.5 Å
    # scatterers). EARL surfaces this as an explicit user toggle; the library
    # keeps them and labels them so nothing is silently dropped.
    include_waters: bool = True
    # Drop HIS/CYS backbone N/C/O from the crop (a modelling choice for
    # RPATH≈6 Å EXAFS clusters). Recorded in provenance.
    drop_backbone: bool = True
    # Add hydrogens via Open Babel. OFF by default: FEFF strips H anyway, and
    # Open Babel places H poorly — the app's DFT block does interactive H
    # editing instead (design doc 04 §3.4). Requires the ``dft`` optional extra.
    add_hydrogens: bool = False

    # --- misc ---
    # Skip structures whose resolution is worse than this (Å). None = keep all.
    resolution_cutoff: Optional[float] = None
    # Only process MODEL 1 of multi-model (NMR) entries.
    first_model_only: bool = False
    # Emit one SiteCandidate per altloc variant (as the CLI does). When False,
    # altlocs are collapsed (blank/A preferred) into a single candidate.
    altloc_split: bool = True


@dataclass(frozen=True)
class SiteCandidate:
    """One absorber site extracted from a PDB entry.

    A polynuclear cluster yields **one SiteCandidate per absorber-element atom**
    (each with its own absorber-centred xyz), all sharing ``cluster_id`` and
    listing each other in ``sibling_target_indices``. That is exactly what lets
    EARL offer "average over the N equivalent sites" (design decision A26 /
    review F1).
    """

    source: Literal["pdb"]
    pdb_id: str
    model: Optional[int]
    cluster_id: str
    cluster_type: str  # "homo" | "multi_homo" | "multi_hetero"

    # PDB serial of the absorber atom this candidate is centred on.
    target_atom_index: int
    # Serials of the other absorber-element atoms in the same cluster (the
    # equivalent sites to average over). Empty for a mononuclear site.
    sibling_target_indices: tuple[int, ...]

    altloc_case: Optional[int]
    altloc_label: Optional[str]
    resolution_A: Optional[float]

    # Geometry: {"label", "CN", "RMS_theta", "distorted", "planar", ...}
    geometry: dict
    # First shell, one entry per element:
    #   {"element", "count", "rmin", "rmax", "is_water", "residues": [str, ...]}
    coordination: list
    # Human-readable coordinating residues, e.g. ["CYS A 12 SG", "HIS A 45 ND1"].
    ligand_residues: list

    # Absorber-centred xyz text: absorber first, title-cased symbols, the
    # provenance comment line, and per-atom ``# RES=… CHAIN=…`` tails preserved.
    xyz_text: str
    hydrogens_added: bool

    # Free-form provenance: pdb_id, model, cluster_id, altloc, resolution,
    # title/method/deposition (when known), and the modelling flags applied.
    provenance: dict

    @property
    def n_sites_in_cluster(self) -> int:
        """How many absorber sites this cluster averages over (1 = mononuclear)."""
        return 1 + len(self.sibling_target_indices)


@dataclass
class ExtractResult:
    """Return value of :func:`scrape_pdb.api.extract_sites`."""

    sites: list = field(default_factory=list)
    # Why sites/clusters were dropped, e.g. {"no_target_component": 1}. Lets the
    # UI explain an empty result in plain words.
    rejections: Counter = field(default_factory=Counter)
    warnings: list = field(default_factory=list)
    # Entry-level facts: pdb_id, n_models, n_clusters, resolution_A, etc.
    stats: dict = field(default_factory=dict)
