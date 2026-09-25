#!/usr/bin/env python3
"""Residue-aware, geometry-checked coordination analysis for configuration searches.

The legacy coordination test (:func:`scrape_pdb.cluster.select_coordinating_neighbors`)
keeps every donor-element atom in a distance window. That miscounts sites in
three ways that matter for "Zn coordinated by exactly N Cys/His" searches:

* it counts **atoms**, so a His with both ND1 and NE2 inside a generous window
  counts twice (a 3-His site then passes as 4-His);
* it ignores carbon, so a His whose ring is modelled flipped (CE1 ~2.0 A from the
  metal, the labelled N ~3 A away) still counts through its far N;
* ligands that aren't in the requirement list (sulfate, phosphate, Asp/Glu,
  inhibitors, waters) are never seen, so mixed-ligand sites pass.

:func:`analyze_coordination` instead picks ONE donor atom per residue, checks
per-element distance windows and His ring geometry, and (given the ligand
requirements) checks the site has no extra ligand and the exact coordination
number. It is opt-in via ``validation.strict_coordination.enabled``; with it off
the pipeline behaves exactly as before, and the library ``extract`` path never
uses it.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from .cluster import select_coordinating_neighbors
from .constants import ALL_METALS, COORDINATION_DONOR_ELEMENTS, WATER_RESIDUES
from .models import Atom
from .utils import dist
from .validation import passes_ligand_requirements

REASONS = ("carbon_contact", "donor_distance", "his_geometry", "extra_ligand",
           "coordination_number", "ligand_requirements")

# Ring neighbours of each imidazole N; the N lone pair points away from their midpoint.
HIS_RING_NEIGHBORS = {"ND1": ("CG", "CE1"), "NE2": ("CD2", "CE1")}
HIS_RESNAMES = {"HIS", "HID", "HIE", "HIP"}


class StrictCoordinationConfig(BaseModel):
    """Settings for :func:`analyze_coordination` (``validation.strict_coordination``)."""

    enabled: bool = False
    # A residue is a candidate ligand if its nearest heavy atom is this close (A).
    contact_radius: float = 2.8
    # Allowed metal-donor distance (A) per donor element.
    donor_windows: Dict[str, Tuple[float, float]] = Field(default_factory=lambda: {
        "S": (1.95, 2.60),
        "N": (1.85, 2.45),
        "O": (1.80, 2.60),
    })
    # Fallback window for donor elements not listed above (halides, P, Se, ...).
    default_window: Tuple[float, float] = (1.80, 2.80)
    # His: max angle (deg) between metal and the donor-N lone pair, and how much
    # farther (A) the ring carbons next to the donor N must be than the N.
    his_max_lone_pair_deg: float = 35.0
    his_ring_c_margin: float = 0.6
    # With ligand requirements: coordinating residues must equal the summed counts.
    exact_coordination_number: bool = True
    # Count waters as ligands (a Zn-bound water makes the site 5-coordinate).
    count_waters: bool = True

    def window_for(self, element: str) -> Tuple[float, float]:
        return tuple(self.donor_windows.get(element.upper(), self.default_window))


@dataclass
class DonorContact:
    """The closest contact one residue makes with the metal."""

    resname: str
    chain: str
    resseq: str
    icode: str
    atom_name: str
    element: str
    distance: float
    atom: Atom
    angle_deg: Optional[float] = None  # His N only: metal vs lone-pair direction


@dataclass
class CoordinationAnalysis:
    contacts: List[DonorContact] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    reasons: set = field(default_factory=set)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def donors(self) -> List[Atom]:
        """One donor atom per coordinating residue (carbon contacts excluded)."""
        return [c.atom for c in self.contacts if c.element in COORDINATION_DONOR_ELEMENTS]

    @property
    def coordination_number(self) -> int:
        return len(self.donors)

    @property
    def blocking_reasons(self) -> set:
        """Reasons other than the plain requirement count (which the parser reports itself)."""
        return self.reasons - {"ligand_requirements"}

    def _fail(self, reason: str, msg: str) -> None:
        self.reasons.add(reason)
        self.errors.append(f"{reason}: {msg}")


def _elem(a: Atom) -> str:
    return (a.element or "").strip().upper()


def _name(a: Atom) -> str:
    return (a.atom_name or "").strip().upper()


def _his_lone_pair_angle(center: Atom, donor: Atom, ring: Dict[str, Atom]) -> Optional[float]:
    nbrs = HIS_RING_NEIGHBORS.get(_name(donor))
    if not nbrs or any(n not in ring for n in nbrs):
        return None
    a, b = ring[nbrs[0]], ring[nbrs[1]]
    lp = [donor.coord[i] - (a.coord[i] + b.coord[i]) / 2 for i in range(3)]
    v = [center.coord[i] - donor.coord[i] for i in range(3)]
    nl = math.sqrt(sum(x * x for x in lp))
    nv = math.sqrt(sum(x * x for x in v))
    if nl == 0 or nv == 0:
        return None
    cos = max(-1.0, min(1.0, sum(p * q for p, q in zip(lp, v)) / (nl * nv)))
    return math.degrees(math.acos(cos))


def _req_get(req: Any, key: str, default: Any = None) -> Any:
    if isinstance(req, dict):
        return req.get(key, default)
    return getattr(req, key, default)


def _req_resnames(req: Any) -> List[str]:
    names = _req_get(req, "resnames") or ([_req_get(req, "resname")] if _req_get(req, "resname") else [])
    return [str(n).strip().upper() for n in names if str(n).strip()]


def _req_count(req: Any) -> int:
    try:
        return max(0, int(_req_get(req, "count", _req_get(req, "min_count", 1))))
    except (TypeError, ValueError):
        return 1


def _matches_requirement(contact: DonorContact, reqs: Sequence[Any]) -> bool:
    for req in reqs:
        if contact.resname not in _req_resnames(req):
            continue
        atom_names = [str(x).strip().upper() for x in (_req_get(req, "atom_names") or [])]
        if not atom_names or contact.atom_name in atom_names:
            return True
    return False


def analyze_coordination(
    center: Atom,
    atoms: Iterable[Atom],
    cfg: Optional[StrictCoordinationConfig] = None,
    ligand_requirements: Optional[Sequence[Any]] = None,
) -> CoordinationAnalysis:
    """Classify the first coordination shell of ``center`` one residue at a time.

    ``atoms`` is any heavy-atom pool around the metal (hydrogens, the center and
    other metals are ignored). See the module docstring for the rules; geometry
    reasons are always evaluated, ``extra_ligand`` / ``coordination_number`` only
    when ``ligand_requirements`` is given.
    """
    cfg = cfg or StrictCoordinationConfig()
    out = CoordinationAnalysis()

    # Group every nearby heavy atom by residue instance.
    residues: Dict[Tuple[str, str, str, str], List[Tuple[float, Atom]]] = {}
    for a in atoms:
        if a.serial == center.serial and _name(a) == _name(center):
            continue
        e = _elem(a)
        if e in ("H", "D") or e in ALL_METALS:
            continue
        key = (a.resname.strip().upper(), a.chain, a.resseq.strip(), a.icode)
        residues.setdefault(key, []).append((dist(a.coord, center.coord), a))

    for (resname, chain, resseq, icode), members in sorted(residues.items()):
        members.sort(key=lambda t: t[0])
        nearest_d, nearest = members[0]
        if nearest_d > cfg.contact_radius:
            continue
        if resname in WATER_RESIDUES and not cfg.count_waters:
            continue
        tag = f"{resname}{chain}{resseq}{icode}"
        donors = [(d, a) for d, a in members if _elem(a) in COORDINATION_DONOR_ELEMENTS]
        if _elem(nearest) not in COORDINATION_DONOR_ELEMENTS:
            # A non-donor (carbon) is this residue's closest atom to the metal.
            if not donors or donors[0][0] > nearest_d:
                far = f", nearest donor {_name(donors[0][1])} {donors[0][0]:.2f}" if donors else ""
                out._fail("carbon_contact",
                          f"{tag} {_name(nearest)} at {nearest_d:.2f} A{far}")
                out.contacts.append(DonorContact(resname, chain, resseq, icode, _name(nearest),
                                                 _elem(nearest), nearest_d, nearest))
                continue
        d, donor = donors[0]
        contact = DonorContact(resname, chain, resseq, icode, _name(donor), _elem(donor), d, donor)
        out.contacts.append(contact)

        lo, hi = cfg.window_for(contact.element)
        if not lo <= d <= hi:
            out._fail("donor_distance", f"{tag} {contact.atom_name} {d:.2f} A outside [{lo}, {hi}]")

        if resname in HIS_RESNAMES:
            if contact.atom_name not in HIS_RING_NEIGHBORS:
                out._fail("his_geometry", f"{tag} binds through {contact.atom_name}, not ND1/NE2")
                continue
            ring = {_name(a): a for _, a in members
                    if _name(a) in ("CG", "ND1", "CD2", "CE1", "NE2")}
            ang = _his_lone_pair_angle(center, donor, ring)
            if ang is None:
                out._fail("his_geometry", f"{tag} incomplete imidazole ring")
                continue
            contact.angle_deg = ang
            if ang > cfg.his_max_lone_pair_deg:
                out._fail("his_geometry",
                          f"{tag} metal {ang:.0f} deg off {contact.atom_name} lone pair")
            nbr_d = [dist(ring[n].coord, center.coord) for n in HIS_RING_NEIGHBORS[contact.atom_name]]
            if min(nbr_d) - d < cfg.his_ring_c_margin:
                out._fail("his_geometry",
                          f"{tag} ring C only {min(nbr_d) - d:.2f} A farther than {contact.atom_name}")

    if ligand_requirements:
        donor_contacts = [c for c in out.contacts if c.element in COORDINATION_DONOR_ELEMENTS]
        # passes_ligand_requirements reads attributes; accept plain dicts too.
        reqs = [SimpleNamespace(**r) if isinstance(r, dict) else r for r in ligand_requirements]
        if not passes_ligand_requirements(out.donors, reqs):
            out._fail("ligand_requirements", "donor counts do not match the ligand requirements")
        for c in donor_contacts:
            if not _matches_requirement(c, ligand_requirements):
                out._fail("extra_ligand", f"{c.resname}{c.chain}{c.resseq}{c.icode} "
                                          f"{c.atom_name} at {c.distance:.2f} A")
        if cfg.exact_coordination_number:
            want = sum(_req_count(r) for r in ligand_requirements)
            if out.coordination_number != want:
                out._fail("coordination_number",
                          f"{out.coordination_number} coordinating residues, expected {want}")
    return out


def coordination_for_config(
    center: Atom,
    atoms: List[Atom],
    validation: Any,
) -> CoordinationAnalysis:
    """Coordination analysis driven by a ``ValidationConfig``.

    Strict mode off (default): wraps the legacy donor-window selection unchanged
    and always reports ``ok``. Strict mode on: :func:`analyze_coordination` with
    the configured ligand requirements.
    """
    strict = getattr(validation, "strict_coordination", None)
    if strict is None or not strict.enabled:
        legacy = select_coordinating_neighbors(
            center, atoms,
            validation.coordination_distance_min,
            validation.coordination_distance_max,
        )
        res = CoordinationAnalysis()
        res.contacts = [DonorContact(a.resname.strip().upper(), a.chain, a.resseq.strip(), a.icode,
                                     _name(a), _elem(a), dist(a.coord, center.coord), a)
                        for a in legacy]
        return res
    return analyze_coordination(center, atoms, strict, validation.ligand_requirements)
