#!/usr/bin/env python3
"""Polynuclear extraction contract (design doc 04 §3.1, TODO 1.4/1.7).

Uses small committed slices of two real entries (no network):
  * 5xp6_zn_site.pdb  — the di-zinc NDM-1 active site.
  * 1oao_ni_site.pdb  — one ACS/CODH Ni site with its neighbouring Fe4S4.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scrape_pdb.api import ExtractConfig, extract_sites

DATA = Path(__file__).resolve().parent / "data"


def _sibling_metal_distance(site, symbol):
    """Distance (Å) from the absorber (at origin) to the nearest same-symbol
    metal in the crop — used to check the Zn–Zn separation."""
    best = None
    for line in site.xyz_text.splitlines()[2:]:
        parts = line.split()
        if parts and parts[0] == symbol:
            d = math.sqrt(sum(float(v) ** 2 for v in parts[1:4]))
            if d > 0.01 and (best is None or d < best):
                best = d
    return best


def test_5xp6_two_zn_sites_averaged(tmp_path):
    res = extract_sites(DATA / "5xp6_zn_site.pdb", element="ZN",
                        config=ExtractConfig(workdir=tmp_path))

    # One SiteCandidate per absorber atom -> two Zn sites in one shared cluster.
    assert len(res.sites) == 2
    assert all(s.cluster_type == "multi_homo" for s in res.sites)
    assert len({s.cluster_id for s in res.sites}) == 1

    # Each site lists the other as a sibling (this is what "average over sites"
    # consumes).
    serials = {s.target_atom_index for s in res.sites}
    for s in res.sites:
        assert set(s.sibling_target_indices) == serials - {s.target_atom_index}
        assert s.n_sites_in_cluster == 2

    # Zn–Zn separation ~3.95 Å.
    d = _sibling_metal_distance(res.sites[0], "Zn")
    assert d is not None
    assert abs(d - 3.95) <= 0.01, f"Zn-Zn distance {d} not ~3.95"


def test_1oao_ni_sites_only_iron_excluded(tmp_path):
    fixture = DATA / "1oao_ni_site.pdb"
    res = extract_sites(fixture, element="NI", config=ExtractConfig(workdir=tmp_path))
    assert res.sites

    lines = fixture.read_text().splitlines()
    ni_serials = {
        int(l[6:11]) for l in lines
        if l.startswith("HETATM") and l[12:16].strip().upper() == "NI"
    }
    fe_count = sum(
        1 for l in lines
        if l.startswith("HETATM") and l[76:78].strip().upper() == "FE"
    )

    # Every candidate is centred on a Ni; the Fe4S4 irons are present in the
    # crop but never become absorber sites (target="NI").
    assert all(s.target_atom_index in ni_serials for s in res.sites)
    assert fe_count > 0
    assert all(s.geometry["CN"] >= 1 for s in res.sites)
