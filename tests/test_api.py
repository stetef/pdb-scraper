#!/usr/bin/env python3
"""Contract tests for the EARL library API (scrape_pdb.api).

Pure-function tests on committed sample PDBs — no network. Network access is
only exercised through a mocked ``fetch_pdb``.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scrape_pdb.api import ExtractConfig, FetchError, extract_sites, fetch_entry
from scrape_pdb.site import SiteCandidate

DATA = Path(__file__).resolve().parent / "data"
ZN_FIXTURE = DATA / "5xp6_zn_site.pdb"


# --- fetch_entry (network mocked) ------------------------------------------

@pytest.mark.parametrize("bad", ["", "zn", "12345", "0abc", "ab!c", "xxxx"])
def test_fetch_entry_rejects_bad_id(tmp_path, bad):
    with pytest.raises(FetchError) as ei:
        fetch_entry(bad, workdir=tmp_path)
    assert ei.value.kind == "invalid_id"


def test_fetch_entry_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr("scrape_pdb.api.fetch_pdb", lambda *a, **k: None)
    with pytest.raises(FetchError) as ei:
        fetch_entry("1abc", workdir=tmp_path, max_retries=1)
    assert ei.value.kind == "not_found"


def test_fetch_entry_size_cap(tmp_path, monkeypatch):
    def fake(pid, dest, timeout=30):
        p = Path(dest) / f"{pid}.pdb"
        p.write_bytes(b"x" * 100)
        return str(p)

    monkeypatch.setattr("scrape_pdb.api.fetch_pdb", fake)
    with pytest.raises(FetchError) as ei:
        fetch_entry("1abc", workdir=tmp_path, size_cap_bytes=10)
    assert ei.value.kind == "too_large"


def test_fetch_entry_success(tmp_path, monkeypatch):
    def fake(pid, dest, timeout=30):
        p = Path(dest) / f"{pid}.pdb"
        p.write_text("HEADER\nEND\n")
        return str(p)

    monkeypatch.setattr("scrape_pdb.api.fetch_pdb", fake)
    p = fetch_entry("1ABC", workdir=tmp_path)  # case-insensitive
    assert p.exists() and p.suffix == ".pdb"


# --- extract_sites (committed fixture, no network) -------------------------

def test_extract_sites_contract(tmp_path):
    res = extract_sites(ZN_FIXTURE, element="ZN", config=ExtractConfig(workdir=tmp_path))
    assert res.sites
    assert all(isinstance(s, SiteCandidate) for s in res.sites)

    s = res.sites[0]
    assert s.source == "pdb"
    assert "5xp6" in s.pdb_id

    lines = s.xyz_text.splitlines()
    # line 0 = count, line 1 = provenance comment, line 2 = first atom (the absorber)
    assert lines[1].startswith("PDB=")
    first = lines[2].split()
    assert first[0] == "Zn"                       # title-cased symbol
    assert first[1:4] == ["0.000000", "0.000000", "0.000000"]  # absorber at origin, first
    assert "# RES=" in lines[2]                   # per-atom provenance tail kept
    assert s.hydrogens_added is False             # default: no Open Babel


def test_extract_sites_is_pure(tmp_path):
    before = ZN_FIXTURE.read_bytes()
    res = extract_sites(ZN_FIXTURE, element="ZN", config=ExtractConfig(workdir=tmp_path))
    assert res.sites
    assert ZN_FIXTURE.read_bytes() == before          # input never modified
    assert list(tmp_path.iterdir()) == []             # nothing written/left behind


def test_extract_sites_rejects_unknown_element(tmp_path):
    with pytest.raises(ValueError):
        extract_sites(ZN_FIXTURE, element="XX", config=ExtractConfig(workdir=tmp_path))


def test_extract_sites_absorber_absent(tmp_path):
    # Copper is a valid metal but not present in this Zn fixture -> no sites,
    # with a rejection the UI can explain.
    res = extract_sites(ZN_FIXTURE, element="CU", config=ExtractConfig(workdir=tmp_path))
    assert res.sites == []
    assert sum(res.rejections.values()) > 0
