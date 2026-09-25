#!/usr/bin/env python3
"""The library search-by-PDB path is pure cluster carving and must not be
affected by the opt-in strict coordination check used by configuration searches.

1thk's Zn has a flipped His96 ring (CE1 1.99 A from Zn), which strict mode
rejects. The library, with its default "show every site" policy, must still
return that site.
"""

from pathlib import Path

import scrape_pdb.extract as extract_mod
from scrape_pdb.api import extract_sites
from scrape_pdb.site import ExtractConfig

FIXTURE = Path(__file__).parent / "data" / "coordination" / "1thk_zn_A262.pdb"


def test_extract_config_has_no_strict_switch():
    assert not hasattr(ExtractConfig, "strict_coordination")


def test_extract_path_does_not_use_strict_coordination():
    src = Path(extract_mod.__file__).read_text()
    assert "coordination_for_config" not in src
    assert "analyze_coordination" not in src


def test_library_still_carves_site_strict_mode_rejects(tmp_path):
    result = extract_sites(FIXTURE, element="ZN", config=ExtractConfig(workdir=tmp_path))
    # Zn A262 is PDB serial 2057 in the fixture.
    assert [s.target_atom_index for s in result.sites] == [2057]
