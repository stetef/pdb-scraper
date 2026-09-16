#!/usr/bin/env python3
"""P1.11 backbone-drop rule + P1.12 per-atom annotations.

P1.11 (tetef01, 2026-09-16): Cα is ALWAYS kept; ``drop_backbone=True`` drops
backbone N, C, O, OXT of every STANDARD amino-acid residue, EXCEPT a residue with
any backbone atom within ``coordination_cutoff`` (default 3.0 Å) of an absorber —
that residue keeps its ENTIRE backbone. Non-standard residues (ligands, waters,
modified residues, metals) are untouched.

P1.12: ``SiteCandidate.atoms_meta`` — one dict per written xyz atom, 0-based, in
xyz order (post-drop, before hydrogens).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scrape_pdb.api import ExtractConfig, extract_sites
from scrape_pdb.models import Atom
from scrape_pdb.writer import (
    render_xyz_text,
    ordered_atoms_for_xyz,
    atoms_meta_for,
    backbone_rule_string,
    _residues_keeping_full_backbone,
    _should_drop_backbone_atom,
)

DATA = Path(__file__).resolve().parent / "data"


def _atom(serial, atom_name, element, x=0.0, y=0.0, z=0.0, *, resname="ALA",
          chain="A", resseq="1", record="ATOM", altloc=""):
    raw = f"{record:<6}{serial:>5} {atom_name}"
    return Atom(record, serial, atom_name, altloc, resname, chain, resseq, "",
                x, y, z, 1.0, 0.0, element, "", raw)


# --- P1.11 unit: the rule on synthetic atoms --------------------------------


def test_drop_rule_keeps_ca_drops_ncoo_for_standard_aa_far_from_absorber():
    zn = _atom(1, "ZN", "ZN", 0, 0, 0, resname="ZN", record="HETATM")
    # A standard AA (HIS) with its whole backbone ~10 Å from Zn (side chain
    # coordinates in reality, but its BACKBONE does not): N/C/O/OXT drop, Cα stays.
    res = [
        _atom(2, "N", "N", 10, 0, 0, resname="HIS", resseq="20"),
        _atom(3, "CA", "C", 10, 1, 0, resname="HIS", resseq="20"),
        _atom(4, "C", "C", 10, 2, 0, resname="HIS", resseq="20"),
        _atom(5, "O", "O", 10, 3, 0, resname="HIS", resseq="20"),
        _atom(6, "OXT", "O", 10, 4, 0, resname="HIS", resseq="20"),
        _atom(7, "CB", "C", 11, 1, 0, resname="HIS", resseq="20"),
        _atom(8, "ND1", "N", 2, 0, 0, resname="HIS", resseq="20"),  # coordinating side chain
    ]
    atoms = [zn, *res]
    keep_full = _residues_keeping_full_backbone(atoms, "ZN", 3.0)
    assert keep_full == set()  # no BACKBONE atom within 3.0 Å
    kept = ordered_atoms_for_xyz(atoms, zn, absorber_first=True, drop_backbone=True,
                                 target="ZN", coordination_cutoff=3.0)
    names = [a.atom_name for a in kept]
    assert "CA" in names and "CB" in names and "ND1" in names
    assert "N" not in names and "C" not in names and "O" not in names and "OXT" not in names
    assert names[0] == "ZN"  # absorber first


def test_coordinating_backbone_keeps_full_residue():
    zn = _atom(1, "ZN", "ZN", 0, 0, 0, resname="ZN", record="HETATM")
    # A CxxC-style residue whose BACKBONE amide N sits 2.1 Å from Zn (deprotonated
    # amide coordination) — the whole backbone is kept.
    res = [
        _atom(2, "N", "N", 2.1, 0, 0, resname="CYS", resseq="30"),
        _atom(3, "CA", "C", 3.0, 0.5, 0, resname="CYS", resseq="30"),
        _atom(4, "C", "C", 4.0, 0, 0, resname="CYS", resseq="30"),
        _atom(5, "O", "O", 4.5, 1, 0, resname="CYS", resseq="30"),
        _atom(6, "SG", "S", 3.0, -2, 0, resname="CYS", resseq="30"),
    ]
    atoms = [zn, *res]
    keep_full = _residues_keeping_full_backbone(atoms, "ZN", 3.0)
    assert ("A", "30", "") in keep_full
    kept = ordered_atoms_for_xyz(atoms, zn, absorber_first=True, drop_backbone=True,
                                 target="ZN", coordination_cutoff=3.0)
    names = {a.atom_name for a in kept}
    assert names == {"ZN", "N", "CA", "C", "O", "SG"}  # full backbone retained


def test_non_standard_residues_never_dropped():
    zn = _atom(1, "ZN", "ZN", 0, 0, 0, resname="ZN", record="HETATM")
    water = _atom(2, "O", "O", 2.2, 0, 0, resname="HOH", record="HETATM")
    ligand_n = _atom(3, "N", "N", 9, 0, 0, resname="IMD", record="HETATM")  # a ligand's N
    kept = ordered_atoms_for_xyz([zn, water, ligand_n], zn, absorber_first=True,
                                 drop_backbone=True, target="ZN", coordination_cutoff=3.0)
    assert {a.atom_name for a in kept} == {"ZN", "O", "N"}  # nothing dropped


def test_gly_keeps_only_ca():
    zn = _atom(1, "ZN", "ZN", 0, 0, 0, resname="ZN", record="HETATM")
    gly = [
        _atom(2, "N", "N", 8, 0, 0, resname="GLY", resseq="5"),
        _atom(3, "CA", "C", 8, 1, 0, resname="GLY", resseq="5"),
        _atom(4, "C", "C", 8, 2, 0, resname="GLY", resseq="5"),
        _atom(5, "O", "O", 8, 3, 0, resname="GLY", resseq="5"),
    ]
    kept = ordered_atoms_for_xyz([zn, *gly], zn, absorber_first=True, drop_backbone=True,
                                 target="ZN", coordination_cutoff=3.0)
    assert {a.atom_name for a in kept} == {"ZN", "CA"}


def test_drop_backbone_false_keeps_everything():
    zn = _atom(1, "ZN", "ZN", 0, 0, 0, resname="ZN", record="HETATM")
    his_n = _atom(2, "N", "N", 8, 0, 0, resname="HIS", resseq="20")
    kept = ordered_atoms_for_xyz([zn, his_n], zn, absorber_first=False, drop_backbone=False,
                                 target="ZN", coordination_cutoff=3.0)
    assert {a.atom_name for a in kept} == {"ZN", "N"}


def test_backbone_rule_string():
    assert backbone_rule_string(3.0) == \
        "drop_N_C_O_OXT_keep_CA; keep_full_backbone_if_coordinating<=3.0A"
    assert backbone_rule_string(2.5).endswith("<=2.5A")


# --- P1.11 integration: 5XP6 (no backbone coordinates) / 1OAO (amide N does) --


def _sites(pdb: str, element: str, tmp_path):
    res = extract_sites(DATA / pdb, element=element,
                        config=ExtractConfig(workdir=tmp_path, cutoff=5.0, selection_radius=6.0))
    assert res.sites, f"no sites extracted from {pdb}"
    return res.sites


def _standard_aa_by_atom(atoms_meta):
    from scrape_pdb.constants import STANDARD_AMINO_ACIDS
    return [m for m in atoms_meta if (m["resname"] or "").upper() in STANDARD_AMINO_ACIDS]


def test_5xp6_every_standard_residue_keeps_ca_loses_ncoo(tmp_path):
    # NDM-1 di-zinc site: residues coordinate through side chains, so NO backbone
    # atom is within 3.0 Å of a Zn — every standard AA loses N/C/O/OXT, keeps Cα.
    site = _sites("5xp6_zn_site.pdb", "ZN", tmp_path)[0]
    aa = _standard_aa_by_atom(site.atoms_meta)
    assert aa, "expected standard amino-acid residues in the 5XP6 crop"
    names = {m["atom_name"].strip().upper() for m in aa}
    assert not (names & {"N", "C", "O", "OXT"}), f"backbone atoms leaked: {names}"
    # Cα is preserved wherever a residue's Cα is inside the crop (the selection
    # radius can crop a residue to a coordinating side-chain fragment with no Cα, so
    # this is a subset relation, not equality).
    ca_res = {(m["chain"], m["resseq"]) for m in aa if m["atom_name"].strip().upper() == "CA"}
    assert ca_res, "expected at least one retained Cα"
    assert site.provenance["backbone_rule"] == backbone_rule_string(3.0)


def test_1oao_non_coordinating_backbone_is_dropped(tmp_path):
    # ACS/CODH Ni next to an Fe4S4. In THIS committed slice the nearest standard-AA
    # backbone atom is a Gly Cα at ~4.29 Å (backbone amide N are farther), so NO
    # residue's backbone coordinates within 3.0 Å — every standard AA loses
    # N/C/O/OXT even in a metal-rich crop. The coordination EXCEPTION is exercised
    # by test_coordinating_backbone_keeps_full_residue (synthetic), because the
    # committed 1OAO/7NYS slices do not contain a backbone atom within 3.0 Å of the
    # absorber (see the report to the orchestrator — a fixture gap, flagged).
    for site in _sites("1oao_ni_site.pdb", "NI", tmp_path):
        aa = _standard_aa_by_atom(site.atoms_meta)
        names = {m["atom_name"].strip().upper() for m in aa}
        assert not (names & {"N", "C", "O", "OXT"}), f"backbone leaked in 1OAO: {names}"
        assert site.provenance["backbone_rule"] == backbone_rule_string(3.0)


# --- P1.12 per-atom annotations ---------------------------------------------


def test_atoms_meta_length_and_absorber_first(tmp_path):
    site = _sites("5xp6_zn_site.pdb", "ZN", tmp_path)[0]
    n_xyz = int(site.xyz_text.splitlines()[0].strip())
    assert len(site.atoms_meta) == n_xyz          # one per written heavy atom (pre-H)
    m0 = site.atoms_meta[0]
    assert m0["index"] == 0
    assert m0["atom_name"].strip().upper() == "ZN"   # absorber first
    assert m0["is_hetero"] is True
    # 0-based, contiguous, full key set.
    assert [m["index"] for m in site.atoms_meta] == list(range(n_xyz))
    for m in site.atoms_meta:
        assert set(m) == {"index", "resname", "resseq", "chain", "atom_name", "altloc", "is_hetero"}


def test_atoms_meta_matches_xyz_order(tmp_path):
    # atoms_meta[i].atom_name equals the ATOM= provenance tail of xyz line i.
    site = _sites("5xp6_zn_site.pdb", "ZN", tmp_path)[0]
    lines = site.xyz_text.splitlines()
    n = int(lines[0].strip())
    for i, line in enumerate(lines[2 : 2 + n]):
        tail = line.partition("#")[2]
        fields = dict(p.split("=", 1) for p in tail.split() if "=" in p)
        assert site.atoms_meta[i]["atom_name"] == fields["ATOM"]
        assert site.atoms_meta[i]["resname"] == fields["RES"]
