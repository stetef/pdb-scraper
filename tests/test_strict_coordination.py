"""Tests for the strict coordination classifier (scrape_pdb.coordination).

Fixtures in tests/data/coordination/ are real PDB entries trimmed to whole
residues within 8 A of one target Zn (see each file's REMARK 999 header).
Expected outcomes come from the clustering-zinc-fingers QC report
(qc_report-sampled-xyz-files-for-val.csv) for the 4his / 1cys3his sets.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from scrape_pdb.cluster import select_coordinating_neighbors
from scrape_pdb.config import load_config
from scrape_pdb.coordination import CoordinationAnalysis, StrictCoordinationConfig, analyze_coordination
from scrape_pdb.parser import collapse_altloc, load_pdb_atoms_all, process_pdb

FIXTURES = Path(__file__).resolve().parent / "data" / "coordination"

GEOMETRY_REASONS = {"carbon_contact", "donor_distance", "his_geometry"}

REQ_4HIS = [{"resname": "HIS", "atom_names": ["ND1", "NE2"], "count": 4}]
REQ_1CYS3HIS = [
    {"resname": "CYS", "atom_names": ["SG"], "count": 1},
    {"resname": "HIS", "atom_names": ["ND1", "NE2"], "count": 3},
]
REQ_4CYS = [{"resname": "CYS", "atom_names": ["SG"], "count": 4}]


# ============================================================================
# HELPERS
# ============================================================================

def load_site(name: str, radius: float = 6.0):
    """Return (zn_center, heavy atoms within radius of it) for a fixture file.

    The target Zn is identified from the file name: <pdbid>_zn_<chain><resseq>.pdb.
    """
    path = FIXTURES / f"{name}.pdb"
    atoms = collapse_altloc(load_pdb_atoms_all(str(path), first_model_only=True))
    tag = name.split("_zn_")[1]
    chain, resseq = tag[0], tag[1:]
    centers = [a for a in atoms if a.element == "ZN" and a.chain == chain and a.resseq == resseq]
    assert len(centers) == 1, f"{name}: expected one target Zn, got {centers}"
    center = centers[0]
    near = [
        a for a in atoms
        if a.serial != center.serial and a.element != "H" and math.dist(a.coord, center.coord) <= radius
    ]
    return center, near


def donor_keys(res: CoordinationAnalysis) -> set[tuple[str, str, str, str]]:
    return {(a.resname, a.chain, a.resseq, a.atom_name) for a in res.donors}


def residue_keys(res: CoordinationAnalysis) -> set[tuple[str, str, str]]:
    return {(a.resname, a.chain, a.resseq) for a in res.donors}


def lone_pair_angle_deg(zn, atoms, resid, n_name) -> float:
    """Angle between Zn-N and the His N lone pair (N minus midpoint of its ring neighbours)."""
    by_name = {a.atom_name: np.array(a.coord) for a in atoms if (a.chain, a.resseq) == resid}
    nbrs = {"ND1": ("CG", "CE1"), "NE2": ("CD2", "CE1")}[n_name]
    n = by_name[n_name]
    lp = n - (by_name[nbrs[0]] + by_name[nbrs[1]]) / 2
    v = np.array(zn.coord) - n
    cos = np.dot(lp, v) / (np.linalg.norm(lp) * np.linalg.norm(v))
    return math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))


def rotate_residue(atoms, resid, pivot, axis, angle_deg):
    """Rotate all atoms of one residue in place about an axis through `pivot` (Rodrigues)."""
    k = np.asarray(axis, float)
    k /= np.linalg.norm(k)
    t = math.radians(angle_deg)
    p = np.asarray(pivot, float)
    for a in atoms:
        if (a.chain, a.resseq) != resid:
            continue
        v = np.array(a.coord) - p
        r = v * math.cos(t) + np.cross(k, v) * math.sin(t) + k * np.dot(k, v) * (1 - math.cos(t))
        a.x, a.y, a.z = (p + r).tolist()


def move_along(center, atom, new_dist):
    """Move `atom` in place along the Zn->atom direction to `new_dist` from Zn."""
    c = np.array(center.coord)
    u = np.array(atom.coord) - c
    u /= np.linalg.norm(u)
    atom.x, atom.y, atom.z = (c + u * new_dist).tolist()


# ============================================================================
# CONFIG DEFAULTS
# ============================================================================

def test_strict_config_defaults():
    cfg = StrictCoordinationConfig()
    assert cfg.enabled is False
    assert cfg.contact_radius == pytest.approx(2.8)
    assert tuple(cfg.donor_windows["S"]) == pytest.approx((1.95, 2.60))
    assert tuple(cfg.donor_windows["N"]) == pytest.approx((1.85, 2.45))
    assert tuple(cfg.donor_windows["O"]) == pytest.approx((1.80, 2.60))
    assert cfg.his_max_lone_pair_deg == pytest.approx(35.0)
    assert cfg.his_ring_c_margin == pytest.approx(0.6)
    assert cfg.exact_coordination_number is True
    assert cfg.count_waters is True


def test_validation_config_strict_disabled_by_default(tmp_path):
    config = load_config(str(_write_pipeline_config(tmp_path, reqs=REQ_4HIS, strict=None)))
    assert config.validation.strict_coordination.enabled is False


# ============================================================================
# REAL SITES
# ============================================================================

# name, requirements, expected ok (with requirements), expected donors (None = don't pin),
# geometry reasons expected without requirements, extra reasons expected with requirements
SITE_CASES = [
    pytest.param(
        "3tio_zn_A1", REQ_4HIS, True,
        {("HIS", "A", "67", "ND1"), ("HIS", "A", "70", "NE2"), ("HIS", "C", "91", "NE2"), ("HIS", "A", "96", "NE2")},
        set(), set(), id="3tio-4his-ND1-control",
    ),
    pytest.param(
        "1pb0_zn_A1301", REQ_4HIS, True,
        {("HIS", "A", "15", "NE2"), ("HIS", "A", "40", "NE2"), ("HIS", "A", "164", "NE2"), ("HIS", "A", "194", "NE2")},
        set(), set(), id="1pb0-4his-control",
    ),
    pytest.param(
        "1v4p_zn_A1001", REQ_1CYS3HIS, True,
        {("CYS", "A", "116", "SG"), ("HIS", "A", "9", "NE2"), ("HIS", "A", "13", "NE2"), ("HIS", "A", "120", "NE2")},
        set(), set(), id="1v4p-1cys3his-control",
    ),
    pytest.param(
        "1kwg_zn_A806", REQ_4CYS, True,
        {("CYS", "A", "106", "SG"), ("CYS", "A", "150", "SG"), ("CYS", "A", "152", "SG"), ("CYS", "A", "155", "SG")},
        set(), set(), id="1kwg-4cys-control",
    ),
    # His96 ring flipped: CE1 1.99 A from Zn, NE2/ND1 ~2.96 A. HOH305 at 2.08 A is also bound.
    pytest.param("1thk_zn_A262", REQ_4HIS, False, None, {"carbon_contact"}, {"extra_ligand"}, id="1thk-flipped-his"),
    # SO4 O1 at 1.84 A is a valid O donor geometrically; His263 NE2 (2.84 A) is outside contact_radius.
    pytest.param(
        "1ozu_zn_B356", REQ_4HIS, False,
        {("HIS", "B", "246", "ND1"), ("HIS", "B", "248", "NE2"), ("HIS", "B", "320", "NE2"), ("SO4", "B", "401", "O1")},
        set(), {"extra_ligand"}, id="1ozu-sulfate",
    ),
    # Asp177 OD2 bound; His175 ring flipped (CE1 2.14 A, NE2 3.10 A).
    pytest.param("2ovz_zn_B445", REQ_4HIS, False, None, {"carbon_contact"}, {"extra_ligand"}, id="2ovz-asp"),
    # Glu B62 OE2 bound; His196 ring flipped (CE1 2.00 A, NE2 2.77 A).
    pytest.param("5act_zn_A1297", REQ_4HIS, False, None, {"carbon_contact"}, {"extra_ligand"}, id="5act-glu"),
    # Cys34 SG at 3.19 A is not a contact; two waters (2.16, 2.38 A) complete the sphere.
    pytest.param(
        "3jyg_zn_E200", REQ_1CYS3HIS, False, None, set(), {"coordination_number"}, id="3jyg-long-cys",
    ),
]


@pytest.mark.parametrize("name,reqs,ok,donors,geom_reasons,req_reasons", SITE_CASES)
def test_geometry_only(name, reqs, ok, donors, geom_reasons, req_reasons):
    center, atoms = load_site(name)
    res = analyze_coordination(center, atoms, StrictCoordinationConfig(enabled=True))
    assert res.reasons & GEOMETRY_REASONS == geom_reasons, res.errors
    # Requirement-based reasons are only evaluated when requirements are given.
    assert not res.reasons & {"extra_ligand", "coordination_number"}, res.errors
    assert res.ok == (not geom_reasons)
    if donors is not None:
        assert donor_keys(res) == donors
        assert res.coordination_number == len(donors)


@pytest.mark.parametrize("name,reqs,ok,donors,geom_reasons,req_reasons", SITE_CASES)
def test_with_requirements(name, reqs, ok, donors, geom_reasons, req_reasons):
    center, atoms = load_site(name)
    res = analyze_coordination(center, atoms, StrictCoordinationConfig(enabled=True), ligand_requirements=reqs)
    assert res.ok is ok, res.errors
    assert geom_reasons | req_reasons <= res.reasons, res.errors
    if ok:
        assert res.reasons == set()
        assert res.errors == []
    else:
        assert res.errors, "a rejected site must explain itself"


@pytest.mark.parametrize("name,reqs,ok,donors,geom_reasons,req_reasons", SITE_CASES)
def test_one_donor_per_residue(name, reqs, ok, donors, geom_reasons, req_reasons):
    center, atoms = load_site(name)
    res = analyze_coordination(center, atoms, StrictCoordinationConfig(enabled=True), ligand_requirements=reqs)
    assert len(residue_keys(res)) == len(res.donors)
    assert res.coordination_number == len(res.donors)
    for a in res.donors:
        assert a.element in {"S", "N", "O"}
        if a.resname == "HIS":
            assert a.atom_name in {"ND1", "NE2"}


def test_flipped_his_not_a_donor_and_counted_once():
    """1thk His96: both ND1 and NE2 lie within the legacy 2.0-3.2 A window.

    Legacy selection counts both ring N atoms (4 "His N" total -> passes HIS x4);
    strict mode records His96 once, as a carbon contact, and never as a donor.
    """
    center, atoms = load_site("1thk_zn_A262")
    legacy = select_coordinating_neighbors(center, atoms, 2.0, 3.2)
    his96_legacy = [a for a in legacy if a.resname == "HIS" and a.resseq == "96"]
    assert {a.atom_name for a in his96_legacy} == {"ND1", "NE2"}
    assert sum(1 for a in legacy if a.resname == "HIS" and a.atom_name in ("ND1", "NE2")) == 4

    res = analyze_coordination(center, atoms, StrictCoordinationConfig(enabled=True), ligand_requirements=REQ_4HIS)
    his96_contacts = [c for c in res.contacts if c.resname == "HIS" and c.resseq == "96"]
    assert len(his96_contacts) == 1
    assert his96_contacts[0].element == "C"
    assert his96_contacts[0].distance == pytest.approx(1.99, abs=0.02)
    assert ("HIS", "A", "96") not in residue_keys(res)
    assert "carbon_contact" in res.reasons


def test_bidentate_glu_is_not_a_carbon_contact():
    """5act Glu B62: OE2 1.95 A with CD 2.73 A behind it is a normal O donor."""
    center, atoms = load_site("5act_zn_A1297")
    res = analyze_coordination(center, atoms, StrictCoordinationConfig(enabled=True))
    assert ("GLU", "B", "62", "OE2") in donor_keys(res)
    carbon = [c for c in res.contacts if c.element == "C"]
    assert [(c.resname, c.resseq) for c in carbon] == [("HIS", "196")]


def test_contact_records_his_angle():
    center, atoms = load_site("3tio_zn_A1")
    res = analyze_coordination(center, atoms, StrictCoordinationConfig(enabled=True))
    by_res = {(c.chain, c.resseq): c for c in res.contacts}
    his67 = by_res[("A", "67")]
    assert (his67.resname, his67.atom_name, his67.element) == ("HIS", "ND1", "N")
    assert his67.distance == pytest.approx(2.07, abs=0.02)
    assert his67.angle_deg == pytest.approx(lone_pair_angle_deg(center, atoms, ("A", "67"), "ND1"), abs=1.0)
    assert his67.angle_deg < 35.0


def test_waters_toggle():
    center, atoms = load_site("3jyg_zn_E200")
    with_w = analyze_coordination(center, atoms, StrictCoordinationConfig(enabled=True))
    without_w = analyze_coordination(center, atoms, StrictCoordinationConfig(enabled=True, count_waters=False))
    assert sum(1 for a in with_w.donors if a.resname == "HOH") == 2
    assert with_w.coordination_number == 5
    assert not any(a.resname == "HOH" for a in without_w.donors)
    assert without_w.coordination_number == 3
    # Cys34 SG at 3.19 A is never a donor.
    assert ("CYS", "E", "34") not in residue_keys(with_w)


# ============================================================================
# SYNTHETIC PERTURBATIONS OF REAL SITES
# ============================================================================

@pytest.mark.parametrize("new_dist", [1.85, 2.70])
def test_donor_distance_outside_window(new_dist):
    """Move a 1kwg Cys SG out of the S window (1.95-2.60) but inside contact_radius."""
    center, atoms = load_site("1kwg_zn_A806")
    sg = next(a for a in atoms if (a.resname, a.resseq, a.atom_name) == ("CYS", "152", "SG"))
    move_along(center, sg, new_dist)
    cys152 = [a for a in atoms if a.resseq == "152" and a.chain == "A"]
    assert min(cys152, key=lambda a: math.dist(a.coord, center.coord)) is sg  # SG still nearest

    res = analyze_coordination(center, atoms, StrictCoordinationConfig(enabled=True), ligand_requirements=REQ_4CYS)
    assert res.ok is False
    assert res.reasons & GEOMETRY_REASONS == {"donor_distance"}


def test_his_off_lone_pair():
    """Tilt 3tio His70 out of plane about NE2 so Zn sits ~55 deg off the lone pair.

    Distance and ring-C margin stay normal, so only the angle rule can reject it.
    """
    center, atoms = load_site("3tio_zn_A1")
    resid = ("A", "70")
    by_name = {a.atom_name: a for a in atoms if (a.chain, a.resseq) == resid}
    ne2 = by_name["NE2"]
    d0 = math.dist(ne2.coord, center.coord)
    axis = np.array(by_name["CE1"].coord) - np.array(by_name["CD2"].coord)
    rotate_residue(atoms, resid, ne2.coord, axis, 65.0)

    his70 = [a for a in atoms if (a.chain, a.resseq) == resid]
    dists = {a.atom_name: math.dist(a.coord, center.coord) for a in his70}
    assert min(dists, key=dists.get) == "NE2"
    assert dists["NE2"] == pytest.approx(d0)
    assert min(dists["CE1"], dists["CD2"]) - dists["NE2"] >= 0.6
    assert lone_pair_angle_deg(center, atoms, resid, "NE2") > 45.0

    res = analyze_coordination(center, atoms, StrictCoordinationConfig(enabled=True), ligand_requirements=REQ_4HIS)
    assert res.ok is False
    assert res.reasons & GEOMETRY_REASONS == {"his_geometry"}

    # A looser angle limit accepts the same tilted ring.
    loose = analyze_coordination(
        center, atoms, StrictCoordinationConfig(enabled=True, his_max_lone_pair_deg=90.0), ligand_requirements=REQ_4HIS
    )
    assert loose.ok is True, loose.errors


def test_exact_coordination_number_toggle():
    """3jyg with waters ignored: 3 His vs CYS1+HIS3 -> CN 3 != 4 unless exact CN is off."""
    center, atoms = load_site("3jyg_zn_E200")
    strict = analyze_coordination(
        center, atoms, StrictCoordinationConfig(enabled=True, count_waters=False), ligand_requirements=REQ_1CYS3HIS
    )
    assert "coordination_number" in strict.reasons

    relaxed = analyze_coordination(
        center, atoms,
        StrictCoordinationConfig(enabled=True, count_waters=False, exact_coordination_number=False),
        ligand_requirements=REQ_1CYS3HIS,
    )
    assert "coordination_number" not in relaxed.reasons
    # Still fails: the required Cys is missing.
    assert relaxed.ok is False


# ============================================================================
# PIPELINE INTEGRATION (parser path)
# ============================================================================

def _write_pipeline_config(tmp_path: Path, *, reqs, strict, dmax: float = 3.2) -> Path:
    validation = {
        "coordination_distance_min": 2.0,
        "coordination_distance_max": dmax,
        "ligand_requirements": reqs,
    }
    if strict is not None:
        validation["strict_coordination"] = strict
    cfg = {
        "search_parameters": {"metal_ion": "ZN"},
        "processing": {
            "temp_directory": str(tmp_path / "downloads"),
            "cutoff": 6.0,
            "target": "ZN",
            "metals_excluded": [],
            "include_waters": True,
        },
        "output": {
            "results_database": str(tmp_path / "results" / "clusters.csv"),
            "checkpoint_file": str(tmp_path / "results" / "checkpoint.db"),
            "log_file": str(tmp_path / "results" / "pipeline.log"),
            "output_dir": str(tmp_path / "results"),
        },
        "validation": validation,
        "log_level": "INFO",
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _run(tmp_path, name, reqs, strict):
    config = load_config(str(_write_pipeline_config(tmp_path, reqs=reqs, strict=strict)))
    written, _stats = process_pdb(str(FIXTURES / f"{name}.pdb"), config)
    return written


def test_legacy_behaviour_unchanged_when_disabled(tmp_path, monkeypatch):
    """With strict mode off, the legacy element-window selection still decides.

    1thk passes HIS x4 under the legacy 2.0-3.2 A window because His96 contributes
    both ring N atoms. This pins the legacy (buggy) result so the strict path
    cannot leak into it.
    """
    import scrape_pdb.coordination as coordination
    import scrape_pdb.parser as parser

    def boom(*a, **k):
        raise AssertionError("analyze_coordination must not run when strict_coordination is disabled")

    monkeypatch.setattr(coordination, "analyze_coordination", boom)
    monkeypatch.setattr(parser, "analyze_coordination", boom, raising=False)

    assert len(_run(tmp_path / "off", "1thk_zn_A262", REQ_4HIS, {"enabled": False})) == 1
    assert len(_run(tmp_path / "absent", "1thk_zn_A262", REQ_4HIS, None)) == 1


def test_strict_mode_rejects_flipped_his_in_pipeline(tmp_path):
    assert _run(tmp_path, "1thk_zn_A262", REQ_4HIS, {"enabled": True}) == []


@pytest.mark.parametrize(
    "name,reqs",
    [("1v4p_zn_A1001", REQ_1CYS3HIS), ("1kwg_zn_A806", REQ_4CYS), ("3tio_zn_A1", REQ_4HIS)],
)
def test_strict_mode_keeps_good_sites_in_pipeline(tmp_path, name, reqs):
    assert len(_run(tmp_path, name, reqs, {"enabled": True})) == 1
