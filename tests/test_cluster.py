import sys
from pathlib import Path

# ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scrape_pdb.cluster import (
    determine_cluster_type,
    select_neighbors_union,
    select_neighbors_from,
    apply_water_toggle,
)
from scrape_pdb.models import Atom


def make_atom(serial, atom_name, element, x=0.0, y=0.0, z=0.0, resname="RES", chain="A", resseq="1", icode=""):
    raw = f"ATOM      {serial} {atom_name} {resseq}"
    return Atom("HETATM", serial, atom_name, "", resname, chain, resseq, icode, x, y, z, 1.0, 0.0, element, "", raw)


def test_determine_cluster_type():
    # single metal -> homo
    m1 = make_atom(1, "ZN", "ZN")
    assert determine_cluster_type([m1], "ZN", "ZN") == "homo"

    # two metals same element -> multi_homo
    m2 = make_atom(2, "ZN", "ZN")
    assert determine_cluster_type([m1, m2], "ZN", "ZN") == "multi_homo"

    # two metals different element -> multi_hetero
    m3 = make_atom(3, "NI", "NI")
    assert determine_cluster_type([m1, m3], "ZN", "ZN") == "multi_hetero"

    # fallback to atom_name when target_element is None
    a1 = make_atom(4, "NI", "NI")
    a2 = make_atom(5, "NI", "NI")
    assert determine_cluster_type([a1, a2], "NI", None) == "multi_homo"


def test_select_neighbors_union_and_from():
    center1 = make_atom(10, "ZN", "ZN", x=0.0, y=0.0, z=0.0)
    center2 = make_atom(11, "ZN", "ZN", x=10.0, y=0.0, z=0.0)
    # atoms near centers
    a1 = make_atom(20, "CA", "C", x=1.0, y=0.0, z=0.0)
    a2 = make_atom(21, "CA", "C", x=9.5, y=0.0, z=0.0)
    a3 = make_atom(22, "CA", "C", x=5.0, y=0.0, z=0.0)

    union = select_neighbors_union([a1, a2, a3], [center1, center2], cutoff=2.0)
    # a1 and a2 within 2A of centers, a3 not
    ids = {a.serial for a in union}
    assert 20 in ids and 21 in ids and 22 not in ids

    # select_neighbors_from
    near = select_neighbors_from(center1, [a1, a2, a3], cutoff=2.0)
    ids2 = {a.serial for a in near}
    assert 20 in ids2 and 21 not in ids2


def test_apply_water_toggle():
    h = make_atom(30, "H1", "H")
    oxy = make_atom(31, "O1", "O", resname="HOH")
    carbon = make_atom(32, "C1", "C")

    # include_waters True: H filtered, water kept
    res1 = apply_water_toggle([h, oxy, carbon], include_waters=True)
    ser = {a.serial for a in res1}
    assert 31 in ser and 32 in ser and 30 not in ser

    # include_waters False: H filtered, water removed
    res2 = apply_water_toggle([h, oxy, carbon], include_waters=False)
    ser2 = {a.serial for a in res2}
    assert 32 in ser2 and 31 not in ser2 and 30 not in ser2
