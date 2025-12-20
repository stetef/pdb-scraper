import sys
from pathlib import Path

# ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scrape_pdb.geometry import classify_geometry, coord_string, HAVE_NUMPY
from scrape_pdb.models import Atom


def make_atom(serial, atom_name, element, x=0.0, y=0.0, z=0.0):
    raw = f"ATOM      {serial} {atom_name} 1"
    return Atom("HETATM", serial, atom_name, "", "RES", "A", "1", "", x, y, z, 1.0, 0.0, element, "", raw)


def test_coord_string_simple():
    a1 = make_atom(1, "N", "N")
    a2 = make_atom(2, "O", "O")
    a3 = make_atom(3, "N", "N")
    s = coord_string([a1, a2, a3])
    assert s == "2N1O"


def test_classify_linear_and_bent():
    center = make_atom(10, "ZN", "ZN", 0.0, 0.0, 0.0)
    # linear neighbors at opposite directions
    n1 = make_atom(11, "L1", "N", 1.0, 0.0, 0.0)
    n2 = make_atom(12, "L2", "N", -1.0, 0.0, 0.0)
    geom, metrics, flags = classify_geometry(center, [n1, n2])
    assert geom.startswith("LIN")

    # bent neighbors (angle < 160 -> BEN)
    b1 = make_atom(21, "B1", "N", 1.0, 0.0, 0.0)
    b2 = make_atom(22, "B2", "N", 0.5, 0.5, 0.0)
    geom2, metrics2, flags2 = classify_geometry(center, [b1, b2])
    assert geom2.startswith("BEN")


def test_classify_octahedral_and_jt():
    center = make_atom(30, "ZN", "ZN", 0.0, 0.0, 0.0)
    neighbors = [
        make_atom(31, "A", "N", 1.0, 0.0, 0.0),
        make_atom(32, "B", "N", -1.0, 0.0, 0.0),
        make_atom(33, "C", "N", 0.0, 1.0, 0.0),
        make_atom(34, "D", "N", 0.0, -1.0, 0.0),
        make_atom(35, "E", "N", 0.0, 0.0, 1.0),
        make_atom(36, "F", "N", 0.0, 0.0, -1.0),
    ]
    geom, metrics, flags = classify_geometry(center, neighbors)
    assert geom.startswith("OH")
    assert flags.get("JT") in ("No", "Yes")

    # introduce Jahn-Teller elongation by moving one ligand out
    neighbors_jt = neighbors.copy()
    neighbors_jt[0] = make_atom(31, "A", "N", 1.3, 0.0, 0.0)
    geom2, metrics2, flags2 = classify_geometry(center, neighbors_jt)
    assert flags2.get("JT") == "Yes"


def test_planar_trigonal():
    center = make_atom(40, "ZN", "ZN", 0.0, 0.0, 0.0)
    # three neighbors in XY plane at 120 deg
    n1 = make_atom(41, "N1", "N", 1.0, 0.0, 0.0)
    n2 = make_atom(42, "N2", "N", -0.5, 0.8660254, 0.0)
    n3 = make_atom(43, "N3", "N", -0.5, -0.8660254, 0.0)
    geom, metrics, flags = classify_geometry(center, [n1, n2, n3])
    # Should detect planar geometry (TP)
    assert flags["planar"] == "Yes"
    assert geom.startswith("TP")
