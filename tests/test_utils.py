import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scrape_pdb import utils


def test_slice_short_line():
    s = "ATOM\n"
    # requesting slice within padded area should not raise
    out = utils._slice(s, 0, 10)
    assert isinstance(out, str)
    assert len(out) == 10


def test_dist_and_centroid():
    a = (0.0, 0.0, 0.0)
    b = (1.0, 0.0, 0.0)
    assert abs(utils.dist(a, b) - 1.0) < 1e-9

    pts = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)]
    c = utils.centroid(pts)
    # centroid x = 2/3, y = 2/3, z = 0
    assert abs(c[0] - (2.0/3.0)) < 1e-9
    assert abs(c[1] - (2.0/3.0)) < 1e-9
    assert abs(c[2] - 0.0) < 1e-9

    # empty list centroid
    assert utils.centroid([]) == (0.0, 0.0, 0.0)


def test_connected_components_empty_and_single():
    assert utils.connected_components([], 3.0) == []
    pts = [(0.0,0.0,0.0)]
    assert utils.connected_components(pts, 3.0) == [[0]]


def test_connected_components_multiple():
    pts = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (5.0, 0.0, 0.0),
        (5.5, 0.0, 0.0),
    ]
    comps = utils.connected_components(pts, cutoff=1.5)
    # expect two components: [0,1] and [2,3]
    assert sorted([sorted(c) for c in comps]) == [[0,1],[2,3]]
