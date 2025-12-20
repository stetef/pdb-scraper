import sys
from pathlib import Path

# ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scrape_pdb.altloc import (
    group_by_id,
    altloc_set_for_atom_id,
    build_altloc_files_for_center,
)
from scrape_pdb.models import Atom, MustHaveSpec


def make_atom(serial, atom_name, altloc, chain="A", resseq="1", icode="", x=0.0, y=0.0, z=0.0, element="N"):
    raw = f"ATOM      {serial} {atom_name} {resseq}"
    return Atom("HETATM", serial, atom_name, altloc, "RES", chain, resseq, icode, x, y, z, 1.0, 0.0, element, "", raw)


def test_group_by_id_and_altloc_set():
    a1 = make_atom(1, "CA", "A", chain="A", resseq="10")
    a2 = make_atom(2, "CA", "B", chain="A", resseq="10")
    a3 = make_atom(3, "CA", "", chain="A", resseq="10")
    a4 = make_atom(4, "CB", "", chain="A", resseq="10")

    atoms = [a1, a2, a3, a4]
    G = group_by_id(atoms)
    key = ("A", "10", "", "CA")
    assert key in G
    assert set(G[key].keys()) == {"A", "B", ""}

    labs_ca = altloc_set_for_atom_id(G, key)
    assert labs_ca == {"A", "B"}

    key_cb = ("A", "10", "", "CB")
    assert altloc_set_for_atom_id(G, key_cb) == set()


def test_build_altloc_case_3(tmp_path, monkeypatch):
    # Situation 3: center unlabeled, neighbors labeled
    center = make_atom(10, "ZN", "", x=0.0, y=0.0, z=0.0, element="ZN")
    # neighbors with labels A and B
    n1 = make_atom(11, "SG", "A", x=1.0, y=0.0, z=0.0, element="S", resseq="2")
    n2 = make_atom(12, "SG", "B", x=1.0, y=1.0, z=0.0, element="S", resseq="3")

    selected = [n1, n2]

    # raw_groups mapping
    G = {
        (n1.chain, n1.resseq, n1.icode, n1.atom_name): {"A": n1},
        (n2.chain, n2.resseq, n2.icode, n2.atom_name): {"B": n2},
        (center.chain, center.resseq, center.icode, center.atom_name): {"": center},
    }

    calls = []

    def fake_write_xyz(path, pdb_id, cluster_index, target, cutoff, origin_kind, centroid_pt, atoms, origin_atom, resolution_angs, extra_comment=""):
        calls.append((path, pdb_id, extra_comment, [a.serial for a in atoms]))

    monkeypatch.setattr("scrape_pdb.altloc.write_xyz", fake_write_xyz)
    # make must_have permissive
    must = MustHaveSpec()

    out = build_altloc_files_for_center(
        pdb_id="1abc",
        center=center,
        selected_atoms_raw=selected,
        raw_groups=G,
        cutoff=5.0,
        out_dir=tmp_path,
        base_name_common="base",
        origin_kind="single_center",
        centroid_pt=(0.0, 0.0, 0.0),
        resolution_angs=None,
        cluster_index=1,
        cluster_type="homo",
        target_upper="ZN",
        include_waters=True,
        must_have=must,
        csv_common={},
        metals_in_comp=[center],
    )

    # Should have created one file per neighbor label (A and B)
    assert len(calls) == 2
    paths = [c[0] for c in calls]
    assert any("altlocLIGA" in p for p in paths)
    assert any("altlocLIGB" in p for p in paths)


def test_build_altloc_case_1_and_2(tmp_path, monkeypatch):
    # Situation 1: center labeled A/B and all neighbors have same labels
    cA = make_atom(20, "ZN", "A", x=0.0, y=0.0, z=0.0, element="ZN")
    cB = make_atom(21, "ZN", "B", x=0.0, y=0.0, z=0.0, element="ZN")
    center = cA  # choose one representative

    # neighbors: both have A and B
    n1A = make_atom(22, "SG", "A", x=1.0, y=0.0, z=0.0, element="S")
    n1B = make_atom(23, "SG", "B", x=1.0, y=0.0, z=0.0, element="S")
    n2A = make_atom(24, "OG", "A", x=1.0, y=1.0, z=0.0, element="O")
    n2B = make_atom(25, "OG", "B", x=1.0, y=1.0, z=0.0, element="O")

    selected = [n1A, n1B, n2A, n2B]

    G = {
        (n1A.chain, n1A.resseq, n1A.icode, n1A.atom_name): {"A": n1A, "B": n1B},
        (n2A.chain, n2A.resseq, n2A.icode, n2A.atom_name): {"A": n2A, "B": n2B},
        (cA.chain, cA.resseq, cA.icode, cA.atom_name): {"A": cA, "B": cB},
    }

    calls = []

    def fake_write_xyz(path, pdb_id, cluster_index, target, cutoff, origin_kind, centroid_pt, atoms, origin_atom, resolution_angs, extra_comment=""):
        calls.append((path, extra_comment, [a.serial for a in atoms]))

    monkeypatch.setattr("scrape_pdb.altloc.write_xyz", fake_write_xyz)
    must = MustHaveSpec()

    out = build_altloc_files_for_center(
        pdb_id="1def",
        center=cA,
        selected_atoms_raw=selected,
        raw_groups=G,
        cutoff=5.0,
        out_dir=tmp_path,
        base_name_common="base2",
        origin_kind="single_center",
        centroid_pt=(0.0, 0.0, 0.0),
        resolution_angs=None,
        cluster_index=2,
        cluster_type="multi_homo",
        target_upper="ZN",
        include_waters=True,
        must_have=must,
        csv_common={},
        metals_in_comp=[cA, cB],
    )

    # Should write one per label A and B
    assert len(calls) == 2
    assert any("ALTLOC_CASE=1" in extra for (_, extra, _) in calls)

    # Situation 2: neighbors may be unlabeled; include unlabeled neighbors each time
    # Make neighbor 2 unlabeled for this test
    n2_unlabeled = make_atom(30, "OG", "", x=1.0, y=1.0, z=0.0, element="O")
    G2 = dict(G)
    G2[(n2A.chain, n2A.resseq, n2A.icode, n2A.atom_name)] = {"": n2_unlabeled, "A": n2A}

    calls2 = []
    monkeypatch.setattr("scrape_pdb.altloc.write_xyz", lambda *args, **kwargs: calls2.append(args[0]))

    out2 = build_altloc_files_for_center(
        pdb_id="1ghi",
        center=cA,
        selected_atoms_raw=[n1A, n1B, n2A, n2_unlabeled],
        raw_groups=G2,
        cutoff=5.0,
        out_dir=tmp_path,
        base_name_common="base3",
        origin_kind="single_center",
        centroid_pt=(0.0, 0.0, 0.0),
        resolution_angs=None,
        cluster_index=3,
        cluster_type="multi_homo",
        target_upper="ZN",
        include_waters=True,
        must_have=must,
        csv_common={},
        metals_in_comp=[cA, cB],
    )

    assert len(calls2) >= 1
