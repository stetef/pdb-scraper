import sys
from pathlib import Path
import csv
import json

# ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scrape_pdb.writer import write_xyz, ensure_csv_headers, write_clusters_csv_row, write_altloc_report_header, append_altloc_rows, append_cache
from scrape_pdb.models import Atom
from scrape_pdb.config import PipelineConfig


def make_atom(serial, atom_name, element, x=0.0, y=0.0, z=0.0, resname="RES", chain="A", resseq="1", icode=""):
    raw = f"ATOM      {serial} {atom_name} {resseq}"
    return Atom("HETATM", serial, atom_name, "", resname, chain, resseq, icode, x, y, z, 1.0, 0.0, element, "", raw)


def make_config(tmp_path):
    cfg = {
        "search_parameters": {"metal_ion": "ZN"},
        "processing": {
            "batch_size": 1,
            "parallel_workers": 1,
            "rate_limit_delay": 0.1,
            "temp_directory": str(tmp_path / "temp"),
            "cutoff": 3.0,
            "target": "ZN",
            "metals_excluded": [],
            "must_have": "",
            "include_waters": True,
            "max_downloads": None
        },
        "output": {
            "results_database": str(tmp_path / "clusters.csv"),
            "checkpoint_file": str(tmp_path / "checkpoint.db"),
            "log_file": str(tmp_path / "pipeline.log"),
            "save_matching_structures": False,
            "matched_structures_dir": str(tmp_path / "matched"),
            "output_dir": str(tmp_path)
        },
        "validation": {}
    }
    return PipelineConfig(**cfg)


def test_write_xyz_and_csv_and_altloc_and_cache(tmp_path):
    config = make_config(tmp_path)

    # Prepare atoms
    a1 = make_atom(1, "ZN", "ZN", x=0.0, y=0.0, z=0.0)
    # Neighbor residue for coordinating CYS 10
    a2 = make_atom(2, "N", "N", x=1.0, y=0.0, z=0.0, resname="ALA", resseq="9")
    a3 = make_atom(3, "CA", "C", x=1.5, y=0.5, z=0.0, resname="ALA", resseq="9")
    # Shared neighbor residue for coordinating CYS 10 and CYS 12
    a4 = make_atom(4, "N", "N", x=0.0, y=2.0, z=0.0, resname="GLY", resseq="11")
    a5 = make_atom(5, "C", "C", x=0.2, y=2.8, z=0.0, resname="GLY", resseq="11")
    # Neighbor residue for coordinating CYS 12
    a6 = make_atom(6, "O", "O", x=-1.2, y=3.3, z=0.0, resname="SER", resseq="13")
    # Coordinating residues themselves (not included in .pc under new behavior)
    a7 = make_atom(7, "SG", "S", x=0.0, y=0.0, z=3.0, resname="CYS", resseq="10")
    a8 = make_atom(8, "SG", "S", x=0.0, y=0.0, z=-3.0, resname="CYS", resseq="12")
    # Coordinating-residue backbone atoms inside 6 A must still be excluded
    a11 = make_atom(11, "N", "N", x=0.5, y=0.0, z=3.2, resname="CYS", resseq="10")
    a12 = make_atom(12, "C", "C", x=0.8, y=0.0, z=-3.1, resname="CYS", resseq="12")
    a13 = make_atom(13, "O", "O", x=1.0, y=0.2, z=-2.7, resname="CYS", resseq="12")
    # Non-neighbor residue inside 6 A from origin: should be included
    a9 = make_atom(9, "CB", "C", x=2.0, y=2.0, z=0.0, resname="VAL", resseq="20")
    # Non-neighbor residue outside 6 A from origin: should be excluded
    a10 = make_atom(10, "CG", "C", x=9.0, y=0.0, z=0.0, resname="VAL", resseq="21")
    atoms = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13]
    coord_residue_keys = {("CYS", "A", "10"), ("CYS", "A", "12")}

    # Write XYZ
    xyz_dir = tmp_path / "xyz_files"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    xyz_path = str(xyz_dir / "test.xyz")
    write_xyz(
        xyz_path,
        "1abc",
        1,
        "ZN",
        config.cutoff,
        "single_center",
        (0.0, 0.0, 0.0),
        atoms,
        a1,
        None,
        extra_comment="TEST",
        coord_residue_keys=coord_residue_keys,
        pc_source_atoms=atoms,
    )

    p = Path(xyz_path)
    assert p.exists()
    lines = p.read_text().splitlines()
    assert int(lines[0].strip()) == len(lines) - 2
    assert int(lines[0].strip()) >= 2
    assert "PDB=1abc" in lines[1]
    assert "CHARGE_UNROUNDED=" in lines[1]
    assert "CHARGE_ROUNDED=" in lines[1]
    assert "MULTIPLICITY=" in lines[1]

    h_lines = [ln for ln in lines[2:] if ln.startswith("H")]
    if h_lines:
        assert all("ATOM=H" in ln and "COORD=" in ln and "BONDEDATOM=" in ln for ln in h_lines)

    # Compact XYZ should contain only coordinating residues + Zn
    xyz_atom_lines = lines[2:]
    assert any("ATOM=ZN" in ln and "COORD=TRUE" in ln for ln in xyz_atom_lines)
    assert any("RESSEQ=10" in ln and "COORD=TRUE" in ln for ln in xyz_atom_lines)
    assert any("RESSEQ=12" in ln and "COORD=TRUE" in ln for ln in xyz_atom_lines)
    assert not any("RESSEQ=9" in ln for ln in xyz_atom_lines)
    assert not any("RESSEQ=11" in ln for ln in xyz_atom_lines)
    assert not any("RESSEQ=13" in ln for ln in xyz_atom_lines)

    # Backbone N/C/O on coordinating residues should be excluded in compact XYZ.
    assert not any("RESSEQ=10" in ln and "ATOM=N" in ln for ln in xyz_atom_lines)
    assert not any("RESSEQ=12" in ln and "ATOM=C" in ln for ln in xyz_atom_lines)
    assert not any("RESSEQ=12" in ln and "ATOM=O" in ln for ln in xyz_atom_lines)

    # Extended file should include coordinating + +/-1 neighbor residues with backbone atoms.
    p_ext = p.with_name(f"{p.stem}-extended{p.suffix}")
    assert p_ext.exists()
    ext_lines = p_ext.read_text().splitlines()[2:]
    assert any("RESSEQ=9" in ln and "COORD=FALSE" in ln for ln in ext_lines)
    assert any("RESSEQ=11" in ln and "COORD=FALSE" in ln for ln in ext_lines)
    assert any("RESSEQ=13" in ln and "COORD=FALSE" in ln for ln in ext_lines)
    assert any("RESSEQ=10" in ln and "ATOM=N" in ln for ln in ext_lines)
    assert any("RESSEQ=12" in ln and "ATOM=C" in ln for ln in ext_lines)
    assert any("RESSEQ=12" in ln and "ATOM=O" in ln for ln in ext_lines)
    assert any("RESSEQ=9" in ln and "ATOM=N" in ln for ln in ext_lines)
    assert any("RESSEQ=11" in ln and "ATOM=C" in ln for ln in ext_lines)
    assert any("RESSEQ=13" in ln and "ATOM=O" in ln for ln in ext_lines)

    # Point-charge file should contain only +/-1 neighboring residues (including N/C/O).
    pc_path = Path(xyz_path).with_suffix(".pc")
    assert pc_path.exists()
    pc_lines = pc_path.read_text().splitlines()
    assert int(pc_lines[0]) >= 5
    parsed = [[float(v) for v in line.split()] for line in pc_lines[1:]]
    coords = {(x, y, z) for _, x, y, z in parsed}
    assert (1.0, 0.0, 0.0) in coords
    assert (1.5, 0.5, 0.0) in coords
    assert (0.0, 2.0, 0.0) in coords
    assert (0.2, 2.8, 0.0) in coords
    assert (-1.2, 3.3, 0.0) in coords
    # Coordinating residues should never appear in point charges.
    assert (0.5, 0.0, 3.2) not in coords
    assert (0.8, 0.0, -3.1) not in coords

    # CSV headers and row
    ensure_csv_headers(config)
    clusters_path = config.clusters_csv
    assert clusters_path.exists()
    # write a row
    row = ["1abc", 1, "homo", 1, "ZN", "ZN", "A", "1", "", "RES", "", 1, "TD", "", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "No", "No", "No", "No", "", "", "", "test.xyz", "NA"]
    write_clusters_csv_row(row, config)
    # read back
    with open(clusters_path, newline="") as f:
        reader = list(csv.reader(f))
    assert any(r and r[0] == "1abc" for r in reader)

    # Altloc report header and append
    write_altloc_report_header(config)
    alt_path = config.altloc_report
    assert alt_path.exists()
    append_altloc_rows([["1abc","HETATM",1,"ZN","ZN","A","RES","A","1",0.0,0.0,0.0,1,"homo"]], config)
    with open(alt_path, newline="") as f:
        rows = list(csv.reader(f))
    assert any(r and r[0] == "1abc" for r in rows)

    # append_cache
    runs = [{"pdb_id": "1abc", "pdb_path": "path/to/1abc.pdb", "clusters": []}]
    append_cache(runs, config)
    cache_file = Path(config.cache)
    assert cache_file.exists()
    payload = json.loads(cache_file.read_text())
    assert "runs" in payload and any(r.get("pdb_id") == "1abc" for r in payload["runs"])


def test_pc_excludes_adjacent_coordinating_residues(tmp_path):
    """If coordinating residues are adjacent in sequence, they must not leak in via +/-1 logic."""
    config = make_config(tmp_path)

    zn = make_atom(1, "ZN", "ZN", x=0.0, y=0.0, z=0.0)
    # Coordinating residues at 10 and 11 (adjacent)
    c10_n = make_atom(2, "N", "N", x=0.5, y=0.0, z=2.0, resname="CYS", resseq="10")
    c10_sg = make_atom(3, "SG", "S", x=0.2, y=0.0, z=2.5, resname="CYS", resseq="10")
    c11_c = make_atom(4, "C", "C", x=-0.5, y=0.0, z=2.1, resname="HIS", resseq="11")
    c11_nd1 = make_atom(5, "ND1", "N", x=-0.3, y=0.2, z=2.8, resname="HIS", resseq="11")
    # True neighbors outside coordinating set
    r9_ca = make_atom(6, "CA", "C", x=1.5, y=0.0, z=0.0, resname="ALA", resseq="9")
    r12_o = make_atom(7, "O", "O", x=-1.5, y=0.0, z=0.0, resname="SER", resseq="12")
    atoms = [zn, c10_n, c10_sg, c11_c, c11_nd1, r9_ca, r12_o]

    xyz_dir = tmp_path / "xyz_files"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    xyz_path = str(xyz_dir / "adjacent.xyz")

    write_xyz(
        xyz_path,
        "2abc",
        1,
        "ZN",
        config.cutoff,
        "single_center",
        (0.0, 0.0, 0.0),
        atoms,
        zn,
        None,
        coord_residue_keys={("CYS", "A", "10"), ("HIS", "A", "11")},
        pc_source_atoms=atoms,
    )

    pc_lines = Path(xyz_path).with_suffix(".pc").read_text().splitlines()
    parsed = [[float(v) for v in line.split()] for line in pc_lines[1:]]
    coords = {(x, y, z) for _, x, y, z in parsed}

    # Neighbor residues should be included.
    assert (1.5, 0.0, 0.0) in coords
    assert (-1.5, 0.0, 0.0) in coords
    # Coordinating residues must be excluded even though each is +/-1 neighbor of the other.
    assert (0.5, 0.0, 2.0) not in coords
    assert (0.2, 0.0, 2.5) not in coords
    assert (-0.5, 0.0, 2.1) not in coords
    assert (-0.3, 0.2, 2.8) not in coords


def test_pc_includes_only_plus_minus_one_neighbor_residues(tmp_path):
    """Point charges should only include +/-1 neighboring residues for the compact QM file."""
    config = make_config(tmp_path)

    zn = make_atom(1, "ZN", "ZN", x=0.0, y=0.0, z=0.0)
    # Coordinating residues
    c10_sg = make_atom(2, "SG", "S", x=0.0, y=0.0, z=2.2, resname="CYS", resseq="10")
    c12_nd1 = make_atom(3, "ND1", "N", x=0.0, y=0.0, z=-2.3, resname="HIS", resseq="12")
    # +/-1 neighbors that should be present in XYZ
    r9_ca = make_atom(4, "CA", "C", x=1.8, y=0.0, z=0.0, resname="ALA", resseq="9")
    r11_o = make_atom(5, "O", "O", x=-1.8, y=0.0, z=0.0, resname="SER", resseq="11")
    r13_cb = make_atom(6, "CB", "C", x=0.0, y=1.8, z=0.0, resname="VAL", resseq="13")
    # +/-2 neighbors should not be eligible for .pc under the new neighbor-only rule.
    r8_ca = make_atom(7, "CA", "C", x=3.2, y=2.2, z=0.0, resname="GLY", resseq="8")
    r14_n = make_atom(8, "N", "N", x=-3.2, y=-2.2, z=0.0, resname="GLY", resseq="14")

    selected_atoms = [zn, c10_sg, c12_nd1, r9_ca, r11_o, r13_cb]
    all_atoms = selected_atoms + [r8_ca, r14_n]

    xyz_dir = tmp_path / "xyz_files"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    xyz_path = str(xyz_dir / "non_overlap.xyz")

    write_xyz(
        xyz_path,
        "3abc",
        1,
        "ZN",
        config.cutoff,
        "single_center",
        (0.0, 0.0, 0.0),
        selected_atoms,
        zn,
        None,
        coord_residue_keys={("CYS", "A", "10"), ("HIS", "A", "12")},
        pc_source_atoms=all_atoms,
    )

    xyz_lines = Path(xyz_path).read_text().splitlines()[2:]
    # Ensure +/-2 residues were not pulled into compact XYZ.
    assert not any("RESSEQ=8" in ln for ln in xyz_lines)
    assert not any("RESSEQ=14" in ln for ln in xyz_lines)

    pc_lines = Path(xyz_path).with_suffix(".pc").read_text().splitlines()
    assert int(pc_lines[0]) == 3
    parsed = [[float(v) for v in line.split()] for line in pc_lines[1:]]
    coords = {(x, y, z) for _, x, y, z in parsed}

    assert (1.8, 0.0, 0.0) in coords
    assert (-1.8, 0.0, 0.0) in coords
    assert (0.0, 1.8, 0.0) in coords
    assert (3.2, 2.2, 0.0) not in coords
    assert (-3.2, -2.2, 0.0) not in coords


def test_pc_does_not_include_plus_minus_two_residues(tmp_path):
    """Residues at RESSEQ +/-2 should not be used in point charges."""
    config = make_config(tmp_path)

    zn = make_atom(1, "ZN", "ZN", x=0.0, y=0.0, z=0.0)
    c10_sg = make_atom(2, "SG", "S", x=0.0, y=0.0, z=2.1, resname="CYS", resseq="10")
    c12_nd1 = make_atom(3, "ND1", "N", x=0.0, y=0.0, z=-2.1, resname="HIS", resseq="12")
    r9_ca = make_atom(4, "CA", "C", x=1.7, y=0.0, z=0.0, resname="ALA", resseq="9")
    r11_ca = make_atom(5, "CA", "C", x=-1.7, y=0.0, z=0.0, resname="SER", resseq="11")
    r13_ca = make_atom(6, "CA", "C", x=0.0, y=1.7, z=0.0, resname="VAL", resseq="13")

    # RESSEQ 8 is outside +/-1 neighborhood and must not be included.
    r8_n_bonded = make_atom(7, "N", "N", x=2.9, y=0.0, z=0.0, resname="GLY", resseq="8")
    r8_o_free = make_atom(8, "O", "O", x=4.8, y=0.0, z=0.0, resname="GLY", resseq="8")

    selected_atoms = [zn, c10_sg, c12_nd1, r9_ca, r11_ca, r13_ca]
    all_atoms = selected_atoms + [r8_n_bonded, r8_o_free]

    xyz_dir = tmp_path / "xyz_files"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    xyz_path = str(xyz_dir / "bond_exclusion.xyz")

    write_xyz(
        xyz_path,
        "4abc",
        1,
        "ZN",
        config.cutoff,
        "single_center",
        (0.0, 0.0, 0.0),
        selected_atoms,
        zn,
        None,
        coord_residue_keys={("CYS", "A", "10"), ("HIS", "A", "12")},
        pc_source_atoms=all_atoms,
    )

    pc_lines = Path(xyz_path).with_suffix(".pc").read_text().splitlines()
    parsed = [[float(v) for v in line.split()] for line in pc_lines[1:]]
    coords = {(x, y, z) for _, x, y, z in parsed}

    assert (2.9, 0.0, 0.0) not in coords
    assert (4.8, 0.0, 0.0) not in coords
