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
    a2 = make_atom(2, "SG", "S", x=1.0, y=0.0, z=0.0)
    atoms = [a1, a2]

    # Write XYZ
    xyz_dir = tmp_path / "xyz_files"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    xyz_path = str(xyz_dir / "test.xyz")
    write_xyz(xyz_path, "1abc", 1, "ZN", config.cutoff, "single_center", (0.0,0.0,0.0), atoms, a1, None, extra_comment="TEST")

    p = Path(xyz_path)
    assert p.exists()
    lines = p.read_text().splitlines()
    assert int(lines[0].strip()) == len(lines) - 2
    assert int(lines[0].strip()) >= len(atoms)
    assert "PDB=1abc" in lines[1]

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
