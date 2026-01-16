import sys
from pathlib import Path
import yaml

# ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scrape_pdb.config import load_config
from scrape_pdb.models import Atom
from scrape_pdb.validation import passes_ligand_requirements


def make_atom(serial, atom_name, resname, element, x=0.0, y=0.0, z=0.0):
    raw = f"HETATM {serial} {atom_name} {resname}"
    return Atom(
        record="HETATM",
        serial=serial,
        atom_name=atom_name,
        altloc="",
        resname=resname,
        chain="A",
        resseq="1",
        icode="",
        x=x,
        y=y,
        z=z,
        occ=1.0,
        bfac=0.0,
        element=element,
        charge="",
        raw_line=raw,
    )


def test_passes_ligand_requirements_cys3_his1():
    neighbors = [
        make_atom(1, "SG", "CYS", "S"),
        make_atom(2, "SG", "CYS", "S"),
        make_atom(3, "SG", "CYS", "S"),
        make_atom(4, "ND1", "HIS", "N"),
        # extra ligands are allowed
        make_atom(5, "O", "HOH", "O"),
        make_atom(6, "OD1", "ASP", "O"),
        make_atom(7, "NZ", "LYS", "N"),
    ]

    reqs = [
        type("R", (), {"resname": "CYS", "atom_names": ["SG"], "min_count": 3})(),
        type("R", (), {"resname": "HIS", "atom_names": ["ND1", "NE2"], "min_count": 1})(),
    ]

    assert passes_ligand_requirements(neighbors, reqs) is True


def test_ligand_requirements_fail_when_missing_his():
    neighbors = [
        make_atom(1, "SG", "CYS", "S"),
        make_atom(2, "SG", "CYS", "S"),
        make_atom(3, "SG", "CYS", "S"),
        make_atom(5, "O", "HOH", "O"),
    ]

    reqs = [
        type("R", (), {"resname": "CYS", "atom_names": ["SG"], "min_count": 3})(),
        type("R", (), {"resname": "HIS", "atom_names": ["ND1", "NE2"], "min_count": 1})(),
    ]

    assert passes_ligand_requirements(neighbors, reqs) is False


def test_config_parses_ligand_requirements(tmp_path):
    cfg = {
        "search_parameters": {"metal_ion": "ZN"},
        "processing": {"temp_directory": str(tmp_path / "temp")},
        "output": {
            "results_database": str(tmp_path / "results" / "clusters.csv"),
            "checkpoint_file": str(tmp_path / "results" / "checkpoint.db"),
            "log_file": str(tmp_path / "results" / "pipeline.log"),
            "output_dir": str(tmp_path / "results"),
        },
        "validation": {
            "ligand_requirements": [
                {"resname": "cys", "atom_names": ["sg"], "min_count": 3},
                {"resname": "his", "atom_names": ["nd1", "ne2"], "min_count": 1},
            ]
        },
    }

    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg))

    config = load_config(str(p))
    reqs = config.validation.ligand_requirements
    assert reqs is not None and len(reqs) == 2
    assert reqs[0].resname == "CYS"
    assert reqs[0].resnames == ["CYS"]
    assert reqs[0].atom_names == ["SG"]
    assert reqs[0].min_count == 3
    assert reqs[1].resname == "HIS"
    assert reqs[1].resnames == ["HIS"]
    assert reqs[1].atom_names == ["ND1", "NE2"]
    assert reqs[1].min_count == 1
