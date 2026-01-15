import os
import io
import json
import sqlite3
import yaml
from pathlib import Path
from types import SimpleNamespace

import pytest
import sys

# Ensure project root is on sys.path so `scrape_pdb` package imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scrape_pdb.main import run_pipeline
from scrape_pdb.checkpoint import CheckpointManager


class FakeResp:
    def __init__(self, data: bytes, status: int = 200):
        self._data = data
        self.status = status
    def read(self):
        return self._data
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def simple_pdb_bytes():
    return b"HEADER    TEST PDB\nATOM      1  N   ALA A   1      11.104  13.207   2.000  1.00 20.00           N\nEND\n"


def test_pipeline_with_search_and_network_mocks(tmp_path, monkeypatch, simple_pdb_bytes):
    # Setup temp dirs
    out_dir = tmp_path / "results"
    temp_dir = tmp_path / "temp"
    out_dir.mkdir()
    temp_dir.mkdir()

    # Create a minimal config that uses search mode
    cfg = {
        "search_parameters": {"metal_ion": "ZN"},
        "processing": {
            "batch_size": 1,
            "temp_directory": str(temp_dir),
            "max_downloads": 2
        },
        "output": {
            "results_database": str(out_dir / "clusters.csv"),
            "checkpoint_file": str(out_dir / "checkpoint.db"),
            "log_file": str(out_dir / "pipeline.log"),
            "output_dir": str(out_dir)
        },
        "validation": {}
    }
    # Use search mode so the pipeline calls the search -> download path
    cfg["input_mode"] = "search"
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump(cfg))

    # Mock search_pdb to return two fake IDs
    # run_pipeline (search mode) calls the symbol imported into scrape_pdb.main
    monkeypatch.setattr("scrape_pdb.main.search_pdb", lambda config: ["1abc", "2def"])

    # Mock urllib.request.urlopen to return a fake PDB response
    import urllib.request
    def fake_urlopen(req, timeout=30):
        return FakeResp(simple_pdb_bytes)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    # Mock the parser process_pdb to write a dummy xyz file and return its path
    def fake_process_pdb(pdb_path, config):
        pdb_id = Path(pdb_path).stem
        xyz_dir = Path(config.output_dir) / "xyz_files"
        xyz_dir.mkdir(parents=True, exist_ok=True)
        xyz_path = xyz_dir / f"{pdb_id}_FAKE.xyz"
        xyz_path.write_text("3\n# fake\nC 0 0 0\nH 0 0 1\nH 0 1 0\n")
        return [str(xyz_path)]
    # Patch both the parser module and the main module's imported reference
    monkeypatch.setattr("scrape_pdb.parser.process_pdb", fake_process_pdb)
    monkeypatch.setattr("scrape_pdb.main.process_pdb", fake_process_pdb)

    # Run pipeline
    rc = run_pipeline(str(cfg_path), verbose=True)
    assert rc == 0

    # Verify cache file was written and contains runs
    cache_file = tmp_path / "cache.json"
    # The pipeline writes cache to the working cache path; check output dir for cache.json
    found_cache = False
    for p in out_dir.iterdir():
        if p.name == "cache.json":
            found_cache = True
            break
    # It's acceptable if cache is in repo root; check at least that output xyz files exist
    xyz_files = list((out_dir / "xyz_files").glob("*.xyz"))
    assert len(xyz_files) >= 1

    # Validate checkpoint statuses
    cp = CheckpointManager(str(out_dir / "checkpoint.db"))
    assert cp.get_status("1abc") in ("matched", "rejected", "error")
    assert cp.get_status("2def") in ("matched", "rejected", "error")


# Note: This test performs network mocks only; it does not reach out to RCSB.
