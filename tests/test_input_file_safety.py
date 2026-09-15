"""P1.10 (04 §2): the pipeline must never delete or overwrite caller-supplied
input files. It may only clean files it downloaded itself into its own
download_dir this run. Regression for the P1.9 hazard, plus a containment unit
test for the deletion-safety helper.
"""

import hashlib
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scrape_pdb.main import (  # noqa: E402
    run_pipeline,
    _is_within_dir,
    _pipeline_may_remove,
    _resolve_path,
)

DATA = Path(__file__).resolve().parent / "data"
FIXTURE_PDB = DATA / "5xp6_zn_site.pdb"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(tmp_path: Path, *, input_mode: str, input_data: list[str], save_matching: bool) -> Path:
    cfg = {
        "search_parameters": {
            "metal_ion": "ZN",
            "resolution_cutoff": 3.0,
            "experimental_method": "X-RAY DIFFRACTION",
            "polymer_type": "Protein",
        },
        "processing": {
            "batch_size": 5,
            "parallel_workers": 1,
            "rate_limit_delay": 0.0,
            "temp_directory": str(tmp_path / "downloads"),
            "cutoff": 6.0,
            "target": "ZN",
            "metals_excluded": [],
            "max_downloads": 10,
            "must_have": "S",
            "include_waters": True,
        },
        "output": {
            "results_database": str(tmp_path / "results" / "clusters.csv"),
            "checkpoint_file": str(tmp_path / "results" / "checkpoint.db"),
            "log_file": str(tmp_path / "results" / "pipeline.log"),
            "save_matching_structures": save_matching,
            "matched_structures_dir": str(tmp_path / "results" / "matched"),
            "output_dir": str(tmp_path / "results"),
        },
        "validation": {"coordination_distance_max": 2.8, "coordination_distance_min": 2.0},
        "input_mode": input_mode,
        "input_data": input_data,
        "log_level": "INFO",
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


def test_rejected_user_input_in_mixed_mode_is_byte_identical(monkeypatch, tmp_path):
    """The P1.9 hazard: a user PDB passed in mixed mode and rejected must survive."""
    user_pdb = tmp_path / "input" / "my_structure.pdb"
    user_pdb.parent.mkdir(parents=True)
    shutil.copy2(FIXTURE_PDB, user_pdb)
    before = _sha(user_pdb)

    # Force a rejection (no clusters) without depending on the parser.
    monkeypatch.setattr("scrape_pdb.main.process_pdb", lambda pdb_path, config: [])

    cfg = _config(tmp_path, input_mode="mixed", input_data=[str(user_pdb)], save_matching=False)
    run_pipeline(str(cfg), verbose=False)

    assert user_pdb.exists(), "the pipeline deleted a caller-supplied input"
    assert _sha(user_pdb) == before, "the pipeline modified a caller-supplied input"


def test_matched_user_input_is_copied_not_moved(monkeypatch, tmp_path):
    """A matched user input with save_matching_structures=True is copied, not moved away."""
    user_pdb = tmp_path / "input" / "keep_me.pdb"
    user_pdb.parent.mkdir(parents=True)
    shutil.copy2(FIXTURE_PDB, user_pdb)
    before = _sha(user_pdb)

    monkeypatch.setattr("scrape_pdb.main.process_pdb", lambda pdb_path, config: ["dummy.xyz"])

    cfg = _config(tmp_path, input_mode="mixed", input_data=[str(user_pdb)], save_matching=True)
    run_pipeline(str(cfg), verbose=False)

    assert user_pdb.exists() and _sha(user_pdb) == before, "matched input was moved/altered"
    assert (tmp_path / "results" / "matched" / "keep_me.pdb").exists(), "no kept copy was made"


def test_downloaded_file_is_still_cleaned(monkeypatch, tmp_path):
    """Files the pipeline downloads into its own download_dir are still removed."""
    download_dir = tmp_path / "downloads"
    download_dir.mkdir(parents=True)

    def fake_fetch(pdb_id, dest_dir):
        dest = Path(dest_dir) / f"{pdb_id.lower()}.pdb"
        shutil.copy2(FIXTURE_PDB, dest)  # the pipeline "downloaded" it here
        return str(dest)

    monkeypatch.setattr("scrape_pdb.downloader.fetch_pdb", fake_fetch)
    monkeypatch.setattr("scrape_pdb.main.process_pdb", lambda pdb_path, config: [])  # rejected

    cfg = _config(tmp_path, input_mode="ids", input_data=["5xp6"], save_matching=False)
    run_pipeline(str(cfg), verbose=False)

    assert not (download_dir / "5xp6.pdb").exists(), "a downloaded file was not cleaned up"


def test_pipeline_may_remove_containment(tmp_path):
    """The deletion-safety helper: removable only if created AND inside download_dir."""
    download = tmp_path / "downloads"
    download.mkdir()
    downloaded = download / "1abc.pdb"
    downloaded.write_text("x")
    user_input = tmp_path / "elsewhere" / "user.pdb"
    user_input.parent.mkdir()
    user_input.write_text("x")
    # A user file that happens to sit inside download_dir but was NOT created by us.
    intruder = download / "user_owned.pdb"
    intruder.write_text("x")

    created = {_resolve_path(downloaded)}

    # created + inside download_dir → removable
    assert _pipeline_may_remove(downloaded, download, created) is True
    # user input: not created and outside → not removable
    assert _pipeline_may_remove(user_input, download, created) is False
    # inside download_dir but not created → still NOT removable (both conditions)
    assert _pipeline_may_remove(intruder, download, created) is False
    # created set membership without containment → not removable
    assert _pipeline_may_remove(user_input, download, {_resolve_path(user_input)}) is False

    assert _is_within_dir(downloaded, download) is True
    assert _is_within_dir(user_input, download) is False
    assert _is_within_dir(download, download) is False  # not itself
