"""Search-mode download outages must not end the run as "No more structures".

A batch in which every fetch fails is retried with exponential backoff
(DOWNLOAD_RETRIES, DOWNLOAD_BACKOFF_S); IDs still failing are marked
download_failed and the run moves on. MAX_FAILED_BATCHES fully-failed batches
in a row abort the run with an error. Partially failed batches are not retried.
"""

import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scrape_pdb.main as main
from scrape_pdb.checkpoint import CheckpointManager
from scrape_pdb.main import run_pipeline

BATCH_SIZE = 3


def make_config_dict(tmp_path: Path) -> dict:
    return {
        "search_parameters": {"metal_ion": "ZN"},
        "processing": {
            "batch_size": BATCH_SIZE,
            "parallel_workers": 1,
            "rate_limit_delay": 0.0,
            "temp_directory": str(tmp_path / "downloads"),
            "cutoff": 3.0,
            "target": "ZN",
            "metals_excluded": [],
            "max_downloads": 1000,
            "must_have": "",
            "include_waters": True,
        },
        "output": {
            "results_database": str(tmp_path / "results" / "clusters.csv"),
            "checkpoint_file": str(tmp_path / "results" / "checkpoint.db"),
            "log_file": str(tmp_path / "results" / "pipeline.log"),
            "save_matching_structures": False,
            "output_dir": str(tmp_path / "results"),
        },
        "validation": {"coordination_distance_max": 2.8, "coordination_distance_min": 2.0},
        "input_mode": "search",
        "input_data": [],
        "log_level": "INFO",
    }


def batches(n_batches: int) -> list[list[str]]:
    """PDB-like ids grouped by the batch the pipeline will fetch them in."""
    return [[f"{b}x{i:02d}" for i in range(BATCH_SIZE)] for b in range(1, n_batches + 1)]


class Harness:
    """Mocked search/fetch/process/sleep around one run_pipeline call.

    `fails(pdb_id, attempt)` decides whether the attempt-th fetch (0-based) of an
    id fails. Every downloaded structure is "rejected" by process_pdb, so the run
    never reaches max_downloads and only runs out of ids or aborts.
    """

    def __init__(self, monkeypatch, tmp_path, ids, fails):
        self.cfg = make_config_dict(tmp_path)
        self.cfg_path = tmp_path / "cfg.yaml"
        self.cfg_path.write_text(yaml.safe_dump(self.cfg))
        Path(self.cfg["processing"]["temp_directory"]).mkdir(parents=True, exist_ok=True)

        self.fetch_attempts: Counter[str] = Counter()
        self.processed: list[str] = []
        self.sleeps: list[float] = []

        def fake_fetch(pdb_id, dest_dir):
            attempt = self.fetch_attempts[pdb_id]
            self.fetch_attempts[pdb_id] += 1
            if fails(pdb_id, attempt):
                return None
            p = Path(dest_dir) / f"{pdb_id}.pdb"
            p.write_text("REMARK   2 RESOLUTION.    1.80 ANGSTROM.\nATOM\n")
            return str(p)

        def fake_process(pdb_path, config):
            self.processed.append(Path(pdb_path).stem)
            return []

        monkeypatch.setattr("scrape_pdb.main.search_pdb", lambda cfg: list(ids))
        monkeypatch.setattr("scrape_pdb.main.fetch_pdb", fake_fetch)
        monkeypatch.setattr("scrape_pdb.main.process_pdb", fake_process)
        monkeypatch.setattr("scrape_pdb.main.time.sleep", self.sleeps.append)

    def run(self) -> int:
        return run_pipeline(str(self.cfg_path), verbose=False)

    def status(self, pdb_id):
        return CheckpointManager(self.cfg["output"]["checkpoint_file"]).get_status(pdb_id)

    @property
    def log(self) -> str:
        return Path(self.cfg["output"]["log_file"]).read_text()


def backoffs(n_retries: int) -> list[float]:
    return [main.DOWNLOAD_BACKOFF_S * 2 ** k for k in range(n_retries)]


def test_constants():
    assert main.DOWNLOAD_RETRIES == 3
    assert main.DOWNLOAD_BACKOFF_S == 30.0
    assert main.MAX_FAILED_BATCHES == 5


def test_transient_outage_is_retried(monkeypatch, tmp_path):
    """Batch 2 fails completely once, then succeeds -> every id is processed."""
    b = batches(4)
    h = Harness(monkeypatch, tmp_path, sum(b, []), lambda pid, attempt: pid in b[1] and attempt == 0)
    assert h.run() == 0

    assert h.sleeps == [30.0]
    assert sorted(h.processed) == sorted(sum(b, []))
    for pid in sum(b, []):
        assert h.status(pid) == "rejected"
    assert all(h.fetch_attempts[pid] == 2 for pid in b[1])
    assert "Aborting" not in h.log


def test_permanent_batch_failure_moves_on(monkeypatch, tmp_path):
    """Batch 2 never downloads -> marked download_failed; batch 3 is still processed."""
    b = batches(3)
    h = Harness(monkeypatch, tmp_path, sum(b, []), lambda pid, attempt: pid in b[1])
    assert h.run() == 0

    assert h.sleeps == backoffs(main.DOWNLOAD_RETRIES)  # 30, 60, 120
    for pid in b[1]:
        assert h.status(pid) == "download_failed"
        assert h.fetch_attempts[pid] == main.DOWNLOAD_RETRIES + 1
    for pid in b[0] + b[2]:
        assert h.status(pid) == "rejected"
    assert sorted(h.processed) == sorted(b[0] + b[2])
    assert "Aborting" not in h.log
    # The run ends because the search list is exhausted, after batch 3.
    assert "No more structures to download." in h.log


def test_sustained_outage_aborts(monkeypatch, tmp_path):
    """Every fetch fails -> abort after MAX_FAILED_BATCHES batches, with an error."""
    n_max = main.MAX_FAILED_BATCHES
    b = batches(n_max + 3)
    h = Harness(monkeypatch, tmp_path, sum(b, []), lambda pid, attempt: True)
    assert h.run() != 0

    assert h.processed == []
    assert h.sleeps == backoffs(main.DOWNLOAD_RETRIES) * n_max
    for pid in sum(b[:n_max], []):
        assert h.status(pid) == "download_failed"
    for pid in sum(b[n_max:], []):
        assert h.fetch_attempts[pid] == 0
        assert h.status(pid) is None
    assert "[ERROR]" in h.log and "Aborting" in h.log
    assert "Run aborted on a download outage" in h.log
    assert "No more structures to download." not in h.log


def test_success_resets_failed_batch_count(monkeypatch, tmp_path):
    """MAX_FAILED_BATCHES counts failures *in a row*: a good batch in between resets it."""
    n_bad = main.MAX_FAILED_BATCHES - 1
    b = batches(2 * n_bad + 2)
    good = {0 + n_bad, 2 * n_bad + 1}  # batch indices that download
    bad_ids = {pid for i, ids in enumerate(b) if i not in good for pid in ids}
    h = Harness(monkeypatch, tmp_path, sum(b, []), lambda pid, attempt: pid in bad_ids)
    assert h.run() == 0

    assert "Aborting" not in h.log
    assert sorted(h.processed) == sorted(pid for i in good for pid in b[i])


@pytest.mark.parametrize("failing_index", [0, 2])
def test_partial_batch_failure_is_not_retried(monkeypatch, tmp_path, failing_index):
    """One id in batch 2 fails -> no backoff; that id is download_failed, the rest are processed."""
    b = batches(3)
    bad = b[1][failing_index]
    h = Harness(monkeypatch, tmp_path, sum(b, []), lambda pid, attempt: pid == bad)
    assert h.run() == 0

    assert h.sleeps == []
    assert h.fetch_attempts[bad] == 1
    assert h.status(bad) == "download_failed"
    others = [pid for pid in sum(b, []) if pid != bad]
    assert sorted(h.processed) == sorted(others)
    assert all(h.status(pid) == "rejected" for pid in others)
