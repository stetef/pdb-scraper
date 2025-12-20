import sys
import sqlite3
from pathlib import Path

# ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scrape_pdb.checkpoint import CheckpointManager


def test_in_progress_reset_on_init(tmp_path):
    db_path = tmp_path / "checkpoint.db"
    # create DB and insert an in_progress row to simulate an interrupted run
    conn = sqlite3.connect(str(db_path))
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                pdb_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                rejection_reason TEXT,
                error_message TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO checkpoints (pdb_id, status, timestamp) VALUES (?, ?, datetime('now'))",
            ("1abc", "in_progress"),
        )
    conn.close()

    # Initialize manager - constructor should reset in_progress -> pending
    mgr = CheckpointManager(str(db_path))
    status = mgr.get_status("1abc")
    assert status == "pending"


def test_update_get_and_pending_ids(tmp_path):
    db_path = tmp_path / "cp2.db"
    mgr = CheckpointManager(str(db_path))

    # Update a few statuses
    mgr.update_status("a1", "matched")
    mgr.update_status("b2", "rejected", rejection_reason="no_metal")

    # get_status
    assert mgr.get_status("a1") == "matched"
    assert mgr.get_status("b2") == "rejected"
    assert mgr.get_status("missing") is None

    # get_pending_ids filters out matched/rejected/error
    all_ids = ["a1", "b2", "c3"]
    pending = mgr.get_pending_ids(all_ids)
    assert pending == ["c3"]


def test_get_stats_counts(tmp_path):
    db_path = tmp_path / "cp3.db"
    mgr = CheckpointManager(str(db_path))

    # Seed statuses
    mgr.update_status("p1", "matched")
    mgr.update_status("p2", "matched")
    mgr.update_status("p3", "rejected")
    mgr.update_status("p4", "error")

    all_ids = ["p1", "p2", "p3", "p4", "p5"]
    stats = mgr.get_stats(all_ids)

    assert stats["total_candidates"] == len(all_ids)
    assert stats["matched"] == 2
    assert stats["rejected"] == 1
    assert stats["error"] == 1
    assert stats["pending"] == 1
