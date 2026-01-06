#!/usr/bin/env python3
"""Manages pipeline progress using a SQLite checkpoint database."""

import sqlite3
import datetime
from pathlib import Path

class CheckpointManager:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._create_table()

    @staticmethod
    def _norm_id(pdb_id: str) -> str:
        return (pdb_id or "").strip().lower()

    def _create_table(self):
        """Creates the checkpoints table if it doesn't exist."""
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    pdb_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    rejection_reason TEXT,
                    error_message TEXT
                )
            """)
            # Reset any "in_progress" from a previous failed run
            self.conn.execute("""
                UPDATE checkpoints SET status = 'pending' WHERE status = 'in_progress'
            """)

    def update_status(self, pdb_id: str, status: str, rejection_reason: str = None, error_message: str = None):
        """Adds or updates the status of a PDB ID."""
        pdb_id = self._norm_id(pdb_id)
        if not pdb_id:
            return
        timestamp = datetime.datetime.now().isoformat()
        with self.conn:
            self.conn.execute("""
                INSERT INTO checkpoints (pdb_id, status, timestamp, rejection_reason, error_message)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(pdb_id) DO UPDATE SET
                    status = excluded.status,
                    timestamp = excluded.timestamp,
                    rejection_reason = excluded.rejection_reason,
                    error_message = excluded.error_message
            """, (pdb_id, status, timestamp, rejection_reason, error_message))

    def get_status(self, pdb_id: str) -> str | None:
        """Gets the status of a single PDB ID."""
        pdb_id = self._norm_id(pdb_id)
        if not pdb_id:
            return None
        cursor = self.conn.execute("SELECT status FROM checkpoints WHERE pdb_id = ?", (pdb_id,))
        result = cursor.fetchone()
        return result[0] if result else None

    def get_pending_ids(self, all_ids: list[str]) -> list[str]:
        """
        Filters a list of PDB IDs, returning only those not already completed.
        """
        normalized_ids = [self._norm_id(pdb_id) for pdb_id in all_ids]
        normalized_ids = [pdb_id for pdb_id in normalized_ids if pdb_id]

        cursor = self.conn.execute(
            "SELECT pdb_id FROM checkpoints WHERE status IN ('matched', 'rejected', 'error', 'download_failed')"
        )
        completed_ids = {self._norm_id(row[0]) for row in cursor.fetchall()}

        # Preserve order while filtering
        pending_ids = [pdb_id for pdb_id in normalized_ids if pdb_id not in completed_ids]
        return pending_ids

    def get_stats(self, all_ids: list[str]) -> dict:
        """Returns a dictionary of status counts."""
        total = len(all_ids)
        cursor = self.conn.execute("SELECT status, COUNT(*) FROM checkpoints GROUP BY status")
        counts = dict(cursor.fetchall())
        
        processed_count = sum(counts.values())
        pending_count = total - processed_count

        stats = {
            'total_candidates': total,
            'matched': counts.get('matched', 0),
            'rejected': counts.get('rejected', 0),
            'error': counts.get('error', 0),
            'pending': pending_count
        }
        return stats

    def get_kept_count(self) -> int:
        """Returns the count of PDB files that have been kept (status='matched')."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM checkpoints WHERE status = 'matched'")
        result = cursor.fetchone()
        return result[0] if result else 0

    def __del__(self):
        """Ensures the database connection is closed."""
        if self.conn:
            self.conn.close()
