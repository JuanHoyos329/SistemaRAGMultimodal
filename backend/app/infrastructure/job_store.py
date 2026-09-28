import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any


class SQLiteJobStore:
    """Small persistent job registry suitable for a single API instance/demo."""

    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS ingestion_jobs (
                job_id TEXT PRIMARY KEY, filename TEXT NOT NULL, status TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0, error_message TEXT, result_summary TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")
            connection.execute(
                "UPDATE ingestion_jobs SET status = 'FAILED', error_message = 'El proceso se reinició durante la ingesta.' "
                "WHERE status IN ('PENDING', 'PROCESSING')"
            )

    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def create(self, job: dict[str, Any]) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO ingestion_jobs(job_id,filename,status,progress) VALUES(?,?,?,?)",
                (job["job_id"], job["filename"], job["status"], job["progress"]),
            )

    def update(self, job_id: str, **values: Any) -> None:
        allowed = {"status", "progress", "error_message", "result_summary"}
        values = {key: value for key, value in values.items() if key in allowed}
        if not values:
            return
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self._lock, self._connect() as connection:
            connection.execute(
                f"UPDATE ingestion_jobs SET {assignments} WHERE job_id = ?",
                (*[json.dumps(value) if isinstance(value, (dict, list)) else value for value in values.values()], job_id),
            )

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM ingestion_jobs WHERE job_id = ?", (job_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        if result.get("result_summary"):
            result["result_summary"] = json.loads(result["result_summary"])
        return result
