"""SQLite persistence for asynchronous jobs and their event streams."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    config_json TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    message TEXT,
    result_json TEXT,
    error TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_jobs_kind_created ON jobs(kind, created_at DESC);

CREATE TABLE IF NOT EXISTS job_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    type TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_job ON job_events(job_id, id);
"""


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.init()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def init(self) -> None:
        with self._lock, self.connect() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    def create_job(self, job_id: str, kind: str, config: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        with self._lock, self.connect() as conn:
            conn.execute(
                "INSERT INTO jobs (id, kind, status, created_at, config_json, progress) "
                "VALUES (?, ?, 'queued', ?, ?, 0)",
                (job_id, kind, now, _dumps(config)),
            )
        return self.get_job(job_id)  # type: ignore[return-value]

    def update_job(self, job_id: str, **fields: Any) -> None:
        allowed = {
            "status", "started_at", "finished_at", "progress", "message",
            "result_json", "error", "cancel_requested",
        }
        clean = {k: v for k, v in fields.items() if k in allowed}
        if not clean:
            return
        assignments = ", ".join(f"{key} = ?" for key in clean)
        values = []
        for key, value in clean.items():
            if key == "result_json" and not isinstance(value, str):
                value = _dumps(value)
            values.append(value)
        values.append(job_id)
        with self._lock, self.connect() as conn:
            conn.execute(f"UPDATE jobs SET {assignments} WHERE id = ?", values)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock, self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._job_row(row) if row else None

    def list_jobs(
        self, kind: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        with self._lock, self.connect() as conn:
            if kind:
                rows = conn.execute(
                    "SELECT * FROM jobs WHERE kind = ? ORDER BY created_at DESC LIMIT ?",
                    (kind, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
        return [self._job_row(row) for row in rows]

    @staticmethod
    def _job_row(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        for source, target in (("config_json", "config"), ("result_json", "result")):
            raw = data.pop(source, None)
            if raw:
                try:
                    data[target] = json.loads(raw)
                except Exception:
                    data[target] = raw
            else:
                data[target] = None
        data["cancel_requested"] = bool(data.get("cancel_requested"))
        return data

    # ------------------------------------------------------------------
    def add_event(self, job_id: str, event: dict[str, Any]) -> int:
        now = time.time()
        with self._lock, self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO job_events (job_id, created_at, type, payload_json) "
                "VALUES (?, ?, ?, ?)",
                (job_id, now, str(event.get("type", "event")), _dumps(event)),
            )
            return int(cur.lastrowid)

    def get_events(
        self, job_id: str, after_id: int = 0, limit: int = 500
    ) -> list[dict[str, Any]]:
        with self._lock, self.connect() as conn:
            rows = conn.execute(
                "SELECT id, created_at, type, payload_json FROM job_events "
                "WHERE job_id = ? AND id > ? ORDER BY id ASC LIMIT ?",
                (job_id, after_id, limit),
            ).fetchall()
        out = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except Exception:
                payload = {"type": row["type"], "raw": row["payload_json"]}
            payload["_event_id"] = row["id"]
            payload["_created_at"] = row["created_at"]
            out.append(payload)
        return out

    def request_cancel(self, job_id: str) -> bool:
        with self._lock, self.connect() as conn:
            cur = conn.execute(
                "UPDATE jobs SET cancel_requested = 1 WHERE id = ? AND status IN ('queued','running')",
                (job_id,),
            )
            return cur.rowcount > 0

    def fail_stale_jobs(self) -> int:
        """Mark jobs left running by a previous process as interrupted."""
        with self._lock, self.connect() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status = 'interrupted', finished_at = ?, "
                "error = COALESCE(error, 'server restarted before the job finished') "
                "WHERE status IN ('queued','running')",
                (time.time(),),
            )
            return cur.rowcount
