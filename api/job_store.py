"""
api/job_store.py
-----------------
SQLite-backed job store for managing analysis pipeline state.
Survives uvicorn --reload restarts. Thread-safe via SQLite WAL mode.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional


def _make_serializable(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable objects (e.g. pandas Timestamps
    used as dict keys) into plain Python types."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(i) for i in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # Handles pandas Timestamp, numpy types, dataclasses, etc.
    try:
        # Pydantic / dataclass
        return _make_serializable(obj.__dict__)
    except AttributeError:
        return str(obj)

_CACHE_DIR = Path(__file__).parent.parent / ".cache"
_CACHE_DIR.mkdir(exist_ok=True)
_DB_PATH = _CACHE_DIR / "job_store.db"


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id  TEXT PRIMARY KEY,
            status  TEXT NOT NULL DEFAULT 'queued',
            request TEXT,
            state   TEXT,
            error   TEXT
        )
    """)
    con.commit()
    return con


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._con = _conn()

    # ── helpers ────────────────────────────────────────────────────────────
    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._con.execute(sql, params)
            self._con.commit()
            return cur

    def _fetchone(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        with self._lock:
            self._con.row_factory = sqlite3.Row
            return self._con.execute(sql, params).fetchone()

    # ── public API ─────────────────────────────────────────────────────────
    def create(self, job_id: str, request: Any) -> None:
        req_json = json.dumps(request if isinstance(request, dict) else request.__dict__
                              if hasattr(request, "__dict__") else str(request))
        self._execute(
            "INSERT OR REPLACE INTO jobs (job_id, status, request) VALUES (?, 'queued', ?)",
            (job_id, req_json),
        )

    def set_running(self, job_id: str) -> None:
        self._execute("UPDATE jobs SET status='running' WHERE job_id=?", (job_id,))

    def set_done(self, job_id: str, state: Any) -> None:
        self._execute(
            "UPDATE jobs SET status='done', state=? WHERE job_id=?",
            (json.dumps(_make_serializable(state)), job_id),
        )

    def set_failed(self, job_id: str, error: str) -> None:
        self._execute(
            "UPDATE jobs SET status='failed', error=? WHERE job_id=?",
            (error, job_id),
        )

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        row = self._fetchone("SELECT * FROM jobs WHERE job_id=?", (job_id,))
        if row is None:
            return None
        result: dict[str, Any] = dict(row)
        if result.get("state"):
            try:
                result["state"] = json.loads(result["state"])
            except (json.JSONDecodeError, TypeError):
                pass
        return result

    def list_jobs(self) -> list[str]:
        with self._lock:
            rows = self._con.execute("SELECT job_id FROM jobs").fetchall()
        return [r[0] for r in rows]


# Module-level singleton
job_store = JobStore()
