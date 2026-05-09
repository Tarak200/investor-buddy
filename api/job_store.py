"""
api/job_store.py
-----------------
In-memory job store for managing analysis pipeline state.
Thread-safe via a simple lock.
"""

from __future__ import annotations

import threading
from typing import Any, Optional


class JobStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def create(self, job_id: str, request: Any) -> None:
        with self._lock:
            self._jobs[job_id] = {
                "status": "queued",
                "request": request,
                "state": None,
                "error": None,
            }

    def set_running(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "running"

    def set_done(self, job_id: str, state: Any) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "done"
                self._jobs[job_id]["state"] = state

    def set_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "failed"
                self._jobs[job_id]["error"] = error

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[str]:
        with self._lock:
            return list(self._jobs.keys())


# Module-level singleton
job_store = JobStore()
