"""In-memory tracker for in-flight /process jobs, keyed by doc_id.

Processing runs in FastAPI's default thread pool, so the dict is guarded by a
threading.Lock. Completed entries are kept for STATUS_TTL_SECONDS so xyne can
still see the final state on a delayed poll, then evicted lazily on access.
"""

import os
import threading
import time
from typing import Any, Callable, Dict, Optional

STATUS_TTL_SECONDS = int(os.getenv("STATUS_TTL_SECONDS", "600"))


class JobTracker:
    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def start(self, doc_id: str, filename: Optional[str] = None) -> None:
        with self._lock:
            self._prune_locked()
            self._jobs[doc_id] = {
                "doc_id":           doc_id,
                "filename":         filename,
                "state":            "running",
                "stage":            None,
                "started_at":       time.time(),
                "completed_at":     None,
                "duration_seconds": None,
                "error":            None,
            }

    def set_stage(self, doc_id: str, stage: str) -> None:
        with self._lock:
            entry = self._jobs.get(doc_id)
            if entry is not None:
                entry["stage"] = stage

    def stage_setter(self, doc_id: str) -> Callable[[str], None]:
        """Return a closure that updates the stage for *doc_id*.

        Lets the processor report progress without importing the tracker.
        """
        def _set(stage: str) -> None:
            self.set_stage(doc_id, stage)
        return _set

    def done(self, doc_id: str) -> None:
        self._finish(doc_id, state="done", error=None)

    def fail(self, doc_id: str, error: str) -> None:
        self._finish(doc_id, state="failed", error=error)

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._prune_locked()
            entry = self._jobs.get(doc_id)
            return dict(entry) if entry else None

    def find(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Look up a job by doc_id (dict key) OR by filename.

        If multiple entries share the same filename, returns the most recent
        one (highest started_at).
        """
        with self._lock:
            self._prune_locked()
            # Fast path: direct doc_id match.
            entry = self._jobs.get(identifier)
            if entry is not None:
                return dict(entry)
            # Fall back: linear scan over filenames; pick newest on tie.
            matches = [e for e in self._jobs.values() if e.get("filename") == identifier]
            if not matches:
                return None
            newest = max(matches, key=lambda e: e.get("started_at") or 0)
            return dict(newest)

    def all(self) -> Dict[str, Any]:
        """Snapshot of every tracked job, sorted newest-first by started_at."""
        with self._lock:
            self._prune_locked()
            jobs = sorted(
                (dict(e) for e in self._jobs.values()),
                key=lambda e: e.get("started_at") or 0,
                reverse=True,
            )
        counts = {"running": 0, "done": 0, "failed": 0}
        for j in jobs:
            state = j.get("state")
            if state in counts:
                counts[state] += 1
        counts["total"] = len(jobs)
        return {"counts": counts, "jobs": jobs}

    def _finish(self, doc_id: str, state: str, error: Optional[str]) -> None:
        with self._lock:
            entry = self._jobs.get(doc_id)
            if entry is None:
                # Job was never registered (shouldn't happen) — synthesize one.
                entry = self._jobs[doc_id] = {
                    "doc_id":     doc_id,
                    "started_at": time.time(),
                }
            now = time.time()
            entry["state"]            = state
            entry["completed_at"]     = now
            entry["duration_seconds"] = round(now - entry.get("started_at", now), 3)
            entry["error"]            = error

    def _prune_locked(self) -> None:
        cutoff = time.time() - STATUS_TTL_SECONDS
        for doc_id in [
            d for d, e in self._jobs.items()
            if e.get("completed_at") is not None and e["completed_at"] < cutoff
        ]:
            self._jobs.pop(doc_id, None)


tracker = JobTracker()
