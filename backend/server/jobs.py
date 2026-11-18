"""Thread-backed asynchronous job manager.

Training and benchmark work is CPU-bound.  A small worker pool keeps the API
responsive without pretending that the work is free or instantaneous.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

from backend.server.db import Database

JobFn = Callable[[Callable[[dict[str, Any]], None], threading.Event], dict[str, Any]]


class JobManager:
    def __init__(self, db: Database, max_workers: int = 1) -> None:
        self.db = db
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, max_workers), thread_name_prefix="will-job"
        )
        self._futures: dict[str, Future] = {}
        self._events: dict[str, threading.Event] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    def submit(self, kind: str, config: dict[str, Any], fn: JobFn) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        job = self.db.create_job(job_id, kind, config)
        stop_event = threading.Event()
        with self._lock:
            self._events[job_id] = stop_event
        future = self._executor.submit(self._run, job_id, kind, fn, stop_event)
        with self._lock:
            self._futures[job_id] = future
        return job  # type: ignore[return-value]

    def _run(
        self,
        job_id: str,
        kind: str,
        fn: JobFn,
        stop_event: threading.Event,
    ) -> None:
        if stop_event.is_set() or self.db.get_job(job_id).get("cancel_requested"):
            self.db.update_job(
                job_id,
                status="cancelled",
                finished_at=time.time(),
                message="cancelled before start",
            )
            self.db.add_event(job_id, {"type": "cancelled", "message": "cancelled before start"})
            self._cleanup(job_id)
            return

        self.db.update_job(
            job_id,
            status="running",
            started_at=time.time(),
            message="running",
        )
        self.db.add_event(job_id, {"type": "start", "kind": kind})

        last_event = [0.0]

        def progress(event: dict[str, Any]) -> None:
            now = time.time()
            completed = event.get("completed")
            total = event.get("total")
            fields: dict[str, Any] = {"message": str(event.get("label") or event.get("type") or "running")}
            if isinstance(completed, (int, float)) and isinstance(total, (int, float)) and total > 0:
                fields["progress"] = max(0.0, min(1.0, float(completed) / float(total)))
            # Persist every event; the UI polls/streams them.  Throttle only
            # identical high-frequency progress events.
            if event.get("type") == "step" and now - last_event[0] < 0.25:
                return
            last_event[0] = now
            self.db.update_job(job_id, **fields)
            self.db.add_event(job_id, event)

        try:
            result = fn(progress, stop_event)
            cancelled = stop_event.is_set() or self.db.get_job(job_id).get("cancel_requested")
            status = "cancelled" if cancelled else "completed"
            self.db.update_job(
                job_id,
                status=status,
                finished_at=time.time(),
                progress=1.0,
                message="cancelled" if cancelled else "completed",
                result_json=result,
            )
            self.db.add_event(
                job_id,
                {
                    "type": "cancelled" if cancelled else "done",
                    "message": "job cancelled" if cancelled else "job completed",
                },
            )
        except Exception as exc:  # pragma: no cover - exercised through API errors
            tb = traceback.format_exc()
            self.db.update_job(
                job_id,
                status="failed",
                finished_at=time.time(),
                message="failed",
                error=tb[-8000:],
            )
            self.db.add_event(
                job_id,
                {"type": "error", "message": str(exc), "traceback": tb[-4000:]},
            )
        finally:
            self._cleanup(job_id)

    def _cleanup(self, job_id: str) -> None:
        with self._lock:
            self._events.pop(job_id, None)
            self._futures.pop(job_id, None)

    # ------------------------------------------------------------------
    def cancel(self, job_id: str) -> dict[str, Any] | None:
        job = self.db.get_job(job_id)
        if job is None:
            return None
        if job["status"] in ("completed", "failed", "cancelled", "interrupted"):
            return job
        self.db.request_cancel(job_id)
        with self._lock:
            event = self._events.get(job_id)
            future = self._futures.get(job_id)
        if event is not None:
            event.set()
        if future is not None and future.cancel():
            self.db.update_job(
                job_id,
                status="cancelled",
                finished_at=time.time(),
                message="cancelled while queued",
            )
            self.db.add_event(job_id, {"type": "cancelled", "message": "cancelled while queued"})
        else:
            self.db.update_job(job_id, message="cancellation requested")
            self.db.add_event(job_id, {"type": "cancelling", "message": "cancellation requested"})
        return self.db.get_job(job_id)

    def shutdown(self) -> None:
        with self._lock:
            events = list(self._events.values())
        for event in events:
            event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
