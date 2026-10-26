"""Background worker. Jobs are rows in the `jobs` table so progress survives a
reload of the UI; a small thread pool runs them. Analysis and render jobs are
CPU-heavy numpy/librosa work that releases the GIL often enough for a couple of
workers to be worthwhile."""

from __future__ import annotations

import logging
import queue
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .db import Database, dumps, loads, new_id, now_iso
from .events import EventBus
from .models import Job


class JobNotFound(KeyError):
    pass

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]
Handler = Callable[["JobContext"], dict[str, Any] | None]


class JobCancelled(Exception):
    pass


@dataclass
class JobContext:
    job_id: str
    kind: str
    payload: dict[str, Any]
    db: Database
    events: EventBus
    _worker: JobWorker

    def progress(self, fraction: float, message: str = "") -> None:
        self._worker.update(self.job_id, progress=max(0.0, min(1.0, fraction)), message=message)
        if self._worker.is_cancelled(self.job_id):
            raise JobCancelled()

    @property
    def cancelled(self) -> bool:
        return self._worker.is_cancelled(self.job_id)


class JobWorker:
    def __init__(self, db: Database, events: EventBus, threads: int = 2):
        self.db = db
        self.events = events
        self._handlers: dict[str, Handler] = {}
        self._queue: queue.PriorityQueue[tuple[int, int, str]] = queue.PriorityQueue()
        self._seq = 0
        self._cancel: set[str] = set()
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._n_threads = threads
        # Paused kinds are parked when dequeued and re-queued on resume, so
        # the other kinds (renders, exports) keep flowing.
        self._paused: set[str] = set()
        self._parked: list[tuple[int, int, str]] = []

    # --- lifecycle --------------------------------------------------------------

    def register(self, kind: str, handler: Handler) -> None:
        self._handlers[kind] = handler

    def start(self) -> None:
        self._requeue_unfinished()
        for i in range(self._n_threads):
            t = threading.Thread(target=self._run, name=f"patine-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the workers and close their database connections. Joining
        matters on Windows, where an open SQLite handle blocks deleting the
        data directory (tests use temp dirs)."""
        self._stop.set()
        for _ in self._threads:
            self._queue.put((-1, 0, ""))
        for t in self._threads:
            t.join(timeout)
        self._threads.clear()

    def _requeue_unfinished(self) -> None:
        rows = self.db.all("SELECT id, kind, status FROM jobs WHERE status IN ('queued','running')")
        for r in rows:
            self.update(r["id"], status="queued", progress=0.0, message="requeued")
            self._enqueue(r["id"], priority=10)

    # --- submission -----------------------------------------------------------------

    def submit(self, kind: str, payload: dict[str, Any], *, priority: int = 5, dedupe_key: str | None = None) -> Job:
        """Queue a job. `dedupe_key` (stored in payload['_dedupe']) returns the
        existing queued/running job instead of adding another."""
        if kind not in self._handlers:
            raise KeyError(f"no handler for job kind {kind!r}")
        if dedupe_key is not None:
            payload = {**payload, "_dedupe": dedupe_key}
            existing = self.db.one(
                "SELECT * FROM jobs WHERE kind = ? AND status IN ('queued','running') AND json_extract(payload, '$._dedupe') = ?",
                (kind, dedupe_key),
            )
            if existing is not None:
                return row_to_job(existing)
        job_id = new_id()
        ts = now_iso()
        with self.db.tx() as conn:
            conn.execute(
                "INSERT INTO jobs (id, kind, status, progress, message, payload, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (job_id, kind, "queued", 0.0, None, dumps(payload), ts, ts),
            )
        job = self.get(job_id)
        self.events.publish("job.updated", job.model_dump())
        self._enqueue(job_id, priority)
        return job

    def _enqueue(self, job_id: str, priority: int) -> None:
        with self._lock:
            self._seq += 1
            self._queue.put((priority, self._seq, job_id))

    def pause(self, kind: str) -> None:
        with self._lock:
            self._paused.add(kind)

    def resume(self, kind: str) -> None:
        with self._lock:
            self._paused.discard(kind)
            parked, self._parked = self._parked, []
        for prio, seq, job_id in parked:
            self._queue.put((prio, seq, job_id))

    def is_paused(self, kind: str) -> bool:
        with self._lock:
            return kind in self._paused

    @property
    def parked_count(self) -> int:
        with self._lock:
            return len(self._parked)

    def cancel(self, job_id: str) -> Job:
        with self._lock:
            self._cancel.add(job_id)
        job = self.get(job_id)
        if job.status == "queued":
            self.update(job_id, status="cancelled", message="cancelled")
        return self.get(job_id)

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancel

    # --- state ---------------------------------------------------------------------

    def get(self, job_id: str) -> Job:
        row = self.db.one("SELECT * FROM jobs WHERE id = ?", (job_id,))
        if row is None:
            raise JobNotFound(job_id)
        return row_to_job(row)

    def list(self, status: str | None = None, limit: int = 100) -> list[Job]:
        if status:
            rows = self.db.all("SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?", (status, limit))
        else:
            rows = self.db.all("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
        return [row_to_job(r) for r in rows]

    def update(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        if "result" in fields and fields["result"] is not None:
            fields["result"] = dumps(fields["result"])
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self.db.tx() as conn:
            conn.execute(f"UPDATE jobs SET {cols}, updated_at = ? WHERE id = ?", (*fields.values(), now_iso(), job_id))
        try:
            self.events.publish("job.updated", self.get(job_id).model_dump())
        except KeyError:
            pass

    # --- execution ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            self._loop()
        finally:
            self.db.close()   # this thread's connection

    def _loop(self) -> None:
        while not self._stop.is_set():
            prio, seq, job_id = self._queue.get()
            if not job_id:
                continue
            if self.is_cancelled(job_id):
                self.update(job_id, status="cancelled", message="cancelled")
                continue
            row = self.db.one("SELECT * FROM jobs WHERE id = ?", (job_id,))
            if row is None or row["status"] != "queued":
                continue
            with self._lock:
                if row["kind"] in self._paused:
                    self._parked.append((prio, seq, job_id))
                    continue
            handler = self._handlers.get(row["kind"])
            if handler is None:
                self.update(job_id, status="failed", error=f"no handler for {row['kind']}")
                continue
            self.update(job_id, status="running", message="started")
            ctx = JobContext(job_id, row["kind"], loads(row["payload"], {}), self.db, self.events, self)
            try:
                result = handler(ctx)
                self.update(job_id, status="done", progress=1.0, message="done", result=result)
            except JobCancelled:
                self.update(job_id, status="cancelled", message="cancelled")
            except Exception as e:  # noqa: BLE001 - job failures are reported, not raised
                log.error("job %s (%s) failed: %s\n%s", job_id, row["kind"], e, traceback.format_exc())
                self.update(job_id, status="failed", error=f"{type(e).__name__}: {e}")
            finally:
                with self._lock:
                    self._cancel.discard(job_id)


def row_to_job(row: Any) -> Job:
    return Job(
        id=row["id"],
        kind=row["kind"],
        status=row["status"],
        progress=row["progress"],
        message=row["message"],
        payload=loads(row["payload"], {}),
        result=loads(row["result"]),
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
