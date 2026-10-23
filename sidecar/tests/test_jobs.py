"""Job worker: dedupe, failure reporting, cancellation, requeue on restart."""

import threading
import time

from patine.db import Database
from patine.events import EventBus
from patine.jobs import JobWorker


def _worker(tmp_data_dir, threads=1):
    db = Database(tmp_data_dir / "jobs.sqlite3")
    return JobWorker(db, EventBus(), threads=threads), db


def _wait(fn, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = fn()
        if v:
            return v
        time.sleep(0.02)
    raise AssertionError("timed out")


def test_run_dedupe_and_result(tmp_data_dir):
    w, _ = _worker(tmp_data_dir)
    seen = []
    w.register("echo", lambda job: (seen.append(job.payload["x"]), {"x": job.payload["x"]})[1])
    gate = threading.Event()
    w.register("slow", lambda job: (gate.wait(5), {"ok": True})[1])
    w.start()
    a = w.submit("slow", {}, dedupe_key="k")
    b = w.submit("slow", {}, dedupe_key="k")
    assert a.id == b.id
    j = w.submit("echo", {"x": 1})
    gate.set()
    done = _wait(lambda: (lambda x: x if x.status == "done" else None)(w.get(j.id)))
    assert done.result == {"x": 1} and seen == [1]
    assert w.get(a.id).status == "done" or _wait(lambda: w.get(a.id).status == "done")
    w.stop()


def test_failure_is_reported_not_raised(tmp_data_dir):
    w, _ = _worker(tmp_data_dir)

    def boom(job):
        raise ValueError("nope")

    w.register("boom", boom)
    w.start()
    j = w.submit("boom", {})
    failed = _wait(lambda: (lambda x: x if x.status == "failed" else None)(w.get(j.id)))
    assert "ValueError: nope" in failed.error
    w.stop()


def test_cancel_queued_and_running(tmp_data_dir):
    w, _ = _worker(tmp_data_dir)
    started = threading.Event()

    def long(job):
        started.set()
        for i in range(200):
            job.progress(i / 200, "working")   # raises JobCancelled once cancelled
            time.sleep(0.01)
        return {}

    w.register("long", long)
    w.start()
    a = w.submit("long", {})
    b = w.submit("long", {})
    started.wait(2)
    assert w.cancel(b.id).status == "cancelled"        # still queued
    w.cancel(a.id)
    _wait(lambda: w.get(a.id).status == "cancelled")
    w.stop()


def test_unfinished_jobs_requeue_on_restart(tmp_data_dir):
    db = Database(tmp_data_dir / "jobs.sqlite3")
    w1 = JobWorker(db, EventBus(), threads=1)
    w1.register("x", lambda job: {})
    j = w1.submit("x", {})            # queued but never started (worker not started)
    w2 = JobWorker(db, EventBus(), threads=1)
    w2.register("x", lambda job: {"ran": True})
    w2.start()
    done = _wait(lambda: (lambda x: x if x.status == "done" else None)(w2.get(j.id)))
    assert done.result == {"ran": True}
    w2.stop()
