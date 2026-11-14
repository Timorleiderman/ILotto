"""One background worker, so the page stays responsive while a model trains.

Jobs run one at a time, in the order they were queued: TensorFlow and the
compiled coverage counters do not share a machine well, and one person is
using the dashboard. A job reports progress and can be asked to stop; it
stops at the next safe point, never in the middle of a fit.
"""

from __future__ import annotations

import logging
import queue
import threading
import uuid
from collections.abc import Callable

from lotto.ui.registry import Invalid

logger = logging.getLogger(__name__)

MAX_QUEUED = 8
KEEP_FINISHED = 60


class Cancelled(Exception):
    """Raised inside a job when the user asked for it to stop."""


class Busy(Exception):
    """The queue is full."""


class Job:
    """One queued, running or finished piece of work."""

    def __init__(self, kind: str, key: str, work: Callable[[Job], dict]):
        self.id = uuid.uuid4().hex[:16]
        self.kind, self.key, self._work = kind, key, work
        self.state = "queued"
        self.stage, self.done, self.total = "Waiting to start", 0, 0
        self.result: dict | None = None
        self.error: dict | None = None
        self._cancel = threading.Event()

    def step(self, stage: str, done: int = 0, total: int = 0) -> None:
        """Report progress, and stop here if cancellation was requested."""
        if self._cancel.is_set():
            raise Cancelled
        self.stage, self.done, self.total = stage, done, total

    def view(self) -> dict:
        """The job as plain data for the page."""
        return {"id": self.id, "kind": self.kind, "state": self.state,
                "progress": {"stage": self.stage, "done": self.done, "total": self.total},
                "result": self.result, "error": self.error}


class JobRunner:
    """A queue and the single thread that works through it."""

    def __init__(self):
        self._queue: queue.Queue[Job] = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        threading.Thread(target=self._run, name="lotto-ui-worker", daemon=True).start()

    def submit(self, kind: str, key: str, work: Callable[[Job], dict]) -> Job:
        """Queue work. The same request already waiting or running is returned, not repeated."""
        with self._lock:
            for job in self._jobs.values():
                if job.key == key and job.state in ("queued", "running", "cancel_requested"):
                    return job
            if sum(j.state == "queued" for j in self._jobs.values()) >= MAX_QUEUED:
                raise Busy
            job = Job(kind, key, work)
            self._jobs[job.id] = job
            finished = [j for j in self._jobs.values() if j.state in ("succeeded", "failed", "cancelled")]
            for old in finished[:-KEEP_FINISHED]:
                del self._jobs[old.id]
        self._queue.put(job)
        return job

    def get(self, job_id: str) -> Job | None:
        """The job with this id, or None once it has been forgotten."""
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> Job | None:
        """Ask a job to stop. A queued job stops at once; a running one at its next safe point."""
        job = self.get(job_id)
        if job is None:
            return None
        with self._lock:
            if job.state == "queued":
                job._cancel.set()
                job.state, job.stage = "cancelled", "Cancelled"
            elif job.state == "running":
                job._cancel.set()
                job.state, job.stage = "cancel_requested", "Cancellation requested; finishing the current stage"
        return job

    def active(self) -> Job | None:
        """The job running now, if any."""
        with self._lock:
            return next((j for j in self._jobs.values() if j.state in ("running", "cancel_requested")), None)

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            with self._lock:
                if job.state != "queued":
                    continue
                job.state = "running"
            try:
                result = job._work(job)
                if job._cancel.is_set():
                    raise Cancelled
                job.result, job.state, job.stage = result, "succeeded", "Done"
            except Cancelled:
                job.state, job.stage = "cancelled", "Cancelled"
            except Invalid as exc:
                job.error, job.state = {"message": "Some settings are not valid.", "fields": exc.fields}, "failed"
            except Exception as exc:  # a failed job must not take the worker down
                logger.exception("Job %s (%s) failed", job.id, job.kind)
                job.error, job.state = {"message": f"{type(exc).__name__}: {exc}", "fields": {}}, "failed"
