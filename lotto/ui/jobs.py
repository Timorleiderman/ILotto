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
        self._lock = threading.Lock()

    def step(self, stage: str, done: int = 0, total: int = 0) -> None:
        """Report progress, and stop here if cancellation was requested."""
        with self._lock:
            if self._cancel.is_set():
                raise Cancelled
            self.stage, self.done, self.total = stage, done, total

    def finish(self, publish: Callable[[], dict]) -> dict:
        """Check cancellation and publish a completed result as one atomic transition."""
        with self._lock:
            if self._cancel.is_set():
                raise Cancelled
            if self.state != "succeeded":
                result = publish()
                self.result, self.state, self.stage = result, "succeeded", "Done"
            return self.result

    def cancel(self) -> None:
        """Cancellation cannot race with committing a finished result."""
        with self._lock:
            if self.state in ("queued", "running"):
                self._cancel.set()
                if self.state == "queued":
                    self.state, self.stage = "cancelled", "Cancelled"
                else:
                    self.state, self.stage = "cancel_requested", "Cancellation requested; finishing the current stage"

    def current_state(self) -> str:
        """Read the job's state under its lock."""
        with self._lock:
            return self.state

    def view(self) -> dict:
        """The job as plain data for the page."""
        with self._lock:
            return {"id": self.id, "kind": self.kind, "state": self.state,
                    "progress": {"stage": self.stage, "done": self.done, "total": self.total},
                    "result": self.result, "error": self.error}


class JobRunner:
    """A queue and the single thread that works through it."""

    def __init__(self):
        # Capacity is enforced on live queued jobs in submit() so cancelled
        # placeholders waiting to be drained never block fresh requests.
        self._queue: queue.Queue[Job | None] = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._worker = threading.Thread(target=self._run, name="lotto-ui-worker", daemon=True)
        self._worker.start()

    def submit(self, kind: str, key: str, work: Callable[[Job], dict]) -> Job:
        """Queue work. The same request already waiting or running is returned, not repeated."""
        with self._lock:
            if self._closed:
                raise Busy
            queued_count = 0
            finished: list[Job] = []
            for job in self._jobs.values():
                state = job.current_state()
                if job.key == key and state in ("queued", "running"):
                    return job
                if state == "queued":
                    queued_count += 1
                elif state in ("succeeded", "failed", "cancelled"):
                    finished.append(job)
            if queued_count >= MAX_QUEUED:
                raise Busy
            job = Job(kind, key, work)
            self._jobs[job.id] = job
            for old in finished[:-KEEP_FINISHED]:
                del self._jobs[old.id]
            self._queue.put_nowait(job)
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
        job.cancel()
        return job

    def close(self, *, wait: bool = False) -> None:
        """Cancel work, optionally waiting for the current stage and its disk writes to stop."""
        with self._lock:
            if not self._closed:
                self._closed = True
                for job in self._jobs.values():
                    job.cancel()
                while True:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        break
                self._queue.put_nowait(None)
        if wait:
            self._worker.join()

    def active(self) -> Job | None:
        """The running job, or the first queued job while the worker is starting."""
        with self._lock:
            queued = None
            for j in self._jobs.values():
                state = j.current_state()
                if state in ("running", "cancel_requested"):
                    return j
                if state == "queued" and queued is None:
                    queued = j
            return queued

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            with job._lock:
                if job.state != "queued":
                    continue
                job.state = "running"
            try:
                result = job._work(job)
                job.finish(lambda: result)
            except Cancelled:
                with job._lock:
                    job.state, job.stage = "cancelled", "Cancelled"
            except Invalid as exc:
                with job._lock:
                    job.error, job.state = {"message": "Some settings are not valid.", "fields": exc.fields}, "failed"
            except Exception:  # a failed job must not take the worker down
                logger.exception("Job %s (%s) failed", job.id, job.kind)
                with job._lock:
                    job.error, job.state = {"message": "The job failed; see the server terminal.", "fields": {}}, "failed"
