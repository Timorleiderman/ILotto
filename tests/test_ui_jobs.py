"""Worker cancellation at computation/publication boundaries, without timing-dependent sleeps."""

import threading

import pytest

from lotto.ui.jobs import Busy, Cancelled, Job, JobRunner


def test_a_cancelled_job_cannot_publish():
    job = Job("test", "test", lambda j: {})
    job.state = "running"
    job.cancel()
    with pytest.raises(Cancelled):
        job.finish(lambda: pytest.fail("cancelled output must not be committed"))
    assert job.result is None


def test_cancelling_a_committed_result_does_not_change_its_state():
    job = Job("test", "test", lambda j: {})
    result = job.finish(lambda: {"saved": True})
    job.cancel()
    assert job.view()["state"] == "succeeded" and job.result == result
    assert job.finish(lambda: pytest.fail("must commit only once")) == result


def test_queued_cancel_and_shutdown_do_not_run_work():
    runner = JobRunner()
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()

    def slow(job):
        entered.set()
        assert release.wait(5)
        try:
            job.step("Complete")
        finally:
            exited.set()
        return {}

    try:
        active = runner.submit("test", "active", slow)
        assert entered.wait(5)
        queued = runner.submit("test", "queued", lambda j: pytest.fail("cancelled queued job ran"))
        assert runner.submit("test", "queued", lambda j: {}) is queued
        assert runner.cancel(queued.id).state == "cancelled"
        runner.close()
        assert active.state == "cancel_requested"
        with pytest.raises(Busy):
            runner.submit("test", "after-close", lambda j: {})
    finally:
        release.set()
        runner.close()
    assert exited.wait(5)
