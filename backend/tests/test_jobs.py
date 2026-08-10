import tempfile
import time
import unittest
from pathlib import Path

from app.core.db import Database
from app.core.jobs import JobQueue, JobState, PermanentError, TransientError


class JobQueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "test.sqlite3")
        self.q = JobQueue(self.db, max_retries=2, backoff_s=0.01)

    def tearDown(self):
        self.q.stop()
        self.db.close()
        self.tmp.cleanup()

    def test_success_path(self):
        self.q.register("ok", lambda job: {"answer": 42})
        self.q.start()
        job = self.q.enqueue("ok", {})
        done = self.q.wait_for(job.id)
        self.assertEqual(done.state, JobState.COMPLETED)
        self.assertEqual(done.result, {"answer": 42})
        self.assertEqual(done.attempts, 1)

    def test_transient_error_retries_then_succeeds(self):
        calls = {"n": 0}

        def flaky(job):
            calls["n"] += 1
            if calls["n"] < 3:
                raise TransientError("backend briefly down")
            return {"ok": True}

        self.q.register("flaky", flaky)
        self.q.start()
        job = self.q.enqueue("flaky", {})
        done = self.q.wait_for(job.id)
        self.assertEqual(done.state, JobState.COMPLETED)
        self.assertEqual(done.attempts, 3)
        states_logged = " ".join(entry["msg"] for entry in done.logs)
        self.assertIn("Retrying", states_logged)

    def test_permanent_error_fails_without_retry(self):
        self.q.register("bad", lambda job: (_ for _ in ()).throw(PermanentError("bad input")))
        self.q.start()
        job = self.q.enqueue("bad", {})
        done = self.q.wait_for(job.id)
        self.assertEqual(done.state, JobState.FAILED)
        self.assertEqual(done.attempts, 1)
        self.assertIn("bad input", done.error or "")

    def test_retry_exhaustion_fails(self):
        self.q.register("always", lambda job: (_ for _ in ()).throw(TransientError("nope")))
        self.q.start()
        job = self.q.enqueue("always", {})
        done = self.q.wait_for(job.id)
        self.assertEqual(done.state, JobState.FAILED)
        self.assertEqual(done.attempts, 3)  # 1 initial + 2 retries

    def test_cancel_pending_job(self):
        # queue not started: job stays pending, cancel must land
        self.q.register("ok", lambda job: {})
        job = self.q.enqueue("ok", {})
        self.assertTrue(self.q.cancel(job.id))
        self.assertEqual(self.q.get(job.id).state, JobState.CANCELLED)
        self.q.start()  # worker must skip the cancelled job without crashing

    def test_cancel_during_retry_backoff_does_not_resume(self):
        """A job cancelled while it reads "Retrying in Ns" must NOT run the
        attempt it was stopped for. The retry loop used to overwrite the
        CANCELLED state with RUNNING after its backoff wait and could even
        complete — a false 'cancelled' plus wasted work."""
        import threading

        # Own temp dir + explicit teardown IN this test: the wider 0.5s
        # backoff means the queue must be stopped and its DB closed before
        # the shared tearDown deletes its own dir (Windows won't unlink an
        # open sqlite file).
        tmp = tempfile.TemporaryDirectory()
        db = Database(Path(tmp.name) / "backoff.sqlite3")
        q = JobQueue(db, max_retries=3, backoff_s=0.5)  # a wide window
        started = threading.Event()
        calls = {"n": 0}

        def flaky(job):
            calls["n"] += 1
            started.set()
            raise TransientError("first attempt fails, then it backs off")

        try:
            q.register("flaky", flaky)
            q.start()
            job = q.enqueue("flaky", {})
            self.assertTrue(started.wait(5), "handler never ran")
            # Wait until attempt 1 has RAISED and the job is asleep in its
            # backoff — cancelling earlier (still RUNNING) would exercise the
            # already-working cooperative path, not the RETRYING-backoff bug.
            deadline = time.monotonic() + 5
            while (q.get(job.id).state is not JobState.RETRYING
                   and time.monotonic() < deadline):
                time.sleep(0.02)
            self.assertEqual(q.get(job.id).state, JobState.RETRYING)
            self.assertTrue(q.cancel(job.id))
            # Wait PAST the 0.5s backoff — the bug only manifests when the
            # sleep expires and the loop resumes. Checking earlier would catch
            # the transient CANCELLED before the buggy resume even happens.
            time.sleep(1.2)
            self.assertEqual(q.get(job.id).state, JobState.CANCELLED)
            # The decisive assertion: the stopped attempt never re-ran.
            self.assertEqual(calls["n"], 1)
        finally:
            q.stop()
            db.close()
            tmp.cleanup()

    def test_manual_retry_of_failed_job(self):
        attempts = {"n": 0}

        def once_broken(job):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise PermanentError("first run mistake")
            return {"fixed": True}

        self.q.register("fixable", once_broken)
        self.q.start()
        job = self.q.enqueue("fixable", {})
        self.assertEqual(self.q.wait_for(job.id).state, JobState.FAILED)
        self.assertTrue(self.q.retry(job.id))
        done = self.q.wait_for(job.id)
        self.assertEqual(done.state, JobState.COMPLETED)

    def test_retry_rejected_for_completed_job(self):
        self.q.register("ok", lambda job: {})
        self.q.start()
        job = self.q.enqueue("ok", {})
        self.q.wait_for(job.id)
        self.assertFalse(self.q.retry(job.id))

    def test_unknown_job_type_rejected(self):
        with self.assertRaises(ValueError):
            self.q.enqueue("nope", {})

    def test_jobs_persisted_to_db(self):
        self.q.register("ok", lambda job: {"x": 1})
        self.q.start()
        job = self.q.enqueue("ok", {})
        self.q.wait_for(job.id)
        rows = self.db.query("SELECT state FROM jobs WHERE id=?", (job.id,))
        self.assertEqual(rows[0]["state"], "completed")

    def test_unserializable_result_does_not_kill_the_worker(self):
        """A handler is supposed to return a JSON-serializable dict. One that
        returns a set/bytes/object would raise TypeError inside _persist,
        killing the worker thread and stopping the whole queue. Persistence
        must survive it: the job still reaches a terminal state, the row is
        written, and the NEXT job still runs."""
        self.q.register("bad", lambda job: {"weird": {1, 2, 3}})  # a set
        self.q.register("ok", lambda job: {"x": 1})
        self.q.start()
        bad = self.q.enqueue("bad", {})
        done = self.q.wait_for(bad.id, timeout=5)
        self.assertEqual(done.state, JobState.COMPLETED)
        # The row persisted with a marker instead of crashing.
        row = self.db.query("SELECT state, result FROM jobs WHERE id=?",
                            (bad.id,))[0]
        self.assertEqual(row["state"], "completed")
        self.assertIn("not serializable", row["result"] or "")
        # The worker is still alive — a following job completes.
        good = self.q.enqueue("ok", {})
        self.assertEqual(self.q.wait_for(good.id, timeout=5).state,
                         JobState.COMPLETED)


if __name__ == "__main__":
    unittest.main()
