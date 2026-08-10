import tempfile
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


if __name__ == "__main__":
    unittest.main()
