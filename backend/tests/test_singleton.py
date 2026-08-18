"""One backend per machine — the pid-file lock and every place that must
respect it. Born from a measured incident (2026-08-18): the updater's
rollback arm raced a slow-booting new version, Windows let both share
:8000, and the loser lived on as a zombie whose monitor spawned duplicate
ComfyUI processes, each holding VRAM."""
import inspect
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from app.core.singleton import acquire, release


class SingletonLock(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lock = Path(self.tmp.name) / "logs" / "backend.pid"

    def tearDown(self):
        self.tmp.cleanup()

    def test_first_caller_wins_and_writes_its_pid(self):
        self.assertTrue(acquire(self.lock))
        self.assertEqual(int(self.lock.read_text()), os.getpid())

    def test_a_second_caller_is_refused_while_the_owner_lives(self):
        # acquire() steals only from the DEAD, and treats the CALLER's own
        # pid as reclaimable — so emulate a foreign live owner with a pid
        # that is always alive: the Windows System process.
        self.lock.parent.mkdir(parents=True, exist_ok=True)
        self.lock.write_text("4")
        self.assertFalse(acquire(self.lock))

    def test_a_stale_lock_is_stolen(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        self.lock.parent.mkdir(parents=True, exist_ok=True)
        self.lock.write_text(str(proc.pid))  # provably dead
        self.assertTrue(acquire(self.lock))
        self.assertEqual(int(self.lock.read_text()), os.getpid())

    def test_release_never_removes_a_foreign_lock(self):
        self.lock.parent.mkdir(parents=True, exist_ok=True)
        self.lock.write_text("4")
        release(self.lock)
        self.assertTrue(self.lock.exists())
        self.lock.write_text(str(os.getpid()))
        release(self.lock)
        self.assertFalse(self.lock.exists())

    def test_every_respawn_path_respects_the_singleton(self):
        from app.core.services import Services
        from app.core.update import UpdateManager

        run_py = (Path(__file__).resolve().parents[1] / "run.py").read_text()
        self.assertIn("acquire(LOCK)", run_py)
        self.assertIn("release(LOCK)", run_py)
        # A backend whose server loop ends must die completely — no
        # zombie monitors against a port it no longer owns.
        self.assertIn("os._exit", run_py)

        helper = inspect.getsource(UpdateManager._schedule_restart)
        self.assertIn("-PassThru", helper)          # know what was started
        self.assertIn("Stop-Process -Id $proc.Id", helper)  # kill before rollback

        monitor = inspect.getsource(Services._monitor_loop)
        self.assertIn("_respawn_comfy_clean", monitor)
        clean = inspect.getsource(Services._respawn_comfy_clean)
        self.assertIn("_comfy_pids", clean)          # strays die first


if __name__ == "__main__":
    unittest.main()
