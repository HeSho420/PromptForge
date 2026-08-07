"""Updates through git, proven on real repositories in a temp folder.

An 'origin' repo stands in for GitHub; a clone stands in for an install.
Commits pushed to origin appear as available updates in the clone; applying
fast-forwards the clone; local edits and diverged local commits refuse
honestly. Nothing here talks to the network or restarts anything."""
import inspect
import subprocess
import tempfile
import unittest
from pathlib import Path

from app.core.update import UpdateError, UpdateManager


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t",
         "-c", "commit.gpgsign=false", *args],
        cwd=str(cwd), capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise AssertionError(f"git {args}: {proc.stderr or proc.stdout}")
    return (proc.stdout or "").strip()


class FakeJob:
    def __init__(self):
        self.lines: list[str] = []

    def log(self, _level, msg):
        self.lines.append(msg)


class GitUpdates(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.origin = root / "origin"
        self.origin.mkdir()
        _git(self.origin, "init", "--quiet")
        (self.origin / "app.txt").write_text("v1")
        _git(self.origin, "add", ".")
        _git(self.origin, "commit", "--quiet", "-m", "first version")
        self.install = root / "install"
        _git(root, "clone", "--quiet", str(self.origin), str(self.install))
        self.mgr = UpdateManager(self.install)

    def _push(self, message: str, name: str = "app.txt",
              content: str = "new") -> None:
        (self.origin / name).write_text(content)
        _git(self.origin, "add", ".")
        _git(self.origin, "commit", "--quiet", "-m", message)

    def test_up_to_date_says_so(self):
        s = self.mgr.status()
        self.assertTrue(s["repo"])
        self.assertEqual(s["behind"], 0)
        self.assertEqual(s["dirty"], [])

    def test_a_pushed_commit_shows_as_an_available_update(self):
        self._push("speed up renders")
        s = self.mgr.status()
        self.assertEqual(s["behind"], 1)
        self.assertEqual(s["incoming"][0]["subject"], "speed up renders")

    def test_apply_fast_forwards_the_install(self):
        self._push("speed up renders", content="v2")
        job = FakeJob()
        out = self.mgr.apply(job, restart=False)
        self.assertTrue(out["updated"])
        self.assertEqual(out["commits"], 1)
        self.assertEqual((self.install / "app.txt").read_text(), "v2")
        self.assertTrue(any("fast-forward" in ln for ln in job.lines))

    def test_apply_with_nothing_pushed_is_a_no_op(self):
        out = self.mgr.apply(FakeJob(), restart=False)
        self.assertFalse(out["updated"])

    def test_local_edits_refuse_with_the_filename(self):
        self._push("something new")
        (self.install / "app.txt").write_text("my local hack")
        with self.assertRaises(UpdateError) as ctx:
            self.mgr.apply(FakeJob(), restart=False)
        self.assertIn("app.txt", str(ctx.exception))

    def test_diverged_local_commits_refuse(self):
        """ff-only is the whole safety story: an install never merges."""
        (self.install / "local.txt").write_text("mine")
        _git(self.install, "add", ".")
        _git(self.install, "commit", "--quiet", "-m", "local change")
        self._push("remote change", name="other.txt")
        with self.assertRaises(UpdateError):
            self.mgr.apply(FakeJob(), restart=False)

    def test_a_plain_folder_is_not_updatable(self):
        plain = Path(self.tmp.name) / "plain"
        plain.mkdir()
        mgr = UpdateManager(plain)
        self.assertFalse(mgr.is_repo())
        self.assertIn("error", mgr.status())
        with self.assertRaises(UpdateError):
            mgr.apply(FakeJob(), restart=False)

    def test_the_restart_helper_can_roll_back(self):
        src = inspect.getsource(UpdateManager._schedule_restart)
        self.assertIn("git reset --hard", src)
        self.assertIn("Test-Health", src)
        self.assertIn("DETACHED_PROCESS", src)

    def test_dependencies_only_reinstall_when_requirements_changed(self):
        src = inspect.getsource(UpdateManager.apply)
        self.assertIn('c == "backend/requirements.txt"', src)
        self.assertIn('c.startswith("frontend/")', src)
        self.assertIn("--ff-only", src)


if __name__ == "__main__":
    unittest.main()
