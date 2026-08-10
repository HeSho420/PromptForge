"""Mock mode makes no ComfyUI connection, and downloads obey their contracts.

Measured live before these gates existed (all from one mock avatar build on
a machine whose resident real ComfyUI was up): the mesh stage passed the
is-ComfyUI-up check via that resident instance and rendered through it;
cancelling the job posted /interrupt into the same instance; the job also
began a multi-GB model download despite auto_install=0, and cancelling did
not stop the transfer (+108 MB in the 18s after the cancel).
"""
import tempfile
import unittest
from pathlib import Path

from app.adapters.mock import OfflineComfyClient
from app.config import Settings
from app.core.jobs import Job, PermanentError, TransientError
from app.core.services import Services


def _services(tmp: str, **overrides) -> Services:
    base = dict(data_dir=Path(tmp), inpaint_backend="mock",
                segment_backend="mock", critic_model="",
                first_run_setup=False, comfyui_dir="",
                llm_url="http://127.0.0.1:9/v1")
    base.update(overrides)
    return Services(Settings(**base))


class OfflineClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = _services(self.tmp.name)
        self.addCleanup(self.s.stop)
        self.job = Job(id="j1", type="avatar", payload={})

    def test_mock_mode_gets_the_offline_client(self):
        self.assertIsInstance(self.s.comfy, OfflineComfyClient)
        self.assertFalse(self.s.comfy.is_up())
        healthy, why = self.s.comfy.health()
        self.assertFalse(healthy)
        self.assertIn("offline", why)

    def test_require_comfy_refuses_in_mock_mode(self):
        """The one gate every real-render path passes through: in mock mode
        it must refuse BEFORE any probe — the old code asked whatever
        ComfyUI answered on this box and then rendered through it."""
        with self.assertRaises(PermanentError) as ctx:
            self.s._require_comfy(self.job)
        self.assertIn("mock", str(ctx.exception).lower())

    def test_stubbed_fake_still_passes_the_gate(self):
        """Dozens of tests drive real-render paths from a mock-configured
        Services by stubbing `services.comfy = Fake()` — the gate keys on
        the client's own `offline` flag, so those fakes sail through."""
        class Fake:
            def is_up(self):
                return True

        self.s.comfy = Fake()
        self.s._require_comfy(self.job)  # must not raise

    def test_unexpected_client_use_fails_loudly(self):
        with self.assertRaises(AttributeError) as ctx:
            self.s.comfy.submit({})
        self.assertIn("offline", str(ctx.exception))

    def test_cancel_of_a_running_render_job_needs_no_comfy(self):
        """cancel_job pokes comfy.interrupt() for running render jobs; with
        the offline client that is a quiet no-op, not an HTTP interrupt
        fired into an instance this process does not own."""
        job = self.s.queue.enqueue("image_edit", {"asset_id": "x",
                                                  "prompt": "p"})
        job.state = type(job.state)("running")  # running, cooperatively
        self.assertTrue(self.s.cancel_job(job.id))
        self.assertFalse(any("interrupt" in e["msg"].lower()
                             for e in job.logs))


class EnsureModelGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.job = Job(id="j2", type="workflow", payload={})

    def _first_missing_model(self, s: Services) -> str:
        name = next(m.name for m in s.registry.list()
                    if not s.registry.is_ready(m.name))
        return name

    def test_render_path_ensure_respects_auto_install_off(self):
        s = _services(self.tmp.name, auto_install=False)
        self.addCleanup(s.stop)
        calls = []
        s.downloader.download = lambda *a, **k: calls.append(a)
        with self.assertRaises(PermanentError) as ctx:
            s._ensure_model(self._first_missing_model(s), self.job)
        self.assertIn("auto-install is off", str(ctx.exception))
        self.assertEqual(calls, [])

    def test_explicit_request_downloads_despite_auto_install_off(self):
        """A Models-page click arrives as a model_download job — that is
        the user asking, not auto-install; it must keep working."""
        s = _services(self.tmp.name, auto_install=False)
        self.addCleanup(s.stop)
        name = self._first_missing_model(s)
        calls = []
        s.downloader.download = lambda n, cb=None: calls.append(n)
        s.model_search.list_weight_files = lambda repo: []
        s._ensure_model(name, self.job, requested=True)
        self.assertEqual(calls, [name])


class DownloadCancelTests(unittest.TestCase):
    def test_progress_callback_aborts_when_cancelled(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        s = _services(tmp.name)
        self.addCleanup(s.stop)
        job = Job(id="j3", type="model_download", payload={})
        progress = s._download_progress(job, "some-model")
        progress(1 << 16, 1 << 30)  # streaming normally: no complaint
        job.cancel_requested = True
        with self.assertRaises(TransientError) as ctx:
            progress(2 << 16, 1 << 30)
        self.assertIn("cancelled", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
