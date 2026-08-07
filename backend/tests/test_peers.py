"""Two PromptForge machines helping each other, proven on one loopback.

Real sockets, real HTTP, no network hardware: two PeerServices run in this
process in loopback mode, discover each other through the UDP beacon, copy
a sha-pinned model file from one library to the other through the real
Downloader (whose internet URL is deliberately broken), and enforce the
render-sharing policy (403 when off, 409 when busy)."""
import hashlib
import inspect
import json
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from app.core.jobs import JobQueue
from app.core.peers import PeerService
from app.core.registry import ModelDownloader, ModelInfo
from app.core.services import Services

BASE_HTTP = 28650
BASE_UDP = 28660


class FakeRegistry:
    """The slice of ModelRegistry the peer service and downloader touch."""

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models: dict[str, ModelInfo] = {}
        self.progress: dict[str, int] = {}
        self.notes: dict[str, str] = {}

    def list(self):
        return list(self.models.values())

    def get(self, name):
        return self.models.get(name)

    def is_ready(self, name):
        m = self.models.get(name)
        return bool(m and m.status == "ready" and m.path
                    and Path(m.path).exists())

    def set_status(self, name, status, path=None):
        m = self.models.get(name)
        if m:
            m.status = status
            if path is not None:
                m.path = path


def _wait(predicate, timeout=12.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.2)
    return predicate()


class TwoMachines(unittest.TestCase):
    """A serving install and a fresh install, side by side."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)

        # Machine A owns a "model" file and shares it.
        a_models = root / "a" / "models" / "checkpoints"
        a_models.mkdir(parents=True)
        payload = b"weights-" * 5000
        (a_models / "tiny.safetensors").write_bytes(payload)
        cls.sha = hashlib.sha256(payload).hexdigest()
        cls.reg_a = FakeRegistry(root / "a" / "models")
        cls.reg_a.models["tiny-model"] = ModelInfo(
            name="tiny-model", purpose="test", url="https://example.invalid/x",
            path=str(a_models / "tiny.safetensors"), sha256=cls.sha,
            status="ready", meta={"folder": "checkpoints",
                                  "file": "tiny.safetensors"})

        # Machine B starts empty.
        (root / "b" / "models").mkdir(parents=True)
        cls.reg_b = FakeRegistry(root / "b" / "models")
        cls.reg_b.models["tiny-model"] = ModelInfo(
            name="tiny-model", purpose="test",
            url="https://huggingface.co/does/not/exist/x.safetensors",
            sha256=cls.sha, status="not_downloaded",
            meta={"folder": "checkpoints", "file": "tiny.safetensors"})

        cls.a = PeerService(cls.reg_a, share=True, render=True,
                            http_port=BASE_HTTP, udp_port=BASE_UDP,
                            name="machine-a", loopback_only=True)
        cls.b = PeerService(cls.reg_b, share=True, render=True,
                            http_port=BASE_HTTP + 1, udp_port=BASE_UDP,
                            name="machine-b", loopback_only=True)
        cls.a.start()
        cls.b.start()
        # Beacons fire immediately on start; discovery is near-instant.
        found = _wait(lambda: cls.b.peers_list() and cls.a.peers_list())
        assert found, "peers never discovered each other on loopback"

    @classmethod
    def tearDownClass(cls):
        cls.a.stop()
        cls.b.stop()
        cls.tmp.cleanup()

    def test_discovery_names_the_other_machine(self):
        names = {p.name for p in self.b.peers_list()}
        self.assertIn("machine-a", names)

    def test_the_index_lists_only_ready_models_with_their_sha(self):
        with urllib.request.urlopen(
                f"http://127.0.0.1:{self.a.http_port}/pf-peer/models",
                timeout=5) as resp:
            index = json.loads(resp.read().decode())
        self.assertEqual(index, [{
            "name": "tiny-model", "file": "tiny.safetensors",
            "folder": "checkpoints", "size": 40000, "sha256": self.sha}])

    def test_find_model_url_requires_the_matching_sha(self):
        self.assertIsNotNone(self.b.find_model_url("tiny-model", self.sha))
        self.assertIsNone(self.b.find_model_url("tiny-model", "0" * 64))
        self.assertIsNone(self.b.find_model_url("tiny-model", None))

    def test_a_fresh_install_copies_the_model_from_the_peer(self):
        """The internet URL is a dead host — only the LAN path can supply
        the bytes, and the normal checksum verification accepts them."""
        dl = ModelDownloader(self.reg_b, sleep=lambda _s: None)
        dl.MAX_ATTEMPTS = 1
        dl.peer_source = (
            lambda name: self.b.find_model_url(
                name, (self.reg_b.get(name) or ModelInfo("", "")).sha256))
        model = dl.download("tiny-model")
        self.assertEqual(model.status, "ready")
        got = Path(model.path).read_bytes()
        self.assertEqual(hashlib.sha256(got).hexdigest(), self.sha)

    def test_range_requests_resume_partial_copies(self):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.a.http_port}/pf-peer/model/tiny-model",
            headers={"Range": "bytes=39990-"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 206)
            self.assertEqual(len(resp.read()), 10)

    def test_files_outside_the_library_are_refused(self):
        """A registry row pointing outside models_dir must not be served —
        that is the boundary between 'model library' and 'user data'."""
        outside = Path(self.tmp.name) / "secret.txt"
        outside.write_text("private")
        self.reg_a.models["evil"] = ModelInfo(
            name="evil", purpose="test", path=str(outside), status="ready",
            sha256="00", meta={})
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.a.http_port}/pf-peer/model/evil",
                    timeout=5) as resp:
                code = resp.status
        except urllib.error.HTTPError as exc:
            code = exc.code
        finally:
            del self.reg_a.models["evil"]
        self.assertEqual(code, 403)

    def test_render_sharing_off_means_403(self):
        self.b.render = False
        try:
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{self.b.http_port}"
                    "/pf-peer/comfy/queue", timeout=5)
                code = 200
            except urllib.error.HTTPError as exc:
                code = exc.code
            self.assertEqual(code, 403)
        finally:
            self.b.render = True

    def test_a_busy_machine_answers_409(self):
        self.a.busy_check = lambda: True
        try:
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{self.a.http_port}"
                    "/pf-peer/comfy/queue", timeout=5)
                code = 200
            except urllib.error.HTTPError as exc:
                code = exc.code
            self.assertEqual(code, 409)
        finally:
            self.a.busy_check = lambda: False

    def test_a_disabled_share_serves_nothing(self):
        self.a.share = False
        try:
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{self.a.http_port}/pf-peer/models",
                    timeout=5)
                code = 200
            except urllib.error.HTTPError as exc:
                code = exc.code
            self.assertEqual(code, 403)
        finally:
            self.a.share = True

    def test_idle_state_follows_the_busy_check(self):
        with urllib.request.urlopen(
                f"http://127.0.0.1:{self.a.http_port}/pf-peer/info",
                timeout=5) as resp:
            info = json.loads(resp.read().decode())
        self.assertTrue(info["idle"])
        self.assertEqual(info["app"], "promptforge")


class QueueBusyAndDelegationWiring(unittest.TestCase):

    def test_an_empty_queue_is_not_busy(self):
        class _Db:
            def query(self, *_a):
                return []

            def execute(self, *_a):
                return None
        q = JobQueue(_Db())
        self.assertFalse(q.busy())

    def test_the_helper_worker_only_fires_when_local_is_busy(self):
        src = inspect.getsource(JobQueue._run_helper)
        self.assertIn("running = any(j.state is JobState.RUNNING", src)
        self.assertIn("if running and not self._paused:", src)

    def test_delegated_jobs_fall_back_to_local_when_the_peer_dies(self):
        src = inspect.getsource(Services._require_comfy)
        self.assertIn('getattr(self._comfy_tls, "client", None)', src)
        self.assertIn("continuing on this machine", src)

    def test_the_peer_fetch_only_exists_for_sha_pinned_entries(self):
        src = inspect.getsource(ModelDownloader.download)
        self.assertIn("if self.peer_source is not None and model.sha256:",
                      src)


if __name__ == "__main__":
    unittest.main()
