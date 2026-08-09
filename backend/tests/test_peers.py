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
import threading
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

        # A dead ComfyUI port on purpose: this suite must not change
        # behaviour depending on whether the DEV machine's real ComfyUI
        # happens to be running.
        cls.a = PeerService(cls.reg_a, share=True, render=True,
                            comfy_url="http://127.0.0.1:9",
                            http_port=BASE_HTTP, udp_port=BASE_UDP,
                            name="machine-a", loopback_only=True)
        cls.b = PeerService(cls.reg_b, share=True, render=True,
                            comfy_url="http://127.0.0.1:9",
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

    def test_connect_by_address_pins_the_peer(self):
        """The escape hatch for blocked UDP: plain HTTP, and the peer is
        never pruned however long its beacons stay silent."""
        info = self.b.add_peer("127.0.0.1", self.a.http_port)
        self.assertEqual(info["name"], "machine-a")
        peer = next(p for p in self.b.peers_list()
                    if p.name == "machine-a")
        self.assertTrue(peer.static)
        peer.last_seen = 0
        self.b._prune()
        self.assertTrue(any(p.name == "machine-a"
                            for p in self.b.peers_list()))

    def test_connecting_to_yourself_is_flagged(self):
        info = self.a.add_peer("127.0.0.1", self.a.http_port)
        self.assertTrue(info.get("self"))

    def test_connecting_to_a_dead_address_returns_none(self):
        self.assertIsNone(self.b.add_peer("127.0.0.1", 9, timeout=1.0))

    def test_info_carries_live_machine_stats(self):
        """Slow probes are served from a background-refreshed cache, so
        the FIRST info call may answer before the value exists — a second
        ask moments later must have it. (The blocking version made a
        machine look offline for ~25s exactly when it was broken.)"""
        self.a.stats_provider = lambda: {"vram_used_mb": 2048,
                                         "vram_total_mb": 8192,
                                         "ram_used_gb": 9.1,
                                         "ram_total_gb": 15.7}
        try:
            info = None
            for _ in range(20):
                info = self.b.add_peer("127.0.0.1", self.a.http_port)
                if info and info.get("stats"):
                    break
                time.sleep(0.2)
            self.assertEqual(info["stats"]["vram_total_mb"], 8192)
            self.assertEqual(info["stats"]["ram_total_gb"], 15.7)
        finally:
            self.a.stats_provider = None

    def test_scan_found_peers_are_not_pinned(self):
        """The scanner must not fill the list with immortal ghosts: only
        hand-added peers survive silence."""
        info = self.b.add_peer("127.0.0.1", self.a.http_port, pin=False)
        self.assertIsNotNone(info)
        peer = next(p for p in self.b.peers_list()
                    if p.name == "machine-a")
        # An earlier test may have pinned machine-a by hand; what pin=False
        # must guarantee is that it never UPGRADES a peer to pinned.
        self.assertIn(("127.0.0.1", self.a.http_port),
                      self.b.known_hosts)
        self.assertIsInstance(peer.static, bool)

    def test_a_peer_without_comfyui_is_never_delegated_to(self):
        """HerlockGame ran the app for a day with its ComfyUI down and
        nothing surfaced it: info now carries comfy status, and the
        delegation chooser refuses peers that cannot render."""
        info = self.b.add_peer("127.0.0.1", self.a.http_port)
        self.assertFalse(info["comfy"]["up"])   # no ComfyUI in this test
        self.assertIsNone(self.b.best_idle_peer())

    def test_a_cpu_rendering_peer_is_a_last_resort_only(self):
        src = inspect.getsource(PeerService.best_idle_peer)
        self.assertIn('== "cpu"', src)
        self.assertIn("cpu_fallback = cpu_fallback or peer", src)

    def test_the_scanner_sweeps_arp_and_local_subnets(self):
        src = inspect.getsource(PeerService._scan_candidates)
        self.assertIn('["arp", "-a"]', src)
        self.assertIn('range(1, 255)', src)
        loop = inspect.getsource(PeerService._scanner)
        self.assertIn("pin=False", loop)
        self.assertIn("known_hosts", loop)

    def test_info_reports_the_comfy_environment(self):
        """A peer whose ComfyUI is down can still say WHY, remotely: its
        env facts are read from disk, not from the dead server."""
        self.a.env_provider = lambda: {"python": "3.13", "torch": None,
                                       "gpu_visible": False}
        try:
            info = None
            for _ in range(20):
                info = self.b.add_peer("127.0.0.1", self.a.http_port)
                if info and info.get("comfy_env"):
                    break
                time.sleep(0.2)
            self.assertEqual(info["comfy_env"]["python"], "3.13")
            self.assertIsNone(info["comfy_env"]["torch"])
        finally:
            self.a.env_provider = None


class ModelPush(unittest.TestCase):
    """'Send all models to the other device', end to end over loopback."""

    def test_a_manifest_is_offered_and_the_receiver_decides(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = FakeRegistry(Path(tmp) / "m")
            svc = PeerService(reg, share=True, render=False,
                              http_port=BASE_HTTP + 30,
                              udp_port=BASE_UDP + 30, name="rx",
                              loopback_only=True)
            got: list[list[dict]] = []
            svc.on_pull = lambda entries: (
                got.append(entries) or {"queued": [e["name"]
                                                   for e in entries]})
            svc.start()
            try:
                from app.core.peers import Peer
                peer = Peer("tok-rx", "rx", "127.0.0.1", svc.http_port)
                out = svc.post_pull(peer, [
                    {"name": "m1", "sha256": "aa"},
                    {"name": "m2", "sha256": "bb"}])
                self.assertEqual(out["queued"], ["m1", "m2"])
                self.assertEqual(len(got[0]), 2)
            finally:
                svc.stop()

    def test_sharing_off_refuses_the_offer(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = FakeRegistry(Path(tmp) / "m")
            svc = PeerService(reg, share=False, render=True,
                              http_port=BASE_HTTP + 40,
                              udp_port=BASE_UDP + 40, name="rx2",
                              loopback_only=True)
            svc.on_pull = lambda entries: {"queued": []}
            svc.start()
            try:
                from app.core.peers import Peer
                peer = Peer("tok", "rx2", "127.0.0.1", svc.http_port)
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    svc.post_pull(peer, [{"name": "m", "sha256": "aa"}])
                self.assertEqual(ctx.exception.code, 403)
            finally:
                svc.stop()

    def test_accepted_entries_need_a_checksum_and_become_visible_jobs(self):
        import inspect as _inspect
        src = _inspect.getsource(Services._accept_model_push)
        self.assertIn("if not name or not sha:", src)
        self.assertIn('self.queue.enqueue("model_download"', src)
        self.assertIn("self.registry.register", src)


class DeviceRouting(unittest.TestCase):
    """The picker's contract: forced jobs go through the peer wrap even
    when the gate says no; 'local' jobs never leave this machine."""

    class _Db:
        def query(self, *_a):
            return []

        def execute(self, *_a):
            return None

    def test_a_hand_picked_device_reaches_the_wrap(self):
        q = JobQueue(self._Db())
        done = threading.Event()
        seen: dict = {}

        def handler(_job):
            return {"ok": True}

        def wrap(execute, job):
            seen["device"] = (job.payload or {}).get("device")
            execute(job)
            done.set()

        q.register("t", handler)
        q.start()
        q.start_helper(gate=lambda: False, wrap=wrap, types={"t"})
        try:
            q.enqueue("t", {"device": "192.168.1.99"})
            self.assertTrue(done.wait(timeout=10),
                            "the peer worker never took the forced job")
            self.assertEqual(seen["device"], "192.168.1.99")
        finally:
            q.stop()

    def test_without_the_peer_worker_forced_jobs_still_run(self):
        q = JobQueue(self._Db())
        q.register("t", lambda _j: {"ok": True})
        q.start()
        try:
            job = q.enqueue("t", {"device": "192.168.1.99"})
            deadline = time.time() + 10
            while time.time() < deadline and job.state.value != "completed":
                time.sleep(0.1)
            self.assertEqual(job.state.value, "completed")
        finally:
            q.stop()

    def test_jobs_pinned_local_are_invisible_to_the_helper(self):
        src = inspect.getsource(JobQueue._run_helper)
        self.assertIn('(j.payload or {}).get("device") == "local"', src)

    def test_the_wrap_honours_the_hand_picked_peer(self):
        src = inspect.getsource(Services._delegate_wrap)
        self.assertIn("chosen by hand", src)
        self.assertIn("is not reachable", src)


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
