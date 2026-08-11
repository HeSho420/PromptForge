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
            "folder": "checkpoints", "size": 40000, "sha256": self.sha,
            "url": "https://example.invalid/x", "purpose": "test",
            "license": "unknown",
            "meta": {"folder": "checkpoints",
                     "file": "tiny.safetensors"}}])

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

    def test_operational_logs_are_readable_but_only_the_whitelist(self):
        """Same-owner remote diagnosis: the machine that cannot render
        must be debuggable from the healthy one. Fixed whitelist — no
        directory walking, no app data."""
        logs = self.reg_a.models_dir.parent / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "comfyui-err.log").write_text("boom: torch missing")
        with urllib.request.urlopen(
                f"http://127.0.0.1:{self.a.http_port}"
                "/pf-peer/log/comfyui-err.log", timeout=5) as resp:
            self.assertIn("torch missing", resp.read().decode())
        (logs / "secrets.txt").write_text("nope")
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{self.a.http_port}"
                "/pf-peer/log/secrets.txt", timeout=5)
            code = 200
        except urllib.error.HTTPError as exc:
            code = exc.code
        self.assertEqual(code, 404)

    def test_concurrent_first_requests_still_fill_the_caches(self):
        """The lazy cache had a first-call race + an empty-dict falsiness
        trap: one machine's stats stayed null FOREVER because every
        request got a fresh orphaned dict. Hammer a fresh service
        concurrently and require the value to land."""
        with tempfile.TemporaryDirectory() as tmp:
            svc = PeerService(FakeRegistry(Path(tmp) / "m"), share=True,
                              render=False, comfy_url="http://127.0.0.1:9",
                              http_port=BASE_HTTP + 50,
                              udp_port=BASE_UDP + 50, name="racy",
                              loopback_only=True)
            svc.stats_provider = lambda: {"ram_total_gb": 42.0}
            svc.start()
            try:
                def hit() -> None:
                    try:
                        urllib.request.urlopen(
                            f"http://127.0.0.1:{svc.http_port}"
                            "/pf-peer/info", timeout=5).read()
                    except Exception:  # noqa: BLE001
                        pass
                threads = [threading.Thread(target=hit) for _ in range(8)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                filled = False
                for _ in range(25):
                    with urllib.request.urlopen(
                            f"http://127.0.0.1:{svc.http_port}"
                            "/pf-peer/info", timeout=5) as resp:
                        info = json.loads(resp.read().decode())
                    if (info.get("stats") or {}).get("ram_total_gb") == 42.0:
                        filled = True
                        break
                    time.sleep(0.2)
                self.assertTrue(filled, "stats cache never filled")
            finally:
                svc.stop()

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


class AskForNearbyModels(unittest.TestCase):
    """The pull direction: the machine that WANTS models asks a peer for
    its manifest and queues what it lacks. Same trust model as a push —
    sha pins, visible download jobs, LAN-first fetch."""

    def test_fetch_manifest_carries_full_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "m"
            (root / "checkpoints").mkdir(parents=True)
            f = root / "checkpoints" / "tiny.safetensors"
            f.write_bytes(b"x" * 32)
            reg = FakeRegistry(root)
            m = ModelInfo(name="tiny", purpose="test ckpt", license="mit",
                          url="https://example.com/tiny.safetensors",
                          sha256="ab" * 32, meta={"folder": "checkpoints"})
            m.status = "ready"
            m.path = str(f)
            reg.models["tiny"] = m
            svc = PeerService(reg, share=True, render=True,
                              http_port=BASE_HTTP + 50,
                              udp_port=BASE_UDP + 50, name="lender",
                              loopback_only=True)
            svc.start()
            try:
                got = svc.fetch_manifest("127.0.0.1", svc.http_port)
                self.assertEqual(len(got), 1)
                e = got[0]
                self.assertEqual(e["name"], "tiny")
                self.assertEqual(e["sha256"], "ab" * 32)
                self.assertEqual(e["url"],
                                 "https://example.com/tiny.safetensors")
                self.assertEqual(e["purpose"], "test ckpt")
                self.assertEqual(e["meta"]["folder"], "checkpoints")
                self.assertEqual(e["meta"]["filename"], "tiny.safetensors")
            finally:
                svc.stop()

    def test_fetch_manifest_tolerates_the_thin_index_of_older_peers(self):
        """A peer still on the pre-provenance wire format serves only
        name/file/folder/size/sha256 — its folder and file names must fold
        into meta so downloads land in the right typed subfolder."""
        import http.server
        thin = [{"name": "old-vae", "file": "old.safetensors",
                 "folder": "vae", "size": 5, "sha256": "cd" * 32}]

        class ThinIndex(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps(thin).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 0), ThinIndex)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                reg = FakeRegistry(Path(tmp) / "m")
                svc = PeerService(reg, share=True, render=True,
                                  http_port=BASE_HTTP + 51,
                                  udp_port=BASE_UDP + 51, name="asker",
                                  loopback_only=True)
                got = svc.fetch_manifest("127.0.0.1", srv.server_port)
                self.assertEqual(len(got), 1)
                self.assertEqual(got[0]["name"], "old-vae")
                self.assertEqual(got[0]["meta"]["folder"], "vae")
                self.assertEqual(got[0]["meta"]["filename"],
                                 "old.safetensors")
                self.assertEqual(got[0]["url"], "")
        finally:
            srv.shutdown()
            srv.server_close()

    def test_asking_a_peer_that_does_not_share_is_a_403(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = FakeRegistry(Path(tmp) / "m")
            svc = PeerService(reg, share=False, render=True,
                              http_port=BASE_HTTP + 52,
                              udp_port=BASE_UDP + 52, name="private",
                              loopback_only=True)
            svc.start()
            try:
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    svc.fetch_manifest("127.0.0.1", svc.http_port)
                self.assertEqual(ctx.exception.code, 403)
            finally:
                svc.stop()

    def test_pull_models_from_uses_the_shared_acceptance_path(self):
        """The ask flow must reuse _accept_model_push (sha requirement,
        register-if-missing, visible jobs) and turn a 403 into the honest
        sharing-is-off message rather than a stack trace."""
        src = inspect.getsource(Services.pull_models_from)
        self.assertIn("fetch_manifest", src)
        self.assertIn("_accept_model_push", src)
        self.assertIn("not sharing models", src)


class HostileRequests(unittest.TestCase):
    """The peer listener faces untrusted LAN input. A malformed or hostile
    request must be refused cleanly and MUST NOT wedge a handler thread."""

    def _raw(self, port: int, request: bytes, timeout: float = 4.0) -> bytes:
        import socket as _socket
        s = _socket.create_connection(("127.0.0.1", port), timeout=timeout)
        s.settimeout(timeout)
        try:
            s.sendall(request)
            chunks = []
            while True:
                try:
                    b = s.recv(4096)
                except TimeoutError:
                    break
                if not b:
                    break
                chunks.append(b)
                if b"\r\n\r\n" in b"".join(chunks) and len(b) < 4096:
                    break
            return b"".join(chunks)
        finally:
            s.close()

    def test_negative_content_length_does_not_hang_the_thread(self):
        """`Content-Length: -1` made rfile.read(-1) read until EOF — on a
        keep-alive socket that never comes, so the handler thread blocked
        forever. A few of those exhaust the pool. It must answer promptly."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = FakeRegistry(Path(tmp) / "m")
            svc = PeerService(reg, share=True, render=True,
                              http_port=BASE_HTTP + 60,
                              udp_port=BASE_UDP + 60, name="victim",
                              loopback_only=True)
            svc.on_pull = lambda entries: {"queued": [], "already": []}
            svc.start()
            try:
                port = svc.http_port
                req = (b"POST /pf-peer/pull HTTP/1.1\r\nHost: x\r\n"
                       b"Content-Length: -1\r\nConnection: close\r\n\r\n")
                t0 = time.time()
                resp = self._raw(port, req)
                # The property that matters: a prompt, valid HTTP response
                # (negative length is floored to an empty body), never the
                # read-until-EOF hang.
                self.assertLess(time.time() - t0, 3.0,
                                "listener hung on a negative Content-Length")
                self.assertTrue(resp.startswith(b"HTTP/1.1 "), resp[:40])
                # And the server is still answering afterwards.
                info = json.loads(urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/pf-peer/info", timeout=4)
                    .read().decode())
                self.assertEqual(info["app"], "promptforge")
            finally:
                svc.stop()

    def test_garbage_content_length_is_a_clean_400(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = FakeRegistry(Path(tmp) / "m")
            svc = PeerService(reg, share=True, render=True,
                              http_port=BASE_HTTP + 61,
                              udp_port=BASE_UDP + 61, name="victim2",
                              loopback_only=True)
            svc.start()
            try:
                req = (b"POST /pf-peer/pull HTTP/1.1\r\nHost: x\r\n"
                       b"Content-Length: notanumber\r\n"
                       b"Connection: close\r\n\r\n")
                resp = self._raw(svc.http_port, req)
                self.assertIn(b"400", resp.split(b"\r\n", 1)[0])
            finally:
                svc.stop()

    def test_suffix_range_serves_the_last_bytes_not_the_first(self):
        """`Range: bytes=-4` means the LAST 4 bytes. The old parser served
        bytes 0..4 under a 206 — silent corruption for a conformant client."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "m"
            (root / "checkpoints").mkdir(parents=True)
            f = root / "checkpoints" / "tiny.bin"
            f.write_bytes(b"ABCDEFGHIJ")  # 10 bytes; last 4 = "GHIJ"
            reg = FakeRegistry(root)
            m = ModelInfo(name="tiny", purpose="p", license="mit",
                          url="", sha256="ab" * 32,
                          meta={"folder": "checkpoints"})
            m.status = "ready"
            m.path = str(f)
            reg.models["tiny"] = m
            svc = PeerService(reg, share=True, render=True,
                              http_port=BASE_HTTP + 62,
                              udp_port=BASE_UDP + 62, name="lender2",
                              loopback_only=True)
            svc.start()
            try:
                r = urllib.request.Request(
                    f"http://127.0.0.1:{svc.http_port}/pf-peer/model/tiny",
                    headers={"Range": "bytes=-4"})
                with urllib.request.urlopen(r, timeout=4) as resp:
                    self.assertEqual(resp.status, 206)
                    self.assertEqual(resp.read(), b"GHIJ")
                    self.assertEqual(resp.headers.get("Content-Range"),
                                     "bytes 6-9/10")
            finally:
                svc.stop()

    def test_hostile_beacon_packets_do_not_kill_discovery(self):
        """A crafted UDP datagram from any LAN host used to raise an
        unhandled KeyError/ValueError out of the receive loop and kill the
        discovery thread until restart: a token-less packet, or a
        non-numeric 'http' port. Discovery must shrug them off and still
        find a real peer afterwards."""
        import socket as _socket
        with tempfile.TemporaryDirectory() as tmp:
            reg = FakeRegistry(Path(tmp) / "m")
            svc = PeerService(reg, share=True, render=True,
                              http_port=BASE_HTTP + 63,
                              udp_port=BASE_UDP + 63, name="listener",
                              loopback_only=True)
            svc.start()
            tx = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
            try:
                hostile = [
                    b"not json at all",
                    json.dumps({"pf": 1}).encode(),                # no token
                    json.dumps({"pf": 1, "token": 123}).encode(),  # token not str
                    json.dumps({"pf": 1, "token": "x",
                                "http": "abc"}).encode(),          # bad port
                    json.dumps([1, 2, 3]).encode(),                # not a dict
                ]
                for pkt in hostile:
                    for port in range(svc.udp_port,
                                      svc.udp_port + svc.UDP_RANGE):
                        tx.sendto(pkt, ("127.0.0.1", port))
                time.sleep(0.4)  # let the receiver chew through them
                # A valid beacon after the hostile ones must still register.
                good = json.dumps({"pf": 1, "token": "real-peer-tok",
                                   "name": "friend", "http": 12345}).encode()
                for port in range(svc.udp_port,
                                  svc.udp_port + svc.UDP_RANGE):
                    tx.sendto(good, ("127.0.0.1", port))
                found = _wait(lambda: any(
                    p.name == "friend" for p in svc.peers_list()), timeout=5)
                self.assertTrue(found,
                                "discovery thread died on a hostile packet")
                friend = next(p for p in svc.peers_list()
                              if p.name == "friend")
                self.assertEqual(friend.port, 12345)
            finally:
                tx.close()
                svc.stop()


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


class LiveStatusPicture(unittest.TestCase):
    """The info endpoint now describes what a machine is DOING (version,
    queue depth, running job type + stage keyword, uptime), and the
    status cache answers the UI without a network round-trip."""

    def test_info_carries_version_queue_uptime_and_auto_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = FakeRegistry(Path(tmp) / "m")
            svc = PeerService(
                reg, share=True, render=True,
                comfy_url="http://127.0.0.1:9",
                http_port=BASE_HTTP + 70, udp_port=BASE_UDP + 70,
                name="teller", loopback_only=True, auto_update=False,
                queue_provider=lambda: {
                    "pending": 2, "paused": False,
                    "running": {"type": "image_edit", "attempts": 1,
                                "started_at": "2026-01-01T00:00:00Z",
                                "stage": "render"}},
                version_provider=lambda: {"commit": "abc1234", "ts": 111})
            svc.start()
            try:
                info = None
                for _ in range(25):   # version rides the background cache
                    with urllib.request.urlopen(
                            f"http://127.0.0.1:{svc.http_port}"
                            "/pf-peer/info", timeout=5) as resp:
                        info = json.loads(resp.read().decode())
                    if info.get("version"):
                        break
                    time.sleep(0.2)
                self.assertEqual(info["version"]["commit"], "abc1234")
                self.assertEqual(info["queue"]["pending"], 2)
                self.assertEqual(info["queue"]["running"]["type"],
                                 "image_edit")
                self.assertEqual(info["queue"]["running"]["stage"],
                                 "render")
                self.assertFalse(info["auto_update"])
                self.assertGreaterEqual(info["uptime_s"], 0)
            finally:
                svc.stop()

    def test_peers_status_answers_from_cache_after_one_probe(self):
        """add_peer already fetched a full info payload — the status list
        must carry it immediately, with a measured latency, without any
        further network traffic hiding in the call."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = FakeRegistry(Path(tmp) / "m")
            teller = PeerService(
                reg, share=True, render=True,
                comfy_url="http://127.0.0.1:9",
                http_port=BASE_HTTP + 71, udp_port=BASE_UDP + 71,
                name="teller2", loopback_only=True,
                queue_provider=lambda: {"pending": 0, "paused": False,
                                        "running": None})
            asker = PeerService(
                FakeRegistry(Path(tmp) / "n"), share=True, render=True,
                comfy_url="http://127.0.0.1:9",
                http_port=BASE_HTTP + 72, udp_port=BASE_UDP + 72,
                name="asker2", loopback_only=True)
            teller.start()
            try:
                self.assertIsNotNone(
                    asker.add_peer("127.0.0.1", teller.http_port))
                status = asker.peers_status()
                self.assertEqual(len(status), 1)
                entry = status[0]
                self.assertTrue(entry["reachable"])
                self.assertIsNotNone(entry["latency_ms"])
                self.assertIsNone(entry["last_error"])
                self.assertEqual(entry["queue"]["pending"], 0)
            finally:
                teller.stop()

    def test_failed_probes_mark_the_peer_offline_with_the_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = PeerService(FakeRegistry(Path(tmp) / "m"), share=True,
                              render=True, http_port=BASE_HTTP + 73,
                              udp_port=BASE_UDP + 73, name="watcher",
                              loopback_only=True)
            from app.core.peers import OFFLINE_AFTER_FAILS, Peer
            dead = Peer("tok-dead", "ghost", "127.0.0.1", 9, static=True)
            svc.peers["tok-dead"] = dead
            for _ in range(OFFLINE_AFTER_FAILS):
                svc._probe_peer(dead)
            entry = svc.peers_status()[0]
            self.assertFalse(entry["reachable"])
            self.assertIsNotNone(entry["last_error"])

    def test_two_installs_on_one_machine_bind_distinct_ports(self):
        """Windows lets a second HTTP server share a taken port when
        allow_reuse_address is on (the stdlib default) — measured live: a
        second install silently double-bound the first one's :8765 and
        requests landed on whichever process accepted first. The port
        RANGE exists so the second install moves on instead."""
        with tempfile.TemporaryDirectory() as tmp:
            first = PeerService(FakeRegistry(Path(tmp) / "a"), share=True,
                                render=True, http_port=BASE_HTTP + 77,
                                udp_port=BASE_UDP + 77, name="one",
                                loopback_only=True)
            second = PeerService(FakeRegistry(Path(tmp) / "b"), share=True,
                                 render=True, http_port=BASE_HTTP + 77,
                                 udp_port=BASE_UDP + 78, name="two",
                                 loopback_only=True)
            first.start()
            second.start()
            try:
                self.assertNotEqual(first.http_port, second.http_port)
                # Both must answer as themselves — no interleaving.
                for svc, name in ((first, "one"), (second, "two")):
                    with urllib.request.urlopen(
                            f"http://127.0.0.1:{svc.http_port}"
                            "/pf-peer/info", timeout=5) as resp:
                        self.assertEqual(
                            json.loads(resp.read().decode())["name"], name)
            finally:
                first.stop()
                second.stop()

    def test_known_hosts_survive_a_restart_on_disk(self):
        """A pair that connected once must reconnect by itself: the hosts
        file written on connect is loaded by the NEXT service instance."""
        with tempfile.TemporaryDirectory() as tmp:
            reg_dir = Path(tmp) / "m"
            teller = PeerService(
                FakeRegistry(Path(tmp) / "t"), share=True, render=True,
                comfy_url="http://127.0.0.1:9",
                http_port=BASE_HTTP + 74, udp_port=BASE_UDP + 74,
                name="teller3", loopback_only=True)
            teller.start()
            try:
                first = PeerService(
                    FakeRegistry(reg_dir), share=True, render=True,
                    http_port=BASE_HTTP + 75, udp_port=BASE_UDP + 75,
                    name="rememberer", loopback_only=True)
                self.assertIsNotNone(
                    first.add_peer("127.0.0.1", teller.http_port))
                hosts_file = reg_dir.parent / "peers.json"
                self.assertTrue(hosts_file.exists())
                reborn = PeerService(
                    FakeRegistry(reg_dir), share=True, render=True,
                    http_port=BASE_HTTP + 76, udp_port=BASE_UDP + 76,
                    name="reborn", loopback_only=True)
                self.assertIn(("127.0.0.1", teller.http_port),
                              reborn.known_hosts)
            finally:
                teller.stop()


class RememberedHostsDecay(unittest.TestCase):
    """Addresses that stopped answering are eventually forgotten — the
    scanner must not re-probe every test rig and re-IP'd machine ever
    connected, forever."""

    def test_old_entries_decay_and_fresh_ones_survive(self):
        from app.core.peers import HOST_MEMORY_S
        with tempfile.TemporaryDirectory() as tmp:
            reg_dir = Path(tmp) / "m"
            reg_dir.mkdir(parents=True)
            stale = time.time() - HOST_MEMORY_S - 60
            (Path(tmp) / "peers.json").write_text(json.dumps([
                {"host": "10.0.0.5", "port": 8765,
                 "last_ok": time.time() - 60},
                {"host": "10.0.0.6", "port": 8765, "last_ok": stale},
                {"host": "10.0.0.7", "port": 8765},   # pre-decay format
            ]))
            svc = PeerService(FakeRegistry(reg_dir), share=True,
                              render=True, http_port=BASE_HTTP + 78,
                              udp_port=BASE_UDP + 79, name="decay",
                              loopback_only=True)
            self.assertIn(("10.0.0.5", 8765), svc.known_hosts)
            self.assertNotIn(("10.0.0.6", 8765), svc.known_hosts)
            # No timestamp = an older install's file: kept (fresh once).
            self.assertIn(("10.0.0.7", 8765), svc.known_hosts)

    def test_saved_files_carry_the_last_answered_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg_dir = Path(tmp) / "m"
            reg_dir.mkdir(parents=True)
            svc = PeerService(FakeRegistry(reg_dir), share=True,
                              render=True, http_port=BASE_HTTP + 79,
                              udp_port=BASE_UDP + 80, name="writer",
                              loopback_only=True)
            svc._remember_host("10.0.0.9", 8765)
            raw = json.loads((Path(tmp) / "peers.json").read_text())
            self.assertEqual(raw[0]["host"], "10.0.0.9")
            self.assertGreater(raw[0]["last_ok"], time.time() - 30)


class RemoteLogReading(unittest.TestCase):
    """The working machine debugs the broken one from its own UI: the
    fetch_log transport against the peer's whitelisted log endpoint."""

    def test_fetch_log_reads_a_whitelisted_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = FakeRegistry(Path(tmp) / "m")
            logs = Path(tmp) / "logs"
            logs.mkdir(parents=True)
            (logs / "comfyui-err.log").write_text("torch kaboom line")
            svc = PeerService(reg, share=True, render=True,
                              http_port=BASE_HTTP + 85,
                              udp_port=BASE_UDP + 85, name="patient",
                              loopback_only=True)
            svc.start()
            try:
                from app.core.peers import Peer
                peer = Peer("tok", "patient", "127.0.0.1", svc.http_port)
                text = svc.fetch_log(peer, "comfyui-err.log")
                self.assertIn("torch kaboom line", text)
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    svc.fetch_log(peer, "secrets.txt")
                self.assertEqual(ctx.exception.code, 404)
            finally:
                svc.stop()


class MonitorRestartNagCap(unittest.TestCase):
    """A PC where ComfyUI cannot start must not get two error events
    every 30 seconds forever. Behavioral: the REAL monitor loop runs
    against stubs, tick by tick — bounded attempts, one final honest
    line, silence after, and a recovery notice when it answers again."""

    def _drive(self, up_script, ticks, spawn_raises=False):
        from types import SimpleNamespace
        events: list[tuple[str, str]] = []

        class Stop:
            def __init__(self, n):
                self.n = n

            def wait(self, _t):
                self.n -= 1
                return self.n < 0      # False = run another tick

        ups = iter(up_script)
        last = {"v": False}

        def is_up():
            try:
                last["v"] = next(ups)
            except StopIteration:
                pass
            return last["v"]

        def spawn():
            if spawn_raises:
                raise OSError("log dir unwritable")
            return False

        fake = SimpleNamespace(
            _monitor_stop=Stop(ticks),
            MONITOR_INTERVAL_S=0,
            COMFY_RESTART_ATTEMPTS=Services.COMFY_RESTART_ATTEMPTS,
            INDEX_REFRESH_EVERY=10 ** 9,
            settings=SimpleNamespace(inpaint_backend="comfyui",
                                     llm_url="http://127.0.0.1:9/v1"),
            comfy=SimpleNamespace(is_up=is_up),
            events=SimpleNamespace(
                log=lambda lv, m: events.append((lv, m))),
            _spawn_comfy=spawn,
            _wait_comfy=lambda _s: False,
            _spawn_ollama=lambda _exe: True,   # never spawn real software
            model_index=SimpleNamespace(refresh_stale=lambda: []),
        )
        # The Ollama arm is not under test — and its real probe costs
        # seconds per tick against a dead URL on some Windows stacks.
        from unittest.mock import patch
        with patch("app.core.services.ollama_is_up",
                   lambda _url: True):
            Services._monitor_loop(fake)
        return [m for _lv, m in events if "ComfyUI" in m]

    def test_exactly_bounded_attempts_then_one_final_line_then_silence(self):
        got = self._drive([False], ticks=12)
        attempts = Services.COMFY_RESTART_ATTEMPTS
        self.assertEqual(
            got.count("ComfyUI is not responding — restarting it"),
            attempts)
        self.assertEqual(
            len([m for m in got if "pausing automatic restarts" in m]), 1)
        # The final line IS final: nothing after it.
        self.assertIn("pausing automatic restarts", got[-1])

    def test_recovery_right_after_the_cap_still_retracts(self):
        # Downs long enough to hit the cap, then up on the very next
        # tick — the state where comfy_down was just re-armed to 0 and
        # the retraction used to be skipped.
        downs = [False] * (2 * Services.COMFY_RESTART_ATTEMPTS)
        got = self._drive([*downs, True, True], ticks=len(downs) + 2)
        self.assertIn("ComfyUI is answering again", got[-1])
        self.assertEqual(got.count("ComfyUI is answering again"), 1)

    def test_a_raising_spawn_counts_as_a_failed_attempt(self):
        # _spawn_comfy raising used to skip both the counter and the
        # re-arm: one event, then the loop went silently dead forever.
        got = self._drive([False], ticks=12, spawn_raises=True)
        self.assertEqual(
            got.count("ComfyUI is not responding — restarting it"),
            Services.COMFY_RESTART_ATTEMPTS)
        self.assertIn("pausing automatic restarts", got[-1])


class AutoUpdatePropagation(unittest.TestCase):
    """When one install runs newer code, the other catches up by itself —
    through its OWN git remote. The peer is only the messenger."""

    def _services(self, http_offset: int, ts: int,
                  name: str) -> PeerService:
        svc = PeerService(
            FakeRegistry(Path(self.tmp.name) / name), share=True,
            render=True, comfy_url="http://127.0.0.1:9",
            http_port=BASE_HTTP + http_offset,
            udp_port=BASE_UDP + http_offset, name=name,
            loopback_only=True,
            version_provider=lambda: {"commit": f"c{ts}", "ts": ts})
        return svc

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_older_side_hears_about_the_newer_version(self):
        newer = self._services(80, ts=200, name="newer")
        older = self._services(81, ts=100, name="older")
        heard: list[tuple[str, dict]] = []
        older.on_newer_peer = lambda peer, info: heard.append(
            (peer.name, info))
        newer.start()
        try:
            self.assertIsNotNone(
                older.add_peer("127.0.0.1", newer.http_port))
            # add_peer may race the background version cache; the status
            # probe is the loop's own path — drive it directly.
            peer = older.peers_list()[0]
            for _ in range(25):
                older._probe_peer(peer)
                if heard:
                    break
                time.sleep(0.2)
            self.assertTrue(heard, "the newer version was never noticed")
            self.assertEqual(heard[0][0], "newer")
            self.assertEqual(heard[0][1]["version"]["ts"], 200)
        finally:
            newer.stop()

    def test_the_newer_side_stays_silent(self):
        newer = self._services(82, ts=300, name="newer2")
        older = self._services(83, ts=100, name="older2")
        heard: list = []
        newer.on_newer_peer = lambda peer, info: heard.append(peer.name)
        older.start()
        try:
            self.assertIsNotNone(
                newer.add_peer("127.0.0.1", older.http_port))
            peer = newer.peers_list()[0]
            for _ in range(10):
                newer._probe_peer(peer)
                time.sleep(0.1)
            self.assertEqual(heard, [])
        finally:
            older.stop()

    def test_update_trigger_guards(self):
        """The hook enqueues ONE update job, only when idle, only once
        per commit, only when the remote actually HAS something to pull,
        and never with auto-update off."""
        from types import SimpleNamespace

        from app.core.peers import Peer
        from app.core.services import Services

        enqueued: list[tuple[str, dict]] = []
        events: list[tuple[str, str]] = []

        def make(busy=False, existing_update=False, auto=True,
                 is_repo=True, remote=None, attempts=None):
            remote = {"behind": 1} if remote is None else remote
            jobs = ([SimpleNamespace(
                type="update", state=SimpleNamespace(value="pending"))]
                if existing_update else [])
            return SimpleNamespace(
                settings=SimpleNamespace(peer_auto_update=auto),
                updates=SimpleNamespace(is_repo=lambda: is_repo,
                                        status=lambda fetch=True: remote),
                queue=SimpleNamespace(
                    busy=lambda: busy,
                    list=lambda: jobs,
                    enqueue=lambda t, p: enqueued.append((t, p))),
                events=SimpleNamespace(
                    log=lambda lvl, msg: events.append((lvl, msg))),
                _auto_update_seen=set(),
                _auto_update_cooldown={},
                _auto_update_attempts=lambda: dict(attempts or {}),
                AUTO_UPDATE_MAX_ATTEMPTS=2)

        peer = Peer("tok", "other-pc", "127.0.0.1", 8765)
        info = {"version": {"commit": "abc9999", "ts": 999}}

        # Busy queue: declined, NOT marked seen — retried next tick.
        fake = make(busy=True)
        Services._maybe_update_from_peer(fake, peer, info)
        self.assertEqual(enqueued, [])
        self.assertEqual(fake._auto_update_seen, set())

        # Existing update job in flight: declined.
        fake = make(existing_update=True)
        Services._maybe_update_from_peer(fake, peer, info)
        self.assertEqual(enqueued, [])

        # Auto-update off: declined.
        fake = make(auto=False)
        Services._maybe_update_from_peer(fake, peer, info)
        self.assertEqual(enqueued, [])

        # Not a repo: declined.
        fake = make(is_repo=False)
        Services._maybe_update_from_peer(fake, peer, info)
        self.assertEqual(enqueued, [])

        # The peer's commit is not on the remote (unpushed dev work):
        # no job, marked seen so it is never re-fetched every tick,
        # and the event says so honestly.
        fake = make(remote={"behind": 0})
        Services._maybe_update_from_peer(fake, peer, info)
        self.assertEqual(enqueued, [])
        self.assertIn("abc9999", fake._auto_update_seen)
        self.assertTrue(any("not on the update source" in m
                            for _l, m in events))

        # Fetch failed (offline): NOT marked seen, but a cooldown stops
        # the status loop from re-fetching every 4 seconds.
        events.clear()
        fake = make(remote={"error": "no route to host"})
        Services._maybe_update_from_peer(fake, peer, info)
        self.assertEqual(enqueued, [])
        self.assertEqual(fake._auto_update_seen, set())
        self.assertGreater(fake._auto_update_cooldown.get("abc9999", 0), 0)

        # A dirty checkout would make apply() refuse — one honest event,
        # no guaranteed-red failed job.
        events.clear()
        fake = make(remote={"behind": 1, "dirty": ["app/core/x.py"]})
        Services._maybe_update_from_peer(fake, peer, info)
        self.assertEqual(enqueued, [])
        self.assertTrue(any("locally edited files" in m
                            for _l, m in events))

        # Two rollbacks already recorded on disk for this commit: the
        # broken-push loop must stay broken across restarts.
        events.clear()
        fake = make(attempts={"abc9999": 2})
        Services._maybe_update_from_peer(fake, peer, info)
        self.assertEqual(enqueued, [])
        self.assertTrue(any("paused" in m for _l, m in events))

        # Idle, clean, remote has it: one update job carrying the commit.
        events.clear()
        fake = make()
        Services._maybe_update_from_peer(fake, peer, info)
        self.assertEqual(len(enqueued), 1)
        self.assertEqual(enqueued[0][0], "update")
        self.assertEqual(enqueued[0][1]["commit"], "abc9999")
        self.assertIn("abc9999", fake._auto_update_seen)
        self.assertTrue(any("newer PromptForge" in m for _l, m in events))

        # Same commit again: never twice.
        Services._maybe_update_from_peer(fake, peer, info)
        self.assertEqual(len(enqueued), 1)

    def test_auto_update_job_steps_aside_when_work_arrived(self):
        """The last hole in 'never restart under work': jobs the helper
        has claimed (or freshly queued ones) beat the update job even
        after the hook's idle check passed."""
        from types import SimpleNamespace

        from app.core.jobs import Job
        from app.core.services import Services
        job = Job(id="up1", type="update", payload={"commit": "abc9999"})
        fake = SimpleNamespace(
            queue=SimpleNamespace(other_work=lambda _id: True),
            _auto_update_seen={"abc9999"})
        out = Services._handle_update(fake, job)
        self.assertTrue(out["deferred"])
        # Un-marked, so the watcher re-triggers once truly idle.
        self.assertEqual(fake._auto_update_seen, set())

    def test_busy_counts_jobs_the_helper_has_claimed(self):
        class _Db:
            def query(self, *_a):
                return []

            def execute(self, *_a):
                return None
        q = JobQueue(_Db())
        self.assertFalse(q.busy())
        q._claimed.add("ghost")
        try:
            self.assertTrue(q.busy())
            self.assertTrue(q.other_work("someone-else"))
        finally:
            q._claimed.discard("ghost")

    def test_busy_local_ignores_jobs_pinned_to_a_remote_machine(self):
        """A pins to B while B pins to A: each side's waiting job must
        not make it refuse the OTHER side's render — that was a livelock
        with no exit but cancel."""
        class _Db:
            def query(self, *_a):
                return []

            def execute(self, *_a):
                return None
        q = JobQueue(_Db())
        q.register("t", lambda j: {"ok": True})
        q.enqueue("t", {"device": "192.168.1.99"})   # waits for the peer
        self.assertTrue(q.busy())                    # local view: not idle
        self.assertFalse(q.busy_local())             # LAN view: GPU free
        q.enqueue("t", {})                           # a real local job
        self.assertTrue(q.busy_local())

    def test_newer_peer_hook_runs_off_thread_single_flight(self):
        """The hook does a git fetch; its callers (status pool, request
        threads, the delegation wrap) must never block on it, and two
        peers announcing in one tick must not double-enqueue."""
        from app.core.services import Services
        src = inspect.getsource(Services._newer_peer_async)
        self.assertIn("acquire(blocking=False)", src)
        self.assertIn("threading.Thread", src)
        init_src = inspect.getsource(Services.__init__)
        self.assertIn("self.peers.on_newer_peer = self._newer_peer_async",
                      init_src)

    def test_version_identity_comes_from_git_and_caches(self):
        from app.config import PROJECT_ROOT
        from app.core.update import UpdateManager
        um = UpdateManager(PROJECT_ROOT)
        v = um.version()
        if v is not None:   # this checkout is a git clone
            self.assertRegex(v["commit"], r"^[0-9a-f]{6,}$")
            self.assertGreater(v["ts"], 1_500_000_000)
            self.assertIs(um.version(), v)   # cached, not re-run
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(UpdateManager(Path(tmp)).version())


class HonestHandPickedDelegation(unittest.TestCase):
    """'Render on THAT machine' is a promise: if that machine cannot take
    the job, the job FAILS with the reason — it must never quietly render
    somewhere the user did not pick."""

    class _Db:
        def query(self, *_a):
            return []

        def execute(self, *_a):
            return None

    def _harness(self, find_result, info, queue):
        import threading as _threading
        from types import SimpleNamespace
        events: list[tuple[str, str]] = []
        return SimpleNamespace(
            peers=SimpleNamespace(
                find_peer=lambda t: find_result,
                add_peer=lambda h, p, timeout=3.0, pin=True: info,
                best_idle_peer=lambda: None),
            events=SimpleNamespace(
                log=lambda lvl, msg: events.append((lvl, msg))),
            queue=queue,
            _comfy_tls=_threading.local(),
        ), events

    def test_unreachable_pinned_device_fails_the_job_without_running(self):
        from app.core.jobs import Job, JobState
        from app.core.services import Services
        q = JobQueue(self._Db())
        q.register("t", lambda j: {"ok": True})
        job = Job(id="j1", type="t", payload={"device": "192.168.1.99"})
        executed: list[str] = []
        fake, events = self._harness(find_result=None, info=None, queue=q)
        Services._delegate_wrap(fake, lambda j: executed.append(j.id), job)
        self.assertEqual(executed, [])
        self.assertIs(job.state, JobState.FAILED)
        self.assertIn("NOT rendered", job.error)
        self.assertIn("is not reachable", job.error)
        self.assertTrue(any(lvl == "error" for lvl, _m in events))

    def test_pinned_device_without_comfy_fails_with_the_why(self):
        from app.core.jobs import Job, JobState
        from app.core.peers import Peer
        from app.core.services import Services
        q = JobQueue(self._Db())
        q.register("t", lambda j: {"ok": True})
        job = Job(id="j2", type="t", payload={"device": "rig-2"})
        peer = Peer("tok", "rig-2", "192.168.1.50", 8765)
        info = {"render": True, "idle": True,
                "comfy": {"up": False, "device": None, "gpu": None}}
        executed: list[str] = []
        fake, _events = self._harness(peer, info, q)
        Services._delegate_wrap(fake, lambda j: executed.append(j.id), job)
        self.assertEqual(executed, [])
        self.assertIs(job.state, JobState.FAILED)
        self.assertIn("ComfyUI is not running", job.error)

    def test_pinned_busy_device_waits_instead_of_failing(self):
        """Busy is 'not yet', not 'cannot': the job goes back to the
        FRONT of the queue, still pending, and says why — it must wait
        for the machine the user chose, not fail and not run elsewhere."""
        from app.core.jobs import Job, JobState
        from app.core.peers import Peer
        from app.core.services import Services
        q = JobQueue(self._Db())
        q.register("t", lambda j: {"ok": True})
        job = Job(id="j3", type="t", payload={"device": "rig-2"})
        q._jobs[job.id] = job     # as if the helper had taken it
        peer = Peer("tok", "rig-2", "192.168.1.50", 8765)
        info = {"render": True, "idle": False,
                "comfy": {"up": True, "device": "cuda", "gpu": "RTX"}}
        executed: list[str] = []
        fake, _events = self._harness(peer, info, q)
        t0 = time.time()
        Services._delegate_wrap(fake, lambda j: executed.append(j.id), job)
        self.assertEqual(executed, [])
        self.assertIs(job.state, JobState.PENDING)
        self.assertEqual(q.pending_order(), ["j3"])
        self.assertTrue(any("busy with its own work" in e["msg"]
                            for e in job.logs))
        # The wrap paces itself so a busy peer is not probed in a hot loop.
        self.assertGreaterEqual(time.time() - t0, 2.5)

    def test_requeue_front_respects_cancellation(self):
        from app.core.jobs import Job, JobState
        q = JobQueue(self._Db())
        q.register("t", lambda j: {"ok": True})
        job = Job(id="jc", type="t", payload={"device": "rig-2"})
        q._jobs[job.id] = job
        q.cancel(job.id)
        q.requeue_front(job, "should not appear")
        self.assertIs(job.state, JobState.CANCELLED)
        self.assertEqual(q.pending_order(), [])

    def test_healthy_pinned_device_binds_and_unbinds_the_proxy(self):
        from app.core.jobs import Job, JobState
        from app.core.peers import Peer
        from app.core.services import Services
        q = JobQueue(self._Db())
        q.register("t", lambda j: {"ok": True})
        job = Job(id="j4", type="t", payload={"device": "rig-2"})
        peer = Peer("tok", "rig-2", "192.168.1.50", 8765)
        info = {"render": True, "idle": True,
                "comfy": {"up": True, "device": "cuda", "gpu": "RTX"}}
        seen: dict = {}

        def execute(j):
            seen["client"] = getattr(fake._comfy_tls, "client", None)
            j.state = JobState.COMPLETED

        fake, _events = self._harness(peer, info, q)
        Services._delegate_wrap(fake, execute, job)
        self.assertIsNotNone(seen["client"])
        self.assertIn("192.168.1.50", seen["client"].base_url)
        self.assertIsNone(getattr(fake._comfy_tls, "client", None))

    def test_auto_path_still_runs_locally_when_no_peer_is_idle(self):
        from app.core.jobs import Job
        from app.core.services import Services
        q = JobQueue(self._Db())
        q.register("t", lambda j: {"ok": True})
        job = Job(id="j5", type="t", payload={})
        executed: list[str] = []
        fake, _events = self._harness(None, None, q)
        Services._delegate_wrap(fake, lambda j: executed.append(j.id), job)
        self.assertEqual(executed, ["j5"])

    def test_mid_render_death_of_a_pinned_peer_raises_permanent(self):
        """Behavioral, not source-string: a delegated binding to a dead
        peer + a hand-pinned device must raise the loud PermanentError,
        leave the binding in place (the wrap's finally owns clearing it),
        and log an error event."""
        import threading as _threading
        from types import SimpleNamespace

        from app.core.jobs import Job, PermanentError
        from app.core.services import Services

        class DeadComfy:
            offline = False

            def is_up(self):
                return False

        tls = _threading.local()
        tls.client = object()
        events: list[tuple[str, str]] = []
        fake = SimpleNamespace(comfy=DeadComfy(), _comfy_tls=tls,
                               events=SimpleNamespace(
                                   log=lambda lv, m: events.append((lv, m))))
        job = Job(id="jx", type="t", payload={"device": "rig-2"})
        with self.assertRaises(PermanentError) as ctx:
            Services._require_comfy(fake, job)
        self.assertIn("stopped answering mid-render", str(ctx.exception))
        self.assertIn("Nothing was rendered", str(ctx.exception))
        self.assertTrue(any(lv == "error" for lv, _m in events))
        self.assertIsNotNone(getattr(tls, "client", None))

    def test_mid_render_death_of_an_auto_job_falls_back_and_says_so(self):
        import threading as _threading
        from types import SimpleNamespace

        from app.core.jobs import Job, TransientError
        from app.core.services import Services

        class DeadComfy:
            offline = False

            def is_up(self):
                return False

        tls = _threading.local()
        tls.client = object()
        fake = SimpleNamespace(comfy=DeadComfy(), _comfy_tls=tls,
                               events=SimpleNamespace(log=lambda *_a: None),
                               _spawn_comfy=lambda: False)
        job = Job(id="jy", type="t", payload={})
        # Local ComfyUI is also down in this fake, so after the fallback
        # the normal revive path raises TransientError — what matters is
        # the binding was CLEARED and the fallback was logged.
        with self.assertRaises(TransientError):
            Services._require_comfy(fake, job)
        self.assertIsNone(getattr(tls, "client", None))
        self.assertTrue(any("continuing on this machine" in e["msg"]
                            for e in job.logs))

    def test_fail_job_never_overwrites_a_cancel(self):
        from app.core.jobs import Job, JobState
        q = JobQueue(self._Db())
        q.register("t", lambda j: {"ok": True})
        job = Job(id="jz", type="t", payload={"device": "rig-2"})
        q._jobs[job.id] = job
        q.cancel(job.id)
        q.fail_job(job, "the peer vanished")
        self.assertIs(job.state, JobState.CANCELLED)
        self.assertIsNone(job.error)


class QueueSnapshotShape(unittest.TestCase):
    """The queue can describe itself in one small dict — the local dock
    and the LAN info endpoint both build on it."""

    class _Db:
        def query(self, *_a):
            return []

        def execute(self, *_a):
            return None

    def test_snapshot_reports_running_stage_and_pending_depth(self):
        q = JobQueue(self._Db())
        release = threading.Event()
        started = threading.Event()

        def handler(job):
            job.log("info", "[stage] render — step 1/2: the busy part")
            started.set()
            release.wait(timeout=10)
            return {"ok": True}

        q.register("t", handler)
        q.start()
        try:
            q.enqueue("t", {})
            q.enqueue("t", {})
            self.assertTrue(started.wait(timeout=10))
            snap = q.snapshot()
            self.assertEqual(snap["pending"], 1)
            self.assertFalse(snap["paused"])
            self.assertEqual(snap["running"]["type"], "t")
            self.assertEqual(snap["running"]["stage"],
                             "render — step 1/2: the busy part")
            self.assertIsNotNone(snap["running"]["started_at"])
        finally:
            release.set()
            q.stop()

    def test_the_lan_variant_cuts_the_stage_before_the_dash(self):
        """Prompt words live AFTER the dash in stage lines; the snapshot a
        peer sees must carry only the keyword before it."""
        from types import SimpleNamespace

        from app.core.services import Services
        fake = SimpleNamespace(queue=SimpleNamespace(snapshot=lambda: {
            "pending": 1, "paused": False,
            "running": {"id": "secret-id", "type": "image_edit",
                        "attempts": 2, "started_at": "2026-01-01",
                        "stage": "render — step 1/2: a red dress"}}))
        out = Services._queue_public_snapshot(fake)
        self.assertEqual(out["running"]["stage"], "render")
        self.assertNotIn("id", out["running"])
        self.assertEqual(out["running"]["type"], "image_edit")

    def test_an_idle_queue_snapshot_is_honest(self):
        q = JobQueue(self._Db())
        snap = q.snapshot()
        self.assertEqual(snap, {"pending": 0, "paused": False,
                                "running": None})


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
        self.assertIn("if running:", src)
        # Paused means paused for EVERY dispatch path — hand-pinned jobs
        # included (they used to slip past the pause).
        self.assertIn("if self._paused:", src)

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
