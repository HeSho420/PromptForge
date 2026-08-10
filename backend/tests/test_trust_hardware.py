"""Tests: download trust gates + LLM judge, hardware tiering, setup job."""
import json
import tempfile
import unittest
from pathlib import Path

from app.config import Settings
from app.core.db import Database
from app.core.hardware import Hardware, llm_model_for, max_auto_download_bytes
from app.core.llm import LLMUnavailableError
from app.core.registry import DownloadError, ModelDownloader, ModelInfo, ModelRegistry
from app.core.services import Services
from app.core.trust import (
    Evidence,
    TrustJudge,
    UntrustedDownloadError,
    check_format,
    check_host,
    rule_verdict,
)
from tests.test_scout_critic_video import OneShotLLM


class HardGateTests(unittest.TestCase):
    def test_untrusted_host_blocked(self):
        with self.assertRaises(UntrustedDownloadError):
            check_host("https://evil.example.com/model.safetensors")

    def test_trusted_hosts_and_local_files_pass(self):
        check_host("https://huggingface.co/org/repo/resolve/main/m.safetensors")
        check_host("https://dl.fbaipublicfiles.com/sam.pth")
        check_host("file:///C:/tmp/weights.bin")  # local: fine (tests, mirrors)

    def test_pickle_over_network_needs_vetting(self):
        with self.assertRaises(UntrustedDownloadError):
            check_format("https://huggingface.co/x/y/resolve/main/m.ckpt", False)
        check_format("https://huggingface.co/x/y/resolve/main/m.ckpt", True)
        check_format("https://huggingface.co/x/y/resolve/main/m.safetensors", False)

    def test_downloader_enforces_gates(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db = Database(Path(tmp.name) / "t.sqlite3")
        self.addCleanup(db.close)
        registry = ModelRegistry(db, Path(tmp.name) / "models")
        registry.register(ModelInfo(
            name="evil", purpose="x", url="https://evil.example.com/m.safetensors"))
        with self.assertRaises(DownloadError) as ctx:
            ModelDownloader(registry).download("evil")
        self.assertIn("allowlist", str(ctx.exception))
        self.assertEqual(registry.get("evil").status, "failed")


class VerdictTests(unittest.TestCase):
    def _ev(self, **kw):
        base = dict(repo_id="someorg/model", filename="m.safetensors",
                    url="https://huggingface.co/someorg/model",
                    size_bytes=1000, sha256="a" * 64, downloads=500)
        base.update(kw)
        return Evidence(**base)

    def test_rules_require_checksum(self):
        self.assertFalse(rule_verdict(self._ev(sha256=None)).proceed)

    def test_rules_reject_unknown_low_adoption(self):
        self.assertFalse(rule_verdict(self._ev(downloads=3)).proceed)

    def test_rules_accept_known_org(self):
        v = rule_verdict(self._ev(repo_id="Comfy-Org/thing", downloads=0))
        self.assertTrue(v.proceed)

    def test_llm_can_reject_but_not_override_rules(self):
        # LLM says proceed, but rules reject (no sha) -> rejected.
        judge = TrustJudge(OneShotLLM(json.dumps({"proceed": True, "reason": "ok"})))
        self.assertFalse(judge.judge(self._ev(sha256=None)).proceed)
        # Rules accept, LLM rejects -> rejected with LLM reason.
        judge = TrustJudge(OneShotLLM(json.dumps(
            {"proceed": False, "reason": "typo-squat"})))
        v = judge.judge(self._ev())
        self.assertFalse(v.proceed)
        self.assertEqual(v.judged_by, "llm")

    def test_llm_down_falls_back_to_rules(self):
        judge = TrustJudge(OneShotLLM(error=LLMUnavailableError("down")))
        v = judge.judge(self._ev())
        self.assertTrue(v.proceed)
        self.assertEqual(v.judged_by, "rules")


class HardwareTests(unittest.TestCase):
    def test_tiers(self):
        self.assertEqual(Hardware("RTX 5090", 24, 64, 500).tier, "high")
        self.assertEqual(Hardware("RTX 4060", 8, 16, 100).tier, "mid")
        self.assertEqual(Hardware(None, 0, 8, 50).tier, "low")

    def test_llm_model_scales_with_hardware(self):
        self.assertEqual(llm_model_for(Hardware("x", 24, 64, 500)), "qwen2.5:14b")
        self.assertEqual(llm_model_for(Hardware("x", 8, 16, 100)), "qwen2.5:7b")
        self.assertEqual(llm_model_for(Hardware(None, 0, 8, 50)), "qwen2.5:3b")

    def test_auto_download_cap_scales_and_bounds(self):
        low = max_auto_download_bytes(Hardware(None, 0, 8, 40))
        mid = max_auto_download_bytes(Hardware("x", 8, 16, 100))
        self.assertLess(low, mid)
        self.assertLessEqual(mid, 12 * 1024**3)
        # never more than half the free disk
        tight = max_auto_download_bytes(Hardware("x", 24, 64, 4))
        self.assertLessEqual(tight, 2 * 1024**3)


class SetupJobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = Settings(
            data_dir=Path(self.tmp.name),
            inpaint_backend="mock", segment_backend="mock", critic_model="")

    def tearDown(self):
        self.tmp.cleanup()

    def test_first_run_enqueues_setup_and_llm_picks(self):
        services = Services(self.settings)
        try:
            # first run: hardware.json written + setup job pending
            self.assertTrue((Path(self.tmp.name) / "hardware.json").exists())
            pending = [j for j in services.queue.list() if j.type == "setup"]
            self.assertEqual(len(pending), 1)
            services.llm = OneShotLLM(json.dumps(
                {"download": ["sd15-inpaint"], "reason": "image checkpoint"}))

            # keep the resulting model_download jobs offline
            registry = services.registry

            class FakeDL:
                def download(inner, name, progress=None):
                    f = Path(self.tmp.name) / "models" / f"{name}.safetensors"
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_bytes(b"w")
                    registry.set_status(name, "ready", path=str(f))
                    return registry.get(name)

            services.downloader = FakeDL()
            services.start()
            done = services.queue.wait_for(pending[0].id, timeout=20)
            self.assertEqual(done.state.value, "completed")
            self.assertIn("sd15-inpaint", done.result["queued"])
        finally:
            services.stop()

    def test_second_run_does_not_repeat_setup(self):
        s1 = Services(self.settings)  # writes hardware.json
        s1.stop()
        s2 = Services(self.settings)
        try:
            # The first run's (interrupted) setup job is restored as history,
            # but no NEW setup may be enqueued on the second run.
            fresh = [j for j in s2.queue.list()
                     if j.type == "setup" and j.state.value == "pending"]
            self.assertEqual(fresh, [])
        finally:
            s2.stop()


if __name__ == "__main__":
    unittest.main()
