"""The resident CLIPSeg worker: fast when healthy, clean when not.

The engine itself needs ComfyUI's venv (torch/transformers live there), so
these tests drive the MANAGER against a stub tool speaking the same
line-JSON protocol with the backend's own interpreter. The real engine's
serve mode was verified live on this machine: cold start 132.5s under
load, then 2.1-4.9s per answer (previously 35-132s for EVERY answer).
"""
import sys
import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image

from app.core.services import _TextMaskWorker

# A stub that answers the worker protocol. Modes ride in phrases[0]:
#   "crash"  -> exit mid-request (no answer)
#   "hang"   -> never answer
#   "faint"  -> answer with a peak below the caller's floor
#   anything else -> confident answer + a real mask file
STUB = """\
import json, sys
from PIL import Image
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    mode = (req.get("phrases") or [""])[0]
    if mode == "crash":
        sys.exit(1)
    if mode == "hang":
        import time
        time.sleep(60)
    Image.new("L", (8, 8), 255).save(req["out"], format="PNG")
    peak = 0.08 if mode == "faint" else 0.9
    print(json.dumps({"peak": peak, "coverage": 0.5,
                      "control_peak": 0.05}), flush=True)
"""


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        tool = Path(self.tmp.name) / "stub_tool.py"
        tool.write_text(STUB)
        self.w = _TextMaskWorker(sys.executable, str(tool))
        self.addCleanup(lambda: self.w.stop(force=True))
        self.out = str(Path(self.tmp.name) / "mask.png")

    def ask(self, phrase: str):
        src = Path(self.tmp.name) / "src.png"
        Image.new("RGB", (8, 8)).save(src, format="PNG")
        return self.w.ask(str(src), self.out, [phrase], ["a control"], 0.4)

    def test_answers_and_stays_warm(self):
        report = self.ask("necklace")
        self.assertIsNotNone(report)
        self.assertEqual(report["peak"], 0.9)
        self.assertTrue(Path(self.out).exists())
        self.assertTrue(self.w.warm)
        # Second ask reuses the process — no new ready handshake needed.
        self.assertEqual(self.ask("shoes")["peak"], 0.9)

    def test_crash_answers_none_and_next_ask_respawns(self):
        self.assertIsNone(self.ask("crash"))
        self.assertFalse(self.w.warm)
        self.assertEqual(self.ask("necklace")["peak"], 0.9)

    def test_hang_is_bounded_and_kills_the_process(self):
        self.w.ASK_TIMEOUT_S = 0.5
        t0 = time.monotonic()
        self.assertIsNone(self.ask("hang"))
        self.assertLess(time.monotonic() - t0, 10)
        self.assertFalse(self.w.warm)

    def test_forced_stop_kills(self):
        self.ask("necklace")
        self.assertTrue(self.w.warm)
        self.w.stop(force=True)
        self.assertFalse(self.w.warm)

    def test_spawn_failure_is_a_quiet_none(self):
        w = _TextMaskWorker(sys.executable + ".does-not-exist", "nope.py")
        self.assertIsNone(w.ask("s", "o", ["x"], [], 0.4))


class ServicesWiringTests(unittest.TestCase):
    """_text_mask consumes worker reports with the same floor/margin gates
    the one-shot path had."""

    def setUp(self):
        import shutil

        from app.config import Settings
        from app.core.services import Services

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # comfyui_dir must exist and hold a "venv" for _comfy_python; point
        # both at the backend's own interpreter via a fake layout.
        fake = Path(self.tmp.name) / "comfy"
        (fake / ".venv" / "Scripts").mkdir(parents=True)
        shutil.copy(sys.executable, fake / ".venv" / "Scripts" / "python.exe")
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name) / "data", inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=str(fake), llm_url="http://127.0.0.1:9/v1"))
        self.addCleanup(self.s.stop)
        tool = Path(self.tmp.name) / "stub_tool.py"
        tool.write_text(STUB)
        self.s._text_mask_worker = _TextMaskWorker(
            str(fake / ".venv" / "Scripts" / "python.exe"), str(tool))
        self.image = Image.new("RGB", (64, 64), (30, 40, 50))

    def test_mock_mode_still_refuses_before_any_worker(self):
        found, report = self.s._text_mask(self.image, ["necklace"])
        self.assertIsNone(found)
        self.assertEqual(report, {})

    def test_confident_answer_returns_the_mask(self):
        self.s.settings.inpaint_backend = "comfyui"
        found, report = self.s._text_mask(self.image, ["necklace"])
        self.assertIsNotNone(found)
        self.assertEqual(found.size, self.image.size)
        self.assertEqual(report["peak"], 0.9)

    def test_faint_answer_is_a_not_found(self):
        self.s.settings.inpaint_backend = "comfyui"
        found, report = self.s._text_mask(self.image, ["faint"])
        self.assertIsNone(found)
        self.assertEqual(report["peak"], 0.08)  # the caller reads the WHY


if __name__ == "__main__":
    unittest.main()
