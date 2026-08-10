"""The health monitor's two promises.

Mock mode means OFFLINE: a mocked instance must never probe, launch or
revive the machine's real ComfyUI/Ollama (measured live before the gate:
a mock instance launched 'ollama serve' once a minute for its whole
lifetime). And in real mode a revival is attempted once per DOWNTIME, not
once per minute forever — the endless identical error lines drowned the
Behind-the-Scenes stream when Ollama could not come back at all.
"""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import Settings
from app.core.services import Services


class _MonitorHarness(unittest.TestCase):
    """A Services instance whose monitor loop runs fast and whose service
    probes are spies. The monitor thread is driven directly — start() would
    also start the queue, peers and first-run machinery."""

    BACKEND = "mock"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend=self.BACKEND,
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir="", llm_url="http://127.0.0.1:9/v1"))
        self.addCleanup(self.s.stop)
        self.s.MONITOR_INTERVAL_S = 0.01  # instance attr shadows the class's
        self.comfy_probes = 0
        self.spawned_ollama: list[str] = []

        def probe():
            self.comfy_probes += 1
            return False  # "down" — the interesting case for both promises

        self.s.comfy.is_up = probe
        self.s._spawn_comfy = lambda: False
        self.s._spawn_ollama = lambda exe: self.spawned_ollama.append(exe)

    def run_monitor(self, seconds: float) -> None:
        self.s._monitor_stop.clear()
        t = threading.Thread(target=self.s._monitor_loop, daemon=True)
        t.start()
        time.sleep(seconds)
        self.s._monitor_stop.set()
        t.join(2)
        self.assertFalse(t.is_alive(), "monitor thread failed to stop")


class MockModeIsOffline(_MonitorHarness):
    BACKEND = "mock"

    def test_mock_mode_never_probes_or_revives_anything(self):
        with patch("app.core.services.ollama_is_up") as up, \
                patch("app.core.services.shutil.which") as which:
            self.run_monitor(0.3)  # ~30 ticks: plenty of both cadences
        self.assertEqual(self.comfy_probes, 0)
        self.assertEqual(self.spawned_ollama, [])
        up.assert_not_called()
        which.assert_not_called()


class RealModeRevivesOncePerDowntime(_MonitorHarness):
    BACKEND = "comfyui"

    def test_ollama_revival_is_once_per_downtime(self):
        alive = {"up": False}
        with patch("app.core.services.ollama_is_up",
                   side_effect=lambda url: alive["up"]), \
                patch("app.core.services.shutil.which",
                      return_value="C:/fake/ollama.exe"):
            self.s._monitor_stop.clear()
            t = threading.Thread(target=self.s._monitor_loop, daemon=True)
            t.start()
            time.sleep(0.3)          # many 4th-ticks while down
            self.assertEqual(len(self.spawned_ollama), 1,
                             "down forever must mean ONE revival, not one "
                             "per minute")
            alive["up"] = True       # it came back …
            time.sleep(0.2)          # … monitor sees it up, re-arms
            alive["up"] = False      # a NEW downtime
            time.sleep(0.3)
            self.s._monitor_stop.set()
            t.join(2)
        self.assertEqual(len(self.spawned_ollama), 2,
                         "a fresh downtime gets a fresh revival")

    def test_real_mode_still_probes_comfy(self):
        with patch("app.core.services.ollama_is_up", return_value=True):
            self.run_monitor(0.3)
        self.assertGreater(self.comfy_probes, 0)


if __name__ == "__main__":
    unittest.main()
