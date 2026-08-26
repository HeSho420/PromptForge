"""Machine faults are recognized, explained, and repaired — never fed to
the LLM graph-repair loop.

Seen live (user screenshot): a cudaErrorNoKernelImageForDevice traceback
wall reached the user AFTER two wasted LLM repairs. No graph edit can fix
a PyTorch build with no kernels for the card; the doctor now reproduces
the fault with a five-line probe, reinstalls matching wheels into
ComfyUI's own interpreter, restarts ComfyUI, and reruns the same graph."""
import inspect
import tempfile
import unittest
from pathlib import Path

from app.config import Settings
from app.core import services as services_module
from app.core.services import (
    Services,
    machine_fault_hint,
    torch_gpu_fault,
)


class TorchGpuFaultTests(unittest.TestCase):
    def test_wheels_signatures_classify(self):
        for text in (
            "CUDA error: no kernel image is available for execution on "
            "the device",
            "cudaErrorNoKernelImageForDevice",
            "RuntimeError: CUDA error: invalid device function",
            "AssertionError: Torch not compiled with CUDA enabled",
        ):
            fault = torch_gpu_fault(text)
            self.assertIsNotNone(fault, text)
            self.assertEqual(fault[0], "wheels", text)

    def test_driver_signatures_classify(self):
        fault = torch_gpu_fault(
            "CUDA driver version is insufficient for CUDA runtime version")
        self.assertEqual(fault[0], "driver")
        self.assertIn("driver", fault[1].lower())

    def test_oom_and_ordinary_errors_do_not(self):
        # OOM is a workload condition with its own machinery — calling it
        # an install fault would trigger a pointless reinstall.
        self.assertIsNone(torch_gpu_fault("CUDA out of memory"))
        self.assertIsNone(torch_gpu_fault("node 5 missing input 'image'"))
        self.assertIsNone(torch_gpu_fault(""))

    def test_machine_fault_hint_composes_both_classes(self):
        self.assertIn("paging file",
                      machine_fault_hint("OS error 1455 blah"))
        self.assertIn("PyTorch",
                      machine_fault_hint("no kernel image is available"))
        self.assertIsNone(machine_fault_hint("some graph error"))


class _Job:
    def __init__(self):
        self.lines = []

    def log(self, level, msg):
        self.lines.append(msg)


class RepairTorchCudaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.comfy_dir = Path(self.tmp.name) / "comfy"
        self.comfy_dir.mkdir()
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name) / "data", inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=str(self.comfy_dir)))
        self.addCleanup(self.s.stop)
        self.s._comfy_python = lambda base: "fake-python.exe"
        self.calls: list[list[str]] = []
        self.real_run = services_module.subprocess.run
        self.addCleanup(self._restore)

    def _restore(self):
        services_module.subprocess.run = self.real_run

    def _fake_run(self, script):
        """script: list of (returncode, stdout, stderr) per call."""
        state = {"i": 0}

        def run(cmd, **kwargs):
            self.calls.append(list(cmd))
            rc, out, err = script[min(state["i"], len(script) - 1)]
            state["i"] += 1

            class P:
                returncode = rc
                stdout = out
                stderr = err
            return P()
        services_module.subprocess.run = run

    def test_auto_install_off_touches_nothing(self):
        self.s.settings.auto_install = False
        self._fake_run([(0, "", "")])
        self.assertFalse(self.s._repair_torch_cuda(_Job()))
        self.assertEqual(self.calls, [])

    def test_a_passing_probe_means_no_reinstall(self):
        self.s.settings.auto_install = True
        self._fake_run([(0, "9.0\n", "")])
        self.assertTrue(self.s._repair_torch_cuda(_Job()))
        self.assertEqual(len(self.calls), 1)      # the probe only
        self.assertNotIn("pip", " ".join(self.calls[0]))

    def test_the_full_repair_path_picks_the_drivers_channel(self):
        self.s.settings.auto_install = True
        self._fake_run([
            (1, "", "CUDA error: no kernel image is available"),  # probe
            (0, "| NVIDIA-SMI ... CUDA Version: 12.7 |", ""),     # smi
            (0, "installed", ""),                                 # pip
            (0, "9.0\n", ""),                                     # re-probe
        ])
        job = _Job()
        self.assertTrue(self.s._repair_torch_cuda(job))
        pip = self.calls[2]
        self.assertIn("pip", pip)
        self.assertIn("--force-reinstall", pip)
        self.assertIn("https://download.pytorch.org/whl/cu126", pip)
        self.assertTrue(any("repaired" in m for m in job.lines))

    def test_a_prehistoric_driver_is_named_not_reinstalled(self):
        self.s.settings.auto_install = True
        self._fake_run([
            (1, "", "no kernel image is available"),              # probe
            (0, "CUDA Version: 11.0", ""),                        # smi
        ])
        job = _Job()
        self.assertFalse(self.s._repair_torch_cuda(job))
        self.assertEqual(len(self.calls), 2)      # never reached pip
        self.assertTrue(any("driver" in m.lower() for m in job.lines))


class RepairLoopRoutingTests(unittest.TestCase):
    def test_gpu_faults_never_reach_the_llm_repair_loop(self):
        src = inspect.getsource(Services)
        self.assertIn("fault = torch_gpu_fault(str(exc))", src)
        self.assertIn("no LLM repairs will be", src)
        # ...and the in-place repair runs at most once per job, then the
        # SAME graph reruns on the fresh build.
        self.assertIn('getattr(job, "_torch_repaired", False)', src)
        self.assertIn('"to load the repaired PyTorch build"', src)


class RunGraphHealedTests(unittest.TestCase):
    """The guarantee: EVERY render path executes graphs through the
    healing wrapper, so the GPU-fault class repairs itself in place and
    can never again halt rendering on a repairable machine."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))
        self.addCleanup(self.s.stop)

    def test_a_wheels_fault_repairs_restarts_and_reruns_the_same_graph(self):
        from app.adapters.comfyui import WorkflowRuntimeError

        calls = {"run": 0, "repair": 0, "restart": 0}

        class Comfy:
            def run_graph(inner, graph):
                calls["run"] += 1
                if calls["run"] == 1:
                    raise WorkflowRuntimeError(
                        "CUDA error: no kernel image is available")
                return "IMAGE", "pid-2"

        self.s.comfy = Comfy()
        self.s._repair_torch_cuda = lambda job: (
            calls.__setitem__("repair", calls["repair"] + 1) or True)
        self.s._restart_comfy = lambda job, why: (
            calls.__setitem__("restart", calls["restart"] + 1))
        job = _Job()
        out = self.s.run_graph_healed(job, {"g": 1})
        self.assertEqual(out, ("IMAGE", "pid-2"))
        self.assertEqual(calls, {"run": 2, "repair": 1, "restart": 1})
        # ...and at most once per job: a second fault raises instead.
        with self.assertRaises(WorkflowRuntimeError):
            calls["run"] = 0
            self.s.run_graph_healed(job, {"g": 1})

    def test_ordinary_errors_pass_straight_through(self):
        from app.adapters.comfyui import WorkflowRuntimeError

        class Comfy:
            def run_graph(inner, graph):
                raise WorkflowRuntimeError("node 7 missing input 'image'")

        self.s.comfy = Comfy()
        self.s._repair_torch_cuda = lambda job: self.fail("must not fire")
        with self.assertRaises(WorkflowRuntimeError):
            self.s.run_graph_healed(_Job(), {})

    def test_every_render_site_is_behind_the_wrapper(self):
        src = inspect.getsource(services_module)
        # the ONLY raw calls are the wrapper's own two.
        self.assertEqual(src.count("self.comfy.run_graph("), 2)
        self.assertGreaterEqual(
            src.count("self.run_graph_healed(job, "), 15)

    def test_startup_runs_the_selfcheck_off_mock(self):
        src = inspect.getsource(Services.start)
        self.assertIn("_startup_gpu_selfcheck", src)
        self.assertIn('inpaint_backend != "mock"', src)


class UiMockupIntentTests(unittest.TestCase):
    """The screenshot's own prompt — a game-menu mock-up with tabs and
    item labels — matched no text intent, so nothing warned that the
    labels would render as gibberish on the general model."""

    def test_the_live_prompt_and_its_class_match(self):
        from app.core.quality import text_render_intent, ui_mockup_intent
        live = ("based on the layout and style from this menu, create a "
                "mock-up for a new menu for item upgrades with tabs for "
                "killstreaks, perks, hunter shop, zombie shop and rank "
                "shop, each tab has multiple items that can be upgraded "
                "with Essence from lvl 1 to 3, use paging")
        self.assertTrue(ui_mockup_intent(live))
        self.assertTrue(text_render_intent(live))
        for yes in ("a wireframe of the settings screen",
                    "an inventory UI with 40 slots",
                    "a dashboard screen with panels and buttons"):
            self.assertTrue(ui_mockup_intent(yes), yes)
        for no in ("a photo of a woman on a beach",
                   "put her in a nightclub",
                   "remove the menu from the restaurant table photo"):
            self.assertFalse(ui_mockup_intent(no), no)

    def test_the_honest_message_names_the_mockup_class(self):
        src = inspect.getsource(Services)
        self.assertIn("text-dense by", src)
        self.assertIn("ui_mockup_intent", src)


if __name__ == "__main__":
    unittest.main()
