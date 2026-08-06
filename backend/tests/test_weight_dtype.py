"""Holding a too-large UNet in half the memory, without downloading anything.

wan2.2_ti2v_5B is 10.0 GB of fp16 weights on an 8.6 GB card with 15.7 GB of
system RAM behind it, and RAM is what OS-kills the load. Asking ComfyUI to
hold those same weights as fp8 halves the footprint; the file on disk does
not change, only how it is held in memory.

The judgement has to be narrow. A checkpoint that is already compressed
carries its own scales, so "casting" it to fp8 is not a precision choice —
it is corruption. And a model that comfortably fits should keep the better
picture that full precision gives it.
"""
import tempfile
import unittest
from pathlib import Path

from app.config import Settings
from app.core.services import Services


class FakeJob:
    def __init__(self):
        self.lines = []

    def log(self, _level, message):
        self.lines.append(message)


class WhichModelsGetCastDown(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))
        self.addCleanup(self.s.stop)
        self.models = self.s.settings.models_dir / "diffusion_models"
        self.models.mkdir(parents=True, exist_ok=True)
        # A card so small that any real file counts as "too big for it",
        # which is the condition under test — not the exact byte count.
        self.s.hardware.vram_gb = 0.001

    def write(self, name, kib=2048):
        """2 MiB against a 0.001 GB "card" stands in for 10 GB against 8.6 —
        comfortably over the threshold, so the test is about the decision and
        not about rounding."""
        (self.models / name).write_bytes(b"\0" * kib * 1024)
        return name

    def test_a_full_precision_model_too_big_for_the_card_is_cast_down(self):
        name = self.write("wan2.2_ti2v_5B_fp16.safetensors")
        self.assertEqual(self.s._weight_dtype_for(name), "fp8_e4m3fn")

    def test_a_model_that_fits_keeps_its_precision(self):
        name = self.write("wan2.2_ti2v_5B_fp16.safetensors")
        self.s.hardware.vram_gb = 64.0
        self.assertIsNone(self.s._weight_dtype_for(name))

    def test_an_already_quantised_model_is_never_touched(self):
        """It carries its own scales; reinterpreting them corrupts it."""
        for name in ("z_image_turbo_int8_convrot.safetensors",
                     "flux1-kontext-dev-Q4_K_S.gguf",
                     "some-model-nf4.safetensors",
                     "already_fp8_e4m3fn.safetensors",
                     "svdq-int4_r32-flux.safetensors"):
            self.write(name)
            self.assertIsNone(self.s._weight_dtype_for(name), name)

    def test_a_model_that_is_not_there_is_not_guessed_about(self):
        self.assertIsNone(self.s._weight_dtype_for("absent.safetensors"))

    def test_an_unknown_gpu_is_left_alone(self):
        """hardware.vram_gb is 0.0 when nvidia-smi is missing or the machine
        is CPU-only. With no floor, 0 * 0.75 = 0 made EVERY model "too big"
        and cast the lot to fp8 — on hardware where fp8 buys nothing."""
        name = self.write("wan2.2_ti2v_5B_fp16.safetensors")
        for vram in (0.0, -1.0):
            self.s.hardware.vram_gb = vram
            self.assertIsNone(self.s._weight_dtype_for(name), f"vram={vram}")

    def test_no_name_at_all(self):
        self.assertIsNone(self.s._weight_dtype_for(""))


class AppliedToTheGraph(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))
        self.addCleanup(self.s.stop)
        models = self.s.settings.models_dir / "diffusion_models"
        models.mkdir(parents=True, exist_ok=True)
        (models / "big_fp16.safetensors").write_bytes(b"\0" * 2048 * 1024)
        self.s.hardware.vram_gb = 0.001
        self.job = FakeJob()

    def graph(self, dtype="default"):
        return {"1": {"class_type": "UNETLoader",
                      "inputs": {"unet_name": "big_fp16.safetensors",
                                 "weight_dtype": dtype}}}

    def test_the_loader_is_rewritten_and_the_reason_is_reported(self):
        out = self.s._apply_hardware_limits(self.graph(), self.job)
        self.assertEqual(out["1"]["inputs"]["weight_dtype"], "fp8_e4m3fn")
        self.assertTrue(any("half the memory" in m for m in self.job.lines))

    def test_an_explicit_choice_in_the_template_is_respected(self):
        """Only 'default' means "nobody decided". A template that names a
        dtype has decided, and this must not overrule it."""
        out = self.s._apply_hardware_limits(self.graph("fp8_e5m2"), self.job)
        self.assertEqual(out["1"]["inputs"]["weight_dtype"], "fp8_e5m2")

    def test_a_gguf_loader_has_no_dtype_to_set(self):
        graph = {"1": {"class_type": "UnetLoaderGGUF",
                       "inputs": {"unet_name": "flux1-kontext-Q4_K_S.gguf"}}}
        out = self.s._apply_hardware_limits(graph, self.job)
        self.assertNotIn("weight_dtype", out["1"]["inputs"])


if __name__ == "__main__":
    unittest.main()
