"""FLUX.1 Kontext: the instruction-edit route, and not reloading 10 GB twice.

Two behaviours are covered here.

The route: Kontext edits from the sentence alone, with no mask, so the class
of failure where the mask found the wrong object cannot happen. It must claim
exactly the operations that are phrased as an instruction about a thing, and
leave the engines that beat it (an exact background matte, a real relighter)
alone.

The memory guard: ComfyUI's cached models are dropped before a large model
loads, because the two together OOM-kill it. But dropping a 10.8 GB stack to
immediately reload the identical 10.8 GB costs about 90 seconds and buys
nothing — the peak footprint is the same either way — so the drop must be
skipped when the incoming graph wants the same weights.
"""
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.config import Settings
from app.core import quality
from app.core.services import Services

KONTEXT = {
    "1": {"class_type": "UnetLoaderGGUF",
          "inputs": {"unet_name": "flux1-kontext-dev-Q4_K_S.gguf"}},
    "2": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
    "3": {"class_type": "KSampler", "inputs": {"steps": 20}},
}
OTHER_GGUF = {
    "1": {"class_type": "UnetLoaderGGUF",
          "inputs": {"unet_name": "wan2.2-i2v-Q4_K_S.gguf"}},
    "2": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
}
LIGHT = {"1": {"class_type": "CheckpointLoaderSimple",
               "inputs": {"ckpt_name": "sd15-inpaint.safetensors"}}}


class FakeJob:
    def __init__(self):
        self.lines = []

    def log(self, _level, message):
        self.lines.append(message)


class HeavySignature(unittest.TestCase):

    def test_the_same_weights_give_the_same_signature(self):
        self.assertEqual(Services._heavy_signature(KONTEXT),
                         Services._heavy_signature(dict(KONTEXT)))

    def test_different_weights_give_different_signatures(self):
        self.assertNotEqual(Services._heavy_signature(KONTEXT),
                            Services._heavy_signature(OTHER_GGUF))

    def test_sampler_settings_are_not_part_of_it(self):
        """Only what gets LOADED matters. Two renders of the same model at
        different step counts must still count as the same weights."""
        fewer = {**KONTEXT, "3": {"class_type": "KSampler",
                                  "inputs": {"steps": 4}}}
        self.assertEqual(Services._heavy_signature(KONTEXT),
                         Services._heavy_signature(fewer))


class MemoryGuard(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))
        self.addCleanup(self.s.stop)
        self.freed = []
        self.s.comfy.free_memory = lambda: (self.freed.append(1), True)[1]
        self.job = FakeJob()

    def test_preparing_a_graph_terminates(self):
        """It delegates to a same-named helper; calling itself instead of the
        helper recursed until the job died, and only a live render caught
        it."""
        self.s._prepare_graph(self.job, KONTEXT)

    def test_a_light_graph_never_touches_the_cache(self):
        self.s._prepare_graph(self.job, LIGHT)
        self.assertEqual(self.freed, [])

    def test_the_first_heavy_load_drops_whatever_was_there(self):
        self.s._prepare_graph(self.job, KONTEXT)
        self.assertEqual(len(self.freed), 1)

    def test_the_same_heavy_model_twice_reloads_nothing(self):
        self.s._prepare_graph(self.job, KONTEXT)
        self.s._prepare_graph(self.job, KONTEXT)
        self.assertEqual(len(self.freed), 1, "reloaded identical weights")
        self.assertTrue(any("still holds these weights" in m
                            for m in self.job.lines))

    def test_a_different_heavy_model_does_drop_the_cache(self):
        self.s._prepare_graph(self.job, KONTEXT)
        self.s._prepare_graph(self.job, OTHER_GGUF)
        self.assertEqual(len(self.freed), 2, "kept the wrong weights resident")

    def test_an_explicit_drop_is_not_forgotten(self):
        """The video path frees the cache directly. After that, ComfyUI holds
        nothing, and the next Kontext render must not assume otherwise."""
        self.s._prepare_graph(self.job, KONTEXT)
        self.s._drop_comfy_cache()
        self.s._prepare_graph(self.job, KONTEXT)
        self.assertIsNotNone(self.s._comfy_heavy_cached)
        self.assertEqual(len(self.freed), 3)

    def test_a_dead_comfyui_holds_nothing(self):
        self.s._prepare_graph(self.job, KONTEXT)
        self.s._comfy_heavy_cached = None  # what a crash/restart sets
        self.s._prepare_graph(self.job, KONTEXT)
        self.assertEqual(len(self.freed), 2)

    def test_a_checkpoint_render_in_between_invalidates_the_belief(self):
        """The real sequence that would OOM: Kontext, then an ordinary
        inpaint/img2img (a CheckpointLoaderSimple graph, which this helper
        returns early for), then Kontext again. If the checkpoint render did
        not clear the belief, the second Kontext render would skip its drop
        and load 10 GB on top of a resident SDXL checkpoint."""
        self.s._prepare_graph(self.job, KONTEXT)
        self.s._prepare_graph(self.job, LIGHT)
        self.assertIsNone(self.s._comfy_heavy_cached)
        self.s._prepare_graph(self.job, KONTEXT)
        self.assertEqual(len(self.freed), 2, "reused a stale belief")

    def test_a_failed_unload_is_not_recorded_as_success(self):
        """If /free did not actually unload, what ComfyUI holds is unknown —
        claiming these weights are resident would make the NEXT identical
        render skip its drop too."""
        self.s.comfy.free_memory = lambda: False
        self.s._prepare_graph(self.job, KONTEXT)
        self.assertIsNone(self.s._comfy_heavy_cached)


class RenderSize(unittest.TestCase):

    def test_a_big_photo_comes_down_to_about_a_megapixel(self):
        out = Services._fit_megapixels(Image.new("RGB", (1486, 1675)), 1.0)
        self.assertLess(out.size[0] * out.size[1], 1.15e6)
        self.assertGreater(out.size[0] * out.size[1], 0.85e6)

    def test_the_aspect_ratio_survives(self):
        """Landing on a multiple of 16 has to move the ratio a little; the
        requirement is that the picture is not visibly stretched."""
        src = Image.new("RGB", (1486, 1675))
        out = Services._fit_megapixels(src, 1.0)
        want = src.size[0] / src.size[1]
        self.assertLess(abs(out.size[0] / out.size[1] - want) / want, 0.02)

    def test_both_sides_land_on_a_multiple_of_sixteen(self):
        out = Services._fit_megapixels(Image.new("RGB", (1486, 1675)), 1.0)
        self.assertEqual((out.size[0] % 16, out.size[1] % 16), (0, 0))

    def test_a_small_photo_is_never_blown_up(self):
        """Upscaling before the sampler invents nothing and costs time."""
        src = Image.new("RGB", (512, 512))
        self.assertEqual(Services._fit_megapixels(src, 1.0).size, (512, 512))


class RestoringResolution(unittest.TestCase):
    """A second render has to be worth it, and must never fail the edit."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))
        self.addCleanup(self.s.stop)
        self.calls = []
        self.s._render_template_step = lambda *a, **k: (
            self.calls.append(a[1]), Image.new("RGB", (4000, 4000)))[1]
        self.s._template_runnable = lambda _t: (True, "")
        self.job = FakeJob()

    def test_an_exact_match_is_returned_untouched(self):
        img = Image.new("RGB", (900, 900))
        self.assertIs(self.s._restore_resolution(self.job, img, (900, 900)),
                      img)

    def test_a_few_pixels_short_is_a_resize_not_a_render(self):
        """ComfyUI's VAE rounds to a multiple of 8, so an edit that was never
        downscaled still comes back a little short. Spending a 4x ESRGAN
        render to recover 6 px is pure waste."""
        img = Image.new("RGB", (1480, 1672))
        out = self.s._restore_resolution(self.job, img, (1486, 1675))
        self.assertEqual(out.size, (1486, 1675))
        self.assertEqual(self.calls, [], "ran an upscale for a 6 px shortfall")

    def test_a_real_shortfall_does_use_the_upscaler(self):
        img = Image.new("RGB", (944, 1056))
        out = self.s._restore_resolution(self.job, img, (1486, 1675))
        self.assertEqual(out.size, (1486, 1675))
        self.assertEqual(self.calls, ["upscale"])

    def test_an_unavailable_upscaler_still_returns_the_edit(self):
        self.s._template_runnable = lambda _t: (False, "not downloaded")
        img = Image.new("RGB", (944, 1056))
        out = self.s._restore_resolution(self.job, img, (1486, 1675))
        self.assertEqual(out.size, (1486, 1675))
        self.assertEqual(self.calls, [])

    def test_a_failing_upscaler_never_loses_the_edit(self):
        def boom(*_a, **_k):
            raise RuntimeError("no upscale model")
        self.s._render_template_step = boom
        img = Image.new("RGB", (944, 1056))
        out = self.s._restore_resolution(self.job, img, (1486, 1675))
        self.assertEqual(out.size, (1486, 1675))


class WhichOperationsMove(unittest.TestCase):

    def test_object_edits_belong_to_kontext(self):
        for op in ("REMOVE_OBJECT", "REPLACE_OBJECT", "CHANGE_ATTRIBUTE",
                   "ADD_OBJECT"):
            self.assertIn(op, quality.KONTEXT_OPERATIONS)

    def test_engines_that_beat_it_keep_their_work(self):
        """Backgrounds have an exact subject matte and lighting has a real
        relighter. A general editor must not take those over."""
        for op in ("REPLACE_BACKGROUND", "CHANGE_LIGHTING", "CHANGE_CAMERA",
                   "COMPOSE", "SWAP_FACE", "OUTPAINT", "UPSCALE", "ANIMATE",
                   "SCENE_3D", "CHANGE_POSE"):
            self.assertNotIn(op, quality.KONTEXT_OPERATIONS)

    def test_every_one_of_them_is_an_inpaint_today(self):
        """The route only ever replaces the masked-inpaint engine, so each
        claimed operation must be one that currently routes to inpaint."""
        for op in quality.KONTEXT_OPERATIONS:
            self.assertEqual(quality.OPERATION_TASK[op], "inpaint")


class RetryWiring(unittest.TestCase):
    """A Kontext retry has to go back through Kontext.

    Seen live: it fell through to the generic template branch, which sends
    the scene-APPENDED prompt at the source's full resolution. Kontext takes
    an instruction, so "remove the hat; scene: a woman in a shop with
    shelves..." read as an order to redraw the scene — the retry returned a
    different room and a different face, at 2.49 MP where the model is also
    off-distribution. The first attempt had been correct."""

    def source(self):
        import inspect
        return inspect.getsource(Services._handle_image_edit)

    def test_the_retry_branch_exists_before_the_generic_fallthrough(self):
        src = self.source()
        branch = src.find('last_step["task"] == "kontext"')
        generic = src.find("candidate = self._render_template_step(\n"
                           "                                job, last_step")
        self.assertGreater(branch, 0, "no Kontext branch in the retry ladder")
        if generic > 0:
            self.assertLess(branch, generic)

    def test_the_retry_sends_the_bare_instruction_not_the_scene_prompt(self):
        src = self.source()
        tail = src[src.find('last_step["task"] == "kontext"'):][:900]
        self.assertIn('self._render_kontext_step(', tail)
        self.assertNotIn("with_scene(", tail.split("elif")[0])


class SceneAnalysisIsSkipped(unittest.TestCase):
    """The vision pass costs minutes (measured: 487 s against a 150 s render)
    and every reader of it belongs to another engine. A plan made only of
    Kontext steps must not pay for it."""

    def source(self):
        import inspect
        return inspect.getsource(Services._handle_image_edit)

    def test_an_all_kontext_plan_skips_the_scene_build(self):
        self.assertIn('all(s["task"] == "kontext" for s in steps)',
                      self.source())

    def test_any_other_engine_still_gets_its_scene(self):
        """The skip must be conditional, not a removal — placement boxes and
        scene-grounded prompts still depend on it."""
        src = self.source()
        self.assertIn("self._scene_graph(job, asset_id, image, real)", src)

    def test_a_drawn_mask_is_bound_to_the_first_inpaint_not_to_index_zero(self):
        """The mask is consumed by whichever step is the first inpaint. An
        index-based exemption would hand that step to Kontext whenever the
        plan opened with something else, discarding the region you drew."""
        src = self.source()
        self.assertIn("first_inpaint", src)
        self.assertNotIn("not (i == 0 and user_mask_b64)", src)

    def test_the_3d_step_does_not_clobber_the_scene_summary(self):
        """`scene` holds the summary string that with_scene() appends to
        prompts; the 3D branch used to overwrite it with its own dict."""
        src = self.source()
        self.assertNotIn("scene = self._render_scene3d_step(", src)


class TemplateOnDisk(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))
        self.addCleanup(self.s.stop)

    def test_the_loader_can_find_it_by_task(self):
        """It shipped as kontext_edit_v1.json declaring task 'img2img', which
        load('img2img') globs as img2img_v*.json — so nothing ever reached
        it. The name and the task have to agree for the route to exist."""
        template = self.s.workflows.load("kontext")
        self.assertEqual(template["task"], "kontext")

    def test_a_card_too_small_for_it_is_refused_with_a_reason(self):
        """flux-kontext-q4 declares vram_gb=7.0 on the ModelInfo, but the fit
        gate only read meta.min_vram_gb — which that entry leaves empty — so
        the gate was a no-op and a 6 GB card would have been routed to a model
        it cannot load, failing mid-render instead of falling back."""
        self.s.hardware.vram_gb, self.s.hardware.ram_gb = 6.0, 32.0
        ok, why = self.s.kontext_ready()
        self.assertFalse(ok)
        self.assertIn("VRAM", why)

    def test_the_machine_it_was_proven_on_is_not_refused_for_size(self):
        """8 GB VRAM / 15.7 GB RAM is where the route was measured working, so
        the size gate must not be what stops it. (In this fixture the weights
        are not downloaded, so readiness is still False — for that reason.)"""
        self.s.hardware.vram_gb, self.s.hardware.ram_gb = 8.0, 15.7
        why = self.s.kontext_ready()[1]
        self.assertNotIn("VRAM", why)
        self.assertNotIn("RAM", why)

    def test_it_takes_an_instruction_and_an_image_and_no_mask(self):
        params = self.s.workflows.load("kontext")["parameters"]
        self.assertIn("prompt", params)
        self.assertIn("image", params)
        self.assertNotIn("mask", params)


class InpaintFallback(unittest.TestCase):
    """The masked route promised when Kontext silently declines an edit.

    Regression: this path called quality.enhance_prompt with a `log` kwarg
    that does not exist and without the required `task` argument — a
    guaranteed TypeError, so every declined Kontext edit died on a raw
    error instead of the promised fallback (found by mypy)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir="", llm_url="http://127.0.0.1:9/v1"))
        self.addCleanup(self.s.stop)

    def test_declined_edit_lands_on_the_masked_route(self):
        image = Image.new("RGB", (64, 64), (20, 30, 40))
        good = Image.new("L", (64, 64), 0)
        good.paste(255, (16, 16, 48, 48))
        self.s.auto_mask = lambda *a, **k: quality.MaskChoice(
            good, "text", "", [])
        self.s._choose_inpaint = lambda job, instr: ("modern", None)
        job = FakeJob()
        out = self.s._inpaint_fallback(job, image,
                                       "change the jacket to red")
        self.assertEqual(out.size, image.size)


if __name__ == "__main__":
    unittest.main()
