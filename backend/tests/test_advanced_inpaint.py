"""Tests: advanced inpainting — variant templates (modern / universal /
hi-res crop&stitch), LLM-selectable inpaint model, crop math, and the chain
driver wiring. Offline: ComfyUI, the scout and the LLM are fakes."""
import base64
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.adapters.base import EditResult, ModelMissingError
from app.adapters.comfyui import (
    ComfyUIInpaintingAdapter,
    WorkflowLibrary,
    WorkflowValidationError,
    build_workflow,
)
from app.config import Settings
from app.core import quality
from app.core.model_scout import ScoutDecision
from app.core.services import Services
from tests.test_workflow_job import DeadLLM, FakeComfy

WORKFLOWS_DIR = Settings().workflows_dir


class FakeClient:
    """Records the graph the adapter submits; returns a blank render."""

    def __init__(self):
        self.graphs = []

    def upload_image(self, image, prefix):
        return f"{prefix}.png"

    def run_graph(self, graph):
        self.graphs.append(graph)
        return Image.new("RGB", (64, 64), (9, 9, 9)), "pid-1"


class FakeRegistry:
    def __init__(self, ready=True):
        self.ready = ready

    def is_ready(self, name):
        return self.ready


def _types(graph):
    return {n["class_type"] for n in graph.values()}


def _adapter(ready=True):
    a = ComfyUIInpaintingAdapter("http://x", WorkflowLibrary(WORKFLOWS_DIR),
                                 FakeRegistry(ready))
    a.client = FakeClient()
    return a


def _mask(size=(64, 64), box=(16, 16, 48, 48)):
    m = Image.new("L", size, 0)
    m.paste(255, box)
    return m


class InpaintVariantAdapterTests(unittest.TestCase):
    def test_modern_variant_uses_soft_inpaint_and_fills_model(self):
        a = _adapter()
        res = a.inpaint(Image.new("RGB", (64, 64)), _mask(), "a red jacket",
                        negative="cartoon", checkpoint="foo-inpaint.safetensors",
                        variant="modern")
        graph = a.client.graphs[0]
        self.assertIn("InpaintModelConditioning", _types(graph))
        self.assertIn("DifferentialDiffusion", _types(graph))
        loaders = [n for n in graph.values()
                   if n["class_type"] == "CheckpointLoaderSimple"]
        self.assertEqual(loaders[0]["inputs"]["ckpt_name"],
                         "foo-inpaint.safetensors")
        negs = [n["inputs"]["text"] for n in graph.values()
                if n["class_type"] == "CLIPTextEncode"]
        self.assertIn("cartoon", negs)
        self.assertEqual(res.meta["variant"], "modern")
        self.assertTrue(res.meta["template"].startswith("inpaint_v"))

    def test_default_inpaint_template_is_now_v3(self):
        t = WorkflowLibrary(WORKFLOWS_DIR).load("inpaint")
        self.assertGreaterEqual(t["version"], 3)
        self.assertIn("InpaintModelConditioning",
                      {n["class_type"] for n in t["graph"].values()})

    def test_universal_variant_lets_any_checkpoint_inpaint(self):
        # Registry has NOTHING staged — an explicit checkpoint (already
        # inside ComfyUI) must still work: that's the whole point.
        a = _adapter(ready=False)
        res = a.inpaint(Image.new("RGB", (64, 64)), _mask(), "a red jacket",
                        checkpoint="photoreal_v5.safetensors",
                        variant="universal")
        graph = a.client.graphs[0]
        self.assertIn("SetLatentNoiseMask", _types(graph))
        self.assertNotIn("InpaintModelConditioning", _types(graph))
        sampler = next(n for n in graph.values()
                       if n["class_type"] == "KSampler")
        self.assertLess(sampler["inputs"]["denoise"], 1.0)
        self.assertEqual(res.meta["variant"], "universal")

    def test_modern_without_checkpoint_requires_staged_model(self):
        a = _adapter(ready=False)
        with self.assertRaises(ModelMissingError):
            a.inpaint(Image.new("RGB", (64, 64)), _mask(), "x")

    def test_hires_variant_crops_and_stitches(self):
        a = _adapter()
        img = Image.new("RGB", (256, 256))
        a.inpaint(img, _mask((256, 256), (100, 100, 140, 140)), "a button",
                  variant="hires")
        graph = a.client.graphs[0]
        for t in ("ImageCrop", "ImageCompositeMasked", "MaskToImage",
                  "ImageToMask", "InpaintModelConditioning"):
            self.assertIn(t, _types(graph))
        crops = [n for n in graph.values() if n["class_type"] == "ImageCrop"]
        comp = next(n for n in graph.values()
                    if n["class_type"] == "ImageCompositeMasked")
        # Image-crop and mask-crop use the SAME rectangle; the stitch node
        # pastes back at the same origin.
        self.assertEqual(len({(c["inputs"]["x"], c["inputs"]["y"],
                               c["inputs"]["width"], c["inputs"]["height"])
                              for c in crops}), 1)
        self.assertEqual(comp["inputs"]["x"], crops[0]["inputs"]["x"])

    def test_crop_params_snap_cover_and_clamp(self):
        img = Image.new("RGB", (640, 480))
        mask = _mask((640, 480), (300, 200, 360, 260))
        p = ComfyUIInpaintingAdapter._crop_params(img, mask)
        for k in ("crop_x", "crop_y", "crop_w", "crop_h", "up_w", "up_h"):
            self.assertEqual(p[k] % 8, 0, k)
        # Rectangle covers the mask bbox and stays inside the image.
        self.assertLessEqual(p["crop_x"], 300)
        self.assertLessEqual(p["crop_y"], 200)
        self.assertGreaterEqual(p["crop_x"] + p["crop_w"], 360)
        self.assertGreaterEqual(p["crop_y"] + p["crop_h"], 260)
        self.assertLessEqual(p["crop_x"] + p["crop_w"], 640)
        self.assertLessEqual(p["crop_y"] + p["crop_h"], 480)
        # Upscale is at most 2x and never shrinks.
        self.assertGreaterEqual(p["up_w"], p["crop_w"])
        self.assertLessEqual(p["up_w"], p["crop_w"] * 2)

    def test_crop_params_with_full_mask_stay_in_bounds(self):
        img = Image.new("RGB", (128, 96))
        p = ComfyUIInpaintingAdapter._crop_params(
            img, Image.new("L", (128, 96), 255))
        self.assertLessEqual(p["crop_x"] + p["crop_w"], 128)
        self.assertLessEqual(p["crop_y"] + p["crop_h"], 96)

    def test_load_named_picks_latest_and_rejects_unknown(self):
        lib = WorkflowLibrary(WORKFLOWS_DIR)
        t = lib.load_named("inpaint_universal")
        self.assertEqual(t["template"], "inpaint_universal")
        self.assertEqual(t.get("task"), "inpaint")
        with self.assertRaises(WorkflowValidationError):
            lib.load_named("no_such_template")

    def test_multi_slot_parameters_fan_out(self):
        t = WorkflowLibrary(WORKFLOWS_DIR).load_named("inpaint_hires")
        graph = build_workflow(t, {"crop_x": 88})
        crops = [n for n in graph.values() if n["class_type"] == "ImageCrop"]
        self.assertTrue(all(c["inputs"]["x"] == 88 for c in crops))

    def test_mask_fraction(self):
        self.assertAlmostEqual(
            quality.mask_fraction(_mask((64, 64), (0, 0, 64, 32))), 0.5)
        self.assertEqual(quality.mask_fraction(Image.new("L", (8, 8), 0)), 0.0)


class FakeScout:
    """Deterministic model choice; records what it was asked."""

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.asked = []

    def choose(self, prompt, task, installed, allow_download,
               progress=None, force_search=False, log=None):
        self.asked.append((prompt, task, list(installed)))
        return ScoutDecision(self.checkpoint, None, "scripted choice")


class VariantInpaint:
    """Real-ish adapter that records the variant/checkpoint it was given."""
    name = "fake-variant"
    is_mock = False
    supports_variants = True

    def __init__(self):
        self.calls = []

    def inpaint(self, image, mask, prompt, *, negative="", checkpoint=None,
                variant="modern", denoise=None):
        self.calls.append({"checkpoint": checkpoint, "variant": variant,
                           "negative": negative, "denoise": denoise})
        return EditResult(image=Image.new("RGB", image.size),
                          adapter=self.name, is_mock=False,
                          meta={"template": "inpaint_v3", "variant": variant,
                                "checkpoint": checkpoint})


class DriverModelChoiceTests(unittest.TestCase):
    """The chain driver lets the LLM pick the inpaint model and routes the
    variant from that choice (+ hi-res upgrade for small regions)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="",
            first_run_setup=False, comfyui_dir=""))
        # Offline: no auto-staging of better models into the live queue.
        self.s.settings.auto_install = False
        self.s.llm = DeadLLM()          # plan falls back to single inpaint
        self.s.scout = FakeScout("juggernaut-inpaint.safetensors")
        self.s.comfy = FakeComfy(
            checkpoints=["sd_xl_base_1.0.safetensors",
                         "juggernaut-inpaint.safetensors"])
        self.adapter = VariantInpaint()
        self.s.inpainting = self.adapter
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), (5, 5, 5)).save(buf, format="PNG")
        self.asset = self.s.store.save_upload("p.png", buf.getvalue())
        self.s.start()

    def tearDown(self):
        self.s.stop()
        self.tmp.cleanup()

    @staticmethod
    def _mask_b64(box):
        m = Image.new("L", (32, 32), 0)
        m.paste(255, box)
        buf = io.BytesIO()
        m.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _edit(self, box):
        job = self.s.queue.enqueue("image_edit", {
            "asset_id": self.asset.id, "prompt": "a red jacket",
            "mask_b64": self._mask_b64(box)})
        done = self.s.queue.wait_for(job.id, timeout=30)
        self.assertEqual(done.state.value, "completed", done.error)
        return done

    def test_inpaint_checkpoint_chosen_by_scout_reaches_adapter(self):
        done = self._edit((0, 0, 32, 24))  # big region: no hi-res upgrade
        call = self.adapter.calls[0]
        self.assertEqual(call["checkpoint"], "juggernaut-inpaint.safetensors")
        self.assertEqual(call["variant"], "modern")
        # Inpaint-capable checkpoints are offered first to the scout.
        self.assertEqual(self.s.scout.asked[0][2][0],
                         "juggernaut-inpaint.safetensors")
        self.assertEqual(done.result["plan"][0]["model"],
                         "juggernaut-inpaint.safetensors")
        # Model choice must be visible in Behind the Scenes, which filters
        # "[llm]" lines — so it must NOT carry that prefix.
        self.assertTrue(any(e["msg"].startswith("Inpaint model:")
                            for e in done.logs))

    def test_non_inpaint_choice_routes_to_universal(self):
        self.s.scout = FakeScout("photoreal_v5.safetensors")
        self._edit((0, 0, 32, 24))
        self.assertEqual(self.adapter.calls[0]["variant"], "universal")

    def test_small_region_upgrades_to_hires(self):
        self._edit((14, 14, 18, 18))  # ~1.6% of the image
        self.assertEqual(self.adapter.calls[0]["variant"], "hires")

    def test_model_knowledge_reaches_the_scout_prompt(self):
        self.s.model_intel._save({"juggernaut-inpaint.safetensors": {
            "best_at": "photoreal portraits", "avoid": "anime",
            "prompt_style": "short sentences", "quality": 9,
            "reason": "top model"}})
        done = self._edit((0, 0, 32, 24))
        asked = self.s.scout.asked[0][0]
        self.assertIn("Model knowledge", asked)
        self.assertIn("photoreal portraits", asked)
        self.assertTrue(any("consulting the model knowledge" in e["msg"]
                            for e in done.logs))

    def test_whole_frame_attribute_change_keeps_composition(self):
        """A whole-frame restyle must run at moderate denoise: replacement
        denoise on a full-frame mask generates a NEW picture that merely
        matches the words — the user's photo is gone."""
        full = Image.new("L", (32, 32), 255)
        self.s._text_mask = lambda *a, **k: (full, {"peak": 0.9})
        job = self.s.queue.enqueue("image_edit", {
            "asset_id": self.asset.id,
            "prompt": "make the sky a warm sunset"})
        done = self.s.queue.wait_for(job.id, timeout=30)
        self.assertEqual(done.state.value, "completed", done.error)
        call = self.adapter.calls[0]
        self.assertEqual(call["variant"], "universal")
        self.assertEqual(call["denoise"], 0.6)
        self.assertTrue(any("whole-frame restyle" in e["msg"]
                            for e in done.logs), done.logs)

    def test_whole_frame_replace_keeps_full_denoise(self):
        """An explicit REPLACE of a frame-filling thing asked for new
        content — structure preservation would fight the request."""
        full = Image.new("L", (32, 32), 255)
        self.s._text_mask = lambda *a, **k: (full, {"peak": 0.9})
        job = self.s.queue.enqueue("image_edit", {
            "asset_id": self.asset.id,
            "prompt": "replace the sky with a starry night sky"})
        done = self.s.queue.wait_for(job.id, timeout=30)
        self.assertEqual(done.state.value, "completed", done.error)
        self.assertIsNone(self.adapter.calls[0]["denoise"])

    def test_invented_outpaint_step_is_dropped_end_to_end(self):
        class PlanLLM:
            source = "local"

            def complete(self, system, prompt, max_tokens=4096):
                import json as _json

                from app.core.llm import LLMReply
                return LLMReply(_json.dumps({"steps": [
                    {"task": "inpaint", "instruction": "recolor the chair"},
                    {"task": "outpaint",
                     "instruction": "extend the background"},
                ]}), "fake", "local")

        self.s.llm = PlanLLM()
        done = self._edit((0, 0, 32, 24))  # prompt has no canvas words
        self.assertEqual([p["task"] for p in done.result["plan"]],
                         ["inpaint"])
        self.assertTrue(any("dropped invented outpaint" in e["msg"]
                            for e in done.logs))

    def test_add_routed_to_img2img_is_coerced_back_to_inpaint(self):
        """The router may mislabel 'add a dog' as img2img (whole-image) —
        the driver coerces add-edits back to regional inpaint (live bug)."""
        import json as _json

        from app.core.llm import LLMReply

        class PlanLLM:
            source = "local"

            def complete(self, system, prompt, max_tokens=4096):
                return LLMReply(_json.dumps({"steps": [
                    {"task": "img2img",
                     "instruction": "add a dog to the background"},
                ]}), "fake", "local")

        self.s.llm = PlanLLM()
        job = self.s.queue.enqueue("image_edit", {
            "asset_id": self.asset.id,
            "prompt": "put a dog in the background"})
        done = self.s.queue.wait_for(job.id, timeout=30)
        self.assertEqual(done.state.value, "completed", done.error)
        # The compiler coerces the add-edit to a regional inpaint, and it
        # gets a placement region (not whole-image segmentation).
        self.assertEqual(done.result["plan"][0]["task"], "inpaint")
        self.assertTrue(any("placement region" in e["msg"]
                            for e in done.logs))

    def test_add_instruction_uses_placement_mask_not_segmentation(self):
        """'put a dog in the background' must NOT segment the background —
        new content gets a placement region (live bug)."""
        def boom(image, instruction):
            raise AssertionError("segmentation must not run for add-edits")

        self.s.segmentation.propose_mask = boom
        job = self.s.queue.enqueue("image_edit", {
            "asset_id": self.asset.id,
            "prompt": "put a dog in the background"})
        done = self.s.queue.wait_for(job.id, timeout=30)
        self.assertEqual(done.state.value, "completed", done.error)
        self.assertTrue(any("placement region" in e["msg"]
                            for e in done.logs))

    def test_custom_plan_step_builds_a_bespoke_workflow(self):
        import json as _json

        from app.core.llm import LLMReply
        from tests.test_workflow_job import GRAPH, ScriptedLLM

        class PlanLLM:
            source = "local"

            def complete(self, system, prompt, max_tokens=4096):
                return LLMReply(_json.dumps({"steps": [
                    {"task": "custom",
                     "instruction": "kaleidoscope-mirror the image"},
                ]}), "fake", "local")

        self.s.llm = PlanLLM()
        self.s.workflow_ai.llm = ScriptedLLM([_json.dumps(GRAPH)])
        self.s.comfy.upload_image = lambda img, prefix: f"{prefix}.png"
        # No user mask: a drawn mask would force step 1 to inpaint.
        job = self.s.queue.enqueue("image_edit", {
            "asset_id": self.asset.id,
            "prompt": "kaleidoscope-mirror the image"})
        done = self.s.queue.wait_for(job.id, timeout=30)
        self.assertEqual(done.state.value, "completed", done.error)
        self.assertEqual(done.result["plan"][0]["task"], "custom")
        self.assertEqual(done.result["plan"][0]["workflow"],
                         "LLM-designed custom workflow")
        self.assertTrue(any("building a custom" in e["msg"]
                            for e in done.logs))

    def test_targeted_segmentation_uses_the_named_target(self):
        """An existing-object edit segments the TARGET, not the whole raw
        instruction.

        The scene-graph location ("car in the center-right of the image") is
        deliberately no longer threaded through. It never did anything:
        SAM's spatial prior scores "car", "car in the center-right of the
        image" and "replace the car with a wolf" identically at 1.0000,
        because it only recognises sky/backdrop/floor and otherwise prefers a
        centred blob. And the chooser now leads with engines that match on
        APPEARANCE, where a position phrase is noise rather than help."""
        captured = {}

        def rec(image, instruction):
            captured["instruction"] = instruction
            # A PLAUSIBLE region. This used to return an empty mask, which
            # the pipeline accepted and rendered — the defect that the
            # deterministic mask verdict now catches. The point of this test
            # is which instruction arrives, not empty-mask handling.
            from PIL import ImageDraw
            mask = Image.new("L", image.size, 0)
            ImageDraw.Draw(mask).rectangle(
                [image.width // 4, image.height // 4,
                 image.width * 3 // 4, image.height * 3 // 4], fill=255)
            return mask

        self.s.segmentation.propose_mask = rec
        # Seed a scene graph for the asset (critic is off in this fixture).
        self.s._scene_cache[self.asset.id] = {
            "scene": "a street", "objects": [
                {"name": "car", "location": "center-right", "size": "large",
                 "cell": 6}]}

        class PlanLLM:
            source = "local"

            def complete(self, system, prompt, max_tokens=4096):
                import json as _json

                from app.core.llm import LLMReply
                return LLMReply(_json.dumps({"steps": [
                    {"operation": "REPLACE_OBJECT", "target": "car",
                     "instruction": "replace the car with a wolf"}]}),
                    "fake", "local")

        self.s.llm = PlanLLM()
        job = self.s.queue.enqueue("image_edit", {
            "asset_id": self.asset.id,
            "prompt": "replace the car with a wolf"})
        done = self.s.queue.wait_for(job.id, timeout=30)
        self.assertEqual(done.state.value, "completed", done.error)
        self.assertIn("car", captured["instruction"])

    def test_animate_operation_routes_to_video(self):
        from app.core.storage import Asset

        calls = {}

        def fake_render(job, asset_id, prompt, **kw):
            calls["prompt"] = prompt
            a = Asset("vid123", "video", "v.webp", "/tmp/v.webp", "", {})
            return a, "pid", 640, 640, 49

        self.s._render_video_asset = fake_render

        class PlanLLM:
            source = "local"

            def complete(self, system, prompt, max_tokens=4096):
                import json as _json

                from app.core.llm import LLMReply
                return LLMReply(_json.dumps({"steps": [
                    {"operation": "ANIMATE", "target": "person",
                     "instruction": "make the person wave"}]}),
                    "fake", "local")

        self.s.llm = PlanLLM()
        job = self.s.queue.enqueue("image_edit", {
            "asset_id": self.asset.id, "prompt": "animate the person waving"})
        done = self.s.queue.wait_for(job.id, timeout=30)
        self.assertEqual(done.state.value, "completed", done.error)
        self.assertEqual(done.result["kind"], "video")
        self.assertEqual(done.result["asset_id"], "vid123")
        self.assertIn("motion", calls["prompt"].lower())

    def test_plain_adapters_keep_the_simple_call(self):
        class PlainInpaint:
            name = "plain"
            is_mock = False
            calls = 0

            def inpaint(self, image, mask, prompt):  # no kwargs at all
                PlainInpaint.calls += 1
                return EditResult(image=Image.new("RGB", image.size),
                                  adapter="plain", is_mock=False, meta={})

        self.s.inpainting = PlainInpaint()
        self._edit((0, 0, 32, 24))
        self.assertEqual(PlainInpaint.calls, 1)


class InpaintModelPolicyTests(unittest.TestCase):
    """Best model first: ranking, and background staging of better models."""

    def _services(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="",
            first_run_setup=False, comfyui_dir=""))
        self.addCleanup(s.stop)
        return s

    def test_rank_prefers_modern_photoreal_inpaint_models(self):
        ranked = sorted([
            "sd_xl_base_1.0.safetensors",
            "sd-v1-5-inpainting.safetensors",
            "juggernautXL_versionXInpaint.safetensors",
            "epicrealism_v10-inpainting.safetensors",
        ], key=Services._inpaint_rank)
        self.assertEqual(ranked[0], "juggernautXL_versionXInpaint.safetensors")
        self.assertEqual(ranked[1], "epicrealism_v10-inpainting.safetensors")
        self.assertEqual(ranked[-1], "sd_xl_base_1.0.safetensors")

    def test_stage_better_inpaint_queues_registered_models_once(self):
        s = self._services()  # queue worker NOT started — jobs stay pending

        class LogJob:
            logs = []

            def log(self, level, msg):
                LogJob.logs.append(msg)

        s._stage_better_inpaint(LogJob(), ["sd-v1-5-inpainting.safetensors"])
        staged = [j for j in s.queue.list() if j.type == "model_download"]
        names = {j.payload["model"] for j in staged}
        self.assertEqual(names, {"epicrealism-inpaint",
                                 "juggernaut-xl-inpaint"})
        self.assertTrue(any("staging better inpaint" in m
                            for m in LogJob.logs))
        s._stage_better_inpaint(LogJob(), ["sd-v1-5-inpainting.safetensors"])
        staged2 = [j for j in s.queue.list() if j.type == "model_download"]
        self.assertEqual(len(staged2), len(staged))  # no duplicates

    def test_stage_skips_when_modern_model_installed(self):
        s = self._services()

        class BoomJob:
            def log(self, level, msg):
                raise AssertionError("nothing should be staged")

        s._stage_better_inpaint(
            BoomJob(), ["juggernautXL_versionXInpaint.safetensors"])
        self.assertEqual([j for j in s.queue.list()
                          if j.type == "model_download"], [])

    def test_universal_v2_uses_differential_diffusion(self):
        t = WorkflowLibrary(WORKFLOWS_DIR).load_named("inpaint_universal")
        self.assertGreaterEqual(t["version"], 2)
        types = {n["class_type"] for n in t["graph"].values()}
        self.assertIn("DifferentialDiffusion", types)
        self.assertIn("SetLatentNoiseMask", types)

    def test_gallery_skips_assets_whose_file_vanished(self):
        s = self._services()
        a = s.store.save_upload("gone.png", b"\x89PNG\r\n\x1a\n123")
        Path(a.path).unlink()
        b = s.store.save_upload("here.png", b"\x89PNG\r\n\x1a\n456")
        ids = [g["asset"]["id"] for g in s.store.gallery()]
        self.assertIn(b.id, ids)
        self.assertNotIn(a.id, ids)


if __name__ == "__main__":
    unittest.main()
