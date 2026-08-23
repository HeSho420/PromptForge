"""Combining two or more images.

The pipeline's promise here is specific: the subject of the SECOND photo ends
up in the FIRST one — the real person, not a lookalike painted from a
description. These tests pin the routing, the placement maths and the honest
degradation when the matting engine is missing.
"""
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.adapters.comfyui import ALLOWED_NODE_TYPES, ALLOWED_TASKS, WorkflowLibrary
from app.config import Settings
from app.core import quality
from app.core.services import Services

WORKFLOWS = Path(__file__).parent.parent / "app" / "workflows"


class _StubJob:
    def log(self, *_a, **_k):
        pass


class OneJson:
    source = "local"

    def __init__(self, payload):
        self.payload = payload

    def complete(self, system, prompt, max_tokens=4096):
        from app.core.llm import LLMReply
        return LLMReply(json.dumps(self.payload), "fake", "local")


class PlacementTests(unittest.TestCase):
    """Where the second image's subject lands."""

    TARGET = (768, 512)

    def _mask(self, box):
        m = Image.new("L", self.TARGET, 0)
        m.paste(255, box)
        return m

    def test_a_drawn_region_is_obeyed(self):
        box = quality.placement_box(self._mask((400, 180, 632, 412)),
                                    self.TARGET, (512, 512))
        self.assertEqual(box, {"x": 400, "y": 180, "w": 232, "h": 232})

    def test_the_subject_keeps_its_own_proportions(self):
        """A person stretched to fill a square region reads as wrong long
        before anyone works out why."""
        box = quality.placement_box(self._mask((400, 180, 632, 412)),
                                    self.TARGET, (300, 900))
        self.assertAlmostEqual(box["w"] / box["h"], 300 / 900, places=1)
        self.assertLessEqual(box["w"], 232)
        self.assertLessEqual(box["h"], 232)

    def test_a_short_subject_stands_on_the_region_not_in_its_middle(self):
        box = quality.placement_box(self._mask((100, 100, 500, 400)),
                                    self.TARGET, (400, 100))
        self.assertEqual(box["y"] + box["h"], 400)  # bottom-aligned

    def test_without_a_region_it_stands_centre_front(self):
        box = quality.placement_box(None, self.TARGET, (512, 512))
        self.assertGreater(box["x"], 0)
        self.assertLess(box["x"] + box["w"], self.TARGET[0])
        self.assertEqual(box["y"] + box["h"], self.TARGET[1])

    def test_it_never_falls_outside_the_photo(self):
        for subject in ((4000, 100), (100, 4000), (1, 1), (9999, 9999)):
            for mask in (None, self._mask((700, 460, 768, 512))):
                b = quality.placement_box(mask, self.TARGET, subject)
                self.assertGreaterEqual(b["x"], 0, subject)
                self.assertGreaterEqual(b["y"], 0, subject)
                self.assertLessEqual(b["x"] + b["w"], self.TARGET[0], subject)
                self.assertLessEqual(b["y"] + b["h"], self.TARGET[1], subject)

    def test_a_degenerate_mask_is_ignored_rather_than_obeyed(self):
        """A stray dot of brush is not a placement region."""
        tiny = quality.placement_box(self._mask((10, 10, 14, 14)),
                                     self.TARGET, (512, 512))
        none = quality.placement_box(None, self.TARGET, (512, 512))
        self.assertEqual(tiny, none)


class RoutingTests(unittest.TestCase):
    def test_compose_is_a_real_task_with_an_operation(self):
        self.assertEqual(quality.OPERATION_TASK["COMPOSE"], "compose")
        self.assertIn("compose", quality.EDIT_TASKS)
        self.assertIn("compose", ALLOWED_TASKS)

    def test_compose_runs_before_every_other_edit(self):
        """Later steps must act on the combined picture, not on half of it."""
        ordered = quality.order_steps([
            {"task": "outpaint"}, {"task": "img2img"},
            {"task": "compose"}, {"task": "upscale"}])
        self.assertEqual([s["task"] for s in ordered],
                         ["compose", "img2img", "outpaint", "upscale"])

    def test_a_plan_may_name_compose_directly(self):
        llm = OneJson({"steps": [{"operation": "COMPOSE", "target": "person",
                                  "instruction": "put her on the bench"}]})
        steps = quality.plan_edit(llm, "put her on the bench", has_mask=False)
        self.assertEqual([s["task"] for s in steps], ["compose"])


class TemplateTests(unittest.TestCase):
    def test_the_matting_node_is_allowed_and_documented(self):
        from app.core.workflow_ai import NODE_GUIDE, NODE_OUTPUTS
        self.assertIn("BiRefNetRMBG", ALLOWED_NODE_TYPES)
        self.assertIn("BiRefNetRMBG", NODE_GUIDE)
        self.assertEqual(NODE_OUTPUTS["BiRefNetRMBG"], ["IMAGE", "MASK", "IMAGE"])

    def test_the_template_mattes_the_subject_and_shrinks_before_feathering(self):
        """Measured: SAM returns 8.7% of a person (their shirt); BiRefNet
        returns the whole figure. And the matte must be SHRUNK before it is
        feathered, or the subject's old backdrop survives as a halo."""
        t = WorkflowLibrary(WORKFLOWS).load("compose")
        graph = t["graph"]
        matte = next(nid for nid, n in graph.items()
                     if n["class_type"] == "BiRefNetRMBG")
        grow = next(n for n in graph.values() if n["class_type"] == "GrowMask")
        feather = next(n for n in graph.values()
                       if n["class_type"] == "FeatherMask")
        composite = next(n for n in graph.values()
                         if n["class_type"] == "ImageCompositeMasked")
        self.assertEqual(grow["inputs"]["mask"], [matte, 1])   # output 1 = MASK
        self.assertLess(grow["inputs"]["expand"], 0)           # shrink, not grow
        self.assertEqual(feather["inputs"]["mask"][0],
                         next(nid for nid, n in graph.items()
                              if n["class_type"] == "GrowMask"))
        self.assertEqual(composite["inputs"]["mask"][0],
                         next(nid for nid, n in graph.items()
                              if n["class_type"] == "FeatherMask"))

    def test_the_matte_is_taken_of_the_SCALED_subject(self):
        """Masks have no scaler in ComfyUI. Matting after the scale keeps the
        matte and the pixels the same size without a round trip."""
        t = WorkflowLibrary(WORKFLOWS).load("compose")
        graph = t["graph"]
        scale = next(nid for nid, n in graph.items()
                     if n["class_type"] == "ImageScale")
        matte = next(n for n in graph.values()
                     if n["class_type"] == "BiRefNetRMBG")
        self.assertEqual(matte["inputs"]["image"], [scale, 0])

    def test_the_harmonisation_pass_blends_without_redrawing(self):
        t = WorkflowLibrary(WORKFLOWS).load("compose")
        ks = next(n for n in t["graph"].values()
                  if n["class_type"] == "KSampler")
        self.assertLessEqual(ks["inputs"]["denoise"], 0.35)  # above: face melts
        self.assertGreaterEqual(ks["inputs"]["denoise"], 0.12)  # below: pasted

    def test_it_builds_with_the_parameters_the_pipeline_supplies(self):
        from app.adapters.comfyui import build_workflow, validate_workflow
        t = WorkflowLibrary(WORKFLOWS).load("compose")
        graph = build_workflow(t, {
            "checkpoint": "x.safetensors", "image": "bg.png",
            "subject": "person.png", "sub_w": 232, "sub_h": 232,
            "pos_x": 400, "pos_y": 180, "prompt": "p", "negative": "n",
            "seed": 1, "denoise": 0.22, "steps": 24, "cfg": 6.0})
        validate_workflow(graph)
        composite = next(n for n in graph.values()
                         if n["class_type"] == "ImageCompositeMasked")
        self.assertEqual((composite["inputs"]["x"], composite["inputs"]["y"]),
                         (400, 180))


class ServiceWiringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))
        self.addCleanup(self.s.stop)

    def test_the_harmoniser_avoids_inpainting_checkpoints(self):
        """An inpainting checkpoint expects a mask it is never given here;
        among the plain bases the SDXL one wins — harmonisation runs at
        the destination's native size (~1.8 MP), where the SD1.5 base is
        far off-distribution (the measured root cause of the soft
        environments, and the compose inspector's recurring 'seam
        between the two women')."""
        self.s.comfy.installed_checkpoints = lambda: [
            "juggernautXL_inpaint.safetensors",
            "sd-v1-5-inpainting.safetensors",
            "v1-5-pruned-emaonly.safetensors",
            "sd_xl_base_1.0.safetensors"]
        self.assertEqual(self.s._best_compose_checkpoint(),
                         "sd_xl_base_1.0.safetensors")

    def test_the_harmoniser_is_chosen_deliberately_not_alphabetically(self):
        """Seen live: the pass picked `nsfw_v10` purely because it sorted
        first among the non-inpainting checkpoints. With no plain XL base
        installed, the registry's SD1.5 base still beats an arbitrary
        community name."""
        self.s.comfy.installed_checkpoints = lambda: [
            "nsfw_v10.safetensors",
            "v1-5-pruned-emaonly.safetensors"]
        self.assertEqual(self.s._best_compose_checkpoint(),
                         "v1-5-pruned-emaonly.safetensors")

    def test_a_surprising_name_is_not_picked_by_accident(self):
        """With no neutral base installed it still must not reach for one of
        these first — it is a general-purpose blend, and the name gets quoted
        back in the job log."""
        self.s.comfy.installed_checkpoints = lambda: [
            "nsfw_v10.safetensors", "dreamshaper_8.safetensors"]
        self.assertEqual(self.s._best_compose_checkpoint(),
                         "dreamshaper_8.safetensors")

    def test_it_copes_when_nothing_suitable_is_installed(self):
        self.s.comfy.installed_checkpoints = lambda: [
            "sd-v1-5-inpainting.safetensors"]
        self.assertIsNone(self.s._best_compose_checkpoint())
        self.s.comfy.installed_checkpoints = lambda: []
        self.assertIsNone(self.s._best_compose_checkpoint())


class FaceSwapTests(unittest.TestCase):
    """Swapping a FACE is not the same request as moving a whole person.

    Seen live: "face swap" planned as COMPOSE(face) and would have matted the
    entire subject with BiRefNet — pasting a complete stranger into the photo
    instead of replacing a face."""

    def test_the_request_is_recognised(self):
        for text in ("face swap", "swap the face", "replace her face",
                     "put her face on him", "swap heads",
                     "change the face to hers", "faceswap this"):
            self.assertTrue(quality.face_intent(text), text)

    def test_ordinary_sentences_are_not_a_face_swap(self):
        for text in ("put the person in the field", "a face in the crowd",
                     "faceted glass", "she has a kind face",
                     "make the facade brighter"):
            self.assertFalse(quality.face_intent(text), text)

    def test_it_has_its_own_operation_and_task(self):
        self.assertEqual(quality.OPERATION_TASK["SWAP_FACE"], "faceswap")
        self.assertIn("faceswap", quality.EDIT_TASKS)
        # Deliberately NOT in ALLOWED_TASKS: that gate guards template files,
        # and a face swap has no template of its own — it composites in
        # Python and then borrows img2img for the harmonisation pass.
        self.assertNotIn("faceswap", ALLOWED_TASKS)

    def test_a_face_swap_runs_before_other_edits(self):
        ordered = quality.order_steps([
            {"task": "img2img"}, {"task": "faceswap"}, {"task": "outpaint"}])
        self.assertEqual(ordered[0]["task"], "faceswap")

    def test_the_face_matte_excludes_hair_and_ears(self):
        """Including them swaps a whole head, and the hairline never matches
        the target's lighting — it reads as a pasted cut-out."""
        parts = set(Services._FACE_PARTS)
        self.assertIn("Skin", parts)
        self.assertIn("Left-eye", parts)
        self.assertIn("Nose", parts)
        for excluded in ("Hair", "Left-ear", "Right-ear", "Neck", "Hat"):
            self.assertNotIn(excluded, parts)

    def test_the_face_is_located_by_features_not_by_skin(self):
        """"Skin" is the parser's skin class over the WHOLE photo — on a beach
        shot it labels arms and torso. A nose does not appear on a shoulder,
        so the features are what say where the face is."""
        features = set(Services._FACE_FEATURES)
        self.assertNotIn("Skin", features)
        self.assertIn("Nose", features)
        self.assertIn("Left-eye", features)
        self.assertIn("Skin", set(Services._FACE_SURFACE))

    def _services_with_mask(self, box, size=(600, 900)):
        """A Services whose face segmenter returns a mask with `box` set."""
        import tempfile
        from pathlib import Path as P

        from app.config import Settings
        from app.core.services import Services as S

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        s = S(Settings(data_dir=P(tmp.name), inpaint_backend="mock",
                       segment_backend="mock", critic_model="",
                       first_run_setup=False, comfyui_dir=""))
        self.addCleanup(s.stop)
        mask = Image.new("L", size, 0)
        if box:
            mask.paste(255, box)
        buf = io.BytesIO()
        mask.convert("RGB").save(buf, format="PNG")
        payload = buf.getvalue()

        s._pack_active = lambda slug: True
        s.comfy.upload_image = lambda image, prefix: "x.png"
        s.comfy.submit = lambda graph: "pid"
        # Two passes: the features locate the face, the skin pass fills it.
        s.comfy.wait_for_output_all = lambda pid: [
            (payload, "pf_facefeatures_0001.png"),
            (payload, "pf_facesurface_0001.png")]
        # These cases are about the whole-frame pass. The head-crop retry has
        # its own test; left live it would call out to a real segmenter.
        s._head_crop = lambda im: None
        return s, Image.new("RGB", size, (10, 10, 10))

    def test_a_torso_shaped_region_is_refused_not_used(self):
        """The segmenter's "Skin" class is ALL skin, not facial skin.

        Measured on a real photo: it returned a 148x445 box spanning 58% of
        the frame height — face through torso — and the swap landed on the
        subject's chest. A confident wrong answer is worse than an honest
        refusal, so the region has to be face-shaped."""
        s, photo = self._services_with_mask((257, 124, 405, 569))
        self.assertIsNone(s._face_region(photo, _StubJob()))

    def test_a_face_shaped_region_is_accepted(self):
        s, photo = self._services_with_mask((150, 100, 405, 365))
        found = s._face_region(photo, _StubJob())
        self.assertIsNotNone(found)
        box = found[1]
        # The features are the eyes-nose-mouth cluster, so the FACE is bigger:
        # grown up for the forehead, out for the cheeks, a little down.
        self.assertLessEqual(box[0], 150)
        self.assertLessEqual(box[1], 100)
        self.assertGreaterEqual(box[2], 405)
        self.assertGreaterEqual(box[3], 365)

    def test_a_tall_thin_region_is_refused(self):
        """A vertical strip of skin is not a face, whatever the parser says."""
        s, photo = self._services_with_mask((280, 100, 340, 700))
        self.assertIsNone(s._face_region(photo, _StubJob()))

    def test_a_small_face_is_retried_on_a_head_crop(self):
        """The parser reads its input at a fixed internal resolution, so on a
        full-length photograph the face arrives about 25 px across and it
        finds nothing — measured, it refused all four real photographs tried.
        Cropping to the head first took that to two of four, and both
        remaining refusals were photos where the face really is hidden (a
        phone held over it, a hand over the mouth)."""
        s, photo = self._services_with_mask(None, size=(600, 900))
        calls = []

        def fake_region(image, job):
            calls.append(image.size)
            if image.size == photo.size:
                return None                      # too small in the full frame
            mask = Image.new("L", image.size, 0)
            mask.paste(255, (10, 10, image.width - 10, image.height - 10))
            return mask, (10, 10, image.width - 10, image.height - 10)

        s._face_region_in = fake_region
        s._head_crop = lambda im: (200, 40, 400, 240)
        found = s._face_region(photo, _StubJob())
        self.assertIsNotNone(found)
        self.assertEqual(calls, [(600, 900), (200, 200)])
        # ...and the box comes back in FULL-image coordinates, not crop ones.
        self.assertEqual(found[1], (210, 50, 390, 230))
        self.assertEqual(found[0].size, photo.size)

    def test_a_region_covering_the_whole_photo_is_refused(self):
        s, photo = self._services_with_mask((5, 5, 595, 895))
        self.assertIsNone(s._face_region(photo, _StubJob()))

    def test_no_detection_at_all_is_refused(self):
        s, photo = self._services_with_mask(None)
        self.assertIsNone(s._face_region(photo, _StubJob()))

    def test_the_segmenter_is_allowed_and_documented(self):
        from app.core.workflow_ai import NODE_GUIDE, NODE_OUTPUTS
        self.assertIn("FaceSegment", ALLOWED_NODE_TYPES)
        self.assertIn("FaceSegment", NODE_GUIDE)
        self.assertEqual(NODE_OUTPUTS["FaceSegment"], ["IMAGE", "MASK", "IMAGE"])

    def test_a_planned_face_swap_does_not_also_get_a_compose(self):
        """Seen live: the planner emitted SWAP_FACE itself, the compose
        coercion saw no compose step and added one, and the job did BOTH — a
        whole stranger transplanted and then their face swapped."""
        import inspect

        from app.core.services import Services as S
        src = inspect.getsource(S._handle_image_edit)
        guard = 's["task"] in ("compose", "faceswap")'
        self.assertIn(guard, src,
                      "the compose coercion must stand down when the plan "
                      "already handles the second image")

    def test_a_face_request_beats_the_generic_combine_route(self):
        """Both coercions can match — the face one has to win, or the second
        photo's whole subject gets transplanted."""
        import inspect

        from app.core.services import Services as S
        src = inspect.getsource(S._handle_image_edit)
        face_at = src.find("wants_face")
        compose_at = src.find("elif references and not any(")
        self.assertGreater(face_at, 0, "no face coercion found")
        self.assertGreater(compose_at, face_at,
                           "the compose coercion must be the ELSE branch")


class RetryWiringTests(unittest.TestCase):
    """A compose retry must re-run the COMPOSE.

    Seen live: it fell through to the generic template path, which sends only
    an input image — the subject filename stayed empty and ComfyUI died on
    'input directory does not exist'. The retry cost a render and bought
    nothing."""

    def test_the_retry_branch_passes_the_reference_and_placement(self):
        import inspect

        from app.core.services import Services
        src = inspect.getsource(Services._handle_image_edit)
        # The retry dispatch must name compose BEFORE the generic else.
        compose_at = src.find('last_step["task"] == "compose"')
        generic_at = src.rfind("candidate = self._render_template_step(")
        self.assertGreater(compose_at, 0,
                           "the retry loop has no compose branch")
        self.assertLess(compose_at, generic_at,
                        "compose must be handled before the generic path")
        branch = src[compose_at:compose_at + 700]
        self.assertIn("_render_compose_step", branch)
        self.assertIn("references[0]", branch)   # the second photo
        self.assertIn("base_mask", branch)       # the same placement region


class ApiValidationTests(unittest.TestCase):
    """A bad reference image must be a 404 the user sees now, not a job that
    dies minutes later."""

    def setUp(self):
        from app.api.routes import create_app
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.services = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))
        self.addCleanup(self.services.stop)
        app = create_app(self.services)
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def _asset(self, name="a.png"):
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), (9, 9, 9)).save(buf, format="PNG")
        return self.services.store.save_upload(name, buf.getvalue())

    def test_a_reference_image_reaches_the_job(self):
        a, b = self._asset("a.png"), self._asset("b.png")
        r = self.client.post("/api/edits", json={
            "asset_id": a.id, "prompt": "put him in the scene",
            "reference_asset_ids": [b.id]})
        self.assertEqual(r.status_code, 202)
        job = self.services.queue.get(r.get_json()["id"])
        self.assertEqual(job.payload["reference_asset_ids"], [b.id])

    def test_an_unknown_reference_is_refused_immediately(self):
        a = self._asset()
        r = self.client.post("/api/edits", json={
            "asset_id": a.id, "prompt": "x",
            "reference_asset_ids": ["nope"]})
        self.assertEqual(r.status_code, 404)

    def test_an_image_cannot_be_composed_with_itself(self):
        a = self._asset()
        r = self.client.post("/api/edits", json={
            "asset_id": a.id, "prompt": "x", "reference_asset_ids": [a.id]})
        self.assertEqual(r.status_code, 400)

    def test_a_malformed_field_is_refused(self):
        a = self._asset()
        r = self.client.post("/api/edits", json={
            "asset_id": a.id, "prompt": "x", "reference_asset_ids": "b"})
        self.assertEqual(r.status_code, 400)

    def test_plain_edits_still_work_untouched(self):
        a = self._asset()
        r = self.client.post("/api/edits", json={"asset_id": a.id,
                                                 "prompt": "make it night"})
        self.assertEqual(r.status_code, 202)
        self.assertNotIn("reference_asset_ids",
                         self.services.queue.get(r.get_json()["id"]).payload)


if __name__ == "__main__":
    unittest.main()
