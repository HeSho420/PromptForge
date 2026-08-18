"""Regression tests for the July 2026 live test report (D1-D26).

Each test names the defect it pins down. The reproduction cases are the
report's own: the same wordings, the same plans, the same numbers.
"""
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.adapters.base import EditResult
from app.config import Settings
from app.core import quality
from app.core.llm import LLMReply
from app.core.registry import ModelInfo
from app.core.services import Services
from tests.test_workflow_job import DeadLLM


class OneJson:
    source = "local"

    def __init__(self, payload):
        self.payload = payload

    def complete(self, system, prompt, max_tokens=4096):
        return LLMReply(json.dumps(self.payload), "fake", "local")


class RoutedLLM:
    """Answers the plan prompt with one payload and the checklist prompt with
    another — the edit loop consults the same client for both."""

    source = "local"

    def __init__(self, plan, checks=None):
        self.plan = plan
        self.checks = checks or {}

    def complete(self, system, prompt, max_tokens=4096):
        if "list of checks" in system:
            return LLMReply(json.dumps(self.checks), "fake", "local")
        if "editing-program compiler" in system:
            return LLMReply(json.dumps(self.plan), "fake", "local")
        return LLMReply("{}", "fake", "local")


class ScriptedCritic:
    def __init__(self, replies):
        self.replies = list(replies)
        self.questions = []

    def ask(self, image, question):
        self.questions.append(question)
        return self.replies.pop(0)


def plan(payload, prompt, has_mask=False):
    return quality.plan_edit(OneJson(payload), prompt, has_mask=has_mask)


class BackgroundIntentClauseTests(unittest.TestCase):
    """D6 — the seven-row table from report section 3, decided on meaning
    rather than on how long the clause before "background" happens to be."""

    def test_report_table(self):
        rows = [
            ("change the background to a beach", True),
            ("remove the hat, change the background to a beach", True),
            ("remove the hat and change the background to a beach", True),
            ("delete the logo, change the background to a studio", True),
            ("remove her earrings, change the background to a beach", True),
            ("remove the trash can and change the background to a forest",
             True),
            ("remove the hat very, change the background to a beach", True),
        ]
        for text, want in rows:
            with self.subTest(text=text):
                self.assertEqual(quality.background_intent(text), want)

    def test_exclusions_still_hold(self):
        for text in ("blur the background", "keep the background the same",
                     "remove the background",
                     "change the background to transparent"):
            with self.subTest(text=text):
                self.assertFalse(quality.background_intent(text))


class CompoundSplitTests(unittest.TestCase):
    """D5 — the splitter must be reachable when the planner labels its single
    overloaded step with the capability itself."""

    C01A = ("change the top to a red leather jacket and change the "
            "background to a snowy mountain")

    def test_c01a_one_step_carrying_both_is_split(self):
        steps = plan({"steps": [{
            "operation": "REPLACE_BACKGROUND", "target": "top",
            "instruction": self.C01A}]}, self.C01A)
        self.assertEqual(len(steps), 2)
        tasks = [s["task"] for s in steps]
        self.assertIn("background", tasks)
        self.assertIn("inpaint", tasks)
        bg = next(s for s in steps if s["task"] == "background")
        self.assertNotIn("jacket", bg["instruction"])
        self.assertEqual(bg["target"], "")          # no garment target (D5)
        edit = next(s for s in steps if s["task"] == "inpaint")
        self.assertIn("jacket", edit["instruction"])
        self.assertNotIn("background", edit["instruction"])

    def test_c01b_mislabelled_step_recovers_the_other_half(self):
        # The live plan: one step, REPLACE_BACKGROUND(shirt), whose
        # instruction was LITERALLY the garment half — the snowy mountain
        # was nowhere in the program.
        prompt = ("change the shirt to a red leather jacket and change the "
                  "background to a snowy mountain")
        steps = plan({"steps": [{
            "operation": "REPLACE_BACKGROUND", "target": "shirt",
            "instruction": "change the shirt to a red leather jacket"}]},
            prompt)
        self.assertEqual(len(steps), 2)
        bg = next(s for s in steps if s["task"] == "background")
        self.assertIn("snowy mountain", bg["instruction"])
        edit = next(s for s in steps if s["task"] == "inpaint")
        self.assertIn("leather jacket", edit["instruction"])

    def test_c03a_three_part_request_keeps_all_three(self):
        prompt = ("remove the hat, change the background to a beach and "
                  "make it look like an oil painting")
        steps = plan({"steps": [
            {"operation": "REMOVE_OBJECT", "target": "hat",
             "instruction": "remove the hat"},
            {"operation": "CHANGE_STYLE", "target": "",
             "instruction": "make it look like an oil painting"}]}, prompt)
        tasks = [s["task"] for s in steps]
        self.assertIn("inpaint", tasks)
        self.assertIn("background", tasks)      # was silently dropped (D6)
        self.assertIn("img2img", tasks)

    def test_deterministic_across_runs(self):
        for _ in range(5):
            steps = plan({"steps": [{
                "operation": "REPLACE_BACKGROUND", "target": "top",
                "instruction": self.C01A}]}, self.C01A)
            self.assertEqual(len(steps), 2)


class InventedStepTests(unittest.TestCase):
    """D3 + D13 — the planner routes; it does not get to invent work."""

    def test_m01_fabricated_face_swap_is_dropped(self):
        prompt = "place the man from the second photo standing beside her"
        steps = plan({"steps": [
            {"operation": "SWAP_FACE", "target": "man",
             "instruction": "replace his face with the one from the second "
                            "photo"},
            {"operation": "COMPOSE", "target": "",
             "instruction": prompt}]}, prompt)
        self.assertEqual([s["task"] for s in steps], ["compose"])

    def test_requested_face_swap_survives(self):
        prompt = "swap her face with the face in the second photo"
        steps = plan({"steps": [{
            "operation": "SWAP_FACE", "target": "face",
            "instruction": prompt}]}, prompt)
        self.assertEqual([s["task"] for s in steps], ["faceswap"])

    def test_m03_invented_outpaint_is_dropped(self):
        prompt = "turn this into a 3d scene I can walk around in"
        steps = plan({"steps": [
            {"operation": "OUTPAINT", "target": "",
             "instruction": "extend the scene"},
            {"operation": "SCENE_3D", "target": "",
             "instruction": prompt}]}, prompt)
        self.assertEqual([s["task"] for s in steps], ["scene3d"])


class RemovalConditioningTests(unittest.TestCase):
    """D1 — the object's name must never be positive conditioning."""

    def test_object_moves_to_negative(self):
        enh = quality.removal_conditioning("remove the hat", "hat",
                                           "blurry, low quality")
        self.assertNotIn("hat", enh["positive"])
        self.assertIn("hat", enh["negative"])
        self.assertIn("blurry", enh["negative"])

    def test_object_extracted_from_instruction_when_no_target(self):
        enh = quality.removal_conditioning("remove the parked cars", "", "")
        self.assertNotIn("car", enh["positive"])
        self.assertIn("parked cars", enh["negative"])

    def test_positive_asks_for_the_scene(self):
        enh = quality.removal_conditioning("erase the necklace", "necklace",
                                           "")
        self.assertIn("background", enh["positive"])

    def test_large_removal_refuses_an_invented_subject(self):
        """Measured live: 'remove the bench' on a wide grass shot (27%
        coverage) grew a standing person in the hole. A large emptied
        region names the usual fillers in the negative."""
        big = quality.removal_fillers_negative(0.27)
        self.assertIn("person", big)
        self.assertIn("animal", big)

    def test_small_removal_keeps_the_negative_untouched(self):
        """A hat is a small region; 'person' in the negative would fight the
        hair and forehead the fill must reconstruct."""
        self.assertEqual(quality.removal_fillers_negative(0.03), "")


class ViewIntentTests(unittest.TestCase):
    """D26 + D2 — 'show her from the side' must route AND be reachable."""

    def test_named_side_view_is_view_intent(self):
        for text in ("show her from the side", "show him in profile",
                     "side view of the car", "show her from behind"):
            with self.subTest(text=text):
                self.assertTrue(quality.view_intent(text))

    def test_lighting_from_the_side_is_not_a_camera_move(self):
        self.assertFalse(quality.view_intent("light it from the side"))
        self.assertFalse(
            quality.view_intent("soft lighting coming from the left"))

    def test_named_azimuths(self):
        self.assertEqual(quality.requested_azimuths("show her from the side"),
                         [90])
        self.assertEqual(quality.requested_azimuths("show her from behind"),
                         [180])
        self.assertEqual(quality.requested_azimuths("make it prettier"), [])


class FormatArithmeticTests(unittest.TestCase):
    """Step 5b — for a format request the aspect ratio IS the requirement."""

    def test_r08a_wide_landscape_delivered(self):
        # The live case: 0.887 portrait → 1.115 landscape, and the verifier
        # still said "still missing: a wide landscape format".
        self.assertTrue(quality.format_delivered(
            "extend this into a wide landscape format",
            (1486, 1675), (1861, 1668)))

    def test_landscape_not_delivered_when_still_portrait(self):
        self.assertFalse(quality.format_delivered(
            "extend this into a wide landscape format",
            (1486, 1675), (1486, 1675)))

    def test_non_format_request_is_none(self):
        self.assertIsNone(quality.format_delivered(
            "remove the hat", (100, 100), (100, 100)))

    def test_about_format(self):
        self.assertTrue(quality.about_format("a wide landscape format"))
        self.assertFalse(quality.about_format("the hat is gone"))

    def test_live_extend_left_right_settles_by_width(self):
        # The live outpaint (2026-08-18): 1471->1855 wide, and the vision
        # verifier still reported 0% twice, burning a full retry render.
        req = "extend the picture to the left and right"
        self.assertTrue(quality.format_delivered(
            req, (1471, 1828), (1855, 1828)))
        self.assertTrue(quality.about_format(req))

    def test_growth_on_the_wrong_axis_is_not_delivery(self):
        # The words name the axis: a left+right extension does not deliver
        # "extend the picture upward".
        self.assertFalse(quality.format_delivered(
            "extend the picture upward", (1471, 1828), (1855, 1828)))
        self.assertTrue(quality.format_delivered(
            "extend the picture upward", (1471, 1828), (1471, 2212)))

    def test_content_extends_are_not_format_requests(self):
        # "extend her dress" measured the unchanged canvas as a FAILED
        # format request and appended a phantom, never-satisfiable missing
        # entry — a content edit must return None here.
        self.assertIsNone(quality.format_delivered(
            "extend her dress to the floor", (1024, 1024), (1024, 1024)))
        self.assertIsNone(quality.format_delivered(
            "extend her dress and make the left sleeve red",
            (1024, 1024), (1024, 1024)))

    def test_comparative_picture_phrasings_stay_covered(self):
        self.assertTrue(quality.format_delivered(
            "make the picture wider", (1000, 1000), (1400, 1000)))
        self.assertIsNone(quality.format_delivered(
            "make her smile bigger", (1000, 1000), (1000, 1000)))


class ObjectiveChecksTests(unittest.TestCase):
    """Step 5b / D20 — arithmetic that catches what the model scored 90."""

    @staticmethod
    def _noise(size=(64, 64), seed=7):
        rng = np.random.default_rng(seed)
        return Image.fromarray(
            rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8),
            "RGB")

    def test_added_grain_is_flagged(self):
        before = Image.new("RGB", (64, 64), (128, 128, 128))
        report = quality.objective_report(before, self._noise())
        self.assertGreater(report["sharpness_ratio"], quality.GRAIN_RATIO_MAX)
        flags = quality.objective_flags(report, "relight")
        self.assertTrue(any("grain" in f for f in flags))

    def test_softened_render_is_flagged(self):
        report = quality.objective_report(self._noise(),
                                          Image.new("RGB", (64, 64),
                                                    (128, 128, 128)))
        flags = quality.objective_flags(report, "img2img")
        self.assertTrue(any("softer" in f for f in flags))

    def test_upscale_may_sharpen(self):
        before = Image.new("RGB", (64, 64), (128, 128, 128))
        report = quality.objective_report(before, self._noise())
        self.assertEqual(quality.objective_flags(report, "upscale"), [])

    def test_mask_leak_is_flagged(self):
        before = Image.new("RGB", (64, 64), (10, 10, 10))
        after = before.copy()
        after.paste((250, 250, 250), (48, 0, 64, 64))   # change on the right
        mask = Image.new("L", (64, 64), 0)
        mask.paste(255, (0, 0, 16, 64))                 # region on the left
        report = quality.objective_report(before, after, mask)
        self.assertGreater(report["outside_mask_fraction"],
                           quality.MASK_LEAK_MAX)
        flags = quality.objective_flags(report, "inpaint")
        self.assertTrue(any("outside the selected region" in f
                            for f in flags))

    def test_size_ratio_reports_the_silent_crop(self):
        report = quality.objective_report(
            Image.new("RGB", (1486, 1675)), Image.new("RGB", (1480, 1672)))
        self.assertLess(report["size_ratio"], 1.0)


class RecolourTests(unittest.TestCase):
    """D22 — a recolour is not a replacement."""

    def test_recolour_detected(self):
        self.assertTrue(quality.is_recolour(
            "change the parked car to bright red"))
        self.assertTrue(quality.is_recolour("change the color of the car"))

    def test_replacement_is_not_a_recolour(self):
        # "red" modifies the jacket — this is the garment swap that WORKED
        # live (R01a) and must not be throttled to recolour denoise.
        self.assertFalse(quality.is_recolour(
            "change the top to a red leather jacket"))


class OverallGateTests(unittest.TestCase):
    """D18 — the headline number is gated on adherence, not averaged."""

    def test_r10a_scores_its_accuracy(self):
        r10a = {"realism": 80, "prompt_accuracy": 20,
                "identity_preservation": 100, "scene_consistency": 90,
                "artifact_free": 90, "visual_quality": 40}
        self.assertEqual(quality.overall(r10a), 20)


class EscalationHardwareTests(unittest.TestCase):
    """D8 — the retry ladder consults the hardware it runs on."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))

    def tearDown(self):
        self.s.stop()
        self.tmp.cleanup()

    def test_model_declaring_the_whole_card_does_not_fit(self):
        # The live pick: 6.94 GB on disk, declares 8.0 GB, card has 8.0 GB.
        self.s.registry.register(ModelInfo(
            name="juggernautXL_inpaint", purpose="inpaint",
            vram_gb=8.0, meta={"min_vram_gb": 8.0}))
        self.s.hardware.vram_gb = 8.0
        fits, why = self.s._checkpoint_fits_retry(
            "juggernautXL_inpaint.safetensors")
        self.assertFalse(fits)
        self.assertIn("headroom", why)

    def test_model_with_headroom_fits(self):
        self.s.registry.register(ModelInfo(
            name="epicrealism_inpaint", purpose="inpaint",
            vram_gb=4.0, meta={"min_vram_gb": 4.0}))
        self.s.hardware.vram_gb = 8.0
        fits, _ = self.s._checkpoint_fits_retry(
            "epicrealism_inpaint.safetensors")
        self.assertTrue(fits)

    def test_unknown_model_passes(self):
        fits, _ = self.s._checkpoint_fits_retry("mystery.safetensors")
        self.assertTrue(fits)


class PreviewRegionTests(unittest.TestCase):
    """D11 — the preview shows what the engine will actually use."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))
        self.image = Image.new("RGB", (48, 48), (30, 30, 30))

    def tearDown(self):
        self.s.stop()
        self.tmp.cleanup()

    def test_background_request_gets_background_region(self):
        choice = self.s.preview_region(
            self.image, "change the background to a snowy mountain")
        self.assertIsNotNone(choice)
        self.assertEqual(choice.source, "background")
        self.assertTrue(choice.notes)

    def test_pose_request_says_whole_frame(self):
        choice = self.s.preview_region(self.image, "make her sit down")
        self.assertIsNotNone(choice)
        self.assertEqual(choice.source, "whole-frame")
        self.assertTrue(any("painted region" in n for n in choice.notes))

    def test_regional_edit_uses_the_normal_chooser(self):
        self.assertIsNone(self.s.preview_region(self.image,
                                                "remove the hat"))


class RemovalPipelineIntegrationTests(unittest.TestCase):
    """D1 end to end: the inpaint adapter must receive removal conditioning,
    and D24: the saved result must be restored to the input's exact size."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="fake",
            first_run_setup=False, comfyui_dir="", quality_rounds=2,
            quality_target=95))
        self.s.scout.llm = DeadLLM()

        calls = []

        class CaptureInpaint:
            name = "fake-real"
            is_mock = False
            supports_variants = True

            def inpaint(self, image, mask, prompt, *, negative="",
                        checkpoint=None, variant="modern", denoise=None):
                calls.append({"prompt": prompt, "negative": negative,
                              "variant": variant, "denoise": denoise})
                # Return the /8-rounded size every real route returns (D24).
                w = image.width - image.width % 8
                h = image.height - image.height % 8
                return EditResult(image=Image.new("RGB", (w, h)),
                                  adapter="fake-real", is_mock=False, meta={})

        self.calls = calls
        self.s.inpainting = CaptureInpaint()
        buf = io.BytesIO()
        Image.new("RGB", (70, 67), (5, 5, 5)).save(buf, format="PNG")
        self.asset = self.s.store.save_upload("p.png", buf.getvalue())
        self.s._scene_cache[self.asset.id] = {"scene": "", "objects": []}

    def tearDown(self):
        self.s.stop()
        self.tmp.cleanup()

    @staticmethod
    def _score_json(v):
        return json.dumps(dict.fromkeys(quality.SCORE_KEYS, v))

    def test_remove_the_hat_never_paints_a_hat(self):
        self.s.llm = RoutedLLM({"steps": [{
            "operation": "REMOVE_OBJECT", "target": "hat",
            "instruction": "remove the hat"}]})
        self.s.critic = ScriptedCritic([
            '{"match": true, "why": "ok"}',            # verify_mask
            '{"issues": []}', self._score_json(96),    # inspect + score
        ])
        self.s.start()
        job = self.s.queue.enqueue("image_edit", {
            "asset_id": self.asset.id, "prompt": "remove the hat"})
        done = self.s.queue.wait_for(job.id, timeout=20)
        self.assertEqual(done.state.value, "completed")
        self.assertEqual(len(self.calls), 1)
        self.assertNotIn("hat", self.calls[0]["prompt"])       # D1
        self.assertIn("hat", self.calls[0]["negative"])        # D1
        logs = " ".join(e["msg"] for e in done.logs)
        self.assertIn("removal conditioning", logs)

    def test_saved_result_is_restored_to_input_size(self):
        self.s.llm = RoutedLLM({"steps": [{
            "operation": "REMOVE_OBJECT", "target": "hat",
            "instruction": "remove the hat"}]})
        self.s.critic = ScriptedCritic([
            '{"match": true, "why": "ok"}',
            '{"issues": []}', self._score_json(96),
        ])
        self.s.start()
        job = self.s.queue.enqueue("image_edit", {
            "asset_id": self.asset.id, "prompt": "remove the hat"})
        done = self.s.queue.wait_for(job.id, timeout=20)
        self.assertEqual(done.state.value, "completed")
        version = self.s.store.get_version(done.result["version_id"])
        with Image.open(version.path) as out:
            self.assertEqual(out.size, (70, 67))               # D24

    def test_static_verdict_stops_spending_renders(self):
        """D7 — a verdict that does not move across a full re-render may not
        spend the remaining retry budget."""
        self.s.llm = RoutedLLM(
            {"steps": [{"operation": "CHANGE_ATTRIBUTE", "target": "scarf",
                        "instruction": "add a red scarf around her neck"}]},
            checks={"checks": [{"need": "a red scarf",
                                "probe": "What is around the neck?",
                                "expect": "red scarf"}]})
        self.s.critic = ScriptedCritic([
            '{"match": true, "why": "ok"}',            # verify_mask
            '{"issues": []}', self._score_json(80),    # attempt 1
            '{"answer": "nothing there"}',             # probe
            '{"answer": "nothing there"}',             # confirm probe
            '{"issues": []}', self._score_json(82),    # round 1
            '{"answer": "nothing there"}',             # probe (identical)
            '{"answer": "nothing there"}',             # confirm probe
            # No further replies: a round 2 would crash the scripted critic,
            # which is exactly the point — it must not happen.
        ])
        self.s.start()
        job = self.s.queue.enqueue("image_edit", {
            "asset_id": self.asset.id,
            "prompt": "add a red scarf around her neck"})
        done = self.s.queue.wait_for(job.id, timeout=20)
        self.assertEqual(done.state.value, "completed")
        self.assertEqual(done.result["rounds"], 1)
        logs = " ".join(e["msg"] for e in done.logs)
        self.assertIn("did not change across a full re-render", logs)


class SafetyRestoredTests(unittest.TestCase):
    """D15 — the built-in protections match the categories the suite (and
    the settings UI) has always declared."""

    def test_builtin_categories(self):
        from app.core.safety import BUILTIN_RULES
        self.assertEqual([c for c, _ in BUILTIN_RULES],
                         ["minors", "exposure", "deepfake", "nonconsensual"])


if __name__ == "__main__":
    unittest.main()
