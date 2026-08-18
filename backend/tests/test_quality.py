"""Tests for the staged quality pipeline: scene analysis + routing, mask
adjustment/verification, seam inspection (model + deterministic stats), the
0-100 scorecard, and the iterate-to-target edit loop."""
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.adapters.base import EditResult
from app.config import Settings
from app.core import quality
from app.core.llm import LLMReply
from app.core.services import Services
from tests.test_workflow_job import DeadLLM


class OneJson:
    source = "local"

    def __init__(self, payload):
        self.payload = payload

    def complete(self, system, prompt, max_tokens=4096):
        return LLMReply(json.dumps(self.payload), "fake", "local")


class ScriptedCritic:
    """critique() unused; ask() returns scripted replies in order."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.questions = []

    def ask(self, image, question):
        self.questions.append(question)
        return self.replies.pop(0)


class AnalyzeTests(unittest.TestCase):
    def test_routes_and_clamps(self):
        a = quality.analyze_scene(OneJson({
            "task": "img2img", "mask_adjust": "grow", "adjust_px": 400,
            "denoise": 5, "reason": "restyle"}), "make it winter", False)
        self.assertEqual(a["task"], "img2img")
        self.assertEqual(a["adjust_px"], 64)   # clamped
        self.assertEqual(a["denoise"], 0.9)    # clamped
        self.assertEqual(a["mask_adjust"], "grow")

    def test_user_mask_forces_inpaint(self):
        a = quality.analyze_scene(OneJson({"task": "img2img"}),
                                  "make it winter", has_mask=True)
        self.assertEqual(a["task"], "inpaint")

    def test_failures_return_none(self):
        self.assertIsNone(quality.analyze_scene(DeadLLM(), "x", False))
        self.assertIsNone(quality.analyze_scene(
            OneJson({"task": "teleport"}), "x", False))


class PlanEditTests(unittest.TestCase):
    def test_compound_request_becomes_multiple_steps(self):
        steps = quality.plan_edit(OneJson({"steps": [
            {"task": "inpaint", "instruction": "change the tshirt"},
            {"task": "outpaint", "instruction": "extend the image"},
        ]}), "change the tshirt and outpaint the image", False)
        self.assertEqual([s["task"] for s in steps], ["inpaint", "outpaint"])
        self.assertEqual(steps[0]["instruction"], "change the tshirt")

    def test_single_object_shape_still_works(self):
        steps = quality.plan_edit(OneJson({"task": "img2img"}), "restyle", False)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["task"], "img2img")

    def test_user_mask_forces_first_step_inpaint_keeps_rest(self):
        steps = quality.plan_edit(OneJson({"steps": [
            {"task": "img2img", "instruction": "recolor"},
            {"task": "outpaint", "instruction": "extend"},
        ]}), "recolor and extend", has_mask=True)
        self.assertEqual([s["task"] for s in steps], ["inpaint", "outpaint"])

    def test_outpaint_listed_first_still_runs_after_the_content_edit(self):
        """Combined jobs ('change the clothing and change the format') are
        two steps, and the canvas change always runs AFTER the content edit
        regardless of how the LLM ordered them."""
        steps = quality.plan_edit(OneJson({"steps": [
            {"task": "outpaint", "instruction": "change the format"},
            {"task": "inpaint", "instruction": "change the clothing"},
        ]}), "change the clothing and change the format of the image", False)
        self.assertEqual([s["task"] for s in steps], ["inpaint", "outpaint"])
        self.assertEqual(steps[0]["instruction"], "change the clothing")

    def test_format_change_is_coerced_to_outpaint(self):
        """Seen live: qwen routed 'change the format of the image to a wider
        landscape format' to CHANGE_STYLE — an img2img restyle that never
        grows the canvas. The coercion is deterministic."""
        steps = quality.plan_edit(OneJson({"steps": [
            {"operation": "CHANGE_ATTRIBUTE", "target": "car",
             "instruction": "change the car's color to blue"},
            {"operation": "CHANGE_STYLE", "target": "",
             "instruction": "change the format of the image to a wider "
                            "landscape format"},
        ]}), "change the car's color to blue and change the format of the "
             "image to a wider landscape format", False)
        self.assertEqual([s["task"] for s in steps], ["inpaint", "outpaint"])
        self.assertEqual(steps[1]["operation"], "OUTPAINT")

    def test_style_and_text_wording_is_not_mistaken_for_format(self):
        steps = quality.plan_edit(OneJson({"steps": [
            {"operation": "CHANGE_STYLE",
             "instruction": "make it look like an oil painting on canvas"},
            {"operation": "CHANGE_TEXT", "target": "sign",
             "instruction": "change the format of the text on the sign"},
        ]}), "restyle it and fix the sign", False)
        self.assertEqual([s["task"] for s in steps], ["img2img", "inpaint"])
        steps = quality.plan_edit(OneJson({"steps": [
            {"task": "outpaint", "instruction": "extend the canvas"},
            {"task": "img2img", "instruction": "recolor the shirt"},
        ]}), "recolor the shirt and extend the canvas", has_mask=True)
        self.assertEqual([s["task"] for s in steps], ["inpaint", "outpaint"])
        self.assertEqual(steps[0]["instruction"], "recolor the shirt")

    def test_caps_at_three_steps_and_failsafe(self):
        steps = quality.plan_edit(OneJson({"steps": [
            {"task": "inpaint", "instruction": str(i)} for i in range(6)
        ]}), "x", False)
        self.assertEqual(len(steps), 3)
        self.assertIsNone(quality.plan_edit(DeadLLM(), "x", False))


class PruneInventedStepsTests(unittest.TestCase):
    """The LLM routes, but it doesn't get to invent work (seen live: plans
    padded with outpaint/'ensure' steps the user never asked for)."""

    @staticmethod
    def _step(task, instruction):
        return {"task": task, "instruction": instruction,
                "mask_adjust": "keep", "adjust_px": 0, "denoise": 0.6,
                "reason": ""}

    def test_uninvited_outpaint_is_dropped(self):
        kept, dropped = quality.prune_invented_steps(
            "she sits on the beach", [
                self._step("inpaint", "edit the figure"),
                self._step("outpaint", "extend the beach background")])
        self.assertEqual([s["task"] for s in kept], ["inpaint"])
        self.assertEqual(dropped[0]["task"], "outpaint")
        self.assertIn("never asks", dropped[0]["why"])

    def test_requested_outpaint_is_kept(self):
        kept, dropped = quality.prune_invented_steps(
            "change the shirt and outpaint the image to the sides", [
                self._step("inpaint", "change the shirt"),
                self._step("outpaint", "extend the canvas")])
        self.assertEqual([s["task"] for s in kept], ["inpaint", "outpaint"])
        self.assertEqual(dropped, [])

    def test_aspect_ratio_counts_as_canvas_intent(self):
        kept, _ = quality.prune_invented_steps(
            "outpaint the image to 9:16 scaling", [
                self._step("outpaint", "extend to 9:16")])
        self.assertEqual(len(kept), 1)

    def test_format_change_counts_as_canvas_intent(self):
        """'change the format of the image' IS a canvas request — the legit
        outpaint step of a combined job must survive the pruner."""
        kept, dropped = quality.prune_invented_steps(
            "change the clothing and change the format of the image", [
                self._step("inpaint", "change the clothing"),
                self._step("outpaint", "extend the canvas")])
        self.assertEqual([s["task"] for s in kept], ["inpaint", "outpaint"])
        self.assertEqual(dropped, [])

    def test_orientation_wording_counts_as_canvas_intent(self):
        kept, _ = quality.prune_invented_steps(
            "make it a portrait orientation picture", [
                self._step("outpaint", "extend vertically")])
        self.assertEqual(len(kept), 1)

    def test_ensure_paraphrase_step_is_dropped(self):
        kept, dropped = quality.prune_invented_steps(
            "make the wall bright green", [
                self._step("inpaint", "make the wall bright green"),
                self._step("img2img",
                           "ensure the wall is green as requested")])
        self.assertEqual([s["task"] for s in kept], ["inpaint"])
        self.assertIn("re-checks", dropped[0]["why"])

    def test_duplicates_are_dropped(self):
        kept, dropped = quality.prune_invented_steps(
            "swap the sky", [
                self._step("inpaint", "swap the sky"),
                self._step("inpaint", "swap the sky")])
        self.assertEqual(len(kept), 1)
        self.assertIn("duplicate", dropped[0]["why"])

    def test_single_legit_step_untouched(self):
        step = self._step("inpaint", "change the shirt to a red jacket")
        kept, dropped = quality.prune_invented_steps(
            "change the shirt to a red jacket", [step])
        self.assertEqual(kept, [step])
        self.assertEqual(dropped, [])


class OrderStepsTests(unittest.TestCase):
    @staticmethod
    def _step(task, instruction):
        return {"task": task, "instruction": instruction}

    def test_video_always_runs_last(self):
        # The video step ENDS the chain (it returns a video asset) — ordered
        # first it would silently drop every step after it.
        steps = [self._step("video", "animate her"),
                 self._step("inpaint", "give her a red shirt")]
        self.assertEqual([s["task"] for s in quality.order_steps(steps)],
                         ["inpaint", "video"])

    def test_canonical_edit_outpaint_upscale_order(self):
        steps = [self._step("upscale", "sharpen"),
                 self._step("outpaint", "wider"),
                 self._step("inpaint", "blue dress")]
        self.assertEqual([s["task"] for s in quality.order_steps(steps)],
                         ["inpaint", "outpaint", "upscale"])

    def test_stable_within_a_tier(self):
        steps = [self._step("inpaint", "shirt"), self._step("custom", "fx"),
                 self._step("inpaint", "hat")]
        self.assertEqual(
            [s["instruction"] for s in quality.order_steps(steps)],
            ["shirt", "fx", "hat"])


class BetterCandidateTests(unittest.TestCase):
    """The undo-protection rule. Seen live: a retry drifted back toward the
    ORIGINAL photo, averaged higher (identity/consistency soar as the edit
    fades) and replaced the attempt that actually made the change."""

    MADE_IT = {"realism": 55, "prompt_accuracy": 95,
               "identity_preservation": 85, "scene_consistency": 70,
               "artifact_free": 80, "visual_quality": 75}    # overall 77
    UNDONE = {"realism": 90, "prompt_accuracy": 20,
              "identity_preservation": 100, "scene_consistency": 95,
              "artifact_free": 100, "visual_quality": 85}    # mean 82, gated 20

    def test_reverted_edit_never_replaces_the_best(self):
        # The reverted image averages higher across the six categories — that
        # was the trap. overall() now gates on adherence, so the headline
        # number itself can no longer be fooled by an edit that undid the
        # request (D18), and better_candidate refuses it independently.
        self.assertEqual(quality.overall(self.UNDONE), 20)
        self.assertGreater(quality.overall(self.MADE_IT),
                           quality.overall(self.UNDONE))
        self.assertFalse(quality.better_candidate(self.UNDONE, self.MADE_IT))

    def test_material_accuracy_gain_wins_despite_lower_average(self):
        best = dict.fromkeys(quality.SCORE_KEYS, 80) | {"prompt_accuracy": 40}
        cand = dict.fromkeys(quality.SCORE_KEYS, 64) | {"prompt_accuracy": 90}
        # best sits below the adherence gate, so its headline is capped at
        # its accuracy; the candidate that actually performed the request
        # wins on both counts.
        self.assertEqual(quality.overall(best), 40)
        self.assertGreater(quality.overall(cand), quality.overall(best))
        self.assertTrue(quality.better_candidate(cand, best))

    def test_overall_gates_on_adherence_not_averages(self):
        # "make her sit down" left her standing: accuracy 20, identity 100,
        # artifact_free 90 — the old mean reported 70 (seen live, D18).
        r10a = {"realism": 80, "prompt_accuracy": 20,
                "identity_preservation": 100, "scene_consistency": 90,
                "artifact_free": 90, "visual_quality": 40}
        self.assertEqual(quality.overall(r10a), 20)
        # At or above the gate the mean still stands.
        fine = dict.fromkeys(quality.SCORE_KEYS, 80)
        self.assertEqual(quality.overall(fine), 80)

    def test_higher_average_with_stable_accuracy_wins(self):
        best = dict.fromkeys(quality.SCORE_KEYS, 70)
        cand = dict.fromkeys(quality.SCORE_KEYS, 85) | {"prompt_accuracy": 68}
        self.assertTrue(quality.better_candidate(cand, best))

    def test_small_accuracy_noise_is_tolerated_big_drop_is_not(self):
        best = dict.fromkeys(quality.SCORE_KEYS, 80)
        noise = dict.fromkeys(quality.SCORE_KEYS, 86) | {"prompt_accuracy": 76}
        drop = dict.fromkeys(quality.SCORE_KEYS, 86) | {"prompt_accuracy": 70}
        self.assertTrue(quality.better_candidate(noise, best))
        self.assertFalse(quality.better_candidate(drop, best))

    def test_none_handling_matches_the_old_fallbacks(self):
        s = dict.fromkeys(quality.SCORE_KEYS, 80)
        self.assertFalse(quality.better_candidate(None, s))
        self.assertTrue(quality.better_candidate(s, None))


class PlacementMaskTests(unittest.TestCase):
    def test_classify_edit(self):
        for text in ("put a dog in the background", "add a red balloon",
                     "insert a lamp next to the sofa", "place a hat on him"):
            self.assertEqual(quality.classify_edit(text), "add", text)
        for text in ("change the shirt to a red jacket", "remove the car",
                     "make the wall green", "replace the sky"):
            self.assertEqual(quality.classify_edit(text), "modify", text)

    def test_box_mask_lands_in_the_right_cell(self):
        m = quality.box_mask((300, 300), cell=3, obj_size="small")  # top-right
        self.assertEqual(m.size, (300, 300))
        # Mass concentrated top-right, none bottom-left.
        self.assertGreater(sum(m.crop((200, 0, 300, 100)).getdata()),
                           sum(m.crop((0, 200, 100, 300)).getdata()))
        # Feathered edges (some mid-gray values).
        self.assertTrue(any(0 < v < 255 for v in set(m.getdata())))

    def test_propose_placement_uses_the_vision_model(self):
        class Critic:
            def ask(self, image, question):
                return '{"cell": 1, "size": "large"}'

        m = quality.propose_placement(Critic(), Image.new("RGB", (300, 300)),
                                      "add a dog")
        self.assertGreater(sum(m.crop((0, 0, 150, 150)).getdata()),
                           sum(m.crop((150, 150, 300, 300)).getdata()))

    def test_propose_placement_falls_back_to_center(self):
        m = quality.propose_placement(None, Image.new("RGB", (300, 300)),
                                      "add a dog")
        self.assertGreater(sum(m.crop((100, 100, 200, 200)).getdata()), 0)

        class Broken:
            def ask(self, image, question):
                raise OSError("down")

        m2 = quality.propose_placement(Broken(), Image.new("RGB", (90, 90)),
                                       "add a cat")
        self.assertGreater(sum(m2.getdata()), 0)


class EnhancePromptTests(unittest.TestCase):
    def test_append_only_user_words_always_survive(self):
        out = quality.enhance_prompt(
            OneJson({"add": "photorealistic, no artifacts",
                     "negative": "blurry"}),
            "remove the background buildings", "inpaint")
        self.assertTrue(out["positive"].startswith(
            "remove the background buildings"))
        self.assertIn("photorealistic", out["positive"])
        self.assertEqual(out["negative"], "blurry")

    def test_llm_cannot_rewrite_or_censor(self):
        # Even a reply that tries to replace the prompt only APPENDS.
        out = quality.enhance_prompt(
            OneJson({"add": "a nice landscape instead", "negative": ""}),
            "remove the buildings", "inpaint")
        self.assertTrue(out["positive"].startswith("remove the buildings"))

    def test_fail_open_with_stock_boosters(self):
        out = quality.enhance_prompt(DeadLLM(), "remove the chair", "inpaint")
        self.assertTrue(out["positive"].startswith("remove the chair"))
        self.assertIn("photorealistic", out["positive"])


class MaskToolTests(unittest.TestCase):
    def _mask(self):
        m = Image.new("L", (64, 64), 0)
        m.paste(255, (24, 24, 40, 40))
        return m

    def test_grow_and_shrink(self):
        m = self._mask()
        grown = quality.adjust_mask(m, "grow", 8)
        shrunk = quality.adjust_mask(m, "shrink", 8)
        self.assertGreater(sum(grown.getdata()), sum(m.getdata()))
        self.assertLess(sum(shrunk.getdata()), sum(m.getdata()))
        self.assertIs(quality.adjust_mask(m, "keep", 8), m)

    def test_verify_mask_parses_and_is_failsafe(self):
        img = Image.new("RGB", (64, 64), (10, 10, 10))
        ok = quality.verify_mask(
            ScriptedCritic(['{"match": true, "why": "covers the chair"}']),
            img, self._mask(), "remove the chair")
        self.assertTrue(ok["match"])
        self.assertIsNone(quality.verify_mask(object(), img, self._mask(), "x"))

    def test_size_mismatch_never_raises(self):
        """ComfyUI's VAE rounds render sizes to multiples of 8, so the edited
        image is often a few px smaller than the mask. Regression for the
        'images do not match' crash: every quality check must align sizes."""
        mask = self._mask()                       # 64x64
        edited = Image.new("RGB", (56, 56), (30, 30, 30))  # VAE-rounded
        edited.paste((250, 40, 40), (21, 21, 35, 35))
        issues = quality.seam_stats(edited, mask)  # must not raise
        self.assertTrue(any("color mismatch" in i for i in issues))
        # inspect_seams + verify_mask survive the mismatch too
        self.assertIsInstance(
            quality.inspect_seams(object(), edited, mask), list)
        ok = quality.verify_mask(
            ScriptedCritic(['{"match": true, "why": "ok"}']),
            edited, mask, "x")
        self.assertTrue(ok["match"])

    def test_seam_stats_flags_color_discontinuity(self):
        # inside of the mask is bright red, outside is dark grey → seam
        img = Image.new("RGB", (64, 64), (30, 30, 30))
        img.paste((250, 40, 40), (24, 24, 40, 40))
        issues = quality.seam_stats(img, self._mask())
        self.assertTrue(any("color mismatch" in i for i in issues))
        # a uniform image has no seam issues
        self.assertEqual(
            quality.seam_stats(Image.new("RGB", (64, 64), (90, 90, 90)),
                               self._mask()), [])


class ScorecardTests(unittest.TestCase):
    IMG = Image.new("RGB", (32, 32), (5, 5, 5))

    def test_structured_scores_parse_and_clamp(self):
        critic = ScriptedCritic([json.dumps({
            "realism": 97, "prompt_accuracy": 120, "identity_preservation": 96,
            "scene_consistency": 95, "artifact_free": -3, "visual_quality": 99})])
        s = quality.scorecard(critic, self.IMG, "x")
        self.assertEqual(s["prompt_accuracy"], 100)
        self.assertEqual(s["artifact_free"], 0)
        self.assertFalse(quality.meets_target(s, 95))
        self.assertEqual(quality.weakest(s)[0], "artifact_free")

    def test_falls_back_to_critique_and_none(self):
        class CritOnly:
            def critique(self, image, prompt):
                class C:
                    score = 8.5
                return C()

        s = quality.scorecard(CritOnly(), self.IMG, "x")
        self.assertEqual(s["realism"], 85)
        self.assertIsNone(quality.scorecard(object(), self.IMG, "x"))

    def test_meets_target_and_overall(self):
        good = dict.fromkeys(quality.SCORE_KEYS, 96)
        self.assertTrue(quality.meets_target(good, 95))
        self.assertEqual(quality.overall(good), 96)
        self.assertFalse(quality.meets_target(None, 95))


class EditPipelineIntegrationTests(unittest.TestCase):
    """The full loop on a real-ish (non-mock) edit path with fakes."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="fake",
            first_run_setup=False, comfyui_dir="", quality_rounds=2,
            quality_target=95))
        self.s.scout.llm = DeadLLM()

        class RealishInpaint:
            name = "fake-real"
            is_mock = False
            calls = 0

            def inpaint(self, image, mask, prompt):
                RealishInpaint.calls += 1
                return EditResult(image=Image.new("RGB", image.size),
                                  adapter="fake-real", is_mock=False, meta={})

        self.RealishInpaint = RealishInpaint
        self.s.inpainting = RealishInpaint()
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), (5, 5, 5)).save(buf, format="PNG")
        self.asset = self.s.store.save_upload("p.png", buf.getvalue())
        # Pre-seed the scene graph so the (scripted) critic's replies feed the
        # edit loop, not the up-front scene analysis. These tests exercise the
        # loop; scene analysis has its own tests.
        self.s._scene_cache[self.asset.id] = {"scene": "", "objects": []}

    def tearDown(self):
        self.s.stop()
        self.tmp.cleanup()

    @staticmethod
    def _score_json(v):
        return json.dumps(dict.fromkeys(quality.SCORE_KEYS, v))

    def test_iterates_until_target_met_and_reports_scores(self):
        # analysis → inpaint; verify → match; inspect → issues; score 80;
        # round 1: inspect clean + score 97 → target met, stop early.
        self.s.llm = OneJson({"task": "inpaint", "mask_adjust": "keep",
                              "adjust_px": 0, "reason": "regional edit"})
        self.s.critic = ScriptedCritic([
            '{"match": true, "why": "ok"}',                    # verify_mask
            '{"issues": ["visible seam"]}',                    # inspect #1
            self._score_json(80),                              # score #1
            '{"issues": []}',                                  # inspect #2
            self._score_json(97),                              # score #2
        ])
        self.s.start()
        job = self.s.queue.enqueue("image_edit", {
            "asset_id": self.asset.id, "prompt": "remove the chair"})
        done = self.s.queue.wait_for(job.id, timeout=20)
        self.assertEqual(done.state.value, "completed")
        self.assertTrue(done.result["passed"])
        self.assertEqual(done.result["overall"], 97)
        self.assertEqual(done.result["rounds"], 1)
        self.assertEqual(self.RealishInpaint.calls, 2)
        logs = " ".join(e["msg"] for e in done.logs)
        for marker in ("[stage] analyze", "[stage] inspect", "[stage] score",
                       "[stage] verify", "production-ready",
                       "[mask] refined mask applied"):
            self.assertIn(marker, logs)

    def test_round_budget_keeps_best_and_reports_honestly(self):
        self.s.llm = DeadLLM()  # analysis fails → default inpaint route
        self.s.critic = ScriptedCritic([
            '{"match": true, "why": "ok"}',           # verify_mask
            '{"issues": []}', self._score_json(70),   # attempt 1
            '{"issues": []}', self._score_json(60),   # round 1 (worse)
            '{"issues": []}', self._score_json(75),   # round 2 (better)
        ])
        self.s.start()
        job = self.s.queue.enqueue("image_edit", {
            "asset_id": self.asset.id, "prompt": "remove the chair"})
        done = self.s.queue.wait_for(job.id, timeout=20)
        self.assertEqual(done.state.value, "completed")
        self.assertFalse(done.result["passed"])
        self.assertEqual(done.result["overall"], 75)  # best of three kept
        logs = " ".join(e["msg"] for e in done.logs)
        self.assertIn("Round 1 discarded", logs)
        self.assertIn("Round 2 kept", logs)
        self.assertIn("weakest category", logs)

    def test_whole_frame_restyle_retry_escalates_denoise(self):
        """Measured live (RTX 4060 A/B): 0.6 keeps composition but can
        undershoot the look. A retry must spend MORE denoise (0.8), never
        re-roll the same 0.6 — and never fall back to the template default,
        which regenerates the whole picture (the recipe used to drop the
        denoise entirely)."""
        class VariantInpaint:
            name = "fake-variant"
            is_mock = False
            supports_variants = True
            calls: list[dict] = []

            def inpaint(self, image, mask, prompt, *, negative="",
                        checkpoint=None, variant="modern", denoise=None):
                VariantInpaint.calls.append(
                    {"variant": variant, "denoise": denoise})
                return EditResult(image=Image.new("RGB", image.size),
                                  adapter=self.name, is_mock=False, meta={})

        VariantInpaint.calls = []
        self.s.inpainting = VariantInpaint()
        self.s.llm = DeadLLM()
        full = Image.new("L", (32, 32), 255)
        self.s._text_mask = lambda *a, **k: (full, {"peak": 0.9})
        self.s._next_edit_recipe = lambda *a, **k: (None, None)
        # Consumption order (traced): verify_mask eats reply 1 as a harmless
        # advisory no-op (no "match" key), inspect eats reply 2, scorecard
        # eats reply 3 (no SCORE_KEYS → None), and attempt 1's adherence
        # fallback eats reply 4. The 70 makes attempt 1 miss; round 1's
        # judges hit an empty script and keep the best — the assertions are
        # settled by then, because the retry RENDER itself carries the 0.8.
        self.s.critic = ScriptedCritic([
            '{"issues": []}', self._score_json(70),
            '{"issues": []}', self._score_json(96),
        ])
        self.s.start()
        job = self.s.queue.enqueue("image_edit", {
            "asset_id": self.asset.id, "prompt": "make the sky a warm sunset"})
        done = self.s.queue.wait_for(job.id, timeout=20)
        self.assertEqual(done.state.value, "completed", done.error)
        denoises = [c["denoise"] for c in VariantInpaint.calls]
        self.assertEqual(denoises[0], 0.6)         # the measured floor
        self.assertEqual(denoises[-1], 0.8)        # the escalation rung
        logs = " ".join(e["msg"] for e in done.logs)
        self.assertIn("Retry raises denoise 0.6 → 0.8", logs)

    def test_retry_that_undoes_the_edit_is_rejected(self):
        """Regression (seen live): a retry drifting back toward the ORIGINAL
        photo averages higher — identity/consistency soar as the edit fades —
        and used to replace the attempt that actually made the change. The
        user sees their edit 'undone'. Fidelity now gates keep-best."""
        self.s.llm = DeadLLM()
        made_it = json.dumps(BetterCandidateTests.MADE_IT)   # overall 77
        undone = json.dumps(BetterCandidateTests.UNDONE)     # overall 82 (!)
        self.s.critic = ScriptedCritic([
            '{"match": true, "why": "ok"}',    # verify_mask
            '{"issues": []}', made_it,         # attempt 1: edit performed
            '{"issues": []}', undone,          # round 1: edit reverted
            '{"issues": []}', undone,          # round 2: edit reverted
        ])
        self.s.start()
        job = self.s.queue.enqueue("image_edit", {
            "asset_id": self.asset.id, "prompt": "remove the chair"})
        done = self.s.queue.wait_for(job.id, timeout=20)
        self.assertEqual(done.state.value, "completed")
        self.assertEqual(done.result["overall"], 77)  # the real edit survived
        logs = " ".join(e["msg"] for e in done.logs)
        self.assertIn("never allowed to undo the edit", logs)

    def test_rejected_mask_gets_one_corrective_pass(self):
        """A mask the vision check rejects is re-cut once with the objection
        folded in; the corrected mask is kept only when it verifies."""
        self.s.llm = DeadLLM()
        self.s.critic = ScriptedCritic([
            '{"match": false, "why": "it covers the sky, not the chair"}',
            '{"match": true, "why": "now covers the chair"}',   # re-check
            '{"issues": []}', self._score_json(96),             # final score
        ])
        self.s.start()
        job = self.s.queue.enqueue("image_edit", {
            "asset_id": self.asset.id, "prompt": "remove the chair"})
        done = self.s.queue.wait_for(job.id, timeout=20)
        self.assertEqual(done.state.value, "completed")
        logs = " ".join(e["msg"] for e in done.logs)
        self.assertIn("Mask corrected", logs)
        self.assertIn("[mask] corrected mask applied", logs)

    def test_img2img_intent_routes_to_template(self):
        submitted = []

        class Comfy:
            def is_up(self):
                return True

            def upload_image(self, image, prefix):
                return "edit_src.png"

            def run_graph(self, graph):
                submitted.append(graph)
                return Image.new("RGB", (16, 16)), "pid-1"

        self.s.comfy = Comfy()
        self.s.llm = OneJson({"task": "img2img", "denoise": 0.55,
                              "reason": "whole-image restyle"})
        self.s.critic = ScriptedCritic([self._score_json(96)])
        self.s.start()
        job = self.s.queue.enqueue("image_edit", {
            "asset_id": self.asset.id, "prompt": "make it a snowy winter day"})
        done = self.s.queue.wait_for(job.id, timeout=20)
        self.assertEqual(done.state.value, "completed")
        self.assertEqual(done.result["route"], "img2img")
        self.assertTrue(done.result["passed"])
        self.assertEqual(self.RealishInpaint.calls, 0)  # inpaint bypassed
        # the template got the analysis denoise + the prompt
        ks = next(n for n in submitted[0].values()
                  if n["class_type"] == "KSampler")
        self.assertAlmostEqual(ks["inputs"]["denoise"], 0.55)

    def test_compound_request_chains_workflows_and_reports_them(self):
        """'change the tshirt and outpaint' → inpaint THEN outpaint, output
        feeding input, with workflow+model reported per step."""
        rendered = []

        class Comfy:
            def is_up(self):
                return True

            def upload_image(self, image, prefix):
                return "edit_src.png"

            def run_graph(self, graph):
                rendered.append(graph)
                return Image.new("RGB", (48, 32)), "pid-1"

        self.s.comfy = Comfy()
        self.s.llm = OneJson({"steps": [
            {"task": "inpaint", "instruction": "change the tshirt"},
            {"task": "outpaint", "instruction": "outpaint the image"},
        ]})
        self.s.critic = ScriptedCritic([
            '{"match": true, "why": "ok"}',   # mask check (step 1)
            '{"issues": []}',                 # seam inspection: left band
            '{"issues": []}',                 # seam inspection: right band
            self._score_json(96),             # final scorecard
        ])
        self.s.start()
        job = self.s.queue.enqueue("image_edit", {
            "asset_id": self.asset.id,
            "prompt": "change the tshirt and outpaint the image"})
        done = self.s.queue.wait_for(job.id, timeout=20)
        self.assertEqual(done.state.value, "completed")
        # both steps ran, in order: inpaint (adapter) then outpaint (comfy)
        self.assertEqual(self.RealishInpaint.calls, 1)
        self.assertEqual(len(rendered), 1)  # the outpaint template render
        plan = done.result["plan"]
        self.assertEqual([s["task"] for s in plan], ["inpaint", "outpaint"])
        self.assertTrue(all("workflow" in s and "model" in s for s in plan))
        logs = " ".join(e["msg"] for e in done.logs)
        self.assertIn("step 1/2", logs)
        self.assertIn("step 2/2", logs)
        self.assertIn("[mask] auto-generated mask applied", logs)  # visible
        self.assertIn("[preview] after step 1/2", logs)  # live progress
        self.assertNotIn("route", done.result)  # multi-step ≠ single route
        # final image (48x32 from the outpaint step) was the one saved
        v = self.s.store.get_version(done.result["version_id"])
        self.assertEqual(Image.open(v.path).size, (48, 32))

    def test_outpaint_uses_continuation_prompts_and_inspects_the_seam(self):
        """Outpaint realism rules: the render prompt describes scene
        CONTINUATION (never leading with the subject), the negative blocks
        extra people, the best installed inpaint model does the blending, and
        the added margins are seam-inspected like any inpaint edit."""
        rendered = []

        class Comfy:
            def is_up(self):
                return True

            def upload_image(self, image, prefix):
                return "edit_src.png"

            def installed_checkpoints(self):
                return ["sd-v1-5-inpainting.safetensors",
                        "epicrealismNaturalSin_inpaint.safetensors"]

            def run_graph(self, graph):
                rendered.append(graph)
                return Image.new("RGB", (96, 32)), "pid-1"  # canvas GREW

        self.s.comfy = Comfy()
        self.s.llm = OneJson({"steps": [
            {"task": "outpaint",
             "instruction": "make it a wide landscape format"},
        ]})
        self.s._scene_cache[self.asset.id] = {
            "scene": "a woman on a beach at sunset", "objects": []}
        self.s.critic = ScriptedCritic([
            '{"issues": []}',        # seam inspection over the pad mask
            self._score_json(96),    # scorecard — target met, no retry
        ])
        self.s.start()
        job = self.s.queue.enqueue("image_edit", {
            "asset_id": self.asset.id,
            "prompt": "make it a wide landscape format"})
        done = self.s.queue.wait_for(job.id, timeout=20)
        self.assertEqual(done.state.value, "completed")

        graph = rendered[0]
        ckpt = next(n for n in graph.values()
                    if n["class_type"] == "CheckpointLoaderSimple")
        self.assertEqual(ckpt["inputs"]["ckpt_name"],
                         "epicrealismNaturalSin_inpaint.safetensors")
        node_types = {n["class_type"] for n in graph.values()}
        self.assertIn("DifferentialDiffusion", node_types)   # soft recipe
        self.assertIn("InpaintModelConditioning", node_types)
        self.assertNotIn("VAEEncodeForInpaint", node_types)  # hard-seam legacy
        texts = [n["inputs"]["text"] for n in graph.values()
                 if n["class_type"] == "CLIPTextEncode"]
        positive = next(t for t in texts if "continuation" in t)
        negative = next(t for t in texts if "extra person" in t)
        self.assertTrue(positive.startswith("seamless continuation"))
        self.assertIn("scene: a woman on a beach at sunset", positive)
        self.assertIn("request: make it a wide landscape format", positive)
        self.assertIn("additional people", negative)
        logs = " ".join(e["msg"] for e in done.logs)
        self.assertIn("soft outpaint", logs)
        self.assertIn("Outpaint model: epicrealism", logs)
        self.assertIn("[stage] inspect", logs)  # outpaint seams now inspected

    def test_upscale_step_runs_without_prompt_parameter(self):
        """The faithful model upscaler takes ONLY an image — the runner must
        not push prompt/seed at templates that don't declare them (live
        failure: 'Template has no parameter prompt', retried 4x)."""
        rendered = []

        class Comfy:
            def is_up(self):
                return True

            def upload_image(self, image, prefix):
                return "edit_src.png"

            def run_graph(self, graph):
                rendered.append(graph)
                return Image.new("RGB", (64, 64)), "pid-1"

        self.s.comfy = Comfy()
        self.s.llm = OneJson({"steps": [
            {"task": "upscale", "instruction": "upscale the image"},
        ]})
        self.s.critic = ScriptedCritic([self._score_json(96)])
        self.s.start()
        job = self.s.queue.enqueue("image_edit", {
            "asset_id": self.asset.id, "prompt": "upscale the image"})
        done = self.s.queue.wait_for(job.id, timeout=20)
        self.assertEqual(done.state.value, "completed")
        self.assertEqual(len(rendered), 1)
        node_types = {n["class_type"] for n in rendered[0].values()}
        self.assertIn("ImageUpscaleWithModel", node_types)
        self.assertNotIn("CLIPTextEncode", node_types)  # promptless template

    def test_animate_intent_detection(self):
        for text in ("animate this photo", "make it move slowly",
                     "turn this picture into a video", "img2video please",
                     "bring the old portrait to life", "make an animated "
                     "version of this", "make a short clip of it"):
            self.assertTrue(quality.animate_intent(text), text)
        for text in ("animated style poster", "remove the animal",
                     "make the sky dramatic", "animation style please"):
            self.assertFalse(quality.animate_intent(text), text)

    def test_plan_coerces_animate_to_video_step(self):
        # A 7B model routing "animate" to CHANGE_STYLE must still yield a
        # video step — deterministically.
        plan = quality.plan_edit(
            OneJson({"steps": [{"task": "img2img",
                                "instruction": "animate the photo"}]}),
            "animate the photo", has_mask=False)
        self.assertEqual([s["task"] for s in plan], ["video"])
        self.assertEqual(plan[0]["operation"], "ANIMATE")
        # ...and a chain gains the missing FINAL video step.
        plan2 = quality.plan_edit(
            OneJson({"steps": [{"task": "inpaint",
                                "instruction": "change the shirt"}]}),
            "change the shirt and animate it", has_mask=False)
        self.assertEqual([s["task"] for s in plan2], ["inpaint", "video"])

    def test_count_request_parses_image_counts(self):
        self.assertEqual(quality.count_request("make 4 images of a cat"),
                         (4, "a cat"))
        count, cleaned = quality.count_request(
            "generate three pictures of a red barn at dawn")
        self.assertEqual((count, cleaned), (3, "a red barn at dawn"))
        self.assertEqual(quality.count_request("a lighthouse at dusk"),
                         (1, "a lighthouse at dusk"))
        self.assertEqual(quality.count_request("make 30 images of dogs")[0], 8)
        self.assertEqual(quality.count_request("make 1 image of a dog")[0], 1)

    def test_video_dims_adapt_to_source_and_memory(self):
        # 1080p source on a 768px budget: aspect preserved, 16-aligned.
        self.assertEqual(Services._video_dims_for((1920, 1080), 768, None),
                         (768, 432))
        # Low commit headroom steps down another 25% up front.
        self.assertEqual(Services._video_dims_for((1920, 1080), 768, 8.0),
                         (576, 320))
        # A small source renders at its own size — never upscaled going in.
        self.assertEqual(Services._video_dims_for((512, 512), 768, None),
                         (512, 512))
        # Upscale target: back toward the source, capped at 2x the render.
        self.assertEqual(Services._video_upscale_target((768, 432),
                                                        (1920, 1080)),
                         (1536, 864))
        self.assertIsNone(Services._video_upscale_target((640, 640),
                                                         (640, 640)))

    def test_outpaint_prompt_helper_and_pad_mask_geometry(self):
        pos, neg = Services._outpaint_prompts("a beach", "wider", "blurry")
        self.assertTrue(pos.startswith("seamless continuation"))
        self.assertIn("scene: a beach", pos)
        self.assertIn("request: wider", pos)
        self.assertIn("extra person", neg)
        self.assertIn("blurry", neg)
        m = Services._pad_mask((64, 48), (128, 48))
        self.assertIsNotNone(m)
        self.assertEqual(m.size, (128, 48))
        self.assertEqual(m.getpixel((0, 24)), 255)     # added left margin
        self.assertEqual(m.getpixel((64, 24)), 0)      # original content
        self.assertEqual(m.getpixel((127, 24)), 255)   # added right margin
        self.assertIsNone(Services._pad_mask((64, 48), (64, 48)))
        self.assertIsNone(Services._pad_mask((64, 48), (48, 32)))

    def test_prompt_enhancement_logged_and_applied(self):
        class SeqLLM:
            source = "local"

            def __init__(self):
                self.n = 0

            def complete(self, system, prompt, max_tokens=4096):
                self.n += 1
                if "decompose" in system.lower():
                    return LLMReply(json.dumps(
                        {"task": "inpaint", "instruction": "remove the car"}),
                        "fake", "local")
                return LLMReply(json.dumps(
                    {"add": "photorealistic, no artifacts",
                     "negative": "blurry"}), "fake", "local")

        captured = []

        class CapturingInpaint:
            name = "fake-real"
            is_mock = False

            def inpaint(self, image, mask, prompt):
                captured.append(prompt)
                return EditResult(image=Image.new("RGB", image.size),
                                  adapter="fake-real", is_mock=False, meta={})

        self.s.inpainting = CapturingInpaint()
        self.s.llm = SeqLLM()
        self.s.critic = ScriptedCritic([
            '{"match": true, "why": "ok"}',
            '{"issues": []}', self._score_json(96),
        ])
        self.s.start()
        job = self.s.queue.enqueue("image_edit", {
            "asset_id": self.asset.id, "prompt": "remove the car"})
        done = self.s.queue.wait_for(job.id, timeout=20)
        self.assertEqual(done.state.value, "completed")
        # the render used the ENHANCED prompt, starting with the user's words
        self.assertTrue(captured[0].startswith("remove the car"))
        self.assertIn("photorealistic", captured[0])
        logs = " ".join(e["msg"] for e in done.logs)
        self.assertIn("[llm] prompt enhanced:", logs)

    def test_user_mask_never_reroutes_away_from_inpaint(self):
        self.s.llm = OneJson({"task": "img2img", "reason": "restyle"})
        self.s.critic = ScriptedCritic([
            '{"match": true, "why": "ok"}',
            '{"issues": []}', self._score_json(96),
        ])
        self.s.start()
        import base64
        mbuf = io.BytesIO()
        m = Image.new("L", (32, 32), 0)
        m.paste(255, (8, 8, 24, 24))
        m.save(mbuf, format="PNG")
        job = self.s.queue.enqueue("image_edit", {
            "asset_id": self.asset.id, "prompt": "make it winter",
            "mask_b64": base64.b64encode(mbuf.getvalue()).decode()})
        done = self.s.queue.wait_for(job.id, timeout=20)
        self.assertEqual(done.state.value, "completed")
        self.assertNotIn("route", done.result)          # stayed on inpaint
        self.assertGreaterEqual(self.RealishInpaint.calls, 1)


class MaskViewBoxTests(unittest.TestCase):
    """The inspector's zoom must follow the mask's SHAPE, not its bbox.
    Measured 2026-08-18 on a real left+right outpaint: the two bands'
    joint bbox is the whole frame, and the full-frame view produced only
    complaints about the untouched (byte-identical) subject 3/3 runs while
    MISSING a real junction stripe 2/2 — the per-band views caught the
    stripe 2/2 and cannot name pixels they are never shown."""

    class _Recorder:
        def __init__(self, reply='{"issues": ["spot"]}'):
            self.sizes = []
            self.reply = reply

        def ask(self, image, question):
            self.sizes.append(image.size)
            return self.reply

    @staticmethod
    def _bands_mask(w=800, h=600, pad=100):
        m = Image.new("L", (w, h), 0)
        m.paste(255, (0, 0, pad, h))
        m.paste(255, (w - pad, 0, w, h))
        return m

    def test_two_bands_become_two_views(self):
        boxes = quality._mask_view_boxes(self._bands_mask())
        self.assertEqual(boxes, [(0, 0, 100, 600), (700, 0, 800, 600)])

    def test_center_blob_keeps_the_single_bbox_zoom(self):
        m = Image.new("L", (800, 600), 0)
        m.paste(255, (300, 200, 500, 400))
        self.assertEqual(quality._mask_view_boxes(m),
                         [(300, 200, 500, 400)])

    def test_hollow_ring_becomes_its_edge_bands(self):
        m = Image.new("L", (800, 600), 255)
        m.paste(0, (100, 100, 700, 500))   # all-side outpaint ring
        boxes = quality._mask_view_boxes(m)
        self.assertEqual(len(boxes), 4)
        self.assertIn((0, 0, 100, 600), boxes)     # left band
        self.assertIn((700, 0, 800, 600), boxes)   # right band
        self.assertIn((0, 0, 800, 100), boxes)     # top band
        self.assertIn((0, 500, 800, 600), boxes)   # bottom band

    def test_single_side_band_stays_one_tight_view(self):
        m = Image.new("L", (800, 600), 0)
        m.paste(255, (0, 0, 800, 120))     # top-only outpaint
        self.assertEqual(quality._mask_view_boxes(m), [(0, 0, 800, 120)])

    def test_inspect_asks_once_per_band_with_located_issues(self):
        critic = self._Recorder()
        edited = Image.new("RGB", (800, 600), (20, 20, 20))
        issues = quality.inspect_seams(critic, edited, self._bands_mask())
        self.assertEqual(len(critic.sizes), 2)
        # each view is band + 30% context, never the full frame
        self.assertTrue(all(w < 300 for w, _h in critic.sizes),
                        critic.sizes)
        self.assertIn("left region: spot", issues)
        self.assertIn("right region: spot", issues)

    def test_inspect_single_view_issues_stay_unprefixed(self):
        critic = self._Recorder()
        m = Image.new("L", (800, 600), 0)
        m.paste(255, (300, 200, 500, 400))
        issues = quality.inspect_seams(
            critic, Image.new("RGB", (800, 600), (20, 20, 20)), m)
        self.assertEqual(len(critic.sizes), 1)
        self.assertIn("spot", issues)
        self.assertNotIn("middle region: spot", issues)


if __name__ == "__main__":
    unittest.main()
