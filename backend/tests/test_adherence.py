"""The prompt is the contract.

These tests pin the behaviour the render pipeline was revised for: a render is
checked against what the REQUEST actually asked for, and when it misses, the
next attempt changes the recipe — a different model, or a different workflow —
instead of re-rolling the seed of something already shown to miss.
"""
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.adapters.comfyui import ALLOWED_NODE_TYPES, ALLOWED_TASKS
from app.config import Settings
from app.core import quality
from app.core.services import Services, _Attempt

IMG = Image.new("RGB", (16, 16), (7, 7, 7))


class ProbingCritic:
    """A vision model that answers probe questions from a dict of scripted
    answers, keyed by a substring of the probe."""

    def __init__(self, answers: dict[str, str], default: str = "none"):
        self.answers = answers
        self.default = default
        self.asked: list[str] = []

    def ask(self, image, question):
        self.asked.append(question)
        for needle, answer in self.answers.items():
            if needle.lower() in question.lower():
                return json.dumps({"answer": answer})
        return json.dumps({"answer": self.default})


class OneJson:
    """An LLM that always replies with the same JSON object."""

    source = "local"

    def __init__(self, payload):
        self.payload = payload

    def complete(self, system, prompt, max_tokens=4096):
        from app.core.llm import LLMReply
        return LLMReply(json.dumps(self.payload), "fake", "local")


class DeadLLM:
    source = "local"

    def complete(self, system, prompt, max_tokens=4096):
        from app.core.llm import LLMUnavailableError
        raise LLMUnavailableError("offline test")


CHECKS = [
    {"need": "a red car", "probe": "What colour is the car?",
     "expect": "red"},
    {"need": "at night", "probe": "Is this daytime or nighttime?",
     "expect": "nighttime"},
]


class ChecklistTests(unittest.TestCase):
    def test_checks_carry_a_neutral_probe_and_an_expected_answer(self):
        llm = OneJson({"checks": [
            {"need": "a red car", "probe": "What colour is the car?",
             "expect": "red"},
            {"need": "high detail", "probe": "Is it detailed?",
             "expect": "yes"},          # decoration — must be dropped
            {"need": "", "probe": "?", "expect": ""},   # malformed
        ]})
        checks = quality.request_checklist(llm, "a red car, high detail")
        self.assertEqual([c["need"] for c in checks], ["a red car"])
        self.assertEqual(checks[0]["expect"], "red")

    def test_unavailable_llm_yields_no_checklist(self):
        self.assertEqual(quality.request_checklist(DeadLLM(), "a cat"), [])


class AnswerMatchingTests(unittest.TestCase):
    def test_literal_and_token_matches(self):
        self.assertTrue(quality.answer_satisfies("A red car", "red"))
        self.assertTrue(quality.answer_satisfies("two people", "2"))
        self.assertTrue(quality.answer_satisfies("It says OPEN", "OPEN"))
        self.assertFalse(quality.answer_satisfies("a blue car", "red"))

    def test_a_denial_never_satisfies_a_positive_expectation(self):
        """The measured llava failure mode: confident prose that describes an
        absence. 'no car is visible' must never satisfy 'red'."""
        self.assertFalse(quality.answer_satisfies("none", "red"))
        self.assertFalse(quality.answer_satisfies("No, there is no car",
                                                  "red"))
        self.assertFalse(quality.answer_satisfies("not visible", "OPEN"))
        # ...and it DOES satisfy an expectation of absence.
        self.assertTrue(quality.answer_satisfies("none", "none"))

    def test_a_synonym_is_not_a_miss(self):
        """The documented false zero: a render that unambiguously showed a
        sunlit meadow scored '0% - missing: a sunlit meadow', because the
        examiner answered in different words and every expected token had to
        appear literally. Two wasted renders per occurrence."""
        self.assertTrue(quality.answer_satisfies(
            "a grassy field in bright sunlight", "sunlit meadow"))
        self.assertTrue(quality.answer_satisfies("tall trees", "tree"))

    def test_a_partial_match_is_inconclusive_not_absent(self):
        """Some of what was asked for, but not most, is not evidence either
        way — and must not cost an escalation rung."""
        self.assertIs(quality.answer_verdict("a dress and a hat",
                                             "red silk dress"), None)
        self.assertIs(quality.answer_verdict("there is no dress",
                                             "red dress"), False)
        self.assertIs(quality.answer_verdict("a bright red dress",
                                             "red dress"), True)

    def test_an_unrelated_answer_still_counts_against(self):
        """Loosening this must not make everything pass: an answer with
        nothing at all in common is still reported (and then confirmed)."""
        self.assertIs(quality.answer_verdict("a wooden chair", "red"), False)

    def test_a_wrong_count_is_not_met_by_substring(self):
        """A checklist count probe expects "2"; "12 people" CONTAINS "2" as a
        substring, and the old fast-path marked the wrong count satisfied —
        so the correction retry never fired. Whole-word containment only."""
        self.assertIs(quality.answer_verdict("12 people", "2"), False)
        self.assertIs(quality.answer_verdict("there are 21 birds", "2"), False)
        # The right count still passes, by digit and by number word.
        self.assertIs(quality.answer_verdict("2 people", "2"), True)
        self.assertIs(quality.answer_verdict("two people", "2"), True)

    def test_the_text_model_settles_a_wording_disagreement(self):
        """When the words share nothing, the TEXT model is asked whether they
        mean the same thing. It never sees the image, so it cannot do what
        the vision model does when shown the request — agree with it."""
        critic = ProbingCritic({"setting": "an open pasture in daylight"})
        checks = [{"need": "a sunlit meadow", "probe": "What is the setting?",
                   "expect": "sunlit meadow"}]
        report = quality.verify_adherence(critic, IMG, "x", checks,
                                          llm=OneJson({"satisfies": True}))
        self.assertEqual(report["missing"], [])
        self.assertEqual(report["accuracy"], 100)

    def test_without_a_text_model_a_real_miss_is_still_a_miss(self):
        """The fallback must not turn every unmatched answer into 'unclear',
        or nothing would ever be reported on a machine with no LLM."""
        critic = ProbingCritic({"colour": "blue", "daytime": "nighttime"})
        report = quality.verify_adherence(critic, IMG, "x", CHECKS, llm=None)
        self.assertEqual(report["missing"], ["a red car"])


class VerifyAdherenceTests(unittest.TestCase):
    def test_the_request_is_never_shown_to_the_examiner(self):
        """Showing the vision model the prompt makes it rubber-stamp — a
        blank image scored 40% adherence that way. The probes must not leak
        it."""
        critic = ProbingCritic({"colour": "red", "daytime": "nighttime"})
        report = quality.verify_adherence(critic, IMG,
                                          "a red car at night", CHECKS)
        self.assertEqual(report["missing"], [])
        self.assertEqual(report["accuracy"], 100)
        for question in critic.asked:
            self.assertNotIn("red car at night", question)
            self.assertNotIn("request", question.lower())

    def test_an_unmet_requirement_is_named(self):
        critic = ProbingCritic({"colour": "blue", "daytime": "nighttime"})
        report = quality.verify_adherence(critic, IMG, "a red car at night",
                                          CHECKS)
        self.assertEqual(report["missing"], ["a red car"])
        self.assertEqual(report["met"], ["at night"])
        self.assertEqual(report["source"], "checklist")

    def test_a_failure_is_confirmed_before_it_costs_a_render(self):
        """One hallucinated 'none' must not spend an escalation rung: the
        probe is asked again, and a second, different answer clears it."""
        class Flaky(ProbingCritic):
            def __init__(self):
                super().__init__({})
                self.n = 0

            def ask(self, image, question):
                self.asked.append(question)
                if "colour" in question:
                    self.n += 1
                    return json.dumps({"answer": "none" if self.n == 1
                                       else "red"})
                return json.dumps({"answer": "nighttime"})

        critic = Flaky()
        report = quality.verify_adherence(critic, IMG, "x", CHECKS)
        self.assertEqual(report["missing"], [])
        self.assertEqual(critic.n, 2)  # re-asked exactly once

    def test_an_examiner_that_cannot_answer_returns_no_verdict(self):
        class Mute:
            def ask(self, image, question):
                return "I am not sure what you mean."

        self.assertIsNone(quality.verify_adherence(Mute(), IMG, "x", CHECKS))
        self.assertIsNone(quality.verify_adherence(object(), IMG, "x", CHECKS))
        self.assertIsNone(quality.verify_adherence(
            ProbingCritic({}), IMG, "x", []))


class EscalationPlanTests(unittest.TestCase):
    def test_ladder_changes_model_then_workflow(self):
        plan = quality.escalation_plan(
            ["a red car"], models=["a.safetensors", "b.safetensors"],
            workflows=["generate", "generate_xl"],
            current_model="a.safetensors", current_workflow="generate",
            max_rungs=3)
        self.assertEqual([s.kind for s in plan],
                         ["emphasize", "model", "workflow"])
        self.assertEqual(plan[1].checkpoint, "b.safetensors")
        self.assertEqual(plan[2].workflow, "generate_xl")

    def test_a_capability_gap_jumps_the_queue(self):
        """No amount of re-seeding an SDXL graph makes text legible — the
        template built for it comes FIRST."""
        plan = quality.escalation_plan(
            ["a readable sign that says OPEN"], models=["a", "b"],
            workflows=["generate_zimage"], current_model="a",
            current_workflow="generate", max_rungs=3)
        self.assertEqual(plan[0].kind, "workflow")
        self.assertEqual(plan[0].workflow, "generate_zimage")

    def test_lens_language_is_not_a_viewpoint_gap(self):
        """'wide-angle framing' is one shot, not a camera move: escalating it
        to orbital view synthesis would answer a question nobody asked."""
        self.assertIsNone(quality.capability_gap(["wide-angle framing"]))
        self.assertIsNone(quality.capability_gap(["a low-angle shot"]))
        self.assertEqual(quality.capability_gap(["shown from another angle"]),
                         "angles")

    def test_nothing_is_ever_tried_twice(self):
        plan = quality.escalation_plan(
            [], models=["a", "b"], workflows=["w1", "w2"],
            current_model="a", current_workflow="w1", max_rungs=10)
        keys = [s.key() for s in plan]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertNotIn(("w1", "a"), keys)

    def test_a_named_model_is_never_swapped_out(self):
        plan = quality.escalation_plan(
            [], models=["a", "b"], workflows=[], current_model="a",
            allow_model_change=False, max_rungs=5)
        self.assertNotIn("model", [s.kind for s in plan])

    def test_emphasize_restates_what_was_missed(self):
        text = quality.emphasize("a red car", ["a red car", "at night"])
        self.assertIn("(a red car:1.3)", text)
        self.assertIn("(at night:1.3)", text)


class AttemptComparisonTests(unittest.TestCase):
    """What counts as 'better' — the ordering the whole ladder rests on."""

    SETTINGS = Settings(data_dir=Path("."), first_run_setup=False,
                        comfyui_dir="")

    @staticmethod
    def _attempt(missing=None, realism=None, source="checklist", acc=None):
        from app.core.critic import Critique
        adherence = None
        if source:
            miss = list(missing or [])
            adherence = {"accuracy": acc if acc is not None
                         else (0 if miss else 100),
                         "missing": miss, "met": [], "source": source}
        return _Attempt(
            image=IMG, prompt_id="p", gen=None, adherence=adherence,
            crit=None if realism is None
            else Critique(score=realism, issues=[], model="fake"))

    def test_delivering_the_request_beats_being_prettier(self):
        on_prompt = self._attempt(missing=[], realism=5.0)
        pretty = self._attempt(missing=["a red car"], realism=9.5)
        self.assertFalse(pretty.beats(on_prompt))
        self.assertTrue(on_prompt.beats(pretty))

    def test_realism_only_decides_when_the_request_is_equally_met(self):
        worse = self._attempt(missing=[], realism=6.0)
        better = self._attempt(missing=[], realism=8.0)
        self.assertTrue(better.beats(worse))
        self.assertFalse(worse.beats(better))

    def test_incomparable_scales_are_not_compared(self):
        """A checklist share and llava's free-form prompt_accuracy are
        different measurements; deciding between them would be noise."""
        from_checklist = self._attempt(missing=[], realism=7.0)
        from_score = self._attempt(missing=[], realism=8.0, source="score",
                                   acc=40)
        self.assertTrue(from_score.beats(from_checklist))  # decided on realism

    def test_a_deliberately_unphotoreal_request_is_satisfied(self):
        """'a flat cartoon drawing' scores terribly for realism and that is
        the CORRECT result — escalating on it would spend the whole budget
        undoing the request."""
        cartoon = self._attempt(missing=[], realism=2.0)
        self.assertTrue(cartoon.satisfies(self.SETTINGS))

    def test_a_missed_requirement_is_never_satisfied(self):
        self.assertFalse(
            self._attempt(missing=["a red car"], realism=10.0)
            .satisfies(self.SETTINGS))

    def test_an_unjudged_attempt_never_displaces_a_judged_one(self):
        judged = self._attempt(missing=[], realism=7.0)
        unjudged = _Attempt(image=IMG, prompt_id="p", gen=None)
        self.assertFalse(unjudged.beats(judged))


class RoutingTests(unittest.TestCase):
    """Requests whose engine is not the default one."""

    def test_viewpoint_requests_reach_the_viewpoint_engine(self):
        for text in ("show this from 3 angles", "render it from another "
                     "viewpoint", "give me a 360 turntable", "the back view"):
            self.assertTrue(quality.view_intent(text), text)
        for text in ("a wide-angle shot of a kitchen", "a low-angle shot",
                     "a dutch angle portrait"):
            self.assertFalse(quality.view_intent(text), text)
        self.assertEqual(quality.view_count("from 4 different angles"), 4)
        self.assertEqual(quality.view_count("from another angle"), 3)

    def test_lighting_requests_reach_the_relighting_engine(self):
        for text in ("redo the lighting", "relight this photo",
                     "change the light coming from the left",
                     "make it golden hour"):
            self.assertTrue(quality.light_intent(text), text)
        for text in ("a light blue shirt", "a lightweight jacket"):
            self.assertFalse(quality.light_intent(text), text)

    def test_operations_route_to_engines_that_can_perform_them(self):
        """img2img cannot move a light source and cannot move the camera —
        routing these there was a silent no-op."""
        self.assertEqual(quality.OPERATION_TASK["CHANGE_LIGHTING"], "relight")
        self.assertEqual(quality.OPERATION_TASK["CHANGE_CAMERA"], "angles")
        self.assertIn("relight", quality.EDIT_TASKS)
        self.assertIn("angles", quality.EDIT_TASKS)

    def test_a_mislabelled_plan_is_corrected(self):
        """A 7B planner calling 'show it from three angles' a restyle must
        not cost the user the capability."""
        llm = OneJson({"steps": [{"operation": "CHANGE_STYLE",
                                  "instruction": "show it from three angles",
                                  "target": ""}]})
        steps = quality.plan_edit(llm, "show it from three angles",
                                  has_mask=False)
        self.assertEqual([s["task"] for s in steps], ["angles"])

        llm = OneJson({"steps": [{"operation": "CHANGE_STYLE",
                                  "instruction": "redo the lighting",
                                  "target": ""}]})
        steps = quality.plan_edit(llm, "redo the lighting", has_mask=False)
        self.assertEqual([s["task"] for s in steps], ["relight"])

    def test_viewpoints_end_the_chain(self):
        """A set of viewpoints is a new set of pictures, not a further edit —
        anything sequenced after it would silently never run."""
        ordered = quality.order_steps([
            {"task": "angles"}, {"task": "inpaint"}, {"task": "outpaint"}])
        self.assertEqual([s["task"] for s in ordered],
                         ["inpaint", "outpaint", "angles"])


class RelightWiringTests(unittest.TestCase):
    """The relight template is only reachable if every gate is open."""

    def test_the_loader_admits_the_relight_task_and_its_nodes(self):
        self.assertIn("relight", ALLOWED_TASKS)
        self.assertIn("LoadAndApplyICLightUnet", ALLOWED_NODE_TYPES)
        self.assertIn("ICLightConditioning", ALLOWED_NODE_TYPES)

    def test_the_template_is_wired_the_way_ic_light_requires(self):
        from app.adapters.comfyui import WorkflowLibrary
        lib = WorkflowLibrary(Path(__file__).parent.parent / "app" / "workflows")
        t = lib.load_named("relight")
        graph = t["graph"]
        ks = next(n for n in graph.values() if n["class_type"] == "KSampler")
        cond = next(nid for nid, n in graph.items()
                    if n["class_type"] == "ICLightConditioning")
        # The sampler must start from ICLightConditioning's THIRD output (a
        # zero latent) — its own outputs 0/1 are the conditioning pair.
        self.assertEqual(ks["inputs"]["latent_image"], [cond, 2])
        self.assertEqual(ks["inputs"]["positive"], [cond, 0])
        self.assertEqual(ks["inputs"]["negative"], [cond, 1])
        # Subject preservation comes from the foreground latent, not from a
        # low denoise: measured, anything under ~0.8 collapses to flat grey.
        self.assertGreaterEqual(ks["inputs"]["denoise"], 0.85)
        # IC-Light can only patch a 4-channel SD1.5 base.
        self.assertIn("sd15-base", t["required_models"])


class LadderCandidateTests(unittest.TestCase):
    """A rung that cannot produce an image is worse than no rung — it costs
    the same minutes and returns nothing."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))
        self.addCleanup(self.s.stop)

    INSTALLED = ["epicrealism_v10-inpainting.safetensors",
                 "juggernautXL_inpaint.safetensors",
                 "sd_xl_base_1.0.safetensors", "sv3d_u.safetensors"]

    def test_generate_never_escalates_to_a_model_that_cannot_generate(self):
        models, _ = self.s._ladder_candidates(
            "generate", None, "sd_xl_base_1.0.safetensors", self.INSTALLED)
        self.assertEqual(models, [])  # every other installed model is unusable

    def test_inpaint_may_use_the_inpainting_checkpoints(self):
        models, _ = self.s._ladder_candidates(
            "inpaint", None, "sd-v1-5-inpainting.safetensors", self.INSTALLED)
        self.assertIn("epicrealism_v10-inpainting.safetensors", models)

    def test_a_template_that_pins_its_model_offers_no_model_rung(self):
        self.s.registry.is_ready = lambda n: True
        models, _ = self.s._ladder_candidates(
            "generate", "generate_zimage", "x", ["a.safetensors"])
        self.assertEqual(models, [])

    def test_never_escalates_INTO_a_draft_template(self):
        """A draft template is deliberately lower quality: 'try harder' must
        never mean 'try worse'."""
        self.s.registry.is_ready = lambda n: True
        _, workflows = self.s._ladder_candidates(
            "generate", "generate", "x", [])
        self.assertNotIn("generate_draft", workflows)
        self.assertIn("generate_xl", workflows)

    def test_a_draft_template_needs_the_user_to_ask_for_a_draft(self):
        """Seen live: the router picked generate_draft (4 steps, cfg 1) for
        'a sign that clearly reads OPEN' because its own reasoning said
        'fast photorealistic'. A preview is not what was asked for."""
        self.assertTrue(self.s._WANTS_DRAFT.search("a quick draft of a cat"))
        self.assertTrue(self.s._WANTS_DRAFT.search("just show me roughly"))
        self.assertFalse(self.s._WANTS_DRAFT.search(
            "a wooden shop sign that clearly reads OPEN"))

    def test_a_capability_says_plainly_when_it_cannot_run(self):
        """Relighting needs a model that may not be downloaded. The pipeline
        must be able to SAY so and approximate, rather than fail the job or
        pretend it relit anything."""
        ok, why = self.s._template_runnable("relight")
        self.assertFalse(ok)
        self.assertIn("sd15-base", why)
        self.s.registry.is_ready = lambda n: True
        self.assertTrue(self.s._template_runnable("relight")[0])

    def test_a_named_model_is_recognised_but_common_words_are_not(self):
        ckpts = ["juggernautXL_inpaint.safetensors", "portrait.safetensors"]
        self.assertEqual(
            self.s._named_model("use juggernautXL for this", ckpts),
            "juggernautXL_inpaint.safetensors")
        # "portrait" is an ordinary word — it must not pin the model and
        # silently delete the change-model rung.
        self.assertIsNone(
            self.s._named_model("a portrait of a woman", ckpts))


class CameraAngleWordingTests(unittest.TestCase):
    """The imperative phrasing, which was missing and cost a real job.

    Live: "make the girl completely naked and change the angle of the camera"
    planned step 2 as img2img — an engine that repaints a picture and cannot
    move a camera. And because CAPABILITY_WORKFLOW consults the SAME pattern,
    the escalation ladder never reached its change-the-workflow rung: it
    retried with a new seed, then a new checkpoint, scored an honest 0% three
    times, and kept the best of three renders that could not have worked."""

    def test_the_imperative_form_is_a_viewpoint_request(self):
        for text in ("change the angle of the camera", "change the camera angle",
                     "adjust the camera position", "a different camera angle",
                     "move the viewpoint", "change the perspective",
                     "shift the point of view",
                     "make the girl completely naked and change the angle "
                     "of the camera"):
            self.assertTrue(quality.view_intent(text), text)

    def test_a_framing_choice_is_still_not_a_camera_move(self):
        """Escalating "wide-angle shot" to orbital view synthesis would
        answer a question nobody asked."""
        for text in ("a wide-angle shot", "a low-angle shot", "dutch angle",
                     "the camera angle stays the same",
                     "change the angle of the light"):
            self.assertFalse(quality.view_intent(text), text)

    def test_it_does_not_steal_the_neighbouring_routes(self):
        for text in ("make her sit down", "change the pose", "make this 3d",
                     "change the background to a forest"):
            self.assertFalse(quality.view_intent(text), text)

    def test_the_ladder_can_now_reach_the_right_workflow(self):
        self.assertEqual(
            quality.capability_gap(["change the angle of the camera"]),
            "angles")

    def test_the_plan_routes_it_even_when_the_planner_says_style(self):
        llm = OneJson({"steps": [
            {"operation": "REMOVE_OBJECT", "target": "clothing",
             "instruction": "a"},
            {"operation": "CHANGE_STYLE", "target": "",
             "instruction": "change the angle of the camera"}]})
        steps = quality.plan_edit(
            llm, "a and change the angle of the camera", has_mask=False)
        self.assertIn("angles", [s["task"] for s in steps])


class ViewpointTests(unittest.TestCase):
    """Which frames of a 21-frame orbit answer the request."""

    def test_a_turntable_spreads_evenly_around_the_subject(self):
        self.assertEqual(Services._orbit_frames(21, 3, 360.0), [0, 7, 14])

    def test_a_plain_viewpoint_request_stays_near_the_front(self):
        """SV3D invents the back of a subject it never saw; past roughly
        +/-60 degrees a face stops being that face."""
        picks = Services._orbit_frames(21, 3, 120.0)
        self.assertEqual(picks, [19, 0, 2])
        degrees = sorted(((p * 360 / 21) + 180) % 360 - 180 for p in picks)
        self.assertLess(max(abs(d) for d in degrees), 60)

    def test_one_extra_view_is_never_the_view_we_were_given(self):
        """Frame 0 IS the input: returning it reads as a bug, not an answer."""
        self.assertNotIn(0, Services._orbit_frames(21, 1, 120.0))

    def test_the_subject_is_restaged_before_the_orbit(self):
        """SV3D was trained on one object on a plain background; handed a
        photo with its background it rotates the whole frame like a picture
        on a turntable."""
        image = Image.new("RGB", (200, 100), (10, 200, 10))
        mask = Image.new("L", (200, 100), 0)
        mask.paste(255, (80, 20, 120, 60))
        staged = Services._stage_for_orbit(image, mask, 576)
        self.assertEqual(staged.size, (576, 576))       # square, model-native
        self.assertEqual(staged.getpixel((5, 5)), (128, 128, 128))  # neutral bg

    def test_it_survives_having_no_mask(self):
        image = Image.new("RGB", (200, 100), (10, 200, 10))
        self.assertEqual(Services._stage_for_orbit(image, None, 576).size,
                         (576, 576))


class BoostTests(unittest.TestCase):
    def test_guidance_and_steps_rise_but_stay_sane(self):
        graph = {"1": {"class_type": "KSampler",
                       "inputs": {"cfg": 7.0, "steps": 30, "denoise": 0.6}}}
        out = Services._boost_graph(graph, {"cfg": 1.0, "steps": 1.25,
                                            "denoise": 0.05})
        self.assertEqual(out["1"]["inputs"]["cfg"], 8.0)
        self.assertEqual(out["1"]["inputs"]["steps"], 38)
        self.assertEqual(out["1"]["inputs"]["denoise"], 0.65)
        self.assertEqual(graph["1"]["inputs"]["cfg"], 7.0)  # not mutated

    def test_distilled_models_are_left_alone(self):
        """A 4-step cfg-1 model's settings are part of the model. Raising
        them does not try harder, it burns the image."""
        graph = {"1": {"class_type": "KSampler",
                       "inputs": {"cfg": 1.0, "steps": 4, "denoise": 1.0}}}
        out = Services._boost_graph(graph, {"cfg": 1.0, "steps": 1.25,
                                            "denoise": 0.05})
        self.assertEqual(out["1"]["inputs"], graph["1"]["inputs"])


if __name__ == "__main__":
    unittest.main()
