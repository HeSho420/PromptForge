"""Changing the background, and nothing else.

The promise here is unusually strong: not "the subject should survive the
repaint" but "the subject cannot change". The graph inverts an exact BiRefNet
subject matte to decide what gets repainted, then composites the ORIGINAL
subject pixels back through a shrunk copy of that same matte — so the
subject's interior is literally the input pixels.

These tests pin the routing (an ordinary inpaint would ask SAM for a
"background" mask, and SAM is a part segmenter), the graph wiring that makes
the guarantee true, and the honest refusal when the matting engine is absent.
"""
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.adapters.comfyui import ALLOWED_NODE_TYPES, ALLOWED_TASKS, WorkflowLibrary, build_workflow
from app.config import Settings
from app.core import quality
from app.core.services import PermanentError, Services

WORKFLOWS = Path(__file__).parent.parent / "app" / "workflows"


class _StubJob:
    def __init__(self):
        self.logs = []

    def log(self, _level, msg):
        self.logs.append(msg)


class OneJson:
    """An LLM that always returns the same plan."""
    source = "local"

    def __init__(self, payload):
        self.payload = payload

    def complete(self, system, prompt, max_tokens=4096):
        from app.core.llm import LLMReply
        return LLMReply(json.dumps(self.payload), "fake", "local")


class IntentTests(unittest.TestCase):
    """Which sentences mean 'replace what is behind the subject'."""

    REPLACE = [
        "change the background to a snowy forest",
        "replace the background with a beach at sunset",
        "swap the backdrop for a city skyline",
        "put a different background behind her",
        "make the background a plain white studio",
        "change background to space",
        "set the backdrop to a library",
        "I want a new background",
        "the background should be a mountain range",
        "update the background please",
        "shoot her with a snowy mountain background",
    ]
    # Mentions the background but must NOT be routed to a repaint.
    LEAVE_ALONE = [
        "blur the background",
        "add background blur",
        "darken the background a little",
        # A cutout is a different job — repainting is not what was asked.
        "remove the background",
        "make the background transparent",
        "background removal",
        # Motion transfer says this constantly; mistaking it for a request to
        # REPLACE the background would invert the user's intent exactly.
        "keep the background",
        "preserve the background exactly",
        "keep the clip background unchanged",
        "leave the original background",
        # No background request at all.
        "give her a red dress",
        "swap the faces",
        "make it look like a painting",
        "add a hat",
        "a portrait with a blurred forest background",
        "the background is fine, change her shirt",
    ]

    def test_replacement_requests_are_recognised(self):
        for text in self.REPLACE:
            with self.subTest(text=text):
                self.assertTrue(quality.background_intent(text))

    def test_other_background_talk_is_left_alone(self):
        for text in self.LEAVE_ALONE:
            with self.subTest(text=text):
                self.assertFalse(quality.background_intent(text))


class RoutingTests(unittest.TestCase):
    def test_the_operation_maps_to_its_own_engine(self):
        self.assertEqual(quality.OPERATION_TASK["REPLACE_BACKGROUND"],
                         "background")
        self.assertIn("background", quality.EDIT_TASKS)
        self.assertIn("background", ALLOWED_TASKS)

    def test_background_runs_after_compose_and_before_outpaint(self):
        """Order matters: repainting everything around the subject before a
        subject has been placed would repaint over the wrong picture, and
        doing it after an outpaint would waste the extension."""
        steps = [{"task": "outpaint"}, {"task": "background"},
                 {"task": "compose"}]
        self.assertEqual([s["task"] for s in quality.order_steps(steps)],
                         ["compose", "background", "outpaint"])

    # What a small planner actually returns for a background request: the
    # right intent, routed to the engine that cannot deliver it.
    MISLABELED = {"steps": [{
        "task": "inpaint", "operation": "REPLACE_OBJECT",
        "target": "background", "instruction": "a snowy forest",
        "mask_adjust": "keep", "adjust_px": 0, "denoise": 0.6,
        "reason": "replace the backdrop"}]}

    def test_a_mislabeled_plan_is_coerced(self):
        """REPLACE_OBJECT(background) sends this to SAM for the one mask
        measured wrong on this machine — coerce it deterministically."""
        steps = quality.plan_edit(
            OneJson(self.MISLABELED),
            "change the background to a snowy forest", has_mask=False)
        self.assertEqual(steps[0]["task"], "background")
        self.assertEqual(steps[0]["operation"], "REPLACE_BACKGROUND")

    def test_a_drawn_mask_still_wins(self):
        """If the user painted a region, they said where — obey that rather
        than a whole-background matte."""
        steps = quality.plan_edit(
            OneJson(self.MISLABELED),
            "change the background to a snowy forest", has_mask=True)
        self.assertEqual(steps[0]["task"], "inpaint")

    def test_one_step_carrying_two_requests_becomes_two_workflows(self):
        """The reported bug. Small planners answer "change the clothing and
        change the background to a tropical resort" with a SINGLE step;
        coercing that step to the background engine delivered the resort and
        silently dropped the clothes. Two requests, one workflow, and no sign
        that half of it had been thrown away."""
        prompt = ("change the clothing and change the background to a "
                  "tropical resort")
        steps = quality.plan_edit(
            OneJson({"steps": [{"operation": "CHANGE_ATTRIBUTE",
                                "target": "clothing",
                                "instruction": prompt}]}),
            prompt, has_mask=False)
        self.assertEqual([s["task"] for s in steps], ["inpaint", "background"])
        self.assertNotIn("background", steps[0]["instruction"])
        self.assertIn("tropical resort", steps[1]["instruction"])

    def test_an_already_split_plan_is_left_alone(self):
        prompt = ("change the clothing and change the background to a "
                  "tropical resort")
        steps = quality.plan_edit(
            OneJson({"steps": [
                {"operation": "CHANGE_ATTRIBUTE", "target": "clothing",
                 "instruction": "change the clothing"},
                {"operation": "REPLACE_BACKGROUND", "target": "",
                 "instruction": "change the background to a tropical "
                                "resort"}]}),
            prompt, has_mask=False)
        self.assertEqual([s["task"] for s in steps], ["inpaint", "background"])

    def test_a_single_request_is_never_torn_in_half(self):
        """"and" inside one description is not a second workflow. Every
        leftover clause has to read as an instruction of its own."""
        for text in ("change the background to a red and blue mural",
                     "change the background to a tropical resort",
                     "put her in a black and white striped dress"):
            self.assertIsNone(
                quality.split_capability_clause(text,
                                                quality.background_intent),
                text)

    def test_the_split_survives_the_invented_step_pruner(self):
        """A step this adds is a REAL request, not planner padding — the
        pruner must not treat it as one."""
        prompt = "remove the trash can and change the background to a forest"
        steps = quality.plan_edit(
            OneJson({"steps": [{"operation": "REMOVE_OBJECT",
                                "target": "trash can",
                                "instruction": prompt}]}),
            prompt, has_mask=False)
        self.assertIn("background", [s["task"] for s in steps])
        self.assertIn("inpaint", [s["task"] for s in steps])

    def test_an_unmet_background_requirement_escalates_here(self):
        self.assertEqual(quality.capability_gap(["a snowy forest background"]),
                         "background")

    def test_a_lighting_miss_still_goes_to_relight(self):
        """'sunlit background' is a lighting miss, not a backdrop miss."""
        self.assertEqual(
            quality.capability_gap(["warm sunlight in the background"]),
            "relight")


class TemplateTests(unittest.TestCase):
    """The wiring that makes 'and nothing else' true."""

    def setUp(self):
        self.template = WorkflowLibrary(WORKFLOWS).load("background")
        self.graph = build_workflow(self.template, {
            "image": "src.png", "prompt": "a snowy forest",
            "seed": 7, "rmbg_model": "BiRefNet-portrait"})

    def test_the_task_and_nodes_are_allowed(self):
        self.assertEqual(self.template["task"], "background")
        for node in self.graph.values():
            self.assertIn(node["class_type"], ALLOWED_NODE_TYPES)

    def test_the_repainted_region_is_the_inverted_subject_matte(self):
        """Slot 1 of BiRefNetRMBG is the subject MASK; the inpaint must be
        fed the INVERSION of it, never the matte itself."""
        matte_node = next(n for n, d in self.graph.items()
                          if d["class_type"] == "BiRefNetRMBG")
        invert = next(n for n, d in self.graph.items()
                      if d["class_type"] == "InvertMask")
        self.assertEqual(self.graph[invert]["inputs"]["mask"],
                         [matte_node, 1])
        # …and the mask reaching the sampler descends from that inversion.
        cond = next(n for n, d in self.graph.items()
                    if d["class_type"] == "InpaintModelConditioning")
        self.assertIn(invert, self._mask_chain(cond),
                      "the sampled mask must descend from InvertMask")

    def _mask_chain(self, start: str) -> list[str]:
        """Every node the mask input of `start` traces back through."""
        chain, node, seen = [], start, set()
        while node not in seen:
            seen.add(node)
            ins = self.graph[node]["inputs"]
            if "mask" not in ins or not isinstance(ins["mask"], list):
                break
            node = ins["mask"][0]
            chain.append(node)
        return chain

    def test_the_original_subject_is_composited_back_over_the_result(self):
        """This is the guarantee. The saved image must be the composite, and
        the composite's SOURCE must be the untouched input image."""
        comp = next(n for n, d in self.graph.items()
                    if d["class_type"] == "ImageCompositeMasked")
        load = next(n for n, d in self.graph.items()
                    if d["class_type"] == "LoadImage")
        decode = next(n for n, d in self.graph.items()
                      if d["class_type"] == "VAEDecode")
        ins = self.graph[comp]["inputs"]
        self.assertEqual(ins["source"], [load, 0],
                         "the subject must come from the ORIGINAL image")
        self.assertEqual(ins["destination"], [decode, 0],
                         "the repaint is what gets covered, not the reverse")
        save = next(n for n, d in self.graph.items()
                    if d["class_type"] == "SaveImage")
        self.assertEqual(self.graph[save]["inputs"]["images"], [comp, 0],
                         "saving the raw decode would drop the guarantee")

    def test_the_reapplied_matte_is_shrunk_not_grown(self):
        """A composite through the full matte re-pastes a hard edge; the
        subject copy is eroded a couple of pixels so the rim blends."""
        comp = next(n for n, d in self.graph.items()
                    if d["class_type"] == "ImageCompositeMasked")
        expands = [self.graph[n]["inputs"]["expand"]
                   for n in self._mask_chain(comp)
                   if self.graph[n]["class_type"] == "GrowMask"]
        self.assertTrue(expands and all(e < 0 for e in expands),
                        f"expected an eroding GrowMask, got {expands}")

    def test_no_feathermask_anywhere(self):
        """Measured live, twice. FeatherMask tapers all four IMAGE borders,
        not just the subject outline. On the background mask that left the
        outermost pixels unrepainted — a visible frame around the new scene
        (SD then leaned into it and painted a framed poster on a wall). On
        the subject mask it blended generated pixels into the subject along
        the bottom edge she stood on: 215 px at y=767. Edge softening comes
        from BiRefNet's own mask_blur, which only softens the matte."""
        kinds = [d["class_type"] for d in self.graph.values()]
        self.assertNotIn("FeatherMask", kinds)
        node = next(d for d in self.graph.values()
                    if d["class_type"] == "BiRefNetRMBG")
        self.assertGreater(node["inputs"]["mask_blur"], 0,
                           "with no FeatherMask, mask_blur is the only thing "
                           "softening the subject edge")

    def test_every_birefnet_optional_is_passed(self):
        """The node reads its optional inputs unconditionally and raises
        KeyError when one is missing — hit live once already in compose."""
        node = next(d for d in self.graph.values()
                    if d["class_type"] == "BiRefNetRMBG")
        for key in ("sensitivity", "mask_blur", "mask_offset", "invert_output",
                    "refine_foreground", "background", "background_color"):
            self.assertIn(key, node["inputs"])
        self.assertFalse(node["inputs"]["invert_output"],
                         "inverting here would swap what gets repainted")


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))
        self.addCleanup(self.s.stop)

    def test_people_get_the_portrait_matte(self):
        self.assertEqual(self.s._matte_model("a woman on a beach"),
                         Services._MATTE_PORTRAIT)
        self.assertEqual(self.s._matte_model("put her in a forest"),
                         Services._MATTE_PORTRAIT)

    def test_objects_get_the_general_matte(self):
        self.assertEqual(self.s._matte_model("a red sports car"),
                         Services._MATTE_GENERAL)
        self.assertEqual(self.s._matte_model(""), Services._MATTE_GENERAL)

    def test_it_refuses_honestly_without_the_matting_pack(self):
        """Degrading to SAM would repaint the subject — say so instead."""
        self.s._pack_active = lambda _name: False
        self.s._require_comfy = lambda _job: None
        with self.assertRaises(PermanentError) as caught:
            self.s._render_background_step(
                _StubJob(), Image.new("RGB", (64, 64)), "a forest", "")
        msg = str(caught.exception).lower()
        self.assertIn("rmbg", msg)
        self.assertIn("matte", msg)


if __name__ == "__main__":
    unittest.main()
