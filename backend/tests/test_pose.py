"""Moving a body, which is not the same as repainting one.

This route was deliberately UN-ROUTED for a long time: pose_v1 was a txt2img
graph, so it generated a fresh picture guided by a skeleton and discarded the
photograph being edited. Routing to it made every request fail permanently.
These tests pin the three things that make pose_v2 shippable — the request
reaches the engine, the photograph survives, and a face is only pasted back
when a rigid paste is actually valid.
"""
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from app.adapters.comfyui import ALLOWED_NODE_TYPES, ALLOWED_TASKS, WorkflowLibrary, build_workflow
from app.config import Settings
from app.core import quality
from app.core.services import Services

WORKFLOWS = Path(__file__).parent.parent / "app" / "workflows"


class _StubJob:
    def __init__(self):
        self.messages = []

    def log(self, _level, message):
        self.messages.append(message)


class IntentTests(unittest.TestCase):
    def test_a_body_being_moved_is_detected(self):
        for text in ("make her sit down", "change the pose",
                     "put him in a different pose", "use this reference pose",
                     "same pose as the second photo", "make her kneel",
                     "change her posture", "standing instead",
                     "copy the pose", "in a relaxed pose",
                     "adjust her stance", "make them raise their arms"):
            self.assertTrue(quality.pose_intent(text), text)

    def test_something_else_with_a_position_is_not_a_pose(self):
        """"Position" is the loose word: a sun, a camera and a logo all have
        one and none of them has a pose."""
        for text in ("change the position of the sun",
                     "reposition the camera", "move the logo",
                     "change the background to a forest",
                     "change her shirt to red", "brighten the image"):
            self.assertFalse(quality.pose_intent(text), text)

    def test_a_pose_is_not_a_camera_move_and_not_a_3d_scene(self):
        """Three routes that all sound like 'change how it looks' and are
        completely different engines."""
        self.assertTrue(quality.pose_intent("make her sit down"))
        self.assertFalse(quality.view_intent("make her sit down"))
        self.assertFalse(quality.scene3d_intent("make her sit down"))
        self.assertFalse(quality.pose_intent("show it from another angle"))
        self.assertFalse(quality.pose_intent("make this 3d"))


class RoutingTests(unittest.TestCase):
    def test_the_operation_reaches_the_engine(self):
        self.assertEqual(quality.OPERATION_TASK["CHANGE_POSE"], "pose")
        self.assertIn("pose", quality.EDIT_TASKS)
        self.assertIn("pose", ALLOWED_TASKS)

    def test_a_mislabelled_plan_is_coerced(self):
        """Small planners label this worst — "make her sit down" comes back
        as CHANGE_ATTRIBUTE, which repaints the photo and leaves her
        standing."""
        steps = [{"task": "img2img", "operation": "CHANGE_ATTRIBUTE",
                  "target": "", "instruction": "make her sit down",
                  "mask_adjust": "keep", "adjust_px": 0, "denoise": 0.5,
                  "reason": ""}]
        quality._coerce_matching(steps, quality.pose_intent, "pose",
                                 "CHANGE_POSE", ("inpaint", "img2img",
                                                 "custom"))
        self.assertEqual(steps[0]["task"], "pose")
        self.assertEqual(steps[0]["operation"], "CHANGE_POSE")


class TemplateTests(unittest.TestCase):
    def setUp(self):
        self.template = WorkflowLibrary(WORKFLOWS).load("pose")

    def test_v2_is_what_loads(self):
        self.assertEqual(self.template["version"], 2)

    def test_it_repaints_the_photo_instead_of_generating_a_new_one(self):
        """The whole reason v1 could not be routed: EmptyLatentImage means
        the photograph being edited is never an input at all."""
        kinds = {n["class_type"] for n in self.template["graph"].values()}
        self.assertNotIn("EmptyLatentImage", kinds)
        self.assertIn("InpaintModelConditioning", kinds)
        self.assertIn("LoadImage", kinds)

    def test_every_node_is_allowed(self):
        graph = build_workflow(self.template, {
            "image": "a.png", "mask": "m.png", "prompt": "p",
            "negative": "n", "seed": 1, "pose_reference": "r.png"})
        for node in graph.values():
            self.assertIn(node["class_type"], ALLOWED_NODE_TYPES)

    def test_the_pose_detector_is_configured_for_this_machine(self):
        """Two settings that are not defaults and both matter: the union
        ControlNet is xinsir's, which wants the rescaled stick figure; and
        onnxruntime here has no CUDA provider, so the .onnx detector runs on
        CPU at over ten minutes."""
        dw = next(n for n in self.template["graph"].values()
                  if n["class_type"] == "DWPreprocessor")
        self.assertEqual(dw["inputs"]["scale_stick_for_xinsr_cn"], "enable")
        self.assertTrue(dw["inputs"]["bbox_detector"].endswith(".torchscript.pt"))

    def test_without_a_reference_the_control_branch_is_removed_cleanly(self):
        graph = build_workflow(self.template, {
            "image": "a.png", "mask": "m.png", "prompt": "p",
            "negative": "n", "seed": 1, "pose_reference": "r.png"})
        pruned = Services._prune_pose_control(graph)
        kinds = {n["class_type"] for n in pruned.values()}
        self.assertNotIn("ControlNetApplyAdvanced", kinds)
        self.assertNotIn("DWPreprocessor", kinds)
        # the sampler must be rewired, not left pointing at a deleted node
        sampler = pruned["8"]["inputs"]
        self.assertEqual(sampler["positive"], ["6", 0])
        self.assertEqual(sampler["negative"], ["6", 1])
        for node, spec in pruned.items():
            for key, value in spec["inputs"].items():
                if isinstance(value, list) and len(value) == 2:
                    self.assertIn(str(value[0]), pruned, f"{node}.{key}")


class RegionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))
        self.addCleanup(self.s.stop)

    def test_the_repaint_area_is_bigger_than_the_body(self):
        """Masked to the CURRENT silhouette, "sit down" can only redraw a
        standing-shaped region and comes back standing."""
        photo = Image.new("RGB", (400, 600))
        mask = Image.new("L", (400, 600), 0)
        ImageDraw.Draw(mask).rectangle([160, 200, 240, 500], fill=255)
        region = self.s._pose_region(photo, mask)
        box = region.point(lambda v: 255 if v > 127 else 0).getbbox()
        self.assertLess(box[0], 160)
        self.assertLess(box[1], 200)
        self.assertGreater(box[2], 240)
        self.assertGreater(box[3], 500)

    def test_it_never_leaves_the_frame(self):
        photo = Image.new("RGB", (400, 600))
        mask = Image.new("L", (400, 600), 0)
        ImageDraw.Draw(mask).rectangle([0, 0, 399, 599], fill=255)
        region = self.s._pose_region(photo, mask)
        self.assertEqual(region.size, photo.size)

    def test_with_no_matte_the_whole_frame_is_fair_game(self):
        photo = Image.new("RGB", (400, 600))
        region = self.s._pose_region(photo, None)
        self.assertGreater(self.s._mask_fraction(region), 0.5)


class FaceRestoreTests(unittest.TestCase):
    """The original face goes back on, but only when a rigid paste is valid.

    The compositor scales the source face to the target box and centres it.
    Measured: a 569 px face squeezed onto a 212 px one produced a smeared
    mask over the head — visibly worse than the face the model drew."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))
        self.addCleanup(self.s.stop)
        self.posed = Image.new("RGB", (400, 600), (10, 20, 30))
        self.original = Image.new("RGB", (400, 600), (30, 20, 10))
        self.s._render_faceswap_step = lambda *a, **k: "PASTED"

    def _boxes(self, src, dst):
        mask = Image.new("L", (400, 600), 255)
        self.s._face_region = lambda image, job: (
            (mask, src) if image is self.original else (mask, dst))

    def test_a_comparable_face_is_restored(self):
        self._boxes((100, 100, 200, 240), (110, 120, 214, 266))
        job = _StubJob()
        self.assertEqual(
            self.s._restore_face(job, self.posed, self.original, "p", "n"),
            "PASTED")

    def test_a_much_smaller_face_is_left_alone(self):
        self._boxes((100, 100, 200, 240), (100, 100, 140, 152))
        job = _StubJob()
        out = self.s._restore_face(job, self.posed, self.original, "p", "n")
        self.assertIs(out, self.posed)
        self.assertTrue(any("changed size too much" in m
                            for m in job.messages), job.messages)

    def test_a_turned_head_is_left_alone(self):
        self._boxes((100, 100, 240, 240), (100, 100, 140, 240))
        job = _StubJob()
        out = self.s._restore_face(job, self.posed, self.original, "p", "n")
        self.assertIs(out, self.posed)
        self.assertTrue(any("turned too far" in m for m in job.messages),
                        job.messages)

    def test_no_face_anywhere_is_reported_not_crashed(self):
        self.s._face_region = lambda image, job: None
        job = _StubJob()
        out = self.s._restore_face(job, self.posed, self.original, "p", "n")
        self.assertIs(out, self.posed)
        self.assertTrue(any("could not be located" in m
                            for m in job.messages), job.messages)


class DrawnMaskTests(unittest.TestCase):
    """A hand-drawn region pins step 1 to a regional inpaint — but only where
    that is a coherent instruction.

    The rule used to be written as a list of EXCEPTIONS ("not video, not
    angles"), and the list did not grow when the engines did. A mask left
    over from a previous edit would therefore have rewritten a background
    swap back into the inpaint whose mask was measurably wrong — the exact
    failure that engine exists to fix — turned a repose into the repaint it
    is built to avoid, and asked the 3D scene builder to edit a rectangle."""

    def test_painting_a_region_still_means_edit_here(self):
        """The default is deliberately to OBEY the drawn region — including
        for a background swap, where painting a region says "not the whole
        backdrop, this bit". Only the engines that cannot use one opt out."""
        for task in ("background", "img2img", "inpaint", "custom",
                     "outpaint", "relight", "upscale"):
            self.assertNotIn(task, quality.UNMASKABLE_TASKS, task)

    def test_the_engines_that_cannot_use_one_opt_out(self):
        """video/angles/scene3d work on the whole frame; compose and faceswap
        already CONSUME the mask as a placement region, so rewriting them to
        inpaint destroys the operation; and for a repose the drawn region is
        where the body IS, which is the one place it is not going."""
        for task in ("video", "angles", "scene3d", "compose", "faceswap",
                     "motion_transfer", "pose"):
            self.assertIn(task, quality.UNMASKABLE_TASKS, task)

    def test_a_leftover_mask_cannot_hijack_a_pose_request(self):
        """The reported symptom: the mask outlived the step it belonged to."""
        llm = _PlanStub({"steps": [{"operation": "CHANGE_POSE", "target": "",
                                    "instruction": "make her sit down"}]})
        steps = quality.plan_edit(llm, "make her sit down", has_mask=True)
        self.assertEqual(steps[0]["task"], "pose")

    def test_a_leftover_mask_cannot_hijack_a_3d_scene(self):
        llm = _PlanStub({"steps": [{"operation": "CHANGE_STYLE", "target": "",
                                    "instruction": "make this 3d"}]})
        steps = quality.plan_edit(llm, "make this 3d", has_mask=True)
        self.assertIn("scene3d", [s["task"] for s in steps])


class _PlanStub:
    source = "local"

    def __init__(self, payload):
        self.payload = payload

    def complete(self, system, prompt, max_tokens=4096):
        import json

        from app.core.llm import LLMReply
        return LLMReply(json.dumps(self.payload), "fake", "local")


if __name__ == "__main__":
    unittest.main()
