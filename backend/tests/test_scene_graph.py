"""Tests: the Image Understanding Engine (persistent scene graph) and the
atomic-operation compiler. All offline."""
import json
import unittest

from PIL import Image

from app.core import quality, scene_graph
from tests.test_quality import OneJson
from tests.test_workflow_job import DeadLLM

SCENE = {
    "scene": "A red car parked on a street at dusk",
    "setting": "street", "lighting": "warm low sun from the right",
    "perspective": "eye-level", "has_person": False,
    "objects": [
        {"name": "car", "location": "center-right", "size": "large"},
        {"name": "street", "location": "bottom-center", "size": "large"},
    ],
}


class FakeCritic:
    def __init__(self, payload):
        self.payload = payload

    def ask(self, image, question):
        return json.dumps(self.payload) if isinstance(self.payload, dict) \
            else self.payload


class SceneGraphTests(unittest.TestCase):
    def test_build_parses_objects_and_maps_cells(self):
        g = scene_graph.build(Image.new("RGB", (64, 64)), FakeCritic(SCENE))
        self.assertEqual(g["setting"], "street")
        self.assertTrue(g["lighting"].startswith("warm low sun"))
        names = [o["name"] for o in g["objects"]]
        self.assertEqual(names, ["car", "street"])
        car = g["objects"][0]
        self.assertEqual(car["location"], "center-right")
        self.assertEqual(car["cell"], 6)          # center-right → 6
        self.assertTrue(g["palette"])              # deterministic colours

    def test_build_without_critic_is_minimal_but_has_palette(self):
        g = scene_graph.build(Image.new("RGB", (32, 32), (200, 30, 30)), None)
        self.assertEqual(g["objects"], [])
        self.assertEqual(g["scene"], "")
        self.assertTrue(g["palette"])

    def test_build_survives_a_broken_vision_model(self):
        class Boom:
            def ask(self, image, question):
                raise OSError("vision down")

        g = scene_graph.build(Image.new("RGB", (16, 16)), Boom())
        self.assertEqual(g["objects"], [])

    def test_build_tolerates_garbage_json(self):
        g = scene_graph.build(Image.new("RGB", (16, 16)),
                              FakeCritic("not json at all"))
        self.assertEqual(g["objects"], [])

    def test_summary_and_find_object_and_placement_context(self):
        g = scene_graph.build(Image.new("RGB", (64, 64)), FakeCritic(SCENE))
        s = scene_graph.summary(g)
        self.assertIn("red car", s)
        self.assertIn("lighting", s)
        # target matching (exact + fuzzy)
        self.assertEqual(scene_graph.find_object(g, "car")["name"], "car")
        self.assertEqual(scene_graph.find_object(g, "the red car")["name"],
                         "car")
        self.assertIsNone(scene_graph.find_object(g, "helicopter"))
        ctx = scene_graph.placement_context(g)
        self.assertIn("camera", ctx)
        self.assertIn("lighting", ctx)

    def test_summary_none_when_empty(self):
        self.assertIsNone(scene_graph.summary(scene_graph.build(
            Image.new("RGB", (8, 8)), None)))


class OperationCompilerTests(unittest.TestCase):
    def _plan(self, payload, prompt="x", has_mask=False):
        return quality.plan_edit(OneJson(payload), prompt, has_mask)

    def test_operation_maps_to_task(self):
        # Each operation is paired with a request that genuinely ASKS for it.
        # A bare "x" used to be enough, but plan_edit now prunes steps the
        # request never asked for, and an OUTPAINT with no canvas wording in
        # the prompt is exactly such a step — the plan came back empty and
        # this test read None[0]. The prompt matters now, so it is spelled
        # out per operation rather than shared.
        cases = {
            "ADD_OBJECT": ("inpaint", "add a hat to the car"),
            "REMOVE_OBJECT": ("inpaint", "remove the car"),
            "REPLACE_OBJECT": ("inpaint", "replace the car with a van"),
            "CHANGE_ATTRIBUTE": ("inpaint", "make the car red"),
            "CHANGE_STYLE": ("img2img", "make it look like a painting"),
            # Lighting and viewpoint have engines of their own: img2img
            # repaints a picture, it cannot move a light source or a camera.
            "CHANGE_LIGHTING": ("relight", "light it like a sunset"),
            "CHANGE_CAMERA": ("angles", "show the car from another angle"),
            "MULTI_VIEW": ("angles", "show the car from several angles"),
            "OUTPAINT": ("outpaint", "extend this into a wider format"),
            "UPSCALE": ("upscale", "upscale this to more detail"),
            "ANIMATE": ("video", "animate the car driving away"),
        }
        for op, (task, prompt) in cases.items():
            steps = self._plan({"steps": [
                {"operation": op, "target": "car",
                 "instruction": prompt}]}, prompt=prompt)
            self.assertIsNotNone(steps, op)
            self.assertEqual(steps[0]["task"], task, op)
            self.assertEqual(steps[0]["operation"], op)
            self.assertEqual(steps[0]["target"], "car")

    def test_a_plan_of_nothing_but_invented_steps_is_no_plan(self):
        """Pruning may empty a plan, and plan_edit then returns None.

        The caller treats that as "the planner gave me nothing usable" and
        falls back to a single default step, which is right — but the drops
        must still be reported, so the reason is passed back through
        `dropped` rather than lost."""
        dropped: list[dict] = []
        steps = quality.plan_edit(
            OneJson({"steps": [{"operation": "OUTPAINT", "target": "",
                                "instruction": "widen the canvas"}]}),
            "make her hair red", False, dropped=dropped)
        self.assertIsNone(steps)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["task"], "outpaint")
        self.assertIn("canvas", dropped[0]["why"])

    def test_legacy_task_only_plans_still_work(self):
        steps = self._plan({"task": "img2img", "instruction": "make it winter"})
        self.assertEqual(steps[0]["task"], "img2img")
        self.assertEqual(steps[0]["operation"], "CHANGE_STYLE")  # inferred

    def test_add_object_via_img2img_is_coerced_to_inpaint(self):
        steps = self._plan({"steps": [
            {"operation": "CHANGE_STYLE", "instruction": "add a dog"}]})
        self.assertEqual(steps[0]["task"], "inpaint")

    def test_user_mask_forces_first_step_inpaint(self):
        steps = self._plan({"steps": [
            {"operation": "CHANGE_STYLE", "instruction": "make it winter"}]},
            has_mask=True)
        self.assertEqual(steps[0]["task"], "inpaint")

    def test_dead_llm_returns_none(self):
        self.assertIsNone(quality.plan_edit(DeadLLM(), "x", False))


if __name__ == "__main__":
    unittest.main()
