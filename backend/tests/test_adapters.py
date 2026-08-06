import unittest

from PIL import Image

from app.adapters.base import BadMaskError, validate_mask
from app.adapters.comfyui import (
    WorkflowLibrary,
    WorkflowNotAllowedError,
    WorkflowValidationError,
    build_workflow,
    validate_workflow,
)
from app.adapters.mock import MockInpaintingAdapter, MockSegmentationAdapter
from app.config import Settings


def _image(color=(120, 90, 60), size=(64, 48)) -> Image.Image:
    return Image.new("RGB", size, color)


class MockAdapterTests(unittest.TestCase):
    def test_mock_adapters_are_labeled(self):
        self.assertTrue(MockSegmentationAdapter.is_mock)
        self.assertTrue(MockInpaintingAdapter.is_mock)

    def test_segmentation_returns_matching_mask(self):
        img = _image()
        mask = MockSegmentationAdapter().propose_mask(img, "remove the chair")
        self.assertEqual(mask.size, img.size)
        self.assertEqual(mask.mode, "L")
        self.assertNotEqual(mask.getextrema(), (0, 0))  # not empty

    def test_sky_prompt_masks_top_region(self):
        img = _image(size=(100, 100))
        mask = MockSegmentationAdapter().propose_mask(img, "change the sky to sunset")
        top = mask.crop((0, 0, 100, 20)).getextrema()[1]
        bottom = mask.crop((0, 80, 100, 100)).getextrema()[1]
        self.assertGreater(top, 200)
        self.assertLess(bottom, 50)

    def test_inpaint_changes_only_masked_pixels(self):
        img = _image(size=(80, 80))
        mask = Image.new("L", img.size, 0)
        for x in range(40, 80):
            for y in range(40, 80):
                mask.putpixel((x, y), 255)
        result = MockInpaintingAdapter().inpaint(img, mask, "change to sunset")
        self.assertTrue(result.is_mock)
        self.assertIn("MOCK", result.meta["note"])
        out = result.image
        self.assertNotEqual(out.getpixel((60, 60)), img.getpixel((60, 60)))  # masked: changed
        self.assertEqual(out.getpixel((5, 5)), img.getpixel((5, 5)))          # unmasked: intact

    def test_empty_mask_rejected(self):
        img = _image()
        with self.assertRaises(BadMaskError):
            MockInpaintingAdapter().inpaint(img, Image.new("L", img.size, 0), "x")

    def test_size_mismatch_rejected(self):
        with self.assertRaises(BadMaskError):
            validate_mask(_image(size=(64, 48)), Image.new("L", (10, 10), 255))


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.lib = WorkflowLibrary(Settings().workflows_dir)

    def test_load_latest_inpaint_template(self):
        t = self.lib.load("inpaint")
        self.assertEqual(t["template"], "inpaint")
        self.assertGreaterEqual(t["version"], 1)
        self.assertIn("graph", t)

    def test_disallowed_task_rejected(self):
        with self.assertRaises(WorkflowNotAllowedError):
            self.lib.load("txt2video_person_swap")

    def test_missing_version_rejected(self):
        with self.assertRaises(WorkflowValidationError):
            self.lib.load("inpaint", version=999)

    def test_build_workflow_injects_parameters(self):
        t = self.lib.load("inpaint")
        graph = build_workflow(t, {"prompt": "a studio wall", "seed": 7})
        texts = [n["inputs"].get("text") for n in graph.values()
                 if n["class_type"] == "CLIPTextEncode"]
        self.assertIn("a studio wall", texts)
        sampler = next(n for n in graph.values()
                       if n["class_type"] == "KSampler")
        self.assertEqual(sampler["inputs"]["seed"], 7)

    def test_build_workflow_rejects_unknown_parameter(self):
        t = self.lib.load("inpaint")
        with self.assertRaises(WorkflowValidationError):
            build_workflow(t, {"rm_rf": "/"})

    def test_validation_rejects_disallowed_node_type(self):
        with self.assertRaises(WorkflowValidationError):
            validate_workflow({"1": {"class_type": "ExecutePythonCode", "inputs": {}}})

    def test_validation_rejects_dangling_link(self):
        graph = {"1": {"class_type": "SaveImage",
                       "inputs": {"images": ["99", 0], "filename_prefix": "x"}}}
        with self.assertRaises(WorkflowValidationError):
            validate_workflow(graph)

    def test_validation_rejects_empty_graph(self):
        with self.assertRaises(WorkflowValidationError):
            validate_workflow({})


if __name__ == "__main__":
    unittest.main()
