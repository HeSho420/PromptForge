"""Tests: live /object_info schema validation of LLM-generated workflows
(unknown inputs, missing required, wrong dropdown values) and the scene
context helper. All offline."""
import json
import unittest

from PIL import Image

from app.core import quality
from app.core.workflow_ai import (
    WorkflowGenerator,
    live_schema_errors,
)
from tests.test_workflow_job import ScriptedLLM

# A minimal live-schema stand-in: legacy COMBO for ckpt, wrapped COMBO for
# sampler — both shapes must be understood.
INFO = {
    "CheckpointLoaderSimple": {"input": {"required": {
        "ckpt_name": [["real.safetensors", "other.safetensors"], {}]}}},
    "KSampler": {"input": {"required": {
        "model": ["MODEL", {}],
        "seed": ["INT", {}],
        "steps": ["INT", {}],
        "sampler_name": [["euler", "dpmpp_2m"], {}],
        "scheduler": ["COMBO", {"options": ["normal", "karras"]}],
        "positive": ["CONDITIONING", {}],
        "negative": ["CONDITIONING", {}],
        "latent_image": ["LATENT", {}],
        "cfg": ["FLOAT", {}],
        "denoise": ["FLOAT", {}],
    }}},
    "SaveImage": {"input": {"required": {
        "images": ["IMAGE", {}], "filename_prefix": ["STRING", {}]}}},
    "CLIPTextEncode": {"input": {"required": {
        "text": ["STRING", {}], "clip": ["CLIP", {}]}}},
    "EmptyLatentImage": {"input": {"required": {
        "width": ["INT", {}], "height": ["INT", {}],
        "batch_size": ["INT", {}]}}},
    "VAEDecode": {"input": {"required": {
        "samples": ["LATENT", {}], "vae": ["VAE", {}]}}},
}


def _good_graph(ckpt="real.safetensors", sampler="euler", scheduler="normal",
                extra_input=None, drop=None):
    g = {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "a cat", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "blurry", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "5": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["2", 0],
                         "negative": ["3", 0], "latent_image": ["4", 0],
                         "seed": 1, "steps": 20, "cfg": 7.0,
                         "sampler_name": sampler, "scheduler": scheduler,
                         "denoise": 1.0}},
        "6": {"class_type": "VAEDecode",
              "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage",
              "inputs": {"images": ["6", 0], "filename_prefix": "x"}},
    }
    if extra_input:
        g["5"]["inputs"][extra_input] = 1
    if drop:
        del g["5"]["inputs"][drop]
    return g


class LiveSchemaErrorTests(unittest.TestCase):
    def test_valid_graph_passes(self):
        self.assertEqual(live_schema_errors(_good_graph(), INFO), [])

    def test_wrong_checkpoint_filename_reports_options(self):
        errs = live_schema_errors(_good_graph(ckpt="made_up.safetensors"),
                                  INFO)
        self.assertEqual(len(errs), 1)
        self.assertIn("made_up.safetensors", errs[0])
        self.assertIn("real.safetensors", errs[0])  # shows the real options

    def test_wrapped_combo_format_is_understood(self):
        errs = live_schema_errors(_good_graph(scheduler="bogus"), INFO)
        self.assertTrue(any("bogus" in e and "karras" in e for e in errs))

    def test_invented_input_name_is_reported(self):
        errs = live_schema_errors(_good_graph(extra_input="strength"), INFO)
        self.assertTrue(any("unknown input 'strength'" in e for e in errs))

    def test_missing_required_input_is_reported(self):
        errs = live_schema_errors(_good_graph(drop="denoise"), INFO)
        self.assertTrue(any("'denoise' is missing" in e for e in errs))

    def test_uninstalled_node_type_is_reported(self):
        g = {"1": {"class_type": "FancyCustomNode", "inputs": {}}}
        errs = live_schema_errors(g, INFO)
        self.assertIn("not installed", errs[0])

    def test_uploaded_loadimage_filename_is_not_rejected(self):
        """A just-uploaded LoadImage file may be absent from the cached
        /object_info options — it must NOT trigger a repair round."""
        info = {**INFO, "LoadImage": {"input": {"required": {
            "image": [["already_there.png"], {}]}}}}
        g = {"1": {"class_type": "LoadImage",
                   "inputs": {"image": "custom_src_freshly_uploaded.png"}},
             "2": {"class_type": "SaveImage",
                   "inputs": {"images": ["1", 0], "filename_prefix": "x"}}}
        self.assertEqual(live_schema_errors(g, info), [])

    def test_error_count_is_capped(self):
        g = {str(i): {"class_type": "Nope", "inputs": {}} for i in range(20)}
        self.assertLessEqual(len(live_schema_errors(g, INFO)), 6)


class GeneratorLiveSchemaLoopTests(unittest.TestCase):
    def test_bad_dropdown_value_is_bounced_back_and_repaired(self):
        bad = _good_graph(sampler="dpmpp_2m_karras")  # not a real sampler
        good = _good_graph(sampler="dpmpp_2m")
        llm = ScriptedLLM([json.dumps(bad), json.dumps(good)])
        gen = WorkflowGenerator(llm, schema_provider=lambda: INFO)
        wf = gen.generate("generate", "a cat")
        self.assertEqual(wf.graph["5"]["inputs"]["sampler_name"], "dpmpp_2m")
        # The retry message carried the live options to the LLM.
        self.assertIn("Live schema check failed", llm.prompts[1])
        self.assertIn("dpmpp_2m", llm.prompts[1])

    def test_provider_failure_never_blocks(self):
        def boom():
            raise OSError("comfy down")

        llm = ScriptedLLM([json.dumps(_good_graph())])
        gen = WorkflowGenerator(llm, schema_provider=boom)
        wf = gen.generate("generate", "a cat")
        self.assertIn("5", wf.graph)


class DescribeSceneTests(unittest.TestCase):
    def test_description_is_cleaned_and_capped(self):
        class Critic:
            def describe(self, image, question):
                return "  A woman   in a garden,\n bright sunlight. "

        out = quality.describe_scene(Critic(), Image.new("RGB", (8, 8)))
        self.assertEqual(out, "A woman in a garden, bright sunlight")

    def test_fakes_without_describe_are_skipped(self):
        class AskOnly:
            def ask(self, image, question):
                raise AssertionError("ask must not be consumed")

        self.assertIsNone(
            quality.describe_scene(AskOnly(), Image.new("RGB", (8, 8))))

    def test_failures_return_none(self):
        class Broken:
            def describe(self, image, question):
                raise OSError("no model")

        self.assertIsNone(
            quality.describe_scene(Broken(), Image.new("RGB", (8, 8))))
        class Rambler:
            def describe(self, image, question):
                return "   "

        self.assertIsNone(
            quality.describe_scene(Rambler(), Image.new("RGB", (8, 8))))


if __name__ == "__main__":
    unittest.main()
