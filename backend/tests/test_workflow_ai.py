"""Workflow generator tests — offline, fake LLM with scripted replies."""
import json
import unittest

from app.adapters.comfyui import WorkflowNotAllowedError
from app.core.llm import LLMReply, LLMUnavailableError
from app.core.workflow_ai import (
    MAX_NODES,
    GeneratedWorkflow,
    WorkflowGenerationError,
    WorkflowGenerator,
    validate_generated,
)

VALID_GRAPH = {
    "1": {"class_type": "CheckpointLoaderSimple",
          "inputs": {"ckpt_name": "model.safetensors"}},
    "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "clip": ["1", 1]}},
    "3": {"class_type": "EmptyLatentImage",
          "inputs": {"width": 512, "height": 512, "batch_size": 1}},
    "4": {"class_type": "KSampler",
          "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["2", 0],
                     "latent_image": ["3", 0], "seed": 1, "steps": 20, "cfg": 7,
                     "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
    "5": {"class_type": "VAEDecode", "inputs": {"samples": ["4", 0], "vae": ["1", 2]}},
    "6": {"class_type": "SaveImage",
          "inputs": {"images": ["5", 0], "filename_prefix": "gen"}},
}


class ScriptedLLM:
    """Returns queued replies in order and records every prompt it received."""

    source = "local"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, system, prompt, max_tokens=4096):
        self.prompts.append(prompt)
        if not self.replies:
            raise AssertionError("LLM called more times than scripted")
        return LLMReply(text=self.replies.pop(0), model="fake-model", source=self.source)


class GenerateTests(unittest.TestCase):
    def test_valid_first_attempt(self):
        llm = ScriptedLLM([json.dumps(VALID_GRAPH)])
        result = WorkflowGenerator(llm).generate("generate", "a cat")
        self.assertIsInstance(result, GeneratedWorkflow)
        self.assertEqual(result.graph, VALID_GRAPH)
        self.assertEqual(result.provenance,
                         {"source": "local", "model": "fake-model", "attempts": 1})

    def test_markdown_fences_are_tolerated(self):
        llm = ScriptedLLM(["```json\n" + json.dumps(VALID_GRAPH) + "\n```"])
        result = WorkflowGenerator(llm).generate("generate", "a cat")
        self.assertEqual(result.graph, VALID_GRAPH)

    def test_invalid_json_repaired_on_second_attempt(self):
        llm = ScriptedLLM(["this is not json{", json.dumps(VALID_GRAPH)])
        result = WorkflowGenerator(llm).generate("generate", "a cat")
        self.assertEqual(result.provenance["attempts"], 2)
        # the repair prompt carries the rejection reason back to the model
        self.assertIn("rejected", llm.prompts[1])
        self.assertIn("not valid JSON", llm.prompts[1])

    def test_disallowed_node_type_is_rejected_and_repaired(self):
        evil = {"1": {"class_type": "ExecutePythonCode", "inputs": {}},
                "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}}}
        llm = ScriptedLLM([json.dumps(evil), json.dumps(VALID_GRAPH)])
        result = WorkflowGenerator(llm).generate("generate", "a cat")
        self.assertEqual(result.provenance["attempts"], 2)
        self.assertIn("disallowed", llm.prompts[1])

    def test_exhausted_attempts_raise(self):
        llm = ScriptedLLM(["nope", "still nope", "nope again"])
        with self.assertRaises(WorkflowGenerationError) as ctx:
            WorkflowGenerator(llm, max_attempts=3).generate("generate", "a cat")
        self.assertIn("3 attempts", str(ctx.exception))

    def test_disallowed_task_rejected_without_llm_call(self):
        llm = ScriptedLLM([])
        with self.assertRaises(WorkflowNotAllowedError):
            WorkflowGenerator(llm).generate("txt2video_person_swap", "x")
        self.assertEqual(llm.prompts, [])

    def test_llm_unavailability_propagates(self):
        class DeadLLM:
            source = "local"

            def complete(self, system, prompt, max_tokens=4096):
                raise LLMUnavailableError("down")

        with self.assertRaises(LLMUnavailableError):
            WorkflowGenerator(DeadLLM()).generate("generate", "a cat")


class RepairTests(unittest.TestCase):
    def test_runtime_error_fed_back_and_fixed(self):
        llm = ScriptedLLM([json.dumps(VALID_GRAPH)])
        broken = dict(VALID_GRAPH)
        result = WorkflowGenerator(llm).repair(
            "generate", broken, "KSampler: value 999 for cfg is out of range")
        self.assertEqual(result.graph, VALID_GRAPH)
        self.assertIn("out of range", llm.prompts[0])   # error reached the model
        self.assertIn("KSampler", llm.prompts[0])


class ValidateGeneratedTests(unittest.TestCase):
    def test_missing_save_image_rejected(self):
        graph = {k: v for k, v in VALID_GRAPH.items() if k != "6"}
        from app.adapters.comfyui import WorkflowValidationError
        with self.assertRaises(WorkflowValidationError) as ctx:
            validate_generated(graph)
        self.assertIn("SaveImage", str(ctx.exception))

    def test_out_of_range_output_index_rejected(self):
        graph = dict(VALID_GRAPH)
        graph["6"] = {"class_type": "SaveImage",
                      "inputs": {"images": ["4", 1]}}  # KSampler has only 0
        from app.adapters.comfyui import WorkflowValidationError
        with self.assertRaises(WorkflowValidationError) as ctx:
            validate_generated(graph)
        self.assertIn("0=LATENT", str(ctx.exception))

    def test_oversized_graph_rejected(self):
        graph = {str(i): {"class_type": "SaveImage", "inputs": {}}
                 for i in range(MAX_NODES + 1)}
        from app.adapters.comfyui import WorkflowValidationError
        with self.assertRaises(WorkflowValidationError):
            validate_generated(graph)


if __name__ == "__main__":
    unittest.main()
