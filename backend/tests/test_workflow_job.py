"""Workflow-job tests: inventory, auto-install, and the execute→repair loop.

Offline: ComfyUI and the LLM are replaced by fakes injected into Services.
"""
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.adapters.comfyui import WorkflowRuntimeError
from app.config import Settings
from app.core.llm import LLMReply
from app.core.services import Services

GRAPH = {
    "1": {"class_type": "CheckpointLoaderSimple",
          "inputs": {"ckpt_name": "real.safetensors"}},
    "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
}


class FakeComfy:
    def __init__(self, up=True, checkpoints=None, fail_times=0):
        self.up = up
        self.checkpoints = checkpoints if checkpoints is not None else ["real.safetensors"]
        self.fail_times = fail_times
        self.runs = 0

    def is_up(self):
        return self.up

    def installed_checkpoints(self):
        return list(self.checkpoints)

    def run_graph(self, graph):
        self.runs += 1
        if self.runs <= self.fail_times:
            raise WorkflowRuntimeError(f"node error on run {self.runs}")
        return Image.new("RGB", (8, 8), (10, 200, 30)), f"pid-{self.runs}"


class ScriptedLLM:
    source = "local"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, system, prompt, max_tokens=4096):
        self.prompts.append(prompt)
        return LLMReply(self.replies.pop(0), "fake-model", "local")


class DeadLLM:
    """Deterministic offline stand-in: always unavailable."""

    source = "local"

    def complete(self, system, prompt, max_tokens=4096):
        from app.core.llm import LLMUnavailableError
        raise LLMUnavailableError("offline test")


class WorkflowJobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.services = Services(Settings(
            data_dir=Path(self.tmp.name),
            inpaint_backend="mock", segment_backend="mock",
            critic_model="", first_run_setup=False, comfyui_dir=""))
        # Keep the scout deterministic and offline: it falls back to the
        # first installed checkpoint when its LLM is unavailable.
        self.services.scout.llm = DeadLLM()
        self.services.llm = DeadLLM()
        self.services.start()

    def tearDown(self):
        self.services.stop()
        self.tmp.cleanup()

    def _run(self, comfy, llm_replies, prompt="a cat"):
        self.services.comfy = comfy
        self.services.workflow_ai.llm = ScriptedLLM(llm_replies)
        job = self.services.queue.enqueue("workflow",
                                          {"task": "generate", "prompt": prompt})
        return self.services.queue.wait_for(job.id)

    def test_success_saves_asset_with_provenance(self):
        comfy = FakeComfy()
        job = self._run(comfy, [json.dumps(GRAPH)])
        self.assertEqual(job.state.value, "completed")
        self.assertEqual(job.result["provenance"]["source"], "local")
        self.assertEqual(job.result["repairs"], 0)
        asset = self.services.store.get_asset(job.result["asset_id"])
        self.assertIsNotNone(asset)
        self.assertTrue(Path(asset.path).exists())

    def test_inventory_reaches_the_llm_prompt(self):
        comfy = FakeComfy(checkpoints=["special-model.safetensors"])
        self.services.comfy = comfy
        llm = ScriptedLLM([json.dumps(GRAPH)])
        self.services.workflow_ai.llm = llm
        job = self.services.queue.enqueue("workflow",
                                          {"task": "generate", "prompt": "x"})
        self.services.queue.wait_for(job.id)
        self.assertIn("special-model.safetensors", llm.prompts[0])

    def test_runtime_error_triggers_llm_repair_then_succeeds(self):
        comfy = FakeComfy(fail_times=1)
        job = self._run(comfy, [json.dumps(GRAPH), json.dumps(GRAPH)])
        self.assertEqual(job.state.value, "completed")
        self.assertEqual(job.result["repairs"], 1)
        logs = " ".join(entry["msg"] for entry in job.logs)
        self.assertIn("repair", logs)

    def test_repair_budget_exhausted_fails_permanently(self):
        comfy = FakeComfy(fail_times=99)
        replies = [json.dumps(GRAPH)] * 4
        job = self._run(comfy, replies)
        self.assertEqual(job.state.value, "failed")
        self.assertIn("repair", (job.error or ""))
        self.assertEqual(job.attempts, 1)  # permanent: no queue-level retries

    def test_comfy_down_is_transient(self):
        comfy = FakeComfy(up=False)
        job = self._run(comfy, [])
        self.assertEqual(job.state.value, "failed")  # after queue retries
        self.assertIn("not running", (job.error or ""))
        self.assertGreater(job.attempts, 1)  # transient: it retried

    def test_no_checkpoints_and_no_candidates_fails_actionably(self):
        # Remove every download URL so auto-install has nothing to fetch.
        from app.core.registry import ModelInfo
        for m in self.services.registry.list():
            self.services.registry.register(ModelInfo(
                name=m.name, purpose=m.purpose, license=m.license, url=None))
        comfy = FakeComfy(checkpoints=[])
        job = self._run(comfy, [])
        self.assertEqual(job.state.value, "failed")
        self.assertIn("checkpoint", (job.error or "").lower())

    def test_downloaded_but_invisible_checkpoint_explains_itself(self):
        # Registry says the checkpoint is ready on disk, ComfyUI can't see it:
        # must NOT re-download; must point at extra_model_paths.yaml.
        models_dir = self.services.settings.models_dir
        fake = models_dir / "sd-v1-5-inpainting.safetensors"
        fake.write_bytes(b"weights")
        self.services.registry.set_status("sd15-inpaint", "ready", path=str(fake))
        comfy = FakeComfy(checkpoints=[])
        job = self._run(comfy, [])
        self.assertEqual(job.state.value, "failed")
        self.assertIn("extra_model_paths", (job.error or ""))


class _StubJob:
    def log(self, *_a, **_k):
        pass


class TemplateFastPathTests(unittest.TestCase):
    """A triage-chosen VALIDATED template must actually render its tuned
    graph — not fall through to a custom LLM-designed SDXL graph."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.services = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))
        # LIFO: temp-dir cleanup registered FIRST so it runs LAST — the DB is
        # closed by services.stop() before the folder is removed (else the
        # open SQLite file → WinError 32 on Windows).
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.services.stop)

    def test_ready_template_renders_its_own_graph(self):
        # generate_zimage is a real shipped template; pretend its models are
        # ready + the machine is big enough, and confirm we build ITS graph
        # (Z-Image UNET, not SDXL).
        ready = {"zimage-turbo", "zimage-text-encoder", "flux-ae"}
        self.services.registry.is_ready = lambda n: n in ready
        self.services.hardware.ram_gb = 32.0  # Z-Image needs ~24 GB RAM
        gen = self.services._template_workflow(
            _StubJob(), {"workflow": "generate_zimage"}, "a shop sign", None)
        self.assertIsNotNone(gen)
        self.assertEqual(gen.provenance["source"], "template")
        self.assertEqual(gen.provenance["model"], "generate_zimage")
        classes = {n["class_type"] for n in gen.graph.values()}
        self.assertIn("UNETLoader", classes)          # Z-Image, not SDXL
        self.assertNotIn("CheckpointLoaderSimple", classes)
        # the prompt landed in the graph
        texts = [n["inputs"].get("text") for n in gen.graph.values()
                 if n["class_type"] == "CLIPTextEncode"]
        self.assertIn("a shop sign", texts)

    def test_missing_models_falls_back_to_custom(self):
        self.services.registry.is_ready = lambda n: False
        gen = self.services._template_workflow(
            _StubJob(), {"workflow": "generate_zimage"}, "x", None)
        self.assertIsNone(gen)  # -> custom LLM design path

    def test_memory_gate_blocks_templates_that_wont_fit(self):
        # Z-Image's encoder declares min_ram_gb 24; on this fake 16 GB
        # machine the template must be refused (→ custom SDXL that fits),
        # never routed to and left to OOM.
        self.services.registry.is_ready = lambda n: True
        self.services.hardware.ram_gb = 16.0
        self.services.hardware.vram_gb = 8.0
        gen = self.services._template_workflow(
            _StubJob(), {"workflow": "generate_zimage"}, "x", None)
        self.assertIsNone(gen)
        fits, why = self.services._models_fit_machine(["zimage-text-encoder"])
        self.assertFalse(fits)
        self.assertIn("RAM", why)
        # A 32 GB machine runs it.
        self.services.hardware.ram_gb = 32.0
        self.assertTrue(
            self.services._models_fit_machine(["zimage-text-encoder"])[0])

    def test_image_template_without_input_falls_back(self):
        self.services.registry.is_ready = lambda n: True
        gen = self.services._template_workflow(
            _StubJob(), {"workflow": "img2img_canny"}, "x", None)
        self.assertIsNone(gen)  # needs an input image → custom path

    def test_unknown_or_empty_workflow_is_none(self):
        self.assertIsNone(self.services._template_workflow(
            _StubJob(), {"workflow": "no_such_template"}, "x", None))
        self.assertIsNone(self.services._template_workflow(
            _StubJob(), {"workflow": ""}, "x", None))
        self.assertIsNone(self.services._template_workflow(
            _StubJob(), None, "x", None))


class ForgeAccuracyRetryTests(unittest.TestCase):
    """Realism can pass while the REQUEST was missed — the adherence ladder
    re-renders with the user's words attention-weighted, and keeps the attempt
    that actually delivers the request."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.services = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="fake",
            first_run_setup=False, comfyui_dir=""))
        self.services.scout.llm = DeadLLM()
        self.services.llm = DeadLLM()
        self.services.start()
        self.addCleanup(self.services.stop)

    def test_low_prompt_accuracy_triggers_weighted_retry(self):
        from app.core.critic import Critique
        from app.core.quality import SCORE_KEYS

        class AccCritic:
            asks = 0

            def critique(self, image, prompt):
                return Critique(score=9.0, issues=[], model="fake")

            def ask(self, image, question):
                AccCritic.asks += 1
                scores = dict.fromkeys(SCORE_KEYS, 90)
                if AccCritic.asks == 1:
                    scores["prompt_accuracy"] = 40  # request was missed
                return json.dumps(scores)

        self.services.critic = AccCritic()
        self.services.comfy = FakeComfy()
        llm = ScriptedLLM([json.dumps(GRAPH), json.dumps(GRAPH)])
        self.services.workflow_ai.llm = llm
        job = self.services.queue.enqueue(
            "workflow", {"task": "generate", "prompt": "a cat"})
        done = self.services.queue.wait_for(job.id, timeout=30)
        self.assertEqual(done.state.value, "completed", done.error)
        logs = " ".join(e["msg"] for e in done.logs)
        self.assertIn("missed part of the request", logs)
        # The second plan carried the weighted prompt.
        self.assertIn("(a cat:1.3)", llm.prompts[1])
        # The better-matching retry replaced the first render.
        self.assertIn("Round 1 kept", logs)

    def test_a_matching_render_is_never_retried(self):
        """The ladder costs a full render per rung: a first attempt that
        already does what was asked must end the job."""
        from app.core.critic import Critique
        from app.core.quality import SCORE_KEYS

        class GoodCritic:
            asks = 0

            def critique(self, image, prompt):
                return Critique(score=9.0, issues=[], model="fake")

            def ask(self, image, question):
                GoodCritic.asks += 1
                return json.dumps(dict.fromkeys(SCORE_KEYS, 96))

        self.services.critic = GoodCritic()
        self.services.comfy = FakeComfy()
        llm = ScriptedLLM([json.dumps(GRAPH)])  # only ONE plan available
        self.services.workflow_ai.llm = llm
        job = self.services.queue.enqueue(
            "workflow", {"task": "generate", "prompt": "a cat"})
        done = self.services.queue.wait_for(job.id, timeout=30)
        self.assertEqual(done.state.value, "completed", done.error)
        self.assertEqual(len(llm.prompts), 1)  # no retry was planned
        self.assertNotIn("[stage] retry",
                         " ".join(e["msg"] for e in done.logs))


if __name__ == "__main__":
    unittest.main()
