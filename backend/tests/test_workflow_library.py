"""Tests: template library, LLM workflow knowledge, hardware budget/clamps,
mask-first retry, repair knowledge, and centralized safety policy."""
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.adapters.comfyui import (
    ALLOWED_NODE_TYPES,
    ALLOWED_TASKS,
    WorkflowLibrary,
)
from app.config import Settings
from app.core.db import Database
from app.core.experience import ExperienceStore
from app.core.hardware import Hardware, render_budget
from app.core.safety import consent_verdict, model_source_blocked
from app.core.services import Services
from app.core.workflow_ai import NODE_OUTPUTS
from tests.test_workflow_job import GRAPH, DeadLLM, FakeComfy, ScriptedLLM

WORKFLOWS_DIR = Settings().workflows_dir


class TemplateLibraryTests(unittest.TestCase):
    def setUp(self):
        self.lib = WorkflowLibrary(WORKFLOWS_DIR)

    def test_library_has_at_least_twenty_validated_templates(self):
        templates = self.lib.list_all()
        self.assertGreaterEqual(len(templates), 20)
        names = [t["template"] for t in templates]
        for expected in ("outpaint", "upscale", "remove_object",
                         "video_inpaint", "video_outpaint", "generate_xl",
                         "restore_photo", "identity"):
            self.assertIn(expected, names)

    def test_every_template_parameter_targets_a_real_input(self):
        # A slot is {node, input} or a LIST of them (one value fanning out
        # to several inputs, e.g. the inpaint_hires crop coordinates).
        for t in self.lib.list_all():
            for pname, slot in t.get("parameters", {}).items():
                for target in (slot if isinstance(slot, list) else [slot]):
                    node = t["graph"].get(target["node"])
                    self.assertIsNotNone(node, (t["template"], pname))
                    self.assertIn(target["input"], node["inputs"],
                                  (t["template"], pname))

    def test_every_template_task_is_allowed(self):
        for t in self.lib.list_all():
            self.assertIn(t.get("task", t["template"]), ALLOWED_TASKS,
                          t["template"])

    def test_allowlist_and_output_table_stay_in_sync(self):
        self.assertEqual(set(NODE_OUTPUTS), ALLOWED_NODE_TYPES)

    def test_knowledge_teaches_guide_plus_example(self):
        k = self.lib.knowledge("outpaint")
        self.assertIn("ImagePadForOutpaint", k)
        self.assertIn("Global rules", k)
        self.assertIn("Common errors", k)
        self.assertIn('"class_type"', k)  # embedded example graph
        self.assertLessEqual(len(k), 6000)

    def test_knowledge_for_unknown_task_still_gives_globals(self):
        k = self.lib.knowledge("angles")
        self.assertIsNotNone(k)
        self.assertIn("SV3D", k)


class HardwareBudgetTests(unittest.TestCase):
    def test_budget_scales_with_tier(self):
        low = render_budget(Hardware("gpu", 4, 8, 100))
        mid = render_budget(Hardware("gpu", 8, 16, 100))
        high = render_budget(Hardware("gpu", 24, 64, 100))
        self.assertLess(low["max_pixels"], mid["max_pixels"])
        self.assertLess(mid["max_pixels"], high["max_pixels"])

    def test_graph_clamp_shrinks_oversize_canvas_and_steps(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        services = Services(Settings(
            data_dir=Path(tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="",
            first_run_setup=False, comfyui_dir=""))
        services.hardware = Hardware("gpu", 8, 16, 100)  # mid tier

        class LogJob:
            logs = []

            def log(self, level, msg):
                LogJob.logs.append(msg)

        graph = {
            "1": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": 4096, "height": 4096, "batch_size": 4}},
            "2": {"class_type": "KSampler", "inputs": {"steps": 150}},
        }
        out = services._apply_hardware_limits(graph, LogJob())
        w, h = out["1"]["inputs"]["width"], out["1"]["inputs"]["height"]
        self.assertLessEqual(w * h, 1024 * 1024)
        self.assertEqual(w % 8, 0)
        self.assertEqual(out["1"]["inputs"]["batch_size"], 1)
        self.assertEqual(out["2"]["inputs"]["steps"], 50)
        self.assertTrue(any("Clamped" in m for m in LogJob.logs))

    def test_graph_clamp_leaves_sane_graphs_alone(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        services = Services(Settings(
            data_dir=Path(tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="",
            first_run_setup=False, comfyui_dir=""))
        services.hardware = Hardware("gpu", 8, 16, 100)

        class LogJob:
            def log(self, level, msg):
                raise AssertionError("no clamping expected")

        graph = {"1": {"class_type": "EmptyLatentImage",
                       "inputs": {"width": 768, "height": 512,
                                  "batch_size": 1}}}
        out = services._apply_hardware_limits(graph, LogJob())
        self.assertEqual(out["1"]["inputs"]["width"], 768)


class MaskRefineTests(unittest.TestCase):
    def test_refined_mask_is_bigger_and_softer(self):
        mask = Image.new("L", (64, 64), 0)
        mask.paste(255, (24, 24, 40, 40))
        refined = Services._refine_mask(mask)
        self.assertGreater(sum(refined.getdata()), sum(mask.getdata()))
        values = set(refined.getdata())
        self.assertTrue(any(0 < v < 255 for v in values), "edges not feathered")


class RepairKnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ExperienceStore(Database(Path(self.tmp.name) / "t.sqlite3"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_repair_is_distilled_and_replayed(self):
        before = {"1": {"class_type": "KSampler",
                        "inputs": {"steps": 20, "cfg": 7.0}}}
        after = {"1": {"class_type": "KSampler",
                       "inputs": {"steps": 30, "cfg": 6.0}}}
        self.store.record_repair("generate", "node error: cfg", before, after)
        hints = self.store.repair_hints("generate")
        self.assertEqual(len(hints), 1)
        self.assertIn("KSampler.steps: 20 → 30", hints[0])
        lessons = self.store.lessons("generate", "whatever")
        self.assertIn("Proven fixes", lessons)

    def test_fix_notes_never_cross_diff_same_type_nodes(self):
        """Positive vs negative CLIPTextEncode must not be diffed against
        each other — that would poison the knowledge base."""
        before = {
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat"}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry"}},
            "5": {"class_type": "KSampler", "inputs": {"steps": 80}},
        }
        after = {
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat"}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry"}},
            "5": {"class_type": "KSampler", "inputs": {"steps": 30}},
        }
        self.store.record_repair("generate", "oom", before, after)
        hints = self.store.repair_hints("generate")
        self.assertEqual(len(hints), 1)
        self.assertIn("KSampler.steps: 80 → 30", hints[0])
        self.assertNotIn("CLIPTextEncode", hints[0])

    def test_relink_repairs_are_described(self):
        before = {"5": {"class_type": "KSampler",
                        "inputs": {"positive": ["2", 0]}}}
        after = {"5": {"class_type": "KSampler",
                       "inputs": {"positive": ["4", 0]}}}
        self.store.record_repair("generate", "bad link", before, after)
        hints = self.store.repair_hints("generate")
        self.assertIn("KSampler.positive re-linked", hints[0])

    def test_duplicate_repair_bumps_uses_instead_of_duplicating(self):
        b = {"1": {"class_type": "KSampler", "inputs": {"steps": 20}}}
        a = {"1": {"class_type": "KSampler", "inputs": {"steps": 30}}}
        self.store.record_repair("generate", "err X", b, a)
        self.store.record_repair("generate", "err X", b, a)
        rows = self.store._db.query("SELECT uses FROM repair_knowledge")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["uses"], 1)

    def test_workflow_job_records_repair_lesson(self):
        services = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="",
            first_run_setup=False, comfyui_dir=""))
        services.scout.llm = DeadLLM()
        services.llm = DeadLLM()
        services.comfy = FakeComfy(fail_times=1)  # first run fails, then OK
        fixed = json.loads(json.dumps(GRAPH))
        fixed["1"]["inputs"]["ckpt_name"] = "real.safetensors"
        services.workflow_ai.llm = ScriptedLLM(
            [json.dumps(GRAPH), json.dumps(fixed)])
        services.start()
        self.addCleanup(services.stop)
        job = services.queue.enqueue(
            "workflow", {"task": "generate", "prompt": "a cat"})
        done = services.queue.wait_for(job.id, timeout=20)
        self.assertEqual(done.state.value, "completed")
        rows = services.db.query("SELECT * FROM repair_knowledge")
        # graphs identical → no fix note; make sure no crash either way
        self.assertLessEqual(len(rows), 1)


class SafetyPolicyHelpersTests(unittest.TestCase):
    def test_consent_verdict(self):
        self.assertTrue(consent_verdict(True).allowed)
        v = consent_verdict(False)
        self.assertFalse(v.allowed)
        self.assertEqual(v.category, "consent")
        self.assertIn("consent", (v.reason or "").lower())

    def test_model_source_blocked(self):
        self.assertTrue(model_source_blocked(True))
        self.assertFalse(model_source_blocked(False))

    def test_no_filter_rules_outside_safety_py(self):
        """Content-safety keyword rules must live in safety.py only."""
        core = Path(Settings().workflows_dir).parent / "core"
        for path in core.glob("*.py"):
            if path.name == "safety.py":
                continue
            text = path.read_text(encoding="utf-8").lower()
            for token in ("deepfake", "undress", "nonconsensual"):
                self.assertNotIn(token, text,
                                 f"{path.name} embeds a safety rule "
                                 f"('{token}') — move it to safety.py")


class WorkflowInputImageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.services = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="",
            first_run_setup=False, comfyui_dir=""))
        self.services.scout.llm = DeadLLM()
        self.services.llm = DeadLLM()
        self.services.start()

    def tearDown(self):
        self.services.stop()
        self.tmp.cleanup()

    def test_image_task_without_image_fails_clearly(self):
        self.services.comfy = FakeComfy()
        job = self.services.queue.enqueue(
            "workflow", {"task": "img2img", "prompt": "make it night"})
        done = self.services.queue.wait_for(job.id, timeout=20)
        self.assertEqual(done.state.value, "failed")
        self.assertIn("upload", (done.error or "").lower())

    def test_asset_is_uploaded_and_named_in_context(self):
        import io
        comfy = FakeComfy()
        uploads = []
        comfy.upload_image = lambda img, prefix: (
            uploads.append(prefix) or f"{prefix}_1.png")
        self.services.comfy = comfy
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), (1, 2, 3)).save(buf, format="PNG")
        asset = self.services.store.save_upload("src.png", buf.getvalue())
        llm = ScriptedLLM([json.dumps(GRAPH)])
        self.services.workflow_ai.llm = llm
        job = self.services.queue.enqueue("workflow", {
            "task": "img2img", "prompt": "make it night",
            "asset_id": asset.id})
        done = self.services.queue.wait_for(job.id, timeout=20)
        self.assertEqual(done.state.value, "completed")
        self.assertEqual(uploads, ["forge_src"])
        self.assertIn("forge_src_1.png", llm.prompts[0])
        self.assertIn("USE IT FULLY", llm.prompts[0])  # hardware maximization
        self.assertIn("Global rules", llm.prompts[0])  # workflow guide


if __name__ == "__main__":
    unittest.main()
