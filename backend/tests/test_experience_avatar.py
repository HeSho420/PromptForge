"""Tests: workflow memory, forced model search, avatar intake job."""
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.config import Settings
from app.core.db import Database
from app.core.experience import ExperienceStore
from app.core.registry import DownloadError
from app.core.services import Services
from tests.test_scout_critic_video import OneShotLLM
from tests.test_workflow_job import GRAPH, DeadLLM, FakeComfy, ScriptedLLM


class ExperienceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ExperienceStore(Database(Path(self.tmp.name) / "t.sqlite3"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_best_example_prefers_high_realism_and_similar_prompt(self):
        self.store.record("generate", "a red car", {"1": {"a": 1}}, True, realism=5)
        self.store.record("generate", "a lighthouse in fog", {"2": {"b": 2}},
                          True, realism=9)
        best = self.store.best_example("generate", "foggy lighthouse at dusk")
        self.assertEqual(best, {"2": {"b": 2}})

    def test_failures_are_not_examples_but_are_pitfalls(self):
        self.store.record("generate", "x", {"1": {}}, False,
                          errors=["bad_linked_input: negative"])
        self.assertIsNone(self.store.best_example("generate", "x"))
        self.assertIn("bad_linked_input: negative",
                      self.store.known_pitfalls("generate"))

    def test_lessons_block_contains_both(self):
        self.store.record("generate", "cat", {"1": {"c": 3}}, True, realism=8)
        self.store.record("generate", "cat", None, False, errors=["oom"])
        lessons = self.store.lessons("generate", "cat photo")
        self.assertIn('{"1": {"c": 3}}', lessons)
        self.assertIn("oom", lessons)

    def test_no_history_returns_none(self):
        self.assertIsNone(self.store.lessons("generate", "anything"))


class WorkflowMemoryIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.services = Services(Settings(
            data_dir=Path(self.tmp.name),
            inpaint_backend="mock", segment_backend="mock", critic_model="",
            first_run_setup=False, comfyui_dir=""))
        self.services.scout.llm = DeadLLM()
        self.services.llm = DeadLLM()
        self.services.start()

    def tearDown(self):
        self.services.stop()
        self.tmp.cleanup()

    def test_success_recorded_and_lessons_fed_to_next_plan(self):
        self.services.comfy = FakeComfy()
        self.services.workflow_ai.llm = ScriptedLLM([json.dumps(GRAPH)])
        j1 = self.services.queue.enqueue("workflow",
                                         {"task": "generate", "prompt": "a cat"})
        self.services.queue.wait_for(j1.id)
        rows = self.services.db.query("SELECT * FROM workflow_memory")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["success"], 1)

        # Second run: the planner prompt must contain the past graph.
        llm2 = ScriptedLLM([json.dumps(GRAPH)])
        self.services.workflow_ai.llm = llm2
        j2 = self.services.queue.enqueue("workflow",
                                         {"task": "generate", "prompt": "a cat again"})
        self.services.queue.wait_for(j2.id)
        self.assertIn("past workflow", llm2.prompts[0].lower())

    def test_forced_search_when_prompt_demands_it(self):
        self.services.comfy = FakeComfy()
        self.services.workflow_ai.llm = ScriptedLLM([json.dumps(GRAPH)])
        # scout LLM refuses to search; code must derive a query and try anyway
        self.services.scout.llm = OneShotLLM(json.dumps({"use": "real.safetensors"}))
        searched = []
        self.services.scout.search.search = lambda q, limit=6: (
            searched.append(q) or [])
        self.services.scout.search.search_civitai = lambda q, limit=8: []
        job = self.services.queue.enqueue("workflow", {
            "task": "generate",
            "prompt": "a portrait — please search online for a better model"})
        done = self.services.queue.wait_for(job.id)
        self.assertEqual(done.state.value, "completed")  # falls back gracefully
        self.assertTrue(searched, "forced search must hit the hub")
        logs = " ".join(e["msg"] for e in done.logs)
        self.assertIn("explicitly asks for a model search", logs)


class GatedMirrorTests(unittest.TestCase):
    """_ensure_model must self-heal when the source is license-gated."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.services = Services(Settings(
            data_dir=Path(self.tmp.name),
            inpaint_backend="mock", segment_backend="mock", critic_model="",
            first_run_setup=False, comfyui_dir=""))

    def tearDown(self):
        self.tmp.cleanup()

    def test_gated_download_switches_to_verified_mirror(self):
        calls = []
        real_registry = self.services.registry

        class FlakyDownloader:
            def download(inner, name, progress=None):
                model = real_registry.get(name)
                calls.append(model.url)
                if "gated-org" in (model.url or ""):
                    raise DownloadError("HTTP Error 401: Unauthorized")
                f = Path(self.tmp.name) / "models" / "m.safetensors"
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(b"w")
                real_registry.set_status(name, "ready", path=str(f))

        routes = {
            "/api/models?": [
                {"id": "gated-org/thing", "downloads": 99, "likes": 0,
                 "pipeline_tag": None, "gated": True},
                {"id": "mirror/thing", "downloads": 500, "likes": 0,
                 "pipeline_tag": None, "gated": False},
            ],
            "/tree/main": [{"type": "file", "path": "thing_u.safetensors",
                            "size": 123, "lfs": {"oid": "e" * 64}}],
        }
        from app.core.model_search import ModelSearch
        from app.core.registry import ModelInfo
        self.services.model_search = ModelSearch(
            real_registry,
            http_get=lambda url, t: next(
                data for frag, data in routes.items() if frag in url))
        from app.core.trust import TrustJudge
        self.services.trust = TrustJudge(None)  # rules-only for determinism
        self.services.downloader = FlakyDownloader()
        real_registry.register(ModelInfo(
            name="thing", purpose="test model", license="x",
            url="https://huggingface.co/gated-org/thing/resolve/main/thing_u.safetensors",
            meta={"repo": "gated-org/thing", "file": "thing_u.safetensors",
                  "folder": "checkpoints"}))

        class LogJob:
            def log(self, level, msg):
                pass

        self.services._ensure_model("thing", LogJob())
        updated = real_registry.get("thing")
        self.assertIn("mirror/thing", updated.url)
        self.assertEqual(updated.sha256, "e" * 64)
        self.assertTrue(real_registry.is_ready("thing"))
        self.assertEqual(len(calls), 2)  # gated attempt, then mirror


class FakeSegPoint:
    name = "sam-vit-b"
    is_mock = False

    def point_mask(self, image, x, y):
        m = Image.new("L", image.size, 0)
        m.paste(255, (0, 0, image.width // 2, image.height))
        return m


class FakeCriticViews:
    """A view classifier that gives the same answer twice for a photo.

    The real one is asked twice on purpose — it is a vision model and it is
    not deterministic, so an answer that does not repeat is not acted on. A
    fake that answers once per photo does not model that, and popping a
    fixed list ran dry on the second ask."""

    def __init__(self, views):
        self.views = list(views)
        self.asked = 0

    def ask(self, image, question):
        index = self.asked // 2
        self.asked += 1
        if index >= len(self.views):
            raise AssertionError("classified more photos than were given")
        return json.dumps({"view": self.views[index]})

    def critique(self, image, prompt):
        raise AssertionError("not used here")


class AvatarJobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.services = Services(Settings(
            data_dir=Path(self.tmp.name),
            inpaint_backend="mock", segment_backend="mock",
            critic_model="fake", auto_install=False, first_run_setup=False, comfyui_dir=""))
        self.services.segmentation = FakeSegPoint()
        # Offline, like every other fixture. Without this the failure path
        # (_diagnose_and_record) makes a REAL Ollama call, and the test's
        # runtime becomes "however long qwen takes to load today" — it passed
        # or timed out depending on what else was using the GPU.
        self.services.llm = DeadLLM()
        self.services.start()

    def tearDown(self):
        self.services.stop()
        self.tmp.cleanup()

    def _photo(self, name):
        buf = io.BytesIO()
        Image.new("RGB", (64, 96), (150, 120, 100)).save(buf, format="PNG")
        return self.services.store.save_upload(name, buf.getvalue())

    def test_consent_is_mandatory(self):
        a = self._photo("p1.png")
        job = self.services.queue.enqueue("avatar",
                                          {"asset_ids": [a.id], "consent": False})
        done = self.services.queue.wait_for(job.id)
        self.assertEqual(done.state.value, "failed")
        self.assertIn("consent", (done.error or "").lower())

    def test_intake_reports_coverage_and_cutouts(self):
        a1, a2 = self._photo("p1.png"), self._photo("p2.png")
        self.services.critic = FakeCriticViews(["front", "left"])

        class ComfyDown:
            def is_up(self):
                return False

        self.services.comfy = ComfyDown()
        job = self.services.queue.enqueue(
            "avatar", {"asset_ids": [a1.id, a2.id], "consent": True})
        done = self.services.queue.wait_for(job.id, timeout=30)
        self.assertEqual(done.state.value, "completed")
        r = done.result
        self.assertEqual(r["photos"], 2)
        self.assertEqual(r["coverage"]["front"], [a1.id])
        self.assertEqual(r["coverage"]["left"], [a2.id])
        self.assertIn("back", r["missing"])
        # SAM cutout versions were stored on the assets
        versions = self.services.store.gallery()
        cutouts = [v for e in versions for v in e["versions"]
                   if v["adapter"] == "sam-cutout"]
        self.assertEqual(len(cutouts), 2)


if __name__ == "__main__":
    unittest.main()
