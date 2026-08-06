"""Tests: the model knowledge base — online-researched capability notes +
ratings per checkpoint, consulted at planning time, refreshed on download.
All offline (search + LLM are fakes)."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from app.config import Settings
from app.core.jobs import Job
from app.core.model_intel import ModelIntel
from app.core.registry import ModelInfo
from app.core.services import Services
from tests.test_quality import OneJson
from tests.test_workflow_job import DeadLLM

NOTES = {"best_at": "photoreal portraits, skin texture",
         "avoid": "anime", "prompt_style": "short natural sentences",
         "quality": 9, "reason": "top community model"}


class FakeSearch:
    def __init__(self, hits=None, boom=False):
        self.hits = hits or []
        self.boom = boom
        self.queries = []

    def search_civitai_rich(self, query, type_key="checkpoint", limit=12):
        self.queries.append(query)
        if self.boom:
            raise OSError("offline")
        return self.hits


class ModelIntelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "model_knowledge.json"

    def test_research_stores_notes_with_online_source(self):
        search = FakeSearch(hits=[{
            "name": "Juggernaut XL", "creator": "KandooAI",
            "downloads": 47905, "rating": 5000, "base_model": "SDXL 1.0",
            "description": "Photorealistic all-rounder.",
            "trigger_words": [],
            "filename": "juggernautXL_inpaint.safetensors"}])
        intel = ModelIntel(self.path, search)
        logs = []
        entry = intel.research("juggernautXL_inpaint.safetensors",
                               OneJson(NOTES), log=logs.append)
        self.assertEqual(entry["quality"], 9)
        self.assertIn("civitai:Juggernaut XL", entry["source"])
        self.assertTrue(any(m.startswith("[search]") for m in logs))
        # Persisted to the standalone file, human-readable JSON.
        on_disk = json.loads(self.path.read_text())
        self.assertEqual(
            on_disk["juggernautXL_inpaint.safetensors"]["best_at"],
            "photoreal portraits, skin texture")

    def test_unrelated_search_hits_are_rejected(self):
        """A wrong model page must not poison the notes (seen live) — weak
        matches fall back to the LLM's own knowledge."""
        search = FakeSearch(hits=[{
            "name": "CineScapeXL beta", "creator": "x", "downloads": 10,
            "rating": 1, "base_model": "SDXL 1.0",
            "description": "Unrelated landscape model.",
            "filename": "cinescapexl_beta.safetensors"}])
        intel = ModelIntel(self.path, search)
        entry = intel.research("sd-v1-5-inpainting.safetensors",
                               OneJson(NOTES))
        self.assertEqual(entry["source"], "llm-knowledge")

    def test_research_works_offline_from_llm_knowledge(self):
        intel = ModelIntel(self.path, FakeSearch(boom=True))
        entry = intel.research("epicrealism_v10-inpainting.safetensors",
                               OneJson(NOTES))
        self.assertIsNotNone(entry)
        self.assertEqual(entry["source"], "llm-knowledge")

    def test_dead_llm_returns_none_and_stores_nothing(self):
        intel = ModelIntel(self.path, FakeSearch())
        self.assertIsNone(intel.research("x.safetensors", DeadLLM()))
        self.assertEqual(intel.load(), {})

    def test_quality_rating_is_clamped(self):
        intel = ModelIntel(self.path, None)
        entry = intel.research("x.safetensors",
                               OneJson({**NOTES, "quality": 42}))
        self.assertEqual(entry["quality"], 10)

    def test_summary_ranks_best_first_and_missing_lists_gaps(self):
        intel = ModelIntel(self.path, None)
        intel._save({
            "weak.safetensors": {**NOTES, "quality": 4, "best_at": "sketches"},
            "strong.safetensors": {**NOTES, "quality": 9},
        })
        s = intel.summary(["weak.safetensors", "strong.safetensors",
                           "unknown.safetensors"])
        self.assertTrue(s.startswith("Model knowledge"))
        self.assertLess(s.index("strong.safetensors"),
                        s.index("weak.safetensors"))
        self.assertIsNone(intel.summary(["unknown.safetensors"]))
        self.assertEqual(intel.missing(["strong.safetensors",
                                        "unknown.safetensors"]),
                         ["unknown.safetensors"])

    def test_query_for_cleans_filenames(self):
        self.assertEqual(
            ModelIntel._query_for("juggernautXL_v9rdphoto2Inpaint.safetensors"),
            "juggernaut XL v9rdphoto2 Inpaint")


class ResearchWiringTests(unittest.TestCase):
    """Services wiring: research queued when models lack notes and after a
    download completes; the knowledge block reaches the scout prompt."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="",
            first_run_setup=False, comfyui_dir=""))  # not started: no worker

    def test_queue_model_research_once_per_missing_model(self):
        class LogJob:
            logs = []

            def log(self, level, msg):
                LogJob.logs.append(msg)

        self.s._queue_model_research(LogJob(), ["a.safetensors",
                                                "b.safetensors"])
        self.s._queue_model_research(LogJob(), ["a.safetensors"])
        jobs = [j for j in self.s.queue.list() if j.type == "model_research"]
        self.assertEqual({j.payload["file"] for j in jobs},
                         {"a.safetensors", "b.safetensors"})
        self.assertEqual(len(jobs), 2)  # deduped on the second call

    def test_download_completion_queues_research(self):
        payload = b"weights" * 512
        src = Path(self.tmp.name) / "95864"  # civitai-style bare id
        src.write_bytes(payload)
        self.s.registry.register(ModelInfo(
            name="test-ckpt", purpose="checkpoint for tests", license="x",
            url=src.as_uri(), sha256=hashlib.sha256(payload).hexdigest(),
            meta={"folder": "checkpoints",
                  "file": "testCkpt_v1.safetensors"}))
        job = Job(id="t1", type="model_download",
                  payload={"model": "test-ckpt"})
        result = self.s._handle_model_download(job)
        self.assertEqual(result["status"], "ready")
        research = [j for j in self.s.queue.list()
                    if j.type == "model_research"]
        self.assertEqual(research[0].payload["file"],
                         "testCkpt_v1.safetensors")

    def test_research_handler_writes_the_knowledge_file(self):
        self.s.llm = OneJson(NOTES)
        self.s.model_intel.search = None  # offline
        job = Job(id="t2", type="model_research",
                  payload={"file": "some_model.safetensors"})
        result = self.s._handle_model_research(job)
        self.assertTrue(result["researched"])
        self.assertEqual(
            self.s.model_intel.get("some_model.safetensors")["quality"], 9)

    def test_research_handler_degrades_gracefully(self):
        self.s.llm = DeadLLM()
        self.s.model_intel.search = None
        job = Job(id="t3", type="model_research",
                  payload={"file": "some_model.safetensors"})
        self.assertFalse(self.s._handle_model_research(job)["researched"])


if __name__ == "__main__":
    unittest.main()
