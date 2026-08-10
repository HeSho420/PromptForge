"""Tests for this batch: Ollama liveness/revive, prompt triage, diagnose+learn,
and the Improve-LLM discover/approve flow."""
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path

from app.config import Settings
from app.core.llm import ollama_is_up
from app.core.services import Services
from tests.test_workflow_job import GRAPH, DeadLLM, FakeComfy, ScriptedLLM


def _services(tmp, **kw):
    base = dict(data_dir=Path(tmp), inpaint_backend="mock",
                segment_backend="mock", critic_model="",
                first_run_setup=False, comfyui_dir="")
    base.update(kw)
    return Services(Settings(**base))


class OllamaLivenessTests(unittest.TestCase):
    def test_ollama_is_up_true_and_false(self):
        # urlopen is monkeypatched via the module import in llm.
        import app.core.llm as llm

        orig = llm.urllib.request.urlopen
        try:
            llm.urllib.request.urlopen = lambda *a, **k: _Ctx()
            self.assertTrue(ollama_is_up("http://127.0.0.1:11434/v1"))

            def boom(*a, **k):
                raise urllib.error.URLError("refused")
            llm.urllib.request.urlopen = boom
            self.assertFalse(ollama_is_up("http://127.0.0.1:11434/v1"))
        finally:
            llm.urllib.request.urlopen = orig


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TriageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.services = _services(self.tmp.name)
        self.services.scout.llm = DeadLLM()
        self.services.start()

    def tearDown(self):
        self.services.stop()
        self.tmp.cleanup()

    def test_triage_logs_choice_and_is_failsafe(self):
        # A dead planner LLM must not break the render; triage just skips.
        self.services.comfy = FakeComfy()
        self.services.workflow_ai.llm = ScriptedLLM([json.dumps(GRAPH)])
        self.services.llm = DeadLLM()  # triage LLM unavailable
        job = self.services.queue.enqueue(
            "workflow", {"task": "generate", "prompt": "a cat"})
        done = self.services.queue.wait_for(job.id, timeout=20)
        self.assertEqual(done.state.value, "completed")
        logs = " ".join(e["msg"] for e in done.logs)
        self.assertIn("triage skipped", logs)

    def test_triage_fetches_only_registry_models(self):
        from app.core.llm import LLMReply

        class TriageLLM:
            source = "local"

            def complete(self, system, prompt, max_tokens=4096):
                # names one real model + one hallucinated one
                return LLMReply(json.dumps({
                    "workflow": "generate",
                    "needed_models": ["sd15-inpaint", "totally-made-up"],
                    "reason": "sd15 fits"}), "fake", "local")

        self.services.comfy = FakeComfy()
        self.services.workflow_ai.llm = ScriptedLLM([json.dumps(GRAPH)])
        self.services.llm = TriageLLM()
        fetched = []
        self.services._ensure_model = lambda name, job: fetched.append(name)
        # sd15-inpaint not ready in this temp env, so triage will try to fetch it
        job = self.services.queue.enqueue(
            "workflow", {"task": "generate", "prompt": "a portrait"})
        done = self.services.queue.wait_for(job.id, timeout=20)
        self.assertEqual(done.state.value, "completed")
        self.assertIn("sd15-inpaint", fetched)
        self.assertNotIn("totally-made-up", fetched)  # hallucination filtered


class DiagnoseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.services = _services(self.tmp.name)

    def tearDown(self):
        self.services.stop()
        self.tmp.cleanup()

    def test_diagnose_records_failure_to_experience(self):
        class Job:
            logs = []

            def log(self, level, msg):
                Job.logs.append(msg)

        self.services.llm = DeadLLM()  # no diagnosis text, but must not crash
        self.services._diagnose_and_record(Job(), "video", "waves", "node X missing")
        rows = self.services.db.query(
            "SELECT * FROM workflow_memory WHERE task='video'")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["success"], 0)


class DiscoverApproveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.services = _services(self.tmp.name)
        self.services.start()

    def tearDown(self):
        self.services.stop()
        self.tmp.cleanup()

    def _good_candidate(self):
        return {"name": "twopass_demo", "task": "generate",
                "description": "two-pass demo", "required_models": [],
                "graph": GRAPH}

    def test_discover_validates_and_offers_candidates(self):
        from app.core.llm import LLMReply

        class AuthorLLM:
            source = "local"

            def complete(self, system, prompt, max_tokens=4096):
                bad = {"name": "broken", "task": "generate",
                       "graph": {"1": {"class_type": "Nope", "inputs": {}}}}
                good = {"name": "twopass_demo", "task": "generate",
                        "description": "demo", "required_models": [],
                        "graph": GRAPH}
                return LLMReply(json.dumps([bad, good]), "fake", "local")

        self.services.llm = AuthorLLM()
        job = self.services.queue.enqueue("discover", {})
        done = self.services.queue.wait_for(job.id, timeout=20)
        self.assertEqual(done.state.value, "completed")
        cands = done.result["candidates"]
        self.assertEqual(len(cands), 1)  # invalid one dropped
        self.assertEqual(cands[0]["name"], "twopass_demo")
        self.assertNotIn("graph", cands[0])  # graph kept server-side only

        cand_id = cands[0]["id"]
        saved = self.services.save_candidate(cand_id, live_test=False)
        self.assertEqual(saved["saved"], "twopass_demo")
        self.assertEqual(saved["verified"], "structural")
        # it's now in the library
        names = [t["template"] for t in self.services.workflows.list_all()]
        self.assertIn("twopass_demo", names)

    def test_approve_unknown_candidate_fails(self):
        from app.core.jobs import PermanentError
        with self.assertRaises(PermanentError):
            self.services.save_candidate("nope")


if __name__ == "__main__":
    unittest.main()
