"""Offline tests for the model scout, the vision critic, and the video job."""
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.config import Settings
from app.core.critic import CriticUnavailable, Critique, ImageCritic
from app.core.db import Database
from app.core.llm import LLMReply, LLMUnavailableError
from app.core.model_scout import ModelScout
from app.core.model_search import ModelSearch
from app.core.registry import ModelRegistry
from app.core.services import Services


class OneShotLLM:
    source = "local"

    def __init__(self, text=None, error=None):
        self.text, self.error = text, error

    def complete(self, system, prompt, max_tokens=4096):
        if self.error:
            raise self.error
        return LLMReply(self.text, "fake", "local")


class ScoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        db = Database(Path(self.tmp.name) / "t.sqlite3")
        self.registry = ModelRegistry(db, Path(self.tmp.name) / "models")
        self.downloads = []

        class FakeDownloader:
            def download(inner, name, progress=None):
                self.downloads.append(name)

        self.downloader = FakeDownloader()

    def tearDown(self):
        self.tmp.cleanup()

    def _scout(self, llm, routes=None):
        from app.core.trust import TrustJudge
        search = ModelSearch(self.registry, http_get=lambda url, t: next(
            (data for frag, data in (routes or {}).items() if frag in url), []))
        # Rules-only trust here (judge behavior has its own tests); fixture
        # repos carry enough adoption to pass the rule verdict.
        return ModelScout(llm, search, self.downloader, self.registry,
                          trust=TrustJudge(None))

    def test_picks_installed_checkpoint(self):
        scout = self._scout(OneShotLLM(json.dumps({"use": "real.safetensors"})))
        d = scout.choose("a cat", "generate", ["real.safetensors"], True)
        self.assertEqual(d.checkpoint, "real.safetensors")
        self.assertIsNone(d.downloaded)

    def test_unknown_choice_falls_back(self):
        scout = self._scout(OneShotLLM(json.dumps({"use": "nope.safetensors"})))
        d = scout.choose("a cat", "generate", ["a.safetensors"], True)
        self.assertEqual(d.checkpoint, "a.safetensors")

    def test_llm_down_falls_back(self):
        scout = self._scout(OneShotLLM(error=LLMUnavailableError("down")))
        d = scout.choose("a cat", "generate", ["a.safetensors"], True)
        self.assertEqual(d.checkpoint, "a.safetensors")

    def test_search_downloads_verified_file(self):
        routes = {
            "/api/models?": [{"id": "author/photo-model", "downloads": 500,
                              "likes": 1, "pipeline_tag": None, "gated": False}],
            "/tree/main": [{"type": "file", "path": "photo.safetensors",
                            "size": 1000, "lfs": {"oid": "c" * 64}}],
        }
        scout = self._scout(
            OneShotLLM(json.dumps({"search": "photo model", "reason": "style"})),
            routes)
        d = scout.choose("hyperreal portrait", "generate", ["inpaint-only.safetensors"], True)
        self.assertEqual(d.checkpoint, "photo.safetensors")
        self.assertEqual(self.downloads, ["scout-photo-model"])
        self.assertIsNotNone(self.registry.get("scout-photo-model"))

    def test_search_skips_unverified_and_falls_back(self):
        routes = {
            "/api/models?": [{"id": "a/x", "downloads": 1, "likes": 0,
                              "pipeline_tag": None, "gated": False}],
            "/tree/main": [{"type": "file", "path": "x.safetensors",
                            "size": 1000}],  # no LFS sha → not eligible
        }
        scout = self._scout(OneShotLLM(json.dumps({"search": "x"})), routes)
        d = scout.choose("p", "generate", ["fallback.safetensors"], True)
        self.assertEqual(d.checkpoint, "fallback.safetensors")
        self.assertEqual(self.downloads, [])


class CivitaiTests(unittest.TestCase):
    def test_search_civitai_extracts_hashed_safetensors(self):
        payload = {"items": [{
            "name": "Photo Model XL", "nsfw": False,
            "creator": {"username": "gooduser"},
            "stats": {"downloadCount": 12345},
            "modelVersions": [{
                "id": 777,
                "files": [
                    {"name": "photo.safetensors", "sizeKB": 2048.0,
                     "downloadUrl": "https://civitai.com/api/download/models/777",
                     "hashes": {"SHA256": "AB" * 32}},
                    {"name": "photo.ckpt", "sizeKB": 2048.0,
                     "hashes": {"SHA256": "CD" * 32}},  # pickle: skipped
                ],
            }],
        }]}
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        registry = ModelRegistry(Database(Path(tmp.name) / "t.sqlite3"),
                                 Path(tmp.name) / "models")
        ms = ModelSearch(registry, http_get=lambda url, t: payload)
        out = ms.search_civitai("photo model")
        self.assertEqual(len(out), 1)
        c = out[0]
        self.assertEqual(c["filename"], "photo.safetensors")
        self.assertEqual(c["sha256"], "ab" * 32)
        self.assertEqual(c["size_bytes"], 2048 * 1024)
        model = ms.propose_civitai(c, name="civ-photo", purpose="test")
        self.assertIn("civitai.com", model.url)
        self.assertEqual(model.sha256, "ab" * 32)
        self.assertEqual(model.meta["folder"], "checkpoints")


class CriticTests(unittest.TestCase):
    def _img(self):
        return Image.new("RGB", (32, 32), (100, 120, 140))

    def test_parses_score_and_issues(self):
        def post(url, payload, timeout):
            assert url.endswith("/api/chat")
            assert payload["messages"][0]["images"]
            return {"message": {"content": json.dumps(
                {"score": 4, "issues": ["visible seam", "warped hand"]})}}

        crit = ImageCritic("http://127.0.0.1:11434/v1", "llava", http_post=post)
        c = crit.critique(self._img(), "a portrait")
        self.assertEqual(c.score, 4.0)
        self.assertIn("visible seam", c.issues)
        self.assertIn("realism 4/10", c.summary())

    def test_salvages_bare_number(self):
        def post(url, payload, timeout):
            return {"message": {"content": "I'd say 7 out of 10 overall."}}

        crit = ImageCritic("http://127.0.0.1:11434/v1", "llava", http_post=post)
        self.assertEqual(crit.critique(self._img(), "x").score, 7.0)

    def test_unreachable_raises_unavailable(self):
        def post(url, payload, timeout):
            raise OSError("refused")

        crit = ImageCritic("http://127.0.0.1:11434/v1", "llava", http_post=post)
        with self.assertRaises(CriticUnavailable):
            crit.critique(self._img(), "x")


class FakeCriticModel:
    """Stands in for ImageCritic inside Services."""

    def __init__(self, scores):
        self.scores = list(scores)

    def critique(self, image, prompt):
        return Critique(score=self.scores.pop(0), issues=["fake issue"], model="fake")


class FakeComfyVideo:
    def __init__(self):
        self.uploaded = []
        self.submitted = None

    def is_up(self):
        return True

    def upload_image(self, image, prefix):
        self.uploaded.append(prefix)
        return f"{prefix}.png"

    def submit(self, graph):
        self.submitted = graph
        return "vid-1"

    def wait_for_output_file(self, prompt_id):
        return b"RIFFfakewebpdata", "promptforge_video_00001_.webp"


class VideoJobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.services = Services(Settings(
            data_dir=Path(self.tmp.name),
            inpaint_backend="mock", segment_backend="mock", critic_model="",
            first_run_setup=False, comfyui_dir=""))
        # Pretend the WAN models are downloaded and on disk.
        for name in ("wan22-ti2v-5b", "wan-umt5-xxl", "wan22-vae"):
            f = self.services.settings.models_dir / f"{name}.safetensors"
            f.write_bytes(b"w")
            self.services.registry.set_status(name, "ready", path=str(f))
        from app.core.hardware import Hardware
        self.services.hardware = Hardware("gpu", 8, 16, 100)  # mid tier
        self.services.comfy = FakeComfyVideo()
        self.services.start()

    def tearDown(self):
        self.services.stop()
        self.tmp.cleanup()

    def test_video_job_renders_and_saves_animation(self):
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (5, 5, 5)).save(buf, format="PNG")
        asset = self.services.store.save_upload("src.png", buf.getvalue())

        job = self.services.queue.enqueue("video", {
            "asset_id": asset.id, "prompt": "waves rolling", "length": 500})
        done = self.services.queue.wait_for(job.id, timeout=20)
        self.assertEqual(done.state.value, "completed")
        self.assertEqual(done.result["length"], 81)  # clamped
        saved = self.services.store.get_asset(done.result["asset_id"])
        self.assertTrue(saved.filename.endswith(".webp"))
        # The template got the prompt + uploaded image injected.
        g = self.services.comfy.submitted
        self.assertEqual(g["4"]["inputs"]["text"], "waves rolling")
        self.assertEqual(g["6"]["inputs"]["image"], "video_src.png")


class CommitExhaustionTests(unittest.TestCase):
    """OS error 1455 (Windows commit charge / paging file exhausted) is a
    machine setting, not a graph bug — it must fail fast with the real cure,
    never with 'update ComfyUI for the WAN nodes' advice or LLM repairs."""

    # The exact live failure (Dutch Windows), 2026-07-15.
    NL_1455 = ("ComfyUI execution failed: Het wisselbestand is te klein voor "
               "het voltooien van deze bewerking. (os error 1455)")
    EN_1455 = ("The paging file is too small for this operation to complete. "
               "(os error 1455)")

    def test_hint_recognizes_localized_1455(self):
        from app.core.services import commit_exhausted_hint
        for text in (self.NL_1455, self.EN_1455, "OSError: error 1455"):
            hint = commit_exhausted_hint(text)
            self.assertIsNotNone(hint, text)
            self.assertIn("paging file", hint)
            self.assertIn("Virtual memory", hint)
        self.assertIsNone(commit_exhausted_hint("node error on run 2"))
        self.assertIsNone(commit_exhausted_hint("CUDA out of memory"))

    def test_available_commit_probe_is_sane(self):
        import sys

        from app.core.hardware import available_commit_gb
        commit = available_commit_gb()
        if sys.platform == "win32":
            self.assertIsInstance(commit, float)
            self.assertGreater(commit, 0)
        else:
            self.assertIsNone(commit)

    def test_video_job_fails_fast_with_the_real_cure(self):
        from app.adapters.comfyui import WorkflowRuntimeError
        from tests.test_workflow_job import DeadLLM

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        services = Services(Settings(
            data_dir=Path(tmp.name),
            inpaint_backend="mock", segment_backend="mock", critic_model="",
            first_run_setup=False, comfyui_dir=""))
        # The render failure runs _diagnose_and_record, which asks the LLM to
        # explain the error. Unstubbed that is a real Ollama round trip, and
        # the test's runtime stops being about the code under test.
        services.llm = DeadLLM()
        self.addCleanup(services.stop)
        for name in ("wan22-ti2v-5b", "wan-umt5-xxl", "wan22-vae"):
            f = services.settings.models_dir / f"{name}.safetensors"
            f.write_bytes(b"w")
            services.registry.set_status(name, "ready", path=str(f))
        from app.core.hardware import Hardware
        services.hardware = Hardware("gpu", 8, 16, 100)

        class Comfy1455(FakeComfyVideo):
            def submit(self, graph):
                raise WorkflowRuntimeError(CommitExhaustionTests.NL_1455)

        services.comfy = Comfy1455()
        services.start()
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (5, 5, 5)).save(buf, format="PNG")
        asset = services.store.save_upload("src.png", buf.getvalue())
        job = services.queue.enqueue("video", {
            "asset_id": asset.id, "prompt": "waves rolling"})
        done = services.queue.wait_for(job.id, timeout=20)
        self.assertEqual(done.state.value, "failed")
        self.assertEqual(done.attempts, 1)  # permanent: no pointless retries
        self.assertIn("paging file", done.error)
        self.assertIn("Virtual memory", done.error)
        self.assertNotIn("WAN", done.error)  # the old, misleading advice


class CriticRetryTests(unittest.TestCase):
    """Workflow job: low realism triggers one strategy-change replan."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.services = Services(Settings(
            data_dir=Path(self.tmp.name),
            inpaint_backend="mock", segment_backend="mock",
            critic_model="fake", critic_retries=1, first_run_setup=False, comfyui_dir=""))
        self.services.start()

    def tearDown(self):
        self.services.stop()
        self.tmp.cleanup()

    def test_low_score_replans_and_keeps_better(self):
        from tests.test_workflow_job import GRAPH, DeadLLM, FakeComfy, ScriptedLLM
        self.services.comfy = FakeComfy()
        self.services.llm = DeadLLM()  # triage/diagnose LLM offline in tests
        self.services.critic = FakeCriticModel([3.0, 8.0])  # bad, then good
        # scout reply + two plans
        self.services.workflow_ai.llm = ScriptedLLM(
            [json.dumps(GRAPH), json.dumps(GRAPH)])
        self.services.scout.llm = OneShotLLM(
            json.dumps({"use": "real.safetensors"}))
        job = self.services.queue.enqueue(
            "workflow", {"task": "generate", "prompt": "a photo"})
        done = self.services.queue.wait_for(job.id, timeout=20)
        self.assertEqual(done.state.value, "completed")
        self.assertEqual(done.result["realism"], 8.0)
        logs = " ".join(entry["msg"] for entry in done.logs)
        self.assertIn("[stage] retry", logs)


if __name__ == "__main__":
    unittest.main()
