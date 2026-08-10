"""Tests: identity renders (movable avatar), download telemetry/auth,
job-history restore, and LLM-transparency logging."""
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.config import Settings
from app.core.db import Database
from app.core.jobs import JobQueue
from app.core.registry import DownloadError, ModelDownloader, ModelRegistry
from app.core.services import Services
from tests.test_workflow_job import GRAPH, DeadLLM, FakeComfy, ScriptedLLM


def _services(tmp, **kw):
    base = dict(data_dir=Path(tmp), inpaint_backend="mock",
                segment_backend="mock", critic_model="", first_run_setup=False, comfyui_dir="")
    base.update(kw)
    return Services(Settings(**base))


class FakeComfyIdentity:
    """Identity + video capable fake ComfyUI."""

    def __init__(self):
        self.submitted = []
        self.uploaded = []

    def is_up(self):
        return True

    def upload_image(self, image, prefix):
        self.uploaded.append(prefix)
        return f"{prefix}.png"

    def submit(self, graph):
        self.submitted.append(graph)
        return f"pid-{len(self.submitted)}"

    def wait_for_output(self, prompt_id):
        return Image.new("RGB", (16, 16), (90, 80, 70))

    def wait_for_output_file(self, prompt_id):
        # A REAL webp, as ComfyUI's SaveAnimatedWEBP produces — the saved
        # asset is now decode-validated (an animated .webp is an image
        # kind), so fake bytes would be rejected exactly as a corrupt
        # upload is.
        buf = io.BytesIO()
        Image.new("RGB", (16, 16), (40, 60, 80)).save(buf, format="WEBP")
        return buf.getvalue(), "avatar_video_00001_.webp"


class AvatarRenderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.services = _services(self.tmp.name)
        self.services.scout.llm = DeadLLM()
        self.services.llm = DeadLLM()
        # Identity + video models "already downloaded".
        for name in ("sdxl-base", "photomaker-v1", "wan22-ti2v-5b",
                     "wan-umt5-xxl", "wan22-vae"):
            f = self.services.settings.models_dir / f"{name}.safetensors"
            f.write_bytes(b"w")
            self.services.registry.set_status(name, "ready", path=str(f))
        from app.core.hardware import Hardware
        self.services.hardware = Hardware("gpu", 8, 16, 100)  # mid tier
        self.services.comfy = FakeComfyIdentity()
        self.services.start()

    def tearDown(self):
        self.services.stop()
        self.tmp.cleanup()

    def _avatar(self):
        buf = io.BytesIO()
        Image.new("RGB", (64, 96), (150, 120, 100)).save(buf, format="PNG")
        asset = self.services.store.save_upload("face.png", buf.getvalue())
        return self.services.store.create_avatar(
            "Test", [asset.id], [], asset.id, meta={"consent": True})

    def test_identity_render_uses_photomaker_trigger(self):
        avatar = self._avatar()
        job = self.services.queue.enqueue("avatar_render", {
            "avatar_id": avatar.id, "prompt": "on a mountain ridge"})
        done = self.services.queue.wait_for(job.id, timeout=20)
        self.assertEqual(done.state.value, "completed")
        self.assertEqual(done.result["avatar_id"], avatar.id)
        saved = self.services.store.get_asset(done.result["asset_id"])
        self.assertIsNotNone(saved)
        graph = self.services.comfy.submitted[0]
        encode = next(n for n in graph.values()
                      if n["class_type"] == "PhotoMakerEncode")
        self.assertIn("photomaker", encode["inputs"]["text"])
        self.assertIn("on a mountain ridge", encode["inputs"]["text"])
        self.assertEqual(encode["inputs"]["image"], ["3", 0])

    def test_identity_render_with_video(self):
        avatar = self._avatar()
        job = self.services.queue.enqueue("avatar_render", {
            "avatar_id": avatar.id, "prompt": "walking on a beach",
            "video": True, "length": 33})
        done = self.services.queue.wait_for(job.id, timeout=20)
        self.assertEqual(done.state.value, "completed")
        video = self.services.store.get_asset(done.result["video_asset_id"])
        self.assertTrue(video.filename.endswith(".webp"))

    def test_unknown_avatar_fails_permanently(self):
        job = self.services.queue.enqueue("avatar_render", {
            "avatar_id": "nope", "prompt": "x"})
        done = self.services.queue.wait_for(job.id, timeout=10)
        self.assertEqual(done.state.value, "failed")
        self.assertIn("Avatar not found", done.error)

    def test_avatar_job_saves_profile(self):
        buf = io.BytesIO()
        Image.new("RGB", (64, 96), (10, 20, 30)).save(buf, format="PNG")
        a = self.services.store.save_upload("p.png", buf.getvalue())

        class ComfyDown:
            def is_up(self):
                return False

        self.services.comfy = ComfyDown()
        job = self.services.queue.enqueue(
            "avatar", {"asset_ids": [a.id], "consent": True, "name": "Ada"})
        done = self.services.queue.wait_for(job.id, timeout=30)
        self.assertEqual(done.state.value, "completed")
        profile = self.services.store.get_avatar(done.result["avatar_id"])
        self.assertEqual(profile.name, "Ada")
        self.assertEqual(profile.source_assets, [a.id])
        self.assertEqual(profile.face_asset, a.id)


class TransparencyTests(unittest.TestCase):
    """The job log must show what the LLM is thinking."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.services = _services(self.tmp.name)
        self.services.scout.llm = DeadLLM()
        self.services.llm = DeadLLM()
        self.services.start()

    def tearDown(self):
        self.services.stop()
        self.tmp.cleanup()

    def test_workflow_logs_llm_plan_and_scout(self):
        self.services.comfy = FakeComfy()
        self.services.workflow_ai.llm = ScriptedLLM([json.dumps(GRAPH)])
        job = self.services.queue.enqueue(
            "workflow", {"task": "generate", "prompt": "a cat"})
        done = self.services.queue.wait_for(job.id, timeout=20)
        logs = " ".join(e["msg"] for e in done.logs)
        self.assertIn("[llm] plan:", logs)
        self.assertIn("Checkpoint[real.safetensors]", logs)
        self.assertIn("[llm] scout:", logs)  # scout narration is job-scoped

    def test_render_jobs_log_expected_time(self):
        self.services.comfy = FakeComfy()
        self.services.workflow_ai.llm = ScriptedLLM([json.dumps(GRAPH)])
        job = self.services.queue.enqueue(
            "workflow", {"task": "generate", "prompt": "a cat"})
        done = self.services.queue.wait_for(job.id, timeout=20)
        logs = " ".join(e["msg"] for e in done.logs)
        self.assertIn("Estimated time remaining:", logs)

    def test_forge_result_carries_generation_recipe(self):
        self.services.comfy = FakeComfy()
        self.services.workflow_ai.llm = ScriptedLLM([json.dumps(GRAPH)])
        job = self.services.queue.enqueue(
            "workflow", {"task": "generate", "prompt": "a cat"})
        done = self.services.queue.wait_for(job.id, timeout=20)
        self.assertEqual(done.state.value, "completed")
        recipe = done.result["recipe"]
        self.assertEqual(recipe["task"], "generate")
        self.assertIn("LLM-planned", recipe["workflow"])
        self.assertIn("local:fake-model", recipe["planned_by"])
        self.assertEqual(recipe["checkpoint"], "real.safetensors")
        self.assertEqual(recipe["nodes"], len(GRAPH))
        steps = [s["step"] for s in recipe["trail"]]
        self.assertIn("plan", steps)
        self.assertIn("render", steps)
        self.assertIn("save", steps)
        self.assertTrue(all("t" in s for s in recipe["trail"]))
        # the recipe is also persisted on the saved asset for later viewing
        asset = self.services.store.get_asset(done.result["asset_id"])
        self.assertEqual(asset.meta["recipe"]["task"], "generate")

    def test_human_time_formatting(self):
        self.assertEqual(Services._human_time(20), "20s")
        self.assertEqual(Services._human_time(300), "5 min")
        self.assertTrue(Services._human_time(3).endswith("s"))

    def test_estimate_uses_history_median_when_available(self):
        # With >=3 completed jobs of a type, the estimate is measured, not a
        # heuristic — this is what makes the ETA self-calibrate.
        for _ in range(3):
            self.services.comfy = FakeComfy()
            self.services.workflow_ai.llm = ScriptedLLM([json.dumps(GRAPH)])
            j = self.services.queue.enqueue(
                "workflow", {"task": "generate", "prompt": "x"})
            self.services.queue.wait_for(j.id, timeout=20)
        _, samples = self.services._estimate_seconds("workflow")
        self.assertGreaterEqual(samples, 3)

    def test_graph_summary_is_readable(self):
        summary = Services._graph_summary({
            "1": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": "m.safetensors"}},
            "2": {"class_type": "KSampler",
                  "inputs": {"steps": 30, "cfg": 7.0,
                             "sampler_name": "dpmpp_2m", "scheduler": "karras"}},
        })
        self.assertEqual(
            summary,
            "Checkpoint[m.safetensors] → KSampler(30 steps, cfg 7.0, "
            "dpmpp_2m/karras)")


class DownloadTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "t.sqlite3")
        self.registry = ModelRegistry(self.db, Path(self.tmp.name) / "models")

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_civitai_token_goes_into_query_not_registry(self):
        from app.core.registry import ModelInfo
        dl = ModelDownloader(self.registry, civitai_token="sekret")
        model = ModelInfo(name="c", purpose="x",
                          url="https://civitai.com/api/download/models/1")
        self.assertEqual(dl._fetch_url(model),
                         "https://civitai.com/api/download/models/1?token=sekret")
        self.assertNotIn("sekret", model.url)
        self.assertEqual(dl._scrub("boom sekret boom"), "boom *** boom")
        # non-civitai URLs are untouched
        model.url = "https://huggingface.co/a/b/resolve/main/f.safetensors"
        self.assertEqual(dl._fetch_url(model), model.url)

    def test_civitai_auth_failure_is_permanent_with_hint(self):
        tmp2 = tempfile.TemporaryDirectory()
        self.addCleanup(tmp2.cleanup)
        services = _services(tmp2.name)

        class Auth403Downloader:
            def download(inner, name, progress=None):
                services.registry.set_status(name, "failed")
                raise DownloadError(
                    f"Download of '{name}' failed: HTTP Error 403: Forbidden. "
                    "Civitai requires a (free) account API token for this "
                    "file: create one under civitai.com → Account settings → "
                    "API Keys and set PROMPTFORGE_CIVITAI_TOKEN before "
                    "launching.")

        from app.core.registry import ModelInfo
        services.registry.register(ModelInfo(
            name="civ-thing", purpose="x", sha256="a" * 64,
            url="https://civitai.com/api/download/models/9"))
        services.downloader = Auth403Downloader()
        services.start()
        self.addCleanup(services.stop)
        job = services.queue.enqueue("model_download", {"model": "civ-thing"})
        done = services.queue.wait_for(job.id, timeout=10)
        self.assertEqual(done.state.value, "failed")
        self.assertEqual(done.attempts, 1)  # permanent: no pointless retries
        self.assertIn("PROMPTFORGE_CIVITAI_TOKEN", done.error)


class HistoryRestoreTests(unittest.TestCase):
    """Job history must survive a backend restart."""

    def test_finished_jobs_reload_and_stale_running_marked_failed(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db = Database(Path(tmp.name) / "t.sqlite3")
        self.addCleanup(db.close)

        q1 = JobQueue(db)
        q1.register("noop", lambda job: {"ok": True})
        q1.start()
        done = q1.enqueue("noop", {})
        q1.wait_for(done.id)
        q1.stop()
        # Simulate a job that was mid-flight when the process died.
        db.execute(
            """INSERT INTO jobs (id, type, state, attempts, payload, logs,
                                 created_at, updated_at)
               VALUES ('deadbeef0001', 'noop', 'running', 1, '{}', '[]',
                       '2026-07-11T00:00:00.000+00:00',
                       '2026-07-11T00:00:00.000+00:00')""")

        q2 = JobQueue(db)
        restored = {j.id: j for j in q2.list()}
        self.assertIn(done.id, restored)
        self.assertEqual(restored[done.id].state.value, "completed")
        self.assertEqual(restored[done.id].result, {"ok": True})
        self.assertEqual(restored["deadbeef0001"].state.value, "failed")
        self.assertIn("restart", restored["deadbeef0001"].error)


if __name__ == "__main__":
    unittest.main()
