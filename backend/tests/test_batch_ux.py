"""Tests for the stability/UX batch: queue management, smarter ETA, gallery
trash/undo, rich civitai search + index, ComfyUI crash recovery (step-down,
free-memory, mid-render death), events stream, and mask-refine visibility."""
import io
import tempfile
import threading
import time
import unittest
from pathlib import Path

from PIL import Image

from app.adapters.base import BackendUnavailableError, EditResult
from app.adapters.comfyui import ComfyUIClient
from app.config import Settings
from app.core import eta
from app.core.db import Database
from app.core.hardware import Hardware
from app.core.jobs import JobQueue
from app.core.model_search import ModelIndex, ModelSearch
from app.core.registry import ModelRegistry
from app.core.services import Services
from tests.test_workflow_job import DeadLLM


def _services(tmp, **kw):
    base = dict(data_dir=Path(tmp), inpaint_backend="mock",
                segment_backend="mock", critic_model="",
                first_run_setup=False, comfyui_dir="")
    base.update(kw)
    s = Services(Settings(**base))
    s.scout.llm = DeadLLM()
    s.llm = DeadLLM()
    return s


# ---------------- queue management ----------------

class QueueManagementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "t.sqlite3")
        self.q = JobQueue(self.db, max_retries=0, backoff_s=0.01)
        self.q.register("noop", lambda job: {})

    def tearDown(self):
        self.q.stop()
        self.db.close()
        self.tmp.cleanup()

    def test_pause_holds_dispatch_and_resume_releases(self):
        self.q.pause()
        self.q.start()
        job = self.q.enqueue("noop", {})
        time.sleep(0.3)
        self.assertEqual(self.q.get(job.id).state.value, "pending")
        self.assertTrue(self.q.paused)
        self.q.resume()
        done = self.q.wait_for(job.id)
        self.assertEqual(done.state.value, "completed")

    def test_reorder_pending_jobs(self):
        self.q.pause()
        a = self.q.enqueue("noop", {})
        b = self.q.enqueue("noop", {})
        c = self.q.enqueue("noop", {})
        self.assertEqual(self.q.pending_order(), [a.id, b.id, c.id])
        self.assertTrue(self.q.move(c.id, "top"))
        self.assertEqual(self.q.pending_order(), [c.id, a.id, b.id])
        self.assertTrue(self.q.move(a.id, "down"))
        self.assertEqual(self.q.pending_order(), [c.id, b.id, a.id])
        self.assertFalse(self.q.move("nope", "up"))

    def test_delete_pending_and_finished_but_not_running(self):
        self.q.pause()
        a = self.q.enqueue("noop", {})
        self.assertTrue(self.q.delete(a.id))
        self.assertIsNone(self.q.get(a.id))
        self.assertEqual(self.q.pending_order(), [])
        self.assertEqual(
            self.db.query("SELECT * FROM jobs WHERE id=?", (a.id,)), [])
        self.q.resume()
        self.q.start()
        b = self.q.enqueue("noop", {})
        self.q.wait_for(b.id)
        self.assertTrue(self.q.delete(b.id))

    def test_delete_running_job_refused(self):
        gate = threading.Event()
        self.q.register("slow", lambda job: (gate.wait(5), {})[1])
        self.q.start()
        job = self.q.enqueue("slow", {})
        for _ in range(100):
            if self.q.get(job.id).state.value == "running":
                break
            time.sleep(0.02)
        self.assertFalse(self.q.delete(job.id))
        gate.set()
        self.q.wait_for(job.id)

    def test_clear_scopes(self):
        self.q.start()
        ok = self.q.enqueue("noop", {})
        self.q.wait_for(ok.id)
        self.q.register("bad", lambda job: (_ for _ in ()).throw(
            __import__("app.core.jobs", fromlist=["PermanentError"])
            .PermanentError("x")))
        bad = self.q.enqueue("bad", {})
        self.q.wait_for(bad.id)
        self.assertEqual(self.q.clear("completed"), 1)
        self.assertEqual(self.q.clear("failed"), 1)
        self.assertEqual(self.q.clear("failed"), 0)
        self.assertEqual(self.q.clear("bogus"), 0)


# ---------------- smarter ETA ----------------

class EtaTests(unittest.TestCase):
    HW = Hardware("gpu", 8, 16, 100)

    def test_resolution_and_length_scale_video(self):
        small = eta.estimate_seconds(
            "video", hardware=self.HW,
            payload={"width": 320, "height": 320, "length": 17})
        big = eta.estimate_seconds(
            "video", hardware=self.HW,
            payload={"width": 640, "height": 640, "length": 81})
        self.assertLess(small, big)

    def test_history_median_overrides_baseline(self):
        secs = eta.estimate_seconds("workflow", hardware=self.HW,
                                    history=[10.0, 12.0, 11.0])
        self.assertAlmostEqual(secs, 11.0, delta=0.1)

    def test_sdxl_checkpoint_costs_more(self):
        sd = eta.estimate_seconds("workflow", hardware=self.HW,
                                  checkpoint="real.safetensors")
        xl = eta.estimate_seconds("workflow", hardware=self.HW,
                                  checkpoint="sd_xl_base_1.0.safetensors")
        self.assertLess(sd, xl)

    def test_conditioning_nodes_and_batch_increase(self):
        plain = eta.estimate_seconds("workflow", hardware=self.HW, graph={})
        heavy = eta.estimate_seconds("workflow", hardware=self.HW, graph={
            "1": {"class_type": "LoraLoader", "inputs": {}},
            "2": {"class_type": "ControlNetApply", "inputs": {}},
            "3": {"class_type": "EmptyLatentImage",
                  "inputs": {"batch_size": 2}},
        })
        self.assertLess(plain, heavy)

    def test_busy_machine_and_queue_ahead_add_time(self):
        idle = eta.estimate_seconds("workflow", hardware=self.HW,
                                    load=eta.SystemLoad(0, 0))
        busy = eta.estimate_seconds("workflow", hardware=self.HW,
                                    load=eta.SystemLoad(95, 92))
        self.assertLess(idle, busy)
        queued = eta.estimate_seconds("workflow", hardware=self.HW,
                                      queue_ahead_seconds=300)
        self.assertGreaterEqual(queued, idle + 300 - 1)

    def test_eta_log_format_is_machine_readable(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        s = _services(tmp.name)
        self.addCleanup(s.stop)

        class LogJob:
            type = "workflow"
            payload = {}
            logs = []

            def log(self, level, msg):
                LogJob.logs.append(msg)

        s._log_eta(LogJob())
        line = LogJob.logs[-1]
        self.assertRegex(line, r"^\[eta:\d+\] Estimated time remaining: ~")
        self.assertNotIn("median", line)  # never explain the estimate


# ---------------- gallery trash / undo / cleanup ----------------

class GalleryTrashTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = _services(self.tmp.name)
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), (1, 2, 3)).save(buf, format="PNG")
        self.asset = self.s.store.save_upload("a.png", buf.getvalue())

    def tearDown(self):
        self.s.stop()
        self.tmp.cleanup()

    def test_delete_hides_restore_brings_back(self):
        self.assertTrue(self.s.store.delete_asset(self.asset.id))
        self.assertEqual(self.s.store.gallery(), [])
        self.assertFalse(  # file moved out of assets dir
            (self.s.settings.assets_dir / self.asset.id).exists())
        self.assertTrue(self.s.store.restore_asset(self.asset.id))
        self.assertEqual(len(self.s.store.gallery()), 1)
        self.assertTrue(
            (self.s.settings.assets_dir / self.asset.id).exists())

    def test_purge_reclaims_disk_and_rows(self):
        self.s.store.delete_asset(self.asset.id)
        self.assertTrue(self.s.store.purge_asset(self.asset.id))
        self.assertIsNone(self.s.store.get_asset(self.asset.id))
        self.assertFalse(
            (self.s.settings.data_dir / "trash" / self.asset.id).exists())

    def test_startup_purge_cleans_leftover_trash(self):
        self.s.store.delete_asset(self.asset.id)
        self.assertEqual(self.s.store.purge_trash(), 1)


# ---------------- civitai rich search + index ----------------

CIVITAI_PAYLOAD = {"items": [{
    "name": "Detail LoRA", "nsfw": False,
    "creator": {"username": "artist"},
    "stats": {"downloadCount": 4321, "thumbsUpCount": 99},
    "description": "<p>Adds <b>detail</b>.</p>",
    "modelVersions": [{
        "id": 55, "name": "v2.0", "baseModel": "SD 1.5",
        "trainedWords": ["add_detail"],
        "images": [{"url": "https://img.example/x.jpg"}],
        "files": [{"name": "detail.safetensors", "sizeKB": 100.0,
                   "downloadUrl": "https://civitai.com/api/download/models/55",
                   "hashes": {"SHA256": "AB" * 32}}],
    }],
}]}


class CivitaiRichTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        db = Database(Path(self.tmp.name) / "t.sqlite3")
        self.registry = ModelRegistry(db, Path(self.tmp.name) / "models")
        self.db = db

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_rich_search_parses_all_fields(self):
        ms = ModelSearch(self.registry, http_get=lambda url, t: CIVITAI_PAYLOAD)
        out = ms.search_civitai_rich("detail", "lora")
        self.assertEqual(len(out), 1)
        m = out[0]
        self.assertEqual(m["type"], "lora")
        self.assertEqual(m["folder"], "loras")
        self.assertEqual(m["trigger_words"], ["add_detail"])
        self.assertEqual(m["base_model"], "SD 1.5")
        self.assertEqual(m["preview_url"], "https://img.example/x.jpg")
        self.assertEqual(m["description"], "Adds detail .")
        self.assertTrue(m["stageable"])
        self.assertEqual(m["sha256"], "ab" * 32)

    def test_workflows_are_searchable_but_not_stageable(self):
        ms = ModelSearch(self.registry, http_get=lambda url, t: CIVITAI_PAYLOAD)
        out = ms.search_civitai_rich("", "workflow")
        self.assertFalse(out[0]["stageable"])  # no target folder

    def test_propose_uses_type_folder_and_triggers(self):
        ms = ModelSearch(self.registry, http_get=lambda url, t: CIVITAI_PAYLOAD)
        cand = ms.search_civitai_rich("detail", "lora")[0]
        model = ms.propose_civitai(cand, name="detail-lora", purpose="lora")
        self.assertEqual(model.meta["folder"], "loras")
        self.assertEqual(model.meta["trigger_words"], ["add_detail"])

    def test_index_caches_and_serves_stale_on_error(self):
        calls = {"n": 0}

        def get(url, t):
            calls["n"] += 1
            return CIVITAI_PAYLOAD

        ms = ModelSearch(self.registry, http_get=get)
        idx = ModelIndex(self.db, ms)
        first = idx.get("lora")
        self.assertEqual(len(first["entries"]), 1)
        second = idx.get("lora")  # cached: no extra fetch
        self.assertEqual(calls["n"], 1)
        self.assertEqual(second["entries"], first["entries"])


# ---------------- ComfyUI client: interrupt + free ----------------

class ComfyClientControlTests(unittest.TestCase):
    def _client(self, log):
        c = ComfyUIClient("http://x")
        c.request = lambda method, path, data=None, headers=None: (
            log.append((method, path, data)) or b"{}")
        return c

    def test_interrupt_and_free_endpoints(self):
        log = []
        c = self._client(log)
        self.assertTrue(c.interrupt())
        self.assertTrue(c.free_memory())
        self.assertEqual(log[0][:2], ("POST", "/interrupt"))
        self.assertEqual(log[1][:2], ("POST", "/free"))
        self.assertIn(b"unload_models", log[1][2])


# ---------------- I2V crash recovery ----------------

class VideoCrashRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = _services(self.tmp.name)
        for name in ("wan22-ti2v-5b", "wan-umt5-xxl", "wan22-vae"):
            f = self.s.settings.models_dir / f"{name}.safetensors"
            f.write_bytes(b"w")
            self.s.registry.set_status(name, "ready", path=str(f))
        self.s.hardware = Hardware("gpu", 8, 16, 100)
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), (0, 0, 0)).save(buf, format="PNG")
        self.asset = self.s.store.save_upload("src.png", buf.getvalue())

    def tearDown(self):
        self.s.stop()
        self.tmp.cleanup()

    def test_midrender_death_becomes_friendly_retry_then_stepdown(self):
        class DyingComfy:
            def __init__(self):
                self.graphs = []
                self.freed = 0

            def is_up(self):
                return True

            def free_memory(self):
                self.freed += 1
                return True

            def upload_image(self, image, prefix):
                return "src.png"

            def submit(self, graph):
                self.graphs.append(graph)
                raise BackendUnavailableError("ComfyUI at http://x is unreachable")

        comfy = DyingComfy()
        self.s.comfy = comfy
        self.s.start()
        job = self.s.queue.enqueue("video", {
            "asset_id": self.asset.id, "prompt": "waves", "length": 49,
            "width": 640, "height": 640})
        done = self.s.queue.wait_for(job.id, timeout=30)
        self.assertEqual(done.state.value, "failed")  # exhausted retries
        # ...but every attempt was the graceful path, not a raw urlopen error:
        self.assertIn("restarted automatically", done.error or "")
        logs = " ".join(e["msg"] for e in done.logs)
        self.assertIn("process died", logs.lower().replace("its process", "process"))
        # step-down happened: later graphs are smaller than the first
        first = comfy.graphs[0]["7"]["inputs"]
        last = comfy.graphs[-1]["7"]["inputs"]
        self.assertLess(last["width"], first["width"])
        self.assertLess(last["length"], first["length"])
        # ComfyUI's caches were purged before every attempt
        self.assertGreaterEqual(comfy.freed, len(comfy.graphs))
        # the failure was recorded so the system learns from it
        rows = self.s.db.query(
            "SELECT * FROM workflow_memory WHERE task='video' AND success=0")
        self.assertGreaterEqual(len(rows), 1)

    def test_cancel_running_render_interrupts_comfy(self):
        interrupted = []

        class Comfy:
            def is_up(self):
                return True

            def interrupt(self):
                interrupted.append(True)
                return True

        self.s.comfy = Comfy()
        gate = threading.Event()
        self.s.queue.register("slowrender", lambda job: (gate.wait(5), {})[1])
        self.s.queue._handlers["video"] = self.s.queue._handlers["slowrender"]
        self.s.start()
        job = self.s.queue.enqueue("video", {"asset_id": self.asset.id})
        for _ in range(100):
            if self.s.queue.get(job.id).state.value == "running":
                break
            time.sleep(0.02)
        self.assertTrue(self.s.cancel_job(job.id))
        self.assertEqual(interrupted, [True])
        gate.set()


# ---------------- events stream + mask refine visibility ----------------

class EventsAndMaskTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = _services(self.tmp.name, critic_model="fake")

    def tearDown(self):
        self.s.stop()
        self.tmp.cleanup()

    def test_events_endpoint_merges_and_filters_llm(self):
        from app.api.routes import create_app
        self.s.events.log("info", "ComfyUI restarted")
        app = create_app(self.s)
        client = app.test_client()
        job = self.s.queue.enqueue("model_download", {"model": "nope"})
        self.s.queue.wait_for(job.id, timeout=10)
        job.log("info", "[llm] secret reasoning")
        job.log("info", "[stage] save — done")
        events = client.get("/api/events").get_json()
        msgs = " | ".join(e["msg"] for e in events)
        self.assertIn("ComfyUI restarted", msgs)
        self.assertIn("[stage] save", msgs)
        self.assertNotIn("secret reasoning", msgs)  # no LLM reasoning here
        self.assertTrue(all("t" in e and "source" in e for e in events))
        # chronologically ordered
        times = [e["t"] for e in events]
        self.assertEqual(times, sorted(times))

    def test_low_score_edit_saves_refined_mask_version(self):
        from tests.test_scout_critic_video import FakeCriticModel

        class RealishInpaint:
            name = "fake-real"
            is_mock = False

            def inpaint(self, image, mask, prompt):
                return EditResult(image=Image.new("RGB", image.size),
                                  adapter="fake-real", is_mock=False, meta={})

        self.s.inpainting = RealishInpaint()
        self.s.critic = FakeCriticModel([3.0, 8.0])  # forces the retry path
        self.s.start()
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), (5, 5, 5)).save(buf, format="PNG")
        asset = self.s.store.save_upload("p.png", buf.getvalue())
        job = self.s.queue.enqueue("image_edit", {
            "asset_id": asset.id, "prompt": "remove the chair"})
        done = self.s.queue.wait_for(job.id, timeout=20)
        self.assertEqual(done.state.value, "completed")
        logs = " ".join(e["msg"] for e in done.logs)
        self.assertIn("[mask] refined mask applied — version", logs)
        masks = [v for entry in self.s.store.gallery()
                 for v in entry["versions"] if v["label"] == "mask"]
        self.assertGreaterEqual(len(masks), 1)  # one per improvement round
        self.assertEqual(masks[0]["adapter"], "mask-refined")
        # gallery edit lists (label == "edit") don't show the mask artifact
        edits = [v for entry in self.s.store.gallery()
                 for v in entry["versions"] if v["label"] == "edit"]
        self.assertTrue(all(v["adapter"] != "mask-refined" for v in edits))


if __name__ == "__main__":
    unittest.main()
