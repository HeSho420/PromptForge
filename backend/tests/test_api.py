import io
import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image

from app.api.routes import create_app
from app.config import Settings
from app.core.services import Services


def _png_bytes(size=(64, 48), color=(200, 150, 100)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class ApiIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # Tests run offline: explicitly select the mock backends. The LLM
        # endpoint MUST be dead too — the default points at the machine's
        # real Ollama, and whenever that happened to be running these jobs
        # cold-loaded a 7B model per planning call and blew every timeout
        # (the long-blamed "load flakes" were exactly this).
        settings = Settings(data_dir=Path(self.tmp.name),
                            inpaint_backend="mock", segment_backend="mock",
                            llm_url="http://127.0.0.1:9/v1",
                            critic_model="", first_run_setup=False, comfyui_dir="")
        self.services = Services(settings)
        app = create_app(self.services)
        app.testing = True
        self.client = app.test_client()

    def tearDown(self):
        self.services.stop()
        self.tmp.cleanup()

    def _upload(self) -> str:
        resp = self.client.post("/api/assets", data={
            "file": (io.BytesIO(_png_bytes()), "photo.png")})
        self.assertEqual(resp.status_code, 201)
        return resp.get_json()["id"]

    def _wait_job(self, job_id: str, timeout=30.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.client.get(f"/api/jobs/{job_id}").get_json()
            if job["state"] in ("completed", "failed", "cancelled"):
                return job
            time.sleep(0.05)
        self.fail("job did not finish")

    def test_health_reports_mock_adapter_honestly(self):
        data = self.client.get("/api/health").get_json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["inpaint_is_mock"])

    def test_the_polled_jobs_list_is_trimmed_but_detail_stays_complete(self):
        """Measured live: 100 history jobs weighed 4.3 MB per poll, 3.7 MB
        of it payload.mask_b64 nothing reads from a list. The LIST elides
        bulk fields and caps finished logs; the detail endpoint and
        ?full=1 keep everything."""
        asset_id = self._upload()
        big_mask = "data:image/png;base64," + "A" * 100_000
        resp = self.client.post("/api/edits", json={
            "asset_id": asset_id, "prompt": "change the sky",
            "mask_b64": big_mask})
        self.assertEqual(resp.status_code, 202)
        job_id = resp.get_json()["id"]
        self._wait_job(job_id)

        listed = next(j for j in self.client.get("/api/jobs").get_json()
                      if j["id"] == job_id)
        self.assertEqual(listed["payload"]["mask_b64"], "<elided from list>")
        self.assertLessEqual(len(listed["logs"]), 3)

        full_row = next(
            j for j in self.client.get("/api/jobs?full=1").get_json()
            if j["id"] == job_id)
        self.assertEqual(full_row["payload"]["mask_b64"], big_mask)

        detail = self.client.get(f"/api/jobs/{job_id}").get_json()
        self.assertEqual(detail["payload"]["mask_b64"], big_mask)
        self.assertGreater(len(detail["logs"]), 3)

    def test_full_edit_flow_creates_before_and_after(self):
        asset_id = self._upload()

        # auto mask preview
        resp = self.client.post("/api/masks/preview",
                                json={"asset_id": asset_id, "prompt": "change the sky to sunset"})
        self.assertEqual(resp.status_code, 200)
        preview = resp.get_json()
        self.assertTrue(preview["mask_b64"].startswith("data:image/png;base64,"))
        self.assertTrue(preview["is_mock"])

        # queue the edit with the (user-approved) mask
        resp = self.client.post("/api/edits", json={
            "asset_id": asset_id, "prompt": "change the sky to sunset",
            "mask_b64": preview["mask_b64"]})
        self.assertEqual(resp.status_code, 202)
        job = self._wait_job(resp.get_json()["id"])
        self.assertEqual(job["state"], "completed")
        self.assertTrue(job["result"]["is_mock"])

        # gallery shows original + edit versions, both downloadable
        gallery = self.client.get("/api/gallery").get_json()
        entry = next(g for g in gallery if g["asset"]["id"] == asset_id)
        labels = [v["label"] for v in entry["versions"]]
        self.assertIn("original", labels)
        self.assertIn("edit", labels)
        edit = next(v for v in entry["versions"] if v["label"] == "edit")
        self.assertTrue(edit["meta"]["is_mock"])
        file_resp = self.client.get(f"/api/versions/{edit['id']}/file")
        self.assertEqual(file_resp.status_code, 200)

    def test_continue_from_result_promotes_the_edit_version(self):
        """'Continue from this result': after promoting, the asset's working
        file IS the rendered version — the next edit/mask/video builds on it,
        and the stale scene-graph cache is dropped."""
        asset_id = self._upload()
        resp = self.client.post("/api/edits", json={
            "asset_id": asset_id, "prompt": "remove the chair"})
        job = self._wait_job(resp.get_json()["id"])
        self.assertEqual(job["state"], "completed")
        version_id = job["result"]["version_id"]

        self.services._scene_cache[asset_id] = {"scene": "stale", "objects": []}
        resp = self.client.post(f"/api/versions/{version_id}/promote")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["id"], asset_id)
        self.assertNotIn(asset_id, self.services._scene_cache)

        asset = self.services.store.get_asset(asset_id)
        version = self.services.store.get_version(version_id)
        self.assertEqual(asset.path, version.path)
        served = self.client.get(f"/api/assets/{asset_id}/file").data
        self.assertEqual(served, Path(version.path).read_bytes())

    def test_promote_is_reversible_via_the_original_version(self):
        asset_id = self._upload()
        resp = self.client.post("/api/edits", json={
            "asset_id": asset_id, "prompt": "remove the chair"})
        job = self._wait_job(resp.get_json()["id"])
        self.client.post(f"/api/versions/{job['result']['version_id']}/promote")

        gallery = self.client.get("/api/gallery").get_json()
        entry = next(g for g in gallery if g["asset"]["id"] == asset_id)
        original = next(v for v in entry["versions"] if v["label"] == "original")
        resp = self.client.post(f"/api/versions/{original['id']}/promote")
        self.assertEqual(resp.status_code, 200)
        asset = self.services.store.get_asset(asset_id)
        self.assertEqual(asset.path,
                         self.services.store.get_version(original["id"]).path)

    def test_promote_refuses_masks_and_unknown_versions(self):
        self.assertEqual(
            self.client.post("/api/versions/doesnotexist/promote").status_code, 404)
        asset_id = self._upload()
        p = self.services.store.new_version_path(asset_id)
        Path(p).write_bytes(_png_bytes())
        mask_v = self.services.store.add_aux_version(
            asset_id, str(p), "auto mask", "test")
        self.assertEqual(
            self.client.post(f"/api/versions/{mask_v.id}/promote").status_code, 404)

    def test_make_n_images_queues_n_sequential_renders(self):
        """'make 3 images of X' queues 3 separate workflow jobs (the
        hardware rarely batches) and reports the batch in the response."""
        self.client.post("/api/queue/pause")  # keep them pending to count
        resp = self.client.post("/api/workflows/run", json={
            "task": "generate", "prompt": "make 3 images of a fox"})
        self.assertEqual(resp.status_code, 202)
        body = resp.get_json()
        self.assertEqual(body["batch_count"], 3)
        self.assertEqual(len(body["batch_job_ids"]), 3)
        jobs = self.client.get("/api/jobs").get_json()
        batch = [j for j in jobs if j["id"] in body["batch_job_ids"]]
        self.assertEqual(len(batch), 3)
        for j in batch:
            self.assertEqual(j["payload"]["prompt"], "a fox")  # count removed

    def test_quick_draft_toggle_rides_the_payload(self):
        """The Studio's Quick draft toggle sends draft:true — it lands in
        the generate payload; an unflagged request carries nothing."""
        self.client.post("/api/queue/pause")
        with_flag = self.client.post("/api/workflows/run", json={
            "task": "generate", "prompt": "a red barn",
            "draft": True}).get_json()["id"]
        without = self.client.post("/api/workflows/run", json={
            "task": "generate", "prompt": "a red barn"}).get_json()["id"]
        jobs = {j["id"]: j for j in self.client.get("/api/jobs").get_json()}
        self.assertTrue(jobs[with_flag]["payload"].get("draft"))
        self.assertNotIn("draft", jobs[without]["payload"])

    def test_remove_background_delivers_a_transparent_png(self):
        """'Remove the background' is a cutout DELIVERABLE (alpha PNG),
        never a repaint — the version stored on disk carries real
        transparency."""
        import io as _io

        from PIL import Image as PILImage
        asset_id = self._upload()
        resp = self.client.post("/api/edits", json={
            "asset_id": asset_id, "prompt": "remove the background"})
        job = self._wait_job(resp.get_json()["id"])
        self.assertEqual(job["state"], "completed")
        self.assertEqual(job["result"]["route"], "cutout")
        self.assertEqual(job["result"]["adapter"], "birefnet-cutout")
        f = self.client.get(
            f"/api/versions/{job['result']['version_id']}/file")
        self.assertEqual(f.status_code, 200)
        img = PILImage.open(_io.BytesIO(f.data))
        self.assertEqual(img.mode, "RGBA")

    def test_edit_without_mask_uses_auto_segmentation(self):
        asset_id = self._upload()
        resp = self.client.post("/api/edits",
                                json={"asset_id": asset_id, "prompt": "remove the chair"})
        self.assertEqual(resp.status_code, 202)
        job = self._wait_job(resp.get_json()["id"])
        self.assertEqual(job["state"], "completed")

    def test_delete_behind_the_scenes_log(self):
        """DELETE /api/events wipes the system event buffer AND the stored
        log lines of finished jobs — but the job records (and their prompts)
        stay untouched."""
        asset_id = self._upload()
        resp = self.client.post("/api/edits", json={
            "asset_id": asset_id, "prompt": "remove the chair"})
        job = self._wait_job(resp.get_json()["id"])
        self.assertTrue(job["logs"])  # the job produced log lines
        self.services.events.log("info", "seed system event")

        resp = self.client.delete("/api/events")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertGreaterEqual(body["events_cleared"], 1)
        self.assertGreaterEqual(body["jobs_stripped"], 1)
        after = self.client.get(f"/api/jobs/{job['id']}").get_json()
        self.assertEqual(after["logs"], [])
        self.assertEqual(after["payload"]["prompt"], "remove the chair")
        # ...and the wipe reached the DB row, not just memory.
        row = self.services.db.query(
            "SELECT logs FROM jobs WHERE id=?", (job["id"],))[0]
        self.assertEqual(row["logs"], "[]")
        # The stream holds only the deletion confirmation now.
        events = self.client.get("/api/events").get_json()
        self.assertEqual(len(events), 1)
        self.assertIn("deleted by the user", events[0]["msg"])

    def test_delete_prompt_history_removes_finished_jobs_only(self):
        asset_id = self._upload()
        resp = self.client.post("/api/edits", json={
            "asset_id": asset_id, "prompt": "remove the chair"})
        job_id = resp.get_json()["id"]
        self._wait_job(job_id)
        # A finished row that only exists in SQLite (older than the in-memory
        # rehydration window) must be swept too — that's the privacy point.
        self.services.db.execute(
            """INSERT INTO jobs (id, type, state, attempts, payload, result,
                                 error, logs, created_at, updated_at)
               VALUES ('oldrow01', 'image_edit', 'completed', 1,
                       '{"prompt": "ancient secret prompt"}', NULL, NULL,
                       '[]', '2020-01-01T00:00:00', '2020-01-01T00:00:00')""")
        # ...and so must the verbatim prompts in the workflow-learning memory.
        self.services.experience.record(
            "generate", "secret forge words", {"1": {}}, True, realism=9.0)

        resp = self.client.delete("/api/history/prompts")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertGreaterEqual(body["cleared"], 2)
        self.assertGreaterEqual(body["prompts_scrubbed"], 1)
        # The finished job — and with it the prompt — is gone.
        self.assertEqual(
            self.client.get(f"/api/jobs/{job_id}").status_code, 404)
        prompts = [j.get("payload", {}).get("prompt", "")
                   for j in self.client.get("/api/jobs").get_json()]
        self.assertNotIn("remove the chair", prompts)
        # Nothing prompt-bearing survives in SQLite.
        self.assertEqual(self.services.db.query(
            "SELECT id FROM jobs WHERE id='oldrow01'"), [])
        remnants = self.services.db.query(
            "SELECT prompt FROM workflow_memory WHERE prompt != ''")
        self.assertEqual(remnants, [])

    def test_safety_blocked_prompt_creates_no_job(self):
        # General adult content is allowed now; the hardcoded categories are
        # not. An undressing edit of a real photo (NCII) stays blocked.
        asset_id = self._upload()
        before = len(self.client.get("/api/jobs").get_json())
        resp = self.client.post("/api/edits",
                                json={"asset_id": asset_id,
                                      "prompt": "undress the woman"})
        self.assertEqual(resp.status_code, 422)
        self.assertTrue(resp.get_json()["error"]["code"].startswith("safety_"))
        self.assertEqual(len(self.client.get("/api/jobs").get_json()), before)

    def test_custom_safety_rule_crud_and_enforcement(self):
        # built-ins are listed but locked; adding a custom rule blocks live.
        listing = self.client.get("/api/safety/rules").get_json()
        self.assertTrue(all(b["locked"] for b in listing["builtin"]))
        self.assertEqual(listing["custom"], [])

        added = self.client.post("/api/safety/rules",
                                 json={"pattern": "unicorn", "reason": "nope"})
        self.assertEqual(added.status_code, 201)
        self.assertEqual(added.headers.get("Cache-Control"),
                         "no-store, no-cache, must-revalidate")
        rid = added.get_json()["id"]

        asset_id = self._upload()
        blocked = self.client.post(
            "/api/edits", json={"asset_id": asset_id, "prompt": "add a unicorn"})
        self.assertEqual(blocked.status_code, 422)
        self.assertEqual(blocked.get_json()["error"]["code"], "safety_custom")

        self.client.delete(f"/api/safety/rules/{rid}")
        ok = self.client.post(
            "/api/edits", json={"asset_id": asset_id, "prompt": "add a unicorn"})
        self.assertEqual(ok.status_code, 202)

    def test_reserved_rule_category_is_rejected(self):
        resp = self.client.post("/api/safety/rules",
                                json={"pattern": "x", "category": "minors"})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()["error"]["code"], "invalid_rule")

    def test_avatar_delete_removes_profile_and_frames(self):
        asset_id = self._upload()
        frame_id = self._upload()
        services = self.services
        profile = services.store.create_avatar(
            "Del", [asset_id], [frame_id], asset_id, meta={"consent": True})
        resp = self.client.delete(f"/api/avatars/{profile.id}?frames=1")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["frames_removed"], 1)
        self.assertIsNone(services.store.get_avatar(profile.id))
        # the frame asset was soft-deleted (out of the gallery)
        gallery_ids = [e["asset"]["id"] for e in services.store.gallery()]
        self.assertNotIn(frame_id, gallery_ids)
        self.assertEqual(
            self.client.delete("/api/avatars/nope").status_code, 404)

    def test_unsupported_upload_rejected(self):
        resp = self.client.post("/api/assets", data={
            "file": (io.BytesIO(b"hello"), "notes.txt")})
        self.assertEqual(resp.status_code, 415)

    def test_corrupt_image_content_is_rejected_at_upload(self):
        """A .png extension does not prove the bytes are a PNG. A renamed
        text file or a truncated download was accepted as a first-class
        image (measured live: HTTP 201) and only failed deep in the render
        pipeline. It must be refused at upload, like an unreadable video."""
        for label, data in (
            ("renamed text", b"this is definitely not an image"),
            ("png signature only", b"\x89PNG\r\n\x1a\n" + b"garbage"),
            ("empty", b""),
        ):
            resp = self.client.post("/api/assets", data={
                "file": (io.BytesIO(data), "photo.png")})
            self.assertEqual(resp.status_code, 415, label)
            self.assertIn("could not be read",
                          resp.get_json()["error"]["message"], label)

    def test_valid_image_upload_records_its_dimensions(self):
        resp = self.client.post("/api/assets", data={
            "file": (io.BytesIO(_png_bytes(size=(64, 48))), "photo.png")})
        self.assertEqual(resp.status_code, 201)
        meta = resp.get_json()["meta"]
        self.assertEqual((meta["width"], meta["height"]), (64, 48))

    def test_edit_on_missing_asset_404s(self):
        resp = self.client.post("/api/edits",
                                json={"asset_id": "nope", "prompt": "remove the chair"})
        self.assertEqual(resp.status_code, 404)

    def test_models_listed_and_download_requires_url(self):
        models = self.client.get("/api/models").get_json()
        by_name = {m["name"]: m for m in models}
        # seeded models ship with real sources and published checksums
        for name in ("sd15-inpaint", "sam-vit-b"):
            self.assertIn(name, by_name)
            self.assertTrue(by_name[name]["url"])
            self.assertRegex(by_name[name]["sha256"], r"^[0-9a-f]{64}$")
        # a model without a configured URL still fails loudly, with no job
        from app.core.registry import ModelInfo
        self.services.registry.register(ModelInfo(
            name="no-url-model", purpose="test", license="MIT",
            url=None, sha256=None, vram_gb=None))
        resp = self.client.post("/api/models/no-url-model/download")
        self.assertEqual(resp.status_code, 409)  # no URL configured -> clear error, no job

    def test_cancel_endpoint_rejects_finished_job(self):
        asset_id = self._upload()
        resp = self.client.post("/api/edits",
                                json={"asset_id": asset_id, "prompt": "remove the chair"})
        job = self._wait_job(resp.get_json()["id"])
        resp = self.client.post(f"/api/jobs/{job['id']}/cancel")
        self.assertEqual(resp.status_code, 409)


if __name__ == "__main__":
    unittest.main()
