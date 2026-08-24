"""The persona promise: renders of an avatar ARE that person, measured.

Live calibration behind these pins (2026-08-24, RTX 4060 8 GB): the
same person measures 0.88-0.98 across pixel-preserving edits and
0.6-0.8 across full re-renders; PhotoMaker's generic look-alike scored
0.213 while InstantID scored 0.781 on the identical prompt — and
InstantID, exiled by a 12 GB / 24 GB paper gate, rendered in 260 s on
this 8 GB card under ComfyUI 0.28's weight streaming."""
import inspect
import py_compile
import tempfile
import unittest
from pathlib import Path

from app.config import Settings
from app.core.services import DEFAULT_MODELS, Services


class InstantIdGateTests(unittest.TestCase):
    def test_gates_state_the_measured_floor_not_file_size_sums(self):
        self.assertEqual(Services._INSTANTID_VRAM_GB, 8.0)
        self.assertEqual(Services._INSTANTID_RAM_GB, 15.0)
        for name in ("instantid-ipadapter", "instantid-controlnet"):
            info = next(m for m in DEFAULT_MODELS if m.name == name)
            self.assertEqual(info.meta.get("min_vram_gb"), 8.0, name)
            self.assertEqual(info.meta.get("min_ram_gb"), 15.0, name)

    def test_the_engine_router_prefers_instantid_when_it_fits(self):
        src = inspect.getsource(Services._identity_engine)
        self.assertIn('"template": "identity_face"', src)
        self.assertIn("strongest likeness", src)


class IdentityMeasurementTests(unittest.TestCase):
    def test_the_similarity_tool_exists_and_compiles(self):
        tool = (Path(__file__).resolve().parent.parent / "app" / "tools"
                / "face_similarity.py")
        self.assertTrue(tool.exists())
        py_compile.compile(str(tool), doraise=True)

    def test_mock_mode_measures_nothing(self):
        from PIL import Image
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        s = Services(Settings(
            data_dir=Path(tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))
        self.addCleanup(s.stop)

        class _J:
            def log(self, *a):
                pass

        img = Image.new("RGB", (32, 32))
        self.assertIsNone(s._face_similarity(_J(), img, img))

    def test_every_identity_render_is_measured_and_reported(self):
        src = inspect.getsource(Services._handle_avatar_render)
        self.assertIn("ArcFace likeness", src)
        self.assertIn('"identity_match"', src)
        # A drifted InstantID render buys ONE harder-locked retry, and the
        # better MEASURED likeness is kept.
        self.assertIn("render_once(positive, weight=0.95)", src)
        self.assertIn('engine["template"] == "identity_face"', src)

    def test_the_weight_dial_only_reaches_templates_that_have_one(self):
        src = inspect.getsource(Services._handle_avatar_render)
        self.assertIn('"weight" in template.get(', src)

    def test_photoreal_checkpoint_rides_the_dial_when_ready(self):
        # "Cannot tell it's AI" starts at the checkpoint: the plain SDXL
        # base has the telltale AI look. RealVisXL is used when ready and
        # the base stays the honest fallback.
        src = inspect.getsource(Services._handle_avatar_render)
        self.assertIn('"RealVisXL_V5.0_fp16.safetensors"', src)
        self.assertIn('"checkpoint" in template.get(', src)
        from app.core.services import DEFAULT_MODELS as models
        info = next(m for m in models if m.name == "realvisxl-v5")
        self.assertTrue(info.sha256)

    def test_the_persona_polish_stays_removed_by_measurement(self):
        # Measured twice: FaceDetailer at 0.45 denoise sharpened the face
        # out of her likeness (ArcFace 0.84 -> 0.62); at 0.25 it measured
        # zero critic gain, -0.03 likeness, and 5:22 of render time.
        # Persona renders are portrait-scale - the detailer exists for
        # small mushy faces in full-body shots, and the generate path
        # keeps it. A future polish must carry InstantID conditioning
        # into the detailer graph and beat both referees.
        src = inspect.getsource(Services._handle_avatar_render)
        self.assertNotIn("self._face_polish(", src)
        self.assertIn("REMOVED the same day", src)

    def test_full_body_requests_render_a_tall_frame(self):
        # "full body" came back as a 3/4 portrait: InstantID anchors the
        # face at the reference's scale and a square latent leaves no
        # room below it. Tall latent + framing words push the same way.
        from app.core.quality import full_body_intent
        for yes in ("standing in a park, full body",
                    "head to toe portrait", "full-length shot",
                    "show her whole figure"):
            self.assertTrue(full_body_intent(yes), yes)
        for no in ("a close portrait", "dancing at a festival"):
            self.assertFalse(full_body_intent(no), no)
        src = inspect.getsource(Services._handle_avatar_render)
        self.assertIn("832, 1216", src)
        self.assertIn("full body from head to toe", src)

    def test_the_reference_set_measures_its_own_coherence(self):
        # Measured live: padding a persona with app-GENERATED renders of
        # the person diluted the averaged identity embedding (0.803
        # single-ref -> 0.717 with three generated refs). Every extra
        # photo is measured against the primary and reported; a distant
        # one warns, never blocks.
        src = inspect.getsource(Services._handle_persona)
        self.assertIn('"reference_coherence": coherence', src)
        self.assertIn("below the", src)
        self.assertIn("real ", src.lower())

    def test_each_template_gets_its_own_reference_param_and_trigger(self):
        # Both hid behind the 12 GB paper gate until it fell: the handler
        # passed PhotoMaker's "image" name to InstantID's "face" parameter
        # (every render failed), and kept the PhotoMaker trigger token in
        # InstantID's prompt (noise it was never trained on).
        src = inspect.getsource(Services._handle_avatar_render)
        self.assertIn('"face" if "face" in template.get(', src)
        self.assertIn('if engine["template"] == "identity"', src)


class PersonaIntentTests(unittest.TestCase):
    """The prompt box speaks persona: 'use persona X' renders one,
    'make a persona from this image' creates one (consent-gated)."""

    def test_use_requests_parse_name_and_scene(self):
        from app.core.quality import persona_use_request
        self.assertEqual(
            persona_use_request("use persona 'Mira': hiking a trail"),
            ("Mira", "hiking a trail"))
        self.assertEqual(
            persona_use_request('generate the persona "Mira B" at a cafe'),
            ("Mira B", "at a cafe"))
        self.assertEqual(
            persona_use_request("persona Mira walking on a beach"),
            ("Mira", "walking on a beach"))
        # a USE outranks its own creation verbs
        name, scene = persona_use_request("show the persona 'Mira' smiling")
        self.assertEqual(name, "Mira")
        self.assertIsNone(persona_use_request("remove the tree"))
        self.assertIsNone(
            persona_use_request("make a persona from this image"))

    def test_create_intent_table(self):
        from app.core.quality import persona_create_intent
        for yes in ("make a persona from this image",
                    "create a persona of her",
                    "save this as a persona",
                    "build a persona from my photo"):
            self.assertTrue(persona_create_intent(yes), yes)
        for no in ("use persona 'Mira': on a beach",
                   "remove the background",
                   "put her in a nightclub"):
            self.assertFalse(persona_create_intent(no), no)

    def test_resolve_persona_matches_exact_case_and_prefix(self):
        from types import SimpleNamespace
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        s = Services(Settings(
            data_dir=Path(tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))
        self.addCleanup(s.stop)
        s.store.list_avatars = lambda: [
            SimpleNamespace(name="Mira (test persona)", id="a1", meta={}),
            SimpleNamespace(name="Bob", id="a2", meta={})]
        self.assertEqual(s.resolve_persona("bob").id, "a2")
        self.assertEqual(s.resolve_persona("Mira").id, "a1")
        self.assertEqual(s.resolve_persona("MIRA (TEST PERSONA)").id, "a1")
        self.assertIsNone(s.resolve_persona("Zoe"))
        self.assertIsNone(s.resolve_persona(""))


class PersonaRouteTests(unittest.TestCase):
    """The /api/edits prompt box routes personas end to end."""

    def setUp(self):
        import io as _io

        from PIL import Image as _Image

        from app.api.routes import create_app
        self.tmp = tempfile.TemporaryDirectory()
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", llm_url="http://127.0.0.1:9/v1",
            critic_model="", first_run_setup=False, comfyui_dir=""))
        app = create_app(self.s)
        app.testing = True
        self.client = app.test_client()
        buf = _io.BytesIO()
        _Image.new("RGB", (64, 64), (90, 90, 90)).save(buf, format="PNG")
        resp = self.client.post("/api/assets", data={
            "file": (_io.BytesIO(buf.getvalue()), "p.png")})
        self.asset_id = resp.get_json()["id"]

    def tearDown(self):
        self.s.stop()
        self.tmp.cleanup()

    def test_use_routes_to_an_identity_render(self):
        from types import SimpleNamespace
        self.s.store.list_avatars = lambda: [
            SimpleNamespace(name="Mira", id="av9", meta={})]
        resp = self.client.post("/api/edits", json={
            "asset_id": self.asset_id,
            "prompt": "use persona 'Mira': hiking a mountain trail"})
        self.assertEqual(resp.status_code, 202)
        job = resp.get_json()
        self.assertEqual(job["type"], "avatar_render")
        self.assertEqual(job["payload"]["avatar_id"], "av9")
        self.assertEqual(job["payload"]["prompt"],
                         "hiking a mountain trail")

    def test_unknown_persona_names_the_saved_ones(self):
        from types import SimpleNamespace
        self.s.store.list_avatars = lambda: [
            SimpleNamespace(name="Mira", id="av9", meta={})]
        resp = self.client.post("/api/edits", json={
            "asset_id": self.asset_id,
            "prompt": "use persona 'Zoe': at the beach"})
        self.assertEqual(resp.status_code, 404)
        body = resp.get_json()["error"]
        self.assertEqual(body["code"], "persona_not_found")
        self.assertIn("'Mira'", body["message"])

    def test_creation_never_proceeds_without_consent(self):
        resp = self.client.post("/api/edits", json={
            "asset_id": self.asset_id,
            "prompt": "make a persona from this image"})
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.get_json()["error"]["code"],
                         "persona_consent_required")

    def test_creation_with_consent_enqueues_the_2d_intake(self):
        # The typed "make a persona" wants the character card (2D, about
        # a minute), NOT the 36-minute 3D avatar build.
        resp = self.client.post("/api/edits", json={
            "asset_id": self.asset_id,
            "prompt": "make a persona from this image",
            "consent": True, "persona_name": "Test P"})
        self.assertEqual(resp.status_code, 202)
        job = resp.get_json()
        self.assertEqual(job["type"], "persona")
        self.assertEqual(job["payload"]["asset_ids"], [self.asset_id])
        self.assertTrue(job["payload"]["consent"])

    def _wait(self, job_id: str, timeout=30.0) -> dict:
        import time
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.client.get(f"/api/jobs/{job_id}").get_json()
            if job["state"] in ("completed", "failed", "cancelled"):
                return job
            time.sleep(0.05)
        self.fail("job did not finish")

    def test_persona_lifecycle_create_list_prefer_delete(self):
        """An avatar is a 3D rigged character; a persona is the 2D card.
        The two lists never mix, and "use persona" prefers the card."""
        resp = self.client.post("/api/personas", json={
            "asset_ids": [self.asset_id], "consent": True, "name": "Zoe"})
        self.assertEqual(resp.status_code, 202)
        job = self._wait(resp.get_json()["id"])
        self.assertEqual(job["state"], "completed", job.get("error"))
        pid = job["result"]["persona_id"]
        personas = self.client.get("/api/personas").get_json()
        self.assertEqual([p["id"] for p in personas], [pid])
        self.assertEqual(personas[0]["meta"]["kind"], "persona")
        # ...and the 3D avatar list does NOT show the 2D card.
        self.assertEqual(self.client.get("/api/avatars").get_json(), [])
        # Same-name preference: a persona outranks an avatar for
        # "use persona".
        from types import SimpleNamespace
        real = self.s.store.list_avatars
        self.s.store.list_avatars = lambda: [
            SimpleNamespace(name="Zoe", id="old3d", meta={}),
            *real()]
        try:
            self.assertEqual(self.s.resolve_persona("Zoe").id, pid)
        finally:
            self.s.store.list_avatars = real
        # Render route answers for the persona id.
        r = self.client.post(f"/api/personas/{pid}/render",
                             json={"prompt": "at a lake"})
        self.assertEqual(r.status_code, 202)
        self.assertEqual(r.get_json()["type"], "avatar_render")
        # Delete removes it from the persona list.
        d = self.client.delete(f"/api/personas/{pid}")
        self.assertEqual(d.status_code, 200)
        self.assertEqual(self.client.get("/api/personas").get_json(), [])


if __name__ == "__main__":
    unittest.main()
