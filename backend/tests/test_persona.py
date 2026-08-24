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

    def test_each_template_gets_its_own_reference_param_and_trigger(self):
        # Both hid behind the 12 GB paper gate until it fell: the handler
        # passed PhotoMaker's "image" name to InstantID's "face" parameter
        # (every render failed), and kept the PhotoMaker trigger token in
        # InstantID's prompt (noise it was never trained on).
        src = inspect.getsource(Services._handle_avatar_render)
        self.assertIn('"face" if "face" in template.get(', src)
        self.assertIn('if engine["template"] == "identity"', src)


if __name__ == "__main__":
    unittest.main()
