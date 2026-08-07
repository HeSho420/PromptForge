"""The avatar's skeleton and its invented back view: wiring guarantees.

The heavy lifting happens in Blender (rig_avatar.py) and ComfyUI (Kontext),
neither of which runs in this suite — what is tested here is the part the
app owes the user regardless: rigging degrades to an unrigged mesh instead
of failing the job, the invented back view only ever ships when the quality
gate measured an improvement, and both are recorded in the asset's metadata
rather than passed off as photographs.
"""
import inspect
import tempfile
import unittest
from pathlib import Path

from app.config import Settings
from app.core.services import Services


class FakeJob:
    id = "t"

    def __init__(self):
        self.lines = []

    def log(self, _level, message):
        self.lines.append(message)


class RigDegradesGracefully(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))
        self.addCleanup(self.s.stop)

    def test_no_blender_returns_the_mesh_unchanged(self):
        self.s._find_blender = lambda: None
        job = FakeJob()
        out, report = self.s._rig_avatar(job, b"GLBBYTES")
        self.assertEqual(out, b"GLBBYTES")
        self.assertEqual(report, {})
        self.assertTrue(any("unrigged" in m for m in job.lines))

    def test_env_override_wins(self):
        import os
        exe = Path(self.tmp.name) / "blender.exe"
        exe.write_bytes(b"x")
        os.environ["PROMPTFORGE_BLENDER"] = str(exe)
        try:
            self.assertEqual(self.s._find_blender(), str(exe))
        finally:
            del os.environ["PROMPTFORGE_BLENDER"]

    def test_back_synthesis_needs_kontext_and_a_front_view(self):
        self.s.kontext_ready = lambda: (False, "not downloaded")
        self.assertIsNone(self.s._synthesize_back_view(FakeJob()))
        self.s.kontext_ready = lambda: (True, "")
        self.s._avatar_front_view = None
        self.assertIsNone(self.s._synthesize_back_view(FakeJob()))


class Wiring(unittest.TestCase):

    def source(self):
        return inspect.getsource(Services._build_mesh)

    def test_the_rig_flag_reaches_build_mesh(self):
        self.assertIn("rig: bool = True", self.source())
        handler = inspect.getsource(Services._handle_avatar)
        self.assertIn('p.get("rig", True) is not False', handler)

    def test_synthesized_views_are_retextured_from_bare_geometry(self):
        """Re-texturing the already-textured GLB would compound the vertex
        split; the second pass must start from the untextured mesh."""
        src = self.source()
        self.assertIn("self._texture_mesh(\n                            "
                      "job, bare, photos + synth,", src)

    def test_every_uncovered_arc_gets_a_generated_view(self):
        """The back, left and right arcs are all checked — an honest render
        showed the uncovered SIDES carrying the worst junk on the figure."""
        src = self.source()
        for arc in ('(180.0, "back")', '(90.0, "left")', '(270.0, "right")'):
            self.assertIn(arc, src)

    def test_an_invented_back_only_ships_when_coverage_rose(self):
        self.assertIn('(report.get("seen_pct") or 0) + 1', self.source())

    def test_the_metadata_says_what_was_invented_and_what_was_rigged(self):
        src = self.source()
        for key in ('"rigged"', '"rig_bones"', '"back_synthesized"'):
            self.assertIn(key, src)


class FullFigure(unittest.TestCase):
    """A waist-up photo with margins is not 'cut off', but it is not a whole
    person either — it used to sail through completion and ship as a bust."""

    def test_a_partial_figure_is_detected_without_touching_an_edge(self):
        src = inspect.getsource(Services._complete_subject)
        self.assertIn("_figure_aspect", src)
        self.assertIn('pad["bottom"]', src)

    def test_the_intake_and_the_rigger_agree_on_what_whole_means(self):
        self.assertEqual(Services._FULL_FIGURE_ASPECT, 1.9)
        rig = (Path(__file__).resolve().parent.parent / "app" / "tools"
               / "rig_avatar.py").read_text(encoding="utf-8")
        self.assertIn("aspect > 1.9", rig)

    def test_the_orbit_source_is_cut_out_even_after_extension(self):
        """`is not open_asset_image(...)` compared against a FRESH copy, so
        'extended' was always true, and the None mask it then passed staged
        the photo UNCUT — SV3D rotated the room, which is where the
        hallucinated-wall views came from."""
        src = inspect.getsource(Services._handle_avatar)
        self.assertIn("extended = source_image is not original", src)
        self.assertIn("orbit_mask = self._subject_matte(source_image)", src)
        self.assertNotIn("None if extended", src)

    def test_the_completed_body_is_what_the_reconstruction_sees(self):
        """When the source was extended, its frame slot must carry the
        COMPLETED image — staging the raw waist-up original there is how a
        job that logged 'the body is complete now' still shipped a bust."""
        src = inspect.getsource(Services._handle_avatar)
        self.assertIn("use_completed = real_id == source_id and extended",
                      src)


class TextureRefinement(unittest.TestCase):
    """The repaint loop may only ever ship measured improvements."""

    def test_a_repainted_view_must_prove_it_got_cleaner(self):
        src = inspect.getsource(Services._refine_texture)
        self.assertIn("s_after >= s_before * 0.92", src)
        self.assertIn('drift >= (72.0 if strong else 55.0)', src)
        self.assertIn("skipped.append", src)

    def test_refinement_is_wired_behind_a_flag_and_never_fatal(self):
        build = inspect.getsource(Services._build_mesh)
        self.assertIn("if textured_here and refine:", build)
        self.assertIn("Texture refinement skipped", build)
        handler = inspect.getsource(Services._handle_avatar)
        self.assertIn('refine=p.get("refine", True) is not False', handler)

    def test_the_rasteriser_and_paster_exist_in_the_tool(self):
        tool = (Path(__file__).resolve().parent.parent / "app" / "tools"
                / "texture_mesh.py").read_text(encoding="utf-8")
        for needle in ("def rasterize_tile", "def paste_tile",
                      "def _tile_frame", '"islands_flipped"'):
            self.assertIn(needle, tool)


if __name__ == "__main__":
    unittest.main()
