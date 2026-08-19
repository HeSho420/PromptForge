"""Tests: environment-aware editing's measurement layer.

Synthetic geometry with KNOWN ground truth: a matte whose feet are drawn
at known columns, normal/depth renders synthesized from a known camera
pitch and horizon. The analysis must recover the numbers it was built
from — and stay silent (None) wherever nothing can be measured."""
import json
import math
import unittest

import numpy as np
from PIL import Image

from app.core import scene_geometry
from app.core.scene_geometry import SceneCard


def _figure_matte(w=300, h=400, feet=((100, 360), (150, 358)),
                  cut=False):
    """A crude standing figure: a torso column plus feet blobs whose
    bottoms are the KNOWN contact rows."""
    m = Image.new("L", (w, h), 0)
    import PIL.ImageDraw as D
    d = D.Draw(m)
    d.rectangle((105, 80, 145, 320), fill=255)          # torso+legs
    for cx, cy in feet:
        d.ellipse((cx - 14, cy - 18, cx + 14, cy), fill=255)
    if cut:
        d.rectangle((105, 320, 145, h - 1), fill=255)   # legs leave frame
    return m


class SubjectGeometryTests(unittest.TestCase):
    def test_contacts_are_the_matte_bottom_clusters(self):
        info = scene_geometry.subject_geometry(_figure_matte())
        self.assertFalse(info["cut_at_bottom"])
        self.assertEqual(len(info["contact_points"]), 2)
        xs = sorted(c[0] for c in info["contact_points"])
        self.assertLess(abs(xs[0] - 100), 8)
        self.assertLess(abs(xs[1] - 150), 8)
        self.assertAlmostEqual(info["contact_y_frac"], 360 / 400, delta=0.01)

    def test_frame_cropped_subject_has_no_contacts(self):
        info = scene_geometry.subject_geometry(_figure_matte(cut=True))
        self.assertTrue(info["cut_at_bottom"])
        self.assertEqual(info["contact_points"], [])
        self.assertIsNone(info["contact_y_frac"])

    def test_empty_matte_measures_nothing(self):
        self.assertEqual(
            scene_geometry.subject_geometry(Image.new("L", (64, 64), 0)), {})


def _synthetic_probe(w=320, h=240, pitch_deg=18.0, y_h=-40,
                     ground_top=140, encode="depth"):
    """Normal/depth/valid renders for a flat ground seen by a camera
    pitched down, with the horizon at the KNOWN row y_h (above frame)."""
    p = math.radians(pitch_deg)
    n = np.zeros((h, w, 3), dtype=np.float32)
    n[:ground_top] = (0.0, 0.0, 1.0)                     # facing wall
    n[ground_top:] = (0.0, math.cos(p), math.sin(p))     # the ground
    normal = Image.fromarray(((n + 1) * 127.5).astype(np.uint8), "RGB")
    ys = np.arange(h, dtype=np.float64)
    if encode == "depth":     # PNG stores affine-mapped DEPTH
        z = np.where(ys >= ground_top, 1.0 / (ys - y_h), 1.0 / (h * 4))
        z = (z - z.min()) / (z.max() - z.min())
    else:                     # PNG stores affine-mapped DISPARITY
        z = np.where(ys >= ground_top, (ys - y_h), 0.0)
        z = z / z.max()
    depth = Image.fromarray(
        np.repeat((z * 255).astype(np.uint8)[:, None], w, axis=1), "L")
    valid = Image.new("L", (w, h), 255)
    return normal, depth.convert("RGB"), valid.convert("RGB")


class GroundGeometryTests(unittest.TestCase):
    def _check(self, encode):
        normal, depth, valid = _synthetic_probe(encode=encode)
        g = scene_geometry.ground_geometry(normal, depth, valid, None)
        self.assertGreater(g["ground_frac"], 0.3)
        self.assertAlmostEqual(g["camera_pitch_deg"], 18.0, delta=2.0)
        self.assertIn("horizon_y_frac", g)
        self.assertAlmostEqual(g["horizon_y_frac"], -40 / 240, delta=0.06)
        self.assertGreaterEqual(g["horizon_r2"], 0.95)

    def test_recovers_pitch_and_horizon_from_depth_encoding(self):
        self._check("depth")

    def test_recovers_pitch_and_horizon_from_disparity_encoding(self):
        self._check("disparity")

    def test_ground_running_to_the_horizon_still_fits(self):
        # Live plaza case: disparity-encoded depth whose far rows saturate
        # to EXACTLY 0 and whose ground runs all the way to the horizon —
        # the root lands on the boundary and the saturated rows must be
        # dropped, not fitted.
        normal, _depth, valid = _synthetic_probe(ground_top=96, y_h=90)
        ys = np.arange(240, dtype=np.float64)
        disp = np.clip(ys - 90, 0, None)
        disp[:110] = 0.0                       # saturated far band
        z = disp / disp.max()
        depth = Image.fromarray(
            np.repeat((z * 255).astype(np.uint8)[:, None], 320, axis=1),
            "L").convert("RGB")
        g = scene_geometry.ground_geometry(normal, depth, valid, None)
        self.assertIn("horizon_y_frac", g)
        self.assertAlmostEqual(g["horizon_y_frac"], 90 / 240, delta=0.06)

    def test_no_ground_reports_only_the_fraction(self):
        # all-wall scene: nothing to stand on, no pitch, no horizon
        normal, depth, valid = _synthetic_probe(ground_top=240)
        g = scene_geometry.ground_geometry(normal, depth, valid, None)
        self.assertLess(g["ground_frac"], 0.02)
        self.assertNotIn("camera_pitch_deg", g)

    def test_subject_matte_is_excluded_from_the_plane(self):
        normal, depth, valid = _synthetic_probe()
        matte = Image.new("L", (320, 240), 0)
        matte.paste(255, (0, 140, 320, 240))   # subject covers ALL ground
        g = scene_geometry.ground_geometry(normal, depth, valid, matte)
        self.assertLess(g["ground_frac"], 0.02)


class GuidanceDepthTests(unittest.TestCase):
    """The perspective guide: measured subject + measured ground + the
    plane's ramp to the measured horizon; far/free above it. No confident
    horizon = no guide (fabricated geometry is forbidden)."""

    def _bits(self):
        normal, _d, valid = _synthetic_probe()        # ground rows 140+
        h, w = 240, 320
        ys = np.arange(h, dtype=np.float64)
        disp = np.clip(ys + 40, 0, None)
        disp = disp / disp.max()
        depth = Image.fromarray(
            np.repeat((disp * 255).astype(np.uint8)[:, None], w, axis=1),
            "L").convert("RGB")
        matte = Image.new("L", (w, h), 0)
        matte.paste(255, (150, 40, 170, 140))         # subject above ground
        return normal, depth, valid, matte

    def test_guide_composes_ramp_ground_and_subject(self):
        normal, depth, valid, matte = self._bits()
        card = SceneCard(horizon_y_frac=90 / 240, horizon_r2=0.99)
        g = scene_geometry.guidance_depth(card, depth, normal, valid,
                                          matte, (640, 480))
        self.assertEqual(g.size, (640, 480))
        a = np.asarray(g.convert("L"), dtype=np.float32)
        self.assertLess(a[100, 40], 12)      # above the horizon: far, free
        self.assertGreater(a[430, 320], 140)  # near ground: measured, bright
        # the subject keeps its measured depth against a darker ramp
        self.assertGreater(a[200, 320], 80)
        self.assertLess(a[200, 40], 40)

    def test_no_confident_horizon_means_no_guide(self):
        normal, depth, valid, matte = self._bits()
        self.assertIsNone(scene_geometry.guidance_depth(
            SceneCard(), depth, normal, valid, matte, (64, 64)))
        weak = SceneCard(horizon_y_frac=0.4, horizon_r2=0.5)
        self.assertIsNone(scene_geometry.guidance_depth(
            weak, depth, normal, valid, matte, (64, 64)))


class PostureVetoTests(unittest.TestCase):
    def test_matte_aspect_vetoes_impossible_answers(self):
        tall = (0, 0, 100, 260)
        wide = (0, 0, 260, 100)
        self.assertIsNone(scene_geometry.posture_veto("lying", tall))
        self.assertIsNone(scene_geometry.posture_veto("standing", wide))
        self.assertEqual(scene_geometry.posture_veto("standing", tall),
                         "standing")
        self.assertEqual(scene_geometry.posture_veto("lying", wide), "lying")
        self.assertIsNone(scene_geometry.posture_veto("unknown", tall))
        self.assertIsNone(scene_geometry.posture_veto("flying", tall))


class SpatialPromptTests(unittest.TestCase):
    SPEC = {"environment": "a sunlit swimming pool area",
            "relationship": "standing on the pool deck beside the pool",
            "ground_surface": "wet ceramic tiles",
            "elements": ["pool edge", "clear water", "lounge chairs"],
            "lighting_wish": "keep"}

    def _card(self, **kw):
        base = {"camera_pitch_deg": 2.0, "horizon_y_frac": 0.42,
                "horizon_r2": 0.99, "lighting": "soft daylight from left",
                "contact_y_frac": 0.9}
        base.update(kw)
        return SceneCard(**base)

    def test_compiles_contract_camera_horizon_and_lighting(self):
        pos, neg = scene_geometry.spatial_prompt(
            self.SPEC, self._card(), "base", "bad")
        for needle in ("standing on the pool deck beside the pool",
                       "wet ceramic tiles under and around the subject",
                       "photographed at eye level",
                       "horizon line across the middle of the frame",
                       "lighting: soft daylight from left",
                       "extending behind and around the subject"):
            self.assertIn(needle, pos)
        for needle in ("flat backdrop", "extra person", "bad"):
            self.assertIn(needle, neg)

    def test_cropped_subject_gets_no_ground_contract(self):
        pos, _ = scene_geometry.spatial_prompt(
            self.SPEC, self._card(cut_at_bottom=True, contact_y_frac=None),
            "base", "")
        self.assertNotIn("under and around the subject's feet", pos)

    def test_unmeasured_geometry_says_nothing(self):
        pos, _ = scene_geometry.spatial_prompt(
            self.SPEC, SceneCard(), "base", "")
        self.assertNotIn("photographed", pos.split("photograph")[0])
        self.assertNotIn("horizon", pos)

    def test_without_spec_the_base_prompt_survives(self):
        pos, _ = scene_geometry.spatial_prompt(None, None, "a forest", "")
        self.assertTrue(pos.startswith("a forest"))
        self.assertIn("extending behind and around the subject", pos)

    def test_lighting_wish_overrides_the_measured_light(self):
        spec = dict(self.SPEC, lighting_wish="warm sunset glow")
        pos, _ = scene_geometry.spatial_prompt(spec, self._card(),
                                               "base", "")
        self.assertIn("lighting: warm sunset glow", pos)
        self.assertNotIn("soft daylight from left", pos)


class _SpecLLM:
    def __init__(self, payload):
        self.payload = payload
        self.asks = []

    def complete(self, system, prompt, max_tokens=0):
        self.asks.append((system, prompt))

        class R:
            text = json.dumps(self.payload)
        return R()


class EnvironmentSpecTests(unittest.TestCase):
    def test_spec_parses_and_carries_the_facts(self):
        llm = _SpecLLM(SpatialPromptTests.SPEC)
        spec = scene_geometry.environment_spec(
            llm, "change the background to a swimming pool",
            "standing", False, "a bedroom")
        self.assertEqual(spec["environment"], "a sunlit swimming pool area")
        self.assertIn("the subject is standing", llm.asks[0][1])
        self.assertIn("visible in frame", llm.asks[0][1])
        self.assertIn("a bedroom", llm.asks[0][1])

    def test_unknown_facts_are_not_claimed(self):
        llm = _SpecLLM(SpatialPromptTests.SPEC)
        scene_geometry.environment_spec(llm, "beach please", None, None)
        self.assertNotIn("visible in frame", llm.asks[0][1])
        self.assertNotIn("OUTSIDE the frame", llm.asks[0][1])
        self.assertIn("none measured yet", llm.asks[0][1])

    def test_planner_is_conservative_without_facts(self):
        # The first live plan put a standing subject "in the shallow end
        # of the pool" on no evidence — the system prompt must demand the
        # conservative relationship when posture is unstated.
        self.assertIn("most conservative", scene_geometry._SPEC_SYSTEM)
        self.assertIn("never inside water", scene_geometry._SPEC_SYSTEM)

    def test_no_llm_or_bad_reply_means_none(self):
        self.assertIsNone(scene_geometry.environment_spec(
            None, "beach", None, None))

        class Bad:
            def complete(self, *a, **k):
                raise RuntimeError("down")
        self.assertIsNone(scene_geometry.environment_spec(
            Bad(), "beach", None, None))


class EnvironmentMissTests(unittest.TestCase):
    def _card(self):
        return SceneCard(ground_frac=0.3, camera_pitch_deg=15.0,
                         horizon_y_frac=0.3, horizon_r2=0.99)

    def test_lost_ground_is_a_floating_subject(self):
        misses = scene_geometry.environment_misses(
            self._card(), {"ground_frac": 0.0})
        self.assertTrue(any("floating" in m for m in misses))

    def test_pitch_and_horizon_breaks_are_named(self):
        misses = scene_geometry.environment_misses(
            self._card(), {"ground_frac": 0.3, "camera_pitch_deg": 40.0,
                           "horizon_y_frac": 0.7, "horizon_r2": 0.99})
        self.assertTrue(any("camera angle changed" in m for m in misses))
        self.assertTrue(any("horizon moved" in m for m in misses))

    def test_matching_geometry_passes(self):
        self.assertEqual(scene_geometry.environment_misses(
            self._card(), {"ground_frac": 0.28, "camera_pitch_deg": 12.0,
                           "horizon_y_frac": 0.34, "horizon_r2": 0.99}), [])

    def test_water_under_the_feet_is_named(self):
        # Measured live: camera and horizon PASSED while the subject stood
        # ankle-deep in the pool — only the under-foot window catches it.
        card = SceneCard(contact_points=[(100, 360)], ground_frac=0.3)
        misses = scene_geometry.environment_misses(
            card, {"ground_frac": 0.3, "contact_ground_frac": 0.0})
        self.assertTrue(any("standing in water or floating" in m
                            for m in misses))
        self.assertEqual(scene_geometry.environment_misses(
            card, {"ground_frac": 0.3, "contact_ground_frac": 0.8}), [])

    def test_contact_window_reads_the_normals_under_the_feet(self):
        normal, _d, _v = _synthetic_probe()       # ground from row 140 down
        # a foot resting ON the ground band: window below is up-facing
        on = scene_geometry.contact_ground_frac(
            normal, [(160, 150)], (320, 240))
        # a "foot" hanging over the wall region: nothing walkable below
        off = scene_geometry.contact_ground_frac(
            normal, [(160, 40)], (320, 240))
        self.assertGreater(on, 0.9)
        self.assertLess(off, 0.1)
        self.assertIsNone(scene_geometry.contact_ground_frac(
            normal, [], (320, 240)))

    def test_solid_ground_plan_forbids_flooded_foreground(self):
        _pos, neg = scene_geometry.spatial_prompt(
            SpatialPromptTests.SPEC, SceneCard(), "base", "")
        self.assertIn("subject standing in water", neg)
        water_spec = dict(SpatialPromptTests.SPEC,
                          ground_surface="shallow water")
        _pos, neg2 = scene_geometry.spatial_prompt(
            water_spec, SceneCard(), "base", "")
        self.assertNotIn("subject standing in water", neg2)

    def test_unmeasured_sides_stay_silent(self):
        self.assertEqual(
            scene_geometry.environment_misses(SceneCard(), {}), [])
        # a cropped subject never triggers the floating check
        card = SceneCard(ground_frac=0.3, cut_at_bottom=True)
        self.assertEqual(scene_geometry.environment_misses(
            card, {"ground_frac": 0.0}), [])


class ContactSurfaceProbeTests(unittest.TestCase):
    """Normals cannot tell pool water from a pool deck — both are
    up-facing planes (measured live: pitch, horizon AND the up-normal
    window all passed while the subject stood ankle-deep in the pool).
    Geometry says WHERE to look; the region-scoped vision probe says
    WHAT is there."""

    def _services(self):
        import tempfile
        from pathlib import Path

        from app.config import Settings
        from app.core.services import Services
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        s = Services(Settings(data_dir=Path(self.tmp.name),
                              inpaint_backend="mock",
                              segment_backend="mock", critic_model="",
                              first_run_setup=False, comfyui_dir=""))
        self.addCleanup(s.stop)
        return s

    class _Critic:
        def __init__(self, reply):
            self.reply = reply
            self.views = []

        def ask(self, image, question, schema=None):
            self.views.append((image.size, question))
            return self.reply

    def test_wrong_surface_is_a_named_miss(self):
        s = self._services()
        s.critic = self._Critic(
            '{"on_expected": false, "seen": "pool water"}')
        card = SceneCard(contact_points=[(512, 900)])
        img = Image.new("RGB", (1024, 1024), (40, 40, 40))
        miss = s._contact_surface_miss(
            img, card, {"ground_surface": "pool deck tiles"})
        self.assertIn("pool water", miss)
        self.assertIn("pool deck tiles", miss)
        # the probe sees the contact neighbourhood, not the whole frame,
        # and judges SUPPORT, not adjectives (live: "dry tiles" was
        # rejected against a planned "wet tiles")
        (size, q), = s.critic.views
        self.assertLess(size[0], 1024)
        self.assertIn("pool deck tiles", q)
        self.assertIn("could not physically support", q)
        self.assertIn("do NOT count", q)

    def test_right_surface_and_unprobeable_cases_stay_silent(self):
        s = self._services()
        s.critic = self._Critic('{"on_expected": true, "seen": "tiles"}')
        card = SceneCard(contact_points=[(512, 900)])
        img = Image.new("RGB", (1024, 1024))
        self.assertIsNone(s._contact_surface_miss(
            img, card, {"ground_surface": "pool deck"}))
        cut = SceneCard(contact_points=[], cut_at_bottom=True)
        self.assertIsNone(s._contact_surface_miss(
            img, cut, {"ground_surface": "pool deck"}))
        self.assertIsNone(s._contact_surface_miss(img, card, None))


class EnvironmentIntentTests(unittest.TestCase):
    """Relocation phrasing must reach the environment pipeline. Live:
    'put her in a nightclub' was planned as a style change, coerced to
    inpaint, and failed hunting the picture for a nightclub to mask."""

    def test_relocations_match(self):
        from app.core.quality import environment_intent
        for yes in ("put her in a nightclub",
                    "put me on a beach at sunset",
                    "place him inside a medieval castle",
                    "put the subject in Tokyo",
                    "set them in a futuristic laboratory",
                    "move her to a rooftop",
                    "put me in front of a Ferrari"):
            self.assertTrue(environment_intent(yes), yes)

    def test_object_insertion_clothing_and_pose_do_not(self):
        from app.core.quality import environment_intent
        for no in ("put a dog in the background",
                    "put her in a red dress",
                    "put him in a business suit",
                    "put her hand on her hip",
                    "place a vase on the table",
                    "put them in a different pose",
                    "put me in the foreground",
                    "change her shoes"):
            self.assertFalse(environment_intent(no), no)


class EnvironmentChecklistTests(unittest.TestCase):
    def test_background_steps_verify_the_place_not_the_far_field(self):
        # "What is the new background?" was answered — correctly — with
        # "trees and mountains" on a pool-dominated scene (the far field
        # IS the background), so every run burned a retry on a
        # requirement no honest answer could name. Background steps must
        # derive their check from the plan and ask about the PLACE.
        import inspect

        from app.core.services import Services
        src = inspect.getsource(Services._handle_image_edit)
        self.assertIn("Where does this photo appear", src)
        self.assertIn('step["_checklist"] = [{', src)

    def test_relight_after_background_is_pruned(self):
        # The environment step lights its own scene non-destructively; a
        # trailing CHANGE_LIGHTING step re-ran the FULL IC-Light redraw —
        # measured live: 92% of face pixels moved, identity gone.
        import inspect

        from app.core.services import Services
        src = inspect.getsource(Services._handle_image_edit)
        self.assertIn("dropped the separate relight", src)
        self.assertIn('s["task"] != "relight"', src)


class ProbeFileRoutingTests(unittest.TestCase):
    def test_files_route_by_prefix(self):
        import io as _io

        def png():
            b = _io.BytesIO()
            Image.new("RGB", (4, 4)).save(b, format="PNG")
            return b.getvalue()
        shots = scene_geometry.parse_probe_files([
            (png(), "pfprobe_depth_00001_.png"),
            (png(), "pfprobe_normal_00001_.png"),
            (png(), "pfprobe_valid_00001_.png"),
            (png(), "unrelated.png")])
        self.assertEqual(set(shots), {"depth", "normal", "valid"})


if __name__ == "__main__":
    unittest.main()
