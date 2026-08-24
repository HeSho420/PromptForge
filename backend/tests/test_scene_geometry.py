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

    def test_a_feathered_crop_edge_is_still_a_crop(self):
        # Measured live: BiRefNet's soft edge stopped fem.png's matte at
        # row 1439 of 1444 — a 4px gap against the old fixed 3px test —
        # so 53% of the frame width became four phantom "contact points"
        # and EVERY environment render failed "nothing walkable under
        # the subject's feet" for a subject cropped at mid-thigh. The
        # tolerance scales with the frame now.
        m = Image.new("L", (300, 500), 0)
        import PIL.ImageDraw as D
        D.Draw(m).rectangle((100, 60, 200, 495), fill=255)  # 4px short
        info = scene_geometry.subject_geometry(m)
        self.assertTrue(info["cut_at_bottom"])
        self.assertEqual(info["contact_points"], [])

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


class ChangedRegionProbeTests(unittest.TestCase):
    """A 'missing' verdict gets one last, region-scoped look: crop what
    the edit actually changed and ask what it shows. Live in one day:
    a colorization, a placed second woman and a huge stone statue were
    each reported missing twice by whole-frame probes."""

    def test_changed_bbox_finds_the_edit(self):
        from app.core.quality import changed_bbox
        before = Image.new("RGB", (200, 200), (50, 50, 50))
        after = before.copy()
        after.paste((200, 200, 200), (20, 30, 120, 170))
        box = changed_bbox(before, after)
        self.assertEqual(box, (20, 30, 120, 170))
        self.assertIsNone(changed_bbox(before, before))

    def test_region_probe_overrules_missing_to_met_only(self):
        from app.core import quality

        class C:
            def __init__(self):
                self.calls = []

            def ask(self, image, q, schema=None):
                self.calls.append((image.size, q))
                if "main thing shown" in q:
                    return '{"answer": "a large stone statue"}'
                return '{"answer": "a beach"}'   # whole-frame probe: wrong

        before = Image.new("RGB", (300, 300), (10, 10, 10))
        after = before.copy()
        after.paste((240, 240, 240), (10, 10, 150, 290))
        checklist = [{"need": "a large stone statue",
                      "probe": "What large object is present?",
                      "expect": "stone statue"}]
        report = quality.verify_adherence(C(), after, "x", checklist,
                                          before=before)
        self.assertEqual(report["missing"], [])
        self.assertEqual(report["met"], ["a large stone statue"])
        self.assertTrue(report["region_settled"])
        self.assertEqual(report["accuracy"], 100)


class LargeMaskCheckpointTests(unittest.TestCase):
    def test_large_mask_xl_switch_stays_reverted(self):
        # Measured and REVERTED same day: the XL swap on a drawn mask
        # overlapping the subject redrew the person at accuracy 20 where
        # SD15 soft-inpaint had kept her intact at 0.60x sharpness. The
        # revert note must survive until a subject-protecting design
        # replaces it.
        import inspect

        from app.core.services import Services
        src = inspect.getsource(Services._handle_image_edit)
        self.assertNotIn("Inpaint model switched to", src)
        self.assertIn("REVERTED same day", src)


class PlacementCorrectionTests(unittest.TestCase):
    """A person joining a photo stands BESIDE whoever is already there,
    at a comparable size, on the same line. Live: the scene-picked spot
    pasted the newcomer on the existing subject's torso at 26% scale."""

    def _existing(self, w=1000, h=1000):
        m = Image.new("L", (w, h), 0)
        m.paste(255, (400, 100, 600, 900))    # a centred standing figure
        return m

    def test_sticker_box_becomes_a_neighbour(self):
        from app.core.quality import placement_correction
        box = {"x": 450, "y": 400, "w": 120, "h": 180}   # on the torso
        fixed, notes = placement_correction(box, self._existing(),
                                            (1000, 1000))
        self.assertGreaterEqual(fixed["h"], int(800 * 0.9))   # ~her height
        self.assertAlmostEqual(fixed["y"] + fixed["h"], 900, delta=5)
        # moved off the torso: overlap with her matte is now small
        self.assertTrue(fixed["x"] + fixed["w"] <= 420
                        or fixed["x"] >= 580,
                        f"still overlapping at x={fixed['x']}")
        self.assertEqual(len(notes), 3)

    def test_good_boxes_and_missing_matte_pass_through(self):
        from app.core.quality import placement_correction
        good = {"x": 40, "y": 130, "w": 200, "h": 770}
        fixed, notes = placement_correction(dict(good), self._existing(),
                                            (1000, 1000))
        self.assertEqual((fixed["x"], fixed["h"]), (40, 770))
        self.assertEqual(notes, [])
        same, notes2 = placement_correction(dict(good), None, (1000, 1000))
        self.assertEqual(notes2, [])

    def test_matte_group_count(self):
        from app.core.scene_geometry import matte_group_count
        one = Image.new("L", (400, 400), 0)
        one.paste(255, (150, 50, 250, 380))
        self.assertEqual(matte_group_count(one), 1)
        two = Image.new("L", (400, 400), 0)
        two.paste(255, (40, 50, 140, 380))
        two.paste(255, (260, 50, 360, 380))
        self.assertEqual(matte_group_count(two), 2)
        self.assertEqual(matte_group_count(Image.new("L", (64, 64), 0)), 0)

    def test_compose_checklist_is_a_count_question(self):
        import inspect

        from app.core.services import Services
        src = inspect.getsource(Services._handle_image_edit)
        self.assertIn("How many people are visible", src)
        self.assertIn("matte_group_count", src)

    def test_checklists_never_ask_about_other_photos(self):
        from app.core.quality import strip_provenance
        self.assertEqual(
            strip_provenance("add the woman from the second photo "
                             "standing in this garden"),
            "add the woman standing in this garden")
        self.assertEqual(strip_provenance("remove the hat"),
                         "remove the hat")


class ColorizeSettlerTests(unittest.TestCase):
    """Colorization has an arithmetic truth. Live: chroma 0.0 → 67.8 with
    natural colours while the verifier reported 'missing: natural
    realistic colors' across two renders."""

    def _img(self, color):
        return Image.new("RGB", (32, 32), color)

    def test_grayscale_to_colour_is_delivered(self):
        from app.core.quality import colorize_delivered
        bw = self._img((120, 120, 120))
        colour = self._img((180, 90, 40))
        self.assertTrue(colorize_delivered("colorize this photo",
                                           bw, colour))
        self.assertFalse(colorize_delivered("colorize this photo",
                                            bw, self._img((90, 90, 90))))

    def test_not_settleable_cases_stay_none(self):
        from app.core.quality import colorize_delivered
        colour = self._img((180, 90, 40))
        # request is not a colorization
        self.assertIsNone(colorize_delivered("remove the hat",
                                             self._img((120,) * 3), colour))
        # the input already had colour
        self.assertIsNone(colorize_delivered("colorize this photo",
                                             colour, colour))

    def test_wiring_pin(self):
        import inspect

        from app.core.services import Services
        src = inspect.getsource(Services._handle_image_edit)
        self.assertIn("colorize_delivered", src)
        self.assertIn("chroma settles it", src)


class TextRenderIntentTests(unittest.TestCase):
    """Readable in-image text routes to the text-rendering engine.
    Live: SD lettering came out 'CDOSED / LOSSE' while the triage reason
    itself said readable text was required."""

    def test_text_phrasings_match(self):
        from app.core.quality import text_render_intent
        for yes in ("a sign that says CLOSED",
                    'a poster with the text "SALE TODAY"',
                    "a neon sign reading OPEN ALL NIGHT",
                    "a t-shirt with the word CHAMPION",
                    "OPEN written across the door",
                    "a mug labelled WORLD'S BEST DAD"):
            self.assertTrue(text_render_intent(yes), yes)

    def test_wordless_prompts_do_not(self):
        from app.core.quality import text_render_intent
        for no in ("a man reading a book in a cafe",
                   "a lighthouse at dusk, heavy fog",
                   "a woman walking her dog",
                   "graffiti-covered alley at night"):
            self.assertFalse(text_render_intent(no), no)

    def test_generate_route_pins_the_coercion(self):
        import inspect

        from app.core.services import Services
        src = inspect.getsource(Services._workflow_inner)
        self.assertIn("text_render_intent", src)
        self.assertIn('"generate_zimage"', src)


class StyleJudgingTests(unittest.TestCase):
    """A deliberate medium change is judged AS that medium. Measured: a
    delivered watercolor scored realism 20 against the photograph frame
    while verify passed every round — three renders of a success."""

    def test_style_departure_table(self):
        from app.core.quality import style_departure
        for yes in ("turn this photo into a watercolor painting",
                    "make it an anime scene",
                    "convert this to a pencil sketch",
                    "pixel art version please",
                    "in the style of ukiyo-e",
                    "make a caricature of him"):
            self.assertTrue(style_departure(yes), yes)
        for no in ("change the background to a beach",
                   "make the lighting softer",
                   "remove the car",
                   "make it look professional",
                   "put her in a nightclub"):
            self.assertFalse(style_departure(no), no)

    def test_a_photo_destination_is_not_a_style_departure(self):
        # "sketch" names the SOURCE here — measured live: the delivered
        # photo was judged as a deliberate style piece while verify
        # false-missed "realistic photograph" twice.
        from app.core.quality import photo_target, style_departure
        for s in ("turn this sketch into a realistic photograph",
                  "turn my drawing into a photo",
                  "make this painting photorealistic"):
            self.assertFalse(style_departure(s), s)
            self.assertTrue(photo_target(s), s)
        # ...and a photographic SOURCE keeps its style departure.
        self.assertTrue(style_departure(
            "turn this photo into a watercolor painting"))
        self.assertFalse(photo_target("remove the car"))

    def test_photo_target_checklist_is_deterministic(self):
        # The 7B built "Was a sketch turned into a photo?" — unanswerable
        # from one image — and answered "Is the image photorealistic?"
        # with "no" on a delivered photo. Both burned a Kontext re-render.
        from app.core import quality
        checks = quality.request_checklist(
            object(), "turn this sketch into a realistic photograph")
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0]["expect"], "photograph")
        self.assertIn("photograph or a drawing", checks[0]["probe"])
        self.assertTrue(quality.answer_satisfies(
            "It is a photograph.", checks[0]["expect"]))
        self.assertFalse(quality.answer_satisfies(
            "A drawing.", checks[0]["expect"]))

    def test_scorecard_uses_the_art_frame_for_styles(self):
        from app.core import quality
        rec: list[str] = []

        class C:
            def ask(self, image, q, schema=None):
                rec.append(q)
                return ('{"realism": 90, "prompt_accuracy": 90, '
                        '"identity_preservation": 90, '
                        '"scene_consistency": 90, "artifact_free": 90, '
                        '"visual_quality": 90}')

        img = Image.new("RGB", (8, 8))
        quality.scorecard(C(), img, "watercolor of x", style=True)
        quality.scorecard(C(), img, "remove the hat", style=False)
        self.assertIn("DELIBERATELY transformed", rec[0])
        self.assertIn("NOT photographic realism", rec[0])
        self.assertIn("authentic, unedited", rec[1])

    def test_soft_flag_skipped_for_style_edits_grain_kept(self):
        from app.core.quality import objective_flags
        rep = {"sharpness_ratio": 0.2}
        self.assertTrue(any("softer" in f
                            for f in objective_flags(rep, "img2img")))
        self.assertEqual(
            [f for f in objective_flags(rep, "img2img", style=True)
             if "softer" in f], [])
        self.assertTrue(any("grain" in f for f in objective_flags(
            {"sharpness_ratio": 9.0}, "img2img", style=True)))

    def test_soft_flag_skipped_for_medium_shifts(self):
        # A pencil sketch's stroke texture out-measures any photograph:
        # 0.12x flagged on a DELIVERED sketch-to-photo success.
        from app.core.quality import objective_flags
        rep = {"sharpness_ratio": 0.12}
        self.assertEqual(
            [f for f in objective_flags(rep, "kontext", medium_shift=True)
             if "softer" in f], [])
        self.assertTrue(any("softer" in f
                            for f in objective_flags(rep, "kontext")))

    def test_soft_flag_skipped_for_background_swaps_grain_kept(self):
        # A background swap re-authors the frame: a dim bar against a
        # sparkly beach measured 0.11-0.32x on renders whose kept subject
        # was pixel-preserved — "much softer" flagged on 6/6 club runs.
        from app.core.quality import objective_flags
        rep = {"sharpness_ratio": 0.12}
        self.assertEqual(
            [f for f in objective_flags(rep, "background")
             if "softer" in f], [])
        self.assertTrue(any("softer" in f
                            for f in objective_flags(rep, "inpaint")))
        self.assertTrue(any("grain" in f for f in objective_flags(
            {"sharpness_ratio": 6.0}, "background")))


class CutoutIntentTests(unittest.TestCase):
    """'Remove the background' delivers TRANSPARENCY (ChatGPT-editor
    parity): the request class was previously excluded from the repaint
    route and then handled by nothing at all."""

    def test_removal_phrasings_match(self):
        from app.core.quality import cutout_intent
        for yes in ("remove the background",
                    "delete the background",
                    "erase the backdrop",
                    "background removal please",
                    "make the background transparent",
                    "transparent background",
                    "i want this without a background",
                    "cut her out",
                    "cut out the person",
                    "turn this into a sticker",
                    "isolate the subject"):
            self.assertTrue(cutout_intent(yes), yes)

    def test_object_removal_and_repaints_do_not(self):
        from app.core.quality import cutout_intent
        for no in ("remove the man in the background",
                   "remove the hat",
                   "blur the background",
                   "change the background to a pool",
                   "make the background darker",
                   "put a sticker on the wall",
                   "cut the image in half",
                   "erase the tattoo"):
            self.assertFalse(cutout_intent(no), no)

    def test_product_shots_route_to_the_background_engine(self):
        # "on a white background" read as ADD_OBJECT before this arm —
        # a product shot is a background replacement
        from app.core.quality import default_edit_step
        for t in ("put this on a white background",
                  "put the product on a plain white background",
                  "white studio background"):
            self.assertEqual(default_edit_step(t)["task"], "background", t)
        self.assertEqual(default_edit_step("blur the background")["task"],
                         "inpaint")
        self.assertEqual(
            default_edit_step("keep the background the same")["task"],
            "inpaint")

    def test_default_step_routes_cutout_before_background(self):
        from app.core.quality import default_edit_step
        step = default_edit_step("remove the background")
        self.assertEqual(step["task"], "cutout")
        self.assertEqual(step["operation"], "CUTOUT")
        # replacement phrasing still reaches the repaint route
        self.assertEqual(
            default_edit_step("change the background to a beach")["task"],
            "background")


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
        # The plan's elements travel as consistent-evidence terms.
        self.assertIn('"consistent": list(', src)

    def test_a_planned_element_answer_is_unclear_not_missing(self):
        # Measured 3/3 on the nightclub environment: the render showed the
        # club's own bar, the examiner honestly said "a bar or a lounge",
        # the synonym judge (correctly) refused to equate that with
        # "nightclub", and every run burned a re-render on a place that
        # was consistent with its own plan.
        from app.core import quality

        class C:
            def ask(self, image, q, schema=None):
                # Short, as the "few words" schema probe answers live —
                # one shared token with an element, not most of one.
                return '{"answer": "a bar or a lounge"}'

        class RefusingSynonymLLM:
            def complete(self, system, prompt, max_tokens=4096):
                class R:
                    text = '{"satisfies": false}'
                return R()

        checklist = [{"need": "nightclub",
                      "probe": "Where does this photo appear to be taken?",
                      "expect": "nightclub",
                      "consistent": [
                          "glowing red and blue lights",
                          "crowd of people in casual attire",
                          "bar counter with bottles and glasses",
                          "large speakers mounted on the wall"]}]
        img = Image.new("RGB", (64, 64))
        report = quality.verify_adherence(C(), img, "x", checklist,
                                          llm=RefusingSynonymLLM())
        # The one item lands unclear, so the examiner "answered too little
        # to be trusted" and adherence falls back to the scorecard —
        # crucially NOT a "missing" that buys a re-render.
        self.assertIsNone(report)

    def test_an_unrelated_answer_still_misses(self):
        from app.core import quality

        class C:
            def ask(self, image, q, schema=None):
                return '{"answer": "a sunny beach with the ocean"}'

        checklist = [{"need": "nightclub", "probe": "Where?",
                      "expect": "nightclub",
                      "consistent": ["bar", "dance floor", "neon lights"]}]
        img = Image.new("RGB", (64, 64))
        report = quality.verify_adherence(C(), img, "x", checklist)
        self.assertEqual(report["missing"], ["nightclub"])

    def test_style_steps_are_kontext_eligible(self):
        # img2img restyled with an INPAINTING checkpoint and was measured
        # catastrophic (10s across the board, watercolor missing, 65% of
        # face pixels moved) — whole-image style instructions go to
        # Kontext when its weights are installed.
        import inspect

        from app.core.services import Services
        src = inspect.getsource(Services._handle_image_edit)
        self.assertIn('s.get("operation") == "CHANGE_STYLE"', src)

    def test_relight_after_background_is_pruned(self):
        # The environment step lights its own scene non-destructively; a
        # trailing CHANGE_LIGHTING step re-ran the FULL IC-Light redraw —
        # measured live: 92% of face pixels moved, identity gone.
        import inspect

        from app.core.services import Services
        src = inspect.getsource(Services._handle_image_edit)
        self.assertIn("dropped the separate relight", src)
        self.assertIn('s["task"] != "relight"', src)


class RestoreIntentTests(unittest.TestCase):
    """A restoration's requirement is the ABSENCE of damage. The 7B
    checklist builder extracted "old damaged photo" as the deliverable
    from "restore this old damaged photo" — measured live: the verifier
    then reported the restored result missing its own damage and burned
    a full Kontext re-render on a success."""

    def test_restoration_phrasings_match(self):
        from app.core.quality import restore_intent
        self.assertTrue(restore_intent("restore this old damaged photo"))
        self.assertTrue(restore_intent("fix up this old photo"))
        self.assertTrue(restore_intent("remove the scratches"))
        self.assertTrue(restore_intent("repair the damage"))
        self.assertFalse(restore_intent("restore the deleted layer"))
        self.assertFalse(restore_intent("fix the car"))
        self.assertFalse(restore_intent("put her in a nightclub"))

    def test_restore_checklist_is_deterministic_and_inverted(self):
        from app.core import quality
        checks = quality.request_checklist(
            object(), "restore this old damaged photo")
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0]["expect"], "no")
        self.assertIn("damage", checks[0]["probe"])
        self.assertTrue(quality.answer_satisfies(
            "No, it looks clean.", checks[0]["expect"]))
        self.assertFalse(quality.answer_satisfies(
            "Yes, there are scratches.", checks[0]["expect"]))

    def test_kontext_fade_steering_pin(self):
        # Measured: Kontext's first live restore repaired every scratch
        # and blotch but kept the sepia palette (chroma 42.5 -> 46.2
        # against a 69.3 ground truth). Faded colour is measured, B&W is
        # left to the colorize route.
        import inspect

        from app.core.services import Services
        src = inspect.getsource(Services._handle_image_edit)
        self.assertIn("restore the original natural colours", src)
        self.assertIn("8 <= quality.mean_chroma(current) < 50", src)


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


class LightingPromptTests(unittest.TestCase):
    """IC-Light conditioning is LIGHTING ONLY — the compiled environment
    prompt used to ride along with a hard-coded "natural light on the
    subject", which fights every dim scene (the daylight-in-a-club
    partway shift)."""

    def test_the_planned_lighting_leads(self):
        p = scene_geometry.lighting_prompt(
            "dim moody nightclub lighting with colored neon accents")
        self.assertTrue(p.startswith("dim moody nightclub lighting"))
        self.assertNotIn("natural light", p)
        self.assertIn("colour temperature", p)

    def test_no_plan_means_natural_light(self):
        self.assertTrue(scene_geometry.lighting_prompt(None)
                        .startswith("natural light on the subject"))
        self.assertTrue(scene_geometry.lighting_prompt("  ")
                        .startswith("natural light on the subject"))

    def test_a_directive_is_not_lighting_language(self):
        # The spec planner answered lighting_wish "keep" on the live
        # nightclub spec — a directive, which would have led the
        # conditioning with "keep on the subject, ...".
        self.assertTrue(scene_geometry.lighting_prompt("keep")
                        .startswith("natural light on the subject"))
        self.assertTrue(scene_geometry.lighting_prompt("Unchanged.")
                        .startswith("natural light on the subject"))

    def test_services_thread_the_wish_not_the_env_prompt(self):
        import inspect

        from app.core import services as services_module
        src = inspect.getsource(services_module)
        self.assertGreaterEqual(
            src.count("lighting=(env_spec or {}).get("), 2,
            "the first render AND the ladder retry must both match the "
            "lighting — a kept retry without the match delivered a "
            "daylight subject in a dim scene (subject luma -0.1 while "
            "the scene dimmed 54 levels)")
        self.assertIn("scene_geometry.lighting_prompt(lighting)", src)


if __name__ == "__main__":
    unittest.main()
