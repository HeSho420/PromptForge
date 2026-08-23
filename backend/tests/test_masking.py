"""Choosing the region an edit will change.

The auto-masker was measured on two real photographs before this was written,
and the numbers in these docstrings are from that run. The headline was that
for a large class of requests the region was chosen WITHOUT READING THE
REQUEST: SAM scores its candidates on geometry alone, so "remove the
necklace" and "change her shoes" produced byte-identical masks — 9.9% on one
photo, 1.7% on the other.
"""
import base64
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from app.adapters.base import EditResult
from app.config import Settings
from app.core import quality
from app.core.services import Services
from tests.test_workflow_job import DeadLLM


def box_mask(box, size=(200, 300)):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rectangle(box, fill=255)
    return m


class VerdictTests(unittest.TestCase):
    """Deterministic gates. No model, so no opinion — just geometry."""

    SUBJECT = box_mask((60, 40, 140, 260))

    def test_a_solid_region_on_the_subject_passes(self):
        v = quality.mask_verdict(box_mask((70, 60, 130, 200)), self.SUBJECT,
                                 confine=True)
        self.assertTrue(v["ok"])
        self.assertFalse(v["repaired"])

    def test_nothing_at_all_is_reported_not_rendered(self):
        v = quality.mask_verdict(Image.new("L", (200, 300), 0))
        self.assertFalse(v["ok"])
        self.assertIn("nothing matching", v["reason"])

    def test_a_speck_is_not_a_garment(self):
        """Measured: "change the trousers" on a photo of someone in a bikini
        returned a mask covering 0.2% of the frame. The inpaint then edited a
        few hundred pixels and reported success."""
        v = quality.mask_verdict(box_mask((10, 10, 14, 14)))
        self.assertFalse(v["ok"])
        self.assertIn("too little", v["reason"])

    def test_the_whole_frame_is_not_a_region(self):
        v = quality.mask_verdict(box_mask((0, 0, 199, 299)))
        self.assertFalse(v["ok"])
        self.assertIn("not a region", v["reason"])

    def test_scattered_specks_are_rejected(self):
        m = Image.new("L", (200, 300), 0)
        d = ImageDraw.Draw(m)
        for x in range(10, 190, 20):
            for y in range(10, 290, 20):
                d.rectangle([x, y, x + 2, y + 2], fill=255)
        v = quality.mask_verdict(m)
        self.assertFalse(v["ok"])
        self.assertIn("scattered", v["reason"])

    def test_a_mask_off_the_subject_is_trimmed_not_used(self):
        """Measured leak before this: up to 21.3% of a clothing mask lay on
        the background. The subject matte is the one mask measured exact
        here, so it referees the others."""
        v = quality.mask_verdict(box_mask((20, 60, 130, 200)), self.SUBJECT,
                                 confine=True)
        self.assertTrue(v["ok"])
        self.assertTrue(v["repaired"])
        self.assertGreater(v["trimmed"], 10)
        self.assertLess(quality.mask_fraction(v["mask"]),
                        quality.mask_fraction(box_mask((20, 60, 130, 200))))

    def test_a_mask_entirely_off_the_subject_is_refused(self):
        v = quality.mask_verdict(box_mask((2, 270, 40, 298)), self.SUBJECT,
                                 confine=True)
        self.assertFalse(v["ok"])

    def test_a_request_about_an_object_is_not_confined_to_a_person(self):
        """A car or the sky is not on the subject, and clipping it to a
        person's silhouette would delete the request."""
        self.assertTrue(quality.about_the_subject("change her shirt"))
        self.assertTrue(quality.about_the_subject("change the bikini"))
        self.assertTrue(quality.about_the_subject("her hair"))
        self.assertFalse(quality.about_the_subject("change the car to red"))
        self.assertFalse(quality.about_the_subject("make the sky stormy"))


class AdjustSpeedTests(unittest.TestCase):
    def test_growing_reaches_the_same_distance(self):
        """Repeated 3-px passes instead of one large kernel: dilation is
        associative, so the distance is the same and the cost drops from
        quadratic in the kernel to linear. It was ~11 s at 65 px."""
        m = box_mask((90, 140, 110, 160))
        grown = quality.adjust_mask(m, "grow", 24)
        self.assertGreater(quality.mask_fraction(grown),
                           quality.mask_fraction(m) * 2)
        box = grown.point(lambda v: 255 if v > 127 else 0).getbbox()
        self.assertLess(box[0], 90)
        self.assertGreater(box[2], 110)

    def test_shrinking_shrinks(self):
        m = box_mask((60, 60, 160, 240))
        self.assertLess(quality.mask_fraction(quality.adjust_mask(m, "shrink", 12)),
                        quality.mask_fraction(m))

    def test_a_no_op_is_a_no_op(self):
        m = box_mask((60, 60, 160, 240))
        self.assertIs(quality.adjust_mask(m, "keep", 10), m)
        self.assertIs(quality.adjust_mask(m, "grow", 0), m)


class ChooserTests(unittest.TestCase):
    """The order the engines are tried in, and what the caller is told."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))
        self.addCleanup(self.s.stop)
        self.image = Image.new("RGB", (200, 300), (20, 30, 40))
        self.s._pack_active = lambda slug: False        # no matte engine
        self.s._multi_region_mask = lambda *a, **k: None
        self.s._text_mask = lambda *a, **k: (None, {})

    def test_named_parts_win_when_the_vocabulary_matches(self):
        named = box_mask((70, 60, 130, 200))
        self.s._multi_region_mask = lambda *a, **k: named
        choice = self.s.auto_mask(self.image, "change the bikini")
        self.assertEqual(choice.source, "named-part")

    def test_text_is_preferred_over_geometry(self):
        self.s._text_mask = lambda *a, **k: (box_mask((70, 60, 130, 200)),
                                             {"peak": 0.9})
        self.s.segmentation.propose_mask = lambda *a: box_mask((0, 0, 20, 20))
        choice = self.s.auto_mask(self.image, "remove the necklace")
        self.assertEqual(choice.source, "text")

    def test_a_confident_not_found_is_an_answer(self):
        """CLIPSeg reporting a low peak means it looked and did not see it.
        Measured: shoes 0.081 and necklace 0.168 on a waist-up photo where
        neither is present, against 0.88-0.94 for things that are."""
        self.s._text_mask = lambda *a, **k: (None, {"peak": 0.08})
        choice = self.s.auto_mask(self.image, "change her shoes")
        self.assertFalse(choice.ok)
        self.assertEqual(choice.source, "none")
        self.assertIn("nothing matching", choice.reason)

    def test_falling_back_to_geometry_is_declared(self):
        """SAM cannot read. If it is what answered, the caller has to be told
        so, because the region is a guess about shape and position."""
        self.s.segmentation.propose_mask = lambda *a: box_mask((70, 60, 130, 200))
        choice = self.s.auto_mask(self.image, "remove the necklace")
        self.assertEqual(choice.source, "sam")
        self.assertTrue(any("cannot read" in n for n in choice.notes),
                        choice.notes)

    def test_an_unusable_mask_is_refused_rather_than_rendered(self):
        self.s.segmentation.propose_mask = lambda *a: Image.new("L", (200, 300), 0)
        choice = self.s.auto_mask(self.image, "remove the necklace")
        self.assertFalse(choice.ok)

    def test_text_full_coverage_routes_to_whole_frame(self):
        """CLIPSeg confidently finding the request EVERYWHERE is a
        whole-frame edit, not a missing answer. Measured live: a sky-only
        photo asked for a warmer sky scored peak 0.889 at 98% coverage and
        died with "nothing matching is clearly visible" — the opposite of
        what the engine had said."""
        full = box_mask((0, 0, 199, 299))
        self.s._text_mask = lambda *a, **k: (full, {"peak": 0.89})
        choice = self.s.auto_mask(self.image, "make the sky a warm sunset")
        self.assertTrue(choice.ok)
        self.assertEqual(choice.source, "whole-frame")
        self.assertGreater(quality.mask_fraction(choice.mask), 0.99)
        self.assertTrue(any("whole picture" in n for n in choice.notes),
                        choice.notes)

    def test_subject_confined_full_coverage_keeps_the_strict_path(self):
        """A garment mask covering ~everything is a segmenter error, not a
        whole-frame request — the subject-confined path must keep its trim
        and rejection, never repaint the frame."""
        full = box_mask((0, 0, 199, 299))
        self.s._text_mask = lambda *a, **k: (full, {"peak": 0.9})
        choice = self.s.auto_mask(self.image, "change her dress to red")
        self.assertNotEqual(choice.source, "whole-frame")
        self.assertFalse(choice.ok)


class LadderFallThroughTests(unittest.TestCase):
    """A REJECTED candidate is one engine failing, not the chooser's answer.

    The chooser used to pick a single candidate up front and treat its
    rejection as the final verdict, so a named-part mask that failed the
    geometry gates ended the search — auto_mask returned "no usable region
    could be selected" and the caller raised BadMaskError, without ever
    asking the engine that can read the request. Measured: the garment
    segmenter's speckled output on a hard photo is rejected at 0.11%.
    """

    SPECKLE = None      # built in setUp: a real mask that fails the gates
    GOOD = None

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))
        self.addCleanup(self.s.stop)
        self.image = Image.new("RGB", (200, 300), (20, 30, 40))
        self.s._pack_active = lambda slug: False
        self.GOOD = box_mask((70, 60, 130, 200))
        speck = Image.new("L", (200, 300), 0)
        draw = ImageDraw.Draw(speck)
        for x in range(10, 190, 9):
            draw.rectangle((x, x % 280, x + 2, x % 280 + 2), fill=255)
        self.SPECKLE = speck
        self.assertFalse(quality.mask_verdict(speck)["ok"],
                         "the fixture must be a mask the gates reject")
        # The geometry engine is watched, not just stubbed: two of these
        # tests assert it is NEVER consulted.
        self.sam_calls = []

        def sam(*a):
            self.sam_calls.append(a)
            return self.GOOD

        self.s.segmentation.propose_mask = sam
        self.s._multi_region_mask = lambda *a, **k: None
        self.s._text_mask = lambda *a, **k: (None, {})

    def test_a_rejected_named_part_asks_the_engine_that_reads(self):
        self.s._multi_region_mask = lambda *a, **k: self.SPECKLE
        self.s._text_mask = lambda *a, **k: (self.GOOD, {"peak": 0.9})
        choice = self.s.auto_mask(self.image, "change the bikini")
        self.assertTrue(choice.ok, choice.reason)
        self.assertEqual(choice.source, "text")
        self.assertEqual(self.sam_calls, [], "geometry must not be needed")

    def test_a_rejected_named_part_reaches_geometry_when_nothing_can_read(self):
        """The reading engine could not RUN (no ComfyUI dir, no CLIPSeg), so
        the geometric guess is the honest last rung rather than a dead end."""
        self.s._multi_region_mask = lambda *a, **k: self.SPECKLE
        choice = self.s.auto_mask(self.image, "change the bikini")
        self.assertTrue(choice.ok, choice.reason)
        self.assertEqual(choice.source, "sam")
        self.assertEqual(len(self.sam_calls), 1)

    def test_a_rejected_named_part_still_obeys_a_confident_not_found(self):
        """Falling through must not weaken D4: when the reading engine RAN and
        said the thing is not in the picture, that is the answer, and the
        geometric guess (whose default candidate on a portrait is the face)
        is not consulted."""
        self.s._multi_region_mask = lambda *a, **k: self.SPECKLE
        self.s._text_mask = lambda *a, **k: (None, {"peak": 0.08})
        choice = self.s.auto_mask(self.image, "change her shoes")
        self.assertFalse(choice.ok)
        self.assertIn("nothing matching", choice.reason)
        self.assertEqual(self.sam_calls, [], "D4: no geometric guess here")

    def test_a_rejected_text_region_does_not_fall_to_geometry(self):
        """The other half of D4: the reading engine found its best region and
        that region failed the gates. Evidence about the picture, not a
        licence to guess."""
        self.s._text_mask = lambda *a, **k: (self.SPECKLE, {"peak": 0.9})
        choice = self.s.auto_mask(self.image, "remove the necklace")
        self.assertFalse(choice.ok)
        self.assertIn("did not survive", choice.reason)
        self.assertEqual(self.sam_calls, [], "D4: no geometric guess here")

    def test_nothing_available_is_still_reported_as_no_region(self):
        self.s.segmentation.propose_mask = lambda *a: Image.new("L", (200, 300), 0)
        choice = self.s.auto_mask(self.image, "remove the necklace")
        self.assertFalse(choice.ok)
        self.assertEqual(choice.source, "none")


class RealisticInpaint:
    """A non-mock inpaint adapter. `real` is derived from is_mock, and the
    advisory mask check only runs on a real render — so a mock adapter cannot
    reach the code this class exists to exercise."""

    name = "fake-inpaint"
    is_mock = False
    supports_variants = False

    def __init__(self):
        self.calls = []

    def inpaint(self, image, mask, prompt, **kw):
        self.calls.append({"mask": mask, "prompt": prompt})
        return EditResult(image=Image.new("RGB", image.size),
                          adapter=self.name, is_mock=False, meta={})


class ObjectingCritic:
    """Answers the mask-coverage question with "no". Every other question gets
    a reply the parsers discard, so only the objection is under test."""

    def __init__(self):
        self.asked = []

    def ask(self, image, question):
        self.asked.append(question)
        if "translucent red overlay" in question:
            return '{"match": false, "why": "the region misses the shoes"}'
        return "{}"

    def describe(self, image, question):
        return "a person in a room"


class ApprovedRegionSurvivesTheCheckTests(unittest.TestCase):
    """A region the user settled is reported on, never silently re-cut.

    This is the regression test for a hard crash. The check site read the
    chooser's `choice` to decide whether a replacement came from a strong
    enough engine, but `choice` is only assigned on the auto-segmentation
    branch — so an edit that arrived WITH a mask hit
    `UnboundLocalError: cannot access local variable 'choice'` the moment the
    advisory check objected, and every attempt failed the same way. That is
    the ordinary Studio flow: proposing a mask loads it into the canvas, which
    exports it, so "Propose mask -> Run edit" posts it as the user's region.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir="", quality_rounds=0))
        self.addCleanup(self.s.stop)
        self.s.settings.auto_install = False
        self.s.llm = DeadLLM()          # plan falls back to one inpaint step
        self.adapter = RealisticInpaint()
        self.s.inpainting = self.adapter
        self.critic = ObjectingCritic()
        self.s.critic = self.critic
        # If the region is ever re-cut, this records it.
        self.corrections = []
        self.s._correct_mask = lambda *a, **k: (
            self.corrections.append(a) or a[3])
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (5, 5, 5)).save(buf, format="PNG")
        self.asset = self.s.store.save_upload("p.png", buf.getvalue())
        self.s.start()

    @staticmethod
    def _mask_b64(box):
        m = Image.new("L", (64, 64), 0)
        m.paste(255, box)
        buf = io.BytesIO()
        m.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def test_an_objection_to_a_supplied_mask_does_not_crash_the_job(self):
        job = self.s.queue.enqueue("image_edit", {
            "asset_id": self.asset.id, "prompt": "change her shoes",
            "mask_b64": self._mask_b64((8, 8, 40, 40))})
        done = self.s.queue.wait_for(job.id, timeout=60)
        self.assertEqual(done.state.value, "completed", done.error)
        self.assertIsNone(done.error)
        # The specific defect, named so a reintroduction is unmistakable.
        self.assertNotIn("choice", str(done.error or ""))

    def test_the_objection_is_reported_and_the_region_is_kept(self):
        job = self.s.queue.enqueue("image_edit", {
            "asset_id": self.asset.id, "prompt": "change her shoes",
            "mask_b64": self._mask_b64((8, 8, 40, 40))})
        done = self.s.queue.wait_for(job.id, timeout=60)
        self.assertEqual(done.state.value, "completed", done.error)
        msgs = [e["msg"] for e in done.logs]
        self.assertTrue(any("may not match" in m for m in msgs),
                        "the objection must still be surfaced")
        self.assertTrue(any("Keeping the region you approved" in m
                            for m in msgs), msgs)
        self.assertEqual(self.corrections, [],
                         "a region the user approved must not be re-cut")
        # And the region that rendered is the one that was supplied.
        self.assertEqual(len(self.adapter.calls), 1)
        self.assertEqual(self.adapter.calls[0]["mask"].getbbox(),
                         (8, 8, 40, 40))


class SubjectShieldTests(unittest.TestCase):
    """A drawn mask that sweeps across a person the request never mentioned.

    Measured 2026-08-20: a 0.44-fraction drawn mask overlapping the subject
    sent her through the repaint with the rest of the region — softened to
    0.60x sharpness on the SD15 inpaint, redrawn as a DIFFERENT PERSON at
    SDXL full denoise. The shield hands the sampler the drawn region minus
    the person's core, and stands down when the region mostly IS the
    person — the user outranks it."""

    MATTE = box_mask((60, 40, 140, 260))

    def test_a_sweeping_mask_loses_the_subjects_core(self):
        drawn = box_mask((0, 0, 130, 299))
        out = quality.subject_shield(drawn, self.MATTE)
        self.assertTrue(out["applied"])
        shielded = out["mask"]
        self.assertEqual(shielded.getpixel((100, 150)), 0,
                         "the subject's core must be protected")
        self.assertEqual(shielded.getpixel((20, 150)), 255,
                         "the rest of the drawn region must still render")
        self.assertEqual(shielded.getpixel((62, 150)), 255,
                         "a rim of the subject stays repaintable so the "
                         "new content can blend at the silhouette")

    def test_a_mask_barely_touching_the_subject_is_untouched(self):
        drawn = box_mask((0, 0, 70, 299))
        out = quality.subject_shield(drawn, self.MATTE)
        self.assertFalse(out["applied"])
        self.assertIs(out["mask"], drawn)
        self.assertIn("barely", str(out["note"]))

    def test_a_mask_that_is_the_subject_outranks_the_shield(self):
        drawn = box_mask((58, 38, 142, 262))
        out = quality.subject_shield(drawn, self.MATTE)
        self.assertFalse(out["applied"])
        self.assertIn("deliberate", str(out["note"]))

    def test_a_whole_frame_matte_is_no_evidence(self):
        matte = box_mask((0, 0, 199, 299))
        out = quality.subject_shield(box_mask((0, 0, 130, 299)), matte)
        self.assertFalse(out["applied"])
        self.assertIn("no reliable subject matte", str(out["note"]))


class _ShieldJob:
    def __init__(self):
        self.lines = []

    def log(self, level, msg):
        self.lines.append(msg)


class _ShieldCritic:
    def __init__(self, answer):
        self.answer = answer
        self.asked = 0

    def ask(self, image, question, **kwargs):
        self.asked += 1
        return self.answer


class SubjectShieldGateTests(unittest.TestCase):
    """When the shield may not touch the drawn region at all.

    BiRefNet mattes whatever is salient — protecting a vase from "replace
    the vase" would block the very edit — so the shield only stands between
    the sampler and a confirmed PERSON, and never when the request already
    names them."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))
        self.addCleanup(self.s.stop)
        self.s._pack_active = lambda name: True
        self.s._region_mask = lambda *a, **k: box_mask((60, 40, 140, 260))
        self.img = Image.new("RGB", (200, 300), (90, 110, 130))
        self.drawn = box_mask((0, 0, 130, 299))
        self.job = _ShieldJob()

    def _run(self, instruction, critic):
        self.s.critic = critic
        return self.s._shield_subject(self.job, self.img, self.drawn,
                                      instruction, True)

    def test_a_request_about_the_person_stands_unshielded(self):
        critic = _ShieldCritic("yes")
        out = self._run("give her a red jacket", critic)
        self.assertIs(out, self.drawn)
        self.assertEqual(critic.asked, 0,
                         "no model round-trip when the words already "
                         "settle it")

    def test_a_salient_object_that_is_not_a_person_is_not_protected(self):
        out = self._run("replace the tree with a statue",
                        _ShieldCritic("No, it is a large vase."))
        self.assertIs(out, self.drawn)

    def test_a_confirmed_person_is_protected(self):
        out = self._run("replace the tree with a statue",
                        _ShieldCritic("Yes."))
        self.assertIsNot(out, self.drawn)
        self.assertEqual(out.getpixel((100, 150)), 0)
        self.assertEqual(out.getpixel((20, 150)), 255)
        self.assertTrue(any("Subject shield" in m for m in self.job.lines))

    def test_mock_runs_leave_the_region_alone(self):
        self.s.critic = _ShieldCritic("yes")
        out = self.s._shield_subject(self.job, self.img, self.drawn,
                                     "replace the tree with a statue", False)
        self.assertIs(out, self.drawn)

    def test_the_hook_only_amends_drawn_masks(self):
        """The driver applies the shield inside the mask_source == "user"
        branch and nowhere else — auto masks already have the named-part
        and face guarantees."""
        import inspect

        from app.core import services as services_module
        src = inspect.getsource(services_module)
        self.assertIn('if mask_source == "user":', src)
        self.assertIn("shielded = self._shield_subject(", src)


if __name__ == "__main__":
    unittest.main()
