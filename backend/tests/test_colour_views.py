"""Which photographs colour the avatar mesh, and from which cameras.

The texturer can only be as right as what it is told. Two measured mistakes
shaped these rules: the old code told it a frame sat at its bin's canonical
angle (0/90/180/270) when the frame really sat up to 35° away — the texture
was then projected from a camera that was somewhere else — and the orbit
model bakes hallucinated surroundings into some renders, which the texturer
then painted onto the avatar as if they were the subject.
"""
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.config import Settings
from app.core.services import Services


class FakeJob:
    def __init__(self):
        self.lines = []

    def log(self, _level, message):
        self.lines.append(message)


def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (90, 90, 90)).save(buf, format="PNG")
    return buf.getvalue()


class ColourViewSelection(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))
        self.addCleanup(self.s.stop)
        self.matted: list[float] = []
        self.job = FakeJob()

    def frame(self, azimuth, synthetic=True):
        asset = self.s.store.save_upload(
            f"angle_{int(azimuth):03d}.png", png_bytes(),
            meta={"azimuth": azimuth, "synthetic": synthetic})
        return asset.id

    def photos(self, frame_ids):
        def fake_matte(job, image, _az=None):
            return image
        original = self.s._matte_on_grey

        def recording(job, image):
            self.matted.append(True)
            return image
        self.s._matte_on_grey = recording
        try:
            return self.s._colour_photos(self.job, frame_ids)
        finally:
            self.s._matte_on_grey = original

    def test_the_true_azimuth_is_passed_not_the_bin_name(self):
        """A frame at 86° must be reported as 86, not as 'the 90 bin'. The
        texturer's refinement window is narrower than the bin tolerance, so
        the canonical angle put the camera somewhere the photo was not."""
        ids = [self.frame(az) for az in (0, 86, 171, 274)]
        azimuths = [az for az, _ in self.photos(ids)]
        self.assertEqual(sorted(azimuths), [0.0, 86.0, 171.0, 274.0])

    def test_eight_bins_are_offered_when_the_orbit_has_them(self):
        ids = [self.frame(az) for az in range(0, 360, 17)]
        self.assertEqual(len(self.photos(ids)), 8)

    def test_one_frame_never_stands_in_for_two_bins(self):
        """A sparse orbit: a frame at 22° is within tolerance of both the 0
        and the 45 bin, and must be used once."""
        ids = [self.frame(22)]
        photos = self.photos(ids)
        self.assertEqual(len(photos), 1)

    def test_a_frame_too_far_from_every_bin_is_not_used(self):
        ids = [self.frame(30)]  # 30 from 0, 15 from 45 — used for 45 only
        photos = self.photos(ids)
        self.assertEqual(len(photos), 1)
        ids2 = [self.frame(203)]  # 23 from 180: inside tolerance
        self.assertEqual(len(self.photos(ids2)), 1)

    def test_the_front_view_comes_first(self):
        """The texturer chains its tone matching outward from the first
        view; the front is the reference exposure."""
        ids = [self.frame(az) for az in (171, 86, 0, 274)]
        photos = self.photos(ids)
        self.assertEqual(photos[0][0], 0.0)

    def test_synthetic_frames_are_matted_and_real_ones_are_not(self):
        """The intake already staged the real photographs on clean grey;
        the orbit renders arrive with hallucinated surroundings baked in."""
        ids = [self.frame(0, synthetic=False), self.frame(90, synthetic=True)]
        self.photos(ids)
        self.assertEqual(len(self.matted), 1)

    def test_no_frames_gives_no_photos(self):
        self.assertEqual(self.photos([]), [])


class MattingFallback(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))
        self.addCleanup(self.s.stop)

    def test_no_rmbg_pack_returns_the_image_unchanged(self):
        """Matting is an upgrade, never a requirement — the texturer's own
        quality gate is the second line of defence."""
        self.s._pack_active = lambda slug: False
        img = Image.new("RGB", (8, 8), (10, 20, 30))
        self.assertIs(self.s._matte_on_grey(FakeJob(), img), img)

    def test_a_matting_failure_returns_the_image_unchanged(self):
        self.s._pack_active = lambda slug: True

        def boom(*_a, **_k):
            raise RuntimeError("comfy is down")
        self.s.comfy.upload_image = boom
        img = Image.new("RGB", (8, 8), (10, 20, 30))
        job = FakeJob()
        self.assertIs(self.s._matte_on_grey(job, img), img)
        self.assertTrue(any("Could not matte" in m for m in job.lines))


if __name__ == "__main__":
    unittest.main()
