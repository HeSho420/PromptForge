"""The animation is checked against the photograph it was made from.

Live twice: /api/video returned a clean, well-formed clip of a completely
different person after twenty minutes, and every stage reported success. The
graph is wired correctly — the executed workflow carries the uploaded frame
into WanImageToVideo's start_image — but wan22-ti2v-5b at denoise 1.0 does
not reliably hold a likeness, and nothing downstream ever looked at the
pixels. This is the look.
"""
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.config import Settings
from app.core.jobs import Job
from app.core.services import Services


def clip(path, colour, frames=4):
    """A tiny animated WEBP of a flat colour."""
    imgs = [Image.new("RGB", (32, 32), colour) for _ in range(frames)]
    imgs[0].save(path, format="WEBP", save_all=True,
                 append_images=imgs[1:], duration=41)


class Asset:
    def __init__(self, path):
        self.path = str(path)
        self.id = "vid"


class VideoDriftTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))
        self.addCleanup(self.s.stop)
        self.job = Job(id="v", type="video", payload={})

    def _run(self, source_colour, clip_colour):
        p = Path(self.tmp.name) / "c.webp"
        clip(p, clip_colour)
        return self.s._check_video_kept_subject(
            self.job, Image.new("RGB", (64, 64), source_colour), Asset(p))

    def messages(self):
        return " ".join(str(e.get("msg", "")) for e in self.job.logs)

    def test_a_clip_of_the_same_thing_passes_quietly(self):
        drift = self._run((30, 90, 180), (30, 90, 180))
        self.assertIsNotNone(drift)
        self.assertLess(drift, self.s._VIDEO_DRIFT_LIMIT)
        self.assertIn("still matches your photograph", self.messages())

    def test_a_clip_of_something_else_is_called_out(self):
        """Black against white is the extreme case. On real pictures the
        limit of 0.18 sits above the same photo edited (0.022-0.039) and
        below two unrelated photographs (0.230-0.250); both clips that came
        back as a different person measured 0.376."""
        drift = self._run((0, 0, 0), (255, 255, 255))
        self.assertGreater(drift, self.s._VIDEO_DRIFT_LIMIT)
        self.assertIn("drifted a long way", self.messages())

    def test_the_limit_separates_an_edit_from_a_different_picture(self):
        """The threshold has to sit between those two measured bands, or it
        would either fire on every ordinary edit or never fire at all — the
        0.50 first tried here would have missed both real failures."""
        self.assertGreater(self.s._VIDEO_DRIFT_LIMIT, 0.05)
        self.assertLess(self.s._VIDEO_DRIFT_LIMIT, 0.23)

    def test_the_check_never_fails_the_job(self):
        """A drifted clip is still the only clip the user has."""
        missing = Asset(Path(self.tmp.name) / "does-not-exist.webp")
        self.assertIsNone(self.s._check_video_kept_subject(
            self.job, Image.new("RGB", (8, 8)), missing))


if __name__ == "__main__":
    unittest.main()
