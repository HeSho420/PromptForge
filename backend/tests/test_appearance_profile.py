"""An avatar's appearance description must be a description.

Seen live: an avatar was saved whose age_range was literally
"<e.g. mid 20s to early 30s>" and whose hair was "<length, texture, colour>" —
the vision model had echoed the question and nothing checked. That text is fed
into identity renders, so a stored placeholder becomes part of every prompt
made from that avatar.
"""
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.config import Settings
from app.core.jobs import Job
from app.core.services import Services, _is_placeholder


class PlaceholderTests(unittest.TestCase):
    def test_the_prompt_s_own_examples_are_recognised(self):
        for echoed in ("<e.g. mid 20s to early 30s>",
                       "<e.g. slim and athletic / broad-shouldered>",
                       "<length, texture, colour>",
                       "<face shape and notable features>",
                       "e.g. tall / average / petite"):
            self.assertTrue(_is_placeholder(echoed), echoed)

    def test_real_descriptions_are_not(self):
        for real in ("mid 20s to early 30s", "lean and athletic",
                     "light-medium with a warm undertone",
                     "short, straight hair with blonde highlights",
                     "oval face, large green eyes, high cheekbones",
                     "", "none"):
            self.assertFalse(_is_placeholder(real), real)


class Critic:
    """A vision model that answers the first photo with the question."""

    def __init__(self, replies):
        self.replies = list(replies)

    def ask(self, _image, _question):
        return self.replies.pop(0) if self.replies else "{}"


class AppearanceProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))
        self.addCleanup(self.s.stop)
        self.job = Job(id="a", type="avatar", payload={})
        img = Image.new("RGB", (32, 32), (200, 180, 160))
        self.asset = self.s.store.save_upload("p.png", _png(img))

    def test_an_echoed_template_is_not_stored(self):
        self.s.critic = Critic([json.dumps({
            "age_range": "<e.g. mid 20s to early 30s>",
            "build": "<e.g. slim and athletic / broad-shouldered / heavy-set>",
            "hair": "<length, texture, colour>"})])
        profile = self.s._appearance_profile(self.job, [self.asset.id])
        self.assertNotIn("age_range", profile)
        self.assertNotIn("build", profile)
        self.assertNotIn("hair", profile)
        self.assertIn("unanswered appearance field",
                      " ".join(str(e.get("msg", "")) for e in self.job.logs))

    def test_a_real_answer_still_comes_through(self):
        self.s.critic = Critic([json.dumps({
            "age_range": "late 20s to early 30s",
            "hair": "short, straight, blonde highlights"})])
        profile = self.s._appearance_profile(self.job, [self.asset.id])
        self.assertEqual(profile["age_range"], "late 20s to early 30s")
        self.assertEqual(profile["estimated"], "true")

    def test_a_partly_echoed_answer_keeps_the_real_fields(self):
        self.s.critic = Critic([json.dumps({
            "age_range": "<e.g. mid 20s to early 30s>",
            "hair": "long, straight, dark brown"})])
        profile = self.s._appearance_profile(self.job, [self.asset.id])
        self.assertNotIn("age_range", profile)
        self.assertEqual(profile["hair"], "long, straight, dark brown")


def _png(image: Image.Image) -> bytes:
    import io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


if __name__ == "__main__":
    unittest.main()
