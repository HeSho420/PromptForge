"""Reading and writing real video files.

Everything here goes through an actual encode/decode round trip — a fake would
prove nothing about whether a user's uploaded clip can be read on a machine
with no ffmpeg on PATH, which is exactly the case this module exists for.
"""
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from app.core import video


def moving_dot(n: int = 24, size=(320, 240)) -> list[Image.Image]:
    """A clip with unmistakable motion: a dot crossing the frame."""
    frames = []
    for i in range(n):
        im = Image.new("RGB", size, (20, 24, 30))
        x = 20 + int(i * (size[0] - 80) / max(1, n - 1))
        ImageDraw.Draw(im).ellipse((x, 100, x + 40, 140), fill=(220, 90, 60))
        frames.append(im)
    return frames


def dot_x(im: Image.Image) -> float:
    """Horizontal centre of the dot, or -1 when it isn't there."""
    px = im.convert("RGB").load()
    xs = [x for x in range(im.width) if px[x, 120][0] > 150]
    return sum(xs) / len(xs) if xs else -1.0


class RoundTripTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = video.write_video(moving_dot(), Path(self.tmp.name) / "d.mp4",
                                      fps=12)

    def test_a_written_clip_reads_back_with_its_own_numbers(self):
        info = video.probe(self.path)
        self.assertEqual((info.width, info.height), (320, 240))
        self.assertAlmostEqual(info.fps, 12.0, places=1)
        self.assertEqual(info.frame_count, 24)

    def test_every_frame_comes_back(self):
        frames = video.read_frames(self.path)
        self.assertEqual(len(frames), 24)
        self.assertEqual(frames[0].size, (320, 240))

    def test_thinning_keeps_the_whole_motion_not_just_the_start(self):
        """A long clip is sampled across its full length. Truncating instead
        would render the first second of a dance and call it finished."""
        thinned = video.read_frames(self.path, max_frames=8)
        self.assertEqual(len(thinned), 8)
        travel = dot_x(thinned[-1]) - dot_x(thinned[0])
        full = dot_x(video.read_frames(self.path)[-1])
        self.assertGreater(travel, 0.75 * full)  # still crosses the frame

    def test_every_nth_frame(self):
        self.assertEqual(len(video.read_frames(self.path, every=2)), 12)

    def test_a_thumbnail_is_not_the_first_frame(self):
        """Frame 0 of a real clip is very often black or a fade-in."""
        thumb = video.thumbnail(self.path)
        self.assertLessEqual(max(thumb.size), 512)
        self.assertGreater(dot_x(thumb), 0)

    def test_odd_sizes_are_made_encodable_without_rescaling_the_content(self):
        """H.264 needs even dimensions; ffmpeg would otherwise silently
        rescale, which shows up as a soft, slightly-wrong-size clip."""
        odd = video.write_video(moving_dot(size=(321, 241)),
                                Path(self.tmp.name) / "odd.mp4", fps=12)
        info = video.probe(odd)
        self.assertEqual((info.width, info.height), (320, 240))


class HonestFailureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_a_file_that_is_not_a_video_says_so(self):
        junk = Path(self.tmp.name) / "not-a-video.mp4"
        junk.write_bytes(b"this is not a video")
        with self.assertRaises(video.VideoError):
            video.probe(junk)

    def test_encoding_nothing_is_an_error_not_an_empty_file(self):
        with self.assertRaises(video.VideoError):
            video.write_video([], Path(self.tmp.name) / "empty.mp4")


class SampleIndexTests(unittest.TestCase):
    def test_spread_evenly_across_the_whole_clip(self):
        self.assertEqual(video.sample_indices(24, 8),
                         [0, 3, 6, 9, 12, 15, 18, 21])

    def test_asking_for_more_than_there_are_returns_all_of_them(self):
        self.assertEqual(video.sample_indices(5, 8), [0, 1, 2, 3, 4])

    def test_degenerate_inputs_are_empty_not_a_crash(self):
        self.assertEqual(video.sample_indices(0, 5), [])
        self.assertEqual(video.sample_indices(10, 0), [])

    def test_indices_never_repeat_and_stay_in_range(self):
        for total in (7, 30, 81, 240):
            for wanted in (1, 5, 16, 49):
                idx = video.sample_indices(total, wanted)
                self.assertEqual(len(idx), len(set(idx)), (total, wanted))
                self.assertTrue(all(0 <= i < total for i in idx))


class AnimationEncodingTests(unittest.TestCase):
    """Every frame that goes in has to come back out.

    A lossy animated WEBP drops frames it judges near-identical and raises
    nothing at all — measured on this machine, 25 frames in and 23 out. The
    clip is simply shorter than the render that made it, which is the worst
    kind of bug: no error, no log line, just a quietly wrong result."""

    @staticmethod
    def _near_identical(n=25):
        """Frames a lossy encoder is tempted to merge: a still scene with one
        pixel creeping across it."""
        out = []
        for i in range(n):
            im = Image.new("RGB", (64, 64), (40, 90, 160))
            im.putpixel((i % 64, i % 64), (41, 91, 161))
            out.append(im)
        return out

    def test_a_lossy_encoder_really_does_drop_them(self):
        """The premise, not an assumption: written the old way, the frames
        are lost. If PIL ever stops doing this the guard below is redundant
        and this test says so."""
        frames = self._near_identical()
        buf = io.BytesIO()
        frames[0].save(buf, format="WEBP", save_all=True,
                       append_images=frames[1:], duration=41, loop=0,
                       quality=90)
        self.assertLessEqual(video.count_animation_frames(buf.getvalue()),
                             len(frames))

    def test_the_shared_encoder_keeps_every_frame(self):
        frames = self._near_identical()
        data = video.encode_animation(frames, fps=24.0)
        self.assertEqual(video.count_animation_frames(data), len(frames))

    def test_a_lost_frame_is_an_error_not_a_silence(self):
        original = video.count_animation_frames
        video.count_animation_frames = lambda _d: 23
        try:
            with self.assertRaises(video.VideoError) as caught:
                video.encode_animation(self._near_identical())
            self.assertIn("lost frames", str(caught.exception))
        finally:
            video.count_animation_frames = original

    def test_nothing_to_encode_is_refused(self):
        with self.assertRaises(video.VideoError):
            video.encode_animation([])


if __name__ == "__main__":
    unittest.main()
