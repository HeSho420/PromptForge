"""Sizing and stitching a motion-transfer clip.

The user's requirement was explicit: split a long clip into pieces rather than
lose quality, and never crash the renderer. These tests pin both halves — the
memory ceiling that decides a window, and the join that must not be visible.
"""
import unittest

from PIL import Image

from app.adapters.comfyui import ALLOWED_TASKS
from app.core import motion


class LengthTests(unittest.TestCase):
    def test_lengths_are_always_valid_for_the_renderer(self):
        """WanVaceToVideo's latent maths is ((length-1)//4)+1 — a length that
        is not 4n+1 is rejected outright, so a clamp that produces 32 is a
        crash rather than a smaller render."""
        for n in range(5, 200):
            aligned = motion.align_length(n)
            self.assertEqual((aligned - 1) % motion.FRAME_STEP, 0, n)
            self.assertLessEqual(aligned, n)

    def test_it_never_returns_something_unrenderable(self):
        for n in (-100, 0, 1, 2, 3, 4, 5):
            self.assertGreaterEqual(motion.align_length(n), 5)
            self.assertEqual((motion.align_length(n) - 1) % 4, 0)


class BudgetTests(unittest.TestCase):
    """Guarding the PRODUCT, not each axis: the per-axis limits would permit
    512x512x33, which is 13% more pixel-frames than the run that already
    peaked at 7.99 of 8.00 GB VRAM."""

    def test_the_measured_safe_point_is_allowed_whole(self):
        self.assertEqual(motion.fit_window(480, 480, 25), 25)

    def test_the_measured_ceiling_is_allowed_but_not_exceeded(self):
        self.assertEqual(motion.fit_window(480, 832, 33), 33)
        self.assertLess(motion.fit_window(480, 832, 81), 81)

    def test_a_bigger_canvas_buys_fewer_frames(self):
        small = motion.fit_window(320, 320, 81)
        large = motion.fit_window(832, 832, 81)
        self.assertGreater(small, large)
        self.assertLessEqual(832 * 832 * large, motion.MAX_PIXEL_FRAMES)

    def test_every_result_is_renderable_and_within_budget(self):
        for w, h in ((320, 320), (480, 480), (480, 832), (832, 832), (1024, 1024)):
            for want in (17, 33, 81, 240):
                n = motion.fit_window(w, h, want)
                self.assertEqual((n - 1) % 4, 0, (w, h, want))
                self.assertGreaterEqual(n, 5)
                self.assertLessEqual(n, want)


class ChunkTests(unittest.TestCase):
    def test_a_short_clip_is_one_window(self):
        self.assertEqual(motion.plan_chunks(25, 25), [(0, 25)])
        self.assertEqual(motion.plan_chunks(17, 25), [(0, 17)])

    def test_a_long_clip_is_split_with_overlap(self):
        chunks = motion.plan_chunks(100, 25, overlap=8)
        self.assertGreater(len(chunks), 1)
        for (s1, e1), (s2, _e2) in zip(chunks, chunks[1:], strict=False):
            self.assertLess(s2, e1)          # they really do overlap
            self.assertGreaterEqual(e1 - s2, 2)
            self.assertLess(s1, s2)          # and always move forward

    def test_the_whole_clip_is_covered_start_to_finish(self):
        for total in (26, 60, 100, 241):
            chunks = motion.plan_chunks(total, 25, overlap=8)
            self.assertEqual(chunks[0][0], 0)
            self.assertEqual(chunks[-1][1], total)
            covered = set()
            for s, e in chunks:
                covered |= set(range(s, e))
            self.assertEqual(covered, set(range(total)), total)

    def test_no_window_is_a_useless_sliver(self):
        """A 3-frame tail renders worse than it costs: the last window is
        pulled back to full length and simply overlaps more."""
        for total in (27, 28, 51, 77):
            lens = [e - s for s, e in motion.plan_chunks(total, 25, overlap=8)]
            self.assertTrue(all(n >= 12 for n in lens), (total, lens))

    def test_it_terminates_on_hostile_input(self):
        for total in (1, 2, 5, 1000):
            for window in (5, 9, 25, 4000):
                for overlap in (0, 2, 8, 9999):
                    chunks = motion.plan_chunks(total, window, overlap)
                    self.assertTrue(chunks)
                    self.assertLessEqual(len(chunks), total + 2)


class StitchTests(unittest.TestCase):
    @staticmethod
    def _frames(n, value):
        return [Image.new("RGB", (16, 16), (value, value, value))
                for _ in range(n)]

    def test_a_join_is_a_fade_not_a_cut(self):
        joined = motion.crossfade(self._frames(10, 0), self._frames(10, 255), 4)
        self.assertEqual(len(joined), 16)          # 10 + 10 - 4
        band = [f.getpixel((8, 8))[0] for f in joined[6:10]]
        self.assertEqual(band, sorted(band))       # monotonic
        self.assertGreater(band[-1], band[0])      # and actually travels
        self.assertLess(band[0], 128)
        self.assertGreater(band[-1], 128)

    def test_no_overlap_is_a_plain_join(self):
        self.assertEqual(
            len(motion.crossfade(self._frames(5, 0), self._frames(5, 9), 0)), 10)

    def test_an_empty_window_does_not_destroy_the_clip(self):
        self.assertEqual(len(motion.crossfade(self._frames(5, 0), [], 4)), 5)
        self.assertEqual(len(motion.crossfade([], self._frames(5, 0), 4)), 5)

    def test_assembling_windows_reproduces_the_clip_length(self):
        chunks = motion.plan_chunks(60, 25, overlap=8)
        windows = [self._frames(e - s, 10 * i) for i, (s, e) in enumerate(chunks)]
        out = motion.assemble(windows, chunks)
        self.assertEqual(len(out), 60)

    def test_assembly_survives_a_window_that_failed_to_render(self):
        chunks = motion.plan_chunks(60, 25, overlap=8)
        windows = [self._frames(e - s, 10 * i) for i, (s, e) in enumerate(chunks)]
        windows[1] = []
        self.assertGreater(len(motion.assemble(windows, chunks)), 0)


class WiringTests(unittest.TestCase):
    def test_the_task_is_allowed_and_the_template_matches_the_engine(self):
        from pathlib import Path

        from app.adapters.comfyui import WorkflowLibrary
        self.assertIn("motion_transfer", ALLOWED_TASKS)
        t = WorkflowLibrary(
            Path(__file__).parent.parent / "app" / "workflows"
        ).load("motion_transfer")
        graph = t["graph"]
        vace = next(nid for nid, n in graph.items()
                    if n["class_type"] == "WanVaceToVideo")
        trim = next(n for n in graph.values()
                    if n["class_type"] == "TrimVideoLatent")
        ks = next(n for n in graph.values() if n["class_type"] == "KSampler")
        # The reference image is prepended as a latent frame and must be
        # trimmed back off, or it appears as a still frame at the start.
        self.assertEqual(trim["inputs"]["trim_amount"], [vace, 3])
        self.assertEqual(ks["inputs"]["latent_image"], [vace, 2])
        self.assertEqual(ks["inputs"]["positive"], [vace, 0])
        self.assertEqual(ks["inputs"]["negative"], [vace, 1])
        # Keeping the driving scene requires masks; without them VACE
        # regenerates the whole frame including the background.
        self.assertIn("control_masks", graph[vace]["inputs"])
        self.assertIn("reference_image", graph[vace]["inputs"])
        self.assertEqual((graph[vace]["inputs"]["length"] - 1) % 4, 0)

    def test_the_output_is_lossless(self):
        """Measured: a lossy animated WEBP silently drops near-identical
        frames — 25 in came back as 23 — which desynchronises the clip."""
        from pathlib import Path

        from app.adapters.comfyui import WorkflowLibrary
        t = WorkflowLibrary(
            Path(__file__).parent.parent / "app" / "workflows"
        ).load("motion_transfer")
        save = next(n for n in t["graph"].values()
                    if n["class_type"] == "SaveAnimatedWEBP")
        self.assertTrue(save["inputs"]["lossless"])


if __name__ == "__main__":
    unittest.main()
