"""SAM adapter tests — all offline: candidates come from a fake generator, so
the grounding/selection logic and the registry gate are exercised without
torch or a downloaded checkpoint."""
import unittest

from PIL import Image, ImageDraw

from app.adapters.base import BadMaskError, ModelMissingError
from app.adapters.sam import MaskCandidate, SamSegmentationAdapter, select_candidate


def _mask(size, box=None, ellipse=None, invert_box=None):
    """Build an L-mode mask: filled box/ellipse, or everything EXCEPT a box."""
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    if box:
        d.rectangle(box, fill=255)
    if ellipse:
        d.ellipse(ellipse, fill=255)
    if invert_box:
        d.rectangle([0, 0, size[0], size[1]], fill=255)
        d.rectangle(invert_box, fill=0)
    return m


class FakeRegistry:
    def __init__(self, ready=True, path="C:/fake/sam.pth"):
        self._ready, self._path = ready, path

    def is_ready(self, name):
        return self._ready

    def get(self, name):
        class M:
            path = self._path
        return M()


class FakeGenerator:
    def __init__(self, candidates):
        self.candidates = candidates
        self.seen_sizes = []

    def generate(self, image):
        self.seen_sizes.append(image.size)
        return self.candidates


def _adapter(candidates, registry=None, **kw):
    gen = FakeGenerator(candidates)
    ad = SamSegmentationAdapter(registry or FakeRegistry(),
                                generator_factory=lambda path: gen, **kw)
    return ad, gen


class SelectionTests(unittest.TestCase):
    SIZE = (100, 100)

    def test_sky_prompt_prefers_top_band(self):
        sky = MaskCandidate(_mask(self.SIZE, box=[0, 0, 100, 35]))
        floor = MaskCandidate(_mask(self.SIZE, box=[0, 70, 100, 100]))
        self.assertIs(select_candidate([floor, sky], "change the sky to sunset"), sky)

    def test_floor_prompt_prefers_bottom_band(self):
        sky = MaskCandidate(_mask(self.SIZE, box=[0, 0, 100, 35]))
        floor = MaskCandidate(_mask(self.SIZE, box=[0, 70, 100, 100]))
        self.assertIs(select_candidate([sky, floor], "replace the floor with grass"), floor)

    def test_background_prompt_prefers_border_region(self):
        subject = MaskCandidate(_mask(self.SIZE, ellipse=[30, 20, 70, 90]))
        backdrop = MaskCandidate(_mask(self.SIZE, invert_box=[25, 15, 75, 95]))
        self.assertIs(select_candidate([subject, backdrop], "blur the background"), backdrop)

    def test_object_prompt_prefers_centered_midsize_region(self):
        speck = MaskCandidate(_mask(self.SIZE, box=[48, 48, 51, 51]))
        full = MaskCandidate(_mask(self.SIZE, box=[0, 0, 100, 100]))
        chair = MaskCandidate(_mask(self.SIZE, ellipse=[35, 40, 65, 80]))
        self.assertIs(select_candidate([speck, full, chair], "remove the chair"), chair)

    def test_model_confidence_breaks_ties(self):
        weak = MaskCandidate(_mask(self.SIZE, ellipse=[35, 40, 65, 80]), model_score=0.4)
        strong = MaskCandidate(_mask(self.SIZE, ellipse=[35, 40, 65, 80]), model_score=0.9)
        self.assertIs(select_candidate([weak, strong], "remove the chair"), strong)

    def test_no_usable_candidate_raises(self):
        with self.assertRaises(BadMaskError):
            select_candidate([], "remove the chair")
        empty = MaskCandidate(Image.new("L", self.SIZE, 0))
        with self.assertRaises(BadMaskError):
            select_candidate([empty], "remove the chair")


class AdapterTests(unittest.TestCase):
    def test_is_labeled_real(self):
        ad, _ = _adapter([])
        self.assertFalse(ad.is_mock)
        self.assertEqual(ad.name, "sam-vit-b")

    def test_missing_model_raises_before_generator_is_built(self):
        built = []
        ad = SamSegmentationAdapter(
            FakeRegistry(ready=False),
            generator_factory=lambda path: built.append(path))
        with self.assertRaises(ModelMissingError):
            ad.propose_mask(Image.new("RGB", (64, 64)), "remove the chair")
        self.assertEqual(built, [])

    def test_propose_mask_returns_image_sized_l_mask(self):
        img = Image.new("RGB", (64, 48))
        ad, _ = _adapter([MaskCandidate(_mask((64, 48), ellipse=[20, 14, 44, 36]))])
        mask = ad.propose_mask(img, "remove the chair")
        self.assertEqual(mask.size, img.size)
        self.assertEqual(mask.mode, "L")
        self.assertNotEqual(mask.getextrema(), (0, 0))

    def test_large_images_are_downscaled_and_mask_restored(self):
        img = Image.new("RGB", (128, 96))
        ad, gen = _adapter([MaskCandidate(_mask((64, 48), ellipse=[20, 14, 44, 36]))],
                           max_side=64)
        mask = ad.propose_mask(img, "remove the chair")
        self.assertEqual(gen.seen_sizes, [(64, 48)])   # SAM saw the downscaled image
        self.assertEqual(mask.size, (128, 96))          # caller gets full size back

    def test_generator_is_cached_across_calls(self):
        factory_calls = []
        gen = FakeGenerator([MaskCandidate(_mask((64, 64), ellipse=[20, 20, 44, 44]))])

        def factory(path):
            factory_calls.append(path)
            return gen

        ad = SamSegmentationAdapter(FakeRegistry(), generator_factory=factory)
        img = Image.new("RGB", (64, 64))
        ad.propose_mask(img, "remove the chair")
        ad.propose_mask(img, "remove the chair")
        self.assertEqual(len(factory_calls), 1)


if __name__ == "__main__":
    unittest.main()
