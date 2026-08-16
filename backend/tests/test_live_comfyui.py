"""End-to-end test against a *live* ComfyUI server.

Skipped by default so the suite stays offline-safe. To run it:

    1. Start ComfyUI (default http://127.0.0.1:8188) with the
       sd-v1-5-inpainting.safetensors checkpoint visible to it
       (the launcher starts and verifies it).
    2. PROMPTFORGE_LIVE_COMFYUI=1 python3 -m unittest tests.test_live_comfyui -v

This is deliberately a real render — nothing is mocked. It verifies the two
integration details that unit tests cannot: that the LoadImageMask red-channel
convention selects the painted region, and that only masked pixels change.
"""
import os
import unittest

from PIL import Image

LIVE = os.environ.get("PROMPTFORGE_LIVE_COMFYUI") == "1"


@unittest.skipUnless(LIVE, "set PROMPTFORGE_LIVE_COMFYUI=1 with a running ComfyUI to enable")
class LiveComfyUITests(unittest.TestCase):
    def setUp(self):
        import tempfile

        from app.config import Settings
        from app.core.services import Services

        self.tmp = tempfile.TemporaryDirectory()
        settings = Settings()
        settings.data_dir = self.tmp.name  # type: ignore[assignment]
        settings.inpaint_backend = "comfyui"
        settings.ensure_dirs()
        self.services = Services(settings)
        # the checkpoint must already be marked ready (downloaded + visible)
        model = self.services.registry.get("sd15-inpaint")
        assert model is not None
        if model.status != "ready":
            self.skipTest("sd15-inpaint not downloaded — run the download first")

    def tearDown(self):
        self.services.shutdown()
        self.tmp.cleanup()

    def test_inpaint_changes_only_masked_region(self):
        image = Image.new("RGB", (512, 512), (40, 90, 160))
        mask = Image.new("L", (512, 512), 0)
        # mask the right half
        for x in range(256, 512):
            for y in range(512):
                mask.putpixel((x, y), 255)

        result = self.services.inpainting.inpaint(
            image=image, mask=mask, prompt="a field of red poppies")
        out = result.image
        self.assertEqual(out.size, image.size)
        self.assertFalse(result.is_mock)

        # unmasked left edge should be (near-)unchanged; masked right half
        # should differ substantially from the flat blue input
        left_orig = image.crop((0, 0, 64, 512))
        left_new = out.crop((0, 0, 64, 512))
        diff_left = _mean_abs_diff(left_orig, left_new)
        right_orig = image.crop((300, 0, 512, 512))
        right_new = out.crop((300, 0, 512, 512))
        diff_right = _mean_abs_diff(right_orig, right_new)
        self.assertLess(diff_left, 12.0, "unmasked pixels changed too much")
        self.assertGreater(diff_right, 20.0, "masked region was not edited")


def _mean_abs_diff(a: Image.Image, b: Image.Image) -> float:
    pa, pb = a.tobytes(), b.tobytes()
    return sum(abs(x - y) for x, y in zip(pa, pb, strict=False)) / len(pa)


if __name__ == "__main__":
    unittest.main()
