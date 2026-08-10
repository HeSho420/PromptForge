"""MOCK ADAPTERS — not real AI models.

These exist so the entire pipeline (upload -> mask -> job -> inpaint ->
versioned result) can run and be tested without multi-GB downloads or a GPU.
Both classes set is_mock=True, stamp "mock" into result metadata, and the UI
labels their output accordingly. They must never be silently substituted for
a real backend.

MockSegmentationAdapter: keyword heuristics ("sky" -> upper band,
  "background" -> border region, "floor/ground" -> lower band, otherwise a
  centered ellipse). A real prompt-driven segmenter (e.g. SAM + grounding)
  replaces this behind the same interface.

MockInpaintingAdapter: fills the masked region with heavily blurred
  surrounding content (a crude content-aware fill) and, for a few prompt
  keywords, applies an obvious tint so results are visibly prompt-dependent.
"""
from __future__ import annotations

import re

from PIL import Image, ImageDraw, ImageFilter

from .base import EditResult, validate_mask


class OfflineComfyClient:
    """Mock mode's stand-in for the ComfyUI client: every probe answers
    OFFLINE, and anything that would actually talk to a server raises.

    Mock mode means this process makes NO connection to a ComfyUI — one
    answering on this box belongs to some other setup. Measured live before
    this existed: a mock avatar build passed the is-ComfyUI-up gate via the
    machine's resident real instance and rendered its mesh there, and
    cancelling that job posted /interrupt into the same instance — a mocked
    build interfering with renders it does not own. Probes are quiet no-ops
    so health checks stay honest; everything else fails loudly so a new
    render path cannot leak through unnoticed."""

    offline = True

    def is_up(self) -> bool:
        return False

    def health(self) -> tuple[bool, str]:
        return False, "mock mode is offline — renders never contact ComfyUI."

    def interrupt(self) -> bool:
        return False  # nothing of ours is running anywhere; cancel goes on

    def __getattr__(self, name: str):
        raise AttributeError(
            f"OfflineComfyClient has no '{name}': mock mode is offline, so "
            "no ComfyUI call is available — gate the calling path on mock "
            "mode instead.")


class MockSegmentationAdapter:
    name = "mock-segmentation"
    is_mock = True

    def propose_mask(self, image: Image.Image, prompt: str) -> Image.Image:
        w, h = image.size
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        p = (prompt or "").lower()

        if re.search(r"\bsky|clouds?|sunset|sunrise\b", p):
            draw.rectangle([0, 0, w, int(h * 0.40)], fill=255)
        elif re.search(r"\bbackground|backdrop|wall\b", p):
            draw.rectangle([0, 0, w, h], fill=255)
            draw.ellipse([w * 0.22, h * 0.12, w * 0.78, h * 0.92], fill=0)
        elif re.search(r"\bfloor|ground|grass|road\b", p):
            draw.rectangle([0, int(h * 0.65), w, h], fill=255)
        else:
            draw.ellipse([w * 0.30, h * 0.30, w * 0.70, h * 0.70], fill=255)

        return mask.filter(ImageFilter.GaussianBlur(max(2, min(w, h) // 200)))


class MockInpaintingAdapter:
    name = "mock-inpaint"
    is_mock = True

    _TINTS = {
        r"sunset|sunrise|warm|orange": (255, 140, 60),
        r"night|dark|midnight": (25, 30, 70),
        r"studio|white|clean": (240, 240, 238),
        r"forest|green|grass": (60, 130, 70),
        r"ocean|sea|blue|water": (50, 110, 190),
    }

    def inpaint(self, image: Image.Image, mask: Image.Image, prompt: str) -> EditResult:
        img = image.convert("RGB")
        mask = validate_mask(img, mask)

        # Crude content-aware fill: paste a heavily blurred copy through the mask.
        radius = max(8, min(img.size) // 24)
        filled = img.filter(ImageFilter.GaussianBlur(radius))

        tint_applied = None
        for pattern, rgb in self._TINTS.items():
            if re.search(pattern, (prompt or "").lower()):
                overlay = Image.new("RGB", img.size, rgb)
                filled = Image.blend(filled, overlay, 0.45)
                tint_applied = rgb
                break

        out = img.copy()
        out.paste(filled, (0, 0), mask)
        return EditResult(
            image=out,
            adapter=self.name,
            is_mock=True,
            meta={
                "note": "MOCK RESULT — blurred fill, not a generative model.",
                "blur_radius": radius,
                "tint": tint_applied,
            },
        )
