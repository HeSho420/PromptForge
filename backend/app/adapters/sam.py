"""Segment Anything adapter (real segmentation — roadmap Phase 1.2).

What works today, fully offline and unit-tested:
  * model-readiness checks against the ModelRegistry (`sam-vit-b`)
  * candidate selection ("grounding"): scoring SAM's mask proposals against
    the prompt with spatial/size priors — pure PIL, no torch needed
  * downscaling large images before segmentation and restoring mask size

What requires torch + segment-anything installed (and is exercised only then):
  * building SamAutomaticMaskGenerator from the downloaded checkpoint and
    running it. Missing packages or checkpoint raise ModelMissingError with
    an actionable message; nothing is silently mocked.

Honesty note: SAM proposes real object masks (is_mock=False), but SAM itself
is not text-conditioned. The prompt picks the best candidate via transparent
keyword/spatial priors (see select_candidate). A learned text-grounding stage
(e.g. CLIP ranking) can replace `select_candidate` behind the same signature;
until then the heuristic selection is documented here. The
user always reviews/corrects the mask in the editor before rendering.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from PIL import Image

from ..core.registry import ModelRegistry
from .base import BadMaskError, ModelMissingError

# Largest image side fed to SAM. Bigger inputs are downscaled first (SAM was
# trained at 1024px); the chosen mask is resized back to the original size.
DEFAULT_MAX_SIDE = 1024

INSTALL_HINT = ("The SAM backend needs extra packages: "
                "pip install -r requirements-sam.txt (torch, segment-anything).")


@dataclass
class MaskCandidate:
    """One proposed region: an L-mode mask plus the model's own confidence."""
    mask: Image.Image
    model_score: float = 1.0


class CandidateGenerator(Protocol):
    def generate(self, image: Image.Image) -> list[MaskCandidate]: ...


# -- prompt grounding (pure, offline-tested) -----------------------------------

def _on_frac(mask: Image.Image) -> float:
    """Fraction of pixels that are 'on' (>127) in an L-mode mask."""
    hist = mask.convert("L").histogram()
    total = sum(hist)
    return sum(hist[128:]) / total if total else 0.0


def _region_frac(mask: Image.Image, box: tuple[int, int, int, int]) -> float:
    """Share of the mask's on-pixels that fall inside `box` (0..1)."""
    w, h = mask.size
    whole = _on_frac(mask) * w * h
    if whole <= 0:
        return 0.0
    bw, bh = box[2] - box[0], box[3] - box[1]
    inside = _on_frac(mask.crop(box)) * bw * bh
    return inside / whole


def _border_frac(mask: Image.Image, margin_frac: float = 0.04) -> float:
    """Share of the mask's on-pixels inside a thin frame along the borders."""
    w, h = mask.size
    m = max(1, int(min(w, h) * margin_frac))
    whole = _on_frac(mask) * w * h
    if whole <= 0:
        return 0.0
    inner = _on_frac(mask.crop((m, m, w - m, h - m))) * (w - 2 * m) * (h - 2 * m)
    return max(0.0, whole - inner) / whole


def _size_preference(area_frac: float, lo: float = 0.02, hi: float = 0.50) -> float:
    """1.0 for mid-sized regions, tapering to 0 for specks and full-frame masks."""
    if area_frac <= 0.0:
        return 0.0
    if area_frac < lo:
        return area_frac / lo
    if area_frac > hi:
        return max(0.0, (1.0 - area_frac) / (1.0 - hi))
    return 1.0


def _spatial_prior(mask: Image.Image, prompt: str) -> float:
    """How well a candidate matches the prompt's spatial intent (0..1).

    Keyword vocabulary intentionally mirrors MockSegmentationAdapter so the
    two backends interpret prompts consistently.
    """
    w, h = mask.size
    p = (prompt or "").lower()
    area = _on_frac(mask)

    if re.search(r"\bsky|clouds?|sunset|sunrise\b", p):
        # Mostly in the top band, and not a sliver.
        return _region_frac(mask, (0, 0, w, int(h * 0.40))) * min(1.0, area / 0.10)
    if re.search(r"\bbackground|backdrop|wall\b", p):
        # Large and hugging the borders.
        return _border_frac(mask) * min(1.0, area / 0.30)
    if re.search(r"\bfloor|ground|grass|road\b", p):
        return _region_frac(mask, (0, int(h * 0.65), w, h)) * min(1.0, area / 0.10)

    # Default: a salient object — mostly inside the center box, mid-sized.
    center = (int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75))
    return _region_frac(mask, center) * _size_preference(area)


def select_candidate(candidates: list[MaskCandidate], prompt: str) -> MaskCandidate:
    """Pick the candidate whose region best matches the prompt.

    Score = model confidence × spatial prior. Raises BadMaskError when SAM
    proposed nothing usable — the user can still paint a mask manually.
    """
    scored = [(c.model_score * _spatial_prior(c.mask, prompt), i, c)
              for i, c in enumerate(candidates)]
    scored = [(s, i, c) for s, i, c in scored if s > 0.0]
    if not scored:
        raise BadMaskError(
            "SAM found no region matching the prompt — paint the mask manually.")
    return max(scored, key=lambda t: (t[0], -t[1]))[2]


# -- real backend (imports torch lazily) ----------------------------------------

class _SamGenerator:
    """Wraps SamAutomaticMaskGenerator; built only when the backend is used."""

    def __init__(self, checkpoint: str, model_type: str, points_per_side: int):
        try:
            import torch
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        except ImportError as exc:
            raise ModelMissingError(INSTALL_HINT) from exc
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device)
        self._amg = SamAutomaticMaskGenerator(sam, points_per_side=points_per_side)

    def generate(self, image: Image.Image) -> list[MaskCandidate]:
        import numpy as np
        arr = np.asarray(image.convert("RGB"))
        out: list[MaskCandidate] = []
        for raw in self._amg.generate(arr):
            mask = Image.fromarray(raw["segmentation"].astype("uint8") * 255, mode="L")
            score = (float(raw.get("predicted_iou", 0.5))
                     * float(raw.get("stability_score", 1.0)))
            out.append(MaskCandidate(mask=mask, model_score=score))
        return out


class _SamPointer:
    """Wraps SamPredictor for point-click segmentation with an embedding
    cache: clicking around the same image reuses the (expensive) image
    encoding, so follow-up clicks answer in well under a second on GPU."""

    def __init__(self, checkpoint: str, model_type: str):
        try:
            import torch
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise ModelMissingError(INSTALL_HINT) from exc
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device)
        self._predictor = SamPredictor(sam)
        self._cached_key: bytes | None = None

    def point_mask(self, image: Image.Image, x: int, y: int) -> Image.Image:
        import numpy as np
        arr = np.asarray(image.convert("RGB"))
        key = arr[:: max(1, arr.shape[0] // 16), :: max(1, arr.shape[1] // 16)].tobytes()
        if key != self._cached_key:
            self._predictor.set_image(arr)
            self._cached_key = key
        masks, scores, _ = self._predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=True)
        best = masks[int(scores.argmax())]
        return Image.fromarray(best.astype("uint8") * 255, mode="L")


class SamSegmentationAdapter:
    """Prompt-guided mask proposals from Segment Anything.

    Two modes:
      propose_mask(image, prompt)  — automatic candidates ranked against the
                                     prompt (used by the edit pipeline);
      point_mask(image, x, y)      — "click anything": exact segmentation of
                                     whatever is under the cursor, the way SAM
                                     is designed to be used. This is the
                                     reliable path for arbitrary objects.
    """

    is_mock = False

    def __init__(self, registry: ModelRegistry, model_name: str = "sam-vit-b",
                 generator_factory: Callable[[str], CandidateGenerator] | None = None,
                 max_side: int = DEFAULT_MAX_SIDE, points_per_side: int = 32,
                 pointer_factory: Callable[[str], Any] | None = None):
        self.name = model_name
        self.registry = registry
        self.model_name = model_name
        self.max_side = max_side
        self.points_per_side = points_per_side
        self._factory = generator_factory or self._default_factory
        self._generator: CandidateGenerator | None = None
        self._pointer_factory = pointer_factory or self._default_pointer_factory
        self._pointer: Any = None

    def _default_factory(self, checkpoint: str) -> CandidateGenerator:
        # "sam-vit-b" -> registry key "vit_b" in segment_anything.
        model_type = self.model_name.removeprefix("sam-").replace("-", "_")
        return _SamGenerator(checkpoint, model_type, self.points_per_side)

    def _default_pointer_factory(self, checkpoint: str) -> _SamPointer:
        model_type = self.model_name.removeprefix("sam-").replace("-", "_")
        return _SamPointer(checkpoint, model_type)

    def _checkpoint_path(self) -> str:
        if not self.registry.is_ready(self.model_name):
            raise ModelMissingError(
                f"Model '{self.model_name}' is not downloaded. "
                "Download it from the Models page first.")
        model = self.registry.get(self.model_name)
        assert model is not None and model.path
        return model.path

    @property
    def is_loaded(self) -> bool:
        """Whether a SAM model is resident — the UI uses this to show
        'Loading SAM model…' only on genuinely cold requests."""
        return self._pointer is not None or self._generator is not None

    def release(self) -> None:
        """Drop the loaded SAM weights to free RAM/VRAM before memory-heavy
        renders (video). SAM lazily reloads on the next mask request."""
        self._pointer = None
        self._generator = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — cache clearing is best-effort
            pass

    def point_mask(self, image: Image.Image, x: int, y: int) -> Image.Image:
        """Segment exactly what is at (x, y) in original-image coordinates."""
        if self._pointer is None:
            self._pointer = self._pointer_factory(self._checkpoint_path())
        work = image.convert("RGB")
        scale = 1.0
        if max(work.size) > self.max_side:
            scale = self.max_side / max(work.size)
            work = work.resize((max(1, round(work.width * scale)),
                                max(1, round(work.height * scale))),
                               Image.Resampling.LANCZOS)
        mask = self._pointer.point_mask(work, int(x * scale), int(y * scale))
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        return mask

    def _get_generator(self) -> CandidateGenerator:
        if self._generator is None:
            self._generator = self._factory(self._checkpoint_path())
        return self._generator

    def propose_mask(self, image: Image.Image, prompt: str) -> Image.Image:
        generator = self._get_generator()

        work = image.convert("RGB")
        if max(work.size) > self.max_side:
            scale = self.max_side / max(work.size)
            work = work.resize((max(1, round(work.width * scale)),
                                max(1, round(work.height * scale))),
                               Image.Resampling.LANCZOS)

        candidates = generator.generate(work)
        best = select_candidate(candidates, prompt)

        mask = best.mask.convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        return mask
