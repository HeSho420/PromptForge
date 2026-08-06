"""AI backend adapter contracts.

The app never talks to a model directly — it talks to adapters. This is the
seam where ComfyUI, diffusers, photogrammetry or Gaussian-splatting backends
plug in later. Two adapter families exist in the MVP:

  SegmentationAdapter: prompt + image -> proposed edit mask (L-mode PIL image,
                       255 = edit here, 0 = keep)
  InpaintingAdapter:   image + mask + prompt -> edited image

Every adapter declares `name` and `is_mock`. Mock adapters MUST set
is_mock=True; the flag is surfaced through the API and UI so mock output can
never be mistaken for a real model result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from PIL import Image


class AdapterError(RuntimeError):
    """Base class for adapter failures."""


class BackendUnavailableError(AdapterError):
    """The backend process (e.g. ComfyUI) is not reachable — transient."""


class ModelMissingError(AdapterError):
    """A required model is not downloaded/ready — permanent until resolved."""


class BadMaskError(AdapterError):
    """Mask is empty, the wrong size, or unusable — permanent."""


@dataclass
class EditResult:
    image: Image.Image
    adapter: str
    is_mock: bool
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SegmentationAdapter(Protocol):
    name: str
    is_mock: bool

    def propose_mask(self, image: Image.Image, prompt: str) -> Image.Image:
        """Return an L-mode mask the size of `image` (255 = edit region)."""
        ...


@runtime_checkable
class InpaintingAdapter(Protocol):
    name: str
    is_mock: bool

    def inpaint(self, image: Image.Image, mask: Image.Image, prompt: str) -> EditResult:
        ...


def validate_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Normalize and sanity-check a mask against its image. Raises BadMaskError."""
    if mask.size != image.size:
        raise BadMaskError(
            f"Mask size {mask.size} does not match image size {image.size}.")
    mask = mask.convert("L")
    extrema = mask.getextrema()
    if extrema == (0, 0):
        raise BadMaskError("Mask is empty — paint the region you want to edit.")
    return mask
