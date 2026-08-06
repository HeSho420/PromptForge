"""Segment what the words actually name, rather than what sits in the middle.

Run with ComfyUI's interpreter, not PromptForge's: transformers and torch
already live there, and transformers 5.13 supports CLIPSeg natively.

    python text_mask.py <image.png> <out.png> --phrase "necklace" [--phrase ...]
                        [--threshold 0.40] [--json]

Why this exists. The general segmenter here is SAM, which proposes every
region it can find and then picks one with a purely GEOMETRIC prior — a
handful of hardcoded branches for sky, backdrop and floor, and otherwise
"centred and mid-sized". The request is never read. Measured on two real
photographs, "remove the necklace" and "change her shoes" produced the
IDENTICAL mask, twice: 9.9% on one photo, 1.7% on the other.

The two obvious fixes are both unavailable on this machine, which is why this
file is small and local rather than a dependency:

  GroundingDINO  installed, but its BERT wrapper calls get_head_mask, removed
                 in transformers 5.13. Fixing it means downgrading the library
                 the working image pipeline runs on.
  CLIPSeg node   the Impact pack's provider is a stub that requires a separate
                 extension which is not installed.

CLIPSeg itself is a first-class transformers model, so it needs no node pack
and no patched library — only its ~150 MB weights, fetched once.

The output is deliberately COARSE and says so: CLIPSeg reasons at 352x352, so
this locates the right thing rather than tracing its outline. The caller
refines and validates it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_ID = "CIDAS/clipseg-rd64-refined"
# CLIPSeg's own working resolution. Anything finer is interpolation, not
# detail, and pretending otherwise would be the sort of claim this codebase
# spends its time removing.
NATIVE = 352


def load_model():
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    processor = CLIPSegProcessor.from_pretrained(MODEL_ID)
    model = CLIPSegForImageSegmentation.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model


def heatmaps(image: Image.Image, phrases: list[str]) -> np.ndarray:
    """One 0..1 map per phrase, at CLIPSeg's own resolution."""
    import torch

    processor, model = load_model()
    rgb = image.convert("RGB")
    inputs = processor(text=phrases, images=[rgb] * len(phrases),
                       padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    if logits.ndim == 2:                      # a single phrase loses the batch
        logits = logits.unsqueeze(0)
    return torch.sigmoid(logits).cpu().numpy()


def to_mask(maps: np.ndarray, size: tuple[int, int],
            threshold: float) -> tuple[Image.Image, dict]:
    """Union the phrase maps and threshold them into one mask."""
    union = maps.max(axis=0)
    peak = float(union.max())
    # A relative floor as well as an absolute one: CLIPSeg's confidence
    # depends on the phrase, so a fixed cut alone either loses faint-but-real
    # detections or accepts noise on an image where nothing matches.
    cut = max(threshold, peak * 0.5)
    binary = (union >= cut).astype(np.uint8) * 255
    mask = Image.fromarray(binary, mode="L").resize(size, Image.BILINEAR)
    mask = mask.point(lambda v: 255 if v >= 128 else 0)
    return mask, {"peak": round(peak, 3), "cut": round(cut, 3)}


def main() -> int:
    argv = sys.argv[1:]
    positional, phrases, controls, threshold = [], [], [], 0.40
    i = 0
    while i < len(argv):
        if argv[i] == "--phrase":
            phrases.append(argv[i + 1])
            i += 2
        elif argv[i] == "--control":
            controls.append(argv[i + 1])
            i += 2
        elif argv[i] == "--threshold":
            threshold = float(argv[i + 1])
            i += 2
        elif argv[i].startswith("--"):
            i += 1
        else:
            positional.append(argv[i])
            i += 1
    if len(positional) < 2 or not phrases:
        print(__doc__)
        return 2

    image = Image.open(positional[0])
    # Controls ride along in the SAME batch: phrases naming things that are
    # certainly not in the photo. Their peaks measure what confidence "not
    # there" attracts on THIS image — a single global floor cannot, because
    # an absent object scored 0.61-0.65 on one photo and below the floor on
    # another (measured live). The caller compares the request's peak
    # against the controls' and requires a margin.
    maps = heatmaps(image, phrases + controls)
    request_maps = maps[:len(phrases)]
    control_maps = maps[len(phrases):]
    mask, detail = to_mask(request_maps, image.size, threshold)
    out = Path(positional[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    mask.save(out, format="PNG")

    covered = float(np.asarray(mask).mean() / 255.0)
    report = {
        "phrases": phrases,
        "coverage": round(covered, 4),
        "per_phrase_peak": [round(float(m.max()), 3) for m in request_maps],
        "native_px": NATIVE,
        **detail,
    }
    if controls:
        report["controls"] = controls
        report["control_peak"] = round(
            max(float(m.max()) for m in control_maps), 3)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
