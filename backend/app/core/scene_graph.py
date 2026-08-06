"""Image Understanding Engine — a persistent Scene Graph.

One rich analysis is built per image and REUSED by every editing step:
planning reads it to pick operations, masking reads object locations to
segment the right thing, placement reads perspective + lighting to put new
content where it belongs, and every render prompt carries a compact summary
so the diffusion model never invents content blind.

Fully fail-safe: an unavailable vision model degrades to a minimal graph
(deterministic colour palette only) — the pipeline can lose richness, never
a render.
"""
from __future__ import annotations

import re
from typing import Any

from PIL import Image

from .quality import _parse_json

# 3x3 grid names so the graph, placement masks and targeted masks all speak
# the same spatial language.
_CELL_NAME = {
    1: "top-left", 2: "top-center", 3: "top-right",
    4: "center-left", 5: "center", 6: "center-right",
    7: "bottom-left", 8: "bottom-center", 9: "bottom-right",
}
_NAME_CELL = {v: k for k, v in _CELL_NAME.items()}

_SCENE_SYSTEM = """You are an image analyst for a photo editor. Study the
image and reply with ONLY JSON:
{"scene": "<one short sentence: subject + setting>",
 "setting": "<indoor|outdoor|studio|street|nature|other>",
 "lighting": "<direction and quality, e.g. 'soft daylight from upper left'>",
 "perspective": "<eye-level|low-angle|high-angle|close-up|wide>",
 "has_person": <true|false>,
 "objects": [{"name": "<noun>", "location":
   "<top-left|top-center|top-right|center-left|center|center-right|
     bottom-left|bottom-center|bottom-right>",
   "size": "<small|medium|large>"}]}
List the 1-6 most prominent, editable objects. Be concrete and factual —
no opinions, no marketing. Ground everything in what is actually visible."""


def _palette(image: Image.Image, k: int = 5) -> list[list[int]]:
    """Dominant colours (deterministic) — always available, even offline."""
    small = image.convert("RGB").resize((64, 64))
    quant = small.quantize(colors=k, method=Image.Quantize.FASTOCTREE)
    pal = quant.getpalette() or []
    counts = sorted(quant.getcolors() or [], reverse=True)
    out: list[list[int]] = []
    for _count, idx in counts[:k]:
        out.append([pal[idx * 3], pal[idx * 3 + 1], pal[idx * 3 + 2]])
    return out


def _clean(text: Any, limit: int = 140) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().strip(".")[:limit]


def _minimal(image: Image.Image) -> dict[str, Any]:
    return {
        "scene": "", "setting": "", "lighting": "", "perspective": "",
        "palette": _palette(image), "has_person": False, "objects": [],
    }


def build(image: Image.Image, critic: Any,
          segmentation: Any = None) -> dict[str, Any]:
    """Build the scene graph. `critic` is the vision model (llava); it may be
    None. `segmentation` is unused here but kept for signature stability —
    object masks are cut on demand at mask time, not eagerly."""
    graph = _minimal(image)
    if critic is None or not hasattr(critic, "ask"):
        return graph
    try:
        data = _parse_json(critic.ask(image, _SCENE_SYSTEM))
    except Exception:  # noqa: BLE001 — analysis is a bonus, never a blocker
        return graph
    if not isinstance(data, dict):
        return graph
    graph["scene"] = _clean(data.get("scene"))
    graph["setting"] = _clean(data.get("setting"), 32)
    graph["lighting"] = _clean(data.get("lighting"), 80)
    graph["perspective"] = _clean(data.get("perspective"), 32)
    graph["has_person"] = bool(data.get("has_person"))
    objects: list[dict[str, Any]] = []
    for obj in data.get("objects") or []:
        if not isinstance(obj, dict) or not str(obj.get("name", "")).strip():
            continue
        loc = str(obj.get("location", "center")).strip().lower()
        objects.append({
            "name": _clean(obj.get("name"), 40),
            "location": loc if loc in _NAME_CELL else "center",
            "size": (str(obj.get("size", "medium")).lower()
                     if str(obj.get("size", "")).lower()
                     in ("small", "medium", "large") else "medium"),
            "cell": _NAME_CELL.get(loc, 5),
        })
    graph["objects"] = objects[:6]
    return graph


def summary(graph: dict[str, Any], max_chars: int = 320) -> str | None:
    """Compact scene context appended to render prompts. None when empty."""
    if not graph or not graph.get("scene"):
        return None
    parts = [graph["scene"]]
    if graph.get("lighting"):
        parts.append(f"lighting: {graph['lighting']}")
    if graph.get("perspective"):
        parts.append(f"perspective: {graph['perspective']}")
    return "; ".join(parts)[:max_chars]


def find_object(graph: dict[str, Any], target: str) -> dict[str, Any] | None:
    """The scene-graph object best matching `target` (the noun an operation
    acts on) — used to tell segmentation WHERE the thing is. None if the
    graph doesn't know it."""
    if not graph or not target:
        return None
    t = target.lower().strip()
    t_tokens = set(re.findall(r"[a-z]{3,}", t))
    best, best_score = None, 0
    for obj in graph.get("objects", []):
        name = obj["name"].lower()
        if name in t or t in name:
            return obj
        score = len(t_tokens & set(re.findall(r"[a-z]{3,}", name)))
        if score > best_score:
            best, best_score = obj, score
    return best if best_score else None


def placement_context(graph: dict[str, Any]) -> str:
    """Scene facts that help place NEW content believably (perspective,
    ground plane, lighting) — appended to the placement question."""
    bits = []
    if graph.get("perspective"):
        bits.append(f"camera: {graph['perspective']}")
    if graph.get("lighting"):
        bits.append(f"lighting: {graph['lighting']}")
    if graph.get("setting"):
        bits.append(f"setting: {graph['setting']}")
    return ("Scene: " + "; ".join(bits) + ". ") if bits else ""
