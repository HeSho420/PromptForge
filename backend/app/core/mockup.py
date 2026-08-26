"""The layout engine: interface mock-ups drawn, not diffused.

A structured menu/UI mock-up is text-dense by construction — every tab,
item and price is a label — and a diffusion model letters gibberish
(measured live: CDOSED). This engine splits the request the way it
actually decomposes: the VISION model reads the reference image's style
(colours, corner vibe), the local TEXT model plans the menu as data
(tabs, items, levels, costs), and Pillow draws it deterministically —
every glyph perfect, paging real, nothing hallucinated at render time.

Pure functions throughout; services orchestrates.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .critic import ask_with_schema
from .llm import LLMClient, complete_with_schema

# -- the menu as DATA ---------------------------------------------------------

SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "currency": {"type": "string"},
        "max_level": {"type": "integer"},
        "active_tab": {"type": "string"},
        "tabs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "level": {"type": "integer"},
                                "cost": {"type": "integer"},
                            },
                            "required": ["name", "level", "cost"],
                        },
                    },
                },
                "required": ["name", "items"],
            },
        },
    },
    "required": ["title", "currency", "tabs"],
}

_SPEC_SYSTEM = (
    "You design game and app menu mock-ups. Reply with ONLY JSON matching "
    "the schema. Use EXACTLY the tabs the request names (same names, same "
    "order). Invent 8-12 fitting items per tab with game-appropriate "
    "names; 'level' is each item's current upgrade level (spread between "
    "0 and max_level so the mock-up shows progression), 'cost' is a "
    "plausible integer price for the NEXT level in the named currency. "
    "Keep names under 24 characters.")


def mockup_spec(llm: LLMClient | None, prompt: str) -> dict[str, Any] | None:
    """The request as a structured menu plan, or None without a planner."""
    if llm is None:
        return None
    try:
        reply = complete_with_schema(
            llm, _SPEC_SYSTEM, f"Request: {prompt}",
            max_tokens=1600, schema=SPEC_SCHEMA)
        data = json.loads(re.sub(r"^```(?:json)?|```$", "",
                                 reply.text.strip(), flags=re.M).strip())
    except Exception:  # noqa: BLE001 — the caller falls back honestly
        return None
    if not isinstance(data, dict) or not data.get("tabs"):
        return None
    data.setdefault("currency", "Coins")
    data.setdefault("title", "Upgrades")
    data["max_level"] = int(data.get("max_level") or 3)
    tabs = []
    for tab in data["tabs"][:8]:
        items = [{"name": str(i.get("name", "?"))[:24],
                  "level": max(0, min(int(i.get("level", 0)),
                                      data["max_level"])),
                  "cost": max(0, int(i.get("cost", 0)))}
                 for i in (tab.get("items") or [])[:24]
                 if isinstance(i, dict)]
        if items:
            tabs.append({"name": str(tab.get("name", "?"))[:20],
                         "items": items})
    if not tabs:
        return None
    data["tabs"] = tabs
    if not data.get("active_tab") or not any(
            t["name"] == data["active_tab"] for t in tabs):
        data["active_tab"] = tabs[0]["name"]
    return data


# "tabs for killstreaks, perks, hunter shop, zombie shop and rank shop" —
# the requested tab names, so the drawn menu can be VERIFIED against the
# request deterministically.
_TABS_CLAUSE = re.compile(
    r"\btabs?\s+(?:for|of|named|called)\s+([^.;]+)", re.IGNORECASE)


def requested_tabs(prompt: str) -> list[str]:
    m = _TABS_CLAUSE.search(prompt or "")
    if not m:
        return []
    clause = re.split(r"\b(?:each|with|that|which|where)\b", m.group(1))[0]
    parts = re.split(r",|\band\b|&", clause)
    return [p.strip(" .'\"") for p in parts if p.strip(" .'\"")]


def missing_tabs(prompt: str, spec: dict[str, Any]) -> list[str]:
    """Requested tab names the spec failed to include (case-insensitive)."""
    have = [t["name"].lower() for t in spec.get("tabs", [])]
    return [want for want in requested_tabs(prompt)
            if not any(want.lower() in h or h in want.lower()
                       for h in have)]


# -- the STYLE, read from the reference --------------------------------------

DEFAULT_STYLE: dict[str, str] = {
    "background": "#1d2129", "panel": "#2b313d", "border": "#4a5468",
    "accent": "#e8a33d", "text": "#e8e6df", "vibe": "rounded",
}

_STYLE_SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "string"} for k in DEFAULT_STYLE},
    "required": list(DEFAULT_STYLE),
}

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def style_from_reference(critic: Any, image: Any) -> dict[str, str]:
    """The reference UI's palette and corner vibe, defaults where the
    reading fails — a mock-up always renders."""
    style = dict(DEFAULT_STYLE)
    if critic is None or image is None:
        return style
    try:
        data = json.loads(ask_with_schema(
            critic, image,
            "This is a game or app interface. Reply ONLY JSON: the "
            "interface's dominant background colour, panel colour, "
            "border colour, one accent/highlight colour and main text "
            'colour as hex strings, plus "vibe": "pixel" for blocky '
            'square-cornered pixel-art interfaces or "rounded" '
            "otherwise.", _STYLE_SCHEMA))
    except Exception:  # noqa: BLE001 — defaults are honest
        return style
    if not isinstance(data, dict):
        return style
    for key in ("background", "panel", "border", "accent", "text"):
        value = str(data.get(key, "")).strip()
        if _HEX.match(value):
            style[key] = value
    if str(data.get("vibe", "")).lower() == "pixel":
        style["vibe"] = "pixel"
    return style


# -- the RENDERER: deterministic Pillow drawing ------------------------------

_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\seguisb.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\arial.ttf",
)


def _font(size: int) -> Any:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001 — try the next candidate
            continue
    return ImageFont.load_default()


def _rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
          fill: str, outline: str | None, vibe: str, width: int = 2) -> None:
    radius = 0 if vibe == "pixel" else 10
    draw.rounded_rectangle(box, radius=radius, fill=fill,
                           outline=outline, width=width)


def _ellipsize(draw: ImageDraw.ImageDraw, text: str, font: Any,
               max_w: int) -> str:
    if draw.textlength(text, font=font) <= max_w:
        return text
    while text and draw.textlength(text + "…", font=font) > max_w:
        text = text[:-1]
    return text + "…"


PER_PAGE = 6


def render_mockup(spec: dict[str, Any],
                  style: dict[str, str] | None = None,
                  size: tuple[int, int] = (1280, 832)) -> Image.Image:
    """Draw page 1 of the active tab. Every label is REAL text."""
    st = {**DEFAULT_STYLE, **(style or {})}
    vibe = st["vibe"]
    w, h = size
    img = Image.new("RGB", size, st["background"])
    d = ImageDraw.Draw(img)
    f_title, f_tab = _font(int(h * 0.045)), _font(int(h * 0.028))
    f_item, f_small = _font(int(h * 0.030)), _font(int(h * 0.022))
    pad = int(w * 0.03)

    # title band
    _rect(d, (pad, pad, w - pad, pad + int(h * 0.085)),
          st["panel"], st["border"], vibe)
    d.text((pad * 2, pad + int(h * 0.020)), spec["title"],
           font=f_title, fill=st["text"])

    # tab row — every requested tab, the active one in the accent
    tabs = spec["tabs"]
    active = spec.get("active_tab") or tabs[0]["name"]
    tx = pad
    ty = pad + int(h * 0.105)
    tab_h = int(h * 0.055)
    for tab in tabs:
        label = tab["name"]
        tw = int(d.textlength(label, font=f_tab)) + pad
        is_active = tab["name"] == active
        _rect(d, (tx, ty, tx + tw, ty + tab_h),
              st["accent"] if is_active else st["panel"], st["border"], vibe)
        d.text((tx + pad // 2, ty + int(tab_h * 0.22)), label, font=f_tab,
               fill=st["background"] if is_active else st["text"])
        tx += tw + int(w * 0.008)

    # item cards for the active tab, page 1
    items = next(t["items"] for t in tabs if t["name"] == active)
    pages = max(1, math.ceil(len(items) / PER_PAGE))
    shown = items[:PER_PAGE]
    top = ty + tab_h + int(h * 0.03)
    bottom = h - pad - int(h * 0.075)
    cols, rows = 2, 3
    gap = int(w * 0.015)
    cw = (w - 2 * pad - gap) // cols
    ch = (bottom - top - (rows - 1) * gap) // rows
    max_level = int(spec.get("max_level") or 3)
    for idx, item in enumerate(shown):
        cx = pad + (idx % cols) * (cw + gap)
        cy = top + (idx // cols) * (ch + gap)
        _rect(d, (cx, cy, cx + cw, cy + ch), st["panel"], st["border"], vibe)
        d.text((cx + pad // 2, cy + int(ch * 0.10)),
               _ellipsize(d, item["name"], f_item, cw - pad * 3),
               font=f_item, fill=st["text"])
        # level pips + "Lv x/max"
        pip = int(ch * 0.14)
        py = cy + int(ch * 0.44)
        for i in range(max_level):
            box = (cx + pad // 2 + i * (pip + 6), py,
                   cx + pad // 2 + i * (pip + 6) + pip, py + pip)
            filled = i < int(item["level"])
            _rect(d, box, st["accent"] if filled else st["panel"],
                  st["border"], vibe, width=2)
        d.text((cx + pad // 2 + max_level * (pip + 6) + 8, py),
               f"Lv {item['level']}/{max_level}", font=f_small,
               fill=st["text"])
        # cost + upgrade chip
        cost_y = cy + int(ch * 0.70)
        d.text((cx + pad // 2, cost_y),
               f"{item['cost']:,} {spec['currency']}", font=f_small,
               fill=st["accent"])
        chip_w = int(d.textlength("Upgrade", font=f_small)) + pad
        _rect(d, (cx + cw - chip_w - pad // 2, cost_y - 4,
                  cx + cw - pad // 2, cost_y + int(h * 0.034)),
              st["accent"], None, vibe)
        d.text((cx + cw - chip_w, cost_y), "Upgrade", font=f_small,
               fill=st["background"])

    # paging footer: real page count, arrows drawn as polygons (no font
    # gamble on glyph coverage)
    fy = h - pad - int(h * 0.05)
    label = f"Page 1 / {pages}"
    lw = d.textlength(label, font=f_tab)
    cx = w // 2
    d.text((cx - lw // 2, fy), label, font=f_tab, fill=st["text"])
    ah = int(h * 0.018)
    d.polygon([(cx - lw // 2 - 3 * ah, fy + ah),
               (cx - lw // 2 - ah, fy),
               (cx - lw // 2 - ah, fy + 2 * ah)], fill=st["accent"])
    d.polygon([(cx + lw // 2 + 3 * ah, fy + ah),
               (cx + lw // 2 + ah, fy),
               (cx + lw // 2 + ah, fy + 2 * ah)], fill=st["accent"])
    return img
