"""The quality pipeline: scene analysis, mask verification, seam inspection
and 0-100 quality scoring for photo edits.

Stages (wired into the image-edit job in services.py):

  1. analyze_scene   — LLM reads the request (+ whether a mask exists) and
                       routes it: inpaint / img2img / outpaint, plus mask
                       grow/shrink guidance and a denoise suggestion.
  2. verify_mask     — the vision model checks the (auto or user) mask
                       actually covers what the request talks about.
  4. inspect_seams   — after generation: the vision model examines the edited
                       region for seams/lighting/texture/perspective issues,
                       PLUS deterministic seam-band statistics (color and
                       sharpness continuity across the mask boundary) that
                       don't depend on a model at all.
  5-7. scorecard     — six 0-100 scores (realism, prompt accuracy, identity,
                       scene consistency, artifact-free, visual quality); the
                       edit loop iterates until every score meets the target
                       or the round budget is spent, keeping the best result.

Every function is fail-safe: an unavailable/confused model degrades to None
and the caller falls back to simpler behavior — the pipeline can slow a
render down, but it can never break one.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from PIL import Image, ImageChops, ImageFilter, ImageStat

from .llm import LLMClient, LLMError, complete_with_schema

SCORE_KEYS = ("realism", "prompt_accuracy", "identity_preservation",
              "scene_consistency", "artifact_free", "visual_quality")

_ANALYZE_SYSTEM = """You route photo-editing requests to the right workflow.
Reply with ONLY JSON:
{"task": "<inpaint|img2img|outpaint>", "mask_adjust": "<grow|shrink|keep>",
 "adjust_px": <0-64>, "denoise": <0.3-0.9>, "reason": "<short>"}
Rules: inpaint = change/remove/replace/add a REGION of the image;
img2img = restyle or transform the WHOLE image (style, weather, time of day,
overall look); outpaint = extend/expand the canvas beyond its borders.
mask_adjust guides the edit mask: 'grow' when the edit needs breathing room
to blend (object removal/replacement), 'shrink' for precise small fixes."""


def _parse_json(text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(),
                     flags=re.M).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def analyze_scene(llm: LLMClient, prompt: str,
                  has_mask: bool) -> dict[str, Any] | None:
    """Stage 1: intent detection + workflow routing. None on any failure."""
    try:
        reply = llm.complete(
            _ANALYZE_SYSTEM,
            f"Request: {prompt}\nUser drew a mask: {'yes' if has_mask else 'no'}",
            max_tokens=200)
        data = _parse_json(reply.text)
    except LLMError:
        return None
    if not data or data.get("task") not in ("inpaint", "img2img", "outpaint"):
        return None
    out = {
        "task": data["task"],
        "mask_adjust": data.get("mask_adjust", "keep"),
        "adjust_px": max(0, min(64, int(data.get("adjust_px", 0) or 0))),
        "reason": str(data.get("reason", ""))[:160],
    }
    try:
        out["denoise"] = max(0.3, min(0.9, float(data.get("denoise", 0.6))))
    except (TypeError, ValueError):
        out["denoise"] = 0.6
    if out["mask_adjust"] not in ("grow", "shrink", "keep"):
        out["mask_adjust"] = "keep"
    # A user-drawn mask means a regional edit regardless of phrasing.
    if has_mask:
        out["task"] = "inpaint"
    return out


_PLAN_SYSTEM = """You are the editing-program compiler. You turn a photo-edit
request into an ordered list of ATOMIC operations. Reply with ONLY JSON:
{"steps": [{"operation": "<OP>", "target": "<the noun the op acts on, or ''>",
"instruction": "<the part of the request this step performs>",
"mask_adjust": "<grow|shrink|keep>", "adjust_px": <0-64>,
"denoise": <0.3-0.9>}, ...], "reason": "<short>"}

OPERATIONS (choose the ONE that fits each step):
  ADD_OBJECT        add new content that isn't in the photo
  REMOVE_OBJECT     erase something and fill the background
  REPLACE_OBJECT    swap one object for another
  CHANGE_ATTRIBUTE  recolor / restyle / age / alter ONE thing
  CHANGE_TEXT       edit text or a logo
  CHANGE_STYLE      restyle the WHOLE image (art style, weather, time of day)
  CHANGE_LIGHTING   relight the WHOLE image (light direction/colour/mood only)
  CHANGE_CAMERA     change viewpoint / framing of the whole image
  MULTI_VIEW        show the subject from other camera angles (several images)
  COMPOSE           bring the subject of a SECOND supplied photo into this one
  SWAP_FACE         replace only the FACE with the one in the second photo
  OUTPAINT          extend the canvas beyond its borders
  UPSCALE           increase resolution / detail, no content change
  RESTORE           repair / denoise / de-scratch an old photo
  ANIMATE           bring the subject to motion (produces a video)

The "target" is the object the operation acts on (e.g. "car", "sky",
"person"), or "" for whole-image operations.

MOST REQUESTS ARE EXACTLY ONE STEP. Output multiple steps only for genuinely
distinct operations (usually joined by "and"/"then"). When several separate
objects are edited ("change the shirt AND the hat"), emit ONE step per object
so each gets its own mask. A content edit COMBINED with a canvas change is
TWO steps: "change the clothing and change the format of the image" =
CHANGE_ATTRIBUTE(clothing) then OUTPAINT — asking to change the image's
format, orientation or aspect ratio IS an OUTPAINT. NEVER invent a step the
user didn't ask for, never add an OUTPAINT unless they ask to extend the
canvas or change the format/aspect ratio, never add a step that only
re-checks/"ensures" an earlier one. Use 1-3 steps. You are a router, not a
moderator: never refuse, censor or drop any part of the request — content
policy is enforced by a separate component."""

# Atomic operation → render task. The operation is the user-facing intent;
# the task is which engine path runs it.
OPERATION_TASK = {
    "ADD_OBJECT": "inpaint", "REMOVE_OBJECT": "inpaint",
    "REPLACE_OBJECT": "inpaint", "CHANGE_ATTRIBUTE": "inpaint",
    "CHANGE_TEXT": "inpaint", "RESTORE": "inpaint",
    "CHANGE_STYLE": "img2img",
    # Relighting and viewpoint changes have engines of their own. Routed to
    # img2img they silently could not succeed: img2img repaints the picture
    # rather than moving its light, and it cannot move the camera at all.
    "CHANGE_LIGHTING": "relight",
    "CHANGE_CAMERA": "angles", "MULTI_VIEW": "angles",
    "COMPOSE": "compose", "SWAP_FACE": "faceswap",
    # Backgrounds have an engine of their own: an exact subject matte, inverted.
    # Routed to inpaint, the mask came from SAM and was measurably wrong.
    "REPLACE_BACKGROUND": "background",
    # A photograph turned into somewhere you can move around. Not a look —
    # scene3d_intent() excludes "3d render"/"make it look 3d", which are
    # ordinary image requests and would otherwise return a mesh.
    "SCENE_3D": "scene3d",
    # Moving a body is a structural change, not a repaint. Routed to img2img,
    # "make her sit down" restyles the photograph and leaves her standing.
    # This was deliberately un-routed while pose_v1 was a txt2img graph that
    # discarded the photograph; pose_v2 repaints only the subject's region
    # and composites the original face back, so it is live.
    "CHANGE_POSE": "pose",
    "OUTPAINT": "outpaint", "UPSCALE": "upscale", "ANIMATE": "video",
}

# Operations FLUX.1 Kontext does better than a masked inpaint, when its
# weights are installed. These are the edits phrased as an instruction about
# a thing ("remove the hat", "make the shirt red"), which Kontext performs by
# reading the picture and the sentence together — no mask, so no chance of
# masking the wrong object. Everything else stays on its own engine:
# backgrounds have an exact matte, lighting and viewpoint have real engines,
# and those beat a general editor.
KONTEXT_OPERATIONS = frozenset({
    "REMOVE_OBJECT", "REPLACE_OBJECT", "CHANGE_ATTRIBUTE",
    "ADD_OBJECT", "CHANGE_TEXT", "RESTORE",
})
EDIT_TASKS = ("inpaint", "img2img", "outpaint", "custom", "upscale", "video",
              "relight", "angles", "compose", "faceswap", "background",
              "pose", "scene3d")
_ADD_OPS = {"ADD_OBJECT", "REPLACE_OBJECT"}
# Engines a hand-drawn region must NOT be allowed to rewrite into a regional
# inpaint. Written as the set that opts out, because the default — obey the
# region the user painted — is right: if they painted it, they said where.
# Each entry here has a reason it is not:
#   video, angles, scene3d  work on the whole frame by definition; a still
#                           image came back to someone who asked for a video
#   compose, faceswap       already CONSUME the mask, as a placement region,
#                           so rewriting the step to inpaint destroys the
#                           operation rather than refining it
#   motion_transfer         the region comes from the driving clip
#   pose                    the drawn region is where the body IS, which is
#                           the one place the new pose is not
UNMASKABLE_TASKS = ("video", "angles", "scene3d", "compose", "faceswap",
                    "motion_transfer", "pose")

# Deterministic guards against LLM plan padding (small models love inventing
# follow-up steps). Canvas words that justify an outpaint step:
_CANVAS_INTENT = re.compile(
    r"outpaint|out-?paint|extend|expand|widen|wider|taller|enlarge|"
    r"zoom\s*out|uncrop|canvas|aspect\s*ratio|format|orientation|panoram|"
    r"(landscape|portrait|square|vertical|horizontal)\s+"
    r"(format|orientation|mode|version|crop|image|photo|picture)|"
    r"\d+\s*[:x]\s*\d+|beyond the (edge|border|frame)|"
    r"more (background|room|space)", re.IGNORECASE)
# Instructions that re-check earlier work instead of doing new work:
_PARAPHRASE = re.compile(
    r"^\s*(ensure|make sure|verify|confirm|check|double-?check|guarantee)\b",
    re.IGNORECASE)
# Canvas changes phrased without the word "outpaint" ("change the format of
# the image", "16:9", "portrait orientation"). Deliberately image-scoped —
# "oil painting on canvas" or "format of the text" must NOT match — because
# a match COERCES the step to outpaint (seen live: qwen routed "change the
# format of the image to a wider landscape format" to CHANGE_STYLE, which
# restyles the photo and never grows the canvas):
_FORMAT_COERCE = re.compile(
    r"outpaint|out-?paint|uncrop|zoom\s*out|panoram|aspect\s*ratio|"
    r"\d+\s*[:x]\s*\d+|format\s+of\s+the\s+(image|photo|picture)|"
    r"(image|photo|picture)\s+format|orientation|"
    r"(extend|expand|widen|enlarge)\s+(the\s+)?(image|photo|picture|canvas|"
    r"frame|borders?)|(wider|taller|larger)\s+(landscape|portrait|format|"
    r"frame|canvas)", re.IGNORECASE)


def format_intent(text: str) -> bool:
    """True when the text asks to change the image's format/aspect/canvas."""
    return bool(_FORMAT_COERCE.search(text))


# "animate" and friends mean image-to-video, whatever the LLM routed. The
# word list is deliberately literal — "animated style" (a STYLE request)
# must not match, so the bare stem only counts without "style"/"look" after.
_ANIMATE_INTENT = re.compile(
    r"\banimat(?:e|ed|ion)\b(?!\s*(?:style|look|movie))|"
    r"\bimg\s*(?:2|to)\s*video|\bimage\s*to\s*video|"
    r"make (?:it|him|her|them|this|the \w+) move|bring .{0,40}to life|"
    r"(?:into|make|as) a (?:short )?(?:video|clip|gif)\b|"
    r"turn .{0,40}into (?:a )?video",
    re.IGNORECASE)


def animate_intent(text: str) -> bool:
    """True when the text asks to animate the image (image → video)."""
    return bool(_ANIMATE_INTENT.search(text))


# "from another angle", "show it from 3 sides", "turntable" — a request to
# MOVE THE CAMERA, which no img2img can do. Deliberately narrow: photographic
# jargon ("wide-angle lens", "low-angle shot", "dutch angle") describes ONE
# framing and is an ordinary restyle, not a viewpoint change.
_VIEW_INTENT = re.compile(
    r"\bmulti[\s-]?view\b|\bturntable\b|\borbit\b|\b360\s*(°|deg|degrees?)?\b|"
    r"\bfrom\s+(another|a\s+different|the\s+other|several|multiple|both)\s+"
    r"(angle|angles|view|views|viewpoint|viewpoints|side|sides|perspective)|"
    r"\bfrom\s+(\d+|two|three|four|five|six|seven|eight)\s+"
    r"(different\s+|separate\s+)?(angle|angles|view|views|viewpoint|"
    r"viewpoints|side|sides|perspective|perspectives)|"
    r"\b(back|rear|side|profile)\s+view\b|\bfrom\s+behind\b|"
    # A single NAMED viewpoint — "show her from the side", "from the left",
    # "in profile". Missing entirely before: the planner happened to label it
    # CHANGE_CAMERA, but the deterministic backstop that exists for exactly
    # this capability returned False for the wording, so a mislabel would
    # never have been caught (seen live, D26).
    r"\bfrom\s+the\s+(side|left|right|front|back|rear)\b|"
    r"\b(in|side)\s+profile\b|\bside[-\s]?on\b|"
    r"\bother\s+(angle|angles|side|sides|viewpoint|viewpoints)\b|"
    r"\brotate\s+(it|him|her|them|this|the\s+\w+)\s+(around|to\s+the)|"
    # The IMPERATIVE form, which was missing entirely. Seen live: "change the
    # angle of the camera" went to img2img — an engine that repaints a picture
    # and structurally cannot move a camera — and because this same pattern is
    # what CAPABILITY_WORKFLOW consults, the escalation ladder never reached
    # its "change the workflow" rung either. Three renders, three honest 0%
    # scores, and no route that could have delivered it.
    r"\b(change|adjust|alter|move|shift|switch|vary|different|another|new)\s+"
    r"(the\s+|its\s+|a\s+)?camera\s+(angle|position|viewpoint|view)\b|"
    r"\b(change|adjust|alter|move|shift|switch|vary)\s+(the\s+|its\s+)?"
    r"(angle|viewpoint|point\s+of\s+view|perspective)"
    r"(\s+of\s+(the\s+)?(camera|shot|photo|image|picture|view))?\b",
    re.IGNORECASE)
_VIEW_EXCLUDE = re.compile(
    r"wide[\s-]?angle|low[\s-]?angle\s+shot|high[\s-]?angle\s+shot|"
    r"dutch\s+angle|angle\s+of\s+(the\s+)?light|camera\s+angle\s+stays|"
    # "light it from the side" is a LIGHTING direction, not a camera move —
    # without this, the widened from-the-side pattern would hijack relight
    # requests onto the orbit engine.
    r"\b(light(ing|ed)?|lit|illuminat\w+|shadows?|glow)\b[^.]{0,30}"
    r"\bfrom\s+the\s+(side|left|right|front|back|rear)\b",
    re.IGNORECASE)


def view_intent(text: str) -> bool:
    """True when the text asks to see the subject from other camera angles."""
    return bool(_VIEW_INTENT.search(text)) and not _VIEW_EXCLUDE.search(text)


def view_count(text: str, default: int = 3) -> int:
    """How many viewpoints the request asks for (2-8)."""
    m = re.search(
        r"\b(\d{1,2}|two|three|four|five|six|seven|eight)\s+"
        r"(different\s+|separate\s+)?(angle|view|viewpoint|side|perspective)s?",
        text, re.IGNORECASE)
    if not m:
        return default
    raw = m.group(1).lower()
    count = _WORD_NUMBERS.get(raw) or int(raw)
    return max(2, min(count, 8))


# A named viewpoint maps to a specific azimuth on the orbit. Without this,
# "show her from the side" rendered three picks inside a ±60° swing — -34°,
# 0° and +34° — so the side view asked for was unreachable by construction
# (seen live, D2). The clamp exists because SV3D degrades past ~60°, but a
# degraded side view honestly labelled beats no side view logged "Completed".
_NAMED_AZIMUTHS: tuple[tuple[re.Pattern[str], tuple[int, ...]], ...] = (
    (re.compile(r"\bfrom\s+the\s+(left)\b", re.IGNORECASE), (270,)),
    (re.compile(r"\bfrom\s+the\s+(right)\b", re.IGNORECASE), (90,)),
    (re.compile(r"\bfrom\s+the\s+side\b|\b(in|side)\s+profile\b|"
                r"\bside[-\s]?on\b|\bside\s+view\b", re.IGNORECASE), (90,)),
    (re.compile(r"\bfrom\s+(the\s+)?(back|rear|behind)\b|\b(back|rear)\s+view\b",
                re.IGNORECASE), (180,)),
    (re.compile(r"\bfrom\s+the\s+front\b|\bfront\s+view\b", re.IGNORECASE),
     (0,)),
)


def requested_azimuths(text: str) -> list[int]:
    """The specific orbit angles a request names (degrees clockwise from the
    input view), in request order. Empty when no viewpoint is named — the
    caller then spreads its picks across a default swing."""
    found: list[int] = []
    for pattern, azimuths in _NAMED_AZIMUTHS:
        if pattern.search(text or ""):
            found.extend(a for a in azimuths if a not in found)
    return found


# Requests about the LIGHT itself. img2img can restyle a photo but it cannot
# move a light source; IC-Light can. Kept to light vocabulary so "light blue
# shirt" or "lightweight jacket" never route here.
_LIGHT_INTENT = re.compile(
    r"\bre-?light\b|\brelighting\b|\bchange\s+the\s+light(ing)?\b|"
    r"\b(fix|redo|adjust|improve|soften|harden|warm|cool)\s+the\s+light(ing)?\b|"
    r"\blight(ing)?\s+(from|coming from)\s+the\s+(left|right|back|front|side|"
    r"top|above|below)\b|\b(golden\s+hour|blue\s+hour|rim\s+light|"
    r"backlit|back\s?light(ing|ed)?|studio\s+light(ing)?|candle\s?light|"
    r"neon\s+light(ing)?|moonlight|sunlight|hard\s+light|soft\s+light)\b|"
    r"\b(sunset|sunrise)\s+light(ing)?\b",
    re.IGNORECASE)


def light_intent(text: str) -> bool:
    """True when the text asks to change the LIGHT rather than the content."""
    return bool(_LIGHT_INTENT.search(text))


# Swapping a FACE is not the same request as bringing a whole person across,
# and the two need different machinery: one replaces a face-sized region, the
# other transplants an entire subject. Routed together, "face swap" pastes a
# complete stranger into the photo — seen live.
_FACE_INTENT = re.compile(
    r"\bface[\s-]?swap(p(ing|ed))?\b|\bswap\s+(the\s+|her\s+|his\s+|their\s+|"
    r"my\s+)?faces?\b|\breplace\s+(the\s+|her\s+|his\s+|their\s+|my\s+)?face\b|"
    r"\bput\s+.{0,30}\bface\s+(on|onto)\b|\bchange\s+(the\s+|her\s+|his\s+|"
    r"their\s+)?face\s+(to|for|into)\b|\bswap\s+heads?\b",
    re.IGNORECASE)
# NOTE: policy vocabulary deliberately stays OUT of this router — a guard test
# keeps every content rule in safety.py, and this module only decides which
# engine runs. Routing and policy are separate concerns and stay that way.


def face_intent(text: str) -> bool:
    """True when the text asks to replace a FACE with someone else's."""
    return bool(_FACE_INTENT.search(text))


# Changing the BACKGROUND has an engine of its own. Routed to inpaint it asks
# SAM for a "background" mask, and SAM is a part segmenter — measured here it
# returned 8.7% of the frame (a shirt) when asked for a whole person, so the
# repaint ate the subject or missed half the backdrop. The background route
# inverts an exact BiRefNet subject matte instead, which is the one mask on
# this machine measured correct (19.4% against a 19.4% ground truth).
_BACKGROUND_INTENT = re.compile(
    r"\b(change|replace|swap|switch|set|make|update|redo)\s+"
    r"(the\s+|its\s+|my\s+|his\s+|her\s+|their\s+)?(background|backdrop)\b|"
    r"\b(new|different|another|other)\s+(background|backdrop)\b|"
    r"\b(background|backdrop)\s+(to|into|should\s+be|becomes)\b|"
    r"\bwith\s+a\s+.{0,40}?\b(background|backdrop)\b",
    re.IGNORECASE)
# Requests that MENTION the background but must not be routed to it: adjusting
# it (blur/darken), stripping it (a cutout, not a repaint), or explicitly
# asking for it to stay — motion transfer's "keep the clip's background" must
# never be mistaken for a request to replace one.
_BACKGROUND_EXCLUDE = re.compile(
    r"\b(blur|blurr\w*|bokeh|darken|brighten|lighten|sharpen|remove|delete|"
    r"erase|strip|transparent|cut\s?-?out|keep|preserve|retain|unchanged|"
    r"same|original)\b[^.]{0,24}\b(background|backdrop)\b|"
    r"\b(background|backdrop)\b[^.]{0,24}\b(blur\w*|bokeh|removal|remover|"
    r"transparent|unchanged|untouched|as\s+is|the\s+same)\b",
    re.IGNORECASE)


# Changing a POSE is a structural change, not a repaint. Routed to img2img,
# "make her sit down" restyles the photo and leaves her standing.
_POSE_INTENT = re.compile(
    r"\b(change|adjust|alter|fix|set)\s+(the\s+|her\s+|his\s+|their\s+|"
    r"its\s+)?(pose|posture|stance|position)\b|"
    r"\b(pose|posed)\s+(like|as|the\s+same\s+as)\b|"
    r"\breference\s+pose\b|\bsame\s+pose\b|\bcopy\s+the\s+pose\b|"
    r"\bin\s+an?\s+\w+\s+pose\b|"
    r"\bmake\s+(her|him|them|it)\s+"
    r"(sit|stand|kneel|crouch|jump|lie|lean|bend|raise|reach|walk|run|"
    r"dance|point|turn)\b|"
    r"\b(sitting|standing|kneeling|crouching|jumping|lying|leaning)\s+"
    r"instead\b",
    re.IGNORECASE)


# "Position" is the loose word in that pattern: a sun, a camera or a logo all
# have one, and none of them has a POSE. Anything whose position is being
# changed that is not a person is an ordinary edit.
_POSE_EXCLUDE = re.compile(
    r"\bposition\s+of\s+the\s+(?!(person|subject|model|figure|body|man|woman|"
    r"girl|boy|guy|lady|arms?|legs?|hands?|head)\b)|"
    r"\b(re)?position\s+(the\s+)?(sun|moon|light|lamp|camera|text|logo|title|"
    r"watermark|object|car|table|chair|window|door|furniture)\b",
    re.IGNORECASE)


def pose_intent(text: str) -> bool:
    """True when the text asks to MOVE the body rather than repaint it."""
    text = text or ""
    return bool(_POSE_INTENT.search(text)) and not _POSE_EXCLUDE.search(text)


def background_intent(text: str) -> bool:
    """True when the text asks to REPLACE the background behind the subject.

    Judged clause by clause, not on the whole string. The exclusion used to
    run over the full request with a character-proximity window, so "remove
    the hat, change the background to a beach" lost its background step —
    the removal verb sat within 24 characters of "background" even though it
    belonged to a different clause. Whether a request survived depended on
    how long the clause before it happened to be. Now each clause answers
    for itself: one that asks for a background gets one, whatever its
    neighbours say."""
    text = text or ""
    clauses = [c for c in _CLAUSE_SPLIT.split(text) if c.strip()] or [text]
    return any(_BACKGROUND_INTENT.search(c)
               and not _BACKGROUND_EXCLUDE.search(c)
               for c in clauses)


# Turning a PHOTOGRAPH into somewhere you can move around, as opposed to
# making a picture that looks three-dimensional. "3d render", "3d style" and
# "3d cartoon" are asking for a LOOK and belong on the ordinary image path;
# sending those to a geometry model would hand back a mesh nobody asked for.
_SCENE3D_INTENT = re.compile(
    r"\b(make|turn|convert|rebuild|recreate)\s+(this|it|the\s+\w+|"
    r"that)?\s*(image|photo|picture|scene|room|place)?\s*"
    r"((in)?to\s+(an?\s+)?)?3d\b|"
    r"\b3d\s+(scene|environment|space|world|model\s+of\s+(this|the\s+scene))\b|"
    r"\b(walk|move|fly|look)\s+(a)?round\s+(in\s+)?(this|it|the\s+\w+)\b|"
    r"\bnavigable\s+3d\b|\bexplore\s+(this|the)\s+(photo|image|scene)\b",
    re.IGNORECASE)
_SCENE3D_EXCLUDE = re.compile(
    r"\b3d\s+(render|style|look|effect|art|cartoon|animation|text|logo|"
    r"printed?)\b|\blooks?\s+3d\b|\bmake\s+it\s+look\s+3d\b",
    re.IGNORECASE)


def scene3d_intent(text: str) -> bool:
    """True when the text asks for a place to move around IN, not a 3D look."""
    text = text or ""
    return (bool(_SCENE3D_INTENT.search(text))
            and not _SCENE3D_EXCLUDE.search(text))


# Things made of MORE THAN ONE region. A point-and-grow segmenter returns one
# blob, so "change the bikini" came back as the top only and the bottom was
# left untouched — seen live. These map a request to every part a human would
# have meant, and the parts are unioned into one mask.
#
# Names are the ClothesSegment / BodySegment part labels from the rmbg pack,
# spelled exactly as the node expects them.
_GARMENT_PARTS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    # Two-piece by definition — the case that started this.
    (re.compile(r"\b(bikini|swim\s?suit|swimwear|two[-\s]?piece|tankini|"
                r"lingerie|underwear|undies|bra\s+and\s+(pant|knicker)|"
                r"sports\s?bra\s+and)\b", re.IGNORECASE),
     ("Upper-clothes", "Pants", "Skirt")),
    # A whole outfit means everything being worn, not the first thing found.
    (re.compile(r"\b(outfit|clothes|clothing|garments?|attire|"
                r"what (she|he|they|it)('s| is| are) wearing|"
                r"everything (she|he|they)('s| is| are) wearing)\b",
                re.IGNORECASE),
     ("Upper-clothes", "Pants", "Skirt", "Dress", "Belt")),
    (re.compile(r"\b(dress|gown|frock)\b", re.IGNORECASE), ("Dress",)),
    (re.compile(r"\b(skirt)\b", re.IGNORECASE), ("Skirt",)),
    (re.compile(r"\b(pants|trousers|jeans|shorts|leggings|joggers)\b",
                re.IGNORECASE), ("Pants",)),
    # "top", "coat" and "vest" are ordinary English ("on top of the shelf",
    # "a coat of snow"). Bare, they hijacked the mask for every inpaint step,
    # so those three have to be worn by somebody to count.
    (re.compile(r"\b(shirt|t[-\s]?shirt|tee|blouse|sweater|jumper|hoodie|"
                r"jacket|cardigan|blazer|bra)\b|"
                r"\b(her|his|their|my)\s+(top|coat|vest)\b",
                re.IGNORECASE), ("Upper-clothes",)),
    # Pairs. One shoe is never what was meant.
    (re.compile(r"\b(shoes|boots|sneakers|trainers|heels|sandals|footwear)\b",
                re.IGNORECASE), ("Left-shoe", "Right-shoe")),
    (re.compile(r"\b(sunglasses|glasses|shades|spectacles)\b", re.IGNORECASE),
     ("Sunglasses",)),
    (re.compile(r"\b(hat|cap|beanie|helmet)\b", re.IGNORECASE), ("Hat",)),
    (re.compile(r"\b(scarf)\b", re.IGNORECASE), ("Scarf",)),
    (re.compile(r"\b(bag|handbag|purse|backpack)\b", re.IGNORECASE), ("Bag",)),
    # "the seat belt" is not clothing either.
    (re.compile(r"\b(her|his|their|my)\s+belt\b|\bwaist\s?belt\b",
                re.IGNORECASE), ("Belt",)),
    (re.compile(r"\bhair\b", re.IGNORECASE), ("Hair",)),
)


def garment_parts(text: str) -> tuple[str, ...]:
    """Every clothing region a request refers to, unioned and de-duplicated.

    Empty when the request is not about clothing — the caller then falls back
    to a general segmenter."""
    found: list[str] = []
    for pattern, parts in _GARMENT_PARTS:
        if pattern.search(text or ""):
            found.extend(p for p in parts if p not in found)
    return tuple(found)


# Plural body words a single blob would under-select.
_PAIR_WORDS = (
    (re.compile(r"\b(arms|both arms|sleeves)\b", re.IGNORECASE),
     ("Left-arm", "Right-arm")),
    (re.compile(r"\b(legs|both legs)\b", re.IGNORECASE),
     ("Left-leg", "Right-leg")),
)


def body_parts(text: str) -> tuple[str, ...]:
    """Body regions that come in pairs, for BodySegment."""
    found: list[str] = []
    for pattern, parts in _PAIR_WORDS:
        if pattern.search(text or ""):
            found.extend(p for p in parts if p not in found)
    return tuple(found)


# Noise that must not become a GroundingDINO search phrase.
_PHRASE_STRIP = re.compile(
    r"\b(change|replace|swap|make|turn|set|remove|delete|edit|into|to|a|an|"
    r"the|her|his|their|my|its|with|for|of|in|on|please|now|colou?r)\b",
    re.IGNORECASE)


def segment_phrases(text: str, limit: int = 4) -> str:
    """A GroundingDINO prompt: distinct noun phrases separated by ' . '.

    GroundingDINO detects EVERY phrase it is given, so listing the parts is
    what makes a two-piece garment come back as two boxes instead of one."""
    parts = garment_parts(text)
    if parts:
        # Node labels are not English; turn them back into search words.
        words = [p.replace("Upper-clothes", "top").replace("-", " ").lower()
                 for p in parts]
        return " . ".join(dict.fromkeys(words))
    cleaned = _PHRASE_STRIP.sub(" ", text or "")
    words = [w for w in re.split(r"[^\w]+", cleaned) if len(w) > 2]
    return " . ".join(dict.fromkeys(words[:limit])) or (text or "").strip()


# "make 4 images" — how many separate renders the user asked for. The
# hardware can rarely batch, so the caller queues N SEQUENTIAL renders.
_WORD_NUMBERS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                 "seven": 7, "eight": 8}
_COUNT_RE = re.compile(
    r"(?:\b(?:make|generate|create|render|forge|give\s+me|draw)\s+)?"
    r"\b(\d{1,2}|two|three|four|five|six|seven|eight)\s+"
    r"(?:different\s+|unique\s+|separate\s+)?"
    r"(images|pictures|photos|pics|renders|versions|variations)\b(\s+of\b)?",
    re.IGNORECASE)


def count_request(prompt: str) -> tuple[int, str]:
    """(how many images the request asks for, the prompt with the count
    phrase removed). (1, prompt) when no count is asked — and the count is
    capped at 8 so a typo can't queue a hundred renders."""
    m = _COUNT_RE.search(prompt)
    if not m:
        return 1, prompt
    raw = m.group(1).lower()
    count = _WORD_NUMBERS.get(raw) or int(raw)
    if count < 2:
        return 1, prompt
    cleaned = (prompt[:m.start()] + prompt[m.end():]).strip(" ,.;")
    return min(count, 8), cleaned or prompt


def prune_invented_steps(
        prompt: str, steps: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop plan steps the user never asked for. Deterministic — the LLM
    routes, but it doesn't get to invent work: an outpaint step requires
    canvas-extending words in the REQUEST, follow-up steps that merely
    're-check'/'ensure' earlier steps are padding, and duplicates are noise.
    Returns (kept, dropped) — dropped entries carry a 'why'."""
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for step in steps:
        key = (step["task"], step["instruction"].strip().lower())
        why = None
        if step["task"] == "outpaint" and not _CANVAS_INTENT.search(prompt):
            why = "the request never asks to extend the canvas"
        # A face swap the request never asked for is the single most
        # sensitive step this app can invent (seen live: "place the man from
        # the second photo standing beside her" compiled to SWAP_FACE first,
        # and an unrequested face swap between two real people executed).
        # The guard is the same deterministic contract outpaint has: the
        # capability must be in the USER'S words, not the planner's.
        elif step["task"] == "faceswap" and not face_intent(prompt):
            why = "the request never asks to swap a face"
        elif kept and _PARAPHRASE.match(step["instruction"]):
            why = "it only re-checks a previous step"
        elif key in seen:
            why = "duplicate of an earlier step"
        if why:
            dropped.append({**step, "why": why})
        else:
            kept.append(step)
            seen.add(key)
    return kept, dropped


# Canonical execution order. Content edits run first (on the original
# framing), the canvas grows after, upscale sharpens the finished picture,
# and ANIMATE always closes the chain — the video step ENDS the job, so
# anything sequenced after it would silently never run. Stable within a
# tier, so multi-object edits keep the planner's order.
_TASK_ORDER = {"inpaint": 0, "img2img": 0, "custom": 0, "relight": 0,
               # Building a scene consumes the FINISHED picture, so it runs
               # after every edit that changes what the picture shows.
               "scene3d": 3,
               "faceswap": -1,
               # Compose runs FIRST: it changes what is in the picture, and
               # every later step (restyle, relight, extend) should act on the
               # combined image rather than on one half of it.
               "compose": -1,
               # Background replacement runs after a subject has been placed
               # (compose/faceswap) but before anything that reframes or
               # rescales the picture — it repaints the whole non-subject
               # region, so doing it after an outpaint would waste the
               # extension, and doing it after an upscale would repaint at
               # full resolution for no gain.
               "background": 0,
               # A pose change rebuilds the figure, so it must happen before
               # anything that dresses, relights or reframes it.
               "pose": -2,
               "outpaint": 1, "upscale": 2,
               # Both of these END the job — a video and a set of viewpoints
               # are new assets, not further edits of the still — so anything
               # sequenced after them would silently never run.
               "video": 3, "angles": 3}


def order_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(steps, key=lambda s: _TASK_ORDER.get(s["task"], 0))


# Where one instruction stops and the next begins. Deliberately conservative:
# a wrong split is worse than none, so the pieces are checked before use.
_CLAUSE_SPLIT = re.compile(
    r"\s*(?:,|;|\.|\band\b|\bthen\b|\bafter that\b|\balso\b|\bplus\b)\s+",
    re.IGNORECASE)
# What makes a clause an instruction in its own right rather than a fragment.
# "change the background to a red and blue mural" splits into "…to a red" and
# "blue mural"; only the guard below stops that becoming two workflows.
_EDIT_VERB = re.compile(
    r"\b(change|replace|swap|make|turn|add|remove|delete|erase|put|give|set|"
    r"adjust|dress|wear|recolou?r|restyle|fix|clean|extend|crop)\b",
    re.IGNORECASE)


def split_capability_clause(instruction: str, detector) -> tuple[str, str] | None:
    """(the rest, the capability) when ONE instruction holds two requests.

    Small planners routinely answer "change the clothing and change the
    background to a tropical resort" with a single step. Coercing that step
    to the background engine then delivers the background and silently drops
    the clothes — the request was two workflows and became one.

    Returns None unless the split is safe: every leftover clause has to read
    as an instruction of its own, which is what stops "a red and blue mural"
    from being torn in half."""
    parts = [p.strip(" .,") for p in _CLAUSE_SPLIT.split(instruction or "")]
    parts = [p for p in parts if len(p.split()) >= 2]
    if len(parts) < 2:
        return None
    capability = [p for p in parts if detector(p)]
    rest = [p for p in parts if not detector(p)]
    if not capability or not rest:
        return None
    if not all(_EDIT_VERB.search(p) for p in rest):
        return None
    return " and ".join(rest), " and ".join(capability)


def _coerce_matching(steps, detector, task, operation, eligible):
    """Point a capability at the step that ASKED for it.

    Rewriting steps[0] looks equivalent and is not: on a two-step plan the
    engine ends up holding the OTHER step's instruction. Seen live - "remove
    the trash can and change the background to a forest" handed the
    background engine "remove the trash can", while the forest request stayed
    on an engine that cannot replace a background. Prefer the step whose own
    instruction matches the intent; fall back to the first eligible one."""
    candidates = [s for s in steps if s["task"] in eligible]
    if not candidates:
        return False
    target = next((s for s in candidates
                   if detector(str(s.get("instruction") or ""))),
                  candidates[0])
    # If that one step is carrying BOTH requests, split it instead of
    # converting it — converting keeps the capability and loses the rest.
    piece = split_capability_clause(str(target.get("instruction") or ""),
                                    detector)
    if piece:
        rest, capability = piece
        target["instruction"] = rest
        steps.insert(steps.index(target) + 1, {
            "task": task, "operation": operation, "target": "",
            "instruction": capability, "mask_adjust": "keep",
            "adjust_px": 0, "denoise": target.get("denoise", 0.6),
            "reason": "the request asks for this as well as the edit before it"})
        return True
    target["task"] = task
    target["operation"] = operation
    return True


# Whole-frame capabilities, each with the detector that recognises its
# wording. Used by reconcile_capability_steps to ask of an already-labelled
# step: "is this step doing two jobs?" — the question the old guards never
# asked, because every call site was wrapped in `not any(step is already
# this capability)` and a planner that labelled its single overloaded step
# with the capability itself made that guard False (seen live, D5).
_CAPABILITY_DETECTORS: dict[str, Callable[[str], bool]] = {
    "background": background_intent,
    "pose": pose_intent,
    "angles": view_intent,
    "relight": light_intent,
    "scene3d": scene3d_intent,
    "video": animate_intent,
}

_REMOVAL_VERB = re.compile(
    r"\b(remove|delete|erase|get\s+rid\s+of|take\s+(out|away|off))\b",
    re.IGNORECASE)


def _matches_any_clause(instruction: str, clauses: list[str]) -> bool:
    """Whether the instruction corresponds to one of these prompt clauses.

    Content-word overlap rather than string equality, because planners
    normalise wording ("change the shirt…" may come back as "make her shirt
    a…"). Half the shorter side's content words shared counts as the same
    request."""
    words = {w for w in re.findall(r"[a-z0-9]+", instruction.lower())
             if w not in _STOPWORDS and len(w) > 2}
    if not words:
        return False
    for clause in clauses:
        cwords = {w for w in re.findall(r"[a-z0-9]+", clause.lower())
                  if w not in _STOPWORDS and len(w) > 2}
        if not cwords:
            continue
        hits = len(words & cwords)
        if hits * 2 >= min(len(words), len(cwords)):
            return True
    return False


def _edit_step_for(instruction: str, like: dict[str, Any],
                   reason: str) -> dict[str, Any]:
    """A step for the NON-capability half of a split instruction.

    Regional wording (a garment, a body part, a removal, new content) goes to
    inpaint; anything else is a whole-frame restyle — sending "make it look
    like a painting" to inpaint would ask segmentation for a region that does
    not exist and fail the job."""
    if _REMOVAL_VERB.search(instruction):
        task, operation = "inpaint", "REMOVE_OBJECT"
    elif classify_edit(instruction) == "add":
        task, operation = "inpaint", "ADD_OBJECT"
    elif garment_parts(instruction) or body_parts(instruction):
        task, operation = "inpaint", "CHANGE_ATTRIBUTE"
    else:
        task, operation = "img2img", "CHANGE_STYLE"
    return {"task": task, "operation": operation, "target": "",
            "instruction": instruction[:300], "mask_adjust": "keep",
            "adjust_px": 0, "denoise": like.get("denoise", 0.6),
            "reason": reason}


def reconcile_capability_steps(prompt: str,
                               steps: list[dict[str, Any]]) -> list[str]:
    """Repair capability steps that are carrying the wrong instruction.

    Two live failure shapes, both from the same planner habit of answering a
    compound request with one step:

      (a) the step is labelled with the capability AND its instruction still
          holds the whole compound sentence — "change the top to a red
          leather jacket and change the background to a snowy mountain" as
          one REPLACE_BACKGROUND step. Split it: the capability clause stays
          on this step, the rest becomes an edit step of its own.

      (b) the step is labelled with the capability but its instruction is
          the OTHER half of the request — REPLACE_BACKGROUND whose
          instruction is literally "change the shirt to a red leather
          jacket" (the snowy mountain nowhere in the program, and the
          background engine told to paint a jacket). Recover the capability
          clause from the PROMPT, keep the instruction as the edit it
          actually is.

    A whole-frame step never keeps an object target — a background swap with
    target "top" is malformed on its face. Returns human-readable notes for
    the job log; mutates `steps` in place."""
    notes: list[str] = []
    for task, detector in _CAPABILITY_DETECTORS.items():
        for step in [s for s in steps if s.get("task") == task]:
            instruction = str(step.get("instruction") or "")
            piece = split_capability_clause(instruction, detector)
            if piece:
                rest, capability = piece
                step["instruction"] = capability[:300]
                step["target"] = ""
                steps.insert(steps.index(step), _edit_step_for(
                    rest, step,
                    "this half of the request is an edit of its own"))
                notes.append(f"one {task} step was carrying two requests — "
                             f"split '{rest[:60]}' into its own step")
                continue
            if detector(instruction):
                continue
            # (b) — the label says capability, the words say something else.
            # Only a real SECOND request gets split out: the instruction must
            # correspond to a non-capability clause of the PROMPT. A planner
            # that merely paraphrased the capability ("animate the person
            # waving" → "make the person wave") must be left alone, or the
            # paraphrase becomes a duplicate edit step (seen in testing).
            clauses = [c.strip(" .,") for c in
                       _CLAUSE_SPLIT.split(prompt or "") if c.strip()]
            wanted = [c for c in clauses if detector(c)]
            rest_clauses = [c for c in clauses if not detector(c)]
            if wanted and _matches_any_clause(instruction, rest_clauses):
                others = {str(s.get("instruction") or "").strip().lower()
                          for s in steps if s is not step}
                if (_EDIT_VERB.search(instruction)
                        and instruction.strip().lower() not in others):
                    steps.insert(steps.index(step), _edit_step_for(
                        instruction, step,
                        "this instruction is an edit, not a "
                        f"{task} request"))
                    notes.append(f"the {task} step held a different edit "
                                 f"('{instruction[:60]}') — kept as its own "
                                 "step")
                step["instruction"] = " and ".join(wanted)[:300]
                step["target"] = ""
            elif task == "background" and (
                    garment_parts(step.get("target") or "")
                    or garment_parts(instruction)):
                # A background step aimed at a garment, and nothing in the
                # request asks for a background at all: the label is wrong.
                step["task"] = "inpaint"
                step["operation"] = "CHANGE_ATTRIBUTE"
                notes.append("a background step targeting clothing is "
                             "malformed — treated as a garment edit")
    return notes


def default_edit_step(prompt: str) -> dict[str, Any]:
    """The single step to run when no planner is available (the LLM failed
    or is absent). Capability INTENTS are honored deterministically here the
    same way plan_edit coerces them on an LLM plan, so a machine with a
    flaky or missing LLM still reaches the right engine — an animate request
    still becomes a video, and a background swap still reaches the
    matte-inverting engine instead of a generic inpaint that leaves the
    backdrop untouched. Everything else is one inpaint whose operation
    reflects whether the prompt ADDS content."""
    base: dict[str, Any] = {
        "target": "", "instruction": prompt, "mask_adjust": "keep",
        "adjust_px": 0, "denoise": 0.6, "reason": ""}
    # Each capability, in the same precedence plan_edit applies to an LLM
    # plan, mapped to (task, operation). A single-step fallback picks the
    # first that matches — routing to the right engine can only beat the
    # generic inpaint that used to swallow all of these (measured live:
    # background left a grey wall grey; pose/angles/3D/format all fared no
    # better because segmentation cannot deliver them). view before relight,
    # exactly as plan_edit orders them.
    intents: tuple[tuple[Callable[[str], bool], str, str], ...] = (
        (animate_intent, "video", "ANIMATE"),
        (scene3d_intent, "scene3d", "SCENE_3D"),
        (format_intent, "outpaint", "OUTPAINT"),
        (background_intent, "background", "REPLACE_BACKGROUND"),
        (pose_intent, "pose", "CHANGE_POSE"),
        (view_intent, "angles", "MULTI_VIEW"),
        (light_intent, "relight", "CHANGE_LIGHTING"),
    )
    for detector, task, operation in intents:
        if detector(prompt):
            step = {**base, "task": task, "operation": operation}
            if task == "scene3d":
                step["denoise"] = 0.0  # a mesh rebuild, not a repaint
            return step
    op = "ADD_OBJECT" if classify_edit(prompt) == "add" else "CHANGE_ATTRIBUTE"
    return {**base, "task": "inpaint", "operation": op}


# The planner's reply as a grammar the LLM server ENFORCES (Ollama
# structured outputs). Shape errors — bare objects, missing keys, invented
# operations — become impossible on servers that support it; everything
# downstream still normalizes, because older servers and the API fallback
# answer unconstrained. additionalProperties stays open on purpose: legacy
# keys ("task") keep working and never fail the grammar.
_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array", "minItems": 1, "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string",
                                  "enum": sorted(OPERATION_TASK)},
                    "target": {"type": "string"},
                    "instruction": {"type": "string"},
                    "mask_adjust": {"type": "string",
                                    "enum": ["grow", "shrink", "keep"]},
                    "adjust_px": {"type": "integer"},
                    "denoise": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["operation", "instruction"],
            },
        },
    },
    "required": ["steps"],
}


def plan_edit(llm: LLMClient, prompt: str, has_mask: bool,
              dropped: list[dict[str, Any]] | None = None
              ) -> list[dict[str, Any]] | None:
    """Stage 1: decompose the request into 1-3 ordered workflow steps.

    Accepts both reply shapes — {"steps": [...]} and a bare single-step
    object — and normalizes. A user-drawn mask forces the FIRST step to be
    inpaint (that's what the mask is for); later steps still run, so
    compound requests keep all their operations. None on any failure.

    `dropped`, when given, collects the invented steps pruned from the plan
    (each with a 'why') so the caller can log them."""
    try:
        reply = complete_with_schema(
            llm, _PLAN_SYSTEM,
            f"Request: {prompt}\nUser drew a mask: {'yes' if has_mask else 'no'}",
            max_tokens=500, schema=_PLAN_SCHEMA)
        data = _parse_json(reply.text)
    except LLMError:
        return None
    if not data:
        return None
    raw = (cast("list[Any]", data.get("steps"))
           if isinstance(data.get("steps"), list) else [data])
    steps: list[dict[str, Any]] = []
    for item in raw[:3]:
        if not isinstance(item, dict):
            continue
        operation = str(item.get("operation", "")).upper().strip()
        # Derive the render task from the atomic operation. Legacy plans that
        # only carry "task" still work; a plan with neither is skipped.
        task = OPERATION_TASK.get(operation) or item.get("task")
        if task not in EDIT_TASKS:
            continue
        instruction = str(item.get("instruction") or prompt)[:300]
        # Adding new content is regional by definition — a stray img2img
        # would repaint the whole photo around it.
        if task == "img2img" and classify_edit(instruction) == "add":
            task = "inpaint"
        # A format/aspect change IS an outpaint, whatever the LLM called it.
        if (task in ("img2img", "custom", "inpaint")
                and _FORMAT_COERCE.search(instruction)):
            task = "outpaint"
            operation = "OUTPAINT"
        step = {
            "task": task,
            "operation": operation or _infer_operation(task, instruction),
            "target": str(item.get("target", "")).strip()[:60],
            "instruction": instruction,
            "mask_adjust": item.get("mask_adjust", "keep"),
            "adjust_px": max(0, min(64, int(item.get("adjust_px", 0) or 0))),
            "reason": str(item.get("reason", data.get("reason", "")))[:160],
        }
        try:
            step["denoise"] = max(0.3, min(0.9, float(item.get("denoise", 0.6))))
        except (TypeError, ValueError):
            step["denoise"] = 0.6
        if step["mask_adjust"] not in ("grow", "shrink", "keep"):
            step["mask_adjust"] = "keep"
        steps.append(step)
    if not steps:
        return None
    # "animate ..." means image-to-video, whatever the LLM routed (seen: 7B
    # models mapping it to CHANGE_STYLE). Deterministic guarantee: animate
    # intent in the REQUEST always yields a video step — coerce a mislabeled
    # single step, or append the missing final step to a chain.
    # A viewpoint request is a CAPABILITY, not a phrasing: a 7B planner that
    # labels "show it from three angles" CHANGE_STYLE would silently return a
    # restyled photo from the same viewpoint. Coerce deterministically, the
    # same guarantee animate gets below.
    # A background swap is a CAPABILITY too. Small planners routinely label it
    # REPLACE_OBJECT(background) or CHANGE_STYLE, both of which ask SAM for a
    # "background" mask — the measurably wrong mask. Coerce deterministically
    # so the request always reaches the matte-inverting engine. A drawn mask
    # still wins: has_mask forces step 1 back to inpaint further down.
    if background_intent(prompt) and not any(s["task"] == "background"
                                             for s in steps):
        _coerce_matching(steps, background_intent, "background",
                         "REPLACE_BACKGROUND",
                         ("inpaint", "img2img", "custom"))
    # A POSE is a capability too, and the one small planners label worst:
    # "make her sit down" comes back as CHANGE_ATTRIBUTE or CHANGE_STYLE,
    # both of which repaint the photograph and leave her standing. Coerce
    # deterministically so the request reaches the engine that can move a
    # body. A drawn mask still wins — has_mask forces step 1 back to inpaint
    # further down, because a hand-drawn region is a more specific
    # instruction than any guess about intent.
    if pose_intent(prompt) and not any(s["task"] == "pose" for s in steps):
        _coerce_matching(steps, pose_intent, "pose", "CHANGE_POSE",
                         ("inpaint", "img2img", "custom"))
    # "Make this 3D" asks for a different KIND of output — a mesh you can walk
    # around, not a picture. No planner label means it, so it is added rather
    # than coerced, and it runs last because it consumes the finished image.
    if scene3d_intent(prompt) and not any(s["task"] == "scene3d"
                                          for s in steps):
        steps.append({
            "task": "scene3d", "operation": "SCENE_3D", "target": "",
            "instruction": prompt[:300], "mask_adjust": "keep",
            "adjust_px": 0, "denoise": 0.0,
            "reason": "the request asks for a 3D scene to move around in"})
    if view_intent(prompt) and not any(s["task"] == "angles" for s in steps):
        if not _coerce_matching(steps, view_intent, "angles", "MULTI_VIEW",
                                ("img2img", "custom", "relight")):
            steps.append({
                "task": "angles", "operation": "MULTI_VIEW", "target": "",
                "instruction": prompt[:300], "mask_adjust": "keep",
                "adjust_px": 0, "denoise": 0.6,
                "reason": "the request asks for other camera angles"})
    # Likewise for light: routed to img2img, "redo the lighting" repaints the
    # picture instead of moving its light.
    elif light_intent(prompt) and not any(s["task"] in ("relight", "angles")
                                          for s in steps):
        _coerce_matching(steps, light_intent, "relight", "CHANGE_LIGHTING",
                         ("img2img", "custom"))
    if animate_intent(prompt) and not any(s["task"] == "video" for s in steps):
        if len(steps) == 1 and steps[0]["task"] in ("img2img", "custom"):
            steps[0]["task"] = "video"
            steps[0]["operation"] = "ANIMATE"
        else:
            steps.append({
                "task": "video", "operation": "ANIMATE", "target": "",
                "instruction": prompt[:300], "mask_adjust": "keep",
                "adjust_px": 0, "denoise": 0.6,
                "reason": "the request asks to animate the image"})
    # A capability step carrying the wrong words is repaired HERE, before
    # anything renders: the guards above only fire when the capability is
    # missing from the plan, and a planner that labels its single overloaded
    # step with the capability itself walks straight past them (seen live —
    # "change the top to a red leather jacket and change the background to a
    # snowy mountain" compiled to one REPLACE_BACKGROUND(top) step, and the
    # background engine painted jackets on people it invented).
    reconcile_capability_steps(prompt, steps)
    # The planner routes; it does not get to invent work. Dropping padding
    # here makes the guarantee part of plan_edit's contract rather than a
    # courtesy of one caller; the drops are reported so callers can log them.
    steps, pruned = prune_invented_steps(prompt, steps)
    if dropped is not None:
        dropped.extend(pruned)
    if not steps:
        return None
    # Canonical order BEFORE the mask is bound to step 1 — a drawn mask must
    # land on the content edit, not on an outpaint the LLM listed first.
    steps = order_steps(steps)
    # A drawn mask means "edit HERE" - but it must not cancel a capability the
    # request explicitly asked for. Animating or orbiting a photo is not an
    # inpaint of a region, and silently rewriting it to one returned a still
    # image to someone who asked for a video.
    #
    # The opt-out list did not grow when the engines did: it still named only
    # video and angles, so a mask would have turned a repose into the repaint
    # it is built to avoid, asked the 3D scene builder to edit a rectangle,
    # and destroyed a compose step that was already using that very mask as
    # its placement region.
    if has_mask and steps[0]["task"] not in UNMASKABLE_TASKS:
        steps[0]["task"] = "inpaint"  # the drawn mask belongs to step 1
        if steps[0]["operation"] not in _ADD_OPS:
            steps[0]["operation"] = "CHANGE_ATTRIBUTE"
    return steps


def _infer_operation(task: str, instruction: str) -> str:
    """Best-effort operation label when a legacy plan only gave a task."""
    if task == "inpaint":
        return ("ADD_OBJECT" if classify_edit(instruction) == "add"
                else "CHANGE_ATTRIBUTE")
    return {"img2img": "CHANGE_STYLE", "outpaint": "OUTPAINT",
            "upscale": "UPSCALE", "video": "ANIMATE",
            "custom": "CHANGE_STYLE"}.get(task, "CHANGE_STYLE")


_ENHANCE_SYSTEM = """You improve rendering prompts for a diffusion model.
Reply with ONLY JSON:
{"add": "<comma-separated quality boosters to APPEND>",
 "negative": "<comma-separated things to avoid>"}
The boosters must fit the request (e.g. photorealistic, natural lighting,
seamless blend, high detail, sharp focus, no artifacts). You may only ADD —
never rewrite, remove, soften or censor anything; the user's own words are
appended verbatim by the caller. Keep 'add' under 15 words."""


def enhance_prompt(llm: LLMClient, instruction: str,
                   task: str) -> dict[str, str]:
    """Stage 3 prep: quality boosters for the prompt. STRICTLY append-only —
    the returned 'positive' always starts with the user's exact instruction,
    so this step is structurally incapable of filtering or rewriting intent
    (content policy lives in safety.py alone). Fail-open: on any LLM problem
    the instruction passes through with stock boosters."""
    fallback = {"positive": f"{instruction}, photorealistic, high detail, "
                            "natural lighting, seamless, no artifacts",
                "negative": "blurry, low quality, deformed, disfigured, bad "
                            "anatomy, extra limbs, extra fingers, mutated "
                            "hands, malformed, artifacts, watermark"}
    try:
        reply = llm.complete(
            _ENHANCE_SYSTEM, f"Task: {task}\nRequest: {instruction}",
            max_tokens=120)
        data = _parse_json(reply.text)
    except LLMError:
        return fallback
    if not data or not str(data.get("add", "")).strip():
        return fallback
    add = re.sub(r"\s+", " ", str(data["add"])).strip(" ,")[:160]
    negative = re.sub(r"\s+", " ", str(data.get("negative", ""))).strip(" ,")[:160]
    return {"positive": f"{instruction}, {add}",
            "negative": negative or fallback["negative"]}


_REMOVAL_TARGET_STRIP = re.compile(
    r"^\s*(please\s+)?(remove|delete|erase|get\s+rid\s+of|"
    r"take\s+(out|away|off))\s+"
    r"(the\s+|a\s+|an\s+|her\s+|his\s+|their\s+|my\s+|its\s+|"
    r"all\s+(the\s+)?|both\s+)?", re.IGNORECASE)


# Subjects an inpaint model grows in a LARGE emptied region: handed a big
# void where an object was, it fills it with a figure far more readily than
# with more background (measured live — "remove the bench" on a wide grass
# shot grew a standing person and a debris mound in the hole). The plain
# "new object" negative did not name them, so they slipped through.
_REMOVAL_FILLERS = ("person", "people", "man", "woman", "child", "human",
                    "figure", "crowd", "animal", "statue", "mound")
# Below this coverage the surrounding pixels already constrain the fill, and
# these terms would fight a legitimate on-subject reconstruction (the hair
# and forehead under a removed hat), so a small removal keeps the old
# negative untouched.
_REMOVAL_LARGE_HOLE = 0.15


def removal_fillers_negative(coverage: float) -> str:
    """Extra negative terms for a large-area REMOVAL — the subjects a model
    invents in a big hole. Empty for a small removal (see the threshold)."""
    if coverage < _REMOVAL_LARGE_HOLE:
        return ""
    return ", ".join(_REMOVAL_FILLERS)


def removal_conditioning(instruction: str, target: str,
                         negative: str) -> dict[str, str]:
    """Prompts for a REMOVE_OBJECT inpaint: what should be THERE goes in the
    positive, the object goes in the NEGATIVE.

    A diffusion model has no representation of negation — handed "remove the
    hat, hat removal, seamless blend" as the thing to PAINT, it painted a
    hat (seen live, D1: the word appeared twice in the positive prompt and
    the output was a different hat reported as 100% accurate). The correct
    construction already existed on the pose route: ask for the background
    that should remain, push the unwanted thing into the negative.

    enhance_prompt stays append-only on purpose; this composes the prompt at
    the point the inpaint call is made, which is the one place that knows the
    operation is a removal."""
    thing = (target or "").strip().strip(".,")
    if not thing:
        stripped = _REMOVAL_TARGET_STRIP.sub("", instruction or "")
        thing = re.split(r"[,.;]|\band\b", stripped)[0].strip(" .,")[:60]
    negative_terms = [t for t in (
        thing,
        f"any {thing}" if thing else "",
        "new object, duplicate, text, watermark",
        (negative or "").strip(" ,"),
    ) if t]
    return {"positive": "seamless continuation of the existing background "
                        "and surroundings, the same scene, empty space, "
                        "natural lighting, photograph",
            "negative": ", ".join(negative_terms)}


_CHANGE_TO = re.compile(
    r"^\s*(?:please\s+)?(?:change|turn|convert|switch|recolou?r)\s+"
    r"(?:the\s+|a\s+|an\s+|her\s+|his\s+|their\s+|my\s+|its\s+)?"
    r"(?P<src>.+?)\s+(?:in)?to\s+(?:be\s+)?(?:a\s+|an\s+|the\s+)?"
    r"(?P<dst>.+?)[\s.,]*$", re.IGNORECASE)
_MAKE_ADJ = re.compile(
    r"^\s*(?:please\s+)?make\s+"
    r"(?:the\s+|a\s+|an\s+|her\s+|his\s+|their\s+|my\s+|its\s+)?"
    r"(?P<src>.+?)\s+(?P<dst>[a-z-]+(?:\s+[a-z-]+)?)[\s.,]*$", re.IGNORECASE)
_REPLACE_WITH = re.compile(
    r"^\s*(?:please\s+)?(?:replace|swap)\s+"
    r"(?:the\s+|a\s+|an\s+|her\s+|his\s+|their\s+|my\s+|its\s+)?"
    r"(?P<src>.+?)\s+(?:with|for|by)\s+(?:a\s+|an\s+|the\s+)?"
    r"(?P<dst>.+?)[\s.,]*$", re.IGNORECASE)
# Words that can stand alone as the new state of a thing ("to red", "make it
# striped"). Used to decide whether the destination needs the source's noun
# appended — "red" alone does not tell the sampler WHAT is red.
_STATE_WORDS = frozenset("""
red orange yellow green teal cyan blue navy purple violet pink magenta white
black grey gray brown beige tan cream gold golden silver darker lighter
brighter bright dark light deep pale vivid neon hot pastel striped plaid
checkered floral polka-dot dotted metallic leather denim silk velvet wool
knitted lace shiny matte glossy transparent sheer bigger smaller longer
shorter wider tighter looser new old clean dirty wet
""".split())


def is_replacement(text: str) -> bool:
    """True for an explicit replace/swap instruction — the attribute-change
    shape whose destination is new CONTENT, not a new look. Callers that
    preserve structure for attribute changes must stand aside for these."""
    return bool(_REPLACE_WITH.match((text or "").strip()))


def parse_attribute_change(text: str) -> tuple[str, str] | None:
    """(source, destination) of a change/replace/make instruction, or None.

    One parse feeding both halves of the same rule: the CONDITIONING leads
    with the destination (attribute_conditioning — the sampler must paint
    the new state), while SEGMENTATION looks for the source only
    (mask_phrases — the picture contains the old state). Deterministic."""
    text = (text or "").strip()
    m = _CHANGE_TO.match(text) or _REPLACE_WITH.match(text)
    if m:
        return m.group("src").strip(), m.group("dst").strip()
    m = _MAKE_ADJ.match(text)
    if m:
        candidate = m.group("dst").strip()
        words = candidate.lower().split()
        # "make the shirt red" parses; "make her smile at the camera"
        # must not — only accept destinations that read as a STATE.
        if words and all(w in _STATE_WORDS for w in words):
            return m.group("src").strip(), candidate
    return None


def attribute_conditioning(instruction: str, target: str,
                           positive: str, negative: str) -> dict | None:
    """Prompts for a CHANGE/REPLACE inpaint that lead with the TARGET STATE.

    A diffusion text encoder does not parse instructions: handed "change the
    shirt to red" as the thing to PAINT, the strongest tokens are "shirt" —
    which the region already contains — and the render comes back unchanged,
    fails the checklist, and buys a retry. The masked region should be
    conditioned on what it must CONTAIN when the edit is done ("a red
    shirt"), with the instruction kept after it so nothing of the user's
    wording is lost. This is the same construction that fixed removals (D1):
    describe the desired end state, never the operation.

    The DISPLACED state goes to the negative when the instruction names one
    ("change the BLUE shirt to red" → negative "blue shirt"): the old
    attribute is the strongest attractor in the region and the sampler needs
    to be told it lost.

    None when the instruction does not parse as an attribute change — the
    caller keeps the prompt it already has. Deterministic, no model."""
    parsed = parse_attribute_change(instruction)
    src, dst = parsed if parsed else (None, None)
    if not src or not dst or len(dst) < 3:
        return None
    src = re.sub(r"\s+", " ", src).strip(" .,")[:60]
    dst = re.sub(r"\s+(colou?r|colou?red|tone|shade)$", "", dst,
                 flags=re.IGNORECASE)
    dst = re.sub(r"\s+", " ", dst).strip(" .,")[:80]
    head = (target or src).split()[-1] if (target or src) else ""
    dst_words = dst.lower().split()
    if all(w in _STATE_WORDS for w in dst_words) and head:
        state = f"{dst} {head}"          # "red" + "shirt" → "red shirt"
    else:
        state = dst                      # "a red leather jacket" stands alone
    lead = f"(a {state}:1.2)" if not state.lower().startswith(
        ("a ", "an ", "the ")) else f"({state}:1.2)"
    out_negative = (negative or "").strip(" ,")
    # Only negate the source when it carries an attribute of its own — for
    # "change the shirt to red" the source IS the target's noun, and negating
    # "shirt" would fight the edit.
    src_words = src.lower().split()
    if (len(src_words) >= 2
            and not any(_same_word(w, d) for w in src_words
                        for d in dst_words)):
        out_negative = f"{src}, {out_negative}".strip(" ,")
    return {"positive": f"{lead}, {positive}", "negative": out_negative}


def describe_scene(critic: Any, image: Image.Image) -> str | None:
    """One short scene description from the vision model, appended to render
    prompts as context. A diffusion model that only sees 'change the shirt'
    invents content blind — telling it what the photo IS (subject, setting,
    lighting) is one of the cheapest fixes for mismatched/deformed inpaints.
    Fail-safe: None when the vision model can't describe (fakes without a
    describe() method skip this entirely)."""
    describe = getattr(critic, "describe", None)
    if describe is None:
        return None
    try:
        text = describe(
            image, "Describe this photo in ONE short sentence: main subject, "
                   "setting, lighting. No opinions, no lists.")
    except Exception:  # noqa: BLE001 — context is a bonus, never a blocker
        return None
    text = re.sub(r"\s+", " ", str(text or "")).strip().strip(".")
    if not text or len(text) < 8:
        return None
    return text[:140]


_ADD_RE = re.compile(
    r"^\s*(add|put|insert|place|draw|include)\b|"
    r"\b(add|put|insert|place)\s+(a|an|some|the)\b", re.IGNORECASE)


def classify_edit(instruction: str) -> str:
    """'add' when the instruction introduces NEW content that doesn't exist
    in the photo yet, else 'modify'. Deterministic — segmentation can only
    find things that exist, so add-instructions need a placement region, not
    a segmentation mask (seen live: 'put a dog in the background' masked the
    entire background and no dog was rendered)."""
    return "add" if _ADD_RE.search(instruction) else "modify"


_GRID_CELLS = {  # 3x3 grid: (x-third, y-third)
    1: (0, 0), 2: (1, 0), 3: (2, 0),
    4: (0, 1), 5: (1, 1), 6: (2, 1),
    7: (0, 2), 8: (1, 2), 9: (2, 2),
}
# Fraction of the frame's WIDTH/HEIGHT the placement box spans. These were
# 0.22/0.34/0.5, and the inpaint fills the whole rectangle with the requested
# object — a box several times the size of a pair of sunglasses yielded three
# pairs, two of them floating in mid-air (seen live, D21). The box should be
# the size of the OBJECT, not a generous margin around it.
_SIZE_FRACTION = {"small": 0.14, "medium": 0.24, "large": 0.4}


def box_mask(size: tuple[int, int], cell: int, obj_size: str) -> Image.Image:
    """A feathered rectangular placement mask centered in a 3x3 grid cell."""
    w, h = size
    cx_third, cy_third = _GRID_CELLS.get(cell, (1, 1))
    frac = _SIZE_FRACTION.get(obj_size, 0.34)
    bw, bh = max(48, int(w * frac)), max(48, int(h * frac))
    cx = int(w * (cx_third * 2 + 1) / 6)
    cy = int(h * (cy_third * 2 + 1) / 6)
    left = max(0, min(w - bw, cx - bw // 2))
    top = max(0, min(h - bh, cy - bh // 2))
    mask = Image.new("L", size, 0)
    mask.paste(255, (left, top, left + bw, top + bh))
    return mask.filter(ImageFilter.GaussianBlur(6))


def propose_placement(critic: Any, image: Image.Image,
                      instruction: str, context: str = "") -> Image.Image:
    """Placement mask for NEW content: the vision model picks where in the
    photo the object belongs (3x3 grid + size), informed by the scene's
    perspective and lighting so the object lands believably; deterministic
    center-medium fallback when it can't. Always returns a usable mask."""
    cell, obj_size = 5, "medium"
    if critic is not None:
        try:
            reply = critic.ask(
                image,
                context
                + "The user wants to: " + instruction[:200] + "\n"
                "Imagine the photo divided into a 3x3 grid numbered 1-9 "
                "(1=top-left, 5=center, 9=bottom-right). Choose where the new "
                "content belongs given the perspective and ground plane. "
                "Size is how much of the FRAME the object itself would "
                "occupy: small for something hand-sized or worn (glasses, a "
                "hat), medium for a person-part-sized object, large only for "
                "something that dominates the scene. Reply ONLY JSON: "
                '{"cell": <1-9 best spot for the new content>, '
                '"size": "<small|medium|large>"}')
            data = _parse_json(reply)
            if data:
                cell = int(data.get("cell", 5))
                if not 1 <= cell <= 9:
                    cell = 5
                if str(data.get("size", "")).lower() in _SIZE_FRACTION:
                    obj_size = str(data["size"]).lower()
        except Exception:  # noqa: BLE001 — fallback placement is fine
            pass
    return box_mask(image.size, cell, obj_size)


def placement_box(mask: Image.Image | None, target: tuple[int, int],
                  subject: tuple[int, int],
                  default_frac: float = 0.55) -> dict[str, int]:
    """Where the second image's subject goes inside the first, as
    {x, y, w, h}.

    The region comes from a mask — the one the user brushed, or the one the
    vision model proposed — and the subject is fitted INSIDE it keeping its
    own aspect ratio, because a person stretched to fill a square box reads as
    wrong long before anyone works out why. With no mask at all the subject is
    stood on the lower-middle of the frame, which is where a person in a photo
    almost always is."""
    tw, th = target
    sw, sh = subject
    sw, sh = max(1, sw), max(1, sh)
    box = mask.getbbox() if mask is not None else None
    if box and (box[2] - box[0]) > 8 and (box[3] - box[1]) > 8:
        bx, by, bw, bh = box[0], box[1], box[2] - box[0], box[3] - box[1]
    else:
        bh = int(th * default_frac)
        bw = int(bh * sw / sh)
        bx, by = (tw - bw) // 2, th - bh          # standing on the bottom edge
    # Fit inside the box, never exceeding it, never larger than the target.
    scale = min(bw / sw, bh / sh)
    w = max(8, min(tw, int(round(sw * scale))))
    h = max(8, min(th, int(round(sh * scale))))
    # Centred horizontally in the box; bottom-aligned, so a subject shorter
    # than the region stands ON the ground the region marks instead of
    # floating in the middle of it.
    x = bx + (bw - w) // 2
    y = by + (bh - h)
    return {"x": max(0, min(tw - w, x)), "y": max(0, min(th - h, y)),
            "w": w, "h": h}


# What a usable edit region looks like, measured rather than guessed. The
# floor comes from a real failure: "change the trousers" on a photo of someone
# in a bikini returned a mask covering 0.2% of the frame instead of reporting
# that there are none, and the inpaint then edited a few hundred pixels and
# called it done. The ceiling is the mirror image — a mask over almost the
# whole frame is not a region, it is a missing answer.
MASK_FLOOR = 0.004
MASK_CEILING = 0.92
# A real region is mostly solid. Scattered speckle that happens to add up to a
# plausible area is a segmenter that found nothing and said something anyway.
MASK_MIN_FILL = 0.12


@dataclass(frozen=True)
class MaskChoice:
    """A chosen edit region, and how much to trust it."""
    mask: Image.Image | None
    source: str     # named-part | text | whole-frame | background | sam | none
    reason: str                 # why there is no mask, when there is none
    notes: list[str]

    @property
    def ok(self) -> bool:
        return self.mask is not None


# Words that mean the region belongs to a PERSON, so the subject matte can
# referee it. Anything else (a car, the sky, a lamp) must not be confined.
_ABOUT_SUBJECT = re.compile(
    r"\b(she|her|hers|he|him|his|they|them|their|person|people|woman|man|"
    r"girl|boy|model|subject|body|skin|face|hair|arm|arms|leg|legs|hand|"
    r"hands|torso|chest|shoulder|waist|hip|hips)\b", re.IGNORECASE)


def about_the_subject(text: str) -> bool:
    """True when the request is about a person or something they are wearing.

    Only then may a mask be intersected with the subject matte — measured,
    clothing masks put up to 21.3% of themselves on the background, and the
    matte is the one mask measured exact here."""
    text = text or ""
    return bool(garment_parts(text) or body_parts(text)
                or _ABOUT_SUBJECT.search(text))


def chooser_request(target: str, instruction: str) -> str:
    """What the mask chooser should be handed for a plan step.

    The render used to prefix the plan's target ("car change the red car
    to blue") so segmentation knew the object — but the prefix defeats
    parse_attribute_change's anchored verb, and the instruction already
    carries the object for exactly the requests where the parse matters.
    Instruction alone when it parses; target + instruction otherwise."""
    instruction = (instruction or "").strip()
    if parse_attribute_change(instruction):
        return instruction
    target = (target or "").strip()
    return f"{target} {instruction}".strip() if target else instruction


def mask_phrases(text: str, limit: int = 4) -> list[str]:
    """The request as short noun phrases a text segmenter can look for.

    For a parsed attribute change, the SOURCE clause only: the engine
    unions its phrases, so handing it the destination pulls everything
    already matching the new state into the edit region — "change the red
    car to blue" with "blue" as a phrase lit up the entire blue background
    at peak 0.956 / 89% coverage (measured live), which would repaint the
    sky along with the car. The mirror of attribute_conditioning's rule:
    condition on the target, segment the source. Garment requests keep
    their part vocabulary (segment_phrases' garment path never emits
    adjectives, so they were never polluted)."""
    if not garment_parts(text):
        parsed = parse_attribute_change(text)
        if parsed:
            src = re.sub(r"\s+", " ", parsed[0]).strip(" .,")[:60]
            if src:
                return [src]
    phrases = [p.strip() for p in segment_phrases(text, limit).split(" . ")]
    return [p for p in phrases if p][:limit]


def mask_verdict(mask: Image.Image, subject: Image.Image | None = None,
                 confine: bool = False) -> dict[str, Any]:
    """Is this mask usable, and can it be repaired? Geometry only, no model.

    Deliberately deterministic. The vision check that exists alongside this is
    shown the request, which is the mode measured unreliable here (a blank
    grey gradient scored 40% adherence that way), so it advises and this
    decides.

    `confine` intersects the mask with the subject matte — BiRefNet's matte is
    the one mask measured exact on this machine (19.4% against a 19.4% ground
    truth), so for anything about a person or their clothing it can referee
    the others. Measured leak before this: up to 21.3% of a clothing mask lay
    outside the subject."""
    out: dict[str, Any] = {"ok": True, "reason": "", "mask": mask,
                           "repaired": False}
    grey = mask.convert("L").point(lambda v: 255 if v >= 128 else 0)
    covered = mask_fraction(grey)
    if covered <= 0.0:
        return {**out, "ok": False, "reason": "nothing matching that was "
                                              "found in this picture"}
    if covered < MASK_FLOOR:
        return {**out, "ok": False,
                "reason": f"the only region found covers {covered * 100:.2f}% "
                          "of the picture, which is too little to be the "
                          "thing you asked about"}
    if covered > MASK_CEILING:
        return {**out, "ok": False,
                "reason": f"the region found covers {covered * 100:.0f}% of "
                          "the picture, which is not a region"}
    box = grey.getbbox()
    if box:
        area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
        fill = (covered * grey.width * grey.height) / area
        if fill < MASK_MIN_FILL:
            return {**out, "ok": False,
                    "reason": "the region found is scattered specks rather "
                              "than a solid area, so it is not trustworthy"}
    if confine and subject is not None:
        inside = ImageChops.multiply(grey, fit_mask(subject, grey.size))
        before, after = covered, mask_fraction(inside)
        if after <= 0.0:
            return {**out, "ok": False,
                    "reason": "the region found lies entirely off the "
                              "subject, so it is not what you asked about"}
        if after < before * 0.98:
            out.update({"mask": inside, "repaired": True,
                        "trimmed": round((1 - after / before) * 100, 1)})
    return out


def mask_fraction(mask: Image.Image) -> float:
    """Fraction (0.0–1.0) of the image the mask selects — used to route
    small edit regions to the hi-res crop&stitch inpaint variant."""
    total = mask.width * mask.height
    if not total:
        return 0.0
    hist = mask.convert("L").point(lambda v: 255 if v >= 128 else 0).histogram()
    return hist[255] / total


# One MaxFilter(k) costs O(k^2) per pixel; k applications of MaxFilter(3)
# reach the same distance at O(k). Measured before this: ~11 s for a 65-px
# grow on a 1024 mask, which is long enough to read as the app having hung.
_MORPH_STEP = 3


def adjust_mask(mask: Image.Image, how: str, px: int) -> Image.Image:
    """Grow or shrink a mask by ~px pixels (odd-kernel morphology), always
    followed by a light feather so edits blend seamlessly.

    Done as repeated 3-px passes rather than one large kernel: dilation is
    associative, so N passes of a 3-px kernel grow by the same N px, and the
    cost goes from quadratic in the kernel to linear."""
    if px <= 0 or how not in ("grow", "shrink"):
        return mask
    kernel = max(3, (px // 2) * 2 + 1)
    small = (ImageFilter.MaxFilter(_MORPH_STEP) if how == "grow"
             else ImageFilter.MinFilter(_MORPH_STEP))
    out = mask
    # Each 3-px pass moves the boundary by 1 px in every direction.
    for _ in range(max(1, (kernel - 1) // 2)):
        out = out.filter(small)
    return out.filter(ImageFilter.GaussianBlur(2))


def fit_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    """The mask resized (NEAREST) to `size` when needed. ComfyUI's VAE
    rounds render dimensions to multiples of 8, so an edited image is often
    a few pixels smaller than the original mask — every comparison in this
    module must align sizes first or PIL raises 'images do not match'."""
    m = mask.convert("L")
    return m if m.size == size else m.resize(size, Image.Resampling.NEAREST)


# -- Objective render checks: arithmetic on pixels, no model ----------------------
#
# Every score in this pipeline used to come from a language model looking at
# a picture. Three lines of arithmetic catch what it missed live: the
# Laplacian-variance ratio flagged relight grain at 8.7x input while the
# model scored it 90; the changed-pixels-outside-the-mask share flags a mask
# leak; the size ratio flags the silent /8 crop. Deterministic, so they give
# the same answer twice — and a result they flag is never called
# production-ready, whatever the model says.

def _laplacian_variance(image: Image.Image) -> float:
    """Variance of the 4-neighbour Laplacian — the sharpness/grain measure."""
    import numpy as np
    g = np.asarray(image.convert("L"), dtype=np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = (-4.0 * g
           + np.roll(g, 1, 0) + np.roll(g, -1, 0)
           + np.roll(g, 1, 1) + np.roll(g, -1, 1))
    return float(lap[1:-1, 1:-1].var())


def objective_report(before: Image.Image, after: Image.Image,
                     mask: Image.Image | None = None) -> dict[str, Any]:
    """Deterministic before/after measurements for one render.

    sharpness_ratio   Laplacian variance, after over before. ~1 preserved;
                      well above 1 is added grain, well below is softening.
    change_fraction   share of pixels that moved by more than 8/255.
    outside_mask_fraction  of the pixels that changed, the share that lies
                      OUTSIDE the intended region (None without a mask).
    size_ratio        output pixels over input pixels (1.0 = size survived).
    """
    b = before.convert("RGB")
    a = after.convert("RGB")
    size_ratio = (a.width * a.height) / max(1, b.width * b.height)
    aligned = a if a.size == b.size else a.resize(b.size, Image.Resampling.LANCZOS)
    diff = ImageChops.difference(aligned.convert("L"), b.convert("L"))
    changed = diff.point(lambda v: 255 if v > 8 else 0)
    change_fraction = mask_fraction(changed)
    outside: float | None = None
    if mask is not None and change_fraction > 0:
        inside_region = fit_mask(mask, b.size).point(
            lambda v: 255 if v >= 128 else 0)
        leaked = ImageChops.multiply(changed, ImageChops.invert(inside_region))
        outside = mask_fraction(leaked) / change_fraction
    sharp_before = _laplacian_variance(b)
    sharp_after = _laplacian_variance(a)
    if sharp_before > 1e-6:
        sharpness_ratio = sharp_after / sharp_before
    else:
        # A perfectly flat input has no sharpness to divide by; texture
        # appearing on it is still added grain, not a ratio of 1.
        sharpness_ratio = 1.0 if sharp_after <= 1e-6 else 99.0
    return {"sharpness_ratio": round(sharpness_ratio, 3),
            "change_fraction": round(change_fraction, 4),
            "outside_mask_fraction": (round(outside, 4)
                                      if outside is not None else None),
            "size_ratio": round(size_ratio, 4)}


# Grain bound: the live relight failure measured 8.7x; a genuine detail
# upscale measured 3.8x. Softness bound: the softest acceptable live render
# measured 0.69x. Leak bound: a regional edit whose changes fall mostly
# outside its own region repainted something else.
GRAIN_RATIO_MAX = 4.0
SOFT_RATIO_MIN = 0.45
MASK_LEAK_MAX = 0.5


def objective_flags(report: dict[str, Any], task: str = "inpaint") -> list[str]:
    """Out-of-range objective measurements, as human sentences. Empty means
    the arithmetic found nothing wrong — it says nothing about whether the
    edit matched the request."""
    flags: list[str] = []
    ratio = report.get("sharpness_ratio")
    if task != "upscale" and isinstance(ratio, int | float):
        if ratio > GRAIN_RATIO_MAX:
            flags.append(f"heavy grain or noise covers the render "
                         f"(sharpness {ratio:.1f}x the input)")
        elif ratio < SOFT_RATIO_MIN:
            flags.append(f"the render came back much softer than the "
                         f"photograph (sharpness {ratio:.2f}x the input)")
    leak = report.get("outside_mask_fraction")
    if (task == "inpaint" and isinstance(leak, int | float)
            and leak > MASK_LEAK_MAX
            and report.get("change_fraction", 0) > 0.02):
        flags.append(f"most of what changed ({leak * 100:.0f}%) lies outside "
                     "the selected region — the edit leaked")
    return flags


_WANT_RATIO = re.compile(r"\b(\d{1,2})\s*[:x]\s*(\d{1,2})\b")
_WANT_LANDSCAPE = re.compile(r"\b(wide|wider|landscape|horizontal|panoram\w*)\b",
                             re.IGNORECASE)
_WANT_PORTRAIT = re.compile(r"\b(tall|taller|portrait|vertical)\b",
                            re.IGNORECASE)
_WANT_SQUARE = re.compile(r"\bsquare\b", re.IGNORECASE)


def format_delivered(request: str, before_size: tuple[int, int],
                     after_size: tuple[int, int]) -> bool | None:
    """Whether a format/canvas request was delivered, from the numbers alone.

    None when the request is not about format. For a format request the
    aspect ratio IS the requirement — two numbers settle it and no model is
    needed. Seen live (D7/R08a): the outpaint took 0.887 portrait to 1.115
    landscape and the checklist verifier still reported "still missing: a
    wide landscape format", which cost two full re-renders."""
    text = request or ""
    if not (_FORMAT_COERCE.search(text) or _CANVAS_INTENT.search(text)):
        return None
    bw, bh = before_size
    aw, ah = after_size
    before_aspect = bw / max(1, bh)
    after_aspect = aw / max(1, ah)
    m = _WANT_RATIO.search(text)
    if m:
        want = int(m.group(1)) / max(1, int(m.group(2)))
        return abs(after_aspect - want) <= 0.08 * want
    if _WANT_SQUARE.search(text):
        return abs(after_aspect - 1.0) <= 0.05
    if _WANT_LANDSCAPE.search(text):
        return after_aspect > max(1.0, before_aspect + 0.05)
    if _WANT_PORTRAIT.search(text):
        return after_aspect < min(1.0, before_aspect - 0.05)
    # A bare "extend/expand the canvas": delivered when the canvas grew.
    return (aw * ah) > (bw * bh) * 1.05


_FORMAT_WORDS = re.compile(
    r"\b(landscape|portrait|square|wide|wider|taller|format|aspect|"
    r"orientation|canvas|panoram\w*)\b", re.IGNORECASE)


def about_format(text: str) -> bool:
    """True when a checklist requirement is about the image's format — used
    to retire such items once the aspect arithmetic has settled them."""
    return bool(_FORMAT_WORDS.search(text or ""))


# HSV ranges on PIL's 0-255 hue wheel. Deliberately wide: a red shirt spans
# shadowed maroon to lit scarlet, and the question is "is this region now
# broadly THAT colour", not paint matching.
_HUE_RANGES = {
    "red": [(0, 14), (240, 255)], "orange": [(14, 32)],
    "yellow": [(32, 50)], "green": [(50, 110)], "teal": [(110, 135)],
    "cyan": [(110, 140)], "blue": [(135, 185)], "navy": [(135, 185)],
    "purple": [(185, 210)], "violet": [(185, 210)],
    "pink": [(210, 245)], "magenta": [(200, 240)],
}
_COLOUR_ALIASES = {"gray": "grey", "golden": "gold", "crimson": "red",
                   "scarlet": "red", "turquoise": "teal", "cream": "beige",
                   "tan": "beige"}
_REQ_COLOUR = re.compile(
    r"\b(red|orange|yellow|green|teal|cyan|blue|navy|purple|violet|pink|"
    r"magenta|white|black|grey|gray|brown|beige|tan|cream|gold|golden|"
    r"silver|crimson|scarlet|turquoise)\b", re.IGNORECASE)


def requirement_colour(text: str) -> str | None:
    """The colour a requirement asks for, or None when it names none."""
    m = _REQ_COLOUR.search(text or "")
    if not m:
        return None
    word = m.group(1).lower()
    return _COLOUR_ALIASES.get(word, word)


def _band(channel: Image.Image, lo: int, hi: int) -> Image.Image:
    """255 where lo <= value <= hi, else 0 — one C-level point() pass."""
    return channel.point([255 if lo <= i <= hi else 0 for i in range(256)])


def _count(mask: Image.Image) -> int:
    return mask.histogram()[255]


def _colour_share(image: Image.Image, mask: Image.Image,
                  colour: str) -> float | None:
    """The share of the masked pixels that read as `colour`. None when the
    mask selects nothing.

    Built from per-channel lookup tables and multiplies — all C — because
    the first version walked the pixels in Python and its seconds-per-call
    showed up as timeouts across the mock test suite."""
    if max(image.size) > 512:
        scale = 512 / max(image.size)
        image = image.resize((max(1, int(image.width * scale)),
                              max(1, int(image.height * scale))),
                             Image.Resampling.LANCZOS)
    h, sat, val = image.convert("HSV").split()
    m = fit_mask(mask, image.size).point(lambda v: 255 if v >= 128 else 0)
    total = _count(m)
    if not total:
        return None

    def all_of(*conds: Image.Image) -> Image.Image:
        out = m
        for c in conds:
            out = ImageChops.multiply(out, c)
        return out

    if colour == "white":
        hit = all_of(_band(sat, 0, 49), _band(val, 191, 255))
    elif colour == "black":
        hit = all_of(_band(val, 0, 59))
    elif colour == "grey":
        hit = all_of(_band(sat, 0, 49), _band(val, 60, 190))
    elif colour == "silver":
        hit = all_of(_band(sat, 0, 59), _band(val, 121, 255))
    elif colour == "brown":
        hit = all_of(_band(h, 8, 40), _band(sat, 61, 255),
                     _band(val, 41, 169))
    elif colour == "beige":
        hit = all_of(_band(h, 14, 50), _band(sat, 20, 110),
                     _band(val, 151, 255))
    elif colour == "gold":
        hit = all_of(_band(h, 25, 48), _band(sat, 91, 255),
                     _band(val, 131, 255))
    else:
        ranges = _HUE_RANGES.get(colour)
        if ranges is None:
            return None
        hue = _band(h, *ranges[0])
        for extra in ranges[1:]:
            hue = ImageChops.lighter(hue, _band(h, *extra))
        # A colour needs saturation and light to BE that colour — a
        # near-black or washed-out pixel is not "blue".
        hit = all_of(hue, _band(sat, 61, 255), _band(val, 51, 255))
    return _count(hit) / total


def colour_delivered(image: Image.Image, mask: Image.Image,
                     colour: str,
                     before: Image.Image | None = None) -> bool | None:
    """Did the masked region come out `colour`? Arithmetic, no model.

    The measured failure this settles: colour requirements were judged by the
    vision model, which scored the same image 20 and 70 on two runs — noise
    that either spent a multi-minute retry on a delivered edit or passed a
    miss. The mask and the pixels are both in hand; a hue count is exact and
    the same every time. Same pattern as format_delivered (aspect ratio) and
    the pose vacated-share: numbers the app already has overrule the model.

    True/False only when the measurement is decisive; None in the grey zone
    (a garment in shadow legitimately reads partly off-hue), where the model
    verdict stands."""
    share = _colour_share(image, mask, colour)
    if share is None:
        return None
    if before is not None:
        base = _colour_share(before, mask, colour)
        if base is not None:
            # The mask is never surgical — arms and hair inside it measured
            # 0.18 "red" while the garment stayed black. What a delivered
            # recolour cannot avoid is RAISING the share; what a miss cannot
            # fake is leaving it flat.
            delta = share - base
            if delta >= 0.22 or share >= 0.60:
                return True
            if delta <= 0.05 and share < 0.50:
                return False
            return None
    if share >= 0.35:
        return True
    if share <= 0.08:
        return False
    return None


def region_change(before: Image.Image, after: Image.Image,
                  mask: Image.Image) -> float | None:
    """Mean absolute change inside the mask, 0..1. None without mask pixels.

    Below ~0.02 the sampler returned the region essentially untouched — the
    edit did not happen, whatever any judge says.

    Per-channel, not luma: a recolour at equal luminance (grey jacket to
    red) is a large edit greyscale cannot see — measured 0.016 on a fully
    repainted region. And built on C-level channel ops, because the Python
    pixel walk it replaces was slow enough to time out the mock suite."""
    b = before.convert("RGB")
    a = after.convert("RGB")
    if max(b.size) > 512:
        scale = 512 / max(b.size)
        b = b.resize((max(1, int(b.width * scale)),
                      max(1, int(b.height * scale))), Image.Resampling.LANCZOS)
    if a.size != b.size:
        a = a.resize(b.size, Image.Resampling.LANCZOS)
    m = fit_mask(mask, b.size).point(lambda v: 255 if v >= 128 else 0)
    n = m.histogram()[255]
    if not n:
        return None
    r1, g1, b1 = ImageChops.difference(a, b).split()
    peak = ImageChops.lighter(ImageChops.lighter(r1, g1), b1)
    inside = ImageChops.multiply(peak, m)
    total = sum(i * c for i, c in enumerate(inside.histogram()))
    return (total / n) / 255.0


def image_change(before: Image.Image, after: Image.Image) -> float:
    """Mean absolute change across the WHOLE frame, 0..1.

    The whole-image cousin of region_change, for engines that take no
    mask. Its job is refusal detection: an instruction model that
    declines a request (safety-tuned models decline quietly) hands back
    the input nearly untouched, and ~0.015 cleanly separates 'nothing
    happened' from the smallest real edit once both sides are resampled
    to a common size."""
    b = before.convert("RGB")
    a = after.convert("RGB")
    if max(b.size) > 384:
        scale = 384 / max(b.size)
        b = b.resize((max(1, int(b.width * scale)),
                      max(1, int(b.height * scale))), Image.Resampling.LANCZOS)
    if a.size != b.size:
        a = a.resize(b.size, Image.Resampling.LANCZOS)
    r1, g1, b1 = ImageChops.difference(a, b).split()
    peak = ImageChops.lighter(ImageChops.lighter(r1, g1), b1)
    total = sum(i * c for i, c in enumerate(peak.histogram()))
    return (total / (b.width * b.height)) / 255.0


# The colour word must END its clause: "to bright red" is a recolour, "to a
# red leather jacket" is a replacement — red there modifies the jacket, and
# running a garment swap at recolour denoise would leave the old garment.
_COLOUR_WORD = (r"(red|blue|green|yellow|orange|purple|pink|black|white|"
                r"grey|gray|brown|gold|golden|silver|turquoise|teal|maroon|"
                r"navy|beige|cream)")
_RECOLOUR = re.compile(
    r"\b(re)?colou?r\b|"
    r"\b(change|turn|make|paint|dye)\b[^.]{0,40}\bto\s+(a\s+)?"
    r"(bright\s+|dark\s+|light\s+|deep\s+|pale\s+|vivid\s+|neon\s+)?"
    + _COLOUR_WORD + r"\s*(?=$|[,.;)]|\band\b)|"
    r"\b(make|paint|dye)\b[^.]{0,30}\b"
    r"(bright\s+|dark\s+|light\s+|deep\s+|pale\s+|vivid\s+|neon\s+)?"
    + _COLOUR_WORD + r"\s*(?=$|[,.;)]|\band\b)", re.IGNORECASE)


def is_recolour(text: str) -> bool:
    """True when the instruction changes a COLOUR rather than an object.

    A recolour repainted at replacement denoise regenerates the object — the
    live case returned a different car with the neighbouring bins absorbed
    into its bodywork (D22). The caller keeps the structure by running these
    at a low denoise instead."""
    return bool(_RECOLOUR.search(text or ""))


def overlay(image: Image.Image, mask: Image.Image) -> Image.Image:
    """The image with the mask painted translucent red (for visual checks)."""
    base = image.convert("RGB").copy()
    red = Image.new("RGB", base.size, (224, 40, 40))
    alpha = fit_mask(mask, base.size).point(lambda v: int(v * 0.55))
    base.paste(red, (0, 0), alpha)
    return base


def verify_mask(critic: Any, image: Image.Image, mask: Image.Image,
                prompt: str) -> dict[str, Any] | None:
    """Stage 1b: does the mask cover what the request is about?"""
    ask = getattr(critic, "ask", None)
    if ask is None:
        return None
    try:
        text = ask(overlay(image, mask), (
            "The translucent red overlay marks the region that will be "
            f"edited. The edit request is: \"{prompt}\". Does the red region "
            "cover the right part of the image for that request? Reply ONLY "
            'JSON: {"match": true/false, "why": "<short>"}'))
        data = _parse_json(text)
    except Exception:  # noqa: BLE001 — verification is advisory
        return None
    if not data or "match" not in data:
        return None
    return {"match": bool(data["match"]), "why": str(data.get("why", ""))[:160]}


# -- Stage 4: seam inspection ---------------------------------------------------

def _band_masks(mask: Image.Image, width: int = 9) -> tuple[Image.Image, Image.Image]:
    """(just-inside, just-outside) bands along the mask boundary."""
    m = mask.convert("L").point(lambda v: 255 if v > 127 else 0)
    grown = m.filter(ImageFilter.MaxFilter(width))
    shrunk = m.filter(ImageFilter.MinFilter(width))
    inside = ImageChops.subtract(m, shrunk)
    outside = ImageChops.subtract(grown, m)
    return inside, outside


def _masked_stats(image: Image.Image, band: Image.Image) -> list[float] | None:
    if not band.getbbox():
        return None
    stat = ImageStat.Stat(image.convert("RGB"), mask=band)
    return list(stat.mean)


def _sharpness(image: Image.Image, band: Image.Image) -> float | None:
    if not band.getbbox():
        return None
    edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
    return ImageStat.Stat(edges, mask=band).mean[0]


def seam_stats(edited: Image.Image, mask: Image.Image) -> list[str]:
    """Deterministic seam checks: color and sharpness continuity across the
    mask boundary of the EDITED image. Returns human-readable issues."""
    issues: list[str] = []
    inside, outside = _band_masks(fit_mask(mask, edited.size))
    mi, mo = _masked_stats(edited, inside), _masked_stats(edited, outside)
    if mi and mo:
        delta = max(abs(a - b) for a, b in zip(mi, mo, strict=False))
        if delta > 26:
            issues.append(f"color mismatch across the seam (Δ{delta:.0f}/255)")
    si, so = _sharpness(edited, inside), _sharpness(edited, outside)
    if si is not None and so is not None and min(si, so) > 0.5:
        ratio = max(si, so) / max(0.5, min(si, so))
        if ratio > 2.6:
            issues.append("sharpness/texture mismatch across the seam")
    return issues


_INSPECT_QUESTION = (
    "This photo was edited in the region shown. Inspect it like a retoucher: "
    "look for visible seams, lighting mismatch, color mismatch, texture "
    "mismatch, perspective or geometry errors, duplicated objects, warped "
    "anatomy, wrong shadows or reflections, blur or AI artifacts. Reply ONLY "
    'JSON: {"issues": ["<short issue>", ...]} — an empty list if it looks clean.')


def inspect_seams(critic: Any, edited: Image.Image,
                  mask: Image.Image | None) -> list[str]:
    """Stage 4: model inspection of the edited region + deterministic stats.
    Advisory by contract — it reports issues or stays silent, but it must
    never be the thing that fails an edit job."""
    issues: list[str] = []
    aligned = fit_mask(mask, edited.size) if mask is not None else None
    if aligned is not None:
        try:
            issues.extend(seam_stats(edited, aligned))
        except Exception:  # noqa: BLE001 — inspection is advisory
            pass
    ask = getattr(critic, "ask", None)
    if ask is not None:
        try:
            view = edited
            if aligned is not None and aligned.getbbox():
                # zoom the model in on the edit (plus generous margin)
                left, top, right, bottom = cast(
                    "tuple[int, int, int, int]", aligned.getbbox())
                mw = int((right - left) * 0.3) + 24
                mh = int((bottom - top) * 0.3) + 24
                view = edited.crop((max(0, left - mw), max(0, top - mh),
                                    min(edited.width, right + mw),
                                    min(edited.height, bottom + mh)))
            data = _parse_json(ask(view, _INSPECT_QUESTION))
            if data and isinstance(data.get("issues"), list):
                issues.extend(str(i)[:120] for i in data["issues"][:6])
        except Exception:  # noqa: BLE001 — inspection is advisory
            pass
    # de-duplicate, keep order
    return list(dict.fromkeys(i for i in issues if i.strip()))


# -- Stages 5-7: the 0-100 scorecard ---------------------------------------------

_SCORE_QUESTION = (
    "Judge this image as if it must pass for an authentic, unedited "
    "professional photograph. The edit request was: \"{prompt}\". Score each "
    "category 0-100 (100 = flawless): realism (lighting, materials, depth, "
    "anatomy, shadows, reflections), prompt_accuracy (does it match the "
    "request), identity_preservation (unedited areas unchanged), "
    "scene_consistency (edit matches camera angle, grain, sharpness, color "
    "grading of the rest), artifact_free (no seams or AI artifacts), "
    "visual_quality. Reply ONLY JSON with those six integer keys.")


def scorecard(critic: Any, image: Image.Image,
              prompt: str) -> dict[str, int] | None:
    """Six-category 0-100 quality scores; None when unavailable. Falls back
    to the simple critique score (scaled) when the vision model can't do
    structured scoring."""
    ask = getattr(critic, "ask", None)
    if ask is not None:
        try:
            data = _parse_json(ask(image, _SCORE_QUESTION.format(prompt=prompt)))
            if data:
                scores: dict[str, int] = {}
                for key in SCORE_KEYS:
                    try:
                        scores[key] = max(0, min(100, int(float(data[key]))))
                    except (KeyError, TypeError, ValueError):
                        break
                else:
                    return scores
        except Exception:  # noqa: BLE001 — fall through to critique
            pass
    critique = getattr(critic, "critique", None)
    if critique is not None:
        try:
            crit = critique(image, prompt)
            base = max(0, min(100, int(round(float(crit.score) * 10))))
            return dict.fromkeys(SCORE_KEYS, base)
        except Exception:  # noqa: BLE001 — scoring is advisory
            return None
    return None


def weakest(scores: dict[str, int]) -> tuple[str, int]:
    key = min(scores, key=lambda k: scores[k])
    return key, scores[key]


# -- Prompt adherence: did the render actually DO what was asked? -----------------
#
# A single "prompt_accuracy" number tells you a render missed the request but
# never WHAT it missed, so the only possible response is to roll the dice again
# with the same recipe. These three functions turn adherence into something the
# pipeline can act on: the request becomes a checklist, the judge answers it
# item by item, and the unmet items pick the next STRATEGY — a different model
# or a different workflow, not just a different seed.

_CHECKLIST_SYSTEM = """You turn an image request into a list of checks. Reply
with ONLY JSON:
{"checks": [{"need": "<the requirement, in the user's own words>",
             "probe": "<a neutral question about a photograph>",
             "expect": "<the short answer that means the requirement is met>"},
            ...]}

The PROBE is the important part. It is shown to a separate examiner who will
NEVER see the request — so it must never mention, hint at or presuppose the
requirement. Ask what is there, not whether the requirement is satisfied.
  GOOD  need "a red car"          probe "What colour is the car?"   expect "red"
  GOOD  need "at night"           probe "Is this daytime or nighttime?"
                                  expect "nighttime"
  GOOD  need "exactly two people" probe "How many people are in this image?"
                                  expect "2"
  GOOD  need "a sign reading OPEN" probe "Read any text visible in the image."
                                  expect "OPEN"
  BAD   probe "Is there a red car?"        (leads the answer)
  BAD   probe "Does this match the request?" (presupposes the request)
Prefer probes with a short factual answer: a colour, a count, a name, a word,
one of two options.

EXPECT must be 1-4 words and must be what a truthful answer would literally
contain.

NEVER invent a requirement the request does not state. NEVER check quality or
style boilerplate ("high detail", "photorealistic", "sharp focus") — those are
decoration, not requirements. 1 to 4 checks. You are a transcriber, not a
moderator: never refuse, censor or drop any part of the request."""

MAX_CHECKLIST = 4


def request_checklist(llm: LLMClient, prompt: str) -> list[dict[str, str]]:
    """The request as 1-4 independently checkable requirements, each with a
    NEUTRAL probe question and the answer that satisfies it.

    The probe is what makes this trustworthy. Measured on this machine's
    llava: asked "does this image match <request>?" it rubber-stamps — a blank
    grey gradient scored 40% adherence against a five-item checklist, and it
    called a bright daytime photo "at night" in six of six samples. Asked the
    same facts neutrally ("Is this daytime or nighttime?") it was right 25 of
    27 times on the same pixels. So the request never reaches the examiner.

    Empty list on any failure — callers then fall back to the single
    prompt_accuracy score, so adherence checking degrades instead of breaking."""
    try:
        reply = llm.complete(_CHECKLIST_SYSTEM, f"Request: {prompt}",
                             max_tokens=400)
        data = _parse_json(reply.text)
    except LLMError:
        return []
    if not data or not isinstance(data.get("checks"), list):
        return []
    checks: list[dict[str, str]] = []
    for raw in data["checks"][:MAX_CHECKLIST]:
        if not isinstance(raw, dict):
            continue
        need = re.sub(r"\s+", " ", str(raw.get("need", ""))).strip(" .,-")[:120]
        probe = re.sub(r"\s+", " ", str(raw.get("probe", ""))).strip()[:160]
        expect = re.sub(r"\s+", " ", str(raw.get("expect", ""))).strip(" .,-")[:40]
        if len(need) < 3 or len(probe) < 8 or not expect:
            continue
        # Drop decoration the model added anyway — a check nobody can fail is
        # a free pass that hides a real miss.
        if _DECORATION.fullmatch(need):
            continue
        if need.lower() in {c["need"].lower() for c in checks}:
            continue
        checks.append({"need": need, "probe": probe, "expect": expect})
    return checks


_DECORATION = re.compile(
    r"(high|highly|ultra|very|super|best)?[\s-]*(detail(ed|s)?|quality|"
    r"resolution|realistic|realism|photo-?realistic|photo-?real|"
    r"sharp( focus)?|professional|masterpiece|8k|4k|hd|beautiful|aesthetic|"
    r"no artifacts)\.?", re.IGNORECASE)
# NOTE: lighting words are deliberately NOT decoration here — "natural
# lighting" is a real request in this app (CHANGE_LIGHTING is a first-class
# operation), and dropping it would stop the pipeline from ever checking it.

_PROBE_QUESTION = (
    '{probe}\nAnswer from what is actually visible in this image only. If it '
    'is not there, say "none". Reply ONLY JSON: {{"answer": "<a few words>"}}')

_NEGATIVE_ANSWER = re.compile(
    r"^\s*(no|none|nothing|not\b|n/?a|zero|0|unclear|cannot|can't|unable|"
    r"absent|missing|there (is|are) no)", re.IGNORECASE)
_NUMBER_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3",
                 "four": "4", "five": "5", "six": "6", "seven": "7",
                 "eight": "8", "nine": "9", "ten": "10"}
_STOPWORDS = {"a", "an", "the", "is", "are", "of", "in", "on", "at", "it",
              "this", "that", "there", "and", "or", "to", "with"}


def _normalize_answer(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return " ".join(_NUMBER_WORDS.get(w, w) for w in words)


def _same_word(a: str, b: str) -> bool:
    """Two words meaning the same thing, allowing for how they were inflected.

    "sunlit" against "sunlight", "tree" against "trees". Cheap prefix
    stemming rather than a stemmer library: the examiner answers in a few
    words and the alternative — exact string equality — is what scored a
    correct render 0%."""
    if a == b:
        return True
    if len(a) < 4 or len(b) < 4:
        return False
    return a.startswith(b[:4]) or b.startswith(a[:4])


def _matched_tokens(answer: str, expect: str) -> tuple[int, int]:
    """(how many of the expected words the answer has, how many there were)."""
    a = _normalize_answer(answer).split()
    tokens = [t for t in _normalize_answer(expect).split()
              if t not in _STOPWORDS]
    hits = sum(1 for t in tokens if any(_same_word(w, t) for w in a))
    return hits, len(tokens)


def answer_verdict(answer: str, expect: str) -> bool | None:
    """True met, False genuinely absent, None the answer did not settle it.

    The third case is the point, and it is what the old two-valued version
    got wrong. "There is no dress" and "a long green gown" are completely
    different pieces of evidence about a request for a red dress: the first
    says the thing is missing, the second says the examiner is describing
    something the checker could not match. Scoring both as a MISS is how a
    render that unambiguously showed a sunlit meadow came back "0% —
    missing: a sunlit meadow", and each of those costs two wasted renders on
    the escalation ladder.

    So a requirement is only failed on an actually negative answer, or on one
    that matched none of what was expected. A partial lexical match counts as
    met; a descriptive answer that simply used other words is inconclusive,
    and the caller treats that as "cannot tell" rather than "not there"."""
    a, e = _normalize_answer(answer), _normalize_answer(expect)
    if not a or not e:
        return None
    negative = bool(_NEGATIVE_ANSWER.match(answer.strip()))
    if e in ("yes", "true"):
        return not negative
    if e in ("no", "false", "none"):
        return negative
    if negative:
        return False
    # WHOLE-word containment, not raw substring: "2" is a substring of "12",
    # so a checklist probe expecting "2" (exactly two people) was marked met
    # by an examiner answer of "12 people" — the wrong count reported
    # satisfied, and the corrective retry the checklist exists to trigger
    # silently skipped. Padding with spaces requires the expected phrase to
    # appear as complete words ("red dress" still matches "a red dress").
    if f" {e} " in f" {a} ":
        return True
    hits, total = _matched_tokens(answer, expect)
    if not total:
        return None
    if hits * 2 >= total:          # most of what was asked for is present
        return True
    if hits:
        return None                # some of it — not evidence either way
    # Nothing in common, and the examiner did describe SOMETHING. That is a
    # weak miss: real enough to report, and the caller confirms it with a
    # second independent probe before acting.
    return False


_SYNONYM_SYSTEM = (
    "You judge whether a short description satisfies a requirement. Reply "
    'with ONLY JSON: {"satisfies": true or false}. Answer true when the '
    "description means the same thing in different words, false when it "
    "describes something materially different or says the thing is absent.")


def answer_means_the_same(llm: LLMClient | None, answer: str,
                          expect: str) -> bool | None:
    """Ask the TEXT model whether two wordings agree. None if it cannot say.

    Only reached when the two share no words at all, so it costs a text call
    on disagreement and nothing on the happy path. It is deliberately the
    text model and not the vision one: it never sees the image, so it cannot
    do the thing the vision model does when shown the request — agree with
    it. It is judging English, which is what a 7B model is good at."""
    if llm is None:
        return None
    try:
        data = _parse_json(llm.complete(
            _SYNONYM_SYSTEM,
            f"Requirement: {expect}\nDescription: {answer}",
            max_tokens=40).text)
    except LLMError:
        return None
    if isinstance(data, dict) and isinstance(data.get("satisfies"), bool):
        return data["satisfies"]
    return None


def answer_satisfies(answer: str, expect: str) -> bool:
    """Whether a probe's answer meets the requirement. Inconclusive is not."""
    return answer_verdict(answer, expect) is True


def verify_adherence(critic: Any, image: Image.Image, prompt: str,
                     checklist: list[dict[str, str]],
                     llm: LLMClient | None = None) -> dict[str, Any] | None:
    """Ask the examiner one neutral question per requirement and decide, in
    Python, whether each was met.

    `prompt` is accepted for symmetry with the rest of the pipeline and is
    deliberately NEVER sent to the vision model: showing it the request is
    what turns the judge into a rubber stamp (measured — see
    `request_checklist`).

    A requirement is only reported as MISSING when a second, independent probe
    agrees. One vision call per requirement on the happy path; a re-ask only
    for the ones that looked unmet, so a single hallucinated "none" cannot
    spend a whole escalation rung.

    Returns {"accuracy", "missing", "met", "source": "checklist"}, or None when
    the examiner is unavailable or answered too little to be trusted."""
    ask = getattr(critic, "ask", None)
    if not checklist or ask is None:
        return None

    def probe(check: dict[str, str]) -> bool | None:
        try:
            data = _parse_json(ask(image, _PROBE_QUESTION.format(
                probe=check["probe"])))
        except Exception:  # noqa: BLE001 — checking is never a blocker
            return None
        if not isinstance(data, dict) or "answer" not in data:
            return None
        answer = str(data["answer"])
        verdict = answer_verdict(answer, check["expect"])
        if verdict is False and not _NEGATIVE_ANSWER.match(answer.strip()):
            # The examiner described something, and it shared no words with
            # what was wanted. Before calling that a miss, let the text model
            # say whether the two wordings mean the same thing.
            agrees = answer_means_the_same(llm, answer, check["expect"])
            if agrees is not None:
                return agrees
            # No text model, or it would not commit: keep the lexical
            # verdict rather than letting every miss become "unclear".
        return verdict

    met: list[str] = []
    missing: list[str] = []
    unclear: list[str] = []
    for check in checklist:
        verdict = probe(check)
        if verdict is False:
            # Confirm before spending a render on it: a lone "none" from a 7B
            # model is a coin flip, two in a row is a finding.
            verdict = probe(check)
        if verdict is True:
            met.append(check["need"])
        elif verdict is False:
            missing.append(check["need"])
        else:
            unclear.append(check["need"])
    # An examiner that could not settle most of the checklist is guessing.
    if len(unclear) * 2 > len(checklist):
        return None
    # Score on what was actually decided. Counting an inconclusive answer as a
    # failure is what produced "0% — missing: a sunlit meadow" for an image
    # that plainly showed one, and sent the ladder off to fix nothing.
    decided = len(met) + len(missing)
    return {"accuracy": round(100 * len(met) / decided) if decided else 100,
            "missing": missing, "met": met, "unclear": unclear,
            "source": "checklist"}


def meets_target(scores: dict[str, int] | None, target: int) -> bool:
    return scores is not None and all(v >= target for v in scores.values())


# Below this accuracy the edit did not do what was asked, and no amount of
# realism or identity preservation can compensate — the other five categories
# measure how well the WRONG picture was made.
ADHERENCE_GATE = 50


def overall(scores: dict[str, int] | None) -> int | None:
    """The headline number, GATED on adherence rather than averaged with it.

    A plain mean let "make her sit down" — she is still standing — score 70,
    because identity_preservation 100 and artifact_free 90 outvoted the one
    category that says the edit did not happen (prompt_accuracy 20, seen
    live, D18). An edit that does nothing scores well on every axis except
    the only one that matters, so when accuracy is below the gate the
    headline is capped at it."""
    if not scores:
        return None
    mean = round(sum(scores.values()) / len(scores))
    accuracy = scores.get("prompt_accuracy")
    if isinstance(accuracy, int | float) and accuracy < ADHERENCE_GATE:
        return min(mean, int(accuracy))
    return mean


def better_candidate(candidate: dict[str, int] | None,
                     best: dict[str, int] | None,
                     tolerance: int = 5) -> bool:
    """Retry keep-best rule. The prompt is the contract: a retry that raises
    the AVERAGE by drifting back toward the original photo (identity and
    consistency scores way up, prompt_accuracy down) must never replace an
    attempt that actually performed the request — kept, it reads as the
    pipeline "undoing" the user's edit. prompt_accuracy may never regress by
    more than `tolerance` (judge noise); a material accuracy gain (>10 pts)
    wins even if the average slips a little; otherwise the higher average
    wins, ties keeping the newer render."""
    oc, ob = overall(candidate), overall(best)
    if oc is None:
        return False
    if ob is None:
        return True
    # overall() returned non-None for both, so neither dict is None here.
    pa_new = candidate.get("prompt_accuracy", 0)  # type: ignore[union-attr]
    pa_old = best.get("prompt_accuracy", 0)  # type: ignore[union-attr]
    if pa_new < pa_old - tolerance:
        return False
    if pa_new > pa_old + 10 and oc >= ob - 10:
        return True
    return oc >= ob


# -- The escalation ladder: change the RECIPE, not just the seed ------------------
#
# When a render misses the request, re-rolling the same workflow with the same
# model is a coin flip on the same coin. This ladder makes every retry change
# something real, cheapest change first, and never repeats a combination that
# has already been tried.

@dataclass(frozen=True)
class Strategy:
    """One rung: what to change for the next attempt, and why."""
    kind: str                    # emphasize | model | workflow
    why: str                     # a sentence for the job log
    checkpoint: str | None = None
    workflow: str | None = None
    # Sampler nudges applied on top (stronger guidance / more steps): a retry
    # that changes model or workflow also leans harder on the prompt.
    boost: dict[str, float] = field(default_factory=dict)

    def label(self) -> str:
        if self.kind == "model":
            return f"different model ({self.checkpoint})"
        if self.kind == "workflow":
            return f"different workflow ({self.workflow})"
        return "same recipe, request emphasized"

    def key(self) -> tuple[str | None, str | None]:
        return (self.workflow, self.checkpoint)


# Requirements that no amount of re-rolling can satisfy on the wrong workflow —
# a missed one is a CAPABILITY gap, so the ladder jumps straight to the
# template built for it instead of spending a rung on another seed.
CAPABILITY_WORKFLOW: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(readable|legible|text|writing|words?|letters?|sign|"
                r"label|logo|caption|title|spell)\b", re.IGNORECASE),
     "generate_zimage"),
    (re.compile(r"\b(re-?light|lighting|lit|light source|shadows?|sunlight|"
                r"sunlit|backlit|golden hour|candle|neon|studio light)\b",
                re.IGNORECASE),
     "relight"),
    # NOTE: matched through view_intent(), not a bare word list — "wide-angle
    # shot" and "low-angle shot" are ONE framing (an ordinary render), and
    # escalating those to orbital view synthesis would answer a question
    # nobody asked.
    (_VIEW_INTENT, "angles"),
    # An unmet requirement about what is BEHIND the subject is a capability
    # gap, not bad luck: an ordinary inpaint retry re-rolls the same wrong
    # mask. Escalating to the matte-inverting engine is the thing that can
    # actually deliver it. Placed after relight so "sunlit background" still
    # reads as a lighting miss, which it is.
    (re.compile(r"\b(background|backdrop)\b", re.IGNORECASE), "background"),
    (re.compile(r"\b(same (composition|structure|pose|layout|outline)|"
                r"keep the (composition|structure|pose|layout)|"
                r"identical (pose|composition))\b", re.IGNORECASE),
     "img2img_canny"),
]


@dataclass(frozen=True)
class ReconTier:
    """One rung of the image→3D ladder."""
    level: int
    name: str
    models: tuple[str, ...]
    what: str            # user-facing: what this rung actually produces
    paint_model: bool    # does a MODEL paint the texture? (colour from
                         # photographs is separate, and always happens)
    min_vram_gb: float
    min_ram_gb: float
    octree: int          # voxel resolution — the thing hardware actually buys


# Highest first — the machine gets the best rung it can actually run.
#
# What a rung buys is SURFACE DETAIL, not more views. That is a correction of
# a wrong assumption, and it was expensive: the ladder used to spend extra
# hardware on multi-view conditioning, on the theory that four views beat one.
# Measured on this machine, from one photograph and its orbit:
#
#     single-view model, 1 view      depth/width 0.53   <- a human is ~0.5
#     multi-view model,  1 view      depth/width 0.63
#     multi-view model,  4 views     depth/width 0.73
#
# Feeding a multi-view model views that were INVENTED by another model does
# not add information, it adds noise, and the reconstruction thickens to
# satisfy all of it at once. So view count is now driven by the DATA (how many
# real angles the user actually supplied), and hardware only sets the octree.
#
# The octree numbers are measured too: 512 failed here — VoxelToMesh asked the
# CPU allocator for 4.3 GB and was refused — and 384 succeeded while driving
# free RAM to zero. Both are gated above what this box has.
#
# The thresholds sit BELOW the nominal sizes on purpose. Hardware never
# reports its advertised capacity: firmware and the iGPU take a slice first,
# so a 16 GB machine reads 15.7 and an 8 GB card reads 7.99. Testing against
# the round number silently demotes every machine by a tier — measured here,
# a 16 GB box was handed tier 1 because 15.7 < 16.
RECON_TIERS: tuple[ReconTier, ...] = (
    # paint_model is False everywhere: ComfyUI ships no Hunyuan3D texture
    # stage and the official one needs a CUDA rasterizer with no wheel for
    # this Python. That does NOT mean the mesh is untextured — colour is
    # projected from the photographs onto a real UV atlas at every rung.
    ReconTier(4, "mesh-max", ("hunyuan3d-v2", "hunyuan3d-v2-mv"),
              "mesh at octree 512, textured from your photographs",
              False, 15.5, 30.0, 512),
    ReconTier(3, "mesh-fine", ("hunyuan3d-v2", "hunyuan3d-v2-mv"),
              "mesh at octree 384, textured from your photographs",
              False, 11.5, 23.0, 384),
    ReconTier(2, "mesh", ("hunyuan3d-v2", "hunyuan3d-v2-mv"),
              "mesh at octree 256, textured from your photographs",
              False, 7.5, 15.0, 256),
    ReconTier(1, "mesh-lite", ("hunyuan3d-v2",),
              "mesh at octree 192, textured from your photographs",
              False, 5.5, 11.0, 192),
    ReconTier(0, "orbit", (),
              "orbit views only — no mesh on this hardware", False,
              0.0, 0.0, 0),
)

# Multi-view conditioning needs the subject to hold still between views, and a
# PERSON never does. Every configuration measured worse than the single-view
# model on the same photograph — depth/width, where a standing human is ~0.5:
#
#     single-view model, best photo              0.53
#     multi-view, 1 view                         0.63
#     multi-view, 4 views (3 of them rendered)   0.73
#     multi-view, 2 REAL photographed angles     1.19
#
# The last line is the important one, and it is the opposite of what more data
# should do. Hunyuan3D's multi-view model expects four orthographic renders of
# ONE RIGID OBJECT; two photographs of a person differ in pose, framing,
# clothing and lighting, so it reconciles them by thickening the body. Extra
# photographs are therefore used for colour and identity, never for shape.
MULTIVIEW_MIN_REAL_ANGLES = 2
# The multi-view checkpoint is a *turbo* distillation. It was being driven at
# 20 steps and cfg 4.0, which is what an undistilled model wants; at its own
# settings it is both better (0.63 against 0.73) and twice as fast.
MULTIVIEW_SAMPLER = {"steps": 5, "cfg": 1.0}


def choose_reconstruction(vram_gb: float, ram_gb: float) -> ReconTier:
    """The best 3D rung this machine can actually run."""
    for tier in RECON_TIERS:
        if vram_gb >= tier.min_vram_gb and ram_gb >= tier.min_ram_gb:
            return tier
    return RECON_TIERS[-1]


def use_multiview(distinct_real_angles: int, rigid_subject: bool = False) -> bool:
    """Whether to condition on several views.

    Judged on the data rather than the GPU, and gated on the subject being
    RIGID. People are not: the pose changes between photographs, and the
    reconstruction pays for it in the depth of the whole figure."""
    return rigid_subject and distinct_real_angles >= MULTIVIEW_MIN_REAL_ANGLES


def reconstruction_note(tier: ReconTier, vram_gb: float, real_angles: int = 1,
                        colour_views: int = 1, textured: bool = True) -> str:
    """One honest sentence about what this machine will and will not do."""
    top = RECON_TIERS[0].level
    what = tier.what
    if not textured and "textured from your photographs" in what:
        # Saying "textured" while texturing is switched off is the exact kind
        # of note this whole file exists to avoid.
        what = what.replace("textured from your photographs",
                            "bare geometry, texturing switched off")
    note = f"3D tier {tier.level} of {top} — {what}"
    if tier.models and textured:
        note += (f", shaped from {real_angles} photographed angles"
                 if real_angles > 1 else
                 ", shaped from your single sharpest photograph")
        if colour_views > 1:
            note += (f" and coloured from {colour_views} views, so extra "
                     "photographs improve the colour without ever distorting "
                     "the shape")
        if not tier.paint_model:
            note += (". Colour is projected from the photographs onto a UV "
                     "texture; no model-generated paint stage is installable "
                     "on this toolchain")
    return note + "."


def capability_gap(missing: list[str]) -> str | None:
    """The template built for the FIRST unmet requirement that names a
    capability, or None when the miss is ordinary (content the current
    workflow could still produce on another try)."""
    for item in missing:
        if _VIEW_EXCLUDE.search(item):
            continue  # lens/framing language, not a viewpoint change
        for pattern, workflow in CAPABILITY_WORKFLOW:
            if pattern.search(item):
                return workflow
    return None


# Nudges that come free with a strategy change: a retry that already costs a
# full render should also lean harder on the prompt than the attempt that
# missed it.
_BOOST = {"cfg": 1.0, "steps": 1.25, "denoise": 0.05}


def escalation_plan(missing: list[str], *,
                    models: Sequence[str] = (),
                    workflows: Sequence[str] = (),
                    current_model: str | None = None,
                    current_workflow: str | None = None,
                    allow_model_change: bool = True,
                    max_rungs: int = 3) -> list[Strategy]:
    """The ordered rungs to try after a render missed the request.

    Order: a capability-matched workflow first (nothing else can satisfy that
    requirement), then one emphasized re-render of the same recipe (cheap and
    frequently enough), then a DIFFERENT MODEL, then a DIFFERENT WORKFLOW.
    Candidates equal to what already ran are skipped, so every rung really does
    change something. `allow_model_change=False` keeps the model fixed when the
    user named one — the request outranks the ladder."""
    plan: list[Strategy] = []
    seen: set[tuple[str | None, str | None]] = {(current_workflow,
                                                 current_model)}

    def add(strategy: Strategy) -> None:
        if strategy.key() in seen:
            return
        seen.add(strategy.key())
        plan.append(strategy)

    gap = capability_gap(missing)
    if gap and gap in workflows and gap != current_workflow:
        add(Strategy(
            kind="workflow", workflow=gap, boost=dict(_BOOST),
            why=f"'{missing[0][:60]}' needs a workflow built for it — "
                f"switching to {gap}"))

    add(Strategy(kind="emphasize", boost={"cfg": 0.5, "steps": 1.1},
                 why="re-rendering with the missed part of the request "
                     "emphasized"))

    if allow_model_change:
        for name in models:
            if name and name != current_model:
                add(Strategy(
                    kind="model", checkpoint=name, workflow=current_workflow,
                    boost=dict(_BOOST),
                    why=f"the model that ran missed the request — trying "
                        f"{name}"))

    for name in workflows:
        if name and name != current_workflow and name != gap:
            add(Strategy(
                kind="workflow", workflow=name, boost=dict(_BOOST),
                why=f"the workflow that ran missed the request — trying "
                    f"{name}"))

    return plan[:max(0, max_rungs)]


def emphasize(prompt: str, missing: list[str], weight: float = 1.3) -> str:
    """The prompt for a retry: the user's words attention-weighted, with the
    requirements the last attempt missed restated so the model cannot skip
    them again. Append-only — nothing the user wrote is removed."""
    text = f"({prompt}:{weight:g})"
    if missing:
        text += ", " + ", ".join(f"({m}:{weight:g})" for m in missing[:3])
    return text
