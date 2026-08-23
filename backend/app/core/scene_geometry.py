"""Scene geometry: the measured physical facts an environment edit must
respect.

"Change the background to a swimming pool" used to be answered by painting
a pool-shaped picture behind a subject cutout. The missing piece was never
better prompting alone — it was that nothing in the pipeline KNEW where the
subject's feet touch the world, where the camera sits, or where the horizon
belongs, so nothing could ask for (or verify) an environment that shares
the photograph's physical space.

This module owns that knowledge:

  * SceneCard        — the structured representation of one photograph's
                       physics: subject placement and contact points,
                       ground plane, camera pitch, horizon, lighting.
  * subject_geometry — contact points and framing read off the exact
                       BiRefNet matte (no model can argue with the matte's
                       own bottom edge).
  * ground_geometry  — ground plane, camera pitch and horizon fitted from
                       MoGe's camera-space normal + depth renders.
  * environment_spec — the LLM turns the user's words into a structured
                       environment plan (semantics only — every number in
                       the card comes from measurement).
  * spatial_prompt   — compiles card + spec into generation language that
                       states the ground contract, camera height, horizon
                       and lighting explicitly.
  * environment_misses — deterministic post-render validation: the SAME
                       measurements run on the result must agree with the
                       card. Misses are named for the retry ladder.

Doctrine (matches the rest of the project): measured numbers outrank any
model's opinion; a value that cannot be measured is None, never a guess
(the prompt simply says less); every consumer must survive a None."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from PIL import Image, ImageFilter

from .llm import LLMClient, complete_with_schema

# Ground-pixel test: OpenGL camera-space normals, +Y up. A floor's normal
# points up (G channel dominant); 0.75 keeps gentle ramps and pool decks,
# rejects walls (|n_y| ~ 0) and ceilings (n_y ~ -1).
_GROUND_NY = 0.75
# Below this fraction of the frame, "ground" is a few mislabeled pixels,
# not a stand-on surface.
_GROUND_MIN_FRAC = 0.02
# A horizon fit is trusted only when the plane model actually explains the
# depth profile.
_HORIZON_MIN_R2 = 0.95
# Contact points: matte columns whose bottom edge reaches within this
# fraction of the matte's lowest row count as touching the ground.
_CONTACT_TOL_FRAC = 0.04
# A matte that reaches within this many pixels of the frame bottom means
# the subject is cropped by the frame: there IS no visible ground contact.
_BOTTOM_CUT_PX = 3

POSTURES = ("standing", "sitting", "lying", "crouching", "kneeling",
            "leaning", "unknown")


@dataclass
class SceneCard:
    """The physical reading of one photograph. Every field is optional:
    an unmeasurable fact stays None and downstream simply says less."""
    # subject
    subject_box: tuple[int, int, int, int] | None = None
    subject_height_frac: float | None = None
    contact_points: list[tuple[int, int]] = field(default_factory=list)
    contact_y_frac: float | None = None
    cut_at_bottom: bool = False
    posture: str | None = None
    posture_source: str = "none"          # keypoints | vision | none
    # ground & camera (camera space, measured by MoGe)
    ground_frac: float | None = None
    camera_pitch_deg: float | None = None
    horizon_y_frac: float | None = None   # can be < 0: above the frame
    horizon_r2: float | None = None
    # scene reads (vision model, schema-forced, from the understand stage)
    lighting: str = ""
    perspective_note: str = ""
    setting: str = ""

    def camera_words(self) -> str:
        """The measured camera, in language a diffusion model understands."""
        p = self.camera_pitch_deg
        if p is None:
            return ""
        if abs(p) < 4:
            return "photographed at eye level"
        updown = "slightly above" if p > 0 else "slightly below"
        if abs(p) > 12:
            updown = "high above" if p > 0 else "well below"
        return f"photographed from {updown} the subject's eye level"

    def horizon_words(self) -> str:
        h = self.horizon_y_frac
        if h is None or not (self.horizon_r2 or 0) >= _HORIZON_MIN_R2:
            return ""
        if h < 0.05:
            return "the horizon sits above the top of the frame"
        band = ("near the top of the frame" if h < 0.33 else
                "across the middle of the frame" if h < 0.6 else
                "low in the frame")
        return f"the horizon line {band}"

    def to_debug(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if v not in (None, "", [])}
        for k in ("subject_height_frac", "contact_y_frac", "ground_frac",
                  "camera_pitch_deg", "horizon_y_frac", "horizon_r2"):
            if d.get(k) is not None:
                d[k] = round(d[k], 3)
        return d


def subject_geometry(matte: Image.Image) -> dict[str, Any]:
    """Contact points and framing, read off the subject matte itself.

    The matte's bottom edge IS the contact evidence: the columns that reach
    lowest are where the subject meets whatever supports them. A matte that
    reaches the frame's bottom edge is a cropped subject — there is no
    visible contact, and an environment edit must not invent ground for
    feet that are outside the photograph."""
    import numpy as np
    a = np.asarray(matte.convert("L")) > 127
    h, w = a.shape
    cols = np.nonzero(a.any(axis=0))[0]
    rows = np.nonzero(a.any(axis=1))[0]
    if not len(cols) or not len(rows):
        return {}
    box = (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
    bottom = int(rows[-1])
    # The cut test scales with resolution: the matting engine feathers a
    # cropped subject's edge a few rows short of the frame, and a fixed
    # 3px missed it by ONE pixel live — fem.png's matte stopped at row
    # 1439 of 1444, so 53% of the frame width (thighs at the crop edge)
    # became four phantom "contact points", and every environment render
    # failed "nothing walkable under the subject's feet" against feet
    # that are outside the photograph. Contacts this close to the edge
    # are unusable anyway: there are no rows below them to validate
    # ground against.
    cut = bottom >= h - 1 - max(_BOTTOM_CUT_PX, h // 100)
    contacts: list[tuple[int, int]] = []
    if not cut:
        # per-column lowest matte row; the columns within tolerance of the
        # global lowest are the touching ones. Cluster into contact groups
        # (two feet -> two groups) and keep each group's center.
        lowest = h - 1 - np.argmax(a[::-1], axis=0)
        lowest[~a.any(axis=0)] = -1
        tol = max(4, int(h * _CONTACT_TOL_FRAC))
        touch = np.nonzero((lowest >= 0) & (lowest >= bottom - tol))[0]
        if len(touch):
            gaps = np.nonzero(np.diff(touch) > max(8, w // 60))[0]
            for grp in np.split(touch, gaps + 1):
                cx = int(grp.mean())
                contacts.append((cx, int(lowest[grp].max())))
    return {
        "subject_box": box,
        "subject_height_frac": (box[3] - box[1]) / h,
        "contact_points": contacts,
        "contact_y_frac": (bottom / h) if not cut else None,
        "cut_at_bottom": cut,
    }


def _fit_horizon(ys: Any, zs: Any, height: int) -> tuple[float, float] | None:
    """Horizon row from the ground's depth profile, tolerant of the render's
    unknown display encoding.

    For a ground plane, disparity is LINEAR in the image row and reaches
    zero at the horizon. The probe's depth PNG is a display render with an
    unknown affine mapping — so two candidate models are fitted against the
    per-row median profile and the better one wins:

      * reciprocal:  z = A / (y - y_h) + B   (PNG encodes affine DEPTH;
        y_h found by scanning candidate horizons above the ground)
      * linear:      z = A*y + B             (PNG encodes affine DISPARITY;
        the horizon is where the line meets the far-field disparity level,
        which the caller supplies via the top-of-ground limit)

    Returns (horizon_y, r2) or None. The r2 travels with the card: a plane
    that does not explain the profile must not claim a horizon.

    Rows at the render's quantization floor or ceiling (exactly 0 or 1
    after 8-bit encoding) are dropped before fitting: measured live, a
    plaza whose ground ran all the way to the horizon had its top rows
    saturated at 0.0, and those rows both bent the fit and pushed the
    root onto the boundary guard."""
    import numpy as np
    if len(ys) < 24:
        return None
    ys = np.asarray(ys, dtype=np.float64)
    zs = np.asarray(zs, dtype=np.float64)
    keep = (zs > 1.5 / 255) & (zs < 253.5 / 255)
    if keep.sum() < 24:
        return None
    ys, zs = ys[keep], zs[keep]
    var = float(((zs - zs.mean()) ** 2).sum()) or 1e-9

    def r2(pred: Any) -> float:
        return 1.0 - float(((zs - pred) ** 2).sum()) / var

    # linear model (disparity-encoded PNG)
    al, bl = np.polyfit(ys, zs, 1)
    lin_r2 = r2(al * ys + bl)
    # reciprocal model (depth-encoded PNG): scan candidate horizons
    best: tuple[float, float, float] | None = None   # (r2, y_h, slope)
    y_top = float(ys.min())
    for y_h in np.linspace(y_top - 1.6 * height, y_top - 2, 120):
        x = 1.0 / (ys - y_h)
        ar, br = np.polyfit(x, zs, 1)
        rr = r2(ar * x + br)
        if best is None or rr > best[0]:
            best = (rr, float(y_h), float(ar))
    if best is not None and best[0] >= lin_r2:
        return best[1], best[0]
    # linear wins: disparity falls toward the horizon, so extrapolate the
    # line to zero disparity, which must land at or above the ground's
    # top edge (a small tolerance: ground that runs TO the horizon puts
    # the root exactly on the boundary) and within one frame height.
    if al > 1e-9:
        y_h = -bl / al
        if -1.5 * height < y_h < y_top + 0.05 * height:
            return float(y_h), lin_r2
    return None


def contact_ground_frac(normal_png: Image.Image,
                        contacts: list[tuple[int, int]],
                        size: tuple[int, int]) -> float | None:
    """How much of the area directly under the contact points is walkable
    (up-facing) surface. THE first-class check of an environment edit: a
    scene can hold a perfect horizon and still have painted water around
    the subject's ankles — measured live on the first geometry-validated
    render, where camera and horizon passed and the subject stood in the
    pool. Water and walls face the camera, floors face up; the window
    under each foot answers directly. `contacts` are in source-image
    coordinates; `size` is the source (w, h) they refer to."""
    import numpy as np
    if not contacts:
        return None
    n = np.asarray(normal_png.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    h, w = n.shape[:2]
    sx, sy = w / max(1, size[0]), h / max(1, size[1])
    ground = n[:, :, 1] > _GROUND_NY
    vals: list[float] = []
    for cx, cy in contacts:
        x, y = int(cx * sx), int(cy * sy)
        x0, x1 = max(0, x - int(w * 0.03)), min(w, x + int(w * 0.03) + 1)
        y0, y1 = min(h - 1, y + 2), min(h, y + int(h * 0.06) + 2)
        win = ground[y0:y1, x0:x1]
        if win.size:
            vals.append(float(win.mean()))
    return sum(vals) / len(vals) if vals else None


def ground_geometry(normal_png: Image.Image, depth_png: Image.Image,
                    valid_png: Image.Image,
                    matte: Image.Image | None) -> dict[str, Any]:
    """Ground plane, camera pitch and horizon from the MoGe probe renders.

    Normals are camera-space OpenGL (+Y up): up-facing pixels ARE the
    walkable ground, no classifier involved. The mean ground normal's tilt
    toward the viewer is the camera pitch (a camera looking down sees floor
    normals leaning at it). The horizon comes from the ground's depth
    profile via _fit_horizon. The subject is excluded so a full-length
    figure cannot pollute the plane statistics."""
    import numpy as np
    n = np.asarray(normal_png.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    h, w = n.shape[:2]
    valid = np.asarray(
        valid_png.convert("L").resize((w, h), Image.NEAREST)) > 127
    ground = (n[:, :, 1] > _GROUND_NY) & valid
    if matte is not None:
        subj = np.asarray(
            matte.convert("L").resize((w, h), Image.NEAREST)) > 127
        ground &= ~subj
    frac = float(ground.mean())
    out: dict[str, Any] = {"ground_frac": frac}
    if frac < _GROUND_MIN_FRAC:
        return out
    ny = float(n[:, :, 1][ground].mean())
    nz = float(n[:, :, 2][ground].mean())
    out["camera_pitch_deg"] = float(np.degrees(np.arctan2(nz, max(ny, 1e-6))))
    z = np.asarray(depth_png.convert("L").resize((w, h), Image.BILINEAR),
                   dtype=np.float32) / 255.0
    rows, zs = [], []
    for y in range(h):
        m = ground[y]
        if m.sum() >= max(8, w // 40):
            rows.append(y)
            zs.append(float(np.median(z[y][m])))
    fit = _fit_horizon(rows, zs, h)
    if fit is not None:
        out["horizon_y_frac"] = fit[0] / h
        out["horizon_r2"] = fit[1]
    return out


def guidance_depth(card: SceneCard, depth_png: Image.Image,
                   normal_png: Image.Image, valid_png: Image.Image,
                   matte: Image.Image | None,
                   size: tuple[int, int]) -> Image.Image | None:
    """The perspective guide for depth-conditioned environment generation.

    Built from measurement, not imagination: the subject keeps its
    measured disparity (it is composited back anyway, so the scene must
    agree with it), the visible ground keeps its measured disparity (the
    plane the feet stand on), the rest of the frame below the measured
    horizon gets that plane's ramp extended (disparity is linear in the
    row for a plane), and everything above the horizon is left at zero —
    far, free for whatever the new environment wants to put there. The
    probe's depth render was measured disparity-encoded with near=bright,
    which is exactly the ControlNet depth convention.

    None whenever the horizon is not confidently measured — guidance
    built on a guessed horizon would be fabricated geometry (the
    failure-handling doctrine forbids exactly that)."""
    import numpy as np
    if card.horizon_y_frac is None \
            or (card.horizon_r2 or 0) < _HORIZON_MIN_R2:
        return None
    z = np.asarray(depth_png.convert("L"), dtype=np.float32) / 255.0
    h, w = z.shape
    y_h = card.horizon_y_frac * h
    rows = np.arange(h, dtype=np.float32)
    ramp_rows = np.clip((rows - y_h) / max(1.0, h - y_h), 0.0, 1.0)
    # The ground gets the FITTED plane (the ramp), not its measured pixels:
    # the fit explains the plane at r²≈1, and the residue is tile grout and
    # texture — measured live, keeping raw ground disparity made the
    # ControlNet repaint the original plaza's tile grid as striped
    # pavement in the new deck. Only the subject keeps measured values.
    canvas = np.repeat(ramp_rows[:, None], w, axis=1)
    base = Image.fromarray((canvas * 255).astype(np.uint8), "L")
    base = base.filter(ImageFilter.GaussianBlur(3))
    out = np.asarray(base, dtype=np.float32) / 255.0
    if matte is not None:
        subj = np.asarray(
            matte.convert("L").resize((w, h), Image.NEAREST)) > 127
        # the subject's silhouette stays crisp: a blurred depth edge reads
        # as a halo around the person in the conditioned render
        out[subj] = z[subj]
    guide = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8),
                            "L").convert("RGB")
    return guide.resize(size, Image.BILINEAR)


def matte_group_count(matte: Image.Image) -> int:
    """How many separated figures a subject matte holds (column-gap
    split, the _mask_view_boxes trick). People standing apart count
    individually; an embracing pair reads as one — callers treat this as
    a lower bound."""
    import numpy as np
    a = np.asarray(matte.convert("L")) > 127
    idx = np.nonzero(a.any(axis=0))[0]
    if not len(idx):
        return 0
    gap = max(16, a.shape[1] // 12)
    return 1 + int((np.diff(idx) > gap).sum())


def posture_veto(posture: str | None,
                 box: tuple[int, int, int, int] | None) -> str | None:
    """Deterministic sanity check on a vision-model posture answer: the
    matte's own aspect ratio can veto the impossible. A subject whose
    matte is 2.5x taller than wide is not 'lying'; one wider than tall is
    not 'standing'. A vetoed answer degrades to None (unknown) — per the
    failure-handling doctrine, uncertainty is marked, never papered over."""
    if posture not in POSTURES or posture == "unknown":
        return None
    if box is None:
        return posture
    w, h = box[2] - box[0], box[3] - box[1]
    if w <= 0 or h <= 0:
        return posture
    aspect = h / w
    if posture == "lying" and aspect > 1.6:
        return None
    if posture == "standing" and aspect < 1.0:
        return None
    return posture


_SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "environment": {"type": "string"},
        "relationship": {"type": "string"},
        "ground_surface": {"type": "string"},
        "elements": {"type": "array", "items": {"type": "string"},
                     "minItems": 2, "maxItems": 6},
        "lighting_wish": {"type": "string"},
    },
    "required": ["environment", "relationship", "ground_surface",
                 "elements", "lighting_wish"],
}

_SPEC_SYSTEM = (
    "You plan photographic environments. Given an edit request and what is "
    "known about the subject, describe the PHYSICAL environment the subject "
    "should exist inside — never a backdrop behind them. Reply ONLY JSON: "
    '{"environment": "<the place, concise>", '
    '"relationship": "<how the subject physically relates to it, e.g. '
    "'standing on the pool deck beside the pool'>\", "
    '"ground_surface": "<what is under the subject, e.g. \'wet tiles\'; '
    "'none visible' if the subject is cropped above the ground>\", "
    '"elements": ["<2-6 things that make the place real, near AND far>"], '
    '"lighting_wish": "<the lighting this place implies, or \'keep\' to '
    "keep the photo's current light>\"}. Physical plausibility outranks "
    "spectacle. Never invent a second person. When the subject's posture "
    "or ground contact is NOT stated, choose the most conservative "
    "physical relationship: on solid ground, beside or near the place's "
    "features — never inside water, on furniture, or in mid-air. (The "
    "first live plan without this rule put a standing subject 'in the "
    "shallow end of the pool' on no evidence at all.)")


def environment_spec(llm: LLMClient | None, instruction: str,
                     posture: str | None, cut_at_bottom: bool | None,
                     setting: str = "") -> dict[str, Any] | None:
    """The user's words become a structured environment plan (semantics
    only; geometry stays measured). None when no local planner is up —
    the caller falls back to the plain enhanced prompt. `cut_at_bottom`
    may be None (plan-time batching runs before any analysis): an unknown
    fact is simply not stated, never assumed."""
    if llm is None:
        return None
    facts = []
    if posture:
        facts.append(f"the subject is {posture}")
    if cut_at_bottom is not None:
        facts.append("the subject's feet/lower body are OUTSIDE the frame "
                     "— no ground contact is visible" if cut_at_bottom else
                     "the subject's contact with the ground is visible in "
                     "frame")
    if setting:
        facts.append(f"current setting: {setting}")
    ask = (f"Edit request: {instruction}\nSubject facts: "
           + ("; ".join(facts) if facts else "none measured yet"))
    try:
        reply = complete_with_schema(llm, _SPEC_SYSTEM, ask,
                                     max_tokens=420, schema=_SPEC_SCHEMA)
        data = json.loads(re.sub(r"^```(?:json)?|```$", "",
                                 reply.text.strip(), flags=re.M).strip())
    except Exception:  # noqa: BLE001 — a planner failure must not kill a job
        return None
    if not isinstance(data, dict) or not data.get("environment"):
        return None
    data["elements"] = [str(e)[:60] for e in data.get("elements", [])][:6]
    return data


def lighting_prompt(lighting: str | None) -> str:
    """The IC-Light conditioning for a background swap: LIGHTING ONLY.

    The lighting match used to be conditioned on the full compiled
    environment prompt plus a hard-coded "natural light on the subject" —
    for a dim scene (a nightclub) that phrase actively fights the very
    illumination the pass exists to transfer, and the IC-Light template's
    own doc warns that non-lighting words make it re-synthesise content
    instead of relighting. The plan already knows what light the scene
    has (the spec's lighting_wish); lead with that, and say nothing else
    about the scene."""
    light = (lighting or "").strip().rstrip(".")
    # The spec planner sometimes answers with a DIRECTIVE ("keep") rather
    # than a description — seen live on the nightclub spec. A directive
    # is not lighting language; fall back rather than lead the
    # conditioning with it.
    if light.lower() in ("keep", "same", "unchanged", "as is", "as-is",
                         "no change", "none", "current", "original"):
        light = ""
    light = light or "natural light"
    return (f"{light} on the subject, matching the scene, consistent "
            "shadows and colour temperature, photograph")


def spatial_prompt(spec: dict[str, Any] | None, card: SceneCard | None,
                   base_positive: str, base_negative: str
                   ) -> tuple[str, str]:
    """Compile plan + measured card into generation language.

    Every clause is conditional on real knowledge: an unmeasured horizon
    adds no horizon words, a cropped subject adds no ground contract. The
    environment-not-backdrop clause and the no-second-person negative stay
    unconditional — both were bought with measured failures (a framed
    forest poster on the original wall; a second headless figure)."""
    parts: list[str] = []
    if spec:
        parts.append(spec["environment"])
        if spec.get("relationship"):
            parts.append(f"the subject {spec['relationship']}")
        gs = (spec.get("ground_surface") or "").strip()
        if (gs and "none" not in gs.lower()
                and not (card and card.cut_at_bottom)):
            parts.append(f"{gs} under and around the subject's feet, "
                         "continuous with the scene")
        if spec.get("elements"):
            parts.append(", ".join(spec["elements"]))
    else:
        parts.append(base_positive)
    if card:
        for clause in (card.camera_words(), card.horizon_words()):
            if clause:
                parts.append(clause)
        wish = ((spec or {}).get("lighting_wish") or "").strip().lower()
        if card.lighting and wish in ("", "keep"):
            parts.append(f"lighting: {card.lighting}")
        elif wish and wish != "keep":
            parts.append(f"lighting: {wish}")
    parts.append("the surrounding environment, a real place extending "
                 "behind and around the subject, continuous scene, natural "
                 "depth of field, correct perspective, photograph")
    gs = ((spec or {}).get("ground_surface") or "").lower()
    solid_ground = (bool(gs) and "none" not in gs and "water" not in gs
                    and not (card and card.cut_at_bottom))
    negative = ", ".join(t for t in (
        (base_negative or "").strip(" ,"),
        "person, people, human figure, limbs, extra person, duplicate "
        "subject, crowd, text, watermark, flat backdrop, painted wall, "
        "poster, tilted horizon",
        # The plan puts the subject on a SOLID surface: forbid the sampler
        # from flooding the foreground instead. Measured live — a "standing
        # on the pool deck beside the pool" plan was rendered with water
        # across the whole lower frame and the subject ankle-deep in it.
        "water under the subject's feet, subject standing in water, "
        "submerged feet, flooded foreground" if solid_ground else "",
    ) if t)
    return ", ".join(p for p in parts if p), negative


# Post-render validation tolerances, calibrated loose-first: the horizon
# fit's own uncertainty spans a few percent of frame height, and a real
# perspective break (backdrop pasted behind a subject) moves the measured
# horizon by far more than this — or removes the ground entirely.
_H_TOL_FRAC = 0.12
_PITCH_TOL_DEG = 10.0


# Under-foot walkable coverage below this means the subject stands on
# nothing solid. Measured live: a subject painted INTO the pool measured
# 0.00 under both feet while the scene's global geometry passed.
_CONTACT_MIN = 0.25


def environment_misses(card: SceneCard, after: dict[str, Any],
                       subject: dict[str, Any] | None = None) -> list[str]:
    """Deterministic geometry comparison, original card vs the SAME
    measurements on the rendered result. Named misses feed the retry
    ladder; an unmeasurable side stays silent (no fabricated verdicts)."""
    misses: list[str] = []
    cgf = after.get("contact_ground_frac")
    if (card.contact_points and not card.cut_at_bottom
            and cgf is not None and cgf < _CONTACT_MIN):
        misses.append("nothing walkable under the subject's feet "
                      f"(only {cgf:.0%} of the area below the contact "
                      "points is up-facing surface) — they read as "
                      "standing in water or floating")
    if card.ground_frac and card.ground_frac >= _GROUND_MIN_FRAC:
        gf = after.get("ground_frac")
        if not card.cut_at_bottom and gf is not None \
                and gf < _GROUND_MIN_FRAC:
            misses.append("the subject stood on visible ground, but the "
                          "new scene has no walkable ground under them "
                          "(floating-subject look)")
    p0, p1 = card.camera_pitch_deg, after.get("camera_pitch_deg")
    if p0 is not None and p1 is not None \
            and abs(p0 - p1) > _PITCH_TOL_DEG:
        misses.append(f"the camera angle changed: the original ground "
                      f"tilts like a {p0:.0f} degree camera, the new "
                      f"scene like {p1:.0f} degrees")
    h0, h1 = card.horizon_y_frac, after.get("horizon_y_frac")
    if (h0 is not None and h1 is not None
            and (card.horizon_r2 or 0) >= _HORIZON_MIN_R2
            and (after.get("horizon_r2") or 0) >= _HORIZON_MIN_R2
            and abs(h0 - h1) > _H_TOL_FRAC):
        misses.append("the horizon moved: it sat at "
                      f"{h0:.0%} of the frame height and now sits at "
                      f"{h1:.0%} — the new scene's perspective does not "
                      "match the photograph")
    return misses


def parse_probe_files(files: list[tuple[bytes, str]]
                      ) -> dict[str, Image.Image]:
    """Route the probe graph's three renders by filename prefix."""
    import io as _io
    out: dict[str, Image.Image] = {}
    for data, fname in files:
        for key in ("depth", "normal", "valid"):
            if f"pfprobe_{key}" in fname:
                out.setdefault(key,
                               Image.open(_io.BytesIO(data)).convert("RGB"))
    return out
