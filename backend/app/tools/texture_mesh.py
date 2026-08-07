"""Paint a generated mesh with the photographs it was built from.

Run with ComfyUI's interpreter, not PromptForge's: trimesh, embreex and scipy
already live there, so this needs no new dependency and no download. The mesh
model produces geometry only — ComfyUI ships no Hunyuan3D texture stage, and
the official one needs a CUDA rasterizer with no wheel for this Python — so
the colour has to come from the photographs rather than from a paint model.

    python texture_mesh.py <mesh.glb> <out.glb> --view <azimuth>:<image.png>
                           [--view ...] [--tile 2048] [--synth az1,az2]

What has to be true for the paint to land where it belongs, each of which was
once missing and each of which left a measurable scar:

  every view       one photograph sees under half of a figure; each vertex
                   takes the view that sees it best.

  measured angles  the camera azimuth per view is FOUND by silhouette
                   matching, not trusted. The refinement window used to be
                   ±18° and the reference dataset's true residuals were
                   larger: both side views slammed into the wall exactly
                   (solved 234 = 252−18) and the sides were painted from
                   cameras that were provably somewhere else.

  2D alignment     an angle alone is not an alignment. The old projection
                   fitted the mesh's bounding box to the subject's bounding
                   box, so one stray dark pixel in the matte — or hair volume
                   the mesh doesn't have — shifted every sample on that view.
                   Each view now gets a small similarity transform (scale and
                   shift per axis) solved by maximising silhouette overlap.

  one exposure     the views are separate photographs (and SV3D renders) with
                   separate tone. Painted side by side they read as patches.
                   Each view is now colour-matched, on vertices that two
                   cameras both see squarely, to the views already accepted —
                   chained outward from the first (front) view.

  quiet seams      the view borders used to be decided per face by raw
                   argmax, which produced islands of one view inside another;
                   scores are now smoothed over the face graph first (only
                   ever choosing among views that see all three corners), and
                   the atlas UVs keep a gutter so bilinear filtering cannot
                   bleed one view's tile into the next.

  a real texture   views are packed into a UV atlas; detail is limited by the
                   photograph rather than by the vertex count. (There are no
                   vertex colours: glTF texture visuals cannot carry them,
                   and computing what cannot be exported was pure cost.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

GRID = 168          # silhouette resolution for the global orientation search
ALIGN_GRID = 208    # finer grid for the per-view similarity refinement
FACING_MIN = 0.20   # below this a vertex is edge-on to the camera. It was
                    # 0.05, and at that grazing angle a projection error of a
                    # few pixels sweeps across whole regions of the photo —
                    # the skirt was painted around onto the back of the
                    # figure. Surface-filled colour is soft but it is never
                    # somebody's clothing print in the wrong place.
UV_GUTTER = 2.0     # texels kept clear at every atlas tile edge


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    return mesh


def repair_normals(mesh: trimesh.Trimesh) -> bool:
    """Make the winding consistent so the normals actually point outwards.

    The surface-net mesher emits faces wound both ways. A vertex whose normal
    came out inverted fails the facing test AND fires its occlusion ray into
    the model's own interior, where it is immediately blocked — so it is
    silently dropped. Measured on a 110k-vertex figure: coverage from four
    views was 54.3% before this and 94.4% after."""
    if mesh.is_winding_consistent:
        return False
    trimesh.repair.fix_normals(mesh)
    return True


def camera_basis(deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(right, up, forward) for a camera orbiting the Y axis at `deg`.

    forward points FROM the subject TOWARDS the camera, so a vertex whose
    normal agrees with it is facing the lens."""
    t = np.radians(deg)
    right = np.array([np.cos(t), 0.0, -np.sin(t)])
    up = np.array([0.0, 1.0, 0.0])
    forward = np.array([np.sin(t), 0.0, np.cos(t)])
    return right, up, forward


def subject_mask(img: np.ndarray) -> np.ndarray:
    """The staged views sit on neutral grey; anything else is the subject.

    The backdrop colour is measured on the BORDER ring, which is backdrop by
    construction. The old global median stopped being the backdrop the moment
    the subject filled most of the frame — a close-up portrait's median is
    skin, and the mask inverted."""
    from scipy import ndimage
    ring = np.concatenate([img[:6].reshape(-1, 3), img[-6:].reshape(-1, 3),
                           img[:, :6].reshape(-1, 3),
                           img[:, -6:].reshape(-1, 3)])
    bg = np.median(ring, axis=0)
    m = np.abs(img.astype(int) - bg).max(axis=2) > 18
    m = ndimage.binary_closing(m, np.ones((5, 5)))
    return ndimage.binary_fill_holes(m)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """The paintable part of a matte: drop far debris, shave the halo ring.

    Rendered honestly in Blender, the shipped avatar wore two artefacts the
    quality gate cannot catch because they live INSIDE views that pass it:
    beige scraps of hallucinated wall floating near the shoulders (matted as
    subject, painted as subject), and a pink smear where flooded EDGE pixels
    — the matting halo, half skin half backdrop — were dragged across every
    surface no camera saw. Both are properties of the mask, so both are
    fixed here: connected components beyond arm's reach of the main figure
    are debris (a real hand attaches through the body; a wall scrap does
    not), and a few edge pixels are shaved so the flood fill and the
    sampler read clean interior colour instead of the mixed rim."""
    from scipy import ndimage
    if not mask.any():
        return mask
    out = mask
    lab, n = ndimage.label(out)
    if n > 1:
        sizes = ndimage.sum(out, lab, range(1, n + 1))
        main = int(sizes.argmax()) + 1
        dist = ndimage.distance_transform_edt(lab != main)
        reach = 0.02 * max(out.shape)
        drop = [c for c in range(1, n + 1)
                if c != main and dist[lab == c].min() > reach]
        if drop:
            out = out & ~np.isin(lab, drop)
    shave = max(2, int(round(0.003 * max(out.shape))))
    shaved = ndimage.binary_erosion(out, iterations=shave)
    return shaved if shaved.any() else out


def _fit_square(mask: np.ndarray) -> np.ndarray:
    """Crop to content and fit into a fixed square, so two silhouettes are
    compared on shape alone rather than on how they happened to be framed."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return np.zeros((GRID, GRID), bool)
    sub = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = sub.shape
    scale = (GRID - 2) / max(h, w)
    small = Image.fromarray(sub.astype(np.uint8) * 255).resize(
        (max(1, int(w * scale)), max(1, int(h * scale))), Image.NEAREST)
    out = np.zeros((GRID, GRID), bool)
    a = np.asarray(small) > 127
    oy, ox = (GRID - a.shape[0]) // 2, (GRID - a.shape[1]) // 2
    out[oy:oy + a.shape[0], ox:ox + a.shape[1]] = a
    return out


def _splat(px: np.ndarray, py: np.ndarray, size: int) -> np.ndarray:
    from scipy import ndimage
    g = np.zeros((size, size), bool)
    g[np.clip(py, 0, size - 1), np.clip(px, 0, size - 1)] = True
    g = ndimage.binary_closing(g, np.ones((3, 3)))
    return ndimage.binary_fill_holes(g)


def mesh_silhouette(verts: np.ndarray, deg: float) -> np.ndarray:
    right, _, _ = camera_basis(deg)
    u, v = verts @ right, verts[:, 1]
    span = max(np.ptp(u), np.ptp(v)) or 1.0
    px = ((u - u.min()) / span * (GRID - 3)).astype(int) + 1
    py = ((v.max() - v) / span * (GRID - 3)).astype(int) + 1
    return _fit_square(_splat(px, py, GRID))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum()) / float(union) if union else 0.0


def solve_orientation(verts: np.ndarray, views: list[View]
                      ) -> tuple[list[float], float, dict]:
    """Find the camera azimuth for each view by matching silhouettes.

    Solved as ONE global rotation plus a handedness, with only a few degrees
    of per-view slack. Two measured failure modes shaped that:

    A human silhouette is near enough symmetric that front and back match
    each other, so a per-view search from scratch can paint the face onto
    the back of the head; the ensemble breaks that tie.

    And a WIDE per-view refinement is worse than none: the relative spacing
    of the views is the one thing that is actually known — the orbit renders
    them at fixed intervals — while silhouette overlap for a standing figure
    is a shallow, multi-modal objective. Given ±30° of freedom the side
    views wandered to angles no camera occupied (solved spacing 358, 319,
    217, 140 for a rigid 0/86/171/274 orbit) and coverage fell ten points.
    The rigid model plus ±6° of slack keeps the physics; the per-view 2D
    similarity fit absorbs what an angle cannot express anyway."""
    targets = [(_fit_square(v.mask), v.azimuth) for v in views]
    cache: dict[int, np.ndarray] = {}

    def sil(deg: float) -> np.ndarray:
        key = int(round(deg)) % 360
        if key not in cache:
            cache[key] = mesh_silhouette(verts, key)
        return cache[key]

    def ensemble(offset: float, sign: int) -> float:
        return float(np.mean([_iou(sil(offset + sign * nom), t)
                              for t, nom in targets]))

    best = (-1.0, 0.0, 1)
    for offset in range(0, 360, 3):
        for sign in (1, -1):
            score = ensemble(offset, sign)
            if score > best[0]:
                best = (score, float(offset), sign)
    score, offset, sign = best
    offset = max(((ensemble(offset + d, sign), offset + d)
                  for d in range(-2, 3)), key=lambda x: x[0])[1]
    angles: list[float] = []
    for target, nominal in targets:
        base = offset + sign * nominal
        local = max(((_iou(sil(base + d), target), base + d)
                     for d in range(-6, 7, 2)), key=lambda x: x[0])
        angles.append(local[1] % 360)
    detail = {"global_offset": round(offset, 1), "handedness": sign,
              "mean_iou": round(score, 3),
              "azimuths": [round(a, 1) for a in angles]}
    return angles, score, detail


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------
class View:
    """One staged photograph, with everything the projection needs precomputed.

    `filled` is the image with the subject's colours flooded outwards over the
    grey surround. A mesh silhouette never matches a photograph's exactly —
    hair especially — so a rim of vertices always lands just outside. Sampling
    the raw staged frame there paints them backdrop-grey, which is what put a
    grey halo around the head and a grey band down one arm."""

    def __init__(self, azimuth: float, image: np.ndarray):
        from scipy import ndimage
        self.azimuth = azimuth
        self.image = image
        # The RAW matte is what the quality gate must judge — cleaning first
        # would hide exactly the fragmentation the gate exists to catch.
        self.mask_raw = subject_mask(image)
        self.mask = clean_mask(self.mask_raw)
        ys, xs = np.nonzero(self.mask)
        h, w = image.shape[:2]
        if len(ys):
            self.bbox = (xs.min() / (w - 1), ys.min() / (h - 1),
                         xs.max() / (w - 1), ys.max() / (h - 1))
        else:
            self.bbox = (0.0, 0.0, 1.0, 1.0)
        _, idx = ndimage.distance_transform_edt(
            ~self.mask, return_distances=True, return_indices=True)
        self.filled = (image[idx[0], idx[1]] if self.mask.any()
                       else image).astype(float)


def view_quality(image: np.ndarray, mask: np.ndarray) -> dict:
    """Is this view actually a photograph of the subject?

    The orbit model does not always produce one. Measured on a real dataset,
    a third of the rendered views were a flat hallucinated slab with a sliver
    of person embedded in it, and two more were a floating fragment beside a
    disjoint blob. Painted onto the mesh they were the single largest source
    of 'the texturing is off': the whole back of the figure took the slab's
    colour. Three cheap signals separate them cleanly (good views measured
    0.06-0.28 slab share; slab views 0.41-0.89):

      slab_share  share of the matte held by its largest FLAT connected
                  component — clothes and skin carry texture, a painted wall
                  does not
      fragments   share of the matte in its largest connected component —
                  a person mattes as one piece, debris mattes as several
      area        share of the frame the matte covers at all"""
    from scipy import ndimage
    area = float(mask.mean())
    if not mask.any():
        return {"slab_share": 1.0, "fragment_share": 0.0, "area": 0.0}
    grey = image.mean(axis=2)
    grad = np.hypot(ndimage.sobel(grey, axis=1), ndimage.sobel(grey, axis=0))
    flat = mask & (grad < 12)
    slab = 0.0
    lab, n = ndimage.label(flat)
    if n:
        sizes = ndimage.sum(flat, lab, range(1, n + 1))
        slab = float(sizes.max()) / float(mask.sum())
    lab, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    fragment = float(sizes.max()) / float(mask.sum())
    return {"slab_share": round(slab, 2),
            "fragment_share": round(fragment, 2), "area": round(area, 3)}


def usable(quality: dict) -> str | None:
    """None when the view may be painted, else the reason it may not.

    The slab test requires BOTH high flatness and a frame-filling matte. A
    real person in one plain flat garment measures slab_share 0.5-0.7 on a
    clean matte — higher than some genuine hallucinations — so flatness
    alone would throw away the user's own photographs. What a hallucinated
    wall has that a plainly-dressed person does not is SIZE: the measured
    walls covered 45-57% of the frame where staged subjects covered 3-44%."""
    if quality["area"] < 0.04:
        return "almost no subject in frame"
    if quality["slab_share"] > 0.35 and quality["area"] > 0.40:
        return "a flat hallucinated slab filling the frame"
    if quality["fragment_share"] < 0.70:
        return "subject matted as disconnected fragments"
    return None


def _base_uv(verts: np.ndarray, view: View, deg: float) -> np.ndarray:
    """The bounding-box fit: mesh silhouette box onto the subject's box."""
    right, up, _ = camera_basis(deg)
    u, v = verts @ right, verts @ up
    bx0, by0, bx1, by1 = view.bbox
    du, dv = (np.ptp(u) or 1.0), (np.ptp(v) or 1.0)
    uu = bx0 + (u - u.min()) / du * (bx1 - bx0)
    vv = by0 + (v.max() - v) / dv * (by1 - by0)
    return np.stack([uu, vv], axis=1)


def solve_view_mapping(mesh: trimesh.Trimesh, view: View,
                       deg: float) -> dict:
    """A small per-view similarity transform on top of the bounding-box fit.

    The box fit is exact only when the mesh and the matte agree about the
    subject's extent, and they never quite do: hair the mesher smoothed away,
    a shadowed pixel included in the matte, a hand the photo crops. Any of it
    shifts every sample on that view. This solves scale and shift per axis by
    maximising silhouette overlap — coordinate descent from the identity,
    bounded so it can only nudge, never wander."""
    uv0 = _base_uv(mesh.vertices, view, deg)
    target = np.asarray(Image.fromarray(
        view.mask.astype(np.uint8) * 255).resize(
        (ALIGN_GRID, ALIGN_GRID), Image.NEAREST)) > 127
    bx0, by0, bx1, by1 = view.bbox
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2

    def overlap(params: tuple[float, float, float, float]) -> float:
        sx, sy, dx, dy = params
        uu = cx + (uv0[:, 0] - cx) * sx + dx
        vv = cy + (uv0[:, 1] - cy) * sy + dy
        px = (uu * (ALIGN_GRID - 1)).astype(int)
        py = (vv * (ALIGN_GRID - 1)).astype(int)
        return _iou(_splat(px, py, ALIGN_GRID), target)

    params = [1.0, 1.0, 0.0, 0.0]
    best = overlap(tuple(params))
    lo = (0.85, 0.85, -0.06, -0.06)
    hi = (1.18, 1.18, 0.06, 0.06)
    for step in (0.04, 0.02, 0.01, 0.005):
        improved = True
        while improved:
            improved = False
            for i in range(4):
                for direction in (step, -step):
                    trial = list(params)
                    trial[i] = float(np.clip(trial[i] + direction,
                                             lo[i], hi[i]))
                    if trial[i] == params[i]:
                        continue
                    got = overlap(tuple(trial))
                    if got > best + 1e-4:
                        params, best = trial, got
                        improved = True
    return {"sx": params[0], "sy": params[1], "dx": params[2],
            "dy": params[3], "cx": cx, "cy": cy, "iou": round(best, 3)}


def compose_mapping(mapping: dict, extra: tuple[float, float, float, float]
                    ) -> dict:
    """Fold a further similarity (about the same centre) into a mapping."""
    sx, sy, dx, dy = extra
    return {**mapping,
            "sx": mapping["sx"] * sx, "sy": mapping["sy"] * sy,
            "dx": mapping["dx"] * sx + dx, "dy": mapping["dy"] * sy + dy}


def refine_photometric(views: list[View], mappings: list[dict],
                       uvs: list[np.ndarray], scores: list[np.ndarray],
                       order: list[int],
                       base_uvs: list[np.ndarray]) -> list[dict]:
    """Nudge each view's 2D mapping until overlapping views AGREE.

    The silhouette fit lines up the outline, and the outline only. Two views
    that both pass it can still disagree by ten pixels in the middle of the
    face — and at atlas resolution that renders as a second pair of lips at
    every seam. The silhouette carries no information about interior
    features; the photograph does. So after the outline fit, each view is
    nudged (scale and shift, tightly bounded) to minimise the colour
    difference against the views already accepted, on the vertices both see
    squarely — the same chained order the tone match uses, so the front
    view stays the anchor and never moves."""
    report: list[dict] = []
    for i in order[1:]:
        done = [j for j in order[:order.index(i)]]
        score_done = np.stack([scores[j] for j in done], axis=1)
        anchor = score_done.max(axis=1)
        anchor_view = np.array(done)[score_done.argmax(axis=1)]
        shared = (scores[i] > 0.3) & (anchor > 0.3)
        if shared.sum() < 300:
            report.append({"view": i, "skipped": "overlap too small"})
            continue
        idx = np.nonzero(shared)[0]
        if len(idx) > 8000:
            idx = idx[np.linspace(0, len(idx) - 1, 8000).astype(int)]
        ref = np.zeros((len(idx), 3))
        for j in done:
            pick = anchor_view[idx] == j
            if pick.any():
                ref[pick] = sample(views[j].filled, uvs[j][idx][pick])
        m = mappings[i]
        cx, cy = m["cx"], m["cy"]
        u0 = base_uvs[i][idx]

        def cost(extra: tuple[float, float, float, float], *,
                 m=m, cx=cx, cy=cy, u0=u0, img=views[i].filled,
                 ref=ref) -> float:
            mm = compose_mapping(m, extra)
            uu = np.stack(
                [cx + (u0[:, 0] - cx) * mm["sx"] + mm["dx"],
                 cy + (u0[:, 1] - cy) * mm["sy"] + mm["dy"]], axis=1)
            return float(np.abs(sample(img, uu) - ref).mean())

        params = [1.0, 1.0, 0.0, 0.0]
        best = start = cost(tuple(params))
        lo, hi = (0.94, 0.94, -0.025, -0.025), (1.06, 1.06, 0.025, 0.025)
        for step in (0.008, 0.004, 0.002, 0.001):
            improved = True
            while improved:
                improved = False
                for k in range(4):
                    for direction in (step, -step):
                        trial = list(params)
                        trial[k] = float(np.clip(trial[k] + direction,
                                                 lo[k], hi[k]))
                        if trial[k] == params[k]:
                            continue
                        got = cost(tuple(trial))
                        if got < best - 1e-4:
                            params, best = trial, got
                            improved = True
        if best < start * 0.99:
            mm = mappings[i] = compose_mapping(m, tuple(params))
            # Later views in the chain must anchor against the REFINED
            # position of this one, not where it used to be.
            uvs[i] = np.stack(
                [cx + (base_uvs[i][:, 0] - cx) * mm["sx"] + mm["dx"],
                 cy + (base_uvs[i][:, 1] - cy) * mm["sy"] + mm["dy"]], axis=1)
            report.append({"view": i, "colour_err": [round(start, 1),
                                                     round(best, 1)]})
        else:
            report.append({"view": i, "kept": round(start, 1)})
    return report


def project(mesh: trimesh.Trimesh, view: View, deg: float,
            intersector, mapping: dict | None = None
            ) -> tuple[np.ndarray, np.ndarray]:
    """(uv in [0,1] for this camera, per-vertex score).

    Score is the cosine of the viewing angle for vertices the camera can
    actually see, and 0 for the rest — so choosing the highest-scoring view
    per vertex picks the one looking at it most squarely."""
    verts = mesh.vertices
    _right, _up, forward = camera_basis(deg)
    facing = mesh.vertex_normals @ forward
    seen = facing > FACING_MIN
    if intersector is not None and seen.any():
        origins = verts + mesh.vertex_normals * (mesh.scale * 1e-3)
        directions = np.tile(forward, (len(verts), 1))
        try:
            blocked = intersector.intersects_any(origins, directions)
            seen &= ~blocked
        except Exception as exc:                       # noqa: BLE001
            print(f"note: occlusion test failed ({exc})", file=sys.stderr)

    uv = _base_uv(verts, view, deg)
    if mapping is not None:
        cx, cy = mapping["cx"], mapping["cy"]
        uv = np.stack([cx + (uv[:, 0] - cx) * mapping["sx"] + mapping["dx"],
                       cy + (uv[:, 1] - cy) * mapping["sy"] + mapping["dy"]],
                      axis=1)

    # A vertex that lands off the subject is not seen by this view, whatever
    # its normal says — otherwise the backdrop gets painted onto the model.
    h, w = view.mask.shape
    px = np.clip((uv[:, 0] * (w - 1)).round().astype(int), 0, w - 1)
    py = np.clip((uv[:, 1] * (h - 1)).round().astype(int), 0, h - 1)
    on_subject = view.mask[py, px]
    score = np.where(seen & on_subject, np.clip(facing, 0, 1), 0.0)
    return uv, score


def sample(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear sample; nearest-neighbour truncation was costing half a pixel
    everywhere and showing up as a soft edge on every seam."""
    h, w = img.shape[:2]
    x = np.clip(uv[:, 0] * (w - 1), 0, w - 1)
    y = np.clip(uv[:, 1] * (h - 1), 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bot = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def match_moments(src: np.ndarray, ref: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel gain and offset taking src's tone onto ref's, CLAMPED.

    Only ever a nudge: two views of one subject differ by exposure and white
    balance, not by an arbitrary transform, and an unclamped fit to a small
    overlap would happily invert a channel."""
    gain = np.clip(ref.std(axis=0) / np.maximum(src.std(axis=0), 1e-3),
                   0.7, 1.4)
    offset = np.clip(ref.mean(axis=0) - gain * src.mean(axis=0), -35.0, 35.0)
    return gain, offset


def harmonise(views: list[View], uvs: list[np.ndarray],
              scores: np.ndarray, order: list[int]) -> list[dict]:
    """Colour-match every view to the ones already accepted, in place.

    Chained outward from order[0] (the front): each view is matched on the
    vertices that it AND some already-harmonised view both see squarely —
    left against front along the arm they share, back against the sides. The
    front view itself is never altered; it is the reference exposure."""
    done = [order[0]]
    report: list[dict] = []
    for i in order[1:]:
        anchor_score = scores[:, done].max(axis=1)
        anchor_view = np.array(done)[scores[:, done].argmax(axis=1)]
        both = (scores[:, i] > 0.25) & (anchor_score > 0.25)
        if both.sum() < 300:
            report.append({"view": i, "skipped": "overlap too small"})
            done.append(i)
            continue
        src = sample(views[i].filled, uvs[i][both])
        ref = np.zeros_like(src)
        for j in done:
            pick = anchor_view[both] == j
            if pick.any():
                ref[pick] = sample(views[j].filled, uvs[j][both][pick])
        gain, offset = match_moments(src, ref)
        views[i].filled = np.clip(views[i].filled * gain + offset, 0, 255)
        report.append({"view": i, "overlap": int(both.sum()),
                       "gain": [round(float(g), 3) for g in gain],
                       "offset": [round(float(o), 1) for o in offset]})
        done.append(i)
    return report


def smooth_face_scores(mesh: trimesh.Trimesh,
                       face_score: np.ndarray) -> np.ndarray:
    """Average each face's per-view scores with its neighbours', a few rings.

    Raw per-face argmax flips view wherever two scores cross, which they do
    constantly on a curved surface — the result was islands of one view
    embedded in another, each island a slightly different exposure. Smoothing
    first makes the borders follow the geometry instead of the noise."""
    try:
        from scipy.sparse import coo_matrix
    except Exception:                                  # noqa: BLE001
        return face_score
    pairs = mesh.face_adjacency
    if len(pairs) == 0:
        return face_score
    f = len(mesh.faces)
    adj = coo_matrix(
        (np.ones(len(pairs) * 2),
         (np.concatenate([pairs[:, 0], pairs[:, 1]]),
          np.concatenate([pairs[:, 1], pairs[:, 0]]))), shape=(f, f)).tocsr()
    degree = np.maximum(np.asarray(adj.sum(axis=1)).ravel(), 1)[:, None]
    out = face_score.copy()
    for _ in range(3):
        out = 0.5 * out + 0.5 * (adj @ out) / degree
    return out


# --------------------------------------------------------------------------
# atlas
# --------------------------------------------------------------------------
def content_rect(view: View, uv_used: np.ndarray | None
                 ) -> tuple[float, float, float, float]:
    """The part of the frame worth spending texels on, in UV units.

    A staged photograph is mostly backdrop — the reference set measured the
    subject at 30-50% of the frame per axis — and resizing the WHOLE frame
    into an atlas tile spends over half the tile on grey. Cropping each tile
    to the subject (plus every UV the mesh actually maps into it, plus a
    margin) is a >2x gain in texels-on-subject per axis at the same atlas
    size: the difference between a face carried by ~250 pixels and one
    carried by ~600."""
    x0, y0, x1, y1 = view.bbox
    if uv_used is not None and len(uv_used):
        x0 = min(x0, float(uv_used[:, 0].min()))
        y0 = min(y0, float(uv_used[:, 1].min()))
        x1 = max(x1, float(uv_used[:, 0].max()))
        y1 = max(y1, float(uv_used[:, 1].max()))
    mx, my = 0.015 * (x1 - x0), 0.015 * (y1 - y0)
    return (max(0.0, x0 - mx), max(0.0, y0 - my),
            min(1.0, x1 + mx), min(1.0, y1 + my))


def build_atlas(images: list[np.ndarray], tile: int,
                rects: list[tuple[float, float, float, float]] | None = None
                ) -> tuple[Image.Image, int]:
    cols = 2 if len(images) > 1 else 1
    rows = (len(images) + cols - 1) // cols
    atlas = Image.new("RGB", (cols * tile, rows * tile), (128, 128, 128))
    for i, img in enumerate(images):
        pic = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        box = None
        if rects is not None:
            h, w = img.shape[:2]
            r = rects[i]
            box = (r[0] * (w - 1), r[1] * (h - 1),
                   r[2] * (w - 1) + 1, r[3] * (h - 1) + 1)
        pic = pic.resize((tile, tile), Image.LANCZOS, box=box)
        atlas.paste(pic, ((i % cols) * tile, (i // cols) * tile))
    return atlas, cols


def atlas_uv(uv: np.ndarray, index: int, cols: int, rows: int,
             tile: int) -> np.ndarray:
    """Local view UV into the atlas tile, in glTF convention (v runs up).

    The local UV is clamped a gutter short of the tile edge: a coordinate at
    exactly the border makes the sampler's bilinear kernel read the
    NEIGHBOURING view's tile, which drew a one-texel stripe of somebody
    else's photograph along every seam."""
    pad = UV_GUTTER / tile
    local = np.clip(uv, pad, 1.0 - pad)
    cx, cy = index % cols, index // cols
    u = (cx + local[:, 0]) / cols
    v = (cy + local[:, 1]) / rows
    return np.stack([u, 1.0 - v], axis=1)


SMOOTH_ITERATIONS = 8
SMOOTH_WHEN_ROUGHER_THAN = 12.0   # degrees, mean adjacent-face normal angle


def smooth_mesh(mesh: trimesh.Trimesh) -> dict:
    """Take real roughness off the surface — and prove it did, or undo it.

    Every step here is measured, because the obvious version of this function
    was measured making things WORSE. The production mesher (surface nets)
    already relaxes its vertices: the reference figure measured a 6.3° mean
    angle between adjacent face normals — smooth — and a blanket Taubin pass
    RAISED that to 22.8° while the same filter cleaned a noisy sphere just
    fine (11.6° → 2.9°). Something in AI-mesher topology (degenerate slivers,
    odd valences) upsets a uniform Laplacian, and a smoother that cannot
    check its own work would have shipped the damage.

    So: skip when the surface is already smooth; drop degenerate faces
    first; use Humphrey (measured best on the sphere: 11.6° → 1.6°, volume
    1.0000); and REVERT unless the roughness actually fell."""
    mesh.merge_vertices()

    def roughness() -> float:
        angles = mesh.face_adjacency_angles
        return float(np.degrees(angles).mean()) if len(angles) else 0.0

    before = roughness()
    if before < SMOOTH_WHEN_ROUGHER_THAN:
        return {"applied": False, "roughness": round(before, 2),
                "note": "already smooth"}
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    kept = mesh.vertices.copy()
    trimesh.smoothing.filter_humphrey(mesh, alpha=0.1, beta=0.5,
                                      iterations=SMOOTH_ITERATIONS)
    after = roughness()
    if after >= before:
        mesh.vertices = kept
        return {"applied": False, "roughness": round(before, 2),
                "note": "smoothing did not improve this surface; reverted"}
    return {"applied": True, "roughness_before": round(before, 2),
            "roughness_after": round(after, 2)}


def retexture(mesh: trimesh.Trimesh, views: list[View], tile: int,
              synth_azimuths: set[float] | frozenset[float] = frozenset()
              ) -> dict:
    # Refuse to paint from views that are not photographs of the subject.
    # If every view fails, keep the single least-bad one: a mesh coloured
    # from one imperfect photograph still beats a bare grey mesh, and the
    # report says exactly what happened.
    dropped: list[dict] = []
    kept: list[View] = []
    for view in views:
        quality = view_quality(view.image, view.mask_raw)
        reason = usable(quality)
        if reason is None:
            kept.append(view)
        else:
            dropped.append({"azimuth": view.azimuth, "reason": reason,
                            **quality})
            print(f"note: view at {view.azimuth}° dropped — {reason}",
                  file=sys.stderr)
    if not kept and views:
        def least_bad(v: View) -> float:
            q = view_quality(v.image, v.mask_raw)
            # Low flatness AND a decent amount of subject: a big imperfect
            # view beats a textured sliver of nothing.
            return q["slab_share"] - q["area"]
        best = min(views, key=least_bad)
        kept = [best]
        dropped = [d for d in dropped if d["azimuth"] != best.azimuth]
    views = kept

    repaired = repair_normals(mesh)
    smoothing = smooth_mesh(mesh)
    # Centre ONCE, in place. Projecting from a centred copy while ray-testing
    # against the uncentred original silently offsets every ray by the
    # centroid, which reads as "occluded" almost everywhere.
    mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)
    verts = mesh.vertices
    angles, fit, detail = solve_orientation(verts, views)

    intersector = None
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        intersector = RayMeshIntersector(mesh)
    except Exception as exc:                           # noqa: BLE001
        print(f"note: no ray engine ({exc}); facing test only", file=sys.stderr)

    mappings, uvs, scores, base_uvs = [], [], [], []
    for deg, view in zip(angles, views, strict=False):
        mapping = solve_view_mapping(mesh, view, deg)
        uv, sc = project(mesh, view, deg, intersector, mapping)
        mappings.append(mapping)
        uvs.append(uv)
        scores.append(sc)
        base_uvs.append(_base_uv(mesh.vertices, view, deg))
    score = np.stack(scores, axis=1)                   # (V, views)
    seen = score.max(axis=1) > 0
    view_iou = [m["iou"] for m in mappings]
    aligned_fit = float(np.mean(view_iou)) if view_iou else fit

    # --- one exposure across the set, chained outward from the first view.
    order = [0] + sorted(
        range(1, len(views)),
        key=lambda i: min(abs(angles[i] - angles[0]),
                          360 - abs(angles[i] - angles[0])))
    tone = harmonise(views, uvs, score, order) if len(views) > 1 else []

    # --- interior alignment: the outline fit cannot see a feature ten
    # pixels adrift inside the silhouette; the colour difference can.
    photometric: list[dict] = []
    if len(views) > 1:
        photometric = refine_photometric(
            views, mappings, uvs,
            [score[:, j] for j in range(len(views))], order, base_uvs)
        uvs, scores = [], []
        for k, (deg, view) in enumerate(zip(angles, views, strict=False)):
            uv, sc = project(mesh, view, deg, intersector, mappings[k])
            uvs.append(uv)
            scores.append(sc)
        score = np.stack(scores, axis=1)
        seen = score.max(axis=1) > 0

    # --- per-face view: the worst corner decides visibility, smoothing
    # decides among the views that qualify.
    faces = mesh.faces
    face_score = score[faces].min(axis=1)              # (F, views)
    face_seen = face_score.max(axis=1) > 0
    # Squared before smoothing: monotone, so no face's own preference flips,
    # but in the neighbourhood average the squarest view now outweighs an
    # oblique one quadratically — fewer single-face islands of a worse view
    # surviving the smoothing, which is what left seams mid-face.
    # A GENERATED view is evidence of last resort: it exists to cover arcs
    # no camera saw, not to outvote a photograph. The 0.92 handicap means a
    # real view within ~23° of squareness still beats a perfectly square
    # invented one; beyond that the invented view wins, which is the point.
    prio = np.array([0.92 if v.azimuth in synth_azimuths else 1.0
                     for v in views])
    smoothed = smooth_face_scores(mesh, (face_score * prio) ** 2)
    smoothed[face_score <= 0] = -1.0     # never pick a view missing a corner
    face_view = smoothed.argmax(axis=1)
    # Faces no camera sees at all: argmax over a row of zeros used to hand
    # every hidden face the FRONT view's coordinates — bands of unrelated
    # skin across the model. Fall back to the camera the face most nearly
    # points at instead.
    if (~face_seen).any():
        forwards = np.stack([camera_basis(a)[2] for a in angles], axis=1)
        face_view[~face_seen] = (
            mesh.face_normals[~face_seen] @ forwards).argmax(axis=1)

    # --- split only the vertices whose faces disagree about which view to use
    pairs = np.stack([faces.ravel(), np.repeat(face_view, 3)], axis=1)
    uniq, inverse = np.unique(pairs, axis=0, return_inverse=True)
    new_faces = inverse.reshape(-1, 3)
    new_verts = mesh.vertices[uniq[:, 0]]

    rows = (len(views) + 1) // 2 if len(views) > 1 else 1
    cols = 2 if len(views) > 1 else 1
    # Every tile is cropped to its subject plus whatever the mesh actually
    # maps into it, so the atlas spends its texels on the person, not on the
    # backdrop. UVs are expressed in crop-local coordinates to match.
    rects = []
    for i in range(len(views)):
        sel_faces = face_view == i
        used_uv = (uvs[i][np.unique(faces[sel_faces])]
                   if sel_faces.any() else None)
        rects.append(content_rect(views[i], used_uv))
    new_uv = np.zeros((len(uniq), 2))
    for i in range(len(views)):
        sel = uniq[:, 1] == i
        if sel.any():
            r = rects[i]
            span = np.array([max(r[2] - r[0], 1e-6), max(r[3] - r[1], 1e-6)])
            local = (uvs[i][uniq[sel, 0]] - np.array([r[0], r[1]])) / span
            new_uv[sel] = atlas_uv(np.clip(local, 0.0, 1.0),
                                   i, cols, rows, tile)

    # The FLOODED (and now tone-matched) images go in the atlas: a texel that
    # falls a pixel outside the silhouette reads as skin or hair rather than
    # as backdrop.
    atlas, _ = build_atlas([v.filled for v in views], tile, rects)
    textured = trimesh.Trimesh(vertices=new_verts, faces=new_faces,
                               process=False)
    textured.visual = trimesh.visual.TextureVisuals(
        uv=new_uv,
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=atlas, metallicFactor=0.0, roughnessFactor=0.9))

    best_view = score.argmax(axis=1)
    return {"mesh": textured, "seen": seen,
            "fit": aligned_fit, "detail": detail, "repaired": repaired,
            "view_iou": view_iou, "tone": tone, "dropped": dropped,
            "smoothing": smoothing,
            "views_used": [v.azimuth for v in views],
            "synthetic_views": sorted(a for a in synth_azimuths
                                      if any(v.azimuth == a for v in views)),
            "tile_crops": [[round(c, 3) for c in r] for r in rects],
            "photometric": photometric,
            "mappings": [{k: round(float(v), 5) for k, v in m.items()}
                         for m in mappings],
            "per_view": [int((best_view[seen] == i).sum())
                         for i in range(len(views))]}


# --------------------------------------------------------------------------
def main() -> int:
    argv = sys.argv[1:]
    views: list[tuple[float, Path]] = []
    positional: list[str] = []
    synth: set[float] = set()
    tile = 2048     # sources are upscaled to ~2300px and tiles are cropped
                    # to content, so 1024 was the resolution bottleneck
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--view":
            try:
                az, _, path = argv[i + 1].partition(":")
                views.append((float(az), Path(path)))
            except (IndexError, ValueError):
                print(__doc__)
                return 2
            i += 2
        elif a == "--synth":
            try:
                synth = {float(s) for s in argv[i + 1].split(",") if s}
            except (IndexError, ValueError):
                print(__doc__)
                return 2
            i += 2
        elif a == "--tile":
            try:
                tile = int(argv[i + 1])
            except (IndexError, ValueError):
                print(__doc__)
                return 2
            i += 2
        elif a.startswith("--"):
            i += 1
        else:
            positional.append(a)
            i += 1
    if len(positional) < 2 or not views:
        print(__doc__)
        return 2

    mesh_path, out_path = Path(positional[0]), Path(positional[1])
    mesh = load_mesh(mesh_path)
    loaded = [View(az, np.asarray(Image.open(p).convert("RGB")))
              for az, p in views]

    result = retexture(mesh, loaded, tile, synth)
    textured = result["mesh"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    textured.export(str(out_path))

    seen = result["seen"]
    report = {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "textured_vertices": int(len(textured.vertices)),
        "views": len(result["views_used"]),
        "seen_pct": round(float(seen.mean()) * 100, 1),
        "per_view_vertices": result["per_view"],
        "orientation_iou": round(float(result["fit"]), 3),
        "orientation": result["detail"],
        "view_iou": result["view_iou"],
        "tone": result["tone"],
        "views_used": result["views_used"],
        "views_dropped": result["dropped"],
        "synthetic_views": result["synthetic_views"],
        "smoothing": result["smoothing"],
        "tile_crops": result["tile_crops"],
        "photometric": result["photometric"],
        "mappings": result["mappings"],
        "winding_repaired": bool(result["repaired"]),
        "atlas_px": [tile * (2 if len(result["views_used"]) > 1 else 1),
                     tile * max(1, (len(result["views_used"]) + 1) // 2
                                if len(result["views_used"]) > 1 else 1)],
        "bytes": out_path.stat().st_size,
    }
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
