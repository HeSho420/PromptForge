"""Motion transfer: sizing a clip the machine can actually render, and
stitching it back together when it has to be rendered in pieces.

VACE holds every frame of a window in VRAM at once — there is no internal
windowing — so a long clip is not "slower", it is impossible. The answer is to
render overlapping windows and cross-fade them, which costs time instead of
memory and keeps quality intact.

The numbers here are measured on the target machine (RTX 4060 Laptop, 8 GB
VRAM, 16 GB system RAM), not estimated:

  480x480 x 25 frames  -> 268 s, peak 7.58 GB VRAM   (comfortable)
  480x832 x 33 frames  -> 744 s, peak 7.99 GB VRAM   (the ceiling; 0 GB RAM free)

The binding constraint turned out to be SYSTEM RAM, not VRAM: the 6.3 GB text
encoder loads into RAM that is already largely committed, and the machine
page-thrashes for minutes before sampling starts. An identical render was 28%
faster purely because it began with 5.9 GB of RAM free instead of 0.03 GB.
"""
from __future__ import annotations

from PIL import Image

# WanVaceToVideo's latent maths is ((length - 1) // 4) + 1: a length that is
# not 4n+1 is rejected outright.
FRAME_STEP = 4

# width * height * frames the machine survived, from the 480x832x33 run that
# peaked at 7.99 of 8.00 GB. Treated as a hard ceiling, not a target.
MAX_PIXEL_FRAMES = 13_178_880


def align_length(frames: int, *, minimum: int = 5) -> int:
    """The nearest valid VACE length at or below `frames` (always 4n+1)."""
    frames = max(minimum, int(frames))
    return frames - ((frames - 1) % FRAME_STEP)


def fit_window(width: int, height: int, frames: int,
               ceiling: int = MAX_PIXEL_FRAMES) -> int:
    """How many frames of this size fit in one render, as a valid length.

    Guards on the PRODUCT width x height x frames rather than on each axis:
    the per-axis limits would happily permit 512x512x33, which is 13% more
    pixel-frames than the run that already peaked at 7.99 of 8.00 GB."""
    per_frame = max(1, int(width) * int(height))
    return align_length(min(int(frames), max(FRAME_STEP + 1,
                                             ceiling // per_frame)))


def plan_chunks(total: int, window: int,
                overlap: int = 8) -> list[tuple[int, int]]:
    """Split `total` driving frames into overlapping windows.

    Each window is a valid VACE length, and consecutive windows share
    `overlap` frames so the render can be cross-faded rather than cut — a hard
    cut between two independently sampled windows is instantly visible.
    Returns [(start, end)] with end exclusive."""
    total = max(1, int(total))
    window = align_length(min(window, total))
    if total <= window:
        return [(0, total)]
    overlap = max(2, min(int(overlap), window // 2))
    stride = window - overlap
    chunks: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(total, start + window)
        # A final sliver would render worse than it is worth: pull the last
        # window back so it is full length and simply overlaps more.
        if end - start < window and chunks:
            start = max(0, total - window)
            end = total
            if chunks[-1][0] == start:
                break
        chunks.append((start, end))
        if end >= total:
            break
        start += stride
    return chunks


def crossfade(a: list[Image.Image], b: list[Image.Image],
              overlap: int) -> list[Image.Image]:
    """Join two rendered windows that share `overlap` frames.

    The shared frames are blended from all-A to all-B so the join is a fade
    rather than a cut. Both windows saw the same reference image, so they
    agree on appearance; what differs is sampling noise, and that is exactly
    what a fade hides."""
    if not a:
        return list(b)
    if not b:
        return list(a)
    overlap = max(0, min(int(overlap), len(a), len(b)))
    if overlap == 0:
        return list(a) + list(b)
    out = list(a[:len(a) - overlap])
    for i in range(overlap):
        # i = 0 is still essentially A, i = overlap-1 is essentially B.
        t = (i + 1) / (overlap + 1)
        fa = a[len(a) - overlap + i].convert("RGB")
        fb = b[i].convert("RGB")
        if fb.size != fa.size:
            fb = fb.resize(fa.size, Image.Resampling.LANCZOS)
        out.append(Image.blend(fa, fb, t))
    out.extend(b[overlap:])
    return out


def assemble(windows: list[list[Image.Image]],
             chunks: list[tuple[int, int]]) -> list[Image.Image]:
    """Stitch every rendered window back into one continuous clip, using each
    pair's real overlap (which the last window may have enlarged)."""
    out: list[Image.Image] = []
    for i, frames in enumerate(windows):
        if not frames:
            continue
        if not out:
            out = list(frames)
            continue
        prev_end = chunks[i - 1][1]
        start = chunks[i][0]
        out = crossfade(out, list(frames), max(0, prev_end - start))
    return out
