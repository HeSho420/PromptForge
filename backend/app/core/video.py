"""Reading and writing real video files.

The app can already RENDER video, but until now nothing could read one back
in — assets were images, and a clip only ever left the program as an animated
WEBP. Motion transfer needs the opposite direction: an uploaded video has to
become frames the renderer can look at, and frames have to become a clip again.

ffmpeg comes from the `imageio-ffmpeg` wheel, which ships a static binary. That
matters on this machine: there is no ffmpeg on PATH and no decoder in the
environment, so anything relying on a system install would work here and fail
on the user's other computer.

Every function is honest about failure: a file that cannot be decoded raises
VideoError with ffmpeg's own message rather than returning something empty.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


class VideoError(RuntimeError):
    """A video file could not be read or written."""


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    duration_s: float

    @property
    def frame_count(self) -> int:
        return max(1, int(round(self.duration_s * self.fps)))

    def to_dict(self) -> dict[str, Any]:
        return {"width": self.width, "height": self.height,
                "fps": round(self.fps, 3),
                "duration_s": round(self.duration_s, 3),
                "frames": self.frame_count}


def _ffmpeg():
    try:
        import imageio_ffmpeg
    except ImportError as exc:  # pragma: no cover - install is launcher-managed
        raise VideoError(
            "Video support needs the imageio-ffmpeg package. Restart "
            "PromptForge through its launcher and it will install itself."
        ) from exc
    return imageio_ffmpeg


def probe(path: str | Path) -> VideoInfo:
    """Size, frame rate and duration of a video file."""
    iio = _ffmpeg()
    reader = None
    try:
        reader = iio.read_frames(str(path))
        meta = next(reader)
    except StopIteration as exc:
        raise VideoError("The video file contains no readable frames.") from exc
    except Exception as exc:  # noqa: BLE001 — ffmpeg's own words are the message
        raise VideoError(f"Could not read the video: {exc}") from exc
    finally:
        # A rejected file leaves ffmpeg's pipes open otherwise, and on Windows
        # the handle keeps the file locked against the caller's own cleanup.
        if reader is not None:
            reader.close()
    width, height = meta.get("size", (0, 0))
    fps = float(meta.get("fps") or 0)
    duration = float(meta.get("duration") or 0)
    if not width or not height or fps <= 0:
        raise VideoError("The file does not look like a video "
                         "(no frame size or frame rate).")
    return VideoInfo(int(width), int(height), fps, duration)


def sample_indices(total: int, wanted: int) -> list[int]:
    """`wanted` frame indices spread evenly across `total` frames.

    Used when a driving video has more frames than the renderer can afford:
    taking every Nth frame keeps the WHOLE motion (start to finish) instead of
    truncating it, which would silently render the first second of a dance and
    call it done."""
    if wanted <= 0 or total <= 0:
        return []
    if wanted >= total:
        return list(range(total))
    step = total / wanted
    out: list[int] = []
    for i in range(wanted):
        idx = min(total - 1, int(math.floor(i * step)))
        if not out or idx != out[-1]:
            out.append(idx)
    return out


def read_frames(path: str | Path, *, max_frames: int | None = None,
                every: int = 1) -> list[Image.Image]:
    """Decode a video into PIL frames.

    `every` keeps one frame in N (the cheap way to halve a frame rate).
    `max_frames` then samples evenly across whatever is left, so a long clip
    is thinned rather than cut short."""
    iio = _ffmpeg()
    info = probe(path)
    frames: list[Image.Image] = []
    reader = None
    try:
        reader = iio.read_frames(str(path), pix_fmt="rgb24")
        meta = next(reader)
        width, height = meta["size"]
        for i, raw in enumerate(reader):
            if every > 1 and i % every:
                continue
            frames.append(Image.frombytes("RGB", (width, height), raw))
    except Exception as exc:  # noqa: BLE001 — surface ffmpeg's reason
        raise VideoError(f"Could not decode the video: {exc}") from exc
    finally:
        if reader is not None:
            reader.close()
    if not frames:
        raise VideoError("The video decoded to zero frames.")
    if max_frames and len(frames) > max_frames:
        frames = [frames[i] for i in sample_indices(len(frames), max_frames)]
    del info  # probed for the honest error message above
    return frames


def write_video(frames: list[Image.Image], path: str | Path,
                fps: float = 16.0, quality: int = 7) -> Path:
    """Encode frames to an H.264 mp4 — a real video file, playable anywhere,
    unlike the animated WEBP the render path used to emit."""
    if not frames:
        raise VideoError("Nothing to encode: no frames were produced.")
    iio = _ffmpeg()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # H.264 needs even dimensions; ffmpeg's macro_block_size would silently
    # rescale otherwise, which shows up as a soft, slightly-wrong-size clip.
    w, h = frames[0].size
    w, h = w - (w % 2), h - (h % 2)
    size = (w, h)
    writer = None
    try:
        writer = iio.write_frames(str(out), size, fps=max(1.0, float(fps)),
                                  quality=quality, macro_block_size=1)
        writer.send(None)
        for frame in frames:
            rgb = frame.convert("RGB")
            if rgb.size != size:
                rgb = rgb.resize(size, Image.Resampling.LANCZOS)
            writer.send(rgb.tobytes())
    except Exception as exc:  # noqa: BLE001 — surface ffmpeg's reason
        raise VideoError(f"Could not write the video: {exc}") from exc
    finally:
        if writer is not None:
            writer.close()
    return out


def encode_animation(frames: list[Image.Image], fps: float = 24.0) -> bytes:
    """Frames as an animated WEBP that still has all of them when read back.

    One writer for every path that emits one, because getting this wrong is
    invisible. A LOSSY animated WEBP drops frames it considers near-identical
    — measured, 25 frames in and 23 out — and nothing raises: the clip is
    simply, quietly, shorter than the render that produced it.

    So: lossless, and then counted. If the round trip loses a frame the caller
    hears about it instead of shipping a clip that is silently wrong."""
    if not frames:
        raise VideoError("Nothing to encode: no frames were produced.")
    buf = io.BytesIO()
    frames[0].convert("RGB").save(
        buf, format="WEBP", save_all=True,
        append_images=[f.convert("RGB") for f in frames[1:]],
        duration=max(1, int(1000 / max(1.0, fps))), loop=0,
        lossless=True, quality=100, method=0)
    data = buf.getvalue()
    written = count_animation_frames(data)
    if written != len(frames):
        raise VideoError(
            f"The animation lost frames on the way to disk: {len(frames)} in, "
            f"{written} out. That is the lossy-WEBP frame-dropping failure, "
            "and it must never pass silently.")
    return data


def count_animation_frames(data: bytes) -> int:
    """How many frames an encoded animation actually contains."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            return getattr(im, "n_frames", 1)
    except Exception:  # noqa: BLE001 — a count we cannot take is not a crash
        return -1


def thumbnail(path: str | Path, size: int = 512) -> Image.Image:
    """One representative frame, taken a little way in — frame 0 of a real
    clip is very often black or a fade-in."""
    iio = _ffmpeg()
    reader = None
    try:
        reader = iio.read_frames(str(path), pix_fmt="rgb24")
        meta = next(reader)
        width, height = meta["size"]
        wanted = 0
        fps = float(meta.get("fps") or 0)
        duration = float(meta.get("duration") or 0)
        if fps > 0 and duration > 0:
            wanted = min(int(fps * duration) - 1, int(fps))  # ~1 second in
        frame = None
        for i, raw in enumerate(reader):
            frame = raw
            if i >= max(0, wanted):
                break
    except Exception as exc:  # noqa: BLE001 — surface ffmpeg's reason
        raise VideoError(f"Could not read a frame from the video: {exc}") from exc
    finally:
        if reader is not None:
            reader.close()
    if frame is None:
        raise VideoError("The video decoded to zero frames.")
    image = Image.frombytes("RGB", (int(width), int(height)), frame)
    image.thumbnail((size, size))
    return image
