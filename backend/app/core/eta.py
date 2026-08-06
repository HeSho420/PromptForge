"""Multi-factor render-time estimation.

The estimate starts from a per-job-type baseline — the median of this
machine's own past renders when at least three exist, otherwise a hardware-
tier heuristic — and scales it by what the CURRENT job actually asks for:
resolution, steps, video length, checkpoint family, extra conditioning nodes
(LoRA / ControlNet / IPAdapter), batch size, denoise strength, plus the live
state of the machine (GPU busy, RAM pressure) and the queue ahead of the job.

The UI shows only "Estimated time remaining" — never the math.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from .hardware import Hardware

# Baseline render seconds per job kind at the "mid" tier (~8 GB VRAM),
# used only until real history exists.
BASE_SECONDS = {"image_edit": 45, "workflow": 60, "video": 480,
                "avatar": 300, "avatar_render": 120}
TIER_FACTOR = {"low": 1.9, "mid": 1.0, "high": 0.5}

# Reference payloads the baselines correspond to (factors scale against these).
REF_PIXELS = 512 * 512
REF_STEPS = 30
REF_VIDEO_LEN = 49
REF_VIDEO_PIXELS = 640 * 640

# Node types that add meaningful per-step cost when present in a graph.
_COND_NODES = ("LoraLoader", "ControlNetLoader", "ControlNetApply",
               "IPAdapter", "PhotoMakerLoader")


@dataclass
class SystemLoad:
    gpu_util_pct: float = 0.0
    ram_load_pct: float = 0.0


def probe_load(timeout_s: float = 3.0) -> SystemLoad:
    """Best-effort snapshot of live GPU utilization and RAM pressure."""
    gpu = 0.0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout_s, check=True,
        ).stdout.strip().splitlines()[0]
        gpu = float(out)
    except Exception:  # noqa: BLE001 — no GPU / driver: treat as idle
        pass
    ram = 0.0
    if sys.platform == "win32":
        try:
            import ctypes

            class MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = MemStatus()
            status.dwLength = ctypes.sizeof(MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            ram = float(status.dwMemoryLoad)
        except Exception:  # noqa: BLE001
            pass
    return SystemLoad(gpu_util_pct=gpu, ram_load_pct=ram)


def _graph_factors(graph: dict[str, Any] | None) -> float:
    """Extra cost from conditioning/adapter nodes and batch size in a graph."""
    if not graph:
        return 1.0
    factor = 1.0
    for node in graph.values():
        ctype = str(node.get("class_type", ""))
        if any(ctype.startswith(c) for c in _COND_NODES):
            factor += 0.15  # each adds a per-step overhead
        ins = node.get("inputs", {})
        if ctype == "EmptyLatentImage":
            try:
                factor *= max(1, int(ins.get("batch_size", 1) or 1))
            except (TypeError, ValueError):
                pass
        if ctype == "KSampler":
            try:
                denoise = float(ins.get("denoise", 1.0) or 1.0)
                factor *= max(0.25, min(denoise, 1.0))
            except (TypeError, ValueError):
                pass
    return factor


def estimate_seconds(
    job_type: str,
    *,
    hardware: Hardware,
    history: list[float] | None = None,
    payload: dict[str, Any] | None = None,
    graph: dict[str, Any] | None = None,
    checkpoint: str | None = None,
    load: SystemLoad | None = None,
    queue_ahead_seconds: float = 0.0,
) -> float:
    """Predicted seconds until this job finishes (including queue wait)."""
    p = payload or {}
    durations = sorted(history or [])
    if len(durations) >= 3:
        base = durations[len(durations) // 2]  # median of real runs
    else:
        base = (BASE_SECONDS.get(job_type, 60)
                * TIER_FACTOR.get(hardware.tier, 1.0))

    factor = 1.0
    # Resolution: quadratic-ish cost in pixel count.
    try:
        w = int(p.get("width", 0) or 0)
        h = int(p.get("height", 0) or 0)
    except (TypeError, ValueError):
        w = h = 0
    if w and h:
        ref = REF_VIDEO_PIXELS if job_type in ("video",) else REF_PIXELS
        factor *= max(0.3, (w * h) / ref)
    # Video length: linear in frames.
    if job_type in ("video",) or p.get("video"):
        try:
            length = int(p.get("length", REF_VIDEO_LEN) or REF_VIDEO_LEN)
        except (TypeError, ValueError):
            length = REF_VIDEO_LEN
        factor *= max(0.25, length / REF_VIDEO_LEN)
    # Steps (when the payload/graph carries them).
    try:
        steps = int(p.get("steps", 0) or 0)
    except (TypeError, ValueError):
        steps = 0
    if steps:
        factor *= max(0.3, steps / REF_STEPS)
    # Checkpoint family: SDXL-class models are ~2x SD1.5-class per step.
    if checkpoint and ("xl" in checkpoint.lower() or "sdxl" in checkpoint.lower()):
        factor *= 2.0
    factor *= _graph_factors(graph)

    # Live machine state: a busy GPU or paging RAM slows everything down.
    if load is not None:
        if load.gpu_util_pct >= 50:
            factor *= 1.0 + (load.gpu_util_pct / 100) * 0.5
        if load.ram_load_pct >= 85:
            factor *= 1.3

    return max(5.0, base * factor + max(0.0, queue_ahead_seconds))


def human_time(seconds: float) -> str:
    s = int(round(seconds))
    if s < 90:
        return f"{max(5, s)}s"
    m = s / 60
    return f"{m:.0f} min" if m < 10 else f"{int(round(m / 5) * 5)} min"
