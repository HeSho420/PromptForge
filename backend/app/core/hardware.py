"""Hardware detection + tiering — the program adapts to the machine it's on.

probe() measures VRAM (nvidia-smi), RAM and free disk with stdlib-only code;
the tier drives which local LLM the launcher pulls, how large a checkpoint
the scout may auto-download, and what the first-run setup pre-stages.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Hardware:
    gpu_name: str | None
    vram_gb: float
    ram_gb: float
    disk_free_gb: float

    @property
    def tier(self) -> str:
        if self.vram_gb >= 16:
            return "high"
        if self.vram_gb >= 6:
            return "mid"
        return "low"

    def to_dict(self) -> dict:
        return {**asdict(self), "tier": self.tier}


def _probe_gpu() -> tuple[str | None, float]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8, check=True,
        ).stdout.strip().splitlines()[0]
        name, total = [p.strip() for p in out.split(",", 1)]
        return name, round(float(total) / 1024, 1)
    except Exception:  # noqa: BLE001 — no NVIDIA GPU / driver
        return _probe_gpu_registry()


def _probe_gpu_registry() -> tuple[str | None, float]:
    """AMD/Intel VRAM from the display-class registry keys.

    nvidia-smi answers for one brand only; an RX 6700 XT with 12 GB was
    reporting 0 and every VRAM-gated tier treated a capable machine as
    GPU-less. HardwareInformation.qwMemorySize is the value the driver
    itself writes (the WMI AdapterRAM field is a 32-bit relic that caps
    at 4 GB)."""
    if sys.platform != "win32":
        return None, 0.0
    try:
        import winreg
        best_name, best_bytes = None, 0
        cls = (r"SYSTEM\CurrentControlSet\Control\Class"
               r"\{4d36e968-e325-11ce-bfc1-08002be10318}")
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, cls) as root_key:
            for i in range(64):
                try:
                    sub = winreg.EnumKey(root_key, i)
                except OSError:
                    break
                if not sub.isdigit():
                    continue
                try:
                    with winreg.OpenKey(root_key, sub) as dev:
                        size, _ = winreg.QueryValueEx(
                            dev, "HardwareInformation.qwMemorySize")
                        name, _ = winreg.QueryValueEx(
                            dev, "DriverDesc")
                except OSError:
                    continue
                if isinstance(size, int) and size > best_bytes:
                    best_bytes, best_name = size, str(name)
        if best_bytes:
            return best_name, round(best_bytes / 1024**3, 1)
    except Exception:  # noqa: BLE001 — probing is advisory
        pass
    return None, 0.0


def _win_mem_status():
    """GlobalMemoryStatusEx snapshot on Windows, None elsewhere/on failure."""
    if sys.platform != "win32":
        return None
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
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(
                ctypes.byref(status)):
            return None
        return status
    except Exception:  # noqa: BLE001 — probing is advisory
        return None


def _probe_ram_gb() -> float:
    status = _win_mem_status()
    if status is not None:
        return round(status.ullTotalPhys / 1024**3, 1)
    try:
        import os
        return round(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")  # type: ignore[attr-defined]
            / 1024**3, 1)
    except (ValueError, OSError, AttributeError):
        return 8.0


def ram_stats() -> tuple[float, float] | None:
    """(used_gb, total_gb) right now, or None off-Windows/on failure.

    Shared with LAN peers so the Network view can show what each machine
    has left before handing it a render."""
    status = _win_mem_status()
    if status is None:
        return None
    total = status.ullTotalPhys / 1024**3
    return round(total - status.ullAvailPhys / 1024**3, 1), round(total, 1)


def available_commit_gb() -> float | None:
    """Windows: commit charge still available (free RAM + paging-file
    headroom), in GB. Model loads draw from this budget — when it runs out
    Windows refuses the allocation with OS error 1455 ("the paging file is
    too small"), which is how a ~10 GB WAN video load dies on 16 GB
    machines. None on other platforms or on probe failure."""
    status = _win_mem_status()
    if status is None:
        return None
    return status.ullAvailPageFile / 1024**3


def probe(data_dir: Path | None = None) -> Hardware:
    name, vram = _probe_gpu()
    disk_root = data_dir if data_dir and data_dir.exists() else Path.cwd()
    return Hardware(
        gpu_name=name,
        vram_gb=vram,
        ram_gb=_probe_ram_gb(),
        disk_free_gb=round(shutil.disk_usage(disk_root).free / 1024**3, 1),
    )


def llm_model_for(hw: Hardware) -> str:
    """Which Ollama planning model fits this machine."""
    if hw.tier == "high":
        return "qwen2.5:14b"
    if hw.tier == "mid" and hw.ram_gb >= 12:
        return "qwen2.5:7b"
    return "qwen2.5:3b"


def render_budget(hw: Hardware) -> dict:
    """What this machine can render without hard-crashing the GPU. Used both
    to TELL the planner LLM to use the machine fully and to CLAMP generated
    graphs that overshoot (an OOM inside ComfyUI can take the whole server
    down, which is exactly what must never happen again)."""
    if hw.tier == "high":
        return {"max_side": 1536, "max_pixels": 1536 * 1536, "max_steps": 60,
                "max_batch": 2, "max_video_len": 81, "max_video_side": 768}
    if hw.tier == "mid":
        return {"max_side": 1280, "max_pixels": 1024 * 1024, "max_steps": 50,
                "max_batch": 1, "max_video_len": 81, "max_video_side": 768}
    return {"max_side": 768, "max_pixels": 768 * 768, "max_steps": 40,
            "max_batch": 1, "max_video_len": 33, "max_video_side": 512}


def max_auto_download_bytes(hw: Hardware) -> int:
    """Cap for scout auto-downloads: a checkpoint the GPU can plausibly run
    (with offload headroom), never more than half the free disk."""
    by_vram = int(max(3.0, hw.vram_gb * 1.5) * 1024**3)
    by_disk = int(hw.disk_free_gb * 1024**3 // 2)
    return max(1 * 1024**3, min(by_vram, by_disk, 12 * 1024**3))
