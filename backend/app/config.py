"""Application configuration.

Everything is env-overridable so the same code runs in dev, tests and CI.
Tests point PROMPTFORGE_DATA_DIR at a temp directory to stay hermetic.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


@dataclass
class Settings:
    data_dir: Path = field(
        default_factory=lambda: _env_path("PROMPTFORGE_DATA_DIR", PROJECT_ROOT / "data")
    )
    comfyui_url: str = field(
        default_factory=lambda: os.environ.get("PROMPTFORGE_COMFYUI_URL", "http://127.0.0.1:8188")
    )
    # Where ComfyUI is installed — lets the backend restart it automatically
    # when it crashes mid-session. The launcher sets this; the fallback covers
    # the common per-user install.
    comfyui_dir: str = field(
        default_factory=lambda: os.environ.get(
            "PROMPTFORGE_COMFYUI_DIR",
            str(p) if (p := Path.home() / "ComfyUI").exists() else "")
    )
    # Which inpainting backend to use: "comfyui" (default) or "mock".
    # Real backends are the default; mocks are opt-in for tests/demos only.
    inpaint_backend: str = field(
        default_factory=lambda: os.environ.get("PROMPTFORGE_INPAINT_BACKEND", "comfyui")
    )
    # Which segmentation backend to use: "sam" (default) or "mock".
    segment_backend: str = field(
        default_factory=lambda: os.environ.get("PROMPTFORGE_SEGMENT_BACKEND", "sam")
    )
    # Local LLM (OpenAI-compatible chat endpoint: Ollama, LM Studio, llama.cpp)
    # used for AI workflow generation. Prompts stay local on this path.
    llm_url: str = field(
        default_factory=lambda: os.environ.get(
            "PROMPTFORGE_LLM_URL", "http://127.0.0.1:11434/v1")
    )
    llm_model: str = field(
        default_factory=lambda: os.environ.get("PROMPTFORGE_LLM_MODEL", "qwen2.5:7b")
    )
    # API model used ONLY when the local LLM fails; set to "" to disable the
    # fallback entirely (fully local operation).
    llm_api_model: str = field(
        default_factory=lambda: os.environ.get(
            "PROMPTFORGE_LLM_API_MODEL", "claude-fable-5")
    )
    # Automatically download missing required models (checksum-verified,
    # registry-listed sources only) when a workflow job needs them.
    auto_install: bool = field(
        default_factory=lambda: os.environ.get(
            "PROMPTFORGE_AUTO_INSTALL", "1") not in ("0", "false", "no")
    )
    # First run on a machine: profile hardware and let the LLM pre-stage
    # models that fit it (visible as a "setup" job).
    first_run_setup: bool = field(
        default_factory=lambda: os.environ.get(
            "PROMPTFORGE_FIRST_RUN_SETUP", "1") not in ("0", "false", "no")
    )
    # Share downloaded model weights with other PromptForge installs on the
    # local network, and copy from them before touching the internet. Only
    # the model library is ever served — never photos, assets or jobs — and
    # every copied file is verified against the registry's pinned checksum.
    lan_share: bool = field(
        default_factory=lambda: os.environ.get(
            "PROMPTFORGE_LAN_SHARE", "1") not in ("0", "false", "no")
    )
    # Accept render work from LAN peers while this machine is idle, and
    # hand renders to an idle peer when this machine's queue is busy.
    lan_render: bool = field(
        default_factory=lambda: os.environ.get(
            "PROMPTFORGE_LAN_RENDER", "1") not in ("0", "false", "no")
    )
    # Peers connected by address ("host" or "host:port", comma-separated) —
    # for networks where UDP discovery broadcasts never arrive.
    lan_peers: str = field(
        default_factory=lambda: os.environ.get("PROMPTFORGE_PEER_HOSTS", "")
    )
    # How many times a failing workflow may be sent back to the LLM for repair.
    workflow_max_repairs: int = field(
        default_factory=lambda: int(os.environ.get("PROMPTFORGE_WORKFLOW_REPAIRS", "2"))
    )
    # Vision model (Ollama) that judges photorealism of results; "" disables.
    critic_model: str = field(
        default_factory=lambda: os.environ.get("PROMPTFORGE_CRITIC_MODEL", "llava")
    )
    # Minimum acceptable realism score (1-10); below it a strategy change is tried.
    critic_min_score: float = field(
        default_factory=lambda: float(os.environ.get("PROMPTFORGE_CRITIC_MIN", "6"))
    )
    # How many strategy-change rounds the critic may trigger per job.
    critic_retries: int = field(
        default_factory=lambda: int(os.environ.get("PROMPTFORGE_CRITIC_RETRIES", "1"))
    )
    # The prompt is the contract. When a render does not do what the request
    # asked, this many escalation rungs may be spent making it — each rung
    # changes something real (emphasis, then a different MODEL, then a
    # different WORKFLOW), never just the seed. 0 disables the ladder.
    adherence_rounds: int = field(
        default_factory=lambda: int(
            os.environ.get("PROMPTFORGE_ADHERENCE_ROUNDS", "2"))
    )
    # Fallback bar only. When the per-requirement checklist works, "did it do
    # what was asked" is answered by whether any requirement is confirmed
    # unmet — a percentage would be quantised to 1/N and one false negative on
    # a 3-item list could never clear 80. This number is used only when the
    # checklist is unavailable and the single prompt_accuracy score is all
    # there is.
    adherence_target: int = field(
        default_factory=lambda: int(
            os.environ.get("PROMPTFORGE_ADHERENCE_TARGET", "60"))
    )
    # Quality pipeline: every score category must reach this 0-100 target or
    # the edit iterates (bounded by quality_rounds, keeping the best attempt).
    quality_target: int = field(
        default_factory=lambda: int(os.environ.get("PROMPTFORGE_QUALITY_TARGET", "95"))
    )
    # How many improvement rounds an edit may spend chasing the target.
    # Each round is a full re-render + inspection (~1-2 min on 8 GB).
    quality_rounds: int = field(
        default_factory=lambda: int(os.environ.get("PROMPTFORGE_QUALITY_ROUNDS", "2"))
    )
    # Civitai account API token (civitai.com/user/account → API Keys). Many
    # civitai files answer 401/403 to anonymous downloads; the token fixes
    # that. Optional — Hugging Face sources work without it.
    civitai_token: str = field(
        default_factory=lambda: os.environ.get("PROMPTFORGE_CIVITAI_TOKEN", "")
    )
    # Longest driving video the app will accept. Motion transfer renders in
    # chunks, so length costs minutes rather than memory — but a clip nobody
    # will sit through is better refused at upload than three hours in.
    max_video_seconds: int = field(
        default_factory=lambda: int(
            os.environ.get("PROMPTFORGE_MAX_VIDEO_SECONDS", "60"))
    )
    max_upload_mb: int = field(
        default_factory=lambda: int(os.environ.get("PROMPTFORGE_MAX_UPLOAD_MB", "64"))
    )
    job_max_retries: int = field(
        default_factory=lambda: int(os.environ.get("PROMPTFORGE_JOB_MAX_RETRIES", "3"))
    )
    job_retry_backoff_s: float = field(
        default_factory=lambda: float(os.environ.get("PROMPTFORGE_JOB_BACKOFF_S", "0.5"))
    )

    @property
    def db_path(self) -> Path:
        return self.data_dir / "promptforge.sqlite3"

    @property
    def assets_dir(self) -> Path:
        return self.data_dir / "assets"

    @property
    def masks_dir(self) -> Path:
        return self.data_dir / "masks"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def workflows_dir(self) -> Path:
        return BACKEND_ROOT / "app" / "workflows"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.assets_dir, self.masks_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)


ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv"}  # video pipeline: see ROADMAP
# 3D results. GLB only: it is a single self-contained binary file (geometry +
# any textures + materials in one), which keeps a mesh a normal asset rather
# than a folder of loose .obj/.mtl/.png that would need its own plumbing.
ALLOWED_MODEL_EXTS = {".glb"}
