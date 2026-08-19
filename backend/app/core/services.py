"""Composition root: builds and wires every component, registers job handlers.

Framework-agnostic on purpose — the Flask layer (app/api) only translates HTTP
to these calls, so swapping Flask for FastAPI later touches one thin layer.
"""
from __future__ import annotations

import base64
import copy
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from PIL import (
    Image,
    ImageChops,
    ImageDraw,
    ImageFilter,
    ImageMath,
    ImageStat,
    UnidentifiedImageError,
)

from ..adapters.base import (
    AdapterError,
    BackendUnavailableError,
    BadMaskError,
    InpaintingAdapter,
    ModelMissingError,
    SegmentationAdapter,
)
from ..adapters.comfyui import (
    ComfyUIClient,
    ComfyUIInpaintingAdapter,
    WorkflowLibrary,
    WorkflowNotAllowedError,
    WorkflowRuntimeError,
    WorkflowValidationError,
    build_workflow,
    hires_split_graph,
    tiled_vae_graph,
    validate_workflow,
)
from ..adapters.mock import (
    MockInpaintingAdapter,
    MockSegmentationAdapter,
    OfflineComfyClient,
)
from ..adapters.sam import SamSegmentationAdapter
from ..config import PROJECT_ROOT, Settings
from . import eta, motion, node_packs, quality, scene_geometry
from . import scene_graph as scene_module
from . import video as video_io
from .critic import (
    CriticChain,
    CriticUnavailable,
    Critique,
    ImageCritic,
    ask_with_schema,
)
from .db import Database
from .experience import ExperienceStore
from .hardware import _probe_gpu_registry as hw_gpu_registry  # noqa: E501
from .hardware import (
    available_commit_gb,
    max_auto_download_bytes,
    probe,
    render_budget,
)
from .hardware import ram_stats as hw_ram_stats
from .jobs import Job, JobQueue, PermanentError, TransientError
from .llm import (
    ClaudeLLM,
    FallbackLLM,
    LLMClient,
    LLMError,
    LLMRefusedError,
    LLMUnavailableError,
    LocalLLM,
    complete_with_schema,
    ollama_autopull,
    ollama_is_up,
    ollama_unload_all,
)
from .model_intel import ModelIntel
from .model_scout import ModelScout
from .model_search import ModelIndex, ModelSearch
from .peers import Peer, PeerService
from .registry import DownloadError, ModelDownloader, ModelInfo, ModelRegistry
from .safety import SafetyFilter, SafetyRuleStore, consent_verdict
from .storage import AssetStore
from .trust import Evidence, TrustJudge
from .update import UpdateError, UpdateManager
from .workflow_ai import WorkflowGenerationError, WorkflowGenerator


def _url_filename(url: str | None) -> str:
    from urllib.parse import urlparse
    return Path(urlparse(url or "").path).name


class _GreyPixels(Protocol):
    """Pixel access of an in-memory "L" image: one int per coordinate."""

    def __getitem__(self, xy: tuple[int, int]) -> int: ...


# Registry seed: models the ComfyUI path will need. URLs/checksums are filled
# per deployment (see docs/PromptForge-Documentation.pdf); nothing downloads
# automatically without an explicit user action.
DEFAULT_MODELS = [
    ModelInfo(
        name="sd15-inpaint",
        purpose="Stable Diffusion 1.5 inpainting checkpoint for the ComfyUI backend",
        license=("CreativeML OpenRAIL-M — review before commercial use. "
                 "Safetensors mirror of the original runwayml weights "
                 "(original HF repo was removed); ~4.27 GB."),
        url=("https://huggingface.co/webui/stable-diffusion-inpainting/"
             "resolve/main/sd-v1-5-inpainting.safetensors"),
        sha256="0ec8f8585b104417a8c34a9fbcc1e922a70b8c15490ec4553087f01c8cf33673",
        vram_gb=6.0,
        meta={"folder": "checkpoints"},
    ),
    ModelInfo(
        name="sam-vit-b",
        purpose="Segment Anything (ViT-B) for prompt-guided mask proposals",
        license="Apache-2.0 (Meta AI, official checkpoint; ~375 MB, .pth pickle format)",
        url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        sha256="ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
        vram_gb=4.0,
        # Official Meta artifact: pickle format explicitly vetted; the folder
        # keeps it out of ComfyUI's checkpoint list.
        meta={"folder": "segmentation", "allow_pickle": True},
    ),
    # WAN 2.2 image-to-video (Comfy-Org repackaged, official mirrors). sha256
    # is backfilled from the hub's LFS metadata just before first download.
    ModelInfo(
        name="wan22-ti2v-5b",
        purpose="WAN 2.2 TI2V-5B diffusion model for image-to-video (fits 8 GB VRAM with offload)",
        license="Apache-2.0 (Wan-AI; ~10 GB fp16)",
        url=("https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/"
             "main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors"),
        vram_gb=8.0,
        meta={"repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
              "file": "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
              "folder": "diffusion_models"},
    ),
    ModelInfo(
        name="wan-umt5-xxl",
        purpose="UMT5-XXL text encoder for WAN video (fp8, ~6.7 GB)",
        license="Apache-2.0 (Wan-AI / Comfy-Org repackaged)",
        url=("https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/"
             "main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
        meta={"repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
              "file": "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
              "folder": "text_encoders"},
    ),
    ModelInfo(
        name="sv3d",
        purpose="SV3D multi-view synthesis (orbital views from one image) for the avatar pipeline",
        license=("Stability AI Community License (review before commercial "
                 "use). The official stabilityai/sv3d repo is gated, so this "
                 "uses a public mirror whose file size matches the official "
                 "byte-for-byte; ~9.4 GB, sha256-verified."),
        url="https://huggingface.co/camenduru/sv3d/resolve/main/sv3d_u.safetensors",
        sha256="d2c281b817232c492f6db27c9ce597b543187c52229cbad2a3c78e238b06c809",
        vram_gb=8.0,
        meta={"repo": "camenduru/sv3d", "file": "sv3d_u.safetensors",
              "folder": "checkpoints"},
    ),
    # Image → 3D mesh. ComfyUI ships Hunyuan3D v2 support in core, so these
    # are plain checkpoints — no node pack, nothing to compile. The tiers are
    # picked by hardware at runtime (quality.choose_reconstruction).
    ModelInfo(
        name="hunyuan3d-v2",
        purpose="Hunyuan3D v2 — one photo to a real 3D mesh, exported as GLB",
        license=("Tencent Hunyuan Community License — review before "
                 "commercial use; it carries an EU-territory exclusion and a "
                 "100M-MAU clause. Repackaged single-file build by Comfy-Org "
                 "(ungated, ~4.6 GB fp16). SHAPE ONLY: ComfyUI has no "
                 "Hunyuan3D texture stage, so the mesh is untextured."),
        url=("https://huggingface.co/Comfy-Org/hunyuan3D_2.0_repackaged/"
             "resolve/main/split_files/hunyuan3d-dit-v2_fp16.safetensors"),
        vram_gb=6.0,
        meta={"repo": "Comfy-Org/hunyuan3D_2.0_repackaged",
              "file": "split_files/hunyuan3d-dit-v2_fp16.safetensors",
              "folder": "checkpoints",
              "min_vram_gb": 5.5, "min_ram_gb": 11.0},
    ),
    ModelInfo(
        name="hunyuan3d-v2-mv",
        purpose=("Hunyuan3D v2 multi-view — builds the mesh from four views "
                 "(front/left/back/right) instead of one, so the back of the "
                 "subject is reconstructed rather than invented"),
        license=("Tencent Hunyuan Community License (see hunyuan3d-v2). "
                 "Turbo/distilled multi-view build, ~4.6 GB fp16, ungated."),
        url=("https://huggingface.co/Comfy-Org/hunyuan3D_2.0_repackaged/"
             "resolve/main/split_files/"
             "hunyuan3d-dit-v2-mv-turbo_fp16.safetensors"),
        vram_gb=8.0,
        meta={"repo": "Comfy-Org/hunyuan3D_2.0_repackaged",
              "file": ("split_files/"
                       "hunyuan3d-dit-v2-mv-turbo_fp16.safetensors"),
              "folder": "checkpoints",
              "min_vram_gb": 7.5, "min_ram_gb": 15.0},
    ),
    ModelInfo(
        name="hunyuan3d-v21",
        purpose="Hunyuan3D 2.1 — higher-fidelity mesh for larger GPUs",
        license=("Tencent Hunyuan Community License (see hunyuan3d-v2). "
                 "~6.9 GB, ungated."),
        url=("https://huggingface.co/Comfy-Org/hunyuan3D_2.1_repackaged/"
             "resolve/main/hunyuan_3d_v2.1.safetensors"),
        vram_gb=12.0,
        meta={"repo": "Comfy-Org/hunyuan3D_2.1_repackaged",
              "file": "hunyuan_3d_v2.1.safetensors", "folder": "checkpoints",
              "min_vram_gb": 11.5, "min_ram_gb": 23.0},
    ),
    ModelInfo(
        name="moge-v2",
        purpose="MoGe v2 — one photo to a metric 3D point map with a "
                "predicted field of view, which is what turns an ordinary "
                "picture into a scene you can move around in",
        license=("MIT (Microsoft, repackaged by Comfy-Org); ~631 MB fp16. "
                 "Unlike the ControlNet depth preprocessors — which return "
                 "8-bit normalised DISPARITY and cannot be meshed correctly — "
                 "this predicts true metric geometry."),
        url=("https://huggingface.co/Comfy-Org/MoGe/resolve/main/"
             "geometry_estimation/moge_2_vitl_normal_fp16.safetensors"),
        vram_gb=2.0,
        meta={"repo": "Comfy-Org/MoGe",
              "file": "geometry_estimation/moge_2_vitl_normal_fp16.safetensors",
              "folder": "geometry_estimation",
              "min_vram_gb": 2.0, "min_ram_gb": 8.0},
    ),
    # Identity pipeline (avatar renders): PhotoMaker encodes a consented face
    # into SDXL conditioning so the avatar can appear in any prompted scene.
    ModelInfo(
        name="sdxl-base",
        purpose="Stable Diffusion XL base checkpoint (identity renders + high-detail generation)",
        license="CreativeML Open RAIL++-M (Stability AI; ~6.9 GB fp16)",
        url=("https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/"
             "resolve/main/sd_xl_base_1.0.safetensors"),
        sha256="31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b",
        vram_gb=8.0,
        meta={"repo": "stabilityai/stable-diffusion-xl-base-1.0",
              "file": "sd_xl_base_1.0.safetensors", "folder": "checkpoints"},
    ),
    ModelInfo(
        name="photomaker-v1",
        purpose="PhotoMaker identity encoder — renders a consented avatar into prompted scenes",
        license=("Apache-2.0 (TencentARC official; ~934 MB .bin pickle format "
                 "— explicitly vetted, same policy as the Meta SAM weights)"),
        url=("https://huggingface.co/TencentARC/PhotoMaker/resolve/main/"
             "photomaker-v1.bin"),
        vram_gb=2.0,
        meta={"repo": "TencentARC/PhotoMaker", "file": "photomaker-v1.bin",
              "folder": "photomaker", "allow_pickle": True},
    ),
    ModelInfo(
        name="wan22-vae",
        purpose="WAN 2.2 VAE for image-to-video (~1.4 GB)",
        license="Apache-2.0 (Wan-AI / Comfy-Org repackaged)",
        url=("https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/"
             "main/split_files/vae/wan2.2_vae.safetensors"),
        meta={"repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
              "file": "split_files/vae/wan2.2_vae.safetensors",
              "folder": "vae"},
    ),
    # 4× detail upscaler (ESRGAN family) for the model-based upscale template.
    ModelInfo(
        name="upscale-ultrasharp",
        purpose="4x-UltraSharp upscaler for the model-based upscale workflow",
        license="CC-BY-NC-SA 4.0 (Kim2091 — review before commercial use; ~64 MB)",
        url=("https://huggingface.co/Kim2091/UltraSharp/resolve/main/"
             "4x-UltraSharp.safetensors"),
        sha256="36a340b5509b699d2c06cb445ddc1d3d39199ac734d889ed6d7915f60e05bcbc",
        meta={"repo": "Kim2091/UltraSharp", "file": "4x-UltraSharp.safetensors",
              "folder": "upscale_models"},
    ),
    # Face detector for the FaceDetailer polish pass (Impact Pack).
    ModelInfo(
        name="face-yolov8m",
        purpose="YOLOv8m face detector — finds faces for the automatic "
                "face-refinement pass after renders with people",
        license="AGPL-3.0 (Ultralytics weights via Bingsu/adetailer; ~52 MB)",
        url=("https://huggingface.co/Bingsu/adetailer/resolve/main/"
             "face_yolov8m.pt"),
        sha256="717923c19b3f4bbf5250b728f1fa6b2cb72a33aed1d236ea9caf0e21ad943e5f",
        meta={"repo": "Bingsu/adetailer", "file": "face_yolov8m.pt",
              "folder": "ultralytics/bbox", "allow_pickle": True},
    ),
    # WAN 2.1 VACE: video inpainting/outpainting (control video + masks).
    ModelInfo(
        name="wan21-vace-1.3b",
        purpose="WAN 2.1 VACE 1.3B for video inpainting/outpainting (fits 8 GB VRAM)",
        license="Apache-2.0 (Wan-AI / Comfy-Org repackaged; ~4.3 GB fp16)",
        url=("https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/"
             "main/split_files/diffusion_models/wan2.1_vace_1.3B_fp16.safetensors"),
        sha256="640ccc0577e6a5d4bb15cd91b11b699ef914fc55f126c5a1c544e152130784f2",
        vram_gb=6.0,
        meta={"repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
              "file": "split_files/diffusion_models/wan2.1_vace_1.3B_fp16.safetensors",
              "folder": "diffusion_models"},
    ),
    # Top photoreal SD1.5 inpainting checkpoint (2nd most-downloaded inpaint
    # model on civitai) — small (2 GB), fast, far better output than the
    # sd-v1-5-inpainting base; the preferred default on 8 GB GPUs.
    ModelInfo(
        name="epicrealism-inpaint",
        purpose="epiCRealism pureEvolution inpainting checkpoint (SD 1.5) — "
                "photoreal inpaint edits, fast on 8 GB GPUs; selectable by "
                "the LLM",
        license="From civitai.com ('epiCRealism pureEvolution InPainting') — "
                "check the model page for terms; ~2 GB",
        url="https://civitai.com/api/download/models/95864",
        sha256="ec6a1ba63656a7bc9eb69130afff5b30a82aa3585f57eef7b2b9bb0c7f3ba845",
        vram_gb=4.0,
        meta={"folder": "checkpoints", "source": "civitai",
              "file": "epicrealism_v10-inpainting.safetensors"},
    ),
    # Photoreal SDXL inpainting checkpoint the LLM can pick for inpaint
    # steps (most-downloaded inpaint model on civitai).
    ModelInfo(
        name="juggernaut-xl-inpaint",
        purpose="Juggernaut XL inpainting checkpoint (SDXL) — photoreal "
                "results for inpaint edits; selectable by the LLM",
        license="From civitai.com ('Juggernaut XL' inpainting version) — "
                "check the model page for terms; ~6.9 GB",
        url="https://civitai.com/api/download/models/456538",
        sha256="b1689257e6e1b2e61544b1a41fc114e7d798f68854b3f875cd52070bfe1fbc00",
        vram_gb=8.0,
        meta={"folder": "checkpoints", "source": "civitai",
              "file": "juggernautXL_inpaint.safetensors"},
    ),
    ModelInfo(
        name="wan21-vae",
        purpose="WAN 2.1 VAE (required by the VACE video-editing workflows; ~254 MB)",
        license="Apache-2.0 (Wan-AI / Comfy-Org repackaged)",
        url=("https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/"
             "main/split_files/vae/wan_2.1_vae.safetensors"),
        sha256="2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b",
        meta={"repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
              "file": "split_files/vae/wan_2.1_vae.safetensors",
              "folder": "vae"},
    ),
    # ---- Speed LoRAs: 4-step draft rendering (2026-07-22 additions) --------
    ModelInfo(
        name="dmd2-sdxl-lora",
        purpose="DMD2 4-step speed LoRA for SDXL — draft renders ~6x faster "
                "(steps 4, cfg 1); best quality of the 4-step distills",
        license="Research license (tianweiy/DMD2); ~394 MB",
        url=("https://huggingface.co/tianweiy/DMD2/resolve/main/"
             "dmd2_sdxl_4step_lora_fp16.safetensors"),
        sha256="b3d9173815a4b595991c3a7a0e0e63ad821080f314a0b2a3cc31ecd7fcf2cbb8",
        meta={"repo": "tianweiy/DMD2",
              "file": "dmd2_sdxl_4step_lora_fp16.safetensors",
              "folder": "loras", "size_bytes": 393854592},
    ),
    ModelInfo(
        name="lcm-lora-sd15",
        purpose="LCM speed LoRA for SD 1.5 — 4-step draft renders (cfg 1-2, "
                "sampler lcm); ~135 MB",
        license="OpenRAIL (latent-consistency, official)",
        url=("https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/"
             "resolve/main/pytorch_lora_weights.safetensors"),
        sha256="8f90d840e075ff588a58e22c6586e2ae9a6f7922996ee6649a7f01072333afe4",
        meta={"repo": "latent-consistency/lcm-lora-sdv1-5",
              "file": "lcm_lora_sd15.safetensors",
              "folder": "loras", "size_bytes": 134621556},
    ),
    # ---- ControlNet (one union model covers pose/depth/canny/tile) ---------
    ModelInfo(
        name="controlnet-union-sdxl",
        purpose="ControlNet Union SDXL ProMax — 12 control types (canny/"
                "pose/depth/tile...) in one model for guided generation",
        license="Apache-2.0 (xinsir); ~2.5 GB",
        url=("https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/"
             "resolve/main/diffusion_pytorch_model_promax.safetensors"),
        sha256="9fae2e50cb431bfcbe05822b59ec2228df545ef27f711dea8949e9f4ed9f7cdc",
        vram_gb=6.0,
        meta={"repo": "xinsir/controlnet-union-sdxl-1.0",
              "file": "controlnet_union_sdxl_promax.safetensors",
              "folder": "controlnet", "size_bytes": 2513342408},
    ),
    ModelInfo(
        name="controlnet-sd15-depth",
        purpose="ControlNet v1.1 depth for SD15 — pins a generated "
                "environment to the photograph's MEASURED perspective. The "
                "guidance canvas comes from the scene probe (subject depth "
                "+ the ground's disparity ramp to the measured horizon); "
                "words alone held the horizon on about half the draws, so "
                "conditioning is the adherence lever.",
        license="OpenRAIL-M (lllyasviel; fp16 repack by comfyanonymous); "
                "~700 MB",
        url=("https://huggingface.co/comfyanonymous/"
             "ControlNet-v1-1_fp16_safetensors/resolve/main/"
             "control_v11f1p_sd15_depth_fp16.safetensors"),
        vram_gb=1.0,
        meta={"repo": "comfyanonymous/ControlNet-v1-1_fp16_safetensors",
              "file": "control_v11f1p_sd15_depth_fp16.safetensors",
              "folder": "controlnet"},
    ),
    # ---- Motion transfer speed LoRA (WAN 2.1 1.3B) -------------------------
    ModelInfo(
        name="causvid-wan13b-lora",
        purpose="CausVid distill LoRA for WAN 2.1 1.3B — same motion in far "
                "fewer sampling steps, which is what makes a longer clip "
                "affordable on 8 GB",
        license="Apache-2.0 (lightx2v / Kijai repack); ~87 MB",
        url=("https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/"
             "Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors"),
        sha256="8e5331e780ffb16520bf1ff7ba90188ebd271a4698e5be8e618800e809cca704",
        meta={"folder": "loras",
              "file": "Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors",
              "size_bytes": 91233416},
    ),
    # ---- InstantID: face identity from ONE reference photo (SDXL only) -----
    # Used for consented identity work only — the same gate the avatar
    # pipeline applies. Needs the 'instantid' node pack active.
    ModelInfo(
        name="instantid-ipadapter",
        purpose="InstantID face adapter — carries a person's identity from a "
                "single reference photo into an SDXL render",
        license=("Apache-2.0 (InstantX, official repo). ~1.7 GB. Pickle "
                 "format (.bin) — the only form InstantX publishes."),
        url=("https://huggingface.co/InstantX/InstantID/resolve/main/"
             "ip-adapter.bin"),
        sha256="02b3618e36d803784166660520098089a81388e61a93ef8002aa79a5b1c546e1",
        vram_gb=6.0,
        # Official InstantX artifact with a published checksum: the pickle
        # gate is opened deliberately here, exactly as it is for Meta's SAM.
        meta={"folder": "instantid", "file": "ip-adapter.bin",
              "allow_pickle": True, "requires_pack": "instantid",
              # SDXL 6.9 GB + this 1.7 GB + the 2.5 GB ControlNet = 11.1 GB
              # of weights that must be resident together. Measured on the
              # 8 GB / 16 GB dev machine: not survivable. Gated like Kontext,
              # so it auto-stages and runs on a bigger machine instead.
              "min_vram_gb": 12.0, "min_ram_gb": 24.0,
              "size_bytes": 1691134141},
    ),
    ModelInfo(
        name="instantid-controlnet",
        purpose="InstantID ControlNet — holds the reference face's pose and "
                "keypoints so the identity lands in the right place",
        license="Apache-2.0 (InstantX, official repo); ~2.5 GB",
        url=("https://huggingface.co/InstantX/InstantID/resolve/main/"
             "ControlNetModel/diffusion_pytorch_model.safetensors"),
        sha256="c8127be9f174101ebdafee9964d856b49b634435cf6daa396d3f593cf0bbbb05",
        vram_gb=6.0,
        meta={"folder": "controlnet", "file": "instantid_controlnet.safetensors",
              "requires_pack": "instantid", "min_vram_gb": 12.0,
              "min_ram_gb": 24.0, "size_bytes": 2502139136},
    ),
    # ---- IC-Light relighting (SD1.5-based; used via the ic-light pack) -----
    ModelInfo(
        name="iclight-sd15-fc",
        purpose="IC-Light (foreground conditioned) — relight a portrait/"
                "object; needs the 'ic-light' node pack installed",
        license="Apache-2.0 (lllyasviel, official); ~1.7 GB",
        url=("https://huggingface.co/lllyasviel/ic-light/resolve/main/"
             "iclight_sd15_fc.safetensors"),
        sha256="a033fbaaa2f3f7859fa6a4477ee63ebbf9c116bf3569d5811856d2807f3468cd",
        vram_gb=6.0,
        meta={"repo": "lllyasviel/ic-light", "file": "iclight_sd15_fc.safetensors",
              "folder": "diffusion_models", "requires_pack": "ic-light",
              "size_bytes": 1719148312},
    ),
    ModelInfo(
        name="iclight-sd15-fbc",
        purpose="IC-Light (foreground + BACKGROUND conditioned) — relights a "
                "subject to match the scene behind it, which is what makes a "
                "replaced background stop looking pasted on",
        license="Apache-2.0 (lllyasviel, official); ~1.6 GB",
        url=("https://huggingface.co/lllyasviel/ic-light/resolve/main/"
             "iclight_sd15_fbc.safetensors"),
        vram_gb=6.0,
        meta={"repo": "lllyasviel/ic-light",
              "file": "iclight_sd15_fbc.safetensors",
              "folder": "diffusion_models", "requires_pack": "ic-light",
              "size_bytes": 1719148312},
    ),
    ModelInfo(
        name="sd15-base",
        purpose="Stable Diffusion 1.5 base — the 4-channel SD1.5 UNet that "
                "IC-Light relighting requires (inpainting checkpoints are "
                "9-channel and physically cannot run it)",
        license="CreativeML OpenRAIL-M — review before commercial use. "
                "Community re-upload of the original runwayml weights "
                "(the original repo was removed); ~4.27 GB.",
        url=("https://huggingface.co/stable-diffusion-v1-5/"
             "stable-diffusion-v1-5/resolve/main/"
             "v1-5-pruned-emaonly.safetensors"),
        sha256="6ce0161689b3853acaa03779ec93eafe75a02f4ced659bee03f50797806fa2fa",
        vram_gb=4.0,
        meta={"repo": "stable-diffusion-v1-5/stable-diffusion-v1-5",
              "file": "v1-5-pruned-emaonly.safetensors",
              "folder": "checkpoints", "size_bytes": 4265146304},
    ),
    # ---- Z-Image Turbo: fast photoreal + the best legible TEXT on 8 GB -----
    ModelInfo(
        name="zimage-turbo",
        purpose="Z-Image Turbo (int8) — photoreal 1024px in 8 steps and the "
                "best legible text rendering that fits 8 GB VRAM",
        license="Apache-2.0 (Alibaba Tongyi; Comfy-Org repackage); ~6.2 GB",
        url=("https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/"
             "split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors"),
        sha256="be517ebd47c912a5626a588e1aeea43e6be4a43c0cdcd2b48a2a780d9f358635",
        vram_gb=7.0,
        meta={"repo": "Comfy-Org/z_image_turbo",
              "file": "split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors",
              "folder": "diffusion_models", "size_bytes": 6201001296},
    ),
    ModelInfo(
        name="zimage-text-encoder",
        purpose="Qwen3-4B text encoder for Z-Image Turbo (~8 GB). The full "
                "encoder — ComfyUI's CLIPLoader cannot load the fp8 variant.",
        license="Apache-2.0 (Comfy-Org repackage)",
        url=("https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/"
             "split_files/text_encoders/qwen_3_4b.safetensors"),
        sha256="6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a",
        meta={"repo": "Comfy-Org/z_image_turbo",
              "file": "split_files/text_encoders/qwen_3_4b.safetensors",
              "folder": "text_encoders", "size_bytes": 8044982048,
              "min_ram_gb": 24},
    ),
    ModelInfo(
        name="flux-ae",
        purpose="Flux autoencoder (shared by Z-Image Turbo and Flux Kontext; "
                "~335 MB)",
        license="Apache-2.0 (Comfy-Org repackage)",
        url=("https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/"
             "split_files/vae/ae.safetensors"),
        sha256="afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
        meta={"repo": "Comfy-Org/z_image_turbo",
              "file": "split_files/vae/ae.safetensors", "folder": "vae",
              "size_bytes": 335304388},
    ),
    # ---- Flux Kontext: instruction-based editing ("make it night") ---------
    ModelInfo(
        name="flux-kontext-q4",
        purpose="FLUX.1 Kontext dev (Q4_K_S GGUF) — instruction-based image "
                "editing; runs on 8 GB with offload (~2 min/edit); needs the "
                "'gguf' node pack",
        license="FLUX.1 dev non-commercial license (Black Forest Labs); ~6.8 GB",
        url=("https://huggingface.co/QuantStack/FLUX.1-Kontext-dev-GGUF/"
             "resolve/main/flux1-kontext-dev-Q4_K_S.gguf"),
        sha256="cc22ff7a2debb02e63765fa53af8c5ae0b6883b462d0601b9b55f51a15cdd6da",
        vram_gb=7.0,
        meta={"repo": "QuantStack/FLUX.1-Kontext-dev-GGUF",
              "file": "flux1-kontext-dev-Q4_K_S.gguf",
              "folder": "diffusion_models", "requires_pack": "gguf",
              # 24 was a guess and it is wrong: measured live, the whole stack
              # (6.3 GB UNet + 4.6 GB T5 + CLIP + VAE) edits a 1 MP photo in
              # about 2.5 minutes on a 15.7 GB machine with ~13 GB of commit
              # headroom to spare. Declaring 24 would gate the route off on
              # exactly the hardware it was proven on.
              "size_bytes": 6797337888, "min_ram_gb": 12},
    ),
    ModelInfo(
        name="flux-t5-fp8",
        purpose="T5-XXL fp8 text encoder for Flux/Kontext (~4.9 GB)",
        license="Apache-2.0 (comfyanonymous repackage)",
        url=("https://huggingface.co/comfyanonymous/flux_text_encoders/"
             "resolve/main/t5xxl_fp8_e4m3fn.safetensors"),
        sha256="7d330da4816157540d6bb7838bf63a0f02f573fc48ca4d8de34bb0cbfd514f09",
        meta={"repo": "comfyanonymous/flux_text_encoders",
              "file": "t5xxl_fp8_e4m3fn.safetensors",
              "folder": "text_encoders", "size_bytes": 4893934904},
    ),
    ModelInfo(
        name="flux-clip-l",
        purpose="CLIP-L text encoder for Flux/Kontext (~246 MB)",
        license="Apache-2.0 (comfyanonymous repackage)",
        url=("https://huggingface.co/comfyanonymous/flux_text_encoders/"
             "resolve/main/clip_l.safetensors"),
        sha256="660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd",
        meta={"repo": "comfyanonymous/flux_text_encoders",
              "file": "clip_l.safetensors", "folder": "text_encoders",
              "size_bytes": 246144152},
    ),
    # ---- Big-GPU models: auto-staged only on machines that can run them ----
    ModelInfo(
        name="qwen-image-edit-2511",
        purpose="Qwen-Image-Edit 2511 (fp8) — best-in-class instruction "
                "editing; needs a 20 GB+ GPU (auto-staged on such machines)",
        license="Apache-2.0 (Alibaba; Comfy-Org repackage); ~20.5 GB",
        url=("https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/"
             "main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors"),
        sha256="c9fdc158e46d3b61ef75f21ae866ca2fe808bf4a53643120d1c1e87c19280a4e",
        vram_gb=20.0,
        meta={"repo": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
              "file": "split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
              "folder": "diffusion_models", "min_vram_gb": 20,
              "size_bytes": 20533762817},
    ),
]

# InsightFace's antelopev2 pack — the face DETECTOR/embedder InstantID reads a
# reference photo with. Five separate ONNX files that must land in the nested
# layout insightface hard-codes (insightface/models/antelopev2/), so they are
# generated rather than written out five times. Checksums are the published
# LFS hashes, so each one is verified on download like every other model.
_ANTELOPE = [
    ("scrfd_10g_bnkps", "face detection",
     "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91", 16923827),
    ("glintr100", "face recognition embedding",
     "4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf", 260665334),
    ("1k3d68", "3D face landmarks",
     "df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc", 143607619),
    ("2d106det", "2D face landmarks",
     "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf", 5030888),
    ("genderage", "face attribute head",
     "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb", 1322532),
]

DEFAULT_MODELS += [
    ModelInfo(
        name=f"antelopev2-{stem}",
        purpose=f"InsightFace antelopev2 — {what} (needed by InstantID)",
        license="MIT (InsightFace); ONNX weights, public mirror",
        url=f"https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/{stem}.onnx",
        sha256=sha,
        meta={"folder": "insightface/models/antelopev2",
              "file": f"{stem}.onnx", "requires_pack": "instantid",
              "size_bytes": size},
    )
    for stem, what, sha, size in _ANTELOPE
]


# When to use WHICH model — curated one-liners the LLM reads at every
# routing/choice site. Registry `purpose` says what a model IS; these say
# when to PICK it and what it must (not) be combined with. A test keeps
# every registry entry covered.
MODEL_USAGE: dict[str, str] = {
    # The five antelopev2 ONNX files are face-analysis plumbing InstantID
    # loads as a set — never a choice the planner makes.
    f"antelopev2-{stem}": f"InstantID face analysis ({what}) — downloaded as "
                          "a set with instantid-ipadapter, never chosen alone"
    for stem, what, _sha, _size in _ANTELOPE
}
MODEL_USAGE |= {
    "sd15-inpaint": "legacy SD15 inpaint fallback — prefer epicrealism-inpaint",
    "epicrealism-inpaint": "DEFAULT for photoreal SD15 inpaint edits",
    "juggernaut-xl-inpaint": "SDXL inpaint — large regions / big canvases",
    "sdxl-base": "general SDXL txt2img base; pairs with dmd2-sdxl-lora and "
                 "controlnet-union-sdxl (SDXL ONLY)",
    "sam-vit-b": "segmentation masks (internal — never a render model)",
    "wan22-ti2v-5b": "image-to-video (the only i2v that fits 8 GB)",
    "wan-umt5-xxl": "text encoder for WAN video (always with wan models)",
    "wan22-vae": "VAE for WAN 2.2 video",
    "wan21-vace-1.3b": "video inpaint/outpaint (VACE editing)",
    "wan21-vae": "VAE for WAN 2.1 VACE",
    "sv3d": "multi-view orbit synthesis from one image (avatars)",
    "hunyuan3d-v2": "one photo -> a real 3D MESH exported as GLB (shape "
                    "only, untextured; ComfyUI has no texture stage)",
    "hunyuan3d-v2-mv": "four views -> a 3D MESH; the back is reconstructed "
                       "rather than invented (pair with sv3d for the views)",
    "hunyuan3d-v21": "higher-fidelity 3D mesh; 12 GB+ GPUs only (auto-gated)",
    "moge-v2": "one photo -> a metric 3D point map + predicted camera FOV; "
                "the basis for turning a picture into a scene you can move "
                "around in (the depth PREPROCESSORS give 8-bit disparity and "
                "cannot be meshed)",
    "iclight-sd15-fbc": "relights a subject to MATCH the scene behind it - "
                        "what stops a replaced background looking pasted on "
                        "(needs the ic-light pack AND sd15-base)",
    "photomaker-v1": "identity renders from reference photos (SDXL)",
    "upscale-ultrasharp": "faithful 4x pixel upscale, no prompt",
    "face-yolov8m": "face detector for the automatic FaceDetailer "
                    "refinement pass — never a render model itself",
    "dmd2-sdxl-lora": "SPEED: 4-step drafts on SDXL checkpoints ONLY "
                      "(cfg 1, euler/sgm_uniform) — never on SD15",
    "lcm-lora-sd15": "SPEED: 4-step drafts on SD15 checkpoints ONLY "
                     "(sampler lcm, cfg 1-2) — never on SDXL",
    "controlnet-union-sdxl": "structure lock (canny/pose/depth/tile) for "
                             "SDXL ONLY — needs a control image input",
    "controlnet-sd15-depth": "GEOMETRY: depth guidance for SD15 background "
                             "renders — never picked by prompt routing; "
                             "the environment pipeline attaches it itself "
                             "when a measured perspective guide exists",
    "iclight-sd15-fc": "THE relighting engine — changes a photo's LIGHT "
                       "instead of repainting it (needs the ic-light node "
                       "pack active AND sd15-base; inpainting checkpoints "
                       "cannot run it)",
    "sd15-base": "plain 4-channel SD1.5 base — required by IC-Light "
                 "relighting; also a safe general SD1.5 checkpoint",
    "causvid-wan13b-lora": "SPEED: pair with wan21-vace-1.3b for motion "
                           "transfer — same motion in far fewer steps",
    "instantid-ipadapter": "face IDENTITY from one reference photo, SDXL "
                           "ONLY; needs instantid-controlnet + the "
                           "antelopev2 files + the instantid node pack",
    "instantid-controlnet": "the ControlNet half of InstantID — always "
                            "loaded together with instantid-ipadapter",
    "zimage-turbo": "readable text in images + fast photoreal; cfg 1 + "
                    "ConditioningZeroOut, 8 steps; needs zimage-text-encoder "
                    "+ flux-ae and ~24 GB RAM (its full encoder is 8 GB)",
    "zimage-text-encoder": "text encoder for zimage-turbo (CLIPLoader "
                           "type lumina2)",
    "flux-ae": "VAE shared by zimage-turbo and flux-kontext-q4",
    "flux-kontext-q4": "instruction-based WHOLE-image edits ('make it "
                       "night') — needs gguf pack + flux-t5-fp8 + "
                       "flux-clip-l + flux-ae; slow on 8 GB but strongest "
                       "editor",
    "flux-t5-fp8": "text encoder for Kontext (DualCLIPLoader type flux)",
    "flux-clip-l": "clip_l for Kontext (DualCLIPLoader type flux)",
    "qwen-image-edit-2511": "flagship instruction editor — 20 GB+ GPUs "
                            "only (auto-gated)",
}


# Windows refuses a big model load with OS error 1455 when the commit charge
# (RAM + paging file) runs out — the message is localized ("Het wisselbestand
# is te klein..." on a Dutch system, "The paging file is too small..." in
# English). It is a machine setting, not a graph bug: LLM repairs and
# step-down retries cannot fix it, so it must fail fast with the real cure.
_COMMIT_EXHAUSTED = re.compile(
    r"os error 1455|error\s*1455|wisselbestand|paging\s*file|page\s*file|"
    r"commitment limit", re.IGNORECASE)


# A vision model that echoes the question instead of answering it. The
# appearance prompt shows its shape with angle-bracket examples
# ("<e.g. mid 20s to early 30s>"), and a small model will sometimes hand the
# whole thing straight back. Seen live: an avatar saved with age_range
# "<e.g. mid 20s to early 30s>" and hair "<length, texture, colour>".
_PLACEHOLDER = re.compile(r"^\s*<.*>\s*$|^\s*e\.g\.|<[^>]{2,}>", re.IGNORECASE)


def _is_placeholder(value: str) -> bool:
    """True when a model reply is the prompt's own example, not an answer."""
    return bool(_PLACEHOLDER.search(value or ""))


def commit_exhausted_hint(error_text: str) -> str | None:
    """Actionable message when a render died because Windows ran out of
    virtual memory (commit) while loading a model; None otherwise."""
    if not _COMMIT_EXHAUSTED.search(error_text):
        return None
    return ("Windows ran out of virtual memory while loading the model "
            "(OS error 1455: the paging file is too small). This is a "
            "Windows setting, not a render bug — retrying won't help. Fix: "
            "enlarge the paging file (Settings > System > About > Advanced "
            "system settings > Performance Settings > Advanced > Virtual "
            "memory: set a custom size, e.g. initial 16384 MB / maximum "
            "32768 MB, on a drive with free space, then reboot) or close "
            "memory-hungry apps and run the render again.")


@dataclass
class _Attempt:
    """One render and everything known about it, so the adherence ladder can
    compare attempts on the two things that can fail a render: is it
    photoreal, and does it DO what the prompt asked."""
    image: Image.Image
    prompt_id: str
    gen: Any
    crit: Critique | None = None
    adherence: dict[str, Any] | None = None
    repairs: int = 0
    checklist: list[dict[str, str]] = field(default_factory=list)
    strategy: str | None = None

    def accuracy(self) -> int | None:
        return None if self.adherence is None else self.adherence["accuracy"]

    def source(self) -> str | None:
        return (self.adherence or {}).get("source")

    def missing(self) -> list[str]:
        return list((self.adherence or {}).get("missing") or [])

    def satisfies(self, settings: Settings) -> bool:
        """Good enough to stop.

        A checklist verdict is the authority: every requirement met means the
        render did what was asked, full stop — including when the request was
        for something deliberately un-photoreal ("a flat cartoon drawing"),
        where a low realism score is the CORRECT answer and escalating on it
        would spend the whole budget undoing the request.

        Realism only decides when adherence could not be established."""
        if self.source() == "checklist":
            return not self.missing()
        acc = self.accuracy()
        if acc is not None and acc < settings.adherence_target:
            return False
        if self.crit is not None and self.crit.score < settings.critic_min_score:
            return False
        return True

    def beats(self, other: _Attempt) -> bool:
        """Prompt fidelity first, realism second. A prettier render that
        dropped part of the request is a worse answer to the request — that
        ordering is the whole point of the ladder.

        Two checklist verdicts are compared on CONFIRMED UNMET COUNTS, not on
        the percentage: the percentage is quantised to 1/N and moves by a
        whole requirement anyway. A scorecard number and a checklist share are
        different scales entirely, so across sources accuracy is not consulted
        at all — that comparison would be noise dressed as a decision."""
        # Wreckage veto, before any fidelity comparison: a render the judge
        # scores as ruined (<=2/10) never displaces a plausible one (>=3),
        # whatever its checklist says. Vision probes answer "met"
        # alarmingly often on pure noise — measured live: a 3D-mesh
        # checkpoint hijacked the model rung and its realism-1/10 output
        # beat a realism-5/10 image on a hallucinated 100% checklist.
        # (Deliberately un-photoreal requests — flat cartoons — still score
        # above this floor; 2/10 is wreckage, not style. RE-CALIBRATED for
        # the qwen2.5-vl judge, 2026-08-17: the critique asks "for the
        # request", so a proper cartoon render scored 8/10 and a watercolor
        # 8/10 while pure noise and a corrupted render both pinned at 1/10
        # — the band holds, with WIDER separation than llava ever gave.)
        wreck_new = self.crit.score if self.crit else None
        wreck_old = other.crit.score if other.crit else None
        if (wreck_new is not None and wreck_new <= 2
                and wreck_old is not None and wreck_old >= 3):
            return False
        if self.source() == "checklist" == other.source():
            new_gaps, old_gaps = len(self.missing()), len(other.missing())
            if new_gaps != old_gaps:
                return new_gaps < old_gaps
        elif (self.source() == other.source()
                and self.accuracy() is not None
                and other.accuracy() is not None
                and self.accuracy() != other.accuracy()):
            # Non-None per the elif; mypy cannot narrow across method calls.
            return self.accuracy() > other.accuracy()  # type: ignore[operator]
        s_new = self.crit.score if self.crit else None
        s_old = other.crit.score if other.crit else None
        if s_new is None:
            # An unjudged candidate never displaces a judged one: "the judge
            # went quiet" is not evidence of improvement.
            return (s_old is None and self.accuracy() is not None
                    and other.accuracy() is None)
        if s_old is None:
            return True
        # Equal-or-better realism only wins when the request was equally met;
        # a checklist regression already lost above.
        return s_new > s_old

    def summary(self) -> str:
        parts = []
        acc = self.accuracy()
        if acc is not None:
            parts.append(f"matches {acc}% of the request")
        if self.crit is not None:
            parts.append(f"realism {self.crit.score:g}/10")
        return ", ".join(parts) or "unjudged"


class EventLog:
    """In-memory ring buffer of SYSTEM events (health monitor, service
    restarts, index refreshes). The Behind-the-Scenes tab merges these with
    per-job logs into one real-time execution stream."""

    def __init__(self, maxlen: int = 600):
        self._buf: deque[dict[str, str]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def log(self, level: str, msg: str, source: str = "system") -> None:
        entry = {"t": datetime.now(UTC).isoformat(timespec="milliseconds"),
                 "level": level, "source": source, "msg": msg}
        with self._lock:
            self._buf.append(entry)

    def list(self) -> list[dict[str, str]]:
        with self._lock:
            return list(self._buf)

    def clear(self) -> int:
        with self._lock:
            n = len(self._buf)
            self._buf.clear()
            return n


class _TextMaskWorker:
    """The CLIPSeg text engine as a resident subprocess.

    One-shot invocation paid a fresh torch import on EVERY regional edit —
    measured 35s warm and 132s with the machine under load. Resident, the
    model loads once and answers in ~2s (measured). The worker exits by
    itself after 10 idle minutes (its --idle-exit flag) so the ~700 MB it
    holds is only resident while edits are actually flowing; whoever asks
    next just respawns it."""

    IDLE_EXIT_S = 600.0
    # First answer covers a torch import and, on the first run ever, the
    # weight download — the same 300s budget the one-shot always had.
    READY_TIMEOUT_S = 300.0
    ASK_TIMEOUT_S = 120.0  # measured warm: 2-5s; generous for loaded machines

    def __init__(self, python: str, tool: str):
        self.python, self.tool = python, tool
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    @property
    def warm(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    def _readline(self, timeout: float) -> str | None:
        """One stdout line, or None on timeout/EOF. Windows pipes cannot be
        select()ed, so a throwaway reader thread carries the deadline; on
        timeout the caller kills the process and the thread dies on EOF."""
        box: list[str | None] = [None]
        proc = self._proc

        def read() -> None:
            try:
                # proc=None only surfaces here as the except arm's None.
                box[0] = proc.stdout.readline() if proc.stdout else None  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001 — surfaced as None
                box[0] = None

        t = threading.Thread(target=read, daemon=True)
        t.start()
        t.join(timeout)
        return None if t.is_alive() else (box[0] or None)

    def _read_json(self, timeout: float) -> dict | None:
        """The next JSON object line, skipping any library chatter that
        lands on stdout (hub warnings have been seen there)."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            line = self._readline(remaining)
            if line is None:
                return None
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue  # progress noise, not an answer
            if isinstance(data, dict):
                return data

    def _spawn(self) -> bool:
        try:
            flags = 0x08000000 if os.name == "nt" else 0  # no console window
            self._proc = subprocess.Popen(
                [self.python, self.tool, "--serve",
                 "--idle-exit", str(self.IDLE_EXIT_S)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1,
                creationflags=flags)
        except OSError:
            self._proc = None
            return False
        ready = self._read_json(self.READY_TIMEOUT_S)
        if not (ready and ready.get("ready")):
            self.stop(force=True)
            return False
        return True

    def ask(self, src: str, out: str, phrases: list[str],
            controls: list[str], threshold: float) -> dict | None:
        """One segmentation via the resident engine; None means the engine
        could not answer (the caller falls through to its next rung). Any
        failure kills the process so the NEXT ask starts clean."""
        with self._lock:
            if not self.warm and not self._spawn():
                return None
            try:
                assert self._proc is not None and self._proc.stdin is not None
                self._proc.stdin.write(json.dumps(
                    {"src": src, "out": out, "phrases": phrases,
                     "controls": controls, "threshold": threshold}) + "\n")
                self._proc.stdin.flush()
            except (OSError, ValueError, AssertionError):
                self.stop(force=True)
                return None
            answer = self._read_json(self.ASK_TIMEOUT_S)
            if answer is None or answer.get("error"):
                self.stop(force=True)
                return None
            return answer

    def stop(self, force: bool = False) -> None:
        """Kill the resident process. force=False is the memory-pressure
        path: it declines rather than block behind (or corrupt) an ask in
        flight — the idle watchdog reaps the process later anyway."""
        if not force:
            if not self._lock.acquire(blocking=False):
                return
            try:
                self._kill()
            finally:
                self._lock.release()
            return
        self._kill()

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
                # TerminateProcess returns before the process is gone; wait
                # so the interpreter file is actually unlocked (Windows) and
                # no zombie lingers.
                proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass


class _ThreadBoundLLM:
    """Resolves to `services.llm` at CALL time.

    Components built once at startup (scout, trust judge, workflow
    generator) would otherwise capture the main LLM chain forever; through
    this handle their calls follow the per-thread binding, so during peer
    delegation THEIR planning traffic also runs on the render machine —
    and tests that stub `services.llm` after construction are honoured."""

    def __init__(self, services: Services):
        self._services = services

    def complete(self, system: str, prompt: str, max_tokens: int = 4096,
                 schema: dict[str, Any] | None = None):
        # Via the schema-aware helper: services.llm may be a test stub
        # with the plain signature, which must keep working unchanged.
        return complete_with_schema(self._services.llm, system, prompt,
                                    max_tokens=max_tokens, schema=schema)

    @property
    def source(self) -> str:
        return self._services.llm.source

    def __getattr__(self, name: str):
        return getattr(self._services.llm, name)


class _EventLogJob:
    """Job-shaped shim (only .log) for job-taking machinery that runs
    OUTSIDE a queue job — the inline missing-node heal installs a pack in
    the middle of a render and logs to the event feed instead."""

    def __init__(self, events: EventLog, prefix: str = ""):
        self._events = events
        self._prefix = prefix

    def log(self, level: str, msg: str) -> None:
        try:
            self._events.log(level, f"{self._prefix}{msg}")
        except Exception:  # noqa: BLE001 — logging must never break a heal
            pass


class Services:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.settings.ensure_dirs()
        # Runtime settings persisted next to the data (survives restarts and
        # doesn't depend on env vars). A token set here — e.g. via the Settings
        # UI — wins over PROMPTFORGE_CIVITAI_TOKEN.
        self._settings_file = self.settings.data_dir / "settings.json"
        try:
            saved = json.loads(self._settings_file.read_text())
            if saved.get("civitai_token"):
                self.settings.civitai_token = str(saved["civitai_token"])
            if "lan_combine" in saved:
                self.settings.lan_combine = bool(saved["lan_combine"])
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
        self.db = Database(self.settings.db_path)
        self.store = AssetStore(self.db, self.settings)
        self.registry = ModelRegistry(self.db, self.settings.models_dir)
        self.downloader = ModelDownloader(self.registry,
                                          civitai_token=self.settings.civitai_token)
        # Custom safety rules live in the DB and are read live on every check
        # (never cached), on top of the non-removable built-in protections.
        self.safety_rules = SafetyRuleStore(self.db)
        self.safety = SafetyFilter(custom_provider=self.safety_rules.compiled)
        # LLM-discovered workflows are saved to a writable user dir (not the
        # packaged templates), and the library scans both.
        self._user_workflows = self.settings.data_dir / "workflows"
        self._user_workflows.mkdir(parents=True, exist_ok=True)
        self.workflows = WorkflowLibrary(self.settings.workflows_dir,
                                         user_dir=self._user_workflows)

        self.segmentation: SegmentationAdapter = self._build_segmentation()
        self.inpainting: InpaintingAdapter = self._build_inpainting()

        # Thread-local LLM/critic overrides: during peer delegation the
        # worker thread binds its planning and quality-check traffic to the
        # render machine, so the WHOLE job runs there — not just the pixels.
        # Must exist before the `llm`/`critic` properties are first read.
        self._llm_tls = threading.local()
        self._critic_tls = threading.local()
        self.llm = self._build_llm()
        # Components built once at startup get a thread-bound handle, so
        # their LLM calls also follow the per-thread binding at CALL time.
        self.workflow_ai = WorkflowGenerator(
            _ThreadBoundLLM(self), schema_provider=self._live_object_info)
        # Live /object_info per ENGINE (keyed by base_url): during
        # delegation self.comfy is a peer's proxy, and one shared cache
        # let a graph validate against the WRONG machine's nodes —
        # measured live as a BiRefNetRMBG graph sailing past the local
        # gate onto a peer that had never installed the pack.
        self._object_info_cache: dict[str, tuple[float, dict]] = {}
        # Which heavy model ComfyUI is believed to be holding, so a second
        # render of the SAME model doesn't unload and reload 10+ GB for
        # nothing. Cleared whenever the cache is actually dropped.
        self._comfy_heavy_cached: str | None = None
        self.model_search = ModelSearch(self.registry)
        # Thread-local ComfyUI override: the peer-delegation worker binds
        # ITS jobs' render traffic to another machine without touching what
        # every other thread sees. Must exist before the `comfy` property
        # is first read.
        self._comfy_tls = threading.local()
        # Mock mode means OFFLINE: the client itself is the null object, so
        # every probe anywhere in the app answers "down" without a request
        # leaving the process. Tests that stub `services.comfy = Fake()`
        # replace it either way (that is what the setter is for).
        self.comfy = (OfflineComfyClient()
                      if self.settings.inpaint_backend == "mock"
                      else ComfyUIClient(self.settings.comfyui_url))
        # A graph bounced for an uninstalled node type heals instead of
        # failing: install the curated pack here, or reroute a delegated
        # graph back to this machine (and ask the peer to install).
        try:
            self.comfy.on_missing_node = self._on_missing_node
        except Exception:  # noqa: BLE001 — a client without the hook just skips it
            pass
        self._heal_attempted: set[str] = set()
        self.hardware = probe(self.settings.data_dir)
        self.trust = TrustJudge(_ThreadBoundLLM(self))
        self.scout = ModelScout(
            _ThreadBoundLLM(self), self.model_search, self.downloader,
            self.registry, trust=self.trust,
            max_auto_bytes=max_auto_download_bytes(self.hardware))
        # The vision judge. "auto" resolves by hardware to qwen2.5-vl —
        # llava (the old default) was MEASURED hallucinating: a 1/10
        # garbage render got a 100% checklist, a blank gradient scored
        # 40% accuracy; the keep-best wreckage veto exists because of it.
        # During migration the new model may not be pulled yet, so llava
        # stays as the chain's fallback: the first checks still answer
        # while autopull fetches the upgrade in the background.
        self.critic_model = self._resolve_critic_model()
        if not self.critic_model:
            self.critic = None
        elif self.critic_model == "llava":
            self.critic = ImageCritic(self.settings.llm_url, "llava")
        else:
            self.critic = CriticChain(
                ImageCritic(self.settings.llm_url, self.critic_model),
                ImageCritic(self.settings.llm_url, "llava"))
        self.experience = ExperienceStore(self.db)
        self.model_index = ModelIndex(self.db, self.model_search)
        # Per-model capability notes (researched online, LLM-distilled,
        # 1-10 rated) in their own human-readable file.
        self.model_intel = ModelIntel(
            self.settings.data_dir / "model_knowledge.json",
            self.model_search)
        self._intel_queued: set[str] = set()
        self.events = EventLog()
        self._comfy_revive_lock = threading.Lock()
        self._monitor_stop = threading.Event()
        self._monitor: threading.Thread | None = None
        # Better-inpaint-model downloads already queued this session.
        self._inpaint_staged: set[str] = set()
        self._packs_queued: set[str] = set()
        # Face-polish support downloads already queued this session.
        self._polish_staged: set[str] = set()
        # Combine mode: peers currently carrying one of OUR delegated jobs.
        # Reserved by _delegate_wrap for the render's duration so parallel
        # delegation workers never double-book a machine in the seconds
        # before its own busy signal flips.
        self._reserved_peers: set[str] = set()
        self._reserve_lock = threading.Lock()
        # Scene graphs, cached per asset (built once, reused every step).
        self._scene_cache: dict[str, dict[str, Any]] = {}
        # Pending LLM-authored workflow candidates awaiting approval (in-memory
        # until the user approves one, which writes it into the library).
        self._workflow_candidates: dict[str, dict[str, Any]] = {}

        self.queue = JobQueue(self.db,
                              max_retries=self.settings.job_max_retries,
                              backoff_s=self.settings.job_retry_backoff_s)
        # Two PromptForge machines on one network help each other: model
        # weights copy from a peer before the internet (sha-verified), and
        # a busy queue hands whole render jobs to an idle peer. The peer
        # listener serves ONLY the model library and a ComfyUI proxy — the
        # app itself stays on 127.0.0.1.
        self.peers = PeerService(
            self.registry, comfy_url=self.settings.comfyui_url,
            share=self.settings.lan_share, render=self.settings.lan_render,
            # busy_local, not busy: a job PINNED to a remote machine is
            # waiting for that machine, not for this GPU — counting it
            # here made two machines pinned at each other refuse each
            # other's renders forever (a confirmed livelock).
            busy_check=self.queue.busy_local,
            static_hosts=[h for h in self.settings.lan_peers.split(",")
                          if h.strip()],
            stats_provider=self._machine_stats,
            env_provider=self._comfy_env_report,
            queue_provider=self._queue_public_snapshot,
            version_provider=self._version_info,
            auto_update=self.settings.peer_auto_update,
            llm_url=self.settings.llm_url,
            secret=self.settings.peer_secret)
        # A delegating peer whose graph needs a node we lack can ask us to
        # install the curated pack (our auto-install setting decides).
        self.peers.pack_installer = self._peer_pack_install
        # Sockets only open for real rendering setups: the mock backend is
        # what every test fixture uses, and hundreds of tests each opening
        # LAN listeners would fight over the ports for nothing.
        if self.settings.inpaint_backend != "mock":
            try:
                self.peers.start()
            except Exception as exc:  # noqa: BLE001 — LAN help is optional
                self.registry.notes["_peers"] = f"peer service off: {exc}"
        self.downloader.peer_source = self._peer_model_url
        self.peers.on_pull = self._accept_model_push
        self.queue.register("image_edit", self._handle_image_edit)
        self.queue.register("model_download", self._handle_model_download)
        self.queue.register("model_research", self._handle_model_research)
        self.queue.register("workflow", self._handle_workflow)
        self.queue.register("video", self._handle_video)
        self.queue.register("motion_transfer", self._handle_motion_transfer)
        self.queue.register("avatar", self._handle_avatar)
        self.queue.register("avatar_render", self._handle_avatar_render)
        self.queue.register("setup", self._handle_setup)
        self.queue.register("discover", self._handle_discover)
        self.queue.register("node_pack", self._handle_node_pack)
        # Updates arrive the way the project does: through git. The check
        # is an API call; applying is a visible job that pulls, refreshes
        # dependencies, and restarts into the new version.
        self.updates = UpdateManager(PROJECT_ROOT)
        self.queue.register("update", self._handle_update)
        # Version identities cross the LAN so machines can compare; when a
        # peer runs newer code, THIS machine triggers its own normal git
        # update — the peer is only the messenger, code never crosses.
        self._auto_update_seen: set[str] = set()
        self._auto_update_cooldown: dict[str, float] = {}
        # Single-flight: the hook does a git fetch, and it is invoked from
        # threads that must never block (the peer status pool, request
        # threads via add_peer, the delegation wrap). It therefore runs on
        # its own thread, at most one at a time — which also serializes
        # its check-then-act guards.
        self._auto_update_flight = threading.Lock()
        self.peers.on_newer_peer = self._newer_peer_async

        for model in DEFAULT_MODELS:
            existing = self.registry.get(model.name)
            if existing is None:
                self.registry.register(model)
            elif (existing.url != model.url
                  or (existing.meta or {}).get("folder")
                  != (model.meta or {}).get("folder")):
                self.registry.register(model)  # refresh definition; keeps status/path

        # A download interrupted by a crash leaves its model stuck in
        # "downloading" — reset those to not_downloaded so the Download button
        # reappears and the next attempt resumes from the partial .part file.
        for stuck in self.registry.reset_stale():
            self.registry.notes[stuck] = (
                "Previous download was interrupted — click Download to resume.")

        # Real segmentation needs its checkpoint: queue the (verified)
        # download up-front so mask proposals work without a manual step.
        if (self.settings.segment_backend == "sam" and self.settings.auto_install
                and not self.registry.is_ready("sam-vit-b")):
            self.queue.enqueue("model_download", {"model": "sam-vit-b"})

        # First run on this machine: profile the hardware and let the LLM
        # decide what to pre-stage for it (visible as a job in the Queue).
        hw_file = self.settings.data_dir / "hardware.json"
        if (not hw_file.exists() and self.settings.auto_install
                and self.settings.first_run_setup):
            try:
                hw_file.write_text(json.dumps(self.hardware.to_dict(), indent=1))
            except OSError:
                pass
            self.queue.enqueue("setup", {"hardware": self.hardware.to_dict()})

    def set_civitai_token(self, token: str) -> None:
        """Set the Civitai API token at runtime: applies to the live
        downloader immediately (no restart) and persists to settings.json so
        it survives restarts regardless of env vars."""
        token = (token or "").strip()
        self.settings.civitai_token = token
        self.downloader.civitai_token = token
        try:
            data = {}
            if self._settings_file.exists():
                data = json.loads(self._settings_file.read_text())
            data["civitai_token"] = token
            self._settings_file.write_text(json.dumps(data))
        except (OSError, json.JSONDecodeError):
            pass  # applies for this run even if persisting failed

    # Parallel delegation workers in combine mode: one job per idle peer,
    # a small fixed pool — more workers than machines just idle-poll.
    COMBINE_WORKERS = 3

    def set_lan_combine(self, on: bool) -> None:
        """Toggle combine mode at runtime and persist it. Turning it ON
        tops up the delegation workers immediately; turning it OFF lets
        the extra workers idle (eager() reads the live setting, so they
        stop taking beyond-head jobs at once)."""
        self.settings.lan_combine = bool(on)
        if (on and self.settings.lan_render
                and self.settings.inpaint_backend != "mock"):
            self.queue.start_helper(
                gate=self._peer_gate, wrap=self._delegate_wrap,
                types=self._DELEGATABLE, workers=self.COMBINE_WORKERS,
                eager=lambda: self.settings.lan_combine)
        self.events.log("info", "Combine mode "
                        + ("ON — the queue spreads across every connected "
                           "device" if on else "off — one helper again"))
        try:
            data = {}
            if self._settings_file.exists():
                data = json.loads(self._settings_file.read_text())
            data["lan_combine"] = bool(on)
            self._settings_file.write_text(json.dumps(data))
        except (OSError, json.JSONDecodeError):
            pass  # applies for this run even if persisting failed

    def _resolve_critic_model(self) -> str:
        """The vision judge for THIS machine. Explicit names ("llava",
        "qwen2.5vl:32b", anything) are honored verbatim; "" stays
        disabled; "auto" picks by hardware. The 7B is unloaded before
        every render like all Ollama models, so VRAM fit only has to
        cover the judging moments between renders."""
        configured = (self.settings.critic_model or "").strip()
        if configured != "auto":
            return configured
        hw = self.hardware
        if hw.vram_gb >= 6 or hw.ram_gb >= 12:
            return "qwen2.5vl:7b"
        return "qwen2.5vl:3b"

    def _build_llm(self) -> LLMClient:
        local = LocalLLM(self.settings.llm_url, self.settings.llm_model)
        api = (ClaudeLLM(self.settings.llm_api_model)
               if self.settings.llm_api_model else None)
        return FallbackLLM(local, api)

    def _build_segmentation(self) -> SegmentationAdapter:
        if self.settings.segment_backend == "sam":
            return SamSegmentationAdapter(self.registry)
        return MockSegmentationAdapter()

    def _build_inpainting(self) -> InpaintingAdapter:
        if self.settings.inpaint_backend == "comfyui":
            return ComfyUIInpaintingAdapter(
                self.settings.comfyui_url, self.workflows, self.registry)
        return MockInpaintingAdapter()


    def start(self) -> None:
        self.queue.start()
        # Downloads get their own lane on real setups: a multi-GB model
        # fetch must never make a render wait behind it. (Mock fixtures
        # keep the single-worker behaviour their assertions rely on.)
        if self.settings.inpaint_backend != "mock":
            self.queue.start_downloader()
        # The peer worker only ever takes a job when this machine is BUSY
        # and a discovered peer is idle — otherwise it sleeps. Combine
        # mode runs several workers so the whole queue spreads across
        # every connected device at once.
        if (self.settings.lan_render
                and self.settings.inpaint_backend != "mock"):
            self.queue.start_helper(
                gate=self._peer_gate, wrap=self._delegate_wrap,
                types=self._DELEGATABLE,
                workers=self.COMBINE_WORKERS
                if self.settings.lan_combine else 1,
                eager=lambda: self.settings.lan_combine)
        # Make sure the vision judge is actually on disk. `ollama pull` of
        # a present model is a near-instant no-op; a missing one downloads
        # in the background while the critic chain's llava fallback keeps
        # quality checks answering — this is what migrates every machine
        # to the new judge without a single failed check.
        if self.critic_model and self.settings.inpaint_backend != "mock":
            try:
                ollama_autopull(self.critic_model)
            except Exception:  # noqa: BLE001 — the 404 path re-triggers it
                pass
        # Reclaim disk for images deleted in previous sessions.
        try:
            purged = self.store.purge_trash()
            if purged:
                self.events.log("info", f"Cleaned up {purged} deleted "
                                        "image(s) from disk")
        except Exception:  # noqa: BLE001 — cleanup must never block startup
            pass
        # Health monitor: watches ComfyUI/Ollama and restarts them when they
        # die, and keeps the online model index fresh in the background.
        if self._monitor is None or not self._monitor.is_alive():
            self._monitor_stop.clear()
            self._monitor = threading.Thread(
                target=self._monitor_loop, name="pf-monitor", daemon=True)
            self._monitor.start()

    def stop(self) -> None:
        self._monitor_stop.set()
        self.queue.stop()
        try:
            self.peers.stop()
        except Exception:  # noqa: BLE001
            pass
        if self._text_mask_worker is not None:
            self._text_mask_worker.stop(force=True)
        if self._monitor:
            self._monitor.join(timeout=2)
        # Workers are joined; release the per-thread SQLite connections so
        # Windows can delete the database file (tests tear their temp dirs
        # down right after this).
        self.db.close()

    # -- peer network -------------------------------------------------------------
    # Job types worth sending across the network: the render-heavy ones.
    # Setup, downloads and research stay local by definition.
    _DELEGATABLE = frozenset({"image_edit", "workflow", "video",
                              "motion_transfer", "avatar", "avatar_render"})

    @property
    def comfy(self) -> Any:
        """This thread's ComfyUI. Delegated worker threads see the peer's
        proxy; every other thread sees the local client. The setter keeps
        the dozens of tests that stub `services.comfy = Fake()` working."""
        return getattr(self._comfy_tls, "client", None) or self._comfy_main

    @comfy.setter
    def comfy(self, client: Any) -> None:
        self._comfy_main = client

    @property
    def llm(self) -> Any:
        """This thread's LLM. During peer delegation the worker thread sees
        a chain whose FIRST stop is the render machine's Ollama (local as
        fallback); every other thread — and every test that stubs
        `services.llm = Fake()` via the setter — sees the main chain."""
        return getattr(self._llm_tls, "client", None) or self._llm_main

    @llm.setter
    def llm(self, client: Any) -> None:
        self._llm_main = client

    @property
    def critic(self) -> Any:
        """This thread's vision critic, peer-bound during delegation with
        the local critic as silent fallback. May be None (no critic model
        configured) — same contract as before."""
        return getattr(self._critic_tls, "client", None) or self._critic_main

    @critic.setter
    def critic(self, client: Any) -> None:
        self._critic_main = client

    def _handle_update(self, job: Job) -> dict[str, Any]:
        """Pull what was pushed, refresh what changed, restart into it.

        An AUTO-triggered update (payload carries the peer-announced
        commit) steps aside if any work arrived between its idle check
        and this moment — applying would restart the app underneath that
        work. A user-clicked update keeps its meaning: the user chose to
        restart now."""
        commit = str((job.payload or {}).get("commit") or "")
        if commit and self.queue.other_work(job.id):
            self._auto_update_seen.discard(commit)
            job.log("info", "Work arrived while this update waited — "
                            "stepping aside; the update retries by "
                            "itself once the queue is idle")
            return {"deferred": True, "commit": commit}
        if commit:
            # Counted BEFORE apply: a restart-into-rollback must still
            # register as a failed attempt on this machine.
            self._bump_auto_update_attempts(commit)
        try:
            return self.updates.apply(job)
        except UpdateError as exc:
            raise PermanentError(str(exc)) from exc

    _stats_cache: tuple[float, dict] | None = None
    _comfy_env_cache: tuple[float, dict] | None = None

    def _comfy_env_report(self) -> dict[str, Any]:
        """What ComfyUI's own environment holds — python, torch, GPU
        visibility — read straight from its venv on disk.

        Works while ComfyUI itself is DOWN, which is exactly when the
        other machine needs to see why: a peer that shows 'no ComfyUI'
        can now also show 'because its env is Python 3.13 with no torch',
        remotely."""
        cached = self._comfy_env_cache
        if cached is not None and time.time() - cached[0] < 60.0:
            return cached[1]
        out: dict[str, Any] = {}
        base = (Path(self.settings.comfyui_dir)
                if self.settings.comfyui_dir else None)
        if base is not None and base.exists():
            try:
                py = self._comfy_python(base)
                # One marker line, parsed from the END: importing a broken
                # GPU torch SPRAYS warnings onto stdout (measured live:
                # 'failed to run amdgpu-arch' landed where the version
                # belonged), so positional line parsing lies.
                code = ("import sys\n"
                        "v = sys.version.split()[0]\n"
                        "try:\n"
                        "    import torch\n"
                        "    t = torch.__version__\n"
                        "    try:\n"
                        "        g = int(torch.cuda.is_available())\n"
                        "    except Exception:\n"
                        "        g = 0\n"
                        "    try:\n"
                        "        import torch_directml\n"
                        "        g = g or int(torch_directml."
                        "device_count() > 0)\n"
                        "    except Exception:\n"
                        "        pass\n"
                        "except Exception:\n"
                        "    t, g = 'none', 0\n"
                        "print(f'PFENV|{v}|{t}|{g}')\n")
                probe = subprocess.run([py, "-c", code],
                                       capture_output=True, text=True,
                                       timeout=25)
                for line in reversed(
                        (probe.stdout or "").strip().splitlines()):
                    if line.startswith("PFENV|"):
                        _, ver, torch_v, gpu = (line.split("|") + [""])[:4]
                        out["python"] = ver
                        out["torch"] = None if torch_v == "none" else torch_v
                        out["gpu_visible"] = gpu == "1"
                        break
            except Exception:  # noqa: BLE001 — a blank report is honest
                pass
        self._comfy_env_cache = (time.time(), out)
        return out

    def _machine_stats(self) -> dict[str, Any]:
        """Live GPU/RAM numbers, shared with LAN peers (3s cache: the
        Network view on the other machine polls, and nvidia-smi is not
        free)."""
        cached = self._stats_cache
        # 10s: the WMI fallback for non-NVIDIA GPU names costs ~a second,
        # and dashboard freshness does not need more.
        if cached is not None and time.time() - cached[0] < 10.0:
            return cached[1]
        stats: dict[str, Any] = {}
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,memory.total,name",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=4, check=True,
            ).stdout.strip().splitlines()[0].split(", ")
            stats.update(gpu_util_pct=int(out[0]),
                         vram_used_mb=int(out[1]),
                         vram_total_mb=int(out[2]), gpu_name=out[3])
        except Exception:  # noqa: BLE001 — no NVIDIA GPU is not an error
            pass
        if "gpu_name" not in stats:
            # AMD/Intel machines have no nvidia-smi. The display-class
            # registry has both the NAME and the real VRAM total (the
            # driver writes qwMemorySize there) — no subprocess, no '0 GB
            # VRAM' for a perfectly good Radeon or Arc. Live utilisation
            # is NVIDIA-only; total still lets peers size delegation.
            try:
                name, vram_gb = hw_gpu_registry()
                if name:
                    stats["gpu_name"] = name
                if vram_gb > 0:
                    stats["vram_total_mb"] = int(vram_gb * 1024)
            except Exception:  # noqa: BLE001
                pass
        ram = hw_ram_stats()
        if ram is not None:
            stats.update(ram_used_gb=ram[0], ram_total_gb=ram[1])
        self._stats_cache = (time.time(), stats)
        return stats

    def _queue_public_snapshot(self) -> dict[str, Any]:
        """The queue picture a PEER may see: depth, pause state, and the
        running job's type/timing plus its stage KEYWORD only. Stage lines
        read '[stage] render — step 1/2: …' where the tail can carry the
        user's words — everything from the dash on is cut, so what crosses
        the LAN is 'render', never the prompt."""
        snap = self.queue.snapshot()
        running = snap.get("running")
        if running:
            stage = (running.get("stage") or "").split("—")[0].strip()
            snap["running"] = {
                "type": running.get("type"),
                "attempts": running.get("attempts"),
                "started_at": running.get("started_at"),
                "stage": stage or None,
            }
        return snap

    def _version_info(self) -> dict[str, Any] | None:
        """This install's version identity for peers to compare against.
        None when this is not a git clone — such installs neither trigger
        nor receive automatic peer updates."""
        updates = getattr(self, "updates", None)
        return updates.version() if updates is not None else None

    def _newer_peer_async(self, peer, info: dict[str, Any]) -> None:
        """Run the update check on its own thread, one at a time.

        The callers (status pool, request threads, the delegation wrap)
        must never block on the git fetch inside; the single-flight lock
        also serializes the guards so two peers announcing in the same
        tick cannot double-enqueue."""
        if not self._auto_update_flight.acquire(blocking=False):
            return

        def run() -> None:
            try:
                self._maybe_update_from_peer(peer, info)
            except Exception:  # noqa: BLE001 — a check must never crash
                logging.getLogger("promptforge.services").debug(
                    "auto-update check failed", exc_info=True)
            finally:
                self._auto_update_flight.release()

        threading.Thread(target=run, daemon=True,
                         name="pf-auto-update-check").start()

    # After this many failed apply attempts on one commit, automatic
    # updating for it stops (persisted across restarts): a push that
    # breaks only THIS machine would otherwise loop update → broken boot
    # → rollback → update, forever.
    AUTO_UPDATE_MAX_ATTEMPTS = 2

    def _auto_update_attempts(self) -> dict[str, int]:
        try:
            raw = json.loads((self.settings.data_dir
                              / "auto-update.json").read_text())
            return {str(k): int(v) for k, v in raw.items()
                    if isinstance(v, int | float)}
        except (OSError, ValueError, AttributeError):
            return {}

    def _bump_auto_update_attempts(self, commit: str) -> None:
        state = self._auto_update_attempts()
        state[commit] = state.get(commit, 0) + 1
        try:
            (self.settings.data_dir / "auto-update.json").write_text(
                json.dumps(state))
        except OSError:
            pass

    def _maybe_update_from_peer(self, peer, info: dict[str, Any]) -> None:
        """A LAN peer runs newer code than this install: catch up.

        Called (via the single-flight wrapper) by the peer status loop.
        The peer only tells us a newer version EXISTS — the update itself
        is the ordinary, visible git job: fast-forward from this
        install's own remote, dependency refresh, restart with automatic
        rollback. Every guard errs toward doing nothing: a wrong 'no'
        costs a manual update, a wrong 'yes' restarts the app under
        someone's work."""
        if not self.settings.peer_auto_update:
            return
        # Shutdown sets the monitor flag before the queue stops: a
        # version notice landing in that window must not enqueue into a
        # stopping queue (the orphaned row rehydrates as a spurious red
        # "interrupted" job at the next boot).
        if self._monitor_stop.is_set():
            return
        theirs = (info.get("version") or {})
        commit = str(theirs.get("commit") or "")
        if not commit or commit in self._auto_update_seen:
            return
        now = time.time()
        if now < self._auto_update_cooldown.get(commit, 0.0):
            return
        if not self.updates.is_repo():
            return
        # Never restart the app under running or queued work; the status
        # loop re-fires on its next tick once the queue drains (the seen
        # set is only marked once the version is actually dealt with).
        if self.queue.busy():
            return
        # One update job at a time, ever.
        if any(j.type == "update" and j.state.value in
               ("pending", "running", "retrying")
               for j in self.queue.list()):
            return
        # The peer announcing a commit does not mean OUR remote has it —
        # a dev machine runs local commits it never pushed, and pulling
        # would find nothing. Fetch first and only queue a job that will
        # actually change something; a fetch that fails (offline) retries
        # after a cooldown instead of hammering every status tick.
        status = self.updates.status(fetch=True)
        if status.get("error"):
            # Say so ONCE (the key outlives its expiry, so this event
            # cannot repeat), then retry quietly on a long cooldown —
            # offline is normal life for a LAN pair.
            if commit not in self._auto_update_cooldown:
                self.events.log(
                    "info",
                    f"'{peer.name}' runs newer code ({commit}) but the "
                    "update source could not be checked "
                    f"({str(status['error'])[:80]}) — retrying every "
                    "15 minutes")
            self._auto_update_cooldown[commit] = now + 900
            return
        if not status.get("behind"):
            self._auto_update_seen.add(commit)
            self.events.log(
                "info",
                f"'{peer.name}' runs {commit}, which is not on the "
                "update source yet (unpushed work?) — nothing to pull, "
                "leaving this install as it is")
            return
        if status.get("dirty"):
            # apply() would refuse a dirty checkout anyway — say it here
            # as one honest event instead of a guaranteed-red failed job.
            self._auto_update_seen.add(commit)
            self.events.log(
                "info",
                f"An update to {commit} is available, but this install "
                "has locally edited files — not updating automatically. "
                "Commit, stash or restore them, then update from "
                "Settings.")
            return
        if (self._auto_update_attempts().get(commit, 0)
                >= self.AUTO_UPDATE_MAX_ATTEMPTS):
            # Two applies of this commit already ended in rollback on
            # THIS machine (persisted across restarts — the in-memory
            # sets die with the process, the broken-push loop must not
            # revive with them). Stop trying; the person decides.
            self._auto_update_seen.add(commit)
            self.events.log(
                "error",
                f"Updating to {commit} failed {self.AUTO_UPDATE_MAX_ATTEMPTS} "
                "time(s) on this machine and was rolled back — automatic "
                "updating for it is paused. Update by hand from Settings "
                "when the cause is fixed.")
            return
        self._auto_update_seen.add(commit)
        self.events.log(
            "info",
            f"'{peer.name}' is running a newer PromptForge ({commit}) — "
            "updating this machine automatically (pulled from the normal "
            "update source, not from the peer)")
        self.queue.enqueue("update", {"reason": f"peer '{peer.name}' "
                                                f"announced {commit}",
                                      "commit": commit})

    def _peer_model_url(self, name: str) -> str | None:
        model = self.registry.get(name)
        return self.peers.find_model_url(name,
                                         model.sha256 if model else None)

    def _accept_model_push(self, entries: list[dict]) -> dict[str, Any]:
        """A peer offered its model library; queue what this machine lacks.

        Every accepted entry must carry a sha256 — the peer is untrusted,
        the pin is what later accepts the bytes — and each download runs
        as a normal, visible model_download job whose fetch tries the LAN
        first. Entries this machine never heard of are registered with the
        ORIGINAL internet URL from the manifest, so provenance survives
        even though the bytes arrive from next door."""
        queued: list[str] = []
        already: list[str] = []
        skipped: list[str] = []
        for entry in entries:
            name = str(entry.get("name") or "").strip()
            sha = str(entry.get("sha256") or "").strip()
            if not name or not sha:
                if name:
                    skipped.append(name)
                continue
            if self.registry.is_ready(name):
                already.append(name)
                continue
            if self.registry.get(name) is None:
                self.registry.register(ModelInfo(
                    name=name,
                    purpose=str(entry.get("purpose") or "shared by a "
                                "PromptForge on your network"),
                    license=str(entry.get("license") or "unknown"),
                    url=str(entry.get("url") or "") or None,
                    sha256=sha,
                    meta=dict(entry.get("meta") or {})))
            self.queue.enqueue("model_download", {"model": name})
            queued.append(name)
        if queued:
            self.events.log("info", f"A network peer offered its model "
                                    f"library — downloading {len(queued)} "
                                    "model(s) over the LAN (visible in the "
                                    "Queue)")
        return {"queued": queued, "already": already,
                "skipped_no_checksum": skipped}

    def push_models_to(self, host: str, port: int = 8765) -> dict[str, Any]:
        """Offer every ready, sha-pinned model to a peer, which queues the
        ones it is missing and fetches them over the LAN."""
        info = self.peers.add_peer(host, port)
        if info is None or info.get("self"):
            raise PermanentError(
                f"No other PromptForge answered at {host}:{port}.")
        peer = self.peers.find_peer(host) or self.peers.find_peer(
            str(info.get("name") or ""))
        if peer is None:
            raise PermanentError(f"Peer at {host} vanished mid-request.")
        manifest = []
        for m in self.registry.list():
            if m.status == "ready" and m.sha256 and m.path \
                    and Path(m.path).exists():
                manifest.append({
                    "name": m.name, "purpose": m.purpose,
                    "license": m.license, "url": m.url,
                    "sha256": m.sha256, "meta": m.meta or {}})
        result = self.peers.post_pull(peer, manifest)
        self.events.log("info", f"Offered {len(manifest)} model(s) to "
                                f"'{peer.name}' — it queued "
                                f"{len(result.get('queued') or [])} and "
                                f"already had "
                                f"{len(result.get('already') or [])}")
        return {"offered": len(manifest), **result}

    def pull_models_from(self, host: str, port: int = 8765) -> dict[str, Any]:
        """Ask a peer for its model library and queue everything this
        machine lacks — the inverse of push_models_to, run from the machine
        that WANTS the models. Same trust model as a push: every entry
        needs a sha256 pin, and the downloads are ordinary visible
        model_download jobs whose fetch tries the LAN copy first."""
        info = self.peers.add_peer(host, port)
        if info is None or info.get("self"):
            raise PermanentError(
                f"No other PromptForge answered at {host}:{port}.")
        peer_name = str(info.get("name") or host)
        try:
            entries = self.peers.fetch_manifest(host, port)
        except urllib.error.HTTPError as exc:
            if exc.code == 403:
                raise PermanentError(
                    f"'{peer_name}' is not sharing models — turn on "
                    "sharing in its Settings → Network.") from exc
            raise TransientError(
                f"'{peer_name}' refused its model list: HTTP {exc.code}"
            ) from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise TransientError(
                f"Could not read the model list from '{peer_name}': {exc}"
            ) from exc
        result = self._accept_model_push(entries)
        self.events.log("info",
                        f"Asked '{peer_name}' for its models — "
                        f"{len(entries)} offered, "
                        f"{len(result.get('queued') or [])} queued, "
                        f"{len(result.get('already') or [])} already here")
        return {"peer": peer_name, "offered": len(entries), **result}

    def _peer_gate(self) -> bool:
        with self._reserve_lock:
            taken = frozenset(self._reserved_peers)
        return self.peers.best_idle_peer(exclude=taken) is not None

    def _delegate_wrap(self, execute, job) -> None:
        """Run one job with its ComfyUI traffic bound to another machine.

        Two ways in: the user picked a device by hand (payload.device
        carries its host or name — honoured even when this machine is
        free), or automatic delegation found an idle peer while this
        machine was busy. The promise differs:

          hand-picked  the user said WHERE. If that machine cannot take
                       the job — gone, no ComfyUI, refusing renders —
                       the job FAILS with the reason, loudly: quietly
                       rendering on a machine the user did not pick looks
                       like success and is the one outcome they cannot
                       see. A machine that is merely BUSY is different —
                       "not yet" is not "cannot" — so the job waits at
                       the front of the queue until it frees up.

          automatic    the user said "whatever is fastest" — no reachable
                       idle peer simply means the job renders here."""
        target = (job.payload or {}).get("device")
        peer: Peer | None = None
        if target and target not in ("auto", "local"):
            found = self.peers.find_peer(target)
            info = (self.peers.add_peer(found.host, found.port,
                                        timeout=3.0)
                    if found is not None else None)
            problem: str | None = None
            if found is None or not info:
                problem = (f"'{target}' is not reachable on the network. "
                           "Check that the machine is on, PromptForge is "
                           "running there, and the firewall allows it "
                           "(allow-lan.ps1 on both machines).")
            elif not info.get("render"):
                problem = (f"'{found.name}' does not accept renders — "
                           "turn on render sharing in its Settings → "
                           "Network.")
            elif not (info.get("comfy") or {}).get("up"):
                problem = (f"'{found.name}' cannot render right now: its "
                           "ComfyUI is not running. Launch PromptForge "
                           "there (the launcher starts ComfyUI) or run "
                           "doctor.ps1 on that machine.")
            elif (job.type in ("video", "motion_transfer")
                  and str((info.get("comfy") or {}).get("device") or "")
                  .lower() in self.peers.VIDEO_INCAPABLE_DEVICES):
                problem = (f"'{found.name}' renders through "
                           f"{(info.get('comfy') or {}).get('device')}, "
                           "which cannot run WAN video (it crashes the "
                           "engine). Pick a machine with an NVIDIA or "
                           "native-ROCm GPU for video jobs.")
            elif not info.get("idle"):
                # Busy is temporary: hold the job at the front of the
                # queue and re-check in a few seconds. The pause also
                # stops this loop from hammering the peer with probes.
                self.queue.requeue_front(
                    job, f"[peer] '{found.name}' is busy with its own "
                         "work — waiting for it (pick Render: auto to "
                         "use whichever machine frees up first)")
                time.sleep(3.0)
                return
            if problem is not None or found is None:
                msg = (f"This job was pinned to '{target}' and was NOT "
                       f"rendered: {problem}")
                self.events.log("error", msg)
                self.queue.fail_job(job, msg)
                return
            peer = found
            job.log("info", f"[peer] rendering on '{found.name}' "
                            f"({found.host}) — chosen by hand")
            self.events.log("info", f"'{job.type}' renders on "
                                    f"'{found.name}' — chosen by hand")
        elif target != "local":
            with self._reserve_lock:
                taken = frozenset(self._reserved_peers)
            peer = self.peers.best_idle_peer(
                exclude=taken,
                video=job.type in ("video", "motion_transfer"))
            if peer is not None:
                verb = ("combine mode" if self.settings.lan_combine
                        else "this machine is busy")
                job.log("info", f"[peer] {verb} — '{peer.name}' "
                                f"({peer.host}) is idle, its GPU renders "
                                "this job")
                self.events.log("info", f"'{job.type}' renders on idle "
                                        f"'{peer.name}' ({verb})")
        if peer is None:
            execute(job)
            return
        # Reserve for the render's duration: combine mode runs several of
        # these wraps in parallel and each peer must carry ONE of our jobs.
        with self._reserve_lock:
            self._reserved_peers.add(peer.host)
        engine = ComfyUIClient(f"{peer.base}/pf-peer/comfy")
        engine.on_missing_node = self._on_missing_node
        self._comfy_tls.client = engine
        # The WHOLE job moves: planning and quality checks bind to the
        # render machine's Ollama too (proxied at /pf-peer/ollama), with
        # this machine's own LLM/critic as silent fallback — an older peer
        # without the proxy, or one missing a model, degrades gracefully
        # instead of failing the job. autopull=False: a 404 over there
        # must not pull models into THIS machine's Ollama; the peer heals
        # its own gap (its proxy starts the pull itself).
        peer_llm_base = f"{peer.base}/pf-peer/ollama/v1"
        self._llm_tls.client = FallbackLLM(
            LocalLLM(peer_llm_base, self.settings.llm_model,
                     autopull=False),
            self._llm_main)
        if self._critic_main is not None and self.critic_model:
            self._critic_tls.client = CriticChain(
                ImageCritic(peer_llm_base, self.critic_model),
                self._critic_main)
        job.log("info", f"[peer] the whole job runs on '{peer.name}' — "
                        "planning and quality checks included")
        try:
            execute(job)
        finally:
            self._comfy_tls.client = None
            self._llm_tls.client = None
            self._critic_tls.client = None
            with self._reserve_lock:
                self._reserved_peers.discard(peer.host)

    # -- health monitoring --------------------------------------------------------
    MONITOR_INTERVAL_S = 15
    INDEX_REFRESH_EVERY = 4 * 60  # monitor ticks (~1h) between index sweeps

    # Consecutive failed automatic ComfyUI restarts before the monitor
    # stops trying (and stops filling the event log). A crash restart
    # usually works on attempt one; a machine where ComfyUI CANNOT start
    # (not installed, no dir configured) used to get two error events
    # every 30 seconds forever — measured live on the test install.
    COMFY_RESTART_ATTEMPTS = 3

    def _monitor_loop(self) -> None:
        comfy_down = 0
        comfy_restart_fails = 0
        ollama_nagged = False
        tick = 0
        while not self._monitor_stop.wait(self.MONITOR_INTERVAL_S):
            tick += 1
            # Mock mode means OFFLINE (the _live_object_info rule): a mocked
            # instance must never probe, launch or revive the real services —
            # a resident ComfyUI or Ollama belongs to some other setup.
            # start() already applies this rule to the peer helper; without
            # it here this loop launches 'ollama serve' every 4th tick for
            # as long as a mock instance runs (the job-path twin of this
            # bug was measured live spending 33s per edit).
            revive = self.settings.inpaint_backend != "mock"
            # ComfyUI: two consecutive failed probes → restart it, a
            # bounded number of times per downtime. Jobs that need it
            # keep their own revival attempt (_require_comfy) and their
            # own honest failure messages either way.
            try:
                if revive and not self.comfy.is_up():
                    comfy_down += 1
                    if (comfy_down == 2
                            and comfy_restart_fails
                            < self.COMFY_RESTART_ATTEMPTS):
                        self.events.log("error", "ComfyUI is not responding — "
                                                 "restarting it")
                        # A RAISING spawn (unwritable log dir, broken env)
                        # is a failed attempt like any other: without this
                        # it skipped both the counter and the re-arm, and
                        # the loop went silently dead at comfy_down > 2.
                        try:
                            revived = self._respawn_comfy_clean()
                        except Exception:  # noqa: BLE001
                            revived = False
                        if revived:
                            self.events.log("info", "ComfyUI restarted and "
                                                    "healthy again")
                            comfy_restart_fails = 0
                        else:
                            comfy_restart_fails += 1
                            if (comfy_restart_fails
                                    >= self.COMFY_RESTART_ATTEMPTS):
                                self.events.log(
                                    "error",
                                    "ComfyUI could not be started after "
                                    f"{comfy_restart_fails} attempts — "
                                    "pausing automatic restarts until it "
                                    "answers again. Run doctor.ps1 (or "
                                    "the launcher) to fix its install; "
                                    "renders that need it will still say "
                                    "exactly what failed.")
                            else:
                                self.events.log("error",
                                                "ComfyUI restart failed; "
                                                "will keep trying")
                        comfy_down = 0  # re-arm the two-strike counter
                else:
                    # Fires on fails alone — comfy_down may already be 0
                    # right after the capping attempt, and the retraction
                    # must still happen or "pausing automatic restarts"
                    # stands as a false last word.
                    if comfy_restart_fails >= self.COMFY_RESTART_ATTEMPTS \
                            and revive:
                        self.events.log("info", "ComfyUI is answering again")
                    comfy_down = 0
                    comfy_restart_fails = 0
            except Exception:  # noqa: BLE001 — the monitor must never die
                pass
            # Ollama: same idea, every 4th tick (~1 min) — but one revival
            # per downtime, not one per minute: when it cannot come back
            # (not installed where llm_url points, or the URL is another
            # product entirely) the old loop spawned it and logged the same
            # error line forever, drowning Behind the Scenes. A job that
            # needs the LLM still gets its own attempt via _revive_ollama.
            try:
                if revive and tick % 4 == 0:
                    if ollama_is_up(self.settings.llm_url):
                        ollama_nagged = False
                    elif not ollama_nagged:
                        exe = shutil.which("ollama")
                        if exe:
                            self.events.log("error",
                                            "Ollama is not responding — "
                                            "restarting it")
                            self._spawn_ollama(exe)
                            ollama_nagged = True
            except Exception:  # noqa: BLE001
                pass
            # Model index: keep the online catalog fresh (network-quiet).
            try:
                if tick % self.INDEX_REFRESH_EVERY == 0:
                    refreshed = self.model_index.refresh_stale()
                    if refreshed:
                        self.events.log("info", "Refreshed online model index: "
                                                + ", ".join(refreshed))
            except Exception:  # noqa: BLE001
                pass
            # New models on DISK (dropped in by hand, or synced outside a
            # download job): research what each is best at, so the planner
            # can route prompts to it. Every ~10 min, a few at a time.
            try:
                if revive and tick % 40 == 0:
                    self._research_new_disk_models()
            except Exception:  # noqa: BLE001
                pass

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _refine_mask(mask: Image.Image) -> Image.Image:
        """Grow the edit region (~4 px) and feather its border so the blend
        has room to be seamless — the mask is corrected before the prompt is."""
        from PIL import ImageFilter
        grown = mask.filter(ImageFilter.MaxFilter(9))
        return grown.filter(ImageFilter.GaussianBlur(3))

    @staticmethod
    def decode_mask_b64(mask_b64: str) -> Image.Image:
        try:
            raw = base64.b64decode(mask_b64.split(",")[-1], validate=True)
            return Image.open(io.BytesIO(raw)).convert("L")
        except Exception as exc:
            raise BadMaskError(f"Mask could not be decoded: {exc}") from exc

    @staticmethod
    def encode_image_b64(image: Image.Image, fmt: str = "PNG") -> str:
        buf = io.BytesIO()
        image.save(buf, format=fmt)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def invalidate_asset_caches(self, asset_id: str) -> None:
        """The asset's working file changed (a version was promoted) — the
        cached scene graph describes the OLD pixels and must be rebuilt."""
        self._scene_cache.pop(asset_id, None)

    def open_asset_image(self, asset_id: str) -> Image.Image:
        asset = self.store.get_asset(asset_id)
        if asset is None:
            raise PermanentError(f"Asset {asset_id} does not exist.")
        if asset.kind != "image":
            raise PermanentError("Prompt-based video editing is on the roadmap; "
                                 "this build edits images.")
        try:
            return Image.open(asset.path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError) as exc:
            raise PermanentError(f"Asset file is missing or unreadable: {exc}") from exc

    def open_asset_frames(self, asset_id: str,
                          max_frames: int | None = None
                          ) -> tuple[list[Image.Image], float]:
        """An asset as a sequence of frames plus its frame rate.

        Handles the three things a "clip" can be here: an uploaded video, an
        animated image (the app's own renders are animated WEBP), and a plain
        still — which is simply a one-frame clip rather than an error.
        `open_asset_image` keeps its promise of returning ONE image; this is
        the parallel road for anything that moves."""
        asset = self.store.get_asset(asset_id)
        if asset is None:
            raise PermanentError(f"Asset {asset_id} does not exist.")
        path = Path(asset.path)
        if not path.exists():
            raise PermanentError(f"Asset file is missing: {path.name}")
        if asset.kind == "video":
            try:
                info = video_io.probe(path)
                frames = video_io.read_frames(path, max_frames=max_frames)
            except video_io.VideoError as exc:
                raise PermanentError(str(exc)) from exc
            return frames, info.fps
        try:
            with Image.open(path) as im:
                count = getattr(im, "n_frames", 1)
                wanted = (video_io.sample_indices(count, max_frames)
                          if max_frames and count > max_frames
                          else range(count))
                frames = []
                for idx in wanted:
                    im.seek(idx)
                    frames.append(im.convert("RGB"))
                # Animated images carry per-frame durations in ms.
                delay = (im.info or {}).get("duration") or 0
        except (FileNotFoundError, UnidentifiedImageError, EOFError) as exc:
            raise PermanentError(
                f"Asset file is missing or unreadable: {exc}") from exc
        if not frames:
            raise PermanentError("That asset contains no frames.")
        return frames, (1000.0 / delay if delay else 16.0)

    # -- job handlers ------------------------------------------------------------
    @staticmethod
    def _log_scores(job: Job, scores: dict[str, int] | None) -> None:
        if scores:
            job.log("info", "[llm] scores: "
                    + ", ".join(f"{k} {v}" for k, v in scores.items()))

    def _handle_image_edit(self, job: Job) -> dict[str, Any]:
        """The staged edit pipeline, now multi-step: the LLM decomposes the
        request into an ordered plan of workflows (inpaint / img2img /
        outpaint), each step's output feeds the next, every step reports the
        workflow + model it used, and the final image is inspected, scored
        and iterated toward the quality target."""
        p = job.payload
        asset_id, prompt = p["asset_id"], p["prompt"]
        self._log_eta(job)
        if self.critic is not None:
            self._revive_ollama(job)  # the vision judge runs on Ollama
        job.log("info", f"Loading asset {asset_id}")
        image = self.open_asset_image(asset_id)
        real = not self.inpainting.is_mock
        user_mask_b64 = p.get("mask_b64")
        # The payload fact, kept separately: user_mask_b64 is cleared once
        # the drawn mask is consumed by its step, so it cannot answer "did
        # the user draw one?" at verify time.
        drew_mask = bool(user_mask_b64)
        # Extra photos to draw FROM. Their presence is the whole reason the
        # user attached them, so it also decides the route below.
        references: list[Image.Image] = []
        for ref_id in (p.get("reference_asset_ids") or [])[:3]:
            try:
                references.append(self.open_asset_image(ref_id))
            except Exception as exc:  # noqa: BLE001 — name the bad one
                raise PermanentError(
                    f"The second image ({ref_id}) could not be opened: "
                    f"{exc}") from exc
        if references:
            job.log("info", f"{len(references)} reference image(s) attached — "
                            "their subject will be brought into this photo")

        # Stage 1 — the LLM reads the request and builds the step plan.
        steps: list[dict[str, Any]] | None = None
        invented: list[dict[str, Any]] = []
        if real:
            job.log("info", "[stage] analyze — reading the request and "
                            "planning the workflow steps")
            # The LLM routes, but it doesn't get to invent work: plan_edit
            # prunes padding deterministically and reports the drops here.
            steps = quality.plan_edit(self.llm, prompt,
                                      has_mask=bool(user_mask_b64),
                                      dropped=invented)
        # Logged whether or not anything survived. When EVERY step was
        # invented the plan comes back empty, and that is exactly the case
        # the user most needs told: the request falls through to the default
        # single step below, and without this line the only trace was
        # "single inpaint step (default)" with no hint that a step had been
        # thrown away. Seen live: "place the man from the second photo
        # standing beside her" planned SWAP_FACE alone, which is dropped —
        # correctly — leaving nothing.
        for d in invented:
            job.log("info", f"[llm] plan: dropped invented {d['task']} "
                            f"step ('{d['instruction'][:60]}') — "
                            f"{d['why']}")
        if steps:
            for s in steps:
                # Adding NEW content is a regional edit by definition —
                # img2img would repaint the whole photo around it (seen
                # live: "put a dog in the background" routed to img2img and
                # no dog appeared). Deterministic, content-neutral coercion.
                if (s["task"] == "img2img"
                        and quality.classify_edit(s["instruction"]) == "add"):
                    s["task"] = "inpaint"
                    job.log("info", "[llm] plan: adding new content is a "
                                    "regional edit — routed to inpaint with "
                                    "a placement region")
                # A format/aspect-ratio change extends the CANVAS — routed
                # anywhere else it would restyle or repaint the photo and
                # the frame would never actually change (seen live: qwen
                # sent "change the format of the image" to CHANGE_STYLE).
                elif (s["task"] in ("img2img", "custom", "inpaint")
                        and quality.format_intent(s["instruction"])):
                    s["task"] = "outpaint"
                    s["operation"] = "OUTPAINT"
                    job.log("info", "[llm] plan: format/aspect change is a "
                                    "canvas extension — routed to outpaint")
        if not steps:
            # No planner: one deterministic step. Capability intents (animate,
            # background) reach their own engine even with no LLM at all; an
            # add-edit still gets a placement region instead of segmentation.
            step = quality.default_edit_step(prompt)
            steps = [step]
            if real:
                detail = {
                    "video": "animate request — image-to-video",
                    "background": "background replacement — inverting the "
                                  "subject matte",
                    "outpaint": "canvas extension",
                    "pose": "pose change",
                    "angles": "new camera angles",
                    "scene3d": "3D scene rebuild",
                    "relight": "relighting",
                    "inpaint": "single inpaint step",
                }.get(step["task"], step["task"])
                job.log("info", f"[llm] plan: {detail} (default, no planner)")
        # A second image attached IS the request to combine them — whatever
        # the planner called the step. Deterministic, like the animate and
        # viewpoint guarantees: the user does not lose a capability because a
        # 7B model chose a different label.
        # Swapping a FACE and transplanting a whole PERSON are different jobs.
        # Seen live: "face swap" planned as COMPOSE(face), which mattes the
        # entire subject — it would have pasted a complete stranger into the
        # photo rather than replacing a face.
        wants_face = references and quality.face_intent(prompt)
        if wants_face and not any(s["task"] == "faceswap" for s in steps):
            for s in steps:
                if s["task"] in ("inpaint", "img2img", "custom", "compose"):
                    s["task"], s["operation"] = "faceswap", "SWAP_FACE"
                    break
            else:
                steps.insert(0, {
                    "task": "faceswap", "operation": "SWAP_FACE", "target": "face",
                    "instruction": prompt[:300], "mask_adjust": "keep",
                    "adjust_px": 0, "denoise": 0.28,
                    "reason": "the request asks to replace a face"})
            steps = quality.order_steps(steps)
            job.log("info", "[llm] plan: this replaces a FACE, not the whole "
                            "person — routed to the face swap")
        # NOTE the task list: a plan that already handles the second image —
        # whether as a compose OR a face swap — must not have a compose added
        # on top. Seen live: the planner emitted SWAP_FACE itself, this branch
        # saw no compose step and added one, and the job did both (a whole
        # stranger transplanted AND their face swapped).
        elif references and not any(s["task"] in ("compose", "faceswap")
                                    for s in steps):
            for s in steps:
                if s["task"] in ("inpaint", "img2img", "custom"):
                    s["task"], s["operation"] = "compose", "COMPOSE"
                    break
            else:
                steps.insert(0, {
                    "task": "compose", "operation": "COMPOSE", "target": "",
                    "instruction": prompt[:300], "mask_adjust": "keep",
                    "adjust_px": 0, "denoise": 0.22,
                    "reason": "a second image was attached to combine"})
            steps = quality.order_steps(steps)
            job.log("info", "[llm] plan: a second image is attached — the "
                            "first step combines them")
        # Decide NOW whether the transplant can actually happen, so a step
        # that has to degrade is already the thing it degrades to before the
        # renderer starts walking the plan.
        if real and any(s["task"] == "compose" for s in steps):
            ok, why = self._template_runnable("compose")
            if ok and not self._pack_active("rmbg"):
                ok, why = False, "the rmbg matting pack is not active"
            if not ok:
                # Without a matte this pastes a rectangle, which is worse than
                # not trying. Paint the subject into the scene instead, and be
                # explicit that the result will RESEMBLE the reference rather
                # than be it.
                job.log("info", f"The subject cannot be transplanted ({why}) "
                                "— painting it into the scene instead. The "
                                "result will resemble your second photo "
                                "rather than contain it.")
                for s in steps:
                    if s["task"] == "compose":
                        s["task"], s["operation"] = "inpaint", "ADD_OBJECT"
        # FLUX.1 Kontext edits from the sentence alone, with no mask at all,
        # which removes the largest single source of wrong edits here: a
        # masked inpaint must be told WHERE the object is, and when the region
        # was wrong "remove the hat" painted a different hat instead of
        # removing one (D1). Only instruction-shaped operations move, only
        # when every weight is already on disk, and never over a region you
        # drew yourself — a mask you drew is an instruction in its own right.
        # A drawn mask is consumed by the FIRST inpaint step, whichever index
        # that is — not by step 0. Exempting index 0 would hand the mask's
        # step to Kontext whenever the plan happened to open with something
        # else, and the region you drew would be silently discarded.
        first_inpaint = next((i for i, s in enumerate(steps)
                              if s["task"] == "inpaint"), None)
        eligible = [i for i, s in enumerate(steps)
                    if s["task"] == "inpaint"
                    and s.get("operation") in quality.KONTEXT_OPERATIONS
                    and not (user_mask_b64 and i == first_inpaint)]
        if real and eligible:
            ok, why = self.kontext_ready()
            if ok:
                for i in eligible:
                    steps[i]["task"] = "kontext"
                job.log("info", f"Editing with FLUX.1 Kontext instead of a "
                                f"masked inpaint for {len(eligible)} step"
                                f"{'s' if len(eligible) > 1 else ''} — it "
                                "reads the whole picture and needs no mask, "
                                "so it cannot edit the wrong region")
            else:
                job.log("info", f"FLUX.1 Kontext would edit this without a "
                                f"mask, but it cannot run here ({why}) — "
                                "using the masked inpaint route")
        if steps:
            job.log("info", "[llm] plan: " + " → ".join(
                f"{i + 1}. {s.get('operation') or s['task']}"
                f"({s.get('target') or '—'})"
                for i, s in enumerate(steps)))

        # Batch the TEXT-model work before the VISION pass. On a single
        # 8 GB GPU the two models evict each other, and every swap is a
        # full reload — measured on a live edit: plan (text, 21.6 s load)
        # → scene (vision, 22.5 s) → enhance (text AGAIN, 17.1 s reload).
        # Enhancing every step here, while the text model is still warm
        # from planning, removes one whole reload per edit job.
        if real and steps:
            for s in steps:
                s["_enh"] = quality.enhance_prompt(
                    self.llm, s["instruction"], s["task"])
            # The adherence checklist reads only the INSTRUCTION, so it
            # belongs in this warm-text batch too. Built after the render
            # it was the judging chain's text-model reload: scorecard
            # (vision) → checklist (text, full reload) → probes (vision
            # AGAIN). Verify falls back if the plan mutated.
            steps[-1]["_checklist"] = quality.request_checklist(
                self.llm, steps[-1].get("instruction") or prompt)
            for s in steps:
                if s["task"] == "background":
                    # Environment planning is TEXT work: do it while the
                    # planner is warm (the same eviction economics as _enh
                    # and _checklist above). No subject facts are measured
                    # yet, so none are claimed — the spec plans the PLACE;
                    # the measured geometry joins at render time.
                    s["_env_spec"] = scene_geometry.environment_spec(
                        self.llm, s.get("instruction") or prompt,
                        None, None)

        # Image Understanding: one rich scene graph is built per image and
        # reused by every step — planning context, placement, targeted
        # masking, and prompt grounding all read it. A diffusion model told
        # only "change the shirt" invents content blind; the graph tells it
        # what the photo IS.
        # ...but only when a step will actually read it. The vision pass costs
        # minutes — measured here at 487 s against a 150 s render — and every
        # reader of it below (placement boxes, targeted masking, scene-grounded
        # prompts) belongs to an engine other than Kontext. Kontext is handed
        # the picture and the sentence and needs neither, so a plan made
        # entirely of Kontext steps would pay for an answer nobody asks for.
        if steps and all(s["task"] == "kontext" for s in steps):
            job.log("info", "Skipping the scene analysis — this edit goes to "
                            "Kontext, which reads the picture itself")
            scene_graph: dict[str, Any] = {}
            scene: str | None = ""
        else:
            scene_graph = self._scene_graph(job, asset_id, image, real)
            scene = scene_module.summary(scene_graph)
            if scene:
                job.log("info", f"Scene understood: {scene[:120]}")

        def with_scene(positive: str) -> str:
            return f"{positive}; scene: {scene}" if scene else positive

        plan_report: list[dict[str, Any]] = []
        current = image
        final_input = image           # input to the LAST step (for retries)
        last_step = steps[-1]
        last_mask: Image.Image | None = None
        last_inpaint: dict[str, Any] | None = None  # variant/model for retries
        last_outpaint: dict[str, Any] | None = None  # prompts/model for retries
        last_positive = prompt
        last_negative = ""
        env_card: scene_geometry.SceneCard | None = None
        env_spec: dict[str, Any] | None = None
        env_guide: Image.Image | None = None
        env_misses: list[str] = []
        result_adapter = self.inpainting.name
        result_is_mock = not real
        rounds = 0
        scores: dict[str, int] | None = None
        objective: dict[str, Any] | None = None
        obj_flags: list[str] = []
        # A 3D scene, when one was asked for. Carried out separately from the
        # image because a GLB is not a version of the photograph.
        scene_result: dict[str, Any] | None = None

        def describe(task: str) -> tuple[str, str]:
            """(workflow name, model name) a step will use — for the report."""
            if task == "inpaint" and not real:
                return "mock inpaint", "mock"
            try:
                t = self.workflows.load(task)
                wf = f"{t['template']}_v{t['version']} template"
                model = (self._recipe_facts(t.get("graph", {}))
                         .get("checkpoint") or "?")
                return wf, str(model)
            except Exception:  # noqa: BLE001 — reporting is best-effort
                return f"{task} workflow", "?"

        try:
            for i, step in enumerate(steps):
                n = len(steps)
                # Prompt optimization (append-only — the user's words always
                # survive verbatim; only safety.py may ever filter content).
                # Enhanced up front, before the vision pass, so the text
                # model is not reloaded mid-job (see the batch above).
                if real:
                    enh = step.get("_enh") or quality.enhance_prompt(
                        self.llm, step["instruction"], step["task"])
                    added = enh["positive"][len(step["instruction"]):]
                    if added:
                        job.log("info", "[llm] prompt enhanced: "
                                        f"+{added.lstrip(', ')[:120]}")
                else:
                    enh = {"positive": step["instruction"], "negative": ""}
                wf_name, model_name = describe(step["task"])
                job.log("info", f"[stage] plan — step {i + 1}/{n}: "
                                f"{step['task']} using {wf_name} "
                                f"(model: {model_name})")
                if i == n - 1:
                    final_input = current
                    last_positive = enh["positive"]
                    last_negative = enh["negative"]

                if step["task"] == "inpaint":
                    # The LLM picks WHICH model inpaints this step (it may
                    # keep the default, use any installed checkpoint, or
                    # search online and download a better one); the inpaint
                    # technique follows the model.
                    variant = ckpt = None
                    supports = real and getattr(self.inpainting,
                                                "supports_variants", False)
                    if supports:
                        variant, ckpt = self._choose_inpaint(
                            job, step["instruction"])
                    job.log("info", f"[stage] mask — step {i + 1}/{n}: "
                                    "selecting the edit region")
                    op = step.get("operation", "")
                    target = step.get("target", "") or ""
                    is_add = (op == "ADD_OBJECT"
                              or (not op and quality.classify_edit(
                                  step["instruction"]) == "add"))
                    is_remove = op == "REMOVE_OBJECT"
                    if is_remove:
                        # Removal means removal. The instruction must never
                        # reach the model as positive conditioning — "remove
                        # the hat" as the thing to PAINT painted a hat (D1).
                        # What should be there goes in the positive, the
                        # object goes in the negative.
                        enh = quality.removal_conditioning(
                            step["instruction"], target, enh["negative"])
                        job.log("info", "[llm] removal conditioning: the "
                                        "model is asked for the scene "
                                        "without the object; the object "
                                        "itself goes to the negative prompt")
                    elif is_add:
                        # The placement box is bigger than the object, and
                        # the model fills what it is given — "add a pair of
                        # sunglasses" returned three pairs (D21). One object
                        # per request unless the words say otherwise.
                        enh = {**enh,
                               "positive": f"{enh['positive']}, a single "
                                           "object, exactly one",
                               "negative": f"{enh['negative']}, multiple, "
                                           "duplicated, repeated, more than "
                                           "one, extra copies".strip(", ")}
                    else:
                        # CHANGE/REPLACE: lead the conditioning with the
                        # TARGET STATE. "Change the shirt to red" hands the
                        # sampler "shirt" as its strongest token — which the
                        # region already contains — and the first attempt
                        # comes back unchanged, fails the checklist, and
                        # buys the retry the user then waits through. Same
                        # construction that fixed removals (D1): describe
                        # the end state, demote the displaced one.
                        cond = quality.attribute_conditioning(
                            step["instruction"], target,
                            enh["positive"], enh["negative"])
                        if cond:
                            enh = cond
                            job.log("info", "[llm] conditioning leads with "
                                            "the target state: "
                                            + cond["positive"][:90])
                    if i == n - 1:
                        last_positive = enh["positive"]
                        last_negative = enh["negative"]
                    # WHERE this region came from, so the advisory check below
                    # knows what it is allowed to re-cut. Bound BEFORE the
                    # branch on purpose: the previous version read the
                    # chooser's `choice` at the check site, which only the
                    # third branch ever assigned, so every edit that arrived
                    # with a mask died on an UnboundLocalError the moment the
                    # check objected.
                    mask_source: str | None = None
                    if user_mask_b64:
                        job.log("info", "Using user-corrected mask")
                        mask = self.decode_mask_b64(user_mask_b64)
                        user_mask_b64 = None  # the drawn mask is step 1's
                        mask_source = "user"
                    elif is_add:
                        # NEW content: segmentation can only find what
                        # already exists — masking "the background" repaints
                        # everything and renders nothing new. Place a region
                        # for the object, informed by the scene graph's
                        # perspective + lighting.
                        job.log("info", "New content requested — choosing a "
                                        "placement region for it (instead of "
                                        "segmenting existing objects)")
                        mask = quality.propose_placement(
                            self.critic if real else None, current,
                            step["instruction"],
                            context=scene_module.placement_context(scene_graph))
                        mask_source = "placement"
                        if real:
                            self._save_step_mask(job, asset_id, mask,
                                                 "placement mask")
                    else:
                        # Existing-object edit: tell segmentation WHICH object
                        # (and, from the scene graph, WHERE it is) so it cuts
                        # the right thing instead of re-deriving from the
                        # whole instruction.
                        # The scene graph's "in the center-right of the image"
                        # hint is no longer threaded in: it existed to steer a
                        # segmenter that could only reason about position, and
                        # the chooser now leads with engines that match on
                        # appearance, where a location phrase is noise.
                        # ONE chooser, shared with the preview, so the region
                        # you approved on screen is the region that renders.
                        # chooser_request keeps the parse alive: a prefixed
                        # target used to defeat segment-the-source and put
                        # the DESTINATION back among the phrases.
                        choice = self.auto_mask(
                            current,
                            quality.chooser_request(target,
                                                    step["instruction"]),
                            job=job)
                        if not choice.ok:
                            raise BadMaskError(
                                f"{choice.reason.capitalize()}. Paint the "
                                "region yourself if it is there and I have "
                                "missed it.")
                        mask = cast(Image.Image, choice.mask)  # ok-checked
                        mask_source = choice.source
                        job.log("info", f"Region selected by {choice.source}"
                                + (f" — {'; '.join(choice.notes)}"
                                   if choice.notes else ""))
                        if real:
                            # Make the auto-generated mask visible in the UI.
                            self._save_step_mask(job, asset_id, mask,
                                                 "auto-generated mask")
                    if step["mask_adjust"] != "keep" and step["adjust_px"]:
                        mask = quality.adjust_mask(mask, step["mask_adjust"],
                                                   step["adjust_px"])
                        job.log("info", f"Mask {step['mask_adjust']}n by "
                                        f"~{step['adjust_px']}px for a "
                                        "seamless blend")
                    if real and self.critic is not None and not is_add:
                        # (Placement masks mark EMPTY space for new content —
                        # coverage checks only make sense for existing objects.)
                        check = quality.verify_mask(self.critic, current,
                                                    mask, step["instruction"])
                        if check is not None:
                            job.log("info", "[llm] mask check: "
                                    + ("matches the request" if check["match"]
                                       else "may not match")
                                    + (f" — {check['why']}"
                                       if check["why"] else ""))
                            if not check["match"]:
                                # Advisory, not decisive: this check is
                                # shown the request, which is the mode
                                # measured unreliable here. It may prompt a
                                # better attempt; it cannot discard a mask
                                # that passed the deterministic gates.
                                #
                                # And it never re-cuts a region the USER
                                # settled. A drawn region — or the proposed
                                # one they reviewed on screen and pressed Run
                                # on — is the most specific instruction there
                                # is, and the preview's whole promise is that
                                # what you approved is what renders. So the
                                # objection is reported and the region stands.
                                # "whole-frame" is deliberate too: the text
                                # engine confidently found the named thing
                                # everywhere, and re-cutting a smaller region
                                # out of that would undo the routing.
                                if mask_source == "whole-frame":
                                    job.log("info",
                                            "Keeping the whole frame — the "
                                            "request names essentially the "
                                            "entire picture, and the check "
                                            "is advisory")
                                elif mask_source in (None, "user"):
                                    job.log("info",
                                            "Keeping the region you approved "
                                            "— the check is advisory and your "
                                            "region is the more specific "
                                            "instruction")
                                else:
                                    mask = self._correct_mask(
                                        job, asset_id, current, mask, step,
                                        check.get("why") or "", mask_source)
                    if (supports and variant == "modern"
                            and quality.mask_fraction(mask) < 0.35):
                        # Regional edit: crop→upscale→inpaint→stitch renders
                        # the region at the model's native resolution — small
                        # regions rendered at full-frame scale come out
                        # low-detail and deformed.
                        variant = "hires"
                        job.log("info", "Inpaint technique: regional edit — "
                                        "hi-res crop&stitch so the region "
                                        "renders at full detail")
                    inpaint_denoise = None
                    if (supports and not is_add and not is_remove
                            and quality.is_recolour(step["instruction"])):
                        # A recolour is a structure-preserving edit: at
                        # replacement denoise the repaint regenerates the
                        # object (a recoloured car came back as a different
                        # vehicle, D22). The universal latent-mask template
                        # is the one that exposes the denoise dial.
                        variant = "universal"
                        inpaint_denoise = 0.45
                        job.log("info", "Inpaint technique: recolour — low "
                                        "denoise so the object keeps its "
                                        "shape and only its colour changes")
                    elif (supports and mask_source == "whole-frame"
                            and not is_add and not is_remove
                            and op != "REPLACE_OBJECT"
                            and not quality.is_replacement(
                                step["instruction"])):
                        # A whole-frame ATTRIBUTE change ("make the sky a
                        # warm sunset" on a photo that is nearly all sky) at
                        # replacement denoise generates a NEW picture that
                        # merely matches the words — the user's photo is
                        # gone. The same structure-preserving principle as
                        # the recolour above, with a little more freedom:
                        # tone and atmosphere may change, composition stays.
                        # An explicit REPLACE keeps full denoise — swapping
                        # the content is what was asked.
                        #
                        # Measured live (RTX 4060, universal template,
                        # 512px skyline test): 0.45 changed almost nothing,
                        # 0.6 kept every building and warmed the light,
                        # 1.0 gave the dramatic sunset. 0.6 is the right
                        # FLOOR — when the checklist later confirms the
                        # asked-for change did not land, escalation (not a
                        # bigger first guess) is the mechanism to spend
                        # more denoise.
                        variant = "universal"
                        inpaint_denoise = 0.6
                        job.log("info", "Inpaint technique: whole-frame "
                                        "restyle — moderate denoise so the "
                                        "picture keeps its composition and "
                                        "only its look changes")
                    # A removal's positive prompt must stay free of the scene
                    # summary too — it can name the very object being removed.
                    step_positive = (enh["positive"] if is_remove
                                     else with_scene(enh["positive"]))
                    step_negative = enh["negative"]
                    if is_remove:
                        # A big emptied region invites an invented subject;
                        # name the usual ones in the negative (coverage-gated
                        # so a small on-subject removal is untouched).
                        fillers = quality.removal_fillers_negative(
                            quality.mask_fraction(mask))
                        if fillers:
                            step_negative = f"{step_negative}, {fillers}"
                            job.log("info", "Removing a large region — the "
                                            "negative also refuses an "
                                            "invented person or creature in "
                                            "the emptied space")
                    job.log("info", f"[stage] render — step {i + 1}/{n}: "
                                    f"inpainting via {self.inpainting.name}"
                                    + ("" if real else " [mock]"))
                    self._free_vram(job)
                    if supports:
                        # denoise only travels when a recolour set it — older
                        # adapter signatures (and test fakes) need not know
                        # the parameter exists.
                        extra = ({"denoise": inpaint_denoise}
                                 if inpaint_denoise is not None else {})
                        # Kwargs the supports-check above proved available.
                        res = self.inpainting.inpaint(  # type: ignore[call-arg]
                            current, mask, step_positive,
                            negative=step_negative,
                            checkpoint=ckpt, variant=variant or "modern",
                            **extra)
                        if res.meta.get("template"):
                            wf_name = f"{res.meta['template']} template"
                        if ckpt:
                            model_name = ckpt
                        if i == n - 1:
                            last_inpaint = {"checkpoint": ckpt,
                                            "variant": variant or "modern"}
                            if inpaint_denoise is not None:
                                # The structure guard must survive into the
                                # retry recipe — without this a whole-frame
                                # restyle retry rendered at the template
                                # DEFAULT denoise, regenerating the very
                                # picture the guard exists to keep.
                                last_inpaint["denoise"] = inpaint_denoise
                    else:
                        res = self.inpainting.inpaint(current, mask,
                                                      step_positive)
                    current = res.image
                    last_mask = mask
                    result_adapter = res.adapter
                    result_is_mock = res.is_mock
                elif step["task"] == "video":
                    # ANIMATE: the edited still becomes the first frame of a
                    # WAN i2v render. Saved as its own asset (a video is not
                    # an image edit), so the chain records it and stops.
                    animated = self._animate_current(
                        job, asset_id, current, step["instruction"],
                        with_scene(enh["positive"]))
                    plan_report.append({
                        "step": i + 1, "task": "video",
                        "operation": step.get("operation", "ANIMATE"),
                        "target": step.get("target", ""),
                        "instruction": step["instruction"][:120],
                        "workflow": "WAN 2.2 image-to-video",
                        "model": "wan2.2-i2v", "video_asset": animated})
                    job.log("info", "[stage] save — animation stored as a "
                                    "video asset")
                    return {"asset_id": animated, "kind": "video",
                            "plan": plan_report,
                            "adapter": "comfyui-video", "is_mock": False}
                elif step["task"] == "faceswap" and not references:
                    job.log("info", "A face swap needs a second photo to take "
                                    "the face FROM — none was attached, so "
                                    "this step is skipped rather than "
                                    "inventing a stranger's face")
                    continue
                elif step["task"] == "compose" and not references:
                    job.log("info", "Combining needs a second photo — none "
                                    "was attached, so this step is skipped")
                    continue
                elif step["task"] == "faceswap":
                    job.log("info", f"[stage] render — step {i + 1}/{n}: "
                                    "replacing the face")
                    current = self._render_faceswap_step(
                        job, current, references[0],
                        with_scene(enh["positive"]), enh["negative"],
                        step.get("denoise"))
                    wf_name = "face swap (face matte + harmonise)"
                    model_name = self._best_compose_checkpoint() or "?"
                    last_mask = None
                    result_adapter = "comfyui-faceswap"
                    result_is_mock = False
                elif step["task"] == "compose":
                    # WHERE it goes: the region the user brushed, else one the
                    # vision model picks from the scene's perspective.
                    place = None
                    drawn = bool(user_mask_b64)
                    if user_mask_b64:
                        job.log("info", "Using the region you drew as the "
                                        "placement area")
                        place = self.decode_mask_b64(user_mask_b64)
                        user_mask_b64 = None
                    elif real and self.critic is not None:
                        place = quality.propose_placement(
                            self.critic, current, step["instruction"],
                            context=scene_module.placement_context(scene_graph))
                        self._save_step_mask(job, asset_id, place,
                                             "placement mask")
                    job.log("info", f"[stage] render — step {i + 1}/{n}: "
                                    "combining the two photos")
                    current = self._render_compose_step(
                        job, current, references[0], place,
                        with_scene(enh["positive"]), enh["negative"],
                        step.get("denoise"), drawn=drawn)
                    wf_name = "compose template (BiRefNet matte)"
                    model_name = self._best_compose_checkpoint() or "?"
                    last_mask = place
                    result_adapter = "comfyui-compose"
                    result_is_mock = False
                elif step["task"] == "pose":
                    job.log("info", f"[stage] render — step {i + 1}/{n}: "
                                    "moving the subject into a new pose")
                    # Reset the vacated-share measurement; the render below
                    # fills it in and the verify stage reads it (D19).
                    self._pose_vacated_share: float | None = None
                    current = self._render_pose_step(
                        job, current, step["instruction"],
                        with_scene(enh["positive"]), enh["negative"],
                        reference=references[0] if references else None,
                        denoise=step.get("denoise"))
                    wf_name = "pose template (masked repaint + ControlNet)"
                    model_name = self._best_pose_checkpoint() or "?"
                    last_mask = None
                    result_adapter = "comfyui-pose"
                    result_is_mock = False
                elif step["task"] == "scene3d":
                    job.log("info", f"[stage] render — step {i + 1}/{n}: "
                                    "rebuilding the photo as a place you can "
                                    "move around in")
                    # NOT `scene`: that name holds the scene SUMMARY string
                    # that with_scene() appends to prompts, and overwriting it
                    # with this dict would interpolate a dict into the prompt
                    # of any step that followed.
                    built = self._render_scene3d_step(job, current)
                    if built is None:
                        raise PermanentError(
                            "A navigable 3D scene could not be built here. "
                            "This needs the geometry model (MoGe) and a "
                            "running ComfyUI.")
                    scene_result = built
                    wf_name = "scene3d template (MoGe metric point map)"
                    model_name = "moge-v2"
                    last_mask = None
                    result_adapter = "comfyui-scene3d"
                    result_is_mock = False
                elif step["task"] == "background":
                    job.log("info", f"[stage] render — step {i + 1}/{n}: "
                                    "rebuilding the environment around the "
                                    "subject")
                    # Environment awareness, in order: MEASURE the photo's
                    # physics (contact points, ground plane, camera pitch,
                    # horizon — MoGe + the exact matte), PLAN the place the
                    # words ask for, then COMPILE both into the prompt so
                    # the new scene is generated around the subject with
                    # the same camera, not painted behind a cutout. Every
                    # stage fails open to the old behaviour, honestly
                    # logged.
                    env_card = (self._scene_card(job, asset_id, current)
                                if real else None)
                    env_spec = step.get("_env_spec")
                    if env_spec is None and real:
                        env_spec = scene_geometry.environment_spec(
                            self.llm, step["instruction"],
                            env_card.posture if env_card else None,
                            env_card.cut_at_bottom if env_card else None,
                            scene or "")
                    if env_spec:
                        job.log("info", "[stage] plan — environment: "
                                        f"{env_spec['environment']}; the "
                                        f"subject {env_spec['relationship']}")
                    # Deliberately NOT with_scene(): the scene summary
                    # describes the backdrop being REPLACED. Seen live —
                    # "scene: person standing in front of mirror" appended to
                    # a forest request made SD paint a framed forest picture
                    # on the original wall instead of putting her in a forest.
                    env_pos, env_neg = scene_geometry.spatial_prompt(
                        env_spec, env_card, enh["positive"], enh["negative"])
                    env_guide = (self._env_guidance(job, asset_id, current,
                                                    env_card)
                                 if env_card is not None else None)
                    before_bg = current
                    current = self._render_background_step(
                        job, current, env_pos, env_neg,
                        subject_hint=f"{scene or ''} {step['instruction']}",
                        compiled=True, guidance=env_guide)
                    # A correctly matted subject still reads as pasted while
                    # its light disagrees with the scene it is now standing
                    # in. This is the step that sells it.
                    current = self._match_lighting(
                        job, current, before_bg, env_pos)
                    if real and env_card is not None:
                        env_misses = self._environment_misses(
                            job, current, env_card, env_spec)
                    wf_name = ("background template (scene-measured "
                               "environment, inverted BiRefNet matte)")
                    model_name = "sd15-inpaint"
                    last_mask = None
                    result_adapter = "comfyui-background"
                    result_is_mock = False
                elif step["task"] == "angles":
                    # MULTI_VIEW: real viewpoint synthesis produces NEW
                    # pictures of the subject, not an edit of this one — so
                    # like ANIMATE it ends the chain and returns its own
                    # assets.
                    views = self._render_viewpoints(
                        job, asset_id, current, step["instruction"], scene,
                        with_scene(enh["positive"]), enh["negative"], real)
                    if views is not None:
                        plan_report.append({
                            "step": i + 1, "task": "angles",
                            "operation": step.get("operation", "MULTI_VIEW"),
                            "target": step.get("target", ""),
                            "instruction": step["instruction"][:120],
                            "workflow": views["workflow"],
                            "model": views["model"]})
                        return {**views["result"], "plan": plan_report}
                    # No viewpoint engine here: fall through to an honest
                    # approximation rather than pretending or failing.
                    job.log("info", f"[stage] render — step {i + 1}/{n}: "
                                    "approximating the new viewpoint with "
                                    "img2img (the multi-view engine is not "
                                    "available on this machine)")
                    current = self._render_template_step(
                        job, "img2img", current,
                        with_scene(enh["positive"]), enh["negative"], 0.72)
                    wf_name = "img2img (viewpoint approximation)"
                    last_mask = None
                    result_adapter = "comfyui-img2img"
                    result_is_mock = False
                    # The step really ran as img2img — record that, or the
                    # quality retry would dispatch on a route that isn't
                    # available and re-run the wrong engine.
                    step["task"] = "img2img"
                elif step["task"] == "relight":
                    ok, why = self._template_runnable("relight")
                    if (not ok and real and self.settings.auto_install
                            and "not downloaded" in why
                            and self._pack_active("ic-light")):
                        job.log("info", "[stage] models — fetching the "
                                        "relighting engine (one-time "
                                        "download)")
                        for name in ("iclight-sd15-fc", "sd15-base"):
                            try:
                                self._ensure_model(name, job)
                            except Exception as exc:  # noqa: BLE001
                                job.log("error", f"Could not fetch {name}: "
                                                 f"{exc}")
                        ok, why = self._template_runnable("relight")
                    if ok and not self._pack_active("ic-light"):
                        ok, why = False, "the ic-light node pack is not active"
                    lit_pos, lit_neg = self._relight_prompts(
                        scene, step["instruction"], enh["negative"])
                    if ok:
                        job.log("info", f"[stage] render — step {i + 1}/{n}: "
                                        "relighting with IC-Light (the light "
                                        "changes, the subject does not)")
                        current = self._render_template_step(
                            job, "relight", current, lit_pos, lit_neg)
                        wf_name = "relight (IC-Light) template"
                        model_name = "iclight_sd15_fc"
                        result_adapter = "comfyui-relight"
                    else:
                        # Honest degradation: img2img cannot move a light
                        # source, so say so instead of silently pretending.
                        job.log("info", f"[stage] render — step {i + 1}/{n}: "
                                        f"the relighting engine is not ready "
                                        f"({why}) — approximating with "
                                        "img2img; download it from the Models "
                                        "page for a true relight")
                        current = self._render_template_step(
                            job, "img2img", current, with_scene(lit_pos),
                            lit_neg, 0.55)
                        wf_name = "img2img (lighting approximation)"
                        result_adapter = "comfyui-img2img"
                        step["task"] = "img2img"  # what actually ran
                    if i == n - 1:
                        last_positive, last_negative = lit_pos, lit_neg
                    last_mask = None
                    result_is_mock = False
                elif step["task"] == "kontext":
                    # Whole-image instruction editing, no mask. The user's own
                    # sentence is what Kontext was trained to follow, so it is
                    # sent as written rather than as the enhanced prompt: the
                    # keyword padding that helps a SD-era sampler ("8k, highly
                    # detailed") is noise to an instruction model.
                    job.log("info", f"[stage] render — step {i + 1}/{n}: "
                                    "editing the whole picture from your "
                                    "instruction, without a mask")
                    current = self._render_kontext_step(
                        job, current, step["instruction"])
                    wf_name = "kontext_v1 template"
                    model_name = "flux1-kontext-dev-Q4_K_S"
                    last_mask = None
                    result_adapter = "comfyui-kontext"
                    result_is_mock = False
                elif step["task"] == "custom":
                    job.log("info", f"[stage] render — step {i + 1}/{n}: "
                                    "no standard workflow fits — building a "
                                    "custom one for this edit")
                    current = self._render_custom_step(
                        job, step["instruction"], current,
                        with_scene(enh["positive"]), enh["negative"])
                    wf_name = "LLM-designed custom workflow"
                    last_mask = None
                    result_adapter = "comfyui-custom"
                    result_is_mock = False
                elif step["task"] == "outpaint":
                    # Outpaint = inpainting the padded margins. Three realism
                    # rules: (1) the prompt describes the CONTINUATION of the
                    # scene, never the subject — or the model paints a second
                    # copy of it in the new space; (2) the best installed
                    # inpaint model does the blending; (3) the added margins
                    # become the seam-inspection mask, so a visible boundary
                    # is caught and retried like any inpaint seam.
                    job.log("info", f"[stage] render — step {i + 1}/{n}: "
                                    "extending the canvas (soft outpaint)")
                    out_pos, out_neg = self._outpaint_prompts(
                        scene, step["instruction"], enh["negative"])
                    out_ckpt = self._best_outpaint_checkpoint() if real else None
                    if out_ckpt:
                        job.log("info", f"Outpaint model: {out_ckpt} "
                                        "(soft-inpaint blending)")
                        model_name = out_ckpt
                    out_dirs = quality.outpaint_directions(
                        step["instruction"])
                    if out_dirs:
                        named = [s for s, px in out_dirs.items() if px]
                        job.log("info", "Extending the canvas on the "
                                        f"requested side{'s' if len(named) > 1 else ''}: "
                                        + ", ".join(named))
                    pre_size = current.size
                    current = self._guarded_outpaint(
                        job, current, out_pos, out_neg, out_ckpt, real,
                        dirs=out_dirs)
                    last_mask = self._pad_mask(pre_size, current.size,
                                               out_dirs)
                    if i == n - 1:
                        last_positive = out_pos
                        last_outpaint = {"positive": out_pos,
                                         "negative": out_neg,
                                         "checkpoint": out_ckpt,
                                         "dirs": out_dirs,
                                         "pre_size": pre_size}
                    result_adapter = "comfyui-outpaint"
                    result_is_mock = False
                else:
                    step_denoise = step.get("denoise")
                    if (step["task"] == "img2img"
                            and step.get("operation") == "CHANGE_STYLE"
                            and (step_denoise or 0.6) > 0.55):
                        # img2img repaints the WHOLE frame; at edit-strength
                        # denoise a style request the model cannot honour
                        # still costs the subject's likeness — one run
                        # returned a clean photograph of somebody else
                        # (D16/D23). Style stays below the identity line.
                        step_denoise = 0.55
                        job.log("info", "Style denoise capped at 0.55 so "
                                        "the person survives the restyle")
                    job.log("info", f"[stage] render — step {i + 1}/{n}: "
                                    f"running the {step['task']} workflow")
                    current = self._render_template_step(
                        job, step["task"], current,
                        with_scene(enh["positive"]),
                        enh["negative"], step_denoise)
                    last_mask = None
                    result_adapter = f"comfyui-{step['task']}"
                    result_is_mock = False
                plan_report.append({
                    "step": i + 1, "task": step["task"],
                    "operation": step.get("operation", ""),
                    "target": step.get("target", ""),
                    "instruction": step["instruction"][:120],
                    "workflow": wf_name, "model": model_name})
                job.log("info", f"Step {i + 1}/{n} done — {wf_name}, "
                                f"model {model_name}")
                if real and i < n - 1:
                    # Live progress on THE image: store an intermediate
                    # preview the UI fades in as the step completes.
                    self._save_step_preview(job, asset_id, current, i + 1, n)

            if real and self.critic is not None:
                issues: list[str] = []
                if last_mask is not None:
                    job.log("info", "[stage] inspect — examining the seam "
                                    "and the edited region")
                    issues = quality.inspect_seams(self.critic, current,
                                                   last_mask)
                    if issues:
                        job.log("info", "Issues found: "
                                        + "; ".join(issues[:5]))
                if env_misses:
                    # Measured geometry disagreements are issues too: they
                    # feed the same avoid-clause the inspector's findings do.
                    issues = [*issues, *env_misses]
                job.log("info", "[stage] score — grading realism, accuracy "
                                "and consistency")
                scores = quality.scorecard(self.critic, current, prompt)
                scores = self._ground_scores(job, scores, current,
                                             last_outpaint)
                scores = self._env_scores(job, scores, env_misses)
                self._log_scores(job, scores)
                # Did the edit DO what was asked? The checklist names the
                # parts that are missing, and those names decide what the
                # next attempt changes.
                #
                # Scoped to the LAST step, because only the last step is ever
                # re-rendered. For "remove the car and make it night" the
                # retry re-runs the night restyle; feeding it "no car in the
                # driveway" would push a car INTO the positive prompt of the
                # step whose job was the sky.
                # Prefer the checklist built at PLAN time, while the text
                # model was warm (the warm-text batch): building it here,
                # between two vision calls, cost a full text-model reload
                # on 8 GB. Presence-checked, not truth-checked — an empty
                # batched checklist is an answer, not a miss.
                if "_checklist" in last_step:
                    checklist = last_step["_checklist"]
                else:
                    checklist = quality.request_checklist(
                        self.llm, last_step.get("instruction") or prompt)
                if checklist:
                    job.log("info", "[llm] the edit must deliver: "
                                    + " · ".join(c["need"] for c in checklist))
                adh = self._adherence(job, current, prompt, checklist, scores)
                target = self.settings.quality_target
                best, best_scores = current, scores
                best_missing = list((adh or {}).get("missing") or [])
                # Objective adherence from numbers the app already has: for a
                # format request the aspect ratio IS the requirement, and two
                # numbers settle it without a model (D7 — the live outpaint
                # went 0.887 → 1.115 and the verifier still reported "still
                # missing: a wide landscape format", costing two re-renders).
                fmt = quality.format_delivered(prompt, image.size,
                                               current.size)
                if fmt is True:
                    settled = [m for m in best_missing
                               if quality.about_format(m)]
                    if settled:
                        best_missing = [m for m in best_missing
                                        if not quality.about_format(m)]
                        job.log("info", "[stage] verify — the measured "
                                        "aspect ratio settles the format "
                                        "requirement: delivered (overruling "
                                        "the verifier)")
                elif fmt is False and not any(quality.about_format(m)
                                              for m in best_missing):
                    best_missing.append("the requested format/aspect change")
                    job.log("info", "[stage] verify — the measured aspect "
                                    "ratio says the format request was NOT "
                                    "delivered")
                # Colour requirements are settled the same way: the mask and
                # the pixels are both in hand, and a hue count is exact where
                # the vision judge scored the same image 20 and 70 on two
                # runs. A decisive measurement overrules the model in BOTH
                # directions — a false "missing" no longer buys a retry, and
                # a false "met" no longer passes a miss.
                def settle_colour(img, missing):
                    if last_step["task"] != "inpaint" or last_mask is None:
                        return missing
                    out = list(missing)
                    for need in list(out):
                        col = quality.requirement_colour(need)
                        if not col:
                            continue
                        if quality.colour_delivered(img, last_mask, col,
                                                    final_input) is True:
                            out.remove(need)
                            job.log("info", "[stage] verify — the measured "
                                            f"hue settles it: the region IS "
                                            f"{col} (overruling the "
                                            "verifier)")
                    want = quality.requirement_colour(
                        last_step.get("instruction") or "")
                    if (want
                            and not any(quality.requirement_colour(m) == want
                                        for m in out)
                            and quality.colour_delivered(
                                img, last_mask, want,
                                final_input) is False):
                        out.append(f"the region actually turning {want}")
                        job.log("info", "[stage] verify — the measured hue "
                                        f"says the region did NOT turn "
                                        f"{want}")
                    return out

                best_missing = settle_colour(current, best_missing)
                # An untouched region is the strongest verdict there is: the
                # recipe ignored the prompt. Re-rolling the SAME recipe with
                # an emphasized prompt re-rolls the same failure — go
                # straight to a different model or technique.
                # A separate FLAG, deliberately never an entry in
                # best_missing and never a retry trigger of its own: a
                # synthetic missing-entry let a worse retry displace a good
                # first attempt, and a forced extra round second-guessed a
                # checklist that was already satisfied (both caught by the
                # integration suite). Its one job: when adherence has
                # already demanded a retry, skip the same-recipe gamble.
                force_swap = False
                if last_step["task"] == "inpaint" and last_mask is not None:
                    moved = quality.region_change(final_input, current,
                                                  last_mask)
                    if moved is not None and moved < 0.02:
                        force_swap = True
                        job.log("info", "[stage] verify — the selected "
                                        f"region changed by only "
                                        f"{moved * 100:.1f}% — the edit did "
                                        "not happen; skipping the "
                                        "same-recipe retry")
                # The pose route already measures the share of frame the
                # repose vacated; 4% means the body did not move. That number
                # was computed, logged and unused (D19) — now it is evidence.
                vacated = (getattr(self, "_pose_vacated_share", None)
                           if last_step["task"] == "pose" else None)
                if (vacated is not None and vacated < 0.05
                        and not any("pose" in m.lower()
                                    for m in best_missing)):
                    best_missing.append("the subject's pose actually "
                                        "changing")
                    job.log("info", "[stage] verify — the repose vacated "
                                    f"only {vacated * 100:.1f}% of the "
                                    "frame, so the body did not move")
                # The mask is refined from the ORIGINAL each round: refining
                # the already-refined one dilates it twice and the edit region
                # creeps outward with every retry.
                base_mask = last_mask
                tried_ckpts = {(last_inpaint or {}).get("checkpoint")}
                tried_variants = {(last_inpaint or {}).get("variant")}
                # An upscale step takes only an image — no prompt, no seed, no
                # model choice — so its graph is byte-identical every time and
                # a "retry" would burn minutes rendering the same pixels.
                retryable = last_step["task"] != "upscale"
                # Swapping the model is only meaningful where the engine can
                # actually take one. An inpaint adapter without variant
                # support (and the mock) has exactly one recipe; and a
                # template that pins its model for a REASON — relight needs a
                # 4-channel SD1.5 base, nothing else physically runs — must
                # keep it.
                can_swap_model = (
                    last_step["task"] in ("inpaint", "img2img", "outpaint")
                    and (last_step["task"] != "inpaint"
                         or getattr(self.inpainting, "supports_variants",
                                    False)))
                if not retryable:
                    job.log("info", "[stage] verify — an upscale has nothing "
                                    "to vary between attempts; keeping it")
                # Stop condition. A checklist verdict is the authority: when
                # the edit delivers every requirement, further rounds chase a
                # quality_target of 95 that llava essentially never awards, so
                # they would spend minutes and keep the first result anyway.
                def keep_going() -> bool:
                    if job.cancel_requested:
                        return False
                    if best_missing:
                        return True
                    if adh and adh.get("source") == "checklist":
                        return not quality.meets_target(best_scores, target) \
                            and (quality.overall(best_scores) or 0) < 85
                    return not quality.meets_target(best_scores, target)

                while (retryable and rounds < self.settings.quality_rounds
                       and keep_going()):
                    rounds += 1
                    # ESCALATION. Round 1 re-renders the same recipe with the
                    # missed part of the request emphasized. From round 2 the
                    # RECIPE changes — a different model, or a different
                    # technique — because rolling the same dice twice is not
                    # a strategy.
                    swap_ckpt = swap_variant = None
                    if (rounds > 1 or force_swap) and can_swap_model:
                        swap_ckpt, swap_variant = self._next_edit_recipe(
                            last_step["task"], tried_ckpts, tried_variants,
                            job=job)
                    if swap_ckpt or swap_variant:
                        job.log("info", f"[stage] retry — the edit still "
                                        f"misses the request (round {rounds}/"
                                        f"{self.settings.quality_rounds}): "
                                        + (f"trying model {swap_ckpt}"
                                           if swap_ckpt else "")
                                        + (f" via the {swap_variant} technique"
                                           if swap_variant else ""))
                    else:
                        job.log("info", f"[stage] retry — regenerating the "
                                        f"final step (round {rounds}/"
                                        f"{self.settings.quality_rounds})")
                    emphasized = last_positive
                    is_removal_step = (last_step.get("operation")
                                       == "REMOVE_OBJECT")
                    if is_removal_step:
                        # The removal conditioning stays exactly as composed:
                        # pushing the missed requirement ("remove the hat")
                        # back into the positive prompt is the defect the
                        # conditioning exists to fix (D1).
                        pass
                    elif best_missing:
                        emphasized = quality.emphasize(last_positive,
                                                       best_missing, 1.25)
                        job.log("info", "Retry emphasizes what was missed: "
                                        + "; ".join(best_missing[:3]))
                    else:
                        weak_key, weak_val = (quality.weakest(best_scores)
                                              if best_scores else (None, None))
                        if (weak_key == "prompt_accuracy"
                                and (weak_val or 100) < 70):
                            emphasized = f"({last_positive}:1.25)"
                            job.log("info", "Retry emphasizes the request "
                                            f"(accuracy was {weak_val}/100)")
                    try:
                        if (last_step["task"] == "inpaint"
                                and base_mask is not None):
                            last_mask = self._refine_mask(base_mask)
                            self._save_step_mask(job, asset_id, last_mask,
                                                 "refined mask")
                            retry_prompt = (
                                f"{emphasized}, seamless blend, "
                                "natural lighting"
                                + (f"; avoid: {'; '.join(issues[:3])}"
                                   if issues else ""))
                            # A removal retry keeps its conditioning free of
                            # the scene summary too — it can name the object.
                            retry_pos = (retry_prompt if is_removal_step
                                         else with_scene(retry_prompt))
                            recipe = dict(last_inpaint or {})
                            if swap_ckpt:
                                recipe["checkpoint"] = swap_ckpt
                            if swap_variant:
                                recipe["variant"] = swap_variant
                            tried_ckpts.add(recipe.get("checkpoint"))
                            tried_variants.add(recipe.get("variant"))
                            if (recipe.get("denoise") is not None
                                    and swap_variant
                                    and swap_variant != "universal"):
                                # The denoise dial belongs to the universal
                                # technique; a different-technique rung runs
                                # that technique's own defaults.
                                recipe.pop("denoise")
                            elif (recipe.get("denoise") == 0.6
                                    and base_mask is not None
                                    and quality.mask_fraction(base_mask)
                                    > quality.MASK_CEILING):
                                # Whole-frame restyle rung, measured live
                                # (RTX 4060 A/B): 0.6 keeps the composition
                                # but can undershoot the asked-for look; 0.8
                                # spends real change while staying short of
                                # the full regeneration that loses the
                                # photo. Rolling the same 0.6 twice is not a
                                # strategy — the ladder's own rule.
                                recipe["denoise"] = 0.8
                                job.log("info",
                                        "Retry raises denoise 0.6 → 0.8: "
                                        "the look still misses the request, "
                                        "so the composition guard loosens "
                                        "one notch (never all the way)")
                            if can_swap_model:
                                # Kwargs gated on the adapter's capability.
                                res2 = self.inpainting.inpaint(  # type: ignore[call-arg]
                                    final_input, last_mask, retry_pos,
                                    negative=last_negative, **recipe)
                            else:
                                res2 = self.inpainting.inpaint(
                                    final_input, last_mask, retry_pos)
                            candidate = res2.image
                        elif last_step["task"] == "faceswap" and references:
                            # Like compose: the generic template path would
                            # ask the loader for a "faceswap" template that
                            # does not exist ("not an allowed workflow type"),
                            # so the retry costs a render and buys nothing.
                            candidate = self._render_faceswap_step(
                                job, final_input, references[0],
                                with_scene(emphasized), last_negative,
                                last_step.get("denoise"))
                        elif last_step["task"] == "compose" and references:
                            # A compose retry must re-run the COMPOSE, with
                            # the same reference photo and the same placement.
                            # Falling through to the generic template path
                            # sends an empty subject filename and ComfyUI
                            # fails on a directory that does not exist (seen
                            # live) — the retry silently buys nothing.
                            candidate = self._render_compose_step(
                                job, final_input, references[0], base_mask,
                                with_scene(emphasized), last_negative,
                                last_step.get("denoise"), drawn=False)
                        elif last_step["task"] == "background":
                            # MUST NOT fall through to the generic template
                            # path: that uses run_graph, which returns the
                            # FIRST output image, and this graph emits the
                            # background plate first (it is ready earlier).
                            # A retry could therefore "win" with the scene
                            # minus the subject.
                            # Recompile the spatial prompt around the
                            # emphasized retry wording: the first retry
                            # implementation fell back to the plain
                            # scene-appended prompt, which dropped the
                            # ground contract and the solid-ground
                            # negatives — measured live, every retry
                            # flooded the foreground again.
                            retry_pos, retry_neg = \
                                scene_geometry.spatial_prompt(
                                    env_spec, env_card, emphasized,
                                    last_negative)
                            candidate = self._render_background_step(
                                job, final_input, retry_pos, retry_neg,
                                compiled=True, guidance=env_guide)
                            if env_card is not None:
                                env_misses = self._environment_misses(
                                    job, candidate, env_card, env_spec)
                        elif last_step["task"] == "kontext":
                            # Kontext follows an INSTRUCTION, and the retry
                            # path's scene-appended prompt is a DESCRIPTION.
                            # Handed "remove the hat; scene: a woman in a
                            # shop with shelves...", it redrew the scene:
                            # the retry came back with a different room and
                            # a different face. It also has to render at its
                            # own ~1 MP size, as the first attempt did, or it
                            # is off-distribution as well as slow.
                            candidate = self._render_kontext_step(
                                job, final_input, last_step["instruction"])
                        elif last_step["task"] == "custom":
                            candidate = self._render_custom_step(
                                job, last_step["instruction"], final_input,
                                with_scene(emphasized), last_negative)
                        elif last_step["task"] == "outpaint" and last_outpaint:
                            # Same continuation prompts + model + directions,
                            # new seed — the subject-emphasized prompt would
                            # invite the extra people the outpaint prompt is
                            # built to avoid.
                            candidate = self._render_template_step(
                                job, "outpaint", final_input,
                                last_outpaint["positive"],
                                last_outpaint["negative"],
                                checkpoint=swap_ckpt
                                or last_outpaint["checkpoint"],
                                extra=last_outpaint.get("dirs"))
                            candidate = self._harmonize_outpaint(
                                job, candidate, final_input,
                                last_outpaint.get("dirs"))
                            if swap_ckpt:
                                tried_ckpts.add(swap_ckpt)
                        else:
                            candidate = self._render_template_step(
                                job, last_step["task"], final_input,
                                with_scene(emphasized), last_negative,
                                last_step.get("denoise"), checkpoint=swap_ckpt)
                            if swap_ckpt:
                                tried_ckpts.add(swap_ckpt)
                    except (PermanentError, AdapterError) as exc:
                        # An escalation rung deliberately tries something new
                        # — an untried checkpoint, another technique — so it
                        # is exactly the thing that can fail. `best` already
                        # holds a finished edit; losing it here would turn a
                        # successful job into a failed one.
                        job.log("info", f"Round {rounds} could not run "
                                        f"({exc}); keeping the best result "
                                        "so far")
                        break
                    if real:
                        # Every attempt is paid for; keep it inspectable.
                        # Only the final image used to survive, which is also
                        # why the verifier's behaviour was so hard to pin
                        # down from outside (D12/Step 8).
                        self._save_attempt(job, asset_id, candidate, rounds)
                    cand_issues = (quality.inspect_seams(self.critic, candidate,
                                                         last_mask)
                                   if last_mask is not None else [])
                    if env_misses:
                        # A kept candidate's measured geometry misses must
                        # reach the NEXT round's avoid-clause too.
                        cand_issues = [*cand_issues, *env_misses]
                    scores2 = quality.scorecard(self.critic, candidate,
                                                prompt)
                    scores2 = self._ground_scores(job, scores2, candidate,
                                                  last_outpaint)
                    scores2 = self._env_scores(job, scores2, env_misses)
                    o2 = quality.overall(scores2)
                    ob = quality.overall(best_scores)
                    if o2 is None:
                        job.log("info", "Judge unavailable — keeping the "
                                        "best attempt so far")
                        break
                    self._log_scores(job, scores2)
                    adh2 = self._adherence(job, candidate, prompt, checklist, scores2)
                    missing2 = settle_colour(
                        candidate, list((adh2 or {}).get("missing") or []))
                    # A retry that also came back untouched keeps the recipe
                    # pressure on: the NEXT round must swap again, not paper
                    # over an unchanged region with an emphasized prompt.
                    force_swap = bool(
                        last_step["task"] == "inpaint"
                        and last_mask is not None
                        and (quality.region_change(final_input, candidate,
                                                   last_mask) or 1.0) < 0.02)
                    # overall(scores2) was non-None, so scores2 is a dict.
                    pa2 = scores2.get("prompt_accuracy", 0)  # type: ignore[union-attr]
                    pab = (best_scores or {}).get("prompt_accuracy", 0)
                    # Prompt fidelity gates first: fewer unmet requirements
                    # wins outright, and more unmet requirements loses
                    # outright, whatever the averages say.
                    if adh and adh2 and len(missing2) != len(best_missing):
                        keep = len(missing2) < len(best_missing)
                        why = (f"it delivers {len(best_missing) - len(missing2)}"
                               " more of the request" if keep else
                               "it drops part of the request the earlier "
                               "attempt delivered")
                    else:
                        keep = quality.better_candidate(scores2, best_scores)
                        why = (f"overall {o2}, accuracy {pa2}" if keep else
                               (f"it drifts from the request (accuracy {pa2} "
                                f"< {pab}); a retry is never allowed to undo "
                                "the edit" if pa2 < pab - 5
                                else f"overall {o2} < {ob}"))
                    verdict_static = (bool(best_missing) and bool(missing2)
                                      and set(missing2) == set(best_missing)
                                      and (adh2 or {}).get("source")
                                      == "checklist")
                    if keep:
                        best, best_scores = candidate, scores2
                        best_missing, issues = missing2, cand_issues
                        job.log("info", f"Round {rounds} kept — {why}")
                    else:
                        # `issues` deliberately NOT updated here: it feeds the
                        # next round's "avoid:" clause, and describing an
                        # image that was thrown away misdirects it.
                        job.log("info", f"Round {rounds} discarded — {why}")
                    if verdict_static:
                        # A full re-render produced a DIFFERENT image and the
                        # verifier returned the identical verdict. A verdict
                        # that never moves is not being computed from the
                        # image, and it must not be allowed to spend another
                        # render on itself — retrying was ~26% of all
                        # pipeline time, as much as rendering (D7).
                        job.log("info", "[stage] verify — the verdict did "
                                        "not change across a full re-render; "
                                        "it is not measuring this image "
                                        "reliably, so no further retries "
                                        "will be spent on it")
                        break
                current, scores = best, best_scores
                # Objective checks — arithmetic on pixels, recorded beside
                # the model's scores and allowed to veto "production-ready".
                # These caught live what the model scored 90: relight grain
                # at 8.7x input sharpness (D20), and they flag mask leaks and
                # silent crops the same way (Step 5b).
                objective = quality.objective_report(
                    final_input, current,
                    last_mask if last_step["task"] == "inpaint" else None)
                obj_flags = quality.objective_flags(objective,
                                                    last_step["task"])
                if vacated is not None and vacated < 0.05:
                    obj_flags.append("the repose vacated only "
                                     f"{vacated * 100:.1f}% of the frame — "
                                     "the body did not move")
                if last_step["task"] in ("img2img", "relight", "custom"):
                    # The identity gate whole-frame repaints never had: how
                    # much of the FACE moved. A style pass that cannot
                    # deliver its style still redraws the person, and
                    # nothing used to notice (D16/D23).
                    drift = self._face_drift(job, image, current)
                    if drift is not None:
                        objective["face_drift"] = round(drift, 4)
                        if drift > 0.5:
                            obj_flags.append(
                                "the subject's face was substantially "
                                f"redrawn ({drift * 100:.0f}% of face "
                                "pixels moved) — identity is not preserved")
                for flag in obj_flags:
                    job.log("info", f"[stage] verify — objective check: "
                                    f"{flag}")
                if best_missing:
                    job.log("info", f"[stage] verify — best of {rounds + 1} "
                                    f"attempt(s) kept; still missing: "
                                    + "; ".join(best_missing[:3]))
                    # A drawn mask is an instruction, so the pipeline never
                    # enlarges it — but a REPLACEMENT that needs more room
                    # than the old object (a t-shirt over a bikini top) can
                    # only appear inside the drawn region. Seen live: the
                    # render reshaped the top inside the ellipse and the
                    # checklist honestly reported the t-shirt missing. Say
                    # WHY, so the miss is actionable instead of mysterious.
                    if (drew_mask
                            and last_step["task"] == "inpaint"
                            and last_step.get("operation")
                            == "REPLACE_OBJECT"):
                        job.log("info", "[stage] verify — note: the drawn "
                                        "region covers the OLD object, and "
                                        "a replacement that needs more room "
                                        "can only appear inside it. Draw a "
                                        "roomier region, or clear the mask "
                                        "to let the app choose the region "
                                        "and engine.")
                elif quality.meets_target(scores, target):
                    job.log("info", f"[stage] verify — the edit does what was "
                                    f"asked and every category ≥ "
                                    f"{target}/100: production-ready")
                elif scores:
                    k, v = quality.weakest(scores)
                    job.log("info", f"[stage] verify — the edit does what was "
                                    f"asked; best of {rounds + 1} attempt(s) "
                                    f"kept; weakest category: {k} {v}/100 "
                                    f"(target {target})")
        except BadMaskError as exc:
            raise PermanentError(str(exc)) from exc
        except ModelMissingError as exc:
            raise PermanentError(str(exc)) from exc
        except BackendUnavailableError as exc:
            raise TransientError(str(exc)) from exc
        except MemoryError as exc:
            job.log("error", "Out of memory during inpainting")
            raise TransientError("Out of memory — will retry.") from exc

        job.log("info", "[stage] save — storing the edit")
        # The VAE rounds render dimensions down to multiples of 8, so every
        # route quietly returned a slightly smaller image and ten successive
        # edits would have cropped 60px (D24 — fit_mask has documented this
        # for as long as it has existed; this is the missing last step).
        if (current.size != image.size
                and abs(current.width - image.width) <= 8
                and abs(current.height - image.height) <= 8):
            current = current.resize(image.size, Image.Resampling.LANCZOS)
            job.log("info", f"Output restored to the input's exact size "
                            f"({image.width}x{image.height}) — the renderer "
                            "works in multiples of 8 and had trimmed the "
                            "difference")
        out_path = self.store.new_version_path(asset_id)
        current.save(out_path, format="PNG")
        version = self.store.add_edit_version(
            asset_id, str(out_path), prompt, result_adapter,
            meta={"is_mock": result_is_mock, "plan": plan_report})
        job.log("info", f"Saved edit version {version.id}")
        out: dict[str, Any] = {"version_id": version.id, "asset_id": asset_id,
                               "adapter": result_adapter,
                               "is_mock": result_is_mock,
                               "plan": plan_report}
        if scene_result:
            # A GLB is not a version of the photo, it is a separate thing the
            # UI opens in the 3D viewer — so it travels as its own asset id.
            out.update(scene_result)
        if len(steps) == 1 and steps[0]["task"] != "inpaint":
            out["route"] = steps[0]["task"]
        if objective is not None:
            # Recorded beside the scores so the arithmetic and the model can
            # be compared after the fact (Step 5b).
            out["objective"] = objective
            if obj_flags:
                out["objective_flags"] = obj_flags
        if scores:
            out.update({"scores": scores, "overall": quality.overall(scores),
                        "rounds": rounds,
                        # An out-of-range objective measurement vetoes
                        # production-ready, whatever the model scored.
                        "passed": quality.meets_target(
                            scores, self.settings.quality_target)
                        and not obj_flags})
        return out

    def _correct_mask(self, job: Job, asset_id: str, current: Image.Image,
                      mask: Image.Image, step: dict[str, Any],
                      why: str, source: str = "sam") -> Image.Image:
        """One corrective pass when the vision check objects to a mask.

        It used to re-cut with SAM every time — including when the rejected
        mask had come from a BETTER engine — so the ladder ran downhill and
        the usual outcome was "second mask attempt no better, keeping the
        first". Now the objection is folded into the request and put back
        through the chooser, which is ordered by strength of evidence, and the
        replacement is only taken if it comes from an engine at least as good
        as the one that produced the original."""
        # whole-frame outranks text: it is the text engine's answer PLUS the
        # evidence that the answer covers everything — a re-cut that comes
        # back whole-frame should replace a doubted text region, not be
        # logged as "weaker".
        rank = {"whole-frame": 3, "named-part": 3, "text": 2, "sam": 1,
                "none": 0}
        hint = step.get("target") or step["instruction"]
        request = f"{hint}. Not: {why}" if why else str(hint)
        try:
            choice = self.auto_mask(current, request, job=job)
        except Exception:  # noqa: BLE001 — correction is best-effort
            return mask
        if not choice.ok:
            job.log("info", "The second attempt found nothing better — "
                            "keeping the first selection")
            return mask
        if rank.get(choice.source, 0) < rank.get(source, 0):
            job.log("info", f"The second attempt fell back to "
                            f"{choice.source}, which is weaker than the "
                            f"{source} selection it would replace — keeping "
                            "the first")
            return mask
        job.log("info", f"Mask corrected via {choice.source}")
        corrected = cast(Image.Image, choice.mask)  # ok-checked above
        self._save_step_mask(job, asset_id, corrected, "corrected mask")
        return corrected

    def _face_drift(self, job: Job, before: Image.Image,
                    after: Image.Image) -> float | None:
        """Share of face-region pixels that moved between input and output.

        Deterministic identity signal for whole-frame repaints: the head box
        comes from the subject matte, the drift is plain pixel arithmetic.
        None when the head cannot be located — unknown, not a pass."""
        try:
            box = self._head_crop(before)
        except Exception:  # noqa: BLE001 — the gate is best-effort
            return None
        if box is None or (box[2] - box[0]) < 16 or (box[3] - box[1]) < 16:
            return None
        b = before.convert("L").crop(box)
        aligned = after
        if aligned.size != before.size:
            aligned = aligned.resize(before.size, Image.Resampling.LANCZOS)
        a = aligned.convert("L").crop(box)
        diff = ImageChops.difference(a, b)
        changed = diff.point(lambda v: 255 if v > 12 else 0)
        return quality.mask_fraction(changed)

    def _save_attempt(self, job: Job, asset_id: str, image: Image.Image,
                      attempt: int) -> None:
        """Store a retry render as an inspectable aux version. Three renders
        were paid for and two thrown away invisibly, which is also what made
        the verifier's behaviour impossible to audit from outside (Step 8)."""
        try:
            path = self.store.new_version_path(asset_id, suffix=".png")
            image.save(path, format="PNG")
            av = self.store.add_aux_version(
                asset_id, str(path), f"retry attempt {attempt}", "attempt",
                meta={"attempt": attempt})
            job.log("info", f"[stage] retry — attempt {attempt} stored for "
                            f"inspection (version {av.id})")
        except OSError:
            pass  # audit trail is a bonus; the retry still counts

    def _save_step_mask(self, job: Job, asset_id: str, mask: Image.Image,
                        note: str) -> None:
        """Store a mask the pipeline created/changed and tell the UI, so the
        canvas always shows the mask that actually renders."""
        try:
            mpath = self.store.new_version_path(asset_id, suffix=".png")
            mask.save(mpath, format="PNG")
            mv = self.store.add_aux_version(asset_id, str(mpath), note,
                                            "mask-refined",
                                            meta={"mask": True})
            job.log("info", f"[mask] {note} applied — version {mv.id}")
        except OSError:
            pass  # preview is cosmetic; the render still runs

    def _save_step_preview(self, job: Job, asset_id: str,
                           image: Image.Image, step: int, total: int) -> None:
        """Store an intermediate result so the UI can fade it onto THE image
        as the pipeline progresses (dynamic-automation view)."""
        try:
            preview = image
            if max(preview.size) > 768:
                scale = 768 / max(preview.size)
                preview = preview.resize(
                    (max(1, round(preview.width * scale)),
                     max(1, round(preview.height * scale))),
                    Image.Resampling.LANCZOS)
            ppath = self.store.new_version_path(asset_id, suffix=".png")
            preview.save(ppath, format="PNG")
            pv = self.store.add_aux_version(
                asset_id, str(ppath), f"preview after step {step}/{total}",
                "preview", meta={"preview": True})
            job.log("info", f"[preview] after step {step}/{total} — "
                            f"version {pv.id}")
        except OSError:
            pass

    def _render_custom_step(self, job: Job, instruction: str,
                            image: Image.Image, positive: str,
                            negative: str = "") -> Image.Image:
        """An edit step no template expresses: the LLM designs a bespoke
        graph against the live inventory (schema-checked before submission),
        hardware-clamped, with one repair round on a runtime failure."""
        self._require_comfy(job)
        image_name = self.comfy.upload_image(image, "custom_src")
        context = ((self.workflow_context() or "")
                   + f"\nInput image file (LoadImage): {image_name}"
                   + f"\nMachine: {self.hardware.gpu_name or 'CPU only'}\n"
                   + (self.workflows.knowledge("img2img") or ""))
        neg = negative or "blurry, low quality, deformed, artifacts"
        request = (f"Transform the uploaded image '{image_name}' as follows: "
                   f"{instruction}\nPositive prompt: {positive}\n"
                   f"Negative prompt: {neg}\n"
                   "Start from LoadImage and end in SaveImage.")
        try:
            gen = self.workflow_ai.generate("img2img", request,
                                            context=context,
                                            log=self._planner_log(job))
        except LLMUnavailableError as exc:
            raise TransientError(str(exc)) from exc
        except (LLMRefusedError, WorkflowGenerationError) as exc:
            raise PermanentError(
                f"Could not build a custom workflow: {exc}") from exc
        graph = self._apply_hardware_limits(gen.graph, job)
        self._free_vram(job)
        try:
            out, _pid = self.comfy.run_graph(graph)
            return out
        except BackendUnavailableError as exc:
            raise self._comfy_died_midrender(job, "custom", positive,
                                             exc) from exc
        except WorkflowRuntimeError as exc:
            hint = commit_exhausted_hint(str(exc))
            if hint:
                # Not a graph problem — LLM repairs can't add RAM.
                raise PermanentError(f"Custom workflow failed: {hint}") from exc
            job.log("error", f"Custom workflow failed: {exc}")
            job.log("info", "Asking the LLM to repair the custom workflow")
            self._diagnose_and_record(job, "img2img", positive, str(exc))
            try:
                gen = self.workflow_ai.repair("img2img", graph, str(exc),
                                              context=context,
                                              log=self._planner_log(job))
            except LLMUnavailableError as inner:
                raise TransientError(str(inner)) from inner
            except (LLMRefusedError, WorkflowGenerationError) as inner:
                raise PermanentError(
                    f"Custom workflow could not be repaired: {inner}"
                ) from inner
            out, _pid = self.comfy.run_graph(
                self._apply_hardware_limits(gen.graph, job))
            return out

    @staticmethod
    def _outpaint_prompts(scene: str | None, instruction: str,
                          base_negative: str) -> tuple[str, str]:
        """Render prompts for a canvas extension. The model paints ONLY the
        new margins, so the positive leads with scene CONTINUATION — a prompt
        that names the subject invites a second copy of it in the new space
        (seen live: outpaints adding extra people). The user's words are
        still appended verbatim (append-only principle)."""
        positive = ("seamless continuation of the existing scene beyond the "
                    "current frame, the same background and environment "
                    "extended naturally, consistent lighting, colors and "
                    "perspective, nothing new added, photorealistic detail")
        if scene:
            positive += f"; scene: {scene}"
        if instruction:
            positive += f"; request: {instruction}"
        negative = ("extra person, additional people, second person, "
                    "duplicated person, cloned subject, twins, new objects, "
                    "split image, collage, mirrored image, frame, border, "
                    "seam, visible edge")
        if base_negative:
            negative += f", {base_negative}"
        return positive, negative

    @staticmethod
    def _relight_prompts(scene: str | None, instruction: str,
                         base_negative: str) -> tuple[str, str]:
        """Render prompts for IC-Light. The model re-synthesises the picture
        from the foreground latent plus this prompt, so the prompt must
        describe the LIGHT — naming the subject makes IC-Light redraw that
        subject instead of relighting the photo it was given. The user's own
        words are still appended verbatim (append-only principle)."""
        positive = (f"{instruction}, cinematic photographic lighting, "
                    "natural light falloff, consistent shadows and "
                    "reflections, the same subject and composition unchanged")
        if scene:
            positive += f"; scene: {scene}"
        negative = ("flat lighting, washed out, overexposed, underexposed, "
                    "different person, changed subject, changed pose, "
                    "changed background, lowres, blurry, artifacts")
        if base_negative:
            negative += f", {base_negative}"
        return positive, negative

    def _pack_active(self, slug: str) -> bool:
        """Is a curated node pack live in ComfyUI right now? (Probed, cached
        — never assumed from disk.)

        Self-healing: when the pack a request needs is missing ENTIRELY,
        auto-install is on and this is a real (non-mock) setup, a visible
        install job is queued once per session — the same policy models
        already follow. 'broken' installs are never reinstalled blindly."""
        try:
            report = self.node_pack_report()
        except Exception:  # noqa: BLE001 — unknown means "don't route there"
            return False
        status = next((p["status"] for p in report if p["name"] == slug), None)
        if status == "active":
            return True
        if (status == "absent"
                and self.settings.auto_install
                and self.settings.inpaint_backend != "mock"
                and slug in node_packs.KNOWN_PACKS
                and slug not in self._packs_queued):
            self._packs_queued.add(slug)
            try:
                self.queue.enqueue("node_pack", {"pack": slug})
                self.events.log("info", f"Auto-installing node pack "
                                        f"'{slug}' — a request needs it")
            except Exception:  # noqa: BLE001 — queueing is best-effort
                pass
        return False

    def _on_missing_node(self, exc: Any, client: Any) -> Any | None:
        """The heal behind run_graph: a graph bounced because a node type is
        not installed on the engine it was sent to. Returns the client to
        re-run the graph through, or None to let the error stand. Two cases:

        peer engine   the delegated machine lacks the node (fresh install).
                      Re-run on THIS machine — the local gate passed to get
                      here — and ask the peer to install the curated pack so
                      next time it renders there. The job never fails.

        local engine  install the curated pack NOW (visible in the event
                      feed), restart ComfyUI, retry once. Once per pack per
                      session — a pack that fails to install must not loop.

        Never raises: any surprise here means the original error stands."""
        try:
            classes = tuple(getattr(exc, "class_types", ()) or ())
            pack = next((p for p in
                         (node_packs.pack_for_node(c) for c in classes)
                         if p is not None), None)
            base_url = str(getattr(client, "base_url", ""))
            if "/pf-peer/comfy" in base_url:
                peer_base = base_url.split("/pf-peer/comfy", 1)[0]
                if pack is not None:
                    self.peers.request_pack_install(peer_base, pack.name)
                local = self._comfy_main
                if local is client or getattr(local, "offline", False):
                    return None
                names = ", ".join(classes) or "a required node"
                self.events.log(
                    "info", f"The render machine does not have {names} — "
                            "rendering this step here instead"
                            + (f"; asked it to install '{pack.title}' for "
                               "next time" if pack else ""))
                # Rebind this worker thread to the local engine: callers
                # re-read `self.comfy` per call (result polling included),
                # so the REST of this job follows the graph to the machine
                # that actually took it instead of bouncing per submit.
                try:
                    self._comfy_tls.client = local
                except Exception:  # noqa: BLE001 — the reroute still works
                    pass
                return local
            if (pack is None
                    or not self.settings.auto_install
                    or self.settings.inpaint_backend == "mock"
                    or pack.name in self._heal_attempted):
                return None
            self._heal_attempted.add(pack.name)
            self.events.log(
                "info", f"'{classes[0] if classes else pack.verify_node}' "
                        f"is not installed — installing node pack "
                        f"'{pack.title}' now, then retrying the render")
            result = self._install_pack_now(
                pack, _EventLogJob(self.events, f"[{pack.name}] "))
            return client if result.get("active") else None
        except Exception:  # noqa: BLE001 — healing must never mask the error
            return None

    def _peer_pack_install(self, slug: str) -> dict[str, Any]:
        """A delegating peer's graph needed a node THIS machine lacks and
        asked us to install the pack. Curated slugs only, this machine's
        auto-install setting has the final word, and the install runs as a
        normal visible job in our own queue."""
        if slug not in node_packs.KNOWN_PACKS:
            return {"queued": False, "error": "not a curated pack"}
        if (not self.settings.auto_install
                or self.settings.inpaint_backend == "mock"):
            return {"queued": False, "error": "auto-install is off here"}
        if slug in self._packs_queued:
            return {"queued": True, "note": "already queued"}
        self._packs_queued.add(slug)
        try:
            self.queue.enqueue("node_pack", {"pack": slug})
            self.events.log("info", f"Node pack '{slug}' queued — a "
                                    "connected machine's render needs it")
            return {"queued": True}
        except Exception:  # noqa: BLE001 — honest refusal beats a crash
            self._packs_queued.discard(slug)
            return {"queued": False, "error": "queueing failed"}

    def _face_polish(self, job: Job, image: Image.Image, prompt: str,
                     checkpoint: str | None = None) -> Image.Image | None:
        """FaceDetailer pass over a finished render — the mushy-face fix.

        Every detected face is re-rendered at guide resolution and blended
        back; images without faces pass through nearly untouched (the
        detector decides). The polished image must PROVE itself: the judge
        scores both versions and a worse polish is discarded. Fail-open at
        every miss — flag off, pack absent, detector model still
        downloading, engine errors — the original ships untouched."""
        if (not self.settings.face_detail
                or self.settings.inpaint_backend == "mock"):
            return None
        try:
            if not self._pack_active("impact-pack"):
                return None
            if not self.registry.is_ready("face-yolov8m"):
                # Same self-heal as models everywhere: queue the download
                # once; this render ships unpolished, the next gets it.
                if (self.settings.auto_install
                        and "face-yolov8m" not in self._polish_staged):
                    self._polish_staged.add("face-yolov8m")
                    self.queue.enqueue("model_download",
                                       {"model": "face-yolov8m"})
                    job.log("info", "Face detector is downloading — this "
                                    "render ships as-is, the next gets "
                                    "the face-refinement pass")
                return None
            template = self.workflows.load("facedetail")
            ckpt = checkpoint or next(iter(self._image_checkpoints()), None)
            if not ckpt:
                return None
            job.log("info", "[stage] faces — refining every detected face "
                            "at native resolution")
            graph = build_workflow(template, {
                "image": self.comfy.upload_image(image, "facedetail_src"),
                "checkpoint": ckpt,
                "prompt": (prompt or "")[:300] or "detailed natural face",
                "seed": int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF,
            })
            polished, _pid = self.comfy.run_graph(graph)
            if polished.size != image.size:
                return None
            if self.critic is not None:
                try:
                    subject = (prompt or "a photo")[:200]
                    before = self.critic.critique(image, subject).score
                    after = self.critic.critique(polished, subject).score
                    if after + 0.5 < before:
                        job.log("info", "Face refinement judged worse "
                                        f"({after:g} vs {before:g}) — "
                                        "keeping the original")
                        return None
                    job.log("info", f"Face refinement kept "
                                    f"({after:g} vs {before:g})")
                except CriticUnavailable:
                    pass  # unjudged polish still ships — denoise is gentle
            return polished
        except Exception as exc:  # noqa: BLE001 — polish must never break a render
            job.log("info", f"Face refinement skipped: {str(exc)[:120]}")
            return None

    def _miopen_tiled_retry(self, job: Job, graph: dict[str, Any],
                            exc: Exception) -> dict[str, Any] | None:
        """When AMD's MIOpen fails inside the VAE decode, hand back a tiled
        variant of the graph to retry ONCE — or None when this is not that
        failure (or the graph has no plain VAEDecode to swap).

        Measured live on the RX 6700 XT's native ROCm stack: a WAN video
        SAMPLED to completion and died only at VAEDecode with
        `miopenStatusUnknownError`. On CUDA the same pressure raises a
        clean OOM that ComfyUI answers with its own tiled fallback; on AMD
        it does not, so the render's last step is where we step in."""
        if "miopen" not in str(exc).lower():
            return None
        tiled = tiled_vae_graph(graph, self._live_object_info())
        if tiled is None:
            return None
        job.log("info", "AMD's GPU library (MIOpen) failed inside the VAE "
                        "decode — the sampling itself succeeded, so the "
                        "same render is retried with tiled decoding, which "
                        "needs far less memory at once")
        return tiled

    @staticmethod
    def _miopen_hint(exc: Exception) -> str | None:
        """An honest message for a miopen failure that survived the tiled
        retry — the stock 'update ComfyUI for WAN nodes' advice is wrong
        for this failure and sent the user chasing the wrong fix."""
        if "miopen" not in str(exc).lower():
            return None
        return ("The render failed inside AMD's GPU library (MIOpen) at "
                "the VAE decode, even after an automatic tiled-decode "
                "retry. Lower the resolution or clip length and try again "
                "— and keep the AMD driver current; the native ROCm stack "
                "on this card is still maturing.")

    def _template_runnable(self, task: str) -> tuple[bool, str]:
        """Can this machine run the task's template RIGHT NOW — template
        present, its models downloaded, its memory needs met? The edit
        pipeline asks before routing to a real capability, so it can either
        use it or say plainly why it can't."""
        try:
            # load_named, not load: `load` gates on the TASK name, but a
            # variant is a FILE prefix ("relight_bg" declares task "relight").
            # Using load here made every variant report "not an allowed
            # workflow type", which silently disabled the whole feature -
            # background relighting and the fast motion path both never ran.
            # load_named still re-checks the declared task, so nothing is
            # loosened.
            template = self.workflows.load_named(task)
        except Exception as exc:  # noqa: BLE001 — no template is an answer
            return False, str(exc)
        required = template.get("required_models") or []
        fits, why = self._models_fit_machine(required)
        if not fits:
            return False, why
        missing = [m for m in required if not self.registry.is_ready(m)]
        if missing:
            return False, f"{', '.join(missing)} not downloaded yet"
        return True, ""

    @staticmethod
    def _pad_mask(before: tuple[int, int], after: tuple[int, int],
                  dirs: dict[str, int] | None = None) -> Image.Image | None:
        """White mask over the margins an outpaint added — lets the seam
        inspector examine the outpaint boundary. None when the canvas did
        not grow. Without `dirs` the original is assumed centered (the
        template's symmetric left+right default); with directional padding
        the named sides carry the whole offset."""
        bw, bh = before
        aw, ah = after
        if aw <= bw and ah <= bh:
            return None
        if dirs:
            left = min(max(0, dirs.get("left", 0)), max(0, aw - bw))
            top = min(max(0, dirs.get("top", 0)), max(0, ah - bh))
        else:
            left = max(0, (aw - bw) // 2)
            top = max(0, (ah - bh) // 2)
        mask = Image.new("L", (aw, ah), 255)
        mask.paste(0, (left, top, min(aw, left + bw), min(ah, top + bh)))
        return mask

    # Outpainting occasionally paints a NEW person into the fresh margin —
    # the continuation prompt and the negative both fight it, yet measured
    # 2026-08-18 (8 production-style renders): 1 in 8 grew a standalone man.
    # BiRefNet-portrait separates the cases perfectly on the same data:
    # every clean margin matted 0.0% person, the intruder 24.6% — so the
    # floor below is nowhere near a judgement call. The inner-slab ceiling
    # tells an INVENTED person (mass only in the margin) from a legitimate
    # completion of a subject who runs off the original edge (mass continues
    # inland past the junction).
    _MARGIN_PERSON_MIN = 0.06
    _MARGIN_INNER_MAX = 0.02
    _MARGIN_INNER_SLAB = 96

    @staticmethod
    def _margin_geometry(image_size: tuple[int, int],
                         pre_size: tuple[int, int],
                         dirs: dict[str, int] | None
                         ) -> tuple[int, int, int, int]:
        """(left, top, right, bottom) margin extents of an outpainted frame.
        `dirs` carries directional padding; without it the original is
        assumed centered (the template's symmetric left+right default).
        Shared by the person guard and the glyph guard so their geometry
        can never drift apart."""
        aw, ah = image_size
        bw, bh = pre_size
        if dirs:
            left = min(max(0, dirs.get("left", 0)), max(0, aw - bw))
            top = min(max(0, dirs.get("top", 0)), max(0, ah - bh))
        else:
            left = max(0, (aw - bw) // 2)
            top = max(0, (ah - bh) // 2)
        return left, top, aw - bw - left, ah - bh - top

    def _margin_intruders(self, image: Image.Image,
                          pre_size: tuple[int, int],
                          dirs: dict[str, int] | None = None) -> list[str]:
        """Names of outpaint margins holding a STANDALONE person the render
        invented. Deterministic (BiRefNet-portrait matte per margin, plus an
        inner slab of the original for the continuation test); empty when the
        canvas did not grow, the rmbg pack is off, or matting fails — the
        guard never blocks an outpaint, it only asks for one re-render."""
        import numpy as np
        if not self._pack_active("rmbg"):
            return []
        aw, ah = image.size
        left, top, right, bottom = self._margin_geometry(
            image.size, pre_size, dirs)
        slab = self._MARGIN_INNER_SLAB
        checks: list[tuple[str, tuple[int, int, int, int], int, str]] = []
        if left >= 32:
            checks.append(("left", (0, 0, min(aw, left + slab), ah),
                           left, "x"))
        if right >= 32:
            checks.append(("right", (max(0, aw - right - slab), 0, aw, ah),
                           right, "x2"))
        if top >= 32:
            checks.append(("top", (0, 0, aw, min(ah, top + slab)),
                           top, "y"))
        if bottom >= 32:
            checks.append(("bottom", (0, max(0, ah - bottom - slab), aw, ah),
                           bottom, "y2"))
        hits: list[str] = []
        for name, box, extent, axis in checks:
            matte = self._region_mask(image.crop(box), "BiRefNetRMBG", {
                "model": self._MATTE_PORTRAIT, "sensitivity": 1.0,
                "mask_blur": 0, "mask_offset": 0, "invert_output": False,
                "refine_foreground": True, "background": "Alpha",
                "background_color": "#222222"})
            if matte is None:  # empty matte or engine hiccup: fail open
                continue
            a = np.asarray(matte) > 127
            if axis == "x":
                margin_part, inner_part = a[:, :extent], a[:, extent:]
            elif axis == "x2":
                margin_part, inner_part = a[:, -extent:], a[:, :-extent]
            elif axis == "y":
                margin_part, inner_part = a[:extent, :], a[extent:, :]
            else:
                margin_part, inner_part = a[-extent:, :], a[:-extent, :]
            if margin_part.size == 0 or inner_part.size == 0:
                continue
            if (float(margin_part.mean()) >= self._MARGIN_PERSON_MIN
                    and float(inner_part.mean()) <= self._MARGIN_INNER_MAX):
                hits.append(name)
        return hits

    def _guarded_outpaint(self, job: Job, src: Image.Image, positive: str,
                          negative: str, checkpoint: str | None,
                          real: bool,
                          dirs: dict[str, int] | None = None) -> Image.Image:
        """One outpaint render, person-guarded: when a margin grew a
        standalone person, re-render ONCE with a fresh seed and keep the
        cleaner of the two. Honest logs either way. `dirs` = directional
        padding, threaded to the template and the margin geometry alike."""
        rendered = self._render_template_step(
            job, "outpaint", src, positive, negative, checkpoint=checkpoint,
            extra=dirs)
        if not real:
            return rendered
        chosen = rendered
        try:
            intruders = self._margin_intruders(rendered, src.size, dirs)
        except Exception:  # noqa: BLE001 — the guard must never kill a render
            intruders = []
        if intruders:
            job.log("info", "[stage] guard — the outpaint invented a person "
                            f"in the {' and '.join(intruders)} margin; "
                            "re-rendering the extension with a fresh seed")
            try:
                second = self._render_template_step(
                    job, "outpaint", src, positive, negative,
                    checkpoint=checkpoint, extra=dirs)
                second_hits = self._margin_intruders(second, src.size, dirs)
                if len(second_hits) < len(intruders):
                    job.log("info", "[stage] guard — the re-rendered "
                                    "margins are clean; keeping the "
                                    "re-render")
                    chosen = second
                else:
                    job.log("info", "[stage] guard — the re-render was no "
                                    "better; keeping the first extension")
            except Exception:  # noqa: BLE001 — keep the result we have
                job.log("info", "[stage] guard — the re-render failed; "
                                "keeping the first extension")
        deglyphed = self._deglyph_outpaint(job, chosen, src, positive,
                                           negative, checkpoint, dirs)
        return self._harmonize_outpaint(job, deglyphed, src, dirs)

    def _harmonize_outpaint(self, job: Job, image: Image.Image,
                            src: Image.Image,
                            dirs: dict[str, int] | None) -> Image.Image:
        """Exposure continuity across outpaint junctions. The sampler paints
        plausible margins with its OWN exposure — measured on three
        independent renders of the same request, the low-frequency colour
        step across the right junction sat in the top percentile of the
        image's own strip statistics every time, the recurring
        "lighting/colour mismatch" of inspection reports. Deterministic and
        fail-safe (any hiccup returns the render unchanged); it touches the
        margins and the feather's inland reach only — everything beyond
        stays byte-identical."""
        try:
            pads = self._margin_geometry(image.size, src.size, dirs)
            fixed, steps = quality.harmonize_margins(image, src, pads)
        except Exception:  # noqa: BLE001 — the guard must never kill a render
            return image
        if steps:
            pretty = ", ".join(f"{side} {step:.0f}/255"
                               for side, step in sorted(steps.items()))
            job.log("info", "[stage] guard — matched the extension's "
                            "exposure to the picture at the junction "
                            f"(measured step: {pretty})")
        return fixed

    def _ground_scores(self, job: Job, scores: dict[str, int] | None,
                       image: Image.Image,
                       last_outpaint: dict[str, Any] | None
                       ) -> dict[str, int] | None:
        """Measured junction defects overrule the whole-frame artifact
        score. The scorecard rated a render carrying a REAL junction
        stripe artifact_free 97 — at frame scale the judge cannot see the
        artifact class outpaints produce, and that score is the gate that
        decides pass/retry. Cycle-21 doctrine: where a cheap measurement
        is exact, it outranks the vision judge — here it can only LOWER
        the score, never raise it."""
        if not scores or not last_outpaint:
            return scores
        try:
            pads = self._margin_geometry(image.size,
                                         last_outpaint["pre_size"],
                                         last_outpaint.get("dirs"))
            flaw = quality.junction_flaws(image, pads)
        except Exception:  # noqa: BLE001 — a score guard must never raise
            return scores
        if flaw and scores.get("artifact_free", 0) > flaw[0]:
            cap, why = flaw
            job.log("info", "[stage] score — the measured junction "
                            f"overrules the artifact score: {why} "
                            f"(artifact_free {scores['artifact_free']} "
                            f"→ {cap})")
            scores = dict(scores)
            scores["artifact_free"] = cap
        return scores

    # Testability seam: the glyph detector, overridable per instance.
    _glyph_rows = staticmethod(quality.glyph_band_rows)

    def _margin_glyph_rows(self, image: Image.Image,
                           pre_size: tuple[int, int],
                           dirs: dict[str, int] | None
                           ) -> tuple[int, int] | None:
        """The union row range of glyph soup across the LEFT/RIGHT margins
        of an outpainted frame, or None when both are clean."""
        left, top, right, _bottom = self._margin_geometry(
            image.size, pre_size, dirs)
        found: list[tuple[int, int]] = []
        for extent, box in (
                (left, (0, 0, left, image.height)),
                (right, (image.width - right, 0, image.width,
                         image.height))):
            if extent < 32:
                continue
            rows = self._glyph_rows(image.crop(box))
            if rows is not None:
                found.append(rows)
        if not found:
            return None
        return (min(r[0] for r in found), max(r[1] for r in found))

    def _deglyph_outpaint(self, job: Job, rendered: Image.Image,
                          src: Image.Image, positive: str, negative: str,
                          checkpoint: str | None,
                          dirs: dict[str, int] | None) -> Image.Image:
        """When the extension continued the source's caption/watermark band
        into a new margin as unreadable glyph soup (measured 4/4 seeds on an
        affected photo — a plain seed retry cannot fix it), re-render ONCE
        from a copy whose band rows are neutralized with the content just
        above them, then restore the ORIGINAL band over the center
        byte-exactly. The photo keeps its overlay; only the margins stop
        pretending to continue it."""
        try:
            soup = self._margin_glyph_rows(rendered, src.size, dirs)
        except Exception:  # noqa: BLE001 — the guard must never kill a render
            return rendered
        if soup is None:
            return rendered
        band_top = max(0, soup[0] - 8)
        if band_top < 1:  # the "band" is the whole strip: not a band
            return rendered
        job.log("info", "[stage] guard — the extension continued the "
                        "picture's caption/watermark band as unreadable "
                        "text; re-rendering from a band-neutralized copy "
                        "(the original band itself is kept)")
        neutral = src.copy()
        fill = (src.crop((0, band_top - 1, src.width, band_top))
                .resize((src.width, src.height - band_top),
                        Image.Resampling.NEAREST)
                .filter(ImageFilter.GaussianBlur(3)))
        neutral.paste(fill, (0, band_top))
        try:
            second = self._render_template_step(
                job, "outpaint", neutral, positive, negative,
                checkpoint=checkpoint, extra=dirs)
            soup2 = self._margin_glyph_rows(second, src.size, dirs)
        except Exception:  # noqa: BLE001 — keep the result we have
            job.log("info", "[stage] guard — the band-neutralized re-render "
                            "failed; keeping the first extension")
            return rendered
        if soup2 is not None and (soup2[1] - soup2[0]) >= (soup[1] - soup[0]):
            job.log("info", "[stage] guard — the re-render was no cleaner; "
                            "keeping the first extension")
            return rendered
        # The graph composited the render over the padded NEUTRAL copy, so
        # the neutralized rows sit in the output's center — put the real
        # band back, byte-exact.
        left, top, _r, _b = self._margin_geometry(second.size, src.size, dirs)
        second.paste(src.crop((0, band_top, src.width, src.height)),
                     (left, top + band_top))
        job.log("info", "[stage] guard — margins re-rendered without the "
                        "text soup; the picture's own band is restored "
                        "unchanged")
        return second

    def _best_outpaint_checkpoint(self) -> str | None:
        """The best installed INPAINT checkpoint for outpainting (outpaint IS
        inpainting of the padded margins). Deterministic — no LLM roundtrip;
        None keeps the template's default."""
        try:
            installed = self._image_checkpoints()
        except Exception:  # noqa: BLE001 — ComfyUI down / fake without the API
            return None
        inpaintable = [c for c in installed if "inpaint" in c.lower()]
        if not inpaintable:
            return None
        return sorted(inpaintable, key=self._inpaint_rank)[0]

    def _render_compose_step(self, job: Job, target: Image.Image,
                             subject: Image.Image, mask: Image.Image | None,
                             positive: str, negative: str,
                             denoise: float | None = None,
                             drawn: bool = False) -> Image.Image:
        """Bring the subject of a SECOND photo into this one.

        The subject is matted out of its own background, scaled into the
        placement region and composited, then one low-denoise pass makes the
        lighting, grain and edges agree. Matting is BiRefNet, not SAM: SAM is
        a part segmenter and returns a shirt where a person was asked for."""
        self._require_comfy(job)
        template = self.workflows.load("compose")
        box = quality.placement_box(mask, target.size, subject.size)
        job.log("info", f"Placing the subject {box['w']}×{box['h']} px at "
                        f"{box['x']},{box['y']} of a "
                        f"{target.width}×{target.height} photo"
                        + (" (the region you painted)" if drawn
                           else " (spot chosen from the scene)" if mask
                           is not None else " (no spot given — centre-front)"))
        params: dict[str, Any] = {
            "image": self.comfy.upload_image(target, "compose_bg"),
            "subject": self.comfy.upload_image(subject, "compose_subject"),
            "sub_w": box["w"], "sub_h": box["h"],
            "pos_x": box["x"], "pos_y": box["y"],
            "prompt": positive, "negative": negative,
            "seed": int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF,
        }
        if denoise is not None:
            # Clamped: past ~0.35 the harmonisation pass stops blending the
            # subject and starts redrawing its face.
            params["denoise"] = max(0.10, min(0.35, float(denoise)))
        ckpt = self._best_compose_checkpoint()
        if ckpt:
            params["checkpoint"] = ckpt
            job.log("info", f"Harmonising with {ckpt}")
        try:
            graph = build_workflow(template, params)
        except WorkflowValidationError as exc:
            raise PermanentError(f"compose template error: {exc}") from exc
        self._free_vram(job)
        try:
            out, _pid = self.comfy.run_graph(graph)
            return out
        except BackendUnavailableError as exc:
            raise self._comfy_died_midrender(job, "compose", positive,
                                             exc) from exc
        except WorkflowRuntimeError as exc:
            self._diagnose_and_record(job, "compose", positive, str(exc))
            raise PermanentError(
                f"compose render failed: "
                f"{commit_exhausted_hint(str(exc)) or exc}") from exc

    # BiRefNet variants, chosen per subject. "-portrait" is trained on human
    # portraits and holds hair edges best; "_lite" is the fast general model
    # already proven on this machine in the compose path (4 s, and it matted a
    # subject at 19.4% against a 19.4% ground truth).
    _MATTE_PORTRAIT = "BiRefNet-portrait"
    _MATTE_GENERAL = "BiRefNet_lite"
    _PERSON_HINT = re.compile(
        r"\b(person|people|man|men|woman|women|boy|girl|guy|lady|human|"
        r"portrait|selfie|model|subject|face|figure|someone|himself|herself|"
        r"myself|themselves|he|she|they|him|her|his|their|me|my)\b",
        re.IGNORECASE)

    def _matte_model(self, hint: str) -> str:
        """Which BiRefNet variant mattes this subject best."""
        return (self._MATTE_PORTRAIT if self._PERSON_HINT.search(hint or "")
                else self._MATTE_GENERAL)

    @staticmethod
    def _transfer_lighting(original: Image.Image, relit: Image.Image,
                           strength: float = 1.0) -> Image.Image:
        """Give `original` the ILLUMINATION of `relit`, keeping its detail.

        A relight pass runs at high denoise, so it re-draws the subject — the
        face comes back subtly different, which is exactly what must never
        happen. Instead of keeping its pixels, keep only its LOW-FREQUENCY
        light: the per-channel ratio between the blurred relit image and the
        blurred original is the lighting change, and multiplying the original
        by that ratio applies the new light to the untouched original detail.
        Skin texture, features and edges are the input's; only the shading,
        direction and colour temperature come from the render."""
        original = original.convert("RGB")
        relit = relit.convert("RGB").resize(original.size, Image.Resampling.LANCZOS)
        # Large enough to carry illumination, far too large to carry features.
        radius = max(6, int(max(original.size) * 0.04))
        blur = ImageFilter.GaussianBlur(radius)
        def relight_channel(a: dict[str, Any]) -> Any:
            # +12 keeps near-black areas from exploding the ratio, and the
            # image expression has to lead the min() — a bare float has no
            # apply().
            scaled = (a["float"](a["o"])
                      * ((a["float"](a["rb"]) + 12.0)
                         / (a["float"](a["ob"]) + 12.0)))
            return a["min"](scaled, 255.0)

        out = []
        for src, lit in zip(original.split(), relit.split(), strict=True):
            ob, rb = src.filter(blur), lit.filter(blur)
            ch = ImageMath.lambda_eval(relight_channel, o=src, rb=rb, ob=ob)
            out.append(ch.convert("L"))
        lit_img = Image.merge("RGB", out)
        if strength >= 0.999:
            return lit_img
        return Image.blend(original, lit_img, max(0.0, min(1.0, strength)))

    def _region_mask(self, image: Image.Image, node: str,
                     inputs: dict[str, Any]) -> Image.Image | None:
        """Run one rmbg segmenter and return its MASK output as an L image."""
        try:
            name = self.comfy.upload_image(image, "seg_src")
            key = "images" if node in ("ClothesSegment", "BodySegment") \
                else "image"
            graph = {
                "1": {"class_type": "LoadImage", "inputs": {"image": name}},
                "2": {"class_type": node,
                      "inputs": {key: ["1", 0], **inputs}},
                "3": {"class_type": "MaskToImage",
                      "inputs": {"mask": ["2", 1]}},
                "4": {"class_type": "SaveImage",
                      "inputs": {"filename_prefix": "pf_seg",
                                 "images": ["3", 0]}},
            }
            validate_workflow(graph)
            data, _f = self.comfy.wait_for_output_file(
                self.comfy.submit(graph))
            mask = Image.open(io.BytesIO(data)).convert("L")
        except Exception:  # noqa: BLE001 — the caller falls back
            return None
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        return mask if mask.getbbox() else None

    # CLIPSeg's own confidence, below which it is saying "I do not see that".
    # Measured on a waist-up photograph: things present peaked at 0.88-0.94
    # (phone 0.941, top 0.935, hair 0.887, face 0.884) and things absent at
    # 0.08-0.17 (shoes 0.081, necklace 0.168). The gap is wide enough that
    # this floor is not a fine judgement call.
    _TEXT_MASK_FLOOR = 0.45
    _text_mask_worker: _TextMaskWorker | None = None
    _text_mask_worker_lock = threading.Lock()
    # Per-image calibration on top of the absolute floor: the request's peak
    # must beat the best ABSENT-object control peak on the same image by
    # this margin. A single global floor cannot separate present from absent
    # across photographs — the same absent phrase scored 0.61-0.65 on one
    # photo and under the floor on another, while present objects scored
    # 0.90+ (Step 9).
    _TEXT_MASK_MARGIN = 0.12
    _TEXT_MASK_CONTROLS = ("a purple octopus", "an igloo", "a unicycle")

    def _text_mask(self, image: Image.Image, phrases: list[str],
                   log: Callable[[str], None] | None = None,
                   ) -> tuple[Image.Image | None, dict]:
        """Segment what the WORDS name, via CLIPSeg.

        Exists because the general segmenter cannot read: SAM scores its
        candidates on geometry alone, and returned the identical mask for
        "remove the necklace" and "change her shoes" on two different photos.
        The two off-the-shelf fixes are both unavailable here — GroundingDINO
        crashes against transformers 5.13, and the CLIPSeg node is a stub for
        an extension that is not installed — so this runs the model directly,
        in ComfyUI's interpreter, the same way the mesh texturer does."""
        # Mock mode means OFFLINE (the _live_object_info rule): a ComfyUI
        # install on this box belongs to some other setup, and borrowing its
        # interpreter here ran a real 35-second CLIPSeg inference inside a
        # supposedly mocked edit (measured live — the mock is for tests and
        # demos, which must stay fast and self-contained).
        if self.settings.inpaint_backend == "mock":
            return None, {}
        base = Path(self.settings.comfyui_dir) if self.settings.comfyui_dir \
            else None
        tool = Path(__file__).resolve().parent.parent / "tools" / \
            "text_mask.py"
        if base is None or not base.exists() or not tool.exists() or not phrases:
            return None, {}
        try:
            python = self._comfy_python(base)
        except Exception:  # noqa: BLE001 — the caller falls back
            return None, {}
        # Init under a lock: the peer-helper thread can run an image_edit
        # concurrently with the main worker, and check-then-set here would
        # give each its own 700 MB engine (the loser only self-reaps at the
        # idle timeout).
        with self._text_mask_worker_lock:
            worker = self._text_mask_worker
            if worker is None or worker.python != python:
                worker = self._text_mask_worker = _TextMaskWorker(
                    python, str(tool))
        if log is not None and not worker.warm:
            # Only the COLD start is slow (torch import, ~35s — minutes
            # under load or on the first run ever while weights download);
            # warm answers take ~2s and need no explanation. Say so — a
            # stalled log line reads as a hang, and this step was one.
            log("Starting the text engine — its model loads once (about "
                "half a minute, longer on a busy machine or while its "
                "weights first download) and later requests answer in "
                "seconds")
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "src.png", Path(tmp) / "mask.png"
            image.convert("RGB").save(src, format="PNG")
            report = worker.ask(str(src), str(out), phrases[:4],
                                list(self._TEXT_MASK_CONTROLS), 0.40)
            if report is None or not out.exists():
                return None, {}
            peak = float(report.get("peak", 0.0))
            if peak < self._TEXT_MASK_FLOOR:
                return None, report
            control_peak = report.get("control_peak")
            if (control_peak is not None
                    and peak < float(control_peak) + self._TEXT_MASK_MARGIN):
                # The request scored barely above what a purple octopus
                # scores on this same image — that is noise wearing a mask.
                report["calibration_rejected"] = True
                return None, report
            mask = Image.open(out).convert("L")
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.Resampling.NEAREST)
            return (mask if mask.getbbox() else None), report

    def preview_region(self, image: Image.Image,
                       request: str) -> quality.MaskChoice | None:
        """The region the RENDER will actually use, for requests that route
        to a whole-frame engine — or None to use the normal chooser.

        The preview used to draw a text-matched patch covering 10-23% of the
        frame for "change the background" while the render inverted a
        BiRefNet subject matte and repainted everything around the person
        (D11). The region approved on screen was not the region repainted.
        Background requests now preview the actual inverted matte; the other
        whole-frame engines say plainly that the whole frame is in play."""
        if quality.background_intent(request):
            matte = None
            if self._pack_active("rmbg"):
                try:
                    matte = self._region_mask(image, "BiRefNetRMBG", {
                        "model": self._matte_model(request),
                        "sensitivity": 1.0, "mask_blur": 0, "mask_offset": 0,
                        "invert_output": False, "refine_foreground": True,
                        "background": "Alpha",
                        "background_color": "#222222"})
                except Exception:  # noqa: BLE001 — fall through to honest note
                    matte = None
            if matte is not None:
                inverted = ImageChops.invert(
                    quality.fit_mask(matte, image.size))
                return quality.MaskChoice(
                    inverted, "background", "",
                    ["a background swap repaints everything AROUND the "
                     "subject — this is that region, from the same matte "
                     "the render uses"])
            return quality.MaskChoice(
                Image.new("L", image.size, 255), "background", "",
                ["a background swap repaints everything around the subject; "
                 "the matting engine is offline, so the whole frame is "
                 "shown"])
        whole = None
        if quality.pose_intent(request):
            whole = ("a pose change rebuilds the subject and repaints where "
                     "the body moves")
        elif quality.view_intent(request):
            whole = ("a viewpoint change re-renders the whole picture from "
                     "a new camera")
        elif quality.animate_intent(request):
            whole = "animation renders new frames of the whole picture"
        elif quality.scene3d_intent(request):
            whole = "a 3D scene rebuild consumes the whole picture"
        if whole:
            return quality.MaskChoice(
                Image.new("L", image.size, 255), "whole-frame", "",
                [whole + " — a painted region would not be used by this "
                         "engine"])
        return None

    def auto_mask(self, image: Image.Image, request: str,
                  job: Job | None = None) -> quality.MaskChoice:
        """The one mask chooser, used by the render AND by the preview.

        They used to differ: /api/masks/preview called the raw segmenter
        directly, so the red overlay you approved was the weakest of the
        three engines and was not what the pipeline would go on to use.

        Order is by strength of evidence, not by convenience:
          named parts   a fixed vocabulary, exact here, and the only path
                        with a structural guarantee that the face is excluded
          text          CLIPSeg — reads the request
          SAM           geometry only; flagged as a guess, because it is one
        """
        def log(message: str) -> None:
            if job is not None:
                job.log("info", message)

        notes: list[str] = []
        subject = None
        confine = quality.about_the_subject(request)
        if confine and self._pack_active("rmbg"):
            subject = self._region_mask(image, "BiRefNetRMBG", {
                "model": self._matte_model(request), "sensitivity": 1.0,
                "mask_blur": 0, "mask_offset": 0, "invert_output": False,
                "refine_foreground": True, "background": "Alpha",
                "background_color": "#222222"})

        def gated(source: str, mask: Image.Image) -> quality.MaskChoice | None:
            """One candidate through the deterministic gates. None when it
            does not survive them — the caller then tries the next rung."""
            verdict = quality.mask_verdict(mask, subject, confine=confine)
            if not verdict["ok"]:
                log(f"The {source} selection was rejected: {verdict['reason']}")
                return None
            if verdict["repaired"]:
                notes.append(f"trimmed {verdict['trimmed']}% that fell "
                             "outside the subject")
                log(f"Trimmed {verdict['trimmed']}% of the selection that "
                    "fell outside the subject")
            if source == "sam":
                # Defence in depth: whatever SAM proposed, the face is not an
                # edit region unless the words asked for one. The named-part
                # engine has this guarantee structurally; the geometry
                # fallback now gets the same one (D4).
                shielded = self._shield_face(image, verdict["mask"], request)
                if shielded is None:
                    log("The geometric selection was mostly the subject's "
                        "face, which the request does not mention — "
                        "rejected")
                    return None
                if shielded is not verdict["mask"]:
                    verdict["mask"] = shielded
                    notes.append("the face area was excluded from this "
                                 "selection")
                    log("The face area was excluded from the geometric "
                        "selection")
                notes.append("chosen by shape and position, not by your "
                             "words — this engine cannot read the request")
                log("No engine here could read the request, so the region "
                    "was chosen by shape and position. Check it before "
                    "rendering, or paint it yourself.")
            return quality.MaskChoice(verdict["mask"], source, "", notes)

        # The rungs are walked LAZILY, and a rejected candidate falls to the
        # next one. The previous version collected a single candidate up front
        # and treated its rejection as the chooser's answer, so a named-part
        # mask that failed the geometry gates ended the whole search — the job
        # died with "no usable region could be selected" without ever asking
        # the engine that can actually read the request.
        named = self._multi_region_mask(job, image, request)
        if named is not None:
            chosen = gated("named-part", named)
            if chosen is not None:
                return chosen
            log("The named-region selection did not survive the geometry "
                "checks — asking the engine that reads the request instead")

        phrases = quality.mask_phrases(request)
        found, report = self._text_mask(image, phrases, log=log)
        if found is not None:
            notes.append(f"read the request as {', '.join(phrases)} "
                         f"(confidence {report.get('peak')})")
            if (not confine
                    and quality.mask_fraction(found) > quality.MASK_CEILING):
                # The ceiling exists to stop GEOMETRY engines from repainting
                # everything on a whim. This engine READ the request and
                # answered, confidently, that the named thing covers
                # essentially the whole picture — "make the sky warmer" on a
                # photo that is nearly all sky. That is not a missing answer,
                # it is a whole-frame edit; failing it with "nothing matching
                # is clearly visible" stated the opposite of what happened
                # (measured live). Subject-confined requests keep the strict
                # path: a garment covering 95% is a segmenter error, and the
                # matte trim below is what fixes it.
                log("What you asked to change covers essentially the whole "
                    "picture — the whole frame is the edit region")
                return quality.MaskChoice(
                    Image.new("L", image.size, 255), "whole-frame", "",
                    notes + ["what you asked to change covers essentially "
                             "the whole picture, so the whole frame will be "
                             "repainted to match the request"])
            chosen = gated("text", found)
            if chosen is not None:
                return chosen
            # It READ the request, found its best region, and that region did
            # not survive. That is evidence about the picture, not a reason to
            # guess geometrically: see the SAM note below.
            return quality.MaskChoice(
                None, "none",
                f"nothing matching '{', '.join(phrases)}' is clearly "
                "visible in this picture — the closest region did not "
                "survive the geometry checks", notes)
        if report:
            # CLIPSeg looked and did not find it. That is an answer.
            log(f"Nothing matching '{', '.join(phrases)}' is visible in "
                f"this picture (confidence {report.get('peak')}, needs "
                f"{self._TEXT_MASK_FLOOR})")
            return quality.MaskChoice(
                None, "none",
                f"nothing matching '{', '.join(phrases)}' is visible in "
                "this picture", notes)

        # ONLY when no engine that reads the request could run at all. When
        # one ran and its region was then rejected, that is evidence the
        # object is NOT there — falling through to SAM there is what put the
        # edit region on the subject's FACE for "change her shoes" on a
        # portrait with no shoes in it (D4): SAM cannot read, so its default
        # candidate is centred and mid-sized, and on a portrait that is the
        # face. Both of those paths return above rather than reaching here.
        try:
            proposed = self.segmentation.propose_mask(image, request)
        except Exception as exc:  # noqa: BLE001 — reported, not raised
            log(f"Segmentation failed: {exc}")
        else:
            chosen = gated("sam", proposed)
            if chosen is not None:
                return chosen
        return quality.MaskChoice(
            None, "none", "no usable region could be selected", notes)

    # Face-shaped words: only these entitle a mask to overlap the face.
    _FACE_WORDS = re.compile(
        r"\b(face|facial|head|hair|eyes?|eyebrows?|mouth|lips?|nose|cheeks?|"
        r"chin|forehead|beard|moustache|mustache|makeup|make-up|skin|"
        r"expression|smile|glasses|sunglasses|hat|cap|beanie|helmet|"
        r"earrings?|complexion)\b", re.IGNORECASE)

    def _shield_face(self, image: Image.Image, mask: Image.Image,
                     request: str) -> Image.Image | None:
        """The mask with the head region subtracted, unless the request is
        ABOUT the face. None when what remains is nothing — i.e. the
        selection WAS the face, so there is no usable region at all.

        Uses the subject-matte head box rather than the full face parser:
        this runs on the fallback path where the region is already a guess,
        and an approximate shield that always runs beats an exact one that
        needs a second model round-trip."""
        if self._FACE_WORDS.search(request or ""):
            return mask
        try:
            box = self._head_crop(image)
        except Exception:  # noqa: BLE001 — the shield is best-effort
            return mask
        if box is None:
            return mask
        grey = mask.convert("L").point(lambda v: 255 if v >= 128 else 0)
        head = Image.new("L", grey.size, 0)
        head.paste(255, box)
        overlap = ImageChops.multiply(grey, head)
        before = quality.mask_fraction(grey)
        overlap_frac = quality.mask_fraction(overlap)
        if before <= 0 or overlap_frac <= 0:
            return mask
        shielded = ImageChops.subtract(grey, head)
        remaining = quality.mask_fraction(shielded)
        # If the face made up most of the selection, there was no real
        # region — report not-found rather than editing the scraps.
        if remaining < quality.MASK_FLOOR or remaining < before * 0.35:
            return None
        return shielded

    def _multi_region_mask(self, job: Job | None, image: Image.Image,
                           instruction: str) -> Image.Image | None:
        """A mask covering EVERY region the request names, not just the first.

        SAM proposes one grown blob, so "change the bikini" selected the top
        and left the bottom untouched — seen live. Clothing and body parts
        have a named-region segmenter that takes all the parts at once, and
        anything else can be described to GroundingDINO as a list of phrases,
        which detects each one. Returns None when neither is available, and
        the caller falls back to the single-region segmenter."""
        if not self._pack_active("rmbg"):
            return None

        def log(message: str) -> None:
            if job is not None:      # the preview has no job to log against
                job.log("info", message)

        garments = quality.garment_parts(instruction)
        body = quality.body_parts(instruction)
        if garments:
            mask = self._region_mask(image, "ClothesSegment",
                                     {p: True for p in garments})
            if mask is not None:
                log(f"Selected {len(garments)} region(s) — "
                    f"{', '.join(garments)}")
                return mask
        if body:
            mask = self._region_mask(image, "BodySegment",
                                     {p: True for p in body})
            if mask is not None:
                log(f"Selected {', '.join(body)}")
                return mask
        # The text-grounded branch used to live here, gated on there being two
        # or more phrases, with the reasoning that "a single phrase is what
        # SAM already does". That was wrong — SAM does not read text at all —
        # and the gate is why it never ran. Text segmentation now happens in
        # auto_mask() for every request, single phrase included, through
        # CLIPSeg: the node used here needs GroundingDINO, which crashes
        # against transformers 5.13 (its BERT wrapper calls get_head_mask,
        # removed from the library).
        return None

    def _best_inpaint_checkpoint(self) -> str | None:
        """The strongest photoreal INPAINTING checkpoint on this machine.

        Background replacement repaints most of the frame, so the scene it
        invents is only as good as the model painting it. The registry's
        plain SD1.5 inpainting base is a floor, not a ceiling — a photoreal
        inpainting checkpoint gives a markedly more convincing environment.
        SDXL inpainting models are skipped: this graph runs at the source
        resolution and an SDXL load on top of BiRefNet is what pushes an
        8 GB card into offloading."""
        try:
            installed = self._image_checkpoints()
        except Exception:  # noqa: BLE001 — keep the template default
            return None
        paint = [c for c in installed
                 if re.search(r"inpaint", c, re.IGNORECASE)
                 and "xl" not in c.lower()
                 and not self._SURPRISING_NAME.search(c)]
        if not paint:
            return None
        # Prefer a photoreal community model over the plain base.
        photoreal = [c for c in paint
                     if not re.match(r"sd-v1-5-inpainting", c, re.IGNORECASE)]
        return (photoreal or paint)[0]

    _POSTURE_QUESTION = (
        "What is the main subject's body posture in this photo? Reply ONLY "
        'JSON: {"posture": "<one of: standing, sitting, lying, crouching, '
        'kneeling, leaning, unknown>"}')
    _POSTURE_SCHEMA = {
        "type": "object",
        "properties": {"posture": {"type": "string",
                                   "enum": list(scene_geometry.POSTURES)}},
        "required": ["posture"],
    }

    def _probe_geometry(self, job: Job, image: Image.Image
                        ) -> dict[str, Image.Image] | None:
        """One MoGe pass over an image → its depth / normal / validity
        renders, or None when the probe graph fails."""
        template = self.workflows.load_named("scene_probe")
        graph = build_workflow(template, {
            "image": self.comfy.upload_image(image, "probe_src")})
        self._free_vram(job)
        files = self.comfy.wait_for_output_all(self.comfy.submit(graph))
        shots = scene_geometry.parse_probe_files(files)
        return shots if {"depth", "normal", "valid"} <= set(shots) else None

    def _scene_card(self, job: Job, asset_id: str, image: Image.Image
                    ) -> scene_geometry.SceneCard | None:
        """Measure the photograph's physics before an environment edit:
        contact points off the exact matte, ground plane / camera pitch /
        horizon off MoGe's camera-space renders, posture from the vision
        model with a deterministic veto, lighting from the scene graph.
        Fails open — no geometry model or no matte means None, and the
        edit runs exactly as before, with the gap logged honestly."""
        import hashlib
        cache: dict[str, Any] = getattr(self, "_scene_cards", {})
        self._scene_cards = cache
        key = hashlib.md5(image.tobytes()).hexdigest()
        if key in cache:
            return cache[key]
        try:
            # BEFORE the capability gates: pack detection asks the live
            # ComfyUI, so a momentarily-down renderer read as "the rmbg
            # pack is off" and silently skipped the whole measurement.
            self._require_comfy(job)
        except Exception:  # noqa: BLE001 — the render step will retry it
            job.log("info", "[stage] analyze — the renderer is not up yet; "
                            "skipping the geometry measurement")
            return None
        ok, why = self._template_runnable("scene_probe")
        if not ok and self.settings.auto_install and "not downloaded" in why:
            job.log("info", "[stage] models — fetching the geometry model "
                            "(MoGe); this happens once")
            try:
                self._ensure_model("moge-v2", job)
                ok, why = self._template_runnable("scene_probe")
            except Exception as exc:  # noqa: BLE001 — analysis is optional
                ok, why = False, str(exc)
        if not ok or not self._pack_active("rmbg"):
            job.log("info", "[stage] analyze — scene geometry unavailable "
                            f"({why if not ok else 'the rmbg pack is off'});"
                            " the environment will be generated without a "
                            "measured camera/ground contract")
            cache[key] = None
            return None
        job.log("info", "[stage] analyze — measuring the subject's ground "
                        "contact, the camera and the horizon")
        try:
            self._require_comfy(job)
            matte = self._region_mask(image, "BiRefNetRMBG", {
                "model": self._matte_model("person subject"),
                "sensitivity": 1.0, "mask_blur": 0, "mask_offset": 0,
                "invert_output": False, "refine_foreground": True,
                "background": "Alpha", "background_color": "#222222"})
            probe = self._probe_geometry(job, image)
            subj = (scene_geometry.subject_geometry(matte)
                    if matte is not None else {})
            ground = (scene_geometry.ground_geometry(
                probe["normal"], probe["depth"], probe["valid"], matte)
                if probe else {})
            graph = self._scene_cache.get(asset_id) or {}
            card = scene_geometry.SceneCard(
                **subj, **ground,
                lighting=str(graph.get("lighting", "")),
                perspective_note=str(graph.get("perspective", "")),
                setting=str(graph.get("setting", "")))
        except Exception as exc:  # noqa: BLE001 — analysis must not kill
            job.log("info", f"[stage] analyze — scene measurement failed "
                            f"({exc}); proceeding without it")
            cache[key] = None
            return None
        ask = getattr(self.critic, "ask", None)
        if ask is not None and card.subject_box is not None:
            try:
                data = json.loads(ask(image, self._POSTURE_QUESTION,
                                      schema=self._POSTURE_SCHEMA))
                card.posture = scene_geometry.posture_veto(
                    data.get("posture"), card.subject_box)
                card.posture_source = "vision" if card.posture else "none"
            except Exception:  # noqa: BLE001 — posture is optional
                pass
        bits = []
        if card.posture:
            bits.append(card.posture)
        if card.cut_at_bottom:
            bits.append("feet outside the frame")
        elif card.contact_points:
            bits.append(f"{len(card.contact_points)} ground contact "
                        f"point(s)")
        if card.camera_pitch_deg is not None:
            bits.append(f"camera pitch {card.camera_pitch_deg:.0f}°")
        if card.horizon_y_frac is not None:
            bits.append(f"horizon at {card.horizon_y_frac:.0%} of frame "
                        f"height (fit r²={card.horizon_r2:.2f})")
        if card.ground_frac is not None:
            bits.append(f"ground covers {card.ground_frac:.0%}")
        job.log("info", "[stage] analyze — measured: "
                        + ("; ".join(bits) if bits else "nothing reliable"))
        if probe:
            # Debug transparency: the measurement images become inspectable
            # aux versions, exactly like auto-masks already do.
            try:
                for label, im in (("scene depth", probe["depth"]),
                                  ("scene normals", probe["normal"])):
                    p = self.store.new_version_path(asset_id)
                    im.save(p, format="PNG")
                    self.store.add_aux_version(asset_id, str(p), label,
                                               "scene_probe")
            except Exception:  # noqa: BLE001 — debug output only
                pass
            # The raw measurements outlive the card: the perspective guide
            # for depth-conditioned generation is built from them later.
            aux_cache: dict[str, Any] = getattr(self, "_scene_aux", {})
            self._scene_aux = aux_cache
            aux_cache[key] = {**probe, "matte": matte}
            while len(aux_cache) > 6:
                aux_cache.pop(next(iter(aux_cache)))
        cache[key] = card
        while len(cache) > 12:
            cache.pop(next(iter(cache)))
        return card

    def _env_guidance(self, job: Job, asset_id: str, image: Image.Image,
                      card: scene_geometry.SceneCard) -> Image.Image | None:
        """The measured-perspective guide for the depth-conditioned
        background render, or None when the horizon was not confidently
        measured (no fabricated geometry) or the probe images are gone."""
        import hashlib
        aux = getattr(self, "_scene_aux", {}).get(
            hashlib.md5(image.tobytes()).hexdigest())
        if not aux:
            return None
        try:
            guide = scene_geometry.guidance_depth(
                card, aux["depth"], aux["normal"], aux["valid"],
                aux.get("matte"), image.size)
        except Exception:  # noqa: BLE001 — guidance is an upgrade, never a gate
            return None
        if guide is None:
            return None
        job.log("info", "[stage] plan — built a perspective guide from the "
                        "measured geometry (subject depth + the ground's "
                        "ramp to the measured horizon)")
        try:
            p = self.store.new_version_path(asset_id)
            guide.save(p, format="PNG")
            self.store.add_aux_version(asset_id, str(p),
                                       "perspective guide", "scene_probe")
        except Exception:  # noqa: BLE001 — debug output only
            pass
        return guide

    _SURFACE_SCHEMA = {
        "type": "object",
        "properties": {"on_expected": {"type": "boolean"},
                       "seen": {"type": "string"}},
        "required": ["on_expected", "seen"],
    }

    def _contact_surface_miss(self, result: Image.Image,
                              card: scene_geometry.SceneCard,
                              spec: dict[str, Any] | None) -> str | None:
        """Is the planned surface actually under the subject's feet?

        Normals cannot answer this: pool water is an up-facing plane and
        measures exactly like a floor — the first geometry-validated
        render passed pitch, horizon AND the up-normal contact window
        while the subject stood ankle-deep in the pool. So geometry says
        WHERE to look (the measured contact band) and the vision model
        says WHAT is there, region-scoped and schema-forced (the same
        doctrine that fixed the seam inspector: small view, concrete
        question)."""
        ask = getattr(self.critic, "ask", None)
        gs = str((spec or {}).get("ground_surface") or "")
        if (ask is None or not gs or "none" in gs.lower()
                or not card.contact_points or card.cut_at_bottom):
            return None
        xs = [c[0] for c in card.contact_points]
        ys = [c[1] for c in card.contact_points]
        w, h = result.size
        view = result.crop((max(0, min(xs) - int(w * 0.14)),
                            max(0, min(ys) - int(h * 0.10)),
                            min(w, max(xs) + int(w * 0.14)),
                            min(h, max(ys) + int(h * 0.14))))
        # SUPPORT-class only: the first phrasing compared adjectives and
        # rejected "dry tiles" against a planned "wet tiles" — twice, live.
        # The physical contract is that the feet stand on something solid,
        # not that the paint matches the plan word-for-word.
        q = ("In this edited photo the subject should be standing on a "
             f"solid surface ({gs}). Look ONLY at what is directly under "
             "and around the feet in this crop. Set on_expected to false "
             "ONLY if the feet are in water, floating in mid-air, or on "
             "something that could not physically support a standing "
             "person — differences of colour, wetness or material detail "
             "do NOT count. Reply ONLY JSON: "
             '{"on_expected": <true/false>, "seen": "<what the feet are '
             'actually standing on or in>"}')
        try:
            data = json.loads(ask(view, q, schema=self._SURFACE_SCHEMA))
        except Exception:  # noqa: BLE001 — advisory probe
            return None
        if data.get("on_expected") is False:
            seen = str(data.get("seen", "something else"))[:60]
            return (f"the subject's feet are on/in {seen} instead of "
                    f"solid ground ({gs})")
        return None

    def _environment_misses(self, job: Job, result: Image.Image,
                            card: scene_geometry.SceneCard,
                            spec: dict[str, Any] | None = None
                            ) -> list[str]:
        """The SAME geometry measurements, run on the rendered result and
        compared against the card, plus the contact-surface probe.
        Misses feed the retry ladder's avoid-clause and cap the
        consistency score. Each part fails safe independently."""
        after: dict[str, Any] = {}
        try:
            probe = self._probe_geometry(job, result)
            if probe:
                matte = self._region_mask(result, "BiRefNetRMBG", {
                    "model": self._matte_model("person subject"),
                    "sensitivity": 1.0, "mask_blur": 0, "mask_offset": 0,
                    "invert_output": False, "refine_foreground": True,
                    "background": "Alpha", "background_color": "#222222"})
                after = scene_geometry.ground_geometry(
                    probe["normal"], probe["depth"], probe["valid"], matte)
                cgf = scene_geometry.contact_ground_frac(
                    probe["normal"], card.contact_points, result.size)
                if cgf is not None:
                    after["contact_ground_frac"] = cgf
        except Exception:  # noqa: BLE001 — validation must not kill a render
            after = {}
        misses = scene_geometry.environment_misses(card, after)
        try:
            surface = self._contact_surface_miss(result, card, spec)
            if surface:
                misses.append(surface)
        except Exception:  # noqa: BLE001 — validation must not kill a render
            pass
        comparable = ((card.contact_points and not card.cut_at_bottom)
                      or card.camera_pitch_deg is not None
                      or card.horizon_y_frac is not None)
        if misses:
            job.log("info", "[stage] validate — the measured geometry "
                            "disagrees: " + "; ".join(misses))
        elif comparable:
            job.log("info", "[stage] validate — the new scene keeps the "
                            "photograph's camera and ground geometry")
        else:
            job.log("info", "[stage] validate — nothing measurable to "
                            "compare on this photograph")
        return misses

    def _env_scores(self, job: Job, scores: dict[str, int] | None,
                    misses: list[str]) -> dict[str, int] | None:
        """Measured environment-geometry misses overrule the whole-frame
        consistency score, the same doctrine as _ground_scores: the
        scorecard cannot see a pasted-backdrop perspective break, the
        measurement can. Lower-only."""
        cap = 70
        if not scores or not misses:
            return scores
        if scores.get("scene_consistency", 0) > cap:
            job.log("info", "[stage] score — the measured scene geometry "
                            "overrules the consistency score "
                            f"(scene_consistency "
                            f"{scores['scene_consistency']} → {cap})")
            scores = dict(scores)
            scores["scene_consistency"] = cap
        return scores

    def _render_background_step(self, job: Job, image: Image.Image,
                                positive: str, negative: str,
                                subject_hint: str = "",
                                compiled: bool = False,
                                guidance: Image.Image | None = None
                                ) -> Image.Image:
        """Replace what is BEHIND the subject, and nothing else.

        An exact BiRefNet subject matte is INVERTED, so the repainted region is
        the background by construction rather than by whatever SAM thought
        "background" meant. The original subject pixels are then composited
        back through a shrunk, feathered copy of the same matte — so the
        subject's interior is literally the input pixels and cannot drift.

        `guidance` (the measured-perspective depth guide) upgrades the graph
        to the depth-conditioned variant so the new scene inherits the
        photograph's camera instead of inventing its own — words alone held
        the measured horizon on about half the draws. Missing model or
        template → the plain graph, honestly logged."""
        self._require_comfy(job)
        if not self._pack_active("rmbg"):
            raise PermanentError(
                "Changing a background needs the rmbg node pack (BiRefNet) to "
                "cut an exact subject matte. Install it from the Models page. "
                "The alternative, SAM, is a part segmenter — asked for a whole "
                "person here it returned 8.7% of the frame (a shirt), so it "
                "would repaint the subject and leave the backdrop.")
        template = self.workflows.load("background")
        guided = False
        if guidance is not None:
            ok, why = self._template_runnable("background_guided")
            if not ok and self.settings.auto_install \
                    and "not downloaded" in why:
                job.log("info", "[stage] models — fetching the depth "
                                "ControlNet (pins the new scene to the "
                                "measured perspective); this happens once")
                try:
                    self._ensure_model("controlnet-sd15-depth", job)
                    ok, why = self._template_runnable("background_guided")
                except Exception as exc:  # noqa: BLE001 — optional upgrade
                    ok, why = False, str(exc)
            if ok:
                template = self.workflows.load_named("background_guided")
                guided = True
            else:
                job.log("info", f"Perspective conditioning unavailable "
                                f"({why}); rendering with the prompt's "
                                "camera language only")
        matte = self._matte_model(subject_hint or positive)
        job.log("info", f"Matting the subject with {matte}, then repainting "
                        "only the inverted region")
        if not compiled:
            # Ask for an ENVIRONMENT, not a backdrop. Without this the model
            # composes the new scene as a flat picture behind the subject —
            # measured live: a framed forest poster on the original wall.
            # (`compiled` prompts arrive from scene_geometry.spatial_prompt,
            # which already carries this clause plus the measured camera,
            # horizon and ground-contact language.)
            positive = (f"{positive}, the surrounding environment, a real "
                        "place extending behind and around the subject, "
                        "continuous scene, natural depth of field, "
                        "photograph")
            # The repainted region is everything AROUND the subject, and left
            # unguarded the model populates it — a second, headless figure was
            # painted into a snowy mountain scene (D9). The pose route has
            # used this negative for exactly this reason; the background
            # route now does too.
            negative = ", ".join(t for t in (
                (negative or "").strip(" ,"),
                "person, people, human figure, limbs, extra person, "
                "duplicate subject, crowd, text, watermark") if t)
        params: dict[str, Any] = {
            "image": self.comfy.upload_image(image, "bg_src"),
            "rmbg_model": matte,
            "prompt": positive, "negative": negative,
            "seed": int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF,
        }
        if guided and guidance is not None:
            params["control_image"] = self.comfy.upload_image(
                guidance, "bg_guide")
            job.log("info", "Conditioning the new scene on the measured "
                            "perspective guide")
        ckpt = self._best_inpaint_checkpoint()
        if ckpt:
            params["checkpoint"] = ckpt
            job.log("info", f"Repainting the scene with {ckpt}")
        try:
            graph = build_workflow(template, params)
        except WorkflowValidationError as exc:
            raise PermanentError(f"background template error: {exc}") from exc
        self._free_vram(job)
        try:
            # NOT run_graph: it returns only the first image, and this graph
            # deliberately emits two — the finished composite and the plain
            # background plate the relight pass reads the light from. They
            # are told apart by filename prefix, not by order.
            files = self.comfy.wait_for_output_all(self.comfy.submit(graph))
            shots: dict[str, Image.Image] = {}
            for data, fname in files:
                key = "plate" if "bgplate" in fname else "composite"
                shots.setdefault(key, Image.open(io.BytesIO(data)).convert("RGB"))
            out = shots.get("composite")
            if out is None:
                raise WorkflowRuntimeError(
                    "the background graph produced no composite")
            self._last_bg_plate = shots.get("plate")
            return out
        except BackendUnavailableError as exc:
            raise self._comfy_died_midrender(job, "background", positive,
                                             exc) from exc
        except WorkflowRuntimeError as exc:
            self._diagnose_and_record(job, "background", positive, str(exc))
            raise PermanentError(
                f"background render failed: "
                f"{commit_exhausted_hint(str(exc)) or exc}") from exc

    def _match_lighting(self, job: Job, composed: Image.Image,
                        original: Image.Image, scene: str) -> Image.Image:
        """Relight the subject so it belongs in its new background.

        A perfectly matted subject dropped into a new scene still reads as
        fake, because its light does not agree with the scene's: a person lit
        by a ceiling bulb does not belong in a forest at sunset. IC-Light's
        background-conditioned variant reads the light out of the ACTUAL new
        background rather than from a description.

        The render is never used directly. It runs at high denoise and redraws
        the subject, so only its low-frequency illumination is kept and
        applied to the original pixels (_transfer_lighting) — the new light
        lands on the real face rather than on a re-imagined one.

        Returns the input unchanged, with a log line, whenever the relight is
        unavailable — a correctly masked background beats a broken subject."""
        ok, why = self._template_runnable("relight_bg")
        if not ok and self.settings.auto_install and "not downloaded" in why:
            try:
                self._ensure_model("iclight-sd15-fbc", job)
                ok, why = self._template_runnable("relight_bg")
            except Exception as exc:  # noqa: BLE001 — optional polish
                ok, why = False, str(exc)
        if not ok:
            job.log("info", f"Skipping the lighting match ({why}); the "
                            "subject keeps its original light")
            return composed
        try:
            # load_named: this is a VARIANT of the relight task, so the
            # file prefix differs from the task it declares.
            template = self.workflows.load_named("relight_bg")
            # The background alone is the light source: the subject is cut
            # out of the composite so IC-Light sees only the new scene.
            # The plate is the scene WITHOUT the subject. Handing IC-Light
            # the composite instead makes it read the subject's own colours
            # back as ambient light — measured: it pushed a person standing
            # in snow warmer, not cooler.
            plate = getattr(self, "_last_bg_plate", None) or composed
            graph = build_workflow(template, {
                "image": self.comfy.upload_image(composed, "relight_fg"),
                "background": self.comfy.upload_image(plate, "relight_bg"),
                "rmbg_model": self._matte_model(scene),
                "prompt": f"{scene}, natural light on the subject matching "
                          "the scene, consistent shadows and colour "
                          "temperature, photograph",
                "seed": int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF,
            })
            graph = self._apply_hardware_limits(graph, job)
            self._free_vram(job)
            relit, _pid = self.comfy.run_graph(graph)
        except Exception as exc:  # noqa: BLE001 — polish, never fatal
            job.log("info", f"Lighting match unavailable ({exc}); the subject "
                            "keeps its original light")
            return composed
        job.log("info", "Matched the subject's light to the new scene "
                        "(illumination only — the face keeps its own detail)")
        return self._transfer_lighting(composed, relit)

    # The parts that make up "a face" for segmentation. Deliberately excludes
    # hair, ears and neck: including them swaps a whole head, which reads as a
    # pasted cut-out because the hairline never matches the target's lighting.
    # Two passes, because one class is doing something different from the
    # others. FEATURES are things that exist only on a face — you do not have
    # a nose on your shoulder — so they say WHERE the face is. "Skin" is the
    # face parser's skin class run over the whole photograph, and on a beach
    # shot it labels arms and torso too: measured, it returned a 148x445 box
    # spanning 58% of the frame and the swap landed on the subject's chest.
    #
    # So the features locate the face and the skin fills it in, clipped to
    # where the features said the face was. That is the difference between a
    # detector that refuses (the old behaviour, and what the documentation
    # still called a limitation) and one that works.
    _FACE_FEATURES = ("Nose", "Left-eye", "Right-eye", "Left-eyebrow",
                      "Right-eyebrow", "Mouth", "Upper-lip", "Lower-lip")
    _FACE_SURFACE = ("Skin", *_FACE_FEATURES)
    _FACE_PARTS = _FACE_SURFACE      # what ends up in the returned matte
    # A face is roughly this much taller than the eyes-to-chin span, and this
    # much wider — enough for forehead and cheeks, not enough for the neck.
    _FACE_GROW_UP, _FACE_GROW_DOWN, _FACE_GROW_SIDE = 1.05, 0.30, 0.34

    def _head_crop(self, image: Image.Image) -> tuple[int, int, int, int] | None:
        """Roughly where the head is, from the subject matte.

        The face parser works on the frame it is given at a fixed internal
        resolution, so on a full-length photograph the face arrives about 25
        pixels across and it finds nothing — measured, it refused all four
        real photographs tried. Cropping to the head first turns the same
        model from "no face found" into a working detector, and the crop
        needs no face detector of its own: the subject matte is already
        measured exact here (19.4% against a 19.4% ground truth), and a
        standing person's head is the top of it."""
        if not self._pack_active("rmbg"):
            return None
        mask = self._region_mask(image, "BiRefNetRMBG", {
            "model": self._matte_model("person subject"), "sensitivity": 1.0,
            "mask_blur": 0, "mask_offset": 0, "invert_output": False,
            "refine_foreground": True, "background": "Alpha",
            "background_color": "#222222"})
        if mask is None:
            return None
        box = mask.convert("L").point(lambda v: 255 if v > 127 else 0).getbbox()
        if not box:
            return None
        x0, y0, x1, y1 = box
        height = y1 - y0
        # The top quarter of a figure holds the head with room to spare, and
        # is still generous enough for a half-length or seated shot.
        band = mask.crop((x0, y0, x1, y0 + max(24, int(height * 0.25))))
        inner = band.convert("L").point(lambda v: 255 if v > 127 else 0).getbbox()
        if not inner:
            return None
        hx0, hy0, hx1, hy1 = (x0 + inner[0], y0 + inner[1],
                              x0 + inner[2], y0 + inner[3])
        side = int(max(hx1 - hx0, hy1 - hy0) * 1.5)
        cx, cy = (hx0 + hx1) // 2, (hy0 + hy1) // 2
        return (max(0, cx - side // 2), max(0, cy - side // 2),
                min(image.width, cx + side // 2),
                min(image.height, cy + side // 2))

    def _face_region(self, image: Image.Image, job: Job,
                     allow_crop: bool = True
                     ) -> tuple[Image.Image, tuple[int, int, int, int]] | None:
        """(face mask, bounding box) for the largest face in an image.

        Tries the whole frame first — that is enough for a portrait — and
        falls back to parsing a crop around the head when it is not."""
        found = self._face_region_in(image, job)
        if found is not None or not allow_crop:
            return found
        crop = self._head_crop(image)
        if crop is None or (crop[2] - crop[0]) < 32:
            return None
        job.log("info", "The face is small in this frame, so the head is "
                        "cropped out and read on its own")
        inner = self._face_region_in(image.crop(crop), job)
        if inner is None:
            return None
        sub_mask, sub_box = inner
        mask = Image.new("L", image.size, 0)
        mask.paste(sub_mask, (crop[0], crop[1]))
        return mask, (crop[0] + sub_box[0], crop[1] + sub_box[1],
                      crop[0] + sub_box[2], crop[1] + sub_box[3])

    def _face_region_in(self, image: Image.Image,
                        job: Job) -> tuple[Image.Image, tuple[int, int, int, int]] | None:
        """(face mask, bounding box) for the largest face in an image.

        Uses the rmbg pack's face segmenter. None when it is unavailable or
        finds no face — the caller then says so rather than swapping
        something that is not a face."""
        if not self._pack_active("rmbg"):
            return None
        try:
            name = self.comfy.upload_image(image, "face_src")
            graph = {
                "1": {"class_type": "LoadImage", "inputs": {"image": name}},
                "2": {"class_type": "FaceSegment",
                      "inputs": {"images": ["1", 0],
                                 **{p: True for p in self._FACE_FEATURES}}},
                "3": {"class_type": "MaskToImage", "inputs": {"mask": ["2", 1]}},
                "4": {"class_type": "SaveImage",
                      "inputs": {"filename_prefix": "pf_facefeatures",
                                 "images": ["3", 0]}},
                "5": {"class_type": "FaceSegment",
                      "inputs": {"images": ["1", 0],
                                 **{p: True for p in self._FACE_SURFACE}}},
                "6": {"class_type": "MaskToImage", "inputs": {"mask": ["5", 1]}},
                "7": {"class_type": "SaveImage",
                      "inputs": {"filename_prefix": "pf_facesurface",
                                 "images": ["6", 0]}},
            }
            validate_workflow(graph)   # same gate every other graph passes
            outputs = self.comfy.wait_for_output_all(self.comfy.submit(graph))
            found: dict[str, Image.Image] = {}
            for data, fname in outputs:
                key = "features" if "features" in fname else "surface"
                found[key] = Image.open(io.BytesIO(data)).convert("L")
            features, surface = found.get("features"), found.get("surface")
            if features is None:
                return None
        except Exception as exc:  # noqa: BLE001 — no face is an answer
            job.log("info", f"Face detection failed: {exc}")
            return None
        if features.size != image.size:
            features = features.resize(image.size, Image.Resampling.NEAREST)
        anchor = features.point(lambda v: 255 if v > 127 else 0).getbbox()
        if not anchor or (anchor[2] - anchor[0]) < 12 or \
                (anchor[3] - anchor[1]) < 12:
            job.log("info", "No eyes, nose or mouth were found, so there is "
                            "no face here to work with")
            return None
        # Plausibility, now measured on the FEATURES. Eyes-nose-mouth is a
        # wide, short cluster; anything tall and thin is not a face however
        # confident the segmenter was.
        fw, fh = anchor[2] - anchor[0], anchor[3] - anchor[1]
        if not 0.7 <= fw / max(1, fh) <= 3.2:
            job.log("info", f"The facial features found are {fw}×{fh}, which "
                            "is not a face shape — no face was reliably found")
            return None
        if fh > image.height * 0.6 or fw > image.width * 0.8:
            job.log("info", "The features found cover most of the photo, so "
                            "they are not a face — no face was reliably found")
            return None
        # Grow the feature cluster into a whole face: up for the forehead,
        # a little down for the chin, out for the cheeks.
        box = (max(0, int(anchor[0] - fw * self._FACE_GROW_SIDE)),
               max(0, int(anchor[1] - fh * self._FACE_GROW_UP)),
               min(image.width, int(anchor[2] + fw * self._FACE_GROW_SIDE)),
               min(image.height, int(anchor[3] + fh * self._FACE_GROW_DOWN)))
        if (box[2] - box[0]) < 16 or (box[3] - box[1]) < 16:
            return None
        # The matte is the skin pass CLIPPED to that box, so torso skin can
        # never join the face however much of it the parser labelled.
        if surface is not None and surface.size != image.size:
            surface = surface.resize(image.size, Image.Resampling.NEAREST)
        mask = Image.new("L", image.size, 0)
        source = surface if surface is not None else features
        mask.paste(source.crop(box), (box[0], box[1]))
        return mask, box

    def _render_faceswap_step(self, job: Job, target: Image.Image,
                              reference: Image.Image, positive: str,
                              negative: str,
                              denoise: float | None = None) -> Image.Image:
        """Replace the face in `target` with the face in `reference`.

        Both faces are located, the reference's is cut to the target's face
        box, and the composite gets one harmonisation pass so skin tone and
        grain agree. This is a COMPOSITING swap: it moves real pixels, so it
        is at its best when the two faces are at a similar angle, and at its
        worst across very different poses or lighting. The identity-model
        route (InstantID) is better but needs 12 GB of VRAM."""
        src = self._face_region(reference, job)
        if src is None:
            raise PermanentError(
                "No face could be located in the second photo. Face swapping "
                "needs a clear, front-facing, reasonably close-up face — a "
                "full-body or side-on shot will not do. Crop it to the head "
                "and try again.")
        dst = self._face_region(target, job)
        if dst is None:
            raise PermanentError(
                "No face could be located in the photo being edited. This "
                "works on portraits where the face is a decent share of the "
                "frame; on a full-body shot the face is usually too small to "
                "find. Crop in on the head and try again.")
        src_mask, src_box = src
        _dst_mask, dst_box = dst
        job.log("info", f"Face found in both photos — moving a "
                        f"{src_box[2] - src_box[0]}×{src_box[3] - src_box[1]} "
                        f"face onto a "
                        f"{dst_box[2] - dst_box[0]}×{dst_box[3] - dst_box[1]} "
                        "one")
        # Cut the reference face out with its own matte, on a transparent
        # background, then crop tight so the compositor scales the FACE and
        # not the whole photo it came from.
        cut = Image.new("RGBA", reference.size, (0, 0, 0, 0))
        cut.paste(reference.convert("RGB"), (0, 0), src_mask)
        face = cut.crop(src_box)
        # Fit into the target's face box, keeping proportions.
        tw, th = dst_box[2] - dst_box[0], dst_box[3] - dst_box[1]
        scale = min(tw / face.width, th / face.height)
        face = face.resize((max(1, int(face.width * scale)),
                            max(1, int(face.height * scale))), Image.Resampling.LANCZOS)
        # Centre it on the target face's centre, so eyes land near eyes.
        cx = (dst_box[0] + dst_box[2]) // 2
        cy = (dst_box[1] + dst_box[3]) // 2
        ox = max(0, min(target.width - face.width, cx - face.width // 2))
        oy = max(0, min(target.height - face.height, cy - face.height // 2))
        blended = target.convert("RGB").copy()
        alpha = face.split()[-1].filter(ImageFilter.GaussianBlur(3))
        blended.paste(face.convert("RGB"), (ox, oy), alpha)
        # One low-denoise pass over the whole frame reconciles skin tone,
        # grain and edge — the same trick the composite path uses.
        region = Image.new("L", target.size, 0)
        region.paste(255, (ox, oy, ox + face.width, oy + face.height))
        return self._render_template_step(
            job, "img2img", blended,
            f"{positive}, seamless natural skin, matching skin tone and "
            "lighting, consistent grain, photorealistic face",
            negative or "pasted, cutout, hard edge, mismatched skin tone, "
                        "blurry, deformed face",
            max(0.10, min(0.35, float(denoise if denoise is not None else 0.28))),
            checkpoint=self._best_compose_checkpoint())

    def _best_compose_checkpoint(self) -> str | None:
        """A photoreal, NON-inpainting checkpoint for the harmonisation pass.

        An inpainting checkpoint expects a mask it will never be given here,
        so it is the wrong tool even though it is often the best model on the
        machine. Beyond that the pick is DELIBERATE rather than alphabetical:
        a general-purpose base is what this pass wants, and picking whichever
        community checkpoint happens to sort first is both worse at the job
        and a surprising thing to read in a log for an innocuous edit."""
        try:
            installed = self._image_checkpoints()
        except Exception:  # noqa: BLE001 — keep the template default
            return None
        plain = [c for c in installed if not self._NOT_A_GENERATOR.search(c)]
        if not plain:
            return None
        # 1. The plain SD1.5 base the registry ships for exactly this kind of
        #    work. 2. Any other neutral-looking SD1.5. 3. Whatever is left.
        base = self.registry.get("sd15-base")
        wanted = Path((base.meta or {}).get("file") or "").name if base else ""
        for name in plain:
            if wanted and name == wanted:
                return name
        neutral = [c for c in plain
                   if "xl" not in c.lower()
                   and not self._SURPRISING_NAME.search(c)]
        return (neutral or plain)[0]

    # Checkpoint names that would be jarring to see quoted back in the job log
    # of an ordinary edit. Never a capability judgement — purely about not
    # picking one by accident when a neutral base is sitting right there.
    _SURPRISING_NAME = re.compile(r"nsfw|porn|hentai|nude", re.IGNORECASE)

    # How far outside the subject to repaint. A new pose needs somewhere to
    # put the limbs: masked to the CURRENT silhouette, "make her sit down"
    # can only redraw a standing-shaped region and comes back standing.
    _POSE_MARGIN = 0.16
    _POSE_FEATHER = 24

    def _pose_region(self, image: Image.Image,
                     mask: Image.Image | None) -> Image.Image:
        """The area a repose is allowed to repaint: the subject's box, grown.

        A BOX rather than the silhouette, because the whole point is that the
        body ends up somewhere the body is not now. Everything outside it is
        the original photograph, untouched."""
        box = None
        if mask is not None:
            box = mask.convert("L").point(
                lambda v: 255 if v > 127 else 0).getbbox()
        if not box:
            box = (0, 0, image.width, image.height)
        w, h = box[2] - box[0], box[3] - box[1]
        pad_x, pad_y = int(w * self._POSE_MARGIN), int(h * self._POSE_MARGIN)
        region = Image.new("L", image.size, 0)
        region.paste(255, (max(0, box[0] - pad_x), max(0, box[1] - pad_y),
                           min(image.width, box[2] + pad_x),
                           min(image.height, box[3] + pad_y)))
        # Feathered here rather than with FeatherMask, which tapers all four
        # IMAGE borders — and a standing subject's box usually touches one.
        return region.filter(ImageFilter.GaussianBlur(self._POSE_FEATHER))

    def _render_pose_step(self, job: Job, image: Image.Image,
                          instruction: str, positive: str, negative: str,
                          reference: Image.Image | None = None,
                          denoise: float | None = None) -> Image.Image:
        """Put the subject in a different pose, and give them their face back.

        Three parts, in order of how much each one matters:

        the region   only the subject's box is repainted, so the rest of the
                     photograph is literally the original pixels.

        the skeleton when a reference photo is attached, its pose is extracted
                     and drives a ControlNet, which is the difference between
                     asking for a pose and specifying one. Without a
                     reference the wording alone has to carry it, and the
                     result is the model's reading of the words.

        the face     the original face is composited back afterwards. This is
                     the requirement that makes a repose usable at all: a
                     regenerated body with a stranger's face is not the
                     person, however good the pose is."""
        self._require_comfy(job)
        mask = None
        if self._pack_active("rmbg"):
            mask = self._region_mask(image, "BiRefNetRMBG", {
                "model": self._matte_model("person subject"),
                "sensitivity": 1.0, "mask_blur": 0, "mask_offset": 0,
                "invert_output": False, "refine_foreground": True,
                "background": "Alpha", "background_color": "#222222"})
            if mask is not None and self._mask_fraction(mask) < 0.04:
                job.log("info", "The subject cut-out is too small to trust; "
                                "reposing the whole frame instead")
                mask = None
        region = self._pose_region(image, mask)
        template = self.workflows.load_named("pose")
        params: dict[str, Any] = {
            "image": self.comfy.upload_image(image, "pose_src"),
            "mask": self.comfy.upload_image(region.convert("RGB"), "pose_reg"),
            "prompt": positive,
            "negative": negative or (
                "deformed, extra limbs, missing limbs, fused fingers, "
                "distorted anatomy, disfigured, bad proportions, blurry"),
            "seed": int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF,
            "denoise": max(0.6, min(1.0, float(
                denoise if denoise is not None else 1.0))),
        }
        checkpoint = self._best_pose_checkpoint()
        if checkpoint:
            params["checkpoint"] = checkpoint
        if reference is not None:
            params["pose_reference"] = self.comfy.upload_image(
                reference, "pose_ref")
            job.log("info", "Reading the pose out of your reference photo "
                            "(DWPose) and locking the body to it")
        graph = build_workflow(template, params)
        if reference is None:
            graph = self._prune_pose_control(graph)
            job.log("info", "No reference pose attached, so the wording alone "
                            "decides the pose — attach a photo of the pose you "
                            "want and it is followed instead of interpreted")
        graph = self._apply_hardware_limits(graph, job)
        self._prepare_heavy_render(job, need_gb=10.0)
        self._free_vram(job)
        try:
            posed, _pid = self.comfy.run_graph(graph)
        except BackendUnavailableError as exc:
            # exc, not negative: passing the negative prompt here stored
            # "process died: <negative prompt>" in the failure-learning
            # records (found by mypy — the only call site of ten that
            # dropped the exception).
            raise self._comfy_died_midrender(job, "pose", positive,
                                             exc) from None
        except WorkflowRuntimeError as exc:
            self._diagnose_and_record(job, "pose", positive, str(exc))
            raise
        posed = self._keep_original_scene(job, posed, image, mask)
        return self._restore_face(job, posed, image, positive, negative)

    def _keep_original_scene(self, job: Job, posed: Image.Image,
                             original: Image.Image,
                             subject: Image.Image | None) -> Image.Image:
        """Put the photograph's own background back behind the new pose.

        The repaint region has to be a BOX — a body moving needs somewhere to
        move to — and when the subject fills the frame that box is most of the
        picture. Measured on a real photo: asked to sit down, she sat down in
        a different street, in different clothes. The pose was right and the
        photograph was gone.

        So only the PERSON is taken from the render. The background comes from
        the original, with the area they used to occupy painted in, which is
        the same background-plate the relighting path already builds."""
        if subject is None:
            return posed
        if posed.size != original.size:
            posed = posed.resize(original.size, Image.Resampling.LANCZOS)
        new_subject = None
        if self._pack_active("rmbg"):
            new_subject = self._region_mask(posed, "BiRefNetRMBG", {
                "model": self._matte_model("person subject"),
                "sensitivity": 1.0, "mask_blur": 2, "mask_offset": 0,
                "invert_output": False, "refine_foreground": True,
                "background": "Alpha", "background_color": "#222222"})
        if new_subject is None or self._mask_fraction(new_subject) < 0.04:
            job.log("info", "Could not isolate the reposed subject, so the "
                            "whole render is kept — the background may have "
                            "changed with the pose")
            return posed
        # Start from the PHOTOGRAPH, not from a repainted plate. The first
        # version of this reused the background-REPLACEMENT route to try to
        # preserve a background, which is a contradiction in terms: asked for
        # "the same place, empty" it invented open bushland. The original
        # pixels are already the right answer everywhere the old body was not.
        new_mask = quality.fit_mask(new_subject, original.size)
        out = original.convert("RGB").copy()
        out.paste(posed.convert("RGB"), (0, 0), new_mask)
        # What is left is the sliver the body vacated: inside the old
        # silhouette, outside the new one. Only that gets painted in.
        old_mask = quality.fit_mask(subject, original.size)
        hole = ImageChops.subtract(old_mask.convert("L"),
                                   new_mask.convert("L"))
        hole = hole.point(lambda v: 255 if v > 110 else 0)
        hole = hole.filter(ImageFilter.MaxFilter(9))     # cover the seam
        share = self._mask_fraction(hole)
        # The verify stage reads this: a real change of pose vacates a large
        # region, and ~4% means the body did not move (D19). Computed and
        # logged before; now it is also USED.
        self._pose_vacated_share = share
        if share < 0.005:
            job.log("info", "Your own background is kept as it is — the new "
                            "pose covers where the old one was")
            return out
        try:
            # `negative` is unknown to simple adapters; the except keeps
            # the plain paste usable either way.
            filled = self.inpainting.inpaint(  # type: ignore[call-arg]
                out, hole,
                "continue the existing background, empty scene, no people",
                negative="person, figure, limbs, duplicate, text, watermark")
            job.log("info", "Kept your own background: only the person comes "
                            f"from the new render, and the {share * 100:.0f}% "
                            "they left behind was painted back in")
            return filled.image
        except Exception as exc:  # noqa: BLE001 — the paste alone is usable
            job.log("info", f"The space the old pose left could not be "
                            f"painted in ({exc}); it may show as a smudge")
            return out

    @staticmethod
    def _prune_pose_control(graph: dict[str, Any]) -> dict[str, Any]:
        """Drop the ControlNet branch and feed the sampler directly.

        The conditioning nodes are OPTIONAL in the sense that the graph is
        valid without them — but only if the sampler is rewired, because it
        reads its conditioning from the ControlNet node's outputs."""
        out = {k: v for k, v in graph.items()
               if k not in ("20", "21", "22", "23")}
        if "8" in out:
            out["8"]["inputs"]["positive"] = ["6", 0]
            out["8"]["inputs"]["negative"] = ["6", 1]
        return out

    def _best_pose_checkpoint(self) -> str | None:
        """An SDXL inpainting checkpoint — the ControlNet here is SDXL-only."""
        entry = self.registry.get("juggernaut-xl-inpaint")
        wanted = Path((entry.meta or {}).get("file") or "").name if entry else ""
        try:
            installed = self._image_checkpoints()
        except Exception:  # noqa: BLE001 — keep the template default
            return None
        if wanted and wanted in installed:
            return wanted
        return next((c for c in installed
                     if "xl" in c.lower() and "inpaint" in c.lower()), None)

    def _restore_face(self, job: Job, posed: Image.Image,
                      original: Image.Image, positive: str,
                      negative: str) -> Image.Image:
        """Put the original face back on a regenerated body.

        Never fatal, and never forced. A repose that turned the head produces
        a face box of a very different shape, and pasting a front-facing face
        onto a profile is worse than the face the model drew — so the two are
        compared first and the paste is skipped when they disagree."""
        src = self._face_region(original, job)
        dst = self._face_region(posed, job)
        if src is None or dst is None:
            job.log("info", "The face could not be located in both the "
                            "original and the reposed image, so the reposed "
                            "face is the model's own work rather than yours")
            return posed
        (sx0, sy0, sx1, sy1), (dx0, dy0, dx1, dy1) = src[1], dst[1]
        src_ratio = (sx1 - sx0) / max(1, sy1 - sy0)
        dst_ratio = (dx1 - dx0) / max(1, dy1 - dy0)
        if not 0.65 <= (dst_ratio / max(1e-6, src_ratio)) <= 1.55:
            job.log("info", f"The head turned too far to put the original "
                            f"face back on it (face shape {src_ratio:.2f} "
                            f"against {dst_ratio:.2f}); the reposed face is "
                            "the model's own work")
            return posed
        # SIZE as well as shape. The compositor scales the source face to the
        # target box and centres it, which is only safe while the two are
        # comparable: asked for a big repose, a 569 px face was squeezed onto
        # a 212 px one and the result was a smeared mask over the head. A
        # regenerated face is a worse likeness but it is not a defect.
        scale = (dy1 - dy0) / max(1, sy1 - sy0)
        if not 0.6 <= scale <= 1.7:
            job.log("info", f"The head changed size too much between the two "
                            f"({sy1 - sy0} px to {dy1 - dy0} px) for the "
                            "original face to be pasted back cleanly, so the "
                            "reposed face is the model's own work. Aligning "
                            "on eye landmarks rather than on the box would "
                            "lift this limit.")
            return posed
        job.log("info", "[stage] face — restoring the original face onto the "
                        "reposed body")
        try:
            return self._render_faceswap_step(job, posed, original, positive,
                                              negative, denoise=0.24)
        except Exception as exc:  # noqa: BLE001 — a pose without the face
            job.log("info", f"Face restoration did not run ({exc}); the "
                            "reposed face is the model's own work")
            return posed

    # Flux Kontext is trained on roughly 1-megapixel buckets. Measured here on
    # the same edit and seed: a 2.49 MP photo took 535 s natively, and 0.98 MP
    # took 171 s and removed the hat just as cleanly. Resolution is restored
    # afterwards by the upscaler instead of being paid for in the sampler.
    _KONTEXT_MP = 1.0

    @staticmethod
    def _fit_megapixels(image: Image.Image, megapixels: float) -> Image.Image:
        """Scale DOWN to about `megapixels`, on a multiple of 16. Never scales
        up — a small photo is already inside the bucket.

        Both sides are rounded to the NEAREST multiple rather than floored:
        flooring shrinks every side, which biases the aspect ratio and throws
        away up to 15 px of real picture on each edge."""
        w, h = image.size
        if w * h <= megapixels * 1e6:
            return image
        scale = (megapixels * 1e6 / (w * h)) ** 0.5
        return image.resize((max(256, round(w * scale / 16) * 16),
                             max(256, round(h * scale / 16) * 16)),
                            Image.Resampling.LANCZOS)

    def kontext_ready(self) -> tuple[bool, str]:
        """Can FLUX.1 Kontext run right now — template, weights, memory, and
        the GGUF node pack that loads a quantised UNet? Asked BEFORE routing,
        so an edit is never sent to an engine this machine cannot start."""
        ok, why = self._template_runnable("kontext")
        if ok and not self._pack_active("gguf"):
            return False, "the gguf node pack is not active"
        return ok, why

    # Below this whole-frame change, an instruction model returned the
    # picture untouched — it declined the request. Kontext is safety-tuned
    # far beyond this app's own policy (it also refuses content the user
    # has explicitly allowed here), and its refusals are SILENT: no error,
    # just the input handed back.
    _KONTEXT_NOOP = 0.015

    def _require_video_capable(self, job: Job) -> None:
        """WAN-class video cannot run on a DirectML backend — the missing
        torch ops KILL the ComfyUI process mid-request (measured live as a
        connection reset on an RX 6700 XT). Fail before the crash, with
        the way out spelled out: render video on a capable machine."""
        if self._render_device() != "privateuseone":
            return
        raise PermanentError(
            "This render backend runs through DirectML, which cannot "
            "execute WAN video (its torch is missing the ops — the engine "
            "crashes mid-load). Nothing was rendered. Render video on a "
            "machine with an NVIDIA or native-ROCm GPU: pick it under "
            "Render, or leave Render on auto — video jobs are only ever "
            "delegated to capable machines.")

    def _render_device(self) -> str:
        """The ACTIVE ComfyUI's render device type — "cuda",
        "privateuseone" (DirectML), "cpu", ... — cached briefly PER
        BACKEND: self.comfy is thread-locally rebound to a peer's engine
        during delegation, and a cache keyed on time alone would answer
        with the wrong machine's device. "" when unknowable; advisory —
        capability gates read it, nothing hard-fails on it."""
        base = str(getattr(self.comfy, "base_url", ""))
        cached = getattr(self, "_render_device_cache", None)
        if (cached and cached[0] == base
                and time.time() - cached[1] < 120):
            return cached[2]
        dev = ""
        try:
            stats = self.comfy.request("GET", "/system_stats")
            dev = str((stats.get("devices") or [{}])[0].get("type") or "")
        except Exception:  # noqa: BLE001 — advisory only
            dev = ""
        self._render_device_cache = (base, time.time(), dev)
        return dev

    def _kontext_blocked(self) -> bool:
        """Has Kontext been proven un-runnable on the ACTIVE backend?
        Per backend URL: a DirectML peer must not disable Kontext for
        this machine's own CUDA renders, or the other way round."""
        base = str(getattr(self.comfy, "base_url", ""))
        return base in getattr(self, "_kontext_unsupported", set())

    def _block_kontext(self) -> None:
        unsup: set[str] = getattr(self, "_kontext_unsupported", set())
        unsup.add(str(getattr(self.comfy, "base_url", "")))
        self._kontext_unsupported = unsup

    def _render_kontext_step(self, job: Job, image: Image.Image,
                             instruction: str) -> Image.Image:
        """One whole-image instruction edit through FLUX.1 Kontext.

        No mask. Kontext reads the picture and the sentence together and
        changes what the sentence names, leaving the rest alone — which is
        why it answers the removal defect directly. A masked inpaint has to
        be TOLD where the hat is, and when the mask was wrong it painted a
        fresh hat instead of removing the one that was there.

        Kontext also declines some requests this app allows, and it
        declines them silently by returning the input unchanged. That is
        DETECTED here (whole-frame change below the no-op floor) and the
        step automatically switches to the masked inpaint engine, which
        has no such opinions. The verdict is remembered per job so
        retries do not pay for another declined render."""
        memo: set[str] = getattr(job, "_kontext_declined", set())
        if instruction in memo:
            job.log("info", "[route] Kontext already declined this "
                            "instruction — going straight to the inpaint "
                            "engine")
            return self._inpaint_fallback(job, image, instruction)
        if self._kontext_blocked():
            job.log("info", "[route] Kontext cannot run on this render "
                            "backend — going straight to the masked "
                            "inpaint engine")
            return self._inpaint_fallback(job, image, instruction)
        self._require_comfy(job)
        if self._render_device() == "privateuseone":
            # DirectML: Flux-class sampling needs torch ops the frozen
            # DirectML backend does not implement (measured live on an
            # RX 6700 XT: 'Cannot access storage of OpaqueTensorImpl'
            # at KSampler). Say so once and use the engine that works.
            self._block_kontext()
            job.log("info", "[route] This machine renders through DirectML, "
                            "which cannot run Flux Kontext — using the "
                            "masked inpaint engine instead (real renders, "
                            "same request)")
            return self._inpaint_fallback(job, image, instruction)
        template = self.workflows.load("kontext")
        small = self._fit_megapixels(image, self._KONTEXT_MP)
        if small.size != image.size:
            job.log("info", f"[stage] edit — rendering at {small.size[0]}×"
                            f"{small.size[1]}, the size Kontext was trained "
                            "on (about 3x faster than native, and the model "
                            "is on-distribution there)")
        graph = build_workflow(template, {
            "prompt": instruction,
            "image": self.comfy.upload_image(small, "kontext_src"),
            "seed": int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF,
        })
        self._free_vram(job)
        self._prepare_graph(job, graph)
        try:
            out, _pid = self.comfy.run_graph(graph)
        except BackendUnavailableError as exc:
            raise self._comfy_died_midrender(job, "kontext", instruction,
                                             exc) from exc
        except WorkflowRuntimeError as exc:
            if re.search(r"OpaqueTensorImpl|NotImplementedError",
                         str(exc)):
                # The backend cannot execute this model class AT ALL —
                # permanent for Kontext on this backend, but not for the
                # request: the masked inpaint engine renders it instead,
                # and the per-backend flag stops future jobs paying for
                # another crash to learn the same fact.
                self._block_kontext()
                job.log("info", "[route] Kontext is not runnable on this "
                                "machine's render backend (missing torch "
                                "ops) — switching this step to the masked "
                                "inpaint engine")
                return self._inpaint_fallback(job, image, instruction)
            self._diagnose_and_record(job, "kontext", instruction, str(exc))
            raise PermanentError(
                f"kontext render failed: "
                f"{commit_exhausted_hint(str(exc)) or exc}") from exc
        if quality.image_change(small, out) < self._KONTEXT_NOOP:
            job.log("info", "[route] Kontext returned the picture "
                            "unchanged — it silently declines some "
                            "content, including things allowed here. "
                            "Switching this step to the masked inpaint "
                            "engine.")
            memo.add(instruction)
            # Deliberately dynamic: a per-job memo, read back via getattr.
            job._kontext_declined = memo  # type: ignore[attr-defined]
            return self._inpaint_fallback(job, image, instruction)
        return self._restore_resolution(job, out, image.size)

    def _inpaint_fallback(self, job: Job, image: Image.Image,
                          instruction: str) -> Image.Image:
        """The masked route for a step an instruction model declined.

        The same building blocks the planned inpaint path uses — the one
        mask chooser, the LLM's model pick, the refined mask — in one
        compact unit, so a declined Kontext step lands on a real
        alternative instead of an error."""
        choice = self.auto_mask(image, instruction, job=job)
        if not choice.ok or choice.mask is None:
            raise PermanentError(
                "Kontext declined this edit and no region could be "
                f"segmented for the inpaint fallback ({choice.reason}). "
                "Paint the region by hand and run it again.")
        mask = self._refine_mask(choice.mask)
        variant, checkpoint = self._choose_inpaint(job, instruction)
        enh = quality.enhance_prompt(self.llm, instruction, "inpaint")
        try:
            # Kwargs beyond the base signature; the except handles adapters
            # without them.
            result = self.inpainting.inpaint(  # type: ignore[call-arg]
                image, mask, enh["positive"],
                negative=enh["negative"],
                checkpoint=checkpoint,
                variant=variant or "modern")
        except TypeError:
            # Mock/simple adapters take the positional core only.
            result = self.inpainting.inpaint(image, mask, enh["positive"])
        return result.image

    def _restore_resolution(self, job: Job, image: Image.Image,
                            size: tuple[int, int]) -> Image.Image:
        """Put an edit rendered at model size back at the source's size.

        The model upscaler goes first because it reconstructs detail rather
        than blurring what is there; a plain resize is the fallback, so a
        missing upscale model can never fail an edit that already worked.

        The shortfall has to be MATERIAL to be worth a second render. ComfyUI's
        VAE returns sizes rounded to a multiple of 8, so an edit that was never
        downscaled still comes back a few pixels short; upscaling 4x to recover
        6 px would cost a whole render for nothing a resize cannot do."""
        if image.size == size:
            return image
        # Below ~10% short in either direction, LANCZOS is indistinguishable
        # from a model upscale and free.
        if min(size[0] / max(1, image.size[0]),
               size[1] / max(1, image.size[1])) < 1.1:
            return image.resize(size, Image.Resampling.LANCZOS)
        if not self._template_runnable("upscale")[0]:
            return image.resize(size, Image.Resampling.LANCZOS)
        try:
            bigger = self._render_template_step(job, "upscale", image, "")
            job.log("info", f"[stage] upscale — restored {size[0]}×{size[1]} "
                            "with the detail-preserving upscaler")
            return bigger.resize(size, Image.Resampling.LANCZOS)
        except Exception as exc:  # noqa: BLE001 — the edit itself is done
            job.log("info", f"The upscaler did not run ({exc}); the edit was "
                            "resized back to its original size instead")
            return image.resize(size, Image.Resampling.LANCZOS)

    def _render_template_step(self, job: Job, task: str, image: Image.Image,
                              positive: str, negative: str = "",
                              denoise: float | None = None,
                              checkpoint: str | None = None,
                              extra: dict[str, Any] | None = None
                              ) -> Image.Image:
        """Run one non-inpaint edit step (img2img / outpaint) through its
        validated template on an in-memory image; returns the result image.
        `extra` carries template-specific slots the caller computed (the
        outpaint direction paddings) — only declared slots are passed."""
        self._require_comfy(job)
        template = self.workflows.load(task)
        image_name = self.comfy.upload_image(image, "edit_src")
        # Only pass parameters the template actually declares — the faithful
        # model upscaler, for example, takes ONLY an image (no prompt, no
        # seed), and build_workflow rejects unknown keys.
        tparams = template.get("parameters", {})
        params: dict[str, Any] = {"image": image_name}
        if "prompt" in tparams:
            params["prompt"] = positive
        if "seed" in tparams:
            params["seed"] = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF
        if negative and "negative" in tparams:
            params["negative"] = negative
        if task == "img2img" and denoise is not None and "denoise" in tparams:
            params["denoise"] = denoise
        if checkpoint and "checkpoint" in tparams:
            params["checkpoint"] = checkpoint
        for key, value in (extra or {}).items():
            if key in tparams:
                params[key] = value
        try:
            graph = build_workflow(template, params)
        except WorkflowValidationError as exc:
            # A template/parameter mismatch can never succeed on retry.
            raise PermanentError(f"{task} template error: {exc}") from exc
        self._free_vram(job)
        self._prepare_graph(job, graph)
        try:
            try:
                out, _pid = self.comfy.run_graph(graph)
                return out
            except WorkflowRuntimeError as first:
                tiled = self._miopen_tiled_retry(job, graph, first)
                if tiled is None:
                    raise
                out, _pid = self.comfy.run_graph(tiled)
                return out
        except BackendUnavailableError as exc:
            raise self._comfy_died_midrender(job, task, positive, exc) from exc
        except WorkflowRuntimeError as exc:
            self._diagnose_and_record(job, task, positive, str(exc))
            raise PermanentError(
                f"{task} render failed: "
                f"{commit_exhausted_hint(str(exc)) or self._miopen_hint(exc) or exc}") from exc

    def _live_object_info(self, engine: Any | None = None) -> dict | None:
        """The live /object_info schema, cached ~5 min PER ENGINE. None when
        that ComfyUI is unreachable — the schema check is advisory, never a
        blocker.

        Keyed by the engine's base_url because during delegation self.comfy
        is a peer's proxy: one shared cache served the LOCAL machine's node
        list while validating a graph bound for the PEER (measured live —
        a BiRefNetRMBG graph passed the gate onto a machine without the
        pack). Callers that ask about THIS machine pass self._comfy_main.

        Mock mode means OFFLINE: a ComfyUI answering on this box belongs to
        some other setup (measured live: the dev machine's real instance
        made every mocked test job spend seconds probing it, and slower
        answers under load broke the whole test_api module — the long-
        misdiagnosed 'load flakes')."""
        if self.settings.inpaint_backend == "mock":
            return None
        comfy = engine if engine is not None else self.comfy
        key = str(getattr(comfy, "base_url", "local"))
        now = time.time()
        hit = self._object_info_cache.get(key)
        if hit and now - hit[0] < 300:
            return hit[1]
        try:
            info = comfy.object_info()
        except Exception:  # noqa: BLE001 — ComfyUI down: skip the check
            return None
        if isinstance(info, dict) and info:
            self._object_info_cache[key] = (now, info)
            return info
        return None

    def model_usage_guide(self, ready_only: bool = False) -> str:
        """'When to use which model' — curated usage lines + live readiness,
        for every prompt where the LLM picks models."""
        lines = []
        for m in self.registry.list():
            usage = MODEL_USAGE.get(m.name)
            if not usage:
                continue
            ready = self.registry.is_ready(m.name)
            if ready_only and not ready:
                continue
            state = "ready" if ready else "not downloaded"
            lines.append(f"- {m.name} ({state}): {usage}")
        return "Model guide (when to use which):\n" + "\n".join(lines)

    # Files that live in ComfyUI's checkpoints folder but are NOT prompt-
    # renderable image checkpoints (no CLIP/text encoder inside): 3D mesh
    # DiTs, image-to-video latents, identity encoders. Loading one through
    # CheckpointLoaderSimple yields clip=None and CLIPTextEncode dies -
    # measured live when the adherence ladder's model rung picked a
    # Hunyuan3D mesh model and a garbage render nearly shipped.
    _NON_IMAGE_CKPT = re.compile(
        r"hunyuan3d|sv3d|stable[_-]?video|(^|[^a-z])svd([^a-z]|$)"
        r"|photomaker|animatediff", re.IGNORECASE)

    def _image_checkpoints(self) -> list[str]:
        """ComfyUI's loadable checkpoints MINUS everything that cannot take
        a text prompt - the only list any model choice may draw from."""
        try:
            ckpts = self.comfy.installed_checkpoints()
        except Exception:  # noqa: BLE001 - callers all handle empty
            return []
        return [c for c in ckpts if not self._NON_IMAGE_CKPT.search(c)]

    def workflow_context(self) -> str | None:
        """Live inventory for the LLM planner; None when ComfyUI is down."""
        try:
            ckpts = self._image_checkpoints()
        except Exception:
            return None
        if not ckpts:
            return None
        return ("Available checkpoints — set ckpt_name to EXACTLY one of "
                "these: " + ", ".join(ckpts) + "\n"
                + self.model_usage_guide(ready_only=True)
                + "\nCompatibility is HARD: SDXL LoRAs/ControlNets only "
                "with SDXL checkpoints, SD15 only with SD15; cfg-1 models "
                "(Z-Image, Kontext, speed LoRAs) use ConditioningZeroOut, "
                "never a text negative.")

    def _download_progress(self, job: Job, name: str):
        def progress(done: int, total: int | None) -> None:
            # The one place a transfer visits on every chunk — so it is
            # where a cancel takes effect. Without this, cancelling the job
            # marked it cancelled and the download kept pulling gigabytes
            # (measured live: +108 MB in the 18s after the cancel).
            if job.cancel_requested:
                raise TransientError(
                    f"Download of '{name}' stopped — the job was cancelled.")
            if total and done % (total // 10 or 1) < ModelDownloader.CHUNK:
                job.log("info", f"Downloading {name}: {done * 100 // total}%")
        return progress

    _GATED_RE = re.compile(r"40[13]|unauthorized|forbidden|gated", re.IGNORECASE)

    def _find_public_mirror(self, model: ModelInfo,
                            job: Job) -> tuple[str, str, str] | None:
        """Locate a public mirror of a gated model file on the hub.

        A candidate must be non-gated, carry the same filename, publish an
        LFS sha256, and — when we know the expected size — match it exactly.
        Returns (repo_id, filename, sha256) or None.
        """
        target = Path((model.meta or {}).get("file")
                      or _url_filename(model.url)).name
        if not target:
            return None
        query = target.rsplit(".", 1)[0].split("_")[0]  # "sv3d_u" -> "sv3d"
        expected_size = (model.meta or {}).get("size_bytes")
        job.log("info", f"[search] Hugging Face: public mirror of '{target}'")
        for repo in self.model_search.search(query, limit=10):
            if repo.gated:
                continue
            try:
                files = self.model_search.list_weight_files(repo.repo_id)
            except Exception:  # noqa: BLE001 — try the next candidate
                continue
            for f in files:
                if (Path(f.filename).name == target and f.sha256
                        and (not expected_size or f.size_bytes == expected_size)):
                    verdict = self.trust.judge(Evidence(
                        repo_id=repo.repo_id, filename=f.filename,
                        url=f"https://huggingface.co/{repo.repo_id}",
                        size_bytes=f.size_bytes, sha256=f.sha256,
                        downloads=repo.downloads, likes=repo.likes,
                        gated=repo.gated))
                    job.log("info", f"Trust check ({verdict.judged_by}) for "
                                    f"mirror {repo.repo_id}: "
                                    f"{'PROCEED' if verdict.proceed else 'REJECTED'}"
                                    f" — {verdict.reason}")
                    if verdict.proceed:
                        return repo.repo_id, f.filename, f.sha256
        return None

    def _ensure_model(self, name: str, job: Job, *,
                      requested: bool = False) -> None:
        """Download a registry model if missing, backfilling its sha256 from
        the hub's LFS metadata first so the download stays verified. If the
        source turns out to be gated (401/403), a checksum-published public
        mirror is located automatically and the download retried.

        `requested` marks an explicit ask (the model_download job type) as
        opposed to a render path discovering a missing dependency — only
        the latter is what the auto_install setting governs."""
        if self.registry.is_ready(name):
            return
        model = self.registry.get(name)
        if model is None:
            raise PermanentError(f"Model '{name}' is not in the registry.")
        # This IS the behavior the auto_install setting names ("download
        # missing required models when a job needs them") — it was enforced
        # for checkpoints below but not here, so a required mesh model
        # started a multi-GB download on a build that had said no. A
        # download the user ASKED for (a Models-page click arrives as a
        # model_download job) is not auto-install and stays allowed.
        if not requested and not self.settings.auto_install:
            raise PermanentError(
                f"'{name}' is not downloaded and auto-install is off — "
                "download it from the Models page and run this again.")
        if not model.sha256 and (model.meta or {}).get("repo"):
            try:
                # meta is non-None: the repo key above came out of it.
                for f in self.model_search.list_weight_files(
                        model.meta["repo"]):  # type: ignore[index]
                    if (f.filename == model.meta.get("file")  # type: ignore[union-attr]
                            and f.sha256):
                        model.sha256 = f.sha256
                        self.registry.register(model)
                        job.log("info", f"Verified checksum for '{name}' "
                                        "fetched from the hub.")
                        break
            except Exception as exc:  # noqa: BLE001 — checksum stays visible-missing
                job.log("error", f"Could not fetch checksum for '{name}': {exc}")
        job.log("info", f"Downloading '{name}' "
                        f"({model.purpose.split('(')[0].strip()})…")
        try:
            self.downloader.download(name, self._download_progress(job, name))
        except DownloadError as exc:
            if not self._GATED_RE.search(str(exc)):
                raise
            if "civitai.com" in (model.url or ""):
                # Not a license gate — civitai wants an account token; the
                # DownloadError already carries the how-to-fix message.
                raise PermanentError(str(exc)) from exc
            job.log("error", f"Source for '{name}' is gated/denied: {exc}")
            mirror = self._find_public_mirror(model, job)
            if mirror is None:
                raise PermanentError(
                    f"'{name}' is license-gated and no verified public mirror "
                    "was found. Accept the license on the official page and "
                    "stage it via Models → Find models online.") from exc
            repo_id, filename, sha = mirror
            job.log("info", f"Using verified mirror {repo_id}/{filename} "
                            f"(sha256 {sha[:12]}…)")
            model.url = (f"https://huggingface.co/{repo_id}/resolve/main/"
                         f"{filename}")
            model.sha256 = sha
            model.meta = {**(model.meta or {}), "repo": repo_id,
                          "file": filename, "mirror": True}
            self.registry.register(model)
            self.registry.set_status(name, "not_downloaded")
            self.downloader.download(name, self._download_progress(job, name))
        job.log("info", f"Model '{name}' ready.")

    def _ensure_checkpoint(self, job: Job) -> list[str]:
        """Return ComfyUI's loadable checkpoints; auto-install one if none."""
        ckpts = self._image_checkpoints()
        if ckpts:
            return ckpts
        if not self.settings.auto_install:
            raise PermanentError(
                "ComfyUI has no checkpoints installed and auto-install is "
                "disabled — download one from the Models page.")
        registry_ckpts = [m for m in self.registry.list()
                          if "checkpoint" in (m.purpose or "").lower()]
        ready = [m for m in registry_ckpts if self.registry.is_ready(m.name)]
        if ready:
            # Already on disk — ComfyUI just can't see the folder (or needs a
            # rescan). Don't re-download 4 GB; say what's actually wrong.
            raise PermanentError(
                f"'{ready[0].name}' is already downloaded, but ComfyUI cannot "
                "see it. Point ComfyUI at PromptForge's models folder (the "
                "launcher writes extra_model_paths.yaml) and restart ComfyUI.")
        candidates = [m for m in registry_ckpts if m.url]
        if not candidates:
            raise PermanentError(
                "ComfyUI has no checkpoints and the registry has no "
                "downloadable checkpoint to auto-install. Stage one via the "
                "Models page (Find models online).")
        # Prefer safetensors sources over pickle formats.
        candidates.sort(key=lambda m: 0 if ".safetensors" in (m.url or "") else 1)
        name = candidates[0].name
        job.log("info", f"No checkpoints in ComfyUI — auto-installing "
                        f"'{name}' (checksum-verified; this can take a while).")

        def progress(done: int, total: int | None) -> None:
            if total and done % (total // 10 or 1) < ModelDownloader.CHUNK:
                job.log("info", f"Downloading {name}: {done * 100 // total}%")

        self.downloader.download(name, progress)
        job.log("info", f"Model '{name}' downloaded.")
        ckpts = self._image_checkpoints()
        if not ckpts:
            raise PermanentError(
                f"'{name}' was downloaded, but ComfyUI cannot see it. Point "
                "ComfyUI at PromptForge's models folder (the launcher writes "
                "extra_model_paths.yaml) and restart ComfyUI.")
        return ckpts

    @staticmethod
    def _graph_summary(graph: dict[str, Any]) -> str:
        """One readable line describing what the LLM decided to build —
        surfaced in the job log so the user sees the model's actual plan."""
        def order(nid: str) -> int:
            return int(nid) if str(nid).isdigit() else 10**9
        parts = []
        for nid in sorted(graph, key=order):
            node = graph[nid]
            ctype = node.get("class_type", "?")
            ins = node.get("inputs", {})
            if ctype == "KSampler":
                parts.append(f"KSampler({ins.get('steps')} steps, "
                             f"cfg {ins.get('cfg')}, {ins.get('sampler_name')}"
                             f"/{ins.get('scheduler')})")
            elif ctype == "CheckpointLoaderSimple":
                parts.append(f"Checkpoint[{ins.get('ckpt_name')}]")
            elif ctype == "EmptyLatentImage":
                parts.append(f"Canvas {ins.get('width')}×{ins.get('height')}")
            elif ctype == "CLIPTextEncode":
                text = str(ins.get("text", ""))
                parts.append(f'Text("{text[:48]}{"…" if len(text) > 48 else ""}")')
            else:
                parts.append(ctype)
        return " → ".join(parts)

    @staticmethod
    def _recipe_facts(graph: dict[str, Any]) -> dict[str, Any]:
        """The key generation parameters actually used, read from the FINAL
        executed graph (post-repairs, post-clamps) — for the recipe card."""
        facts: dict[str, Any] = {}
        loras = controlnets = 0
        for node in graph.values():
            ctype = str(node.get("class_type", ""))
            ins = node.get("inputs", {})
            if ctype == "CheckpointLoaderSimple":
                facts["checkpoint"] = ins.get("ckpt_name")
            elif ctype in ("UNETLoader", "UnetLoaderGGUF"):
                # A quantised UNet is still the model that made the picture;
                # without this the recipe card and the plan log read "model: ?"
                # for every Kontext edit.
                facts["checkpoint"] = ins.get("unet_name")
            elif ctype == "KSampler":
                facts.update(sampler=ins.get("sampler_name"),
                             scheduler=ins.get("scheduler"),
                             steps=ins.get("steps"), cfg=ins.get("cfg"),
                             seed=ins.get("seed"))
                if ins.get("denoise") not in (None, 1, 1.0):
                    facts["denoise"] = ins.get("denoise")
            elif ctype == "EmptyLatentImage":
                facts["resolution"] = f"{ins.get('width')}×{ins.get('height')}"
            elif ctype.startswith("LoraLoader"):
                loras += 1
            elif ctype.startswith("ControlNet"):
                controlnets += 1
        if loras:
            facts["loras"] = loras
        if controlnets:
            facts["controlnets"] = controlnets
        return facts

    @staticmethod
    def _recipe_steps(job: Job) -> list[dict[str, str]]:
        """The stage sequence this job actually went through, timestamped."""
        steps: list[dict[str, str]] = []
        for entry in job.logs:
            m = re.match(r"\[stage\] (\w+)(?:\s*—\s*(.*))?", entry["msg"])
            if m:
                steps.append({"t": entry["t"][11:19], "step": m.group(1),
                              "detail": (m.group(2) or "")[:90]})
        return steps

    @staticmethod
    def _inpaint_rank(name: str) -> tuple:
        """Sort key: best inpaint checkpoint first. Dedicated inpaint models
        beat regular ones; modern SDXL beats SD1.5; well-known photoreal
        community models beat base releases; the legacy sd-v1-5-inpainting
        default ranks last among inpaint models."""
        n = name.lower()
        return (
            0 if "inpaint" in n else 1,
            0 if ("xl" in n or "sdxl" in n) else 1,
            0 if any(k in n for k in ("juggernaut", "epicrealism", "realvis",
                                      "realistic", "photo")) else 1,
            1 if n.startswith(("sd-v1-5", "sd15", "v1-5")) else 0,
            n,
        )

    # Best registered inpaint checkpoints, small-and-quick first.
    _BETTER_INPAINT = ("epicrealism-inpaint", "juggernaut-xl-inpaint")

    def _stage_better_inpaint(self, job: Job, installed: list[str]) -> None:
        """Self-improvement: when no modern photoreal inpaint checkpoint is
        installed yet, queue the best registered ones for download in the
        background (visible in the Queue). The current edit continues with
        what's installed; the NEXT edit gets the better model."""
        if not self.settings.auto_install:
            return
        have_modern = any("inpaint" in c.lower()
                          and any(k in c.lower() for k in
                                  ("juggernaut", "epicrealism", "realvis"))
                          for c in installed)
        if have_modern:
            return
        for name in self._BETTER_INPAINT:
            if name in self._inpaint_staged or self.registry.is_ready(name):
                continue
            self._inpaint_staged.add(name)
            self.queue.enqueue("model_download", {"model": name})
            job.log("info", f"[llm] inpaint setup: staging better inpaint "
                            f"model '{name}' in the background — future "
                            "edits will use it automatically")

    # Extra VRAM a retry candidate must leave free. A model whose declared
    # need EQUALS the card is a guaranteed thrash: observed live at 7753 MB
    # of 8188 MB with the GPU pinned at 100% for minutes (D8).
    _VRAM_HEADROOM_GB = 0.5

    def _checkpoint_fits_retry(self, filename: str) -> tuple[bool, str]:
        """Whether an escalation candidate can actually run HERE, with
        headroom. Looked up in the registry by name, stem, and file path;
        models the registry knows nothing about pass — the guard is against
        KNOWN oversubscription, not a ban on the unknown."""
        stem = Path(filename).stem
        m = self.registry.get(filename) or self.registry.get(stem)
        if m is None:
            try:
                m = next((c for c in self.registry.list()
                          if c.path and Path(c.path).name == filename), None)
            except Exception:  # noqa: BLE001 — inventory is best-effort
                m = None
        if m is None:
            return True, ""
        need = float((m.meta or {}).get("min_vram_gb") or m.vram_gb or 0)
        have = float(self.hardware.vram_gb or 0)
        if need and have < need + self._VRAM_HEADROOM_GB:
            return False, (f"declares {need:g} GB VRAM against {have:g} GB "
                           "on this card — no headroom to run it")
        return True, ""

    def _next_edit_recipe(self, task: str, tried_ckpts: set[str | None],
                          tried_variants: set[str | None],
                          job: Job | None = None
                          ) -> tuple[str | None, str | None]:
        """The escalation rung for an edit retry: the next (checkpoint,
        technique) pair that has NOT been tried yet AND fits this machine.

        Models come first (a different model is the biggest single change an
        edit can make), and when every installed model has had a turn the
        TECHNIQUE changes instead — modern soft-inpaint, universal latent
        mask, or hi-res crop&stitch are genuinely different workflows, not
        different seeds. (None, None) when there is nothing new left.

        The hardware check is not optional politeness: round 2 used to pick
        the next checkpoint by inpaint rank alone and selected a 6.94 GB
        model declaring 8.0 GB on an 8.0 GB card (D8) — a retry that costs
        more than the render it retries and thrashes the whole machine."""
        try:
            installed = sorted(self._image_checkpoints(),
                               key=self._inpaint_rank)
        except Exception:  # noqa: BLE001 — no inventory: keep the recipe
            installed = []
        for name in installed:
            if name in tried_ckpts:
                continue
            fits, why = self._checkpoint_fits_retry(name)
            if not fits:
                if job is not None:
                    job.log("info", f"Escalation skips {name}: {why}")
                continue
            if task != "inpaint":
                return name, None
            return name, ("modern" if "inpaint" in name.lower()
                          else "universal")
        if task == "inpaint":
            for variant in ("universal", "modern", "hires"):
                if variant not in tried_variants:
                    return None, variant
        return None, None

    def _scene_graph(self, job: Job, asset_id: str, image: Image.Image,
                     real: bool) -> dict[str, Any]:
        """The asset's scene graph, built once (vision analysis) and cached.
        The Image Understanding Engine every later step reads."""
        cached = self._scene_cache.get(asset_id)
        if cached is not None:
            return cached
        if not real or self.critic is None:
            graph = scene_module.build(image, None)
        else:
            job.log("info", "[stage] understand — analyzing the scene "
                            "(objects, lighting, perspective)")
            graph = scene_module.build(image, self.critic, self.segmentation)
            objs = ", ".join(o["name"] for o in graph.get("objects", []))
            if objs:
                job.log("info", f"[llm] scene objects: {objs[:140]}")
        self._scene_cache[asset_id] = graph
        return graph

    def _handle_model_research(self, job: Job) -> dict[str, Any]:
        """Research one model's strengths online + rate it, into the
        knowledge file the planner reads on every prompt."""
        fname = job.payload["file"]
        job.log("info", f"[stage] research — learning what '{fname}' "
                        "performs best at")
        entry = self.model_intel.research(
            fname, self.llm, log=lambda m: job.log("info", m))
        if entry is None:
            job.log("info", "Research unavailable right now (LLM offline or "
                            "nothing usable found) — it will be retried when "
                            "this model is next considered")
            self._intel_queued.discard(fname)  # allow a later retry
            return {"file": fname, "researched": False}
        job.log("info", f"[llm] model notes saved: quality "
                        f"{entry['quality']}/10 — best at "
                        f"{entry['best_at'][:110]}")
        return {"file": fname, "researched": True,
                "quality": entry["quality"]}

    # Folders whose files the planner ROUTES BETWEEN — those are worth
    # capability research. Encoders/VAEs/upscalers are plumbing, not a
    # choice, and researching them would only pollute the knowledge file.
    _RESEARCH_FOLDERS = frozenset({"checkpoints", "diffusion_models",
                                   "loras"})

    def _research_new_disk_models(self, cap: int = 4) -> None:
        """Detect models that appeared on DISK without a download job and
        queue capability research for them — a hand-copied checkpoint gets
        the same treatment as a downloaded one. Bounded per sweep so a
        freshly synced library does not flood the queue."""
        if not self.settings.auto_install:
            return
        names: list[str] = []
        root = self.settings.data_dir / "models"
        for folder in sorted(self._RESEARCH_FOLDERS):
            d = root / folder
            if not d.is_dir():
                continue
            for f in d.iterdir():
                if f.is_file() and f.suffix.lower() in (
                        ".safetensors", ".ckpt", ".gguf"):
                    names.append(f.name)
        queued = 0
        for fname in self.model_intel.missing(names):
            if fname in self._intel_queued:
                continue
            self._intel_queued.add(fname)
            self.queue.enqueue("model_research", {"file": fname})
            self.events.log("info", f"New model detected on disk: '{fname}' "
                                    "— researching what it does best")
            queued += 1
            if queued >= cap:
                break

    def _queue_model_research(self, job: Job, ckpts: list[str]) -> None:
        """Queue background research for installed checkpoints that have no
        capability notes yet (once per session each; jobs show in the Queue)."""
        if not self.settings.auto_install:
            return
        for f in self.model_intel.missing(ckpts):
            if f in self._intel_queued:
                continue
            self._intel_queued.add(f)
            self.queue.enqueue("model_research", {"file": f})
            job.log("info", f"[llm] queued online research for '{f}' — its "
                            "strengths will inform future model choices")

    def _choose_inpaint(self, job: Job,
                        instruction: str) -> tuple[str | None, str | None]:
        """Let the LLM pick the inpaint model for a step. The scout may keep
        the default, pick any installed checkpoint, or search the hubs
        (civitai/Hugging Face) and download a better one. The technique
        follows the model: dedicated inpaint checkpoints run the modern
        soft-inpaint template; any other checkpoint runs the universal
        latent-mask template (which lets ANY model inpaint).
        Returns (variant, checkpoint) — (None, None) keeps template defaults."""
        try:
            ckpts = self._image_checkpoints()
        except Exception:  # noqa: BLE001 — ComfyUI hiccup: template default
            ckpts = []
        if not ckpts:
            return None, None
        # Quality-ranked, best first, so the scout's no-LLM fallback (first
        # installed) is already the strongest inpaint default — modern
        # photoreal SDXL/community inpaint models beat the legacy
        # sd-v1-5-inpainting base that used to win by being alphabetical.
        ckpts = sorted(ckpts, key=self._inpaint_rank)
        self._stage_better_inpaint(job, ckpts)
        self._queue_model_research(job, ckpts)
        intel = self.model_intel.summary(ckpts)
        if intel:
            job.log("info", "[llm] consulting the model knowledge file for "
                            "this prompt")
        try:
            decision = self.scout.choose(
                f"inpainting edit: {instruction}\n"
                + (intel + "\n" if intel else "")
                + "Preference: pick the checkpoint whose known strengths "
                "match THIS edit (see model knowledge above) — modern "
                "community inpaint models (Juggernaut XL inpaint, epiCRealism "
                "inpainting) beat the legacy sd-v1-5-inpainting base.",
                "inpaint", ckpts,
                allow_download=self.settings.auto_install,
                progress=self._download_progress(job, "inpaint model"),
                log=self._scout_log(job))
        except Exception as exc:  # noqa: BLE001 — choice is best-effort
            job.log("info", f"[llm] inpaint setup: scout failed ({exc}); "
                            "keeping the template default")
            return None, None
        ckpt = decision.checkpoint
        variant = "modern" if "inpaint" in ckpt.lower() else "universal"
        how = ("modern soft inpaint (differential diffusion)"
               if variant == "modern"
               else "universal latent-mask (any checkpoint can inpaint)")
        # No [llm] prefix: the model choice must be visible in Behind the
        # Scenes (which filters [llm] reasoning lines), not just the Studio.
        job.log("info", f"Inpaint model: '{ckpt}' via {how} — "
                        f"{decision.note[:120]}")
        return variant, ckpt

    @staticmethod
    def _scout_log(job: Job):
        """Job-scoped scout narration; "[search]" lines pass through
        unprefixed so the GUI renders them as live search activity."""
        def emit(m: str) -> None:
            job.log("info", m if m.startswith("[search]")
                    else f"[llm] scout: {m}")
        return emit

    @staticmethod
    def _planner_log(job: Job):
        return lambda m: job.log("info", f"[llm] planner: {m}")

    def _triage(self, job: Job, task: str, prompt: str) -> dict[str, Any] | None:
        """Show the LLM the prompt + workflow menu + model registry so it can
        choose the right workflow and pre-fetch the models it needs, before
        any planning. Advisory and fully fail-safe (skips on any error).

        Custom ComfyUI *plugins* are deliberately NOT auto-installed — the
        node-type allowlist would reject unknown nodes anyway, and silently
        installing third-party code is a stability/security risk. Only
        registry-listed, checksum-verified models are fetched here."""
        try:
            templates = self.workflows.list_all()
        except Exception:  # noqa: BLE001 — triage is optional
            templates = []
        # Only offer templates this machine can actually run — a template
        # whose models need more VRAM/RAM than we have is hidden here so the
        # LLM never routes to it (it would OOM); the SAME template stays
        # available on a bigger machine.
        runnable = [t for t in templates
                    if self._models_fit_machine(
                        t.get("required_models") or [])[0]]
        menu = "\n".join(
            f"- {t['template']} (task {t.get('task', t['template'])}): "
            f"{(t.get('description') or '')[:110]} "
            f"[models: {', '.join(t.get('required_models') or []) or 'none'}]"
            for t in runnable)
        models = [{"name": m.name, "purpose": (m.purpose or "")[:70],
                   "ready": self.registry.is_ready(m.name)}
                  for m in self.registry.list() if m.url]
        system = (
            "You route an image/video request to the best ComfyUI workflow and "
            "list any models that must be downloaded first. Reply with ONLY "
            'JSON: {"workflow": "<template name from the menu, or empty for a '
            'custom plan>", "needed_models": ["<registry model name>", ...], '
            '"reason": "<short>"}. Name models ONLY from the provided registry '
            "list; prefer ones already ready. Do not invent names. "
            "Choosing the BEST tool: generate_zimage whenever the image must "
            "contain READABLE TEXT (signs, posters, labels) or the user asks "
            "for fast photoreal; generate_draft when they want a quick "
            "draft/preview to iterate on; kontext for whole-image "
            "instruction edits ('make it night'); img2img_canny to restyle "
            "while keeping the exact structure. Never pick a workflow whose "
            "required node pack is not active — fall back and mention it.")
        packs = [{"name": p["name"], "status": p["status"],
                  "unlocks": p["unlocks"]} for p in self.node_pack_report()]
        ask = (f"Task: {task}\nPrompt: {prompt}\n\nWorkflow menu:\n{menu}\n\n"
               f"Registry models (name/ready): {json.dumps(models)}\n"
               f"{self.model_usage_guide()}\n"
               f"Node packs (user installs them in the Models tab): "
               f"{json.dumps(packs)}")
        try:
            reply = self.llm.complete(system, ask, max_tokens=400)
            data = json.loads(re.sub(r"^```(?:json)?|```$", "",
                                     reply.text.strip(), flags=re.M).strip())
        except (LLMError, json.JSONDecodeError, AttributeError, TypeError,
                ValueError) as exc:  # LLMError covers all LLM failures
            job.log("info", f"[llm] triage skipped ({type(exc).__name__})")
            return None
        workflow = str(data.get("workflow", "")).strip()
        # A draft template is deliberately lower quality (4 steps, cfg 1). It
        # is the right answer ONLY when the user asked for something quick —
        # a 7B router picks it for "fast photorealistic" in its own reasoning
        # and quietly hands back a preview when a finished image was wanted.
        if (re.search(r"draft|preview|fast", workflow, re.IGNORECASE)
                and not self._WANTS_DRAFT.search(prompt)):
            job.log("info", f"[llm] triage picked the '{workflow}' draft "
                            "template but the request never asked for a "
                            "quick draft — using a full-quality workflow "
                            "instead")
            workflow = ""
        names = {m["name"] for m in models}
        needed = [n for n in (data.get("needed_models") or []) if n in names]
        job.log("info", f"[llm] triage: workflow "
                        f"'{workflow or 'custom plan'}', extra models "
                        f"{needed or 'none'} — {str(data.get('reason', ''))[:160]}")
        if needed and self.settings.auto_install:
            for n in needed:
                if not self.registry.is_ready(n):
                    job.log("info", f"[stage] models — fetching '{n}' "
                                    "(chosen by triage)")
                    try:
                        self._ensure_model(n, job)
                    except Exception as exc:  # noqa: BLE001 — keep going
                        job.log("error", f"Could not fetch '{n}': {exc}")
        return {"workflow": workflow or None, "fetched_models": needed,
                "reason": str(data.get("reason", ""))[:160]}

    # The user asking for speed over quality — the only reason to route a
    # request to a draft/preview template.
    _WANTS_DRAFT = re.compile(
        r"\b(draft|preview|rough|quick(ly)?|fast|sketch|thumbnail|"
        r"low[\s-]?(quality|res)|test render|just show me)\b", re.IGNORECASE)

    def _models_fit_machine(self, names: list[str]) -> tuple[bool, str]:
        """Can this machine actually RUN these models? A model may declare
        meta.min_vram_gb / min_ram_gb (the heavy models do). Returns
        (fits, reason). Keeps the LLM from routing to a model that will OOM
        here while the SAME model auto-stages and runs on a bigger machine."""
        for n in names:
            m = self.registry.get(n)
            meta = (m.meta or {}) if m else {}
            # ModelInfo.vram_gb is the field most entries actually fill in —
            # flux-kontext-q4 declares 7.0 there and nothing in meta, so
            # reading meta alone made this whole gate a no-op for it and a
            # 6 GB card would have been routed to a model it cannot load.
            need_v = float(meta.get("min_vram_gb")
                           or getattr(m, "vram_gb", 0) or 0)
            need_r = float(meta.get("min_ram_gb") or 0)
            if need_v and self.hardware.vram_gb < need_v:
                return False, f"{n} needs {need_v:g} GB VRAM"
            if need_r and self.hardware.ram_gb < need_r:
                return False, f"{n} needs {need_r:g} GB RAM"
        return True, ""

    @staticmethod
    def _boost_graph(graph: dict[str, Any],
                     boost: dict[str, float]) -> dict[str, Any]:
        """Lean harder on the prompt: raise cfg/steps (and denoise, where the
        sampler has one) on every sampler in a graph. Used by the adherence
        ladder, so a retry that costs a full render also pushes the model
        toward the words it ignored.

        cfg-1 models (Z-Image, Kontext, the speed LoRAs) are left alone — for
        those, raising guidance burns the image instead of improving it."""
        if not boost:
            return graph
        out = copy.deepcopy(graph)
        for node in out.values():
            if node.get("class_type") != "KSampler":
                continue
            inputs = node.get("inputs", {})
            cfg = inputs.get("cfg")
            distilled = isinstance(cfg, int | float) and cfg <= 1.5
            if isinstance(cfg, int | float) and not distilled and "cfg" in boost:
                inputs["cfg"] = round(min(12.0, cfg + boost["cfg"]), 2)
            steps = inputs.get("steps")
            # A distilled model's step count is part of the model, not a
            # quality dial: 4 steps means 4 steps, and "trying harder" by
            # running 5 makes it worse, not better.
            if isinstance(steps, int) and not distilled and "steps" in boost:
                inputs["steps"] = int(min(60, round(steps * boost["steps"])))
            den = inputs.get("denoise")
            # denoise 1.0 is a from-scratch render — there is nothing to raise.
            if (isinstance(den, int | float) and den < 0.99
                    and "denoise" in boost):
                inputs["denoise"] = round(min(0.95, den + boost["denoise"]), 2)
        return out

    def _template_workflow(self, job: Job, triage: dict[str, Any] | None,
                           prompt_used: str, image_name: str | None,
                           seed: int | None = None,
                           checkpoint: str | None = None,
                           boost: dict[str, float] | None = None):
        """When triage picked a VALIDATED template whose models are all
        ready, build that template's tuned graph directly — this is how the
        LLM's best-workflow choice actually takes effect (e.g. Z-Image for
        readable text instead of a custom SDXL graph). Returns a
        GeneratedWorkflow, or None to fall back to custom LLM design."""
        from .workflow_ai import GeneratedWorkflow
        name = (triage or {}).get("workflow", "")
        if not name:
            return None
        try:
            template = self.workflows.load_named(name)
        except Exception:  # noqa: BLE001 — no such template → custom path
            return None
        required = template.get("required_models", [])
        fits, why = self._models_fit_machine(required)
        if not fits:
            job.log("info", f"Template '{name}' can't run on this machine "
                            f"({why}) — designing a custom workflow instead")
            return None
        missing = [m for m in required if not self.registry.is_ready(m)]
        if missing:
            job.log("info", f"Template '{name}' needs {', '.join(missing)} "
                            "— not ready, designing a custom workflow instead")
            return None
        params = template.get("parameters", {})
        if "image" in params:
            if not image_name:
                return None  # image template but no input image → fall back
            values: dict[str, Any] = {"image": image_name}
        else:
            values = {}
        if "prompt" in params:
            values["prompt"] = prompt_used
        if "seed" in params:
            values["seed"] = (seed if seed is not None
                              else int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF)
        # The adherence ladder's "different model" rung: only templates that
        # declare a checkpoint slot can swap one, and a template with a fixed
        # model (Z-Image, Kontext) simply keeps its own.
        if checkpoint and "checkpoint" in params:
            values["checkpoint"] = checkpoint
        try:
            graph = build_workflow(template, values)
        except WorkflowValidationError as exc:
            job.log("info", f"Template '{name}' could not be filled ({exc}) "
                            "— designing a custom workflow instead")
            return None
        if boost:
            graph = self._boost_graph(graph, boost)
        return GeneratedWorkflow(
            graph=graph, task=template.get("task", "generate"),
            provenance={"source": "template", "model": name, "attempts": 1,
                        "checkpoint": checkpoint})

    def _plan_context(self, job: Job, task: str, prompt: str, chosen: str,
                      ckpts: list[str], image_context: str) -> str:
        """Everything the planner LLM needs to design a graph for THIS render:
        which checkpoint to use, what else is installed, the input image, what
        this machine can survive, the task's guide + a validated example, and
        the lessons of past renders. Rebuilt (not reused) when the adherence
        ladder changes model, so 'use checkpoint X' always names the model the
        rung actually decided on."""
        budget = render_budget(self.hardware)
        context = (f"Use checkpoint '{chosen}' — set ckpt_name to EXACTLY "
                   f"this. (All installed: {', '.join(ckpts)})"
                   + image_context
                   + f"\nMachine: {self.hardware.gpu_name or 'CPU only'}, "
                     f"{self.hardware.vram_gb:g} GB VRAM "
                     f"({self.hardware.tier} tier). USE IT FULLY: canvas "
                     f"up to {budget['max_side']} px per side "
                     f"(≤{budget['max_pixels']} px²), steps 25–"
                     f"{budget['max_steps']}, batch_size "
                     f"{budget['max_batch']}. Never exceed these — the "
                     "GPU will crash.")
        knowledge = self.workflows.knowledge(task)
        if knowledge:
            context = f"{context}\n{knowledge}"
        lessons = self.experience.lessons(task, prompt)
        if lessons:
            job.log("info", "Applying lessons from past renders "
                            "(workflow memory)")
            context = f"{context}\n{lessons}"
        return context

    def _repair_missing(self, job: Job, best: _Attempt,
                        prompt: str) -> _Attempt | None:
        """Surgical accuracy rescue: repaint ONLY the requirements the
        checklist confirmed missing, on the best image so far.

        A whole-frame re-roll gambles everything that already works to fix
        one absent hat; this rung keeps the frame and fixes the hat. A
        region that exists but is wrong is segmented by the words that
        failed; content that does not exist yet gets a placement box from
        the vision model. Fail-open everywhere — None means 'nothing was
        repaired, climb the normal ladder'."""
        missing = best.missing()
        if not missing or best.image is None:
            return None
        if self.settings.inpaint_backend == "mock":
            return None   # offline — no engine to repair with
        image = best.image
        variant = checkpoint = None
        fixed: list[str] = []
        for need in missing[:2]:
            need = str(need).strip()
            if not need:
                continue
            job.log("info", f"[stage] repair — fixing only: {need}")
            mask = None
            try:
                choice = self.auto_mask(image, need, job=job)
                if (choice.ok and choice.mask is not None
                        and choice.mask.getbbox()):
                    mask = choice.mask
            except Exception:  # noqa: BLE001 — fall through to placement
                mask = None
            if mask is None:
                # Nothing matching exists yet — this is an ADD: the vision
                # model picks where the new content belongs.
                try:
                    mask = quality.propose_placement(self.critic, image,
                                                     need)
                except Exception:  # noqa: BLE001
                    mask = None
            if mask is None or not mask.getbbox():
                job.log("info", f"No region could be chosen for '{need}' — "
                                "left to the re-render rungs")
                continue
            if checkpoint is None and variant is None:
                variant, checkpoint = self._choose_inpaint(job, need)
            enh = quality.enhance_prompt(self.llm, need, "inpaint")
            try:
                try:
                    result = self.inpainting.inpaint(  # type: ignore[call-arg]
                        image, self._refine_mask(mask), enh["positive"],
                        negative=enh["negative"], checkpoint=checkpoint,
                        variant=variant or "modern")
                except TypeError:
                    result = self.inpainting.inpaint(
                        image, self._refine_mask(mask), enh["positive"])
            except (PermanentError, TransientError) as exc:
                job.log("info", f"Repair of '{need}' could not render "
                                f"({exc}) — left to the re-render rungs")
                continue
            image = result.image
            fixed.append(need)
        if not fixed:
            return None
        job.log("info", "Repaired in place: " + "; ".join(fixed))
        crit2 = self._critique(job, image, prompt)
        adh2 = self._adherence(job, image, prompt, best.checklist)
        return _Attempt(image=image, prompt_id=best.prompt_id, gen=best.gen,
                        crit=crit2, adherence=adh2, repairs=best.repairs,
                        checklist=best.checklist, strategy="targeted repair")

    def _pursue_request(self, job: Job, *, task: str, prompt: str,
                        prompt_used: str, context: str, image_context: str,
                        triage: dict[str, Any] | None, used_template: bool,
                        image_name: str | None, ckpts: list[str],
                        current_model: str, errors_seen: list[str],
                        state: _Attempt) -> tuple[int, _Attempt]:
        """Climb the escalation ladder until the render does what the prompt
        asked (or the budget runs out). Returns (rounds spent, best attempt).

        Rung order — cheapest real change first: emphasize the missed part of
        the request, then a DIFFERENT MODEL, then a DIFFERENT WORKFLOW. The
        combination that already ran is never retried, so no rung is spent
        re-rolling a recipe that has already been shown to miss.

        Fail-open: with no vision judge there is nothing to escalate on and the
        first render stands, exactly as before."""
        budget = max(0, self.settings.adherence_rounds)
        if budget == 0 or self.critic is None:
            return 0, state

        best = state
        if best.satisfies(self.settings):
            return 0, best
        # Rung 0 — surgical repair: the checklist NAMES what is missing;
        # when the rest of the frame is already right, repaint only those
        # pixels on the best image instead of gambling the whole frame on
        # a re-roll. Keep-best applies, so this can only improve things.
        rounds = 0
        repaired = self._repair_missing(job, best, prompt)
        if repaired is not None:
            rounds = 1
            if repaired.beats(best):
                job.log("info", f"Repair kept — {repaired.summary()}")
                best = repaired
            else:
                job.log("info", "The repair did not improve the checklist "
                                "— keeping the previous best")
            if best.satisfies(self.settings):
                return rounds, best
        current_workflow = (triage or {}).get("workflow") if used_template \
            else None
        models, workflows = self._ladder_candidates(task, current_workflow,
                                                    current_model, ckpts)
        # The request outranks the ladder: when the user named a model, a rung
        # is never allowed to quietly swap it out.
        named = self._named_model(prompt, ckpts)
        if named:
            job.log("info", f"The request names '{named}' — the model stays "
                            "fixed; only the workflow may change")
        plan = quality.escalation_plan(
            best.missing(), models=models, workflows=workflows,
            current_model=current_model, current_workflow=current_workflow,
            allow_model_change=not named, max_rungs=budget)
        if not plan:
            return rounds, best
        # Say the ceiling ONCE, before any of it happens: a live countdown
        # that quietly runs four minutes past its estimate reads as a hang,
        # and that is when people kill the job.
        job.log("info", f"[stage] retry — the render missed part of the "
                        f"request; trying up to {len(plan)} more approach(es). "
                        "The best result so far is already kept, so this can "
                        "only improve it.")
        self._log_eta(job, extra_renders=len(plan))

        for strategy in plan:
            if best.satisfies(self.settings) or job.cancel_requested:
                break
            rounds += 1
            job.log("info", f"[stage] retry — round {rounds}/{len(plan)}: "
                            f"{strategy.why}")
            # Weight the USER's words, not the quality boosters that were
            # appended to them — weighting the whole blob dilutes the request
            # it is supposed to reinforce.
            # TWO different strings, and mixing them up is a silent killer:
            # `render_prompt` is what the diffusion model sees (weights, no
            # prose — English instructions rendered as subject matter, and a
            # blob past CLIP's 77 tokens, would degrade the very render they
            # are meant to fix); `planner_request` is what the LLM designing a
            # graph reads, and it wants the diagnosis.
            tail = (prompt_used[len(prompt):]
                    if prompt_used.startswith(prompt) else "")
            render_prompt = quality.emphasize(prompt, best.missing()) + tail
            planner_request = render_prompt
            if best.missing():
                planner_request += ("\nThe previous attempt did not deliver: "
                                    + "; ".join(best.missing()[:3])
                                    + ". Every one must be clearly visible.")
            if best.crit and best.crit.score < self.settings.critic_min_score:
                planner_request += (
                    f"\nIt also scored {best.crit.score:g}/10 for realism "
                    f"({'; '.join(best.crit.issues[:3]) or 'not photoreal'}). "
                    "Use a different sampler/steps/cfg and a stronger "
                    "photorealism negative prompt.")
            try:
                gen2, note = self._strategy_workflow(
                    job, strategy, task=task, render_prompt=render_prompt,
                    planner_request=planner_request,
                    prompt=prompt, triage=triage, image_name=image_name,
                    image_context=image_context, ckpts=ckpts,
                    current_model=current_model, context=context,
                    used_template=used_template)
            except (PermanentError, TransientError) as exc:
                # A rung that cannot even be BUILT must not fail a job that
                # already has a usable image — drop the rung, keep the result.
                job.log("info", f"Strategy '{strategy.label()}' is not "
                                f"available ({exc}); keeping the best result "
                                "so far")
                continue
            if gen2 is None:
                job.log("info", f"Strategy '{strategy.label()}' has nothing "
                                "to run; trying the next one")
                rounds -= 1  # nothing was rendered — don't charge for it
                continue
            # A rung's failures belong to the rung. Merged into the job's
            # error trail only if its image is kept, or a discarded attempt
            # would teach the workflow memory lessons about a render nobody
            # ever saw.
            rung_errors: list[str] = []
            try:
                image2, pid2, rep2, gen2 = self._render(
                    job, task, gen2, note, rung_errors, speculative=True)
            except TransientError as exc:
                # ComfyUI or the LLM went down mid-ladder. The remaining rungs
                # would fail the same way, and `best` is a real image — stop
                # cleanly rather than failing a job that already succeeded.
                job.log("info", f"The {strategy.label()} attempt could not "
                                f"run ({exc}); keeping the best result so far")
                break
            except PermanentError as exc:
                job.log("info", f"The {strategy.label()} attempt failed "
                                f"({exc}); keeping the best result so far")
                continue
            crit2 = self._critique(job, image2, prompt)
            # No scorecard passed on purpose: _adherence only needs one when
            # the checklist probes fail, and on this machine each vision call
            # costs 4-20 seconds.
            adh2 = self._adherence(job, image2, prompt, best.checklist)
            cand = _Attempt(image=image2, prompt_id=pid2, gen=gen2, crit=crit2,
                            adherence=adh2, repairs=best.repairs + rep2,
                            checklist=best.checklist,
                            strategy=strategy.label())
            if cand.beats(best):
                job.log("info", f"Round {rounds} kept — {strategy.label()}: "
                                f"{cand.summary()}")
                errors_seen.extend(rung_errors)
                best = cand
            else:
                job.log("info", f"Round {rounds} discarded — "
                                f"{cand.summary()} is not better than "
                                f"{best.summary()}; keeping the earlier render")
        if job.cancel_requested:
            job.log("info", "[stage] verify — stopped at your request; the "
                            "best render so far is being saved")
        elif best.satisfies(self.settings):
            job.log("info", "[stage] verify — the result does what the "
                            "prompt asked")
        elif rounds:
            job.log("info", "[stage] verify — best of "
                            f"{rounds + 1} attempts kept; {best.summary()}")
        return rounds, best

    # Checkpoint filename stems that are ordinary English and would pin the
    # model on any prompt that happens to use the word.
    _WEAK_STEM = re.compile(
        r"^(photo|realistic|portrait|anime|base|model|image|art|style|"
        r"dream|real|final|full|main|test|v?\d+)$", re.IGNORECASE)

    def _named_model(self, prompt: str, ckpts: list[str]) -> str | None:
        """The checkpoint the USER named in the prompt, if any.

        Deliberately hard to trigger: community checkpoints are called things
        like `portrait.safetensors`, and a bare substring test would pin the
        model on every prompt containing "portrait" — silently deleting the
        change-model rung the user asked for."""
        low = prompt.lower()
        for name in ckpts:
            stem = name.rsplit(".", 1)[0]
            tokens = [t for t in re.split(r"[_\-.\s]+", stem)
                      if len(t) >= 4 and not self._WEAK_STEM.match(t)]
            hits = [t for t in tokens
                    if re.search(rf"\b{re.escape(t.lower())}\b", low)]
            if len(hits) >= 2:
                return name
            if hits and re.search(
                    rf"\b(use|using|with|model|checkpoint)\b[^.]{{0,20}}"
                    rf"\b{re.escape(hits[0].lower())}\b", low):
                return name
        return None

    def _strategy_workflow(self, job: Job, strategy, *, task: str,
                           render_prompt: str, planner_request: str,
                           prompt: str, triage: dict[str, Any] | None,
                           image_name: str | None, image_context: str,
                           ckpts: list[str], current_model: str,
                           context: str, used_template: bool):
        """Build the next attempt for one ladder rung, and the planner context
        that goes with it. Returns (GeneratedWorkflow | None, context)."""
        if strategy.kind == "workflow" and strategy.workflow:
            gen = self._template_workflow(
                job, {"workflow": strategy.workflow}, render_prompt,
                image_name, boost=strategy.boost)
            if gen is not None:
                return gen, context
            # No such template here (or its models aren't ready): ask the
            # planner for a graph built around what the request still misses.
            job.log("info", f"No runnable '{strategy.workflow}' template — "
                            "designing a workflow for the missing part "
                            "instead")
            return self._plan(job, task, planner_request, context), context

        if strategy.kind == "model" and strategy.checkpoint:
            if used_template:
                gen = self._template_workflow(
                    job, triage, render_prompt, image_name,
                    checkpoint=strategy.checkpoint, boost=strategy.boost)
                if gen is not None:
                    facts = self._recipe_facts(gen.graph)
                    if facts.get("checkpoint") == strategy.checkpoint:
                        return gen, context
                    # A template that pins its own model (Z-Image, Kontext)
                    # has no model rung — _ladder_candidates should already
                    # have withheld one, so this is a belt-and-braces skip
                    # rather than a licence to abandon a working template for
                    # a minutes-long custom design.
                    # Reaching a template rung implies triage ran.
                    job.log("info",
                            f"The '{triage.get('workflow')}' "  # type: ignore[union-attr]
                            "template pins its own model — skipping the "
                            "change-model step")
                    return None, context
            new_context = self._plan_context(job, task, prompt,
                                             strategy.checkpoint, ckpts,
                                             image_context)
            return (self._plan(job, task, planner_request, new_context),
                    new_context)

        # "emphasize": the same recipe, the missed words weighted, new seed.
        if used_template:
            gen = self._template_workflow(job, triage, render_prompt,
                                          image_name, boost=strategy.boost)
            if gen is not None:
                return gen, context
        return self._plan(job, task, planner_request, context), context

    def _plan(self, job: Job, task: str, request: str, context: str):
        job.log("info", f"[llm] planner request: {request[:220]}")
        try:
            gen = self.workflow_ai.generate(task, request, context=context,
                                            log=self._planner_log(job))
        except LLMUnavailableError as exc:
            raise TransientError(str(exc)) from exc
        except (LLMRefusedError, WorkflowNotAllowedError, WorkflowGenerationError) as exc:
            raise PermanentError(str(exc)) from exc
        job.log("info", f"Plan ready via {gen.provenance['source']}:"
                        f"{gen.provenance['model']} "
                        f"({gen.provenance['attempts']} attempt(s), "
                        f"{len(gen.graph)} nodes)")
        job.log("info", f"[llm] plan: {self._graph_summary(gen.graph)}")
        return gen

    @staticmethod
    def _human_time(seconds: float) -> str:
        return eta.human_time(seconds)

    def _estimate_seconds(self, job_type: str,
                          payload: dict[str, Any] | None = None,
                          graph: dict[str, Any] | None = None,
                          checkpoint: str | None = None,
                          with_queue: bool = True,
                          with_load: bool = True) -> tuple[float, int]:
        """Multi-factor prediction (see core/eta.py): job type, resolution,
        video length, steps, checkpoint family, conditioning nodes, live
        GPU/RAM load, jobs queued ahead, and this machine's own history."""
        history = self.queue.recent_durations(job_type)
        queue_ahead = 0.0
        if with_queue:
            for jid in self.queue.pending_order():
                ahead = self.queue.get(jid)
                if ahead is not None:
                    hist = self.queue.recent_durations(ahead.type)
                    queue_ahead += eta.estimate_seconds(
                        ahead.type, hardware=self.hardware, history=hist,
                        payload=ahead.payload, load=None,
                        queue_ahead_seconds=0.0)
        secs = eta.estimate_seconds(
            job_type, hardware=self.hardware, history=history,
            payload=payload, graph=graph, checkpoint=checkpoint,
            load=eta.probe_load() if with_load else None,
            queue_ahead_seconds=queue_ahead)
        return secs, len(history)

    def _log_eta(self, job: Job, required_models: list[str] | None = None,
                 graph: dict[str, Any] | None = None,
                 checkpoint: str | None = None,
                 extra_renders: int = 0) -> None:
        """Emit an [eta:<seconds>] line: the GUI shows a live countdown as
        'Estimated time remaining' (and never explains the math).

        `extra_renders` re-states the estimate when the adherence ladder is
        about to spend more attempts — a countdown that hits zero and then
        keeps going for minutes reads as a hang."""
        # with_queue=False: this logs at execution start, so jobs still in the
        # queue run AFTER this one and must not inflate its estimate.
        secs, _ = self._estimate_seconds(job.type, payload=job.payload,
                                         graph=graph, checkpoint=checkpoint,
                                         with_queue=False)
        if extra_renders:
            # Each rung is a full render plus its judging, and the boosted
            # step count makes it a little longer than the first.
            secs *= extra_renders * 1.2
        msg = (f"[eta:{int(secs)}] Estimated time remaining: "
               f"~{self._human_time(secs)}")
        missing = [m for m in (required_models or [])
                   if not self.registry.is_ready(m)]
        if missing:
            msg += " (plus a one-time model download)"
        job.log("info", msg)

    def _free_vram(self, job: Job) -> None:
        """Unload LLMs (and idle SAM weights) from the shared GPU/RAM so the
        renderer gets as much of the machine as possible.

        DELEGATED renders skip it entirely: the pixels are computed on the
        peer's GPU (which does its own unload in the render proxy), so
        unloading here only forced a pointless cold reload of the local
        planner — paid once per delegated job, and in combine mode that is
        every job the helpers carry."""
        if "/pf-peer/comfy" in str(getattr(self.comfy, "base_url", "")):
            return
        freed = ollama_unload_all(self.settings.llm_url)
        if freed:
            job.log("info", "Freed GPU memory for rendering "
                            f"(unloaded {', '.join(freed)})")
        release = getattr(self.segmentation, "release", None)
        if release and getattr(self.segmentation, "is_loaded", False):
            release()
            job.log("info", "Released SAM weights to free memory "
                            "(reloads on next mask request)")
        # The resident text engine holds ~700 MB of CPU RAM — real money on
        # the machines whose heavy renders are RAM-killed, so it yields
        # here. Never at the cost of an ask in flight: stop(force=False)
        # declines if the engine is busy (the idle watchdog reaps it later).
        worker = self._text_mask_worker
        if worker is not None and worker.warm:
            worker.stop()
            if not worker.warm:
                job.log("info", "Stopped the idle text engine to free "
                                "memory (restarts on next mask request)")

    @staticmethod
    def _heavy_signature(graph: dict[str, Any]) -> str:
        """Which weight files a graph loads, as a stable string.

        Two graphs with the same signature put the same models in memory, so
        one can reuse what the other left cached."""
        parts: list[str] = []
        for node in graph.values():
            if not isinstance(node, dict):
                continue
            cls = str(node.get("class_type", ""))
            if "Loader" not in cls:
                continue
            names = sorted(str(v) for v in (node.get("inputs") or {}).values()
                           if isinstance(v, str))
            if names:
                parts.append(f"{cls}({','.join(names)})")
        return "|".join(sorted(parts))

    def _drop_comfy_cache(self) -> bool:
        """Unload ComfyUI's cached models and forget what it was holding."""
        self._comfy_heavy_cached = None
        free = getattr(self.comfy, "free_memory", None)
        return bool(free and free())

    def _apply_weight_dtype(self, job: Job | None,
                            graph: dict[str, Any]) -> None:
        """Hold an over-large full-precision UNet at half precision.

        Only "default" is overridden: that is the value meaning nobody chose,
        and a template naming its own dtype has chosen."""
        for node in graph.values():
            if not isinstance(node, dict):
                continue
            ins = node.get("inputs") or {}
            if (node.get("class_type") != "UNETLoader"
                    or ins.get("weight_dtype") != "default"):
                continue
            dtype = self._weight_dtype_for(str(ins.get("unet_name") or ""))
            if dtype:
                ins["weight_dtype"] = dtype
                if job is None:
                    continue
                job.log("info", f"Loading {ins['unet_name']} as {dtype} "
                                "instead of full precision — half the "
                                "memory, which is what lets this model run "
                                "here at all")

    def _prepare_graph(self, job: Job | None,
                       graph: dict[str, Any]) -> None:
        """Get a graph ready to run on THIS machine: choose the precision its
        weights are held at, then make room for them. Every path that submits
        a graph loading a large model calls this."""
        self._apply_weight_dtype(job, graph)
        self._prepare_graph_memory(job, graph)

    def _prepare_graph_memory(self, job: Job | None,
                              graph: dict[str, Any]) -> None:
        """Make room for a graph that loads a large standalone model.

        Z-Image, Flux/Kontext GGUF and WAN commit 10+ GB on load, so an
        earlier render's cached checkpoint underneath them OOM-kills ComfyUI.
        The cache is dropped for that reason — but ONLY when the incoming
        graph wants different weights. Unloading a 10.8 GB Kontext stack to
        immediately reload the identical 10.8 GB costs about 90 s and buys
        nothing, because the peak footprint is the same either way."""
        if not any(n.get("class_type") in ("UNETLoader", "UnetLoaderGGUF")
                   for n in graph.values() if isinstance(n, dict)):
            # A checkpoint graph ran instead. It loads weights of its own, so
            # whatever was cached before is no longer ALL that is resident and
            # the belief is now false — forget it rather than let the next
            # heavy render skip its drop and load 10 GB on top of an SDXL
            # checkpoint. Reuse is an optimisation; a wrong belief is an OOM.
            self._comfy_heavy_cached = None
            return
        signature = self._heavy_signature(graph)
        if signature and signature == self._comfy_heavy_cached:
            if job is not None:
                job.log("info", "[stage] prepare — ComfyUI still holds "
                                "these weights; reusing them instead of "
                                "reloading")
        elif self._drop_comfy_cache():
            if job is not None:
                job.log("info", "[stage] prepare — cleared ComfyUI's "
                                "cached models to make room for a large "
                                "model")
        else:
            # The unload did not happen (no /free endpoint, or it failed), so
            # what ComfyUI holds is unknown. Claiming these weights are the
            # resident set would make the NEXT identical render skip its drop.
            self._comfy_heavy_cached = None
            return
        self._comfy_heavy_cached = signature or None

    def _prepare_heavy_render(self, job: Job, need_gb: float = 12.0) -> None:
        """Give a big model load the best chance this machine can offer.

        A WAN motion render loads ~4 GB of UNet plus a 6.3 GB text encoder.
        On a 16 GB box that only fits if nothing else is holding memory —
        measured: an identical render ran 28% faster purely because it started
        with 5.9 GB free instead of 0.03 GB, and when the headroom is gone
        Windows kills ComfyUI outright rather than failing the allocation.

        So: unload the LLMs, drop ComfyUI's cached models, then WAIT a moment
        for the OS to actually reclaim the pages before submitting. Advisory —
        it reports what it could not free rather than refusing to try."""
        self._free_vram(job)
        self._drop_comfy_cache()
        commit = available_commit_gb()
        if commit is None:
            return
        if commit < need_gb:
            # Reclaiming is not instant; give the OS a beat, then re-measure.
            time.sleep(3)
            commit = available_commit_gb() or commit
        if commit < need_gb:
            job.log("info", f"Only {commit:.1f} GB of memory headroom is free "
                            f"(this render wants about {need_gb:.0f} GB). "
                            "Close other heavy apps if it fails — the render "
                            "is starting anyway.")
        else:
            job.log("info", f"{commit:.1f} GB of memory headroom free — "
                            "loading the video model")

    def _comfy_died_midrender(self, job: Job, task: str, prompt: str,
                              exc: Exception) -> TransientError:
        """A render was in flight when ComfyUI stopped answering — the process
        was killed (usually by the OS, out of memory). Learn from it, restart
        ComfyUI, and hand back a friendly retryable error so the job (and the
        frontend) never see a raw connection-refused failure."""
        job.log("error", "ComfyUI stopped responding mid-render — its process "
                         "died (out-of-memory is the usual cause on this "
                         "machine class)")
        self._comfy_heavy_cached = None  # the process died; it holds nothing
        self.events.log("error", f"ComfyUI died during a {task} render")
        self._diagnose_and_record(job, task, prompt, f"process died: {exc}")
        try:
            self._require_comfy(job)  # restart + wait; raises TransientError itself if it can't
        except TransientError:
            pass  # the retry below will hit _require_comfy again
        return TransientError(
            "ComfyUI crashed during the render and was restarted "
            "automatically — retrying at reduced settings.")

    def _spawn_comfy(self) -> bool:
        """Start the ComfyUI process (shared by the per-job guard and the
        health monitor; a lock prevents double-spawning). On machines with
        Cached leftovers stacked under the ~17 GB WAN video stack is what
        OOM-killed it, which --disable-smart-memory used to prevent. That flag
        is a blunt instrument: it also throws away the weights a render is
        about to ask for again, and reloading a Kontext stack costs ~90 s, so
        the cache is ALSO managed deliberately now (_drop_comfy_cache at every
        heavy path; _prepare_graph_memory keeps weights the next graph reuses).

        That deliberate management is not a substitute for the flag on a small
        machine, and this was tested the hard way: raising the threshold to 12
        GB so this 15.7 GB box would cache produced the FIRST fatal ComfyUI
        crash in the logs — 386 prompts of history contain none. A WAN render
        sampled all 20 steps, then "0 models unloaded" was followed by an
        out-of-memory in VAE decode and a hard abort on the next prompt. The
        threshold is back at 20 GB: the checkpoint paths (the inpaint adapter
        submits its own graphs) never drop the cache, so below that the flag
        is still the only thing unloading them."""
        base = Path(self.settings.comfyui_dir) if self.settings.comfyui_dir else None
        if base and (base / "ComfyUI").exists():
            base = base / "ComfyUI"
        main = base / "main.py" if base else None
        if not (main and main.exists()):
            return False
        with self._comfy_revive_lock:
            if self.comfy.is_up():  # someone else already revived it
                return True
            try:
                # main.exists() above proves base is a real Path.
                py = self._comfy_python(cast(Path, base))
            except PermanentError:
                py = sys.executable
            args = [py, str(main), "--listen", "127.0.0.1"]
            if self.hardware.ram_gb <= 20:
                args.append("--disable-smart-memory")
            # Flag decisions read the RESOLVED interpreter's site-packages —
            # base/.venv is wrong for the nested and portable layouts.
            pyp = Path(py)
            env_root = (pyp.parent.parent
                        if pyp.parent.name.lower() == "scripts" else pyp.parent)
            site = env_root / "Lib" / "site-packages"
            # INT8 attention: measured here at 11% faster with the picture
            # unchanged (PSNR 54.8 dB, sharpness identical). Only offered when
            # the package is actually installed — the flag aborts startup
            # otherwise, which would take the renderer down with it.
            if (site / "sageattention").is_dir():
                args.append("--use-sage-attention")
            # torch-directml present means the launcher chose the DirectML
            # swap for this Radeon (a natively-working ROCm-SDK stack is kept
            # WITHOUT torch-directml, so this stays off there).
            if (site / "torch_directml").is_dir():
                args.append("--directml")
            # A fresh process holds no weights.
            self._comfy_heavy_cached = None
            log_dir = self.settings.data_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            try:
                with open(log_dir / "comfyui-revive.log", "ab") as out:
                    flags = 0x08000008 if os.name == "nt" else 0  # DETACHED|NO_WINDOW
                    subprocess.Popen(args, cwd=str(base), stdout=out,
                                     stderr=out, creationflags=flags)
                return True
            except OSError:
                return False

    def _wait_comfy(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.comfy.is_up():
                return True
            time.sleep(2)
        return False

    # -- custom node packs (curated third-party ComfyUI extensions) -----------
    def _comfy_base(self) -> Path | None:
        """The ComfyUI install directory (same resolution as _spawn_comfy)."""
        base = Path(self.settings.comfyui_dir) if self.settings.comfyui_dir else None
        if base and (base / "ComfyUI").exists():
            base = base / "ComfyUI"
        return base if base and (base / "main.py").exists() else None

    def _comfy_python(self, base: Path) -> str:
        """ComfyUI's OWN interpreter — the one that must import a pack's
        dependencies. Repo layout: base\\.venv; nested layout (AMD's
        ROCm-SDK guides): the venv one level ABOVE the code dir; portable
        build: <parent>\\python_embeded\\python.exe. Never falls back to the
        backend venv (a different site-packages ComfyUI cannot see)."""
        candidates = [
            base / ".venv" / "Scripts" / "python.exe",
            base / "venv" / "Scripts" / "python.exe",
            base.parent / ".venv" / "Scripts" / "python.exe",
            base.parent / "python_embeded" / "python.exe",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        raise PermanentError(
            "Could not find ComfyUI's Python environment (looked for "
            ".venv and python_embeded). Node-pack dependencies must install "
            "into ComfyUI's own interpreter — start ComfyUI once via the "
            "launcher so its environment is created, then retry.")

    def node_pack_report(self) -> list[dict[str, Any]]:
        """Probed status of every curated pack (never assumed). Reads the
        SHARED /object_info cache — this runs inside triage on every render,
        and the raw schema is a multi-megabyte response nobody should refetch
        per job."""
        base = self._comfy_base()
        # THIS machine's engine, explicitly: pack status is a statement
        # about what this install can do, and must not flip when the
        # calling thread happens to be delegation-bound to a peer.
        info = self._live_object_info(self._comfy_main)
        live = set(info.keys()) if info else None
        return [node_packs.pack_status(p, base, live)
                for p in node_packs.KNOWN_PACKS.values()]

    @staticmethod
    def _comfy_pids() -> list[int]:
        """Every running ComfyUI process, found by command line rather than
        by port — an orphan that lost :8188 still holds VRAM."""
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\""
                 " | Where-Object { $_.CommandLine -like '*ComfyUI*main.py*' }"
                 " | ForEach-Object { $_.ProcessId }"],
                capture_output=True, text=True, timeout=20).stdout
        except Exception:  # noqa: BLE001 — an empty answer is a safe answer
            return []
        pids = []
        for line in out.splitlines():
            line = line.strip()
            if line.isdigit():
                pids.append(int(line))
        return pids

    @staticmethod
    def _port_holders(port: int = 8188) -> list[int]:
        """PIDs currently LISTENING on a port."""
        pids: list[int] = []
        try:
            out = subprocess.run(["netstat", "-ano", "-p", "TCP"],
                                 capture_output=True, text=True,
                                 timeout=15).stdout
        except Exception:  # noqa: BLE001 — no inventory is an empty answer
            return pids
        for line in out.splitlines():
            if f":{port}" in line and "LISTENING" in line.upper():
                try:
                    pids.append(int(line.split()[-1]))
                except ValueError:
                    continue
        return pids

    def _respawn_comfy_clean(self) -> bool:
        """Clear stray ComfyUI processes, then spawn ONE and wait for it.

        The monitor's revive used to spawn without looking. Strays are
        real: a killed instance loses :8188 but keeps its CUDA context
        and its share of an 8 GB card (seen live: three at once), and a
        half-killed venv PAIR (the 3.13 launcher shim and its child
        share one fate — killing either strands nothing, but killing by
        wrong process-picking has) leaves the survivor holding memory.
        In this path ComfyUI is DOWN by definition, so everything
        matching its command line is safe to clear before the one fresh
        spawn."""
        for pid in self._comfy_pids():
            try:
                subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                               capture_output=True, timeout=15)
            except Exception:  # noqa: BLE001 — spawn is still attempted
                pass
        return self._spawn_comfy() and self._wait_comfy(120)

    def _restart_comfy(self, job: Job,
                       why: str = "to load the new nodes") -> None:
        """Restart ComfyUI. It has no restart API — terminate whatever holds
        its port, WAIT for the port to actually come free, then respawn.

        The wait is not politeness: spawning while the old process still owns
        :8188 makes the new one die on bind with WinError 10048, which then
        looks exactly like a crash."""
        job.log("info", f"Restarting ComfyUI {why}")
        # EVERY ComfyUI process, not just whichever one currently owns the
        # port. Seen live: three were running at once — a killed instance
        # loses :8188 but keeps its CUDA context and its share of an 8 GB
        # card, so restarting "the" one leaves the memory pressure that
        # caused the restart, and can wedge the driver outright.
        pids = set(self._port_holders()) | set(self._comfy_pids())
        if len(pids) > 1:
            job.log("info", f"Closing {len(pids)} ComfyUI processes "
                            "(orphans from earlier restarts were still "
                            "holding graphics memory)")
        for pid in pids:
            try:
                subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                               capture_output=True, timeout=15)
            except Exception as exc:  # noqa: BLE001 — spawn still attempted
                job.log("info", f"Could not stop ComfyUI cleanly: {exc}")
        deadline = time.monotonic() + 30
        while self._port_holders() and time.monotonic() < deadline:
            time.sleep(1)
        time.sleep(1)  # a beat for the socket to leave TIME_WAIT
        # The node inventory may have changed: drop the cached /object_info so
        # new nodes are visible immediately instead of after the TTL.
        self._object_info_cache.clear()
        if not self._spawn_comfy() or not self._wait_comfy(180):
            raise TransientError("ComfyUI did not come back after being "
                                 "restarted — the job will retry.")

    def _download_pack_zip(self, job: Job, repo: str, dest_dir: Path) -> None:
        """Fetch a pack's GitHub archive and place its tree at dest_dir."""
        last_error: Exception | None = None
        for url in node_packs.zip_urls(repo):
            # mkstemp returns an OPEN descriptor — close it, or the cleanup
            # unlink below hits WinError 32 (file in use by this process).
            fd, tmp_name = tempfile.mkstemp(suffix=".zip")
            os.close(fd)
            tmp_zip = Path(tmp_name)
            tmp_dir = Path(tempfile.mkdtemp(prefix="pf-pack-"))
            try:
                job.log("info", f"Downloading {url}")
                req = urllib.request.Request(url, headers={
                    "User-Agent": "PromptForge/1.0"})
                with urllib.request.urlopen(req, timeout=120) as resp, \
                        open(tmp_zip, "wb") as out:
                    total = 0
                    while chunk := resp.read(262144):
                        total += len(chunk)
                        if total > 300 * 1024 * 1024:
                            raise PermanentError(
                                "pack archive exceeds the 300 MB safety cap")
                        out.write(chunk)
                with zipfile.ZipFile(tmp_zip) as zf:
                    zf.extractall(tmp_dir)
                inner = next((d for d in tmp_dir.iterdir() if d.is_dir()), None)
                if inner is None:
                    raise PermanentError("pack archive contained no directory")
                dest_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(inner), str(dest_dir))
                return
            except PermanentError:
                raise
            except Exception as exc:  # noqa: BLE001 — try the next branch name
                last_error = exc
            finally:
                # Cleanup must never fail an install that already succeeded.
                try:
                    tmp_zip.unlink(missing_ok=True)
                except OSError:
                    pass
                shutil.rmtree(tmp_dir, ignore_errors=True)
        raise TransientError(f"Could not download {repo}: {last_error}")

    def _handle_node_pack(self, job: Job) -> dict[str, Any]:
        """Install one curated node pack: download → pip install its
        requirements into ComfyUI's own environment → restart ComfyUI →
        verify the nodes actually registered."""
        pack = node_packs.KNOWN_PACKS.get(str(job.payload.get("pack", "")))
        if pack is None:
            raise PermanentError(
                "Unknown node pack. Only the curated packs can be installed: "
                + ", ".join(sorted(node_packs.KNOWN_PACKS)))
        return self._install_pack_now(pack, job)

    def _install_pack_now(self, pack: node_packs.NodePack,
                          job: Any) -> dict[str, Any]:
        """The install itself, callable from the queue job AND inline from
        the missing-node heal (`job` only needs .log). Restarts ComfyUI."""
        base = self._comfy_base()
        if base is None:
            raise PermanentError(
                "No ComfyUI installation found — set PROMPTFORGE_COMFYUI_DIR "
                "or install ComfyUI first.")
        # Requirements MUST go into ComfyUI's OWN interpreter — that is the
        # one that imports the custom nodes. Falling back to the backend
        # venv (a different Python) installed the deps where ComfyUI never
        # sees them, so nodes showed up "broken". Resolve it once, honestly.
        comfy_py = self._comfy_python(base)
        job.log("info", f"[stage] install — node pack '{pack.title}' "
                        f"({pack.repo})")
        job.log("info", f"Installing requirements into {comfy_py}")
        pip_ok = True
        targets = [(pack.repo, pack.dir_name), *pack.extra_repos]
        for repo, dir_name in targets:
            dest = base / "custom_nodes" / dir_name
            if dest.exists():
                job.log("info", f"{dir_name} already present — keeping it")
                continue
            self._download_pack_zip(job, repo, dest)
            req = dest / "requirements.txt"
            if req.exists():
                py = comfy_py
                job.log("info", f"Installing {dir_name} requirements")
                proc = subprocess.run(
                    [py, "-m", "pip", "install", "--retries", "8",
                     "--timeout", "120", "-r", str(req)],
                    capture_output=True, text=True, timeout=1800)
                if proc.returncode != 0:
                    pip_ok = False
                    tail = (proc.stderr or proc.stdout or "")[-400:]
                    job.log("error", f"pip failed for {dir_name}: {tail}")
                else:
                    # Log what pip actually DID — a "successful" pip that
                    # installed nothing has happened; make it diagnosable.
                    lines = (proc.stdout or "").strip().splitlines()
                    job.log("info", "pip: " + (lines[-1][:200] if lines
                                               else "(no output)"))
        self._restart_comfy(job)
        live = set()
        try:
            # THIS machine's engine explicitly: the heal can run on a
            # delegation-bound thread whose self.comfy is a peer proxy.
            live = self._comfy_main.installed_node_types()
        except Exception:  # noqa: BLE001 — verified below as missing
            pass
        active = pack.verify_node in live
        if active:
            job.log("info", f"Verified: node '{pack.verify_node}' is live — "
                            f"{pack.unlocks} is now available")
            self.events.log("info", f"Node pack '{pack.title}' installed — "
                                    f"{pack.unlocks}")
        else:
            job.log("error", f"Pack files are installed but ComfyUI does not "
                             f"expose '{pack.verify_node}'"
                             + ("" if pip_ok else " (its pip install failed "
                                "— likely missing build tools)"))
        return {"pack": pack.name, "active": active, "pip_ok": pip_ok,
                "status": "active" if active else "broken"}

    def _require_comfy(self, job: Job) -> None:
        """ComfyUI must be up; if it crashed, restart it automatically.

        A generated workflow once hard-crashed ComfyUI and every render after
        that failed until a manual reboot — this is the guard against that.
        """
        # Mock mode means OFFLINE, and this is the one gate every real-render
        # path passes through. Without it the check was answered by whatever
        # ComfyUI happened to be listening on this box (a mock avatar build
        # rendered its mesh through the resident real instance, measured
        # live) — and on failure the recovery below would LAUNCH one. The
        # flag, not the settings, is consulted: tests legitimately stub
        # `services.comfy = Fake()` on a mock-configured Services to drive
        # these paths, and their fakes must keep working.
        if getattr(self.comfy, "offline", False):
            raise PermanentError(
                "This build runs mock renders only — real rendering and "
                "reconstruction need the ComfyUI backend.")
        # A DELEGATED job talks to another machine's ComfyUI. That machine
        # is not ours to restart: if the peer stops answering, what happens
        # next depends on the promise. A job the user PINNED to that
        # machine fails loudly — rendering it here would be doing the one
        # thing they said not to do, invisibly. An auto-delegated job
        # drops the binding and carries on with the local checks below —
        # it simply finishes on this machine instead, and says so.
        if getattr(self._comfy_tls, "client", None) is not None:
            if self.comfy.is_up():
                return
            device = str((job.payload or {}).get("device") or "")
            if device and device not in ("auto", "local"):
                # "Down" through the proxy is AMBIGUOUS: the peer answers
                # 409 while busy with its own work, and is_up() reads
                # that as down. Ask the peer itself which it is — busy
                # means WAIT (the same promise the wrap makes), dead
                # means fail loudly.
                peer = self.peers.find_peer(device)
                info = (self.peers.add_peer(peer.host, peer.port,
                                            timeout=3.0)
                        if peer is not None else None)
                if (info and info.get("render")
                        and (info.get("comfy") or {}).get("up")
                        and not info.get("idle")):
                    job.log("info", f"[peer] '{device}' got busy with "
                                    "its own work mid-render — waiting "
                                    "for it")
                    deadline = time.time() + 15 * 60
                    while time.time() < deadline:
                        if job.cancel_requested:
                            raise TransientError(
                                f"cancelled while waiting for '{device}'")
                        time.sleep(3.0)
                        if self.comfy.is_up():
                            job.log("info", f"[peer] '{device}' is free "
                                            "again — continuing")
                            return
                    msg = (f"'{device}' stayed busy with its own work "
                           "for 15 minutes. Nothing was rendered on this "
                           "machine. Retry when it is free, or set "
                           "Render: this PC.")
                    self.events.log("error", msg)
                    raise PermanentError(msg)
                msg = (f"'{device}' stopped answering mid-render. Nothing "
                       "was rendered on this machine. Check that the "
                       "other PC is on and PromptForge is running there, "
                       "then press Retry — or set Render: this PC to run "
                       "the job locally.")
                self.events.log("error", msg)
                raise PermanentError(msg)
            job.log("info", "[peer] the delegated machine stopped "
                            "answering — continuing on this machine")
            self.events.log("info", "A delegated render's peer stopped "
                                    "answering mid-job — continuing on "
                                    "this machine")
            self._comfy_tls.client = None
        # health() is an OPTIONAL capability, probed the same way free_memory()
        # is. The adapter boundary here is duck-typed and every fake in the
        # tests implements is_up() only — requiring the richer method broke
        # 43 of them at once.
        probe = getattr(self.comfy, "health", None)
        if probe is not None:
            healthy, why = probe()
        else:
            healthy, why = self.comfy.is_up(), "ComfyUI is not listening."
        if healthy:
            return
        # A wedged graphics driver is NOT the same failure as a stopped
        # process: spawning another instance inherits the broken context, so
        # the fix is to close every ComfyUI first. Doing that here turns four
        # identical failed attempts into one recovery.
        if "driver" in why or "CUDA" in why:
            job.log("error", f"{why} Closing all of them and starting fresh.")
            self.events.log("error", "ComfyUI's graphics driver wedged — "
                                     "restarting it from scratch")
            try:
                self._restart_comfy(job, "to clear the graphics-driver error")
                job.log("info", "ComfyUI is back up — continuing")
                return
            except TransientError:
                raise TransientError(
                    "ComfyUI's graphics driver is in a bad state and it did "
                    "not recover. Close every ComfyUI window (or reboot) and "
                    "the job will retry."
                ) from None
        job.log("error", "ComfyUI is down — restarting it automatically")
        self.events.log("error", "ComfyUI is down — restarting it")
        if not self._spawn_comfy():
            raise TransientError(
                "ComfyUI is not running — start it (the launcher does this "
                "automatically) and the job will retry.")
        if self._wait_comfy(120):
            job.log("info", "ComfyUI is back up — continuing")
            self.events.log("info", "ComfyUI restarted and healthy again")
            return
        raise TransientError(
            f"ComfyUI did not come back within 120s. {why} "
            "The job will retry.")

    def _spawn_ollama(self, exe: str) -> bool:
        log_dir = self.settings.data_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(log_dir / "ollama-revive.log", "ab") as out:
                flags = 0x08000008 if os.name == "nt" else 0  # DETACHED|NO_WINDOW
                subprocess.Popen([exe, "serve"], stdout=out, stderr=out,
                                 creationflags=flags)
            return True
        except OSError:
            return False

    def _revive_ollama(self, job: Job) -> bool:
        """Best-effort restart of a crashed Ollama server ('ollama serve').
        Returns True if it is (or came back) up. The planner also has the
        Claude API fallback, so this never blocks the job — it just prefers
        keeping generation local, as the project intends."""
        if ollama_is_up(self.settings.llm_url):
            return True
        # Mock mode means OFFLINE (the _live_object_info rule): an Ollama
        # binary on this box belongs to some other setup. A running one is
        # still used (the up-check above) — but a mocked demo must not
        # LAUNCH other software and block 30s waiting on it (measured live:
        # this wait was the first half of a 'stuck' mock edit).
        if self.settings.inpaint_backend == "mock":
            return False
        exe = shutil.which("ollama")
        if not exe:
            return False
        job.log("error", "Ollama is down — restarting it automatically")
        self.events.log("error", "Ollama is down — restarting it")
        if not self._spawn_ollama(exe):
            job.log("error", "Could not start Ollama")
            return False
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if ollama_is_up(self.settings.llm_url):
                job.log("info", "Ollama is back up — local planning restored")
                return True
            time.sleep(1.5)
        job.log("error", "Ollama did not come back; using the API fallback "
                         "if configured")
        return False

    # Formats that are ALREADY compressed. Casting one of these to fp8 does
    # not save memory and can corrupt it — an int8-convrot checkpoint carries
    # its own scales, and reinterpreting them is not a precision choice.
    _QUANTISED = ("int8", "fp8", "nf4", "int4", "gguf", "q2_", "q3_", "q4_",
                  "q5_", "q6_", "q8_", "svdq")

    def _weight_dtype_for(self, unet_name: str) -> str | None:
        """fp8 for a full-precision UNet too big for this GPU, else None.

        The video stack is the case that matters: wan2.2_ti2v_5B is 10.0 GB
        of fp16 on an 8.6 GB card with 15.7 GB of system RAM behind it, and
        RAM is what OS-kills the load. Half of 10 GB fits where 10 GB does
        not, and no download is involved — the weights on disk are unchanged,
        only how they are held in memory."""
        name = unet_name.lower()
        if not name or any(q in name for q in self._QUANTISED):
            return None
        # An unknown GPU reports 0.0 VRAM (no nvidia-smi, or CPU-only). With
        # no floor that made every full-precision UNet "too big" and cast the
        # lot to fp8 — on a CPU-only machine, where fp8 buys nothing and is
        # not what the user asked for. Unknown means leave it alone.
        vram_gb = self.hardware.vram_gb
        if not vram_gb or vram_gb <= 0:
            return None
        path = self.settings.models_dir / "diffusion_models" / unet_name
        try:
            size_gb = path.stat().st_size / 1024 ** 3
        except OSError:
            return None  # not where we expect it: do not guess about it
        # Comfortably smaller than the card: full precision is the better
        # picture and it fits, so leave it alone.
        if size_gb <= vram_gb * 0.75:
            return None
        # e4m3fn, not e4m3fn_fast: the fast variant additionally routes the
        # matmuls through Ada's FP8 tensor cores, which is a speed claim this
        # has not measured and which needs hardware this cannot detect here.
        return "fp8_e4m3fn"

    def _apply_hardware_limits(self, graph: dict[str, Any],
                               job: Job) -> dict[str, Any]:
        """Clamp resource-hungry parameters in a generated graph to what this
        GPU survives. An over-budget canvas is scaled down (aspect kept); the
        planner is told the budget up front, so clamps should be rare."""
        b = render_budget(self.hardware)
        # Precision is part of "what this GPU survives". Safe to run twice —
        # once a dtype is chosen the input is no longer "default".
        self._apply_weight_dtype(job, graph)
        for node in graph.values():
            ins = node.get("inputs", {})
            ctype = node.get("class_type")
            if ctype == "EmptyLatentImage":
                try:
                    w, h = int(ins.get("width", 512)), int(ins.get("height", 512))
                except (TypeError, ValueError):
                    continue
                scale = min(1.0, b["max_side"] / max(w, h, 1),
                            (b["max_pixels"] / max(w * h, 1)) ** 0.5)
                if scale < 1.0:
                    ins["width"] = max(64, int(w * scale) // 8 * 8)
                    ins["height"] = max(64, int(h * scale) // 8 * 8)
                    job.log("info", f"Clamped canvas {w}×{h} → "
                                    f"{ins['width']}×{ins['height']} "
                                    "(VRAM budget)")
                if int(ins.get("batch_size", 1) or 1) > b["max_batch"]:
                    ins["batch_size"] = b["max_batch"]
            elif ctype == "KSampler":
                try:
                    steps = int(ins.get("steps", 20))
                except (TypeError, ValueError):
                    continue
                if steps > b["max_steps"]:
                    ins["steps"] = b["max_steps"]
                    job.log("info", f"Clamped steps {steps} → {b['max_steps']}")
            elif ctype in ("WanImageToVideo", "SV3D_Conditioning",
                           "WanVaceToVideo"):
                for key, cap in (("width", b["max_video_side"]),
                                 ("height", b["max_video_side"]),
                                 ("length", b["max_video_len"]),
                                 ("video_frames", 25)):
                    try:
                        val = int(ins.get(key, 0) or 0)
                    except (TypeError, ValueError):
                        continue
                    if val > cap:
                        ins[key] = cap
                        job.log("info", f"Clamped {ctype}.{key} {val} → {cap}")
                if ctype == "WanVaceToVideo":
                    # Per-axis caps are not enough for VACE: it holds every
                    # frame in VRAM at once, so the cost is the PRODUCT.
                    # (And a clamp that lands on a non-4n+1 length is not a
                    # smaller render, it is a hard error.)
                    try:
                        w = int(ins.get("width", 0) or 0)
                        h = int(ins.get("height", 0) or 0)
                        n = int(ins.get("length", 0) or 0)
                    except (TypeError, ValueError):
                        continue
                    fitted = motion.fit_window(w, h, n)
                    if fitted != n:
                        ins["length"] = fitted
                        job.log("info", f"Clamped the clip to {fitted} frames "
                                        f"— {w}×{h}×{n} is more than this "
                                        "machine can hold at once")
        # An SD1.5-class model asked for a big canvas in ONE pass breaks
        # down beyond its native scale (measured 2026-08-18: doubled
        # irises, waxy skin, duplicated objects at 1024² — while a
        # 512-base + refine of the same seed was both better AND
        # faster). Oversized single-pass txt2img becomes two-pass.
        split = hires_split_graph(graph)
        if split is not None:
            job.log("info", "Large canvas on an SD1.5-class model — "
                            "rendering base-then-refine (hires fix): "
                            "single-pass at this size deforms anatomy "
                            "and duplicates detail")
            return split
        return graph

    def _render(self, job: Job, task: str, gen, context: str,
                errors_seen: list[str] | None = None,
                speculative: bool = False):
        """Run a generated graph, repairing via the LLM on ComfyUI errors.
        Successful repairs are distilled into long-term repair knowledge.

        `speculative` marks an adherence-ladder attempt: a result is already
        safe, so this render's problems are reported as an alternative that
        didn't work out rather than as job errors — a planned detour should
        not fill the log with red at the exact moment the user is wondering
        why the job is taking longer than the estimate."""
        level = "info" if speculative else "error"
        aside = (" — this is an extra attempt; the earlier result is safe"
                 if speculative else "")
        repairs = 0
        last_error: str | None = None
        broken_graph: dict[str, Any] | None = None
        while True:
            gen.graph = self._apply_hardware_limits(gen.graph, job)
            try:
                self._free_vram(job)
                self._prepare_graph(job, gen.graph)
                try:
                    image, prompt_id = self.comfy.run_graph(gen.graph)
                except BackendUnavailableError as exc:
                    raise self._comfy_died_midrender(job, task, "",
                                                     exc) from exc
                if repairs and last_error and broken_graph is not None:
                    # The LLM fixed a real error: remember error→fix forever.
                    self.experience.record_repair(task, last_error,
                                                  broken_graph, gen.graph)
                    job.log("info", "[llm] repair lesson saved to long-term "
                                    "memory")
                return image, prompt_id, repairs, gen
            except WorkflowRuntimeError as exc:
                if job.cancel_requested:
                    # The user pressed stop and ComfyUI reported the interrupt
                    # as an error. Repairing it would make the job do MORE
                    # work in response to being told to stop.
                    raise PermanentError("Stopped at your request.") from exc
                hint = commit_exhausted_hint(str(exc))
                if hint:
                    # Not a graph problem — LLM repairs can't add RAM.
                    raise PermanentError(f"Render failed: {hint}") from exc
                if errors_seen is not None:
                    errors_seen.append(str(exc))
                repairs += 1
                if repairs > self.settings.workflow_max_repairs:
                    raise PermanentError(
                        f"Workflow still failing after {repairs - 1} LLM "
                        f"repair(s). Last ComfyUI error: {exc}") from exc
                if not self.comfy.is_up():
                    self._require_comfy(job)  # crashed mid-render: revive
                job.log(level, f"ComfyUI error: {exc}{aside}")
                job.log("info", f"Asking the LLM to repair the workflow "
                                f"(repair {repairs}/"
                                f"{self.settings.workflow_max_repairs})…")
                last_error, broken_graph = str(exc), gen.graph
                try:
                    gen = self.workflow_ai.repair(task, gen.graph, str(exc),
                                                  context=context,
                                                  log=self._planner_log(job))
                except LLMUnavailableError as inner:
                    raise TransientError(str(inner)) from inner
                except (LLMRefusedError, WorkflowGenerationError) as inner:
                    raise PermanentError(str(inner)) from inner

    def _diagnose_and_record(self, job: Job, task: str, prompt: str,
                             error: str) -> None:
        """Learn from ANY render failure (template renders included, not just
        LLM-planned graphs): record it to the experience store and ask the LLM
        for a short human diagnosis + likely fix. Fully fail-safe."""
        try:
            self.experience.record(task, prompt or "", None, success=False,
                                   errors=[error])
        except Exception:  # noqa: BLE001 — learning is best-effort
            pass
        try:
            reply = self.llm.complete(
                "You are a ComfyUI troubleshooter. In 1-2 sentences, explain "
                "this error and the single most likely fix (missing custom "
                "node/model, update ComfyUI, reduce resolution for VRAM, etc.). "
                "Plain text, no preamble.",
                f"Task: {task}\nError: {error[:800]}", max_tokens=160)
            if reply.text.strip():
                job.log("info", f"[llm] diagnosis: {reply.text.strip()[:300]}")
        except (LLMError, AttributeError):  # base LLMError = any LLM failure
            pass

    def _adherence(self, job: Job, image: Image.Image, prompt: str,
                   checklist: list[dict[str, str]],
                   scores: dict[str, int] | None = None
                   ) -> dict[str, Any] | None:
        """Did this render actually DO what the prompt asked?

        Prefers the item-by-item checklist verdict (it names WHAT is missing,
        which is what picks the next strategy); falls back to the scorecard's
        single prompt_accuracy number when the judge can't answer a checklist.
        None when there is no vision model at all — adherence checking is an
        improvement to the pipeline, never a requirement of it.

        `scores` is an already-computed scorecard: pass it wherever the caller
        has one, so this never costs a second vision round-trip per attempt."""
        if self.critic is None:
            return None
        if not checklist and getattr(self.critic, "ask", None) is None:
            # Without a checklist AND without a question-answering judge, the
            # scorecard degenerates to the realism score repeated six times —
            # it would cost another vision pass to learn nothing the critic
            # has not already said.
            return None
        # The text model settles wording disagreements — it never sees the
        # image, so it cannot rubber-stamp the way the vision model does.
        report = quality.verify_adherence(self.critic, image, prompt,
                                          checklist, llm=self.llm)
        if report is not None:
            unclear = report.get("unclear") or []
            if report["missing"]:
                job.log("info", f"[stage] verify — the render matches "
                                f"{report['accuracy']}% of what could be "
                                f"checked; missing: "
                                f"{'; '.join(report['missing'][:3])}")
            else:
                job.log("info", "[stage] verify — every part of the request "
                                "that could be checked is in the image")
            if unclear:
                job.log("info", "Could not settle: " + "; ".join(unclear[:3])
                        + " — not counted either way, so it cannot send the "
                          "retry ladder after something that is already there")
            return report
        if scores is None:
            scores = quality.scorecard(self.critic, image, prompt)
        acc = (scores or {}).get("prompt_accuracy")
        if acc is None:
            return None
        job.log("info", f"[stage] verify — the render matches the request "
                        f"{acc}/100")
        # source="score": ONE opaque number. Its empty `missing` list is
        # ignorance, not proof that everything was delivered — every caller
        # must treat the two sources as different evidence.
        return {"accuracy": int(acc), "missing": [], "met": [],
                "source": "score", "scores": scores}

    # Checkpoints that cannot serve a from-scratch or whole-image render:
    # 9-channel inpainting UNets need a mask, and image-only models (SV3D)
    # aren't text-to-image at all. Offering one as a "different model" rung
    # buys a guaranteed multi-minute failure, not a second opinion.
    _NOT_A_GENERATOR = re.compile(r"inpaint|sv3d|refiner|vae|lora|control",
                                  re.IGNORECASE)

    def _ladder_candidates(self, task: str, current_workflow: str | None,
                           current_model: str | None,
                           ckpts: list[str]) -> tuple[list[str], list[str]]:
        """(models, workflows) the adherence ladder may escalate to.

        Both lists are filtered to things that can actually RUN here: a model
        whose architecture suits the task, and a template whose own models are
        downloaded and whose memory needs this machine meets. A rung that
        cannot produce an image is worse than no rung — it costs the same
        minutes and returns nothing."""
        models = [c for c in ckpts
                  if c != current_model
                  and (task == "inpaint" or not self._NOT_A_GENERATOR.search(c))]
        # A template that hard-codes its model has no model rung: filling one
        # in is silently ignored, and abandoning the template to honour it
        # costs a full custom design.
        if current_workflow:
            try:
                pinned = "checkpoint" not in (
                    self.workflows.load_named(current_workflow)
                    .get("parameters", {}))
            except Exception:  # noqa: BLE001 — unknown template: assume pinned
                pinned = True
            if pinned:
                models = []
        workflows: list[str] = []
        try:
            templates = self.workflows.list_all()
        except Exception:  # noqa: BLE001 — the ladder degrades, never breaks
            templates = []
        for t in templates:
            name = t.get("template", "")
            if not name or t.get("task", name) != task:
                continue
            # Never escalate INTO a draft/preview template: it is deliberately
            # lower quality, so "try harder" would mean "try worse".
            if re.search(r"draft|preview|fast", name, re.IGNORECASE):
                continue
            required = t.get("required_models") or []
            if not self._models_fit_machine(required)[0]:
                continue
            if any(not self.registry.is_ready(m) for m in required):
                continue
            workflows.append(name)
        return models, workflows

    def _critique(self, job: Job, image: Image.Image, prompt: str) -> Critique | None:
        """Score a result's realism; None when the critic is off/unavailable."""
        if self.critic is None:
            return None
        try:
            crit = self.critic.critique(image, prompt)
        except CriticUnavailable as exc:
            job.log("error", f"Realism check skipped: {exc}")
            return None
        job.log("info", f"[llm] critic ({crit.model}): {crit.summary()}"
                        + (f" — issues: {'; '.join(crit.issues[:3])}"
                           if crit.issues else ""))
        return crit

    # The user explicitly asking for a model search overrides the scout's
    # prefer-installed default — the program adapts to what the prompt says.
    _FORCE_SEARCH = re.compile(
        r"search (?:online|the (?:web|internet|hub))|find (?:a |an )?"
        r"(?:better|new|different) (?:image |video )?model|"
        r"download (?:a |an )?(?:better|new|different) (?:image |video )?model",
        re.IGNORECASE)

    def _handle_workflow(self, job: Job) -> dict[str, Any]:
        p = job.payload
        task, prompt = p.get("task", "generate"), p["prompt"]
        errors_seen: list[str] = []
        try:
            return self._workflow_inner(job, task, prompt, errors_seen)
        except Exception as exc:
            # Failures are lessons too: remembered for future planning.
            self.experience.record(task, prompt, None, success=False,
                                   errors=errors_seen + [str(exc)])
            raise

    # Tasks that transform an existing image and therefore need one attached.
    IMAGE_TASKS = {"img2img", "upscale", "outpaint", "inpaint"}

    def _workflow_inner(self, job: Job, task: str, prompt: str,
                        errors_seen: list[str]) -> dict[str, Any]:
        self._require_comfy(job)
        self._revive_ollama(job)  # keep planning local when possible
        self._log_eta(job)

        job.log("info", "[stage] models — checking what is installed")
        ckpts = self._ensure_checkpoint(job)

        # Let the LLM read the prompt FIRST: pick the fitting workflow and
        # pre-fetch any models it decides are needed, before planning.
        # EXCEPT a ready draft: the deterministic coercion below would
        # override triage's answer anyway, and the triage call itself was
        # measured at 33 s (cold planner load + routing) on a job whose
        # render takes ~5 s.
        triage: dict[str, Any] | None = None
        draft_ready = False
        # Draft intent arrives two ways: the words of the prompt, or the
        # Studio's "Quick draft" toggle riding the payload — same meaning.
        if task == "generate" and (job.payload.get("draft")
                                   or quality.draft_intent(prompt)):
            try:
                needed = (self.workflows.load_named("generate_draft")
                          .get("required_models") or [])
                draft_ready = all(self.registry.is_ready(m) for m in needed)
            except Exception:  # noqa: BLE001 — the normal path always works
                draft_ready = False
            if not draft_ready:
                job.log("info", "A draft was asked for, but the speed "
                                "template isn't ready on this machine — "
                                "rendering full quality instead")
        if draft_ready:
            triage = {"workflow": "generate_draft"}
            job.log("info", "Draft requested — skipping workflow triage, "
                            "the 4-step speed template renders this")
        else:
            triage = self._triage(job, task, prompt)

        # Default prompt optimization: quality boosters are APPENDED — the
        # user's words always survive verbatim (only safety.py filters).
        # A DRAFT skips it twice over: the enhancement LLM call was the
        # draft path's last model load (~19 s cold), and a draft exists to
        # preview YOUR wording — enhancing it changes the thing being
        # tested. With this skip the whole draft path is LLM-free.
        if draft_ready:
            prompt_used = prompt
            job.log("info", "Draft renders your words verbatim — no "
                            "enhancement pass")
        else:
            enh = quality.enhance_prompt(self.llm, prompt, task)
            prompt_used = enh["positive"]
            added = prompt_used[len(prompt):].lstrip(", ")
            if added:
                job.log("info", f"[llm] prompt enhanced: +{added[:120]}")

        # Attach the input image for image-transform tasks so the LLM plans
        # around a real uploaded file instead of inventing a filename.
        image_context = ""
        uploaded_image: str | None = None
        if job.payload.get("asset_id"):
            src = self.open_asset_image(job.payload["asset_id"])
            uploaded_image = self.comfy.upload_image(src, "forge_src")
            image_context = (f"\nInput image file: '{uploaded_image}' "
                             f"({src.width}×{src.height}) — load it with "
                             f"LoadImage using EXACTLY this filename.")
            job.log("info", f"Input image uploaded as {uploaded_image}")
        elif task in self.IMAGE_TASKS:
            raise PermanentError(
                f"The '{task}' task transforms an existing image — upload "
                "one first.")

        # Draft intent is a CAPABILITY, not a phrasing (the background/
        # animate/viewpoint doctrine): a READY draft already skipped triage
        # above; this is only the honest message for the not-ready case.
        if (task == "generate" and not draft_ready
                and quality.draft_intent(prompt)):
            job.log("info", "Draft requested, but the speed model(s) are "
                            "not downloaded yet — rendering normal quality")

        # BEST-WORKFLOW FAST PATH: if triage chose a validated template whose
        # models are ready, render THAT template's tuned graph — a custom LLM
        # design only happens when no template fits. Without this, triage's
        # choice was cosmetic and every generate became a custom SDXL graph.
        template_gen = self._template_workflow(
            job, triage, prompt_used, uploaded_image)
        used_template = template_gen is not None
        # The planning context is built either way: the adherence ladder may
        # fall through to a custom design on a later rung, and a blind planner
        # (no checkpoint list, no image filename) invents filenames.
        chosen = ckpts[0]

        if used_template:
            gen = template_gen
            # used_template is only ever set from a non-None triage.
            model_note = f"template: {triage['workflow']}"  # type: ignore[index]
            job.log("info", f"[stage] plan — using the "
                            f"'{triage['workflow']}' template (chosen by "  # type: ignore[index]
                            "triage; its models are ready)")
            context = self._plan_context(job, task, prompt, chosen, ckpts,
                                         image_context)
        else:
            # Prompt-aware model choice; may search the hub and download a
            # better-fitting checkpoint (verified, size-capped).
            model_note = f"template default ({chosen})"
            if task in ("generate", "img2img"):
                forced = bool(self._FORCE_SEARCH.search(prompt))
                if forced:
                    job.log("info", "Prompt explicitly asks for a model "
                                    "search — obeying it")
                self._queue_model_research(job, ckpts)
                intel = self.model_intel.summary(ckpts)
                if intel:
                    job.log("info", "[llm] consulting the model knowledge "
                                    "file for this prompt")
                decision = self.scout.choose(
                    prompt + (f"\n{intel}\nPick the model whose known "
                              "strengths match this prompt." if intel else ""),
                    task, ckpts, allow_download=self.settings.auto_install,
                    progress=self._download_progress(job, "scout model"),
                    force_search=forced, log=self._scout_log(job))
                job.log("info", f"Model choice: {decision.note}")
                if decision.downloaded:
                    ckpts = self._image_checkpoints() or ckpts
                chosen = decision.checkpoint
                model_note = decision.note
            context = self._plan_context(job, task, prompt, chosen, ckpts,
                                         image_context)
            job.log("info", "[stage] plan — the LLM designs the workflow")
            gen = self._plan(job, task, prompt_used, context)

        job.log("info", "[stage] render — ComfyUI is working")
        image, prompt_id, repairs, gen = self._render(job, task, gen, context,
                                                      errors_seen)

        # THE PROMPT IS THE CONTRACT. One ladder judges the render on both
        # counts that can fail it — is it photoreal, and does it actually DO
        # what was asked — and every rung it climbs changes something real:
        # the emphasis, then the MODEL, then the WORKFLOW. Re-rolling the seed
        # of a recipe that already missed is a coin flip on the same coin.
        # A DRAFT is explicitly a preview: measured on a live draft job,
        # the quality ladder (critic + checklist + adherence + verify)
        # cost 62 s on top of a ~5 s render — twelve times the render it
        # was judging. Drafts skip the ladder and the polish; the honest
        # trade is stated in the log.
        is_draft = (used_template
                    and (triage or {}).get("workflow") == "generate_draft")
        if is_draft:
            job.log("info", "Draft mode — skipping quality checks and "
                            "retries (a draft is for iterating on the "
                            "wording; ask for a final when it is right)")
            crit = None
            rounds = 0
        else:
            job.log("info", "[stage] check — judging realism and whether "
                            "the render did what the prompt asked")
            crit = self._critique(job, image, prompt)
            checklist = (quality.request_checklist(self.llm, prompt)
                         if self.critic is not None else [])
            if checklist:
                job.log("info", "[llm] the render must deliver: "
                                + " · ".join(c["need"] for c in checklist))
            adh = self._adherence(job, image, prompt, checklist)
            rounds, chose = self._pursue_request(
                job, task=task, prompt=prompt, prompt_used=prompt_used,
                context=context, image_context=image_context,
                triage=triage, used_template=used_template,
                image_name=uploaded_image, ckpts=ckpts, current_model=chosen,
                errors_seen=errors_seen,
                state=_Attempt(image=image, prompt_id=prompt_id, gen=gen,
                               crit=crit, adherence=adh, repairs=repairs,
                               checklist=checklist))
            image, prompt_id, gen = chose.image, chose.prompt_id, chose.gen
            crit, repairs = chose.crit, chose.repairs
            if chose.strategy:
                model_note = f"{model_note} → {chose.strategy}"

        # Faces get one native-resolution refinement pass before saving —
        # judged, so it can only ever improve the shipped image. Drafts
        # skip it: polish belongs on finals.
        polished = None
        if not is_draft:
            own_ckpt = next(
                (n["inputs"].get("ckpt_name") for n in gen.graph.values()
                 if isinstance(n, dict)
                 and n.get("class_type") == "CheckpointLoaderSimple"
                 and n.get("inputs", {}).get("ckpt_name")), None)
            polished = self._face_polish(job, image, prompt_used,
                                         checkpoint=own_ckpt)
            if polished is not None:
                image = polished

        job.log("info", "[stage] save — storing the result")
        # The recipe card: how this exact image was made — the workflow
        # decision, the models/params actually executed, and the step trail.
        recipe: dict[str, Any] = {
            "task": task,
            "prompt": prompt[:300],
            "prompt_enhanced": (prompt_used[:300]
                                if prompt_used != prompt else None),
            # used_template is only ever set from a non-None triage.
            "workflow": (f"{triage['workflow']} template" if used_template  # type: ignore[index]
                         else "LLM-planned custom graph"
                         + (f" (reference template: {triage['workflow']})"
                            if triage and triage.get("workflow") else "")),
            "planned_by": f"{gen.provenance.get('source')}:"
                          f"{gen.provenance.get('model')} "
                          f"({gen.provenance.get('attempts')} attempt(s))",
            "model_choice": model_note,
            "nodes": len(gen.graph),
            "repairs": repairs,
            "strategy_rounds": rounds,
            "face_refined": polished is not None,
            "draft": is_draft,
            "realism": crit.score if crit else None,
            **self._recipe_facts(gen.graph),
            "trail": self._recipe_steps(job),
        }
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        asset = self.store.save_upload(f"forge_{job.id}.png", buf.getvalue(),
                                       meta={"recipe": recipe})
        job.log("info", f"Saved result as asset {asset.id}")
        self.experience.record(task, prompt, gen.graph, success=True,
                               realism=crit.score if crit else None,
                               repairs=repairs, errors=errors_seen)
        return {"asset_id": asset.id, "task": task, "repairs": repairs,
                "prompt_id": prompt_id, "provenance": gen.provenance,
                "realism": crit.score if crit else None,
                "recipe": recipe}

    # SV3D orbits a SINGLE OBJECT on a plain background — it was trained on
    # object renders and has no scene or face prior. Given a real photo it
    # rotates the whole frame like a picture on a turntable, so the subject is
    # cut out and re-staged before it ever sees the image.
    VIEW_SIZE = 576
    VIEW_FRAMES = 21

    @staticmethod
    def _stage_for_orbit(image: Image.Image, mask: Image.Image | None,
                         size: int) -> Image.Image:
        """The subject, cut out, centered and square-padded on neutral grey —
        the input shape SV3D was trained on."""
        subject = image.convert("RGB")
        box = mask.getbbox() if mask is not None else None
        if box:
            pad_x = int((box[2] - box[0]) * 0.08) + 8
            pad_y = int((box[3] - box[1]) * 0.08) + 8
            cut = Image.new("RGB", subject.size, (128, 128, 128))
            # box came from mask.getbbox(), so mask cannot be None here.
            cut.paste(subject, (0, 0),
                      quality.fit_mask(cast(Image.Image, mask), subject.size))
            subject = cut.crop((max(0, box[0] - pad_x), max(0, box[1] - pad_y),
                                min(subject.width, box[2] + pad_x),
                                min(subject.height, box[3] + pad_y)))
        side = max(subject.size)
        square = Image.new("RGB", (side, side), (128, 128, 128))
        square.paste(subject, ((side - subject.width) // 2,
                               (side - subject.height) // 2))
        return square.resize((size, size), Image.Resampling.LANCZOS)

    @staticmethod
    def _orbit_frames(total: int, wanted: int, span: float) -> list[int]:
        """Which of the orbit's frames to keep. Frame 0 IS the input view, so
        a single extra viewpoint deliberately skips it — returning the picture
        we were given reads as a bug, not an answer."""
        if wanted <= 1:
            return [max(1, round(total * span / 720)) % total]
        step = total * (span / 360.0) / wanted
        start = 0 if span >= 359 else -step * (wanted - 1) / 2
        out: list[int] = []
        for k in range(wanted):
            idx = int(round(start + k * step)) % total
            if idx not in out:
                out.append(idx)
        return out

    @staticmethod
    def _contact_sheet(frames: list[Image.Image],
                       labels: list[str]) -> Image.Image:
        """The viewpoints as one labelled picture — the edit view can show
        exactly one image, and N unlabelled squares explain nothing."""
        cols = 2 if len(frames) == 4 else min(len(frames), 3)
        rows = (len(frames) + cols - 1) // cols
        cell = frames[0].width
        strip = 22
        sheet = Image.new("RGB", (cols * cell, rows * (cell + strip)),
                          (18, 18, 18))
        draw = ImageDraw.Draw(sheet)
        for i, frame in enumerate(frames):
            x, y = (i % cols) * cell, (i // cols) * (cell + strip)
            sheet.paste(frame, (x, y))
            draw.text((x + 6, y + cell + 5), labels[i], fill=(210, 210, 210))
        return sheet

    def _orbit_from_photo(self, job: Job, image: Image.Image,
                          mask: Image.Image | None, frames: int | None = None,
                          prefix: str = "orbit_src"
                          ) -> list[tuple[bytes, str]]:
        """Orbit ONE photo's subject with SV3D; returns the rendered frames.

        The staging is the whole point and the reason this is shared rather
        than copied. SV3D expects a cut-out subject on neutral grey, square
        and 576px. Handed a raw photo it rotates the FRAME — a picture on a
        turntable — which is exactly what the avatar intake used to do.
        Raises on failure; callers decide whether that is fatal."""
        staged = self._stage_for_orbit(image, mask, self.VIEW_SIZE)
        template = self.workflows.load("angles")
        graph = build_workflow(template, {
            "image": self.comfy.upload_image(staged, prefix),
            "frames": frames or self.VIEW_FRAMES,
            "seed": int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF})
        graph = self._apply_hardware_limits(graph, job)
        self._free_vram(job)
        # sv3d_u commits ~9.4 GB: drop ComfyUI's cached models first, the same
        # guard the other heavy models get.
        self._drop_comfy_cache()
        # NOT run_graph: it returns only the FIRST image, which for an orbit
        # is (approximately) the view we started from.
        return self.comfy.wait_for_output_all(self.comfy.submit(graph))

    def _render_viewpoints(self, job: Job, asset_id: str, image: Image.Image,
                           instruction: str, scene: str | None,
                           positive: str, negative: str,
                           real: bool) -> dict[str, Any] | None:
        """Real viewpoint synthesis: orbit the subject with SV3D and return
        the requested views as their own assets plus one labelled contact
        sheet. None when this machine has no viewpoint engine — the caller
        then approximates and says so."""
        if not real:
            return None
        wanted = quality.view_count(instruction)
        ok, why = self._template_runnable("angles")
        if not ok and self.settings.auto_install and "not downloaded" in why:
            job.log("info", "[stage] models — fetching the multi-view engine "
                            "(SV3D, ~9.4 GB); this happens once")
            try:
                self._ensure_model("sv3d", job)
                ok, why = self._template_runnable("angles")
            except Exception as exc:  # noqa: BLE001 — fall back, never fail
                ok, why = False, str(exc)
        if not ok:
            job.log("info", f"True multi-view synthesis is unavailable "
                            f"({why})")
            return None
        try:
            self._require_comfy(job)
            job.log("info", f"[stage] render — orbiting the subject for "
                            f"{wanted} viewpoint(s) (SV3D)")
            # Whole-subject matte, NOT a SAM part mask: _stage_for_orbit
            # crops to the mask bbox, so a part mask orbits that part. This
            # is the same failure that made an avatar orbit a bikini.
            mask = None
            if self._pack_active("rmbg"):
                mask = self._region_mask(image, "BiRefNetRMBG", {
                    "model": self._matte_model(instruction or "subject"),
                    "sensitivity": 1.0, "mask_blur": 0, "mask_offset": 0,
                    "invert_output": False, "refine_foreground": True,
                    "background": "Alpha", "background_color": "#222222"})
                if mask is not None and self._mask_fraction(mask) < 0.04:
                    job.log("info", "The cut-out is too small to be the "
                                    "subject; orbiting the full frame")
                    mask = None
            if mask is None:
                try:
                    mask = self.segmentation.propose_mask(
                        image, instruction or "the main subject")
                except Exception as exc:  # noqa: BLE001 — staging is a bonus
                    job.log("info", f"Subject cut-out unavailable ({exc}); "
                                    "orbiting the full frame")
            files = self._orbit_from_photo(job, image, mask)
        except (BackendUnavailableError, WorkflowRuntimeError,
                WorkflowValidationError, PermanentError) as exc:
            job.log("error", f"Multi-view render failed: {exc}")
            self._diagnose_and_record(job, "angles", instruction, str(exc))
            return None
        total = len(files)
        # The azimuth comes from the frames ACTUALLY rendered — the hardware
        # clamp may have executed fewer than the template asked for.
        named = quality.requested_azimuths(instruction)
        if named:
            # The request names a SPECIFIC viewpoint. The old behaviour
            # spread three picks across a ±60° swing, so "show her from the
            # side" rendered -34°, 0° and +34° and the side view was
            # unreachable by construction (D2). Render what was asked, with
            # the original view alongside for comparison, and say plainly
            # that far angles are the engine's invention.
            picks = [0]    # the original view rides along for comparison
            for az in named:
                idx = round(az / 360 * total) % total
                if idx not in picks:
                    picks.append(idx)
            job.log("info", "The request names a specific viewpoint — "
                            "rendering "
                            + ", ".join(f"{az}°" for az in named)
                            + " directly. Past about 60° the engine has "
                              "never seen that side of the subject and "
                              "invents it; judge the result with that in "
                              "mind")
        else:
            # A full turntable only when one was asked for. Otherwise stay
            # within a modest swing: SV3D invents the back of a subject it
            # never saw, and past roughly ±60° a face or a detailed object
            # stops being itself.
            span = 360.0 if re.search(r"360|turntable|orbit|all\s+(the\s+)?"
                                      r"(sides|angles)|around",
                                      instruction or "",
                                      re.IGNORECASE) else 120.0
            picks = self._orbit_frames(total, wanted, span)
        decoded: list[tuple[int, Image.Image, bytes, str]] = []
        for idx in picks:
            data, fname = files[idx]
            decoded.append((idx, Image.open(io.BytesIO(data)).convert("RGB"),
                            data, fname))
        # VERIFY — this route used to skip the quality pipeline entirely: it
        # returned a garment floating on grey three times and logged
        # "Completed" (D2). At minimum, the subject must be IN the output.
        job.log("info", "[stage] verify — checking the subject survived the "
                        "orbit")
        present = self._views_contain_subject(
            job, [im for _, im, _, _ in decoded])
        if present is False:
            job.log("error", "The rendered viewpoints do not contain a "
                             "recognisable subject — the orbit locked onto "
                             "something else. This result is NOT being "
                             "saved as a success.")
            self._diagnose_and_record(job, "angles", instruction,
                                      "subject absent from rendered views")
            return None
        view_ids: list[str] = []
        frames: list[Image.Image] = []
        labels: list[str] = []
        for idx, frame_img, data, fname in decoded:
            azimuth = round(idx * 360 / max(1, total))
            asset = self.store.save_upload(
                f"view_{azimuth:03d}_{job.id}{Path(fname).suffix or '.png'}",
                data, meta={"synthetic": True, "engine": "sv3d",
                            "azimuth": azimuth, "source_asset": asset_id})
            view_ids.append(asset.id)
            frames.append(frame_img)
            labels.append("original view" if azimuth == 0
                          else f"{azimuth}° around")
        sheet_path = self.store.new_version_path(asset_id)
        self._contact_sheet(frames, labels).save(sheet_path, format="PNG")
        version = self.store.add_edit_version(
            asset_id, str(sheet_path), instruction, "comfyui-angles",
            meta={"is_mock": False, "views": view_ids, "synthetic": True,
                  "engine": "sv3d",
                  "verified": {"subject_present": present}})
        job.log("info", f"[stage] save — {len(view_ids)} viewpoint(s) saved "
                        f"as their own images, plus a labelled sheet "
                        f"({version.id})")
        return {
            "workflow": "angles (SV3D multi-view) template",
            "model": "sv3d_u",
            "result": {"version_id": version.id, "asset_id": asset_id,
                       "kind": "views", "views": view_ids,
                       "adapter": "comfyui-angles", "is_mock": False,
                       "verified": {"subject_present": present}},
        }

    def _views_contain_subject(self, job: Job,
                               frames: list[Image.Image]) -> bool | None:
        """Whether the rendered viewpoints still contain the subject.

        Mattes the farthest-from-original frame (the most likely to have
        lost the person) and requires a plausible subject share. None when
        no matting engine is available — unknown, not a pass; the caller
        proceeds but the result carries the verdict."""
        if not frames:
            return False
        if not self._pack_active("rmbg"):
            job.log("info", "No matting engine available to verify the "
                            "views — the subject-present check could not "
                            "run")
            return None
        probe = frames[-1]
        try:
            matte = self._region_mask(probe, "BiRefNetRMBG", {
                "model": self._matte_model("person subject"),
                "sensitivity": 1.0, "mask_blur": 0, "mask_offset": 0,
                "invert_output": False, "refine_foreground": False,
                "background": "Alpha", "background_color": "#222222"})
        except Exception as exc:  # noqa: BLE001 — unknown beats a lie
            job.log("info", f"Subject-present check unavailable ({exc})")
            return None
        if matte is None:
            return None
        share = self._mask_fraction(matte)
        job.log("info", f"Subject occupies {share * 100:.1f}% of the "
                        "rendered view")
        return share >= 0.05

    # Overlap between rendered windows. Enough frames for a fade nobody
    # notices; more would cost re-rendered frames for no visible gain.
    MOTION_OVERLAP = 8

    def _motion_canvas(self, source: tuple[int, int]) -> tuple[int, int]:
        """Render size for a motion transfer: the driving video's aspect,
        snapped to /16 (VACE's step) and held inside this machine's budget."""
        budget = render_budget(self.hardware)
        cap = min(int(budget.get("max_video_side", 480) or 480), 832)
        sw, sh = source
        scale = min(cap / max(1, max(sw, sh)), 1.0)
        w = max(256, int(sw * scale) // 16 * 16)
        h = max(256, int(sh * scale) // 16 * 16)
        return w, h

    def _matte_frames(self, job: Job, frames: list[Image.Image],
                      size: tuple[int, int]) -> str | None:
        """Per-frame mattes of the person in the driving video, uploaded as
        one animated file.

        This is what keeps the driving scene: VACE regenerates wherever the
        mask is 1 and copies the driving pixels wherever it is 0, so the
        person is replaced and the room is not. Returns None when matting is
        unavailable — the caller then renders the whole frame instead and
        says so."""
        if not self._pack_active("rmbg"):
            return None
        try:
            scaled = [f.convert("RGB").resize(size, Image.Resampling.LANCZOS)
                      for f in frames]
            name = self.comfy.upload_frames(scaled, "motion_src")
            graph = {
                "1": {"class_type": "LoadImage", "inputs": {"image": name}},
                "2": {"class_type": "BiRefNetRMBG",
                      "inputs": {"image": ["1", 0], "model": "BiRefNet_lite",
                                 "sensitivity": 1.0, "mask_blur": 4,
                                 "mask_offset": 6, "invert_output": False,
                                 "refine_foreground": False,
                                 "background": "Alpha",
                                 "background_color": "#222222"}},
                "3": {"class_type": "MaskToImage", "inputs": {"mask": ["2", 1]}},
                "4": {"class_type": "SaveAnimatedWEBP",
                      "inputs": {"images": ["3", 0],
                                 "filename_prefix": "pf_motion_mask",
                                 "fps": 16.0, "lossless": True,
                                 "quality": 100, "method": "default"}},
            }
            job.log("info", f"[stage] mask — finding the person in "
                            f"{len(frames)} frame(s) so the scene behind them "
                            "is kept exactly")
            prompt_id = self.comfy.submit(graph)
            data, _fname = self.comfy.wait_for_output_file(prompt_id)
            masks = []
            with Image.open(io.BytesIO(data)) as m:
                for i in range(getattr(m, "n_frames", 1)):
                    m.seek(i)
                    masks.append(m.convert("RGB"))
            return self.comfy.upload_frames(masks, "motion_mask")
        except Exception as exc:  # noqa: BLE001 — fall back to whole-frame
            job.log("info", f"Could not isolate the person ({exc}); the whole "
                            "frame will be re-rendered, so the background "
                            "will change too")
            return None

    def _handle_motion_transfer(self, job: Job) -> dict[str, Any]:
        """Make the person in a still photo perform a driving video's motion.

        Rendered in overlapping windows and cross-faded: VACE keeps a whole
        window in VRAM at once, so a long clip is not slow, it is impossible.
        Splitting costs time and keeps quality, which is the trade the user
        asked for."""
        p = job.payload
        ref_id, drive_id = p["reference_asset_id"], p["driving_asset_id"]
        self._require_comfy(job)
        self._require_video_capable(job)
        self._log_eta(job)

        reference = self.open_asset_image(ref_id)
        cap_frames = max(1, int(p.get("max_frames") or 0)) if p.get("max_frames") \
            else None
        frames, fps = self.open_asset_frames(drive_id, max_frames=cap_frames)
        w, h = self._motion_canvas(frames[0].size)
        window = motion.fit_window(w, h, len(frames))
        chunks = motion.plan_chunks(len(frames), window, self.MOTION_OVERLAP)
        job.log("info", f"[stage] plan — {len(frames)} driving frame(s) at "
                        f"{fps:.0f} fps → {w}×{h}, "
                        + (f"one render of {window} frames"
                           if len(chunks) == 1 else
                           f"{len(chunks)} overlapping renders of up to "
                           f"{window} frames (this machine cannot hold more "
                           "in memory at once)"))

        # Template choice is a real quality decision, so it is made on the
        # clip's length rather than guessed. A short clip gets the full
        # 20-step render. A long one would spend hours that way, so it uses
        # the CausVid distill LoRA — visibly a little softer per frame, but
        # the difference between a finished clip and no clip.
        fast = (p.get("fast") if p.get("fast") is not None
                else len(chunks) > 2)
        name = "motion_transfer_fast" if fast else "motion_transfer"
        if fast and not self._template_runnable(name)[0]:
            fast, name = False, "motion_transfer"
        template = self.workflows.load_named(name)
        for model in template.get("required_models", []):
            self._ensure_model(model, job)
        job.log("info", f"[stage] plan — {'fast' if fast else 'full-quality'} "
                        f"render ({template['graph']['12']['inputs']['steps']} "
                        f"steps per part)"
                        + (" — the clip is long enough that the full-quality "
                           "path would take hours" if fast else ""))
        positive = (p.get("prompt") or "").strip() or \
            "the same person, same clothing, natural motion, photorealistic"
        keep_scene = bool(p.get("preserve_background", True))
        mask_name = (self._matte_frames(job, frames, (w, h))
                     if keep_scene else None)
        if keep_scene and mask_name is None:
            job.log("info", "The driving video's background cannot be kept "
                            "on this run — the whole frame is being rebuilt")

        # A CLEAN ComfyUI for the video model. The WAN stack is ~4 GB of UNet
        # on top of a 5.6 GB text encoder, and on a 16 GB machine that only
        # fits in a process that has not already loaded something else — the
        # matting pass above is enough to tip it, and the failure mode is the
        # OS killing ComfyUI outright rather than a catchable error. Uploaded
        # frames live on disk in ComfyUI's input folder, so they survive this.
        if self._comfy_base() is not None:
            try:
                self._restart_comfy(job, "so the video model gets a clean "
                                         "slate (it needs almost all of this "
                                         "machine's memory)")
            except TransientError:
                job.log("info", "Could not restart ComfyUI — continuing with "
                                "the session that is already running")

        windows: list[list[Image.Image]] = []
        for idx, (start, end) in enumerate(chunks, 1):
            if job.cancel_requested:
                job.log("info", "Stopped at your request")
                break
            piece = frames[start:end]
            length = motion.align_length(len(piece))
            piece = piece[:length]
            job.log("info", f"[stage] render — part {idx}/{len(chunks)}: "
                            f"frames {start}–{start + length}")
            params: dict[str, Any] = {
                "prompt": positive,
                "reference": self.comfy.upload_image(reference, "motion_ref"),
                "control": self.comfy.upload_frames(
                    [f.resize((w, h), Image.Resampling.LANCZOS) for f in piece],
                    "motion_ctrl", fps),
                "width": w, "height": h, "length": length,
                "strength": float(p.get("strength") or 1.0),
                "seed": int(p.get("seed") or 0) or
                (int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF),
                "fps": float(fps),
            }
            if p.get("negative"):
                params["negative"] = str(p["negative"])[:400]
            if mask_name:
                params["mask"] = mask_name
            else:
                # The template always wires control_masks, so an all-white
                # mask is how "rebuild everything" is expressed.
                params["mask"] = self.comfy.upload_frames(
                    [Image.new("RGB", (w, h), (255, 255, 255))] * length,
                    "motion_allmask", fps)
            graph = build_workflow(template, params)
            self._apply_weight_dtype(job, graph)
            self._prepare_heavy_render(job, need_gb=12.0)
            try:
                try:
                    prompt_id = self.comfy.submit(graph)
                    data, _fname = self.comfy.wait_for_output_file(prompt_id)
                except WorkflowRuntimeError as first:
                    tiled = self._miopen_tiled_retry(job, graph, first)
                    if tiled is None:
                        raise
                    prompt_id = self.comfy.submit(tiled)
                    data, _fname = self.comfy.wait_for_output_file(prompt_id)
            except BackendUnavailableError as exc:
                # ComfyUI vanished mid-render. On this machine that is almost
                # always the OS killing it during the 6.3 GB text-encoder
                # load, and reporting a raw socket error would send the user
                # hunting for a network problem that does not exist.
                raise self._comfy_died_midrender(
                    job, "motion_transfer", positive, exc) from exc
            except WorkflowRuntimeError as exc:
                hint = (commit_exhausted_hint(str(exc))
                        or self._miopen_hint(exc))
                self._diagnose_and_record(job, "motion_transfer", positive,
                                          str(exc))
                raise PermanentError(
                    f"Motion transfer failed on part {idx}: "
                    f"{hint or exc}") from exc
            out: list[Image.Image] = []
            with Image.open(io.BytesIO(data)) as clip:
                for i in range(getattr(clip, "n_frames", 1)):
                    clip.seek(i)
                    out.append(clip.convert("RGB"))
            job.log("info", f"Part {idx}/{len(chunks)} rendered "
                            f"({len(out)} frames)")
            windows.append(out)

        if not windows:
            raise PermanentError("No part of the clip rendered.")
        final = motion.assemble(windows, chunks[:len(windows)])
        job.log("info", f"[stage] save — joining {len(windows)} part(s) into "
                        f"{len(final)} frames")
        out_path = self.settings.assets_dir / f"motion_{job.id}.mp4"
        video_io.write_video(final, out_path, fps=fps)
        asset = self.store.save_upload(
            f"motion_{job.id}.mp4", out_path.read_bytes(),
            meta={"motion_from": drive_id, "reference": ref_id,
                  "parts": len(windows), "preserved_background": bool(mask_name)})
        out_path.unlink(missing_ok=True)
        job.log("info", f"Saved the animated clip as asset {asset.id}")
        return {"asset_id": asset.id, "kind": "video",
                "frames": len(final), "fps": round(fps, 2),
                "parts": len(windows),
                "preserved_background": bool(mask_name),
                "resolution": f"{w}x{h}", "adapter": "comfyui-motion",
                "is_mock": False}

    def _animate_current(self, job: Job, source_asset_id: str,
                         image: Image.Image, instruction: str,
                         positive: str) -> str:
        """ANIMATE operation: turn the (possibly already-edited) still into a
        WAN image-to-video render. Returns the new video asset id."""
        self._require_comfy(job)
        job.log("info", "[stage] animate — planning motion, then WAN "
                        "image-to-video (identity preserved from the still)")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        frame = self.store.save_upload(f"animate_src_{job.id}.png",
                                       buf.getvalue())
        motion = f"{positive}. Camera holds steady; motion: {instruction}"
        asset, _pid, _w, _h, _len = self._render_video_asset(
            job, frame.id, motion)
        return asset.id

    @staticmethod
    def _video_dims_for(source: tuple[int, int], max_side: int,
                        commit_gb: float | None) -> tuple[int, int]:
        """The render size ADAPTED to the source image: same aspect ratio,
        fitted into the hardware's video cap, 16-aligned. When Windows'
        commit headroom is already tight, step down another 25% up front —
        failing AFTER the multi-minute model load costs far more than
        pixels, and the post-render upscale wins them back."""
        w, h = source
        scale = min(1.0, max_side / max(w, h))
        if commit_gb is not None and commit_gb < 12:
            scale *= 0.75
        return (max(256, int(w * scale) // 16 * 16),
                max(256, int(h * scale) // 16 * 16))

    @staticmethod
    def _video_upscale_target(rendered: tuple[int, int],
                              source: tuple[int, int]) -> tuple[int, int] | None:
        """Post-render upscale target: back toward the SOURCE resolution,
        never more than 2x the render, long side capped at 1536. None when
        the render already matches the source (nothing was lost)."""
        rw, rh = rendered
        sw, sh = source
        if max(sw, sh) <= max(rw, rh):
            return None
        scale = min(max(sw, sh) / max(rw, rh), 2.0, 1536 / max(rw, rh))
        if scale <= 1.05:
            return None
        return (max(8, int(rw * scale) // 8 * 8),
                max(8, int(rh * scale) // 8 * 8))

    def _maybe_upscale_video(self, job: Job, video_asset,
                             rendered: tuple[int, int],
                             source_size: tuple[int, int]):
        """Regain the resolution the hardware cap took away: run every frame
        through the AI upscale model and reassemble the clip at the target
        size. Advisory by contract — any failure keeps the original video."""
        target = self._video_upscale_target(rendered, source_size)
        if target is None:
            return None
        try:
            frames: list[Image.Image] = []
            with Image.open(video_asset.path) as clip:
                n = getattr(clip, "n_frames", 1)
                if n > 81:
                    return None
                template = self.workflows.load("upscale")
                for name in template.get("required_models", []):
                    self._ensure_model(name, job)
                job.log("info", f"[stage] upscale — restoring resolution: "
                                f"{n} frames → {target[0]}×{target[1]} "
                                "(AI-upscaled frame by frame)")
                self._free_vram(job)
                self._drop_comfy_cache()  # drop the ~17 GB WAN stack
                for i in range(n):
                    clip.seek(i)
                    frame_name = self.comfy.upload_image(
                        clip.convert("RGB"), "vup")
                    graph = build_workflow(template, {"image": frame_name})
                    out, _pid = self.comfy.run_graph(graph)
                    frames.append(out.resize(target, Image.Resampling.LANCZOS))
                    if (i + 1) % 12 == 0:
                        job.log("info", f"Upscaled {i + 1}/{n} frames")
            # Was written lossy at quality=90, which is exactly the setting
            # that silently drops near-identical frames (25 in, 23 out).
            upscaled = self.store.save_upload(
                f"video_upscaled_{job.id}.webp",
                video_io.encode_animation(frames, fps=24.0),
                meta={"upscaled_from": video_asset.id,
                      "resolution": f"{target[0]}x{target[1]}"})
            job.log("info", f"Video upscaled to {target[0]}×{target[1]} — "
                            "both versions are in the gallery")
            return upscaled
        except Exception as exc:  # noqa: BLE001 — enhancement is advisory
            job.log("info", f"Video upscale skipped: {exc}")
            return None

    def _render_video_asset(self, job: Job, asset_id: str, prompt: str,
                            width: int | None = None,
                            height: int | None = None,
                            length: int = 49):
        """Animate an image asset with the versioned WAN template. Shared by
        the video job and avatar renders. Returns (asset, prompt_id, w, h, l).
        Width/height None = adapt to the source image's aspect ratio within
        the hardware budget (the resolution-adaptive path)."""
        template = self.workflows.load("video")
        job.log("info", "[stage] models — WAN video models "
                        "(first run downloads ~18 GB, verified)")
        if not self.settings.auto_install:
            missing = [m for m in template.get("required_models", [])
                       if not self.registry.is_ready(m)]
            if missing:
                raise PermanentError(
                    "Missing video models (auto-install disabled): "
                    + ", ".join(missing))
        for name in template.get("required_models", []):
            self._ensure_model(name, job)

        job.log("info", "[stage] render — animating (this takes several "
                        "minutes on an 8 GB GPU)")
        image = self.open_asset_image(asset_id)
        image_name = self.comfy.upload_image(image, "video_src")
        b = render_budget(self.hardware)  # template path gets clamped too
        if width is None or height is None:
            commit = available_commit_gb()
            width, height = self._video_dims_for(
                image.size, b["max_video_side"], commit)
            note = (" (low memory headroom — stepped down; the finished "
                    "clip is upscaled back)" if commit is not None
                    and commit < 12 else "")
            job.log("info", f"Video resolution adapted to the image: "
                            f"{width}×{height} from source "
                            f"{image.width}×{image.height}, hardware cap "
                            f"{b['max_video_side']}px{note}")
        else:
            width = max(256, min(int(width), b["max_video_side"]))
            height = max(256, min(int(height), b["max_video_side"]))
        length = max(9, min(int(length), b["max_video_len"]))
        job.log("info", "Video model: wan22-ti2v-5b — the largest "
                        "image-to-video model that fits "
                        f"{self.hardware.vram_gb:g} GB VRAM")
        # Step-down retries: if an earlier attempt took ComfyUI down (OOM),
        # each retry renders smaller so the job converges instead of
        # crash-looping at the same size.
        if job.attempts > 1:
            shrink = 0.75 ** (job.attempts - 1)
            width = max(256, int(width * shrink) // 16 * 16)
            height = max(256, int(height * shrink) // 16 * 16)
            length = max(9, int(length * shrink) // 4 * 4 + 1)
            job.log("info", f"Retry attempt {job.attempts}: stepping down to "
                            f"{width}×{height}, {length} frames")
        graph = build_workflow(template, {
            "prompt": prompt,
            "image": image_name,
            "seed": int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF,
            "width": width, "height": height, "length": length,
        })
        try:
            self._free_vram(job)
            # Choose the precision the 10 GB UNet is held at, and drop models
            # ComfyUI still has cached from earlier renders — leftovers under
            # the WAN stack are what OOM-kill it. (Cached weights are kept
            # when they are the SAME ones this graph wants: the peak is
            # identical either way, and reloading them costs minutes.)
            self._prepare_graph(job, graph)
            # The WAN stack commits ~12+ GB while loading. Warn BEFORE the
            # multi-minute load when Windows' commit budget (free RAM +
            # paging file) can't carry it — the failure would be OS error
            # 1455, which no retry can fix.
            commit = available_commit_gb()
            if commit is not None and commit < 12:
                job.log("info", f"Low virtual memory: only {commit:.1f} GB "
                                "commit headroom for a ~12 GB video stack. "
                                "If this render fails with OS error 1455, "
                                "enlarge the Windows paging file or close "
                                "memory-hungry apps.")
            try:
                prompt_id = self.comfy.submit(graph)
                data, filename = self.comfy.wait_for_output_file(prompt_id)
            except WorkflowRuntimeError as first:
                tiled = self._miopen_tiled_retry(job, graph, first)
                if tiled is None:
                    raise
                prompt_id = self.comfy.submit(tiled)
                data, filename = self.comfy.wait_for_output_file(prompt_id)
        except BackendUnavailableError as exc:
            raise self._comfy_died_midrender(job, "video", prompt, exc) from exc
        except WorkflowRuntimeError as exc:
            self._diagnose_and_record(job, "video", prompt, str(exc))
            hint = commit_exhausted_hint(str(exc)) or self._miopen_hint(exc)
            if hint:
                raise PermanentError(f"Video render failed: {hint}") from exc
            raise PermanentError(
                f"Video render failed: {exc}. If ComfyUI cannot find the WAN "
                "nodes, update ComfyUI to a recent version.") from exc

        job.log("info", "[stage] save — storing the animation")
        ext = Path(filename).suffix or ".webp"
        asset = self.store.save_upload(f"video_{job.id}{ext}", data)
        job.log("info", f"Saved video as asset {asset.id} "
                        f"({width}x{height}, {length} frames)")
        self._check_video_kept_subject(job, image, asset)
        return asset, prompt_id, width, height, length

    # How far the first frame may drift from the photograph before the clip
    # is no longer that photograph animated. Mean absolute RGB difference at
    # 64x64, calibrated on this machine's own pictures:
    #
    #   0.022 - 0.039   the SAME photo with a real edit applied to it
    #   0.206           a photo against flat grey
    #   0.230 - 0.250   two unrelated photographs, any pair
    #   0.376           both clips that came back as a different person
    #
    # 0.18 sits above every same-photo pair and below every unrelated one,
    # with the observed failures at twice the limit. Deliberately loose: the
    # point is to catch a STRANGER, not to grade the animation.
    _VIDEO_DRIFT_LIMIT = 0.18

    def _check_video_kept_subject(self, job: Job, source: Image.Image,
                                  asset: Any) -> float | None:
        """Say so when the animation is not of the photograph it was given.

        The wiring is correct — the executed graph carries the uploaded frame
        into WanImageToVideo's start_image — but wan22-ti2v-5b at these
        settings does not always hold the subject, and twice returned a clean
        video of a completely different person after twenty minutes. Nothing
        downstream looks at the pixels, so the job reported success. This
        compares frame one with the source and states the number; it never
        fails the job, because a drifted clip is still the only clip the user
        has and deleting it would help nobody."""
        try:
            with Image.open(asset.path) as clip:
                clip.seek(0)
                first = clip.convert("RGB").resize((64, 64), Image.Resampling.LANCZOS)
        except Exception:  # noqa: BLE001 — advisory only, never fails a job
            return None
        ref = source.convert("RGB").resize((64, 64), Image.Resampling.LANCZOS)
        a = [p / 255.0 for p in ref.tobytes()]
        b = [p / 255.0 for p in first.tobytes()]
        drift = sum(abs(x - y)
                    for x, y in zip(a, b, strict=True)) / max(1, len(a))
        if drift > self._VIDEO_DRIFT_LIMIT:
            job.log("info",
                    f"The animation drifted a long way from your photograph "
                    f"(first frame differs by {drift:.0%}) — this model keeps "
                    f"motion better than it keeps a likeness, so the person "
                    f"in the clip may not be the person you uploaded")
        else:
            job.log("info", f"The first frame still matches your photograph "
                            f"(differs by {drift:.0%})")
        return drift

    def _handle_video(self, job: Job) -> dict[str, Any]:
        """Image-to-video via the versioned WAN template."""
        p = job.payload
        self._require_comfy(job)
        self._require_video_capable(job)
        self._log_eta(job, required_models=["wan22-ti2v-5b", "wan-umt5-xxl",
                                            "wan22-vae"])
        asset, prompt_id, width, height, length = self._render_video_asset(
            job, p["asset_id"], p.get("prompt", ""),
            width=p.get("width"), height=p.get("height"),
            length=p.get("length", 49))
        result = {"asset_id": asset.id, "prompt_id": prompt_id,
                  "width": width, "height": height, "length": length}
        if p.get("upscale", True):
            try:
                source_size = self.open_asset_image(p["asset_id"]).size
            except Exception:  # noqa: BLE001 — upscale is advisory
                source_size = (width, height)
            upscaled = self._maybe_upscale_video(
                job, asset, (width, height), source_size)
            if upscaled is not None:
                result.update({"asset_id": upscaled.id,
                               "raw_asset_id": asset.id,
                               "upscaled": True})
        return result

    # -- digital human intake (see docs/digital_human_pipeline) --------------------
    VIEW_BINS = ["front", "front-right", "right", "back-right",
                 "back", "back-left", "left", "front-left"]

    def _classify_view(self, image: Image.Image) -> str:
        """Which way is the person facing? One of VIEW_BINS (or 'unknown')."""
        if self.critic is None:
            return "unknown"
        try:
            # The enum includes "unknown" on purpose: a schema without an
            # escape hatch would force the model to invent a bin when the
            # photo genuinely does not show one.
            text = ask_with_schema(self.critic, image, (
                "From which side is the person in this photo captured? Reply "
                "with ONLY JSON: {\"view\": \"<one of: "
                + ", ".join(self.VIEW_BINS) + ">\"}"),
                {"type": "object",
                 "properties": {"view": {
                     "type": "string",
                     "enum": [*self.VIEW_BINS, "unknown"]}},
                 "required": ["view"]})
            view = str(json.loads(text).get("view", "")).strip().lower()
            return view if view in self.VIEW_BINS else "unknown"
        except (CriticUnavailable, json.JSONDecodeError, AttributeError):
            return "unknown"

    def _confident_view(self, image: Image.Image) -> tuple[str, bool]:
        """(view, whether two independent asks agreed).

        The classifier is a vision model and is NOT deterministic: the same
        photograph came back 'left' on one run and 'front' on the next, and a
        later run put all nine photos of a set in 'front'. That matters far
        beyond a label — a photo binned as 'back' is spliced into the orbit at
        180 degrees, so a front view ends up painted onto the back of the
        model and, when multi-view conditioning was still on, deforming it.

        Asking twice does not make the model deterministic; it makes the
        pipeline only ACT on answers that repeat. 'front' is the honest
        fallback: it is overwhelmingly the common case for a portrait, and it
        is the one bin where being wrong costs least."""
        first = self._classify_view(image)
        if first == "unknown":
            return "front", False
        second = self._classify_view(image)
        return (first, True) if second == first else ("front", False)

    @staticmethod
    def _mask_fraction(mask: Image.Image) -> float:
        """How much of the frame a mask covers, 0..1."""
        grey = mask.convert("L")
        hist = grey.histogram()
        lit = sum(hist[128:])
        return lit / max(1, grey.width * grey.height)

    @staticmethod
    def _focus_score(image: Image.Image) -> float:
        """How sharp a photo is — edge energy, higher is sharper.

        Used to pick the best photo out of a set rather than the first one.
        Resized first so the score compares like with like across a dataset
        of mixed resolutions."""
        grey = image.convert("L").resize((256, 256), Image.Resampling.LANCZOS)
        return float(ImageStat.Stat(grey.filter(ImageFilter.FIND_EDGES)).var[0])

    # Bins ordered by how much of the subject's front they show. SV3D was
    # trained to orbit FROM a front view; started from behind it invents the
    # face, which is the one thing an avatar cannot get wrong.
    _ORBIT_PREFERENCE = ("front", "front-right", "front-left",
                         "right", "left", "back-right", "back-left", "back")

    _FRONTAL_BINS = ("front", "front-right", "front-left")

    @staticmethod
    def _framing_penalty(mask: Image.Image) -> float:
        """How badly the frame cuts the subject off. 0 is clean, 1 is severe.

        Purely geometric on purpose. The obvious alternative — asking the
        vision model "is the whole person visible?" — was tried and measured
        useless here: it answered yes for all nine photos of a set including
        one plainly cropped at the hips. What IS measurable is how far the
        silhouette runs along a frame edge, and a subject running along more
        than half an edge really is cut off there.

        A squat silhouette makes it worse: a standing person is much taller
        than wide, so a wide short cut-out that also hugs an edge is a
        fragment, and a fragment reconstructs as a fragment."""
        m = mask.convert("L").point(lambda v: 255 if v > 127 else 0)
        w, h = m.size
        px = cast(_GreyPixels, m.load())
        box = m.getbbox()
        if not box or w < 2 or h < 2:
            return 1.0
        runs = [sum(1 for x in range(w) if px[x, 0]) / w,
                sum(1 for x in range(w) if px[x, h - 1]) / w,
                sum(1 for y in range(h) if px[0, y]) / h,
                sum(1 for y in range(h) if px[w - 1, y]) / h]
        aspect = (box[3] - box[1]) / max(1, box[2] - box[0])
        # Feet at the bottom of a full-length shot touch the edge too, so the
        # penalty only bites once the contact is wide.
        cut = max(0.0, max(runs) - 0.35) / 0.65
        squat = 1.0 if aspect < 1.4 else 0.0
        return min(1.0, cut * (1.0 + squat))

    def _best_orbit_source(self, coverage: dict[str, list[str]],
                           asset_ids: list[str], focus: dict[str, float],
                           framing: dict[str, float] | None = None) -> str:
        """Which photo to orbit from: the cleanest sharp frontal one.

        The old code took `coverage["front"][0]` or `asset_ids[0]` — so a
        blurry snapshot beat a sharp one purely by arriving first, and every
        photo after the first changed nothing at all."""
        cut = framing or {}

        def rank(aid: str) -> tuple:
            bins = [v for v in self._ORBIT_PREFERENCE
                    if aid in coverage.get(v, [])]
            preference = -self._ORBIT_PREFERENCE.index(bins[0]) if bins else -99
            # Framing first, in coarse steps so a slightly cropped but much
            # sharper photo still wins; only real fragments are demoted.
            return (-round(cut.get(aid, 0.0) * 2) / 2,
                    preference, focus.get(aid, 0.0))

        pool = [a for v in self._FRONTAL_BINS for a in coverage.get(v, [])]
        if not pool:
            pool = [a for v in self._ORBIT_PREFERENCE
                    for a in coverage.get(v, [])]
        return max(pool or asset_ids, key=rank)

    def _best_face_photo(self, coverage: dict[str, list[str]],
                         asset_ids: list[str],
                         focus: dict[str, float]) -> str:
        """The identity reference: sharpest frontal face available."""
        for view in ("front", "front-right", "front-left"):
            if coverage.get(view):
                return max(coverage[view], key=lambda a: focus.get(a, 0.0))
        return max(asset_ids, key=lambda a: focus.get(a, 0.0))

    # What a render actually needs to look like the person. Written as prompt
    # words rather than categories on purpose: "warm fair skin, freckles,
    # long blonde hair" steers a diffusion model, where a label like an
    # ethnicity class or a weight in kilos does not — and a weight read off a
    # photo would be invented. Age is kept as a RANGE and marked an estimate,
    # because that is what a look-based guess honestly is.
    _APPEARANCE_QUESTION = (
        "Describe only what this person visibly looks like, as short phrases "
        "an artist would use to draw them. Reply with ONLY JSON: "
        '{"age_range": "<e.g. mid 20s to early 30s>", '
        '"build": "<e.g. slim and athletic / broad-shouldered / heavy-set>", '
        '"height_impression": "<e.g. tall / average / petite>", '
        '"skin_tone": "<e.g. deep brown with warm undertones>", '
        '"hair": "<length, texture, colour>", '
        '"face": "<face shape and notable features>", '
        '"distinctive": "<glasses, freckles, beard, tattoos, or empty>"}')

    def _appearance_profile(self, job: Job,
                            asset_ids: list[str]) -> dict[str, str]:
        """How this person looks, in words a renderer can use.

        Read from the photos themselves so an identity render is not starting
        from nothing but a face crop. Every value is the vision model's
        estimate from appearance alone — recorded as such, never as fact."""
        if self.critic is None or not asset_ids:
            return {}
        merged: dict[str, list[str]] = {}
        # Two photos is enough to steady the estimate without doubling the
        # intake time; the sharpest are already at the front of the list.
        for aid in asset_ids[:2]:
            try:
                text = self.critic.ask(self.open_asset_image(aid),
                                       self._APPEARANCE_QUESTION)
                data = json.loads(text)
            except Exception as exc:  # noqa: BLE001 — appearance is a bonus
                # Deliberately broad: this only enriches the render prompt, and
                # nothing about it is worth failing a consented dataset over.
                job.log("info", f"Appearance read unavailable: {exc}")
                continue
            if not isinstance(data, dict):
                continue
            for key, value in data.items():
                value = str(value).strip()
                if not value or value.lower() in ("none", "n/a", "unknown"):
                    continue
                if _is_placeholder(value):
                    # The model echoed the question instead of answering it.
                    # Seen live: an avatar was saved whose age_range was
                    # literally "<e.g. mid 20s to early 30s>" and whose hair
                    # was "<length, texture, colour>". This text is fed into
                    # identity renders, so a stored placeholder becomes part
                    # of every prompt made from that avatar.
                    job.log("info", f"Ignored an unanswered appearance field "
                                    f"({key}) — the model returned the "
                                    f"question, not a description")
                    continue
                merged.setdefault(key, []).append(value)
        if not merged:
            return {}
        # Keep the first (sharpest photo's) answer for each field; the second
        # only fills gaps.
        profile = {k: v[0] for k, v in merged.items()}
        profile["estimated"] = "true"
        job.log("info", "Appearance (estimated from the photos): "
                        + "; ".join(f"{k} {v}" for k, v in profile.items()
                                    if k != "estimated"))
        return profile

    @staticmethod
    def appearance_phrase(profile: dict[str, Any] | None) -> str:
        """The appearance profile as a prompt fragment."""
        if not profile:
            return ""
        order = ("age_range", "build", "height_impression", "skin_tone",
                 "hair", "face", "distinctive")
        bits = [str(profile[k]).strip() for k in order
                if str(profile.get(k) or "").strip()]
        return ", ".join(bits)

    # How much of an edge the subject must occupy before we call it "cut off
    # by the frame" rather than "standing near the edge". A leg crossing the
    # bottom of the picture covers a good slice of that row; a shoulder
    # brushing the side does not.
    _CUTOFF_FRACTION = 0.12

    def _subject_edges(self, matte: Image.Image) -> dict[str, float]:
        """How much of each frame edge the subject occupies, 0..1."""
        w, h = matte.size
        px = cast(_GreyPixels, matte.convert("L").load())
        thresh = 128
        top = sum(1 for x in range(w) if px[x, 0] > thresh) / max(1, w)
        bottom = sum(1 for x in range(w) if px[x, h - 1] > thresh) / max(1, w)
        left = sum(1 for y in range(h) if px[0, y] > thresh) / max(1, h)
        right = sum(1 for y in range(h) if px[w - 1, y] > thresh) / max(1, h)
        return {"top": top, "bottom": bottom, "left": left, "right": right}

    def _subject_matte(self, image: Image.Image) -> Image.Image | None:
        """The BiRefNet matte of the whole subject, or None off-pack."""
        if not self._pack_active("rmbg"):
            return None
        return self._region_mask(image, "BiRefNetRMBG", {
            "model": self._MATTE_GENERAL, "sensitivity": 1.0, "mask_blur": 0,
            "mask_offset": 0, "invert_output": False,
            "refine_foreground": False, "background": "Alpha",
            "background_color": "#222222"})

    @staticmethod
    def _plain_backdrop(image: Image.Image) -> bool:
        """Is this image already a staged subject on a plain background?

        Judged on the border ring, which is backdrop by construction. Used
        as the escape hatch when no matte can be computed: a photo that is
        already subject-on-plain stages safely uncut, and anything else
        must not — a 3D reconstruction handed a photo WITH its background
        models the photograph itself as a flat slab."""
        import numpy as np
        arr = np.asarray(image.convert("RGB").resize((256, 256)), float)
        ring = np.concatenate([arr[:6].reshape(-1, 3),
                               arr[-6:].reshape(-1, 3),
                               arr[:, :6].reshape(-1, 3),
                               arr[:, -6:].reshape(-1, 3)])
        return float(ring.std(axis=0).max()) < 14.0

    def _cut_edges(self, image: Image.Image) -> dict[str, float]:
        """Which frame edges the subject runs off, and by how much."""
        matte = self._subject_matte(image)
        if matte is None:
            return {}
        return {k: v for k, v in self._subject_edges(matte).items()
                if v >= self._CUTOFF_FRACTION}

    def _is_cut_off(self, image: Image.Image) -> bool:
        return bool(self._cut_edges(image))

    # A standing person is much taller than wide. Below this, a figure is
    # partial — the rigger uses the same threshold to tell a bust from a
    # full body, so the two ends of the pipeline agree on what "whole" is.
    _FULL_FIGURE_ASPECT = 1.9

    def _figure_aspect(self, image: Image.Image) -> float | None:
        """Subject bounding-box height over width, from the BiRefNet matte."""
        matte = self._subject_matte(image)
        if matte is None:
            return None
        box = matte.convert("L").point(
            lambda v: 255 if v > 127 else 0).getbbox()
        if not box:
            return None
        return (box[3] - box[1]) / max(1, box[2] - box[0])

    def _complete_subject(self, job: Job,
                          image: Image.Image) -> Image.Image:
        """Grow the picture until the whole subject is inside it.

        A mesh built from a photo cropped at the thigh produces a person
        cropped at the thigh — the reconstruction can only model what it can
        see. When the subject runs off an edge, outpaint that edge first so
        there is a whole body to reconstruct. The invented part is a guess and
        is logged as one.

        A subject can also be partial WITHOUT touching an edge: a waist-up
        photo with margin all round is the common case, and it used to sail
        through this check and reconstruct as a floating bust. A figure whose
        silhouette is much wider than a standing person's is treated as
        ending at the waist and extended downward the same way.

        Returns the input unchanged whenever the figure is whole or the
        outpaint is unavailable."""
        cut = self._cut_edges(image)
        partial = False
        if not cut:
            aspect = self._figure_aspect(image)
            partial = (aspect is not None
                       and aspect < self._FULL_FIGURE_ASPECT)
            if not partial:
                return image
        w, h = image.size
        # Enough room to hold what is missing without ballooning the canvas —
        # a subject cut at the hips needs roughly its own height again.
        pad = {"left": 0, "right": 0, "top": 0, "bottom": 0}
        for edge in cut:
            span = h if edge in ("top", "bottom") else w
            pad[edge] = int(span * (0.55 if edge == "bottom" else 0.35))
        if partial:
            pad["bottom"] = int(h * 0.9)
            job.log("info", "The photo shows a partial figure (its "
                            "silhouette is squatter than any standing "
                            "person) — extending it downward so a whole "
                            "body can be reconstructed (the added part is "
                            "invented)")
        else:
            job.log("info", "The subject runs off the "
                            + ", ".join(sorted(cut))
                            + " of the frame — extending it to reconstruct "
                              "a whole body (the added part is invented)")
        try:
            template = self.workflows.load("outpaint")
            graph = build_workflow(template, {
                "image": self.comfy.upload_image(image, "complete_src"),
                "prompt": "full body photograph, head to toe, the complete "
                          "figure standing, legs and feet visible, natural "
                          "proportions, plain background",
                "negative": "cropped, cut off, extra limbs, duplicate legs, "
                            "deformed anatomy, floating body parts, second "
                            "person",
                "seed": int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF,
                **pad})
            graph = self._apply_hardware_limits(graph, job)
            self._free_vram(job)
            out, _pid = self.comfy.run_graph(graph)
        except Exception as exc:  # noqa: BLE001 — a partial mesh beats none
            job.log("info", f"Could not extend the frame ({exc}); "
                            "reconstructing from what is visible")
            return image
        # Did it actually finish the body? "No longer touches an edge" is
        # NOT the answer — a live run logged "the body is complete now"
        # while the outpaint had extended the canvas and painted background
        # below the hips, and the mesh shipped as a bust anyway. The honest
        # test is the silhouette's aspect; when it still is not a standing
        # figure, the instruction model gets one shot at zooming the camera
        # out — it rebuilt this same subject's back convincingly, and
        # inventing the lower body is the same class of task.
        aspect = self._figure_aspect(out)
        if aspect is not None and aspect < self._FULL_FIGURE_ASPECT:
            job.log("info", "The extended frame still does not hold a full "
                            "figure — zooming the camera out instead (the "
                            "lower body is invented)")
            # Two attempts, fresh seed each: the zoom-out is a generation,
            # and a single unlucky seed was measured producing a figure
            # still too squat while the next one measured 2.9x.
            for _attempt in range(2):
                zoomed = self._zoom_out_full_body(job, image)
                z_aspect = (self._figure_aspect(zoomed)
                            if zoomed is not None else None)
                if z_aspect is not None and z_aspect > aspect:
                    out, aspect = zoomed, z_aspect
                if aspect >= self._FULL_FIGURE_ASPECT:
                    break
        if aspect is not None:
            if aspect >= self._FULL_FIGURE_ASPECT:
                job.log("info", "The figure is full-length now (silhouette "
                                f"{aspect:.1f}x taller than wide), with "
                                "everything below the original photo "
                                "invented")
            else:
                job.log("info", "The figure could not be completed — the "
                                "mesh will end where the photograph does")
            return out
        still = self._cut_edges(out)
        if still:
            job.log("info", "Extended the frame, but the subject still "
                            f"reaches the {', '.join(sorted(still))} — the "
                            "model will end where the photograph does. A "
                            "full-length shot reconstructs properly.")
        else:
            job.log("info", f"The body is complete now: {image.width}x"
                            f"{image.height} to {out.width}x{out.height}, "
                            "with the part below the crop invented")
        return out

    # The instruction Kontext follows to invent the missing lower body.
    _FULL_BODY_PROMPT = (
        "zoom the camera out to show the person's whole body from head to "
        "toe: the same person in the same outfit, the clothing continuing "
        "naturally into matching bottoms, legs and feet fully visible, "
        "standing, same plain background")

    def _zoom_out_full_body(self, job: Job,
                            image: Image.Image) -> Image.Image | None:
        """A generated full-length view of the subject, or None."""
        ok, _why = self.kontext_ready()
        if not ok:
            return None
        try:
            template = self.workflows.load("kontext")
            graph = build_workflow(template, {
                "prompt": self._FULL_BODY_PROMPT,
                "image": self.comfy.upload_image(image, "fullbody_src"),
                "seed": int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF,
            })
            self._free_vram(job)
            self._prepare_graph(job, graph)
            out, _pid = self.comfy.run_graph(graph)
            return out.convert("RGB")
        except Exception as exc:  # noqa: BLE001 — the figure stays partial
            job.log("info", f"Could not zoom out to a full body ({exc})")
            return None

    # Which orbit frame stands in for each Hunyuan3D view input. The mesh
    # model wants front/left/back/right; the orbit runs anticlockwise from
    # the source photo, so these are azimuths, not frame indices.
    _MESH_VIEWS = {"front": 0, "right": 90, "back": 180, "left": 270}
    # Which graph nodes carry each view, so unused ones can be pruned rather
    # than filled with a stand-in. Hunyuan3Dv2ConditioningMultiView declares
    # all four inputs optional and indexes the positional embedding by slot,
    # so a missing view is handled correctly and a WRONG one is not.
    _MV_NODES = {"front": ("2", "6"), "left": ("3", "7"),
                 "back": ("4", "8"), "right": ("5", "9")}
    # How far a frame may sit from a canonical azimuth and still stand in for
    # it. Without a limit, min() always returns something: an orbit covering
    # only 0-90 degrees handed a near-front frame to the BACK input and told
    # the model the two sides of the subject look identical.
    _MESH_VIEW_TOLERANCE = 35.0
    # Silhouette agreement between the finished mesh and the views it was
    # built from. Measured: 0.85 from a clean full-length photograph, 0.51
    # from a cropped mirror selfie with a phone across the chest.
    _MESH_FIT_FLOOR = 0.70

    def _frame_angles(self, frame_ids: list[str]) -> list[tuple[float, str, bool]]:
        """(azimuth, asset id, is a real photograph) for every tagged frame."""
        out: list[tuple[float, str, bool]] = []
        for aid in frame_ids:
            asset = self.store.get_asset(aid)
            meta = (asset.meta or {}) if asset else {}
            if "azimuth" in meta:
                out.append((float(meta["azimuth"]) % 360, aid,
                            not meta.get("synthetic", True)))
        return out

    def _views_for_mesh(self, frame_ids: list[str], real_only: bool = False,
                        tolerance: float | None = None) -> dict[str, str]:
        """The frame standing in for each canonical azimuth, or nothing.

        Reads the azimuth off each frame's own meta rather than assuming an
        even spread — the hardware clamp can render fewer frames than the
        template asked for."""
        angles = [(az, aid) for az, aid, real in self._frame_angles(frame_ids)
                  if real or not real_only]
        if not angles:
            return {}
        limit = self._MESH_VIEW_TOLERANCE if tolerance is None else tolerance
        picked: dict[str, str] = {}
        for name, want in self._MESH_VIEWS.items():
            # Circular distance — 350 deg is 10 deg from front, not 350.
            best = min(angles, key=lambda a: min(abs(a[0] - want),
                                                 360 - abs(a[0] - want)))
            if min(abs(best[0] - want), 360 - abs(best[0] - want)) <= limit:
                picked[name] = best[1]
        return picked

    @staticmethod
    def _subject_aspect(image: Image.Image) -> float:
        """Height over width of the subject in a staged frame.

        The staging step crops each photo to its own subject and blows it up
        to fill the frame, so a head-and-shoulders shot and a full-length one
        arrive the same size. Measured on a real dataset, the photographs fed
        to the mesh had silhouette aspects of 1.8 and 1.46 while the rendered
        views of the same person had 4.45 — three descriptions of three
        different objects, handed to a model that assumes one."""
        grey = Image.new("RGB", image.size, (128, 128, 128))
        diff = ImageChops.difference(image.convert("RGB"), grey).convert("L")
        box = diff.point(lambda v: 255 if v > 18 else 0).getbbox()
        if not box:
            return 0.0
        width = max(1, box[2] - box[0])
        return (box[3] - box[1]) / width

    def _consistent_real_views(self, job: Job,
                               views: dict[str, str]) -> dict[str, str]:
        """Drop real photographs that are not framed like the others.

        Multi-view reconstruction assumes four renders of ONE object at ONE
        scale. Two photographs staged independently are not that."""
        if len(views) < 2:
            return views
        aspects = {name: self._subject_aspect(self.open_asset_image(aid))
                   for name, aid in views.items()}
        front = aspects.get("front") or (
            sorted(aspects.values())[len(aspects) // 2])
        keep = {name: aid for name, aid in views.items()
                if front and 0.75 <= (aspects[name] / front) <= 1.33}
        for name in views.keys() - keep.keys():
            job.log("info",
                    f"The {name} photograph frames the subject differently "
                    f"(silhouette {aspects[name]:.2f} tall against "
                    f"{front:.2f} for the front), so it is used for colour "
                    "but not for shape")
        return keep

    # Colour is sampled per 45° around the subject — twice the density of the
    # four shape bins, because colour views are cheap and every extra camera
    # shrinks the surface that has to make do with its neighbours' colour.
    _COLOUR_BINS = (0, 45, 90, 135, 180, 225, 270, 315)
    _COLOUR_TOLERANCE = 25.0

    def _matte_on_grey(self, job: Job, image: Image.Image) -> Image.Image:
        """The subject alone, composited on neutral grey, via BiRefNet.

        The orbit model bakes hallucinated surroundings into its renders —
        measured on a real dataset, a third of the views were mostly a flat
        wall with a sliver of person in it. The texturer samples anything the
        matte calls subject, so what stands around the subject ends up ON the
        avatar. Returns the input unchanged when matting is unavailable; the
        texturer's own quality gate is the second line of defence."""
        if not self._pack_active("rmbg"):
            return image
        try:
            name = self.comfy.upload_image(image.convert("RGB"), "mesh_matte")
            graph = {
                "1": {"class_type": "LoadImage", "inputs": {"image": name}},
                "2": {"class_type": "BiRefNetRMBG",
                      "inputs": {"image": ["1", 0], "model": "BiRefNet_lite",
                                 "sensitivity": 1.0, "mask_blur": 0,
                                 "mask_offset": 0, "invert_output": False,
                                 "refine_foreground": True,
                                 "background": "Color",
                                 "background_color": "#808080"}},
                "3": {"class_type": "SaveImage",
                      "inputs": {"images": ["2", 0],
                                 "filename_prefix": "pf_mesh_matte"}},
            }
            out, _pid = self.comfy.run_graph(graph)
            return out.convert("RGB")
        except Exception as exc:  # noqa: BLE001 — the gate still protects
            job.log("info", f"Could not matte a colour view ({exc}); "
                            "passing it as rendered")
            return image

    def _colour_photos(self, job: Job,
                       frame_ids: list[str]) -> list[tuple[float, Image.Image]]:
        """(true azimuth, staged image) for up to eight colour views.

        Three rules, each the correction of a measured mistake:

        TRUE azimuths, not bin names. The old code told the texturer a frame
        sat at its bin's canonical angle; the bins tolerate 35° and the
        texturer refined ±18°, so a view could be projected from a camera
        17° away from where the photograph was really taken.

        The frame nearest each bin, deduplicated — a sparse orbit must not
        hand one frame to two bins.

        Synthetic frames are matted first (see _matte_on_grey); the real
        photographs were already staged onto clean grey by the intake."""
        angles = self._frame_angles(frame_ids)
        picked: dict[str, tuple[float, bool, float]] = {}
        for want in self._COLOUR_BINS:
            best = None
            for az, aid, real in angles:
                dist = min(abs(az - want), 360 - abs(az - want))
                if dist <= self._COLOUR_TOLERANCE and (
                        best is None or dist < best[0]):
                    best = (dist, az, aid, real)
            if best is None:
                continue
            dist, az, aid, real = best
            if aid not in picked or dist < picked[aid][2]:
                picked[aid] = (az, real, dist)
        # Front first: the texturer chains its tone-matching outward from the
        # first view, and the front is the one most likely to be the user's
        # own photograph.
        ordered = sorted(picked.items(),
                         key=lambda kv: min(kv[1][0], 360 - kv[1][0]))
        photos: list[tuple[float, Image.Image]] = []
        can_upscale = self._template_runnable("upscale")[0]
        upscaled = 0
        for aid, (az, real, _dist) in ordered:
            image = self.open_asset_image(aid)
            if not real:
                image = self._matte_on_grey(job, image)
            # The orbit stages everything at 576px and the atlas tiles are
            # 1024: plain resampling would fill the texture with blur. The
            # detail-reconstructing upscaler is on disk — use it, and fall
            # back silently because sharpness is a bonus, not a requirement.
            if can_upscale and min(image.size) < 1024:
                try:
                    image = self._render_template_step(job, "upscale",
                                                       image, "")
                    upscaled += 1
                except Exception:  # noqa: BLE001
                    can_upscale = False
            photos.append((float(az), image))
        if upscaled:
            job.log("info", f"[stage] texture — {upscaled} colour view(s) "
                            "upscaled with the detail model before "
                            "projection, so the texture is limited by the "
                            "photograph rather than by the orbit's 576px")
        return photos

    # The instructions Kontext follows to invent a missing camera. Fixed
    # wording, measured working for the back; the sides use the same shape.
    # In this pipeline's convention the camera at azimuth 90° sees the
    # subject's LEFT profile (and 270° the right).
    _VIEW_PROMPTS = {
        "back": (
            "turn the camera to directly behind the person, showing them "
            "from the back: back of the head, back of any hat, hair falling "
            "down the back, back of the clothing, same plain grey "
            "background"),
        "left": (
            "turn the camera to the person's left side, showing their left "
            "profile: left side of the face and of any hat, left ear and "
            "cheek, hair falling over the left shoulder, left arm and the "
            "left side of the clothing, same plain grey background"),
        "right": (
            "turn the camera to the person's right side, showing their "
            "right profile: right side of the face and of any hat, right "
            "ear and cheek, hair falling over the right shoulder, right arm "
            "and the right side of the clothing, same plain grey "
            "background"),
    }

    def _synthesize_view(self, job: Job, which: str) -> Image.Image | None:
        """A generated photograph of the subject from a camera nobody held.

        The orbit model hallucinates walls off the front arc — measured, a
        third of its views were unusable, leaving the back and both sides
        with no camera at all. Kontext, handed the STAGED FRONT photo,
        renders the same person from the missing angle well enough to pass
        the texturer's quality gate; the back view alone lifted real
        coverage from 66% to 77% and replaced the surface-fill smear with
        actual hair, and an honest Blender render showed the uncovered
        SIDES carrying the worst junk on the whole figure. Every generated
        view is judged by the same gate as a real one and clearly labelled
        invented: no photograph of it exists."""
        ok, _why = self.kontext_ready()
        if not ok:
            return None
        front = getattr(self, "_avatar_front_view", None)
        if front is None:
            return None
        try:
            template = self.workflows.load("kontext")
            graph = build_workflow(template, {
                "prompt": self._VIEW_PROMPTS[which],
                "image": self.comfy.upload_image(front, f"{which}_synth"),
                "seed": int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF,
            })
            self._free_vram(job)
            self._prepare_graph(job, graph)
            out, _pid = self.comfy.run_graph(graph)
            # A silent refusal hands back (a near-copy of) the front photo.
            # Painting the FRONT onto the BACK is the one mistake worse
            # than leaving the arc approximate — discard and say so.
            if quality.image_change(front, out) < 0.03:
                job.log("info", f"The generated {which} view came back as "
                                "a copy of the front (the instruction "
                                "model declines some content) — discarded; "
                                "that arc keeps its neighbours' colour")
                return None
            return self._matte_on_grey(job, out.convert("RGB"))
        except Exception as exc:  # noqa: BLE001 — the arc stays approximate
            job.log("info", f"Could not generate a {which} view ({exc}); "
                            "that arc keeps its neighbours' colour")
            return None

    def _synthesize_back_view(self, job: Job) -> Image.Image | None:
        return self._synthesize_view(job, "back")

    @staticmethod
    def _find_blender() -> str | None:
        """blender.exe, from the env override or the standard install dirs."""
        override = os.environ.get("PROMPTFORGE_BLENDER")
        if override and Path(override).exists():
            return override
        for base in (Path(os.environ.get("ProgramFiles",
                                         r"C:\Program Files"))
                     / "Blender Foundation",):
            if base.exists():
                exes = sorted(base.glob("Blender */blender.exe"),
                              reverse=True)
                if exes:
                    return str(exes[0])
        return None

    def _rig_avatar(self, job: Job, glb: bytes) -> tuple[bytes, dict]:
        """Give the textured mesh a posable humanoid skeleton, via Blender.

        Headless: the mesh goes through app/tools/rig_avatar.py, which
        measures the figure (neck and shoulders from band widths, bust
        against full body from the aspect ratio), builds the armature,
        cleans the 60k+ doubled vertices that make Blender's heat solver
        fail, binds — falling back to envelope weights rather than refusing
        — and exports a GLB any DCC or engine can pose. Returns the input
        unchanged when Blender is not installed or anything fails: a rig is
        an upgrade, never a requirement."""
        blender = self._find_blender()
        if blender is None:
            job.log("info", "Blender is not installed, so the mesh ships "
                            "unrigged (install Blender to get a posable "
                            "skeleton for free)")
            return glb, {}
        tool = Path(__file__).resolve().parent.parent / "tools" /             "rig_avatar.py"
        if not tool.exists():
            return glb, {}
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "mesh.glb"
            dst = Path(tmp) / "rigged.glb"
            src.write_bytes(glb)
            job.log("info", "[stage] rig — building a posable skeleton "
                            "(joints measured from the mesh, weights from "
                            "Blender's heat solver)")
            try:
                proc = subprocess.run(
                    [blender, "--background", "--factory-startup",
                     "--python", str(tool), "--", str(src), str(dst)],
                    capture_output=True, text=True, timeout=600)
            except Exception as exc:  # noqa: BLE001
                job.log("info", f"Rigging unavailable ({exc}); the mesh "
                                "ships unrigged")
                return glb, {}
            report: dict = {}
            for line in reversed((proc.stdout or "").strip().splitlines()):
                if line.startswith("{"):
                    try:
                        report = json.loads(line)
                    except json.JSONDecodeError:
                        pass
                    break
            if proc.returncode != 0 or not dst.exists() or "error" in report:
                job.log("info", "Rigging did not complete "
                                f"({(report.get('error') or proc.stderr or '')[:120]}); "
                                "the mesh ships unrigged")
                return glb, {}
            job.log("info", f"Rigged with {report.get('bones')} bones "
                            f"({'full figure' if report.get('full_body') else 'bust'}, "
                            f"{report.get('weights')} weights) — pose it in "
                            "Blender, Unity, Unreal or any glTF viewer with "
                            "bone support")
            return dst.read_bytes(), report

    # The instruction Kontext follows to repaint one rendered view clean.
    # Identity-preserving by model design; the acceptance test below is the
    # deterministic guarantee that it actually cleaned rather than redrew.
    _REFINE_PROMPT = (
        "enhance this into a sharp, clean, photorealistic photograph: "
        "remove all speckles, noise, stray patches and artifacts, make "
        "skin, hair and fabric look real and continuous, keep the person's "
        "identity, pose, clothing, colors and framing exactly the same, "
        "plain grey background")

    @staticmethod
    def _speckle_and_drift(before: Image.Image, after: Image.Image,
                           alpha: Image.Image) -> tuple[float, float, float]:
        """(speckle_before, speckle_after, drift), all on the subject only.

        Speckle is high-frequency energy: how far each pixel sits from its
        own 3x3 median — single-face colour specks light this metric up and
        clean fabric or skin does not. Drift is the mean colour change, the
        guard that the model cleaned the image rather than replaced it."""
        import numpy as np
        size = (512, 512)
        mask = np.asarray(alpha.convert("L").resize(size,
                                                    Image.Resampling.NEAREST)) > 127
        if not mask.any():
            return 0.0, 0.0, 255.0
        out: list[float] = []
        arrs: list[Any] = []
        for img in (before, after):
            small = img.convert("RGB").resize(size, Image.Resampling.LANCZOS)
            grey = small.convert("L")
            med = grey.filter(ImageFilter.MedianFilter(3))
            g = np.asarray(grey, float)
            m = np.asarray(med, float)
            out.append(float(np.abs(g - m)[mask].mean()))
            arrs.append(np.asarray(small, float))
        drift = float(np.abs(arrs[0] - arrs[1]).max(axis=2)[mask].mean())
        return out[0], out[1], drift

    def _refine_texture(self, job: Job, glb: bytes,
                        report: dict) -> tuple[bytes, dict]:
        """Repaint each view of the textured mesh clean, in place.

        Winner-take-all projection from views that partly disagree has a
        quality ceiling: even perfectly aligned, it can only choose pixels,
        never repaint a coherent surface — and rendered honestly the result
        carries specks and patchwork that read as noise. This is the
        standard render -> diffusion-refine -> reproject loop (TEXTure,
        Paint3D), with one structural advantage: the atlas tiles ARE
        per-view camera frames, so the rasteriser renders each view in
        pixel-registration with its own tile and re-projection is a paste —
        no alignment step exists to go wrong.

        Every view must PROVE the refinement helped: the speckle metric
        (distance from the 3x3 median) has to fall while the mean colour
        drift stays bounded, or that view keeps its projected tile."""
        needed = ("mappings", "tile", "cols", "views_used", "view_bbox",
                  "tile_crops")
        if not all(report.get(k) is not None for k in needed):
            return glb, {}
        ok, _why = self.kontext_ready()
        if not ok:
            return glb, {}
        base = Path(self.settings.comfyui_dir) if self.settings.comfyui_dir \
            else None
        if base is None or not base.exists():
            return glb, {}
        tool = Path(__file__).resolve().parent.parent / "tools" / \
            "texture_mesh.py"
        try:
            python = self._comfy_python(base)
        except Exception:  # noqa: BLE001 — refinement is a bonus
            return glb, {}
        n = len(report["views_used"])
        job.log("info", f"[stage] texture — repainting {n} view(s) of the "
                        "textured mesh into clean photographs (each view "
                        "ships only if its measured noise actually fell)")
        refined: list[float] = []
        skipped: list[dict] = []
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            current = work / "current.glb"
            current.write_bytes(glb)
            (work / "report.json").write_text(json.dumps(report))
            template = self.workflows.load("kontext")
            # Free once, then let the SAME model serve every view: a
            # measured first run that dropped the cache per iteration spent
            # 67 minutes mostly reloading Kontext, against ~2 minutes of
            # actual repainting.
            self._free_vram(job)
            for i in range(n):
                raster = work / f"raster_{i}.png"
                alpha = work / f"alpha_{i}.png"
                proc = subprocess.run(
                    [python, str(tool), "rasterize", str(current),
                     str(work / "report.json"), str(i), str(raster),
                     str(alpha), "1536"],
                    capture_output=True, text=True, timeout=420)
                if proc.returncode != 0 or not raster.exists():
                    skipped.append({"view": i, "why": "rasterize failed"})
                    continue
                try:
                    before = Image.open(raster).convert("RGB")
                    small = before.copy()
                    small.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                    graph = build_workflow(template, {
                        "prompt": self._REFINE_PROMPT,
                        "image": self.comfy.upload_image(
                            small, f"refine_{i}"),
                        "seed": int.from_bytes(os.urandom(4),
                                               "big") & 0x7FFFFFFF,
                    })
                    self._prepare_graph(job, graph)
                    out, _pid = self.comfy.run_graph(graph)
                    # Plain resampling to tile size on purpose: routing
                    # through the upscale model evicted Kontext every
                    # single view and the reload dwarfed the repaint.
                    out = out.convert("RGB").resize(
                        (report["tile"], report["tile"]), Image.Resampling.LANCZOS)
                except Exception as exc:  # noqa: BLE001
                    skipped.append({"view": i, "why": str(exc)[:80]})
                    continue
                s_before, s_after, drift = self._speckle_and_drift(
                    before, out, Image.open(alpha))
                # A repaint must clean (speckle down ≥8%) without redrawing
                # (drift bounded) — but a DRAMATIC clean may move more
                # pixels, because repainting patchwork into one coherent
                # surface IS movement. Both thresholds measured on a live
                # set: a 38%-cleaner view was rejected at drift 57 while a
                # near-no-op was rightly rejected at 68.
                strong = s_after < s_before * 0.75
                if (s_after >= s_before * 0.92
                        or drift >= (72.0 if strong else 55.0)):
                    skipped.append({"view": i,
                                    "why": f"speckle {s_before:.1f}->"
                                           f"{s_after:.1f}, drift "
                                           f"{drift:.0f}"})
                    job.log("info", f"View {int(report['views_used'][i])}° "
                                    "kept its projected texture (the "
                                    "repaint measured no clear improvement)")
                    continue
                refined_png = work / f"refined_{i}.png"
                out.save(refined_png)
                nxt = work / f"step_{i}.glb"
                proc = subprocess.run(
                    [python, str(tool), "paste", str(current),
                     str(work / "report.json"), str(i), str(refined_png),
                     str(alpha), str(nxt)],
                    capture_output=True, text=True, timeout=300)
                if proc.returncode == 0 and nxt.exists():
                    current = nxt
                    refined.append(report["views_used"][i])
                    job.log("info", f"View {int(report['views_used'][i])}° "
                                    f"repainted clean (noise "
                                    f"{s_before:.1f} -> {s_after:.1f})")
                else:
                    skipped.append({"view": i, "why": "paste failed"})
            if refined:
                job.log("info", f"{len(refined)} of {n} view(s) repainted; "
                                "the rest measured no improvement and kept "
                                "their projected texture")
                return current.read_bytes(), {"refined": refined,
                                              "skipped": skipped}
        job.log("info", "No view measured cleaner after repainting — the "
                        "projected texture ships unchanged")
        return glb, {"refined": [], "skipped": skipped}

    def _texture_mesh(self, job: Job, glb: bytes,
                      photos: list[tuple[float, Image.Image]],
                      synth_azimuths: list[float] | None = None
                      ) -> tuple[bytes, dict]:
        """Paint the mesh with every view of it that exists.

        The mesh model produces geometry only, and the official texture stage
        cannot run here — it needs a CUDA rasterizer with no wheel for this
        Python, on a machine that also lacks the spare RAM for its offload
        strategy. So the colour comes from the photographs: each vertex takes
        the view that sees it most squarely, and the views are packed into a
        UV atlas so detail is limited by the photograph rather than by the
        vertex count.

        One view covered 43.5% of the surface; four cover 92.9%.

        Runs in ComfyUI's interpreter because trimesh/embreex/scipy already
        live there — no new dependency, no download. Returns the input
        unchanged if any of that is unavailable."""
        base = Path(self.settings.comfyui_dir) if self.settings.comfyui_dir \
            else None
        if base is None or not base.exists() or not photos:
            return glb, {}
        tool = Path(__file__).resolve().parent.parent / "tools" / \
            "texture_mesh.py"
        if not tool.exists():
            return glb, {}
        try:
            python = self._comfy_python(base)
        except Exception:  # noqa: BLE001 — colour is a bonus
            return glb, {}
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "mesh.glb"
            out = Path(tmp) / "textured.glb"
            src.write_bytes(glb)
            args = [python, str(tool), str(src), str(out), "--tile", "2048"]
            if synth_azimuths:
                args += ["--synth",
                         ",".join(str(a) for a in synth_azimuths)]
            for i, (azimuth, photo) in enumerate(photos):
                pic = Path(tmp) / f"view_{i}.png"
                photo.convert("RGB").save(pic, format="PNG")
                args += ["--view", f"{azimuth}:{pic}"]
            try:
                proc = subprocess.run(args, capture_output=True, text=True,
                                      timeout=900)
            except Exception as exc:  # noqa: BLE001
                job.log("info", f"Mesh colouring unavailable ({exc}); "
                                "the mesh stays untextured")
                return glb, {}
            if proc.returncode != 0 or not out.exists():
                job.log("info", "Mesh colouring did not run "
                                f"({(proc.stderr or '').strip()[:160]}); "
                                "the mesh stays untextured")
                return glb, {}
            try:
                report = json.loads(
                    (proc.stdout or "{}").strip().splitlines()[-1])
            except (json.JSONDecodeError, IndexError):
                report = {}
            seen = report.get("seen_pct")
            fit = report.get("orientation_iou")
            used = report.get("views", len(photos))
            job.log("info",
                    f"Textured the mesh from {used} view(s)"
                    + (f" — {seen}% of the surface is covered by a real view; "
                       "the rest takes its neighbours' colour along the "
                       "surface and is approximate" if seen is not None else "")
                    + (f" (camera alignment solved from the silhouettes, "
                       f"IoU {fit})" if fit is not None else ""))
            dropped = report.get("views_dropped") or []
            if dropped:
                job.log("info",
                        f"{len(dropped)} rendered view(s) were not usable "
                        "for colour and were left out: "
                        + "; ".join(f"{int(d['azimuth'])}° — {d['reason']}"
                                    for d in dropped)
                        + ". A rejected view's surface takes neighbouring "
                        "colour instead — painting an unusable view onto "
                        "the avatar is what used to smear it.")
            broken = report.get("components_dropped") or 0
            if broken:
                job.log("info", f"The reconstruction contained {broken} "
                                "disconnected fragment(s) — usually an "
                                "invented body part that did not fuse to "
                                "the figure — and they were removed. A "
                                "standing, full-length photo reconstructs "
                                "in one piece.")
            return out.read_bytes(), report

    def _render_scene3d_step(self, job: Job,
                             image: Image.Image) -> dict[str, Any] | None:
        """A photograph as a place you can move around in.

        Different problem from an avatar: that is an OBJECT you orbit, this is
        a SCENE you stand inside. A geometry model predicts a metric point map
        and the camera's field of view, the triangulation drops faces that
        would span a depth discontinuity — without which every silhouette
        drags a rubber sheet back to the horizon — and the photograph is its
        own texture, so nothing has to be unwrapped.

        TWO layers, because one photograph is one camera frustum: everything
        the lens could not see is simply absent, and stepping sideways opens
        black wedges behind whatever was in front. So the foreground is
        matted out, the hole is inpainted, and THAT image is meshed too and
        put behind the first. Moving now reveals a reconstruction of what was
        probably there instead of a void. It is still a guess, and it is
        labelled as one."""
        ok, why = self._template_runnable("scene3d")
        if not ok and self.settings.auto_install and "not downloaded" in why:
            job.log("info", "[stage] models — fetching the geometry model "
                            "(MoGe); this happens once")
            try:
                self._ensure_model("moge-v2", job)
                ok, why = self._template_runnable("scene3d")
            except Exception as exc:  # noqa: BLE001
                ok, why = False, str(exc)
        if not ok:
            job.log("error", f"A navigable 3D scene is not available here "
                             f"({why})")
            return None
        layers: list[tuple[str, bytes]] = []
        try:
            self._require_comfy(job)
            job.log("info", "[stage] scene — reading the geometry of the "
                            "photograph")
            layers.append(("front", self._mesh_one_layer(job, image)))
        except PermanentError:
            # A job pinned to a peer that died mid-render must fail NOW,
            # loudly — swallowing it here kept the job visibly running
            # against a dead binding for whole retry rounds.
            raise
        except Exception as exc:  # noqa: BLE001
            job.log("error", f"Scene reconstruction failed: {exc}")
            self._diagnose_and_record(job, "scene3d", "", str(exc))
            return None
        # The layer behind. Never fatal: one layer is a working scene with
        # holes, which is what the previous version always produced.
        try:
            plate = self._background_plate(job, image)
            if plate is not None:
                job.log("info", "[stage] scene — reconstructing what stood "
                                "behind the foreground, so moving the camera "
                                "does not open black gaps")
                layers.append(("behind", self._mesh_one_layer(job, plate)))
        except Exception as exc:  # noqa: BLE001
            job.log("info", f"The hidden layer could not be built ({exc}); "
                            "moving sideways will show gaps where the "
                            "foreground was")
        data = layers[0][1]
        if len(layers) > 1:
            merged = self._merge_meshes(job, [d for _n, d in layers])
            if merged is not None:
                data = merged
        asset = self.store.save_upload(
            f"scene_{job.id}.glb", data,
            meta={"synthetic": True, "engine": "moge-v2", "kind3d": "scene",
                  "layers": len(layers),
                  "disocclusion": len(layers) > 1})
        job.log("info", f"[stage] save — 3D scene saved "
                        f"({len(data) // 1024} KB, {asset.id}); walk around "
                        "it in Studio or download the GLB")
        return {"scene_asset": asset.id, "layers": len(layers)}

    def _mesh_one_layer(self, job: Job, image: Image.Image) -> bytes:
        """One image through the geometry model, out as a textured GLB."""
        template = self.workflows.load_named("scene3d")
        graph = build_workflow(template, {
            "image": self.comfy.upload_image(image, "scene_src"),
            # 0 asks the model to predict the field of view rather than being
            # told a wrong one; a wrong focal length shears the whole scene.
            "fov_x_degrees": 0.0})
        self._prepare_heavy_render(job, need_gb=6.0)
        self._free_vram(job)
        self._drop_comfy_cache()
        data, _fname = self.comfy.wait_for_mesh(self.comfy.submit(graph))
        return data

    def _background_plate(self, job: Job,
                          image: Image.Image) -> Image.Image | None:
        """The photograph with its foreground removed and painted over.

        Reuses the background route's own machinery — the matte measured
        exact here, inverted — rather than a second implementation."""
        if not self._pack_active("rmbg"):
            return None
        try:
            self._render_background_step(
                job, image, "the same place, empty, nothing in the foreground",
                "people, person, figure, subject, text, watermark",
                subject_hint="the main subject")
        except Exception as exc:  # noqa: BLE001
            job.log("info", f"Could not clear the foreground ({exc})")
            return None
        # That route already exports the scene WITHOUT the subject — it needs
        # it to read the light for relighting. Same picture, second use.
        return getattr(self, "_last_bg_plate", None)

    def _merge_meshes(self, job: Job, parts: list[bytes]) -> bytes | None:
        """Combine layered GLBs into one file, in ComfyUI's interpreter."""
        base = Path(self.settings.comfyui_dir) if self.settings.comfyui_dir \
            else None
        tool = Path(__file__).resolve().parent.parent / "tools" / \
            "merge_meshes.py"
        if base is None or not base.exists() or not tool.exists():
            return None
        try:
            python = self._comfy_python(base)
        except Exception:  # noqa: BLE001
            return None
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, data in enumerate(parts):
                p = Path(tmp) / f"layer_{i}.glb"
                p.write_bytes(data)
                paths.append(str(p))
            out = Path(tmp) / "merged.glb"
            try:
                proc = subprocess.run([python, str(tool), str(out), *paths],
                                      capture_output=True, text=True,
                                      timeout=600)
            except Exception as exc:  # noqa: BLE001
                job.log("info", f"Could not merge the layers ({exc})")
                return None
            if proc.returncode != 0 or not out.exists():
                job.log("info", "Could not merge the layers "
                                f"({(proc.stderr or '').strip()[:160]})")
                return None
            job.log("info", (proc.stdout or "").strip().splitlines()[-1][:200])
            return out.read_bytes()

    def _build_mesh(self, job: Job, frame_ids: list[str], fallback_id: str,
                    texture: bool = True, rig: bool = True,
                    refine: bool = True) -> dict[str, Any] | None:
        """Turn the orbit into a real 3D mesh, saved as a .glb asset.

        Hardware sets the octree resolution — the surface detail — and the
        DATA sets how many views condition the shape. That split is the
        correction of a measured mistake: the ladder used to spend a bigger
        GPU on multi-view conditioning, and conditioning on views another
        model invented made the figure 40% too deep. None when this machine
        has no mesh tier."""
        hw = self.hardware
        tier = quality.choose_reconstruction(
            hw.vram_gb if hw else 0.0, hw.ram_gb if hw else 0.0)
        if not tier.models:
            job.log("info", "[stage] mesh — " + quality.reconstruction_note(
                tier, hw.vram_gb if hw else 0.0))
            return None
        # Shape comes from REAL photographs only. A view SV3D invented is not
        # evidence about the subject, and the reconstruction treats it as if
        # it were: measured, four views (three of them invented) produced
        # depth/width 0.73 where the same photograph alone produced 0.53.
        real_views = self._views_for_mesh(frame_ids, real_only=True)
        real_views = self._consistent_real_views(job, real_views)
        # A person is not a rigid object: the pose changes between shots, so
        # multi-view conditioning is off for avatars. Measured on this very
        # dataset, two real photographed angles produced depth/width 1.19
        # against 0.68 from the single best photograph.
        multi = quality.use_multiview(len(real_views), rigid_subject=False)
        # Colour, unlike shape, is happy with rendered views: they are real
        # pixels of the right surface even when the geometry behind them was
        # guessed, and a view that covers the back beats smearing the front.
        try:
            colour_photos = (self._colour_photos(job, frame_ids)
                             if texture else [])
        except Exception as exc:  # noqa: BLE001 — colour is a bonus
            job.log("info", f"Could not prepare colour views ({exc}); "
                            "the mesh will use the fallback photograph")
            colour_photos = []
        note = quality.reconstruction_note(
            tier, hw.vram_gb if hw else 0.0,
            len(real_views) if multi else 1, len(colour_photos) or 1,
            textured=texture)
        job.log("info", f"[stage] mesh — {note}")
        try:
            self._require_comfy(job)
            model = "hunyuan3d-v2-mv" if multi else "hunyuan3d-v2"
            self._ensure_model(model, job)
            seed = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF
            params: dict[str, Any] = {"seed": seed,
                                      "octree_resolution": tier.octree}
            if multi:
                template = self.workflows.load_named("reconstruct_mv")
                # A turbo distillation wants its own settings. Driven at the
                # undistilled 20 steps / cfg 4.0 it was both slower and worse.
                params.update(quality.MULTIVIEW_SAMPLER)
                for name, aid in real_views.items():
                    params[name] = self.comfy.upload_image(
                        self.open_asset_image(aid), f"mesh_{name}")
            else:
                template = self.workflows.load_named("reconstruct")
                best = (real_views.get("front")
                        or self._views_for_mesh(frame_ids).get("front")
                        or fallback_id)
                shape_src = self.open_asset_image(best)
                # The reconstruction models whatever rectangle it is
                # handed. Orbit frames arrive staged; the raw-photo
                # fallback does not — matte it, or the photo itself comes
                # back as a textured slab (seen live as a literal box).
                if not self._plain_backdrop(shape_src):
                    try:
                        shape_src = self._matte_on_grey(job, shape_src)
                    except Exception:  # noqa: BLE001 — keep the original
                        pass
                params["image"] = self.comfy.upload_image(
                    shape_src, "mesh_front")
            # Pin the checkpoint to the model actually ensured. The templates
            # carry a default filename, and picking the wrong one downloaded
            # 4.6 GB that was then never used.
            entry = self.registry.get(model)
            wanted = (Path((entry.meta or {}).get("file") or "").name
                      if entry else "")
            if wanted and "checkpoint" in template.get("parameters", {}):
                params["checkpoint"] = wanted
            graph = build_workflow(template, params)
            if multi:
                graph = self._prune_mesh_views(graph, set(real_views))
            # The checkpoint is ~4.6 GB and RAM, not VRAM, is what OS-kills a
            # load on this class of machine — the same guard the video path
            # learned to take.
            self._prepare_heavy_render(job, need_gb=10.0)
            self._free_vram(job)
            self._drop_comfy_cache()
            data, fname = self.comfy.wait_for_mesh(self.comfy.submit(graph))
        except Exception as exc:  # noqa: BLE001 — a mesh is a bonus, not the job
            job.log("error", f"Mesh reconstruction unavailable: {exc}")
            self._diagnose_and_record(job, "reconstruct", "", str(exc))
            return None
        # Geometry only is what the model gives; the colour comes from the
        # photographs. Guarded: a mesh without colour still beats no mesh.
        textured_here = False
        report: dict[str, Any] = {}
        if not texture:
            job.log("info", "Texturing is switched off — the mesh is bare "
                            "geometry, which is what you want if you are "
                            "going to paint or sculpt it yourself")
        else:
            try:
                # The fallback is a RAW upload with its real background —
                # matte it, or the room gets painted onto the avatar, which
                # is the exact smear class this pipeline exists to prevent.
                photos = colour_photos or [
                    (0.0, self._matte_on_grey(
                        job, self.open_asset_image(fallback_id)))]
                # Remember the front view: it is what the back synthesis
                # shows Kontext when the back arc has no usable camera.
                self._avatar_front_view = photos[0][1] if photos else None
                bare = data          # the untextured geometry, for re-runs
                coloured, report = self._texture_mesh(job, bare, photos)
                textured_here = coloured is not bare
                data = coloured
                # A whole arc with no camera is the one gap texture
                # arithmetic cannot close — and rendered honestly, the
                # uncovered arcs (usually the back AND both sides) carry
                # the worst pixels on the figure. Generate every missing
                # view and re-texture THE BARE GEOMETRY with them added;
                # keep the result only when coverage actually rose — the
                # quality gate judges a generated view like any other.
                used = report.get("views_used") or []

                def _covered(lo: float, hi: float) -> bool:
                    return any(lo <= a <= hi for a in used)

                # NOT `wanted` — this function already binds that name to
                # the checkpoint filename above, and the shadow poisoned
                # every type inference below it (found by mypy).
                missing_views: list[tuple[float, str]] = []
                if textured_here and not _covered(145, 215):
                    missing_views.append((180.0, "back"))
                if textured_here and not _covered(55, 125):
                    missing_views.append((90.0, "left"))
                if textured_here and not _covered(235, 305):
                    missing_views.append((270.0, "right"))
                if missing_views:
                    names = ", ".join(w for _, w in missing_views)
                    job.log("info", "[stage] texture — no usable camera on "
                                    f"the {names} arc(s); generating the "
                                    "missing view(s) (invented, and "
                                    "recorded as such)")
                    synth = [(az, img) for az, which in missing_views
                             if (img := self._synthesize_view(job, which))
                             is not None]
                    if synth:
                        coloured2, report2 = self._texture_mesh(
                            job, bare, photos + synth,
                            synth_azimuths=[az for az, _ in synth])
                        gained = ((report2.get("seen_pct") or 0)
                                  > (report.get("seen_pct") or 0) + 1)
                        if gained:
                            data, report = coloured2, report2
                            azs = [az for az, _ in synth]
                            report["synthesized_views"] = azs
                            if 180.0 in azs:
                                report["back_synthesized"] = True
                            job.log("info", f"{len(synth)} generated "
                                            "view(s) passed the quality "
                                            "gate and cover the missing "
                                            "arcs")
            except Exception as exc:  # noqa: BLE001
                job.log("info", f"Mesh colouring skipped ({exc})")
        # Projection can only CHOOSE pixels; repainting each view of the
        # painted mesh is what removes the specks and patchwork choosing
        # leaves behind. Guarded and self-verifying: a view ships repainted
        # only when its measured noise fell.
        if textured_here and refine:
            try:
                data, refined = self._refine_texture(job, data, report)
                if refined.get("refined"):
                    report["texture_refined"] = refined["refined"]
            except Exception as exc:  # noqa: BLE001 — refinement is a bonus
                job.log("info", f"Texture refinement skipped ({exc})")
        # The texturing step has to solve where the cameras are, and how well
        # it can is a free measurement of whether the reconstruction actually
        # agrees with the photographs. Measured: 0.85 for a mesh built from a
        # clean full-length photo, 0.51 for one built from a mirror selfie
        # with a phone held across the chest. Reported rather than hidden,
        # because the fix is a better photo and only the user has one.
        fit = report.get("orientation_iou")
        if fit is not None and fit < self._MESH_FIT_FLOOR:
            job.log("warn",
                    f"This reconstruction only matches your photos to "
                    f"{fit:.2f} of 1.0 — well below the {self._MESH_FIT_FLOOR} "
                    "a clean source reaches. That usually means the photo it "
                    "was built from has an arm or an object across the body, "
                    "or cuts the subject off. A plain, full-length, "
                    "front-facing photo reconstructs far better.")
        rig_report: dict = {}
        if rig:
            data, rig_report = self._rig_avatar(job, data)
        asset = self.store.save_upload(
            f"avatar_mesh_{job.id}{Path(fname).suffix or '.glb'}", data,
            limit_mb=256,
            meta={"synthetic": True, "engine": "hunyuan3d",
                  "rigged": bool(rig_report),
                  "rig_bones": rig_report.get("bones"),
                  "rig_weights": rig_report.get("weights"),
                  "rig_full_body": rig_report.get("full_body"),
                  "back_synthesized": bool(report.get("back_synthesized")),
                  "views_synthesized": report.get("synthesized_views"),
                  "texture_refined": report.get("texture_refined"),
                  "tier": tier.name, "octree": tier.octree,
                  "textured": textured_here,
                  "colour": "photo-projected" if textured_here else "none",
                  "shape_views": len(real_views) if multi else 1,
                  "colour_views": (report.get("views",
                                              len(colour_photos))
                                   if textured_here else 0),
                  "colour_views_dropped": len(report.get("views_dropped")
                                              or []),
                  "surface_covered_pct": report.get("seen_pct"),
                  "photo_agreement": report.get("orientation_iou")})
        job.log("info", f"[stage] save — 3D mesh saved ({len(data) // 1024} KB, "
                        f"{asset.id}); download it as GLB or orbit it in Studio")
        return {"mesh_asset": asset.id, "tier": tier.level,
                "tier_name": tier.name, "textured": textured_here,
                "rigged": bool(rig_report),
                "surface_covered_pct": report.get("seen_pct"),
                "photo_agreement": report.get("orientation_iou"),
                "note": note}

    @staticmethod
    def _prune_mesh_views(graph: dict[str, Any],
                          keep: set[str]) -> dict[str, Any]:
        """Drop the view inputs there is no real photograph for.

        The alternative — repeating the front photo into the empty slots —
        tells the model the subject looks identical from every side, and it
        produced a flat sheet: one measured mesh came out 1.81 x 1.97 x 0.03."""
        out = dict(graph)
        conditioning = out.get("10", {}).get("inputs", {})
        for name, (loader, encoder) in Services._MV_NODES.items():
            if name in keep:
                continue
            conditioning.pop(name, None)
            out.pop(loader, None)
            out.pop(encoder, None)
        return out

    def _handle_avatar(self, job: Job) -> dict[str, Any]:
        """Avatar dataset intake: consent → segment → coverage → synth angles.

        Implements the intake stages of the digital-human pipeline: consented
        photos are background-segmented (SAM), their view angles classified
        (vision model), and missing angles synthesized with the SV3D template
        — synthetic views are clearly labeled and never replace real photos.
        Reconstruction/rigging build on this dataset (roadmap Phase 4).
        """
        p = job.payload
        asset_ids: list[str] = p.get("asset_ids") or []
        consent = consent_verdict(bool(p.get("consent")))
        if not consent.allowed:  # policy lives in safety.py
            raise PermanentError(consent.reason or "Consent required.")
        if len(asset_ids) < 1:
            raise PermanentError("Add at least one photo to the dataset.")

        self._log_eta(job, required_models=["sv3d"])
        job.log("info", f"[stage] segment — isolating the subject in "
                        f"{len(asset_ids)} photo(s)")
        point_capable = hasattr(self.segmentation, "point_mask")
        coverage: dict[str, list[str]] = {b: [] for b in self.VIEW_BINS}
        unknown: list[str] = []
        # Whether the view classification repeated on a second, independent
        # ask. Only confirmed answers are allowed to move a photograph away
        # from the front of the orbit.
        confident: dict[str, bool] = {}
        # Kept, not discarded: the cutout mask is what lets SV3D orbit the
        # SUBJECT instead of the frame, and the focus score is how the best
        # photo in the set gets chosen rather than whichever arrived first.
        masks: dict[str, Image.Image] = {}
        focus: dict[str, float] = {}
        for aid in asset_ids:
            image = self.open_asset_image(aid)
            focus[aid] = self._focus_score(image)
            # WHOLE-SUBJECT matte, not a SAM point click. SAM is a PART
            # segmenter: clicking the centre of a photo lands on whatever
            # happens to be there, and on a swimwear shot that is the
            # swimwear — measured live, the avatar orbited a black triangle
            # because SAM had cut out the bikini rather than the person.
            # BiRefNet is the matte measured correct here (19.4% against a
            # 19.4% ground truth).
            mask = None
            if self._pack_active("rmbg"):
                mask = self._region_mask(image, "BiRefNetRMBG", {
                    "model": self._matte_model("person subject"),
                    "sensitivity": 1.0, "mask_blur": 0, "mask_offset": 0,
                    "invert_output": False, "refine_foreground": True,
                    "background": "Alpha", "background_color": "#222222"})
            if mask is None and point_capable:
                try:
                    # hasattr-probed above (point_capable); SAM-only method.
                    mask = self.segmentation.point_mask(  # type: ignore[attr-defined]
                        image, image.width // 2, image.height // 2)
                    job.log("info", "BiRefNet unavailable — fell back to a "
                                    "SAM click, which often selects only part "
                                    "of the subject")
                except (ModelMissingError, Exception) as exc:  # noqa: BLE001
                    job.log("error", f"Cutout failed for {aid}: {exc}")
            if mask is not None:
                cover = self._mask_fraction(mask)
                # A plausible person fills a real share of a portrait. Far
                # below that and the matte found a garment or a prop, and
                # orbiting it produces the black-wedge result.
                if cover < 0.04:
                    job.log("info", f"Photo {aid}: the cut-out covers only "
                                    f"{cover * 100:.1f}% of the frame — too "
                                    "little to be the subject, so the whole "
                                    "frame is used instead")
                    mask = None
                else:
                    masks[aid] = mask
                    cutout = Image.new("RGBA", image.size, (0, 0, 0, 0))
                    cutout.paste(image, (0, 0), mask)
                    out = self.store.new_version_path(aid, suffix=".png")
                    cutout.save(out, format="PNG")
                    self.store.add_edit_version(
                        aid, str(out), "subject cutout", "sam-cutout",
                        meta={"avatar": True, "coverage": round(cover, 4)})
            view, sure = self._confident_view(image)
            confident[aid] = sure
            (coverage[view] if view in coverage else unknown).append(aid)
            job.log("info", f"Photo {aid}: view = {view}"
                            + ("" if sure else " (not confirmed on a second "
                                              "ask, so treated as front)")
                            + f", sharpness {focus[aid]:.0f}")

        job.log("info", "[stage] coverage — classifying view angles")
        # How this person looks, in prompt words — so an identity render is
        # not working from a face crop alone.
        appearance = self._appearance_profile(
            job, sorted(asset_ids, key=lambda a: -focus.get(a, 0.0)))
        present = [b for b in self.VIEW_BINS if coverage[b]]
        missing = [b for b in self.VIEW_BINS if not coverage[b]]
        job.log("info", f"Coverage: {len(present)}/8 bins "
                        f"({', '.join(present) or 'none'}); missing: "
                        f"{', '.join(missing) or 'none'}")

        frame_ids: list[str] = []
        synthetic: list[str] = []
        synth_bins: dict[str, int] = {}
        real_bins: dict[str, int] = {}
        # Fewer than 8 photos can never cover all 8 view bins — synthesize
        # the missing angles with the multi-view template, then proceed.
        if missing or len(asset_ids) < 8:
            job.log("info", "[stage] angles — synthesizing missing views "
                            "(SV3D, clearly labeled synthetic)")
            try:
                # ComfyUI may have died during the (slow) SAM/classify loop —
                # bring it back before the multi-view render.
                self._require_comfy(job)
                template = self.workflows.load("angles")
                if self.settings.auto_install:
                    for name in template.get("required_models", []):
                        self._ensure_model(name, job)
                # A subject the frame cuts in half reconstructs as a subject
                # cut in half, however sharp the photograph is.
                framing = {aid: self._framing_penalty(m)
                           for aid, m in masks.items()}
                source_id = self._best_orbit_source(coverage, asset_ids, focus,
                                                    framing)
                if framing.get(source_id, 0.0) > 0.5:
                    job.log("info", "Every photo you gave cuts the subject off "
                                    "at the frame edge, so the 3D model will "
                                    "end where the photograph does. One "
                                    "full-length shot would fix that.")
                # A body cropped by the frame reconstructs as a cropped body.
                # Extend the picture first when the subject runs off an edge.
                original = self.open_asset_image(source_id)
                source_image = original
                if p.get("complete_body", True) is not False:
                    source_image = self._complete_subject(job, original)
                elif self._is_cut_off(original):
                    job.log("info", "The subject runs off the frame and "
                                    "completing the body is switched off, so "
                                    "the model will end where the photograph "
                                    "does")
                # SV3D orbits FROM the photo it is given, so frame 0 is that
                # photo's own viewpoint - not the front. Labelling frame 0 as
                # 0 degrees rotated every angle by however far off-front the
                # source was: seen live, a left-profile source produced an
                # orbit whose "front" frames were actually the back.
                source_view = next(
                    (v for v, ids in coverage.items() if source_id in ids),
                    "front")
                # A wrong non-front answer rotates the WHOLE orbit — a bogus
                # "left" shifts every label by 270 degrees — so a non-zero
                # offset is only taken from a classification that repeated.
                offset = 0
                if (source_view in self.VIEW_BINS and source_view != "front"
                        and confident.get(source_id)):
                    offset = self.VIEW_BINS.index(source_view) * 45
                elif source_view != "front":
                    job.log("info", f"'{source_view}' for the orbit source was "
                                    "not confirmed on a second ask, so the "
                                    "orbit starts from the front")
                    source_view = "front"
                job.log("info", f"Orbiting from {source_id} — the sharpest, "
                                f"most frontal photo in the set (a "
                                f"{source_view} view, so the orbit starts at "
                                f"{offset} degrees)")
                # If the frame was extended the old matte no longer lines
                # up, so cut a FRESH one. Two bugs lived on this line: the
                # old `is not open_asset_image(...)` compared against a
                # freshly opened copy, so "extended" was ALWAYS true — and
                # the mask it then passed was None, which _stage_for_orbit
                # answers by not cutting the subject out at all. Every
                # avatar orbit was fed the photo WITH its background, and
                # SV3D rotated the room: that is where the "hallucinated
                # wall" views came from.
                extended = source_image is not original
                orbit_mask = masks.get(source_id)
                if extended or orbit_mask is None:
                    orbit_mask = self._subject_matte(source_image)
                    if orbit_mask is None:
                        # One retry behind a revive: seen live, ComfyUI was
                        # briefly unreachable, the matte silently failed,
                        # and the uncut photo reconstructed as a textured
                        # BOX — the picture modelled as an object.
                        try:
                            self._require_comfy(job)
                            orbit_mask = self._subject_matte(source_image)
                        except PermanentError:
                            raise   # pinned peer died mid-render: fail NOW
                        except Exception:  # noqa: BLE001
                            orbit_mask = None
                if orbit_mask is None and \
                        not self._plain_backdrop(source_image):
                    raise RuntimeError(
                        "the subject could not be isolated from the "
                        "background (segmentation unavailable) — orbiting "
                        "the whole photograph would model the picture "
                        "itself as a flat slab")
                files = self._orbit_from_photo(
                    job, source_image, orbit_mask, prefix="avatar_src")
                used_real: set[str] = set()
                for i, (data, fname) in enumerate(files):
                    angle = round((offset + i * 360 / max(1, len(files)))
                                  % 360)
                    bin_name = self.VIEW_BINS[round(angle / 45) % 8]
                    suffix = Path(fname).suffix or ".png"
                    is_synth = True
                    # A REAL photo of this side beats an invented one. Staged
                    # the same way as the orbit frames (cut out, neutral grey,
                    # square) so it sits in the sequence without jumping, and
                    # so the frame list stays dense and evenly spaced — the
                    # viewer indexes it by angle.
                    #
                    # Two conditions, both learned the hard way. The bin has to
                    # be CONFIRMED, because an unconfirmed 'back' splices a
                    # front view in at 180 degrees and paints a face onto the
                    # back of the model. And the best candidate is taken
                    # rather than whichever happened to be first in the list.
                    real_id = max(
                        (a for a in coverage.get(bin_name, [])
                         if a not in used_real and confident.get(a)),
                        key=lambda a: (-round(framing.get(a, 0.0) * 2) / 2,
                                       focus.get(a, 0.0)),
                        default=None)
                    if real_id:
                        # The real photo of this side is better evidence
                        # than an invented frame — EXCEPT when it is the
                        # very photo the body completion just extended.
                        # Staging the RAW original here handed the waist-up
                        # crop back to the reconstruction as its "real
                        # front", which is exactly how a job that logged
                        # "the body is complete now" still shipped a bust.
                        # The completed image's upper half IS the original
                        # pixels, so nothing real is lost by staging it.
                        use_completed = real_id == source_id and extended
                        staged = self._stage_for_orbit(
                            source_image if use_completed
                            else self.open_asset_image(real_id),
                            orbit_mask if use_completed
                            else masks.get(real_id), self.VIEW_SIZE)
                        buf = io.BytesIO()
                        staged.save(buf, format="PNG")
                        data, suffix, is_synth = buf.getvalue(), ".png", False
                        used_real.add(real_id)
                    a = self.store.save_upload(
                        f"angle_{angle:03d}_{job.id}{suffix}", data,
                        meta={"synthetic": is_synth, "azimuth": angle,
                              "view_bin": bin_name,
                              "engine": "sv3d" if is_synth else "photo",
                              "source_asset": real_id or source_id})
                    frame_ids.append(a.id)
                    if is_synth:
                        synthetic.append(a.id)
                        synth_bins[bin_name] = synth_bins.get(bin_name, 0) + 1
                    else:
                        real_bins[bin_name] = real_bins.get(bin_name, 0) + 1
                job.log("info",
                        f"{len(frame_ids)} orbit views: {len(frame_ids) - len(synthetic)} "
                        f"from your own photos, {len(synthetic)} synthesized "
                        "(labelled synthetic — for reconstruction only)")
            except Exception as exc:  # noqa: BLE001 — report, keep the dataset
                # Never lose the consented dataset because angle synth failed;
                # learn from the error and continue with what we have.
                job.log("error", f"Angle synthesis unavailable: {exc}")
                self._diagnose_and_record(job, "angles", "", str(exc))

        face_asset_hint = self._best_face_photo(coverage, asset_ids, focus)
        # A real 3D mesh, when the hardware can build one. Never fatal: an
        # avatar with orbit frames and no mesh is still a working avatar.
        mesh = (self._build_mesh(job, frame_ids, face_asset_hint,
                                 texture=p.get("texture", True) is not False,
                                 rig=p.get("rig", True) is not False,
                                 refine=p.get("refine", True) is not False)
                if frame_ids or asset_ids else None)

        job.log("info", "[stage] save — saving the avatar profile")
        still_missing = [b for b in missing if not synth_bins.get(b)]
        # The face reference is the sharpest FRONTAL photo, not whichever id
        # sorted first — it is the single image every identity render is
        # conditioned on, so its quality sets the ceiling for all of them.
        face_asset = face_asset_hint
        # Distinct bins covered, out of 8. Adding present+synthetic counted
        # the same bin twice and reported "10 of 8".
        covered = {b for b in self.VIEW_BINS
                   if coverage[b] or synth_bins.get(b) or real_bins.get(b)}
        profile = self.store.create_avatar(
            name=p.get("name") or f"Avatar {face_asset[:6]}",
            source_assets=asset_ids, frames=frame_ids, face_asset=face_asset,
            meta={"coverage_bins": len(covered),
                  "real_views": len(frame_ids) - len(synthetic),
                  "synthetic": len(synthetic), "consent": True,
                  "appearance": appearance,
                  "mesh": mesh or {}})
        job.log("info", f"Avatar '{profile.name}' saved ({profile.id}) — "
                        "movable in the orbit viewer, renderable via prompts")
        return {
            "photos": len(asset_ids),
            "avatar_id": profile.id,
            "avatar_name": profile.name,
            "frames": frame_ids,
            "coverage": {b: coverage[b] for b in self.VIEW_BINS},
            "coverage_synthetic": synth_bins,
            "coverage_real": real_bins,
            "appearance": appearance,
            "mesh": mesh or {},
            "unknown": unknown,
            "missing": still_missing,
            "synthetic_assets": synthetic,
            "next": ("Avatar ready: drag the orbit view to move it, and use "
                     "‘Render with this avatar’ to put the person in any "
                     "prompted image or video (photoreal, identity-preserving)."
                     if frame_ids else
                     "Avatar saved for identity renders. Orbit frames are "
                     "missing — start ComfyUI (with the SV3D model) and "
                     "rebuild to make it movable."),
        }

    # PhotoMaker stacks reference images to build one identity embedding, and
    # more angles of the same face gives a markedly steadier likeness. More
    # than this stops helping and costs VRAM on an 8 GB card.
    _MAX_IDENTITY_REFS = 4

    def _identity_reference(self, job: Job, avatar: Any, face_id: str) -> str:
        """Upload every consented photo of this person as ONE image batch.

        PhotoMakerEncode takes a single IMAGE input, but a ComfyUI IMAGE is a
        batch — and an animated WEBP arrives through core LoadImage as its
        full frame batch (verified on this machine: 25 frames in, 25 out).
        So the whole consented set reaches the identity encoder without the
        template needing to change shape. Falls back to the single best photo
        if anything about the batch path fails."""
        ids = [face_id] + [a for a in (avatar.source_assets or [])
                           if a != face_id]
        ids = ids[:self._MAX_IDENTITY_REFS]
        if len(ids) > 1 and hasattr(self.comfy, "upload_frames"):
            try:
                frames = [self.open_asset_image(a).convert("RGB")
                          for a in ids]
                # One batch means one shape: match the reference photo.
                size = frames[0].size
                frames = [f if f.size == size else f.resize(size,
                                                            Image.Resampling.LANCZOS)
                          for f in frames]
                name = self.comfy.upload_frames(frames, "identity_ref", 1)
                job.log("info", f"Identity built from {len(frames)} consented "
                                "photos of this person")
                return name
            except Exception as exc:  # noqa: BLE001 — one photo still works
                job.log("info", f"Multi-photo identity unavailable ({exc}); "
                                "using the single best reference")
        job.log("info", "Identity built from 1 reference photo")
        return self.comfy.upload_image(self.open_asset_image(face_id),
                                       "identity_ref")

    # InstantID holds a likeness noticeably better than PhotoMaker, but it
    # needs SDXL + adapter + ControlNet resident at once (~11.1 GB), so it is
    # gated on hardware exactly like the 3D tiers. Its template shipped in the
    # repo but nothing ever loaded it; this is where it earns its place.
    _INSTANTID_VRAM_GB = 12.0
    _INSTANTID_RAM_GB = 24.0

    def _identity_engine(self) -> dict[str, str]:
        """The best identity engine this machine can hold in memory."""
        hw = self.hardware
        if (hw and hw.vram_gb >= self._INSTANTID_VRAM_GB
                and hw.ram_gb >= self._INSTANTID_RAM_GB
                and self._pack_active("instantid")):
            return {"template": "identity_face",
                    "why": "InstantID — strongest likeness, needs ~11 GB"}
        why = "SDXL + PhotoMaker"
        if hw:
            why += (f"; InstantID needs {self._INSTANTID_VRAM_GB:.0f} GB VRAM "
                    f"and this GPU has {hw.vram_gb:.0f}")
        return {"template": "identity", "why": why}

    def _handle_avatar_render(self, job: Job) -> dict[str, Any]:
        """Identity render: put a consented avatar into a prompted scene.

        PhotoMaker (ComfyUI core) encodes the avatar's face reference into
        SDXL conditioning; the critic judges realism with one strategy-change
        retry; optionally the result is animated with the WAN template.
        """
        p = job.payload
        avatar = self.store.get_avatar(p.get("avatar_id", ""))
        if avatar is None:
            raise PermanentError("Avatar not found — build one on the Avatar "
                                 "page first.")
        prompt = (p.get("prompt") or "").strip()
        if not prompt:
            raise PermanentError("A prompt is required.")
        self._require_comfy(job)
        eta_models = ["sdxl-base", "photomaker-v1"]
        if p.get("video"):
            eta_models += ["wan22-ti2v-5b", "wan-umt5-xxl", "wan22-vae"]
        self._log_eta(job, required_models=eta_models)

        engine = self._identity_engine()
        job.log("info", f"[stage] models — identity pipeline ({engine['why']})")
        # load_named handles both "identity" and the "identity_face"
        # variant; both declare the allowed task "identity".
        template = self.workflows.load_named(engine["template"])
        if not self.settings.auto_install:
            missing = [m for m in template.get("required_models", [])
                       if not self.registry.is_ready(m)]
            if missing:
                raise PermanentError(
                    "Missing identity models (auto-install disabled): "
                    + ", ".join(missing))
        for name in template.get("required_models", []):
            self._ensure_model(name, job)

        job.log("info", "[stage] reference — preparing the face reference")
        face_id = avatar.face_asset or (avatar.source_assets[0]
                                        if avatar.source_assets else None)
        if face_id is None:
            raise PermanentError("This avatar has no reference photo.")
        image_name = self._identity_reference(job, avatar, face_id)

        # 'photomaker' is the trigger token PhotoMakerEncode replaces with the
        # identity embedding — it must directly follow the class word.
        # The appearance read at intake steers build, colouring and age,
        # which a single face crop cannot carry on its own.
        look = self.appearance_phrase((avatar.meta or {}).get("appearance"))
        positive = (f"photo of a person photomaker. {prompt}"
                    + (f", {look}" if look else "")
                    + ", photorealistic, natural skin texture, detailed face, "
                      "sharp focus")
        job.log("info", f"[llm] identity prompt: {positive}")

        def render_once(text: str):
            graph = build_workflow(template, {
                "prompt": text, "image": image_name,
                "seed": int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF})
            self._free_vram(job)
            try:
                prompt_id = self.comfy.submit(graph)
                return self.comfy.wait_for_output(prompt_id), prompt_id
            except BackendUnavailableError as exc:
                raise self._comfy_died_midrender(job, "identity", prompt,
                                                 exc) from exc
            except WorkflowRuntimeError as exc:
                self._diagnose_and_record(job, "identity", prompt, str(exc))
                hint = commit_exhausted_hint(str(exc))
                if hint:
                    raise PermanentError(
                        f"Identity render failed: {hint}") from exc
                raise PermanentError(
                    f"Identity render failed: {exc}. If ComfyUI cannot find "
                    "the PhotoMaker nodes, update ComfyUI to a recent "
                    "version.") from exc

        job.log("info", "[stage] render — rendering the avatar into the scene")
        image, prompt_id = render_once(positive)

        job.log("info", "[stage] check — judging realism and identity")
        crit = self._critique(job, image, prompt)
        if (crit is not None and crit.score < self.settings.critic_min_score
                and self.settings.critic_retries > 0):
            job.log("info", "[stage] retry — not convincing enough, "
                            "changing strategy")
            retry_text = (positive
                          + (f"; avoid: {'; '.join(crit.issues[:3])}"
                             if crit.issues else "; avoid: uncanny, waxy skin"))
            image2, pid2 = render_once(retry_text)
            crit2 = self._critique(job, image2, prompt)
            if crit2 is not None and crit2.score >= crit.score:
                job.log("info", f"Retry kept ({crit2.score:g} ≥ {crit.score:g})")
                image, prompt_id, crit = image2, pid2, crit2
            else:
                job.log("info", "Retry discarded; keeping the first render")

        job.log("info", "[stage] save — storing the render")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        asset = self.store.save_upload(
            f"avatar_{avatar.id}_{job.id}.png", buf.getvalue())
        result: dict[str, Any] = {
            "asset_id": asset.id, "avatar_id": avatar.id,
            "prompt_id": prompt_id,
            "realism": crit.score if crit else None}

        if p.get("video"):
            job.log("info", "[stage] animate — bringing the render to life")
            vasset, vpid, w, h, length = self._render_video_asset(
                job, asset.id, prompt, length=p.get("length", 49))
            result.update({"video_asset_id": vasset.id, "video_frames": length})
        return result

    # -- "Improve the LLM": grow the workflow library over time ------------------
    def _handle_discover(self, job: Job) -> dict[str, Any]:
        """The LLM proposes ADVANCED workflows the library doesn't have yet,
        built only from allowlisted nodes. Each candidate is structurally
        validated; the survivors are offered for approval (see save_candidate),
        which is what lets the tool keep getting smarter.

        Note on 'searching online': arbitrary third-party ComfyUI graphs can't
        be run safely (the node-type allowlist would reject unknown custom
        nodes, and running unvetted node code is a security risk). So this uses
        the model's own broad knowledge (local first, Claude fallback) to
        AUTHOR advanced recipes from safe building blocks, then verifies them."""
        from ..adapters.comfyui import ALLOWED_NODE_TYPES
        from .workflow_ai import NODE_OUTPUTS, validate_generated
        self._revive_ollama(job)
        existing = [t["template"] for t in self.workflows.list_all()]
        job.log("info", "[stage] discover — the LLM is proposing advanced "
                        f"workflows beyond the {len(existing)} we already have")
        outputs = "; ".join(
            f"{t}={','.join(o) or 'none'}" for t, o in sorted(NODE_OUTPUTS.items()))
        system = (
            "You are a ComfyUI workflow author. Propose NEW, genuinely useful "
            "workflows that are NOT already in the library. Reply with ONLY a "
            'JSON array; each item: {"name": "<snake_case, new>", "task": '
            '"<generate|img2img|inpaint|outpaint|upscale|video>", '
            '"description": "<one line>", "required_models": [], "graph": '
            "{<ComfyUI API-format graph>}}. Use ONLY these node types: "
            f"{', '.join(sorted(ALLOWED_NODE_TYPES))}. Valid output indices: "
            f"{outputs}. Every graph needs a SaveImage (or SaveAnimatedWEBP) "
            "node. Keep graphs correct and minimal.")
        ask = (f"Library already has: {', '.join(existing)}.\n"
               "Propose up to 3 advanced workflows we are missing (e.g. "
               "multi-pass refinement, tiled upscaling, mask compositing).")
        try:
            reply = self.llm.complete(system, ask, max_tokens=2000)
            data = json.loads(re.sub(r"^```(?:json)?|```$", "",
                                     reply.text.strip(), flags=re.M).strip())
        except (LLMError, json.JSONDecodeError) as exc:
            raise PermanentError(f"Could not get workflow ideas: {exc}") from exc
        if isinstance(data, dict):
            data = data.get("workflows") or data.get("candidates") or [data]
        accepted: list[dict[str, Any]] = []
        for item in data if isinstance(data, list) else []:
            if not isinstance(item, dict) or "graph" not in item:
                continue
            name = re.sub(r"[^a-z0-9_]", "", str(item.get("name", "")).lower())
            if not name or name in existing:
                continue
            try:
                validate_generated(item["graph"])
            except Exception as exc:  # noqa: BLE001 — reject invalid proposals
                job.log("info", f"Rejected '{name}': {exc}")
                continue
            # Bounded memory: keep only the most recent unapproved candidates.
            while len(self._workflow_candidates) >= 40:
                self._workflow_candidates.pop(
                    next(iter(self._workflow_candidates)))
            cand_id = uuid.uuid4().hex[:8]
            cand = {"id": cand_id, "name": name,
                    "task": str(item.get("task", "generate")),
                    "description": str(item.get("description", ""))[:200],
                    "required_models": [str(m) for m in
                                        (item.get("required_models") or [])],
                    "graph": item["graph"],
                    "nodes": len(item["graph"]),
                    "provenance": {"source": reply.source, "model": reply.model}}
            self._workflow_candidates[cand_id] = cand
            accepted.append({k: v for k, v in cand.items() if k != "graph"})
            job.log("info", f"[llm] candidate '{name}' ({cand['nodes']} nodes) "
                            "— validated, awaiting approval")
        if not accepted:
            job.log("info", "No new valid workflows this round.")
        return {"candidates": accepted}

    def _glob_template_exists(self, name: str) -> bool:
        return any(t["template"] == name for t in self.workflows.list_all())

    # Job kinds whose running work executes inside ComfyUI (interruptible).
    _COMFY_JOBS = {"workflow", "video", "image_edit", "avatar",
                   "avatar_render", "motion_transfer"}

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job. For a RUNNING render this also asks ComfyUI to
        interrupt the in-flight prompt so the GPU stops immediately."""
        job = self.queue.get(job_id)
        was_running = job is not None and job.state.value == "running"
        ok = self.queue.cancel(job_id)
        if ok and was_running and job is not None \
                and job.type in self._COMFY_JOBS:
            if self.comfy.interrupt():
                job.log("info", "Asked ComfyUI to interrupt the running render")
        return ok

    def save_candidate(self, cand_id: str, live_test: bool = True) -> dict[str, Any]:
        """Approve a discovered workflow: re-validate, optionally live-test a
        self-contained one, and write it into the library so it's usable and
        teaches the planner from now on."""
        from .workflow_ai import validate_generated
        cand = self._workflow_candidates.get(cand_id)
        if cand is None:
            raise PermanentError("Unknown or expired candidate.")
        validate_generated(cand["graph"])  # never save an invalid graph
        tested = "structural"
        needs_input = any(
            n.get("class_type") in ("LoadImage", "LoadImageMask")
            for n in cand["graph"].values())
        models_ready = all(self.registry.is_ready(m)
                           for m in cand["required_models"])
        if (live_test and not needs_input and models_ready
                and self.comfy.is_up()):
            try:
                # A discovered workflow can load a UNet as large as any
                # other; approving one must not be the single submit path
                # that skips the precision choice and the cache drop.
                self._prepare_graph(None, cand["graph"])
                self.comfy.run_graph(cand["graph"])
                tested = "live"
            except Exception as exc:  # noqa: BLE001 — approval fails loudly
                raise PermanentError(
                    f"Live test failed, not saving: {exc}") from exc
        template = {"template": cand["name"], "version": 1,
                    "task": cand["task"], "description": cand["description"],
                    "required_models": cand["required_models"],
                    "parameters": {}, "graph": cand["graph"],
                    "authored_by": cand["provenance"]}
        # cand["name"] is already sanitized to [a-z0-9_]; guard anyway so a
        # candidate can never write outside the user workflows dir.
        safe = re.sub(r"[^a-z0-9_]", "", cand["name"])
        if not safe or self._glob_template_exists(safe):
            raise PermanentError(
                f"A template named '{cand['name']}' already exists.")
        path = self._user_workflows / f"{safe}_v1.json"
        path.write_text(json.dumps(template, indent=1))
        self._workflow_candidates.pop(cand_id, None)
        return {"saved": cand["name"], "verified": tested, "path": str(path)}

    def _handle_setup(self, job: Job) -> dict[str, Any]:
        """First-run setup: the LLM looks at this machine's hardware and the
        registry and decides what to pre-download; conservative rules decide
        when the LLM is unavailable."""
        hw = job.payload.get("hardware", self.hardware.to_dict())
        job.log("info", f"[stage] hardware — {hw.get('gpu_name') or 'no NVIDIA GPU'}, "
                        f"{hw.get('vram_gb')} GB VRAM, {hw.get('ram_gb')} GB RAM, "
                        f"{hw.get('disk_free_gb')} GB free (tier: {hw.get('tier')})")

        candidates = [
            {"name": m.name, "purpose": m.purpose, "vram_gb": m.vram_gb,
             "ready": self.registry.is_ready(m.name)}
            for m in self.registry.list() if m.url]
        picks: list[str] = []
        try:
            reply = self.llm.complete(
                "You configure an AI image studio for a specific machine. "
                "Given the hardware and available models, reply ONLY JSON: "
                '{"download": ["<model name>", ...], "reason": "<short>"}. '
                "Pick only models the GPU can run (model vram_gb <= machine "
                "VRAM + 2), skip ones already ready, and prefer the minimal "
                "useful set (an image checkpoint first; video/multiview only "
                "on >= 8 GB VRAM and ample disk). On >= 8 GB also consider "
                "the speed LoRAs and zimage-turbo (+ zimage-text-encoder + "
                "flux-ae) for fast photoreal and legible text; on >= 20 GB "
                "VRAM stage qwen-image-edit-2511, the flagship instruction "
                "editor.",
                f"Hardware: {json.dumps(hw)}\nModels: {json.dumps(candidates)}",
                max_tokens=300)
            data = json.loads(re.sub(r"^```(?:json)?|```$", "",
                                     reply.text.strip(), flags=re.M).strip())
            names = {c["name"] for c in candidates}
            picks = [p for p in data.get("download", []) if p in names]
            job.log("info", f"LLM setup plan ({reply.source}:{reply.model}): "
                            f"{', '.join(picks) or 'nothing'} — "
                            f"{data.get('reason', '')}")
        except (LLMError, json.JSONDecodeError, AttributeError,
                TypeError) as exc:
            job.log("error", f"LLM unavailable for setup ({exc}); using rules")
            vram_rule = float(hw.get("vram_gb") or 0)
            if vram_rule >= 6:
                picks = ["sd15-inpaint"]
            if vram_rule >= 20:
                # Big-GPU machines get the flagship editor staged up front.
                picks.append("qwen-image-edit-2511")

        # Deterministic hardware gate ON TOP of whatever was picked: a model
        # declaring meta.min_vram_gb is never staged on a smaller GPU, no
        # matter who chose it — this is how a powerful machine automatically
        # receives the heavy models while an 8 GB one never wastes 20 GB.
        vram = float(hw.get("vram_gb") or 0)
        ram = float(hw.get("ram_gb") or 0)
        gated: list[str] = []
        for name in list(picks):
            m = self.registry.get(name)
            meta = (m.meta or {}) if m else {}
            need_v = float(meta.get("min_vram_gb") or 0)
            need_r = float(meta.get("min_ram_gb") or 0)
            if (need_v and vram < need_v) or (need_r and ram < need_r):
                picks.remove(name)
                gated.append(name)
        if gated:
            job.log("info", "Skipped (needs a bigger machine): "
                            + ", ".join(gated))

        job.log("info", "[stage] models — staging downloads")
        queued = []
        for name in picks:
            if not self.registry.is_ready(name):
                self.queue.enqueue("model_download", {"model": name})
                queued.append(name)
                job.log("info", f"Queued download: {name}")
        return {"hardware": hw, "queued": queued}

    def _handle_model_download(self, job: Job) -> dict[str, Any]:
        name = job.payload["model"]
        try:
            # Shares the workflow path's machinery: checksum backfill from the
            # hub, gated-source self-healing via verified public mirrors.
            # requested=True: this job type IS the user's explicit ask (or an
            # enqueue site that applied its own auto_install check), so the
            # auto-install gate does not apply.
            self._ensure_model(name, job, requested=True)
        except PermanentError:
            raise
        except DownloadError as exc:
            msg = str(exc)
            # Bad source / bad file / needs user action: not worth retrying.
            permanent = ("Checksum mismatch" in msg or "not in the registry" in msg
                         or "no download URL" in msg or "blocked" in msg
                         or "civitai requires" in msg.lower())
            raise (PermanentError if permanent else TransientError)(msg) from exc
        model = self.registry.get(name)
        job.log("info", f"Model {name} ready at {model.path if model else '?'}")
        # New model on board: research what it performs best at so the
        # planner can route prompts to it. Not just checkpoints — the
        # diffusion_models folder holds the strongest generators (Z-Image,
        # Kontext, WAN) and LoRAs change what a prompt can achieve; leaving
        # them out meant the knowledge file steered around the best tools.
        if (model and model.status == "ready"
                and (model.meta or {}).get("folder", "checkpoints")
                in self._RESEARCH_FOLDERS):
            fname = Path(str((model.meta or {}).get("file")
                             or model.path or "")).name
            if fname and fname not in self._intel_queued:
                self._intel_queued.add(fname)
                self.queue.enqueue("model_research", {"file": fname})
                job.log("info", f"Queued capability research for '{fname}' — "
                                "its strengths will guide future model "
                                "choices")
        return {"model": name,
                "status": model.status if model else "unknown",
                "path": model.path if model else None}
