"""Curated ComfyUI node-pack installer.

Custom node packs extend ComfyUI with capabilities core nodes don't have
(face detailing, pose preprocessing, frame interpolation, GGUF loading...).
They are THIRD-PARTY CODE, so this module only knows a CURATED list of
well-maintained packs pinned to their official GitHub repos — nothing is
ever installed from a name the LLM invented. Installation is a visible job
in the queue; after install ComfyUI is restarted and the pack is verified
against the live /object_info (a pack either provably registered its nodes
or the status says exactly what happened).

Status vocabulary (honest, probed — never assumed):
  absent          not on disk
  installed       directory exists, ComfyUI not up to confirm the nodes
  active          directory exists AND its marker node is live in ComfyUI
  broken          directory exists but ComfyUI does not expose its nodes
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NodePack:
    name: str                 # short id used in the API/UI
    title: str
    purpose: str              # user-facing: what does installing this unlock?
    repo: str                 # owner/repo on GitHub
    dir_name: str             # folder name under custom_nodes/
    verify_node: str          # a node type that MUST appear when active
    unlocks: str              # which PromptForge feature waits on this pack
    extra_repos: tuple = ()   # (repo, dir_name) companions installed with it
    note: str = ""            # honest caveats (build tools, size, speed)
    provides: tuple = ()      # node class_types our graphs use from this pack
                              # (beyond verify_node) — lets a missing-node
                              # error name the pack that heals it


KNOWN_PACKS: dict[str, NodePack] = {p.name: p for p in [
    NodePack(
        name="impact-pack",
        title="Impact Pack (FaceDetailer)",
        purpose="Automatic face detection + high-res re-render of every "
                "face — fixes the classic 'mushy face' in full-body shots.",
        repo="ltdrdata/ComfyUI-Impact-Pack",
        dir_name="ComfyUI-Impact-Pack",
        verify_node="FaceDetailer",
        unlocks="face refinement pass after renders and edits",
        extra_repos=(("ltdrdata/ComfyUI-Impact-Subpack",
                      "ComfyUI-Impact-Subpack"),),
        note="Downloads its detector models (~450 MB) on first use.",
        provides=("UltralyticsDetectorProvider", "SAMLoader"),
    ),
    NodePack(
        name="controlnet-aux",
        title="ControlNet preprocessors",
        purpose="Pose / depth / lineart / scribble extraction from images — "
                "unlocks pose-guided and depth-guided generation (canny "
                "works without this pack).",
        repo="Fannovel16/comfyui_controlnet_aux",
        dir_name="comfyui_controlnet_aux",
        verify_node="CannyEdgePreprocessor",
        unlocks="pose/depth ControlNet guidance (with controlnet-union-sdxl)",
        provides=("DWPreprocessor", "DepthAnythingPreprocessor",
                  "LineArtPreprocessor"),
    ),
    NodePack(
        name="frame-interpolation",
        title="RIFE frame interpolation",
        purpose="Smooths videos: 24 fps WAN clips become 48/60 fps, or "
                "slow-motion — the biggest perceived video-quality win.",
        repo="Fannovel16/ComfyUI-Frame-Interpolation",
        dir_name="ComfyUI-Frame-Interpolation",
        verify_node="RIFE VFI",
        unlocks="smooth 48/60 fps video output",
        note="Downloads RIFE weights (~100 MB) on first use.",
    ),
    NodePack(
        name="rmbg",
        title="Background removal (RMBG / BiRefNet)",
        purpose="One-click clean subject cutouts and text-prompted "
                "segmentation masks.",
        repo="1038lab/ComfyUI-RMBG",
        dir_name="ComfyUI-RMBG",
        verify_node="RMBG",
        unlocks="transparent cutouts + better selection masks",
        note="Downloads its matting models (~1.5 GB) on first use.",
        provides=("BiRefNetRMBG", "ClothesSegment", "BodySegment",
                  "FaceSegment", "FashionSegment", "Segment", "SegmentV2"),
    ),
    NodePack(
        name="ic-light",
        title="IC-Light relighting",
        purpose="Move the light source: golden-hour any portrait, match a "
                "subject's lighting to a new background.",
        repo="kijai/ComfyUI-IC-Light",
        dir_name="ComfyUI-IC-Light",
        verify_node="LoadAndApplyICLightUnet",
        unlocks="relighting edits (with the iclight-sd15-fc model)",
        provides=("ICLightConditioning",),
    ),
    NodePack(
        name="instantid",
        title="InstantID identity",
        purpose="Strong identity-preserving renders from ONE photo — "
                "stronger identity lock than PhotoMaker.",
        repo="cubiq/ComfyUI_InstantID",
        dir_name="ComfyUI_InstantID",
        verify_node="ApplyInstantID",
        unlocks="single-photo identity renders",
        provides=("InstantIDModelLoader", "InstantIDFaceAnalysis"),
        note="Its InsightFace dependency sometimes needs Microsoft C++ "
             "Build Tools on Windows — the install reports honestly if "
             "that happens.",
    ),
    NodePack(
        name="gguf",
        title="GGUF model loader",
        purpose="Loads quantized GGUF models — REQUIRED for Flux Kontext "
                "instruction editing on 8 GB GPUs.",
        repo="city96/ComfyUI-GGUF",
        dir_name="ComfyUI-GGUF",
        verify_node="UnetLoaderGGUF",
        unlocks="Flux Kontext instruction-based editing (kontext)",
        provides=("DualCLIPLoaderGGUF", "CLIPLoaderGGUF"),
    ),
]}


def pack_for_node(class_type: str) -> NodePack | None:
    """The curated pack that ships a node type, or None. This is what turns
    a ComfyUI 'missing_node_type' rejection into an actionable install —
    only ever a curated pack, never a guess from the node's name."""
    for pack in KNOWN_PACKS.values():
        if class_type == pack.verify_node or class_type in pack.provides:
            return pack
    return None


def zip_urls(repo: str) -> list[str]:
    """Candidate archive URLs for a repo's default branch (branch name
    varies across projects — try the common ones)."""
    return [f"https://github.com/{repo}/archive/refs/heads/{b}.zip"
            for b in ("main", "master", "Main")]


def pack_status(pack: NodePack, comfy_base: Path | None,
                live_nodes: set[str] | None) -> dict[str, Any]:
    """Probed status for one pack. `live_nodes` is the live object_info key
    set (None when ComfyUI is down — we then only know what's on disk)."""
    on_disk = bool(comfy_base
                   and (comfy_base / "custom_nodes" / pack.dir_name).exists())
    if not on_disk:
        status = "absent"
    elif live_nodes is None:
        status = "installed"
    elif pack.verify_node in live_nodes:
        status = "active"
    else:
        status = "broken"
    return {"name": pack.name, "title": pack.title, "purpose": pack.purpose,
            "repo": pack.repo, "status": status, "unlocks": pack.unlocks,
            "note": pack.note, "verify_node": pack.verify_node}
