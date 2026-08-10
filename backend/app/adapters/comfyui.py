"""ComfyUI adapter (prepared backend — see README "Implemented vs prepared").

What works today, fully offline and unit-tested:
  * loading versioned workflow templates from app/workflows/
  * generating a concrete workflow from user intent — ONLY for allowed task
    types; anything else raises WorkflowNotAllowedError
  * structural validation of a workflow graph before it may execute
  * model-readiness checks against the ModelRegistry

What requires a running ComfyUI instance (and is exercised only then):
  * submit(): POST /prompt, poll /history, fetch the output image.
    Connection failures raise BackendUnavailableError so the job queue
    retries/reports instead of crashing.
"""
from __future__ import annotations

import copy
import io
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from ..core.registry import ModelRegistry
from .base import (
    AdapterError,
    BackendUnavailableError,
    EditResult,
    ModelMissingError,
    validate_mask,
)

# Task types the app is willing to translate into workflows — via versioned
# templates or via the LLM generator (core/workflow_ai.py). Everything else is
# rejected regardless of what a template file or a model reply might contain.
ALLOWED_TASKS = {"inpaint", "generate", "img2img", "upscale", "outpaint",
                 "video", "video_inpaint", "video_outpaint", "angles",
                 "identity", "relight", "compose", "motion_transfer",
                 "background", "reconstruct", "pose", "scene3d", "kontext"}

# Node classes permitted in generated/validated workflows. Extend deliberately.
ALLOWED_NODE_TYPES = {
    "KSampler", "CheckpointLoaderSimple", "CLIPTextEncode", "VAEDecode",
    "VAEEncode", "VAEEncodeForInpaint", "SaveImage", "LoadImage", "LoadImageMask",
    "EmptyLatentImage", "LatentUpscale", "ImageScale",
    # Video (WAN 2.2) — used by the versioned video template.
    "UNETLoader", "VAELoader", "CLIPLoader", "ModelSamplingSD3",
    "WanImageToVideo", "SaveAnimatedWEBP",
    # Multi-view synthesis (SV3D) — used by the versioned angles template.
    "ImageOnlyCheckpointLoader", "SV3D_Conditioning", "VideoLinearCFGGuidance",
    # Identity-preserving generation (PhotoMaker, ComfyUI core nodes) — used
    # by the versioned identity template for avatar renders.
    "PhotoMakerLoader", "PhotoMakerEncode",
    # Outpainting, model-based upscaling and mask utilities (core nodes).
    "ImagePadForOutpaint", "UpscaleModelLoader", "ImageUpscaleWithModel",
    "GrowMask", "FeatherMask", "InvertMask", "ImageInvert",
    # WAN 2.1 VACE video editing (inpaint/outpaint on videos, core nodes).
    "WanVaceToVideo", "TrimVideoLatent",
    # Image → 3D mesh (Hunyuan3D v2). ALL of these ship in ComfyUI core
    # (comfy_extras.nodes_hunyuan3d) — there is no node pack to install and
    # nothing to compile, which is why this is the reconstruction backend
    # rather than TripoSR/InstantMesh/TRELLIS (all of which pin an older
    # torch/transformers than this machine runs).
    "CLIPVisionEncode", "Hunyuan3Dv2Conditioning",
    "Hunyuan3Dv2ConditioningMultiView", "EmptyLatentHunyuan3Dv2",
    "VAEDecodeHunyuan3D", "VoxelToMesh", "VoxelToMeshBasic", "SaveGLB",
    # Batching references into one IMAGE (multi-photo identity).
    "ImageBatch",
    # Multi-region segmentation (rmbg pack). A point-and-grow segmenter
    # returns ONE blob, so "change the bikini" came back as the top only.
    # These select every named part at once and union them.
    "ClothesSegment", "BodySegment", "Segment", "SegmentV2",
    # Photo -> navigable 3D SCENE (MoGe, ComfyUI core since v0.22). Predicts
    # a METRIC point map plus the camera FOV, then triangulates it with a
    # depth-discontinuity cut and UVs the original photo onto it. The old
    # depth preprocessors returned 8-bit disparity and could not be meshed.
    "LoadMoGeModel", "MoGeInference", "MoGePanoramaInference",
    "MoGePointMapToMesh", "MoGeRender",
    # Advanced inpainting (all core nodes, verified against live ComfyUI):
    # modern conditioning, soft/differential inpainting, any-checkpoint
    # latent-mask method, and crop→upscale→inpaint→stitch plumbing.
    "InpaintModelConditioning", "DifferentialDiffusion", "SetLatentNoiseMask",
    "ImageCrop", "ImageCompositeMasked", "MaskToImage", "ImageToMask",
    # Speed LoRAs + draft modes (core nodes).
    "LoraLoader", "LoraLoaderModelOnly",
    # ControlNet-guided generation/editing (core nodes; canny preprocessing
    # is core, other preprocessors come from the controlnet_aux node pack).
    "ControlNetLoader", "ControlNetApplyAdvanced", "Canny",
    # Regional prompting (core nodes).
    "ConditioningSetMask", "ConditioningCombine",
    # Z-Image Turbo (core nodes since ComfyUI 0.7x) and Flux-family flows
    # (Kontext instruction editing) — the GGUF loader comes from the
    # ComfyUI-GGUF node pack and is a DELIBERATE pack-node allowance.
    "ModelSamplingAuraFlow", "EmptySD3LatentImage", "ConditioningZeroOut",
    "DualCLIPLoader", "ReferenceLatent", "FluxGuidance", "UnetLoaderGGUF",
    # IC-Light relighting — the only way to actually change a photo's LIGHT
    # (img2img repaints the picture instead). Both come from the ic-light
    # node pack and are a DELIBERATE pack-node allowance, like UnetLoaderGGUF.
    "LoadAndApplyICLightUnet", "ICLightConditioning",
    # Subject matting for compositing (rmbg pack). Measured on this machine:
    # BiRefNet returned 19.4% coverage against a 19.4% ground truth, where
    # SAM — a PART segmenter — returned 8.7% (the person's shirt).
    "BiRefNetRMBG",
    # Face-region matte (rmbg pack), for replacing a face rather than a whole
    # head — hair and ears are deliberately left out of the selection.
    "FaceSegment",
    # Pose extraction from a driving video (controlnet_aux pack), for motion
    # transfer. NOTE: bbox_detector MUST be the .torchscript.pt variant —
    # onnxruntime here has no CUDA provider, and the .onnx default runs on
    # CPU at over ten minutes for 25 frames.
    "DWPreprocessor",
    # InstantID face identity (instantid pack). Consent-gated in the
    # pipeline. ApplyInstantID returns model/positive/negative in that order.
    "InstantIDModelLoader", "InstantIDFaceAnalysis", "ApplyInstantID",
}


class WorkflowNotAllowedError(AdapterError):
    pass


class WorkflowValidationError(AdapterError):
    pass


class WorkflowRuntimeError(AdapterError):
    """ComfyUI accepted the graph but a node failed while executing (or the
    graph was rejected at submit). The message carries ComfyUI's own error
    details so the LLM repair loop can act on them."""


class WorkflowLibrary:
    """Loads versioned templates: <task>_v<N>.json.

    Two locations are scanned: the packaged `dir` (the shipped templates) and
    an optional writable `user_dir` where the LLM saves newly-discovered
    workflows. Keeping user-authored templates out of the package dir means
    the library grows without mutating source and stays hermetic in tests.
    """

    def __init__(self, workflows_dir: Path, user_dir: Path | None = None):
        self.dir = workflows_dir
        self.user_dir = user_dir

    def _dirs(self) -> list[Path]:
        return [d for d in (self.dir, self.user_dir) if d and d.exists()]

    def _glob(self, pattern: str) -> list[Path]:
        found: list[Path] = []
        for d in self._dirs():
            found.extend(d.glob(pattern))
        return found

    def load(self, task: str, version: int | None = None) -> dict[str, Any]:
        if task not in ALLOWED_TASKS:
            raise WorkflowNotAllowedError(
                f"Task '{task}' is not an allowed workflow type.")
        if version is None:
            candidates = sorted(self._glob(f"{task}_v*.json"),
                                key=lambda p: p.name)
            if not candidates:
                raise WorkflowValidationError(f"No template found for task '{task}'.")
            path = candidates[-1]
        else:
            found = next((p for p in self._glob(f"{task}_v{version}.json")), None)
            if found is None:
                raise WorkflowValidationError(
                    f"Template {task}_v{version}.json does not exist.")
            path = found
        template = json.loads(path.read_text())
        validate_workflow(template["graph"])
        return template

    def load_named(self, prefix: str) -> dict[str, Any]:
        """Load a template VARIANT by its file prefix (e.g.
        'inpaint_universal'). The variant's declared task must still be an
        allowed one — variants never bypass the task gate."""
        candidates = sorted(self._glob(f"{prefix}_v*.json"),
                            key=lambda p: p.name)
        if not candidates:
            raise WorkflowValidationError(
                f"No template named '{prefix}' in the library.")
        template = json.loads(candidates[-1].read_text())
        if template.get("task", prefix) not in ALLOWED_TASKS:
            raise WorkflowNotAllowedError(
                f"Template '{prefix}' declares a disallowed task.")
        validate_workflow(template["graph"])
        return template

    def knowledge(self, task: str, max_chars: int = 6000) -> str | None:
        """Teaching context for the planner LLM: the matching section of
        WORKFLOW_GUIDE.md plus one validated example template for the task.
        Kept compact — the local model's context window is finite."""
        parts: list[str] = []
        guide = self.dir / "WORKFLOW_GUIDE.md"
        if guide.exists():
            text = guide.read_text(encoding="utf-8")
            sections = re.split(r"^## ", text, flags=re.M)
            for want in ("Global rules", f"Task: {task}",
                         "Common errors and fixes"):
                for sec in sections:
                    if sec.startswith(want):
                        parts.append("## " + sec.strip())
                        break
        examples = [t for t in self.list_all() if t.get("task") == task]
        if examples:
            t = examples[0]
            parts.append(
                f"Validated example template '{t['template']}' "
                f"({t.get('description', '')[:160]}):\n"
                + json.dumps(t["graph"], separators=(",", ":")))
        if not parts:
            return None
        return "\n\n".join(parts)[:max_chars]

    def list_all(self) -> list[dict[str, Any]]:
        """Every template in the library (validated), newest version first
        per name. This is the LLM's example book: variants beyond the
        directly-executed defaults exist to teach it real, working recipes."""
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for path in sorted(self._glob("*_v*.json"),
                           key=lambda p: p.name, reverse=True):
            name = path.stem.rsplit("_v", 1)[0]
            if name in seen:
                continue
            try:
                template = json.loads(path.read_text())
                validate_workflow(template["graph"])
            except (json.JSONDecodeError, KeyError, WorkflowValidationError):
                continue
            seen.add(name)
            out.append(template)
        out.sort(key=lambda t: t.get("template", ""))
        return out


def validate_workflow(graph: dict[str, Any]) -> None:
    """Structural validation of a ComfyUI API-format graph.

    Checks: node shape, allowlisted class types, and that every link
    references an existing node. Raises WorkflowValidationError.
    """
    if not isinstance(graph, dict) or not graph:
        raise WorkflowValidationError("Workflow graph is empty.")
    for node_id, node in graph.items():
        if not isinstance(node, dict) or "class_type" not in node:
            raise WorkflowValidationError(f"Node {node_id} is missing class_type.")
        ctype = node["class_type"]
        if ctype not in ALLOWED_NODE_TYPES:
            raise WorkflowValidationError(
                f"Node {node_id} uses disallowed type '{ctype}'.")
        for key, value in node.get("inputs", {}).items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                if value[0] not in graph:
                    raise WorkflowValidationError(
                        f"Node {node_id} input '{key}' links to missing node {value[0]}.")


def build_workflow(template: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Fill a template's declared parameter slots. Unknown params are
    rejected. A slot may be a single {node, input} or a LIST of them — one
    value fanning out to several inputs (e.g. crop coordinates used by the
    crop, mask-crop and stitch nodes of the hi-res inpaint template)."""
    graph = copy.deepcopy(template["graph"])
    slots = template.get("parameters", {})
    for key, value in params.items():
        if key not in slots:
            raise WorkflowValidationError(f"Template has no parameter '{key}'.")
        slot = slots[key]
        targets = slot if isinstance(slot, list) else [slot]
        for t in targets:
            graph[t["node"]]["inputs"][t["input"]] = value
    validate_workflow(graph)
    return graph


class ComfyUIClient:
    """HTTP client for a ComfyUI server: submit graphs, fetch results, and
    inspect what the server actually has installed (nodes, checkpoints)."""

    def __init__(self, base_url: str, poll_interval: float = 1.0,
                 timeout_s: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.poll_interval = poll_interval
        self.timeout_s = timeout_s

    # -- plumbing ---------------------------------------------------------------
    def request(self, method: str, path: str, data: bytes | None = None,
                headers: dict[str, str] | None = None) -> bytes:
        req = urllib.request.Request(self.base_url + path, data=data,
                                     headers=headers or {}, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            # ComfyUI answers 400 with a JSON body describing exactly which
            # node/input is wrong — surface that for the repair loop.
            try:
                detail = json.loads(exc.read())
            except Exception:
                detail = None
            if detail:
                raise WorkflowRuntimeError(
                    f"ComfyUI rejected the request: {json.dumps(detail)[:2000]}") from exc
            raise WorkflowRuntimeError(
                f"ComfyUI returned HTTP {exc.code} for {path}.") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise BackendUnavailableError(
                f"ComfyUI at {self.base_url} is unreachable: {exc}") from exc

    def is_up(self) -> bool:
        try:
            self.request("GET", "/system_stats")
            return True
        except AdapterError:
            return False

    def health(self) -> tuple[bool, str]:
        """(healthy, why-not). Distinguishes NOT LISTENING from LISTENING BUT
        BROKEN — two failures that need opposite responses.

        Seen live: after a run of out-of-memory kills the CUDA context wedged,
        and ComfyUI answered /system_stats with a 500 carrying
        `torch.AcceleratorError: CUDA error: unknown error`. The old
        boolean probe called that "down", so the app killed and respawned it
        four times into the same broken driver state and told the user it
        "did not come back" — which was both wrong and unactionable."""
        try:
            self.request("GET", "/system_stats")
            return True, ""
        except AdapterError as exc:
            text = str(exc)
            if "cuda" in text.lower() or "AcceleratorError" in text:
                return False, ("ComfyUI is running but its graphics driver is "
                               "in a bad state (CUDA error). Every ComfyUI "
                               "process has to be closed, not just restarted.")
            # Matched on the adapter's OWN wording, not on the OS error text —
            # that arrives in the machine's display language.
            if isinstance(exc, BackendUnavailableError) or "unreachable" in text:
                return False, "ComfyUI is not listening."
            return False, text[:200]

    def interrupt(self) -> bool:
        """Abort the currently-executing prompt (best effort)."""
        try:
            self.request("POST", "/interrupt", b"{}",
                         {"Content-Type": "application/json"})
            return True
        except AdapterError:
            return False

    def free_memory(self) -> bool:
        """Unload cached models inside ComfyUI and release RAM/VRAM.

        Called before memory-heavy renders (video): leftover checkpoints from
        earlier renders stacking on top of the ~17 GB WAN stack is exactly
        what OOM-killed the ComfyUI process on a 16 GB machine."""
        try:
            self.request("POST", "/free",
                         json.dumps({"unload_models": True,
                                     "free_memory": True}).encode(),
                         {"Content-Type": "application/json"})
            return True
        except AdapterError:
            return False

    # -- inventory ("know what tools are already there") --------------------------
    def object_info(self) -> dict[str, Any]:
        return json.loads(self.request("GET", "/object_info"))

    def installed_node_types(self) -> set[str]:
        return set(self.object_info().keys())

    def installed_checkpoints(self) -> list[str]:
        """Checkpoint files ComfyUI can currently load."""
        try:
            info = json.loads(self.request(
                "GET", "/object_info/CheckpointLoaderSimple"))
            options = (info["CheckpointLoaderSimple"]["input"]["required"]
                       ["ckpt_name"][0])
            return [o for o in options if isinstance(o, str)]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            return []

    # -- execution ----------------------------------------------------------------
    def upload_image(self, image: Image.Image, prefix: str) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        boundary = uuid.uuid4().hex
        filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            f"Content-Type: image/png\r\n\r\n"
        ).encode() + buf.getvalue() + f"\r\n--{boundary}--\r\n".encode()
        self.request("POST", "/upload/image", body,
                     {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        return filename

    def upload_frames(self, frames: list[Image.Image], prefix: str,
                      fps: float = 16.0) -> str:
        """Upload a SEQUENCE of frames as one file ComfyUI reads as a batch.

        Encoded as a lossless animated WEBP, because core LoadImage walks
        every frame of an animated file and concatenates them into a single
        IMAGE batch. That is the whole trick behind driving a render from a
        video here: no video-loading node pack, no new node types, just the
        upload endpoint that already exists."""
        if not frames:
            raise WorkflowValidationError("No frames to upload.")
        # One shared encoder, which writes lossless AND counts the frames back
        # off disk: a lossy animated WEBP drops near-identical frames with no
        # error anywhere, and a driving clip quietly one frame short desyncs
        # everything downstream of it.
        from app.core.video import encode_animation
        data = encode_animation(frames, fps=fps)
        boundary = uuid.uuid4().hex
        filename = f"{prefix}_{uuid.uuid4().hex[:8]}.webp"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            f"Content-Type: image/webp\r\n\r\n"
        ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
        self.request("POST", "/upload/image", body,
                     {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        return filename

    def submit(self, graph: dict[str, Any]) -> str:
        payload = json.dumps({"prompt": graph}).encode()
        raw = self.request("POST", "/prompt", payload,
                           {"Content-Type": "application/json"})
        return json.loads(raw)["prompt_id"]

    # A render that is still going must never be reported as a dead backend:
    # the caller answers that by RESTARTING ComfyUI, which throws away the
    # work in flight and steps the settings down. Measured here: a WAN video
    # finished sampling in 5:04 and then spent ~21 more minutes in VAE decode
    # (prompt executed in 26:39), so a flat 10-minute deadline killed four
    # healthy renders in a row and walked 704x1280 down to 256x256.
    # Hard stop, so neither a wedged queue nor a thrashing machine can hang a
    # job for ever. 6x the base timeout is an hour by default, against a
    # measured healthy range of 12:50-26:39 for a WAN clip on an 8 GB card —
    # roughly double the worst honest render.
    #
    # NOTE this only guards the DEADLINE path. A render can also end because
    # ComfyUI stops answering HTTP under memory pressure, in which case
    # request() raises BackendUnavailableError straight out of the loop and
    # never consults these numbers. Measured on this machine: the GPU pinned
    # at 100% with 0.4 GB of system RAM free (swapping) and ComfyUI silent
    # after ~8 minutes. Distinguishing "busy" from "gone" there needs a
    # retry-with-backoff around request(), which this does not yet do.
    _MAX_WAIT_MULTIPLE = 6

    def _still_executing(self, prompt_id: str) -> bool:
        """True when ComfyUI is reachable AND this prompt is still queued or
        running. Anything else (gone, unreachable, finished) is False."""
        try:
            queue = json.loads(self.request("GET", "/queue"))
        except Exception:  # noqa: BLE001 — unreachable means not working
            return False
        for key in ("queue_running", "queue_pending"):
            for item in queue.get(key) or []:
                if not isinstance(item, list | tuple):
                    continue
                if any(str(part) == str(prompt_id) for part in item
                       if isinstance(part, str | int)):
                    return True
        return False

    def _next_deadline(self, prompt_id: str, deadline: float,
                       started: float) -> float | None:
        """The deadline to keep waiting until, or None to give up."""
        if time.monotonic() < deadline:
            return deadline
        if time.monotonic() - started > self.timeout_s * self._MAX_WAIT_MULTIPLE:
            return None
        if self._still_executing(prompt_id):
            return time.monotonic() + self.timeout_s
        return None

    # A single refused connection is NOT proof the renderer is gone. Under
    # memory pressure ComfyUI stops answering HTTP for a stretch and then
    # comes back — measured here, silent from 00:50:34 to 00:58:28 while the
    # GPU sat at 100%. The caller answers "gone" by restarting ComfyUI and
    # throwing the render away, so only SUSTAINED silence may count as death.
    _UNREACHABLE_GRACE_S = 90.0

    def _history_entry(self, prompt_id: str,
                       silent_since: float | None
                       ) -> tuple[dict[str, Any] | None, float | None]:
        """(this prompt's history entry or None, when silence began or None).

        Re-raises the connection error only once ComfyUI has been unreachable
        for longer than the grace period."""
        try:
            raw = self.request("GET", f"/history/{prompt_id}")
        except BackendUnavailableError:
            now = time.monotonic()
            if silent_since is None:
                return None, now
            if now - silent_since > self._UNREACHABLE_GRACE_S:
                raise
            return None, silent_since
        return json.loads(raw).get(prompt_id), None

    def wait_for_output_all(self, prompt_id: str) -> list[tuple[bytes, str]]:
        """All output files of a finished prompt (e.g. one per SV3D view)."""
        started = time.monotonic()
        deadline: float | None = started + self.timeout_s
        silent_since: float | None = None
        while deadline is not None:
            entry, silent_since = self._history_entry(prompt_id,
                                                     silent_since)
            if entry:
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    raise WorkflowRuntimeError(
                        "ComfyUI execution failed: "
                        + json.dumps(status.get("messages", []))[:2000])
                files: list[tuple[bytes, str]] = []
                for node_output in entry.get("outputs", {}).values():
                    for img in node_output.get("images", []):
                        q = urllib.parse.urlencode({
                            "filename": img["filename"],
                            "subfolder": img.get("subfolder", ""),
                            "type": img.get("type", "output")})
                        files.append((self.request("GET", f"/view?{q}"),
                                      img["filename"]))
                if not files:
                    raise WorkflowRuntimeError(
                        "ComfyUI finished but produced no output files.")
                return files
            time.sleep(self.poll_interval)
            deadline = self._next_deadline(prompt_id, deadline,
                                           started)
        raise BackendUnavailableError(
            "Timed out waiting for ComfyUI to render.")

    def wait_for_mesh(self, prompt_id: str) -> tuple[bytes, str]:
        """The .glb a mesh graph produced.

        SaveGLB does not report under `images` — ComfyUI lists 3D results
        under their own key (`result`/`3d`/`meshes` depending on version), so
        this scans every output value for a filename ending in .glb rather
        than assuming one key."""
        started = time.monotonic()
        deadline: float | None = started + self.timeout_s
        silent_since: float | None = None
        while deadline is not None:
            entry, silent_since = self._history_entry(prompt_id,
                                                     silent_since)
            if entry:
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    raise WorkflowRuntimeError(
                        "ComfyUI execution failed: "
                        + json.dumps(status.get("messages", []))[:2000])
                for node_output in entry.get("outputs", {}).values():
                    for group in node_output.values():
                        if not isinstance(group, list):
                            continue
                        for item in group:
                            name = (item.get("filename")
                                    if isinstance(item, dict) else str(item))
                            if not name or not name.lower().endswith(".glb"):
                                continue
                            sub = (item.get("subfolder", "")
                                   if isinstance(item, dict) else "")
                            kind = (item.get("type", "output")
                                    if isinstance(item, dict) else "output")
                            q = urllib.parse.urlencode({
                                "filename": name, "subfolder": sub,
                                "type": kind})
                            return self.request("GET", f"/view?{q}"), name
                raise WorkflowRuntimeError(
                    "ComfyUI finished but produced no .glb mesh.")
            time.sleep(self.poll_interval)
            deadline = self._next_deadline(prompt_id, deadline,
                                           started)
        raise BackendUnavailableError(
            "Timed out waiting for ComfyUI to render.")

    def wait_for_output_file(self, prompt_id: str) -> tuple[bytes, str]:
        """Like wait_for_output, but returns the raw bytes + filename of the
        first output — required for animated results (webp video), where
        decoding through PIL would keep only the first frame."""
        started = time.monotonic()
        deadline: float | None = started + self.timeout_s
        silent_since: float | None = None
        while deadline is not None:
            entry, silent_since = self._history_entry(prompt_id,
                                                     silent_since)
            if entry:
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    raise WorkflowRuntimeError(
                        "ComfyUI execution failed: "
                        + json.dumps(status.get("messages", []))[:2000])
                for node_output in entry.get("outputs", {}).values():
                    for img in node_output.get("images", []):
                        q = urllib.parse.urlencode({
                            "filename": img["filename"],
                            "subfolder": img.get("subfolder", ""),
                            "type": img.get("type", "output")})
                        return self.request("GET", f"/view?{q}"), img["filename"]
                raise WorkflowRuntimeError(
                    "ComfyUI finished but produced no output file.")
            time.sleep(self.poll_interval)
            deadline = self._next_deadline(prompt_id, deadline,
                                           started)
        raise BackendUnavailableError(
            "Timed out waiting for ComfyUI to render.")

    def wait_for_output(self, prompt_id: str) -> Image.Image:
        started = time.monotonic()
        deadline: float | None = started + self.timeout_s
        silent_since: float | None = None
        while deadline is not None:
            entry, silent_since = self._history_entry(prompt_id,
                                                     silent_since)
            if entry:
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    raise WorkflowRuntimeError(
                        "ComfyUI execution failed: "
                        + json.dumps(status.get("messages", []))[:2000])
                for node_output in entry.get("outputs", {}).values():
                    for img in node_output.get("images", []):
                        q = urllib.parse.urlencode({
                            "filename": img["filename"],
                            "subfolder": img.get("subfolder", ""),
                            "type": img.get("type", "output")})
                        data = self.request("GET", f"/view?{q}")
                        return Image.open(io.BytesIO(data)).convert("RGB")
                raise WorkflowRuntimeError(
                    "ComfyUI finished but produced no image output "
                    "(is a SaveImage node wired to the final image?).")
            time.sleep(self.poll_interval)
            deadline = self._next_deadline(prompt_id, deadline,
                                           started)
        raise BackendUnavailableError(
            "Timed out waiting for ComfyUI to render.")

    def run_graph(self, graph: dict[str, Any]) -> tuple[Image.Image, str]:
        """Submit a validated graph and wait for its image. Returns (image, prompt_id)."""
        validate_workflow(graph)  # belt-and-braces: never send unvalidated graphs
        prompt_id = self.submit(graph)
        return self.wait_for_output(prompt_id), prompt_id


class ComfyUIInpaintingAdapter:
    name = "comfyui-inpaint"
    is_mock = False
    # The edit pipeline may pass checkpoint/variant/negative kwargs.
    supports_variants = True

    # Advanced inpainting techniques (all validated core-node templates):
    #   modern    — InpaintModelConditioning + DifferentialDiffusion (soft
    #               inpainting) with a dedicated inpaint checkpoint.
    #   universal — SetLatentNoiseMask: lets ANY checkpoint inpaint, so the
    #               LLM can pick photoreal community models freely.
    #   hires     — crop the masked region (+margin), upscale, inpaint at
    #               high resolution, stitch back — detail for small regions.
    VARIANTS = {"modern": "inpaint", "universal": "inpaint_universal",
                "hires": "inpaint_hires"}

    def __init__(self, base_url: str, workflows: WorkflowLibrary,
                 registry: ModelRegistry, poll_interval: float = 1.0,
                 timeout_s: float = 600.0):
        self.client = ComfyUIClient(base_url, poll_interval, timeout_s)
        self.workflows = workflows
        self.registry = registry

    @staticmethod
    def _crop_params(image: Image.Image, mask: Image.Image,
                     max_up: int = 1024) -> dict[str, int]:
        """Crop rectangle (mask bbox + breathing room, snapped to /8) and the
        upscale size for the hi-res crop→inpaint→stitch variant."""
        bbox = mask.getbbox() or (0, 0, image.width, image.height)
        left, top, right, bottom = bbox
        margin = max(32, int(max(right - left, bottom - top) * 0.25))
        x = max(0, (left - margin) // 8 * 8)
        y = max(0, (top - margin) // 8 * 8)
        w = min(image.width - x, (right + margin - x + 7) // 8 * 8)
        h = min(image.height - y, (bottom + margin - y + 7) // 8 * 8)
        w = max(64, w // 8 * 8)
        h = max(64, h // 8 * 8)
        factor = max(1.0, min(2.0, max_up / max(w, h)))
        return {"crop_x": x, "crop_y": y, "crop_w": w, "crop_h": h,
                "up_w": max(64, int(w * factor) // 8 * 8),
                "up_h": max(64, int(h * factor) // 8 * 8)}

    # -- pipeline -------------------------------------------------------------
    def inpaint(self, image: Image.Image, mask: Image.Image, prompt: str, *,
                negative: str = "", checkpoint: str | None = None,
                variant: str = "modern",
                denoise: float | None = None) -> EditResult:
        mask = validate_mask(image, mask)
        prefix = self.VARIANTS.get(variant, "inpaint")
        template = (self.workflows.load("inpaint") if prefix == "inpaint"
                    else self.workflows.load_named(prefix))

        # With an explicit checkpoint the model comes from ComfyUI's own
        # installed list, so the template's default model needn't be staged.
        if not checkpoint:
            missing = [m for m in template.get("required_models", [])
                       if not self.registry.is_ready(m)]
            if missing:
                raise ModelMissingError(
                    "Required models are not downloaded: " + ", ".join(missing)
                    + ". Download them from the Models page first.")

        image_name = self.client.upload_image(image, "input")
        # The template loads the mask with LoadImageMask channel="red":
        # channel values are used directly (white = edit region), unlike
        # LoadImage's MASK output, which inverts the alpha channel.
        # Ref: docs.comfy.org/custom-nodes/backend/images_and_masks
        mask_name = self.client.upload_image(mask.convert("RGB"), "mask")
        params: dict[str, Any] = {
            "prompt": prompt,
            "image": image_name,
            "mask": mask_name,
            # Random, not clock-derived: a quality retry that lands in the
            # same wall-clock second as its first attempt would otherwise
            # re-render a BIT-IDENTICAL image and "discard" itself.
            "seed": int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF,
        }
        slots = template.get("parameters", {})
        if negative and "negative" in slots:
            params["negative"] = negative
        if checkpoint and "checkpoint" in slots:
            params["checkpoint"] = checkpoint
        # A structure-preserving edit (recolour) needs a LOW denoise — at
        # replacement strength the repaint regenerates the object instead of
        # restyling it (a recoloured car came back as a different car, D22).
        # Only the universal latent-mask template exposes the dial.
        if denoise is not None and "denoise" in slots:
            params["denoise"] = round(max(0.2, min(0.95, denoise)), 2)
        if variant == "hires":
            params.update(self._crop_params(image, mask))
        graph = build_workflow(template, params)
        out, prompt_id = self.client.run_graph(graph)
        return EditResult(image=out, adapter=self.name, is_mock=False,
                          meta={"template": f"{template['template']}_v{template['version']}",
                                "variant": variant,
                                "checkpoint": checkpoint,
                                "prompt_id": prompt_id})
