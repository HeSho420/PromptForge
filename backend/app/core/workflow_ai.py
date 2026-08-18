"""LLM-driven ComfyUI workflow generation with a validate/repair loop.

The generator asks the LLM (local first, API fallback — see llm.py) to emit a
ComfyUI API-format graph for a user's prompt, then checks it *every step of
the way*:

  1. the task type must be in ALLOWED_TASKS (same gate as templates);
  2. the reply must parse as JSON;
  3. the graph must pass validate_workflow — the structural allowlist in
     adapters/comfyui.py (node types, link integrity) — plus generation
     checks (an output node, a bounded node count).

Any failure is fed back to the LLM verbatim for a bounded number of repair
attempts; runtime errors from ComfyUI can be fed back the same way via
`repair()`. The security property is structural, not trust-the-model: no
graph reaches ComfyUI without passing the same allowlist validation that
gates the hand-written templates, so neither a confused model nor a prompt-
injection attempt can smuggle in disallowed node types.

Every result carries provenance (source: local|api, exact model, attempts)
which the API surfaces — mirroring the adapters' is_mock honesty rule.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..adapters.comfyui import (
    ALLOWED_NODE_TYPES,
    ALLOWED_TASKS,
    WorkflowNotAllowedError,
    WorkflowValidationError,
    validate_workflow,
)
from .llm import LLMClient, LLMError

# Upper bound on generated graph size — a sanity brake, not a real workflow
# limit (the shipped templates are <10 nodes).
MAX_NODES = 64

# Output arity/type of every allowed node — used both to teach the LLM the
# valid output indices and to statically reject out-of-range links before a
# graph is ever sent to ComfyUI ("tuple index out of range" class errors).
NODE_OUTPUTS: dict[str, list[str]] = {
    "CheckpointLoaderSimple": ["MODEL", "CLIP", "VAE"],
    "CLIPTextEncode": ["CONDITIONING"],
    "KSampler": ["LATENT"],
    "EmptyLatentImage": ["LATENT"],
    "LatentUpscale": ["LATENT"],
    "VAEEncode": ["LATENT"],
    "VAEEncodeForInpaint": ["LATENT"],
    "VAEDecode": ["IMAGE"],
    "ImageScale": ["IMAGE"],
    "LoadImage": ["IMAGE", "MASK"],
    "LoadImageMask": ["MASK"],
    "SaveImage": [],
    # Video / multi-view nodes (used mainly by versioned templates).
    "UNETLoader": ["MODEL"],
    "VAELoader": ["VAE"],
    "CLIPLoader": ["CLIP"],
    "ModelSamplingSD3": ["MODEL"],
    "WanImageToVideo": ["CONDITIONING", "CONDITIONING", "LATENT"],
    "SaveAnimatedWEBP": [],
    "ImageOnlyCheckpointLoader": ["MODEL", "CLIP_VISION", "VAE"],
    "SV3D_Conditioning": ["CONDITIONING", "CONDITIONING", "LATENT"],
    "VideoLinearCFGGuidance": ["MODEL"],
    "PhotoMakerLoader": ["PHOTOMAKER"],
    "PhotoMakerEncode": ["CONDITIONING"],
    "ImagePadForOutpaint": ["IMAGE", "MASK"],
    "UpscaleModelLoader": ["UPSCALE_MODEL"],
    "ImageUpscaleWithModel": ["IMAGE"],
    "GrowMask": ["MASK"],
    "FeatherMask": ["MASK"],
    "InvertMask": ["MASK"],
    "ImageInvert": ["IMAGE"],
    "WanVaceToVideo": ["CONDITIONING", "CONDITIONING", "LATENT", "INT"],
    "TrimVideoLatent": ["LATENT"],
    "InpaintModelConditioning": ["CONDITIONING", "CONDITIONING", "LATENT"],
    "DifferentialDiffusion": ["MODEL"],
    "SetLatentNoiseMask": ["LATENT"],
    "ImageCrop": ["IMAGE"],
    "ImageCompositeMasked": ["IMAGE"],
    "MaskToImage": ["IMAGE"],
    "ImageToMask": ["MASK"],
    # Speed LoRAs, ControlNet, regional prompting, Z-Image and Flux flows.
    "LoraLoader": ["MODEL", "CLIP"],
    "LoraLoaderModelOnly": ["MODEL"],
    "ControlNetLoader": ["CONTROL_NET"],
    "ControlNetApplyAdvanced": ["CONDITIONING", "CONDITIONING"],
    "Canny": ["IMAGE"],
    "ConditioningSetMask": ["CONDITIONING"],
    "ConditioningCombine": ["CONDITIONING"],
    "ModelSamplingAuraFlow": ["MODEL"],
    "EmptySD3LatentImage": ["LATENT"],
    "ConditioningZeroOut": ["CONDITIONING"],
    "DualCLIPLoader": ["CLIP"],
    "ReferenceLatent": ["CONDITIONING"],
    "FluxGuidance": ["CONDITIONING"],
    "UnetLoaderGGUF": ["MODEL"],
    "LoadAndApplyICLightUnet": ["MODEL"],
    # positive, negative, empty_latent — the 3rd output IS the start latent.
    "ICLightConditioning": ["CONDITIONING", "CONDITIONING", "LATENT"],
    # image, mask, mask_image — output 1 is the matte you actually want.
    "BiRefNetRMBG": ["IMAGE", "MASK", "IMAGE"],
    # Face refinement (impact pack): output 0 is the finished image with
    # every detected face re-rendered and blended back.
    "FaceDetailer": ["IMAGE", "IMAGE", "IMAGE", "MASK", "DETAILER_PIPE",
                     "IMAGE"],
    "UltralyticsDetectorProvider": ["BBOX_DETECTOR", "SEGM_DETECTOR"],
    # Image → 3D mesh (Hunyuan3D v2, ComfyUI core). Signatures read off the
    # live /object_info, not guessed.
    "CLIPVisionEncode": ["CLIP_VISION_OUTPUT"],
    "Hunyuan3Dv2Conditioning": ["CONDITIONING", "CONDITIONING"],
    "Hunyuan3Dv2ConditioningMultiView": ["CONDITIONING", "CONDITIONING"],
    "EmptyLatentHunyuan3Dv2": ["LATENT"],
    "VAEDecodeHunyuan3D": ["VOXEL"],
    "VoxelToMesh": ["MESH"],
    "VoxelToMeshBasic": ["MESH"],
    "SaveGLB": [],
    "ImageBatch": ["IMAGE"],
    # Multi-region segmentation (rmbg pack): IMAGE, MASK, MASK_IMAGE.
    "ClothesSegment": ["IMAGE", "MASK", "IMAGE"],
    "BodySegment": ["IMAGE", "MASK", "IMAGE"],
    "Segment": ["IMAGE", "MASK", "IMAGE"],
    "SegmentV2": ["IMAGE", "MASK", "IMAGE"],
    "LoadMoGeModel": ["MOGE_MODEL"],
    "MoGeInference": ["MOGE_GEOMETRY"],
    "MoGePanoramaInference": ["MOGE_GEOMETRY"],
    "MoGePointMapToMesh": ["MESH"],
    "MoGeRender": ["IMAGE"],
    "FaceSegment": ["IMAGE", "MASK", "IMAGE"],
    "DWPreprocessor": ["IMAGE", "POSE_KEYPOINT"],
    "InstantIDModelLoader": ["INSTANTID"],
    "InstantIDFaceAnalysis": ["FACEANALYSIS"],
    # model, positive, negative — the sampler takes ALL THREE from here.
    "ApplyInstantID": ["MODEL", "CONDITIONING", "CONDITIONING"],
}

# What each allowed node DOES and when to reach for it — the generator model
# otherwise only sees names and has to guess semantics. Kept to one tight
# line per node (context budget); a sync test asserts full coverage of
# ALLOWED_NODE_TYPES.
NODE_GUIDE: dict[str, str] = {
    "CheckpointLoaderSimple": "loads an SD/SDXL checkpoint (model+clip+vae)",
    "CLIPTextEncode": "turns a prompt into conditioning (one per prompt)",
    "KSampler": "the diffusion sampler — where the image is actually made",
    "EmptyLatentImage": "blank canvas for SD/SDXL txt2img (width/height)",
    "EmptySD3LatentImage": "blank canvas for Z-Image/SD3-family models",
    "LatentUpscale": "resizes a latent (creative upscale before a 2nd pass)",
    "VAEEncode": "image -> latent (start of img2img)",
    "VAEEncodeForInpaint": "LEGACY inpaint encode — hard seams; avoid",
    "VAEDecode": "latent -> image (before SaveImage)",
    "ImageScale": "resizes an image in pixel space",
    "LoadImage": "loads the input image (use the given filename verbatim)",
    "LoadImageMask": "loads a mask image (white = editable region)",
    "SaveImage": "writes the result — every graph must end in one",
    "UNETLoader": "loads a bare diffusion model (WAN video, Z-Image)",
    "VAELoader": "loads a standalone VAE (pairs with UNETLoader flows)",
    "CLIPLoader": "loads a standalone text encoder; set the right type "
                  "(wan / lumina2 for Z-Image)",
    "DualCLIPLoader": "loads clip_l + t5 together, type=flux (Kontext/Flux)",
    "ModelSamplingSD3": "shift tuning for WAN video models",
    "ModelSamplingAuraFlow": "shift tuning for Z-Image (shift 3)",
    "WanImageToVideo": "conditions WAN on a start image -> video latent",
    "SaveAnimatedWEBP": "writes a video result (fps 24)",
    "ImageOnlyCheckpointLoader": "loads SV3D (image-conditioned model)",
    "SV3D_Conditioning": "camera-orbit conditioning for SV3D multi-view",
    "VideoLinearCFGGuidance": "cfg ramp that stabilizes SV3D/video sampling",
    "PhotoMakerLoader": "loads the PhotoMaker identity adapter",
    "PhotoMakerEncode": "binds reference face photos into conditioning",
    "ImagePadForOutpaint": "grows the canvas + feathered mask for outpaint",
    "UpscaleModelLoader": "loads a pixel upscale model (4x-UltraSharp)",
    "ImageUpscaleWithModel": "faithful AI upscale (no prompt needed)",
    "GrowMask": "expands a mask by N px (blend headroom)",
    "FeatherMask": "soft mask edges — smooth inpaint transitions",
    "InvertMask": "flips a mask (edit everything EXCEPT the selection)",
    "ImageInvert": "inverts image colors (rarely needed)",
    "WanVaceToVideo": "WAN VACE video editing (video inpaint/outpaint)",
    "TrimVideoLatent": "drops VACE setup frames from the video latent",
    "InpaintModelConditioning": "MODERN inpaint/outpaint conditioning — "
                                "always prefer over VAEEncodeForInpaint",
    "DifferentialDiffusion": "soft per-pixel denoise strength — pairs with "
                             "InpaintModelConditioning for seamless blends",
    "SetLatentNoiseMask": "universal inpaint: ANY checkpoint, denoise ~0.85",
    "ImageCrop": "crops a region (hires inpaint: crop->work->stitch)",
    "ImageCompositeMasked": "pastes a worked region back into the original",
    # Image -> 3D mesh (Hunyuan3D v2, all ComfyUI core).
    "CLIPVisionEncode": "encodes an image for models conditioned on vision, "
                        "not text (Hunyuan3D)",
    "Hunyuan3Dv2Conditioning": "image -> 3D conditioning from ONE view",
    "Hunyuan3Dv2ConditioningMultiView": "image -> 3D conditioning from four "
                                        "views (front/left/back/right)",
    "EmptyLatentHunyuan3Dv2": "the empty 3D latent Hunyuan3D samples into",
    "VAEDecodeHunyuan3D": "3D latent -> VOXEL volume",
    "VoxelToMesh": "VOXEL -> MESH (surface net; 'basic' is faster, rougher)",
    "VoxelToMeshBasic": "VOXEL -> MESH, marching-cubes style",
    "SaveGLB": "writes a MESH as a .glb file (the exportable 3D asset)",
    "ImageBatch": "stacks two images into one IMAGE batch",
    # Named-region segmentation (rmbg pack) - selects EVERY named part at
    # once, which a point-and-grow segmenter cannot do.
    "ClothesSegment": "mask named GARMENTS (Upper-clothes/Skirt/Dress/Pants/"
                      "shoes...); several parts at once",
    "BodySegment": "mask named BODY parts (arms, legs, torso-skin, face...)",
    "Segment": "mask everything matching a text prompt (GroundingDINO+SAM); "
               "separate phrases with ' . ' to find several things",
    "SegmentV2": "as Segment, newer detector",
    "LoadMoGeModel": "loads MoGe (photo -> metric 3D geometry)",
    "MoGeInference": "photo -> metric point map + predicted camera FOV",
    "MoGePanoramaInference": "as MoGeInference, for a 360 equirect panorama",
    "MoGePointMapToMesh": "point map -> MESH with UVs and the photo as its "
                          "texture; discontinuity_threshold cuts the triangles "
                          "that would smear across a depth edge",
    "MoGeRender": "renders a MoGe point map from a new camera",
    "MaskToImage": "mask -> image (visualize or route into image inputs)",
    "ImageToMask": "image channel -> mask",
    "LoraLoader": "applies a LoRA to model+clip (speed or style LoRAs); "
                  "SDXL LoRAs only on SDXL checkpoints, SD15 on SD15",
    "LoraLoaderModelOnly": "applies a model-only LoRA (video speed LoRAs)",
    "ControlNetLoader": "loads a ControlNet (union model covers 12 types)",
    "ControlNetApplyAdvanced": "injects a control image (canny/pose/depth) "
                               "into conditioning; its 2 outputs REPLACE "
                               "positive/negative",
    "Canny": "edge map from an image — the core-node ControlNet input",
    "ConditioningSetMask": "limits a prompt to a masked region (regional "
                           "prompting)",
    "ConditioningCombine": "merges regional conditionings into one",
    "ConditioningZeroOut": "empty negative for cfg-1 models (Z-Image, "
                           "Kontext, speed LoRAs) — no text negative",
    "ReferenceLatent": "gives Kontext the source image to edit",
    "FluxGuidance": "Flux/Kontext guidance strength (~2.5)",
    "UnetLoaderGGUF": "loads a quantized GGUF model (Kontext on 8 GB); "
                      "needs the gguf node pack",
    "LoadAndApplyICLightUnet": "RELIGHTING: patches a 4-channel SD1.5 base "
                               "with the IC-Light unet — SD1.5 *inpainting* "
                               "checkpoints (9-channel) crash it; needs the "
                               "ic-light node pack",
    "InstantIDModelLoader": "IDENTITY: loads the InstantID face adapter; "
                            "needs the instantid node pack",
    "InstantIDFaceAnalysis": "IDENTITY: the face detector InstantID reads a "
                             "reference photo with (antelopev2)",
    "ApplyInstantID": "IDENTITY: carries a face from one reference photo "
                      "into an SDXL render. Returns model, positive, "
                      "negative — the sampler must take ALL THREE from here. "
                      "SDXL only, and heavy (~11 GB resident)",
    "FaceSegment": "FACE region matte (output 1 is the mask). Per-feature "
                   "booleans — select skin/eyes/nose/mouth for a face, and "
                   "leave hair and ears OFF unless a whole head is wanted; "
                   "needs the rmbg pack",
    "BiRefNetRMBG": "COMPOSITING: cuts a whole SUBJECT out of a photo "
                    "(output 1 is the matte). Use this, never SAM, when the "
                    "goal is the whole person or object; needs the rmbg pack",
    "FaceDetailer": "FACES: detects every face and re-renders each at guide "
                    "resolution with low denoise (0.45), blending it back — "
                    "the mushy-face fix for full-body shots. Output 0 is the "
                    "finished image. Wire bbox_detector from "
                    "UltralyticsDetectorProvider; needs the impact pack",
    "UltralyticsDetectorProvider": "FACES: loads a YOLO detector for "
                                   "FaceDetailer — use model_name "
                                   "'bbox/face_yolov8m.pt'; needs the "
                                   "impact pack",
    "DWPreprocessor": "MOTION: turns video frames into OpenPose skeletons to "
                      "drive a render. Set bbox_detector to the "
                      ".torchscript.pt file — the .onnx default has no CUDA "
                      "here and is ~20x slower; needs the controlnet-aux pack",
    "ICLightConditioning": "RELIGHTING: takes the photo's latent as "
                           "'foreground' and emits positive, negative and a "
                           "ZERO start latent (use output 2 as latent_image, "
                           "denoise 0.9 — lower collapses to grey); the "
                           "prompt must describe LIGHT, not the subject",
}

SYSTEM_PROMPT = f"""You generate ComfyUI workflows in API format: a JSON \
object whose top-level keys are node ids ("1", "2", ...) and whose values are \
nodes of the form {{"class_type": ..., "inputs": {{...}}}}. A link is a \
2-item array [node_id, output_index].

Example of the EXACT expected reply shape (txt2img):
{{"1": {{"class_type": "CheckpointLoaderSimple", "inputs": {{"ckpt_name": "model.safetensors"}}}},
 "2": {{"class_type": "CLIPTextEncode", "inputs": {{"text": "a cat", "clip": ["1", 1]}}}},
 "3": {{"class_type": "CLIPTextEncode",
        "inputs": {{"text": "blurry, low quality", "clip": ["1", 1]}}}},
 "4": {{"class_type": "EmptyLatentImage",
        "inputs": {{"width": 768, "height": 512, "batch_size": 1}}}},
 "5": {{"class_type": "KSampler",
        "inputs": {{"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                    "latent_image": ["4", 0], "seed": 42, "steps": 22, "cfg": 7.0,
                    "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}}}},
 "6": {{"class_type": "VAEDecode", "inputs": {{"samples": ["5", 0], "vae": ["1", 2]}}}},
 "7": {{"class_type": "SaveImage", "inputs": {{"images": ["6", 0], "filename_prefix": "out"}}}}}}

Node catalog — what each node is FOR, and its output indices (a link \
[node_id, i] must use a valid index i):
{chr(10).join(
    f"- {t}: {NODE_GUIDE.get(t, '?')} | outputs: "
    + (", ".join(f"{i}={o}" for i, o in enumerate(outs)) or "none")
    for t, outs in sorted(NODE_OUTPUTS.items()))}

Hard rules:
- Reply with ONLY the JSON graph object, exactly like the example — no prose,
  no markdown fences, and NO wrapper keys such as "nodes", "workflow", "links".
- Use ONLY these node types: {", ".join(sorted(ALLOWED_NODE_TYPES))}
- The graph MUST contain a SaveImage node; its "images" input must link to a
  node whose output is IMAGE (e.g. VAEDecode output 0).
- Checkpoint names, image filenames and mask filenames given in the request
  must be used verbatim.
- Use EXACT input names from each node's schema — never invent, rename or
  guess parameters. Every model/sampler/scheduler value must be one this
  machine actually has (they are validated against the live server and any
  mismatch is rejected back to you).
- Keep the graph minimal: no nodes that do not contribute to the output.
"""


def _combo_options(spec: Any) -> list | None:
    """COMBO (dropdown) options from an /object_info input spec. Handles both
    API shapes: legacy `[[opt1, ...], {...}]` and wrapped
    `["COMBO", {"options": [...]}]`."""
    if not isinstance(spec, list | tuple) or not spec:
        return None
    head = spec[0]
    if head == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1].get("options", []) or None
    if isinstance(head, list):
        return head or None
    return None


def live_schema_errors(graph: dict[str, Any], object_info: dict[str, Any],
                       max_errors: int = 6) -> list[str]:
    """Check a graph against the LIVE ComfyUI schema (/object_info): node
    types this server doesn't have, invented/misspelled input names, missing
    required inputs, and literal COMBO values that are not among this
    machine's actual options (wrong model filename, sampler typo, ...).
    Returns precise messages for the repair loop instead of raising — a graph
    that passes this never reaches ComfyUI just to bounce off node
    validation."""
    errors: list[str] = []
    for node_id, node in graph.items():
        ct = node.get("class_type", "")
        nd = object_info.get(ct)
        if not isinstance(nd, dict):
            errors.append(f"Node {node_id}: '{ct}' is not installed in this "
                          "ComfyUI server.")
            continue
        req = (nd.get("input") or {}).get("required") or {}
        opt = (nd.get("input") or {}).get("optional") or {}
        valid = set(req) | set(opt)
        inputs = node.get("inputs", {})
        for key in inputs:
            if valid and key not in valid:
                errors.append(
                    f"Node {node_id} ({ct}): unknown input '{key}' — valid "
                    f"inputs are: {', '.join(sorted(valid))}.")
        for key in req:
            if key not in inputs:
                errors.append(
                    f"Node {node_id} ({ct}): required input '{key}' is "
                    "missing.")
        for key, value in inputs.items():
            if isinstance(value, list):
                continue  # a link — structure is validated elsewhere
            # LoadImage/LoadImageMask "image" is a dropdown of files in
            # ComfyUI's input dir. A file we just uploaded may not be in the
            # (cached, ≤5-min-old) /object_info options yet — validating it
            # here would force a pointless repair round. Skip it; the actual
            # load fails loudly at run time if the file is truly missing.
            if ct in ("LoadImage", "LoadImageMask") and key == "image":
                continue
            options = _combo_options(req.get(key) or opt.get(key))
            if options and value not in options:
                sample = ", ".join(str(o) for o in options[:8])
                more = ", ..." if len(options) > 8 else ""
                errors.append(
                    f"Node {node_id} ({ct}): '{key}' = {value!r} is not "
                    f"available on this machine. Options: {sample}{more}.")
        if len(errors) >= max_errors:
            break
    return errors[:max_errors]


class WorkflowGenerationError(RuntimeError):
    """The LLM could not produce a valid workflow within the attempt budget."""


@dataclass
class GeneratedWorkflow:
    graph: dict[str, Any]
    task: str
    provenance: dict[str, Any] = field(default_factory=dict)


def validate_generated(graph: dict[str, Any]) -> None:
    """Structural allowlist validation + generation-specific checks."""
    validate_workflow(graph)  # node-type allowlist, link integrity
    if len(graph) > MAX_NODES:
        raise WorkflowValidationError(
            f"Graph has {len(graph)} nodes; the limit is {MAX_NODES}.")
    if not any(n.get("class_type") == "SaveImage" for n in graph.values()):
        raise WorkflowValidationError(
            "Graph has no SaveImage node — nothing would be saved.")
    # Reject malformed links and out-of-range output indices before ComfyUI
    # has to (in API format a list input is ALWAYS a [node_id, index] link).
    for node_id, node in graph.items():
        for key, value in node.get("inputs", {}).items():
            if isinstance(value, list) and not (
                    len(value) == 2 and isinstance(value[0], str)
                    and isinstance(value[1], int)):
                raise WorkflowValidationError(
                    f"Node {node_id} input '{key}' is a malformed link "
                    f"{value!r} — a link must be [\"node_id\", output_index].")
            if (isinstance(value, list) and len(value) == 2
                    and isinstance(value[0], str)):
                target = graph.get(value[0], {})
                outs = NODE_OUTPUTS.get(target.get("class_type", ""), None)
                if outs is not None and not (
                        isinstance(value[1], int) and 0 <= value[1] < len(outs)):
                    raise WorkflowValidationError(
                        f"Node {node_id} input '{key}' links to "
                        f"{target.get('class_type')} output {value[1]}, but "
                        f"that node's outputs are "
                        + (", ".join(f"{i}={o}" for i, o in enumerate(outs))
                           or "none") + ".")


def _parse_graph(text: str) -> dict[str, Any]:
    """Parse the LLM reply into a graph dict; tolerate markdown fences."""
    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise WorkflowValidationError(f"Reply is not valid JSON: {exc}") from exc
    # Tolerate common wrapper keys when the inner value is a node map.
    for wrapper in ("graph", "prompt", "workflow", "nodes"):
        if (isinstance(data, dict) and isinstance(data.get(wrapper), dict)
                and all(isinstance(v, dict) for v in data[wrapper].values())):
            data = data[wrapper]
            break
    if isinstance(data, dict) and isinstance(data.get("nodes"), list):
        raise WorkflowValidationError(
            'You used the ComfyUI UI-export format ("nodes" as a list). '
            "Reply in API format: a JSON object keyed by node id, exactly "
            "like the example.")
    if not isinstance(data, dict):
        raise WorkflowValidationError("Reply JSON is not a graph object.")
    return data


class WorkflowGenerator:
    def __init__(self, llm: LLMClient, max_attempts: int = 3,
                 log: Callable[[str], None] | None = None,
                 schema_provider: Callable[[], dict[str, Any] | None]
                 | None = None):
        self.llm = llm
        self.max_attempts = max_attempts
        self._log = log or (lambda msg: None)
        # Optional: returns the live /object_info dict (or None when ComfyUI
        # is unreachable). When present, every generated graph is checked
        # against the REAL schema before it may leave the loop.
        self.schema_provider = schema_provider

    # -- public -----------------------------------------------------------------
    def generate(self, task: str, request: str, context: str | None = None,
                 log: Callable[[str], None] | None = None) -> GeneratedWorkflow:
        """Generate a validated workflow for `request` (a task description).

        `context` carries live inventory ("Available checkpoints: ...") so the
        model plans against what this machine actually has installed. `log`
        (optional) receives per-call narration — pass a job-scoped logger
        instead of mutating self._log, so concurrent callers (job worker +
        HTTP preview) can never write into each other's logs.
        """
        if task not in ALLOWED_TASKS:
            raise WorkflowNotAllowedError(
                f"Task '{task}' is not an allowed workflow type.")
        first = (f"Task type: {task}\n"
                 + (f"{context}\n" if context else "")
                 + f"Request: {request}\n"
                 "Generate the workflow graph now.")
        return self._loop(task, first, log=log)

    def repair(self, task: str, graph: dict[str, Any], error: str,
               context: str | None = None,
               log: Callable[[str], None] | None = None) -> GeneratedWorkflow:
        """Fix a workflow that failed at runtime (e.g. a ComfyUI node error)."""
        if task not in ALLOWED_TASKS:
            raise WorkflowNotAllowedError(
                f"Task '{task}' is not an allowed workflow type.")
        first = (f"Task type: {task}\n"
                 + (f"{context}\n" if context else "")
                 + "This workflow failed when executed:\n"
                 f"{json.dumps(graph, indent=1)}\n"
                 f"Error: {error}\n"
                 "Reply with the corrected workflow graph.")
        return self._loop(task, first, log=log)

    # -- loop -------------------------------------------------------------------
    def _loop(self, task: str, first_message: str,
              log: Callable[[str], None] | None = None) -> GeneratedWorkflow:
        emit = log or self._log
        message = first_message
        last_error = "no attempts made"
        model_used = "?"
        for attempt in range(1, self.max_attempts + 1):
            try:
                reply = self.llm.complete(SYSTEM_PROMPT, message)
            except LLMError:
                raise  # unavailability/refusal is not repairable by retrying here
            model_used = f"{reply.source}:{reply.model}"
            try:
                graph = _parse_graph(reply.text)
                validate_generated(graph)
                if self.schema_provider is not None:
                    info = None
                    try:
                        info = self.schema_provider()
                    except Exception:  # noqa: BLE001 — advisory when down
                        pass
                    if info:
                        schema_errors = live_schema_errors(graph, info)
                        if schema_errors:
                            raise WorkflowValidationError(
                                "Live schema check failed:\n- "
                                + "\n- ".join(schema_errors))
            except WorkflowValidationError as exc:
                last_error = str(exc)
                emit(f"Workflow attempt {attempt}/{self.max_attempts} "
                     f"rejected: {last_error}")
                message = (f"Your previous reply was rejected: {last_error}\n"
                           f"Previous reply:\n{reply.text}\n"
                           "Reply with the corrected JSON graph only.")
                continue
            emit(f"Workflow validated on attempt {attempt} via {model_used}")
            return GeneratedWorkflow(
                graph=graph, task=task,
                provenance={"source": reply.source, "model": reply.model,
                            "attempts": attempt})
        raise WorkflowGenerationError(
            f"No valid workflow after {self.max_attempts} attempts "
            f"(last model: {model_used}). Last error: {last_error}")
