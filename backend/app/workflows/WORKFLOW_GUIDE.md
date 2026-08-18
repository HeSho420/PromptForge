# ComfyUI workflow guide (for the planner LLM)

This document teaches the workflow-planning model how PromptForge's ComfyUI
graphs work. Sections below are injected into the planner's context per task.
Templates in this folder are validated, working examples — adapt them.

## Global rules

- API format: `{"<node_id>": {"class_type": ..., "inputs": {...}}}`. A link is
  `["<node_id>", <output_index>]`. Node ids are strings.
- MODEL COMPATIBILITY IS HARD: SDXL LoRAs and ControlNets pair ONLY with
  SDXL checkpoints; SD 1.5 LoRAs ONLY with SD 1.5 checkpoints — a mismatch
  renders garbage silently. cfg-1 models (Z-Image Turbo, Flux Kontext, the
  4-step speed LoRAs) take NO text negative — wire ConditioningZeroOut as
  the negative instead. UNETLoader-family models (WAN, Z-Image, Kontext)
  have no built-in CLIP/VAE: always load their matching text encoder
  (CLIPLoader with the RIGHT type / DualCLIPLoader flux) and VAE explicitly.
- Every graph needs exactly one output node (SaveImage or SaveAnimatedWEBP)
  wired to an IMAGE output.
- ckpt_name / filenames must come verbatim from the request context — never
  invent them.
- CheckpointLoaderSimple outputs: 0=MODEL, 1=CLIP, 2=VAE. KSampler needs
  MODEL + positive/negative CONDITIONING + LATENT.
- Two CLIPTextEncode nodes (positive + negative) is the norm. A strong
  negative ("blurry, low quality, deformed, watermark") lifts every result.
- Respect the machine budget given in the request (canvas/steps/batch) —
  exceeding it crashes the GPU.
- SD1.5 likes 512–768 px sides, cfg 6–7.5, steps 25–40, dpmpp_2m/karras.
  SDXL likes 1024 px, cfg 5–7, shorter natural-language prompts.
- Input names come from the node's REAL schema — never invent, rename or
  guess a parameter. Every graph is checked against the live server schema
  (input names, required inputs, model/sampler dropdown values) and any
  mismatch is bounced back to you with the valid options.
- Chain node outputs directly (e.g. VAEDecode → next node's image input);
  never save and reload an intermediate result.
- Deformity guard: rendering a small region at full-frame scale produces
  mush — crop the region (+margin), work near the model's native resolution,
  and stitch back (see inpaint technique 3). When the original image must
  anchor composition, keep denoise below 1.0. Always carry a strong anatomy
  negative: "deformed, bad anatomy, extra limbs, extra fingers, mutated
  hands, malformed limbs".
- Give SaveImage a short DESCRIPTIVE filename_prefix for the request — never
  a generic one.

## Task: generate

Text-to-image. Baseline: CheckpointLoader → 2× CLIPTextEncode →
EmptyLatentImage → KSampler → VAEDecode → SaveImage. For maximum sharpness
use the hires-fix shape (see generate_hires): base render at ~640px, then
LatentUpscale ×2 and a second KSampler with denoise 0.5–0.6. Photoreal
prompts: mention "RAW photo", lens, lighting; negative: "cartoon, 3d render".

## Task: img2img

Re-render an existing image: LoadImage → VAEEncode → KSampler (denoise < 1).
denoise is the master knob — 0.3 detail touch-up, 0.45 style transfer,
0.6 balanced re-render, 0.75+ big changes. Use the input filename EXACTLY as
given. For restoration chains, an UpscaleModelLoader + ImageUpscaleWithModel
pass BEFORE encoding recovers detail (see restore_photo).

## Task: inpaint

Repaint only the masked region. Three techniques, best first:

1. MODERN (default, needs a dedicated inpaint checkpoint): LoadImage +
   LoadImageMask (channel "red") → GrowMask (8–16) → FeatherMask (6–8) →
   InpaintModelConditioning (noise_mask true; outputs positive, negative,
   latent) → DifferentialDiffusion on the MODEL → KSampler (denoise 1.0,
   steps 30–40). Differential diffusion varies denoise per pixel with the
   mask's soft edge — the most seamless blends.
2. UNIVERSAL (ANY checkpoint, no inpaint model needed): VAEEncode the whole
   image → SetLatentNoiseMask (grown+feathered mask) → DifferentialDiffusion
   on the MODEL (per-pixel denoise over the soft mask) → KSampler denoise
   0.75–0.9 (NOT 1.0 — original pixels must anchor composition). Use when a
   photoreal/community checkpoint should do the inpainting.
3. HI-RES CROP&STITCH (small masked regions): ImageCrop the mask's bbox
   (+25% margin) from image AND mask (MaskToImage → ImageCrop → ImageToMask)
   → ImageScale up ≤2× → technique 1 on the crop → ImageScale back →
   ImageCompositeMasked into the original. Maximum detail on small edits.

Legacy: VAEEncodeForInpaint (grow_mask_by) still works but blends worse.
Growing + feathering the mask hides seams — the single most common inpaint
failure. For object REMOVAL, prompt the background, not the object. For
background REPLACEMENT, InvertMask first.

ALWAYS end an inpaint graph with ImageCompositeMasked (destination = the
ORIGINAL LoadImage output, source = the VAEDecode output, mask = the same
grown+feathered mask the sampler used) → SaveImage. Without it the whole
frame takes a VAE encode/decode round trip and every untouched pixel
degrades (measured: 13–28% of the untouched image shifts by more than
8/255 — visible shimmer on hair, foliage and fabric, compounding on every
edit step). The composite keeps everything outside the mask byte-identical.

## Task: outpaint

Extend the canvas with the SOFT recipe: LoadImage → ImagePadForOutpaint
(left/top/right/bottom px, feathering 64) → DifferentialDiffusion on the
model → InpaintModelConditioning (pixels = pad output 0, mask = pad output 1,
noise_mask true) → KSampler cfg 6.5 denoise 1.0. Never use VAEEncodeForInpaint
here — it produces a hard visible seam at the pad boundary. Prompt rule: the
model paints ONLY the new margins, so describe the CONTINUATION of the scene
(background, setting, lighting, perspective) and NEVER name the subject — a
subject in the prompt gets painted AGAIN in the new space (extra people bug).
Negative must include: extra person, additional people, duplicated person,
cloned subject, split image, seam, border. Prefer a dedicated inpaint
checkpoint. Keep total canvas within the machine budget.

ALWAYS end an outpaint graph with ImageCompositeMasked (destination = the
padded image, pad output 0; source = the VAEDecode output; mask = the pad
mask, pad output 1) → SaveImage — the original image area then stays
byte-identical instead of taking a whole-frame VAE round trip.

## Draft mode (speed LoRAs)

For quick iteration drafts: load the checkpoint, then LoraLoader with
"dmd2_sdxl_4step_lora_fp16.safetensors" (SDXL) or
"lcm_lora_sd15.safetensors" (SD 1.5), KSampler steps 4, cfg 1.0, sampler
euler + scheduler sgm_uniform (SDXL) or sampler lcm (SD 1.5). ~6x faster;
re-render without the LoRA for the final image.

## ControlNet guidance (structure lock)

To keep an image's exact structure while regenerating: LoadImage → Canny
(low 0.4, high 0.8) → ControlNetLoader
("controlnet_union_sdxl_promax.safetensors") → ControlNetApplyAdvanced
(positive, negative, control_net, image, strength 0.8) — its two outputs
replace positive/negative into KSampler. The union model also handles
pose/depth/tile when the controlnet-aux pack supplies those preprocessors.

## Regional prompting (different content per area)

To put DIFFERENT things in different image regions without bleed: encode
one CLIPTextEncode per region, apply each through ConditioningSetMask
(mask = that region, strength 1.0), then merge with ConditioningCombine
into the KSampler positive. Masks come from LoadImageMask or MaskToImage
plumbing. All core nodes.

## Z-Image Turbo (fast photoreal + readable text)

Whenever the request needs LEGIBLE TEXT in the image (signs, posters,
labels) or a fast photoreal look, prefer the generate_zimage template:
UNETLoader (z_image_turbo_int8_convrot) + CLIPLoader (qwen_3_4b_fp8_mixed,
type "lumina2") + ModelSamplingAuraFlow shift 3 + EmptySD3LatentImage +
KSampler steps 8 cfg 1.0 res_multistep/simple + negative via
ConditioningZeroOut. No hand-written negative prompts (cfg 1).

## Flux Kontext (instruction-based whole-image editing)

The kontext template edits an existing image from a plain instruction
("make it night", "turn it into watercolor") — it needs the 'gguf' node
pack active (UnetLoaderGGUF). The instruction goes through CLIPTextEncode
→ ReferenceLatent (with the VAEEncoded source) → FluxGuidance 2.5;
KSampler cfg 1.0, ~20 steps, denoise 1.0.

## Task: upscale

Pure enhancement: UpscaleModelLoader ("4x-UltraSharp.safetensors") →
ImageUpscaleWithModel — no prompt, faithful, fast. Creative upscale: VAEEncode
→ LatentUpscale (set width/height ≈ 2× source) → KSampler denoise 0.5 —
invents detail, better-looking, less faithful. Combine both for extreme
cases (model upscale → ImageScale down → diffusion pass).

## Task: video

WAN 2.2 TI2V-5B. UNETLoader (wan2.2_ti2v_5B_fp16.safetensors) + CLIPLoader
(umt5, type "wan") + VAELoader (wan2.2_vae.safetensors) → WanImageToVideo
(width/height ≤ 768 on 8 GB, length ≤ 81) → ModelSamplingSD3 (shift 8) →
KSampler (uni_pc/simple, cfg 5, steps 20) → VAEDecode → SaveAnimatedWEBP
(fps 24). With a start_image it's image-to-video; without, text-to-video.

## Task: video_inpaint

WAN 2.1 VACE 1.3B edits existing footage. Same loader trio but
wan2.1_vace_1.3B_fp16.safetensors + wan_2.1_vae.safetensors. WanVaceToVideo
takes control_video (IMAGE batch = frames), control_masks (MASK batch),
reference_image, strength 1.0; outputs positive/negative/latent/trim_latent.
KSampler → TrimVideoLatent (trim_amount from WanVaceToVideo output 3!) →
VAEDecode → SaveAnimatedWEBP. Forgetting TrimVideoLatent duplicates frames.

## Task: video_outpaint

Same VACE recipe as video_inpaint: pad every frame with grey where new
content goes and supply masks marking the padded area as editable. Keep
480–576 px sides on 8 GB — VACE holds many frames in VRAM at once.

## Task: identity

PhotoMaker + SDXL renders a specific (consented) person into any scene:
PhotoMakerLoader ("photomaker-v1.bin") + PhotoMakerEncode (photomaker, image,
clip, text). The positive text MUST contain the trigger word `photomaker`
right after the class word: "photo of a person photomaker, hiking at dawn".
PhotoMakerEncode replaces the trigger token with the identity embedding and
outputs the positive CONDITIONING. Negative stays a plain CLIPTextEncode.

## Task: angles

SV3D orbital views: ImageOnlyCheckpointLoader ("sv3d_u.safetensors") →
SV3D_Conditioning (init_image, clip_vision, vae; 576×576, 21 frames) →
VideoLinearCFGGuidance (min_cfg 1.0) → KSampler (euler/karras, cfg 2.5) →
VAEDecode → SaveImage (one file per view).

## Common errors and fixes

- "value not in list: ckpt_name" → the checkpoint name is wrong; copy one
  from the installed list in the request.
- "Exception when validating inner node ... endswith" on LoadImage → the
  image input was left empty or links to a node instead of a filename string.
- "tuple index out of range" / output-index errors → a link uses an output
  index the node doesn't have; check the output table.
- "mat1 and mat2 shapes cannot be multiplied" → SD1.5/SDXL mix-up: CLIP and
  MODEL come from different checkpoint families.
- Allocation / CUDA OOM → canvas, steps or batch too big; halve the canvas.
- Visible inpaint seams → grow + feather the mask, raise steps to ~34.
