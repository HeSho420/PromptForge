# PromptForge Intelligence Layer — Architecture & Implementation Map

Goal: upload any image, type any instruction, the AI decides everything
(masks, models, workflows, nodes, samplers, ControlNets, settings). No manual
selection. Local-first: Ollama = director, ComfyUI = renderer, SAM =
segmentation, LLaVA = vision, Python = the spine. Architecture is preserved —
this is an intelligence upgrade, not a rewrite.

## Pipeline (unchanged shape, deeper brain)

```
USER → UI → Ollama Orchestrator → Image Understanding (Scene Graph)
     → Editing-Program Compiler (atomic operations) → Workflow Planner
     → ComfyUI → Rendering → AI Quality Inspector → Self-Correction Loop
```

## What already exists (do not rebuild)

| Spec capability | Where it lives today |
|---|---|
| Ollama as reasoning engine (plan/model/workflow/repair) | `core/quality.plan_edit`, `core/workflow_ai`, `core/model_scout` |
| Structured JSON plans | `plan_edit` → ordered steps |
| Dynamic ComfyUI workflow generation + repair loop | `workflow_ai.generate/repair` |
| Node-exists / input / model / live-schema validation | `workflow_ai.validate_generated` + `live_schema_errors` |
| VRAM / hardware clamps | `services._apply_hardware_limits`, `hardware.render_budget` |
| Model intelligence database (best-at, quality 1-10, researched online) | `core/model_intel.py` → `data/model_knowledge.json` |
| Model choice per prompt | `_choose_inpaint`, forge scout, knowledge injection |
| Intelligent masking (SAM for existing, placement for new) | `quality.classify_edit`, `propose_placement`, `sam` adapter |
| 6-category quality scorecard, target 95, ≤N retries, keep best | `quality.scorecard`, edit + forge retry loops |
| Self-improvement / repair memory | `core/experience.py` (repair_knowledge) |
| Animation engine (WAN 2.2 i2v) | `_handle_video` |
| Multi-step chained edits | `_handle_image_edit` chain driver |
| Prompt-accuracy weighted retries | edit loop + forge accuracy pass |
| Per-requirement adherence check (neutral probes, never shows the judge the request) | `quality.request_checklist` / `verify_adherence` |
| Escalation ladder: emphasize → different MODEL → different WORKFLOW, nothing tried twice | `quality.escalation_plan`, `services._pursue_request`, `_next_edit_recipe` |
| Relighting as a real capability (IC-Light) rather than an img2img repaint | `relight_v1.json`, `_relight_prompts`, `OPERATION_TASK["CHANGE_LIGHTING"]` |
| Viewpoint synthesis from the edit pipeline (SV3D orbit + contact sheet) | `_render_viewpoints`, `quality.view_intent` |

## The genuine gaps (this upgrade)

1. **Persistent Scene Graph** — today the image is understood shallowly (a
   one-sentence description + per-step SAM). The spec wants ONE rich analysis
   (objects with location/size/depth, lighting, perspective, identity) built
   up-front and REUSED by every step. → `core/scene_graph.py`.
2. **Atomic-operation compiler** — plans carry a coarse task
   (inpaint/img2img/outpaint/custom). The spec wants a typed operation
   vocabulary (ADD_OBJECT, REMOVE_OBJECT, REPLACE_OBJECT, CHANGE_ATTRIBUTE,
   CHANGE_STYLE, CHANGE_LIGHTING, CHANGE_TEXT, OUTPAINT, UPSCALE, RESTORE,
   ANIMATE, …) plus an explicit `target`. → `operation` + `target` fields on
   every step; deterministic operation→task mapping.
3. **Targeted masking from the scene graph** — segmentation should be told
   WHICH object (its known location) instead of re-deriving it; placement
   should use the scene graph's ground plane / perspective / lighting.
4. **Multi-pass finishing** — optional structural → realism → lighting →
   enhancement passes (staged).
5. **Animate-from-Studio** — the ANIMATE operation routes an edited still
   into the existing WAN i2v pipeline (staged).

## Implementation slices

- **Slice A (this change):** Scene Graph + operation/target planning +
  targeted masking. The keystone — everything downstream reads the graph.
- **Slice B:** multi-pass compositor (lighting/shadow match pass).
- **Slice C:** animate-from-Studio.

## Scene Graph contract (`core/scene_graph.py`)

`build(image, critic, segmentation=None) -> dict`:

```
{
  "scene": "<one-line description>",
  "setting": "<indoor|outdoor|studio|...>",
  "lighting": "<direction + quality, e.g. 'soft daylight from upper left'>",
  "perspective": "<eye-level | low | high | ...>",
  "palette": [ (r,g,b), ... ],          # dominant colors (deterministic)
  "has_person": bool,
  "objects": [
     {"name": "car", "location": "lower-right", "size": "large",
      "cell": <1-9 grid>}
  ]
}
```

Built once per asset, cached in `services`. Every render prompt carries a
compact summary; placement and targeted masking read `objects`, `lighting`,
`perspective`. Fully fail-safe — an unavailable vision model degrades to a
minimal graph (deterministic palette only), never blocks a render.

## Operation → task mapping (deterministic)

```
ADD_OBJECT / REPLACE_OBJECT / REMOVE_OBJECT / CHANGE_ATTRIBUTE /
CHANGE_TEXT / RESTORE           → inpaint  (regional)
CHANGE_STYLE / CHANGE_LIGHTING / CHANGE_CAMERA → img2img (whole image)
OUTPAINT                        → outpaint
UPSCALE                         → upscale
ANIMATE                         → video (WAN i2v)   [Slice C]
(anything the model can't express) → custom (LLM-designed graph)
```

`target` (the noun the operation acts on) drives masking: existing target →
targeted SAM using the scene-graph location; new target (ADD) → placement
mask using scene-graph perspective + lighting.
