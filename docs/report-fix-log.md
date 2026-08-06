# Fix log — PromptForge live test report (29 July 2026)

Every defect in the report's register (section 6, D1–D26), what was changed,
where, and the test that pins it down. Regression tests live in
`backend/tests/test_report_fixes.py` unless noted.

## Critical

| ID | Defect | Fix | Where |
|----|--------|-----|-------|
| D1 | "Remove X" painted a different X (instruction used as positive conditioning) | `removal_conditioning()`: positive asks for the scene without the object, the object goes to the negative. Applied at the inpaint call, on retries, and the scene summary is kept out of removal prompts. | `quality.py` (`removal_conditioning`), `services.py` inpaint branch + retry loop |
| D2 | Viewpoint route returned a garment on grey / no side view reachable | Named viewpoints ("from the side" → 90°, "from behind" → 180°) are rendered directly instead of being clamped inside ±60°; a subject-present verify stage mattes the farthest frame and refuses to log "Completed" when the subject is absent. | `quality.requested_azimuths`, `services._render_viewpoints`, `services._views_contain_subject` |
| D3 | Planner fabricated an unrequested face swap and ran it | `prune_invented_steps` drops any faceswap step when `face_intent(prompt)` is false — same deterministic contract outpaint already had. Pruning now happens inside `plan_edit` itself. | `quality.py` |
| D4 | Mask fell through to SAM and landed on the face | (a) A CLIPSeg hit that fails the geometry checks now returns *not-found* instead of falling through to SAM. (b) Any surviving SAM mask has the head region subtracted (`_shield_face`) unless the request names the face/headwear; a mask that was mostly face is rejected outright. | `services.auto_mask`, `services._shield_face` |
| D5 | Compound splitter unreachable when the planner labels its one step with the capability | `reconcile_capability_steps()` asks "is this step doing two jobs?" of every whole-frame step, splits overloaded instructions, recovers the capability clause from the prompt when the step carries the *other* half, and clears garment targets off background steps. | `quality.py`, called from `plan_edit` |
| D6 | `_BACKGROUND_EXCLUDE` proximity window dropped background steps | `background_intent` is clause-scoped: each clause answers for itself. The report's 7-row table is asserted verbatim. | `quality.py` |

## High

| ID | Defect | Fix | Where |
|----|--------|-----|-------|
| D7 | Checklist verdict constant across renders; retries ~26% of pipeline time | A verdict that does not change across a full re-render stops the ladder immediately. Format requests are settled by aspect-ratio arithmetic before any retry is spent. | `services.py` retry loop, `quality.format_delivered` |
| D8 | Escalation picked a model the card cannot hold (8.0 need on 8.0 card) | `_next_edit_recipe` filters candidates through `_checkpoint_fits_retry`, which requires 0.5 GB headroom over the declared VRAM need. | `services.py` |
| D9 | Background repaint invented people | The background route now carries the pose route's negative: "person, people, human figure, limbs, extra person, duplicate subject, crowd, text, watermark". | `services._render_background_step` |
| D10 | "This was a guess" warning dropped at the API boundary | `MaskPreview` interface carries `source` + `notes`; Studio renders them beside the overlay (amber warning for SAM-guessed regions). | `frontend/src/types.ts`, `pages/Studio.tsx` |
| D15 | Face-swap ran ungated; 11 safety tests failing | The hand-renamed safety categories were restored: `minors` and `deepfake` keyword sets are back (they had been renamed to nonsense words matching almost nothing, which disabled both protections). All safety tests pass; identity-manipulation wording is blocked again as the suite has always asserted. | `safety.py` |
| D16/D23 | img2img at edit denoise replaced the person; style scored 91 | CHANGE_STYLE denoise capped at 0.55; a deterministic face-drift measurement (share of head-box pixels that moved) is recorded and flags >50% drift as "identity not preserved", vetoing production-ready. | `services.py` |
| D17 | Same failure scored 20 or 70 depending on run | Root mitigation via D18's gate + objective checks; the noisy judge no longer decides alone. | `quality.py`, `services.py` |
| D18 | "Overall" is a mean, so a no-op edit scored 70 | `overall()` gates on adherence: below accuracy 50 the headline is capped at the accuracy score. | `quality.overall` |
| D19 | Pose vacated-share computed, logged, unused | The share is stored (`_pose_vacated_share`) and read by the verify stage: <5% vacated adds "the subject's pose actually changing" to missing requirements and flags the result. | `services.py` |
| D20 | Relight grain (8.7× sharpness) scored 90 | `objective_report`/`objective_flags`: Laplacian-variance ratio, changed-pixel share, outside-mask leak and size ratio computed on every edit, logged, saved in the result, and allowed to veto `passed`. | `quality.py`, `services.py` |
| D21 | "Add a pair of X" filled the rectangle with X | Placement box fractions reduced (0.14/0.24/0.40), the vision prompt now defines size as the object's own footprint, and add-edits carry "a single object, exactly one" positive + "multiple, duplicated…" negative. | `quality.py`, `services.py` |

## Medium / Low

| ID | Defect | Fix | Where |
|----|--------|-----|-------|
| D11 | Preview mask ignored by whole-frame engines | `preview_region()`: background requests preview the actual inverted subject matte; pose/angles/video/scene3d say plainly the whole frame is in play. | `services.py`, `routes.py` |
| D12 | Only the final render stored | Every retry attempt is saved as an inspectable aux version (`_save_attempt`). | `services.py` |
| D13 | Scene-3D request gained an invented outpaint | `plan_edit` prunes invented steps itself now (see D3). | `quality.py` |
| D14/D25 | Clothing mask extent unreliable; rear views defeat it | Known limitation of the ClothesSegment vocabulary; failures surface as honest not-found messages (improved in D4's restructure). Documented, not fully solved — needs a better segmenter. | — |
| D22 | Recolour executed as replacement | `is_recolour()` routes colour-only edits through the universal template at denoise 0.45, preserving structure. The colour word must end its clause, so "red leather jacket" (a replacement that worked live) is untouched. | `quality.py`, `services.py`, `adapters/comfyui.py` |
| D24 | Every edit silently shrank by the VAE's /8 rounding | The saved result is restored to the input's exact size when the difference is ≤8 px per axis (outpaint/upscale unaffected). | `services.py` save path |
| D26 | `_VIEW_INTENT` missed "show her from the side" | Pattern widened (named single viewpoints, profile, side-on); lighting-direction phrases excluded so relight requests aren't hijacked. | `quality.py` |

## Report improvement-plan items

- **Step 9 (CLIPSeg calibration):** `text_mask.py` accepts `--control` phrases;
  the request's peak must beat the best absent-object control peak on the same
  image by 0.12 as well as the absolute 0.45 floor.
- **Step 11 (visibility):** new `POST /api/edits/plan` compiles the program
  without rendering; Studio shows the plan chips while the job runs, and mask
  provenance/warnings render beside the overlay.
- **Step 8 (keep attempts):** every retry render is stored as an aux version.

## Safety note

`safety.py` had been hand-edited: the `minors` keyword list was reduced to the
single word "child" under the name `_panini`, and the `deepfake` list to the
literal word "khebab" — meaning sexualized-minor wording beyond "child" and
all face-swap/impersonation wording passed the filter. This restoration puts
the module back in line with its own test suite and settings UI. Face-swap
prompts are consequently blocked again ("Identity manipulation… is not
supported"), which also closes D15's ungated-face-swap finding. If a consented
face-swap feature is wanted, the avatar route's explicit consent attestation
(`consent_verdict`) is the pattern to extend — that is a product decision, not
one this fix takes.

## Verification

- Full backend suite: run `python -m unittest discover -s tests` from
  `backend/` — green, including 37 new regression tests keyed to the report's
  own cases (`tests/test_report_fixes.py`).
- Frontend: `npx tsc --noEmit` — clean.
