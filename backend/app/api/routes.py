"""HTTP layer (Flask). Thin by design: parse -> call Services -> serialize.

Errors follow one shape: {"error": {"code", "message"}} with a proper status.
User-facing messages stay actionable; stack traces stay in server logs.
"""
from __future__ import annotations

import logging
from pathlib import Path

from flask import Blueprint, Flask, jsonify, request, send_file

from ..adapters.base import BackendUnavailableError, BadMaskError, ModelMissingError
from ..adapters.comfyui import WorkflowNotAllowedError
from ..config import PROJECT_ROOT
from ..core import quality
from ..core.llm import LLMRefusedError, LLMUnavailableError
from ..core.model_search import ModelSearchError
from ..core.safety import SafetyRuleError, consent_verdict, model_source_blocked
from ..core.services import Services
from ..core.storage import UnsupportedFormatError
from ..core.workflow_ai import WorkflowGenerationError

log = logging.getLogger("promptforge.api")


def _error(status: int, code: str, message: str):
    return jsonify({"error": {"code": code, "message": message}}), status


def create_app(services: Services | None = None) -> Flask:
    services = services or Services()
    services.start()

    dist = PROJECT_ROOT / "frontend" / "dist"
    app = Flask(
        __name__,
        static_folder=str(dist) if dist.exists() else None,
        static_url_path="/",
    )
    app.config["MAX_CONTENT_LENGTH"] = services.settings.max_upload_mb * 1024 * 1024
    app.extensions["services"] = services

    api = Blueprint("api", __name__, url_prefix="/api")

    # -- health / settings ------------------------------------------------------
    @api.get("/health")
    def health():
        return jsonify({
            "status": "ok",
            "inpaint_adapter": services.inpainting.name,
            "inpaint_is_mock": services.inpainting.is_mock,
            "segmentation_adapter": services.segmentation.name,
            "segmentation_is_mock": services.segmentation.is_mock,
            "llm_local": f"{services.settings.llm_url} ({services.settings.llm_model})",
            "llm_api_fallback": services.settings.llm_api_model or None,
        })

    # -- assets -----------------------------------------------------------------
    @api.post("/assets")
    def upload_asset():
        file = request.files.get("file")
        if file is None or not file.filename:
            return _error(400, "no_file", "Attach a file to the 'file' field.")
        try:
            asset = services.store.save_upload(file.filename, file.read())
        except UnsupportedFormatError as exc:
            return _error(415, "unsupported_format", str(exc))
        return jsonify(asset.to_dict()), 201

    @api.get("/assets")
    def list_assets():
        return jsonify([a.to_dict() for a in services.store.list_assets()])

    @api.get("/assets/<asset_id>/file")
    def asset_file(asset_id: str):
        asset = services.store.get_asset(asset_id)
        if asset is None or not Path(asset.path).exists():
            return _error(404, "not_found", "Asset not found.")
        return send_file(asset.path)

    @api.get("/versions/<version_id>/file")
    def version_file(version_id: str):
        v = services.store.get_version(version_id)
        if v is None or not Path(v.path).exists():
            return _error(404, "not_found", "Version not found.")
        return send_file(v.path)

    @api.post("/versions/<version_id>/promote")
    def promote_version(version_id: str):
        """'Continue from this result': the version becomes the asset's
        working image, so the next edit/mask/video builds on it instead of
        the original upload. Reversible — the original is version-labelled
        'original' and can be promoted back."""
        asset = services.store.promote_version(version_id)
        if asset is None:
            return _error(404, "not_found",
                          "Nothing to continue from — unknown version, a mask "
                          "artifact, or its file is missing.")
        services.invalidate_asset_caches(asset.id)
        services.events.log("info", f"Continuing from result {version_id} — it "
                                    f"is now the working image of {asset.filename}")
        return jsonify(asset.to_dict())

    @api.get("/gallery")
    def gallery():
        return jsonify(services.store.gallery())

    # -- gallery management (soft delete + undo + disk cleanup) --------------------
    @api.delete("/assets/<asset_id>")
    def delete_asset(asset_id: str):
        hard = request.args.get("hard") in ("1", "true")
        if hard:
            if not services.store.purge_asset(asset_id):
                return _error(404, "not_found", "No trashed asset to purge.")
            return jsonify({"purged": asset_id})
        if not services.store.delete_asset(asset_id):
            return _error(404, "not_found", "Asset not found (or already deleted).")
        return jsonify({"deleted": asset_id})

    @api.post("/assets/<asset_id>/restore")
    def restore_asset(asset_id: str):
        if not services.store.restore_asset(asset_id):
            return _error(404, "not_found", "Nothing to restore.")
        return jsonify({"restored": asset_id})

    @api.delete("/gallery")
    def delete_gallery():
        deleted = [a.id for a in services.store.list_assets()
                   if services.store.delete_asset(a.id)]
        services.events.log("info", f"Gallery cleared ({len(deleted)} images "
                                    "moved to trash — undo available)")
        return jsonify({"deleted": deleted})

    # -- masks --------------------------------------------------------------------
    @api.post("/masks/preview")
    def mask_preview():
        body = request.get_json(silent=True) or {}
        asset_id, prompt = body.get("asset_id"), body.get("prompt", "")
        if not asset_id:
            return _error(400, "missing_field", "asset_id is required.")
        verdict = services.safety.check(prompt, editing=True)
        if not verdict.allowed:
            log.warning("Safety block (mask preview): category=%s", verdict.category)
            return _error(422, f"safety_{verdict.category}", verdict.reason or "Blocked.")
        try:
            image = services.open_asset_image(asset_id)
        except Exception as exc:
            return _error(404, "asset_error", str(exc))
        # Real backends can fail in ways the mock never does; map them to
        # actionable statuses instead of a bare 500.
        try:
            # The SAME chooser the render uses. This endpoint used to call the
            # raw segmenter, so the region you approved on screen was picked
            # by a weaker engine than the one that would actually run.
            # Whole-frame engines (background/pose/angles/video/scene3d)
            # never consume the painted region, so for those the preview now
            # shows what the render will ACTUALLY use instead (D11).
            choice = (services.preview_region(image, prompt)
                      or services.auto_mask(image, prompt))
            if not choice.ok:
                return _error(422, "mask_error", choice.reason.capitalize()
                              + ". Paint the region yourself if it is there.")
            mask = choice.mask
        except ModelMissingError as exc:
            return _error(409, "model_missing", str(exc))
        except BadMaskError as exc:
            return _error(422, "mask_error", str(exc))
        except BackendUnavailableError as exc:
            return _error(503, "backend_unavailable", str(exc))
        return jsonify({
            "mask_b64": services.encode_image_b64(mask),
            "adapter": services.segmentation.name,
            "is_mock": services.segmentation.is_mock,
            # How the region was chosen, so the UI can say when it was a guess
            # rather than an understanding of the words.
            "source": choice.source,
            "notes": choice.notes,
            "width": image.width, "height": image.height,
        })

    @api.post("/masks/point")
    def mask_point():
        """Click-anything: segment exactly what's under the given pixel."""
        body = request.get_json(silent=True) or {}
        asset_id = body.get("asset_id")
        x, y = body.get("x"), body.get("y")
        if not asset_id or x is None or y is None:
            return _error(400, "missing_field", "asset_id, x and y are required.")
        try:
            image = services.open_asset_image(asset_id)
        except Exception as exc:
            return _error(404, "asset_error", str(exc))
        point_fn = getattr(services.segmentation, "point_mask", None)
        if point_fn is None:
            return _error(409, "not_supported",
                          "Click-to-select needs the SAM backend "
                          "(PROMPTFORGE_SEGMENT_BACKEND=sam).")
        try:
            mask = point_fn(image, int(x), int(y))
        except ModelMissingError as exc:
            return _error(409, "model_missing", str(exc))
        except BackendUnavailableError as exc:
            return _error(503, "backend_unavailable", str(exc))
        return jsonify({
            "mask_b64": services.encode_image_b64(mask),
            "adapter": services.segmentation.name,
            "is_mock": services.segmentation.is_mock,
            # Same interface as /masks/preview: how the region was chosen.
            "source": "point",
            "notes": [],
            "width": image.width, "height": image.height,
        })

    # -- system (GPU) telemetry ----------------------------------------------------
    @api.get("/peers")
    def peers():
        """Other PromptForge machines discovered on the local network."""
        return jsonify({
            "share": services.settings.lan_share,
            "render": services.settings.lan_render,
            "port": getattr(services.peers, "http_port", None),
            "peers": [p.to_dict() for p in services.peers.peers_list()]})

    @api.get("/system")
    def system_stats():
        import subprocess
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,memory.total,name",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout.strip().splitlines()[0]
            util, used, total, name = [p.strip() for p in out.split(",", 3)]
            return jsonify({"gpu": {"name": name, "util_pct": int(float(util)),
                                    "vram_used_mb": int(float(used)),
                                    "vram_total_mb": int(float(total))}})
        except Exception:
            return jsonify({"gpu": None})

    # -- edits ----------------------------------------------------------------------
    @api.post("/edits/plan")
    def preview_edit_plan():
        """Compile the request into its step plan WITHOUT rendering.

        Every defect class in the July 2026 test report was invisible from
        the UI because the compiled program lived only in the job log
        (Step 11). This shows the program before any minutes are spent, so a
        dropped half of a compound request is a visible wrong plan, not a
        silent wrong picture."""
        body = request.get_json(silent=True) or {}
        prompt = (body.get("prompt") or "").strip()
        verdict = services.safety.check(prompt, editing=True)
        if not verdict.allowed:
            return _error(422, f"safety_{verdict.category}",
                          verdict.reason or "Blocked.")
        steps = quality.plan_edit(services.llm, prompt,
                                  has_mask=bool(body.get("has_mask")))
        if not steps:
            return jsonify({"steps": [], "planned": False})
        return jsonify({"planned": True, "steps": [
            {"step": i + 1, "task": s["task"],
             "operation": s.get("operation", ""),
             "target": s.get("target", ""),
             "instruction": s["instruction"]}
            for i, s in enumerate(steps)]})

    @api.post("/edits")
    def create_edit():
        body = request.get_json(silent=True) or {}
        asset_id, prompt = body.get("asset_id"), (body.get("prompt") or "").strip()
        if not asset_id:
            return _error(400, "missing_field", "asset_id is required.")
        verdict = services.safety.check(prompt, editing=True)
        if not verdict.allowed:
            log.warning("Safety block (edit): category=%s", verdict.category)
            return _error(422, f"safety_{verdict.category}", verdict.reason or "Blocked.")
        if services.store.get_asset(asset_id) is None:
            return _error(404, "not_found", "Asset not found.")
        payload = {"asset_id": asset_id, "prompt": prompt}
        if body.get("mask_b64"):
            payload["mask_b64"] = body["mask_b64"]
        # Extra photos to combine with this one. Validated HERE, so an
        # unknown id is a 404 the user sees immediately rather than a job
        # that fails minutes later.
        refs = body.get("reference_asset_ids") or []
        if not isinstance(refs, list):
            return _error(400, "bad_field",
                          "reference_asset_ids must be a list of asset ids.")
        refs = [str(r) for r in refs][:3]
        for ref in refs:
            if ref == asset_id:
                return _error(400, "bad_field",
                              "A reference image must be a different image "
                              "from the one being edited.")
            ref_asset = services.store.get_asset(ref)
            if ref_asset is None:
                return _error(404, "not_found",
                              f"Reference image {ref} was not found.")
            if ref_asset.kind != "image":
                return _error(400, "bad_field",
                              "Reference images must be images.")
        if refs:
            payload["reference_asset_ids"] = refs
        job = services.queue.enqueue("image_edit", payload)
        return jsonify(job.to_dict()), 202

    @api.post("/avatar")
    def create_avatar():
        body = request.get_json(silent=True) or {}
        asset_ids = body.get("asset_ids") or []
        if not isinstance(asset_ids, list) or not asset_ids:
            return _error(400, "missing_field", "asset_ids (list) is required.")
        consent = consent_verdict(bool(body.get("consent")))
        if not consent.allowed:
            return _error(422, "consent_required", consent.reason or "Blocked.")
        for aid in asset_ids:
            if services.store.get_asset(aid) is None:
                return _error(404, "not_found", f"Asset {aid} not found.")
        job = services.queue.enqueue("avatar", {
            "asset_ids": asset_ids, "consent": True,
            "name": (body.get("name") or "").strip() or None,
            # Both default ON — they are what makes the mesh look like the
            # person and like a whole person. Off is a deliberate choice: bare
            # geometry is easier to sculpt on, and a body the model invented
            # below the crop is a guess you may not want in your file.
            "texture": body.get("texture", True) is not False,
            "complete_body": body.get("complete_body", True) is not False})
        return jsonify(job.to_dict()), 202

    @api.get("/avatars")
    def list_avatars():
        return jsonify([a.to_dict() for a in services.store.list_avatars()])

    @api.delete("/avatars/<avatar_id>")
    def delete_avatar(avatar_id: str):
        """Delete an avatar profile. With ?frames=1 its synthetic orbit
        frames are moved to the trash too (undoable until purge); the
        original source photos always stay in the gallery."""
        profile = services.store.delete_avatar(avatar_id)
        if profile is None:
            return _error(404, "not_found", "Avatar not found.")
        frames_removed = 0
        if request.args.get("frames") in ("1", "true"):
            for fid in profile.frames:
                if services.store.delete_asset(fid):
                    frames_removed += 1
        services.events.log("info", f"Avatar '{profile.name}' deleted"
                            + (f" ({frames_removed} orbit frames trashed)"
                               if frames_removed else ""))
        return jsonify({"deleted": avatar_id, "frames_removed": frames_removed})

    @api.post("/avatars/<avatar_id>/render")
    def render_avatar(avatar_id: str):
        """Prompt-based identity render (image, optionally animated). The
        avatar carries the consent attestation from its creation; exposure
        and adult rules stay fully enforced for real-person identities."""
        body = request.get_json(silent=True) or {}
        prompt = (body.get("prompt") or "").strip()
        if not prompt:
            return _error(400, "missing_field", "prompt is required.")
        if services.store.get_avatar(avatar_id) is None:
            return _error(404, "not_found", "Avatar not found.")
        verdict = services.safety.check(prompt, editing=True)
        if not verdict.allowed:
            log.warning("Safety block (avatar render): category=%s",
                        verdict.category)
            return _error(422, f"safety_{verdict.category}",
                          verdict.reason or "Blocked.")
        payload: dict = {"avatar_id": avatar_id, "prompt": prompt,
                         "video": bool(body.get("video"))}
        if body.get("length") is not None:
            payload["length"] = body["length"]
        job = services.queue.enqueue("avatar_render", payload)
        return jsonify(job.to_dict()), 202

    # -- jobs -------------------------------------------------------------------------
    @api.post("/motion_transfer")
    def create_motion_transfer():
        """Make the person in a photo perform a driving video's motion."""
        body = request.get_json(silent=True) or {}
        ref = body.get("reference_asset_id")
        drive = body.get("driving_asset_id")
        if not ref or not drive:
            return _error(400, "missing_field",
                          "reference_asset_id (the person) and "
                          "driving_asset_id (the motion) are both required.")
        prompt = (body.get("prompt") or "").strip()
        verdict = services.safety.check(prompt, editing=True)
        if not verdict.allowed:
            log.warning("Safety block (motion): category=%s", verdict.category)
            return _error(422, f"safety_{verdict.category}",
                          verdict.reason or "Blocked.")
        ref_asset = services.store.get_asset(ref)
        drive_asset = services.store.get_asset(drive)
        if ref_asset is None or drive_asset is None:
            return _error(404, "not_found", "Asset not found.")
        if ref_asset.kind != "image":
            return _error(400, "bad_field",
                          "The reference must be a photo of the person.")
        payload: dict[str, object] = {"reference_asset_id": ref,
                                      "driving_asset_id": drive,
                                      "prompt": prompt}
        # Coerced and range-checked HERE: an unusable number should be a 400
        # now, not a crash in the worker after the models have loaded.
        for field, lo, hi in (("max_frames", 5, 600), ("seed", 0, 2**31 - 1)):
            if body.get(field) is not None:
                try:
                    payload[field] = max(lo, min(hi, int(body[field])))
                except (TypeError, ValueError):
                    return _error(400, "bad_field",
                                  f"{field} must be a whole number.")
        if body.get("strength") is not None:
            try:
                payload["strength"] = max(0.1, min(2.0, float(body["strength"])))
            except (TypeError, ValueError):
                return _error(400, "bad_field", "strength must be a number.")
        if body.get("preserve_background") is not None:
            payload["preserve_background"] = bool(body["preserve_background"])
        job = services.queue.enqueue("motion_transfer", payload)
        return jsonify(job.to_dict()), 202

    @api.get("/jobs")
    def list_jobs():
        return jsonify([j.to_dict() for j in services.queue.list()])

    @api.get("/jobs/<job_id>")
    def get_job(job_id: str):
        job = services.queue.get(job_id)
        if job is None:
            return _error(404, "not_found", "Job not found.")
        return jsonify(job.to_dict())

    @api.post("/jobs/<job_id>/cancel")
    def cancel_job(job_id: str):
        # services.cancel_job also interrupts an in-flight ComfyUI prompt.
        if not services.cancel_job(job_id):
            return _error(409, "not_cancellable",
                          "Job is already finished or does not exist.")
        return jsonify(services.queue.get(job_id).to_dict())  # type: ignore[union-attr]

    @api.post("/jobs/<job_id>/retry")
    def retry_job(job_id: str):
        if not services.queue.retry(job_id):
            return _error(409, "not_retryable",
                          "Only failed or cancelled jobs can be re-queued.")
        return jsonify(services.queue.get(job_id).to_dict())  # type: ignore[union-attr]

    # -- queue management -----------------------------------------------------------
    @api.delete("/jobs/<job_id>")
    def delete_job(job_id: str):
        if not services.queue.delete(job_id):
            return _error(409, "not_deletable",
                          "Running jobs can't be deleted — cancel first.")
        return jsonify({"deleted": job_id})

    @api.post("/jobs/clear")
    def clear_jobs():
        body = request.get_json(silent=True) or {}
        count = services.queue.clear(str(body.get("scope", "finished")))
        return jsonify({"cleared": count})

    @api.get("/queue/state")
    def queue_state():
        return jsonify({"paused": services.queue.paused,
                        "order": services.queue.pending_order()})

    @api.post("/queue/pause")
    def pause_queue():
        services.queue.pause()
        services.events.log("info", "Queue paused by the user")
        return jsonify({"paused": True})

    @api.post("/queue/resume")
    def resume_queue():
        services.queue.resume()
        services.events.log("info", "Queue resumed")
        return jsonify({"paused": False})

    @api.post("/jobs/<job_id>/move")
    def move_job(job_id: str):
        body = request.get_json(silent=True) or {}
        if not services.queue.move(job_id, str(body.get("to", ""))):
            return _error(409, "not_movable",
                          "Only pending jobs can be reordered.")
        return jsonify({"order": services.queue.pending_order()})

    # -- behind-the-scenes event stream ------------------------------------------------
    @api.get("/events")
    def list_events():
        """Merged real-time execution log: system events (health monitor,
        restarts) + every job's log lines, newest last. LLM-reasoning lines
        are filtered out — this stream shows what the app is DOING."""
        limit = min(int(request.args.get("limit", 300)), 500)
        merged = list(services.events.list())
        for job in services.queue.list()[:30]:
            src = f"{job.type} · {job.id[:6]}"
            for entry in job.logs:
                if entry["msg"].startswith("[llm]"):
                    continue  # execution log, not model reasoning
                merged.append({**entry, "source": src})
        merged.sort(key=lambda e: e["t"])
        return jsonify(merged[-limit:])

    @api.delete("/events")
    def clear_events():
        """Delete the Behind-the-Scenes log: the system event buffer AND the
        stored log lines of every non-running job (the stream merges both).
        Job records themselves — prompts, states, results — are kept."""
        events_cleared = services.events.clear()
        jobs_stripped = services.queue.clear_logs()
        services.events.log("info", "Behind-the-Scenes log deleted by the "
                                    f"user ({events_cleared} system entries, "
                                    f"logs of {jobs_stripped} jobs)")
        return jsonify({"events_cleared": events_cleared,
                        "jobs_stripped": jobs_stripped})

    @api.delete("/history/prompts")
    def clear_prompt_history():
        """Delete the prompt history: every finished (completed / failed /
        cancelled) job record — including its prompt and logs, in memory AND
        across the whole SQLite history — plus the verbatim prompt text kept
        in the workflow-learning memory. Pending and running jobs are
        untouched, and prompts already saved into gallery recipe cards stay
        with their images."""
        cleared = services.queue.clear("finished")
        scrubbed = services.experience.scrub_prompts()
        services.events.log("info", f"Prompt history deleted by the user "
                                    f"({cleared} job records, {scrubbed} "
                                    "learning-memory prompts blanked)")
        return jsonify({"cleared": cleared, "prompts_scrubbed": scrubbed})

    # -- AI workflow generation ---------------------------------------------------------
    @api.post("/workflows/generate")
    def generate_workflow():
        body = request.get_json(silent=True) or {}
        task = body.get("task", "generate")
        prompt = (body.get("prompt") or "").strip()
        if not prompt:
            return _error(400, "missing_field", "prompt is required.")
        verdict = services.safety.check(prompt, editing=(task != "generate"))
        if not verdict.allowed:
            log.warning("Safety block (workflow gen): category=%s", verdict.category)
            return _error(422, f"safety_{verdict.category}", verdict.reason or "Blocked.")
        try:
            # Live inventory (installed checkpoints) makes plans concrete when
            # ComfyUI is running; None is fine when it isn't.
            result = services.workflow_ai.generate(
                task, prompt, context=services.workflow_context())
        except WorkflowNotAllowedError as exc:
            return _error(422, "task_not_allowed", str(exc))
        except WorkflowGenerationError as exc:
            return _error(422, "generation_failed", str(exc))
        except LLMRefusedError as exc:
            return _error(422, "llm_refused", str(exc))
        except LLMUnavailableError as exc:
            return _error(503, "llm_unavailable", str(exc))
        return jsonify({"task": result.task, "graph": result.graph,
                        "provenance": result.provenance})

    @api.post("/workflows/run")
    def run_workflow():
        body = request.get_json(silent=True) or {}
        task = body.get("task", "generate")
        prompt = (body.get("prompt") or "").strip()
        if not prompt:
            return _error(400, "missing_field", "prompt is required.")
        verdict = services.safety.check(prompt, editing=(task != "generate"))
        if not verdict.allowed:
            log.warning("Safety block (workflow run): category=%s", verdict.category)
            return _error(422, f"safety_{verdict.category}", verdict.reason or "Blocked.")
        # "make 4 images of X" queues 4 SEQUENTIAL renders (the hardware
        # rarely batches) — each with its own seed, appearing in the gallery
        # as they finish.
        count, cleaned = (quality.count_request(prompt)
                          if task == "generate" else (1, prompt))
        payload = {"task": task, "prompt": cleaned}
        if body.get("asset_id"):
            if services.store.get_asset(body["asset_id"]) is None:
                return _error(404, "not_found", "Asset not found.")
            payload["asset_id"] = body["asset_id"]
        job = services.queue.enqueue("workflow", payload)
        if count > 1:
            extra_ids = [services.queue.enqueue("workflow", dict(payload)).id
                         for _ in range(count - 1)]
            services.events.log(
                "info", f"Queued {count} renders for \"{cleaned[:80]}\" — "
                        "they run one after another")
            out = job.to_dict()
            out.update({"batch_count": count,
                        "batch_job_ids": [job.id, *extra_ids]})
            return jsonify(out), 202
        return jsonify(job.to_dict()), 202

    # -- runtime settings (Civitai token) -----------------------------------------
    @api.get("/settings")
    def get_settings():
        # Never echo the token itself — only whether one is configured.
        return jsonify({"civitai_token_set": bool(services.settings.civitai_token)})

    @api.post("/settings")
    def update_settings():
        body = request.get_json(silent=True) or {}
        if "civitai_token" in body:
            services.set_civitai_token(str(body["civitai_token"]))
        return jsonify({"civitai_token_set": bool(services.settings.civitai_token)})

    # -- "Improve the LLM": discover + approve new workflows ----------------------
    @api.post("/workflows/discover")
    def discover_workflows():
        job = services.queue.enqueue("discover", {})
        return jsonify(job.to_dict()), 202

    @api.post("/workflows/approve")
    def approve_workflow():
        body = request.get_json(silent=True) or {}
        cand_id = body.get("id")
        if not cand_id:
            return _error(400, "missing_field", "id is required.")
        try:
            saved = services.save_candidate(
                cand_id, live_test=bool(body.get("live_test", True)))
        except Exception as exc:  # noqa: BLE001 — surface the reason
            return _error(422, "approve_failed", str(exc))
        return jsonify(saved), 201

    # -- safety rules (user-editable; built-ins stay locked) ----------------------
    def _no_store(payload, status=200):
        resp = jsonify(payload)
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        return resp, status

    @api.get("/safety/rules")
    def list_safety_rules():
        return _no_store({
            "builtin": services.safety.builtin_summary(),
            "custom": services.safety_rules.list(),
        })

    @api.post("/safety/rules")
    def add_safety_rule():
        body = request.get_json(silent=True) or {}
        try:
            rule = services.safety_rules.add(
                body.get("pattern", ""), body.get("reason", ""),
                body.get("category", "custom"))
        except SafetyRuleError as exc:
            return _error(400, "invalid_rule", str(exc))
        return _no_store(rule, 201)

    @api.delete("/safety/rules/<int:rule_id>")
    def delete_safety_rule(rule_id: int):
        if not services.safety_rules.delete(rule_id):
            return _error(404, "not_found", "Rule not found.")
        return _no_store({"deleted": rule_id})

    @api.post("/video")
    def create_video():
        body = request.get_json(silent=True) or {}
        asset_id = body.get("asset_id")
        prompt = (body.get("prompt") or "").strip()
        if not asset_id:
            return _error(400, "missing_field", "asset_id is required.")
        verdict = services.safety.check(prompt, editing=True)
        if not verdict.allowed:
            log.warning("Safety block (video): category=%s", verdict.category)
            return _error(422, f"safety_{verdict.category}", verdict.reason or "Blocked.")
        if services.store.get_asset(asset_id) is None:
            return _error(404, "not_found", "Asset not found.")
        payload = {"asset_id": asset_id, "prompt": prompt}
        for key in ("width", "height", "length"):
            if body.get(key) is not None:
                payload[key] = body[key]
        job = services.queue.enqueue("video", payload)
        return jsonify(job.to_dict()), 202

    # -- rich civitai search + periodic index ------------------------------------------
    @api.get("/models/civitai")
    def civitai_search():
        query = (request.args.get("q") or "").strip()
        type_key = (request.args.get("type") or "checkpoint").strip()
        try:
            hits = services.model_search.search_civitai_rich(query, type_key)
        except ModelSearchError as exc:
            return _error(503, "search_unavailable", str(exc))
        # Download-time content policy lives in safety.py.
        return jsonify([h for h in hits
                        if not model_source_blocked(h.get("nsfw", False))])

    @api.get("/models/index")
    def model_index():
        type_key = (request.args.get("type") or "checkpoint").strip()
        data = services.model_index.get(type_key)
        data["entries"] = [e for e in data["entries"]
                           if not model_source_blocked(e.get("nsfw", False))]
        return jsonify(data)

    @api.post("/models/propose-civitai")
    def propose_civitai_model():
        body = request.get_json(silent=True) or {}
        cand = body.get("candidate") or {}
        name = (body.get("name") or "").strip()
        required = ("url", "sha256", "filename", "folder")
        if not name or any(not cand.get(k) for k in required):
            return _error(400, "missing_field",
                          "name and a stageable candidate are required.")
        try:
            model = services.model_search.propose_civitai(
                cand, name=name,
                purpose=body.get("purpose") or f"{cand.get('type', 'model')} "
                                               f"from civitai")
        except ModelSearchError as exc:
            return _error(409, "propose_failed", str(exc))
        return jsonify(model.to_dict()), 201

    # -- SAM status (drives the Select-Object loading indicator) -----------------------
    @api.get("/nodepacks")
    def list_node_packs():
        """Curated ComfyUI node packs with PROBED status (absent/installed/
        active/broken) — each unlocks a specific PromptForge capability."""
        return jsonify(services.node_pack_report())

    @api.post("/nodepacks/<name>/install")
    def install_node_pack(name: str):
        from ..core.node_packs import KNOWN_PACKS
        if name not in KNOWN_PACKS:
            return _error(404, "not_found",
                          "Unknown pack. Curated packs: "
                          + ", ".join(sorted(KNOWN_PACKS)))
        job = services.queue.enqueue("node_pack", {"pack": name})
        return jsonify(job.to_dict()), 202

    @api.get("/masks/status")
    def mask_status():
        return jsonify({
            "loaded": bool(getattr(services.segmentation, "is_loaded", False)),
            "adapter": services.segmentation.name,
        })

    # -- model search (Hugging Face hub) ------------------------------------------------
    @api.get("/models/search")
    def search_models():
        query = (request.args.get("q") or "").strip()
        if not query:
            return _error(400, "missing_field", "q is required.")
        try:
            hits = services.model_search.search(query)
        except ModelSearchError as exc:
            return _error(503, "search_unavailable", str(exc))
        return jsonify([vars(h) for h in hits])

    @api.get("/models/files")
    def list_model_files():
        repo = (request.args.get("repo") or "").strip()
        if not repo:
            return _error(400, "missing_field", "repo is required.")
        try:
            files = services.model_search.list_weight_files(repo)
        except ModelSearchError as exc:
            return _error(503, "search_unavailable", str(exc))
        return jsonify([vars(f) for f in files])

    @api.post("/models/propose")
    def propose_model():
        body = request.get_json(silent=True) or {}
        missing = [k for k in ("repo", "file", "name", "purpose") if not body.get(k)]
        if missing:
            return _error(400, "missing_field", f"Required: {', '.join(missing)}.")
        try:
            model = services.model_search.propose(
                body["repo"], body["file"],
                name=body["name"], purpose=body["purpose"],
                vram_gb=body.get("vram_gb"))
        except ModelSearchError as exc:
            return _error(409, "propose_failed", str(exc))
        return jsonify(model.to_dict()), 201

    # -- models --------------------------------------------------------------------------
    @api.get("/models")
    def list_models():
        # progress/note: live download telemetry so a Download click visibly
        # does something right in the Models tab.
        return jsonify([
            m.to_dict() | {
                "progress": services.registry.progress.get(m.name),
                "note": services.registry.notes.get(m.name),
            }
            for m in services.registry.list()
        ])

    @api.post("/models/<name>/download")
    def download_model(name: str):
        model = services.registry.get(name)
        if model is None:
            return _error(404, "not_found", "Model is not in the registry.")
        if not model.url:
            return _error(409, "no_url",
                          "No download URL configured for this model. "
                          "Add one in the registry first (see README).")
        job = services.queue.enqueue("model_download", {"model": name})
        return jsonify(job.to_dict()), 202

    app.register_blueprint(api)

    # -- serve built frontend (if `npm run build` has produced frontend/dist) --------------
    @app.get("/")
    def index():
        if dist.exists():
            return send_file(dist / "index.html")
        return jsonify({
            "app": "PromptForge API",
            "hint": "Frontend not built. Run `npm install && npm run dev` in frontend/ "
                    "or `npm run build` to serve it from here. API lives under /api.",
        })

    @app.errorhandler(413)
    def too_large(_e):
        return _error(413, "too_large",
                      f"Upload exceeds {services.settings.max_upload_mb} MB.")

    @app.errorhandler(500)
    def internal(_e):
        return _error(500, "internal",
                      "Something went wrong on the server. Check the logs panel.")

    return app
