"""Prompt-aware model selection ("scout").

Given a user prompt and the checkpoints ComfyUI can already load, an LLM
decides which installed checkpoint fits best — or, when none fits, proposes a
Hugging Face search, picks a candidate file, and stages it in the registry so
the job can download it (checksum-verified, size-capped).

The scout is advisory and fail-safe: any parsing/search/LLM failure falls
back to the first installed checkpoint, and every decision is logged.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from .llm import LLMClient, LLMError
from .model_search import ModelSearch, ModelSearchError
from .safety import model_source_blocked
from .trust import Evidence, TrustJudge

# Never auto-download single files bigger than this (VRAM/disk sanity brake;
# an SDXL checkpoint is ~7 GB).
MAX_AUTO_BYTES = 8 * 1024**3

SCOUT_SYSTEM = """You select the best Stable-Diffusion-family checkpoint for \
an image prompt. Reply with ONLY a JSON object, one of:
  {"use": "<exact filename from the installed list>"}
  {"search": "<short hugging face search query>", "reason": "<why>"}
Prefer an installed checkpoint unless it clearly cannot serve the prompt's
style (e.g. only an inpainting model is installed and the task is generation,
or the prompt demands a specialised style). Search queries should name the
model family and 'safetensors', e.g. "realistic vision v6 sd15 safetensors".
"""


@dataclass
class ScoutDecision:
    checkpoint: str          # filename ComfyUI should load
    downloaded: str | None   # registry name if the scout staged+fetched one
    note: str                # human-readable trail for the job log


class ModelScout:
    def __init__(self, llm: LLMClient, search: ModelSearch,
                 downloader, registry,
                 log: Callable[[str], None] | None = None,
                 trust: TrustJudge | None = None,
                 max_auto_bytes: int = MAX_AUTO_BYTES):
        self.llm = llm
        self.search = search
        self.downloader = downloader
        self.registry = registry
        self.trust = trust or TrustJudge(llm)
        self.max_auto_bytes = max_auto_bytes
        self._log = log or (lambda m: None)

    # -- public -------------------------------------------------------------------
    def choose(self, prompt: str, task: str, installed: list[str],
               allow_download: bool,
               progress: Callable[[int, int | None], None] | None = None,
               force_search: bool = False,
               log: Callable[[str], None] | None = None) -> ScoutDecision:
        """Pick a checkpoint. With force_search=True (the user explicitly
        asked for a model search) the installed list is only the fallback —
        a hub search WILL be attempted. `log` scopes narration to the calling
        job without mutating shared state."""
        if log is not None:
            self._log = log  # single queue worker; per-call override
        fallback = ScoutDecision(installed[0], None,
                                 f"default: first installed ({installed[0]})")
        ask = (f"Task: {task}\nInstalled checkpoints: {', '.join(installed)}\n"
               f"Prompt: {prompt}\n")
        if force_search:
            ask += ("The user EXPLICITLY asked to find/download a better "
                    "model online. You MUST reply with the search action.\n")
        ask += "Reply with the JSON decision."

        decision: dict = {}
        try:
            decision = self._parse(self.llm.complete(SCOUT_SYSTEM, ask).text)
        except (LLMError, ValueError) as exc:
            if not force_search:
                self._log(f"Scout unavailable ({exc}); using {fallback.checkpoint}")
                return fallback
            self._log(f"Scout LLM unavailable ({exc}); deriving search query "
                      "from the prompt")

        query = str(decision.get("search", "")).strip()
        if force_search and not query:
            # Obey the user even when the LLM won't: derive a query in code.
            words = [w for w in re.findall(r"[a-zA-Z]{4,}", prompt)][:5]
            query = " ".join(words) + " photorealistic sd15 checkpoint safetensors"

        if query and allow_download:
            try:
                return self._download_best(query,
                                           str(decision.get("reason", "")),
                                           installed, progress)
            except (ModelSearchError, Exception) as exc:  # noqa: BLE001 - fail safe
                self._log(f"Scout download path failed ({exc}); "
                          f"using {fallback.checkpoint}")
                return fallback

        if "use" in decision:
            name = str(decision["use"])
            if name in installed:
                return ScoutDecision(name, None, f"scout chose installed '{name}'")
            self._log(f"Scout named unknown checkpoint '{name}'; "
                      f"using {fallback.checkpoint}")
        return fallback

    # -- internals ------------------------------------------------------------------
    @staticmethod
    def _parse(text: str) -> dict:
        cleaned = text.strip()
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
        if fence:
            cleaned = fence.group(1)
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("scout reply is not a JSON object")
        return data

    def _download_best(self, query: str, reason: str, installed: list[str],
                       progress) -> ScoutDecision:
        # "[search]" markers render as live search activity in the GUI.
        self._log(f"[search] Hugging Face: '{query}'"
                  + (f" — {reason}" if reason else ""))
        repos = [r for r in self.search.search(query, limit=6) if not r.gated]
        for repo in repos:
            files = [f for f in self.search.list_weight_files(repo.repo_id)
                     if f.filename.lower().endswith(".safetensors")
                     and f.sha256  # published checksum required for auto-install
                     and f.size_bytes and f.size_bytes <= self.max_auto_bytes]
            if not files:
                continue
            best = files[0]  # list is size-sorted desc: biggest full model
            # Trust judgment: evidence-based proceed/reject, logged either way.
            verdict = self.trust.judge(Evidence(
                repo_id=repo.repo_id, filename=best.filename,
                url=f"https://huggingface.co/{repo.repo_id}",
                size_bytes=best.size_bytes, sha256=best.sha256,
                downloads=repo.downloads, likes=repo.likes, gated=repo.gated))
            self._log(f"Trust check ({verdict.judged_by}) for {repo.repo_id}: "
                      f"{'PROCEED' if verdict.proceed else 'REJECTED'} — "
                      f"{verdict.reason}")
            if not verdict.proceed:
                continue
            reg_name = f"scout-{repo.repo_id.split('/')[-1][:40]}".lower()
            if self.registry.get(reg_name) is None:
                self.search.propose(repo.repo_id, best.filename, name=reg_name,
                                    purpose=f"checkpoint (scout: {query})")
            self._log(f"Scout downloading {repo.repo_id}/{best.filename} "
                      f"({(best.size_bytes or 0) // 1024**2} MB, verified)")
            try:
                self.downloader.download(reg_name, progress)
            except Exception as exc:  # noqa: BLE001 — try the next candidate
                self._log(f"Download failed ({exc}); trying the next candidate")
                continue
            filename = best.filename.split("/")[-1]
            return ScoutDecision(filename, reg_name,
                                 f"scout downloaded '{filename}' from "
                                 f"{repo.repo_id}")

        # Second source: civitai.com (community checkpoints, hashed files).
        self._log(f"[search] Civitai: '{query}'")
        for cand in self.search.search_civitai(query):
            # Content policy lives in safety.py — never embed rules here.
            if model_source_blocked(cand["nsfw"]) or not cand["size_bytes"] \
                    or cand["size_bytes"] > self.max_auto_bytes:
                continue
            verdict = self.trust.judge(Evidence(
                repo_id=f"civitai/{cand['creator']}", filename=cand["filename"],
                url=cand["url"], size_bytes=cand["size_bytes"],
                sha256=cand["sha256"], downloads=cand["downloads"]))
            self._log(f"Trust check ({verdict.judged_by}) for civitai "
                      f"'{cand['name']}' by {cand['creator']}: "
                      f"{'PROCEED' if verdict.proceed else 'REJECTED'} — "
                      f"{verdict.reason}")
            if not verdict.proceed:
                continue
            reg_name = f"scout-{cand['filename'].rsplit('.', 1)[0][:40]}".lower()
            if self.registry.get(reg_name) is None:
                self.search.propose_civitai(cand, name=reg_name,
                                            purpose=f"checkpoint (scout: {query})")
            self._log(f"Scout downloading civitai '{cand['name']}' "
                      f"({cand['size_bytes'] // 1024**2} MB, verified)")
            try:
                self.downloader.download(reg_name, progress)
            except Exception as exc:  # noqa: BLE001 — e.g. 403 without an
                # account token; try the next candidate instead of giving up.
                self._log(f"Download failed ({exc}); trying the next candidate")
                continue
            return ScoutDecision(cand["filename"], reg_name,
                                 f"scout downloaded '{cand['filename']}' "
                                 f"from civitai ({cand['creator']})")
        raise ModelSearchError(f"no suitable verified file for '{query}'")
