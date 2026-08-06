"""Model knowledge base: what each installed model performs best at.

For every checkpoint the studio can load, the studio searches online
(civitai: description, download stats, base model, trigger words), the LLM
distills the findings into concrete capability notes — best at / avoid /
prompt style — plus a 1-10 quality rating, and everything is written to a
separate human-readable file (data/model_knowledge.json). The planner
injects these notes into every model choice, and a research job re-runs
automatically whenever a new model finishes downloading.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .llm import LLMClient, LLMError
from .quality import _parse_json

_INTEL_SYSTEM = """You write factual capability notes for image-generation
models (Stable Diffusion checkpoints and similar). Reply with ONLY JSON:
{"best_at": "<subjects/styles this model excels at, comma-separated>",
 "avoid": "<what it is weak at or wrong for>",
 "prompt_style": "<how to prompt it for its best results>",
 "quality": <1-10 overall output quality rating>,
 "reason": "<short basis for the rating>"}
Ground the notes in the provided research when present (description,
downloads, community rating); otherwise use what is commonly known about the
model family in its filename. Be concrete ("photoreal portraits, skin
texture, natural light" / "prompt in short natural sentences") — never
generic marketing. Rate honestly: an old base model is a 5, a top community
photoreal model an 8-9."""


class ModelIntel:
    """Per-checkpoint capability notes, persisted to a standalone JSON file."""

    def __init__(self, path: Path, search: Any = None):
        self.path = path
        self.search = search  # ModelSearch (optional: offline still works)

    # -- storage ------------------------------------------------------------------
    def load(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save(self, data: dict[str, dict[str, Any]]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True),
                       encoding="utf-8")
        tmp.replace(self.path)

    def get(self, filename: str) -> dict[str, Any] | None:
        return self.load().get(filename)

    def missing(self, filenames: list[str]) -> list[str]:
        """Installed checkpoints that have no notes yet."""
        known = self.load()
        return [f for f in filenames if f not in known]

    # -- planner context ------------------------------------------------------------
    def summary(self, filenames: list[str], max_chars: int = 900) -> str | None:
        """A compact 'what each model is good at' block for model-choice
        prompts. Best-rated first; None when nothing is known yet."""
        known = self.load()
        rows = [(f, known[f]) for f in filenames if f in known]
        if not rows:
            return None
        rows.sort(key=lambda r: -int(r[1].get("quality") or 0))
        lines = ["Model knowledge (researched online):"]
        for f, n in rows[:6]:
            line = (f"- {f}: quality {n.get('quality', '?')}/10 — best at "
                    f"{n.get('best_at', '?')}")
            if n.get("prompt_style"):
                line += f"; prompt: {n['prompt_style']}"
            if n.get("avoid"):
                line += f"; avoid: {n['avoid']}"
            lines.append(line[:220])
        out = "\n".join(lines)
        return out[:max_chars]

    # -- research -----------------------------------------------------------------
    @staticmethod
    def _query_for(filename: str) -> str:
        stem = Path(filename).stem
        words = re.sub(r"[._-]+", " ", stem)
        words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", words)  # camelCase
        words = re.sub(r"\b(v\d+|fp16|fp8|final|pruned|emaonly)\b", " ",
                       words, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", words).strip()

    def _gather(self, filename: str,
                log: Callable[[str], None]) -> tuple[str, str]:
        """(research text, source note) from online search; empty when
        nothing was found or the search backend is unavailable."""
        if self.search is None:
            return "", "llm-knowledge"
        query = self._query_for(filename)
        log(f"[search] civitai: what is '{query}' best at?")
        try:
            hits = self.search.search_civitai_rich(query, "checkpoint",
                                                   limit=5)
        except Exception:  # noqa: BLE001 — offline: notes from LLM knowledge
            return "", "llm-knowledge"
        stem = Path(filename).stem.lower()
        best = None
        for h in hits:
            hit_file = str(h.get("filename") or "").lower()
            if hit_file and Path(hit_file).stem in (stem, stem + "-inpainting"):
                best = h
                break
        if best is None and hits:
            # Fuzzy fallback needs REAL overlap — a wrong model page poisons
            # the notes worse than no page at all (seen live: juggernaut
            # matched to an unrelated 'selectivecolor' derivative).
            tokens = set(re.findall(r"[a-z]{3,}", stem))

            def overlap(h: dict) -> int:
                name_tokens = set(re.findall(
                    r"[a-z]{3,}", str(h.get("name", "")).lower()))
                return sum(1 for t in tokens
                           if any(t in n or n in t for n in name_tokens))

            best = max(hits, key=overlap)
            if overlap(best) < 2:
                best = None
        if not best:
            return "", "llm-knowledge"
        parts = [f"Model page: {best.get('name')} by {best.get('creator')}",
                 f"Downloads: {best.get('downloads')}, community thumbs-up: "
                 f"{best.get('rating')}",
                 f"Base model: {best.get('base_model')}"]
        if best.get("description"):
            parts.append(f"Description: {best['description']}")
        if best.get("trigger_words"):
            parts.append("Trigger words: " + ", ".join(best["trigger_words"]))
        return "\n".join(parts), f"civitai:{best.get('name', '?')}"

    def research(self, filename: str, llm: LLMClient,
                 log: Callable[[str], None] | None = None,
    ) -> dict[str, Any] | None:
        """Search online for what `filename` performs best at, distill via
        the LLM, rate 1-10, persist. Returns the stored entry, or None when
        the LLM is unavailable / produced nothing usable (retry later)."""
        log = log or (lambda m: None)
        research, source = self._gather(filename, log)
        ask = (f"Model file: {filename}\n"
               + (f"Research found online:\n{research}\n" if research
                  else "No online research available — use known facts about "
                       "this model family only.\n")
               + "Reply with the JSON capability notes.")
        try:
            reply = llm.complete(_INTEL_SYSTEM, ask, max_tokens=300)
        except LLMError:
            return None
        data = _parse_json(reply.text)
        if not data or not str(data.get("best_at", "")).strip():
            return None
        try:
            rating = max(1, min(10, int(float(data.get("quality", 5)))))
        except (TypeError, ValueError):
            rating = 5
        entry = {
            "best_at": re.sub(r"\s+", " ", str(data["best_at"]))[:200],
            "avoid": re.sub(r"\s+", " ", str(data.get("avoid", "")))[:160],
            "prompt_style": re.sub(r"\s+", " ",
                                   str(data.get("prompt_style", "")))[:160],
            "quality": rating,
            "reason": re.sub(r"\s+", " ", str(data.get("reason", "")))[:160],
            "source": source,
            "researched_at": datetime.now(UTC).isoformat(
                timespec="seconds"),
        }
        stored = self.load()
        stored[filename] = entry
        self._save(stored)
        return entry
