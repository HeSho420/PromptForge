"""Vision critic: judges whether a render/edit looks realistic.

Uses a local Ollama vision model (llava by default) through the native
/api/chat endpoint with an attached image. Returns a 1-10 realism score plus
concrete issues, which the workflow pipeline feeds back into a strategy
change (different sampler/steps/checkpoint) when the score is too low.

Fail-open by design: if the critic model is missing or errors, jobs proceed
without the check (logged, never fabricated).
"""
from __future__ import annotations

import base64
import io
import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PIL import Image

CRITIC_PROMPT = """Rate how realistic and photoreal this image looks for the \
request: "{prompt}".
Reply with ONLY JSON: {{"score": <1-10>, "issues": ["<short issue>", ...]}}
Score 8-10: convincingly photoreal. 5-7: usable but visibly AI. 1-4: obvious \
artifacts (deformed anatomy, seams, smearing, wrong lighting, garbled text)."""


class CriticUnavailable(RuntimeError):
    pass


@dataclass
class Critique:
    score: float
    issues: list[str]
    model: str

    def summary(self) -> str:
        return (f"realism {self.score:g}/10"
                + (f" — issues: {'; '.join(self.issues[:4])}" if self.issues else ""))


def _http_post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read())


class ImageCritic:
    def __init__(self, base_url: str, model: str, timeout_s: float = 180.0,
                 http_post: Callable[[str, dict[str, Any], float], dict[str, Any]]
                 | None = None):
        # base_url is the OpenAI-compat URL ("…:11434/v1"); the native Ollama
        # API (which accepts images) lives one level up.
        self.native_url = base_url.rstrip("/").removesuffix("/v1")
        self.model = model
        self.timeout_s = timeout_s
        self._post = http_post or _http_post_json

    def ask(self, image: Image.Image, question: str) -> str:
        """Generic vision question against the local model; returns raw text
        (JSON-formatted when the question demands it). Used for critique and
        for view-angle classification in the avatar pipeline."""
        return self._vision(image, question, force_json=True)

    def describe(self, image: Image.Image, question: str) -> str:
        """Free-text vision answer (no JSON forcing) — scene descriptions
        that get appended to render prompts as context."""
        return self._vision(image, question, force_json=False)

    def _vision(self, image: Image.Image, question: str,
                force_json: bool) -> str:
        buf = io.BytesIO()
        # Bound the payload: the model doesn't need full resolution.
        img = image.convert("RGB")
        img.thumbnail((1024, 1024))
        img.save(buf, format="JPEG", quality=88)
        b64 = base64.b64encode(buf.getvalue()).decode()

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [{
                "role": "user",
                "content": question,
                "images": [b64],
            }],
        }
        if force_json:
            payload["format"] = "json"
        try:
            data = self._post(f"{self.native_url}/api/chat", payload, self.timeout_s)
            return data["message"]["content"]
        except urllib.error.HTTPError as exc:
            # Vision model not pulled: heal it in the background instead of
            # leaving quality checks silently off forever on fresh machines.
            if exc.code == 404:
                from .llm import ollama_autopull
                if ollama_autopull(self.model):
                    raise CriticUnavailable(
                        f"Critic model '{self.model}' is downloading in the "
                        "background — checks resume when it lands") from exc
            raise CriticUnavailable(f"Critic model unavailable: {exc}") from exc
        except (urllib.error.URLError, OSError, TimeoutError, KeyError,
                TypeError, json.JSONDecodeError) as exc:
            raise CriticUnavailable(f"Critic model unavailable: {exc}") from exc

    def critique(self, image: Image.Image, prompt: str) -> Critique:
        text = self.ask(image, CRITIC_PROMPT.format(prompt=prompt[:400]))
        try:
            parsed = json.loads(text)
            score = float(parsed["score"])
            issues = [str(i) for i in parsed.get("issues", [])][:8]
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            # Salvage a bare number if the model rambled. Match a WHOLE
            # number with its full decimals, then clamp: the old regex read
            # "0.5" as 5, truncated "8.75" to 8.0, and — worse — could not
            # match a two-digit "12" at all, raising CriticUnavailable and
            # aborting a critique it should have salvaged as a 10.
            m = re.search(r"\d{1,2}(?:\.\d+)?", text)
            if not m:
                raise CriticUnavailable(
                    f"Critic reply unparseable: {text[:200]}") from None
            score, issues = float(m.group(0)), []
        return Critique(score=max(1.0, min(10.0, score)), issues=issues,
                        model=self.model)
