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
import inspect
import io
import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PIL import Image


def ask_with_schema(critic: Any, image: Any, question: str,
                    schema: dict[str, Any] | None) -> str:
    """critic.ask with a JSON schema when the critic supports it.

    The vision twin of llm.complete_with_schema: shape enforcement is an
    upgrade, never a requirement, so every scripted test critic with the
    plain (image, question) signature keeps working unconstrained. The
    signature is inspected rather than TypeError-probed — an error raised
    INSIDE a real ask must never be mistaken for 'no schema support'."""
    if schema is not None:
        try:
            supports = "schema" in inspect.signature(critic.ask).parameters
        except (TypeError, ValueError):
            supports = False
        if supports:
            return critic.ask(image, question, schema=schema)
    return critic.ask(image, question)

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

    def ask(self, image: Image.Image, question: str,
            schema: dict[str, Any] | None = None) -> str:
        """Generic vision question against the local model; returns raw text
        (JSON-formatted when the question demands it). Used for critique and
        for view-angle classification in the avatar pipeline. `schema`
        (a JSON Schema) upgrades "please answer as JSON" into a grammar the
        server enforces — the reply cannot be misshapen."""
        return self._vision(image, question, force_json=True, schema=schema)

    def describe(self, image: Image.Image, question: str) -> str:
        """Free-text vision answer (no JSON forcing) — scene descriptions
        that get appended to render prompts as context."""
        return self._vision(image, question, force_json=False)

    def _vision(self, image: Image.Image, question: str,
                force_json: bool,
                schema: dict[str, Any] | None = None) -> str:
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
        if schema is not None:
            payload["format"] = schema
        elif force_json:
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

    # The critique reply as an enforced grammar: score must be a number,
    # issues must be strings. The bare-number salvage below stays for the
    # unconstrained paths (older servers, scripted fakes).
    CRITIQUE_SCHEMA = {
        "type": "object",
        "properties": {
            "score": {"type": "number"},
            "issues": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["score"],
    }

    def critique(self, image: Image.Image, prompt: str) -> Critique:
        text = self.ask(image, CRITIC_PROMPT.format(prompt=prompt[:400]),
                        schema=self.CRITIQUE_SCHEMA)
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


class CriticChain:
    """Primary critic first, local fallback when it fails — the vision
    counterpart of FallbackLLM. Used during peer delegation so quality
    checks run on the render machine, but a peer without the vision model
    (or an older build without the proxy) silently degrades to checking
    here instead of dropping quality checks on the floor."""

    def __init__(self, primary: ImageCritic, fallback: ImageCritic):
        self.primary = primary
        self.fallback = fallback

    def _call(self, name: str, *args, **kwargs):
        try:
            return getattr(self.primary, name)(*args, **kwargs)
        except Exception:  # noqa: BLE001 — any peer failure means "check here"
            return getattr(self.fallback, name)(*args, **kwargs)

    def ask(self, image: Image.Image, question: str,
            schema: dict[str, Any] | None = None) -> str:
        return self._call("ask", image, question, schema=schema)

    def describe(self, image: Image.Image, question: str) -> str:
        return self._call("describe", image, question)

    def critique(self, image: Image.Image, prompt: str) -> Critique:
        return self._call("critique", image, prompt)

    def __getattr__(self, name: str):
        # Attributes (model name etc.) come from the primary; only the
        # three calls above carry the fallback behaviour.
        return getattr(self.primary, name)
