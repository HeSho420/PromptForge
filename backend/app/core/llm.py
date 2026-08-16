"""Local-first LLM layer (powers AI workflow generation, see workflow_ai.py).

Two backends behind one protocol, chained by FallbackLLM:

  LocalLLM  – an OpenAI-compatible chat endpoint (Ollama, LM Studio,
              llama.cpp server). This is the default path: prompts stay on
              this machine.
  ClaudeLLM – the Anthropic API (Claude Fable 5), used ONLY when the local
              model fails, per project policy. Requires the `anthropic`
              package and credentials (ANTHROPIC_API_KEY or `ant auth login`).
              Requests opt into server-side refusal fallbacks so a safety
              decline is transparently re-served by claude-opus-4-8.

Honesty rule (same spirit as adapters.is_mock): every reply is stamped with
`source` ("local" | "api") and the exact model that produced it, and that
provenance is surfaced through the API — cloud output can never silently pass
as local output.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

_PULLING: set[str] = set()
_PULL_LOCK = threading.Lock()


def ollama_autopull(model: str) -> bool:
    """Kick off `ollama pull <model>` in the background, once per process.

    A missing local model must never be a dead end the user has to fix by
    hand: the current request falls back (API/heuristics, honestly stamped),
    and once the pull lands every later request runs locally again. Returns
    True when a pull was actually started."""
    with _PULL_LOCK:
        if model in _PULLING:
            return False
        _PULLING.add(model)
    exe = shutil.which("ollama")
    if not exe:
        cand = (Path(os.environ.get("LOCALAPPDATA", ""))
                / "Programs" / "Ollama" / "ollama.exe")
        exe = str(cand) if cand.exists() else None
    if not exe:
        return False
    try:
        flags = 0x08000008 if os.name == "nt" else 0  # DETACHED|NO_WINDOW
        subprocess.Popen([exe, "pull", model], creationflags=flags,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except OSError:
        return False


class LLMError(RuntimeError):
    """Base class for LLM failures."""


class LLMUnavailableError(LLMError):
    """No usable backend (unreachable, unconfigured, model not pulled)."""


class LLMRefusedError(LLMError):
    """The API declined the request for safety reasons — do not retry as-is."""


@dataclass
class LLMReply:
    text: str
    model: str
    source: str  # "local" | "api"


class LLMClient(Protocol):
    # Read-only: FallbackLLM derives it with a property.
    @property
    def source(self) -> str: ...

    def complete(self, system: str, prompt: str, max_tokens: int = 4096) -> LLMReply:
        ...


# -- local backend (OpenAI-compatible: Ollama, LM Studio, llama.cpp) -----------

def _http_post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read())


class LocalLLM:
    """Chat completions against a local OpenAI-compatible server."""

    source = "local"

    def __init__(self, base_url: str, model: str, timeout_s: float = 300.0,
                 http_post: Callable[[str, dict[str, Any], float], dict[str, Any]]
                 | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._post = http_post or _http_post_json

    NUM_CTX = 8192  # planner context now carries guides/templates/lessons

    def complete(self, system: str, prompt: str, max_tokens: int = 4096) -> LLMReply:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        # Native Ollama endpoint first: unlike the OpenAI-compat endpoint it
        # honors num_ctx, and the default 4k window would silently truncate
        # the front of a rich planning context. Non-Ollama servers 404 here
        # and get the OpenAI-compatible request below instead.
        native = self.base_url.removesuffix("/v1")
        try:
            data = self._post(native + "/api/chat", {
                "model": self.model, "stream": False, "format": "json",
                "options": {"temperature": 0.2, "num_ctx": self.NUM_CTX,
                            "num_predict": max_tokens},
                "messages": messages,
            }, self.timeout_s)
            text = (data.get("message") or {}).get("content")
            if text is not None:
                return LLMReply(text=text or "",
                                model=data.get("model", self.model),
                                source=self.source)
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().lower()
            except Exception:  # noqa: BLE001 — body is diagnostic only
                body = b""
            if exc.code == 404 and b"model" in body and b"not found" in body:
                started = ollama_autopull(self.model)
                note = (" — downloading it in the background now; this "
                        "request uses the fallback and later ones run "
                        "locally" if started else
                        f" (try: ollama pull {self.model})")
                raise LLMUnavailableError(
                    f"Local LLM server has no model '{self.model}'{note}"
                ) from exc
            # Anything else: probably not Ollama — fall through to OpenAI.
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise LLMUnavailableError(
                f"Local LLM at {self.base_url} is unreachable: {exc}. "
                "Start Ollama/LM Studio or set PROMPTFORGE_LLM_URL.") from exc

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "stream": False,
            # Grammar-constrained JSON + low temperature: small local models
            # otherwise produce syntactically broken graphs under pressure.
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
            "messages": messages,
        }
        try:
            data = self._post(self.base_url + "/chat/completions", payload, self.timeout_s)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                started = ollama_autopull(self.model)
                note = (" — downloading it in the background now" if started
                        else f" (try: ollama pull {self.model})")
                raise LLMUnavailableError(
                    f"Local LLM server has no model '{self.model}'{note}"
                ) from exc
            raise LLMUnavailableError(
                f"Local LLM at {self.base_url} returned HTTP {exc.code}.") from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise LLMUnavailableError(
                f"Local LLM at {self.base_url} is unreachable: {exc}. "
                "Start Ollama/LM Studio or set PROMPTFORGE_LLM_URL.") from exc

        try:
            text = data["choices"][0]["message"]["content"]
            model = data.get("model", self.model)
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Local LLM returned an unexpected response: {exc}") from exc
        return LLMReply(text=text or "", model=model, source=self.source)


# -- API fallback (Anthropic, Claude Fable 5) -----------------------------------

class ClaudeLLM:
    """Anthropic API client — fallback only; never the default path."""

    source = "api"

    def __init__(self, model: str = "claude-fable-5",
                 refusal_fallback_model: str = "claude-opus-4-8",
                 client_factory: Callable[[], Any] | None = None):
        self.model = model
        self.refusal_fallback_model = refusal_fallback_model
        self._client_factory = client_factory or self._default_factory
        self._client: Any = None

    @staticmethod
    def _default_factory() -> Any:
        try:
            import anthropic
        except ImportError as exc:
            raise LLMUnavailableError(
                "API fallback needs the anthropic package: pip install anthropic") from exc
        # Zero-arg client: resolves ANTHROPIC_API_KEY / auth profile itself.
        return anthropic.Anthropic()

    def complete(self, system: str, prompt: str, max_tokens: int = 4096) -> LLMReply:
        if self._client is None:
            self._client = self._client_factory()
        try:
            # Claude Fable 5: thinking is always on — the `thinking` param must
            # be omitted. Server-side fallbacks re-serve a safety decline via
            # refusal_fallback_model inside the same call.
            response = self._client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                betas=["server-side-fallback-2026-06-01"],
                fallbacks=[{"model": self.refusal_fallback_model}],
                messages=[{"role": "user", "content": prompt}],
            )
        except LLMError:
            raise
        except Exception as exc:  # anthropic.* errors, without importing eagerly
            raise LLMUnavailableError(
                f"Anthropic API call failed: {exc}. "
                "Check ANTHROPIC_API_KEY / network, or disable the API fallback.") from exc

        if getattr(response, "stop_reason", None) == "refusal":
            raise LLMRefusedError(
                "The API model declined this request (safety policy).")
        text = "".join(getattr(b, "text", "") for b in response.content
                       if getattr(b, "type", "") == "text")
        # response.model is the model that actually served the reply — it may
        # be the refusal-fallback model; report it honestly.
        return LLMReply(text=text, model=getattr(response, "model", self.model),
                        source=self.source)


def ollama_is_up(base_url: str, timeout_s: float = 4.0) -> bool:
    """True when the Ollama/OpenAI-compatible server answers. Used by the
    backend's auto-restart guard (mirrors the ComfyUI liveness check)."""
    native = base_url.rstrip("/").removesuffix("/v1")
    for probe in (native + "/api/tags", native + "/"):
        try:
            with urllib.request.urlopen(probe, timeout=timeout_s):
                return True
        except (urllib.error.URLError, OSError, TimeoutError):
            continue
    return False


def ollama_unload_all(base_url: str, timeout_s: float = 20.0) -> list[str]:
    """Ask Ollama to release its GPU memory (keep_alive=0 unloads a model).

    On single-GPU machines the LLM and the renderer share VRAM; unloading the
    LLM right before a render prevents ComfyUI from being pushed into a hard
    CUDA out-of-memory crash. Best-effort: returns the unloaded model names,
    or [] when Ollama isn't reachable. Models transparently reload on the
    next LLM call.
    """
    native = base_url.rstrip("/").removesuffix("/v1")
    unloaded: list[str] = []
    try:
        with urllib.request.urlopen(native + "/api/ps", timeout=timeout_s) as resp:
            running = json.loads(resp.read()).get("models", [])
        for entry in running:
            name = entry.get("name")
            if not name:
                continue
            req = urllib.request.Request(
                native + "/api/generate",
                data=json.dumps({"model": name, "keep_alive": 0}).encode(),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=timeout_s).read()
            unloaded.append(name)
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError):
        pass
    return unloaded


# -- chain ----------------------------------------------------------------------

class FallbackLLM:
    """Local first; API only when the local model fails. Never silent about it."""

    def __init__(self, primary: LLMClient | None, fallback: LLMClient | None = None,
                 log: Callable[[str], None] | None = None):
        self.primary = primary
        self.fallback = fallback
        self._log = log or (lambda msg: None)

    @property
    def source(self) -> str:
        return self.primary.source if self.primary else (
            self.fallback.source if self.fallback else "none")

    def complete(self, system: str, prompt: str, max_tokens: int = 4096) -> LLMReply:
        failures: list[str] = []
        for client in (self.primary, self.fallback):
            if client is None:
                continue
            try:
                return client.complete(system, prompt, max_tokens=max_tokens)
            except LLMRefusedError:
                raise  # retrying elsewhere won't (and shouldn't) help
            except LLMUnavailableError as exc:
                failures.append(f"[{client.source}] {exc}")
                self._log(f"LLM backend '{client.source}' unavailable: {exc}")
        raise LLMUnavailableError(
            "No LLM backend produced a reply. " + " | ".join(failures)
            if failures else "No LLM backend is configured.")
