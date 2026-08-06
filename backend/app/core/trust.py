"""Download trust layer — every file fetched from the internet is vetted.

Three lines of defense, applied before a single byte is written:

  1. Host allowlist (hard gate, enforced in the downloader): model files may
     only come from hosts we know serve verifiable content.
  2. Format gate (hard gate): pickle-based weight formats can execute code on
     load, so over the network they are only allowed for explicitly vetted
     registry entries (e.g. Meta's official SAM checkpoint). safetensors/
     gguf/onnx are data-only containers.
  3. Judgment (LLM with rule fallback): before the scout or mirror-resolver
     downloads a community file, the evidence (org, downloads, format,
     checksum, size) is judged — proceed or reject with a reason, logged.
     If the LLM is unavailable the conservative rule verdict decides.

Integrity is separate and unconditional: downloads stream through sha256 and
a mismatch discards the file (registry.py).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib.parse import urlparse

from .llm import LLMClient, LLMError

# Hosts model files may be downloaded from. Extend deliberately.
TRUSTED_HOSTS = {
    "huggingface.co",
    "cdn-lfs.huggingface.co",
    "dl.fbaipublicfiles.com",
    "civitai.com",
}

SAFE_FORMATS = (".safetensors", ".gguf", ".onnx")
PICKLE_FORMATS = (".pt", ".pth", ".ckpt", ".bin")

# Organisations with an established public record; used by the rule fallback.
KNOWN_ORGS = {
    "stabilityai", "comfy-org", "runwayml", "black-forest-labs", "wan-ai",
    "meta-llama", "openai", "google", "microsoft", "qwen", "webui",
}

JUDGE_SYSTEM = """You judge whether an AI model file from a public hub is \
safe to download. Reply ONLY JSON: {"proceed": true/false, "reason": "<short>"}.
Reject when: the file format can execute code (.pt/.pth/.ckpt/.bin) and the \
publisher is unknown; the repo looks like a typo-squat of a known org; there \
is no published checksum; or the size is implausible for the claimed model.
Prefer to proceed for well-known organisations publishing .safetensors with \
checksums."""


class UntrustedDownloadError(RuntimeError):
    """Raised by hard gates; never bypassed by the LLM."""


@dataclass
class Evidence:
    repo_id: str
    filename: str
    url: str
    size_bytes: int | None
    sha256: str | None
    downloads: int = 0
    likes: int = 0
    gated: bool = False

    def summary(self) -> str:
        return (f"repo={self.repo_id} file={self.filename} "
                f"size={self.size_bytes} sha256={'yes' if self.sha256 else 'NO'} "
                f"downloads={self.downloads} likes={self.likes} "
                f"gated={self.gated}")


@dataclass
class Verdict:
    proceed: bool
    reason: str
    judged_by: str  # "llm" | "rules"


def check_host(url: str) -> None:
    """Hard gate: only http(s) downloads from allowlisted hosts (file:// is
    local and used by tests/local mirrors)."""
    parsed = urlparse(url or "")
    if parsed.scheme in ("http", "https"):
        host = (parsed.hostname or "").lower()
        if host not in TRUSTED_HOSTS and not host.endswith(".huggingface.co"):
            raise UntrustedDownloadError(
                f"Host '{host}' is not on the trusted download allowlist "
                f"({', '.join(sorted(TRUSTED_HOSTS))}).")


def check_format(url: str, allow_pickle: bool) -> None:
    """Hard gate: pickle formats over the network need an explicit vetting
    flag on the registry entry (meta.allow_pickle)."""
    parsed = urlparse(url or "")
    if parsed.scheme not in ("http", "https"):
        return
    path = (parsed.path or "").lower()
    if path.endswith(PICKLE_FORMATS) and not allow_pickle:
        raise UntrustedDownloadError(
            "Pickle-based weight formats can execute code on load and are "
            "blocked for unvetted sources — prefer a .safetensors variant.")


def rule_verdict(e: Evidence) -> Verdict:
    """Conservative rules used when the LLM cannot judge."""
    org = e.repo_id.split("/")[0].lower()
    fmt_safe = e.filename.lower().endswith(SAFE_FORMATS)
    if not fmt_safe:
        return Verdict(False, "unsafe format from unvetted source", "rules")
    if not e.sha256:
        return Verdict(False, "no published checksum", "rules")
    if org in KNOWN_ORGS or e.downloads >= 100:
        return Verdict(True, f"safetensors + checksum + known org/adoption "
                             f"({org}, {e.downloads} downloads)", "rules")
    return Verdict(False, f"unknown publisher '{org}' with low adoption", "rules")


class TrustJudge:
    """LLM-backed judgment with the rule verdict as floor and fallback.

    The LLM may only REJECT beyond the rules — it can never override the
    hard gates (host/format) or approve something the rules reject for a
    missing checksum.
    """

    def __init__(self, llm: LLMClient | None):
        self.llm = llm

    def judge(self, e: Evidence) -> Verdict:
        rules = rule_verdict(e)
        if not rules.proceed or self.llm is None:
            return rules
        try:
            reply = self.llm.complete(
                JUDGE_SYSTEM, f"Candidate: {e.summary()}\nJudge it.",
                max_tokens=200)
            cleaned = re.sub(r"^```(?:json)?|```$", "", reply.text.strip(),
                             flags=re.MULTILINE).strip()
            data = json.loads(cleaned)
            if not bool(data.get("proceed", False)):
                return Verdict(False, str(data.get("reason", "LLM rejected")),
                               "llm")
            return Verdict(True, str(data.get("reason", rules.reason)), "llm")
        except (LLMError, json.JSONDecodeError, AttributeError, TypeError):
            return rules
