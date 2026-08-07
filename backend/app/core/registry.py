"""Model registry + safe downloader.

Registry rows carry: name, purpose, license notes, download URL, local path,
sha256, status, VRAM estimate. Statuses:

  not_downloaded -> downloading -> ready
                              \\-> failed            (network / IO error)
                              \\-> checksum_failed   (hash mismatch; file removed)

Downloads stream to a temp file, hash while streaming, verify, then atomically
move into place — a partial or corrupt download can never masquerade as a
ready model.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .db import Database
from .net import urlopen_verified
from .trust import UntrustedDownloadError, check_format, check_host


class DownloadError(RuntimeError):
    pass


# Extensions ComfyUI's loaders actually list. A downloaded weight file with
# anything else (or nothing) is invisible to them, however complete it is.
_MODEL_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf",
                   ".onnx", ".sft"}


# A real browser UA — some CDNs (Civitai) reject the default urllib agent.
_DOWNLOAD_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/121.0 Safari/537.36")


class _TransientDownload(RuntimeError):
    """A download attempt failed in a way that a retry/resume might fix
    (network hiccup, timeout, 5xx). The partial bytes are kept on disk."""


class _PermanentDownload(RuntimeError):
    """A download attempt failed terminally (bad auth, 404, gated). Retrying
    the same URL will not help; the caller may still try a mirror."""


def _content_range_total(resp: Any) -> int | None:
    """Total size parsed from a 206 response's Content-Range: bytes a-b/total."""
    cr = resp.headers.get("Content-Range") if hasattr(resp, "headers") else None
    if cr and "/" in cr:
        tail = cr.rsplit("/", 1)[-1].strip()
        if tail.isdigit():
            return int(tail)
    return None


@dataclass
class ModelInfo:
    name: str
    purpose: str
    license: str = "unknown"
    url: str | None = None
    path: str | None = None
    sha256: str | None = None
    status: str = "not_downloaded"
    vram_gb: float | None = None
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "purpose": self.purpose, "license": self.license,
            "url": self.url, "path": self.path, "sha256": self.sha256,
            "status": self.status, "vram_gb": self.vram_gb, "meta": self.meta or {},
        }


class ModelRegistry:
    def __init__(self, db: Database, models_dir: Path):
        self._db = db
        self.models_dir = models_dir
        models_dir.mkdir(parents=True, exist_ok=True)
        # Live download telemetry (in-memory, this process only): the Models
        # page polls these so a click visibly *does* something.
        self.progress: dict[str, int] = {}   # name -> percent done
        self.notes: dict[str, str] = {}      # name -> last failure reason

    def register(self, model: ModelInfo) -> None:
        self._db.execute(
            """INSERT INTO models (name, purpose, license, url, path, sha256, status, vram_gb, meta)
               VALUES (?,?,?,?,?,?,?,?,?)
               ON CONFLICT(name) DO UPDATE SET
                 purpose=excluded.purpose, license=excluded.license, url=excluded.url,
                 sha256=excluded.sha256, vram_gb=excluded.vram_gb, meta=excluded.meta""",
            (model.name, model.purpose, model.license, model.url, model.path,
             model.sha256, model.status, model.vram_gb, json.dumps(model.meta or {})),
        )

    def get(self, name: str) -> ModelInfo | None:
        rows = self._db.query("SELECT * FROM models WHERE name=?", (name,))
        return self._row_to_model(rows[0]) if rows else None

    def list(self) -> list[ModelInfo]:
        return [self._row_to_model(r) for r in self._db.query("SELECT * FROM models ORDER BY name")]

    def set_status(self, name: str, status: str, path: str | None = None) -> None:
        if path is not None:
            self._db.execute("UPDATE models SET status=?, path=? WHERE name=?",
                             (status, path, name))
        else:
            self._db.execute("UPDATE models SET status=? WHERE name=?", (status, name))

    def is_ready(self, name: str) -> bool:
        m = self.get(name)
        return bool(m and m.status == "ready" and m.path and Path(m.path).exists())

    def reset_stale(self) -> list[str]:
        """Flip any model left in 'downloading' (a crash mid-download) back to
        'not_downloaded' so its Download button reappears and a fresh attempt
        can resume from the partial .part file. Returns the names reset."""
        rows = self._db.query("SELECT name FROM models WHERE status='downloading'")
        names = [r["name"] for r in rows]
        for name in names:
            self._db.execute(
                "UPDATE models SET status='not_downloaded' WHERE name=?", (name,))
            self.progress.pop(name, None)
        return names

    @staticmethod
    def _row_to_model(r: Any) -> ModelInfo:
        return ModelInfo(
            name=r["name"], purpose=r["purpose"], license=r["license"], url=r["url"],
            path=r["path"], sha256=r["sha256"], status=r["status"],
            vram_gb=r["vram_gb"], meta=json.loads(r["meta"] or "{}"),
        )


ProgressCb = Callable[[int, int | None], None]  # (bytes_done, total_or_None)


class ModelDownloader:
    """Streams a model file to disk with checksum validation."""

    CHUNK = 1 << 16
    # Transient HTTP statuses worth retrying (rate-limit / server hiccups).
    RETRYABLE_HTTP = frozenset({408, 425, 429, 500, 502, 503, 504})
    MAX_ATTEMPTS = 4

    def __init__(self, registry: ModelRegistry, civitai_token: str = "",
                 sleep: Callable[[float], None] = time.sleep):
        self.registry = registry
        self.civitai_token = civitai_token
        # Injectable so tests don't actually wait through the retry backoff.
        self._sleep = sleep
        # Optional LAN source: name -> URL on a discovered PromptForge peer
        # that already holds the file. Wired by Services when peer sharing
        # is on; None means the internet path runs exactly as before.
        self.peer_source: Callable[[str], str | None] | None = None

    def _fetch_url(self, model: ModelInfo) -> str:
        """The URL actually requested. Civitai downloads need an account
        token for many files; it goes in the query (their documented form)
        and never into the registry or error messages."""
        url = model.url or ""
        if "civitai.com" in url and self.civitai_token:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}token={urllib.parse.quote(self.civitai_token)}"
        return url

    def _scrub(self, text: str) -> str:
        return text.replace(self.civitai_token, "***") if self.civitai_token else text

    def download(self, name: str, progress: ProgressCb | None = None) -> ModelInfo:
        model = self.registry.get(name)
        if model is None:
            raise DownloadError(f"Model '{name}' is not in the registry.")
        if not model.url:
            raise DownloadError(f"Model '{name}' has no download URL configured.")
        if self.registry.is_ready(name):
            return model

        # Trust gates: allowlisted hosts only; pickle formats only when the
        # registry entry is explicitly vetted (meta.allow_pickle).
        try:
            check_host(model.url)
            check_format(model.url,
                         bool((model.meta or {}).get("allow_pickle")))
        except UntrustedDownloadError as exc:
            self.registry.set_status(name, "failed")
            raise DownloadError(f"Download of '{name}' blocked: {exc}") from exc

        url_name = Path(urllib.parse.urlparse(model.url).path).name
        # Civitai download URLs end in a bare version id ("456538") with no
        # extension — saved as-is, ComfyUI's extension-filtered loaders would
        # NEVER list the file even though the registry says "ready". The
        # entry's meta.file carries the real filename; always prefer it.
        file_hint = Path(str((model.meta or {}).get("file") or "")).name
        if file_hint:
            url_name = file_hint
        # Backstop for entries with no meta.file at all — the scout registers
        # civitai finds straight from a download URL, and a file saved as
        # "501240" is a multi-gigabyte download the registry calls "ready"
        # while ComfyUI's extension-filtered loader never lists it. Give it
        # the extension its loader needs, named after the registry entry.
        if Path(url_name).suffix.lower() not in _MODEL_SUFFIXES:
            url_name = f"{name}.safetensors"
        # Typed subfolder (checkpoints/diffusion_models/vae/...): ComfyUI maps
        # each category to its own directory, so a segmentation .pth can never
        # show up in the checkpoint list.
        folder = (model.meta or {}).get("folder", "checkpoints")
        dest_dir = self.registry.models_dir / folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / (url_name or f"{name}.bin")
        # Downloads stream to a stable "<file>.part" so an interrupted transfer
        # can be resumed on the next attempt instead of starting over.
        part = dest_dir / ((url_name or name) + ".part")
        # Only HTTP(S) transfers are worth retrying; a missing file:// source
        # is permanent, so we don't sit through the backoff for it.
        is_remote = urllib.parse.urlparse(model.url).scheme in ("http", "https")

        self.registry.set_status(name, "downloading")
        self.registry.progress[name] = 0
        self.registry.notes.pop(name, None)

        digest: hashlib._Hash | None = None
        # A machine on the local network that already holds this exact file
        # beats any internet mirror by an order of magnitude. The peer is
        # NOT trusted — the checksum verification below is what accepts the
        # bytes — so this path exists only for sha-pinned entries, skips
        # the host allowlist for that reason alone, and any failure falls
        # straight through to the normal download.
        if self.peer_source is not None and model.sha256:
            try:
                peer_url = self.peer_source(name)
            except Exception:  # noqa: BLE001 — discovery must never block
                peer_url = None
            if peer_url:
                self.registry.notes[name] = ("Copying from another "
                                             "PromptForge on your network…")
                try:
                    digest = self._attempt(name, replace(model, url=peer_url),
                                           part, progress)
                except Exception as exc:  # noqa: BLE001
                    digest = None
                    self.registry.notes[name] = (
                        f"Network copy failed ({str(exc)[:80]}); "
                        "downloading from the internet instead")
        if digest is None:
            for attempt in range(1, self.MAX_ATTEMPTS + 1):
                try:
                    digest = self._attempt(name, model, part, progress)
                    break
                except _PermanentDownload as exc:
                    part.unlink(missing_ok=True)
                    raise self._fail(name, model, str(exc)) from exc
                except _TransientDownload as exc:
                    if not is_remote or attempt == self.MAX_ATTEMPTS:
                        # Keep the .part on a remote give-up so a future
                        # run resumes.
                        if not is_remote:
                            part.unlink(missing_ok=True)
                        raise self._fail(
                            name, model,
                            f"failed after {attempt} attempt(s): {exc}",
                            keep_part=is_remote) from exc
                    wait = min(2 ** attempt, 30)
                    self.registry.notes[name] = (
                        f"Attempt {attempt}/{self.MAX_ATTEMPTS} failed "
                        f"({exc}); resuming in {wait}s…")
                    self._sleep(wait)

        assert digest is not None
        if model.sha256:
            actual = digest.hexdigest()
            if actual.lower() != model.sha256.lower():
                part.unlink(missing_ok=True)
                self.registry.set_status(name, "checksum_failed")
                self.registry.progress.pop(name, None)
                message = (
                    f"Checksum mismatch for '{name}': expected {model.sha256}, got {actual}. "
                    "The file was discarded."
                )
                self.registry.notes[name] = message
                raise DownloadError(message)
        # else: no checksum published — recorded as ready but flagged in meta by caller/UI

        shutil.move(str(part), dest)
        self.registry.set_status(name, "ready", path=str(dest))
        self.registry.progress.pop(name, None)
        self.registry.notes.pop(name, None)
        refreshed = self.registry.get(name)
        assert refreshed is not None
        return refreshed

    def _fail(self, name: str, model: ModelInfo, detail: str,
              keep_part: bool = False) -> DownloadError:
        """Record a terminal download failure and build its DownloadError.
        With keep_part=True the partial bytes survive for a later resume."""
        self.registry.set_status(name, "failed")
        self.registry.progress.pop(name, None)
        message = self._scrub(f"Download of '{name}' {detail}")
        if ("civitai.com" in (model.url or "")
                and ("401" in message or "403" in message)):
            message += (". Civitai requires a (free) account API token "
                        "for this file: create one under civitai.com → "
                        "Account settings → API Keys and set "
                        "PROMPTFORGE_CIVITAI_TOKEN before launching.")
        if keep_part:
            message += " (partial download kept — retrying will resume it)."
        self.registry.notes[name] = message
        return DownloadError(message)

    def _attempt(self, name: str, model: ModelInfo, part: Path,
                 progress: ProgressCb | None) -> hashlib._Hash:
        """A single download attempt that resumes from `part` when bytes are
        already there. Returns the full-file sha256 digest, or raises
        _TransientDownload / _PermanentDownload."""
        start = part.stat().st_size if part.exists() else 0
        digest = hashlib.sha256()
        if start:
            # Re-hash bytes already on disk so a resumed stream still yields the
            # digest of the whole file.
            with open(part, "rb") as fh:
                for chunk in iter(lambda: fh.read(self.CHUNK), b""):
                    digest.update(chunk)
        # A browser-like User-Agent is required by some CDNs (notably Civitai,
        # which 403s the default "Python-urllib" agent even with a valid token).
        headers = {"User-Agent": _DOWNLOAD_UA}
        if start:
            headers["Range"] = f"bytes={start}-"
        req = urllib.request.Request(self._fetch_url(model), headers=headers)
        try:
            with urlopen_verified(req, timeout=60) as resp:
                code = getattr(resp, "status", 200) or 200
                if start and code != 206:
                    # Server ignored the Range request and is sending the whole
                    # file again — restart cleanly rather than corrupt the .part.
                    start, digest = 0, hashlib.sha256()
                clen = resp.headers.get("Content-Length")
                total = _content_range_total(resp) or (
                    (start + int(clen)) if clen else None)
                done = start
                with open(part, "ab" if start else "wb") as fh:
                    while True:
                        chunk = resp.read(self.CHUNK)
                        if not chunk:
                            break
                        fh.write(chunk)
                        digest.update(chunk)
                        done += len(chunk)
                        if total:
                            self.registry.progress[name] = min(99, done * 100 // total)
                        if progress:
                            progress(done, total)
        except urllib.error.HTTPError as exc:
            if exc.code in self.RETRYABLE_HTTP:
                raise _TransientDownload(f"HTTP {exc.code}") from exc
            raise _PermanentDownload(str(exc)) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                OSError, ValueError) as exc:
            # Missing local file (file://) surfaces here too; the caller decides
            # it is permanent because the scheme isn't retryable.
            raise _TransientDownload(str(exc)) from exc
        return digest
