"""Model discovery: search Hugging Face Hub and stage candidates in the registry.

Flow (deliberately conservative — this feeds the checksum-verified downloader
in registry.py, it does not replace it):

  search(query)            -> ranked candidate repos from the HF hub API
  list_weight_files(repo)  -> weight files in a repo with their LFS sha256
  propose(repo, file, ...) -> register a ModelInfo (status not_downloaded)
                              with the resolve URL + published sha256

Nothing is ever downloaded here. A proposed model appears on the Models page
like the pre-seeded ones and downloads only on explicit user action, hashed
and verified against the sha256 the hub published for that exact file. Files
without an LFS sha256 can still be proposed, but the missing checksum stays
visible in the registry (same policy as manual registration).

Only huggingface.co is queried; the resolve URL is constructed, never taken
from response data, so a poisoned search result cannot redirect downloads.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .net import urlopen_verified
from .registry import ModelInfo, ModelRegistry

HF_API = "https://huggingface.co"

# Weight formats worth proposing. Pickle-based formats (.pt/.pth/.ckpt) are
# allowed but flagged in the license notes — prefer safetensors.
WEIGHT_EXTS = (".safetensors", ".gguf", ".onnx", ".pth", ".pt", ".ckpt", ".bin")
PICKLE_EXTS = (".pth", ".pt", ".ckpt", ".bin")


class ModelSearchError(RuntimeError):
    pass


@dataclass
class RepoCandidate:
    repo_id: str
    downloads: int
    likes: int
    pipeline_tag: str | None
    gated: bool


@dataclass
class WeightFile:
    filename: str
    size_bytes: int | None
    sha256: str | None  # from LFS metadata; None for small non-LFS files


def _http_get_json(url: str, timeout_s: float) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "promptforge"})
    try:
        with urlopen_verified(req, timeout=timeout_s) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise ModelSearchError(f"Hugging Face API unreachable: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ModelSearchError(f"Hugging Face API returned invalid JSON: {exc}") from exc


class ModelSearch:
    def __init__(self, registry: ModelRegistry, timeout_s: float = 30.0,
                 http_get: Callable[[str, float], Any] | None = None):
        self.registry = registry
        self.timeout_s = timeout_s
        self._get = http_get or _http_get_json

    # -- discovery --------------------------------------------------------------
    def search(self, query: str, limit: int = 10) -> list[RepoCandidate]:
        q = urllib.parse.urlencode({
            "search": query, "limit": str(limit),
            "sort": "downloads", "direction": "-1", "full": "false",
        })
        raw = self._get(f"{HF_API}/api/models?{q}", self.timeout_s)
        if not isinstance(raw, list):
            raise ModelSearchError("Unexpected search response shape.")
        out = []
        for item in raw:
            if not isinstance(item, dict) or "id" not in item:
                continue
            out.append(RepoCandidate(
                repo_id=item["id"],
                downloads=int(item.get("downloads") or 0),
                likes=int(item.get("likes") or 0),
                pipeline_tag=item.get("pipeline_tag"),
                gated=bool(item.get("gated")),
            ))
        return out

    def list_weight_files(self, repo_id: str) -> list[WeightFile]:
        raw = self._get(f"{HF_API}/api/models/{repo_id}/tree/main?recursive=true",
                        self.timeout_s)
        if not isinstance(raw, list):
            raise ModelSearchError("Unexpected tree response shape.")
        files = []
        for entry in raw:
            if not isinstance(entry, dict) or entry.get("type") != "file":
                continue
            path = entry.get("path", "")
            if not path.lower().endswith(WEIGHT_EXTS):
                continue
            lfs = entry.get("lfs") or {}
            files.append(WeightFile(
                filename=path,
                size_bytes=entry.get("size"),
                sha256=lfs.get("oid"),  # LFS oid IS the file's sha256
            ))
        files.sort(key=lambda f: f.size_bytes or 0, reverse=True)
        return files

    # -- Civitai (second source; sha256 comes from their file metadata) ----------
    def search_civitai(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        """Checkpoint candidates from civitai.com with verified hashes.

        Returns dicts: {name, creator, downloads, filename, size_bytes,
        sha256, url, nsfw} — only safetensors files carrying a SHA256.
        """
        q = urllib.parse.urlencode({
            "query": query, "types": "Checkpoint", "limit": str(limit),
            "sort": "Most Downloaded"})
        raw = self._get(f"https://civitai.com/api/v1/models?{q}", self.timeout_s)
        items = raw.get("items", []) if isinstance(raw, dict) else []
        out: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            stats = item.get("stats") or {}
            for version in (item.get("modelVersions") or [])[:1]:  # latest
                for f in version.get("files") or []:
                    name = str(f.get("name", ""))
                    sha = ((f.get("hashes") or {}).get("SHA256") or "").lower()
                    if not name.lower().endswith(".safetensors") or not sha:
                        continue
                    out.append({
                        "name": item.get("name", name),
                        "creator": (item.get("creator") or {}).get("username", "?"),
                        "downloads": int(stats.get("downloadCount") or 0),
                        "filename": name,
                        "size_bytes": int(float(f.get("sizeKB") or 0) * 1024),
                        "sha256": sha,
                        "url": f.get("downloadUrl")
                               or f"https://civitai.com/api/download/models/{version.get('id')}",
                        "nsfw": bool(item.get("nsfw")),
                    })
        return out

    # Civitai model types the UI can search, mapped to (their API name, the
    # ComfyUI model folder staged files belong in). Workflows are searchable
    # but not stageable — they're graphs, not weights.
    CIVITAI_TYPES: dict[str, tuple[str, str | None]] = {
        "checkpoint": ("Checkpoint", "checkpoints"),
        "lora": ("LORA", "loras"),
        "controlnet": ("Controlnet", "controlnet"),
        "embedding": ("TextualInversion", "embeddings"),
        "vae": ("VAE", "vae"),
        "upscaler": ("Upscaler", "upscale_models"),
        "workflow": ("Workflows", None),
    }

    def search_civitai_rich(self, query: str, type_key: str = "checkpoint",
                            limit: int = 12) -> list[dict[str, Any]]:
        """Full-detail civitai search across model types: preview image,
        description, trigger words, base model, version and download stats.
        Only safetensors files with a published SHA256 are stageable."""
        api_type, folder = self.CIVITAI_TYPES.get(
            type_key, self.CIVITAI_TYPES["checkpoint"])
        params: dict[str, str] = {"types": api_type, "limit": str(limit),
                                  "sort": "Most Downloaded"}
        if query.strip():
            params["query"] = query.strip()
        raw = self._get("https://civitai.com/api/v1/models?"
                        + urllib.parse.urlencode(params), self.timeout_s)
        items = raw.get("items", []) if isinstance(raw, dict) else []
        out: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            stats = item.get("stats") or {}
            versions = item.get("modelVersions") or []
            if not versions:
                continue
            v = versions[0]  # latest
            images = v.get("images") or []
            preview = next((i.get("url") for i in images
                            if isinstance(i, dict) and i.get("url")), None)
            desc = re.sub(r"<[^>]+>", " ", str(item.get("description") or ""))
            desc = re.sub(r"\s+", " ", desc).strip()[:280]
            file_info: dict[str, Any] = {}
            for f in v.get("files") or []:
                fname = str(f.get("name", ""))
                sha = ((f.get("hashes") or {}).get("SHA256") or "").lower()
                if fname.lower().endswith(".safetensors") and sha:
                    file_info = {
                        "filename": fname, "sha256": sha,
                        "size_bytes": int(float(f.get("sizeKB") or 0) * 1024),
                        "url": f.get("downloadUrl")
                               or f"https://civitai.com/api/download/models/{v.get('id')}",
                    }
                    break
            out.append({
                "name": item.get("name", ""), "type": type_key,
                "creator": (item.get("creator") or {}).get("username", "?"),
                "downloads": int(stats.get("downloadCount") or 0),
                "rating": stats.get("thumbsUpCount"),
                "description": desc,
                "trigger_words": [str(w) for w in (v.get("trainedWords") or [])][:8],
                "base_model": v.get("baseModel"),
                "version": v.get("name"),
                "preview_url": preview,
                "nsfw": bool(item.get("nsfw")),
                "folder": folder,
                "stageable": bool(file_info and folder),
                **file_info,
            })
        return out

    def propose_civitai(self, candidate: dict[str, Any], *, name: str,
                        purpose: str) -> ModelInfo:
        """Stage a civitai file in the registry (download stays verified)."""
        if self.registry.get(name) is not None:
            raise ModelSearchError(
                f"A model named '{name}' is already in the registry.")
        meta: dict[str, Any] = {
            "folder": candidate.get("folder") or "checkpoints",
            "source": "civitai",
            "file": candidate["filename"],
            "size_bytes": candidate.get("size_bytes"),
            "proposed_by": "model_search",
        }
        if candidate.get("trigger_words"):
            meta["trigger_words"] = candidate["trigger_words"]
        if candidate.get("base_model"):
            meta["base_model"] = candidate["base_model"]
        model = ModelInfo(
            name=name, purpose=purpose,
            license=(f"From civitai.com ('{candidate['name']}' by "
                     f"{candidate['creator']}) — review the model page's "
                     "license before commercial use."),
            url=candidate["url"], sha256=candidate["sha256"],
            meta=meta,
        )
        self.registry.register(model)
        return model

    # -- staging ----------------------------------------------------------------
    def propose(self, repo_id: str, filename: str, *, name: str,
                purpose: str, vram_gb: float | None = None,
                folder: str = "checkpoints") -> ModelInfo:
        """Register a hub file as a downloadable model (nothing downloads here)."""
        if self.registry.get(name) is not None:
            raise ModelSearchError(
                f"A model named '{name}' is already in the registry.")
        matches = [f for f in self.list_weight_files(repo_id)
                   if f.filename == filename]
        if not matches:
            raise ModelSearchError(
                f"'{filename}' is not a weight file in {repo_id}.")
        wf = matches[0]

        notes = [f"Proposed from huggingface.co/{repo_id} — review the license "
                 "on the repo page before use."]
        if wf.sha256 is None:
            notes.append("No published sha256 (non-LFS file): download "
                         "integrity cannot be verified.")
        if filename.lower().endswith(PICKLE_EXTS):
            notes.append("Pickle-based format — prefer a .safetensors "
                         "variant when one exists.")

        # URL is constructed from the validated repo/file pair, never taken
        # from API response data.
        url = (f"{HF_API}/{urllib.parse.quote(repo_id)}/resolve/main/"
               f"{urllib.parse.quote(filename)}")
        model = ModelInfo(
            name=name, purpose=purpose, license=" ".join(notes),
            url=url, sha256=wf.sha256, vram_gb=vram_gb,
            meta={"repo": repo_id, "file": filename, "folder": folder,
                  "size_bytes": wf.size_bytes, "proposed_by": "model_search"},
        )
        self.registry.register(model)
        return model


class ModelIndex:
    """A periodically-refreshed index of popular models per type, so the app's
    understanding of what's available (and which workflows each model serves)
    stays current instead of relying on static data.

    Entries live in the `model_index` DB table; the health-monitor thread
    calls refresh_stale() in the background, and reads always serve whatever
    is cached (refreshing lazily on first access)."""

    STALE_SECONDS = 6 * 3600

    def __init__(self, db, search: ModelSearch):
        self._db = db
        self._search = search
        db.execute("""
            CREATE TABLE IF NOT EXISTS model_index (
                type TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                fetched_at REAL NOT NULL
            )""")

    def get(self, type_key: str) -> dict[str, Any]:
        rows = self._db.query(
            "SELECT payload, fetched_at FROM model_index WHERE type=?",
            (type_key,))
        if rows and time.time() - rows[0]["fetched_at"] < self.STALE_SECONDS:
            return {"type": type_key, "fetched_at": rows[0]["fetched_at"],
                    "entries": json.loads(rows[0]["payload"])}
        return self.refresh(type_key)

    def refresh(self, type_key: str) -> dict[str, Any]:
        try:
            entries = self._search.search_civitai_rich("", type_key, limit=16)
        except ModelSearchError:
            rows = self._db.query(
                "SELECT payload, fetched_at FROM model_index WHERE type=?",
                (type_key,))
            if rows:  # network down: serve the stale copy rather than nothing
                return {"type": type_key, "fetched_at": rows[0]["fetched_at"],
                        "entries": json.loads(rows[0]["payload"])}
            return {"type": type_key, "fetched_at": 0, "entries": []}
        now = time.time()
        self._db.execute(
            "INSERT INTO model_index (type, payload, fetched_at) VALUES (?,?,?) "
            "ON CONFLICT(type) DO UPDATE SET payload=excluded.payload, "
            "fetched_at=excluded.fetched_at",
            (type_key, json.dumps(entries), now))
        return {"type": type_key, "fetched_at": now, "entries": entries}

    def refresh_stale(self) -> list[str]:
        """Refresh any index entries older than STALE_SECONDS (called from the
        background monitor). Returns the types refreshed."""
        refreshed = []
        for type_key in ModelSearch.CIVITAI_TYPES:
            rows = self._db.query(
                "SELECT fetched_at FROM model_index WHERE type=?", (type_key,))
            if not rows or time.time() - rows[0]["fetched_at"] >= self.STALE_SECONDS:
                self.refresh(type_key)
                refreshed.append(type_key)
        return refreshed
