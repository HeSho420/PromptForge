"""Asset store: uploaded media + edit versions on the local filesystem,
metadata in SQLite. Every edit creates a new version row; the original file is
never overwritten, which gives before/after for free.
"""
from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import ALLOWED_IMAGE_EXTS, ALLOWED_MODEL_EXTS, ALLOWED_VIDEO_EXTS, Settings
from .db import Database


class UnsupportedFormatError(ValueError):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _safe_name(filename: str) -> str:
    name = Path(filename).name
    return re.sub(r"[^A-Za-z0-9._-]", "_", name) or "upload"


@dataclass
class Asset:
    id: str
    kind: str
    filename: str
    path: str
    created_at: str
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "filename": self.filename,
                "created_at": self.created_at, "meta": self.meta}


@dataclass
class Version:
    id: str
    asset_id: str
    label: str
    path: str
    prompt: str | None
    adapter: str | None
    created_at: str
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "asset_id": self.asset_id, "label": self.label,
                "prompt": self.prompt, "adapter": self.adapter,
                "created_at": self.created_at, "meta": self.meta}


@dataclass
class AvatarProfile:
    """A consented digital-human profile: source photos, orbit frames (for
    the movable viewer) and the face reference used for identity renders."""
    id: str
    name: str
    created_at: str
    source_assets: list[str]
    frames: list[str]
    face_asset: str | None
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "created_at": self.created_at,
                "source_assets": self.source_assets, "frames": self.frames,
                "face_asset": self.face_asset, "meta": self.meta}


class AssetStore:
    def __init__(self, db: Database, settings: Settings):
        self._db = db
        self._settings = settings
        settings.ensure_dirs()

    # -- assets ---------------------------------------------------------------
    def save_upload(self, filename: str, data: bytes,
                    meta: dict[str, Any] | None = None,
                    limit_mb: int | None = None) -> Asset:
        """`limit_mb` overrides the size cap for INTERNALLY generated files:
        a rigged avatar mesh carries a texture atlas plus per-vertex
        skinning data and legitimately exceeds a cap sized for user photo
        uploads. Callers pass it explicitly; HTTP requests never can."""
        ext = Path(filename).suffix.lower()
        if ext in ALLOWED_IMAGE_EXTS:
            kind = "image"
        elif ext in ALLOWED_VIDEO_EXTS:
            kind = "video"
        elif ext in ALLOWED_MODEL_EXTS:
            kind = "model"
        else:
            raise UnsupportedFormatError(
                f"Unsupported file type '{ext or 'none'}'. "
                f"Images: {', '.join(sorted(ALLOWED_IMAGE_EXTS))}. "
                f"Video: {', '.join(sorted(ALLOWED_VIDEO_EXTS))}. "
                f"3D: {', '.join(sorted(ALLOWED_MODEL_EXTS))}.")
        cap = limit_mb or self._settings.max_upload_mb
        max_bytes = cap * 1024 * 1024
        if len(data) > max_bytes:
            raise UnsupportedFormatError(
                f"File is larger than the {cap} MB upload limit.")

        asset_id = uuid.uuid4().hex[:12]
        folder = self._settings.assets_dir / asset_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / _safe_name(filename)
        path.write_bytes(data)

        extra: dict[str, Any] = {}
        if kind == "video":
            # Decode it NOW rather than at render time. A clip that cannot be
            # read is a broken upload, and finding that out three minutes into
            # a motion transfer — after the models have loaded — is the worst
            # possible moment. Also gives the gallery something to show.
            try:
                extra = self._describe_video(path, folder)
            except UnsupportedFormatError:
                shutil.rmtree(folder, ignore_errors=True)
                raise
        elif kind == "image":
            # Same principle the video path already applies: the extension
            # only names the format, it does not prove the bytes ARE one.
            # A renamed .txt, a truncated download or a corrupt PNG was
            # accepted here and stored as a first-class image (measured
            # live: 12 bytes of "not an image" → HTTP 201), then failed
            # deep in masking or rendering with a cryptic PIL error. Decode
            # it once, now, and reject a broken upload with a plain message.
            try:
                extra = self._describe_image(path)
            except UnsupportedFormatError:
                shutil.rmtree(folder, ignore_errors=True)
                raise

        asset = Asset(asset_id, kind, path.name, str(path), _now(),
                      {"bytes": len(data), **extra, **(meta or {})})
        self._db.execute(
            "INSERT INTO assets (id, kind, filename, path, created_at, meta) VALUES (?,?,?,?,?,?)",
            (asset.id, asset.kind, asset.filename, asset.path, asset.created_at,
             json.dumps(asset.meta)))
        self._add_version(asset.id, "original", str(path), None, None, {})
        return asset

    def _describe_video(self, path: Path, folder: Path) -> dict[str, Any]:
        """Probe an uploaded clip and write a poster frame beside it.

        Raises UnsupportedFormatError (the caller deletes the upload) when the
        file cannot be decoded or is longer than the pipeline will accept — an
        honest rejection at upload time beats a confusing failure later."""
        from . import video as video_io
        try:
            info = video_io.probe(path)
        except video_io.VideoError as exc:
            raise UnsupportedFormatError(
                f"That video could not be read. {exc}") from exc
        limit = self._settings.max_video_seconds
        if info.duration_s > limit:
            raise UnsupportedFormatError(
                f"That clip is {info.duration_s:.0f} seconds long; the limit "
                f"is {limit}. Trim it and upload again.")
        out: dict[str, Any] = {"video": info.to_dict()}
        try:
            poster = folder / "poster.png"
            video_io.thumbnail(path).save(poster, format="PNG")
            out["poster"] = poster.name
        except Exception:  # noqa: BLE001 — a missing poster is cosmetic
            pass
        return out

    def _describe_image(self, path: Path) -> dict[str, Any]:
        """Decode an uploaded image to prove it is one, and record its size.

        Two opens on purpose: verify() checks structural integrity but
        leaves the file unusable afterwards and does NOT catch a truncated
        body, so a real load() follows to force every pixel through the
        decoder. Either failure is a broken upload; the caller deletes it.
        The width/height also give the gallery and the pipeline the
        dimensions without a second open later."""
        from PIL import Image, UnidentifiedImageError
        try:
            with Image.open(path) as probe:
                probe.verify()
            with Image.open(path) as full:
                full.load()
                width, height = full.size
        except (UnidentifiedImageError, OSError, SyntaxError,
                ValueError) as exc:
            # The PIL message embeds the absolute on-disk path of the
            # upload; that is server-internal and does not belong in a
            # client-facing error. Keep the reason type, drop the path.
            reason = type(exc).__name__
            raise UnsupportedFormatError(
                "That image could not be read — it may be corrupt, "
                f"truncated, or not really an image ({reason}).") from exc
        return {"width": width, "height": height}

    def get_asset(self, asset_id: str) -> Asset | None:
        rows = self._db.query("SELECT * FROM assets WHERE id=?", (asset_id,))
        if not rows:
            return None
        r = rows[0]
        return Asset(r["id"], r["kind"], r["filename"], r["path"], r["created_at"],
                     json.loads(r["meta"]))

    def list_assets(self, include_deleted: bool = False) -> list[Asset]:
        assets = [Asset(r["id"], r["kind"], r["filename"], r["path"], r["created_at"],
                        json.loads(r["meta"]))
                  for r in self._db.query("SELECT * FROM assets ORDER BY created_at DESC")]
        if include_deleted:
            return assets
        return [a for a in assets if not a.meta.get("deleted")]

    # -- deletion (soft, undoable, with disk cleanup) ---------------------------
    @property
    def _trash_dir(self) -> Path:
        d = self._settings.data_dir / "trash"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _set_deleted(self, asset_id: str, deleted: bool) -> None:
        asset = self.get_asset(asset_id)
        if asset is None:
            return
        meta = dict(asset.meta)
        if deleted:
            meta["deleted"] = True
        else:
            meta.pop("deleted", None)
        self._db.execute("UPDATE assets SET meta=? WHERE id=?",
                         (json.dumps(meta), asset_id))

    def delete_asset(self, asset_id: str) -> bool:
        """Soft-delete: the asset folder moves to the trash dir and the asset
        disappears from the gallery. restore_asset() undoes it; purge_asset()
        (or the startup purge) reclaims the disk space for real."""
        asset = self.get_asset(asset_id)
        if asset is None or asset.meta.get("deleted"):
            return False
        folder = self._settings.assets_dir / asset_id
        if folder.exists():
            import shutil
            shutil.move(str(folder), str(self._trash_dir / asset_id))
        self._set_deleted(asset_id, True)
        return True

    def restore_asset(self, asset_id: str) -> bool:
        asset = self.get_asset(asset_id)
        if asset is None or not asset.meta.get("deleted"):
            return False
        trashed = self._trash_dir / asset_id
        if trashed.exists():
            import shutil
            shutil.move(str(trashed), str(self._settings.assets_dir / asset_id))
        self._set_deleted(asset_id, False)
        return True

    def purge_asset(self, asset_id: str) -> bool:
        """Hard-delete a trashed asset: files gone, DB rows gone."""
        asset = self.get_asset(asset_id)
        if asset is None or not asset.meta.get("deleted"):
            return False
        import shutil
        shutil.rmtree(self._trash_dir / asset_id, ignore_errors=True)
        self._db.execute("DELETE FROM versions WHERE asset_id=?", (asset_id,))
        self._db.execute("DELETE FROM assets WHERE id=?", (asset_id,))
        return True

    def purge_trash(self) -> int:
        """Reclaim disk for every soft-deleted asset (called at startup so
        trash never accumulates across sessions). Returns the purge count."""
        count = 0
        for asset in self.list_assets(include_deleted=True):
            if asset.meta.get("deleted") and self.purge_asset(asset.id):
                count += 1
        return count

    # -- versions -------------------------------------------------------------
    def new_version_path(self, asset_id: str, suffix: str = ".png") -> Path:
        folder = self._settings.assets_dir / asset_id
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"edit_{uuid.uuid4().hex[:8]}{suffix}"

    def add_edit_version(self, asset_id: str, path: str, prompt: str,
                         adapter: str, meta: dict[str, Any] | None = None) -> Version:
        return self._add_version(asset_id, "edit", path, prompt, adapter, meta or {})

    def add_aux_version(self, asset_id: str, path: str, note: str,
                        adapter: str, meta: dict[str, Any] | None = None) -> Version:
        """Auxiliary artifacts (e.g. a refined mask): stored and fetchable via
        the versions endpoint but labeled 'mask' so gallery edit lists —
        which filter on label 'edit' — never show them as results."""
        return self._add_version(asset_id, "mask", path, note, adapter, meta or {})

    def _add_version(self, asset_id: str, label: str, path: str,
                     prompt: str | None, adapter: str | None,
                     meta: dict[str, Any]) -> Version:
        v = Version(uuid.uuid4().hex[:12], asset_id, label, path, prompt,
                    adapter, _now(), meta)
        self._db.execute(
            """INSERT INTO versions (id, asset_id, label, path, prompt, adapter, created_at, meta)
               VALUES (?,?,?,?,?,?,?,?)""",
            (v.id, v.asset_id, v.label, v.path, v.prompt, v.adapter, v.created_at,
             json.dumps(v.meta)))
        return v

    def versions(self, asset_id: str) -> list[Version]:
        return [Version(r["id"], r["asset_id"], r["label"], r["path"], r["prompt"],
                        r["adapter"], r["created_at"], json.loads(r["meta"]))
                for r in self._db.query(
                    "SELECT * FROM versions WHERE asset_id=? ORDER BY created_at",
                    (asset_id,))]

    def get_version(self, version_id: str) -> Version | None:
        rows = self._db.query("SELECT * FROM versions WHERE id=?", (version_id,))
        if not rows:
            return None
        r = rows[0]
        return Version(r["id"], r["asset_id"], r["label"], r["path"], r["prompt"],
                       r["adapter"], r["created_at"], json.loads(r["meta"]))

    def promote_version(self, version_id: str) -> Asset | None:
        """Make a version the asset's WORKING file: follow-up edits, masks
        and video renders then build on that result instead of whatever the
        asset pointed at before. Nothing is lost — every render stays in the
        version history and the original upload is itself a version (label
        'original'), so promoting is always reversible. Mask versions are
        auxiliary artifacts, never promotable."""
        v = self.get_version(version_id)
        if v is None or v.label == "mask":
            return None
        asset = self.get_asset(v.asset_id)
        if asset is None or asset.meta.get("deleted") or not Path(v.path).exists():
            return None
        self._db.execute("UPDATE assets SET path=? WHERE id=?", (v.path, asset.id))
        asset.path = v.path
        return asset

    # -- avatars ----------------------------------------------------------------
    def create_avatar(self, name: str, source_assets: list[str],
                      frames: list[str], face_asset: str | None,
                      meta: dict[str, Any] | None = None) -> AvatarProfile:
        profile = AvatarProfile(uuid.uuid4().hex[:12], name, _now(),
                                source_assets, frames, face_asset, meta or {})
        self._db.execute(
            """INSERT INTO avatars (id, name, created_at, source_assets,
                                    frames, face_asset, meta)
               VALUES (?,?,?,?,?,?,?)""",
            (profile.id, profile.name, profile.created_at,
             json.dumps(profile.source_assets), json.dumps(profile.frames),
             profile.face_asset, json.dumps(profile.meta)))
        return profile

    def get_avatar(self, avatar_id: str) -> AvatarProfile | None:
        rows = self._db.query("SELECT * FROM avatars WHERE id=?", (avatar_id,))
        return self._row_to_avatar(rows[0]) if rows else None

    def list_avatars(self) -> list[AvatarProfile]:
        return [self._row_to_avatar(r) for r in self._db.query(
            "SELECT * FROM avatars ORDER BY created_at DESC")]

    def delete_avatar(self, avatar_id: str) -> AvatarProfile | None:
        """Remove an avatar profile. Returns the deleted profile (so callers
        can also clean up its synthetic frames); None if it didn't exist.
        Source photos are ordinary gallery assets and are never touched."""
        profile = self.get_avatar(avatar_id)
        if profile is None:
            return None
        self._db.execute("DELETE FROM avatars WHERE id=?", (avatar_id,))
        return profile

    @staticmethod
    def _row_to_avatar(r: Any) -> AvatarProfile:
        return AvatarProfile(
            r["id"], r["name"], r["created_at"],
            json.loads(r["source_assets"] or "[]"),
            json.loads(r["frames"] or "[]"),
            r["face_asset"], json.loads(r["meta"] or "{}"))

    def gallery(self) -> list[dict[str, Any]]:
        """Assets joined with their versions, newest first — powers the history view."""
        out = []
        for asset in self.list_assets():
            if not Path(asset.path).exists():
                continue  # DB row survived a cleanup but the file is gone
            out.append({"asset": asset.to_dict(),
                        "versions": [v.to_dict() for v in self.versions(asset.id)]})
        return out
