"""Tiny SQLite layer. Single-file DB, WAL mode, thread-safe via per-call connections.

Deliberately minimal for the MVP; the schema is designed so a later move to
Postgres (or an ORM) is a mechanical change.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,               -- image | video
    filename TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS versions (
    id TEXT PRIMARY KEY,
    asset_id TEXT NOT NULL REFERENCES assets(id),
    label TEXT NOT NULL,              -- original | edit
    path TEXT NOT NULL,
    prompt TEXT,
    adapter TEXT,
    created_at TEXT NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_versions_asset ON versions(asset_id);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    state TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    payload TEXT NOT NULL DEFAULT '{}',
    result TEXT,
    error TEXT,
    logs TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS avatars (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    source_assets TEXT NOT NULL DEFAULT '[]',  -- consented photo asset ids
    frames TEXT NOT NULL DEFAULT '[]',         -- orbit frame asset ids, in angle order
    face_asset TEXT,                           -- best identity reference asset id
    meta TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS models (
    name TEXT PRIMARY KEY,
    purpose TEXT NOT NULL,
    license TEXT NOT NULL DEFAULT 'unknown',
    url TEXT,
    path TEXT,
    sha256 TEXT,
    status TEXT NOT NULL DEFAULT 'not_downloaded',
    vram_gb REAL,
    meta TEXT NOT NULL DEFAULT '{}'
);
"""


class Database:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(sql, params).fetchall()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self.connect() as conn:
            conn.execute(sql, params)
