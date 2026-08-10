"""Tiny SQLite layer. Single-file DB, WAL mode, thread-safe via per-THREAD
connections.

Deliberately minimal for the MVP; the schema is designed so a later move to
Postgres (or an ORM) is a mechanical change.

Connections used to be per-CALL — open, pragma, execute, close, every
operation. Measured (cProfile, Windows): 166 such round-trips inside one
Services construction cost 0.76s of its 1.02s, and every runtime operation
paid ~5ms of open/close for a sub-millisecond query. Each thread now keeps
one connection (SQLite's supported concurrency model under WAL); commit
semantics per operation are unchanged. close() exists because Windows will
not delete an open database file — Services.stop() and direct test users
call it; a generation counter lets any straggler thread reopen safely
instead of crashing on a closed handle.
"""
from __future__ import annotations

import sqlite3
import threading
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
        self._tls = threading.local()
        self._lock = threading.Lock()
        self._open: list[sqlite3.Connection] = []
        self._generation = 0
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def _thread_conn(self) -> sqlite3.Connection:
        """This thread's connection, opened on first use (or after close())."""
        if getattr(self._tls, "generation", None) == self._generation:
            return self._tls.conn
        # check_same_thread=False so close() may close it at shutdown; each
        # connection is still USED by its one owning thread only.
        conn = sqlite3.connect(self.path, timeout=30,
                               check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._open.append(conn)
            self._tls.conn = conn
            self._tls.generation = self._generation
        return conn

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = self._thread_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(sql, params).fetchall()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self.connect() as conn:
            conn.execute(sql, params)

    def close(self) -> None:
        """Close every connection this Database opened, from any thread.

        Windows cannot delete an open database file, so shutdown (and test
        teardown) must come through here. Callers whose worker threads are
        already joined lose nothing; a straggler thread that runs afterwards
        gets a fresh connection via the generation bump rather than a crash
        on a closed handle."""
        with self._lock:
            conns, self._open = self._open, []
            self._generation += 1
        for conn in conns:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 — closing is best-effort
                pass
