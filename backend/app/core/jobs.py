"""In-process job queue with a real state machine.

MVP implementation: a worker thread and a queue.Queue. The public surface
(enqueue / get / cancel / retry / handler registry) is deliberately small and
broker-shaped so it can be swapped for Redis + Celery/RQ or BullMQ later
without touching call sites (see ROADMAP).

States: pending -> running -> completed
                         \\-> retrying -> running ...
                         \\-> failed
        pending|retrying -> cancelled

Handlers signal error kinds by raising:
  TransientError  -> retried with exponential backoff up to max_retries
  PermanentError  -> failed immediately (bad input, safety, missing file...)
  anything else   -> treated as transient (crash-safety), retried, then failed
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import traceback
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from .db import Database


class TransientError(RuntimeError):
    """Worth retrying: OOM after cache clear, backend briefly unreachable, ..."""


class PermanentError(RuntimeError):
    """Not worth retrying: invalid input, unsupported format, safety block, ..."""


class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    RETRYING = "retrying"
    FAILED = "failed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


@dataclass
class Job:
    id: str
    type: str
    payload: dict[str, Any]
    state: JobState = JobState.PENDING
    attempts: int = 0
    result: dict[str, Any] | None = None
    error: str | None = None
    logs: list[dict[str, str]] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    cancel_requested: bool = False

    def log(self, level: str, message: str) -> None:
        self.logs.append({"t": _now(), "level": level, "msg": message})
        self.updated_at = _now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "type": self.type, "state": self.state.value,
            "attempts": self.attempts, "payload": self.payload,
            "result": self.result, "error": self.error, "logs": self.logs,
            "created_at": self.created_at, "updated_at": self.updated_at,
        }


Handler = Callable[[Job], dict[str, Any]]


class JobQueue:
    HISTORY = 100  # finished jobs loaded back after a restart

    def __init__(self, db: Database, *, max_retries: int = 3, backoff_s: float = 0.5):
        self._db = db
        # Pending dispatch order lives in a deque (not queue.Queue) so jobs
        # can be reordered, removed and the whole queue paused/resumed.
        self._pending: deque[str] = deque()
        self._jobs: dict[str, Job] = {}
        self._handlers: dict[str, Handler] = {}
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._paused = False
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._helper: threading.Thread | None = None
        self._load_history()

    def _load_history(self) -> None:
        """Rehydrate recent job history so a restart doesn't wipe the Queue
        page. Jobs that were mid-flight when the process died are marked
        failed — their worker thread is gone."""
        try:
            rows = self._db.query(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (self.HISTORY,))
        except Exception:  # noqa: BLE001 — history is a convenience, not critical
            return
        for r in rows:
            try:
                job = Job(
                    id=r["id"], type=r["type"], payload=json.loads(r["payload"]),
                    state=JobState(r["state"]), attempts=r["attempts"],
                    result=json.loads(r["result"]) if r["result"] else None,
                    error=r["error"], logs=json.loads(r["logs"] or "[]"),
                    created_at=r["created_at"], updated_at=r["updated_at"])
            except (ValueError, KeyError, json.JSONDecodeError):
                continue
            if job.state in (JobState.PENDING, JobState.RUNNING, JobState.RETRYING):
                job.state = JobState.FAILED
                job.error = "Interrupted by a server restart — use Retry to re-queue."
                job.log("error", job.error)
                self._persist(job)
            self._jobs[job.id] = job

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._run, name="pf-worker", daemon=True)
        self._worker.start()

    def start_helper(self, gate, wrap, types: set[str]) -> None:
        """A SECOND worker that exists to hand work to an idle network peer.

        It takes a pending job only when the main worker is already busy,
        the job's type is delegatable, and `gate()` — a network probe, so
        never called under the queue lock — confirms a peer can carry it.
        The job then runs through `wrap(execute, job)`, which binds its
        render traffic to the peer and falls back to local on failure."""
        helper = getattr(self, "_helper", None)
        if helper is not None and helper.is_alive():
            return
        self._helper_gate = gate
        self._helper_wrap = wrap
        self._helper_types = set(types)
        self._helper = threading.Thread(target=self._run_helper,
                                        name="pf-worker-peer", daemon=True)
        self._helper.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()  # wake the worker
        if self._worker:
            self._worker.join(timeout=timeout)
        helper = getattr(self, "_helper", None)
        if helper is not None:
            helper.join(timeout=timeout)

    def busy(self) -> bool:
        """Anything running or waiting? The peer service answers delegation
        offers with this — an idle machine is one whose queue is empty."""
        with self._lock:
            return bool(self._pending) or any(
                j.state is JobState.RUNNING for j in self._jobs.values())

    def register(self, job_type: str, handler: Handler) -> None:
        self._handlers[job_type] = handler

    # -- public API ----------------------------------------------------------
    def enqueue(self, job_type: str, payload: dict[str, Any]) -> Job:
        if job_type not in self._handlers:
            raise ValueError(f"No handler registered for job type '{job_type}'")
        job = Job(id=uuid.uuid4().hex[:12], type=job_type, payload=payload)
        job.log("info", f"Job created ({job_type})")
        with self._cv:
            self._jobs[job.id] = job
            self._pending.append(job.id)
            self._cv.notify_all()
        self._persist(job)
        return job

    # -- queue management ------------------------------------------------------
    @property
    def paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        """Stop dispatching new jobs; the running job (if any) continues."""
        with self._cv:
            self._paused = True

    def resume(self) -> None:
        with self._cv:
            self._paused = False
            self._cv.notify_all()

    def pending_order(self) -> list[str]:
        with self._lock:
            return list(self._pending)

    def move(self, job_id: str, to: str) -> bool:
        """Reorder a pending job: 'up', 'down' or 'top'."""
        with self._cv:
            try:
                idx = list(self._pending).index(job_id)
            except ValueError:
                return False
            items = list(self._pending)
            items.pop(idx)
            if to == "top":
                new = 0
            elif to == "up":
                new = max(0, idx - 1)
            elif to == "down":
                new = min(len(items), idx + 1)
            else:
                return False
            items.insert(new, job_id)
            self._pending.clear()
            self._pending.extend(items)
        return True

    def delete(self, job_id: str) -> bool:
        """Remove a job (and its history row). Running jobs can't be deleted —
        cancel first. Pending jobs are pulled out of the dispatch order."""
        with self._cv:
            job = self._jobs.get(job_id)
            if job is None or job.state is JobState.RUNNING:
                return False
            try:
                self._pending.remove(job_id)
            except ValueError:
                pass
            self._jobs.pop(job_id, None)
        self._db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        return True

    def clear(self, scope: str) -> int:
        """Bulk-delete jobs by scope: 'completed', 'failed', 'cancelled',
        'finished' (all three) or 'pending'. Never touches the running job."""
        scopes = {
            "completed": {JobState.COMPLETED},
            "failed": {JobState.FAILED},
            "cancelled": {JobState.CANCELLED},
            "finished": {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED},
            "pending": {JobState.PENDING},
        }
        wanted = scopes.get(scope)
        if wanted is None:
            return 0
        with self._cv:
            victims = [j.id for j in self._jobs.values() if j.state in wanted]
            for jid in victims:
                try:
                    self._pending.remove(jid)
                except ValueError:
                    pass
                self._jobs.pop(jid, None)
        # Sweep the DB by STATE, not by the in-memory ids: only the newest
        # HISTORY rows are rehydrated, so id-based deletes would leave older
        # rows (and the prompts inside their payloads) in SQLite forever.
        names = tuple(s.value for s in wanted)
        rows = self._db.query(
            f"SELECT COUNT(*) AS n FROM jobs WHERE state IN "
            f"({','.join('?' * len(names))})", names)
        db_count = int(rows[0]["n"]) if rows else 0
        self._db.execute(
            f"DELETE FROM jobs WHERE state IN ({','.join('?' * len(names))})",
            names)
        return max(len(victims), db_count)

    def clear_logs(self) -> int:
        """Wipe the log LINES of every job that is not currently running;
        the job records (prompt, state, result) stay. The Behind-the-Scenes
        stream merges these per-job lines with the system event log, so
        clearing that stream must strip both. Sweeps the whole SQLite table —
        rows older than the in-memory rehydration window included. Returns
        how many in-memory jobs were stripped."""
        with self._cv:
            victims = [j for j in self._jobs.values()
                       if j.state is not JobState.RUNNING and j.logs]
            for j in victims:
                j.logs = []
            running = tuple(j.id for j in self._jobs.values()
                            if j.state is JobState.RUNNING)
        if running:
            self._db.execute(
                f"UPDATE jobs SET logs='[]' WHERE id NOT IN "
                f"({','.join('?' * len(running))})", running)
        else:
            self._db.execute("UPDATE jobs SET logs='[]'")
        return len(victims)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            if job.state in (JobState.PENDING, JobState.RETRYING):
                job.state = JobState.CANCELLED
                job.log("info", "Cancelled before execution")
                self._persist(job)
                return True
            if job.state is JobState.RUNNING:
                job.cancel_requested = True  # cooperative: handlers may check
                job.log("info", "Cancellation requested (job is running)")
                return True
            return False

    def retry(self, job_id: str) -> bool:
        """Manual re-run of a failed/cancelled job."""
        with self._cv:
            job = self._jobs.get(job_id)
            if not job or job.state not in (JobState.FAILED, JobState.CANCELLED):
                return False
            job.state = JobState.PENDING
            job.error = None
            job.cancel_requested = False
            job.log("info", "Manually re-queued")
            self._persist(job)
            self._pending.append(job_id)
            self._cv.notify_all()
        return True

    # -- worker --------------------------------------------------------------
    @staticmethod
    def _forced_peer(job: Job) -> str | None:
        """The device the user picked by hand, when it is another machine."""
        device = (job.payload or {}).get("device")
        return device if device and device not in ("auto", "local") else None

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._stop.is_set() and (self._paused or not self._pending):
                    self._cv.wait(timeout=0.5)
                if self._stop.is_set():
                    return
                job_id = self._pick_main()
                if job_id is None:
                    # Only hand-targeted jobs are waiting; the peer worker
                    # owns those while it is alive.
                    self._cv.wait(timeout=1.0)
                    continue
            job = self.get(job_id)
            if job is None or job.state is JobState.CANCELLED:
                continue
            self._execute(job)

    def _pick_main(self) -> str | None:
        """The next job for the MAIN worker: everything except jobs the
        user pinned to another machine (those belong to the peer worker,
        which also handles their fall-back-to-local when the peer is
        gone). If the peer worker is not running, the main worker takes
        them anyway rather than let them starve."""
        helper_alive = self._helper is not None and self._helper.is_alive()
        for jid in list(self._pending):
            job = self._jobs.get(jid)
            if job is None:
                self._pending.remove(jid)
                continue
            if helper_alive and self._forced_peer(job):
                continue
            self._pending.remove(jid)
            return jid
        return None

    def _run_helper(self) -> None:
        while not self._stop.is_set():
            candidate: str | None = None
            forced = False
            with self._lock:
                running = any(j.state is JobState.RUNNING
                              for j in self._jobs.values())
                for jid in self._pending:
                    j = self._jobs.get(jid)
                    if j is None or j.type not in self._helper_types:
                        continue
                    if (j.payload or {}).get("device") == "local":
                        continue      # pinned to this machine by hand
                    if self._forced_peer(j):
                        candidate, forced = jid, True
                        break
                    if running and not self._paused:
                        candidate = jid
                        break
            if candidate is None:
                self._stop.wait(1.5)
                continue
            # Hand-targeted jobs go straight to the wrap — it resolves the
            # chosen machine itself and falls back to local when it is
            # gone. Automatic delegation still asks the gate first. The
            # network probe runs OUTSIDE every lock: it can take seconds
            # and the main worker must never wait on it.
            if not forced:
                try:
                    ready = self._helper_gate()
                except Exception:  # noqa: BLE001 — a broken probe means "no"
                    ready = False
                if not ready:
                    self._stop.wait(4.0)
                    continue
            with self._cv:
                try:
                    self._pending.remove(candidate)
                except ValueError:
                    continue      # the main worker got there first — fine
            job = self.get(candidate)
            if job is None or job.state is JobState.CANCELLED:
                continue
            self._helper_wrap(self._execute, job)

    def _execute(self, job: Job) -> None:
        handler = self._handlers[job.type]
        while True:
            job.state = JobState.RUNNING
            job.attempts += 1
            job.log("info", f"Attempt {job.attempts} started")
            self._persist(job)
            try:
                result = handler(job)
                job.result = result
                job.log("info", "Completed")
                self._finish(job, JobState.COMPLETED)
                return
            except PermanentError as exc:
                job.error = str(exc)
                job.log("error", f"Permanent failure: {exc}")
                self._finish(job, JobState.FAILED)
                return
            except Exception as exc:  # TransientError and unexpected crashes
                kind = "transient" if isinstance(exc, TransientError) else "unexpected"
                job.log("error", f"{kind} error: {exc}")
                if kind == "unexpected":
                    job.log("debug", traceback.format_exc(limit=5))
                if job.cancel_requested:
                    job.log("info", "Cancelled during execution")
                    self._finish(job, JobState.CANCELLED)
                    return
                if job.attempts > self._max_retries:
                    job.error = str(exc)
                    job.log("error", f"Failed after {job.attempts} attempts")
                    self._finish(job, JobState.FAILED)
                    return
                job.state = JobState.RETRYING
                delay = self._backoff_s * (2 ** (job.attempts - 1))
                job.log("info", f"Retrying in {delay:.2f}s")
                self._persist(job)
                if self._stop.wait(delay):
                    return

    def _finish(self, job: Job, state: JobState) -> None:
        """Persist the terminal row BEFORE publishing the in-memory state.

        wait_for/get watch the in-memory state without a DB read; writing the
        row first means anyone who observes a terminal state can rely on the
        DB already agreeing (no completed-in-memory/running-on-disk window).
        """
        self._persist(job, state=state)
        job.state = state

    # -- persistence (job history survives restarts) --------------------------
    def _persist(self, job: Job, state: JobState | None = None) -> None:
        try:
            self._db.execute(
                """INSERT INTO jobs (id, type, state, attempts, payload, result,
                                     error, logs, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     state=excluded.state, attempts=excluded.attempts,
                     result=excluded.result, error=excluded.error,
                     logs=excluded.logs, updated_at=excluded.updated_at""",
                (job.id, job.type, (state or job.state).value, job.attempts,
                 json.dumps(job.payload),
                 json.dumps(job.result) if job.result else None,
                 job.error, json.dumps(job.logs), job.created_at, job.updated_at),
            )
        except sqlite3.OperationalError:
            # The data directory can vanish under a worker mid-shutdown (or in
            # test teardown). Job history persistence is best-effort — losing
            # one late write must never crash the worker thread.
            pass

    def recent_durations(self, job_type: str, limit: int = 20) -> list[float]:
        """Wall-clock RENDER seconds of the most recent COMPLETED jobs of a
        type — used to estimate expected render time from real history.

        Measured from when the job STARTED running (the "Attempt N started"
        log line), not from enqueue, so queue-wait behind a backlog never
        inflates the estimate. Falls back to created_at if no start marker."""
        try:
            rows = self._db.query(
                "SELECT created_at, updated_at, logs FROM jobs WHERE type=? AND "
                "state='completed' ORDER BY created_at DESC LIMIT ?",
                (job_type, limit))
        except Exception:  # noqa: BLE001 — estimation is best-effort
            return []
        out: list[float] = []
        for r in rows:
            try:
                t1 = datetime.fromisoformat(r["updated_at"])
                start_iso = r["created_at"]
                for entry in json.loads(r["logs"] or "[]"):
                    if "started" in entry.get("msg", ""):
                        start_iso = entry["t"]
                        break
                t0 = datetime.fromisoformat(start_iso)
            except (ValueError, TypeError, json.JSONDecodeError, KeyError):
                continue
            secs = (t1 - t0).total_seconds()
            if secs > 0:
                out.append(secs)
        return out

    def wait_for(self, job_id: str, timeout: float = 10.0) -> Job:
        """Test/CLI helper: block until the job reaches a terminal state."""
        deadline = time.monotonic() + timeout
        terminal = {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}
        while time.monotonic() < deadline:
            job = self.get(job_id)
            if job and job.state in terminal:
                return job
            time.sleep(0.02)
        raise TimeoutError(f"Job {job_id} did not finish within {timeout}s")
