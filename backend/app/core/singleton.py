"""One backend per machine, enforced with a liveness-checked pid file.

Why a port probe is not enough: Windows sockets with SO_REUSEADDR let a
second server bind :8000 instead of failing (documented live incident:
a day-old zombie backend executed fresh jobs with outdated code), and
run.py's connect-probe only fires at STARTUP — two instances booting in
the same window (the updater's rollback arm racing a slow new version
is a code-visible path) both pass it, share the port, and the loser
lives on with an active monitor and queue workers. The pid file closes
the class: the second process sees a live owner and exits before it
builds anything.

Process-model note (learned the hard way, 2026-08-18): a Python 3.13
venv's python.exe is a LAUNCHER that runs the real interpreter as its
child — every backend/ComfyUI instance is a two-process PAIR sharing
one fate. Only the CHILD executes run.py and takes this lock; process
counts must group pairs, and killing either member ends both.
"""
from __future__ import annotations

import os
from pathlib import Path


def _pid_alive(pid: int) -> bool:
    """Is the process genuinely RUNNING (not merely a pid someone still
    holds a handle to)? os.kill(pid, 0) cannot tell on Windows: a dead
    child whose Popen handle is still open reports as alive, and access-
    denied answers are ambiguous. OpenProcess + GetExitCodeProcess is the
    real answer — STILL_ACTIVE (259) means running, anything else is an
    exit code of a finished process."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            # Access denied = a protected process EXISTS; anything else
            # (invalid parameter) = no such process.
            return ctypes.get_last_error() == 5
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True  # exists but unqueryable — assume alive
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def acquire(lock_path: Path) -> bool:
    """Claim the single-instance lock, stealing it only from the dead.

    True when this process now owns the lock. False when another LIVE
    process holds it — the caller must exit without touching anything."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):  # first pass may steal a stale lock, then re-claim
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                holder = int(lock_path.read_text().strip() or "0")
            except (OSError, ValueError):
                holder = 0
            if holder != os.getpid() and _pid_alive(holder):
                return False
            # Stale (crashed owner, os._exit restarts, power loss): take it.
            try:
                lock_path.unlink()
            except OSError:
                return False
    return False


def release(lock_path: Path) -> None:
    """Drop the lock if THIS process holds it (never someone else's)."""
    try:
        if int(lock_path.read_text().strip() or "0") == os.getpid():
            lock_path.unlink()
    except (OSError, ValueError):
        pass
