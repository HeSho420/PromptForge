"""Updates arrive the way the project does: through git.

Push to the repository, and every install can pull it — automatically at
launch (launch.ps1) or from inside the app while it runs (this module,
driven by the "update" job). The rules that keep that safe:

  fast-forward only   `git pull --ff-only`: an install never merges or
                      rebases. If local commits diverge from the remote,
                      the update refuses and says so — pushing or resetting
                      them is a decision for a person.

  dirty means no      locally MODIFIED tracked files would be clobbered or
                      cause conflicts, so the update refuses and lists them
                      by name. `data/` is untracked by design, so user
                      photos, models and the database can never block or be
                      touched by an update.

  deps follow code    backend/requirements.txt changed between the old and
                      new commit -> pip install runs; frontend/ changed and
                      npm exists -> the UI rebuilds. Nothing else is
                      reinstalled for nothing.

  restart proves it   the backend restarts through a detached helper that
                      waits for this process to die, starts the new code,
                      and probes /api/health for a minute. If the new code
                      never comes up, the helper resets the checkout back
                      to the recorded commit and starts the old code again
                      — a broken push must not brick the install.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger("promptforge.update")

GIT_TIMEOUT_S = 60
FETCH_TIMEOUT_S = 120


class UpdateError(RuntimeError):
    pass


class UpdateManager:
    def __init__(self, repo_root: Path):
        self.root = Path(repo_root)

    # ------------------------------------------------------------------ git
    def _git(self, *args: str, timeout: int = GIT_TIMEOUT_S) -> str:
        proc = subprocess.run(
            ["git", *args], cwd=str(self.root), capture_output=True,
            text=True, timeout=timeout)
        if proc.returncode != 0:
            raise UpdateError(
                (proc.stderr or proc.stdout or "git failed").strip()[:400])
        return (proc.stdout or "").strip()

    def is_repo(self) -> bool:
        try:
            return self._git("rev-parse", "--is-inside-work-tree") == "true"
        except Exception:  # noqa: BLE001
            return False

    def _dirty_tracked(self) -> list[str]:
        """Locally modified TRACKED files (untracked ones cannot conflict).

        Parsed by splitting off the status columns rather than by slicing:
        `_git` strips the output, which eats the leading space of a
        ' M file' line and made a fixed [3:] slice chop the first letter
        of the first filename — caught by this module's own tests."""
        out = self._git("status", "--porcelain", "--untracked-files=no")
        names: list[str] = []
        for line in out.splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                names.append(parts[1].strip())
        return names

    def _branch(self) -> str:
        return self._git("rev-parse", "--abbrev-ref", "HEAD")

    # --------------------------------------------------------------- status
    def status(self, fetch: bool = True) -> dict[str, Any]:
        """Where this install stands against its remote.

        Never raises for the API: problems come back as {"error": ...} so
        the Settings page can show them instead of a 500."""
        if not self.is_repo():
            return {"repo": False,
                    "error": "This install is not a git clone, so updates "
                             "cannot arrive through git. Clone the "
                             "repository to enable them."}
        try:
            branch = self._branch()
            commit = self._git("rev-parse", "--short", "HEAD")
            if fetch:
                self._git("fetch", "--quiet", "origin",
                          timeout=FETCH_TIMEOUT_S)
            upstream = f"origin/{branch}"
            behind = int(self._git("rev-list", "--count",
                                   f"HEAD..{upstream}"))
            ahead = int(self._git("rev-list", "--count",
                                  f"{upstream}..HEAD"))
            incoming: list[dict[str, str]] = []
            if behind:
                for line in self._git(
                        "log", "--oneline", "--no-decorate",
                        f"HEAD..{upstream}").splitlines()[:20]:
                    sha, _, subject = line.partition(" ")
                    incoming.append({"sha": sha, "subject": subject})
            return {"repo": True, "branch": branch, "commit": commit,
                    "behind": behind, "ahead": ahead,
                    "dirty": self._dirty_tracked(), "incoming": incoming}
        except Exception as exc:  # noqa: BLE001
            return {"repo": True, "error": str(exc)[:300]}

    # ---------------------------------------------------------------- apply
    def apply(self, job, restart: bool = True) -> dict[str, Any]:
        """Pull the update, refresh what changed, then restart into it."""
        if not self.is_repo():
            raise UpdateError("not a git clone — updates arrive through "
                              "git, and there is no repository here")
        dirty = self._dirty_tracked()
        if dirty:
            raise UpdateError(
                "locally modified files would be clobbered: "
                + ", ".join(dirty[:8])
                + (" …" if len(dirty) > 8 else "")
                + ". Commit, stash or restore them first.")
        branch = self._branch()
        job.log("info", "[stage] update — fetching what was pushed")
        self._git("fetch", "--quiet", "origin", timeout=FETCH_TIMEOUT_S)
        behind = int(self._git("rev-list", "--count",
                               f"HEAD..origin/{branch}"))
        if behind == 0:
            job.log("info", "Already up to date — nothing was pushed since "
                            "this version")
            return {"updated": False, "commit": self._git("rev-parse",
                                                          "--short", "HEAD")}
        old = self._git("rev-parse", "HEAD")
        job.log("info", f"{behind} update commit(s) available — applying "
                        "(fast-forward only; your data/ is untracked and "
                        "untouched)")
        self._git("pull", "--ff-only", "origin", branch,
                  timeout=FETCH_TIMEOUT_S)
        new = self._git("rev-parse", "HEAD")
        changed = self._git("diff", "--name-only", old, new).splitlines()

        if any(c == "backend/requirements.txt" for c in changed):
            job.log("info", "[stage] update — requirements changed; "
                            "installing new backend dependencies")
            pip = Path(sys.executable).parent / "pip.exe"
            proc = subprocess.run(
                [str(pip), "install", "-q", "--retries", "6", "-r",
                 str(self.root / "backend" / "requirements.txt")],
                capture_output=True, text=True, timeout=900)
            if proc.returncode != 0:
                job.log("warn", "Dependency install reported a problem "
                                f"({(proc.stderr or '').strip()[:160]}) — "
                                "the restart may fail and roll back")
        if any(c.startswith("frontend/") for c in changed):
            npm = self._find_npm()
            if npm:
                job.log("info", "[stage] update — the UI changed; "
                                "rebuilding it")
                proc = subprocess.run(
                    [npm, "run", "build"], cwd=str(self.root / "frontend"),
                    capture_output=True, text=True, timeout=900,
                    shell=False)
                if proc.returncode != 0:
                    job.log("warn", "UI rebuild failed — the app keeps the "
                                    "previous interface until the next "
                                    "successful build")
            else:
                job.log("info", "The UI changed but Node.js is not "
                                "installed here — keeping the previous "
                                "interface")

        summary = {"updated": True, "from": old[:7], "to": new[:7],
                   "commits": behind, "files_changed": len(changed)}
        if restart:
            job.log("info", "Update applied — restarting into the new "
                            "version (the app is back in ~15 seconds; if "
                            "the new version fails to start, the previous "
                            "one is restored automatically)")
            self._schedule_restart(old_commit=old)
        else:
            job.log("info", "Update applied — restart the app to run the "
                            "new version")
        return summary

    @staticmethod
    def _find_npm() -> str | None:
        for name in ("npm.cmd", "npm"):
            from shutil import which
            found = which(name)
            if found:
                return found
        return None

    # -------------------------------------------------------------- restart
    def _schedule_restart(self, old_commit: str) -> None:
        """Hand control to a detached helper, then exit this process.

        The helper inherits this process's environment, so the launcher's
        PROMPTFORGE_* configuration survives the restart. It waits for the
        port to free, starts the new code, health-probes it, and rolls the
        checkout back to `old_commit` if the new version never answers."""
        helper = self.root / "data" / "logs" / "pf-restart.ps1"
        helper.parent.mkdir(parents=True, exist_ok=True)
        python = sys.executable
        helper.write_text(f"""# Written by PromptForge's updater; safe to delete.
$ErrorActionPreference = 'Continue'
$pidToWait = {os.getpid()}
try {{ Wait-Process -Id $pidToWait -Timeout 30 }} catch {{}}
Start-Sleep -Seconds 2
function Test-Health {{
    try {{
        $r = [System.Net.WebRequest]::Create('http://127.0.0.1:8000/api/health')
        $r.Timeout = 2000
        $r.GetResponse().Close()
        return $true
    }} catch {{ return $false }}
}}
Start-Process -WindowStyle Hidden -WorkingDirectory '{self.root / "backend"}' `
    -RedirectStandardOutput '{self.root / "data" / "logs" / "backend-live.log"}' `
    -RedirectStandardError '{self.root / "data" / "logs" / "backend-live-err.log"}' `
    '{python}' -ArgumentList 'run.py' | Out-Null
$up = $false
for ($i = 0; $i -lt 30; $i++) {{
    Start-Sleep -Seconds 2
    if (Test-Health) {{ $up = $true; break }}
}}
if (-not $up) {{
    # The new version never answered: put the old one back.
    Set-Location '{self.root}'
    git reset --hard {old_commit} 2>&1 | Out-Null
    Start-Process -WindowStyle Hidden -WorkingDirectory '{self.root / "backend"}' `
        -RedirectStandardOutput '{self.root / "data" / "logs" / "backend-live.log"}' `
        -RedirectStandardError '{self.root / "data" / "logs" / "backend-live-err.log"}' `
        '{python}' -ArgumentList 'run.py' | Out-Null
}}
""", encoding="utf-8")

        def _die() -> None:
            subprocess.Popen(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                 "-File", str(helper)],
                creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP
                               | subprocess.DETACHED_PROCESS),
                close_fds=True)
            os._exit(0)

        # A short delay lets the job persist its "completed" state and the
        # HTTP response flush before the process vanishes.
        threading.Timer(2.0, _die).start()
