"""Dev entrypoint: python run.py  (serves API on http://127.0.0.1:8000)"""
import logging
import os
import socket
import sys
from pathlib import Path

from app.core.singleton import acquire, release

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
# The UI polls several endpoints every 1-2s; werkzeug's per-request access
# lines at INFO would flood the console. Errors (4xx/5xx) still surface.
logging.getLogger("werkzeug").setLevel(logging.WARNING)

LOCK = Path(__file__).resolve().parent.parent / "data" / "logs" / "backend.pid"


def _instance_running(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


if __name__ == "__main__":
    # ONE backend per machine, enforced twice over. The pid-file lock is
    # the real guarantee: Windows sockets let a second server share :8000
    # via SO_REUSEADDR instead of failing, and two instances booting in
    # the same window (the updater's rollback arm racing a slow new
    # version — measured live) both pass a connect-probe, share the port,
    # and the loser lives on as a zombie whose monitor spawns duplicate
    # ComfyUI processes. The probe stays as the friendly fast-path error.
    if _instance_running(8000):
        logging.getLogger("promptforge").error(
            "Another PromptForge instance already serves port 8000 — "
            "refusing to start a second one (it would steal queued jobs). "
            "Stop it first, or just use the running app.")
        sys.exit(1)
    if not acquire(LOCK):
        logging.getLogger("promptforge").error(
            "Another PromptForge backend is already starting or running "
            f"(live pid in {LOCK}) — refusing to start a second one.")
        sys.exit(1)
    code = 0
    try:
        from app.api.routes import create_app

        app = create_app()
        # threaded=True matters: the UI polls several endpoints concurrently
        # and a workflow-generation request can take a minute — one slow
        # request must not block the rest of the app.
        app.run(host="127.0.0.1", port=8000, debug=False, threaded=True)
    except Exception:  # noqa: BLE001 — surface it, then die completely
        logging.getLogger("promptforge").exception("Backend crashed")
        code = 1
    finally:
        release(LOCK)
        # If the server loop ever ENDS (socket lost to a sharer, fatal
        # error), no half-alive process may linger running monitors and
        # queue workers against a port it no longer owns.
        os._exit(code)
