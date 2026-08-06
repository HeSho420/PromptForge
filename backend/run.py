"""Dev entrypoint: python run.py  (serves API on http://127.0.0.1:8000)"""
import logging
import socket
import sys

from app.api.routes import create_app

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
# The UI polls several endpoints every 1-2s; werkzeug's per-request access
# lines at INFO would flood the console. Errors (4xx/5xx) still surface.
logging.getLogger("werkzeug").setLevel(logging.WARNING)


def _instance_running(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


if __name__ == "__main__":
    # Windows sockets let a SECOND server bind :8000 (SO_REUSEADDR shares
    # the port instead of failing) — a duplicate instance wouldn't crash, it
    # would silently share the port AND the SQLite job queue, and jobs would
    # run on whichever (possibly stale) process grabs them first. Seen live:
    # a day-old zombie backend executed fresh jobs with outdated code.
    if _instance_running(8000):
        logging.getLogger("promptforge").error(
            "Another PromptForge instance already serves port 8000 — "
            "refusing to start a second one (it would steal queued jobs). "
            "Stop it first, or just use the running app.")
        sys.exit(1)
    app = create_app()
    # threaded=True matters: the UI polls several endpoints concurrently and
    # a workflow-generation request can take a minute — one slow request must
    # not block the rest of the app.
    app.run(host="127.0.0.1", port=8000, debug=False, threaded=True)
