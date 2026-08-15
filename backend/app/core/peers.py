"""Two PromptForge machines on one network help each other.

Two capabilities, one subsystem, both stdlib-only:

  model transfer   a fresh install pulls multi-gigabyte weights from a
                   machine two metres away instead of the internet. The
                   peer is NOT trusted — the registry's pinned SHA-256 is
                   what makes the bytes trustworthy, so only sha-pinned
                   entries ever take this path, and the normal checksum
                   verification runs on the result exactly as it does for
                   an internet download.

  render delegation  when this machine's queue is busy and a peer's is
                   idle, a job's ComfyUI traffic is routed to the peer
                   through a thin reverse proxy the peer controls. The
                   peer enforces its own policy: sharing off means 403,
                   busy means 409, and the delegator falls back to
                   rendering locally on any failure.

Privacy boundary, deliberately hard: the main app keeps listening on
127.0.0.1 only. What this LAN listener serves is the MODEL library (public
weights, path-checked under models_dir) and a ComfyUI proxy — never
assets, photos, prompts or settings. The one deliberate extension: the
info endpoint describes what this machine is DOING — job type, a
code-authored stage keyword ("render", "verify"), queue depth, timings —
because the owner watching from their other PC needs exactly that. Prompt
text and payloads never cross; the stage keyword is cut BEFORE the "—"
where prompt-derived words can appear.

Discovery is a tiny UDP beacon plus an active TCP scanner; a status loop
keeps a live health picture of every known peer (latency, last error,
what it is doing) so the UI answers instantly instead of probing serially
on every poll. Peers that answered once are remembered on disk and
re-probed after a restart — two machines reconnect by themselves.
"""
from __future__ import annotations

import json
import logging
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote, unquote

log = logging.getLogger("promptforge.peers")

BEACON_INTERVAL_S = 6.0
PEER_STALE_S = 25.0
HTTP_TIMEOUT_S = 10.0
STATUS_INTERVAL_S = 4.0      # how often the status loop refreshes peers
STATUS_FRESH_S = 12.0        # cached info younger than this needs no probe
OFFLINE_AFTER_FAILS = 2      # consecutive failed probes before "offline"
BACKOFF_AFTER_FAILS = 4      # then probe every Nth tick, not every tick
BACKOFF_EVERY_TICKS = 6
# A remembered address that has not answered for this long is forgotten:
# without decay, every test rig and re-IP'd machine ever connected would
# be re-probed by the scanner forever.
HOST_MEMORY_S = 14 * 24 * 3600.0


class Peer:
    def __init__(self, token: str, name: str, host: str, port: int,
                 static: bool = False):
        self.token = token
        self.name = name
        self.host = host
        self.port = port
        self.static = static      # added by hand/env: never pruned
        self.last_seen = time.time()
        self.first_seen = time.time()
        # Live health picture, maintained by the status loop (and by any
        # add_peer probe): the last /pf-peer/info payload, when it arrived,
        # how long it took, and the current failure streak. The UI and the
        # delegation chooser read THIS instead of touching the network.
        self.info: dict[str, Any] | None = None
        self.info_at: float = 0.0
        self.latency_ms: float | None = None
        self.fails: int = 0
        self.last_error: str | None = None

    @property
    def base(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def reachable(self) -> bool:
        return self.fails < OFFLINE_AFTER_FAILS

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "host": self.host, "port": self.port,
                "static": self.static,
                "seen_ago_s": round(time.time() - self.last_seen, 1)}

    def status_dict(self) -> dict[str, Any]:
        """Everything the UI shows about this peer, answered from cache —
        no network round-trip hides inside this call."""
        info = self.info or {}
        return {
            **self.to_dict(),
            "reachable": self.reachable,
            "latency_ms": self.latency_ms,
            "last_error": self.last_error,
            "info_age_s": (round(time.time() - self.info_at, 1)
                           if self.info_at else None),
            "idle": info.get("idle"),
            "stats": info.get("stats"),
            "comfy": info.get("comfy"),
            "comfy_env": info.get("comfy_env"),
            "version": info.get("version"),
            "queue": info.get("queue"),
            "uptime_s": info.get("uptime_s"),
            "auto_update": info.get("auto_update"),
        }


class _PeerHandler(BaseHTTPRequestHandler):
    """Module-level so PeerService's route methods can be typed against
    the class that actually carries the _json helper."""

    def _json(self, code: int, payload: Any) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_body(self, cap: int) -> bytes | None:
        """The request body, or None when the Content-Length is unusable.

        A raw `int(Content-Length)` accepts a NEGATIVE value, and
        rfile.read(-1) reads until EOF — on a keep-alive connection that
        never comes, so the handler thread blocks forever. Repeat that and
        the ThreadingHTTPServer's unbounded daemon threads pile up: a
        one-line denial of service from any LAN host. The length is floored
        at zero and capped, so a hostile or garbage header can only ever ask
        for a bounded, finite read."""
        raw = self.headers.get("Content-Length")
        if raw is None:
            return b""
        try:
            declared = int(raw)
        except ValueError:
            return None
        length = max(0, min(declared, cap))
        return self.rfile.read(length) if length else b""


class PeerService:
    """Announce this install, discover others, serve models, proxy renders."""

    def __init__(self, registry, comfy_url: str = "http://127.0.0.1:8188",
                 share: bool = True, render: bool = True,
                 http_port: int = 8765, udp_port: int = 8766,
                 name: str | None = None,
                 busy_check: Callable[[], bool] | None = None,
                 loopback_only: bool = False,
                 static_hosts: list[str] | None = None,
                 stats_provider: Callable[[], dict] | None = None,
                 env_provider: Callable[[], dict] | None = None,
                 queue_provider: Callable[[], dict] | None = None,
                 version_provider: Callable[[], dict | None] | None = None,
                 auto_update: bool = True):
        self.registry = registry
        self.comfy_url = comfy_url.rstrip("/")
        self.share = share
        self.render = render
        self.udp_port = udp_port
        self.http_port = http_port          # actual port set at start()
        self.name = name or socket.gethostname()
        self.busy_check = busy_check or (lambda: False)
        # Loopback mode exists for tests and same-machine pairs: broadcast
        # frames do not reliably loop back on Windows, 127.0.0.1 does.
        self.loopback_only = loopback_only
        self.token = f"{self.name}-{time.time_ns()}"
        self.peers: dict[str, Peer] = {}
        self.static_hosts = list(static_hosts or [])
        # Eagerly initialised: the lazy version had a first-call race and
        # an empty-dict-is-falsy trap that left one machine's caches
        # permanently unfilled (every request got a fresh orphaned dict).
        self._bg_cache: dict[str, tuple[float, Any]] = {}
        self._bg_running: set[str] = set()
        # Injected by Services: accepts a pushed model manifest and queues
        # the downloads on THIS machine (they arrive over the LAN path).
        self.on_pull: Callable[[list[dict]], dict] | None = None
        # Injected by Services: live GPU/RAM numbers, shown on the other
        # machine's rail so "who has headroom?" is answered at a glance.
        self.stats_provider = stats_provider
        # Injected by Services: ComfyUI's venv facts (python/torch/GPU),
        # readable even while ComfyUI is down — remote WHY, not just THAT.
        self.env_provider = env_provider
        # Injected by Services: a prompt-free picture of this machine's
        # queue (depth + running job type/stage/timing) so the owner's
        # other PC can SHOW what this one is doing.
        self.queue_provider = queue_provider
        # Injected by Services: this install's version identity (git short
        # sha + commit timestamp). Peers compare it to their own; the newer
        # side triggers the other's normal git update — code itself never
        # crosses the LAN.
        self.version_provider = version_provider
        # Injected by Services: called (peer, its info) when the status
        # loop sees a peer running a NEWER version than this install.
        self.on_newer_peer: Callable[[Peer, dict], None] | None = None
        # Advertised so the OTHER machine's UI can say honestly whether
        # this one will catch up by itself or needs a manual update.
        self.auto_update = auto_update
        self._started_at = time.time()
        # Every address that ever answered as a PromptForge: the scanner
        # keeps re-probing these, so a peer that reboots comes back on its
        # own without waiting for a beacon to make it through.
        self.known_hosts: set[tuple[str, int]] = set()
        # When each of them last actually answered — old ones decay out.
        self._host_last_ok: dict[tuple[str, int], float] = {}
        self._hosts_saved_at = 0.0
        # ...and they survive restarts: the same file is re-probed at the
        # next start(), so two machines reconnect without rediscovery.
        self._hosts_file = Path(registry.models_dir).parent / "peers.json"
        self._load_hosts()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._httpd: ThreadingHTTPServer | None = None
        self._index_cache: dict[str, tuple[float, list[dict]]] = {}
        self._status_tick = 0

    # ------------------------------------------------------------------ life
    def start(self) -> None:
        if not (self.share or self.render):
            return
        self._stop.clear()
        self._start_http()
        workers = [self._beacon_tx, self._beacon_rx, self._status_loop]
        if not self.loopback_only:
            workers.append(self._scanner)
        for target in workers:
            t = threading.Thread(target=target, daemon=True,
                                 name=f"pf-peer-{target.__name__}")
            t.start()
            self._threads.append(t)
        # Hand-configured peers (PROMPTFORGE_PEER_HOSTS): probed over plain
        # HTTP, so they work even where UDP broadcasts never arrive. Peers
        # remembered from earlier runs are probed the same way — they were
        # real once, and a restart must not orphan the pair.
        remembered = [(h, p) for (h, p) in sorted(self.known_hosts)]
        for entry in self.static_hosts:
            host, _, port = entry.strip().partition(":")
            if host:
                remembered.append(
                    (host, int(port) if port.isdigit() else 8765))
        for rhost, rport in dict.fromkeys(remembered):
            threading.Thread(
                target=self.add_peer, args=(rhost, rport),
                kwargs={"pin": any(e.strip().partition(":")[0] == rhost
                                   for e in self.static_hosts)},
                daemon=True, name="pf-peer-reconnect").start()
        log.info("peer service up: http :%s, beacon :%s (share=%s render=%s)",
                 self.http_port, self.udp_port, self.share, self.render)

    def stop(self) -> None:
        self._stop.set()
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:  # noqa: BLE001
                pass
            self._httpd = None
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()

    # ------------------------------------------------------------------ http
    def _start_http(self) -> None:
        service = self
        bind_host = "127.0.0.1" if self.loopback_only else "0.0.0.0"

        class Handler(_PeerHandler):
            protocol_version = "HTTP/1.1"
            # No timeout means a slow (or hostile) client holds a handler
            # thread forever: dribble the request line, or declare a body
            # and never send it. HTTP/1.1 keep-alive makes the idle case
            # persistent too. 30s is generous for a LAN manifest or graph
            # and applies per socket operation (not per transfer), so a
            # legitimate steady model download is unaffected — only a
            # stalled connection is dropped. BaseHTTPRequestHandler turns
            # the resulting socket timeout into a clean close.
            timeout = 30

            def log_message(self, *_args):  # keep the console quiet
                pass

            def do_GET(self):  # noqa: N802
                try:
                    service._handle_get(self)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as exc:  # noqa: BLE001
                    try:
                        self._json(500, {"error": str(exc)[:200]})
                    except Exception:  # noqa: BLE001
                        pass

            def do_POST(self):  # noqa: N802
                try:
                    service._handle_post(self)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as exc:  # noqa: BLE001
                    try:
                        self._json(500, {"error": str(exc)[:200]})
                    except Exception:  # noqa: BLE001
                        pass

        class Server(ThreadingHTTPServer):
            # allow_reuse_address (the BaseServer default) lets a SECOND
            # server bind an already-taken port on Windows — measured
            # live: a test install silently double-bound the real
            # install's :8765 and requests landed on whichever process
            # accepted first. With reuse off the second bind raises and
            # the port range below moves on to :8766, which is the whole
            # point of having a range.
            allow_reuse_address = False

        last_error: Exception | None = None
        for port in range(self.http_port, self.http_port + 10):
            try:
                self._httpd = Server((bind_host, port), Handler)
                self._httpd.daemon_threads = True
                self.http_port = port
                break
            except OSError as exc:
                last_error = exc
        if self._httpd is None:
            raise RuntimeError(f"no free peer port: {last_error}")
        t = threading.Thread(target=self._httpd.serve_forever, daemon=True,
                             name="pf-peer-http")
        t.start()
        self._threads.append(t)

    # ---- server-side request handling (runs on http threads)
    def _background_cached(self, name: str, fn: Callable[[], Any],
                           ttl: float) -> Any:
        """The last known value NOW; a refresh in the background when it
        is stale. /pf-peer/info must answer within the discovery probes'
        couple of seconds, and a cold torch-import probe was measured
        taking ~25s — long enough that the machine looked 'not answering'
        on every other machine precisely while it had the most to say.

        The spawn decision is taken under the lock: concurrent first
        requests must agree on ONE refresh writing into ONE cache."""
        entry = self._bg_cache.get(name)
        fresh = entry is not None and time.time() - entry[0] < ttl
        if not fresh:
            with self._lock:
                spawn = name not in self._bg_running
                if spawn:
                    self._bg_running.add(name)
            if spawn:
                def refresh() -> None:
                    try:
                        value = fn()
                    except Exception:  # noqa: BLE001
                        value = entry[1] if entry else None
                    self._bg_cache[name] = (time.time(), value)
                    with self._lock:
                        self._bg_running.discard(name)

                threading.Thread(target=refresh, daemon=True,
                                 name=f"pf-peer-{name}").start()
        return entry[1] if entry else None

    def _comfy_status(self) -> dict[str, Any]:
        """Can this machine actually render? Answered by its own ComfyUI.

        A peer whose PromptForge answers but whose ComfyUI is down or
        CPU-bound must never be picked for delegation, and the other
        machine's UI should SAY so — measured live: a gaming PC ran the
        app for a day on the mock renderer and nothing surfaced it."""
        cached = getattr(self, "_comfy_cache", None)
        if cached is not None and time.time() - cached[0] < 5.0:
            return cached[1]
        out: dict[str, Any] = {"up": False, "device": None, "gpu": None}
        try:
            with urllib.request.urlopen(self.comfy_url + "/system_stats",
                                        timeout=2) as resp:
                data = json.loads(resp.read().decode())
            dev = (data.get("devices") or [{}])[0]
            out = {"up": True, "device": str(dev.get("type") or ""),
                   "gpu": dev.get("name")}
        except Exception:  # noqa: BLE001 — down is an answer, not an error
            pass
        self._comfy_cache = (time.time(), out)
        return out

    def _handle_get(self, req: _PeerHandler) -> None:
        path = req.path.split("?", 1)[0]
        if path == "/pf-peer/info":
            # Every slow probe is served from cache and refreshed in the
            # background: this endpoint must answer inside the discovery
            # probes' short timeouts or the machine reads as offline.
            stats = (self._background_cached("stats", self.stats_provider,
                                             10.0)
                     if self.stats_provider is not None else None)
            env = (self._background_cached("env", self.env_provider, 60.0)
                   if self.env_provider is not None else None)
            comfy = self._background_cached("comfy", self._comfy_status,
                                            5.0)
            # Version identity: the first call may run git, so it goes
            # through the background cache too — this endpoint must answer
            # inside the discovery probes' short timeouts, always.
            version = (self._background_cached(
                "version", self.version_provider, 3600.0)
                if self.version_provider is not None else None)
            # The queue picture is an in-memory read — served live. A
            # broken provider must not take the whole endpoint down.
            queue = None
            if self.queue_provider is not None:
                try:
                    queue = self.queue_provider()
                except Exception:  # noqa: BLE001 — absent beats broken
                    queue = None
            req._json(200, {"app": "promptforge", "name": self.name,
                            "token": self.token, "share": self.share,
                            "render": self.render,
                            "idle": not self.busy_check(),
                            "comfy": comfy or {"up": False, "device": None,
                                               "gpu": None},
                            "comfy_env": env,
                            "stats": stats,
                            "version": version,
                            "queue": queue,
                            "auto_update": self.auto_update,
                            "uptime_s": round(
                                time.time() - self._started_at, 1)})
            return
        if path == "/pf-peer/models":
            if not self.share:
                req._json(403, {"error": "model sharing is off"})
                return
            req._json(200, self._model_index())
            return
        if path.startswith("/pf-peer/model/"):
            if not self.share:
                req._json(403, {"error": "model sharing is off"})
                return
            self._serve_model(req, unquote(path[len("/pf-peer/model/"):]))
            return
        if path.startswith("/pf-peer/comfy/"):
            self._proxy(req, body=None)
            return
        if path.startswith("/pf-peer/log/"):
            self._serve_log(req, unquote(path[len("/pf-peer/log/"):]))
            return
        req._json(404, {"error": "unknown path"})

    # Operational logs another machine of the SAME OWNER may read to
    # diagnose this one remotely. A fixed whitelist, never a directory
    # walk: install/crash logs only, no jobs, no prompts-carrying app
    # DBs. Two of these are PASS-THROUGH streams PromptForge does not
    # author (ComfyUI's stdout/stderr can echo node inputs), so serving
    # is additionally limited to machines that are themselves paired
    # PromptForge peers — not any curious device on the LAN.
    LOG_WHITELIST = frozenset({
        "comfyui.log", "comfyui-err.log", "comfyui-repair.log",
        "comfyui-install.log", "directml-install.log",
        "torch-cuda-repair.log", "sage-install.log", "xformers-install.log",
        "sam-install.log", "backend-live.log", "backend-live-err.log",
        "doctor-report.txt", "launch.log",
    })

    def _serve_log(self, req: _PeerHandler, name: str) -> None:
        if not self.share:
            req._json(403, {"error": "sharing is off"})
            return
        # Only machines this install already KNOWS as PromptForge peers
        # (plus loopback pairs) may read logs: the files can carry
        # pass-through text this app does not author, and "shows up in
        # my peer list" is the closest thing a LAN protocol without
        # secrets has to "is my other machine".
        source = req.client_address[0]
        allowed = {p.host for p in self.peers_list()} | {"127.0.0.1"}
        if source not in allowed:
            req._json(403, {"error": "logs are readable by paired "
                                     "PromptForge machines only"})
            return
        if name not in self.LOG_WHITELIST:
            req._json(404, {"error": "not a shareable log"})
            return
        path = self.registry.models_dir.parent / "logs" / name
        try:
            resolved = path.resolve()
            resolved.relative_to(
                (self.registry.models_dir.parent / "logs").resolve())
        except (ValueError, OSError):
            req._json(403, {"error": "outside the log folder"})
            return
        if not resolved.exists():
            req._json(404, {"error": f"{name} does not exist here"})
            return
        data = resolved.read_bytes()[-32_768:]
        req.send_response(200)
        req.send_header("Content-Type", "text/plain; charset=utf-8")
        req.send_header("Content-Length", str(len(data)))
        req.end_headers()
        req.wfile.write(data)

    def _handle_post(self, req: _PeerHandler) -> None:
        path = req.path.split("?", 1)[0]
        if path.startswith("/pf-peer/comfy/"):
            # 256 MB covers any ComfyUI graph or uploaded image with room to
            # spare; the cap is what stops a claimed-huge body from being
            # read wholesale into memory before it is proxied.
            body = req.read_body(256 * 1024 * 1024)
            if body is None:
                req._json(400, {"error": "bad Content-Length"})
                return
            self._proxy(req, body=body)
            return
        if path == "/pf-peer/pull":
            # A peer offers its model manifest; THIS machine decides what
            # to queue. Only sha-pinned entries are ever accepted, and the
            # downloads themselves run through the normal registry path
            # (LAN first, checksum-verified) as visible jobs.
            #
            # The body is drained BEFORE any refusal: answering 403 while
            # the client is still writing races into a connection reset,
            # so the client never sees the honest error (caught live by
            # the suite under load). Capped — a manifest is tiny.
            raw = req.read_body(8 * 1024 * 1024)
            if raw is None:
                req._json(400, {"error": "bad Content-Length"})
                return
            if not self.share:
                req._json(403, {"error": "model sharing is off"})
                return
            if self.on_pull is None:
                req._json(501, {"error": "no pull handler on this build"})
                return
            try:
                body = json.loads(raw.decode() if raw else "{}")
                if not isinstance(body, dict):
                    raise ValueError("manifest must be an object")
                entries = list(body.get("models") or [])[:300]
            except (ValueError, UnicodeDecodeError):
                req._json(400, {"error": "bad manifest"})
                return
            req._json(200, self.on_pull(entries))
            return
        req._json(404, {"error": "unknown path"})

    def _model_index(self) -> list[dict]:
        out: list[dict] = []
        root = self.registry.models_dir.resolve()
        for m in self.registry.list():
            if m.status != "ready" or not m.path:
                continue
            p = Path(m.path)
            try:
                resolved = p.resolve()
                resolved.relative_to(root)   # models only, never user data
            except (ValueError, OSError):
                continue
            if not resolved.exists():
                continue
            out.append({"name": m.name, "file": resolved.name,
                        "folder": (m.meta or {}).get("folder", ""),
                        "size": resolved.stat().st_size,
                        "sha256": m.sha256 or "",
                        # Provenance travels with the entry so a machine
                        # that ASKS for models can register them with the
                        # original internet URL, not just the LAN copy.
                        "url": m.url or "",
                        "purpose": m.purpose or "",
                        "license": m.license or "",
                        "meta": dict(m.meta or {})})
        return out

    def _serve_model(self, req: _PeerHandler, name: str) -> None:
        m = self.registry.get(name)
        root = self.registry.models_dir.resolve()
        if m is None or m.status != "ready" or not m.path:
            req._json(404, {"error": f"model '{name}' not available"})
            return
        p = Path(m.path).resolve()
        try:
            p.relative_to(root)
        except ValueError:
            req._json(403, {"error": "outside the model library"})
            return
        if not p.exists():
            req._json(404, {"error": "file vanished"})
            return
        size = p.stat().st_size
        start, end = 0, size - 1
        rng = req.headers.get("Range")
        partial = False
        if rng and rng.startswith("bytes="):
            try:
                s, _, e = rng[len("bytes="):].partition("-")
                if not s:
                    # Suffix range "bytes=-N" means the LAST N bytes, not
                    # bytes 0..N. The old code read it as the latter and
                    # served the wrong bytes under a 206 — silent corruption
                    # for any conformant client (the transfer client only
                    # sends "bytes=N-", which is why it stayed hidden).
                    n = int(e) if e else 0
                    start, end = max(0, size - n), size - 1
                else:
                    start = int(s)
                    end = int(e) if e else size - 1
                partial = True
            except ValueError:
                start, end, partial = 0, size - 1, False
        start = max(0, min(start, size - 1))
        end = max(start, min(end, size - 1))
        length = end - start + 1
        req.send_response(206 if partial else 200)
        req.send_header("Content-Type", "application/octet-stream")
        req.send_header("Content-Length", str(length))
        req.send_header("Accept-Ranges", "bytes")
        if partial:
            req.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        req.end_headers()
        with p.open("rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(1 << 20, remaining))
                if not chunk:
                    break
                req.wfile.write(chunk)
                remaining -= len(chunk)

    def _proxy(self, req: _PeerHandler, body: bytes | None) -> None:
        """Forward one request to the local ComfyUI, policy first.

        The peer decides for itself: rendering for others can be switched
        off, and a machine whose OWN queue is running never accepts more.
        Delegated ComfyUI load deliberately does not count as busy — the
        delegator streams several graphs per job."""
        if not self.render:
            req._json(403, {"error": "render sharing is off"})
            return
        if self.busy_check():
            req._json(409, {"error": "busy with local work"})
            return
        rest = req.path[len("/pf-peer/comfy"):] or "/"
        target = self.comfy_url + rest
        headers = {"Content-Type": req.headers.get("Content-Type")
                   or "application/octet-stream"}
        inner = urllib.request.Request(target, data=body, headers=headers,
                                       method=req.command)
        try:
            with urllib.request.urlopen(inner, timeout=600) as resp:
                payload = resp.read()
                req.send_response(resp.status)
                ctype = resp.headers.get("Content-Type")
                if ctype:
                    req.send_header("Content-Type", ctype)
                req.send_header("Content-Length", str(len(payload)))
                req.end_headers()
                req.wfile.write(payload)
        except urllib.error.HTTPError as exc:
            payload = exc.read() if hasattr(exc, "read") else b""
            req.send_response(exc.code)
            req.send_header("Content-Length", str(len(payload)))
            req.end_headers()
            req.wfile.write(payload)
        except Exception as exc:  # noqa: BLE001
            req._json(502, {"error": f"local ComfyUI unreachable: "
                                     f"{str(exc)[:160]}"})

    # ---------------------------------------------------------------- beacon
    # Same-machine pairs are real (tests, one PC with two installs): a
    # unicast datagram to a shared port reaches only ONE binder, so each
    # instance binds the first free port in a small range and every beacon
    # is sent to the whole range.
    UDP_RANGE = 4

    @staticmethod
    def _local_ipv4s() -> list[str]:
        ips: set[str] = set()
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None,
                                           socket.AF_INET):
                # AF_INET sockaddr is (host, port); the host is a str.
                ips.add(cast(str, info[4][0]))
        except OSError:
            pass
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe.connect(("8.8.8.8", 80))   # no packet is sent
            ips.add(probe.getsockname()[0])
            probe.close()
        except OSError:
            pass
        return [ip for ip in ips if not ip.startswith("127.")]

    def _beacon_tx(self) -> None:
        """Announce on every interface, not just the default one.

        A machine with Hyper-V/WSL virtual switches routes the plain
        255.255.255.255 broadcast out ONE interface — often the virtual
        one, where no peer will ever hear it. So each local IPv4 gets its
        own sending socket (binding the source address steers the egress
        interface) and its subnet's directed broadcast is targeted too.
        Hand-configured peers additionally get direct unicast beacons."""
        while not self._stop.is_set():
            payload = json.dumps({
                "pf": 1, "token": self.token, "name": self.name,
                "http": self.http_port}).encode()
            if self.loopback_only:
                socks: list[socket.socket] = []
                targets = ["127.0.0.1"]
            else:
                locals_ = self._local_ipv4s()
                socks = []
                for ip in locals_:
                    try:
                        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        s.setsockopt(socket.SOL_SOCKET,
                                     socket.SO_BROADCAST, 1)
                        s.bind((ip, 0))
                        socks.append(s)
                    except OSError:
                        pass
                targets = ["255.255.255.255", "127.0.0.1"]
                targets += [ip.rsplit(".", 1)[0] + ".255" for ip in locals_]
                targets += [p.host for p in self.peers_list() if p.static]
                targets += [e.strip().partition(":")[0]
                            for e in self.static_hosts if e.strip()]
            base = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            base.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            for sock in [base, *socks]:
                for host in dict.fromkeys(targets):
                    for port in range(self.udp_port,
                                      self.udp_port + self.UDP_RANGE):
                        try:
                            sock.sendto(payload, (host, port))
                        except OSError:
                            pass
            for sock in [base, *socks]:
                sock.close()
            self._stop.wait(BEACON_INTERVAL_S)

    def _beacon_rx(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        bound = False
        for port in range(self.udp_port, self.udp_port + self.UDP_RANGE):
            try:
                sock.bind(("", port))
                bound = True
                # The startup line prints the BASE port; on a machine
                # running two installs the actual bound port differs, and
                # the honest number is the one that helps debugging.
                log.info("peer discovery listening on udp :%s", port)
                break
            except OSError:
                continue
        if not bound:
            log.info("peer discovery off (ports %s-%s taken)",
                     self.udp_port, self.udp_port + self.UDP_RANGE - 1)
            return
        sock.settimeout(1.0)
        while not self._stop.is_set():
            try:
                data, (host, _port) = sock.recvfrom(2048)
            except TimeoutError:
                self._prune()
                continue
            except OSError:
                break
            # EVERYTHING derived from the packet is parsed inside this
            # guard. A crafted datagram from any LAN host used to kill the
            # discovery thread until restart: a token-less packet slipped
            # the check below (None != self.token) and then raised KeyError
            # on msg["token"], and a non-numeric "http" raised ValueError on
            # int() — both outside the json guard, both unhandled, both
            # fatal to the one thread that finds other machines.
            try:
                msg = json.loads(data.decode())
                if not isinstance(msg, dict) or msg.get("pf") != 1:
                    continue
                token = msg.get("token")
                if not isinstance(token, str) or not token \
                        or token == self.token:
                    continue
                http_port = int(msg.get("http") or 8765)
                name = str(msg.get("name") or host)
            except (ValueError, TypeError, UnicodeDecodeError):
                continue
            with self._lock:
                peer = self.peers.get(token)
                if peer is None:
                    self.peers[token] = Peer(token, name, host, http_port)
                    log.info("discovered peer '%s' at %s", name, host)
                else:
                    peer.last_seen = time.time()
                    peer.host = host
                    peer.port = http_port
        sock.close()

    # -------------------------------------------------------------- scanner
    SCAN_HUNGRY_S = 20.0     # nothing connected yet: look often
    SCAN_SETTLED_S = 120.0   # peers connected: keep an eye out for more

    def _scan_candidates(self) -> list[str]:
        """Addresses worth asking "are you a PromptForge?".

        The ARP table lists machines that provably exist on this segment;
        the full /24 of each local interface catches the ones ARP has not
        met yet. A home LAN's 254 addresses at a 0.4s connect timeout in a
        small thread pool is a few seconds of background work."""
        hosts: set[str] = set()
        try:
            out = subprocess.run(["arp", "-a"], capture_output=True,
                                 text=True, timeout=10).stdout
            hosts.update(m.group(0) for m in re.finditer(
                r"\b(?:10|172|192)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", out))
        except Exception:  # noqa: BLE001 — ARP is a bonus source
            pass
        for ip in self._local_ipv4s():
            base = ip.rsplit(".", 1)[0]
            hosts.update(f"{base}.{n}" for n in range(1, 255))
        locals_ = set(self._local_ipv4s())
        return [h for h in hosts
                if h not in locals_ and not h.endswith(".255")]

    def _scanner(self) -> None:
        """Actively find peers instead of waiting for beacons to arrive.

        UDP broadcasts die to firewalls and access-point isolation far
        more often than TCP does — measured live: two machines both
        running PromptForge, neither hearing the other. A cheap TCP
        connect sweep answers definitively. Known-good addresses are
        re-probed first, so a rebooted peer reconnects by itself."""
        while not self._stop.is_set():
            try:
                with self._lock:
                    remembered = list(self.known_hosts)
                for host, port in remembered:
                    self.add_peer(host, port, timeout=2.0, pin=False)
                reachable = bool(self.peers_list())
                if not reachable:
                    candidates = self._scan_candidates()
                    from concurrent.futures import ThreadPoolExecutor

                    def knock(host: str) -> str | None:
                        try:
                            s = socket.create_connection(
                                (host, 8765), timeout=0.4)
                            s.close()
                            return host
                        except OSError:
                            return None
                    with ThreadPoolExecutor(max_workers=32) as pool:
                        open_hosts = [h for h in pool.map(knock, candidates)
                                      if h]
                    for host in open_hosts:
                        self.add_peer(host, 8765, timeout=3.0, pin=False)
            except Exception:  # noqa: BLE001 — the scanner must survive
                pass
            self._stop.wait(self.SCAN_HUNGRY_S if not self.peers_list()
                            else self.SCAN_SETTLED_S)

    def _prune(self) -> None:
        now = time.time()
        with self._lock:
            for token in [t for t, p in self.peers.items()
                          if not p.static
                          and now - p.last_seen > PEER_STALE_S]:
                del self.peers[token]

    # ---------------------------------------------------------------- status
    def _status_loop(self) -> None:
        """Keep a live health picture of every known peer.

        ONE place probes; everyone else reads cache. Before this loop the
        API route re-probed every peer serially on every UI poll (three
        quiet peers made the Network page block for six seconds), and the
        delegation gate did the same every few seconds. Now each peer is
        probed at most once per tick, in parallel, and a peer that keeps
        failing is probed at a slower rhythm instead of hammered."""
        while not self._stop.is_set():
            self._status_tick += 1
            tick = self._status_tick
            targets = [p for p in self.peers_list()
                       if p.fails < BACKOFF_AFTER_FAILS
                       or tick % BACKOFF_EVERY_TICKS == 0]
            if targets:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(
                        max_workers=min(8, len(targets))) as pool:
                    list(pool.map(self._probe_peer, targets))
            self._stop.wait(STATUS_INTERVAL_S)

    def _probe_peer(self, peer: Peer) -> None:
        t0 = time.time()
        try:
            info = self._peer_json(peer, "/pf-peer/info", timeout=3.0)
            if not isinstance(info, dict) \
                    or info.get("app") != "promptforge":
                raise ValueError("did not answer as a PromptForge")
        except Exception as exc:  # noqa: BLE001 — recorded, never fatal
            with self._lock:
                peer.fails += 1
                peer.last_error = str(exc)[:160]
            return
        with self._lock:
            peer.info = info
            peer.info_at = time.time()
            peer.latency_ms = round((peer.info_at - t0) * 1000.0, 1)
            peer.fails = 0
            peer.last_error = None
            # An answering peer is ALIVE even where UDP beacons never
            # arrive: HTTP keeps it in the list, prune only kills silence.
            peer.last_seen = peer.info_at
        # A verified answer is exactly what the reconnect memory is for —
        # beacon-discovered peers never pass through add_peer, and
        # without this line they were forgotten at every restart.
        self._remember_host(peer.host, peer.port)
        self._maybe_notify_newer(peer, info)

    def _maybe_notify_newer(self, peer: Peer, info: dict) -> None:
        """Fire on_newer_peer when a peer runs newer code than this side.

        Fired on EVERY tick the difference persists — deliberately: the
        hook's own guards may decline transiently (queue busy) and must
        get another chance once the reason passes. The hook is cheap and
        idempotent; it only ever TRIGGERS this install's normal git
        update — deciding and fetching stay on the trusted path. Both
        sides run this symmetrically, so whichever machine updates first
        drags the other along within a status tick."""
        hook = self.on_newer_peer
        if hook is None or self.version_provider is None:
            return
        try:
            mine = self.version_provider()
            theirs = info.get("version")
            if not (isinstance(mine, dict) and isinstance(theirs, dict)):
                return
            mine_ts, theirs_ts = mine.get("ts"), theirs.get("ts")
            if not (isinstance(mine_ts, int | float)
                    and isinstance(theirs_ts, int | float)):
                return
            if theirs_ts <= mine_ts \
                    or theirs.get("commit") == mine.get("commit"):
                return
            hook(peer, info)
        except Exception:  # noqa: BLE001 — the status loop must survive
            log.debug("newer-version hook failed", exc_info=True)

    # ------------------------------------------------------- remembered hosts
    def _load_hosts(self) -> None:
        try:
            raw = json.loads(self._hosts_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return   # first run, or a corrupt file — rediscovery still works
        now = time.time()
        for e in raw if isinstance(raw, list) else []:
            if not isinstance(e, dict):
                continue
            host = str(e.get("host") or "").strip()
            port = e.get("port")
            last_ok = e.get("last_ok")
            if not (isinstance(last_ok, int | float) and last_ok > 0):
                last_ok = now   # a pre-decay file: treat as fresh once
            if not (host and isinstance(port, int) and 0 < port < 65536):
                continue
            if now - float(last_ok) > HOST_MEMORY_S:
                continue        # decayed: not answered for two weeks
            self.known_hosts.add((host, port))
            self._host_last_ok[(host, port)] = float(last_ok)

    def _save_hosts(self,
                    snapshot: list[tuple[str, int, float]]) -> None:
        """Write a SNAPSHOT taken under the lock — iterating the live set
        here raced concurrent add_peer calls (scanner + request threads +
        reconnect threads) into 'set changed size during iteration'."""
        try:
            self._hosts_file.parent.mkdir(parents=True, exist_ok=True)
            self._hosts_file.write_text(
                json.dumps([{"host": h, "port": p, "last_ok": ok}
                            for h, p, ok in snapshot][:64]),
                encoding="utf-8")
        except OSError:
            pass   # remembering peers is a convenience, never a failure

    def _remember_host(self, host: str, port: int) -> None:
        """Record that this address answered as a PromptForge just now.
        Writes the file when something NEW appears, and otherwise at most
        hourly — the timestamps only need day-scale resolution."""
        now = time.time()
        with self._lock:
            key = (host, port)
            is_new = key not in self.known_hosts
            self.known_hosts.add(key)
            self._host_last_ok[key] = now
            if not is_new and now - self._hosts_saved_at < 3600:
                return
            self._hosts_saved_at = now
            snapshot = sorted(
                (h, p, self._host_last_ok.get((h, p), now))
                for h, p in self.known_hosts
                if now - self._host_last_ok.get((h, p), now)
                <= HOST_MEMORY_S)
        self._save_hosts(snapshot)

    # ---------------------------------------------------------------- client
    def peers_list(self) -> list[Peer]:
        self._prune()
        with self._lock:
            return list(self.peers.values())

    def peers_status(self) -> list[dict[str, Any]]:
        """What the UI shows — answered entirely from the status cache."""
        self._prune()
        with self._lock:
            return [p.status_dict() for p in self.peers.values()]

    def add_peer(self, host: str, port: int = 8765,
                 timeout: float = 5.0,
                 pin: bool = True) -> dict[str, Any] | None:
        """Connect to a peer by address.

        Probes /pf-peer/info over plain HTTP — the escape hatch for
        networks where UDP broadcasts never arrive. A machine added BY
        HAND (or from the environment) is pinned and never pruned; one the
        scanner found is not, so it disappears when it really goes away
        and reappears when the scanner sees it again."""
        t0 = time.time()
        try:
            with urllib.request.urlopen(
                    f"http://{host}:{int(port)}/pf-peer/info",
                    timeout=timeout) as resp:
                info = json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001
            log.info("peer probe %s:%s failed: %s", host, port, exc)
            return None
        latency_ms = round((time.time() - t0) * 1000.0, 1)
        if info.get("app") != "promptforge":
            return None
        if info.get("token") == self.token:
            return {"self": True, **info}
        self._remember_host(host, int(port))
        with self._lock:
            peer = self.peers.get(info["token"])
            if peer is None:
                peer = Peer(info["token"], str(info.get("name") or host),
                            host, int(port), static=pin)
                self.peers[info["token"]] = peer
                log.info("peer '%s' connected at %s:%s",
                         info.get("name"), host, port)
            else:
                peer.static = peer.static or pin
                peer.host = host
                peer.port = int(port)
                peer.last_seen = time.time()
            # The probe just fetched a full info payload — cache it, so
            # a hand-connected peer shows its whole picture immediately
            # instead of waiting for the next status tick.
            peer.info = info
            peer.info_at = time.time()
            peer.latency_ms = latency_ms
            peer.fails = 0
            peer.last_error = None
        self._maybe_notify_newer(peer, info)
        return info

    def _peer_json(self, peer: Peer, path: str,
                   timeout: float = HTTP_TIMEOUT_S) -> Any:
        with urllib.request.urlopen(peer.base + path,
                                    timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def find_model_url(self, name: str, sha256: str | None) -> str | None:
        """Where on the network this model already exists, or None.

        Only a sha match counts: the download path verifies the bytes
        against the registry pin, so a peer advertising a different sha
        would only waste a download."""
        if not sha256:
            return None
        for peer in self.peers_list():
            try:
                cached = self._index_cache.get(peer.token)
                if cached is None or time.time() - cached[0] > 60:
                    cached = (time.time(),
                              self._peer_json(peer, "/pf-peer/models"))
                    self._index_cache[peer.token] = cached
                for entry in cached[1]:
                    if (entry.get("name") == name
                            and (entry.get("sha256") or "").lower()
                            == sha256.lower()):
                        return (f"{peer.base}/pf-peer/model/"
                                f"{quote(name, safe='')}")
            except Exception:  # noqa: BLE001 — a dead peer is not an error
                continue
        return None

    def best_idle_peer(self) -> Peer | None:
        """An idle peer that can actually RENDER, or None.

        A peer with no ComfyUI cannot help however idle it is, and one
        rendering on CPU is a last resort only — delegating a GPU job to
        it would take longer than waiting for this machine. Answered from
        the status cache when it is fresh: the delegation gate asks every
        few seconds, and probing every peer serially on each ask was
        seconds of network time bought for nothing."""
        cpu_fallback: Peer | None = None
        for peer in self.peers_list():
            info = (peer.info
                    if peer.info is not None
                    and time.time() - peer.info_at < STATUS_FRESH_S
                    else None)
            if info is None:
                try:
                    info = self._peer_json(peer, "/pf-peer/info")
                except Exception:  # noqa: BLE001
                    continue
            if not (info.get("render") and info.get("idle")):
                continue
            comfy = info.get("comfy") or {}
            if not comfy.get("up"):
                continue
            if str(comfy.get("device") or "").lower() == "cpu":
                cpu_fallback = cpu_fallback or peer
                continue
            return peer
        return cpu_fallback

    def find_peer(self, target: str) -> Peer | None:
        """A known peer matched by host or name (the device picker sends
        whichever it has)."""
        for peer in self.peers_list():
            if target in (peer.host, peer.name):
                return peer
        return None

    def fetch_manifest(self, host: str, port: int = 8765,
                       timeout: float = 10.0) -> list[dict]:
        """A peer's shareable model list, normalized for acceptance: every
        entry carries name/sha256 plus whatever provenance the peer knows.
        Tolerates older peers that serve only the thin index (no url/meta)
        by folding their folder/file fields into meta so downloads still
        land in the right typed subfolder."""
        with urllib.request.urlopen(
                f"http://{host}:{port}/pf-peer/models",
                timeout=timeout) as resp:
            raw = json.loads(resp.read().decode())
        entries: list[dict] = []
        for e in raw if isinstance(raw, list) else []:
            if not isinstance(e, dict):
                continue
            meta = dict(e.get("meta") or {})
            if e.get("folder") and "folder" not in meta:
                meta["folder"] = e["folder"]
            if e.get("file") and "filename" not in meta:
                meta["filename"] = e["file"]
            entries.append({
                "name": e.get("name"), "sha256": e.get("sha256"),
                "url": e.get("url") or "",
                "purpose": e.get("purpose") or "",
                "license": e.get("license") or "",
                "meta": meta})
        return entries

    def fetch_log(self, peer: Peer, name: str) -> str:
        """One whitelisted operational log from a peer — remote diagnosis:
        the machine that CAN render helps debug the one that cannot. The
        peer enforces its own whitelist; this is only transport. The read
        is CAPPED: a real peer serves at most 32 KiB, and 'the peer is
        not trusted' is this module's doctrine — an impostor streaming an
        endless body must not fill this machine's RAM."""
        with urllib.request.urlopen(
                f"{peer.base}/pf-peer/log/{quote(name, safe='')}",
                timeout=HTTP_TIMEOUT_S) as resp:
            return resp.read(64 * 1024).decode("utf-8", "replace")

    def post_pull(self, peer: Peer, manifest: list[dict]) -> dict[str, Any]:
        """Offer this machine's model manifest to a peer; it queues what
        it is missing and answers with what it did."""
        body = json.dumps({"models": manifest}).encode()
        req = urllib.request.Request(
            peer.base + "/pf-peer/pull", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
