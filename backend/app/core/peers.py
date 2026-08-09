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
assets, photos, jobs or settings. Discovery is a tiny UDP beacon; peers
vanish from the list when their beacons stop.
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
from typing import Any
from urllib.parse import quote, unquote

log = logging.getLogger("promptforge.peers")

BEACON_INTERVAL_S = 6.0
PEER_STALE_S = 25.0
HTTP_TIMEOUT_S = 10.0


class Peer:
    def __init__(self, token: str, name: str, host: str, port: int,
                 static: bool = False):
        self.token = token
        self.name = name
        self.host = host
        self.port = port
        self.static = static      # added by hand/env: never pruned
        self.last_seen = time.time()

    @property
    def base(self) -> str:
        return f"http://{self.host}:{self.port}"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "host": self.host, "port": self.port,
                "static": self.static,
                "seen_ago_s": round(time.time() - self.last_seen, 1)}


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
                 env_provider: Callable[[], dict] | None = None):
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
        # Every address that ever answered as a PromptForge: the scanner
        # keeps re-probing these, so a peer that reboots comes back on its
        # own without waiting for a beacon to make it through.
        self.known_hosts: set[tuple[str, int]] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._httpd: ThreadingHTTPServer | None = None
        self._index_cache: dict[str, tuple[float, list[dict]]] = {}

    # ------------------------------------------------------------------ life
    def start(self) -> None:
        if not (self.share or self.render):
            return
        self._stop.clear()
        self._start_http()
        workers = [self._beacon_tx, self._beacon_rx]
        if not self.loopback_only:
            workers.append(self._scanner)
        for target in workers:
            t = threading.Thread(target=target, daemon=True,
                                 name=f"pf-peer-{target.__name__}")
            t.start()
            self._threads.append(t)
        # Hand-configured peers (PROMPTFORGE_PEER_HOSTS): probed over plain
        # HTTP, so they work even where UDP broadcasts never arrive.
        for entry in self.static_hosts:
            host, _, port = entry.strip().partition(":")
            if host:
                threading.Thread(
                    target=self.add_peer,
                    args=(host, int(port) if port.isdigit() else 8765),
                    daemon=True, name="pf-peer-static").start()
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

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):  # keep the console quiet
                pass

            def _json(self, code: int, payload: Any) -> None:
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

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

        last_error: Exception | None = None
        for port in range(self.http_port, self.http_port + 10):
            try:
                self._httpd = ThreadingHTTPServer((bind_host, port), Handler)
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

    def _handle_get(self, req: BaseHTTPRequestHandler) -> None:
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
            req._json(200, {"app": "promptforge", "name": self.name,
                            "token": self.token, "share": self.share,
                            "render": self.render,
                            "idle": not self.busy_check(),
                            "comfy": comfy or {"up": False, "device": None,
                                               "gpu": None},
                            "comfy_env": env,
                            "stats": stats})
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
    # walk: install/crash logs only, no jobs, no prompts-carrying app DBs.
    LOG_WHITELIST = frozenset({
        "comfyui.log", "comfyui-err.log", "comfyui-repair.log",
        "directml-install.log", "torch-cuda-repair.log", "sage-install.log",
        "sam-install.log", "backend-live.log", "backend-live-err.log",
        "doctor-report.txt",
    })

    def _serve_log(self, req: BaseHTTPRequestHandler, name: str) -> None:
        if not self.share:
            req._json(403, {"error": "sharing is off"})
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

    def _handle_post(self, req: BaseHTTPRequestHandler) -> None:
        path = req.path.split("?", 1)[0]
        if path.startswith("/pf-peer/comfy/"):
            length = int(req.headers.get("Content-Length") or 0)
            body = req.rfile.read(length) if length else b""
            self._proxy(req, body=body)
            return
        if path == "/pf-peer/pull":
            # A peer offers its model manifest; THIS machine decides what
            # to queue. Only sha-pinned entries are ever accepted, and the
            # downloads themselves run through the normal registry path
            # (LAN first, checksum-verified) as visible jobs.
            if not self.share:
                req._json(403, {"error": "model sharing is off"})
                return
            if self.on_pull is None:
                req._json(501, {"error": "no pull handler on this build"})
                return
            length = int(req.headers.get("Content-Length") or 0)
            try:
                body = json.loads(req.rfile.read(length).decode()
                                  if length else "{}")
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
                        "sha256": m.sha256 or ""})
        return out

    def _serve_model(self, req: BaseHTTPRequestHandler, name: str) -> None:
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
                start = int(s) if s else 0
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

    def _proxy(self, req: BaseHTTPRequestHandler, body: bytes | None) -> None:
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
                ips.add(info[4][0])
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
                socks = []
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
            try:
                msg = json.loads(data.decode())
            except (ValueError, UnicodeDecodeError):
                continue
            if msg.get("pf") != 1 or msg.get("token") == self.token:
                continue
            with self._lock:
                peer = self.peers.get(msg["token"])
                if peer is None:
                    self.peers[msg["token"]] = Peer(
                        msg["token"], str(msg.get("name") or host),
                        host, int(msg.get("http") or 8765))
                    log.info("discovered peer '%s' at %s",
                             msg.get("name"), host)
                else:
                    peer.last_seen = time.time()
                    peer.host = host
                    peer.port = int(msg.get("http") or peer.port)
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
                for host, port in list(self.known_hosts):
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

    # ---------------------------------------------------------------- client
    def peers_list(self) -> list[Peer]:
        self._prune()
        with self._lock:
            return list(self.peers.values())

    def add_peer(self, host: str, port: int = 8765,
                 timeout: float = 5.0,
                 pin: bool = True) -> dict[str, Any] | None:
        """Connect to a peer by address.

        Probes /pf-peer/info over plain HTTP — the escape hatch for
        networks where UDP broadcasts never arrive. A machine added BY
        HAND (or from the environment) is pinned and never pruned; one the
        scanner found is not, so it disappears when it really goes away
        and reappears when the scanner sees it again."""
        try:
            with urllib.request.urlopen(
                    f"http://{host}:{int(port)}/pf-peer/info",
                    timeout=timeout) as resp:
                info = json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001
            log.info("peer probe %s:%s failed: %s", host, port, exc)
            return None
        if info.get("app") != "promptforge":
            return None
        if info.get("token") == self.token:
            return {"self": True, **info}
        self.known_hosts.add((host, int(port)))
        with self._lock:
            peer = self.peers.get(info["token"])
            if peer is None:
                self.peers[info["token"]] = Peer(
                    info["token"], str(info.get("name") or host),
                    host, int(port), static=pin)
                log.info("peer '%s' connected at %s:%s",
                         info.get("name"), host, port)
            else:
                peer.static = peer.static or pin
                peer.host = host
                peer.port = int(port)
                peer.last_seen = time.time()
        return info

    def _peer_json(self, peer: Peer, path: str) -> Any:
        with urllib.request.urlopen(peer.base + path,
                                    timeout=HTTP_TIMEOUT_S) as resp:
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
        it would take longer than waiting for this machine."""
        cpu_fallback: Peer | None = None
        for peer in self.peers_list():
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

    def post_pull(self, peer: Peer, manifest: list[dict]) -> dict[str, Any]:
        """Offer this machine's model manifest to a peer; it queues what
        it is missing and answers with what it did."""
        body = json.dumps({"models": manifest}).encode()
        req = urllib.request.Request(
            peer.base + "/pf-peer/pull", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
