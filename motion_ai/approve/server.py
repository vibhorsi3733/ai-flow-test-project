"""Local review server (stdlib http.server)."""

from __future__ import annotations

import json
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .decision import compute_publish_decision
from .media import MediaIndex
from .page import render_page
from .store import (ApprovalError, file_sha256, load_approvals, record_clip,
                    record_package, save_approvals)

MAX_BODY = 16 * 1024
CHUNK = 256 * 1024
_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


def parse_range(header: str | None, size: int):
    """Return (start, end) inclusive for a single 'bytes=' range, None for no range,
    or 'invalid' when unsatisfiable."""
    if not header:
        return None
    m = _RANGE.match(header.strip())
    if not m or (not m.group(1) and not m.group(2)) or size <= 0:
        return "invalid"
    a, b = m.group(1), m.group(2)
    if a == "":  # suffix range: last N bytes
        n = int(b)
        if n == 0:
            return "invalid"
        return max(0, size - n), size - 1
    start = int(a)
    end = min(int(b), size - 1) if b else size - 1
    if start >= size or end < start:
        return "invalid"
    return start, end


class ReviewApp:
    """Holds the manifest (read once, never written) and the approvals state."""

    def __init__(self, manifest_path: Path, clips_dir: Path):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        self.manifest_sha = file_sha256(self.manifest_path)
        self.media = MediaIndex(self.manifest_path, self.manifest, clips_dir)
        self.lock = threading.Lock()

    def state(self):
        approvals = load_approvals(self.manifest_path)
        return approvals, compute_publish_decision(self.manifest, approvals)

    def page(self) -> str:
        approvals, decision = self.state()
        saved_sha = approvals.get("manifest_sha256")
        changed = bool(approvals.get("history")) and saved_sha not in (None, self.manifest_sha)
        return render_page(self.manifest, approvals, decision, self.media,
                           manifest_name=self.manifest_path.name, manifest_changed=changed)

    def decide(self, scope: str, body: dict) -> dict:
        with self.lock:
            approvals = load_approvals(self.manifest_path)
            if scope == "clip":
                entry = record_clip(self.manifest, approvals, body.get("clip_id"),
                                    body.get("decision"), body.get("reviewer"), body.get("note"))
            else:
                entry = record_package(approvals, body.get("decision"),
                                       body.get("reviewer"), body.get("note"))
            approvals = save_approvals(self.manifest_path, self.manifest, approvals)
        return {"entry": entry, "decision": approvals["publish_decision"]}


def make_handler(app: ReviewApp, allowed_hosts: set[str]):
    class Handler(BaseHTTPRequestHandler):
        server_version = "MotionAIReview/1"

        def log_message(self, fmt, *args):  # quieter default logging
            print("[approve] " + (fmt % args))

        def _host_ok(self) -> bool:
            # Blocks DNS-rebinding: only answer to the names we bound.
            host = (self.headers.get("Host") or "").lower()
            return host in allowed_hosts

        def _send(self, code, body: bytes, ctype: str, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            if not self._host_ok():
                return self._send(HTTPStatus.FORBIDDEN, b"forbidden host", "text/plain")
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                return self._send(HTTPStatus.OK, app.page().encode(), "text/html; charset=utf-8",
                                  {"Content-Security-Policy":
                                   "default-src 'self'; style-src 'unsafe-inline'; "
                                   "script-src 'unsafe-inline'; media-src 'self'"})
            if path == "/api/state":
                approvals, decision = app.state()
                return self._json(HTTPStatus.OK, {"approvals": approvals, "decision": decision})
            if path.startswith("/media/"):
                return self._media(path[len("/media/"):])
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

        def _media(self, key: str):
            fpath = app.media.resolve(key)
            if fpath is None:
                return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            size = fpath.stat().st_size
            rng = parse_range(self.headers.get("Range"), size)
            if rng == "invalid":
                return self._send(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, b"", "text/plain",
                                  {"Content-Range": f"bytes */{size}"})
            start, end = rng if rng else (0, size - 1)
            length = end - start + 1 if size else 0
            self.send_response(HTTPStatus.PARTIAL_CONTENT if rng else HTTPStatus.OK)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            self.send_header("X-Content-Type-Options", "nosniff")
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            try:
                with open(fpath, "rb") as f:
                    f.seek(start)
                    left = length
                    while left > 0:
                        buf = f.read(min(CHUNK, left))
                        if not buf:
                            break
                        self.wfile.write(buf)
                        left -= len(buf)
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser cancelled the range request; normal for video

        def do_POST(self):
            if not self._host_ok():
                return self._send(HTTPStatus.FORBIDDEN, b"forbidden host", "text/plain")
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc.lower() not in allowed_hosts:
                return self._json(HTTPStatus.FORBIDDEN, {"error": "cross-origin request refused"})
            if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/json":
                return self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "JSON body required"})
            path = urlsplit(self.path).path
            scope = {"/api/clip": "clip", "/api/package": "package"}.get(path)
            if scope is None:
                return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if n <= 0 or n > MAX_BODY:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "bad body size"})
            try:
                body = json.loads(self.rfile.read(n))
                if not isinstance(body, dict):
                    raise ValueError("object expected")
            except ValueError as e:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": f"invalid JSON: {e}"})
            try:
                result = app.decide(scope, body)
            except ApprovalError as e:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
            self._json(HTTPStatus.OK, result)

    return Handler


def serve(manifest_path: Path, clips_dir: Path, host: str = "127.0.0.1", port: int = 8765):
    app = ReviewApp(manifest_path, clips_dir)
    allowed: set[str] = set()  # filled after bind so port 0 (tests) works
    httpd = ThreadingHTTPServer((host, port), make_handler(app, allowed))
    real_port = httpd.server_address[1]
    names = {host, "127.0.0.1", "localhost"} if host in ("127.0.0.1", "localhost") else {host}
    allowed.update(f"{n}:{real_port}" for n in names)
    if real_port == 80:
        allowed.update(names)
    return app, httpd
