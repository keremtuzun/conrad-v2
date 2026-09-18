"""Tiny read-only stdlib HTTP server for the console, bound to 127.0.0.1 only (ch37 access boundary).

It serves the pre-rendered page at ``/`` and a data-free ``/healthz``. Every non-GET method is answered
with 405: there are no write endpoints and no command capability.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from conrad.console.assets import CSP
from conrad.console.page import render_run

ALLOWED_HOST = "127.0.0.1"


class BindRefusedError(ValueError):
    """The console only binds to the IPv4 loopback address."""


def _handler(page: bytes) -> type[BaseHTTPRequestHandler]:
    class ConsoleHandler(BaseHTTPRequestHandler):
        server_version = "ConradConsole/1"

        def _reply(self, code: int, body: bytes, ctype: str = "text/plain; charset=utf-8") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            if code == 405:
                self.send_header("Allow", "GET, HEAD")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _host_ok(self) -> bool:
            # DNS-rebinding guard: only loopback host names reach the page.
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
            return host in (ALLOWED_HOST, "localhost")

        def do_GET(self) -> None:
            if not self._host_ok():
                self._reply(403, b"forbidden host\n")
            elif self.path in ("/", "/index.html"):
                self._reply(200, page, "text/html; charset=utf-8")
            elif self.path == "/healthz":
                self._reply(200, b"ok\n")
            else:
                self._reply(404, b"not found\n")

        def do_HEAD(self) -> None:
            self.do_GET()

        def _refuse(self) -> None:
            self._reply(405, b"read-only console: method not allowed\n")

        do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _refuse

        def log_message(self, format: str, *args: object) -> None:
            return

    return ConsoleHandler


class ConsoleServer:
    """A bound, running console server. ``url`` is the page address; ``close()`` stops it."""

    def __init__(self, httpd: ThreadingHTTPServer) -> None:
        self._httpd = httpd
        self._thread = threading.Thread(target=httpd.serve_forever, name="conrad-console", daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        return int(self._httpd.server_address[1])

    @property
    def url(self) -> str:
        return f"http://{ALLOWED_HOST}:{self.port}/"

    def wait(self) -> None:
        """Block until interrupted (for a CLI foreground run)."""
        try:
            self._thread.join()
        except KeyboardInterrupt:
            self.close()

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


def serve_console(
    run_dir: str | Path,
    host: str = ALLOWED_HOST,
    port: int = 0,
    object_store: str | Path | None = None,
    include_truth: bool = False,
) -> ConsoleServer:
    """Render the bundle once, then serve it read-only on 127.0.0.1. Any other bind address is refused."""
    if host != ALLOWED_HOST:
        raise BindRefusedError(f"console binds to {ALLOWED_HOST} only; refused bind address {host!r}")
    page = render_run(run_dir, object_store, include_truth).encode("utf-8")
    httpd = ThreadingHTTPServer((ALLOWED_HOST, port), _handler(page))
    return ConsoleServer(httpd)
