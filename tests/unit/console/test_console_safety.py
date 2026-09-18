"""The console is read-only: loopback-only bind, no write endpoints, no command path (ch37 access boundary)."""

from __future__ import annotations

import ast
import http.client
from pathlib import Path

import pytest

from conrad.console import BindRefusedError, serve_console

CONSOLE = Path(__file__).resolve().parents[3] / "conrad" / "console"


def test_console_never_imports_gateway_or_calls_send() -> None:
    violations = []
    files = sorted(CONSOLE.rglob("*.py"))
    assert files
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or "", *(a.name for a in node.names)]
            violations += [f"{path.name}: imports {n}" for n in names if "command_gateway" in n]
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else fn.id if isinstance(fn, ast.Name) else ""
                if name in ("send", "submit", "set_thruster_commands"):
                    violations.append(f"{path.name}:{node.lineno} calls {name}")
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("host", ["0.0.0.0", "", "localhost", "::", "192.168.1.10"])
def test_server_refuses_non_loopback_bind(bundle_dir: Path, host: str) -> None:
    with pytest.raises(BindRefusedError):
        serve_console(bundle_dir, host=host)


def test_server_is_read_only(bundle_dir: Path) -> None:
    server = serve_console(bundle_dir)
    try:
        assert server.url.startswith("http://127.0.0.1:")
        conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
        conn.request("GET", "/")
        resp = conn.getresponse()
        body = resp.read().decode("utf-8")
        assert resp.status == 200 and "belief-status-timeline" in body
        assert "default-src 'none'" in (resp.getheader("Content-Security-Policy") or "")
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            conn.request(method, "/", body=b"{}")
            r = conn.getresponse()
            r.read()
            assert r.status == 405, method
        conn.request("GET", "/api/v1/operator/start")
        r = conn.getresponse()
        r.read()
        assert r.status == 404
        conn.request("GET", "/", headers={"Host": "evil.example"})
        r = conn.getresponse()
        r.read()
        assert r.status == 403
        conn.close()
    finally:
        server.close()
