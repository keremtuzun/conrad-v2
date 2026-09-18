"""Operator and Replay Console (spec ch37): a local, read-only, explanatory surface over run bundles.

It verifies a bundle's digests before reading anything, then renders one self-contained HTML page (inline
CSS/JS/SVG, no network). It is never a source of truth or a decision authority and has no command
capability: it does not import the command gateway and never calls ``send``.
"""

from conrad.console.bundle import IntegrityFailure, LoadedBundle, open_bundle
from conrad.console.page import build_console, integrity_page, render_run
from conrad.console.server import BindRefusedError, ConsoleServer, serve_console

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": [
        "ch37 Operator and Replay Console",
        "ch36 Replay, observability and operator review",
        "ch34 run bundles (SS-04)",
    ],
    "configuration_keys": ["paths.object_store (fallback object store for bundle verification)"],
    "assumptions": [
        "static HTML + stdlib http.server replaces the ch37 React/FastAPI stack (no new dependencies allowed)",
        "mission/ and truth/ JSON shapes are not yet frozen; they are read tolerantly and shown generically",
        "active contradictions = belief heads with U_C > 0 plus CONTRADICTS claim edges; no threshold invented",
        "the estimated pose track is the robot_pose_estimate stamped on stored observations",
    ],
    "baselines": [],
    "acceptance_tests": [
        "tests/unit/console/test_console_render.py",
        "tests/unit/console/test_console_safety.py",
    ],
    "claim_status": "IMPLEMENTED",
}

__all__ = [
    "IMPLEMENTATION_METADATA",
    "BindRefusedError",
    "ConsoleServer",
    "IntegrityFailure",
    "LoadedBundle",
    "build_console",
    "integrity_page",
    "open_bundle",
    "render_run",
    "serve_console",
]
