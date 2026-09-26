"""OS-FM P3 smoke namespace registration stub.

This module does not publish research results.  P3 smoke evidence is produced by
`conrad train run` immutable run directories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run(config: dict[str, Any], seeds: list[int], out_dir: Path) -> dict[str, Any]:
    out = {"registered": 1.0, "seeds": float(len(seeds)), "smoke_namespace": "OSFM"}
    (out_dir / "osfm_registry_stub.json").write_text(str(out), encoding="utf-8")
    return out

