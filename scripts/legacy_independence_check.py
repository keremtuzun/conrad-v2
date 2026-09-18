"""Legacy independence acceptance test (prompt s6, s128).

1. optionally rename the legacy repository away (restored in a ``finally`` block)
2. clean-clone conrad-v2 into a temporary directory (no caches, no .venv)
3. uv sync --all-groups --locked; lint; type-check; tests; conrad doctor; simulation and replay smoke
4. write a JSON report

Usage: python scripts/legacy_independence_check.py [--legacy PATH] [--rename] [--report PATH] [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEGACY = Path.home() / "OneDrive" / "Documents" / "ChatGPT" / "conrad"


def uv_cmd() -> list[str]:
    exe = shutil.which("uv")
    return [exe] if exe else [sys.executable, "-m", "uv"]


def step(name: str, cmd: list[str], cwd: Path, env: dict[str, str], results: list[dict[str, object]]) -> bool:
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
    results.append(
        {
            "step": name,
            "cmd": " ".join(cmd[-6:]),
            "exit": proc.returncode,
            "seconds": round(time.time() - t0, 1),
            "tail": tail,
        }
    )
    print(
        f"[{'OK' if proc.returncode == 0 else 'FAIL'}] {name} ({results[-1]['seconds']}s) {tail[-1] if tail else ''}"
    )
    return proc.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", default=str(DEFAULT_LEGACY))
    ap.add_argument("--rename", action="store_true", help="rename the legacy repo away during the check")
    ap.add_argument("--report", default=str(ROOT / "artifacts" / "legacy_independence" / "report.json"))
    ap.add_argument("--quick", action="store_true", help="skip the full test suite (contract+leakage only)")
    args = ap.parse_args()
    legacy = Path(args.legacy)
    moved = legacy.with_name(legacy.name + ".__moved_for_independence_check__")
    results: list[dict[str, object]] = []
    renamed = False
    ok = True
    try:
        if args.rename and legacy.exists():
            legacy.rename(moved)
            renamed = True
        legacy_absent = not legacy.exists()
        with tempfile.TemporaryDirectory(prefix="conrad-v2-clean-") as tmp:
            clone = Path(tmp) / "conrad-v2"
            env = {
                k: v
                for k, v in os.environ.items()
                if k not in ("VIRTUAL_ENV", "PYTHONPATH", "UV_PROJECT_ENVIRONMENT")
            }
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(ROOT), str(clone)], check=True)
            uv = uv_cmd()
            plan = [
                ("uv sync --all-groups --locked", [*uv, "sync", "--all-groups", "--locked"]),
                ("ruff format --check", [*uv, "run", "ruff", "format", "--check", "."]),
                ("ruff check", [*uv, "run", "ruff", "check", "."]),
                ("mypy conrad tests", [*uv, "run", "mypy", "conrad", "tests"]),
                (
                    "tests",
                    [
                        *uv,
                        "run",
                        "pytest",
                        "-q",
                        "-p",
                        "no:cacheprovider",
                        *(["tests/contract", "tests/leakage"] if args.quick else []),
                    ],
                ),
                ("conrad db migrate", [*uv, "run", "conrad", "db", "migrate"]),
                ("conrad doctor", [*uv, "run", "conrad", "doctor"]),
                (
                    "simulation smoke",
                    [
                        *uv,
                        "run",
                        "conrad",
                        "sim",
                        "run",
                        "--scenario",
                        "GOLDEN-SMOKE",
                        "--config",
                        "configs/sim/default.yaml",
                        "--run-id",
                        "independence-smoke",
                    ],
                ),
                ("replay smoke", [*uv, "run", "conrad", "replay", "run", "--run", "independence-smoke"]),
                (
                    "core training smoke",
                    [*uv, "run", "conrad", "train", "run", "--config", "configs/train/core_smoke.yaml"],
                ),
            ]
            for name, cmd in plan:
                ok = step(name, cmd, clone, env, results) and ok
    finally:
        if renamed:
            moved.rename(legacy)
    report = {
        "legacy_path": str(legacy),
        "legacy_renamed_during_check": renamed,
        "legacy_absent_during_check": legacy_absent if "legacy_absent" in locals() else None,
        "legacy_restored": legacy.exists() if renamed else None,
        "steps": results,
        "ok": ok,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("LEGACY INDEPENDENCE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
