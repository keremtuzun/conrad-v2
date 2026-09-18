"""Fail when a tracked file looks like it contains a credential, or when a forbidden file is tracked."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "aws access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "anthropic/openai style key": re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
    "slack token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    "assigned secret": re.compile(
        r"(?i)\b(password|passwd|secret|api_key|apikey|token)\b\s*[:=]\s*[\"'][^\"'\s]{12,}[\"']"
    ),
}
FORBIDDEN_NAMES = re.compile(r"(^|/)(\.env(\..*)?|.*\.pem|.*\.key|id_rsa|credentials\.json)$")
FORBIDDEN_PREFIXES = ("artifacts/runs/", "artifacts/objects/", "mlruns/")
SKIP_SUFFIXES = (".png", ".jpg", ".npy", ".lock", ".sqlite")


def main() -> int:
    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split("\n")
    problems: list[str] = []
    for name in filter(None, files):
        if FORBIDDEN_NAMES.search(name):
            problems.append(f"{name}: forbidden file is tracked")
        if name.startswith(FORBIDDEN_PREFIXES):
            problems.append(f"{name}: run artifact is tracked")
        if name.endswith(SKIP_SUFFIXES) or name == "scripts/secret_scan.py":
            continue
        try:
            text = Path(name).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                problems.append(f"{name}: possible {label}")
    for p in problems:
        print("SECRET-SCAN:", p)
    print(f"secret scan: {len(problems)} problem(s) in {len(files)} tracked files")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
