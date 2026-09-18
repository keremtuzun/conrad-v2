#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run conrad train run --config "${1:-configs/train/core_smoke.yaml}"
