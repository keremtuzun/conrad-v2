#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run conrad eval run --run "$1" --config "${2:-configs/eval/default.yaml}"
