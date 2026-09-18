#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run conrad runtime start --config "${1:-configs/runtime/default.yaml}"
