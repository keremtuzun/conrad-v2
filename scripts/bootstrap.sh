#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --all-groups && uv run conrad db migrate && uv run conrad doctor
