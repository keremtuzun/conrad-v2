#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# usage: scripts/eval.sh <EXPERIMENT_ID> [config]
uv run conrad eval run --experiment "$1" ${2:+--config "$2"}
