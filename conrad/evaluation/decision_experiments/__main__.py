from __future__ import annotations

import sys

from conrad.evaluation.decision_experiments import run_all

if __name__ == "__main__":
    run_all(sys.argv[1] if len(sys.argv) > 1 else "artifacts/experiments/decision")
