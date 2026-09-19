"""Integrated intelligence benchmarks INT-001..INT-010 and the flagship I4 comparison (spec ch20, ch25).

EVALUATION PLANE: reads run bundles and the evaluation-only truth record.
"""

from __future__ import annotations

EXPERIMENTS = {
    "INT-BENCH": ("conrad.evaluation.int_benchmarks.runner", "configs/sim/mission_default.yaml"),
}

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": ["ch20 Integrated intelligence benchmarks", "ch25 gate I4", "ch28 Acceptance Records"],
    "configuration_keys": ["sim.mission.*", "scenarios"],
    "assumptions": [
        "INT benchmark definitions are one-line in the spec; the concrete fault/degradation choices "
        "live in conrad.sim.mission.scenarios and are SYNTHETIC_ONLY"
    ],
    "baselines": ["FLAGSHIP-I4-FIXEDVIEW"],
    "acceptance_tests": ["tests/acceptance/test_flagship_mission.py"],
    "claim_status": "IMPLEMENTED",
}
