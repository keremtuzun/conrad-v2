"""Shared scenario builders (one world, one set of WorldEntity IDs, sections for 2S / 2T / 2E)."""

from typing import Any

IMPLEMENTATION_METADATA: dict[str, Any] = {
    "implementation_status": "FROZEN_CONTRACT",
    "source_sections": ["ch2 Shared world and scenario", "ch14 Cross-domain 2T/2E interface"],
    "configuration_keys": ["configs/sim/twin2s_default.yaml"],
    "assumptions": ["Material labels are generic SYNTHETIC_ONLY names, not measured material data."],
    "baselines": [],
    "acceptance_tests": ["tests/simulation/twin2s/test_twin2s_scenario_and_observations.py"],
    "claim_status": "IMPLEMENTED",
}
