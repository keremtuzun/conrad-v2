"""Truth side of the integrated mission (Phases 5-11, gates I1-I7, flagship I4). TRUTH PLANE."""

from __future__ import annotations

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": [
        "ch20 Integrated intelligence benchmarks INT-001..INT-010",
        "ch25 gates I1-I7",
        "ch35 Priority 1 vertical slice, Priority 3 golden suite",
        "docs/development/INTEGRATION_BRIEF.md",
    ],
    "configuration_keys": ["sim.mission.world.*", "sim.mission.runtime.*"],
    "assumptions": [
        "Twin2S generic labels are mapped to Twin2T SEGMENT/WELD/SUPPORT and carbon_steel/concrete",
        "the hidden defect is Twin2T initial corrosion depth + crack length of the target segment; its "
        "observability is the Twin2S visibility of a far-side (+Y) surface patch",
        "the transit lane is defined here (-Y of the pipe axis); the shared scenario has none",
        "structural inspection payload FOV/range are SYNTHETIC_ONLY settings",
    ],
    "baselines": ["FLAGSHIP-I4-FIXEDVIEW (A-B1_fixed_inspection)"],
    "acceptance_tests": ["tests/acceptance", "tests/integration/test_i1_spatial_loop.py"],
    "claim_status": "IMPLEMENTED",
}
