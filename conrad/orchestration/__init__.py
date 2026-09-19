"""Deployment-side orchestration: Belief Bus and the supervised MissionRuntime (gates I1-I7, flagship I4).

DEPLOYMENT PLANE: never imports conrad.twins, conrad.sim, conrad.schemas.truth or conrad.evaluation.
"""

from __future__ import annotations

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": [
        "ch2 Belief Bus",
        "ch16-ch20 Model1 / MCBR / BAAC / navigation integration",
        "ch34 runtime process model, command gateway",
        "docs/development/INTEGRATION_BRIEF.md",
    ],
    "configuration_keys": ["sim.mission.runtime.* (MissionRuntimeConfig)"],
    "assumptions": [
        "survey-then-resolve mission structure: MCBR needs are deferred until the lane survey completes",
        "requirements target each segment's surveyed mid-span inspection station",
        "MCBR predicted visibility = belief-map visibility of the still-UNKNOWN registry design surface",
        "UNKNOWN cells block MCBR view prediction with a configured per-cell probability (ENGINEERING_ESTIMATE)",
        "structural association is ambiguous (NO_MATCH) when two design surfaces are within one measurement sigma",
        "BAAC offers are coalesced to the latest revision per belief with a minimum re-offer interval",
    ],
    "baselines": ["A-B1_fixed_inspection planner (FLAGSHIP-I4-FIXEDVIEW)"],
    "acceptance_tests": ["tests/acceptance", "tests/integration", "tests/regression/test_adversarial.py"],
    "claim_status": "IMPLEMENTED",
}
