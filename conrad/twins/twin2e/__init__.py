"""Twin2E: ecological/environmental truth world with the MEIFE candidate engine. TRUTH PLANE ONLY.

Belief/decision packages must never import this package (tests/leakage).
"""

from __future__ import annotations

from conrad.twins.twin2e.baselines import ABLATIONS, BASELINE_SWITCHES, baseline_config
from conrad.twins.twin2e.config import (
    BoundaryCondition,
    MeifeSwitches,
    ObservationLevel,
    Twin2EConfig,
    load_config,
)
from conrad.twins.twin2e.counterfactual import (
    CONFOUNDER_KINDS,
    CounterfactualPair,
    make_confounded_variant,
    make_counterfactual_pair,
)
from conrad.twins.twin2e.observation import Twin2EObservationLevelError
from conrad.twins.twin2e.priors import Twin2EScenarioError, populate_ecological_state
from conrad.twins.twin2e.residual import LearnedResidualNotCalibratedError
from conrad.twins.twin2e.scenario_builder import build_small_scenario
from conrad.twins.twin2e.solver import CFLViolationError, advect_diffuse
from conrad.twins.twin2e.twin import Twin2E, Twin2ENotInitializedError
from conrad.twins.twin2e.validators import validate_twin

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": [
        "ch12 Model 2E and Twin2E (Twin side: fields, entities, MEIFE, disturbances, recovery, observation levels, partial surveys, don't fake synchronization)",
        "ch13 Twin2E ecological and environmental world design",
        "ch33 Model 2E and Twin2E exact implementation (Twin2E freeze)",
        "H-T2E-01 MEIFE baselines and counterfactual test",
    ],
    "configuration_keys": [
        "twin2e.regional_grid",
        "twin2e.local_grid",
        "twin2e.switches.*",
        "twin2e.solver.*",
        "twin2e.ecology.*",
        "twin2e.observation.*",
        "twin2e.learned_residual.*",
        "twin2e.clock_domain",
        "twin2e.allow_prior_sampling",
    ],
    "assumptions": [
        "Twin2E is a controlled synthetic ecological/environmental generator; it does NOT claim comprehensive or accurate marine ecosystem simulation.",
        "No mechanism or prior range is calibrated against real data yet; every unspecified value is sampled from a logged ENGINEERING_ESTIMATE/SYNTHETIC_ONLY prior.",
        "Fields: temperature, turbidity (advected-diffused, first-order upwind finite volume), current (prescribed 1/7-power profile + OU fluctuation, not dynamically solved), light (diagnostic Beer-Lambert with diel cycle).",
        "Flat seabed at each grid's bottom face and a flat sea surface; Twin2S owns canonical geometry and Twin2E only reads entity positions from shared context.",
        "Regional->local coupling by open-boundary inflow plus nudging; local->regional by slow relaxation of overlapped regional cells.",
        "Biofouling is a surface-cover process only: it affects observability, never implies corrosion, and never writes Twin2T state.",
        "Scenario events inside a step are applied at the start of that step.",
        "Ecological time may be accelerated by ecology.time_scale (default 1.0); any acceleration is a logged config value.",
        "False-positive detections in surveys are not modelled; absent mobile groups are never reported.",
        "E2 (sensor/pixel) observations are renderer-owned and refused by Twin2E.",
        "No learned Twin2E component: the residual hook is disabled and raises if enabled without calibration data.",
    ],
    "baselines": sorted(BASELINE_SWITCHES) + sorted(ABLATIONS),
    "acceptance_tests": [
        "tests/unit/twins/twin2e",
        "tests/property/twins/twin2e",
        "tests/simulation/twin2e",
    ],
    "claim_status": "IMPLEMENTED",
}

__all__ = [
    "ABLATIONS",
    "BASELINE_SWITCHES",
    "CONFOUNDER_KINDS",
    "IMPLEMENTATION_METADATA",
    "BoundaryCondition",
    "CFLViolationError",
    "CounterfactualPair",
    "LearnedResidualNotCalibratedError",
    "MeifeSwitches",
    "ObservationLevel",
    "Twin2E",
    "Twin2EConfig",
    "Twin2ENotInitializedError",
    "Twin2EObservationLevelError",
    "Twin2EScenarioError",
    "advect_diffuse",
    "baseline_config",
    "build_small_scenario",
    "load_config",
    "make_confounded_variant",
    "make_counterfactual_pair",
    "populate_ecological_state",
    "validate_twin",
]
