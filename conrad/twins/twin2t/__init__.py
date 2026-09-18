"""Twin2T: structural/degradation truth twin + MCDE (Mechanism-Coupled Degradation Engine). TRUTH PLANE.

Must never be imported by conrad.core / domains / decision / active / communication / robotics / runtime.
Only the Twin2T class, scenario tooling, baselines, validators and curricula are exported; hidden mechanism
parameters and transition functions are intentionally NOT part of the public surface (ch10 truth/model
separation).
"""

from __future__ import annotations

from conrad.twins.twin2t.baselines import GeneratorKind, make_engine, mcde_config_for
from conrad.twins.twin2t.config import (
    MCDEConfig,
    ObservationConfig,
    Twin2TConfig,
    config_from_dict,
    load_config,
)
from conrad.twins.twin2t.curriculum import (
    COVERAGE_LEVELS,
    TIERS,
    SensorDegradationCurriculum,
    contradiction_pair,
    generate_scenario,
    tier_mcde_config,
)
from conrad.twins.twin2t.events import StructuralEventType
from conrad.twins.twin2t.mcde import MCDE, MCDEStepResult, StructuralEvent
from conrad.twins.twin2t.observation import FidelityLevel, SensorLevelHookMissing, coverage_visibility
from conrad.twins.twin2t.registry import Mechanism, RelationshipType
from conrad.twins.twin2t.scenario import build_small_scenario, populate_structural_state
from conrad.twins.twin2t.state import STATE_DIMENSIONS, DegradationStatus
from conrad.twins.twin2t.twin import Twin2T
from conrad.twins.twin2t.validators import NOT_EVALUABLE, ValidationReport, reality_gap, validate_sequence

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": [
        "ch10 Model 2T and Twin2T (hierarchy, relationships, degradation state V1, masks, MCDE, events, "
        "fidelity levels, coverage, contradiction, sensor degradation, truth/model separation)",
        "ch11 Twin2T structural generator design and validation",
        "ch33 Model 2T and Twin2T exact implementation (tick order, logged prior, no learned residual)",
        "ch31 H-T2T-01 MCDE hypothesis and baselines",
    ],
    "configuration_keys": [
        "twin2t.mcde.{mechanism_coupling,topology_coupling,stochastic,events_enabled,interventions_enabled,"
        "environment_conditioning,loading_conditioning,corrosion_model,max_stress_concentration,"
        "support_load_redistribution,crack_coating_length_scale_m,max_crack_length_m,"
        "reference_stress_range_pa,reference_cycles_per_s}",
        "twin2t.observation.{visibility_threshold,default_fidelity,feature_dim,wall_loss_sigma_m,anomaly_sigma,"
        "crack_sigma_m,crack_detection_limit_m,turbidity_noise_gain,biofouling_noise_gain,"
        "corruption_noise_gain,contradiction_magnitude}",
        "twin2t.prior_overrides",
        "twin2t.generator",
        "twin2t.clock_domain",
        "twin2t.generator_version",
    ],
    "assumptions": [
        "corrosion: power-law (ISO 9224/Melchers) or bilinear (Melchers 2003) forms; modifier = "
        "q10^((T-10)/10) * DO/7 * burial * CP residual; parameter ranges are ENGINEERING_ESTIMATE, uncalibrated",
        "coating breakdown: DNV-RP-B401 linear f_c = a + b t, monotone until renewal",
        "fatigue: Paris-Erdogan with dK threshold, dK = Y dsigma sqrt(pi a), closed-form cycle integration; "
        "no crack initiation model (cracks come from the initial state or IMPACT events)",
        "stochastic residual N(0, sigma sqrt(dt_yr)) truncated so loss never reverses without intervention",
        "couplings: wall loss -> net-section stress (t0/(t-d)); coating breakdown -> exposed fraction; "
        "crack -> coating breach; support wall loss -> load redistribution (SUPPORTED_BY/LOAD_TRANSFER only); "
        "environment change shared over CONNECTED_TO/ATTACHED_TO/CONTACTS; ADJACENT_TO carries nothing",
        "per-component random streams from (seed, entity UUID)",
        "visibility absent from SensingContext.degradation means NOT observable",
        "concrete and HDPE have no valid V1 mechanism (fully masked)",
    ],
    "baselines": [g.value for g in GeneratorKind],
    "acceptance_tests": [
        "tests/unit/twins/twin2t",
        "tests/property/twins/twin2t",
        "tests/simulation/twin2t",
    ],
    "claim_status": "IMPLEMENTED",
}

__all__ = [
    "COVERAGE_LEVELS",
    "IMPLEMENTATION_METADATA",
    "MCDE",
    "NOT_EVALUABLE",
    "STATE_DIMENSIONS",
    "TIERS",
    "DegradationStatus",
    "FidelityLevel",
    "GeneratorKind",
    "MCDEConfig",
    "MCDEStepResult",
    "Mechanism",
    "ObservationConfig",
    "RelationshipType",
    "SensorDegradationCurriculum",
    "SensorLevelHookMissing",
    "StructuralEvent",
    "StructuralEventType",
    "Twin2T",
    "Twin2TConfig",
    "ValidationReport",
    "build_small_scenario",
    "config_from_dict",
    "contradiction_pair",
    "coverage_visibility",
    "generate_scenario",
    "load_config",
    "make_engine",
    "mcde_config_for",
    "populate_structural_state",
    "reality_gap",
    "tier_mcde_config",
    "validate_sequence",
]
