"""Model2T: structural BELIEF child (component-level persistent beliefs, TCDP). BELIEF PLANE.

Never imports conrad.twins, conrad.schemas.truth, conrad.sim or conrad.evaluation (static test).
"""

from __future__ import annotations

from conrad.domains.technical.config import (
    ConditionConfig,
    ContextConfig,
    DirectConfig,
    DynamicsConfig,
    LearnedTCDPConfig,
    Model2TConfig,
    PriorConfig,
    PropagationMode,
    TCDPConfig,
    model2t_config_from_dict,
)
from conrad.domains.technical.engine import StructuralBeliefEngine
from conrad.domains.technical.evidence import structured_evidence
from conrad.domains.technical.model import Model2T, Model2TNotInitialized
from conrad.domains.technical.registry import (
    CORROSION_DEPTH,
    CRACK_LENGTH,
    SURFACE_ANOMALY,
    AssetRegistry,
    ComponentSpec,
    DegradationMechanism,
    RegistryRelation,
)
from conrad.domains.technical.tcdp import relational_contamination

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": [
        "ch10 Model 2T (belief representation, heads, TCDP input/message/gate/update, provenance, "
        "direct vs propagated confidence, anti-contamination, temporal dynamics, truth/model separation, "
        "baselines B0/B1/B2/B5/B8/B9, relational benefit/contamination, temporal benefit)",
        "ch33 Model 2T exact implementation (z_T partitions, heads, 3 mechanism-conditioned TCDP layers)",
        "ch2 Model2Child interface",
    ],
    "configuration_keys": [
        "model2t.direct.{sigma_m,min_reliability,conflict_sigma,conflict_min_reliability,uc_gain,"
        "uc_resolve_factor,ua_smoothing}",
        "model2t.dynamics.{corrosion_rate_m_per_yr,corrosion_rate_sd_m_per_yr,crack_rate_m_per_yr,"
        "crack_rate_sd_m_per_yr,level_process_noise,rate_process_noise,support_timescale_s,"
        "epistemic_growth_per_yr}",
        "model2t.prior.{mean,sd,base_epistemic,unknown_material_epistemic}",
        "model2t.tcdp.{mode,mechanism_relations,correlation,min_source_support,min_gate,max_shift_sd,"
        "generic_correlation,generic_iterations}",
        "model2t.context.{biofouling_keys,turbidity_keys,visibility_keys,uo_gain,ua_gain,surface_variance_gain}",
        "model2t.condition.{nominal_wall_m,crack_critical_m,bands,change_sigma}",
        "model2t.learned_tcdp.{d_node_in,d_z,d_r,d_mech,d_hidden,d_gate,d_head,n_layers,n_mechanisms,"
        "partitions,mutable_partitions,dropout}",
    ],
    "assumptions": [
        "all rates, noise levels, correlations and bands are the model's own ENGINEERING_ESTIMATE guesses; "
        "none is read from Twin2T",
        "per-quantity [level, rate] constant-rate Kalman filter as the engineering prior (TBD-T default)",
        "measurement variance = (sigma_q (1 + aleatoric))^2 / reliability",
        "a repeated independence group may move the mean but never shrinks the variance",
        "reliable conflicts: variance floored at the equal-weight two-hypothesis mixture; U_C += uc_gain",
        "TCDP message = Gaussian conditional under an assumed edge correlation rho, gated by source direct "
        "support; only targets without direct lineage are written",
        "sensor health -> reliability table OK 0.9 / DEGRADED 0.5 / FAULT 0.05 (INVALID)",
        "cross-domain context changes U_O/U_A and surface measurement variance only",
        "a large innovation (> conflict_sigma) inflates the prior variance by innov^2 - S (adaptive process "
        "noise) so genuine changes (crack run-away, repair) are followed; reliable ones also record a conflict",
        "TCDP message shift bounded to max_shift_sd target-prior sd (added after run 1 showed a failed 1 m "
        "crack propagating ~300 mm cracks); GENERIC stays unbounded",
        "2T-E001..E004 executed on SYNTHETIC_ONLY Twin2T data (artifacts/experiments/structural); mixed results",
    ],
    "baselines": [
        "LATEST_OBSERVATION (B1)",
        "SINGLE_FRAME (B0)",
        "INDEPENDENT_COMPONENT (B5/B8: no propagation)",
        "GENERIC_RELATIONAL (B9: mechanism-agnostic)",
        "GRU_TEMPORAL (B2)",
    ],
    "acceptance_tests": [
        "tests/unit/domains/technical",
        "tests/property/domains/technical",
        "conrad.evaluation.structural_experiments (2T-E001..E004)",
    ],
    "claim_status": "EVALUATED",
}

__all__ = [
    "CORROSION_DEPTH",
    "CRACK_LENGTH",
    "IMPLEMENTATION_METADATA",
    "SURFACE_ANOMALY",
    "AssetRegistry",
    "ComponentSpec",
    "ConditionConfig",
    "ContextConfig",
    "DegradationMechanism",
    "DirectConfig",
    "DynamicsConfig",
    "LearnedTCDPConfig",
    "Model2T",
    "Model2TConfig",
    "Model2TNotInitialized",
    "PriorConfig",
    "PropagationMode",
    "RegistryRelation",
    "StructuralBeliefEngine",
    "TCDPConfig",
    "model2t_config_from_dict",
    "relational_contamination",
    "structured_evidence",
]
