"""Model 2E: ecological/environmental BELIEF child with Coupled Entity-Field Dynamics (ch12, ch33).

BELIEF PLANE: never imports conrad.twins, conrad.schemas.truth, conrad.sim or conrad.evaluation.
Entity beliefs and field beliefs are two different state types and are never merged.
"""

from __future__ import annotations

from conrad.domains.ecological.baselines import BASELINES, baseline_config, make_baseline
from conrad.domains.ecological.cefd_analytic import DAMAGE_CLAIM, STRESS_CLAIM, AnalyticCEFD
from conrad.domains.ecological.config import (
    BeliefGridConfig,
    CefdSwitches,
    CouplingConfig,
    EntityConfig,
    FieldSpec,
    Model2EConfig,
    load_model2e_config,
)
from conrad.domains.ecological.ecmer_e import EcologicalEncoder, evidence_kind
from conrad.domains.ecological.entity_belief import EntityBelief, EntityBeliefStore, classify_entity_type
from conrad.domains.ecological.field_belief import FieldBelief, FieldBeliefGrid
from conrad.domains.ecological.model2e import Model2E
from conrad.domains.ecological.units import UnitError

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": [
        "ch12 Model 2E (two latent types, ECMER-E, association, CEFD, multi-timescale, UEI, coupling benefit)",
        "ch33 Model 2E exact implementation (entity graph-transformer + 3D U-Net + gated cross-attention)",
        "ch2 Model2Child interface, ch28 knowledge status / provenance",
    ],
    "configuration_keys": [
        "model2e.grid.*",
        "model2e.fields.<name>.*",
        "model2e.entity.*",
        "model2e.coupling.*",
        "model2e.switches.*",
        "model2e.field_model.*",
        "learned CEFD: conrad.domains.ecological.learned.LearnedCEFDConfig",
    ],
    "assumptions": [
        "field belief = hierarchical empirical-Bayes kriging over sensor stations (unknown level + optional "
        "depth trend + SE-kernel local deviation); tau^2, length scales, drift and noise scale learned per run",
        "field temporal model = per-station random walk with an EB drift rate over physical delta_t; "
        "turbidity/temperature local_sd priors are EB estimates from the 2E DEV partition (SYNTHETIC_ONLY)",
        "cover temporal model = bounded random walk with no trend; stress only inflates its process noise",
        "late evidence is applied without rewind, with measurement variance inflated by the lag",
        "survey measurement noise from the turbidity BELIEF via a beam-attenuation law (ENGINEERING_ESTIMATE)",
        "thermal stress likelihood = P(T > threshold) under the temperature belief; INFERRED only",
        "ecological_damage is always UNKNOWN: no analytic decoder of condition from E0/E1 evidence",
        "E1 feature surveys only count as hits; cover decoding from features needs a trained ECMER-E head",
        "mobile-group non-detections are not emitted as negative evidence, so presence decays to the prior",
        "entity->field effect = filtration sink on turbidity; expected to be small",
        "predict() returns PREDICTED messages without mutating or persisting belief state",
    ],
    "baselines": sorted(BASELINES),
    "acceptance_tests": ["tests/unit/domains/ecological", "tests/property/domains/ecological"],
    "claim_status": "IMPLEMENTED",
}

__all__ = [
    "BASELINES",
    "DAMAGE_CLAIM",
    "IMPLEMENTATION_METADATA",
    "STRESS_CLAIM",
    "AnalyticCEFD",
    "BeliefGridConfig",
    "CefdSwitches",
    "CouplingConfig",
    "EcologicalEncoder",
    "EntityBelief",
    "EntityBeliefStore",
    "EntityConfig",
    "FieldBelief",
    "FieldBeliefGrid",
    "FieldSpec",
    "Model2E",
    "Model2EConfig",
    "UnitError",
    "baseline_config",
    "classify_entity_type",
    "evidence_kind",
    "load_model2e_config",
    "make_baseline",
]
