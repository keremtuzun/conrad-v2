"""ECMER: raw observation -> structured evidence (ch4, ch33). Never emits beliefs."""

from conrad.core.ecmer.baselines import build_fusion_baseline
from conrad.core.ecmer.fusion import MODALITIES, EcmerModel, EcmerOutput, EventBatch, sample_modality_dropout
from conrad.core.ecmer.preprocess import PayloadLoader, build_event_batch
from conrad.core.ecmer.quality_features import RawQuality, image_quality, scalar_quality, sonar_quality
from conrad.core.ecmer.service import EcmerEncoder, EncodedEvidence
from conrad.core.ecmer.ssl import ProjectionHead, ReprObjective, corrupt_images, info_nce
from conrad.core.ecmer.training import (
    stage_e1_representation,
    stage_e2_degradation,
    stage_e3_quality,
    stage_e4_correspondence,
)

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": ["ch3 ECMER", "ch4", "ch33 ECMER exact implementation", "ch33 SSL pretraining"],
    "configuration_keys": ["model2_core.ecmer.*", "model2_core.evidence_dim", "model2_core.heads"],
    "assumptions": [
        "context embedding is added to every modality token but not to [EVENT]",
        "reliability ablation forces reliability=1 so downstream sees no reliability signal",
        "early-fusion baseline sums pre-fusion modality features before any joint processing",
        "spatial support of evidence is the robot pose estimate; entity localisation is child-specific",
    ],
    "baselines": ["single_rgb", "single_sonar", "concat", "early", "late", "cross_attention"],
    "acceptance_tests": ["tests/unit/core/test_ecmer.py"],
    "claim_status": "IMPLEMENTED",
}

__all__ = [
    "IMPLEMENTATION_METADATA",
    "MODALITIES",
    "EcmerEncoder",
    "EcmerModel",
    "EcmerOutput",
    "EncodedEvidence",
    "EventBatch",
    "PayloadLoader",
    "ProjectionHead",
    "RawQuality",
    "ReprObjective",
    "build_event_batch",
    "build_fusion_baseline",
    "corrupt_images",
    "image_quality",
    "info_nce",
    "sample_modality_dropout",
    "scalar_quality",
    "sonar_quality",
    "stage_e1_representation",
    "stage_e2_degradation",
    "stage_e3_quality",
    "stage_e4_correspondence",
]
