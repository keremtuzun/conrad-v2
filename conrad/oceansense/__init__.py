"""OceanSense downstream contracts for P5-P10.

These modules sit downstream of the frozen OS-FM representation contract. They are
fixture-safe integration code, not a claim that qualified OSFM-S-PRETRAIN-V1 has
completed final integrated validation.
"""

from conrad.oceansense.alpha import AlphaIntegrationResult, OceanSenseAlpha
from conrad.oceansense.host import HostCapabilityProfile, HostIntentGateway, InspectionIntent
from conrad.oceansense.model2_adapter import Model2EvidenceBundle, RepresentationAdapter
from conrad.oceansense.reasoning import CompetingHypotheses, CrossDomainReasoner, Hypothesis
from conrad.oceansense.task_heads import (
    AnomalyHead,
    ConditionHead,
    DetectionHead,
    SegmentationHead,
    TaskHeadInput,
)

__all__ = [
    "AlphaIntegrationResult",
    "AnomalyHead",
    "CompetingHypotheses",
    "ConditionHead",
    "CrossDomainReasoner",
    "DetectionHead",
    "HostCapabilityProfile",
    "HostIntentGateway",
    "Hypothesis",
    "InspectionIntent",
    "Model2EvidenceBundle",
    "OceanSenseAlpha",
    "RepresentationAdapter",
    "SegmentationHead",
    "TaskHeadInput",
]
