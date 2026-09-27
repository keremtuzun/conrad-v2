"""P5 task-head contracts over frozen OS-FM downstream representations."""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import Field, model_validator

from conrad.schemas.base import VersionedModel
from conrad.schemas.frames import SpatialSupport
from conrad.schemas.observation import Evidence, Modality
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.uncertainty import Uncertainty

OSFM_REPRESENTATION_DIM = 384


class DownstreamValidationStatus(str, Enum):
    DESIGNED = "DESIGNED"
    IMPLEMENTED = "IMPLEMENTED"
    VALIDATED_RUN = "VALIDATED-RUN"


class TaskHeadInput(VersionedModel):
    """Representation plus immutable source metadata handed off by OS-FM."""

    representation: tuple[float, ...] = Field(min_length=OSFM_REPRESENTATION_DIM, max_length=OSFM_REPRESENTATION_DIM)
    source_evidence_id: UUID
    source_observation_id: UUID
    mission_id: UUID
    run_id: UUID
    trace_id: UUID
    modality: Modality
    timestamp: TimeStamp
    provenance_id: UUID
    status: DownstreamValidationStatus = DownstreamValidationStatus.IMPLEMENTED
    checkpoint_label: str = "fixture-or-dev-osfm"


class Detection(VersionedModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    support: SpatialSupport | None = None
    provenance_id: UUID


class SegmentationMask(VersionedModel):
    label: str
    mask_ref: str
    confidence: float = Field(ge=0, le=1)
    provenance_id: UUID


class AnomalyAssessment(VersionedModel):
    score: float = Field(ge=0, le=1)
    reasons: tuple[str, ...]
    provenance_id: UUID


class ConditionAssessment(VersionedModel):
    state: str
    probability: float = Field(ge=0, le=1)
    uncertainty: Uncertainty
    provenance_id: UUID


class TaskHeadOutput(VersionedModel):
    head_name: str
    source_evidence_id: UUID
    status: DownstreamValidationStatus
    payload: dict[str, Any]
    provenance_id: UUID

    @model_validator(mode="after")
    def _no_final_validation_from_fixtures(self) -> TaskHeadOutput:
        if self.status is DownstreamValidationStatus.VALIDATED_RUN and self.payload.get("checkpoint_label") != "OSFM-S-PRETRAIN-V1":
            raise ValueError("VALIDATED-RUN requires qualified OSFM-S-PRETRAIN-V1")
        return self


def _mean_abs(rep: tuple[float, ...]) -> float:
    return sum(abs(x) for x in rep) / len(rep)


class DetectionHead:
    name = "p5-detection-head-v0"

    def infer(self, x: TaskHeadInput) -> TaskHeadOutput:
        score = min(1.0, _mean_abs(x.representation))
        det = Detection(label="asset-feature", confidence=score, provenance_id=x.provenance_id)
        return TaskHeadOutput(
            head_name=self.name,
            source_evidence_id=x.source_evidence_id,
            status=x.status,
            payload={"detections": [det.model_dump(mode="json")], "checkpoint_label": x.checkpoint_label},
            provenance_id=x.provenance_id,
        )


class SegmentationHead:
    name = "p5-segmentation-head-v0"

    def infer(self, x: TaskHeadInput) -> TaskHeadOutput:
        mask = SegmentationMask(
            label="inspectable-region",
            mask_ref=f"derived://{x.source_evidence_id}/segmentation/0",
            confidence=min(1.0, max(0.0, x.representation[0] * 0.5 + 0.5)),
            provenance_id=x.provenance_id,
        )
        return TaskHeadOutput(
            head_name=self.name,
            source_evidence_id=x.source_evidence_id,
            status=x.status,
            payload={"masks": [mask.model_dump(mode="json")], "checkpoint_label": x.checkpoint_label},
            provenance_id=x.provenance_id,
        )


class AnomalyHead:
    name = "p5-anomaly-head-v0"

    def infer(self, x: TaskHeadInput) -> TaskHeadOutput:
        score = min(1.0, max(0.0, _mean_abs(x.representation) * 1.25))
        reasons = ("high-representation-energy",) if score >= 0.5 else ("within-fixture-band",)
        assessment = AnomalyAssessment(score=score, reasons=reasons, provenance_id=x.provenance_id)
        return TaskHeadOutput(
            head_name=self.name,
            source_evidence_id=x.source_evidence_id,
            status=x.status,
            payload={"anomaly": assessment.model_dump(mode="json"), "checkpoint_label": x.checkpoint_label},
            provenance_id=x.provenance_id,
        )


class ConditionHead:
    name = "p5-condition-head-v0"

    def infer(self, x: TaskHeadInput) -> TaskHeadOutput:
        probability = min(1.0, max(0.0, 0.5 + x.representation[1] * 0.25))
        condition = ConditionAssessment(
            state="nominal" if probability >= 0.5 else "requires-review",
            probability=probability,
            uncertainty=Uncertainty(
                aleatoric=0.05,
                epistemic=0.25 if x.checkpoint_label != "OSFM-S-PRETRAIN-V1" else 0.1,
                contradiction=0.0,
                observational=0.2,
            ),
            provenance_id=x.provenance_id,
        )
        return TaskHeadOutput(
            head_name=self.name,
            source_evidence_id=x.source_evidence_id,
            status=x.status,
            payload={"condition": condition.model_dump(mode="json"), "checkpoint_label": x.checkpoint_label},
            provenance_id=x.provenance_id,
        )


def task_input_from_evidence(evidence: Evidence, representation: tuple[float, ...]) -> TaskHeadInput:
    return TaskHeadInput(
        representation=representation,
        source_evidence_id=evidence.evidence_id,
        source_observation_id=evidence.source_observation_id,
        mission_id=evidence.mission_id,
        run_id=evidence.run_id,
        trace_id=evidence.trace_id,
        modality=evidence.modality,
        timestamp=evidence.timestamp,
        provenance_id=evidence.provenance_id,
    )
