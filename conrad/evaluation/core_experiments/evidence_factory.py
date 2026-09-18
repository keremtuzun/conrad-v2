"""Builds schema-valid ``Evidence`` + provenance for experiments and tests (SYNTHETIC_ONLY).

Nothing here carries truth: callers pass the (noisy) measured values they want the belief plane to see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

import numpy as np

from conrad.schemas.frames import SpatialSupport
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Evidence, EvidenceValidity, Modality, QualityContext
from conrad.schemas.provenance import ProvenanceRecord, SourceType
from conrad.schemas.timebase import TimeStamp, stamp

CLOCK = "sandbox_clock"


@dataclass
class EvidenceFactory:
    ids: IdFactory
    embedding_dim: int = 16
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    mission_id: UUID | None = None
    run_id: UUID | None = None

    def __post_init__(self) -> None:
        self.mission_id = self.mission_id or self.ids.new()
        self.run_id = self.run_id or self.ids.new()

    def make(
        self,
        time_s: float,
        measurements: dict[str, float] | None = None,
        *,
        center_m: tuple[float, float, float] | None = (0.0, 0.0, 0.0),
        reliability: float = 0.9,
        aleatoric: float = 0.01,
        ood_score: float | None = 0.05,
        occlusion: float | None = 0.0,
        independence_group: str | None = None,
        modality: Modality = Modality.STRUCTURED,
        validity: EvidenceValidity = EvidenceValidity.VALID,
        embedding: np.ndarray | None = None,
        units: str = "unitless",
        position_sigma_m: float | None = 0.05,
        frame_id: str = "WORLD",
    ) -> tuple[Evidence, ProvenanceRecord]:
        assert self.mission_id is not None and self.run_id is not None
        ts: TimeStamp = stamp(time_s, CLOCK)
        evidence_id, observation_id, record_id = self.ids.new(), self.ids.new(), self.ids.new()
        emb = self.rng.normal(size=self.embedding_dim) if embedding is None else embedding
        meas = dict(measurements or {})
        record = ProvenanceRecord(
            record_id=record_id,
            source_type=SourceType.DIRECT_OBSERVATION,
            source_ids=(observation_id,),
            operation="sandbox.encode",
            module="conrad.evaluation.core_experiments",
            model_version="sandbox-v0",
            timestamp=ts,
            subject_id=evidence_id,
        )
        evidence = Evidence(
            evidence_id=evidence_id,
            source_observation_id=observation_id,
            mission_id=self.mission_id,
            run_id=self.run_id,
            trace_id=self.ids.new(),
            modality=modality,
            timestamp=ts,
            created_time_ns=ts.time_ns,
            embedding=tuple(float(v) for v in emb),
            spatial_support=None
            if center_m is None
            else SpatialSupport(frame_id=frame_id, center_m=center_m, position_sigma_m=position_sigma_m),
            reliability=reliability,
            aleatoric_uncertainty=aleatoric,
            validity=validity,
            sensor_context=QualityContext(ood_score=ood_score, occlusion=occlusion),
            measurements=meas,
            measurement_units=dict.fromkeys(meas, units),
            independence_group=independence_group,
            provenance_id=record_id,
            encoder_version="sandbox-v0",
        )
        return evidence, record
