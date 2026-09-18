"""EcmerEncoder service: Observation -> Evidence (+ DIRECT_OBSERVATION provenance) (ch3, ch4).

ECMER stops at structured evidence; it never emits a belief. ``reliability`` is sensing
reliability from the quality head, never a semantic class confidence (none is produced here).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from conrad.core.config import CoreConfig
from conrad.core.ecmer.fusion import EcmerModel, EcmerOutput
from conrad.core.ecmer.preprocess import PayloadLoader, build_event_batch, payload_array, raw_quality_for
from conrad.core.ecmer.quality_features import RawQuality
from conrad.schemas.frames import SpatialSupport
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Evidence, EvidenceValidity, Observation, QualityContext, SensorHealth
from conrad.schemas.provenance import ProvenanceRecord, SourceType


@dataclass(frozen=True)
class EncodedEvidence:
    evidence: Evidence
    provenance: ProvenanceRecord


def _measurements(obs: Observation) -> tuple[dict[str, float], dict[str, str]]:
    """Interpretable scalars pass through only when the sensor declared their names; never invented."""
    names = obs.sensor_context.get("measurement_names")
    if (
        obs.inline_values is None
        or not isinstance(names, (list, tuple))
        or len(names) != len(obs.inline_values)
    ):
        return {}, {}
    units = obs.inline_units or "unknown"
    values = {str(n): float(v) for n, v in zip(names, obs.inline_values, strict=True) if math.isfinite(v)}
    return values, dict.fromkeys(values, units)


class EcmerEncoder:
    def __init__(
        self,
        cfg: CoreConfig,
        ids: IdFactory,
        loader: PayloadLoader | None = None,
        model: EcmerModel | None = None,
        clock_ns: Callable[[], int] | None = None,
    ) -> None:
        self.cfg = cfg
        self.ids = ids
        self.loader = loader
        self.model = model or EcmerModel(cfg)
        self.clock_ns = clock_ns

    def encode(
        self, observations: Sequence[Observation], delta_t_s: Sequence[float] | None = None
    ) -> list[EncodedEvidence]:
        if not observations:
            return []
        arrays = [payload_array(o, self.loader) for o in observations]
        qualities = [raw_quality_for(o, a) for o, a in zip(observations, arrays, strict=True)]
        batch = build_event_batch(observations, arrays, qualities, self.cfg, delta_t_s)
        self.model.eval()
        with torch.no_grad():
            out = self.model(batch)
        return [
            self._evidence(i, o, q, out) for i, (o, q) in enumerate(zip(observations, qualities, strict=True))
        ]

    def _evidence(self, i: int, obs: Observation, q: RawQuality, out: EcmerOutput) -> EncodedEvidence:
        e = self.cfg.ecmer
        evidence_id, record_id = self.ids.new(), self.ids.new()
        reliability = float(out.reliability[i])
        usable = float(torch.sigmoid(out.usable_logit[i]))
        if obs.sensor_health is SensorHealth.FAULT or (
            q.valid_fraction is not None and q.valid_fraction == 0
        ):
            validity = EvidenceValidity.INVALID
        elif obs.sensor_health is SensorHealth.DEGRADED or usable < e.usable_threshold:
            validity = EvidenceValidity.DEGRADED
        else:
            validity = EvidenceValidity.VALID
        pose = obs.robot_pose_estimate
        sigma = pose.position_sigma_m() if pose is not None else None
        support = (
            None
            if pose is None
            else SpatialSupport(frame_id=pose.frame_id, center_m=pose.position_m, position_sigma_m=sigma)
        )
        range_m = obs.sensor_context.get("range_m")
        quality = QualityContext(
            blur=q.blur,
            lighting=q.brightness,
            occlusion=None,
            range_m=float(range_m) if isinstance(range_m, (int, float)) and range_m >= 0 else None,
            snr_db=q.snr_db,
            sensor_health=obs.sensor_health,
            pose_sigma_m=sigma,
            ood_score=float(torch.sigmoid(out.ood_logit[i])),
        )
        group = obs.sensor_context.get("independence_group", str(obs.observation_id))
        measurements, units = _measurements(obs)
        created = self.clock_ns() if self.clock_ns is not None else obs.timestamp.time_ns
        record = ProvenanceRecord(
            record_id=record_id,
            source_type=SourceType.DIRECT_OBSERVATION,
            source_ids=(obs.observation_id,),
            operation="ecmer.encode",
            module="conrad.core.ecmer",
            model_version=e.encoder_version,
            timestamp=obs.timestamp,
            subject_id=evidence_id,
        )
        evidence = Evidence(
            evidence_id=evidence_id,
            source_observation_id=obs.observation_id,
            mission_id=obs.mission_id,
            run_id=obs.run_id,
            trace_id=obs.trace_id,
            modality=obs.modality,
            timestamp=obs.timestamp,
            created_time_ns=max(created, obs.timestamp.time_ns),
            embedding=tuple(float(v) for v in out.event[i].tolist()),
            spatial_support=support,
            reliability=min(max(reliability, 0.0), 1.0),
            aleatoric_uncertainty=float(torch.exp(out.log_var_a[i])),
            validity=validity,
            sensor_context=quality,
            measurements=measurements,
            measurement_units=units,
            independence_group=str(group) if e.use_source_independence else None,
            provenance_id=record_id,
            encoder_version=e.encoder_version,
        )
        return EncodedEvidence(evidence, record)
