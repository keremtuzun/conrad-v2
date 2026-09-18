"""ECMER-S geometric pass-through (domain-local): Observation -> Evidence + DIRECT_OBSERVATION provenance.

The learned ECMER service lives in ``conrad.core.ecmer``; Model2S consumes Evidence from either. This
encoder only summarises the geometric payload (interpretable embedding, never a belief) so Model2S can be
run without the learned encoder. Reliability is a declared health mapping (ENGINEERING_ESTIMATE); it is
never a semantic confidence. The Evidence points back to its Observation, whose payload Model2S loads
through the ObjectStore.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from conrad.persistence.object_store import ObjectStore
from conrad.schemas.frames import SpatialSupport
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import (
    Evidence,
    EvidenceValidity,
    Modality,
    Observation,
    QualityContext,
    SensorHealth,
)
from conrad.schemas.provenance import ProvenanceRecord, SourceType

HEALTH_RELIABILITY = {
    SensorHealth.OK: 1.0,
    SensorHealth.DEGRADED: 0.6,
    SensorHealth.UNKNOWN: 0.8,
    SensorHealth.FAULT: 0.0,
}
"""Declared sensing-reliability mapping from self-reported health (ENGINEERING_ESTIMATE)."""

GEOMETRIC_MODALITIES = (Modality.DEPTH_RANGE, Modality.POINT_CLOUD, Modality.SONAR)
_MODALITY_ORDER = (Modality.DEPTH_RANGE, Modality.POINT_CLOUD, Modality.SONAR, Modality.RGB)


def capture_group(obs: Observation) -> str:
    """Observations from one sensor at one instant (e.g. a range image and its point cloud) are one capture."""
    return f"{obs.sensor_id}:{obs.timestamp.clock_domain}:{obs.timestamp.time_ns}"


def _summary(obs: Observation, arr: np.ndarray) -> tuple[list[float], float, float | None]:
    """(embedding, valid fraction, median range or None)."""
    a = np.asarray(arr, dtype=np.float64)
    max_r = float(obs.sensor_context.get("settings", {}).get("max_range_m", 1.0)) or 1.0
    if obs.modality in (Modality.DEPTH_RANGE, Modality.POINT_CLOUD):
        r = (
            a.reshape(-1)
            if obs.modality is Modality.DEPTH_RANGE
            else np.linalg.norm(a.reshape(-1, 3), axis=1)
        )
        ok = np.isfinite(r)
        valid = float(ok.mean()) if r.size else 0.0
        if ok.any():
            q25, q50, q75 = np.percentile(r[ok], [25, 50, 75])
            return [valid, q50 / max_r, (q75 - q25) / max_r, float(r[ok].min()) / max_r], valid, float(q50)
        return [valid, 0.0, 0.0, 0.0], valid, None
    if obs.modality is Modality.SONAR:
        return [float(a.mean()), float(a.std()), float(a.max(initial=0.0)), float(np.median(a))], 1.0, None
    return [float(a.mean()) / 255.0, float(a.std()) / 255.0, 0.0, 0.0], 1.0, None


class GeometricEvidenceEncoder:
    def __init__(
        self,
        ids: IdFactory,
        store: ObjectStore,
        encoder_version: str = "m2s-geom-v1",
        clock_ns: Callable[[], int] | None = None,
    ) -> None:
        self.ids = ids
        self.store = store
        self.encoder_version = encoder_version
        self.clock_ns = clock_ns

    def encode(self, obs: Observation) -> tuple[Evidence, ProvenanceRecord]:
        if obs.payload_ref is None:
            raise ValueError(f"observation {obs.observation_id} has no payload to encode")
        arr = self.store.get_array(obs.payload_ref)
        feats, valid, median_r = _summary(obs, arr)
        onehot = [1.0 if obs.modality is m else 0.0 for m in _MODALITY_ORDER]
        pose = obs.robot_pose_estimate
        sigma = pose.position_sigma_m() if pose is not None else None
        settings = obs.sensor_context.get("settings", {})
        if obs.modality is Modality.SONAR:
            span = float(settings.get("max_range_m", 0.0)) - float(settings.get("min_range_m", 0.0))
            aleatoric = span / max(int(settings.get("n_range_bins", 1)), 1) / np.sqrt(12.0)
        else:
            aleatoric = float(settings.get("range_noise_sigma_m", 1.0))
        validity = (
            EvidenceValidity.INVALID
            if obs.sensor_health is SensorHealth.FAULT
            else EvidenceValidity.DEGRADED
            if obs.sensor_health is SensorHealth.DEGRADED
            else EvidenceValidity.VALID
        )
        evidence_id, record_id = self.ids.new(), self.ids.new()
        measurements = {"valid_fraction": valid}
        units = {"valid_fraction": "1"}
        if median_r is not None:
            measurements["median_range_m"] = median_r
            units["median_range_m"] = "m"
        record = ProvenanceRecord(
            record_id=record_id,
            source_type=SourceType.DIRECT_OBSERVATION,
            source_ids=(obs.observation_id,),
            operation="m2s.geometric_encode",
            module="conrad.domains.spatial.evidence",
            model_version=self.encoder_version,
            timestamp=obs.timestamp,
            subject_id=evidence_id,
        )
        created = self.clock_ns() if self.clock_ns is not None else obs.timestamp.time_ns
        ev = Evidence(
            evidence_id=evidence_id,
            source_observation_id=obs.observation_id,
            mission_id=obs.mission_id,
            run_id=obs.run_id,
            trace_id=obs.trace_id,
            modality=obs.modality,
            timestamp=obs.timestamp,
            created_time_ns=max(created, obs.timestamp.time_ns),
            embedding=tuple(float(x) for x in (*feats, *onehot)),
            spatial_support=None
            if pose is None
            else SpatialSupport(frame_id=pose.frame_id, center_m=pose.position_m, position_sigma_m=sigma),
            reliability=HEALTH_RELIABILITY[obs.sensor_health],
            aleatoric_uncertainty=aleatoric,
            validity=validity,
            sensor_context=QualityContext(
                range_m=median_r, sensor_health=obs.sensor_health, pose_sigma_m=sigma
            ),
            measurements=measurements,
            measurement_units=units,
            independence_group=capture_group(obs),
            provenance_id=record_id,
            encoder_version=self.encoder_version,
        )
        return ev, record
