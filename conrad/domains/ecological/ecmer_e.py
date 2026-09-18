"""ECMER-E analytic encoder: ENVIRONMENTAL / STRUCTURED observations -> 2E Evidence (ch12 ECMER-E).

Evidence type is explicit: FIELD evidence (point sensor, one field channel, located at the robot's
ESTIMATED sensor position) or ENTITY evidence (survey hit, located by estimated pose + offset).
The observation timestamp is copied unchanged; streams are never re-timed. No simulator identity
is read: association happens later by position gating against the mission asset registry.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from conrad.domains.ecological.config import Model2EConfig
from conrad.domains.ecological.units import UnitError, require_units, split_compound
from conrad.schemas.frames import SpatialSupport, quat_to_matrix
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

ENCODER_VERSION = "ecmer-e-analytic-0.1.0"
ENTITY_KEYS = ("cover_fraction", "detection", "survey_hit")
_RELIABILITY = {SensorHealth.OK: 0.95, SensorHealth.DEGRADED: 0.6, SensorHealth.FAULT: 0.0}
EvidenceKind = Literal["FIELD", "ENTITY", "OTHER"]


def evidence_kind(ev: Evidence, config: Model2EConfig) -> EvidenceKind:
    if any(k in ev.measurements for k in ENTITY_KEYS):
        return "ENTITY"
    if any(k.split(".")[0] in config.fields for k in ev.measurements):
        return "FIELD"
    return "OTHER"


class EcologicalEncoder:
    def __init__(self, config: Model2EConfig, ids: IdFactory, pose_sigma_m: float = 0.5) -> None:
        self.cfg = config
        self.ids = ids
        self.pose_sigma_m = pose_sigma_m

    def encode(self, obs: Observation, created_time_ns: int) -> tuple[Evidence, ProvenanceRecord]:
        if obs.inline_values is None or obs.inline_units is None:
            raise UnitError("2E encoder needs inline values with units")
        ctx = obs.sensor_context
        kind = ctx.get("kind")
        if obs.modality is Modality.ENVIRONMENTAL and kind == "field_sample":
            meas, units, emb, support, sd = self._field(obs)
        elif obs.modality is Modality.STRUCTURED and kind == "ecological_survey":
            meas, units, emb, support, sd = self._survey(obs)
        else:
            raise ValueError(f"2E encoder does not handle {obs.modality.value}/{kind!r}")
        health = obs.sensor_health
        validity = EvidenceValidity.INVALID if health is SensorHealth.FAULT else EvidenceValidity.VALID
        if support is None and validity is EvidenceValidity.VALID:
            validity = EvidenceValidity.DEGRADED  # no pose estimate: cannot be located
        pid = self.ids.new()
        eid = self.ids.new()
        prov = ProvenanceRecord(
            record_id=pid,
            source_type=SourceType.DIRECT_OBSERVATION,
            source_ids=(obs.observation_id,),
            operation="ecmer_e.encode",
            module="conrad.domains.ecological.ecmer_e",
            model_version=ENCODER_VERSION,
            timestamp=obs.timestamp,
            subject_id=eid,
        )
        range_m = (
            float(np.linalg.norm(emb[-3:])) if "cover_fraction" in meas or "survey_hit" in meas else None
        )
        ev = Evidence(
            evidence_id=eid,
            source_observation_id=obs.observation_id,
            mission_id=obs.mission_id,
            run_id=obs.run_id,
            trace_id=obs.trace_id,
            modality=obs.modality,
            timestamp=obs.timestamp,
            created_time_ns=max(created_time_ns, obs.timestamp.time_ns),
            embedding=tuple(float(v) for v in emb),
            spatial_support=support,
            reliability=_RELIABILITY.get(health, 0.5),
            aleatoric_uncertainty=sd,
            validity=validity,
            sensor_context=QualityContext(
                sensor_health=health,
                pose_sigma_m=None if support is None else support.position_sigma_m,
                range_m=range_m,
            ),
            measurements=meas,
            measurement_units=units,
            provenance_id=pid,
            encoder_version=ENCODER_VERSION,
        )
        return ev, prov

    def _position(self, obs: Observation, offset: np.ndarray | None) -> SpatialSupport | None:
        pose = obs.robot_pose_estimate
        if pose is None:
            return None
        p = np.asarray(pose.position_m, dtype=np.float64)
        sigma = self.pose_sigma_m
        if offset is not None:
            p = p + quat_to_matrix(pose.orientation_wxyz) @ offset
            sigma = self.pose_sigma_m + 0.05 * float(np.linalg.norm(offset))
        return SpatialSupport(
            frame_id=pose.frame_id, center_m=(float(p[0]), float(p[1]), float(p[2])), position_sigma_m=sigma
        )

    def _field(
        self, obs: Observation
    ) -> tuple[dict[str, float], dict[str, str], list[float], SpatialSupport | None, float]:
        channel = str(obs.sensor_context.get("channel"))
        if channel not in self.cfg.fields:
            raise ValueError(f"field channel {channel!r} is not configured in Model2E")
        spec = self.cfg.fields[channel]
        require_units(obs.inline_units, spec.units, f"field {channel}")
        vals = list(obs.inline_values or ())
        if len(vals) != spec.components:
            raise UnitError(f"field {channel} expects {spec.components} values, got {len(vals)}")
        if spec.components == 1:
            meas = {channel: float(vals[0])}
        else:
            meas = {f"{channel}.{i}": float(v) for i, v in enumerate(vals)}
        units = dict.fromkeys(meas, spec.units)
        return meas, units, vals, self._position(obs, None), spec.sensor_sd

    def _survey(
        self, obs: Observation
    ) -> tuple[dict[str, float], dict[str, str], list[float], SpatialSupport | None, float]:
        parts = split_compound(tuple(obs.inline_values or ()), obs.inline_units or "")
        if "offset_m" not in parts:
            raise UnitError("survey observation lacks offset_m")
        off_vals, off_units = parts["offset_m"]
        for u in off_units:
            require_units(u, "m", "survey offset")
        offset = np.asarray(off_vals, dtype=np.float64)
        meas: dict[str, float]
        if obs.sensor_context.get("channel") == "cover_estimate" and "cover_fraction" in parts:
            (cover,), (cu,) = parts["cover_fraction"]
            require_units(cu, "1", "cover_fraction")
            meas, emb = {"cover_fraction": float(cover)}, [float(cover)]
        elif "detection" in parts:
            (det,), (du,) = parts["detection"]
            require_units(du, "1", "detection")
            meas, emb = {"detection": float(det)}, [float(det)]
        elif "feature" in parts:
            # E1 features are not decodable analytically; the hit itself is the only interpretable fact.
            meas, emb = {"survey_hit": 1.0}, list(parts["feature"][0])
        else:
            raise UnitError(f"survey observation has no recognised payload: {obs.inline_units!r}")
        units = dict.fromkeys(meas, "1")
        support = self._position(obs, offset)
        return meas, units, [*emb, *offset.tolist()], support, self.cfg.entity.cover_meas_sd
