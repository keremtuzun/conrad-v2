"""STRUCTURED observation -> Evidence adapter for Model2T (belief-side reading of a sensor payload).

It reads only the Observation contract: ``inline_values`` + ``sensor_context['measurements'/'units']``.
Reliability and aleatoric level come from reported sensor health (own ENGINEERING_ESTIMATE table).
The association hint (``registry_entity_id``) is supplied by the caller's association step.

Independence groups are written as ``sensor:<sensor id>|<reading group>`` (reading group = the caller's group,
else ``obs:<observation id>``). Evidence sharing the full group is one reading; evidence sharing only the
``sensor:`` prefix comes from one sensor and shares its persistent sizing bias (Model2T's bias floor,
docs/audits/MODEL2T_REPAIR.md). The sensor id is a deployment-side identity, never a world-entity id.
"""

from __future__ import annotations

from uuid import UUID

from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import (
    EntityCandidate,
    Evidence,
    EvidenceValidity,
    Modality,
    Observation,
    QualityContext,
    SensorHealth,
)
from conrad.schemas.provenance import ProvenanceRecord, SourceType

HEALTH_RELIABILITY: dict[SensorHealth, tuple[float, float, EvidenceValidity]] = {
    SensorHealth.OK: (0.9, 0.1, EvidenceValidity.VALID),
    SensorHealth.DEGRADED: (0.5, 0.5, EvidenceValidity.DEGRADED),
    SensorHealth.UNKNOWN: (0.5, 0.5, EvidenceValidity.DEGRADED),
    SensorHealth.FAULT: (0.05, 1.0, EvidenceValidity.INVALID),
}
ENCODER_VERSION = "model2t-structured-adapter-0.1.0"


class NotStructuredObservation(ValueError):
    pass


def structured_evidence(
    obs: Observation,
    ids: IdFactory,
    registry_entity_id: UUID | None,
    *,
    association_score: float = 1.0,
    independence_group: str | None = None,
) -> tuple[Evidence, ProvenanceRecord]:
    if obs.modality is not Modality.STRUCTURED or obs.inline_values is None:
        raise NotStructuredObservation("Model2T structured adapter needs inline STRUCTURED values")
    names = [str(n) for n in obs.sensor_context.get("measurements", [])]
    units = [str(u) for u in obs.sensor_context.get("units", str(obs.inline_units).split(","))]
    if len(names) != len(obs.inline_values) or len(units) != len(names):
        raise NotStructuredObservation("measurement names/units do not match inline values")
    rel, ua, validity = HEALTH_RELIABILITY[obs.sensor_health]
    group = f"sensor:{obs.sensor_id}|{independence_group or f'obs:{obs.observation_id}'}"
    pid, eid = ids.new(), ids.new()
    prov = ProvenanceRecord(
        record_id=pid,
        source_type=SourceType.DIRECT_OBSERVATION,
        source_ids=(obs.observation_id,),
        operation="structured_evidence",
        module="conrad.domains.technical.evidence",
        model_version=ENCODER_VERSION,
        timestamp=obs.timestamp,
        subject_id=eid,
    )
    cands = (
        ()
        if registry_entity_id is None
        else (EntityCandidate(registry_entity_id=registry_entity_id, score=association_score),)
    )
    ev = Evidence(
        evidence_id=eid,
        source_observation_id=obs.observation_id,
        mission_id=obs.mission_id,
        run_id=obs.run_id,
        trace_id=obs.trace_id,
        modality=Modality.STRUCTURED,
        timestamp=obs.timestamp,
        created_time_ns=obs.timestamp.time_ns,
        embedding=tuple(float(v) for v in obs.inline_values),
        entity_candidates=cands,
        reliability=rel,
        aleatoric_uncertainty=ua,
        validity=validity,
        sensor_context=QualityContext(sensor_health=obs.sensor_health),
        measurements=dict(zip(names, (float(v) for v in obs.inline_values), strict=True)),
        measurement_units=dict(zip(names, units, strict=True)),
        independence_group=group,
        provenance_id=pid,
        encoder_version=ENCODER_VERSION,
    )
    return ev, prov
