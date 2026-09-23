"""Read-side of Model2E: PREDICTED messages, queries and observability context (no state mutation).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

from conrad.domains.ecological import messages as mb
from conrad.domains.ecological.views import cover_prior_var, ts
from conrad.schemas.belief import (
    BeliefMessage,
    BeliefQuery,
    EcologicalPayload,
    KnowledgeStatus,
    PropertyClaim,
    summarize_status,
)
from conrad.schemas.frames import SpatialSupport
from conrad.schemas.provenance import ProvenanceRecord, SourceType
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.uncertainty import Uncertainty
from conrad.schemas.world import Domain

if TYPE_CHECKING:
    from conrad.domains.ecological.model2e import Model2E

K = KnowledgeStatus
CONTEXT_NOTE = "CROSS_DOMAIN_CONTEXT: consume as context only, never as a direct observation"


def _message(
    m: Model2E,
    belief_id: UUID,
    t: TimeStamp,
    claims: tuple[PropertyClaim, ...],
    u: Uncertainty,
    emb: tuple[float, ...],
    support: SpatialSupport | None,
    payload: EcologicalPayload,
    prov: UUID,
    change: str | None = None,
    prediction: str | None = None,
) -> BeliefMessage:
    head = m.ledger.heads[belief_id]
    return BeliefMessage(
        message_id=m.ids.new(),
        belief_id=belief_id,
        revision=head.revision,
        independent_observation_count=(
            head.message.independent_observation_count if head.message is not None else 0
        ),
        world_entity_id=head.registry_id,
        domain=Domain.ECOLOGICAL,
        timestamp=t,
        state_summary=claims,
        state_embedding=emb,
        knowledge_status=summarize_status(claims) if claims else K.UNKNOWN,
        uncertainty=u,
        change_summary=change,
        prediction_summary=prediction,
        provenance_refs=(prov,),
        spatial_support=support,
        lifecycle=head.lifecycle,
        model_version=m.model_version,
        publisher_availability=m.availability(),
        ecological=payload,
    )


def _repredict(claims: tuple[PropertyClaim, ...]) -> tuple[PropertyClaim, ...]:
    return tuple(c if c.status is K.UNKNOWN else c.model_copy(update={"status": K.PREDICTED}) for c in claims)


def predict_all(m: Model2E, target_ns: int, now: TimeStamp) -> list[BeliefMessage]:
    """PREDICTED readout at now + delta_t; each belief propagates over ITS OWN physical elapsed time."""
    out: list[BeliefMessage] = []
    t = ts(m, target_ns)
    records: list[ProvenanceRecord] = []
    for name, bid in m.field_ids.items():
        mean, var = m.fields.moments_at_time(name, target_ns)
        q, units = mb.field_summary(m.fields, name, mean, var, None)
        ua, ue, uc, _ = m.fields.uncertainty_channels(name)
        u = mb.unc(ua, ue, uc, m.fields.predicted_uo(name, target_ns))
        rec = m.ledger.record(
            SourceType.TEMPORAL_PREDICTION, (bid,), bid, t, (m.ledger.heads[bid].root,), "field.ou_predict"
        )
        records.append(rec)
        claims = _repredict(mb.field_claims(m.fields, name, q, K.OBSERVED, u, rec.record_id))
        out.append(
            _message(
                m,
                bid,
                t,
                claims,
                u,
                mb.field_embedding(q),
                mb.field_support(m.fields),
                EcologicalPayload(kind="FIELD", quantities=q, units=units),
                rec.record_id,
                prediction=f"OU prediction to t={target_ns} ns",
            )
        )
    pv = cover_prior_var(m)
    for bid, b in m.entities.beliefs.items():
        if bid not in m.ledger.heads:
            continue
        gate = m.cefd.gate(m.fields, "temperature", b.position_m) if m.cfg.switches.fields else 0.0
        cover = m.entities.cover_moments_at(b, target_ns, m.cefd.process_inflation(b, gate))
        pres = m.entities.presence_at(b, target_ns)
        rec = m.ledger.record(
            SourceType.TEMPORAL_PREDICTION,
            (bid,),
            bid,
            t,
            (m.ledger.heads[bid].root,),
            "entity.random_walk_predict",
        )
        records.append(rec)
        claims = _repredict(mb.entity_claims(b, cover, pres, K.OBSERVED, rec.record_id, rec.record_id, pv))
        q, units = mb.entity_quantities(b, cover, pres)
        out.append(
            _message(
                m,
                bid,
                t,
                claims,
                mb.entity_uncertainty(b, cover[1], pv),
                mb.entity_embedding(b, cover, pres),
                mb.entity_support(b),
                EcologicalPayload(kind="ENTITY", quantities=q, units=units),
                rec.record_id,
                prediction=f"bounded random walk to t={target_ns} ns (no invented trend)",
            )
        )
    m.ledger.store_records(records)
    return out


def _inside(support: SpatialSupport | None, region: SpatialSupport) -> bool:
    if support is None or support.frame_id != region.frame_id:
        return False
    d = np.abs(np.asarray(support.center_m) - np.asarray(region.center_m))
    return bool(np.all(d <= np.asarray(region.half_extent_m) + np.asarray(support.half_extent_m)))


def _region_field(m: Model2E, name: str, region: SpatialSupport) -> BeliefMessage | None:
    if region.frame_id != m.fields.grid.cfg.frame_id:
        return None
    cells = m.fields.region_cells(np.asarray(region.center_m), np.asarray(region.half_extent_m))
    if not cells.any():
        return None
    fb = m.fields.fields[name]
    head = m.ledger.heads[m.field_ids[name]]
    q, units = mb.field_summary(m.fields, name, fb.mean, fb.var, cells)
    u = mb.unc(*m.fields.uncertainty_channels(name, cells))
    claims = mb.field_claims(m.fields, name, q, K.OBSERVED, u, head.root)
    return _message(
        m,
        m.field_ids[name],
        ts(m, head.time_ns),
        claims,
        u,
        mb.field_embedding(q),
        mb.field_support(m.fields, cells),
        EcologicalPayload(kind="FIELD", quantities=q, units=units),
        head.root,
        change="region summary of the field belief",
    )


def run_query(m: Model2E, q: BeliefQuery) -> list[BeliefMessage]:
    if q.domain not in (None, Domain.ECOLOGICAL):
        return []
    field_ids = {v: k for k, v in m.field_ids.items()}
    out: list[BeliefMessage] = []
    for bid, head in m.ledger.heads.items():
        msg = head.message
        if msg is None or (q.belief_ids and bid not in q.belief_ids):
            continue
        if q.entity_ids and head.registry_id not in q.entity_ids:
            continue
        if q.region is not None:
            if bid in field_ids:
                region_msg = _region_field(m, field_ids[bid], q.region)
                if region_msg is None:
                    continue
                msg = region_msg
            elif not _inside(msg.spatial_support, q.region):
                continue
        if q.time_range_ns is not None and not (
            q.time_range_ns[0] <= msg.timestamp.time_ns <= q.time_range_ns[1]
        ):
            continue
        if (
            q.min_observational_uncertainty is not None
            and msg.uncertainty.observational < q.min_observational_uncertainty
        ):
            continue
        if (
            q.min_contradiction_uncertainty is not None
            and msg.uncertainty.contradiction < q.min_contradiction_uncertainty
        ):
            continue
        if q.requested_fields:
            keep = tuple(c for c in msg.state_summary if c.name.startswith(q.requested_fields))
            if not keep:
                continue
            msg = msg.model_copy(update={"state_summary": keep, "knowledge_status": summarize_status(keep)})
        out.append(msg)
        if len(out) >= q.max_results:
            break
    return out
