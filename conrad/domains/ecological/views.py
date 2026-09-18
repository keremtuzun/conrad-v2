"""Belief -> revision/message conversion for Model2E (create and commit paths).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from conrad.domains.ecological import messages as mb
from conrad.domains.ecological.entity_belief import EntityBelief
from conrad.domains.ecological.ledger import CommitSpec, next_lifecycle
from conrad.schemas.belief import BeliefMessage, EcologicalPayload, KnowledgeStatus, Lifecycle, UpdateKind
from conrad.schemas.observation import Evidence
from conrad.schemas.provenance import ProvenanceRecord, SourceType
from conrad.schemas.timebase import TimeStamp

if TYPE_CHECKING:
    from conrad.domains.ecological.model2e import Model2E

K = KnowledgeStatus


def ts(m: Model2E, t_ns: int) -> TimeStamp:
    return TimeStamp(time_ns=max(0, t_ns), clock_domain=m.clock_domain)


def entity_type_of(b: EntityBelief) -> str:
    return b.asset.entity_type if b.asset is not None else b.eco_class.lower()


def cover_prior_var(m: Model2E) -> float:
    return min(0.25, m.cfg.entity.cover_prior_sd**2)


# ---------------------------------------------------------------------- field
def field_spec(
    m: Model2E,
    name: str,
    status: KnowledgeStatus,
    kind: UpdateKind,
    records: tuple[ProvenanceRecord, ...],
    evidence: Sequence[Evidence],
    lifecycle: Lifecycle,
    t: TimeStamp,
) -> CommitSpec:
    fb = m.fields.fields[name]
    q, units = mb.field_summary(m.fields, name, fb.mean, fb.var, None)
    u = mb.unc(*m.fields.uncertainty_channels(name))
    return CommitSpec(
        belief_id=m.field_ids[name],
        entity_type=f"environmental_field.{name}",
        registry_id=None,
        lifecycle=lifecycle,
        timestamp=t,
        claims=mb.field_claims(m.fields, name, q, status, u, records[-1].record_id),
        uncertainty=u,
        embedding=mb.field_embedding(q),
        support=mb.field_support(m.fields),
        kind=kind,
        payload=EcologicalPayload(kind="FIELD", quantities=q, units=units),
        evidence=tuple(evidence),
        records=records,
    )


def create_field(m: Model2E, name: str, t0: TimeStamp) -> BeliefMessage:
    bid = m.field_ids[name]
    rec = m.ledger.record(SourceType.PRIOR, (), bid, t0, (), "field.create_prior")
    spec = field_spec(m, name, K.UNKNOWN, UpdateKind.CREATE, (rec,), (), Lifecycle.CONFIRMED, t0)
    return m.ledger.commit(spec, m.availability())


def commit_field(m: Model2E, name: str, evidence: Sequence[Evidence], sink: bool) -> BeliefMessage:
    bid = m.field_ids[name]
    head = m.ledger.heads[bid]
    fb = m.fields.fields[name]
    t = ts(m, fb.time_ns or head.time_ns)
    records: list[ProvenanceRecord] = []
    if sink:
        ent = [b.belief_id for b in m.entities.beliefs.values() if b.sessile and b.n_hits]
        parents = [head.root, *(r for r in (m.ledger.root_of(e) for e in ent) if r is not None)]
        records.append(
            m.ledger.record(
                SourceType.RELATIONAL_INFERENCE, ent, bid, t, parents, "cefd.entity_to_field.filtration_sink"
            )
        )
    if evidence:
        parents = [head.root, *(r.record_id for r in records)]
        ev_ids = list(dict.fromkeys(e.evidence_id for e in evidence))
        records.append(
            m.ledger.record(
                SourceType.DIRECT_OBSERVATION, ev_ids, bid, t, parents, "field.gp_lite_kernel_update"
            )
        )
    kind = UpdateKind.DIRECT if evidence else UpdateKind.RELATIONAL
    status = K.OBSERVED if evidence else K.INFERRED
    lifecycle = next_lifecycle(head.lifecycle, fb.n_obs, True)
    uniq = tuple({e.evidence_id: e for e in evidence}.values())
    return m.ledger.commit(
        field_spec(m, name, status, kind, tuple(records), uniq, lifecycle, t), m.availability()
    )


# ---------------------------------------------------------------------- entity
def entity_spec(
    m: Model2E,
    b: EntityBelief,
    kind: UpdateKind,
    records: tuple[ProvenanceRecord, ...],
    evidence: Sequence[Evidence],
    lifecycle: Lifecycle,
    t: TimeStamp,
    summary: str | None = None,
) -> CommitSpec:
    pv = cover_prior_var(m)
    cover = (b.cover_mean, b.cover_var)
    claims = mb.entity_claims(
        b, cover, b.presence_p, K.OBSERVED, m.direct_prov.get(b.belief_id), m.stress_prov.get(b.belief_id), pv
    )
    q, units = mb.entity_quantities(b, cover, b.presence_p)
    return CommitSpec(
        belief_id=b.belief_id,
        entity_type=entity_type_of(b),
        registry_id=None if b.asset is None else b.asset.registry_id,
        lifecycle=lifecycle,
        timestamp=t,
        claims=claims,
        uncertainty=mb.entity_uncertainty(b, b.cover_var, pv),
        embedding=mb.entity_embedding(b, cover, b.presence_p),
        support=mb.entity_support(b),
        kind=kind,
        payload=EcologicalPayload(kind="ENTITY", quantities=q, units=units),
        evidence=tuple(evidence),
        records=records,
        change_summary=summary,
    )


def create_entity(m: Model2E, b: EntityBelief, t0: TimeStamp) -> BeliefMessage:
    rec = m.ledger.record(SourceType.PRIOR, (), b.belief_id, t0, (), "entity.create")
    life = Lifecycle.CONFIRMED if b.asset is not None else Lifecycle.CANDIDATE
    return m.ledger.commit(entity_spec(m, b, UpdateKind.CREATE, (rec,), (), life, t0), m.availability())


def commit_entity(
    m: Model2E, b: EntityBelief, evidence: Sequence[Evidence], stress_gate: float | None
) -> BeliefMessage:
    head = m.ledger.heads[b.belief_id]
    t_ns = max(b.time_ns or 0, b.stress_time_ns or 0, head.time_ns)
    t = ts(m, t_ns)
    records: list[ProvenanceRecord] = []
    summary = None
    if stress_gate is not None:
        temp_id = m.field_ids["temperature"]
        parents = [head.root, m.ledger.heads[temp_id].root]
        rec = m.ledger.record(
            SourceType.RELATIONAL_INFERENCE,
            (temp_id,),
            b.belief_id,
            t,
            parents,
            "cefd.field_to_entity.thermal_stress_likelihood",
        )
        records.append(rec)
        m.stress_prov[b.belief_id] = rec.record_id
        summary = f"stress likelihood INFERRED from temperature belief (gate={stress_gate:.2f}); not damage"
    if evidence:
        ev_ids = list(dict.fromkeys(e.evidence_id for e in evidence))
        parents = [head.root, *(r.record_id for r in records)]
        rec = m.ledger.record(
            SourceType.DIRECT_OBSERVATION, ev_ids, b.belief_id, t, parents, "entity.survey_update"
        )
        records.append(rec)
        m.direct_prov[b.belief_id] = rec.record_id
    kind = UpdateKind.DIRECT if evidence else UpdateKind.RELATIONAL
    life = next_lifecycle(head.lifecycle, b.n_hits, b.asset is not None) if evidence else head.lifecycle
    uniq = tuple({e.evidence_id: e for e in evidence}.values())
    return m.ledger.commit(entity_spec(m, b, kind, tuple(records), uniq, life, t, summary), m.availability())
