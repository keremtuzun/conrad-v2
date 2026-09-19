from __future__ import annotations

from uuid import UUID

import pytest
from m2t_helpers import DAY, claim, evidence, model

from conrad.domains.technical import CORROSION_DEPTH, Model2T, PropagationMode
from conrad.schemas.belief import (
    Availability,
    BeliefMessage,
    BeliefQuery,
    EcologicalPayload,
    KnowledgeStatus,
    Lifecycle,
    UpdateKind,
)
from conrad.schemas.ids import IdFactory
from conrad.schemas.provenance import ProvenanceRecord, SourceType
from conrad.schemas.timebase import stamp
from conrad.schemas.uncertainty import Uncertainty
from conrad.schemas.world import Domain


def _biofouling(ids: IdFactory, target, cover: float) -> BeliefMessage:
    return BeliefMessage(
        message_id=ids.new(),
        belief_id=ids.new(),
        revision=0,
        world_entity_id=target,
        domain=Domain.ECOLOGICAL,
        timestamp=stamp(20.0, "sim"),
        state_summary=(),
        state_embedding=(),
        knowledge_status=KnowledgeStatus.OBSERVED,
        uncertainty=Uncertainty(aleatoric=0.1, epistemic=0.1, contradiction=0.0, observational=0.1),
        provenance_refs=(ids.new(),),
        lifecycle=Lifecycle.ACTIVE,
        model_version="2e-test",
        ecological=EcologicalPayload(kind="FIELD", quantities={"biofouling_cover": cover}),
    )


def test_predict_sets_predicted_status_and_grows_uncertainty():
    m, r, ev = model(mode=PropagationMode.NONE)
    m.ingest([evidence(ev, r["seg_a"], 10.0, wall=1e-3, crack=2e-3)])
    obs = m.update_beliefs(stamp(10.0, "sim"))[0]
    msgs = m.predict(90 * DAY, stamp(10.0 + 90 * DAY, "sim"))
    pred = next(x for x in msgs if x.world_entity_id == r["seg_a"])
    c = claim(pred, CORROSION_DEPTH)
    assert c.status is KnowledgeStatus.PREDICTED
    assert all(x.status is not KnowledgeStatus.OBSERVED for x in pred.state_summary)
    assert c.value > claim(obs, CORROSION_DEPTH).value  # engineering-prior rate is positive
    assert claim(pred, "corrosion_depth_m.variance").value > claim(obs, "corrosion_depth_m.variance").value
    assert pred.uncertainty.observational > obs.uncertainty.observational
    known = {x.status for x in pred.state_summary if x.status is not KnowledgeStatus.UNKNOWN}
    assert known == {KnowledgeStatus.PREDICTED} and not pred.evidence_support
    with pytest.raises(ValueError):
        m.predict(-1.0, stamp(0.0, "sim"))


def test_unknown_stays_unknown_under_prediction():
    m, r, _ = model(mode=PropagationMode.NONE)
    m.predict(365 * DAY, stamp(365 * DAY, "sim"))
    msg = m.query(BeliefQuery(entity_ids=(r["seg_b"],)))[0]
    assert claim(msg, CORROSION_DEPTH).status is KnowledgeStatus.UNKNOWN
    assert msg.uncertainty.observational == 1.0


def test_biofouling_context_raises_uo_ua_not_severity():
    m, r, ev = model()
    m.ingest([evidence(ev, r["seg_a"], 10.0, wall=2e-3, anomaly=0.4)])
    m.update_beliefs(stamp(10.0, "sim"))
    before = m.query(BeliefQuery(entity_ids=(r["seg_a"],)))[0]
    captured = []
    orig = m.engine.drain_provenance

    def spy(rid: UUID) -> list[ProvenanceRecord]:
        records = orig(rid)
        captured.extend(records)
        return records

    m.engine.drain_provenance = spy
    m.receive_context([_biofouling(IdFactory(seed=5), r["seg_a"], 0.9)])
    after = m.query(BeliefQuery(entity_ids=(r["seg_a"],)))[0]
    assert after.uncertainty.observational > before.uncertainty.observational
    assert after.uncertainty.aleatoric > before.uncertainty.aleatoric
    for name in (CORROSION_DEPTH, "severity", "condition"):
        assert claim(after, name).value == claim(before, name).value
    assert after.technical.severity == before.technical.severity
    assert any(p.source_type is SourceType.CROSS_DOMAIN_CONTEXT for p in captured)


def test_reset_preserves_persistent_state():
    m, r, ev = model()
    m.ingest([evidence(ev, r["seg_a"], 10.0, wall=2e-3)])
    m.update_beliefs(stamp(10.0, "sim"))
    before = [(x.belief_id, x.revision, x.state_summary) for x in m.export_beliefs()]
    m.ingest([evidence(ev, r["seg_a"], 11.0, wall=9e-3)])
    m.reset_working_memory()
    assert m.update_beliefs(stamp(12.0, "sim")) == []
    after = [(x.belief_id, x.revision, x.state_summary) for x in m.export_beliefs()]
    assert before == after


def test_availability_and_query_filters():
    fresh = Model2T(IdFactory(seed=1))
    assert fresh.availability() is Availability.UNAVAILABLE
    m, r, ev = model()
    assert m.availability() is Availability.AVAILABLE
    assert m.query(BeliefQuery(domain=Domain.SPATIAL)) == []
    only = m.query(BeliefQuery(entity_ids=(r["seg_a"],), requested_fields=(CORROSION_DEPTH,)))
    assert [c.name for c in only[0].state_summary] == [CORROSION_DEPTH]
    hidden = m.query(BeliefQuery(min_observational_uncertainty=0.99))
    assert len(hidden) == len(m.beliefs)
    stray = evidence(ev, IdFactory(seed=77).new(), 5.0, wall=1e-3)
    m.ingest([stray])
    assert m.availability() is Availability.DEGRADED


def test_persistence_commits_every_kind(repo):
    run_id = IdFactory(seed=9).new()
    m, r, ev = model(repository=repo, run_id=run_id)
    m.ingest([evidence(ev, r["seg_a"], 10.0, wall=3e-3, crack=6e-3, run_id=run_id)])
    m.update_beliefs(stamp(10.0, "sim"))
    m.predict(30 * DAY, stamp(10.0 + 30 * DAY, "sim"))
    m.receive_context([_biofouling(IdFactory(seed=5), r["seg_a"], 0.5)])
    m.ingest([evidence(ev, r["seg_a"], 5.0, wall=3e-3, run_id=run_id)])  # late evidence
    m.update_beliefs(stamp(11.0 + 30 * DAY, "sim"))
    kinds = {rev.update_kind for rev in repo.all_revisions(run_id)}
    assert {
        UpdateKind.CREATE,
        UpdateKind.DIRECT,
        UpdateKind.RELATIONAL,
        UpdateKind.PREDICTED,
        UpdateKind.CONTEXT,
    } <= kinds
    seg_a = m.beliefs[r["seg_a"]]
    head = repo.head(seg_a.belief_id)
    assert head is not None and head.revision == seg_a.revision
    assert any(rev.late for rev in repo.revisions(seg_a.belief_id))
    closure = repo.provenance_closure(head.provenance_root)
    assert any(p.source_type is SourceType.PRIOR for p in closure.values())
    assert len(repo.heads(run_id)) == len(m.beliefs)
