from __future__ import annotations

from m2t_helpers import claim, evidence, model

from conrad.domains.technical import CORROSION_DEPTH, CRACK_LENGTH, PropagationMode, relational_contamination
from conrad.schemas.belief import KnowledgeStatus
from conrad.schemas.provenance import SourceType
from conrad.schemas.timebase import stamp


def _observe(m, r, ev, name, **meas):
    m.ingest([evidence(ev, r[name], 10.0, **meas)])
    return {msg.world_entity_id: msg for msg in m.update_beliefs(stamp(10.0, "sim"))}


def test_propagated_is_inferred_with_relational_provenance():
    m, r, ev = model()
    msgs = _observe(m, r, ev, "seg_a", wall=3e-3, crack=8e-3)
    b = msgs[r["seg_b"]]
    c = claim(b, CORROSION_DEPTH)
    assert c.status is KnowledgeStatus.INFERRED
    assert b.technical.propagated_support > 0 and b.technical.direct_support == 0
    assert b.uncertainty.observational == 1.0  # never reduced as if observed
    assert not b.evidence_support
    assert m.engine.pending.get(r["seg_b"], []) == []  # drained at commit
    a = msgs[r["seg_a"]]
    assert claim(a, CORROSION_DEPTH).status is KnowledgeStatus.OBSERVED
    assert a.technical.direct_support > 0 and a.technical.propagated_support == 0


def test_relational_record_source_type():
    m, r, ev = model()
    captured = []
    orig = m.engine.drain_provenance

    def spy(rid):
        recs = orig(rid)
        captured.extend(recs)
        return recs

    m.engine.drain_provenance = spy
    _observe(m, r, ev, "seg_a", wall=3e-3)
    seg_b_bid = m.beliefs[r["seg_b"]].belief_id
    rel = [p for p in captured if p.subject_id == seg_b_bid]
    assert rel and all(p.source_type is SourceType.RELATIONAL_INFERENCE for p in rel)
    assert m.beliefs[r["seg_a"]].belief_id in rel[0].source_ids


def test_no_propagation_over_invalid_relation_types():
    m, r, ev = model()
    _observe(m, r, ev, "seg_a", wall=4e-3, crack=1e-2)
    beliefs = m.beliefs
    # ADJACENT_TO carries nothing; concrete is not susceptible; SUPPORTED_BY is not a corrosion path
    assert beliefs[r["seg_adj"]].estimates[CORROSION_DEPTH].status is KnowledgeStatus.UNKNOWN
    assert beliefs[r["seg_adj"]].estimates[CRACK_LENGTH].status is KnowledgeStatus.UNKNOWN
    assert beliefs[r["support_concrete"]].estimates[CORROSION_DEPTH].status is KnowledgeStatus.UNKNOWN
    assert beliefs[r["support_steel"]].estimates[CORROSION_DEPTH].status is KnowledgeStatus.UNKNOWN
    # the load path does carry fatigue
    assert beliefs[r["support_steel"]].estimates[CRACK_LENGTH].status is KnowledgeStatus.INFERRED
    # CONNECTED_TO (shared environment) carries corrosion but not fatigue
    assert beliefs[r["seg_b"]].estimates[CORROSION_DEPTH].status is KnowledgeStatus.INFERRED
    assert beliefs[r["seg_b"]].estimates[CRACK_LENGTH].status is KnowledgeStatus.UNKNOWN


def test_generic_baseline_contaminates_invalid_edges():
    m, r, ev = model(mode=PropagationMode.GENERIC)
    _observe(m, r, ev, "seg_a", wall=4e-3)
    assert m.beliefs[r["seg_adj"]].estimates[CORROSION_DEPTH].status is KnowledgeStatus.INFERRED
    assert m.beliefs[r["support_concrete"]].estimates[CORROSION_DEPTH].status is KnowledgeStatus.INFERRED


def test_healthy_observed_neighbour_is_never_overridden():
    def seg_b_state(mode):
        m, r, ev = model(mode=mode)
        m.ingest(
            [evidence(ev, r["seg_b"], 10.0, wall=0.0, crack=0.0), evidence(ev, r["seg_a"], 10.0, wall=6e-3)]
        )
        m.update_beliefs(stamp(10.0, "sim"))
        e = m.beliefs[r["seg_b"]].estimates[CORROSION_DEPTH]
        return e.level, e.level_var, e.status

    for mode in (PropagationMode.TCDP, PropagationMode.GENERIC):
        assert seg_b_state(mode) == seg_b_state(PropagationMode.NONE)
        assert seg_b_state(mode)[2] is KnowledgeStatus.OBSERVED


def test_unreliable_source_is_gated():
    m, r, ev = model()
    _observe(m, r, ev, "seg_a", wall=4e-3, rel=0.1)
    assert m.beliefs[r["seg_b"]].estimates[CORROSION_DEPTH].status is KnowledgeStatus.UNKNOWN


def test_message_is_attenuated():
    m, r, ev = model()
    _observe(m, r, ev, "seg_a", wall=4e-3)
    a = m.beliefs[r["seg_a"]].estimates[CORROSION_DEPTH]
    b = m.beliefs[r["seg_b"]].estimates[CORROSION_DEPTH]
    prior = m.beliefs[r["seg_b"]].prior[CORROSION_DEPTH]
    assert prior.level < b.level < a.level
    assert b.level_var > a.level_var


def test_contamination_metric():
    assert relational_contamination([True, False, True], [True, True, False], [True, True, True]) == 0.5
    assert relational_contamination([True], [False], [True]) is None


def test_message_shift_is_bounded():
    m, r, ev = model()
    _observe(m, r, ev, "seg_a", crack=1.0)  # a failed 1 m crack must not become a 300 mm neighbour crack
    sup = m.beliefs[r["support_steel"]]
    prior = sup.prior[CRACK_LENGTH]
    est = sup.estimates[CRACK_LENGTH]
    assert est.status is KnowledgeStatus.INFERRED
    assert est.level <= prior.level + 3.0 * prior.sd + 1e-12
