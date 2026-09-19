from __future__ import annotations

import numpy as np
import pytest

from conrad.domains.ecological import DAMAGE_CLAIM, STRESS_CLAIM, EntityBelief, FieldBelief
from conrad.schemas.belief import Availability, BeliefQuery, KnowledgeStatus
from conrad.schemas.frames import SpatialSupport
from conrad.schemas.provenance import SourceType
from conrad.schemas.timebase import stamp

K = KnowledgeStatus


def _msgs_by_kind(msgs, kind):
    return [m for m in msgs if m.ecological is not None and m.ecological.kind == kind]


def test_entity_and_field_beliefs_are_separate_state_types(make_model, maker, reg):
    m = make_model()
    out = m.update_beliefs(stamp(1, "SIM"))
    assert out == []
    m.ingest([maker.field("temperature", 18.0, 1.0)])
    msgs = m.update_beliefs(stamp(2, "SIM"))
    assert _msgs_by_kind(msgs, "FIELD")
    # field evidence reaches entities only as RELATIONAL context (stress), never as their evidence
    for em in _msgs_by_kind(msgs, "ENTITY"):
        assert not em.evidence_support
        assert all(c.status is K.UNKNOWN for c in em.state_summary if c.name.startswith("cover_fraction"))
    unc = make_model("uncoupled", with_repo=False)
    unc.ingest([maker.field("temperature", 18.0, 1.0)])
    assert not _msgs_by_kind(unc.update_beliefs(stamp(2, "SIM")), "ENTITY")
    m.ingest([maker.cover(0.4, 3.0, reg["pipe"][1])])
    msgs = m.update_beliefs(stamp(4, "SIM"))
    kinds = {mm.ecological.kind for mm in msgs}
    # the survey created an ENTITY revision; the field stream was not touched by it
    assert "ENTITY" in kinds and m.fields.fields["temperature"].n_obs == 1
    assert all(isinstance(fb, FieldBelief) for fb in m.fields.fields.values())
    assert all(isinstance(b, EntityBelief) for b in m.entities.beliefs.values())
    assert set(m.field_ids.values()).isdisjoint(m.entities.beliefs)
    for mm in m.export_beliefs():
        names = {c.name for c in mm.state_summary}
        if mm.ecological.kind == "FIELD":
            assert not names & {"cover_fraction", "presence_probability"}
        else:
            assert not any(n.startswith(("temperature", "turbidity")) for n in names)
        assert all(u for u in mm.ecological.units.values()), "units are carried everywhere"
        assert set(mm.ecological.quantities) == set(mm.ecological.units)


def test_field_only_baseline_ignores_survey_evidence(make_model, maker, reg):
    m = make_model("field_only")
    assert not m.entities.beliefs
    m.ingest([maker.cover(0.5, 1.0, reg["pipe"][1]), maker.field("turbidity", 4.0, 1.0)])
    msgs = m.update_beliefs(stamp(2, "SIM"))
    assert {mm.ecological.kind for mm in msgs} == {"FIELD"}
    assert m.stats["ignored_entity_evidence"] == 1


def test_variance_grows_away_from_sensor_and_with_time(make_model, maker):
    m = make_model()
    sensor = np.array([-30.0, -30.0, -17.5])
    # turbidity has no depth trend, so the residual variance is purely the local-deviation GP
    m.ingest([maker.field("turbidity", 5.0, t, pos=tuple(sensor)) for t in (9.0, 10.0)])
    m.update_beliefs(stamp(10, "SIM"))
    fb = m.fields.fields["turbidity"]
    d = np.linalg.norm(m.fields.grid.centers - sensor, axis=1)
    near, far = np.argmin(d), np.argmax(d)
    assert fb.resid_var[0, near] < 0.5 * fb.resid_var[0, far]  # local variance
    # far away the local variance returns to the (empirical-Bayes) local scale; resid_var also carries the
    # level/deviation posterior covariance, hence the ~1 % tolerance
    assert fb.resid_var[0, far] == pytest.approx(fb.local_var, rel=2e-2)
    assert fb.var[0, near] < fb.var[0, far]  # total variance too
    k = m.fields.grid.kernel(sensor, fb.spec)  # anisotropic correlation to the sensor
    order = np.argsort(-k)
    assert np.all(np.diff(fb.var[0, order]) >= -1e-9)  # variance grows as correlation falls
    t0 = fb.time_ns
    _, v1 = m.fields.moments_at_time("turbidity", t0 + 3600 * 10**9)
    _, v2 = m.fields.moments_at_time("turbidity", t0 + 36000 * 10**9)
    assert v1[0].min() > fb.var[0].min() and v2[0].min() > v1[0].min()
    p_short = m.predict(60.0, stamp(10, "SIM"))
    p_long = m.predict(86400.0, stamp(10, "SIM"))

    def sd(ms):
        return next(x for x in ms if x.belief_id == m.field_ids["turbidity"]).ecological.quantities["sd"]

    assert sd(p_long) > sd(p_short)
    assert all(c.status in (K.PREDICTED, K.UNKNOWN) for x in p_long for c in x.state_summary)
    assert all(not x.evidence_support for x in p_long)


def test_each_observation_uses_its_own_timestamp(make_model, maker):
    m = make_model()
    m.ingest(
        [
            maker.field("turbidity", 5.0, 25.0),
            maker.field("temperature", 19.0, 10.0),
            maker.field("temperature", 19.5, 500.0),  # future relative to 'now': stays pending
        ]
    )
    m.update_beliefs(stamp(100, "SIM"))
    assert m.fields.fields["temperature"].time_ns == 10 * 10**9
    assert m.fields.fields["turbidity"].time_ns == 25 * 10**9
    assert len(m.pending) == 1
    m.ingest([maker.field("temperature", 18.0, 5.0)])  # older than the head: late, not re-timed
    m.update_beliefs(stamp(100, "SIM"))
    assert m.stats["late_evidence"] == 1
    assert m.fields.fields["temperature"].time_ns == 10 * 10**9


def test_stress_is_inferred_and_never_observed_damage(make_model, maker, reg, repo):
    m = make_model()
    coral_id, coral_pos = reg["coral"]
    m.ingest([maker.field("temperature", 29.0, float(t), pos=coral_pos) for t in range(1, 6)])
    m.ingest([maker.cover(0.6, 6.0, coral_pos)])
    msgs = m.update_beliefs(stamp(10, "SIM"))
    coral = next(x for x in msgs if x.world_entity_id == coral_id)
    stress = next(c for c in coral.state_summary if c.name == STRESS_CLAIM)
    damage = next(c for c in coral.state_summary if c.name == DAMAGE_CLAIM)
    assert stress.status is K.INFERRED and stress.value > 0.9
    assert damage.status is K.UNKNOWN and damage.value is None
    assert coral.knowledge_status is K.MIXED
    rec = repo.provenance_record(stress.provenance_id)
    assert rec.source_type is SourceType.RELATIONAL_INFERENCE
    for x in [*m.export_beliefs(), *m.predict(3600.0, stamp(10, "SIM"))]:
        for c in x.state_summary:
            if c.name == DAMAGE_CLAIM:
                assert c.status is K.UNKNOWN
            if c.name == STRESS_CLAIM:
                assert c.status in (K.INFERRED, K.PREDICTED)
    # stress does not move the cover mean (environmental stress is not ecological change)
    assert m.entities.by_registry(coral_id).cover_mean == pytest.approx(
        next(c.value for c in coral.state_summary if c.name == "cover_fraction")
    )


def test_biofouling_context_message_has_cross_domain_provenance(make_model, maker, reg, repo):
    m = make_model()
    pipe_id, pipe_pos = reg["pipe"]
    m.ingest([maker.cover(0.55, 2.0, pipe_pos), maker.field("turbidity", 8.0, 2.0, pos=pipe_pos)])
    m.update_beliefs(stamp(3, "SIM"))
    ctx = m.observability_context(stamp(3, "SIM"))
    bio = [x for x in ctx if x.world_entity_id == pipe_id]
    assert len(bio) == 1
    msg = bio[0]
    assert "CROSS_DOMAIN_CONTEXT" in msg.change_summary and "not evidence of corrosion" in msg.change_summary
    assert repo.provenance_record(msg.provenance_refs[0]).source_type is SourceType.CROSS_DOMAIN_CONTEXT
    assert all(c.status is K.INFERRED for c in msg.state_summary)
    assert {c.name for c in msg.state_summary} >= {"surface_biofouling_cover", "surface_occlusion_fraction"}
    turb = [x for x in ctx if x.ecological.kind == "FIELD"]
    assert len(turb) == 4 and all("observability of region" in x.change_summary for x in turb)
    assert all(x.ecological.units["turbidity_ntu"] == "NTU" for x in turb)


def test_internal_exception_sets_unavailable_and_is_raised(make_model, maker, monkeypatch):
    m = make_model()
    assert m.availability() is Availability.AVAILABLE

    def boom(*a, **k):
        raise FloatingPointError("injected")

    monkeypatch.setattr(m.fields, "update_point", boom)
    m.ingest([maker.field("temperature", 18.0, 1.0)])
    with pytest.raises(FloatingPointError):
        m.update_beliefs(stamp(2, "SIM"))
    assert m.availability() is Availability.UNAVAILABLE
    with pytest.raises(RuntimeError, match="UNAVAILABLE"):
        m.export_beliefs()
    assert m.availability() is Availability.UNAVAILABLE


def test_reset_working_memory_preserves_persistent_state(make_model, maker, reg, repo):
    m = make_model()
    m.ingest([maker.cover(0.3, 1.0, reg["pipe"][1]), maker.field("temperature", 17.0, 1.0)])
    m.update_beliefs(stamp(2, "SIM"))
    m.ingest([maker.field("temperature", 30.0, 5.0)])  # pending, never applied
    m.receive_context([])
    before = {x.belief_id: (x.revision, x.canonical_json()) for x in m.export_beliefs()}
    heads_before = {c.belief_id: c.revision for c in repo.heads()}
    m.reset_working_memory()
    assert m.pending == [] and m.context == {}
    after = {x.belief_id: (x.revision, x.canonical_json()) for x in m.export_beliefs()}
    assert after == before
    assert {c.belief_id: c.revision for c in repo.heads()} == heads_before
    assert m.availability() is Availability.AVAILABLE


def test_persisted_revisions_and_query(make_model, maker, reg, repo):
    m = make_model()
    pipe_id, pipe_pos = reg["pipe"]
    m.ingest([maker.cover(0.3, 1.0, pipe_pos), maker.cover(0.35, 2.0, pipe_pos)])
    m.update_beliefs(stamp(3, "SIM"))
    b = m.entities.by_registry(pipe_id)
    revs = repo.revisions(b.belief_id)
    assert [r.revision for r in revs] == [0, 1] and len(revs[1].consumed_evidence_ids) == 2
    assert repo.head(b.belief_id).independent_observation_count == 2
    q = m.query(BeliefQuery(entity_ids=(pipe_id,)))
    assert len(q) == 1 and q[0].belief_id == b.belief_id
    region = SpatialSupport(frame_id="WORLD", center_m=(-30.0, -30.0, -17.5), half_extent_m=(10, 10, 2.5))
    rq = m.query(BeliefQuery(region=region, requested_fields=("temperature",)))
    assert len(rq) == 1 and rq[0].ecological.quantities["cells"] == 1.0


def test_spatial_context_is_context_only(make_model, maker, reg):
    m = make_model()
    fake = m.export_beliefs()[0].model_copy(update={"domain": m.domain})
    m.receive_context([fake])  # own-domain messages are never re-ingested
    assert m.context == {}
    assert m.gate_inflation() == 1.0
