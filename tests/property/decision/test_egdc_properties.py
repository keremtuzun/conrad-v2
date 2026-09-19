"""Property tests: closed vocabulary, UIR = 0, constraint determinism, cause-specific needs."""

from hypothesis import given, settings
from hypothesis import strategies as st

from conrad.decision import ACTION_VOCABULARY, EGDC, unsupported_inference_rate
from conrad.evaluation.decision_experiments.fixtures import make_belief, make_context, make_requirement, unc
from conrad.schemas.decision import ActionType, InformationNeed
from conrad.schemas.ids import IdFactory

channel = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


@settings(max_examples=40, deadline=None)
@given(
    ua=channel,
    ue=channel,
    uc=channel,
    uo=channel,
    consequence=channel,
    n_ev=st.integers(0, 3),
    calibrated=st.booleans(),
    battery=st.floats(0.0, 1.0),
    stale=st.booleans(),
)
def test_egdc_invariants(ua, ue, uc, uo, consequence, n_ev, calibrated, battery, stale):
    ids = IdFactory(123)
    b = make_belief(
        ids, uncertainty=unc(ua, ue, uc, uo, calibrated), n_evidence=n_ev, time_s=10.0 if stale else 100.0
    )
    req = make_requirement(ids, belief_ids=[b.belief_id], consequence=consequence)
    ctx = make_context(ids, [b], [req], battery=battery)
    out = EGDC(ids).decide(ctx)
    rec = out.record
    assert all(a.action_type in ACTION_VOCABULARY for a in rec.candidates)
    assert unsupported_inference_rate([rec]) == 0.0
    assert len(rec.constraint_decisions) >= 1
    assert rec.chosen is None or rec.constraint_decisions[-1].accepted
    assert all(not d.accepted for d in rec.constraint_decisions[:-1])
    if rec.chosen is not None and rec.chosen.action_type is ActionType.CONTINUE_MISSION:
        assert battery >= 0.2
        grounded = {c.claim_id for c in rec.claims if c.grounding.value == "GROUNDED"}
        assert set(rec.chosen.supporting_claims) <= grounded
    payload = getattr(out.routed, "payload", None)
    if isinstance(payload, InformationNeed):
        assert payload.target_belief_ids == (b.belief_id,)


@settings(max_examples=20, deadline=None)
@given(ua=channel, ue=channel, uc=channel, uo=channel)
def test_decisions_replay_deterministically(ua, ue, uc, uo):
    def once():
        ids = IdFactory(7)
        b = make_belief(ids, uncertainty=unc(ua, ue, uc, uo))
        ctx = make_context(ids, [b], [make_requirement(ids, belief_ids=[b.belief_id])])
        return EGDC(ids).decide(ctx).record.canonical_json()

    assert once() == once()
