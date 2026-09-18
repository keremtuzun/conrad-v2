"""AnalyticBUO behaviour: each uncertainty channel responds to its own cause; trust != innovation."""

from conrad.core.buo import AnalyticBUO
from conrad.core.state import unknown_state
from conrad.schemas.observation import EvidenceValidity


def _prior(fac, n=3, x=0.5):
    buo, st = AnalyticBUO(), unknown_state()
    for t in range(n):
        st = buo.update(st, [fac.make(float(t), {"p": x}, reliability=0.9, aleatoric=0.0025)[0]]).state
    return buo, st


def test_corrupted_evidence_raises_ua(fac):
    buo, st = _prior(fac)
    clean = buo.update(st, [fac.make(5.0, {"p": 0.5}, reliability=0.9, aleatoric=0.0025)[0]]).state
    bad = buo.update(
        st,
        [fac.make(5.0, {"p": 0.5}, reliability=0.2, aleatoric=0.09, validity=EvidenceValidity.DEGRADED)[0]],
    ).state
    assert bad.ua > clean.ua


def test_ood_score_raises_ue_and_unmeasured_ood_keeps_it(fac):
    buo, st = _prior(fac)
    ind = buo.update(st, [fac.make(5.0, {"p": 0.5}, ood_score=0.05)[0]]).state
    ood = buo.update(st, [fac.make(5.0, {"p": 0.5}, ood_score=0.95)[0]]).state
    unmeasured = buo.update(st, [fac.make(5.0, {"p": 0.5}, ood_score=None)[0]]).state
    assert ood.ue > ind.ue
    assert unmeasured.ue == st.ue


def test_credible_contradiction_raises_uc_and_is_not_averaged_away(fac):
    buo, st = _prior(fac, x=0.2)
    res = buo.update(st, [fac.make(5.0, {"p": 0.9}, reliability=0.95, aleatoric=0.0025)[0]])
    assert res.state.uc > st.uc
    assert res.conflict_ids and res.state.contradictions
    est = res.state.estimates["p"]
    assert est.variance > st.estimates["p"].variance  # disagreement widens, never sharpens
    support = buo.update(st, [fac.make(5.0, {"p": 0.2}, reliability=0.95, aleatoric=0.0025)[0]]).state
    assert support.uc <= st.uc


def test_unreliable_disagreement_is_ambiguous_not_contradiction(fac):
    buo, st = _prior(fac, x=0.2)
    res = buo.update(st, [fac.make(5.0, {"p": 0.9}, reliability=0.2, aleatoric=0.01)[0]])
    assert not res.conflict_ids
    assert res.assessments[0].outcome == "AMBIGUOUS"


def test_trust_is_not_innovation(fac):
    buo, st = _prior(fac, x=0.2)
    reliable_conflict = buo.update(st, [fac.make(5.0, {"p": 0.9}, reliability=0.95)[0]]).assessments[0]
    reliable_support = buo.update(st, [fac.make(5.0, {"p": 0.2}, reliability=0.95)[0]]).assessments[0]
    # identical trust, very different innovation: conflict does NOT lower trust
    assert reliable_conflict.trust == reliable_support.trust
    assert reliable_conflict.innovation > reliable_support.innovation


def test_repeated_independent_support_resolves_contradiction(fac):
    buo, st = _prior(fac, x=0.2)
    st = buo.update(st, [fac.make(5.0, {"p": 0.9}, reliability=0.95)[0]]).state
    uc_peak = st.uc
    for t in range(6, 12):
        st = buo.update(
            st, [fac.make(float(t), {"p": 0.9}, reliability=0.95, independence_group=f"g{t}")[0]]
        ).state
    assert st.uc < uc_peak
    assert abs(st.estimates["p"].mean - 0.9) < 0.1
    assert not st.contradictions


def test_low_coverage_raises_uo(fac):
    buo = AnalyticBUO()
    one = buo.update(unknown_state(), [fac.make(0.0, {"p": 0.5})[0]]).state
    _, many = _prior(fac, n=5)
    occluded = buo.update(unknown_state(), [fac.make(0.0, {"p": 0.5}, occlusion=0.9)[0]]).state
    assert one.uo > many.uo
    assert occluded.uo > one.uo


def test_duplicates_are_ignored(fac):
    buo, st = _prior(fac)
    ev = fac.make(5.0, {"p": 0.55})[0]
    first = buo.update(st, [ev, ev])
    assert first.rejected_duplicate_ids == (ev.evidence_id,)
    again = buo.update(first.state, [ev])
    assert not again.changed and again.state is first.state


def test_near_identical_same_group_evidence_does_not_create_certainty(fac):
    buo = AnalyticBUO()
    base = buo.update(unknown_state(), [fac.make(0.0, {"p": 0.5}, independence_group="frame1")[0]]).state
    st = base
    for _ in range(20):
        st = buo.update(st, [fac.make(0.0, {"p": 0.5}, independence_group="frame1")[0]]).state
    assert st.estimates["p"].variance >= base.estimates["p"].variance - 1e-12
    assert st.uo == base.uo and st.independent_observation_count == 1
    indep = base
    for t in range(20):
        indep = buo.update(indep, [fac.make(float(t + 1), {"p": 0.5}, independence_group=f"f{t}")[0]]).state
    assert indep.estimates["p"].variance < st.estimates["p"].variance


def test_invalid_evidence_does_not_move_state(fac):
    buo, st = _prior(fac)
    res = buo.update(st, [fac.make(5.0, {"p": 0.99}, validity=EvidenceValidity.INVALID)[0]])
    assert res.state.estimates["p"] == st.estimates["p"]
