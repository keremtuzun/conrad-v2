from __future__ import annotations

from m2t_helpers import DAY, claim, evidence, model

from conrad.domains.technical import CORROSION_DEPTH, PropagationMode
from conrad.schemas.belief import KnowledgeStatus
from conrad.schemas.timebase import stamp


def test_direct_update_is_observed_and_reduces_uo():
    m, r, ev = model(mode=PropagationMode.NONE)
    before = m.beliefs[r["seg_a"]].uncertainty().observational
    m.ingest([evidence(ev, r["seg_a"], 10.0, wall=1.0e-3, crack=0.0, anomaly=0.3)])
    msgs = m.update_beliefs(stamp(10.0, "sim"))
    assert len(msgs) == 1
    c = claim(msgs[0], CORROSION_DEPTH)
    assert c.status is KnowledgeStatus.OBSERVED and abs(c.value - 1.0e-3) < 3e-4
    assert msgs[0].technical.direct_support > 0.5 and msgs[0].technical.propagated_support == 0.0
    assert msgs[0].uncertainty.observational < before
    assert msgs[0].evidence_support


def test_contradiction_raises_uc_and_is_not_averaged_away():
    def run(second_wall: float):
        m, r, ev = model(mode=PropagationMode.NONE)
        for i in range(3):
            m.ingest([evidence(ev, r["seg_a"], 10.0 + i, wall=0.5e-3)])
            m.update_beliefs(stamp(10.0 + i, "sim"))
        m.ingest([evidence(ev, r["seg_a"], 20.0, wall=second_wall)])
        msg = m.update_beliefs(stamp(20.0, "sim"))[0]
        return msg, m.beliefs[r["seg_a"]].estimates[CORROSION_DEPTH]

    agree, est_agree = run(0.5e-3)
    clash, est_clash = run(6.0e-3)
    assert clash.uncertainty.contradiction > agree.uncertainty.contradiction
    assert clash.evidence_conflicts and not agree.evidence_conflicts
    # the disagreement survives in the variance instead of collapsing into false certainty
    assert est_clash.level_var > 10 * est_agree.level_var


def test_independence_group_counted_once():
    def variance(groups: list[str | None]) -> float:
        m, r, ev = model(mode=PropagationMode.NONE)
        m.ingest([evidence(ev, r["seg_a"], 10.0, wall=1e-3, group=g) for g in groups])
        m.update_beliefs(stamp(10.0, "sim"))
        return m.beliefs[r["seg_a"]].estimates[CORROSION_DEPTH].level_var

    one = variance(["frame-1"])
    same_group = variance(["frame-1"] * 5)
    independent = variance([None] * 5)
    assert abs(same_group - one) < 1e-18
    assert independent < 0.5 * one


def test_repeated_group_later_never_shrinks_variance():
    m, r, ev = model(mode=PropagationMode.NONE)
    m.ingest([evidence(ev, r["seg_a"], 10.0, wall=1e-3, group="g")])
    m.update_beliefs(stamp(10.0, "sim"))
    v1 = m.beliefs[r["seg_a"]].estimates[CORROSION_DEPTH].level_var
    ds1 = m.beliefs[r["seg_a"]].direct_support
    m.ingest([evidence(ev, r["seg_a"], 10.0, wall=1e-3, group="g")])
    m.update_beliefs(stamp(10.0, "sim"))
    assert m.beliefs[r["seg_a"]].estimates[CORROSION_DEPTH].level_var >= v1
    assert m.beliefs[r["seg_a"]].direct_support == ds1


def test_duplicate_evidence_ignored():
    m, r, ev = model(mode=PropagationMode.NONE)
    e = evidence(ev, r["seg_a"], 10.0, wall=1e-3)
    m.ingest([e, e])
    m.update_beliefs(stamp(10.0, "sim"))
    v1 = m.beliefs[r["seg_a"]].estimates[CORROSION_DEPTH].level_var
    m.ingest([e])
    assert m.update_beliefs(stamp(11.0, "sim")) == []
    assert m.beliefs[r["seg_a"]].estimates[CORROSION_DEPTH].level_var == v1


def test_corrupted_evidence_raises_ua():
    def ua(level: float) -> float:
        m, r, ev = model(mode=PropagationMode.NONE)
        m.ingest([evidence(ev, r["seg_a"], 10.0, wall=1e-3, ua=level, rel=0.9)])
        return m.update_beliefs(stamp(10.0, "sim"))[0].uncertainty.aleatoric

    assert ua(0.9) > ua(0.1)


def test_non_susceptible_component_gets_no_degradation_claim():
    m, r, ev = model(mode=PropagationMode.NONE)
    m.ingest([evidence(ev, r["support_concrete"], 10.0, wall=2e-3, crack=5e-3)])
    msg = m.update_beliefs(stamp(10.0 + DAY, "sim"))[0]
    names = {c.name for c in msg.state_summary}
    assert CORROSION_DEPTH not in names and msg.technical.corrosion_depth_m is None
    assert msg.technical.direct_support > 0  # it WAS inspected
