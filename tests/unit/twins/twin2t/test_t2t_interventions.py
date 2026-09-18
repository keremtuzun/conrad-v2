from __future__ import annotations

from dataclasses import replace

import pytest
from t2t_unit_helpers import YEAR, engine, new_ids, runtime

from conrad.twins.twin2t.config import MCDEConfig
from conrad.twins.twin2t.events import StructuralEventError
from conrad.twins.twin2t.events import StructuralEventType as T
from conrad.twins.twin2t.mcde import StructuralEvent

DET = MCDEConfig(stochastic=False)


def ev(t, target, **p):
    return StructuralEvent(t, target, p)


def test_repair_resets_then_degradation_resumes():
    (a,) = new_ids(1)
    rt = runtime(a, corrosion=0.004, crack=6e-3, coated=True, coating=0.5)
    eng = engine([rt], cfg=DET)
    eng.step(YEAR)
    res = eng.step(1.0, [ev(T.REPAIR, a)])
    assert rt.state.corrosion_depth_m == 0.0 and rt.state.crack_length_m == 0.0
    assert rt.state.coating_breakdown_fraction == 0.0
    assert res.records[a].intervened
    for _ in range(5):
        eng.step(YEAR)
    assert rt.state.corrosion_depth_m > 0.0  # new trajectory after intervention


def test_partial_repair_and_replacement():
    (a,) = new_ids(1)
    rt = runtime(a, corrosion=0.004, crack=6e-3)
    eng = engine([rt], cfg=DET)
    eng.step(1.0, [ev(T.REPAIR, a, restore_fraction=0.5, remove_cracks=False)])
    assert rt.state.corrosion_depth_m == pytest.approx(0.002, rel=1e-3)
    assert rt.state.crack_length_m > 0
    eng.step(1.0, [ev(T.REPLACEMENT, a)])
    assert rt.state.vector() == (0.0,) * 5


def test_inspection_does_not_alter_truth():
    (a,) = new_ids(1)
    r1, r2 = runtime(a, corrosion=1e-3, crack=5e-3), runtime(a, corrosion=1e-3, crack=5e-3)
    engine([r1], cfg=DET).step(YEAR, [ev(T.INSPECTION, a)])
    engine([r2], cfg=DET).step(YEAR)
    assert r1.state == r2.state


def test_impact_and_load_spike():
    (a,) = new_ids(1)
    rt = runtime(a, coated=True, coating=0.05, stress=60e6)
    eng = engine([rt], cfg=DET)
    eng.step(1.0, [ev(T.IMPACT, a, crack_length_m=0.01, coating_damage_fraction=0.3)])
    assert rt.state.crack_length_m == pytest.approx(0.01)
    assert rt.state.coating_breakdown_fraction >= 0.35
    before = rt.state.crack_length_m
    eng.step(1.0, [ev(T.LOAD_SPIKE, a, stress_multiplier=3.0, n_cycles=2000)])
    assert rt.state.crack_length_m > before


def test_reinforcement_lowers_stress_concentration():
    (a,) = new_ids(1)
    rt = runtime(a, corrosion=0.006)
    eng = engine([rt], cfg=DET)
    eng.step(1.0)
    scf = rt.stress_concentration
    eng.step(1.0, [ev(T.REINFORCEMENT, a, added_thickness_m=0.006)])
    eng.step(1.0)
    assert rt.stress_concentration < scf


def test_ablation_switches_ignore_events():
    (a,) = new_ids(1)
    rt = runtime(a, corrosion=0.004)
    eng = engine([rt], cfg=replace(DET, interventions_enabled=False, events_enabled=False))
    res = eng.step(1.0, [ev(T.REPAIR, a), ev(T.IMPACT, a)])
    assert rt.state.corrosion_depth_m > 0.0039 and rt.state.crack_length_m == 0.0
    assert [i["reason"] for i in res.ignored_events] == ["disabled_by_ablation"] * 2


def test_event_errors():
    (a,) = new_ids(1)
    eng = engine([runtime(a)], cfg=DET)
    with pytest.raises(StructuralEventError):
        eng.step(1.0, [ev(T.REPAIR, None)])
    with pytest.raises(StructuralEventError):
        eng.step(1.0, [ev(T.REPAIR, new_ids(2, seed=99)[1])])
    with pytest.raises(StructuralEventError):
        eng.step(1.0, [ev(T.COATING_FAILURE, a, fraction=2.0)])
