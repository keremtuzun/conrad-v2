from __future__ import annotations

from dataclasses import replace

import pytest
from t2t_unit_helpers import YEAR, engine, new_ids, runtime

from conrad.twins.twin2t.config import MCDEConfig
from conrad.twins.twin2t.coupling import COUPLING_GRAPH, export_coupling_graph
from conrad.twins.twin2t.events import StructuralEventType
from conrad.twins.twin2t.mcde import StructuralEvent
from conrad.twins.twin2t.registry import RelationshipType

DET = MCDEConfig(stochastic=False)


def _run(eng, steps=12, dt=YEAR / 12):
    for _ in range(steps):
        eng.step(dt)
    return eng


def test_wall_loss_accelerates_crack_growth_only_with_coupling():
    (a,) = new_ids(1)
    res = {}
    for coupled in (True, False):
        rt = runtime(a, corrosion=0.006, crack=4e-3, stress=60e6, cycles=0.01)
        _run(engine([rt], cfg=replace(DET, mechanism_coupling=coupled)))
        res[coupled] = rt.state.crack_length_m
        if coupled:
            assert rt.stress_concentration > 1.9
    assert res[True] > res[False] > 4e-3


def test_coating_failure_drives_corrosion_only_with_coupling():
    (a,) = new_ids(1)
    out = {}
    for coupled in (True, False):
        rt = runtime(a, coated=True, coating=0.05)
        eng = engine([rt], cfg=replace(DET, mechanism_coupling=coupled))
        eng.step(1.0, [StructuralEvent(StructuralEventType.COATING_FAILURE, a, {"fraction": 1.0})])
        _run(eng)
        out[coupled] = rt.state.corrosion_area_fraction
    # depth is the local loss on exposed metal; coating breakdown governs how much area corrodes
    assert out[True] > 0.5 and out[False] < 0.06


def test_crack_breaches_coating():
    (a,) = new_ids(1)
    rt = runtime(a, coated=True, coating=0.05, crack=0.02, stress=1e6)
    engine([rt], cfg=DET).step(1.0)
    assert rt.state.coating_breakdown_fraction == pytest.approx(0.4)


def test_no_contagion_to_adjacent_healthy_component():
    a, b = new_ids(2)
    traj = []
    for a_damaged in (True, False):
        ra = runtime(a, corrosion=0.009 if a_damaged else 0.0, crack=0.01 if a_damaged else 0.0)
        rb = runtime(b, coated=True, coating=0.05)
        eng = engine([ra, rb], [(a, b, RelationshipType.ADJACENT_TO)], MCDEConfig(), seed=3)
        _run(eng)
        traj.append(rb.state.vector())
    assert traj[0] == traj[1]


def test_load_path_coupling_only_via_supported_by():
    seg, sup = new_ids(2)
    out = {}
    for rel in (RelationshipType.SUPPORTED_BY, RelationshipType.ADJACENT_TO):
        rs = runtime(seg, crack=4e-3, stress=60e6, cycles=0.01)
        rp = runtime(sup, "SUPPORT", corrosion=0.006)
        eng = engine([rs, rp], [(seg, sup, rel)], DET)
        _run(eng)
        out[rel] = (rs.state.crack_length_m, rs.load_multiplier)
    assert out[RelationshipType.SUPPORTED_BY][1] > 1.2
    assert out[RelationshipType.ADJACENT_TO][1] == 1.0
    assert out[RelationshipType.SUPPORTED_BY][0] > out[RelationshipType.ADJACENT_TO][0]


def test_concrete_support_is_masked_and_carries_no_coupling():
    seg, sup = new_ids(2)
    rs = runtime(seg, crack=4e-3)
    rp = runtime(sup, "SUPPORT", "concrete", corrosion=0.006, crack=0.01)
    assert not any(rp.state.mask()) and rp.state.vector() == (0.0,) * 5
    eng = engine([rs, rp], [(seg, sup, RelationshipType.SUPPORTED_BY)], DET)
    _run(eng)
    assert rs.load_multiplier == 1.0 and rp.state.vector() == (0.0,) * 5


def test_shared_environment_change_propagates_only_with_topology():
    a, b, c = new_ids(3)
    for topo in (True, False):
        rts = [runtime(a), runtime(b), runtime(c)]
        eng = engine(
            rts,
            [(a, b, RelationshipType.CONNECTED_TO), (a, c, RelationshipType.ADJACENT_TO)],
            replace(DET, topology_coupling=topo),
        )
        eng.step(1.0, [StructuralEvent(StructuralEventType.ENVIRONMENT_CHANGE, a, {"temperature_c": 25.0})])
        assert rts[0].environment.temperature_c == 25.0
        assert (rts[1].environment.temperature_c == 25.0) is topo
        assert rts[2].environment.temperature_c == 10.0


def test_coupling_graph_is_justified_and_restricted():
    for e in COUPLING_GRAPH:
        assert e.justification
        if e.scope == "CROSS":
            assert e.relationship_types and RelationshipType.ADJACENT_TO not in e.relationship_types
    assert len(export_coupling_graph()) == len(COUPLING_GRAPH)


def test_stochasticity_switch():
    (a,) = new_ids(1)
    outs = []
    for seed in (1, 2):
        rt = runtime(a, crack=5e-3)
        _run(engine([rt], cfg=DET, seed=seed))
        outs.append(rt.state.vector())
    assert outs[0] == outs[1]
    outs = []
    for seed in (1, 2):
        rt = runtime(a, crack=5e-3)
        _run(engine([rt], cfg=MCDEConfig(), seed=seed))
        outs.append(rt.state.vector())
    assert outs[0] != outs[1]
