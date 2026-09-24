from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from conrad.twins.twin2t import GeneratorKind, build_small_scenario, make_engine, validate_sequence
from conrad.twins.twin2t.assembly import assemble
from conrad.twins.twin2t.config import MCDEConfig
from conrad.twins.twin2t.events import StructuralEventType
from conrad.twins.twin2t.mcde import StructuralEvent

DAY = 86400.0
MECHANISTIC = [
    k for k in GeneratorKind if k not in (GeneratorKind.RANDOM_WALK, GeneratorKind.RANDOM_STATIC_DEFECTS)
]
EVENTS = [
    StructuralEventType.IMPACT,
    StructuralEventType.LOAD_SPIKE,
    StructuralEventType.COATING_FAILURE,
    StructuralEventType.REPAIR,
    StructuralEventType.REPLACEMENT,
    StructuralEventType.REINFORCEMENT,
    StructuralEventType.COATING_RENEWAL,
]


def _engine(seed, kind, **cfg):
    asm = assemble(build_small_scenario(seed, n_segments=2), np.random.default_rng(seed))
    return make_engine(kind, asm.graph, asm.runtimes, MCDEConfig(**cfg), seed)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(0, 10_000),
    kind=st.sampled_from(MECHANISTIC),
    dts=st.lists(st.floats(0.0, 400 * DAY), min_size=1, max_size=8),
    ev=st.lists(st.tuples(st.integers(0, 7), st.sampled_from(EVENTS), st.integers(0, 50)), max_size=4),
)
def test_mechanistic_invariants_hold(seed, kind, dts, ev):
    eng = _engine(seed, kind)
    order = eng.order
    results = []
    for i, dt in enumerate(dts):
        batch = [StructuralEvent(t, order[j % len(order)]) for step, t, j in ev if step == i]
        results.append(eng.step(dt, batch))
    rep = validate_sequence(results, eng.runtimes)
    assert rep.passed, rep.violations[:3]


def test_reinforcement_then_impact_seed_401_preserves_crack_depth():
    """The next fatigue tick must use the reinforced, effective wall bound."""
    eng = _engine(401, GeneratorKind.CONFIGURED)
    target = eng.order[0]
    results = [
        eng.step(
            4919074.0,
            [
                StructuralEvent(StructuralEventType.REINFORCEMENT, target),
                StructuralEvent(StructuralEventType.IMPACT, target),
            ],
        ),
        eng.step(1.0),
    ]
    rec = results[1].records[target]
    assert rec.after[4] >= rec.before[4]
    assert validate_sequence(results, eng.runtimes).passed


@settings(max_examples=15, deadline=None)
@given(seed=st.integers(0, 10_000), kind=st.sampled_from(list(GeneratorKind)), n=st.integers(1, 6))
def test_bounds_and_masks_for_every_generator(seed, kind, n):
    eng = _engine(seed, kind)
    for _ in range(n):
        eng.step(180 * DAY)
    arr, mask = eng.truth_arrays()
    assert (arr >= 0).all() and (arr[~mask] == 0).all()
    assert (arr[:, 1:3] <= 1.0).all()
    walls = np.array([eng.runtimes[e].effective_wall_m for e in eng.order])
    assert (arr[:, 0] <= walls + 1e-12).all() and (arr[:, 4] <= walls + 1e-12).all()


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(0, 10_000))
def test_reproducible_sequences(seed):
    a, b = _engine(seed, GeneratorKind.FULL_MCDE), _engine(seed, GeneratorKind.FULL_MCDE)
    for _ in range(4):
        a.step(90 * DAY)
        b.step(90 * DAY)
    assert np.array_equal(a.truth_arrays()[0], b.truth_arrays()[0])
