"""Property tests for the analytic Model2T operators (hypothesis-free, seeded random sweeps)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "unit" / "domains" / "technical"))
from m2t_helpers import DAY, evidence, model

from conrad.domains.technical import CORROSION_DEPTH, CRACK_LENGTH, PropagationMode
from conrad.schemas.belief import KnowledgeStatus
from conrad.schemas.timebase import stamp


@pytest.mark.parametrize("seed", range(6))
def test_status_invariants_under_random_sequences(seed):
    rng = np.random.default_rng(seed)
    m, r, ev = model(seed=seed)
    names = ["seg_a", "seg_b", "weld_a", "support_steel", "support_concrete", "seg_adj"]
    t = 0.0
    observed: set[str] = set()
    for _ in range(8):
        dt = float(rng.uniform(1, 60)) * DAY
        t += dt
        m.predict(dt, stamp(t, "sim"))
        batch = []
        for n in rng.choice(names, size=int(rng.integers(0, 3)), replace=False):
            observed.add(str(n))
            batch.append(
                evidence(ev, r[n], t, wall=float(rng.uniform(0, 4e-3)), crack=float(rng.uniform(0, 1e-2)))
            )
        m.ingest(batch)
        m.update_beliefs(stamp(t, "sim"))
        for n in names:
            b = m.beliefs[r[n]]
            u = b.uncertainty()
            assert all(0.0 <= x <= 1.0 for x in u.as_tuple())
            for q in (CORROSION_DEPTH, CRACK_LENGTH):
                e = b.estimates[q]
                assert e.level_var > 0
                if e.status is KnowledgeStatus.OBSERVED:
                    assert e.direct_lineage
                if e.status is KnowledgeStatus.INFERRED:
                    assert not e.direct_lineage
                if n not in observed:
                    assert e.status in (KnowledgeStatus.UNKNOWN, KnowledgeStatus.INFERRED)
                    assert b.direct_support == 0.0 and u.observational >= 0.999


@pytest.mark.parametrize("seed", range(4))
def test_propagation_never_touches_direct_lineage(seed):
    rng = np.random.default_rng(100 + seed)
    walls = rng.uniform(0, 5e-3, size=2)
    results = {}
    for mode in PropagationMode:
        m, r, ev = model(seed=seed, mode=mode)
        m.ingest([evidence(ev, r["seg_a"], 5.0, wall=walls[0]), evidence(ev, r["seg_b"], 5.0, wall=walls[1])])
        m.update_beliefs(stamp(5.0, "sim"))
        results[mode] = [
            (
                m.beliefs[r[n]].estimates[CORROSION_DEPTH].level,
                m.beliefs[r[n]].estimates[CORROSION_DEPTH].level_var,
            )
            for n in ("seg_a", "seg_b")
        ]
    assert results[PropagationMode.TCDP] == results[PropagationMode.NONE] == results[PropagationMode.GENERIC]
