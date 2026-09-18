from __future__ import annotations

import numpy as np
import pytest
from t2t_unit_helpers import YEAR, engine, new_ids, runtime

from conrad.twins.twin2t import NOT_EVALUABLE, config_from_dict, load_config, reality_gap, validate_sequence
from conrad.twins.twin2t.config import MCDEConfig
from conrad.twins.twin2t.events import StructuralEventType as T
from conrad.twins.twin2t.mcde import StructuralEvent
from conrad.twins.twin2t.priors import PriorSpec
from conrad.twins.twin2t.registry import RelationshipType


def _results():
    a, b, c = new_ids(3)
    rts = [
        runtime(a, corrosion=1e-3, crack=5e-3, cycles=0.01),
        runtime(b, "SUPPORT"),
        runtime(c, "SUPPORT", "concrete"),
    ]
    eng = engine(rts, [(a, b, RelationshipType.SUPPORTED_BY)], MCDEConfig())
    out = [eng.step(YEAR / 4) for _ in range(4)]
    out.append(eng.step(1.0, [StructuralEvent(T.REPAIR, a), StructuralEvent(T.IMPACT, b)]))
    out += [eng.step(YEAR / 4) for _ in range(2)]
    return out, eng


def test_validators_pass_on_generated_sequence():
    res, eng = _results()
    rep = validate_sequence(res, eng.runtimes)
    assert rep.passed, rep.violations
    assert {"bounds", "masks", "monotone", "paris_threshold", "repair_reset"} <= set(rep.checks)


def test_validators_detect_violations():
    res, eng = _results()
    rec = next(iter(res[1].records.values()))
    rec.after = (rec.before[0] - 1e-4, *rec.before[1:])  # material loss reversed without intervention
    rec.delta_k, rec.delta_k_threshold = 1.0, 3.0
    rec.after = (rec.after[0], rec.after[1], rec.after[2], rec.before[3] + 1e-3, rec.after[4])
    rep = validate_sequence(res, eng.runtimes)
    assert any("decreased without intervention" in v for v in rep.violations)
    assert any("below threshold" in v for v in rep.violations)


def test_reality_gap_hooks():
    sim = np.random.default_rng(0).normal(size=(200, 2))
    assert reality_gap(sim, None)["status"] == NOT_EVALUABLE
    assert reality_gap(sim, np.empty((0, 2)))["status"] == NOT_EVALUABLE
    same = reality_gap(sim, np.random.default_rng(1).normal(size=(200, 2)), ["a", "b"])
    shifted = reality_gap(sim, np.random.default_rng(1).normal(3.0, 1.0, size=(200, 2)), ["a", "b"])
    assert same["status"] == "EVALUATED"
    assert shifted["metrics"]["a"]["wasserstein_1"] > same["metrics"]["a"]["wasserstein_1"] + 2
    with pytest.raises(ValueError):
        reality_gap(sim, np.zeros((5, 3)))


def test_config_loading_and_strictness(tmp_path):
    cfg = load_config("configs/sim/twin2t_default.yaml")
    assert cfg.mcde.mechanism_coupling and cfg.observation.default_fidelity == "T0"
    with pytest.raises(ValueError):
        config_from_dict({"mcde": {"magic_contagion": True}})
    with pytest.raises(ValueError):
        config_from_dict({"nonsense": 1})
    bad = tmp_path / "x.yaml"
    bad.write_text("other: {}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad)


def test_priors_never_measured():
    with pytest.raises(ValueError):
        PriorSpec("x", "m", 0.0, 1.0, "uniform", "MEASURED", "cite")
    with pytest.raises(ValueError):
        PriorSpec("x", "m", 0.0, 1.0, "uniform", "ENGINEERING_ESTIMATE", "")
