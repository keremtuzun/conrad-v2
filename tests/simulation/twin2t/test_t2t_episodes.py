from __future__ import annotations

import json

import numpy as np
import pytest

from conrad.persistence.object_store import ObjectStore
from conrad.schemas.frames import Pose
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import SensorHealth
from conrad.schemas.timebase import stamp
from conrad.schemas.world import SensorSpec
from conrad.twins.base import SensingContext
from conrad.twins.twin2t import (
    COVERAGE_LEVELS,
    TIERS,
    SensorDegradationCurriculum,
    Twin2T,
    Twin2TConfig,
    coverage_visibility,
    generate_scenario,
    tier_mcde_config,
    validate_sequence,
)

DAY = 86400.0


def _ctx(deg, t):
    ids = IdFactory(seed=31337)
    pose = Pose(frame_id="WORLD", position_m=(0.0, 0.0, -5.0))
    sensor = SensorSpec(
        sensor_id=ids.new(), modality="STRUCTURED", frame_id="s", mount_pose=pose, rate_hz=1.0
    )
    return SensingContext(
        mission_id=ids.new(),
        run_id=ids.new(),
        trace_id=ids.new(),
        sensor=sensor,
        true_pose=pose,
        estimated_pose=None,
        timestamp=stamp(t, "sim"),
        degradation=deg,
    )


@pytest.mark.parametrize("tier", sorted(TIERS))
def test_procedural_tier_episode(tmp_path, tier):
    rng = np.random.default_rng(tier)
    scenario = generate_scenario(rng, tier, IdFactory(seed=tier), seed=tier)
    cfg = Twin2TConfig(mcde=tier_mcde_config(tier))
    twin = Twin2T(IdFactory(seed=1000 + tier), ObjectStore(tmp_path / "o"), cfg)
    twin.initialize(scenario)
    spec = TIERS[tier]
    results, n_obs = [], 0
    curriculum = SensorDegradationCurriculum()
    for _ in range(10):
        results.append(twin.advance(180 * DAY))
        deg = {
            **coverage_visibility(twin.component_ids, spec.coverage, rng),
            **curriculum.degradation(spec.sensor_degradation, rng),
        }
        n_obs += len(twin.generate_observation(_ctx(deg, twin.time_s)))
    assert twin._mcde is not None
    rep = validate_sequence(results, twin._mcde.runtimes)
    assert rep.passed, rep.violations[:3]
    export = twin.export_domain_state()
    json.dumps(export)
    assert len(export["truth_sequence"]) == 11 and len(export["observation_masks"]) == 10
    if tier == 1:
        assert len(twin.component_ids) == 1 and n_obs == 10
        assert all(r.records[twin.component_ids[0]].after[3] == 0.0 for r in results)  # corrosion only
    if tier == 5:
        assert export["mcde_config"]["mechanism_coupling"] is True


def test_coverage_levels_and_sensor_curriculum(tmp_path):
    rng = np.random.default_rng(0)
    scenario = generate_scenario(rng, 4, IdFactory(seed=40), seed=40)
    twin = Twin2T(IdFactory(seed=4040), ObjectStore(tmp_path / "o"))
    twin.initialize(scenario)
    twin.step(365 * DAY)
    n = len(twin.component_ids)
    for frac in COVERAGE_LEVELS:
        got = twin.generate_observation(_ctx(coverage_visibility(twin.component_ids, frac, rng), 1.0))
        assert len(got) == round(frac * n)
    cur = SensorDegradationCurriculum(missing_modality_prob=0.0)
    full = coverage_visibility(twin.component_ids, 1.0, rng)
    clean = twin.generate_observation(_ctx({**full, **cur.degradation(0.0, rng)}, 1.0))
    heavy = twin.generate_observation(_ctx({**full, **cur.degradation(1.0, rng)}, 1.0))
    assert all(s.observation.sensor_health is SensorHealth.OK for s in clean)
    assert heavy and all(s.observation.sensor_health is not SensorHealth.OK for s in heavy)
