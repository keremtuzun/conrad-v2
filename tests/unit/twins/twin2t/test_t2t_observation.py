from __future__ import annotations

import numpy as np
import pytest

from conrad.persistence.object_store import ObjectStore
from conrad.schemas.frames import Pose
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Modality, SensorHealth
from conrad.schemas.timebase import stamp
from conrad.schemas.world import SensorSpec
from conrad.twins.base import SensingContext
from conrad.twins.twin2t import (
    SensorLevelHookMissing,
    Twin2T,
    build_small_scenario,
    contradiction_pair,
    coverage_visibility,
)

DAY = 86400.0


@pytest.fixture
def twin(tmp_path):
    t = Twin2T(IdFactory(seed=2021), ObjectStore(tmp_path / "obj"))
    t.initialize(build_small_scenario(21))
    for _ in range(12):
        t.step(60 * DAY)
    return t


def ctx(deg, fidelity="T0", modality="STRUCTURED"):
    ids = IdFactory(seed=99)
    pose = Pose(frame_id="WORLD", position_m=(0.0, 0.0, -10.0))
    sensor = SensorSpec(
        sensor_id=ids.new(),
        modality=modality,
        frame_id="sensor",
        mount_pose=pose,
        rate_hz=1.0,
        parameters={"t2t_fidelity": fidelity},
    )
    return SensingContext(
        mission_id=ids.new(),
        run_id=ids.new(),
        trace_id=ids.new(),
        sensor=sensor,
        true_pose=pose,
        estimated_pose=None,
        timestamp=stamp(100.0, "sim"),
        degradation=deg,
    )


def vis_all(t, level=1.0):
    return {f"visibility:{e}": level for e in t.component_ids}


def test_masking_unobservable_yields_nothing(twin):
    assert twin.generate_observation(ctx({})) == []
    assert twin.generate_observation(ctx(vis_all(twin, 0.2))) == []  # below threshold 0.3
    assert twin.generate_observation(ctx({**vis_all(twin), "missing_modality": 1.0})) == []
    assert twin.generate_observation(ctx({**vis_all(twin, 0.5), "occlusion": 0.9})) == []
    one = twin.component_ids[0]
    out = twin.generate_observation(ctx({f"visibility:{one}": 1.0}))
    assert len(out) == 1 and out[0].supervision.true_world_entity_id == one
    mask = twin.export_domain_state()["observation_masks"][-1]["observed"]
    assert sum(mask.values()) == 1 and mask[str(one)]


def test_structured_observation_contains_no_truth(twin):
    samples = twin.generate_observation(ctx(vis_all(twin)))
    assert len(samples) == len(twin.component_ids)
    truth_ids = {str(e) for e in twin.component_ids} | {str(twin._asm.scenario.scenario_id)}
    truth_vals = set()
    for tr in twin.get_truth(stamp(0, "sim")):
        truth_vals |= {repr(v) for k, v in tr.state.items() if isinstance(v, float) and v != 0.0}
    for s in samples:
        o = s.observation
        assert o.modality is Modality.STRUCTURED and o.inline_units == "m,1,m" and len(o.inline_values) == 3
        blob = o.model_dump_json()
        assert not any(i in blob for i in truth_ids)
        assert not any(repr(v) in truth_vals for v in o.inline_values)
        assert set(o.sensor_context) == {"twin2t_fidelity", "measurements", "units"}
        assert s.supervision is not None and "corrosion_depth_m" in s.supervision.targets


def test_noise_grows_with_sensing_degradation(twin):
    one = twin.component_ids[0]
    true_d = twin.get_truth(stamp(0, "sim"))[0].state["corrosion_depth_m"]

    def err(extra):
        vals = [
            twin.generate_observation(ctx({f"visibility:{one}": 1.0, **extra}))[0].observation.inline_values[
                0
            ]
            for _ in range(200)
        ]
        return float(np.std(np.asarray(vals) - true_d))

    clean, dirty = err({}), err({"turbidity": 1.0, "biofouling_cover": 1.0, "corruption": 0.6})
    assert dirty > 3 * clean
    s = twin.generate_observation(ctx({f"visibility:{one}": 1.0, "corruption": 0.95}))[0]
    assert s.observation.sensor_health is SensorHealth.FAULT


def test_contradiction_is_controlled_and_labelled(twin):
    one = twin.component_ids[0]
    severe, plain = contradiction_pair({f"visibility:{one}": 1.0}, 1.0)
    a = [twin.generate_observation(ctx(severe))[0] for _ in range(30)]
    b = [twin.generate_observation(ctx(plain))[0] for _ in range(30)]
    assert (
        np.mean([s.observation.inline_values[1] for s in a])
        > np.mean([s.observation.inline_values[1] for s in b]) + 0.3
    )
    assert (
        abs(
            np.mean([s.observation.inline_values[0] for s in a])
            - np.mean([s.observation.inline_values[0] for s in b])
        )
        < 5e-4
    )
    assert (
        a[0].supervision.targets["contradiction_injected"]
        and not b[0].supervision.targets["contradiction_injected"]
    )


def test_feature_and_sensor_level(twin):
    one = twin.component_ids[0]
    deg = {f"visibility:{one}": 1.0}
    f = twin.generate_observation(ctx(deg, "T1"))[0].observation
    assert len(f.inline_values) == twin.config.observation.feature_dim
    with pytest.raises(SensorLevelHookMissing):
        twin.generate_observation(ctx(deg, "T2", "RGB"))

    def paint(app, rng):
        return np.full((4, 4, 3), app["rust_coverage_fraction"], dtype=np.float32)

    twin.register_sensor_renderer(Modality.RGB, paint)
    o = twin.generate_observation(ctx(deg, "T2", "RGB"))[0].observation
    assert o.modality is Modality.RGB and o.payload_ref is not None and o.payload_ref.shape == (4, 4, 3)
    assert twin.store.get_array(o.payload_ref).shape == (4, 4, 3)


def test_surface_appearance_api(twin):
    app = twin.surface_appearance(twin.component_ids[0])
    assert {"rust_coverage_fraction", "pitting_texture_scale_m", "crack_visible_length_m"} <= set(app)
    assert 0.0 <= app["rust_coverage_fraction"] <= 1.0
    with pytest.raises(KeyError):
        twin.surface_appearance(IdFactory(seed=1234).new())


def test_coverage_levels(twin):
    rng = np.random.default_rng(0)
    ids = twin.component_ids
    for frac in (1.0, 0.5, 0.2, 0.05):
        deg = coverage_visibility(ids, frac, rng)
        assert len(deg) == round(frac * len(ids))
        assert len(twin.generate_observation(ctx(deg))) == len(deg)
