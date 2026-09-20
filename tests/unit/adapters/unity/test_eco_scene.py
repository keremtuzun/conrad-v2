"""Twin2E -> Unity scene conversion (gate I6): no player needed. Development world only (unity_gate development)."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

from conrad.sim.mission.options import MissionWorldOptions
from conrad.sim.mission.scenarios import I6_SPIKE
from conrad.sim.mission.world import MissionWorld, _eco_event
from conrad.sim.unity.eco_scene import (
    EcoSceneOptions,
    attenuation_truth,
    build_optics,
    eco_scene,
    optics_update,
    trilinear,
)
from conrad.sim.unity.twin_scene import twin_scene

DEV_SEED = 7810000  # split("unity_gate", "development", "design")


@pytest.fixture(scope="module")
def world():
    opts = MissionWorldOptions.model_validate({"ecological_enabled": True})
    w = MissionWorld.build(
        DEV_SEED, "I6-ECO-SCENE", opts, uuid.UUID(int=6), Path(tempfile.mkdtemp(prefix="eco-scene-"))
    )
    assert w.t2e is not None
    return w


def test_truth_sampler_matches_twin2e(world):
    rng = np.random.default_rng(0)
    lo, hi = np.asarray(world.t2s.world.bounds_min), np.asarray(world.t2s.world.bounds_max)
    pts = lo + (hi - lo) * rng.random((40, 3))
    o = world.t2e.cfg.observation
    ref = o.beam_attenuation_clear_per_m + o.beam_attenuation_per_m_per_ntu * world.t2e.fields.sample(
        "turbidity", pts
    )
    assert np.allclose(attenuation_truth(world.t2e, pts), ref, atol=1e-12)


def test_grid_interpolation_is_exact_at_cell_centres(world):
    grid = build_optics(world.t2e, world.t2s.world, EcoSceneOptions())
    v = grid.values()
    idx = np.array([[0, 0, 0], [1, 2, 1], [v.shape[0] - 1, v.shape[1] - 1, v.shape[2] - 1]])
    pts = np.asarray(grid.origin_m) + idx * np.asarray(grid.spacing_m)
    assert np.allclose(trilinear(grid, pts), v[idx[:, 0], idx[:, 1], idx[:, 2]])
    far = np.asarray(grid.origin_m) - 10.0  # clamped at the faces, like the C# OpticsField
    assert np.isclose(trilinear(grid, far[None, :])[0], v[0, 0, 0])


def test_scene_extension_and_measured_error(world):
    scene, _ = twin_scene(world.t2s.world)
    base = scene.to_json()
    out, grid, report = eco_scene(world.t2e, world.t2s.world, base, EcoSceneOptions())
    optics = [p for p in out["primitives"] if p["kind"] == "optics_grid"]
    assert len(out["primitives"]) == len(base["primitives"]) + 1 and len(optics) == 1
    assert len(optics[0]["beam_attenuation_per_m"]) == int(np.prod(grid.shape))
    for p in out["primitives"]:
        if "fouling_cover" in p:
            assert p["kind"] in ("box", "capsule") and 0.0 < p["fouling_cover"] <= 1.0
    for old, new in zip(base["primitives"], out["primitives"][:-1], strict=True):
        assert set(new) - set(old) <= {"fouling_cover", "fouling_rgb"}
        assert {k: new[k] for k in old} == old  # geometry untouched
    # measured, not assumed: finite errors, and the grid reproduces Twin2E closely at 1 m spacing
    assert report.optics.points > 0 and report.optics.rays > 0
    assert np.isfinite(report.optics.attenuation_max_abs_per_m) and report.optics.transmission_max_abs < 0.05
    assert report.fouling.entities > 0 and 0.0 <= report.fouling.cover_max_abs <= 1.0


def test_optics_update_after_turbidity_spike(world):
    before = build_optics(world.t2e, world.t2s.world, EcoSceneOptions())
    ev = I6_SPIKE
    world.t2e.step(1e-3, [_eco_event(ev["event_type"], ev["parameters"], world.t2e.t_s)])
    body, after = optics_update(world.t2e, world.t2s.world, EcoSceneOptions())
    assert body["replace"] is False and [p["kind"] for p in body["primitives"]] == ["optics_grid"]
    o = world.t2e.cfg.observation
    rise = after.values().max() - before.values().max()
    assert rise > 0.5 * o.beam_attenuation_per_m_per_ntu * ev["parameters"]["delta_ntu"]
