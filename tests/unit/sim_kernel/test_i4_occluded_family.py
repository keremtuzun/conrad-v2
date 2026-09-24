"""World family ACTIVE_INSPECTION_OCCLUDED_V1 (gate I4): the truth-side construction it promises.

docs/audits/I4_WORLD_FAMILY.md. These are properties of the WORLD, checked without running any planner:
the critical surface is hidden from the nominal route, the occluding structure exists and is unregistered,
the window it leaves is reachable, and the family is deterministic in the seed.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from conrad.sim.mission.occlusion import patch_left, sample_defect, sample_occlusion, tilted
from conrad.sim.mission.options import world_options
from conrad.sim.mission.scenarios import resolve
from conrad.sim.mission.world import MissionWorld

DEV_SEEDS = tuple(range(8000000, 8000024))


def _axis(heading: float) -> list[np.ndarray]:
    d = np.array([math.cos(heading), math.sin(heading), 0.0])
    return [-2.0 * d, 2.0 * d]


def _lane(axis: list[np.ndarray], sign: float) -> list[np.ndarray]:
    d = axis[-1] - axis[0]
    right = np.array([d[1], -d[0], 0.0])
    right = right / np.linalg.norm(right)
    return [p + sign * 2.0 * right for p in axis]


@pytest.mark.parametrize("heading", [0.0, 0.2, 1.4, 2.9, -1.1, -2.5])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_off_lane_points_away_from_the_lane_for_every_heading(heading: float, sign: float) -> None:
    axis = _axis(heading)
    lane = _lane(axis, sign)
    opts = world_options({"defect": {"side": "off_lane"}})
    to_lane = np.mean(np.asarray(lane), axis=0) - 0.5 * (axis[0] + axis[-1])
    assert float(patch_left(axis, opts, lane) @ to_lane) < 0.0
    near = world_options({"defect": {"side": "lane"}})
    assert float(patch_left(axis, near, lane) @ to_lane) > 0.0


@pytest.mark.parametrize("heading", [0.0, 0.2, 1.4, -1.1])
def test_far_and_near_keep_their_historical_meaning(heading: float) -> None:
    axis = _axis(heading)
    lane = _lane(axis, 1.0)
    far = patch_left(axis, world_options({"defect": {"side": "far"}}), lane)
    near = patch_left(axis, world_options({"defect": {"side": "near"}}), lane)
    assert far[1] > 0.0  # the historical rule is "the WORLD +Y normal of the pipe heading"
    assert np.allclose(near, -far)


def test_defect_and_occluder_sampling_are_deterministic_in_the_seed() -> None:
    opts = world_options(resolve("I4-OCCLUDED", {})[0])
    assert sample_defect(opts, 8000000) == sample_defect(opts, 8000000)
    assert sample_defect(opts, 8000000) != sample_defect(opts, 8000001)
    assert sample_occlusion(opts.occlusion, 8000000) == sample_occlusion(opts.occlusion, 8000000)


def test_defect_variation_stays_inside_the_declared_ranges() -> None:
    opts = world_options(resolve("I4-OCCLUDED", {})[0])
    v = opts.defect_variation
    for seed in range(8000000, 8000024):
        d = sample_defect(opts, seed)
        assert v.corrosion_depth_m[0] <= d.corrosion_depth_m <= v.corrosion_depth_m[1]
        assert v.crack_length_m[0] <= d.crack_length_m <= v.crack_length_m[1]
        assert v.patch_tilt_deg[0] <= d.patch_tilt_deg <= v.patch_tilt_deg[1]


def _build(seed: int, scenario: str = "I4-OCCLUDED") -> MissionWorld:
    opts = world_options(resolve(scenario, {})[0])
    root = Path(tempfile.mkdtemp(prefix="i4occ-"))
    return MissionWorld.build(seed, scenario, opts, uuid4(), root / "objects")


@pytest.mark.parametrize("seed", DEV_SEEDS)
def test_the_hidden_patch_is_not_visible_from_the_transit_lane(seed: int) -> None:
    w = _build(seed)
    suite = w.hardware.suite
    assert suite is not None
    patch = suite.targets[0]
    spec = suite.sensors.structural
    mount_yaw = math.radians(w.opts.structural.mount_yaw_deg)
    centre = patch.points.mean(axis=0)
    lane = np.asarray(w.context.transit_lane, dtype=np.float64)
    dense = np.concatenate([np.linspace(lane[i], lane[i + 1], 12) for i in range(len(lane) - 1)])
    from conrad.schemas.frames import Pose, quat_from_euler

    best = 0.0
    for p in dense:
        dx, dy, dz = (float(centre[i] - p[i]) for i in range(3))
        pose = Pose(
            frame_id="WORLD",
            position_m=(float(p[0]), float(p[1]), float(p[2])),
            orientation_wxyz=quat_from_euler(
                0.0, -math.atan2(dz, math.hypot(dx, dy)), math.atan2(dy, dx) - mount_yaw
            ),
        )
        vis = w.t2s.visibility(spec, pose, patch.points, patch.normals)
        best = max(best, float(np.asarray(vis.visible, dtype=bool).mean()))
    assert best == 0.0


@pytest.mark.parametrize("seed", DEV_SEEDS)
def test_the_occluding_structure_is_present_and_unregistered(seed: int) -> None:
    w = _build(seed)
    rec = w.recorder.meta["view_occlusion"]
    assert rec is not None and len(rec["entity_ids"]) >= 2
    ids = {str(e) for e in rec["entity_ids"]}
    # Twin2S geometry only: no scenario world entity, no registry component, no mission-context design entry.
    assert ids.isdisjoint({str(e.id) for e in w.scenario.world_entities})
    assert ids.isdisjoint({str(r) for r in w.mapping.to_world})
    assert ids.isdisjoint({str(c.registry_id) for c in w.context.design})
    assert ids <= {str(e.entity_id) for e in w.t2s.world.entities}
    assert all(str(c.get("registry_id")) not in ids for c in w.context.asset_registry["components"])


@pytest.mark.parametrize("seed", DEV_SEEDS)
def test_the_window_is_reachable_and_the_panels_flank_it(seed: int) -> None:
    w = _build(seed)
    rec = w.recorder.meta["view_occlusion"]
    window = np.asarray(rec["window_direction"], dtype=np.float64)
    assert float(window[2]) >= w.opts.occlusion.min_window_z
    normals = [np.asarray(p["normal"], dtype=np.float64) for p in rec["panels"]]
    span = math.radians(rec["panel_span_deg"])
    for n in normals:  # both panels sit outside the window, by the declared span
        assert math.acos(float(np.clip(n @ window, -1.0, 1.0))) == pytest.approx(span, abs=1e-6)
    assert float(normals[0] @ normals[1]) < float(normals[0] @ window)


def test_existing_scenarios_are_untouched_by_the_new_options() -> None:
    for scenario in ("FLAGSHIP-I4", "I5-NOMINAL", "I7-BANDWIDTH", "INT-002"):
        opts = world_options(resolve(scenario, {})[0])
        assert not opts.occlusion.enabled
        assert not opts.defect_variation.enabled
        assert opts.defect.side in ("far", "near")


def test_the_frozen_family_definition_still_describes_the_scenarios() -> None:
    """configs/eval/i4_occluded_family.yaml pins the resolved world options of the family by digest."""
    import hashlib
    import json

    import yaml

    from conrad.settings import REPO_ROOT

    doc = yaml.safe_load((REPO_ROOT / "configs" / "eval" / "i4_occluded_family.yaml").read_text("utf-8"))
    # Spatial V1 is opt-in. Its new default fields do not change the frozen
    # legacy world declaration or the historical family digest.
    live = {}
    for scenario in doc["scenarios"]:
        options = world_options(resolve(scenario, {})[0])
        assert options.twin2t_truth_model == "legacy"
        assert options.spatial_truth is None and options.spatial_sensor_model is None
        assert options.survey_endpoint_bound_m is None
        live[scenario] = options.model_dump(
            mode="json",
            exclude={
                "twin2t_truth_model",
                "spatial_truth",
                "spatial_sensor_model",
                "survey_endpoint_bound_m",
            },
        )
    assert live == doc["scenarios"]
    digest = hashlib.sha256(
        json.dumps(doc["scenarios"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == doc["config_digest"]


def test_a_panel_is_a_thin_plate_along_the_pipe_axis() -> None:
    """The rotated Box really has half_extents (length, width, thickness) along (axis, tangent, radial)."""
    from conrad.sim.mission.occlusion import _plate

    ex, ey, ez = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    for rot in (0.0, 0.7, 2.1):
        c, s = math.cos(rot), math.sin(rot)
        a, b, n = ex, c * ey + s * ez, -s * ey + c * ez
        box = _plate(np.zeros(3), a, b, n, (1.0, 0.4, 0.05))
        probes = np.array([2.0 * a, 2.0 * b, 2.0 * n])
        d = box.sdf(probes)
        assert d[0] == pytest.approx(1.0, abs=1e-9)  # 2.0 - half length 1.0
        assert d[1] == pytest.approx(1.6, abs=1e-9)  # 2.0 - half width 0.4
        assert d[2] == pytest.approx(1.95, abs=1e-9)  # 2.0 - half thickness 0.05


@pytest.mark.parametrize("seed", DEV_SEEDS)
def test_the_panels_block_most_of_the_candidate_ring_but_never_all_of_it(seed: int) -> None:
    """The family promise: several candidate views exist, only a few resolve the defect, and at least one does."""
    from conrad.active.candidates import SensorOption, ViewpointGenerator
    from conrad.active.config import MCBRConfig
    from conrad.schemas.frames import Pose, quat_from_euler

    w = _build(seed)
    suite = w.hardware.suite
    assert suite is not None
    patch, spec = suite.targets[0], suite.sensors.structural
    mount_yaw = math.radians(w.opts.structural.mount_yaw_deg)
    centre = patch.points.mean(axis=0)
    comp = next(c for c in w.context.design if c.registry_id == w.mapping.to_registry[w.target])
    p = spec.parameters
    option = SensorOption(
        sensor_id=spec.sensor_id,
        modality=spec.modality,
        min_range_m=float(p.get("min_range_m", 0.3)) + 0.7,
        max_range_m=float(p["max_range_m"]),
    )
    cands = ViewpointGenerator(MCBRConfig()).generate(comp.inspection_station(1.0, 0.2), (option,))
    free = np.asarray(w.t2s.world.sdf(np.array([c.pose.position_m for c in cands])), float) > 0.5
    fracs = []
    for c, ok in zip(cands, free, strict=True):
        if not ok:
            fracs.append(0.0)
            continue
        pos = c.pose.position_m
        dx, dy, dz = (float(centre[i] - pos[i]) for i in range(3))
        pose = Pose(
            frame_id="WORLD",
            position_m=pos,
            orientation_wxyz=quat_from_euler(
                0.0, -math.atan2(dz, math.hypot(dx, dy)), math.atan2(dy, dx) - mount_yaw
            ),
        )
        vis = w.t2s.visibility(spec, pose, patch.points, patch.normals)
        fracs.append(float(np.asarray(vis.visible, dtype=bool).mean()))
    resolving = sum(1 for f in fracs if f > 0.30)
    assert max(fracs) > 0.30, "no reachable candidate resolves the defect: the world measures nothing"
    assert 0 < resolving <= 0.25 * len(fracs), f"{resolving}/{len(fracs)} candidates resolve it"


def test_tilted_rotates_towards_plus_z() -> None:
    left = np.array([0.0, 1.0, 0.0])
    up = tilted(left, 90.0)
    assert up[2] == pytest.approx(1.0)
    assert tilted(left, 0.0) == pytest.approx(left)
