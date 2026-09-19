"""The target's non-defect surface is its own truth-side Twin2T component (docs/audits/MODEL2T_REPAIR.md)."""

from __future__ import annotations

from uuid import UUID

import numpy as np

from conrad.persistence.object_store import ObjectStore
from conrad.schemas.frames import Pose
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import stamp
from conrad.schemas.world import SensorSpec
from conrad.sim.mission.options import world_options
from conrad.sim.mission.structure import REGION_OF, rest_region_of, structural_scenario
from conrad.sim.scenarios.pipeline_inspection import build_pipeline_inspection_scenario
from conrad.twins.base import SensingContext
from conrad.twins.twin2t import Twin2T


def _setup(tmp_path, seed=7):
    opts = world_options({})
    scenario = build_pipeline_inspection_scenario(seed, IdFactory(seed).child("scenario"), family=opts.family)
    target = UUID(scenario.spatial_state["groups"]["segments"][0])
    t2t_scenario = structural_scenario(scenario, target, opts, seed)
    twin = Twin2T(IdFactory(seed).child("twin2t"), ObjectStore(tmp_path / "obj"))
    twin.initialize(t2t_scenario)
    return scenario, t2t_scenario, target, twin, opts


def _ctx(entity: UUID, seed=3):
    ids = IdFactory(seed)
    pose = Pose(frame_id="WORLD", position_m=(0.0, 0.0, -10.0))
    sensor = SensorSpec(
        sensor_id=ids.new(),
        modality="STRUCTURED",
        frame_id="sensor",
        mount_pose=pose,
        rate_hz=1.0,
        parameters={"t2t_fidelity": "T0"},
    )
    return SensingContext(
        mission_id=ids.new(),
        run_id=ids.new(),
        trace_id=ids.new(),
        sensor=sensor,
        true_pose=pose,
        estimated_pose=None,
        timestamp=stamp(1.0, "sim"),
        degradation={f"visibility:{entity}": 1.0},
    )


def test_region_is_truth_side_only_and_carries_no_defect(tmp_path):
    scenario, t2t_scenario, target, twin, opts = _setup(tmp_path)
    region = rest_region_of(t2t_scenario, target)
    assert region is not None and region not in {e.id for e in scenario.world_entities}
    ent = next(e for e in t2t_scenario.world_entities if e.id == region)
    assert ent.metadata[REGION_OF] == str(target) and not ent.domain_ownership.spatial
    arr, _ = twin.truth_arrays()
    idx = {eid: i for i, eid in enumerate(twin.component_ids)}
    assert arr[idx[target], 3] == opts.defect.crack_length_m
    assert arr[idx[region], 3] < 3.0e-3 and arr[idx[region], 0] < 1.0e-3  # ordinary prior ranges


def test_near_side_reading_reports_the_region_not_the_defect(tmp_path):
    _, t2t_scenario, target, twin, opts = _setup(tmp_path)
    region = rest_region_of(t2t_scenario, target)
    assert region is not None
    assert len(twin.generate_observation(_ctx(region))) == 1  # only the region is in view: one reading
    walls = []
    for k in range(40):
        (s,) = twin.generate_observation(_ctx(region, seed=100 + k))
        walls.append(s.observation.inline_values[0])
        assert s.supervision is not None and s.supervision.true_world_entity_id == region
    assert float(np.median(walls)) < 0.5 * opts.defect.corrosion_depth_m


def test_region_construction_is_deterministic(tmp_path):
    opts = world_options({})
    seed = 7
    scenario = build_pipeline_inspection_scenario(seed, IdFactory(seed).child("scenario"), family=opts.family)
    target = UUID(scenario.spatial_state["groups"]["segments"][0])
    with_region = structural_scenario(scenario, target, opts, seed).structural_state["entities"]
    region = next(k for k in with_region if k not in {str(e.id) for e in scenario.world_entities})
    others = {k: v for k, v in with_region.items() if k != region}
    again = structural_scenario(scenario, target, opts, seed).structural_state["entities"]
    assert all(again[k] == v for k, v in others.items())
