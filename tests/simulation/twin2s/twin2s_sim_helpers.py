"""Helpers shared by the Twin2S simulation tests (not a test module)."""

from __future__ import annotations

from pathlib import Path

from conrad.persistence.object_store import ObjectStore
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import stamp
from conrad.twins.base import SensingContext
from conrad.twins.twin2s.twin import Twin2S


def make_twin(scenario, root: Path, id_seed: int = 70_011) -> Twin2S:
    # the twin must NOT share the scenario IdFactory seed, or observation IDs repeat entity IDs
    tw = Twin2S(IdFactory(id_seed), ObjectStore(root))
    tw.initialize(scenario)
    return tw


def make_ctx(scenario, sensor, t_s=0.5, true_pose=None, est_pose=None, degradation=None, seq=0):
    ids = IdFactory(99)
    robot = scenario.robots[0]
    return SensingContext(
        mission_id=scenario.mission.mission_id,
        run_id=ids.new(),
        trace_id=ids.new(),
        sensor=sensor,
        true_pose=true_pose or robot.initial_pose,
        estimated_pose=est_pose,
        timestamp=stamp(t_s, "SIM", sequence_index=seq),
        degradation=degradation or {},
    )
