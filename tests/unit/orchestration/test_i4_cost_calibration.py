"""The optional I4 cost correction uses only estimated path length and frozen constants."""

import math
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from pydantic import ValidationError

from conrad.orchestration.deliberation import Deliberation
from conrad.orchestration.mission_config import I4CostCalibration, MissionRuntimeConfig
from conrad.schemas.frames import WORLD, Pose


def test_cost_correction_is_finite_nonnegative_and_monotone():
    model = I4CostCalibration(
        enabled=True, time_offset_s=2.0, time_scale=1.25, energy_offset_j=400.0, energy_per_m_j=120.0
    )
    predicted = [model.estimate(distance, 0.5) for distance in (0.0, 1.0, 4.0)]
    assert all(math.isfinite(value) and value >= 0 for pair in predicted for value in pair)
    assert predicted == sorted(predicted)
    with pytest.raises(ValidationError):
        I4CostCalibration(energy_offset_j=float("inf"))
    with pytest.raises(ValueError):
        model.estimate(-1.0, 0.5)


def test_v4_runtime_keeps_new_switches_off_by_default():
    runtime = MissionRuntimeConfig(planner="V4")
    assert not runtime.trajectory_short_leg_fix
    assert not runtime.i4_cost_calibration.enabled
    corrected = MissionRuntimeConfig(
        planner="V4", i4_cost_calibration={"enabled": True, "energy_offset_j": 1600.0}
    )
    assert corrected.i4_cost_calibration.enabled
    assert corrected.i4_cost_calibration.estimate(2.0, 0.5)[1] == 1680.0


def test_belief_side_navigation_cost_feeds_calibrated_resource_cost():
    # This test supplies only the two collaborators used by navigation_cost.
    deliberation = cast(Any, object.__new__(Deliberation))
    deliberation.ctx = SimpleNamespace(design_distance=lambda points: np.full(len(points), 10.0))
    deliberation.is_free = lambda points: np.ones(len(points), dtype=bool)
    a = Pose(frame_id=WORLD, position_m=(0.0, 0.0, -5.0))
    b = Pose(frame_id=WORLD, position_m=(2.0, 0.0, -5.0))
    deliberation.cfg = MissionRuntimeConfig(planner="V4")
    old = deliberation.navigation_cost(a, b)
    assert old is not None
    deliberation.cfg = MissionRuntimeConfig(
        planner="V4",
        i4_cost_calibration={"enabled": True, "time_offset_s": 2.8, "energy_offset_j": 1600.0},
    )
    corrected = deliberation.navigation_cost(a, b)
    assert corrected is not None
    assert corrected.travel_m == old.travel_m
    assert corrected.time_s == pytest.approx(old.time_s + 2.8)
    assert corrected.energy_j == pytest.approx(old.energy_j + 1600.0)
