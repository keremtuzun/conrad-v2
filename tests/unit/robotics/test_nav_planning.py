import itertools
import math

import numpy as np
import pytest

from conrad.robotics.navigation import (
    AStarPlanner,
    GoalManager,
    GoalRejectedError,
    ObjectiveKind,
    PlannerConfig,
    PlanningError,
    UnknownSpacePolicy,
    incidence_angle,
)
from conrad.schemas.decision import NavigationGoal
from conrad.schemas.frames import FrameError, Pose, SpatialSupport
from conrad.schemas.ids import IdFactory

IDS = IdFactory(seed=9)


def wall_free(p):  # wall x in [2,3], |y| < 2
    return ~((p[:, 0] > 2) & (p[:, 0] < 3) & (np.abs(p[:, 1]) < 2))


def test_astar_routes_around_obstacle():
    planner = AStarPlanner(wall_free, PlannerConfig(resolution_m=0.25))
    path = planner.plan(np.array([0.0, 0, 0]), np.array([5.0, 0, 0]))
    assert np.allclose(path[0], [0, 0, 0]) and np.allclose(path[-1], [5, 0, 0])
    for a, b in itertools.pairwise(path):
        assert planner.segment_clear(a, b)
    assert len(path) >= 3


def test_straight_line_when_clear():
    planner = AStarPlanner(lambda p: np.ones(len(p), bool))
    assert len(planner.plan(np.zeros(3), np.array([3.0, 1, -1]))) == 2


def test_unknown_policy_is_separate_from_the_map():
    free = lambda p: np.ones(len(p), bool)  # noqa: E731
    unknown_band = lambda p: np.where(np.abs(p[:, 0] - 2.5) < 0.5, "UNKNOWN", "OBSERVED")  # noqa: E731
    forbid = AStarPlanner(free, PlannerConfig(bounds_margin_m=1.0), knowledge=unknown_band)
    with pytest.raises(PlanningError):
        forbid.plan(np.zeros(3), np.array([5.0, 0, 0]))  # band spans the whole search box
    allow = AStarPlanner(
        free,
        PlannerConfig(unknown_policy=UnknownSpacePolicy.TRAVERSE_WITH_PENALTY, bounds_margin_m=1.0),
        knowledge=unknown_band,
    )
    assert len(allow.plan(np.zeros(3), np.array([5.0, 0, 0]))) == 2


def test_cost_hook_changes_route():
    free = lambda p: ~((np.abs(p[:, 0] - 2.5) < 0.3) & (np.abs(p[:, 1]) < 1.0))  # noqa: E731
    plain = AStarPlanner(free, PlannerConfig(resolution_m=0.25))
    hook = AStarPlanner(free, PlannerConfig(resolution_m=0.25), cost_hook=lambda a, b: 5.0 * max(0.0, b[1]))
    assert np.min(hook.plan(np.zeros(3), np.array([5.0, 0, 0]))[:, 1]) < 0  # pushed to the cheap side
    assert plain.plan(np.zeros(3), np.array([5.0, 0, 0])) is not None


def goal(**kw):
    base = {
        "goal_id": IDS.new(),
        "trace_id": IDS.new(),
        "position_tolerance_m": 0.3,
        "orientation_tolerance_rad": 0.2,
        "risk_limit": 0.1,
    }
    return NavigationGoal(**{**base, **kw})


def test_goal_manager_primitives():
    gm = GoalManager()
    tp = Pose(frame_id="WORLD", position_m=(5, 0, -5))
    o = gm.to_objective(goal(target_pose=tp, observation_constraints={"via": [[2, 2, -5]]}), np.zeros(3))
    assert o.kind is ObjectiveKind.GO_TO and o.waypoints.shape == (2, 3)
    o = gm.to_objective(
        goal(target_pose=tp, observation_constraints={"primitive": "STATION_KEEP", "duration_s": 9}),
        np.zeros(3),
    )
    assert o.kind is ObjectiveKind.STATION_KEEP and o.hold_duration_s == 9
    poly = [[0, 0, -10], [5, 0, -10]]
    o = gm.to_objective(
        goal(
            target_region=SpatialSupport(frame_id="WORLD", center_m=(2.5, 0, -10)),
            observation_constraints={"primitive": "PIPELINE_FOLLOW", "polyline": poly, "standoff_m": 1.5},
        ),
        np.zeros(3),
    )
    assert o.route_is_prescribed and np.allclose(o.waypoints[:, 2], -8.5)


def test_inspection_respects_incidence_range_and_standoff():
    gm = GoalManager()
    region = SpatialSupport(frame_id="WORLD", center_m=(0, 0, -5))
    oc = {
        "primitive": "INSPECT",
        "standoff_m": 2.0,
        "incidence_range_rad": [0.0, 0.5],
        "surface_normal": [1, 0, 0],
    }
    o = gm.to_objective(
        goal(target_region=region, observation_constraints=oc), np.array([0.0, 6.0, -5.0])
    )  # robot at 90 deg
    vp = o.final_position
    assert np.linalg.norm(vp - [0, 0, -5]) == pytest.approx(2.0)
    assert incidence_angle(vp, np.array([0, 0, -5.0]), np.array([1.0, 0, 0])) == pytest.approx(0.5, abs=1e-6)
    yaw = o.params["final_yaw"]
    assert math.cos(yaw) * (0 - vp[0]) + math.sin(yaw) * (0 - vp[1]) > 0  # camera faces the region


def test_goal_rejections():
    gm = GoalManager()
    region = SpatialSupport(frame_id="WORLD", center_m=(0, 0, -5))
    with pytest.raises(GoalRejectedError) as e:
        gm.to_objective(
            goal(target_region=region, observation_constraints={"primitive": "INSPECT"}), np.zeros(3)
        )
    assert e.value.reason_code == "MALFORMED_CONSTRAINT"
    with pytest.raises(GoalRejectedError):
        gm.to_objective(
            goal(target_region=region, observation_constraints={"primitive": "DANCE"}), np.zeros(3)
        )
    with pytest.raises(FrameError):
        gm.to_objective(goal(target_pose=Pose(frame_id="ASSET", position_m=(0, 0, 0))), np.zeros(3))
