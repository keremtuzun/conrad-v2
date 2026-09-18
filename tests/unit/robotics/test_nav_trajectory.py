import numpy as np
import pytest

from conrad.robotics.hardware.config import load_robot_config
from conrad.robotics.trajectory import TrajectoryConfig, TrajectoryGenerator, TrajectorySampler, YawMode
from conrad.schemas.ids import IdFactory

CFG = load_robot_config("configs/robot/sim_reference.yaml")


def gen(**kw):
    ids = IdFactory(seed=1)
    return TrajectoryGenerator(CFG, ids, TrajectoryConfig(**kw)), ids


def test_speed_and_accel_limits_respected():
    g, ids = gen(sample_dt_s=0.05)
    traj = g.generate(np.array([[0, 0, -5], [10, 0, -5], [10, 5, -8.0]]), ids.new(), ids.new(), start_yaw=0.0)
    t = np.array([p.t_s for p in traj.points])
    v = np.array([p.linear_velocity_mps for p in traj.points])
    speed = np.linalg.norm(v, axis=1)
    assert traj.frame_id == "WORLD" and traj.planner.startswith("NAV-TRAJ-01")
    assert speed.max() <= traj.max_speed_mps + 1e-9 <= 1.0 + 1e-9  # safety.max_speed_mps
    assert speed[0] == 0.0 and speed[-1] == pytest.approx(0.0, abs=1e-9)
    accel = np.abs(np.diff(speed)) / np.diff(t)
    assert accel.max() <= g.accel_limit * 1.1
    assert traj.points[-1].pose.position_m == pytest.approx((10, 5, -8))


def test_corner_slows_down():
    g, ids = gen(sample_dt_s=0.05)
    traj = g.generate(np.array([[0, 0, 0], [6, 0, 0], [6, 6, 0.0]]), ids.new(), ids.new(), start_yaw=0.0)
    p = np.array([q.pose.position_m for q in traj.points])
    v = np.linalg.norm([q.linear_velocity_mps for q in traj.points], axis=1)
    corner = np.argmin(np.linalg.norm(p - [6, 0, 0], axis=1))
    assert v[corner] < 0.5 * v.max()


def test_accel_limit_derived_from_config_and_sampler():
    g, ids = gen()
    assert 0.05 < g.accel_limit < 2.0
    traj = g.generate(
        np.array([[0, 0, 0], [0, 4, 0.0]]), ids.new(), ids.new(), start_yaw=0.0, yaw_mode=YawMode.FACE_TRAVEL
    )
    s = TrajectorySampler(traj)
    ref_mid = s.sample(s.duration_s / 2)
    assert ref_mid.position_world_m[1] == pytest.approx(2.0, abs=0.2)
    end = s.sample(s.duration_s + 10)
    assert np.allclose(end.velocity_world_mps, 0) and end.position_world_m[1] == pytest.approx(4.0)
    yaw_end = 2 * np.arctan2(end.orientation_wxyz[3], end.orientation_wxyz[0])
    assert yaw_end == pytest.approx(np.pi / 2, abs=0.05)  # yaw rate-limited toward travel direction


def test_degenerate_route_is_a_hold():
    g, ids = gen()
    traj = g.generate(np.array([[1, 2, -3.0]]), ids.new(), ids.new(), start_yaw=0.3)
    assert len(traj.points) == 1 and traj.planner.endswith(":hold")
