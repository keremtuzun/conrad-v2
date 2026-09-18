import numpy as np
import pytest

from conrad.robotics.hardware.config import load_robot_config
from conrad.schemas.ids import IdFactory
from conrad.schemas.robot import AllocatedCommand, HealthLevel
from conrad.sim.kernel import FaultType, SimKernelConfig, build_sim_hardware

CFG = load_robot_config("configs/robot/sim_reference.yaml")
IDS = IdFactory(seed=3)


def make(seed=1, **kw):
    return build_sim_hardware(CFG, seed, SimKernelConfig(water_density_kgm3=11.5 / 0.01125, **kw))


def command(hw, values, timeout_s=0.25):
    now = hw.now_ns()
    return AllocatedCommand(
        command_id=IDS.new(),
        mission_id=IDS.new(),
        run_id=IDS.new(),
        trace_id=IDS.new(),
        belief_snapshot_id=None,
        robot_config_digest=hw.robot_config_digest(),
        clock_domain=hw.clock_domain(),
        issued_time_ns=now,
        deadline_ns=now + round(timeout_s * 1e9),
        thruster_commands={t.thruster_id: values.get(t.thruster_id, 0.0) for t in CFG.thrusters},
        source_wrench_id=IDS.new(),
        provenance_root=IDS.new(),
        producer="conrad.robotics.allocation",
    )


def test_identity_and_capabilities():
    hw = make()
    assert hw.clock_domain() == "SIM" and hw.now_ns() == 0
    assert hw.robot_config_digest() == CFG.content_digest()
    caps = hw.capabilities()
    assert caps.imu and caps.depth and caps.thruster_count == 8
    assert set(caps.controllable_dof) == {"surge", "sway", "heave", "roll", "pitch", "yaw"}
    assert hw.get_camera() is None  # no renderer wired


def test_imu_and_depth_are_noisy_measurements():
    hw = make()
    hw.kernel.reset(np.array([0.0, 0.0, -7.0]))
    hw.advance(1.0)
    imu, depth = hw.get_imu(), hw.get_depth()
    assert imu.linear_acceleration_mps2[2] == pytest.approx(9.80665, abs=0.1)
    assert imu.timestamp.time_ns < hw.now_ns()  # measurement time precedes delivery (latency)
    assert depth.depth_m == pytest.approx(7.0, abs=0.1) and depth.depth_m != 7.0


def test_send_applies_and_watchdog_zeroes_after_deadline():
    hw = make()
    ack = hw.send(command(hw, {"H1": 0.5}))  # direct call allowed in this adapter unit test only
    assert ack.accepted
    hw.advance(0.2)
    assert hw.kernel.thrusters.applied_command[0] == 0.5
    hw.advance(0.3)
    assert hw.kernel.thrusters.applied_command[0] == 0.0 and hw.watchdog_trips == 1


def test_thruster_faults_reported():
    hw = make()
    hw.inject_fault(FaultType.THRUSTER_FAILURE, "V1")
    hw.inject_fault(FaultType.THRUSTER_DEGRADATION, "V2", magnitude=0.5)
    hw.kernel.thrusters.command(0.0, {"V1": 1.0, "V2": 1.0})
    hw.advance(2.0)
    states = {s.thruster_id: s for s in hw.get_thruster_state()}
    assert states["V1"].health is HealthLevel.FAULT and states["V1"].estimated_thrust_n == 0.0
    assert states["V2"].estimated_thrust_n == pytest.approx(20.0, rel=1e-2)
    assert hw.get_health().overall is HealthLevel.DEGRADED
    assert len(hw.fault_log) == 2


def test_sensor_faults_power_leak_gust():
    hw = make()
    hw.advance(0.5)
    hw.inject_fault(FaultType.SENSOR_BIAS, "depth", magnitude=1.0)
    hw.inject_fault(FaultType.SENSOR_DROPOUT, "imu", duration_s=1.0)
    stamp = hw.get_imu().timestamp.time_ns
    hw.advance(0.5)
    assert hw.get_imu().timestamp.time_ns <= stamp + 10_000_000  # only the in-flight sample arrives
    assert hw.get_depth().depth_m == pytest.approx(1.0, abs=0.1)
    assert hw.get_health().devices["imu"] is HealthLevel.FAULT
    hw.inject_fault(FaultType.LOW_POWER, magnitude=0.1)
    assert hw.get_power_state().remaining_fraction <= 0.1
    hw.inject_fault(FaultType.CURRENT_GUST, vector=(0.5, 0.0, 0.0), duration_s=5.0)
    hw.advance(5.0)
    assert hw.truth_access().true_state().position_world_m[0] > 0.5
    hw.inject_fault(FaultType.LEAK_SIGNAL)
    health = hw.get_health()
    assert health.leak_detected and health.overall is HealthLevel.FAULT


def test_renderer_receives_true_and_estimated_pose():
    seen = []

    def renderer(name, true_pose, est_pose, stamp):
        seen.append((name, true_pose.position_m, est_pose, stamp.time_ns))
        return None

    hw = make()
    hw.set_renderer(renderer)
    hw.advance(0.1)
    hw.get_sonar()
    hw.get_sonar()  # rate-gated: second call in the same instant renders nothing
    assert len(seen) == 1 and seen[0][0] == "sonar" and seen[0][2] is None
