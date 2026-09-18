import numpy as np
import pytest

from conrad.robotics.allocation import PRODUCER, ThrusterAllocator, ThrusterLayoutUnknownError
from conrad.robotics.hardware.config import load_robot_config
from conrad.schemas.ids import IdFactory
from conrad.schemas.robot import HealthLevel, SourceKind, ThrusterState, WrenchCommand
from conrad.schemas.timebase import stamp

CFG = load_robot_config("configs/robot/sim_reference.yaml")


def alloc():
    return ThrusterAllocator(CFG, IdFactory(seed=1))


def test_matrix_columns_from_geometry():
    a = alloc()
    assert a.matrix.shape == (6, 8)
    assert np.linalg.matrix_rank(a.matrix) == 6
    h1 = a.matrix[:, 0]
    assert h1[:3] == pytest.approx([0.7071068, -0.7071068, 0.0])
    assert h1[5] == pytest.approx(0.2 * -0.7071068 - 0.15 * 0.7071068)


def test_reproduces_wrench_within_limits():
    a = alloc()
    tau = np.array([10.0, -5.0, 8.0, 0.3, -0.2, 0.5])
    r = a.solve(tau)
    assert not r.saturated
    assert a.matrix @ r.thrust_n == pytest.approx(tau, abs=1e-9)
    assert all(abs(u) <= 1.0 for u in r.commands.values())


def test_saturation_is_bounded_and_keeps_direction():
    a = alloc()
    r = a.solve(np.array([500.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert r.saturated
    assert np.all(r.thrust_n <= 40.0 + 1e-9) and np.all(r.thrust_n >= -30.0 - 1e-9)
    assert r.achieved_wrench[0] == pytest.approx(4 * 40 * 0.7071068, rel=1e-6)
    assert abs(r.achieved_wrench[5]) < 1e-6


def test_failed_thruster_reallocates():
    a = alloc()
    states = tuple(
        ThrusterState(
            thruster_id=t,
            command=0.0,
            estimated_thrust_n=0.0,
            health=HealthLevel.FAULT if t == "H1" else HealthLevel.OK,
        )
        for t in a.thruster_ids
    )
    assert a.update_from_thruster_states(states)
    tau = np.array([8.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    r = a.solve(tau)
    assert r.commands["H1"] == 0.0 and r.thrust_n[0] == 0.0
    assert r.achieved_wrench == pytest.approx(tau, abs=1e-6)  # still feasible with 3 horizontal thrusters
    assert a.capability()["fx"] < alloc().capability()["fx"]


def test_command_inversion_asymmetry_and_deadzone():
    a = alloc()
    r = a.solve(np.array([0.0, 0.0, 40.0, 0.0, 0.0, 0.0]))
    assert r.commands["V1"] == pytest.approx(0.5)  # 10 N = 40 * 0.5^2
    r = a.solve(np.array([0.0, 0.0, -40.0, 0.0, 0.0, 0.0]))
    assert r.commands["V1"] == pytest.approx(-np.sqrt(10.0 / 30.0))  # reverse coefficient 30
    r = a.solve(np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0]))
    assert all(u == 0.0 for u in r.commands.values())  # below half the deadzone thrust -> off
    r = a.solve(np.array([0.0, 0.0, 0.32, 0.0, 0.0, 0.0]))
    assert r.commands["V1"] > 0.05  # lifted just above the deadzone


def test_allocated_command_contract():
    ids = IdFactory(seed=2)
    a = ThrusterAllocator(CFG, ids)
    w = WrenchCommand(
        command_id=ids.new(),
        trace_id=ids.new(),
        timestamp=stamp(1.0, "SIM"),
        frame_id="ROBOT",
        force_n=(1, 0, 0),
        torque_nm=(0, 0, 0),
    )
    cmd, _ = a.allocate(w, mission_id=ids.new(), run_id=ids.new(), now_ns=1_000_000_000, clock_domain="SIM")
    assert cmd.producer == PRODUCER == "conrad.robotics.allocation"
    assert cmd.robot_config_digest == CFG.content_digest()
    assert cmd.deadline_ns - cmd.issued_time_ns == 250_000_000
    assert cmd.source_wrench_id == w.command_id and cmd.trace_id == w.trace_id
    assert cmd.safety_authorization is None


def test_unknown_layout_raises():
    with pytest.raises(ThrusterLayoutUnknownError):
        ThrusterAllocator(CFG.model_copy(update={"thrusters": ()}), IdFactory(seed=1))
    t0 = CFG.thrusters[0]
    open_pos = t0.position_body_m.model_copy(update={"value": None, "source": SourceKind.OPEN})
    broken = CFG.model_copy(
        update={"thrusters": (t0.model_copy(update={"position_body_m": open_pos}), *CFG.thrusters[1:])}
    )
    with pytest.raises(ThrusterLayoutUnknownError):
        ThrusterAllocator(broken, IdFactory(seed=1))
