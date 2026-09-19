"""Test-only harness for gate U0 against the real Unity player.

Everything physical here is SYNTHETIC_ONLY: the RobotConfig variants are labelled simulator inputs, the
water density is a scenario value chosen to make ``configs/robot/sim_reference.yaml`` exactly neutral, and
the analytic references use exactly the parameters the player loaded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from conrad_unity_testkit import make_command

from conrad.adapters.unity import StatePacket, UnityRobotHardware
from conrad.persistence.object_store import ObjectStore
from conrad.robotics.hardware.config import load_robot_config
from conrad.runtime.command_gateway import CommandGateway
from conrad.schemas.ids import IdFactory
from conrad.schemas.robot import RobotConfig
from conrad.settings import CommandMode, ExecutionLane, RuntimeSettings
from conrad.sim.unity.player import UnityPlayerSession, scenario_document
from conrad.sim.unity.truth import TruthVehicleState, UnityTruthClient

G = 9.80665  # HydrodynamicsModel.StandardGravity
PHYSICS_DT_NS = 5_000_000
STEP_NS = 100_000_000  # one STEP request advances 20 physics steps
INITIAL_POSITION = (0.0, 0.0, -10.0)
BASE = load_robot_config("configs/robot/sim_reference.yaml")
# Scenario water density that makes the reference config exactly neutral (SYNTHETIC_ONLY scenario value).
RHO_NEUTRAL = float(BASE.mass_kg.value) / float(BASE.displaced_volume_m3.value)  # type: ignore[arg-type]


def synthetic(value: Any, units: str) -> dict[str, Any]:
    return {"value": value, "units": units, "source": "SYNTHETIC_ONLY"}


def variant(**fields: dict[str, Any]) -> RobotConfig:
    """``sim_reference`` with some Sourced fields replaced (each still labelled SYNTHETIC_ONLY or OPEN)."""
    doc = BASE.model_dump(mode="json")
    doc.update(fields)
    doc["config_name"] = "sim_reference_u0_variant"
    return RobotConfig.model_validate(doc)


VARIANTS: dict[str, RobotConfig] = {
    "neutral": BASE,
    "positive": variant(
        displaced_volume_m3=synthetic(float(BASE.displaced_volume_m3.value) * 1.02, "m^3")  # type: ignore[arg-type]
    ),
    "negative": variant(mass_kg=synthetic(float(BASE.mass_kg.value) * 1.02, "kg")),  # type: ignore[arg-type]
    "inertia_x2": variant(
        inertia_diag_kgm2=synthetic([2 * float(v) for v in BASE.inertia_diag_kgm2.value], "kg*m^2")  # type: ignore[union-attr]
    ),
    "cob_forward": variant(center_of_buoyancy_body_m=synthetic([0.005, 0.0, 0.02], "m")),
    "com_aft": variant(center_of_mass_body_m=synthetic([-0.005, 0.0, 0.0], "m")),
}


def environment() -> dict[str, Any]:
    return {"water_density_kgm3": RHO_NEUTRAL, "surface_z_m": 0.0, "turbidity": 0.0}


def scenario_for(robot: RobotConfig, seed: int) -> dict[str, Any]:
    return scenario_document(
        IdFactory(seed=seed), robot, seed=seed, initial_position_m=INITIAL_POSITION, environment=environment()
    )


def session(robot: RobotConfig, run_dir: Path, seed: int = 2026, **kw: Any) -> UnityPlayerSession:
    return UnityPlayerSession(robot, run_dir, scenario_for(robot, seed), **kw)


# ------------------------------------------------------------------------------------------------ allocation
def thrust_matrix(robot: RobotConfig) -> np.ndarray:
    """6 x N: column = [direction; position x direction] in the Conrad body frame."""
    cols = []
    for t in robot.thrusters:
        d = np.asarray(t.direction_body.value, dtype=float)
        p = np.asarray(t.position_body_m.value, dtype=float)
        cols.append(np.concatenate([d, np.cross(p, d)]))
    return np.stack(cols, axis=1)


def allocate(robot: RobotConfig, wrench: tuple[float, ...]) -> dict[str, float]:
    """Exact thruster commands for a body wrench (min-norm thrusts, inverted T = k u|u| thrust curve)."""
    b = thrust_matrix(robot)
    thrust = np.linalg.pinv(b) @ np.asarray(wrench, dtype=float)
    assert np.allclose(b @ thrust, wrench, atol=1e-9), "wrench not reachable"
    out: dict[str, float] = {}
    for t, f in zip(robot.thrusters, thrust, strict=True):
        k = float(t.thrust_coefficient.value)  # type: ignore[arg-type]
        fmax, rmax = float(t.max_forward_thrust_n.value), float(t.max_reverse_thrust_n.value)  # type: ignore[arg-type]
        dead = float(t.deadzone_command.value)  # type: ignore[arg-type]
        if abs(f) < 1e-9:
            out[t.thruster_id] = 0.0
            continue
        u = math.sqrt(f / k) if f > 0 else -math.sqrt(-f / (k * rmax / fmax))
        assert dead <= abs(u) <= 1.0 and (f <= fmax if f > 0 else -f <= rmax), (t.thruster_id, f, u)
        out[t.thruster_id] = u
    return out


# ------------------------------------------------------------------------------------------------ analytics
def terminal_velocity(force: float, d1: float, d2: float) -> float:
    """Positive root of d1 v + d2 v^2 = |F| with the sign of F."""
    f = abs(force)
    v = (-d1 + math.sqrt(d1 * d1 + 4 * d2 * f)) / (2 * d2)
    return math.copysign(v, force)


def reference_1dof(
    force: float,
    m_eff: float,
    d1: float,
    d2: float,
    seconds: float,
    dt: float = PHYSICS_DT_NS / 1e9,
    lag: tuple[float, float] | None = None,
) -> np.ndarray:
    """1-DOF m_eff v' = F(t) - d1 v - d2 v|v| with the player's step order (semi-implicit Euler).

    ``lag = (latency_s, time_constant_s)`` reproduces the thruster's command latency and discrete
    first-order lag; ``None`` applies ``force`` from t = 0 (restoring/buoyancy forces). Returns
    positions after each physics step (displacement from 0), and velocities as a second row.
    """
    n = round(seconds / dt)
    x = v = f_applied = 0.0
    alpha = 0.0 if lag is None else 1.0 - math.exp(-dt / lag[1])
    delay_steps = 0 if lag is None else round(lag[0] / dt)
    out = np.zeros((2, n))
    for k in range(n):
        if lag is None:
            f_applied = force
        else:
            target = force if k >= delay_steps else 0.0
            f_applied += alpha * (target - f_applied)
        a = (f_applied - d1 * v - d2 * v * abs(v)) / m_eff
        v += a * dt
        x += v * dt
        out[0, k], out[1, k] = x, v
    return out


def nose_up_deg(q_wxyz: tuple[float, float, float, float]) -> float:
    w, x, y, z = q_wxyz
    # body +X expressed in WORLD (first column of R)
    fx = 1 - 2 * (y * y + z * z)
    fy = 2 * (x * y + w * z)
    fz = 2 * (x * z - w * y)
    return math.degrees(math.atan2(fz, math.hypot(fx, fy)))


def body_rates(t: TruthVehicleState) -> tuple[np.ndarray, np.ndarray]:
    """World-frame truth velocities rotated into the body frame."""
    w, x, y, z = t.pose.orientation_wxyz
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return r.T @ np.asarray(t.linear_velocity_world_mps), r.T @ np.asarray(t.angular_velocity_world_rps)


# ------------------------------------------------------------------------------------------------ driver
@dataclass
class Driver:
    """Control plane (UnityRobotHardware behind the real CommandGateway) + truth plane for evaluation."""

    session: UnityPlayerSession
    seed: int = 1
    store: ObjectStore | None = None
    ids: IdFactory = field(init=False)
    hw: UnityRobotHardware = field(init=False)
    truth: UnityTruthClient = field(init=False)
    gateway: CommandGateway = field(init=False)
    states: list[StatePacket] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.ids = IdFactory(seed=self.seed)
        self.mission, self.run = self.ids.new(), self.ids.new()
        self.robot = self.session.robot
        self.hw = self.session.hardware(self.ids, self.mission, self.run, payload_store=self.store)
        self.ack = self.hw.connect()
        self.truth = self.session.truth()
        self.truth.connect(self.ids.new().hex)
        self.gateway = CommandGateway(
            self.hw,
            self.robot,
            RuntimeSettings(command_mode=CommandMode.SIMULATED),
            ExecutionLane.SIMULATION,
            self.mission,
            self.run,
        )

    def reset(self, seed: int) -> None:
        self.hw.reset(seed=seed)
        self.states.clear()

    def command(self, values: dict[str, float]) -> None:
        cmd = make_command(self.ids, self.robot, self.mission, self.run, self.hw.now_ns(), values)
        ack = self.gateway.submit(cmd)
        assert ack.accepted, ack.reason_codes

    def wrench(self, wrench: tuple[float, ...]) -> dict[str, float]:
        values = allocate(self.robot, wrench)
        self.command(values)
        return values

    def advance(
        self, seconds: float, step_ns: int = STEP_NS, keep_states: bool = False
    ) -> list[TruthVehicleState]:
        out = []
        for _ in range(round(seconds * 1e9 / step_ns)):
            state = self.hw.step(step_ns)
            if keep_states:
                self.states.append(state)
            out.append(self.truth.get_ground_truth())
        return out

    def close(self) -> None:
        self.hw.close()
