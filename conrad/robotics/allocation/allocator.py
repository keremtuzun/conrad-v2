"""Thruster allocation: tau = B f from RobotConfig geometry, bounded least squares, command inversion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from uuid import UUID

import numpy as np
from pydantic import Field
from scipy.optimize import lsq_linear

from conrad.schemas.base import ConradModel
from conrad.schemas.ids import IdFactory
from conrad.schemas.robot import (
    AllocatedCommand,
    HealthLevel,
    OpenParameterError,
    RobotConfig,
    ThrusterState,
    WrenchCommand,
)
from conrad.schemas.timebase import NS_PER_S

PRODUCER = "conrad.robotics.allocation"


class ThrusterLayoutUnknownError(RuntimeError):
    """Allocation is impossible: no thrusters, or their geometry/limits are OPEN."""


class AllocationConfig(ConradModel):
    wrench_weights: tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 3.0, 3.0, 3.0)
    degraded_effectiveness: float = Field(default=0.5, gt=0, le=1)
    deadzone_compensation: bool = True
    fault_aware: bool = True


@dataclass(frozen=True)
class AllocationResult:
    thrust_n: np.ndarray
    commands: dict[str, float]
    achieved_wrench: np.ndarray
    residual_norm: float
    saturated: bool


@dataclass(frozen=True)
class _Thruster:
    thruster_id: str
    max_fwd: float
    max_rev: float
    deadzone: float
    k_fwd: float
    k_rev: float


class ThrusterAllocator:
    def __init__(
        self, robot_config: RobotConfig, ids: IdFactory, config: AllocationConfig | None = None
    ) -> None:
        if not robot_config.thrusters:
            raise ThrusterLayoutUnknownError("RobotConfig has no thrusters; thruster layout unknown")
        self._ids = ids
        self.config = config or AllocationConfig()
        self._digest = robot_config.content_digest()
        self._timeout_s = float(robot_config.safety.command_timeout_s.require("safety.command_timeout_s"))
        cols, thrusters = [], []
        try:
            for t in robot_config.thrusters:
                p = f"thrusters[{t.thruster_id}]"
                r = np.asarray(t.position_body_m.require(f"{p}.position_body_m"), dtype=np.float64)
                d = np.asarray(t.direction_body.require(f"{p}.direction_body"), dtype=np.float64)
                d = d / np.linalg.norm(d)
                cols.append(np.concatenate([d, np.cross(r, d)]))
                fwd = float(t.max_forward_thrust_n.require(f"{p}.max_forward_thrust_n"))
                rev = float(t.max_reverse_thrust_n.require(f"{p}.max_reverse_thrust_n"))
                k = float(t.thrust_coefficient.require(f"{p}.thrust_coefficient"))
                dz = float(t.deadzone_command.require(f"{p}.deadzone_command"))
                thrusters.append(_Thruster(t.thruster_id, fwd, rev, dz, k, k * rev / fwd if fwd > 0 else k))
        except OpenParameterError as exc:
            raise ThrusterLayoutUnknownError(str(exc)) from exc
        self.matrix = np.stack(cols, axis=1)
        self._thrusters = tuple(thrusters)
        self.thruster_ids = tuple(t.thruster_id for t in thrusters)
        self._eff = np.ones(len(thrusters))
        self._w = np.asarray(self.config.wrench_weights, dtype=np.float64)

    # -- degraded / failed actuators -------------------------------------------------------------
    def set_effectiveness(self, effectiveness: dict[str, float]) -> None:
        for tid, e in effectiveness.items():
            self._eff[self.thruster_ids.index(tid)] = float(np.clip(e, 0.0, 1.0))

    def update_from_thruster_states(self, states: tuple[ThrusterState, ...]) -> bool:
        """Fault-aware reallocation from RHI-reported health. Returns True when capability is reduced."""
        if not self.config.fault_aware:
            return False
        levels = {
            HealthLevel.OK: 1.0,
            HealthLevel.UNKNOWN: 1.0,
            HealthLevel.DEGRADED: self.config.degraded_effectiveness,
            HealthLevel.FAULT: 0.0,
        }
        self.set_effectiveness(
            {s.thruster_id: levels[s.health] for s in states if s.thruster_id in self.thruster_ids}
        )
        return bool(np.any(self._eff < 1.0))

    @property
    def effectiveness(self) -> dict[str, float]:
        return {tid: float(e) for tid, e in zip(self.thruster_ids, self._eff, strict=True)}

    def capability(self) -> dict[str, float]:
        """Max achievable |wrench| per axis (single-axis, other axes unconstrained) with current health."""
        hi = np.array([t.max_fwd for t in self._thrusters]) * self._eff
        lo = np.array([t.max_rev for t in self._thrusters]) * self._eff
        names = ("fx", "fy", "fz", "mx", "my", "mz")
        out = {}
        for i, n in enumerate(names):
            row = self.matrix[i]
            out[n] = float(
                min(
                    np.sum(np.where(row > 0, row * hi, -row * lo)),
                    np.sum(np.where(row > 0, row * lo, -row * hi)),
                )
            )
        return out

    # -- solve ------------------------------------------------------------------------------------
    def _to_command(self, t: _Thruster, thrust: float) -> float:
        mag = abs(thrust)
        k = t.k_fwd if thrust >= 0 else t.k_rev
        if self.config.deadzone_compensation and k > 0:
            minimum = k * t.deadzone * t.deadzone
            if mag < 0.5 * minimum:
                return 0.0
            mag = max(mag, minimum * 1.0001)
        u = math.sqrt(mag / k) if k > 0 else 0.0
        return float(math.copysign(min(u, 1.0), thrust))

    def solve(self, wrench: np.ndarray) -> AllocationResult:
        tau = np.asarray(wrench, dtype=np.float64)
        if tau.shape != (6,) or not np.all(np.isfinite(tau)):
            raise ValueError("wrench must be six finite numbers")
        b_eff = self.matrix * self._eff[None, :]
        a = self._w[:, None] * b_eff
        y = self._w * tau
        hi = np.array([t.max_fwd for t in self._thrusters])
        lo = -np.array([t.max_rev for t in self._thrusters])
        active = self._eff > 0
        f = np.zeros(len(self._thrusters))
        f[active] = np.linalg.pinv(a[:, active]) @ y
        saturated = bool(np.any(f > hi + 1e-9) or np.any(f < lo - 1e-9))
        if saturated:
            sol = lsq_linear(a[:, active], y, bounds=(lo[active], hi[active]), method="bvls")
            f[active] = np.clip(sol.x, lo[active], hi[active])
        commands = {
            t.thruster_id: self._to_command(t, float(fi)) for t, fi in zip(self._thrusters, f, strict=True)
        }
        achieved = b_eff @ f
        return AllocationResult(f, commands, achieved, float(np.linalg.norm(achieved - tau)), saturated)

    def allocate(
        self,
        wrench: WrenchCommand,
        *,
        mission_id: UUID,
        run_id: UUID,
        now_ns: int,
        clock_domain: str,
        belief_snapshot_id: UUID | None = None,
        provenance_root: UUID | None = None,
    ) -> tuple[AllocatedCommand, AllocationResult]:
        result = self.solve(np.array([*wrench.force_n, *wrench.torque_nm]))
        command = AllocatedCommand(
            command_id=self._ids.new(),
            mission_id=mission_id,
            run_id=run_id,
            trace_id=wrench.trace_id,
            belief_snapshot_id=belief_snapshot_id,
            robot_config_digest=self._digest,
            clock_domain=clock_domain,
            issued_time_ns=now_ns,
            deadline_ns=now_ns + max(1, round(self._timeout_s * NS_PER_S)),
            thruster_commands=result.commands,
            source_wrench_id=wrench.command_id,
            provenance_root=provenance_root or wrench.command_id,
            producer=PRODUCER,
        )
        return command, result

    def zero_command(
        self,
        *,
        trace_id: UUID,
        mission_id: UUID,
        run_id: UUID,
        now_ns: int,
        clock_domain: str,
        source_wrench_id: UUID,
    ) -> AllocatedCommand:
        return AllocatedCommand(
            command_id=self._ids.new(),
            mission_id=mission_id,
            run_id=run_id,
            trace_id=trace_id,
            belief_snapshot_id=None,
            robot_config_digest=self._digest,
            clock_domain=clock_domain,
            issued_time_ns=now_ns,
            deadline_ns=now_ns + max(1, round(self._timeout_s * NS_PER_S)),
            thruster_commands=dict.fromkeys(self.thruster_ids, 0.0),
            source_wrench_id=source_wrench_id,
            provenance_root=source_wrench_id,
            producer=PRODUCER,
        )
