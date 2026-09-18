"""TRUTH ACCESS. Evaluation / logging / sensor-rendering infrastructure only.

Importing this module from ``conrad.robotics`` or any inference package is a truth leak.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from conrad.schemas.frames import WORLD, Pose
from conrad.sim.kernel.dynamics import SimKernel
from conrad.sim.kernel.params import SimValidityLevel

TRUTH_MARKER = "__conrad_truth__"


@dataclass(frozen=True)
class TrueVehicleState:
    t_s: float
    position_world_m: np.ndarray
    orientation_wxyz: np.ndarray
    linear_velocity_body_mps: np.ndarray
    angular_velocity_body_rps: np.ndarray
    thrust_n: np.ndarray
    current_world_mps: np.ndarray

    def pose(self) -> Pose:
        p, q = self.position_world_m, self.orientation_wxyz
        return Pose(
            frame_id=WORLD,
            position_m=(float(p[0]), float(p[1]), float(p[2])),
            orientation_wxyz=(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
        )


class TruthAccess:
    """The only sanctioned door to X_true. Hand it to evaluation code, never to the robot stack."""

    __conrad_truth__ = True

    def __init__(self, kernel: SimKernel) -> None:
        self._kernel = kernel

    def true_state(self) -> TrueVehicleState:
        p, q, v, w, _ = self._kernel._snapshot()
        return TrueVehicleState(
            t_s=self._kernel.t_s,
            position_world_m=p,
            orientation_wxyz=q,
            linear_velocity_body_mps=v,
            angular_velocity_body_rps=w,
            thrust_n=self._kernel.thrusters.thrust.copy(),
            current_world_mps=self._kernel.current_at(p, self._kernel.t_s),
        )

    @property
    def collision_count(self) -> int:
        return self._kernel.collision_count

    @property
    def min_clearance_m(self) -> float:
        return self._kernel.min_clearance_m

    @property
    def energy_used_j(self) -> float:
        return self._kernel.energy_used_j

    @property
    def validity_level(self) -> SimValidityLevel:
        return self._kernel.validity_level
