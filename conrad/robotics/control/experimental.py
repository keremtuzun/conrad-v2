"""EXPERIMENTAL controller interfaces (CTRL-B3 MPC, CTRL-B5 RL, CTRL-B6 residual). Not canonical.

These are abstract contracts only. Nothing here is implemented, registered as a default, or allowed
to bypass the allocator / safety supervisor / command gateway. The canonical baseline is
:class:`conrad.robotics.control.pid.CascadedPidController`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from conrad.robotics.control.pid import ControlReference
from conrad.schemas.robot import RobotState, WrenchCommand
from conrad.schemas.timebase import TimeStamp

EXPERIMENTAL_STATUS = "OPEN_BLOCKED"


class ExperimentalWrenchController(ABC):
    """Same signature as the PID baseline so benchmarks can swap controllers; output is a wrench only."""

    name: str = "experimental"
    implementation_status: str = EXPERIMENTAL_STATUS

    @abstractmethod
    def compute(
        self, state: RobotState, reference: ControlReference, dt: float, trace_id: UUID, timestamp: TimeStamp
    ) -> WrenchCommand: ...

    @abstractmethod
    def reset(self) -> None: ...


class MpcControllerInterface(ExperimentalWrenchController):
    """CTRL-B3: to be benchmarked on performance / latency / energy before any adoption."""

    name = "CTRL-B3-mpc"


class LearnedResidualControllerInterface(ExperimentalWrenchController):
    """CTRL-B6: u = u_classical + delta_u_theta; admissible only if it demonstrably beats CTRL-B1."""

    name = "CTRL-B6-residual"
