"""Configuration, result and read-only hardware facade types of the NavigationStack."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pydantic import Field

from conrad.robotics.allocation.allocator import AllocationConfig
from conrad.robotics.control.pid import ControlConfig, ControlReference
from conrad.robotics.estimation.ekf import EkfConfig
from conrad.robotics.hardware.interface import RobotHardwareInterface
from conrad.robotics.navigation.global_planner import PlannerConfig
from conrad.robotics.navigation.local_planner import LocalPlannerConfig
from conrad.robotics.safety.monitors import SafetyAssessment, SafetyConfig
from conrad.robotics.safety.supervisor import SafetyDecision
from conrad.robotics.trajectory.generator import TrajectoryConfig
from conrad.schemas.base import ConradModel
from conrad.schemas.robot import AllocatedCommand, RobotState


class NavigationStackConfig(ConradModel):
    control_period_s: float = Field(default=0.02, gt=0)
    estimator: EkfConfig = EkfConfig()
    control: ControlConfig = ControlConfig()
    allocation: AllocationConfig = AllocationConfig()
    trajectory: TrajectoryConfig = TrajectoryConfig()
    planner: PlannerConfig = PlannerConfig()
    local: LocalPlannerConfig = LocalPlannerConfig()
    safety: SafetyConfig = SafetyConfig()


class GoalStatus(str, Enum):
    IDLE = "IDLE"
    EXECUTING = "EXECUTING"
    ARRIVED = "ARRIVED"
    COMPLETE = "COMPLETE"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class StepResult:
    command: AllocatedCommand  # authorized copy when ``decision.authorized``; submit via CommandGateway
    decision: SafetyDecision
    state: RobotState
    assessment: SafetyAssessment
    reference: ControlReference


class _ReadOnlyHardware:
    """Exposes only the sensing half of the RHI to the stack."""

    def __init__(self, hw: RobotHardwareInterface) -> None:
        self.get_imu, self.get_depth = hw.get_imu, hw.get_depth
        self.get_camera, self.get_sonar = hw.get_camera, hw.get_sonar
        self.get_thruster_state, self.get_power_state = hw.get_thruster_state, hw.get_power_state
        self.get_health, self.now_ns, self.clock_domain = hw.get_health, hw.now_ns, hw.clock_domain
