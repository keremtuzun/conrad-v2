"""RobotHardwareInterface: one contract for the simulated and the physical robot (ch1, ch20).

Upper layers never branch on the concrete adapter type. ``send`` may only be invoked by
``conrad.runtime.command_gateway`` (enforced by tests/leakage/test_actuator_path.py).

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from conrad.schemas.observation import Observation
from conrad.schemas.robot import (
    AllocatedCommand,
    BatteryState,
    CommandAck,
    DepthSample,
    ImuSample,
    RobotCapabilities,
    SystemHealth,
    ThrusterState,
)


class HardwareUnavailableError(RuntimeError):
    pass


class RobotHardwareInterface(ABC):
    """Every reading is timestamped, framed and carries device health; absent sensors return None."""

    adapter_name: str = "abstract"
    is_physical: bool = False

    @abstractmethod
    def capabilities(self) -> RobotCapabilities: ...

    @abstractmethod
    def robot_config_digest(self) -> str: ...

    @abstractmethod
    def clock_domain(self) -> str: ...

    @abstractmethod
    def now_ns(self) -> int: ...

    @abstractmethod
    def get_imu(self) -> ImuSample | None: ...

    @abstractmethod
    def get_depth(self) -> DepthSample | None: ...

    @abstractmethod
    def get_camera(self) -> Observation | None: ...

    @abstractmethod
    def get_sonar(self) -> Observation | None: ...

    @abstractmethod
    def get_thruster_state(self) -> tuple[ThrusterState, ...]: ...

    @abstractmethod
    def get_power_state(self) -> BatteryState | None: ...

    @abstractmethod
    def get_health(self) -> SystemHealth: ...

    @abstractmethod
    def send(self, command: AllocatedCommand) -> CommandAck:
        """Execute thruster commands. Callable ONLY from the Command Gateway."""

    def set_thruster_commands(self, command: AllocatedCommand) -> CommandAck:
        """Spec-named alias kept for contract parity; identical gateway-only restriction."""
        return self.send(command)


class PhysicalRobotHardware(RobotHardwareInterface):
    """Physical adapter boundary. The vendor drivers are the hardware owner's deliverable.

    This class is intentionally NOT a no-op: every call fails closed until a driver-backed
    subclass is supplied (external blocker EXT-HW-01).
    """

    adapter_name = "physical"
    is_physical = True

    def _blocked(self) -> HardwareUnavailableError:
        return HardwareUnavailableError("physical RobotHardwareInterface driver not supplied (EXT-HW-01)")

    def capabilities(self) -> RobotCapabilities:
        raise self._blocked()

    def robot_config_digest(self) -> str:
        raise self._blocked()

    def clock_domain(self) -> str:
        raise self._blocked()

    def now_ns(self) -> int:
        raise self._blocked()

    def get_imu(self) -> ImuSample | None:
        raise self._blocked()

    def get_depth(self) -> DepthSample | None:
        raise self._blocked()

    def get_camera(self) -> Observation | None:
        raise self._blocked()

    def get_sonar(self) -> Observation | None:
        raise self._blocked()

    def get_thruster_state(self) -> tuple[ThrusterState, ...]:
        raise self._blocked()

    def get_power_state(self) -> BatteryState | None:
        raise self._blocked()

    def get_health(self) -> SystemHealth:
        raise self._blocked()

    def send(self, command: AllocatedCommand) -> CommandAck:
        raise self._blocked()
