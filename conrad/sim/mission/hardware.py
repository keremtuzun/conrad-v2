"""MissionHardware: the simulated RHI plus the declared payload sensors. TRUTH-SIDE adapter.

The deployment side calls only RobotHardwareInterface methods and ``get_payload_observations``; the payload
sensors are declared in ``RobotCapabilities.environmental_sensors``.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import numpy as np

from conrad.schemas.frames import Pose
from conrad.schemas.observation import Observation
from conrad.schemas.robot import RobotCapabilities, RobotConfig
from conrad.sim.kernel import SimKernelConfig, SimRobotHardware
from conrad.sim.kernel.dynamics import CurrentField, SignedDistance, SimKernel
from conrad.sim.kernel.params import vehicle_params_from_config
from conrad.sim.mission.sensing import MissionSensorSuite

PAYLOAD_SENSORS = (
    "structural_inspection",
    "structural_inspection_aux",
    "depth_range_imager",
    "imaging_sonar",
    "environmental_probe",
    "ecological_survey",
    "usbl_position_fix",
)


class MissionHardware(SimRobotHardware):
    adapter_name = "sim_kernel_mission"

    def __init__(
        self, kernel: SimKernel, robot_config: RobotConfig, rng: np.random.Generator, ecological: bool
    ) -> None:
        super().__init__(kernel, robot_config, rng)
        self.suite: MissionSensorSuite | None = None
        self._ecological = ecological

    def attach_suite(self, suite: MissionSensorSuite) -> None:
        self.suite = suite

    def capabilities(self) -> RobotCapabilities:
        declared = (
            PAYLOAD_SENSORS
            if self._ecological
            else tuple(s for s in PAYLOAD_SENSORS if s not in ("environmental_probe", "ecological_survey"))
        )
        return super().capabilities().model_copy(update={"environmental_sensors": declared})

    def get_payload_observations(self) -> list[Observation]:
        """Payload readings due at the current sim time. Each carries only the ESTIMATED pose."""
        if self.suite is None:
            return []
        truth = self.truth_access().true_state()
        est: Pose | None = self._estimated_pose()
        return self.suite.collect(truth.t_s, self._stamp(truth.t_s), truth.pose(), est)


def build_mission_hardware(
    robot_config: RobotConfig,
    seed: int,
    kernel_config: SimKernelConfig,
    sdf: SignedDistance,
    current_field: CurrentField | None,
    ecological: bool,
) -> MissionHardware:
    """Same deterministic seeding as ``conrad.sim.kernel.build_sim_hardware``."""
    physics_seq, sensor_seq = np.random.SeedSequence(seed).spawn(2)
    kernel = SimKernel(
        vehicle_params_from_config(robot_config),
        kernel_config,
        np.random.default_rng(physics_seq),
        current_field=current_field,
        sdf=sdf,
    )
    return MissionHardware(kernel, robot_config, np.random.default_rng(sensor_seq), ecological)
