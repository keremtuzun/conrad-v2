"""Python 6-DOF simulation kernel (TRUTH side) and the simulated RobotHardwareInterface adapter."""

from __future__ import annotations

import numpy as np

from conrad.schemas.robot import RobotConfig
from conrad.sim.kernel.dynamics import CurrentField, SignedDistance, SimKernel
from conrad.sim.kernel.faults import FaultRecord, FaultType
from conrad.sim.kernel.hardware import SIM_CLOCK, SensorRenderer, SimRobotHardware
from conrad.sim.kernel.params import (
    SIM_FRAME_CONVENTION,
    SimKernelConfig,
    SimValidityLevel,
    VehicleParams,
    vehicle_params_from_config,
)
from conrad.sim.kernel.truth import TrueVehicleState, TruthAccess

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": ["ch20 Unity 6-DOF dynamics", "ch21 Physics/Sensors/Faults", "ch21 validity levels"],
    "configuration_keys": ["configs/sim/nav_kernel.yaml", "RobotConfig.*"],
    "assumptions": [
        SIM_FRAME_CONVENTION,
        "diagonal added mass and inertia; CoM coupling enters through restoring moments only",
        "reverse thrust coefficient = k * max_reverse/max_forward",
        "sensor lever arms ignored (IMU/depth measured at body origin)",
        "electrical power P = c |T|^1.5 with a SYNTHETIC_ONLY coefficient",
    ],
    "baselines": ["RK4 fixed-step Fossen-form rigid body"],
    "acceptance_tests": ["tests/unit/sim_kernel"],
    "claim_status": "IMPLEMENTED",
    "simulation_validity_level": SimValidityLevel.L1_APPROXIMATE_PHYSICS.value,
}


def build_sim_hardware(
    robot_config: RobotConfig,
    seed: int,
    kernel_config: SimKernelConfig | None = None,
    current_field: CurrentField | None = None,
    sdf: SignedDistance | None = None,
    renderer: SensorRenderer | None = None,
) -> SimRobotHardware:
    """Deterministic construction: one seed feeds independent physics and sensor streams."""
    physics_seq, sensor_seq = np.random.SeedSequence(seed).spawn(2)
    kernel = SimKernel(
        vehicle_params_from_config(robot_config),
        kernel_config or SimKernelConfig(),
        np.random.default_rng(physics_seq),
        current_field=current_field,
        sdf=sdf,
    )
    return SimRobotHardware(kernel, robot_config, np.random.default_rng(sensor_seq), renderer)


__all__ = [
    "IMPLEMENTATION_METADATA",
    "SIM_CLOCK",
    "SIM_FRAME_CONVENTION",
    "CurrentField",
    "FaultRecord",
    "FaultType",
    "SensorRenderer",
    "SignedDistance",
    "SimKernel",
    "SimKernelConfig",
    "SimRobotHardware",
    "SimValidityLevel",
    "TrueVehicleState",
    "TruthAccess",
    "VehicleParams",
    "build_sim_hardware",
    "vehicle_params_from_config",
]
