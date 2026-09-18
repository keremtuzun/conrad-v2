"""Thruster allocation (ch20 Thruster allocation / Fault-tolerant allocation)."""

from conrad.robotics.allocation.allocator import (
    PRODUCER,
    AllocationConfig,
    AllocationResult,
    ThrusterAllocator,
    ThrusterLayoutUnknownError,
)

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": [
        "ch20 Thruster allocation",
        "ch20 Allocation problem",
        "ch20 Fault-tolerant allocation",
    ],
    "configuration_keys": [
        "RobotConfig.thrusters",
        "RobotConfig.safety.command_timeout_s",
        "AllocationConfig",
    ],
    "assumptions": [
        "static thrust model T = k u|u| with reverse coefficient k*max_reverse/max_forward",
        "DEGRADED thruster effectiveness is a configured estimate, not a measurement",
    ],
    "baselines": ["weighted pseudo-inverse", "bounded least squares (BVLS) on saturation"],
    "acceptance_tests": ["tests/unit/robotics/test_nav_allocation.py", "tests/property/robotics"],
    "claim_status": "IMPLEMENTED",
}

__all__ = [
    "IMPLEMENTATION_METADATA",
    "PRODUCER",
    "AllocationConfig",
    "AllocationResult",
    "ThrusterAllocator",
    "ThrusterLayoutUnknownError",
]
