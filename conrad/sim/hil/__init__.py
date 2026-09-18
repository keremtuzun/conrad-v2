"""HIL harness: deployment-equivalent stack loop against any RobotHardwareInterface, plus the HIL gate."""

from conrad.sim.hil.harness import (
    Advance,
    FaultTrial,
    HilConfig,
    HilHarness,
    SensorFrame,
    StackOutput,
    StackStep,
    Submit,
    default_platform_id,
)
from conrad.sim.hil.metrics import (
    NOT_AVAILABLE,
    FixedRateScheduler,
    LatencyStats,
    ResourceReport,
    ResourceSampler,
    default_gpu_probe,
)
from conrad.sim.hil.report import (
    FaultTrialResult,
    GateCheck,
    GateStatus,
    HilGateCriteria,
    HilGateReport,
    HilMode,
    SensorRateResult,
    evaluate_gate,
)
from conrad.sim.hil.settings import HilSettingsError, hil_config, load_hil_config

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": [
        "ch20 HIL architecture",
        "ch20 Why HIL matters",
        "ch20 HIL gate",
        "ch20 Runtime frequencies",
        "ch21 Hardware-in-the-loop",
        "ch22 Sim-to-real ladder R1/R2",
    ],
    "configuration_keys": [
        "sim.hil.mode",
        "sim.hil.rate_hz",
        "sim.hil.cycles",
        "sim.hil.warmup_cycles",
        "sim.hil.sensor_rates_hz",
        "sim.hil.stale_tolerance",
        "sim.hil.target_platform_id",
        "sim.hil.criteria.*",
    ],
    "assumptions": [
        "frame drop = a cycle in which a sensor's newest sample is older than stale_tolerance / rate",
        "observation->command latency is wall time from the start of sensor reads to the gateway ack",
        "memory growth is Python-heap growth (tracemalloc) after warm-up; native memory is NOT_AVAILABLE on hosts"
        " without /proc",
        "gate thresholds are configuration, not spec values; the spec leaves them to hardware",
    ],
    "baselines": ["host-HIL against conrad.sim.kernel SimRobotHardware"],
    "acceptance_tests": ["tests/unit/hil"],
    "claim_status": "IMPLEMENTED",
    "target_hil_status": "BLOCKED_EXTERNAL",
}

__all__ = [
    "IMPLEMENTATION_METADATA",
    "NOT_AVAILABLE",
    "Advance",
    "FaultTrial",
    "FaultTrialResult",
    "FixedRateScheduler",
    "GateCheck",
    "GateStatus",
    "HilConfig",
    "HilGateCriteria",
    "HilGateReport",
    "HilHarness",
    "HilMode",
    "HilSettingsError",
    "LatencyStats",
    "ResourceReport",
    "ResourceSampler",
    "SensorFrame",
    "SensorRateResult",
    "StackOutput",
    "StackStep",
    "Submit",
    "default_gpu_probe",
    "default_platform_id",
    "evaluate_gate",
    "hil_config",
    "load_hil_config",
]
