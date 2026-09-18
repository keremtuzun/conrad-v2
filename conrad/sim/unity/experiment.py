"""ExperimentConfig, deterministic fault schedules and simulation validity assessment (ch20, ch21).

``seed + versions + config`` reproduces the same run: the fault schedule is drawn from a seeded
``numpy.random.Generator`` and its digest goes into the experiment record.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import Field

from conrad.adapters.unity.protocol import FaultInjectionRequest, FaultType, SimulationValidityLevel
from conrad.schemas.base import VersionedModel, digest_of
from conrad.schemas.robot import PHYSICALLY_GROUNDED, RobotConfig, SourceKind

DYNAMICS_PARAMETERS = ("linear_drag", "quadratic_drag", "added_mass_diag")
MASS_PARAMETERS = (
    "mass_kg",
    "displaced_volume_m3",
    "center_of_mass_body_m",
    "center_of_buoyancy_body_m",
    "inertia_diag_kgm2",
)


class ExperimentConfig(VersionedModel):
    """Every simulation run is ``Experiment(config, seed)``."""

    experiment_name: str = Field(min_length=1)
    seed: int
    scenario_version: str
    robot_config_version: str
    robot_config_digest: str = Field(pattern="^[0-9a-f]{64}$")
    physics_config: str
    sensor_config: str
    noise_config: str
    failure_config: str
    mission_config: str
    model_versions: dict[str, str] = Field(default_factory=dict)
    controller_version: str
    physics_dt_ns: int = Field(gt=0)
    lock_step: bool = True


class FaultSpec(VersionedModel):
    """Stochastic fault template: type, target and windows for start time / duration / magnitude."""

    fault_type: FaultType
    target: str | None = None
    probability: float = Field(ge=0, le=1)
    start_window_s: tuple[float, float]
    duration_window_s: tuple[float, float] | None = None
    magnitude_window: tuple[float, float] = (1.0, 1.0)


def draw_fault_schedule(
    specs: tuple[FaultSpec, ...], seed: int
) -> tuple[tuple[FaultInjectionRequest, ...], str]:
    """Deterministic fault schedule and its digest. Same seed -> identical schedule."""
    rng = np.random.default_rng(seed)
    out: list[FaultInjectionRequest] = []
    for i, spec in enumerate(specs):
        # Always draw every variate so later specs are unaffected by earlier outcomes.
        occur = float(rng.random()) < spec.probability
        start_s = float(rng.uniform(*spec.start_window_s))
        dur_s = float(rng.uniform(*spec.duration_window_s)) if spec.duration_window_s else None
        mag = float(rng.uniform(*spec.magnitude_window))
        if not occur:
            continue
        out.append(
            FaultInjectionRequest(
                fault_id=f"fault-{seed}-{i:03d}",
                fault_type=spec.fault_type,
                target=spec.target,
                start_time_ns=round(start_s * 1e9),
                duration_ns=None if dur_s is None else max(1, round(dur_s * 1e9)),
                magnitude=mag,
            )
        )
    schedule = tuple(sorted(out, key=lambda f: (f.start_time_ns, f.fault_id)))
    return schedule, digest_of([f.model_dump(mode="json") for f in schedule])


class ValidityAssessment(VersionedModel):
    level: SimulationValidityLevel
    reasons: tuple[str, ...]
    validated_operating_envelope: bool
    evidence: dict[str, Any] = Field(default_factory=dict)


def _sources(config: RobotConfig, names: tuple[str, ...]) -> set[SourceKind]:
    return {getattr(config, n).source for n in names}


def assess_validity_level(
    config: RobotConfig,
    *,
    physics_enabled: bool = True,
    validation_report_ref: str | None = None,
) -> ValidityAssessment:
    """L0..L4 from the provenance of the RobotConfig plus held-out validation evidence.

    L2 needs MEASURED/IDENTIFIED mass properties and thrusters; L3 additionally IDENTIFIED dynamics;
    L4 additionally a held-out validation report reference. Nothing is inferred from a claim.
    """
    reasons: list[str] = []
    if not physics_enabled:
        return ValidityAssessment(
            level=SimulationValidityLevel.L0_FUNCTIONAL,
            reasons=("physics disabled",),
            validated_operating_envelope=False,
        )
    if config.open_parameters():
        reasons.append("OPEN parameters present")
        return ValidityAssessment(
            level=SimulationValidityLevel.L0_FUNCTIONAL,
            reasons=tuple(reasons),
            validated_operating_envelope=False,
        )
    thruster_sources = {
        getattr(t, f).source
        for t in config.thrusters
        for f in ("thrust_coefficient", "time_constant_s", "deadzone_command", "position_body_m")
    }
    mass_ok = _sources(config, MASS_PARAMETERS) <= PHYSICALLY_GROUNDED
    thr_ok = bool(thruster_sources) and thruster_sources <= PHYSICALLY_GROUNDED
    dyn_ok = _sources(config, DYNAMICS_PARAMETERS) <= {SourceKind.IDENTIFIED}
    level = SimulationValidityLevel.L1_APPROXIMATE_PHYSICS
    if not mass_ok:
        reasons.append("mass properties not MEASURED/IDENTIFIED")
    if not thr_ok:
        reasons.append("thruster parameters not MEASURED/IDENTIFIED")
    if mass_ok and thr_ok:
        level = SimulationValidityLevel.L2_CHARACTERIZED
        if dyn_ok:
            level = SimulationValidityLevel.L3_IDENTIFIED
            if validation_report_ref:
                level = SimulationValidityLevel.L4_VALIDATED_ENVELOPE
            else:
                reasons.append("no held-out validation report")
        else:
            reasons.append("drag/added mass not IDENTIFIED")
    return ValidityAssessment(
        level=level,
        reasons=tuple(reasons),
        validated_operating_envelope=level is SimulationValidityLevel.L4_VALIDATED_ENVELOPE,
        evidence={"validation_report_ref": validation_report_ref},
    )
