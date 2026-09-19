"""Identification log collection protocol (EXT-HW-04) as data: channels, profiles, repetitions.

This module is the machine-readable twin of ``docs/IDENTIFICATION_LOG_PROTOCOL.md``. The intake checker
and the manoeuvre runner both read it, so the document, the checker and the runner cannot drift apart.

Every number here is labelled. ``ENGINEERING_ESTIMATE`` values are starting points chosen before any
physical data exists; the hardware owner may tighten them. Nothing in this file is a vehicle measurement.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from conrad.robotics.hardware.identification.dataset import ExperimentKind

PROTOCOL_VERSION = "identification-protocol.v1"

DEFINITION = (
    "Time-synchronized command, sensor, power and independently measured vehicle-motion records from "
    "repeated controlled maneuvers (static trim, thruster steps, surge, sway, heave, yaw and station "
    "keeping), split into identification and held-out validation runs, for estimating and validating "
    "the robot's physical parameters."
)

ENGINEERING_ESTIMATE = "ENGINEERING_ESTIMATE"
OPEN = "OPEN"


class Variant(str, Enum):
    """Sub-procedure of one experiment kind. Only static trim has more than one."""

    STANDARD = "STANDARD"
    SCALE = "SCALE"  # A: vehicle hung from a submerged scale -> buoyancy
    TILT_WEIGHTS = "TILT_WEIGHTS"  # A: known weights at known lever arms -> CoB-CoG offset
    TILT_THRUSTERS = "TILT_THRUSTERS"  # A: thruster-applied pitch moments (needs B models first)


class MotionReferenceKind(str, Enum):
    """Where the independent pose/velocity comes from. The robot's own estimator is never accepted."""

    DVL = "DVL"
    USBL = "USBL"
    EXTERNAL_TRACKING = "EXTERNAL_TRACKING"
    SIMULATOR_TRUTH = "SIMULATOR_TRUTH"  # synthetic logs only


@dataclass(frozen=True)
class ChannelSpec:
    """One CSV column. ``{tid}`` in the name expands to every thruster id in the manifest."""

    name: str
    units: str
    min_rate_hz: float
    group: str
    rate_checked: bool = True  # command columns are piecewise constant by design
    rate_label: str = ENGINEERING_ESTIMATE

    def expand(self, thruster_ids: tuple[str, ...]) -> tuple[ChannelSpec, ...]:
        if "{tid}" not in self.name:
            return (self,)
        return tuple(
            ChannelSpec(
                self.name.replace("{tid}", t),
                self.units,
                self.min_rate_hz,
                self.group,
                self.rate_checked,
                self.rate_label,
            )
            for t in thruster_ids
        )


COMMAND = (ChannelSpec("cmd_{tid}", "unitless", 20.0, "command", rate_checked=False),)
IMU = tuple(
    ChannelSpec(n, u, 50.0, "imu")
    for n, u in (
        ("imu_ax_mps2", "m/s^2"),
        ("imu_ay_mps2", "m/s^2"),
        ("imu_az_mps2", "m/s^2"),
        ("imu_gx_rps", "rad/s"),
        ("imu_gy_rps", "rad/s"),
        ("imu_gz_rps", "rad/s"),
    )
)
PRESSURE = (ChannelSpec("pressure_depth_m", "m", 5.0, "pressure"),)
POWER = (
    ChannelSpec("battery_voltage_v", "V", 1.0, "power"),
    ChannelSpec("battery_current_a", "A", 1.0, "power"),
)
REFERENCE_ORIENTATION = (
    ChannelSpec("ref_roll_rad", "rad", 10.0, "reference"),
    ChannelSpec("ref_pitch_rad", "rad", 10.0, "reference"),
)
REFERENCE = (
    ChannelSpec("ref_x_m", "m", 10.0, "reference"),
    ChannelSpec("ref_y_m", "m", 10.0, "reference"),
    ChannelSpec("ref_z_m", "m", 10.0, "reference"),
    *REFERENCE_ORIENTATION,
    ChannelSpec("ref_yaw_rad", "rad", 10.0, "reference"),
    ChannelSpec("ref_u_mps", "m/s", 10.0, "reference"),
    ChannelSpec("ref_v_mps", "m/s", 10.0, "reference"),
    ChannelSpec("ref_w_mps", "m/s", 10.0, "reference"),
    ChannelSpec("ref_p_rps", "rad/s", 10.0, "reference"),
    ChannelSpec("ref_q_rps", "rad/s", 10.0, "reference"),
    ChannelSpec("ref_r_rps", "rad/s", 10.0, "reference"),
)
MEASURED_THRUST = (ChannelSpec("thrust_n", "N", 100.0, "load_cell"),)
CURRENT_METER = (
    ChannelSpec("current_x_mps", "m/s", 1.0, "current_meter"),
    ChannelSpec("current_y_mps", "m/s", 1.0, "current_meter"),
)
SCALE = (ChannelSpec("net_weight_n", "N", 1.0, "scale"),)
APPLIED_MOMENT = (ChannelSpec("applied_pitch_moment_nm", "N*m", 1.0, "applied_moment"),)

# Recorded when the hardware exposes them; units are checked if the column is present.
OPTIONAL = (
    ChannelSpec("rpm_{tid}", "rpm", 20.0, "thruster_telemetry"),
    ChannelSpec("thr_current_{tid}_a", "A", 20.0, "thruster_telemetry"),
    ChannelSpec("current_z_mps", "m/s", 1.0, "current_meter"),
    ChannelSpec("wrench_cmd_fx_n", "N", 20.0, "command", rate_checked=False),
    ChannelSpec("wrench_cmd_fy_n", "N", 20.0, "command", rate_checked=False),
    ChannelSpec("wrench_cmd_fz_n", "N", 20.0, "command", rate_checked=False),
    ChannelSpec("wrench_cmd_mx_nm", "N*m", 20.0, "command", rate_checked=False),
    ChannelSpec("wrench_cmd_my_nm", "N*m", 20.0, "command", rate_checked=False),
    ChannelSpec("wrench_cmd_mz_nm", "N*m", 20.0, "command", rate_checked=False),
)

_MOTION = (*COMMAND, *REFERENCE, *IMU, *PRESSURE, *POWER)

REQUIRED_CHANNELS: dict[tuple[ExperimentKind, Variant], tuple[ChannelSpec, ...]] = {
    (ExperimentKind.STATIC_TRIM, Variant.SCALE): (*SCALE, *PRESSURE),
    (ExperimentKind.STATIC_TRIM, Variant.TILT_WEIGHTS): (
        *APPLIED_MOMENT,
        *REFERENCE_ORIENTATION,
        *IMU,
        *PRESSURE,
    ),
    (ExperimentKind.STATIC_TRIM, Variant.TILT_THRUSTERS): (*COMMAND, *REFERENCE, *IMU, *PRESSURE, *POWER),
    (ExperimentKind.THRUSTER_STEP, Variant.STANDARD): (*COMMAND, *MEASURED_THRUST, *POWER),
    (ExperimentKind.SURGE_ACCELERATION, Variant.STANDARD): _MOTION,
    (ExperimentKind.SWAY, Variant.STANDARD): _MOTION,
    (ExperimentKind.HEAVE, Variant.STANDARD): _MOTION,
    (ExperimentKind.YAW_ROTATION, Variant.STANDARD): _MOTION,
    (ExperimentKind.STATION_KEEPING, Variant.STANDARD): (*_MOTION, *CURRENT_METER),
}

# Kinds whose fit input force is derived from thruster commands through the identified B models.
NEEDS_THRUSTER_MODELS = frozenset(
    {
        (ExperimentKind.STATIC_TRIM, Variant.TILT_THRUSTERS),
        (ExperimentKind.SURGE_ACCELERATION, Variant.STANDARD),
        (ExperimentKind.SWAY, Variant.STANDARD),
        (ExperimentKind.HEAVE, Variant.STANDARD),
        (ExperimentKind.YAW_ROTATION, Variant.STANDARD),
        (ExperimentKind.STATION_KEEPING, Variant.STANDARD),
    }
)

# Whole repetitions per group (kind, variant, thruster). ENGINEERING_ESTIMATE.
MIN_IDENTIFICATION_REPETITIONS = 3
MIN_VALIDATION_REPETITIONS = 1

# Primary response channel per kind, used for the repeatability statistic.
PRIMARY_RESPONSE: dict[tuple[ExperimentKind, Variant], str] = {
    (ExperimentKind.STATIC_TRIM, Variant.SCALE): "net_weight_n",
    (ExperimentKind.STATIC_TRIM, Variant.TILT_WEIGHTS): "ref_pitch_rad",
    (ExperimentKind.STATIC_TRIM, Variant.TILT_THRUSTERS): "ref_pitch_rad",
    (ExperimentKind.THRUSTER_STEP, Variant.STANDARD): "thrust_n",
    (ExperimentKind.SURGE_ACCELERATION, Variant.STANDARD): "ref_u_mps",
    (ExperimentKind.SWAY, Variant.STANDARD): "ref_v_mps",
    (ExperimentKind.HEAVE, Variant.STANDARD): "ref_w_mps",
    (ExperimentKind.YAW_ROTATION, Variant.STANDARD): "ref_r_rps",
    (ExperimentKind.STATION_KEEPING, Variant.STANDARD): "ref_x_m",
}

# Body wrench index (fx, fy, fz, mx, my, mz) excited by each wrench manoeuvre.
WRENCH_AXIS: dict[ExperimentKind, int] = {
    ExperimentKind.SURGE_ACCELERATION: 0,
    ExperimentKind.SWAY: 1,
    ExperimentKind.HEAVE: 2,
    ExperimentKind.STATIC_TRIM: 4,
    ExperimentKind.YAW_ROTATION: 5,
}
WRENCH_COLUMNS = (
    ("wrench_cmd_fx_n", "N"),
    ("wrench_cmd_fy_n", "N"),
    ("wrench_cmd_fz_n", "N"),
    ("wrench_cmd_mx_nm", "N*m"),
    ("wrench_cmd_my_nm", "N*m"),
    ("wrench_cmd_mz_nm", "N*m"),
)
REFERENCE_VELOCITY: dict[ExperimentKind, tuple[str, str]] = {
    ExperimentKind.SURGE_ACCELERATION: ("ref_u_mps", "m/s"),
    ExperimentKind.SWAY: ("ref_v_mps", "m/s"),
    ExperimentKind.HEAVE: ("ref_w_mps", "m/s"),
    ExperimentKind.YAW_ROTATION: ("ref_r_rps", "rad/s"),
}

REQUIRED_SEGMENT_METADATA = ("water_density_kgm3", "water_temperature_c")
REQUIRED_CONFIGURATION = ("ballast", "payload", "tether")


def required_channels(
    kind: ExperimentKind, variant: Variant, thruster_ids: tuple[str, ...]
) -> tuple[ChannelSpec, ...]:
    specs = REQUIRED_CHANNELS.get((kind, variant))
    if specs is None:
        raise KeyError(f"{kind.value}/{variant.value} is not a protocol procedure")
    return tuple(x for s in specs for x in s.expand(thruster_ids))


def optional_channels(thruster_ids: tuple[str, ...]) -> tuple[ChannelSpec, ...]:
    return tuple(x for s in OPTIONAL for x in s.expand(thruster_ids))


# -- command profiles (ENGINEERING_ESTIMATE; the safety envelope caps every level) ---------------------


@dataclass(frozen=True)
class StepProfile:
    """B: alternating zero / level plateaus of one thruster's normalised command, both signs."""

    levels: tuple[float, ...]
    on_s: float
    off_s: float
    label: str = ENGINEERING_ESTIMATE

    @property
    def duration_s(self) -> float:
        return len(self.levels) * (self.on_s + self.off_s) + self.off_s


@dataclass(frozen=True)
class WrenchPulseProfile:
    """A-tilt / C / D / E: body-wrench plateaus on one axis, as signed fractions of that axis' capability.

    Like the step profile it starts and ends at zero: off, on, off, on, ..., off (the off phases are the
    coast-downs, and the leading one gives the thruster model its at-rest initial condition)."""

    fractions: tuple[float, ...]
    on_s: float
    off_s: float
    label: str = ENGINEERING_ESTIMATE

    @property
    def duration_s(self) -> float:
        return len(self.fractions) * (self.on_s + self.off_s) + self.off_s


@dataclass(frozen=True)
class HoldProfile:
    """F: hold station against the measured current for ``duration_s``; the settled tail is fitted."""

    duration_s: float
    settled_fraction: float = 0.5
    label: str = ENGINEERING_ESTIMATE


_LADDER = (0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8)

DEFAULT_PROFILES: dict[tuple[ExperimentKind, Variant], StepProfile | WrenchPulseProfile | HoldProfile] = {
    (ExperimentKind.THRUSTER_STEP, Variant.STANDARD): StepProfile(
        levels=tuple(s * x for x in _LADDER for s in (1.0, -1.0)), on_s=2.0, off_s=2.0
    ),
    (ExperimentKind.STATIC_TRIM, Variant.TILT_THRUSTERS): WrenchPulseProfile(
        fractions=(0.01, -0.01, 0.02, -0.02, 0.03, -0.03), on_s=20.0, off_s=10.0
    ),
    (ExperimentKind.SURGE_ACCELERATION, Variant.STANDARD): WrenchPulseProfile(
        fractions=(0.2, -0.2, 0.4, -0.4, 0.6, -0.6), on_s=4.0, off_s=8.0
    ),
    (ExperimentKind.SWAY, Variant.STANDARD): WrenchPulseProfile(
        fractions=(0.2, -0.2, 0.4, -0.4, 0.6, -0.6), on_s=4.0, off_s=8.0
    ),
    (ExperimentKind.HEAVE, Variant.STANDARD): WrenchPulseProfile(
        fractions=(0.2, -0.2, 0.4, -0.4, 0.6, -0.6), on_s=4.0, off_s=8.0
    ),
    (ExperimentKind.YAW_ROTATION, Variant.STANDARD): WrenchPulseProfile(
        fractions=(0.2, -0.2, 0.4, -0.4, 0.6, -0.6), on_s=3.0, off_s=6.0
    ),
    (ExperimentKind.STATION_KEEPING, Variant.STANDARD): HoldProfile(duration_s=120.0),
}
