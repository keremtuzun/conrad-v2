"""Default simulated sensor suite and robot spec. All values are labelled simulation defaults.

Sensor frame convention (simulation default): +X optical/acoustic axis, +Y left, +Z up.
Sensor WORLD pose = robot WORLD pose (o) ``SensorSpec.mount_pose`` (expressed in ROBOT).

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

from typing import Any

from conrad.schemas.frames import ROBOT, Pose
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Modality
from conrad.schemas.world import RobotSpec, SensorSpec

SOURCE = "SYNTHETIC_ONLY"

DEFAULT_SENSOR_PARAMETERS: dict[str, dict[str, Any]] = {
    Modality.RGB.value: {
        "width_px": 64,
        "height_px": 48,
        "hfov_deg": 70.0,
        "max_range_m": 8.0,
        "light_intensity": 1.0,
        "source_kind": SOURCE,
    },
    Modality.DEPTH_RANGE.value: {
        "width_px": 64,
        "height_px": 48,
        "hfov_deg": 70.0,
        "max_range_m": 10.0,
        "range_noise_sigma_m": 0.01,
        "dropout_prob": 0.01,
        "emit_point_cloud": True,
        "source_kind": SOURCE,
    },
    Modality.SONAR.value: {
        "n_beams": 64,
        "n_range_bins": 96,
        "hfov_deg": 90.0,
        "vfov_deg": 20.0,
        "n_elevation_rays": 5,
        "max_range_m": 20.0,
        "min_range_m": 0.3,
        "speckle_sigma": 0.25,
        "source_kind": SOURCE,
    },
    Modality.IMU.value: {
        "gyro_noise_sigma_rps": 0.002,
        "accel_noise_sigma_mps2": 0.02,
        "source_kind": SOURCE,
    },
    Modality.PRESSURE_DEPTH.value: {"depth_noise_sigma_m": 0.01, "source_kind": SOURCE},
}

_MOUNTS: dict[str, tuple[float, float, float]] = {
    Modality.RGB.value: (0.30, 0.0, 0.05),
    Modality.DEPTH_RANGE.value: (0.30, 0.0, -0.05),
    Modality.SONAR.value: (0.35, 0.0, 0.10),
    Modality.IMU.value: (0.0, 0.0, 0.0),
    Modality.PRESSURE_DEPTH.value: (0.0, 0.0, 0.0),
}
_RATES_HZ = {"RGB": 5.0, "DEPTH_RANGE": 5.0, "SONAR": 5.0, "IMU": 50.0, "PRESSURE_DEPTH": 10.0}


def default_sensor_suite(
    ids: IdFactory, overrides: dict[str, dict[str, Any]] | None = None
) -> tuple[SensorSpec, ...]:
    """RGB camera, forward sonar, depth/range, IMU, pressure depth. ``overrides[modality]`` patches parameters."""
    specs = []
    for modality, params in DEFAULT_SENSOR_PARAMETERS.items():
        merged = {**params, **(overrides or {}).get(modality, {})}
        specs.append(
            SensorSpec(
                sensor_id=ids.new(),
                modality=modality,
                frame_id=f"SENSOR_{modality}",
                mount_pose=Pose(frame_id=ROBOT, position_m=_MOUNTS[modality]),
                rate_hz=_RATES_HZ[modality],
                parameters=merged,
                calibration_ref=None,
            )
        )
    return tuple(specs)


def default_robot_spec(
    ids: IdFactory, initial_pose: Pose, overrides: dict[str, dict[str, Any]] | None = None
) -> RobotSpec:
    return RobotSpec(
        robot_id=ids.new(),
        robot_config_ref="configs/robot/sim_reference.yaml",
        initial_pose=initial_pose,
        sensors=default_sensor_suite(ids, overrides),
    )
