"""Truth-plane adapter from a measured surface footprint to an Observation.

The caller must determine the footprint from pose, visibility and occlusion.
This adapter only samples truth within that supplied support.  It does not
attach a Twin entity ID, a belief-cell ID, or ground truth to the Observation.
"""

from __future__ import annotations

from uuid import UUID

import numpy as np

from conrad.schemas.frames import Pose
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Modality, Observation
from conrad.schemas.structural_sensor import StructuralSensorModel
from conrad.schemas.structural_support import CapsuleSurfaceSupport
from conrad.schemas.timebase import TimeStamp
from conrad.twins.twin2t.spatial_field import SpatialStructuralTruth


def emit_spatial_observation(
    truth: SpatialStructuralTruth,
    model: StructuralSensorModel,
    support: CapsuleSurfaceSupport,
    *,
    ids: IdFactory,
    rng: np.random.Generator,
    mission_id: UUID,
    run_id: UUID,
    trace_id: UUID,
    sensor_id: UUID,
    sensor_frame: str,
    timestamp: TimeStamp,
    estimated_pose: Pose,
    measured_range_m: float,
    measured_bearing_rad: float,
    measured_elevation_rad: float,
    range_sigma_m: float,
    angle_sigma_rad: float,
    independence_group: str,
) -> Observation:
    if not independence_group:
        raise ValueError("capture independence group required")
    if measured_range_m < model.range_min_m or measured_range_m > model.range_max_m:
        raise ValueError("measurement outside declared sensor range")
    if range_sigma_m < 0 or angle_sigma_rad < 0:
        raise ValueError("negative geometry uncertainty")
    value = truth.measure(support, model)
    noise = model.noise_sigma_m * rng.standard_normal(2)
    return Observation(
        observation_id=ids.new(),
        mission_id=mission_id,
        run_id=run_id,
        trace_id=trace_id,
        sensor_id=sensor_id,
        modality=Modality.STRUCTURED,
        timestamp=timestamp,
        sensor_frame=sensor_frame,
        robot_pose_estimate=estimated_pose,
        inline_values=(
            max(0.0, value.corrosion_depth_m + float(noise[0])),
            max(0.0, value.crack_length_m + float(noise[1])),
        ),
        inline_units="m",
        structural_support=support,
        sensor_context={
            "measurement_names": ["apparent_wall_loss", "crack_indication_length"],
            "independence_group": independence_group,
            "structural_sensor_version": model.version,
            "structural_sensor_digest": model.digest,
            "measured_range_m": measured_range_m,
            "measured_bearing_rad": measured_bearing_rad,
            "measured_elevation_rad": measured_elevation_rad,
            "range_sigma_m": range_sigma_m,
            "angle_sigma_rad": angle_sigma_rad,
        },
    )
