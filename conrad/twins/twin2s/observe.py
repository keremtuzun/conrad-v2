"""Observation / supervision assembly. Keeps the two planes physically separate. TRUTH PLANE.

Observation: measured payload by PayloadRef, ``robot_pose_estimate = ctx.estimated_pose``, and a
``sensor_context`` holding only sensor settings and self-reported health (no truth, no entity IDs).
SupervisionLabel: per-pixel/per-point true entity ids, true ranges, visibility masks, true poses, lineage.

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from conrad.persistence.object_store import ObjectStore
from conrad.schemas.frames import Pose
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Modality, Observation, PayloadRef, SensorHealth
from conrad.schemas.truth import SupervisionLabel
from conrad.schemas.world import Domain
from conrad.twins.base import SensingContext
from conrad.twins.twin2s.raycast import sensor_world_pose

GRAVITY_WORLD = np.array([0.0, 0.0, -9.80665])
"""Standard gravity in a +Z-up WORLD (the scenario FrameConvention default)."""

SENSOR_INTERNAL_KEYS = ("dropout", "range_noise_scale")
"""Degradations a sensor can self-report. Environmental ones (turbidity, backscatter...) are not health."""


def sensor_health(deg: dict[str, float]) -> SensorHealth:
    if deg.get("fault", 0.0) >= 1.0:
        return SensorHealth.FAULT
    if any(deg.get(k, 0.0) >= 0.3 for k in SENSOR_INTERNAL_KEYS):
        return SensorHealth.DEGRADED
    return SensorHealth.OK


def make_observation(
    ids: IdFactory,
    ctx: SensingContext,
    modality: Modality,
    payload: PayloadRef | None = None,
    inline: tuple[float, ...] | None = None,
    units: str | None = None,
    encoding: dict[str, Any] | None = None,
) -> Observation:
    health = sensor_health(ctx.degradation)
    return Observation(
        observation_id=ids.new(),
        mission_id=ctx.mission_id,
        run_id=ctx.run_id,
        trace_id=ctx.trace_id,
        sensor_id=ctx.sensor.sensor_id,
        modality=modality,
        timestamp=ctx.timestamp,
        sensor_frame=ctx.sensor.frame_id,
        robot_pose_estimate=ctx.estimated_pose,
        payload_ref=payload,
        inline_values=inline,
        inline_units=units,
        sensor_health=health,
        calibration_ref=ctx.sensor.calibration_ref,
        sensor_context={
            "settings": dict(ctx.sensor.parameters),
            "self_reported_health": health.value,
            "encoding": encoding or {},
        },
    )


def make_supervision(
    store: ObjectStore,
    obs: Observation,
    truth_arrays: dict[str, NDArray[np.generic]],
    entity_table: list[UUID],
    ctx: SensingContext,
    lineage: str,
    extra: dict[str, Any] | None = None,
) -> SupervisionLabel:
    """Large arrays are stored in the object store; ``targets`` carries only their digests."""
    rot, origin = sensor_world_pose(ctx.true_pose, ctx.sensor.mount_pose)
    targets: dict[str, Any] = {
        "arrays": {k: store.put_array(v).digest for k, v in sorted(truth_arrays.items())},
        "entity_id_table": [str(u) for u in entity_table],
        "true_robot_pose": ctx.true_pose.model_dump(mode="json"),
        "true_sensor_position_m": [float(x) for x in origin],
        "true_sensor_rotation_wxyz": [float(x) for x in Rotation.from_matrix(rot).as_quat(scalar_first=True)],
        **(extra or {}),
    }
    return SupervisionLabel(
        observation_id=obs.observation_id,
        domain=Domain.SPATIAL,
        targets=targets,
        target_masks=dict.fromkeys(truth_arrays, True),
        lineage=lineage,
    )


@dataclass
class PoseHistory:
    """Last true poses per IMU (truth plane), for finite-difference inertial synthesis."""

    samples: list[tuple[float, Pose]]

    def push(self, t_s: float, pose: Pose) -> None:
        if self.samples and t_s <= self.samples[-1][0]:
            raise ValueError("IMU samples must have strictly increasing time")
        self.samples = [*self.samples[-2:], (t_s, pose)]

    def imu(self) -> tuple[np.ndarray, np.ndarray] | None:
        """(specific force body m/s^2, angular rate body rad/s) or None with fewer than 3 samples."""
        if len(self.samples) < 3:
            return None
        (t0, p0), (t1, p1), (t2, p2) = self.samples
        x0, x1, x2 = (np.asarray(p.position_m) for p in (p0, p1, p2))
        v01, v12 = (x1 - x0) / (t1 - t0), (x2 - x1) / (t2 - t1)
        acc = (v12 - v01) / (0.5 * (t2 - t0))
        r1 = Rotation.from_quat(p1.orientation_wxyz, scalar_first=True)
        r2 = Rotation.from_quat(p2.orientation_wxyz, scalar_first=True)
        omega = (r1.inv() * r2).as_rotvec() / (t2 - t1)
        r2m = r2.as_matrix()
        return r2m.T @ (acc - GRAVITY_WORLD), omega
