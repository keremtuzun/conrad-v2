"""StateEstimator contract (ch20 State estimator interface). Consumes RHI readings only, never sim truth."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from conrad.schemas.observation import Observation
from conrad.schemas.robot import DepthSample, ImuSample, RobotState, ThrusterState

POSITION_FIX_KIND = "POSITION_FIX"
"""``Observation.sensor_context['kind']`` of a USBL/DVL-like or visual WORLD position fix.

Payload: ``inline_values = (x, y, z)`` in WORLD metres, ``sensor_context['sigma_m']`` 1-sigma.
"""


@dataclass(frozen=True)
class MapConstraint:
    """A position constraint from the spatial belief plane (e.g. a re-observed landmark)."""

    position_world_m: np.ndarray
    sigma_m: float
    frame_id: str = "WORLD"


class StateEstimator(ABC):
    name: str = "abstract"

    @abstractmethod
    def predict(self, imu: ImuSample, control: tuple[ThrusterState, ...] | None, dt: float) -> RobotState: ...

    @abstractmethod
    def update_depth(self, depth: DepthSample) -> None: ...

    @abstractmethod
    def update_visual(self, visual: Observation) -> bool: ...

    @abstractmethod
    def update_sonar(self, sonar: Observation) -> bool: ...

    @abstractmethod
    def update_map_constraint(self, constraint: MapConstraint) -> bool: ...

    @abstractmethod
    def get_state(self) -> RobotState: ...
