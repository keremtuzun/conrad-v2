"""NAV-LOCAL-01: potential-field obstacle avoidance applied to the controller reference.

It only bends the reference (position + feed-forward velocity); it never produces thrust.
Obstacle range comes from a local distance query (sonar-derived map, SDF of the belief map, ...).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
from pydantic import Field

from conrad.robotics.control.pid import ControlReference
from conrad.schemas.base import ConradModel

LocalDistance = Callable[[np.ndarray], np.ndarray]
"""``distance(points[N,3]) -> [N]`` distance to the nearest known obstacle surface (metres)."""


class LocalPlannerConfig(ConradModel):
    influence_m: float = Field(default=0.6, gt=0, description="repulsion starts inside D_safe + this")
    gain: float = Field(default=0.6, ge=0)
    max_push_mps: float = Field(default=0.4, ge=0)
    gradient_eps_m: float = Field(default=0.05, gt=0)


class PotentialFieldLocalPlanner:
    name = "NAV-LOCAL-01-potential-field"

    def __init__(self, distance: LocalDistance | None, config: LocalPlannerConfig | None = None) -> None:
        self._distance = distance
        self.config = config or LocalPlannerConfig()
        self.last_clearance_m: float | None = None

    def clearance(self, position: np.ndarray) -> float | None:
        if self._distance is None:
            return None
        return float(np.asarray(self._distance(position[None, :]))[0])

    def adjust(
        self,
        position: np.ndarray,
        velocity_world: np.ndarray,
        reference: ControlReference,
        envelope: Callable[[float], float],
    ) -> ControlReference:
        """``envelope(closing_speed_mps) -> D_safe``: the margin grows with speed TOWARD the obstacle."""
        if self._distance is None:
            return reference
        c = self.config
        d = self.clearance(position)
        self.last_clearance_m = d
        assert d is not None
        e = c.gradient_eps_m
        probes = np.concatenate([position + e * np.eye(3), position - e * np.eye(3)])
        vals = np.asarray(self._distance(probes))
        grad = (vals[:3] - vals[3:]) / (2 * e)
        n = float(np.linalg.norm(grad))
        if n < 1e-9:
            return reference
        normal = grad / n
        d_safe_m = envelope(max(0.0, -float(velocity_world @ normal)))
        if not np.isfinite(d_safe_m):
            return reference
        margin = d - d_safe_m
        if margin >= c.influence_m:
            return reference
        strength = c.gain * (1.0 / max(margin, 0.05) - 1.0 / c.influence_m)
        push = min(c.max_push_mps, max(0.0, strength)) * normal
        v_ref = reference.velocity_world_mps
        inward = float(v_ref @ normal)
        if inward < 0:  # remove the component that drives into the obstacle
            v_ref = v_ref - inward * normal
        target = reference.position_world_m
        to_target = target - position
        toward = float(to_target @ normal)
        if toward < 0 and margin < 0.5 * c.influence_m:  # reference behind the wall: hold the stand-off
            target = target - toward * normal
        return replace(reference, position_world_m=target, velocity_world_mps=v_ref + push)
