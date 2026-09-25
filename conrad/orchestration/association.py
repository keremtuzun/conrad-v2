"""Structural observation -> registry component association by projection and gating. DEPLOYMENT PLANE.

The sensor-frame range/bearing/elevation of a STRUCTURED observation is projected into WORLD through the
ESTIMATED pose (never the true pose) and the sensor mount from the mission context. Each registry
component contributes one transient association cell centred on the closest point of its surveyed design
surface; ``conrad.core.association`` then gates (retrieve_candidates) and decides (nearest neighbour).
NO_MATCH is a legitimate outcome and such evidence is never forced onto a component.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from uuid import UUID

import numpy as np

from conrad.core.association import AssociationDecision, nearest_neighbour_decision, retrieve_candidates
from conrad.core.config import AssociationConfig
from conrad.orchestration.mission_config import AssociationSettings
from conrad.orchestration.mission_context import DesignComponent, MissionContext
from conrad.schemas.belief import BeliefCell, KnowledgeStatus, Lifecycle
from conrad.schemas.frames import WORLD, SpatialSupport, quat_to_matrix
from conrad.schemas.observation import EntityCandidate, Evidence, Observation
from conrad.schemas.uncertainty import unknown_uncertainty
from conrad.schemas.world import Domain

MEASUREMENT_KEYS = ("measured_range_m", "measured_bearing_rad", "measured_elevation_rad")
ASSOCIATION_VERSION = "orchestration.structural_association-0.1"
SPATIAL_ASSOCIATION_VERSION = "spatial-structural-association-axial-v3"


@dataclass(frozen=True)
class ProjectedMeasurement:
    point_m: np.ndarray
    sigma_m: float


def project(obs: Observation, ctx: MissionContext) -> ProjectedMeasurement | None:
    """WORLD point of the observed surface through the ESTIMATED pose, or None if not projectable."""
    pose = obs.robot_pose_estimate
    sc = obs.sensor_context
    if pose is None or pose.frame_id != WORLD or not all(k in sc for k in MEASUREMENT_KEYS):
        return None
    try:
        sensor = ctx.sensor(obs.sensor_id)
    except KeyError:
        return None
    r, az, el = (float(sc[k]) for k in MEASUREMENT_KEYS)
    local = r * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    r_wb = quat_to_matrix(pose.orientation_wxyz)
    r_bs = quat_to_matrix(sensor.mount_pose.orientation_wxyz)
    origin = np.asarray(pose.position_m) + r_wb @ np.asarray(sensor.mount_pose.position_m)
    point = origin + r_wb @ r_bs @ local
    pose_sigma = pose.position_sigma_m()
    pose_sigma = 0.5 if pose_sigma is None else pose_sigma  # unknown covariance: conservative, never zero
    rs, as_ = float(sc.get("range_sigma_m", 0.0)), float(sc.get("angle_sigma_rad", 0.0))
    return ProjectedMeasurement(point, math.sqrt(pose_sigma**2 + rs**2 + (r * as_) ** 2))


def unique_axial_segment(
    point: np.ndarray,
    sigma_m: float,
    candidates: list[DesignComponent],
    sigma_multiplier: float,
) -> UUID | None:
    """Resolve a joint only when one bounded-survey axis contains the point.

    A Gaussian scale is used as an association confidence margin, not a hard
    registration bound or a coverage certificate. A declared endpoint bound
    widens each axial guard by a conservative axis-direction perturbation.
    Unbounded surveys, overlaps, and points near joints keep NO_MATCH.
    """
    if not candidates or any(
        c.shape != "CAPSULE"
        or c.component_type != "SEGMENT"
        or (c.survey_sigma_m and c.survey_endpoint_bound_m is None)
        for c in candidates
    ):
        return None
    inside: list[UUID] = []
    for comp in candidates:
        a, b = np.asarray(comp.p0_m), np.asarray(comp.p1_m)
        axis = b - a
        length = float(np.linalg.norm(axis))
        endpoint_bound = comp.survey_endpoint_bound_m or 0.0
        if length <= 2 * endpoint_bound:
            return None
        unit_axis = axis / length
        # Endpoint displacement <= bound implies axis displacement <= 2*bound.
        # The unit-axis change is bounded by 4*bound/(length-2*bound).
        direction_guard = 4 * endpoint_bound / (length - 2 * endpoint_bound)
        start_guard = sigma_multiplier * sigma_m + endpoint_bound + float(np.linalg.norm(point - a)) * direction_guard
        end_guard = sigma_multiplier * sigma_m + endpoint_bound + float(np.linalg.norm(point - b)) * direction_guard
        start = float((point - a) @ unit_axis)
        end = float((point - b) @ unit_axis)
        if start > start_guard and end < -end_guard:
            inside.append(comp.registry_id)
        elif start >= -start_guard and end <= end_guard:
            return None
    return inside[0] if len(inside) == 1 else None


class StructuralAssociator:
    def __init__(self, ctx: MissionContext, settings: AssociationSettings) -> None:
        self.ctx = ctx
        self.cfg = AssociationConfig(
            gate_distance_m=settings.gate_distance_m,
            gate_sigma_multiplier=settings.gate_sigma_multiplier,
            baseline_gate_distance_m=settings.gate_distance_m,
        )
        self.no_match = 0
        self.matched = 0

    def _cells(self, ev: Evidence, point: np.ndarray) -> list[BeliefCell]:
        cells = []
        for comp in self.ctx.design:
            # Spatial structural support is the inspected segment's capsule
            # surface, not a generic point on nearby supports or joint welds.
            # This filters by declared measurement type, never a truth ID.
            if ev.structural_support is not None and (
                comp.shape != "CAPSULE" or comp.component_type != "SEGMENT"
            ):
                continue
            q = comp.closest_surface_point(point)
            cells.append(
                BeliefCell(
                    belief_id=comp.registry_id,  # transient association cell keyed by registry identity
                    domain=Domain.TECHNICAL,
                    entity_type=comp.component_type,
                    lifecycle=Lifecycle.CONFIRMED,
                    revision=0,
                    timestamp=ev.timestamp,
                    state_embedding=(0.0,),
                    knowledge_status=KnowledgeStatus.UNKNOWN,
                    uncertainty=unknown_uncertainty(),
                    spatial_support=SpatialSupport(
                        frame_id=WORLD,
                        center_m=(float(q[0]), float(q[1]), float(q[2])),
                        position_sigma_m=comp.survey_sigma_m,
                    ),
                    provenance_root=comp.registry_id,
                    model_version=ASSOCIATION_VERSION,
                )
            )
        return cells

    def associate(self, obs: Observation, ev: Evidence) -> tuple[Evidence, AssociationDecision | None]:
        """Returns the evidence with its spatial support and (on a match) the registry hint attached."""
        proj = project(obs, self.ctx)
        if proj is None:
            self.no_match += 1
            return ev, None
        p = proj.point_m
        support = SpatialSupport(
            frame_id=WORLD, center_m=(float(p[0]), float(p[1]), float(p[2])), position_sigma_m=proj.sigma_m
        )
        located = ev.model_copy(update={"spatial_support": support})
        candidates = retrieve_candidates(located, self._cells(located, p), self.cfg, domain=Domain.TECHNICAL)
        decision = nearest_neighbour_decision(located, candidates, self.cfg)
        dists = sorted(
            float(np.linalg.norm(np.asarray(c.spatial_support.center_m) - p))
            for c in candidates
            if c.spatial_support is not None
        )
        separated = len(dists) < 2 or dists[1] - dists[0] >= proj.sigma_m
        if decision.belief_id is not None and (decision.margin < self.cfg.accept_margin or not separated):
            # two design surfaces explain the measurement within its own uncertainty (e.g. at a joint)
            decision = AssociationDecision(
                ev.evidence_id,
                None,
                decision.probability,
                decision.margin,
                decision.candidate_ids,
                "ambiguous",
            )
        if ev.structural_support is not None and len(decision.candidate_ids) > 1 and decision.method != "registry_identity":
            axial_id = unique_axial_segment(
                p,
                proj.sigma_m,
                [self.ctx.component(cid) for cid in decision.candidate_ids],
                self.cfg.gate_sigma_multiplier,
            )
            if axial_id is not None:
                decision = AssociationDecision(
                    ev.evidence_id,
                    axial_id,
                    decision.probability,
                    1.0,
                    decision.candidate_ids,
                    "unique_axial_segment",
                )
            else:
                decision = AssociationDecision(
                    ev.evidence_id,
                    None,
                    decision.probability,
                    decision.margin,
                    decision.candidate_ids,
                    "axial_not_unique",
                )
        if decision.belief_id is None:
            self.no_match += 1
            return located, decision
        self.matched += 1
        hint = EntityCandidate(
            registry_entity_id=decision.belief_id, score=float(np.clip(decision.probability, 0, 1))
        )
        return located.model_copy(update={"entity_candidates": (hint,)}), decision


def registry_of(ev: Evidence) -> UUID | None:
    for c in ev.entity_candidates:
        if c.registry_entity_id is not None:
            return c.registry_entity_id
    return None
