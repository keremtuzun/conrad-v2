"""EGDC deliberation and MCBR active planning over the Belief Bus. DEPLOYMENT PLANE.

Requirements come from the mission context (asset-registry identities only). MCBR sees the belief world
only: Model2S ``is_free`` / ``predicted_visibility`` plus the surveyed design geometry of the registry, and
a navigation-cost estimate built from the same belief. Its ObservationPlan is adopted with a PLAN
provenance record that links plan -> decision, so a command traces back to the raw observations.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import numpy as np

from conrad.active import MCBRConfig, MCBRPlanner, PlanningRequest, SensorOption, make_planners
from conrad.active.planner import PlanResult
from conrad.active.predictive import PredictiveBelief
from conrad.active.production import PRODUCTION, production_planner
from conrad.decision.config import DecisionConfig
from conrad.decision.context import DecisionContext, DecisionSummary, MissionRequirement
from conrad.decision.egdc import EGDC, DecisionOutcome
from conrad.decision.query_engine import BeliefQueryEngine
from conrad.decision.route import ROUTE_OBSERVED_CLAIM
from conrad.domains.spatial.grid import OBSERVED
from conrad.domains.spatial.keys import index_to_center, region_indices
from conrad.domains.spatial.model import Model2S
from conrad.domains.spatial.queries import UnknownPolicy
from conrad.orchestration.belief_bus import BeliefBus
from conrad.orchestration.mission_config import MissionRuntimeConfig
from conrad.orchestration.mission_context import MissionContext
from conrad.orchestration.services import RuntimeServices
from conrad.schemas.belief import BeliefMessage, BeliefQuery, BeliefSnapshot
from conrad.schemas.comms import LinkState
from conrad.schemas.decision import (
    ActionType,
    InformationNeed,
    MissionPhase,
    MissionState,
    ObservationPlan,
    ResourceCost,
    ResourceState,
)
from conrad.schemas.events import EventType
from conrad.schemas.frames import Pose, SpatialSupport
from conrad.schemas.provenance import ProvenanceRecord, SourceType
from conrad.schemas.robot import RobotState, SystemHealth
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.world import Domain

MODULE = "conrad.orchestration.deliberation"
INSPECTED_TYPES = ("SEGMENT", "WELD")
ENERGY_PER_M_J = 40.0  # ENGINEERING_ESTIMATE of the planner's own travel-energy model (not measured)


@dataclass
class AdoptedPlan:
    plan: ObservationPlan
    adoption_record: UUID
    decision_id: UUID
    table: list[dict[str, Any]]


class Deliberation:
    def __init__(
        self,
        s: RuntimeServices,
        ctx: MissionContext,
        cfg: MissionRuntimeConfig,
        bus: BeliefBus,
        m2s: Model2S | None,
    ) -> None:
        self.s, self.ctx, self.cfg, self.bus, self.m2s = s, ctx, cfg, bus, m2s
        ids = s.ids.child("deliberation")
        self.ids = ids
        self.egdc = EGDC(s.ids.child("egdc"), DecisionConfig(**cfg.decision))
        mcbr_cfg = MCBRConfig(**cfg.mcbr)
        if cfg.planner == PRODUCTION:  # default: the frozen, validation-selected planner (configs/active)
            self.planner: MCBRPlanner = production_planner(s.ids.child("mcbr"), mcbr_cfg)
        else:
            planners = make_planners(s.ids.child("mcbr"), mcbr_cfg)
            if cfg.planner not in planners:
                raise KeyError(f"unknown planner {cfg.planner!r}; known {sorted([PRODUCTION, *planners])}")
            self.planner = planners[cfg.planner]
        self.query = BeliefQueryEngine(bus, s.ids.child("query"))
        self.requirements = self._requirements()
        self.history: list[DecisionSummary] = []
        self.decisions: list[DecisionOutcome] = []
        self.plans: list[AdoptedPlan] = []
        self.last_route: tuple[list[SpatialSupport], dict[str, list[str]]] = ([], {})
        # Optional belief-side predictive model of MCBR (I4 repair): beliefs of the need -> PredictiveBelief.
        # Set by MissionRuntime (conrad.orchestration.mission_predictive); None = no predictive model.
        self.predictive_provider: Callable[[Sequence[BeliefMessage]], PredictiveBelief | None] | None = None
        self.sensor = ctx.sensor(ctx.structural_sensor_ids[0])
        # MCBR candidate poses are SENSOR-boresight poses (+X looks at the target); the belief map is queried
        # with the same convention, and ``view_pose`` converts to a vehicle pose through the real mount yaw.
        self.boresight_sensor = self.sensor.model_copy(
            update={"mount_pose": Pose(frame_id=self.sensor.mount_pose.frame_id, position_m=(0.0, 0.0, 0.0))}
        )
        self.mount_yaw = _yaw(self.sensor.mount_pose.orientation_wxyz)
        p = self.sensor.parameters
        self.sensor_option = SensorOption(
            sensor_id=self.sensor.sensor_id,
            modality=self.sensor.modality,
            min_range_m=float(p.get("min_range_m", 0.3)) + 0.7,
            max_range_m=float(p["max_range_m"]),
            duration_s=cfg.inspection_dwell_s,
            measurement_quality=0.8,
        )

    def _requirements(self) -> tuple[MissionRequirement, ...]:
        out = []
        for comp in self.ctx.design:
            if comp.component_type not in INSPECTED_TYPES:
                continue
            critical = comp.registry_id in self.ctx.critical_component_ids
            out.append(
                MissionRequirement(
                    requirement_id=self.ids.new(),
                    description=f"condition of registry component {comp.registry_id}",
                    domain=Domain.TECHNICAL,
                    target_entity_ids=(comp.registry_id,),
                    region=comp.inspection_station(self.cfg.inspection_station_half_m, 0.2),
                    properties=("condition",),
                    consequence=self.cfg.critical_consequence if critical else self.cfg.routine_consequence,
                    context_domains=(Domain.SPATIAL,),
                )
            )
        return tuple(out)

    # ------------------------------------------------------------------ EGDC
    def decide(
        self,
        now: TimeStamp,
        robot: RobotState | None,
        resources: ResourceState | None,
        link: LinkState | None,
        health: SystemHealth,
        motion_permitted: bool,
        notes: dict[str, Any],
        route_messages: Sequence[BeliefMessage] = (),
    ) -> DecisionOutcome:
        trace = self.ids.new()
        snapshot = self.query.retrieve(self.requirements, now.time_ns)
        extra = [m for m in route_messages if m.belief_id not in {x.belief_id for x in snapshot.messages}]
        if extra:  # SPATIAL beliefs that occupy a planned-route leg (route_context) join the snapshot
            revisions = dict(snapshot.provenance.get("revisions", {}))
            revisions.update({str(m.belief_id): m.revision for m in extra})
            snapshot = snapshot.model_copy(
                update={
                    "messages": tuple(
                        sorted([*snapshot.messages, *extra], key=lambda m: (m.domain.value, m.belief_id.int))
                    ),
                    "provenance": {**snapshot.provenance, "revisions": revisions},
                }
            )
        ctx = DecisionContext(
            timestamp=now,
            trace_id=trace,
            mission=MissionState(
                mission_id=self.ctx.mission_id,
                phase=MissionPhase(notes.pop("phase", "TRANSIT")),
                timestamp=now,
                objectives_total=len(self.requirements),
                objectives_done=0,
                notes=notes,
            ),
            mission_spec=self.ctx.spec,
            requirements=self.requirements,
            snapshot=snapshot,
            robot_state=robot,
            resource_state=resources,
            link_state=link,
            system_health=health,
            previous_decisions=tuple(
                h for h in self.history if (now.time_ns - h.time_ns) / 1e9 <= self.cfg.decision_history_s
            ),
            answered_information_requests=self._answered_requests(),
            available_modalities=(self.sensor.modality,),
            motion_permitted=motion_permitted,
            operator_reachable=link is not None and link.status.value != "DOWN",
        )
        out = self.egdc.decide(ctx)
        self.s.repo.put_provenance(self.s.run_id, [out.provenance])
        rec = out.record
        chosen = rec.chosen
        self.history.append(
            DecisionSummary(
                decision_id=rec.decision_id,
                time_ns=now.time_ns,
                action_type=None if chosen is None else chosen.action_type,
                target_belief_ids=() if chosen is None else chosen.target_belief_ids,
                abstained=rec.abstained,
                target_revisions=_revisions(snapshot, () if chosen is None else chosen.target_belief_ids),
            )
        )
        self.decisions.append(out)
        self.s.emit(
            EventType.DECISION_MADE,
            MODULE,
            trace,
            {
                "decision_id": str(rec.decision_id),
                "action": None if chosen is None else chosen.action_type.value,
                "abstained": rec.abstained,
                "targets": [] if chosen is None else [str(b) for b in chosen.target_belief_ids],
                "route": None if out.routed is None or out.routed.target is None else out.routed.target.value,
                "snapshot_id": str(snapshot.snapshot_id),
                "provenance_id": str(out.provenance.record_id),
                "unsupported_claims": len(rec.unsupported_claim_ids),
                "rationale": rec.rationale[:200],
            },
        )
        return out

    def _answered_requests(self) -> dict[UUID, int]:
        """Per belief, the newest revision an information request the runtime CARRIED OUT saw.

        The decision-history window H_t (``decision_history_s``) is the right horizon for counting recent
        attempts, but not for the fact that an independent look was taken: that does not expire. EGDC reads
        this ledger through ``DecisionContext.request_answered``, and still re-checks the belief itself
        (properties OBSERVED, evidence present, no conflict) on every cycle.
        """
        out: dict[UUID, int] = {}
        for h in self.history:
            if h.action_type is not ActionType.REQUEST_INFORMATION or h.executed is False:
                continue
            for belief_id, revision in zip(h.target_belief_ids, h.target_revisions, strict=False):
                if revision > out.get(belief_id, -1):
                    out[belief_id] = revision
        return out

    def mark_not_executed(self, decision_id: UUID) -> None:
        """Routing did not carry this decision's action out (deferred or dropped). EGDC then does not count
        it as an attempt (``DecisionContext.attempts_on``), so a deferred information request cannot use up
        the attempt budget and tip the next decision into escalation."""
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i].decision_id == decision_id:
                self.history[i] = self.history[i].model_copy(update={"executed": False})
                return

    # ------------------------------------------------------------------ planned route
    def route_context(
        self, legs: Sequence[SpatialSupport], now: TimeStamp
    ) -> tuple[dict[str, Any], list[BeliefMessage]]:
        """Publish the planned route and Model2S's cell-level occupancy of each leg for Model 1.

        Returns the notes (``planned_route``: leg supports; ``route_leg_occupancy``: leg index -> the SPATIAL
        belief IDs whose OBSERVED, occupied cells lie inside that leg) and those belief messages, which are
        merged into the decision snapshot. Only belief-plane data is used: Model2S point states, the surveyed
        registry design and the charted seabed (known structure is not an obstacle). Model2S publishes 2.8 m
        blocks, far coarser than a route corridor, so the cell-level check has to be made here.
        """
        notes: dict[str, Any] = {"planned_route": [leg.model_dump(mode="json") for leg in legs]}
        if self.m2s is None or not legs:
            return notes, []
        rc = self.cfg.route
        res = float(self.m2s.cfg.grid.base_voxel_m)
        occupancy: dict[str, list[str]] = {}
        cells: dict[str, int] = {}
        found: dict[UUID, BeliefMessage] = {}
        for i, leg in enumerate(legs):
            idx = region_indices(np.asarray(leg.center_m), np.asarray(leg.half_extent_m), res)
            pts = index_to_center(idx, res)
            if len(pts) == 0:
                continue
            st = self.m2s.occupancy_state(pts)
            occ = (st.status == OBSERVED) & (st.probability >= rc.occupied_probability)
            occ &= self.ctx.design_distance(pts) > rc.design_clearance_m
            n = int(occ.sum())
            if n < rc.min_occupied_cells:
                continue
            hit = pts[occ]
            lo, hi = hit.min(axis=0) - res / 2, hit.max(axis=0) + res / 2
            box = SpatialSupport(
                frame_id=leg.frame_id,
                center_m=tuple(float(v) for v in (lo + hi) / 2),
                half_extent_m=tuple(float(v) for v in (hi - lo) / 2),
            )
            reply = self.bus.query(BeliefQuery(domain=Domain.SPATIAL, region=box), now.time_ns)
            ids = [m for m in reply.messages if any(c.name == ROUTE_OBSERVED_CLAIM for c in m.state_summary)]
            if ids:
                occupancy[str(i)] = sorted(str(m.belief_id) for m in ids)
                cells[str(i)] = n
                found.update({m.belief_id: m for m in ids})
        notes["route_leg_occupancy"] = occupancy
        notes["route_occupied_cells"] = cells
        self.last_route = (list(legs), occupancy)
        return notes, list(found.values())

    # ------------------------------------------------------------------ MCBR
    def plan(
        self,
        need: InformationNeed,
        decision: DecisionOutcome,
        beliefs: Sequence[BeliefMessage],
        robot_pose: Pose,
        now: TimeStamp,
        prior_views: Sequence[Any],
    ) -> AdoptedPlan:
        req = PlanningRequest(
            need=need,
            beliefs=list(beliefs),
            robot_pose=robot_pose,
            sensors=(self.sensor_option,),
            is_free=self.is_free,
            predicted_visibility=self.predicted_visibility,
            navigation_cost=self.navigation_cost,
            now=now,
            prior_views=tuple(prior_views),
            rng=np.random.default_rng(now.time_ns % (2**32)),
            predictive=None if self.predictive_provider is None else self.predictive_provider(beliefs),
        )
        result: PlanResult = self.planner.plan(req)
        adoption = ProvenanceRecord(
            record_id=self.ids.new(),
            source_type=SourceType.PLAN,
            source_ids=(result.plan.plan_id, need.need_id, decision.record.decision_id),
            operation="orchestration.adopt_plan",
            module=MODULE,
            model_version=self.planner.name,
            timestamp=now,
            parent_records=(result.provenance.record_id, decision.provenance.record_id),
            subject_id=result.plan.plan_id,
        )
        self.s.repo.put_provenance(self.s.run_id, [result.provenance, adoption])
        plan = result.plan
        act = plan.primary_action
        self.s.emit(
            EventType.PLAN_PROPOSED,
            MODULE,
            plan.trace_id,
            {
                "plan_id": str(plan.plan_id),
                "need_id": str(need.need_id),
                "decision_id": str(decision.record.decision_id),
                "planner": self.planner.name,
                "status": plan.status.value,
                "candidates": len(result.table),
                "feasible": sum(1 for r in result.table if r.get("feasible")),
                "rejected": len(plan.rejected),
                "view_position_m": None if act is None else list(act.pose.position_m),
                "predicted_visibility": None if act is None else act.predicted_visibility,
                "provenance_id": str(adoption.record_id),
            },
        )
        adopted = AdoptedPlan(plan, adoption.record_id, decision.record.decision_id, result.table)
        self.plans.append(adopted)
        return adopted

    # ------------------------------------------------------------------ belief-derived callables
    def is_free(self, pts: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts, dtype=np.float64))
        ok = self.ctx.design_distance(p) >= self.cfg.candidate_clearance_m
        ok &= (p[:, 2] - self.ctx.seabed_z_m) >= self.cfg.min_altitude_m
        if self.m2s is not None:
            ok &= np.asarray(self.m2s.is_free(p, UnknownPolicy.PERMISSIVE), dtype=bool)
        return np.asarray(ok, dtype=bool)

    def _surface(self, region: SpatialSupport) -> tuple[np.ndarray, np.ndarray]:
        """Registry design-surface samples (points, outward normals) of capsules inside ``region``."""
        c, h = np.asarray(region.center_m), np.asarray(region.half_extent_m) + 1e-6
        pts, nrm = [], []
        for comp in self.ctx.design:
            if comp.shape != "CAPSULE":
                continue
            a, b = np.asarray(comp.p0_m), np.asarray(comp.p1_m)
            d = (b - a) / max(float(np.linalg.norm(b - a)), 1e-9)
            u = np.cross(d, [0.0, 0.0, 1.0])
            u = u / max(float(np.linalg.norm(u)), 1e-9)
            v = np.cross(d, u)
            for t in np.linspace(0.0, 1.0, 9):
                for ang in np.linspace(0.0, 2 * math.pi, 12, endpoint=False):
                    n = math.cos(ang) * u + math.sin(ang) * v
                    p = a + t * (b - a) + comp.radius_m * n
                    if np.all(np.abs(p - c) <= h):
                        pts.append(p)
                        nrm.append(n)
        return np.asarray(pts).reshape(-1, 3), np.asarray(nrm).reshape(-1, 3)

    def predicted_visibility(self, pose: Pose, region: SpatialSupport) -> float:
        """Belief-side visibility of the component's still-UNKNOWN design surface from ``pose``.

        Design-surface samples (registry geometry) whose water cell just outside the surface is still UNKNOWN
        in the 2S map are the unobserved surface; a sample counts, weighted by the cosine of its incidence
        angle, if it faces the sensor and the belief map predicts a clear ray to it. Falls back to the region's UNKNOWN cells when no design surface is known.
        """
        if self.m2s is None:
            return 0.0
        pts, nrm = self._surface(region)
        if len(pts):
            probe = pts + 0.15 * nrm
            unknown = np.asarray([s.value == "UNKNOWN" for s in self.m2s.occupancy_status(probe)], dtype=bool)
            sel = np.nonzero(unknown)[0] if unknown.any() else np.arange(len(pts))
            origin = np.asarray(pose.position_m)
            seen = 0.0
            for i in sel:
                ray = origin - pts[i]
                cos_incidence = float(nrm[i] @ ray) / max(float(np.linalg.norm(ray)), 1e-9)
                if cos_incidence <= 0.0:
                    continue
                cell = SpatialSupport(frame_id=region.frame_id, center_m=tuple(float(x) for x in probe[i]))
                seen += (
                    cos_incidence
                    * self.m2s.predicted_visibility(
                        pose, cell, self.boresight_sensor, self.cfg.mcbr_unknown_block_probability
                    ).mean
                )  # grazing views reveal little of a surface: Lambertian incidence weighting
            return float(seen / len(sel))
        vis = self.m2s.predicted_visibility(
            pose, region, self.boresight_sensor, self.cfg.mcbr_unknown_block_probability
        )
        if len(vis.points_m) == 0:
            return 0.0
        unknown = np.asarray(
            [s.value == "UNKNOWN" for s in self.m2s.occupancy_status(vis.points_m)], dtype=bool
        )
        prob = np.asarray(vis.probability, dtype=np.float64)
        return float(prob[unknown].mean()) if unknown.any() else float(prob.mean())

    def navigation_cost(self, a: Pose, b: Pose) -> ResourceCost | None:
        p, q = np.asarray(a.position_m), np.asarray(b.position_m)
        if not bool(self.is_free(q[None, :])[0]):
            return None
        dist = float(np.linalg.norm(q - p))
        line = p + np.linspace(0.0, 1.0, 24)[:, None] * (q - p)
        clearance = float(np.min(self.ctx.design_distance(line)))
        if clearance < self.cfg.planner_inflation_m:
            dist *= 1.6  # the believed route must go around the surveyed structure
        risk = 0.05 if clearance >= self.cfg.planner_inflation_m else 0.2
        return ResourceCost(
            time_s=dist / self.cfg.cruise_speed_mps, energy_j=ENERGY_PER_M_J * dist, risk=risk, travel_m=dist
        )


def _revisions(snapshot: BeliefSnapshot, belief_ids: Sequence[UUID]) -> tuple[int, ...]:
    """Revision of each target belief as the decision saw it (-1 = not in the snapshot)."""
    seen = {m.belief_id: m.revision for m in snapshot.messages}
    return tuple(seen.get(b, -1) for b in belief_ids)


def _yaw(q: tuple[float, float, float, float]) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def view_pose(plan: ObservationPlan, mount_yaw: float = 0.0) -> Pose:
    """Vehicle pose that points the (level, yaw-mounted) payload along the candidate's boresight."""
    assert plan.primary_action is not None
    pose = plan.primary_action.pose
    yaw = _yaw(pose.orientation_wxyz) - mount_yaw
    return Pose(
        frame_id=pose.frame_id,
        position_m=pose.position_m,
        orientation_wxyz=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
    )
