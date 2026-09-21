"""MissionRuntime: the supervised deployment-side mission loop (gates I1-I7, flagship I4). DEPLOYMENT PLANE.

It sees only the RobotHardwareInterface (plus declared payload sensors), the mission context and config.
One ``tick`` = sense -> encode/associate -> Model2S/2T/2E -> Belief Bus -> (cadenced) EGDC -> MCBR ->
NavigationStack -> safety -> CommandGateway, with BAAC stepping a constrained link to the shore receiver.
The simulation driver (truth side) advances physics between ticks; this class never does.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

from conrad.orchestration.association import StructuralAssociator
from conrad.orchestration.belief_bus import BeliefBus
from conrad.orchestration.children import build_children
from conrad.orchestration.comms import ShoreLink
from conrad.orchestration.deliberation import Deliberation
from conrad.orchestration.executive import MissionExecutive
from conrad.orchestration.mission_config import MissionRuntimeConfig
from conrad.orchestration.mission_context import MissionContext
from conrad.orchestration.mission_predictive import mission_predictive_provider
from conrad.orchestration.multidomain import SensingConditionsGate, sensing_requirements
from conrad.orchestration.perception import Perception
from conrad.orchestration.routing import DecisionRouting
from conrad.orchestration.services import ModuleRunner, RuntimeServices
from conrad.persistence.object_store import ObjectStore
from conrad.persistence.repository import Repository
from conrad.robotics.hardware.config import load_robot_config
from conrad.robotics.hardware.interface import RobotHardwareInterface
from conrad.robotics.navigation import NavigationStack, NavigationStackConfig
from conrad.robotics.safety.monitors import MissionBoundary, SafetyConfig
from conrad.robotics.trajectory.generator import TrajectoryConfig
from conrad.runtime.command_gateway import CommandGateway
from conrad.runtime.event_log import EventLog
from conrad.runtime.health import HealthRegistry
from conrad.runtime.supervisor import RuntimeState, RuntimeSupervisor
from conrad.schemas.belief import Availability, BeliefMessage
from conrad.schemas.events import EventType
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Observation
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.world import Domain
from conrad.settings import ConradSettings, ExecutionLane

MODULE = "conrad.orchestration.mission"
MODULES = ("sensing", "model2s", "model2t", "model2e", "navigation", "egdc", "mcbr", "baac")
CRITICAL = ("navigation",)


class PayloadCapabilityError(RuntimeError):
    pass


class MissionRuntime:
    def __init__(
        self,
        hardware: RobotHardwareInterface,
        context: MissionContext,
        repo: Repository,
        store: ObjectStore,
        log: EventLog,
        settings: ConradSettings,
        ids: IdFactory,
        config: MissionRuntimeConfig,
        seed: int,
        operator_inbox: Path | None = None,
    ) -> None:
        self.hw, self.ctx, self.cfg, self.settings = hardware, context, config, settings
        self.operator_inbox = operator_inbox
        self._inbox_seen = 0
        run_id, mission_id = log.ctx.run_id, log.ctx.mission_id
        self.health = HealthRegistry(hardware.now_ns)
        self.s = RuntimeServices(ids, log, repo, store, self.health, run_id, mission_id, hardware.now_ns)
        for name in MODULES:
            freshness = (
                2.0 if name == "navigation" else 4 * max(config.decision_period_s, config.comms_period_s, 2.0)
            )
            self.health.register(name, "mission-0.1", freshness, name in CRITICAL)
        faults = config.module_faults
        self.runner = ModuleRunner(
            self.s, {"model2e": faults.model2e_crash_at_s, "model2s": faults.model2s_crash_at_s}
        )
        self.supervisor = RuntimeSupervisor(settings, self.health, log.emit, run_id, self._on_safe_hold)
        self.robot_config = load_robot_config(settings.runtime.robot_config)
        caps = hardware.capabilities()
        self.payload = (
            getattr(hardware, "get_payload_observations", None) if caps.environmental_sensors else None
        )
        if caps.environmental_sensors and self.payload is None:
            raise PayloadCapabilityError("payload sensors declared but get_payload_observations is missing")
        clock = hardware.clock_domain()
        eco = config.model2e_enabled and "environmental_probe" in caps.environmental_sensors
        kids = build_children(context, config, ids, store, repo, run_id, clock, hardware.now_ns(), eco)
        self.m2s, self.m2t, self.m2e, encoder = kids.m2s, kids.m2t, kids.m2e, kids.encoder_e
        self.bus = BeliefBus(ids.child("bus"))
        self.bus.attach_child(Domain.SPATIAL, self.m2s.receive_context)
        self.bus.attach_child(Domain.TECHNICAL, self._technical_context)
        if self.m2e is not None:
            self.bus.attach_child(Domain.ECOLOGICAL, self.m2e.receive_context)
        lo, hi = context.spec.boundary_min_m, context.spec.boundary_max_m
        boundary = None if lo is None or hi is None else MissionBoundary(min_xyz_m=lo, max_xyz_m=hi)
        nav_cfg = NavigationStackConfig.for_surface(
            context.water_surface_z_m,
            control_period_s=config.control_period_s,
            safety=SafetyConfig(boundary=boundary),
            trajectory=TrajectoryConfig(cruise_speed_fraction=config.cruise_speed_fraction),
        )
        # NAVIGATION MAP. V3 planned over the SURVEYED REGISTRY DESIGN alone, so the global planner routed
        # straight through structure that is not in the registry even after Model2S had observed it, and the
        # approach to an inspection view ran into it. With the V4 switch on, both planners also read Model2S
        # OBSERVED occupancy (belief plane; the twin is never consulted). UNKNOWN space stays traversable.
        self.stack = NavigationStack(
            hardware,
            self.robot_config,
            ids.child("navigation"),
            mission_id,
            run_id,
            context.launch_pose,
            config=nav_cfg,
            is_free=self._nav_is_free,
            local_distance=self._nav_distance,
        )
        self.gateway = CommandGateway(
            hardware,
            self.robot_config,
            settings.runtime,
            settings.run.lane,
            mission_id,
            run_id,
            emit=log.emit,
            state_age_s=self.stack.state_age_s,
        )
        self.executive = MissionExecutive(self.s, context, config, self.stack, self.gateway)
        self.perception = Perception(
            self.s,
            self.runner,
            context,
            self.m2s,
            self.m2t,
            self.m2e,
            encoder,
            StructuralAssociator(context, config.association),
            self._position_fix,
            settings.runtime.late_evidence_policy,
            clock,
        )
        self.deliberation = Deliberation(self.s, context, config, self.bus, self.m2s)
        # MCBR belief-side predictive model (I4 repair; frozen with the production planner). Same request for
        # every planner, so baselines see identical feasible sets.
        self.deliberation.predictive_provider = mission_predictive_provider(
            self.m2t,
            self.m2s,
            self.deliberation.boresight_sensor,
            config.mcbr_unknown_block_probability,
            track=lambda: self.estimated_track,
        )
        self.shore = ShoreLink(self.s, config, seed, context.critical_component_ids)
        self.routing = DecisionRouting(
            self.s,
            context,
            config,
            self.bus,
            self.deliberation,
            self.executive,
            self.shore,
            self.supervisor,
            hardware,
            self.runner,
        )
        self.sensing_gate: SensingConditionsGate | None = None
        if config.multidomain.enabled and self.m2e is not None:  # gate I6 (off by default)
            self._enable_multidomain()
        self._next_decision_s = config.decision_period_s
        self._next_comms_s = 0.0
        self.estimated_track: list[list[float]] = []
        self.published = 0

    def _enable_multidomain(self) -> None:
        """Gate I6: 2E imaging-conditions requirements join EGDC; the sensing-conditions gate guards MCBR."""
        m2e, md = self.m2e, self.cfg.multidomain
        assert m2e is not None
        belief = m2e.field_ids[md.turbidity_field]
        extra = sensing_requirements(
            self.deliberation.requirements,
            self.ctx.critical_component_ids,
            belief,
            md,
            self.s.ids.child("multidomain"),
        )
        self.deliberation.requirements = (*self.deliberation.requirements, *extra)
        cc = m2e.cfg.coupling
        self.sensing_gate = SensingConditionsGate(
            self.s, self.bus, md, cc.beam_attenuation_clear_per_m, cc.beam_attenuation_per_m_per_ntu
        )
        self.routing.sensing_gate = self.sensing_gate.check

    def _technical_context(self, messages: Sequence[BeliefMessage]) -> None:
        """2T takes 2S coverage only for its own registry components (not every map block) plus 2E context."""
        keep = [m for m in messages if m.domain is Domain.ECOLOGICAL or m.world_entity_id is not None]
        if keep:
            self.runner.call("model2t", lambda: self.m2t.receive_context(keep))

    def _ecological_context(self, now: TimeStamp) -> None:
        """2E observability context views existing 2E heads (no new revision): delivered to 2T as context."""
        m2e = self.m2e
        assert m2e is not None
        msgs = list(self.runner.call("model2e", lambda: m2e.observability_context(now)) or [])
        if msgs:
            self._technical_context(msgs)
            self.s.emit(
                EventType.BELIEF_PUBLISHED,
                "conrad.orchestration.belief_bus",
                msgs[0].message_id,
                {
                    "context_only": True,
                    "source_domain": "ECOLOGICAL",
                    "target_domain": "TECHNICAL",
                    "messages": len(msgs),
                },
            )

    # ------------------------------------------------------------------ navigation map (belief plane)
    def _nav_is_free(self, points: Any) -> Any:
        import numpy as np

        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        free = self.ctx.design_distance(p) > self.cfg.planner_inflation_m
        if self.cfg.view_execution.enabled and self.cfg.view_execution.belief_map_navigation:
            from conrad.domains.spatial.queries import UnknownPolicy

            free &= np.asarray(self.m2s.is_free(p, UnknownPolicy.PERMISSIVE), dtype=bool)
        return np.asarray(free, dtype=bool)

    def _nav_distance(self, points: Any) -> Any:
        """Distance to the nearest known obstacle for the local planner: surveyed design, plus, with the
        V4 switch on, the nearest OBSERVED occupied belief cell that is not part of that design."""
        import numpy as np

        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        d = np.asarray(self.ctx.design_distance(p), dtype=np.float64)
        if not (self.cfg.view_execution.enabled and self.cfg.view_execution.belief_map_navigation):
            return d
        from conrad.domains.spatial.queries import UnknownPolicy

        blocked = ~np.asarray(self.m2s.is_free(p, UnknownPolicy.PERMISSIVE), dtype=bool)
        if blocked.any():  # a believed obstacle is at most one voxel away from the queried point
            d[blocked] = np.minimum(d[blocked], float(self.m2s.cfg.grid.base_voxel_m) * 0.5)
        return d

    # ------------------------------------------------------------------ lifecycle
    def estimated_pose(self) -> Any:
        return self.stack.estimator.get_state().pose

    def start(self, hardware_lane: bool = False) -> None:
        checks = [
            ("robot_config_digest", self._check_digest),
            (
                "mission_context",
                lambda: None if self.ctx.design else (_ for _ in ()).throw(ValueError("empty design")),
            ),
        ]
        problems = self.supervisor.self_test(checks)
        if problems:
            raise RuntimeError("self-test failed: " + "; ".join(problems))
        self.supervisor.connect(lambda: None)
        self.supervisor.hardware_ready.update(adapter=True, safety_supervisor=True, command_gateway=True)
        self.supervisor.start(
            hardware=hardware_lane or self.settings.run.lane in (ExecutionLane.HIL, ExecutionLane.PHYSICAL)
        )
        for name in MODULES:
            self.health.heartbeat(name)
        now = self._now()
        self.s.emit(
            EventType.RUN_STARTED,
            MODULE,
            self.s.run_id,
            {
                "mission_id": str(self.s.mission_id),
                "planner": self.cfg.planner,
                "lane": self.settings.run.lane.value,
            },
        )
        self._publish(self.m2t.export_beliefs(), now)
        self.routing.phase = "TRANSIT"
        self.routing.started_ns = now.time_ns
        self.executive.transit_goal(now)

    def _check_digest(self) -> None:
        if self.hw.robot_config_digest() != self.robot_config.content_digest():
            raise ValueError("hardware RobotConfig digest differs from the runtime RobotConfig")

    def _now(self) -> TimeStamp:
        return TimeStamp(time_ns=self.hw.now_ns(), clock_domain=self.hw.clock_domain())

    def _position_fix(self, obs: Observation) -> bool:
        return bool(self.runner.call("navigation", lambda: self.stack.add_position_fix(obs), obs.trace_id))

    def _on_safe_hold(self, reason: str) -> None:
        self.routing.phase = "HOLDING"
        if self.runner.available("navigation"):
            self.executive.hold_goal(self._now(), "SAFE_HOLD")

    # ------------------------------------------------------------------ one control period
    def tick(self) -> None:
        now = self._now()
        t_s = now.time_ns / 1e9
        obs: list[Observation] = []
        if self.payload is not None:
            got = self.runner.call("sensing", self.payload)
            obs = list(got or [])
        msgs = self.perception.process(obs, now)
        self._publish(msgs, now)
        if (
            self.supervisor.state in (RuntimeState.RUNNING, RuntimeState.DEGRADED)
            and t_s + 1e-9 >= self._next_decision_s
        ):
            self._next_decision_s = t_s + self.cfg.decision_period_s
            if self.m2e is not None and self.runner.available("model2e"):
                self._ecological_context(now)
            self.runner.heartbeat_idle("mcbr")
            self.routing.cycle(now, self.executive.goal_finished(now.time_ns))
            self._poll_operator()
            self.executive.flush()
        if self.runner.available("navigation"):
            self.runner.call("navigation", self.executive.control_tick)
        if t_s + 1e-9 >= self._next_comms_s:
            self._next_comms_s = t_s + self.cfg.comms_period_s
            self.runner.call("baac", lambda: self.shore.step(t_s, self.cfg.comms_period_s))
        if int(t_s * 2) != int((t_s - self.cfg.control_period_s) * 2):
            p = self.estimated_pose().position_m
            self.estimated_track.append([round(t_s, 3), p[0], p[1], p[2]])
        self.supervisor.tick()

    def _publish(self, msgs: list[BeliefMessage], now: TimeStamp) -> None:
        if not msgs:
            return
        accepted = [m for m, r in zip(msgs, self.bus.publish_many(msgs), strict=True) if r.accepted]
        self.published += len(accepted)
        for domain in Domain:
            self.bus.report_availability(domain, self._availability(domain))
        self.bus.dispatch()
        self.s.emit(
            EventType.BELIEF_PUBLISHED,
            "conrad.orchestration.belief_bus",
            msgs[0].message_id,
            {
                "accepted": len(accepted),
                "offered": len(msgs),
                "beliefs": sorted({str(m.belief_id) for m in accepted if m.domain is not Domain.SPATIAL}),
            },
        )
        entity = [m for m in accepted if m.domain is not Domain.SPATIAL or m.world_entity_id is not None]
        self.runner.call("baac", lambda: self.shore.offer(entity, now))

    def _availability(self, domain: Domain) -> Availability:
        name = {Domain.SPATIAL: "model2s", Domain.TECHNICAL: "model2t", Domain.ECOLOGICAL: "model2e"}[domain]
        return Availability.UNAVAILABLE if not self.runner.available(name) else Availability.AVAILABLE

    def _poll_operator(self) -> None:
        if self.operator_inbox is None or not self.operator_inbox.exists():
            return
        lines = [ln for ln in self.operator_inbox.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for line in lines[self._inbox_seen :]:
            req = json.loads(line)
            if req.get("action") == "safe_hold":
                self.supervisor.operator(
                    "safe_hold", str(req.get("operator", "operator")), str(req.get("reason", ""))
                )
        self._inbox_seen = len(lines)

    def finish(self, reason: str = "mission duration reached") -> None:
        self.executive.close_views(self.hw.now_ns())
        self.executive.flush()
        self.s.flush_provenance()
        self.supervisor.stop(reason)


def registry_ids(ctx: MissionContext) -> set[UUID]:
    return {c.registry_id for c in ctx.design}
