"""Phase 1 fake full system (Gate I0). TEST_FIXTURE only: establishes wiring, not intelligence.

world -> sensor -> Observation -> ECMER -> Evidence -> Model2 -> Belief -> Bus -> Model1 ->
InformationNeed -> MCBR -> ObservationPlan -> Navigation -> Control -> Gateway -> Robot ->
new observation -> belief update -> BAAC -> receiver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import numpy as np

from conrad.persistence.repository import BeliefUpdate, Repository
from conrad.robotics.hardware.config import load_robot_config
from conrad.robotics.hardware.interface import RobotHardwareInterface
from conrad.runtime.command_gateway import ALLOCATOR_PRODUCER, CommandGateway
from conrad.runtime.event_log import EventLog, RunContext
from conrad.schemas.belief import (
    BeliefCell,
    BeliefMessage,
    BeliefRevision,
    KnowledgeStatus,
    Lifecycle,
    PropertyClaim,
    TechnicalPayload,
    UpdateKind,
)
from conrad.schemas.comms import (
    DeltaType,
    Fidelity,
    FidelityOption,
    InformationType,
    InformationUnit,
    SemanticDelta,
    Transmission,
)
from conrad.schemas.decision import (
    ActionProposal,
    ActionType,
    ClaimType,
    ConstraintDecision,
    DecisionClaim,
    DecisionRecord,
    GroundingStatus,
    InformationNeed,
    NavigationGoal,
    ObservationAction,
    ObservationPlan,
    PlanStatus,
    QuestionType,
    ResourceCost,
    UncertaintyType,
)
from conrad.schemas.events import EventType
from conrad.schemas.frames import WORLD, Pose, SpatialSupport
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Evidence, Modality, Observation
from conrad.schemas.provenance import ProvenanceRecord, SourceType
from conrad.schemas.robot import (
    AllocatedCommand,
    BatteryState,
    CommandAck,
    DepthSample,
    HealthLevel,
    ImuSample,
    RobotCapabilities,
    RobotConfig,
    SafetyAuthorization,
    SystemHealth,
    ThrusterState,
    WrenchCommand,
)
from conrad.schemas.timebase import TimeStamp, stamp
from conrad.schemas.uncertainty import Uncertainty
from conrad.schemas.world import Domain
from conrad.settings import CommandMode, ExecutionLane, RuntimeSettings

VERSION = "fake-0"
DT = 0.1


class SimClock:
    def __init__(self) -> None:
        self.t = 0.0

    def ns(self) -> int:
        return round(self.t * 1e9)

    def stamp(self, seq: int = 0) -> TimeStamp:
        return stamp(self.t, "SIM", seq, self.t)


class FakeTwin:
    """Truth: one pipeline segment with a hidden severity visible only from its +Y side."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.position = np.array([5.0, 0.0, -10.0])
        self.hidden_severity = float(rng.uniform(0.55, 0.8))


class FakeRobot(RobotHardwareInterface):
    adapter_name = "fake"

    def __init__(self, config: RobotConfig, clock: SimClock) -> None:
        self._config, self._clock = config, clock
        self.true_position = np.array([0.0, -3.0, -10.0])
        self._last: dict[str, float] = {t.thruster_id: 0.0 for t in config.thrusters}
        self.received: list[AllocatedCommand] = []

    def capabilities(self) -> RobotCapabilities:
        return RobotCapabilities(capability_version="fake-1", cameras=("camera",), imu=True, depth=True, thruster_count=len(self._last))

    def robot_config_digest(self) -> str:
        return self._config.content_digest()

    def clock_domain(self) -> str:
        return "SIM"

    def now_ns(self) -> int:
        return self._clock.ns()

    def get_imu(self) -> ImuSample | None:
        return ImuSample(timestamp=self._clock.stamp(), frame_id="ROBOT", linear_acceleration_mps2=(0, 0, 0), angular_velocity_rps=(0, 0, 0))

    def get_depth(self) -> DepthSample | None:
        return DepthSample(timestamp=self._clock.stamp(), frame_id="ROBOT", depth_m=float(-self.true_position[2]))

    def get_camera(self) -> Observation | None:
        return None

    def get_sonar(self) -> Observation | None:
        return None

    def get_thruster_state(self) -> tuple[ThrusterState, ...]:
        return tuple(ThrusterState(thruster_id=k, command=v, estimated_thrust_n=40.0 * v) for k, v in self._last.items())

    def get_power_state(self) -> BatteryState | None:
        return BatteryState(timestamp=self._clock.stamp(), remaining_fraction=0.9)

    def get_health(self) -> SystemHealth:
        return SystemHealth(timestamp=self._clock.stamp(), overall=HealthLevel.OK)

    def send(self, command: AllocatedCommand) -> CommandAck:
        self.received.append(command)
        self._last = dict(command.thruster_commands)
        c = command.thruster_commands
        self.true_position = self.true_position + DT * np.array([c["H1"], c["H2"], c["V1"]])
        return CommandAck(command_id=command.command_id, accepted=True, ack_time_ns=self._clock.ns(), adapter=self.adapter_name)


@dataclass
class FakeRunResult:
    events: EventLog
    decisions: list[DecisionRecord] = field(default_factory=list)
    plans: list[ObservationPlan] = field(default_factory=list)
    messages: list[BeliefMessage] = field(default_factory=list)
    transmissions: list[Transmission] = field(default_factory=list)
    commands: list[AllocatedCommand] = field(default_factory=list)
    belief_id: UUID | None = None
    final_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    true_severity: float = 0.0


class FakeFullSystem:
    def __init__(self, repo: Repository, seed: int, log_path: Path | None = None) -> None:
        self.repo = repo
        self.ids = IdFactory(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.clock = SimClock()
        self.config = load_robot_config("configs/robot/sim_reference.yaml")
        self.run_id, self.mission_id, self.scenario_id = self.ids.new(), self.ids.new(), self.ids.new()
        self.ctx = RunContext(self.run_id, self.mission_id, self.scenario_id, self.ids, self.clock.ns, "SIM", VERSION)
        self.log = EventLog(self.ctx, log_path)
        self.twin = FakeTwin(self.rng)
        self.robot = FakeRobot(self.config, self.clock)
        self.gateway = CommandGateway(
            self.robot, self.config, RuntimeSettings(command_mode=CommandMode.SIMULATED), ExecutionLane.SIMULATION,
            self.mission_id, self.run_id, emit=self.log.emit,
        )
        self.sensor_id = self.ids.new()
        self.result = FakeRunResult(events=self.log, true_severity=self.twin.hidden_severity)
        self._belief: BeliefCell | None = None
        self._receiver_revision: int | None = None
        self._seq = 0

    # ------------------------------------------------------------------ observation plane
    def estimated_pose(self) -> Pose:
        est = self.robot.true_position + self.rng.normal(0, 0.02, 3)  # estimator stand-in: noisy, never exact truth
        cov = [0.0] * 36
        cov[0] = cov[7] = cov[14] = 0.02**2
        return Pose(frame_id=WORLD, position_m=(float(est[0]), float(est[1]), float(est[2])), covariance_6x6=tuple(cov))

    def fake_sensor(self, trace: UUID) -> Observation:
        rel = self.robot.true_position - self.twin.position
        sees_hidden_side = rel[1] > 1.0 and np.linalg.norm(rel) < 4.0
        visible = 1.0 if sees_hidden_side else 0.0
        reading = self.twin.hidden_severity + float(self.rng.normal(0, 0.03)) if sees_hidden_side else float(self.rng.normal(0.2, 0.2))
        self._seq += 1
        obs = Observation(
            observation_id=self.ids.new(), mission_id=self.mission_id, run_id=self.run_id, trace_id=trace,
            sensor_id=self.sensor_id, modality=Modality.STRUCTURED, timestamp=self.clock.stamp(self._seq),
            sensor_frame="SENSOR_CAMERA", robot_pose_estimate=self.estimated_pose(),
            inline_values=(float(np.clip(reading, 0, 1)), visible), inline_units="unitless,coverage",
        )
        self.repo.put_observation(obs)
        self.log.emit(EventType.OBSERVATION_RECEIVED, "fake.sensor", trace, {"observation_id": str(obs.observation_id)}, measurement_time_ns=obs.timestamp.time_ns)
        return obs

    def fake_ecmer(self, obs: Observation) -> tuple[Evidence, ProvenanceRecord]:
        assert obs.inline_values is not None and obs.robot_pose_estimate is not None
        value, coverage = obs.inline_values
        eid, pid = self.ids.new(), self.ids.new()
        prov = ProvenanceRecord(
            record_id=pid, source_type=SourceType.DIRECT_OBSERVATION, source_ids=(obs.observation_id,), operation="fake_encode",
            module="fake.ecmer", model_version=VERSION, timestamp=obs.timestamp, subject_id=eid,
        )
        ev = Evidence(
            evidence_id=eid, source_observation_id=obs.observation_id, mission_id=self.mission_id, run_id=self.run_id,
            trace_id=obs.trace_id, modality=obs.modality, timestamp=obs.timestamp, created_time_ns=max(self.clock.ns(), obs.timestamp.time_ns),
            embedding=(value, coverage, math.sin(value), math.cos(value)), reliability=0.3 + 0.65 * coverage,
            aleatoric_uncertainty=0.03 if coverage else 0.2,
            spatial_support=SpatialSupport(frame_id=WORLD, center_m=(5.0, 0.0, -10.0), half_extent_m=(1, 1, 1), position_sigma_m=obs.robot_pose_estimate.position_sigma_m()),
            measurements={"severity": value, "coverage": coverage}, measurement_units={"severity": "unitless", "coverage": "unitless"},
            independence_group=f"obs:{obs.observation_id}", provenance_id=pid, encoder_version=VERSION,
        )
        self.repo.put_evidence(ev)
        self.log.emit(EventType.EVIDENCE_CREATED, "fake.ecmer", obs.trace_id, {"evidence_id": str(eid), "observation_id": str(obs.observation_id)}, measurement_time_ns=obs.timestamp.time_ns, causation=(obs.observation_id,))
        return ev, prov

    # ------------------------------------------------------------------ belief plane
    def fake_model2(self, ev: Evidence, eprov: ProvenanceRecord) -> BeliefMessage:
        prior = self._belief
        coverage = ev.measurements["coverage"]
        n = 0 if prior is None else prior.independent_observation_count
        old = prior.claim("severity") if prior else None
        old_val = float(old.value) if old and old.value is not None else None
        if coverage > 0.5:
            weight = ev.reliability / (ev.reliability + max(n, 1) * 0.5) if old_val is not None else 1.0
            value: float | None = ev.measurements["severity"] if old_val is None else old_val + weight * (ev.measurements["severity"] - old_val)
            status = KnowledgeStatus.OBSERVED
            best_cov = 1.0
        else:  # the relevant surface was not seen: keep UNKNOWN instead of inventing a value
            value, status = old_val, (old.status if old else KnowledgeStatus.UNKNOWN)
            best_cov = 1.0 if status is KnowledgeStatus.OBSERVED else 0.0
        unc = Uncertainty(
            aleatoric=ev.aleatoric_uncertainty, epistemic=1.0 / (1.0 + n),
            contradiction=0.0 if old_val is None or coverage < 0.5 else min(1.0, abs(ev.measurements["severity"] - old_val)),
            observational=1.0 - best_cov,
        )
        bid = self.ids.new() if prior is None else prior.belief_id
        rev_no = 0 if prior is None else prior.revision + 1
        pid = self.ids.new()
        parents = (eprov.record_id,) if prior is None else (eprov.record_id, prior.provenance_root)
        bprov = ProvenanceRecord(
            record_id=pid, source_type=SourceType.DIRECT_OBSERVATION, source_ids=(ev.evidence_id,), operation="fake_buo",
            module="fake.model2", model_version=VERSION, timestamp=ev.timestamp, parent_records=parents, subject_id=bid,
        )
        claim = PropertyClaim(name="severity", value=value if status is not KnowledgeStatus.UNKNOWN else None, units="unitless", status=status, uncertainty=unc, provenance_id=None if status is KnowledgeStatus.UNKNOWN else pid)
        cell = BeliefCell(
            belief_id=bid, domain=Domain.TECHNICAL, entity_type="pipeline_segment", lifecycle=Lifecycle.CANDIDATE if prior is None else Lifecycle.CONFIRMED,
            revision=rev_no, timestamp=ev.timestamp, state_embedding=(unc.observational, unc.epistemic, float(n), 0.0), claims=(claim,),
            knowledge_status=status, uncertainty=unc, spatial_support=ev.spatial_support, provenance_root=pid, model_version=VERSION,
        )
        revision = BeliefRevision(
            belief_id=bid, revision=rev_no, predecessor_revision=None if prior is None else prior.revision, measurement_time_ns=ev.timestamp.time_ns,
            consumed_evidence_ids=(ev.evidence_id,), provenance_root=pid, update_kind=UpdateKind.DIRECT, cell=cell,
        )
        res = self.repo.commit_update(BeliefUpdate(message_id=self.ids.new(), producer_version=VERSION, run_id=self.run_id, revision=revision, provenance=[eprov, bprov]))
        self._belief = cell.model_copy(update={"independent_observation_count": res.independent_observation_count})
        self.result.belief_id = bid
        self.log.emit(EventType.BELIEF_COMMITTED, "fake.model2", ev.trace_id, {"belief_id": str(bid), "revision": rev_no, "evidence_id": str(ev.evidence_id), "status": status.value}, measurement_time_ns=ev.timestamp.time_ns, causation=(ev.evidence_id,))
        msg = BeliefMessage(
            message_id=self.ids.new(), belief_id=bid, revision=rev_no, domain=Domain.TECHNICAL, timestamp=ev.timestamp, state_summary=(claim,),
            state_embedding=cell.state_embedding, knowledge_status=status, uncertainty=unc, evidence_support=(ev.evidence_id,), provenance_refs=(pid,),
            spatial_support=cell.spatial_support, lifecycle=cell.lifecycle, model_version=VERSION,
            technical=TechnicalPayload(severity=claim.value if isinstance(claim.value, float) else None, direct_support=best_cov),
        )
        self.result.messages.append(msg)  # FakeBeliefBus
        self.log.emit(EventType.BELIEF_PUBLISHED, "fake.bus", ev.trace_id, {"belief_id": str(bid), "revision": rev_no})
        return msg

    # ------------------------------------------------------------------ decision plane
    def fake_model1(self, msg: BeliefMessage, trace: UUID) -> tuple[DecisionRecord, InformationNeed | None]:
        claim = DecisionClaim(
            claim_id=self.ids.new(), claim_type=ClaimType.BELIEF_CLAIM, statement="segment severity", structured_value={"status": msg.knowledge_status.value},
            source_belief_ids=(msg.belief_id,), source_belief_revisions=(msg.revision,), evidence_refs=msg.evidence_support,
            uncertainty=msg.uncertainty, timestamp=msg.timestamp, grounding=GroundingStatus.GROUNDED,
        )
        need = None
        if msg.uncertainty.observational > 0.5:
            action = ActionProposal(action_id=self.ids.new(), action_type=ActionType.REQUEST_INFORMATION, target_belief_ids=(msg.belief_id,), supporting_claims=(claim.claim_id,), score=1.0)
            need = InformationNeed(
                need_id=self.ids.new(), trace_id=trace, target_belief_ids=(msg.belief_id,), question_type=QuestionType.EXTEND_COVERAGE,
                target_properties=("severity",), priority=0.9, originating_claim_ids=(claim.claim_id,),
            )
        else:
            action = ActionProposal(action_id=self.ids.new(), action_type=ActionType.TRANSMIT_INFORMATION, target_belief_ids=(msg.belief_id,), supporting_claims=(claim.claim_id,), score=1.0)
        pid = self.ids.new()
        self.repo.put_provenance(self.run_id, [ProvenanceRecord(
            record_id=pid, source_type=SourceType.DECISION, source_ids=(msg.belief_id,), operation="fake_decide", module="fake.model1",
            model_version=VERSION, timestamp=msg.timestamp, parent_records=msg.provenance_refs, subject_id=action.action_id,
        )])
        record = DecisionRecord(
            decision_id=self.ids.new(), trace_id=trace, mission_id=self.mission_id, timestamp=msg.timestamp, belief_snapshot_id=msg.message_id,
            claims=(claim,), candidates=(action,), chosen=action,
            constraint_decisions=(ConstraintDecision(action_id=action.action_id, accepted=True, engine_version=VERSION),),
            rationale="coverage missing" if need else "belief resolved", provenance_id=pid, model_version=VERSION,
        )
        self.result.decisions.append(record)
        self.log.emit(EventType.DECISION_MADE, "fake.model1", trace, {"decision_id": str(record.decision_id), "action": action.action_type.value, "belief_id": str(msg.belief_id), "belief_revision": msg.revision, "provenance_id": str(pid)})
        return record, need

    def fake_mcbr(self, need: InformationNeed, decision: DecisionRecord) -> ObservationPlan:
        view = Pose(frame_id=WORLD, position_m=(5.0, 2.5, -10.0))
        pid = self.ids.new()
        self.repo.put_provenance(self.run_id, [ProvenanceRecord(
            record_id=pid, source_type=SourceType.PLAN, source_ids=(need.need_id,), operation="fake_plan", module="fake.mcbr",
            model_version=VERSION, timestamp=self.clock.stamp(), parent_records=(decision.provenance_id,), subject_id=need.need_id,
        )])
        act = ObservationAction(
            action_id=self.ids.new(), pose=view, sensor_id=self.sensor_id, target_region=SpatialSupport(frame_id=WORLD, center_m=(5.0, 0.0, -10.0), half_extent_m=(1, 1, 1)),
            duration_s=1.0, expected_cost=ResourceCost(time_s=30, energy_j=500, risk=0.05, travel_m=8), predicted_visibility=0.9, expected_information_gain=0.8,
        )
        plan = ObservationPlan(
            plan_id=self.ids.new(), need_id=need.need_id, trace_id=need.trace_id, status=PlanStatus.PLAN, target_beliefs=need.target_belief_ids,
            primary_action=act, expected_information_gain=0.8, targeted_uncertainty=(UncertaintyType.OBSERVATIONAL,), expected_cost=act.expected_cost, confidence=0.7, provenance=pid,
        )
        self.result.plans.append(plan)
        self.log.emit(EventType.PLAN_PROPOSED, "fake.mcbr", need.trace_id, {"plan_id": str(plan.plan_id), "need_id": str(need.need_id), "provenance_id": str(pid)})
        return plan

    def fake_navigation(self, plan: ObservationPlan) -> None:
        assert plan.primary_action is not None
        goal = NavigationGoal(
            goal_id=self.ids.new(), trace_id=plan.trace_id, target_pose=plan.primary_action.pose, position_tolerance_m=0.15,
            orientation_tolerance_rad=0.2, risk_limit=0.2, source_plan_id=plan.plan_id, source_action_id=plan.primary_action.action_id,
        )
        self.log.emit(EventType.GOAL_ACCEPTED, "fake.navigation", plan.trace_id, {"goal_id": str(goal.goal_id), "plan_id": str(plan.plan_id)})
        target = np.asarray(plan.primary_action.pose.position_m)
        # detour waypoint so the fake robot does not drive through the pipe
        for waypoint in (np.array([1.5, 2.5, -10.0]), target):
            for _ in range(400):
                est = np.asarray(self.estimated_pose().position_m)  # navigation uses the ESTIMATED pose
                err = waypoint - est
                if np.linalg.norm(err) < goal.position_tolerance_m:
                    break
                vel = np.clip(err, -1.0, 1.0)
                self.fake_control(goal, vel)
                self.clock.t += DT

    def fake_control(self, goal: NavigationGoal, vel: np.ndarray) -> None:
        now = self.clock.ns()
        wrench = WrenchCommand(command_id=self.ids.new(), trace_id=goal.trace_id, timestamp=self.clock.stamp(), frame_id="ROBOT", force_n=(float(vel[0]) * 40, float(vel[1]) * 40, float(vel[2]) * 40), torque_nm=(0, 0, 0))
        cmds = {t.thruster_id: 0.0 for t in self.config.thrusters}
        cmds.update({"H1": float(vel[0]), "H2": float(vel[1]), "V1": float(vel[2])})
        cid = self.ids.new()
        pid = self.ids.new()
        self.repo.put_provenance(self.run_id, [ProvenanceRecord(
            record_id=pid, source_type=SourceType.COMMAND, source_ids=(wrench.command_id, goal.goal_id), operation="fake_allocate",
            module=ALLOCATOR_PRODUCER, model_version=VERSION, timestamp=self.clock.stamp(), parent_records=(self.result.plans[-1].provenance,), subject_id=cid,
        )])
        cmd = AllocatedCommand(
            command_id=cid, mission_id=self.mission_id, run_id=self.run_id, trace_id=goal.trace_id, belief_snapshot_id=None,
            robot_config_digest=self.config.content_digest(), clock_domain="SIM", issued_time_ns=now, deadline_ns=now + 250_000_000,
            thruster_commands=cmds, source_wrench_id=wrench.command_id, provenance_root=pid, producer=ALLOCATOR_PRODUCER,
            safety_authorization=SafetyAuthorization(authorization_id=self.ids.new(), command_id=cid, supervisor_version=VERSION, issued_time_ns=now, safety_state="NORMAL"),
        )
        self.log.emit(EventType.WRENCH_REQUESTED, "fake.control", goal.trace_id, {"wrench_id": str(wrench.command_id), "goal_id": str(goal.goal_id)})
        ack = self.gateway.submit(cmd)
        assert ack.accepted, ack.reason_codes
        self.result.commands.append(cmd)

    def fake_baac(self, msg: BeliefMessage, decision: DecisionRecord) -> None:
        delta = SemanticDelta(
            delta_type=DeltaType.NEW_BELIEF if self._receiver_revision is None else DeltaType.STATE_CHANGED, belief_id=msg.belief_id,
            base_revision=self._receiver_revision, new_revision=msg.revision,
            changed_fields={"severity": msg.state_summary[0].value, "status": msg.knowledge_status.value, "uncertainty": list(msg.uncertainty.as_tuple())},
        )
        unit = InformationUnit(
            unit_id=self.ids.new(), trace_id=decision.trace_id, content_type=InformationType.BELIEF_DELTA, belief_ids=(msg.belief_id,),
            evidence_ids=msg.evidence_support, semantic_delta=delta, priority=0.9, mission_value=1.0, created_time_ns=self.clock.ns(),
            fidelity_levels=(FidelityOption(fidelity=Fidelity.F1_STRUCTURED_BELIEF, size_bits=len(delta.canonical_json()) * 8, information_retained=0.8),),
            provenance_id=decision.provenance_id,
        )
        tx = Transmission(
            transmission_id=self.ids.new(), unit_id=unit.unit_id, trace_id=unit.trace_id, link_name="acoustic", fidelity=Fidelity.F1_STRUCTURED_BELIEF,
            bits=unit.fidelity_levels[0].size_bits, sent_time_ns=self.clock.ns(), delivered=True, delivered_time_ns=self.clock.ns() + 1_000_000_000,
            payload=delta.model_dump(mode="json"),
        )
        self._receiver_revision = msg.revision
        self.result.transmissions.append(tx)
        self.log.emit(EventType.TRANSMISSION, "fake.baac", unit.trace_id, {"unit_id": str(unit.unit_id), "bits": tx.bits, "belief_id": str(msg.belief_id), "revision": msg.revision})

    # ------------------------------------------------------------------ loop
    def run(self, max_cycles: int = 4) -> FakeRunResult:
        self.log.emit(EventType.RUN_STARTED, "fake.system", self.run_id, {"seed_stream": "fake"})
        for _ in range(max_cycles):
            trace = self.ids.new()
            obs = self.fake_sensor(trace)
            msg = self.fake_model2(*self.fake_ecmer(obs))
            decision, need = self.fake_model1(msg, trace)
            if need is None:
                self.fake_baac(msg, decision)
                break
            self.fake_navigation(self.fake_mcbr(need, decision))
            self.clock.t += DT
        self.log.emit(EventType.RUN_TERMINATED, "fake.system", self.run_id, {"cycles": len(self.result.decisions)})
        self.result.final_position = tuple(float(v) for v in self.robot.true_position)  # type: ignore[assignment]
        self.log.close()
        return self.result
