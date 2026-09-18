"""Executive NavigationStack: estimator -> goal/planners -> trajectory -> controller -> allocator -> safety.

It READS the RobotHardwareInterface through a read-only facade and returns the (authorized or
refused) AllocatedCommand. It never calls ``send``: only the Command Gateway may.
"""

from __future__ import annotations

from uuid import UUID

import numpy as np

from conrad.robotics.allocation.allocator import ThrusterAllocator
from conrad.robotics.control.pid import CascadedPidController, ControlReference
from conrad.robotics.estimation.ekf import EkfStateEstimator
from conrad.robotics.estimation.interface import MapConstraint
from conrad.robotics.estimation.rotations import quat_from_yaw, quat_to_rot, yaw_of
from conrad.robotics.hardware.interface import RobotHardwareInterface
from conrad.robotics.navigation.global_planner import (
    AStarPlanner,
    CostHook,
    IsFree,
    KnowledgeQuery,
    PlanningError,
)
from conrad.robotics.navigation.goals import GoalManager, GoalRejectedError, NavigationObjective
from conrad.robotics.navigation.local_planner import (
    LocalDistance,
    PotentialFieldLocalPlanner,
)
from conrad.robotics.navigation.stack_types import (
    GoalStatus,
    NavigationStackConfig,
    StepResult,
    _ReadOnlyHardware,
)
from conrad.robotics.navigation.trace import NavRecordType, NavTraceRecord, NavTraceSink
from conrad.robotics.safety.monitors import (
    SafeHoldAction,
    SafetyAssessment,
    SafetyInputs,
    SafetyState,
    collision_envelope,
)
from conrad.robotics.safety.supervisor import SafetySupervisor
from conrad.robotics.trajectory.generator import TrajectoryGenerator, TrajectorySampler
from conrad.schemas.decision import NavigationGoal, Trajectory
from conrad.schemas.frames import ROBOT, Pose
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Observation
from conrad.schemas.robot import RobotConfig, WrenchCommand
from conrad.schemas.timebase import NS_PER_S, TimeStamp


class NavigationStack:
    def __init__(
        self,
        hardware: RobotHardwareInterface,
        robot_config: RobotConfig,
        ids: IdFactory,
        mission_id: UUID,
        run_id: UUID,
        initial_pose: Pose,
        config: NavigationStackConfig | None = None,
        is_free: IsFree | None = None,
        knowledge: KnowledgeQuery | None = None,
        cost_hook: CostHook | None = None,
        local_distance: LocalDistance | None = None,
        sink: NavTraceSink | None = None,
    ) -> None:
        self.config = c = config or NavigationStackConfig()
        self._hw = _ReadOnlyHardware(hardware)
        self._ids, self.mission_id, self.run_id = ids, mission_id, run_id
        self._clock = self._hw.clock_domain()
        now = self._hw.now_ns()
        self.estimator = EkfStateEstimator(
            robot_config, initial_pose, TimeStamp(time_ns=now, clock_domain=self._clock), c.estimator
        )
        self.controller = CascadedPidController(robot_config, ids, c.control)
        self.allocator = ThrusterAllocator(robot_config, ids, c.allocation)
        self.trajectories = TrajectoryGenerator(robot_config, ids, c.trajectory)
        self.planner = AStarPlanner(is_free, c.planner, knowledge, cost_hook) if is_free is not None else None
        self.local = PotentialFieldLocalPlanner(local_distance, c.local)
        self.supervisor = SafetySupervisor(robot_config, ids, c.safety)
        self.goals = GoalManager()
        self._max_depth = self.supervisor.max_depth
        self._sink = sink
        self.records: list[NavTraceRecord] = []
        self.status = GoalStatus.IDLE
        self.objective: NavigationObjective | None = None
        self.trajectory: Trajectory | None = None
        self._sampler: TrajectorySampler | None = None
        self._traj_clock_s = 0.0
        self._hold_pose: tuple[np.ndarray, np.ndarray] | None = None
        self._arrived_at_ns: int | None = None
        self._last_ns: int | None = None
        self._last_depth_ns = -1
        self._trace = ids.new()
        self._last_state = SafetyState.NORMAL

    # -- records ------------------------------------------------------------------------------------
    def _record(self, record_type: str, now_ns: int, **kw: object) -> None:
        rec = NavTraceRecord(record_type, self._trace, now_ns, self._clock, **kw)  # type: ignore[arg-type]
        self.records.append(rec)
        if self._sink is not None:
            self._sink(rec)

    # -- goals ----------------------------------------------------------------------------------------------
    def set_goal(self, goal: NavigationGoal) -> Trajectory | None:
        now = self._hw.now_ns()
        self._trace = goal.trace_id
        est = self.estimator.get_state()
        p = np.asarray(est.pose.position_m)
        try:
            obj = self.goals.to_objective(goal, p)
            self._check_envelope(obj)
            route = np.vstack([p[None, :], obj.waypoints])
            if self.planner is not None and not obj.route_is_prescribed:
                legs = [self.planner.plan(route[i], route[i + 1]) for i in range(len(route) - 1)]
                route = np.vstack([legs[0]] + [leg[1:] for leg in legs[1:]])
        except (GoalRejectedError, PlanningError, ValueError) as exc:
            code = getattr(exc, "reason_code", type(exc).__name__)
            self._record(
                NavRecordType.GOAL_REJECTED,
                now,
                goal_id=goal.goal_id,
                payload={"reason": code, "detail": str(exc)},
            )
            self.status, self.objective, self._sampler = GoalStatus.REJECTED, None, None
            return None
        traj = self.trajectories.generate(
            route,
            goal.goal_id,
            goal.trace_id,
            start_yaw=yaw_of(np.asarray(est.pose.orientation_wxyz)),
            yaw_mode=obj.yaw_mode,
            final_orientation_wxyz=obj.final_orientation_wxyz,
            look_at=obj.look_at,
        )
        self.objective, self.trajectory, self._sampler = obj, traj, TrajectorySampler(traj)
        self._traj_clock_s, self._arrived_at_ns, self._hold_pose = 0.0, None, None
        self.status = GoalStatus.EXECUTING
        self._record(NavRecordType.GOAL_ACCEPTED, now, goal_id=goal.goal_id, payload={"kind": obj.kind.value})
        self._record(
            NavRecordType.TRAJECTORY_PLANNED,
            now,
            goal_id=goal.goal_id,
            trajectory_id=traj.trajectory_id,
            payload={"points": len(traj.points), "duration_s": traj.points[-1].t_s, "waypoints": len(route)},
        )
        return traj

    def _check_envelope(self, obj: NavigationObjective) -> None:
        if np.any(-obj.waypoints[:, 2] > self._max_depth):
            raise GoalRejectedError("GOAL_OUTSIDE_ENVELOPE", "waypoint deeper than safety.max_depth_m")
        boundary = self.config.safety.boundary
        if boundary is not None and not all(boundary.contains(w) for w in obj.waypoints):
            raise GoalRejectedError("GOAL_OUTSIDE_ENVELOPE", "waypoint outside the mission boundary")

    # -- external evidence -------------------------------------------------------------------------------
    def add_position_fix(self, observation: Observation) -> bool:
        return self.estimator.update_visual(observation)

    def add_map_constraint(self, constraint: MapConstraint) -> bool:
        return self.estimator.update_map_constraint(constraint)

    def state_age_s(self) -> float | None:
        """For ``CommandGateway(state_age_s=...)``: age of the estimate in the hardware clock."""
        return (self._hw.now_ns() - self.estimator.get_state().timestamp.time_ns) / NS_PER_S

    # -- one control tick ---------------------------------------------------------------------------------
    def _estimate(self, now: int, dt: float) -> None:
        imu = self._hw.get_imu()
        thrusters = self._hw.get_thruster_state()
        if imu is not None:
            self.estimator.predict(imu, thrusters, dt)
        depth = self._hw.get_depth()
        if depth is not None and depth.timestamp.time_ns > self._last_depth_ns:
            self._last_depth_ns = depth.timestamp.time_ns
            self.estimator.update_depth(depth)
        for obs in (self._hw.get_camera(), self._hw.get_sonar()):
            if obs is not None:
                self.estimator.update_visual(obs)
        self.allocator.update_from_thruster_states(thrusters)

    def _reference(
        self, a: SafetyAssessment, p: np.ndarray, q: np.ndarray, dt: float, now: int
    ) -> ControlReference:
        if a.hold_required or (a.state is not SafetyState.NORMAL and a.state is not SafetyState.DEGRADED):
            if self._hold_pose is None:
                hold_p = p.copy()
                if self.supervisor.safe_hold_action is SafeHoldAction.SURFACE:
                    hold_p[2] = 0.0  # only because RobotConfig says so
                self._hold_pose = (hold_p, quat_from_yaw(yaw_of(q)))
            return ControlReference(self._hold_pose[0], self._hold_pose[1], speed_limit_mps=0.25)
        self._hold_pose = None
        if self._sampler is None:
            self._hold_pose = None
            return ControlReference(p.copy(), quat_from_yaw(yaw_of(q)), speed_limit_mps=0.25)
        self._traj_clock_s += dt * a.speed_scale
        ref = self._sampler.sample(self._traj_clock_s)
        if a.speed_scale < 1.0:
            ref = ControlReference(
                ref.position_world_m,
                ref.orientation_wxyz,
                ref.velocity_world_mps * a.speed_scale,
                ref.speed_limit_mps,
            )
        self._update_status(p, now)
        return ref

    def _update_status(self, p: np.ndarray, now: int) -> None:
        if self.objective is None or self._sampler is None or self.status is GoalStatus.COMPLETE:
            return
        done = self._traj_clock_s >= self._sampler.duration_s
        close = (
            float(np.linalg.norm(p - self.objective.final_position))
            <= self.objective.goal.position_tolerance_m
        )
        if done and close and self._arrived_at_ns is None:
            self._arrived_at_ns, self.status = now, GoalStatus.ARRIVED
        if (
            self._arrived_at_ns is not None
            and (now - self._arrived_at_ns) / NS_PER_S >= self.objective.hold_duration_s
        ):
            self.status = GoalStatus.COMPLETE
            self._record(NavRecordType.GOAL_COMPLETED, now, goal_id=self.objective.goal.goal_id)

    def step(self) -> StepResult:
        now = self._hw.now_ns()
        dt = (
            self.config.control_period_s
            if self._last_ns is None
            else max((now - self._last_ns) / NS_PER_S, 1e-6)
        )
        self._last_ns = now
        self._estimate(now, dt)
        state = self.estimator.get_state()
        p, q = np.asarray(state.pose.position_m), np.asarray(state.pose.orientation_wxyz)
        clearance = self.local.clearance(p)
        ref_err = None
        if self._sampler is not None:
            ref_err = float(np.linalg.norm(self._sampler.sample(self._traj_clock_s).position_world_m - p))
        a = self.supervisor.assess(
            SafetyInputs(
                now_ns=now,
                clock_domain=self._clock,
                state=state,
                estimator_reasons=self.estimator.health_reasons(),
                health=self._hw.get_health(),
                battery=self._hw.get_power_state(),
                thrusters=self._hw.get_thruster_state(),
                obstacle_clearance_m=clearance,
                tracking_error_m=ref_err,
            )
        )
        if a.state is not self._last_state:
            self._record(
                NavRecordType.SAFETY_STATE_CHANGED,
                now,
                payload={"state": a.state.value, "reasons": a.reason_codes},
            )
            self._last_state = a.state
        ref = self._reference(a, p, q, dt, now)
        sigma = state.pose.position_sigma_m()
        v_world = quat_to_rot(q) @ np.asarray(state.linear_velocity_body_mps or (0.0, 0.0, 0.0))
        ref = self.local.adjust(
            p,
            v_world,
            ref,
            lambda closing: collision_envelope(
                self.supervisor.vehicle_radius, closing, sigma, self.config.safety
            ),
        )
        stamp = TimeStamp(time_ns=now, clock_domain=self._clock)
        goal_id = self.objective.goal.goal_id if self.objective else None
        traj_id = self.trajectory.trajectory_id if self.trajectory else None
        if a.zero_thrust_required:
            self.controller.reset()
            wrench = WrenchCommand(
                command_id=self._ids.new(),
                trace_id=self._trace,
                timestamp=stamp,
                frame_id=ROBOT,
                force_n=(0.0, 0.0, 0.0),
                torque_nm=(0.0, 0.0, 0.0),
            )
            cmd = self.allocator.zero_command(
                trace_id=self._trace,
                mission_id=self.mission_id,
                run_id=self.run_id,
                now_ns=now,
                clock_domain=self._clock,
                source_wrench_id=wrench.command_id,
            )
        else:
            wrench = self.controller.compute(state, ref, dt, self._trace, stamp)
            cmd, _ = self.allocator.allocate(
                wrench,
                mission_id=self.mission_id,
                run_id=self.run_id,
                now_ns=now,
                clock_domain=self._clock,
                provenance_root=goal_id,
            )
        self._record(
            NavRecordType.WRENCH_REQUESTED,
            now,
            goal_id=goal_id,
            trajectory_id=traj_id,
            wrench_id=wrench.command_id,
        )
        self._record(
            NavRecordType.COMMAND_ALLOCATED,
            now,
            goal_id=goal_id,
            trajectory_id=traj_id,
            wrench_id=wrench.command_id,
            command_id=cmd.command_id,
        )
        decision = self.supervisor.authorize(cmd, a, now, self._clock)
        if not decision.authorized:
            self._record(
                NavRecordType.SAFETY_DECISION,
                now,
                command_id=cmd.command_id,
                payload={"authorized": False, "reasons": decision.reason_codes},
            )
        return StepResult(decision.command, decision, state, a, ref)
