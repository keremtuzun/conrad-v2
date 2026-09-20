"""NAV-SAFETY-01: deterministic Safety Supervisor. Observes everything, overrides all autonomy.

``assess`` maps monitored conditions to a safety state (worst wins; leak and operator E-stop latch).
``authorize`` validates one AllocatedCommand against that state and either attaches a
SafetyAuthorization or refuses it. Every decision carries reason codes.

Two rules protect a HOLD taken because the state estimate itself is bad (FLAGSHIP-UNITY, 2026-09-20):

* the assessment carries the safe-hold action the consumer must execute. While the estimate is not
  trustworthy (missing, stale, estimator FAULT, LOCALIZATION_LOST, sigma above the limit) it is
  ``SafetyConfig.untrusted_state_hold_action`` (ZERO_THRUST by default), never a station keep that would
  close a control loop on the estimate that was just declared lost;
* the mission boundary is judged on the worst case the estimate admits (inflated by ``boundary_sigma_k``
  sigma), and a real breach while the estimate is untrustworthy latches EMERGENCY_STOP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from conrad.robotics.allocation.allocator import PRODUCER
from conrad.robotics.safety.monitors import (
    SEVERITY,
    Reason,
    SafeHoldAction,
    SafetyAssessment,
    SafetyConfig,
    SafetyInputs,
    SafetyState,
    collision_envelope,
)
from conrad.schemas.ids import IdFactory
from conrad.schemas.robot import AllocatedCommand, HealthLevel, RobotConfig, SafetyAuthorization
from conrad.schemas.timebase import NS_PER_S

SUPERVISOR_VERSION = "conrad.robotics.safety/1.0.0"
MOTION_STATES = frozenset({SafetyState.NORMAL, SafetyState.DEGRADED, SafetyState.HOLD})


@dataclass(frozen=True)
class SafetyDecision:
    authorized: bool
    command: AllocatedCommand
    state: SafetyState
    reason_codes: tuple[str, ...]


class SafetySupervisor:
    def __init__(self, robot_config: RobotConfig, ids: IdFactory, config: SafetyConfig | None = None) -> None:
        s = robot_config.safety
        self.config = config or SafetyConfig()
        self._ids = ids
        action = s.safe_hold_action.require("safety.safe_hold_action")
        try:
            self.safe_hold_action = SafeHoldAction(action)
        except ValueError as exc:
            raise ValueError(
                f"unsupported safe_hold_action {action!r}; supported: {list(SafeHoldAction)}"
            ) from exc
        self.max_depth = float(s.max_depth_m.require("safety.max_depth_m"))
        self.min_altitude = float(s.min_altitude_m.require("safety.min_altitude_m"))
        self.max_speed = float(s.max_speed_mps.require("safety.max_speed_mps"))
        self.max_sigma = float(s.max_pose_sigma_m.require("safety.max_pose_sigma_m"))
        self.min_battery = float(s.min_battery_fraction.require("safety.min_battery_fraction"))
        self.stale_s = float(s.state_stale_after_s.require("safety.state_stale_after_s"))
        self.reserve = float(robot_config.battery.reserve_fraction.require("battery.reserve_fraction"))
        dims = np.asarray(robot_config.dimensions_m.require("dimensions_m"), dtype=np.float64)
        self.vehicle_radius = 0.5 * float(np.max(dims))
        self._digest = robot_config.content_digest()
        self._thrusters = frozenset(t.thruster_id for t in robot_config.thrusters)
        self._latched: list[str] = []
        self._loss_since_ns: int | None = None
        self._degraded_until_ns = -1
        self.state = SafetyState.NORMAL
        self.last: SafetyAssessment | None = None
        self.history: list[tuple[int, SafetyState, tuple[str, ...]]] = []

    # -- operator / latched overrides ------------------------------------------------------------
    def emergency_stop(self, reason: str = Reason.OPERATOR_ESTOP) -> None:
        if reason not in self._latched:
            self._latched.append(reason)

    def reset_latch(self) -> None:
        """Operator acknowledgement. Only clears latches; conditions are re-evaluated next tick."""
        self._latched.clear()

    # -- assessment ------------------------------------------------------------------------------------
    def assess(self, x: SafetyInputs) -> SafetyAssessment:
        found: list[tuple[SafetyState, str]] = []
        add = lambda st, code: found.append((st, code))  # noqa: E731
        c = self.config
        sigma: float | None = None
        speed = 0.0
        # the estimate may not be used as a control reference once any of these holds (see the module docstring)
        untrusted = False
        if x.state is None:
            add(SafetyState.HOLD, Reason.STATE_MISSING)
            untrusted = True
        else:
            age_s = (x.now_ns - x.state.timestamp.time_ns) / NS_PER_S
            if x.state.timestamp.clock_domain != x.clock_domain or age_s > self.stale_s:
                add(SafetyState.HOLD, Reason.STATE_STALE)
                untrusted = True
            sigma = x.state.pose.position_sigma_m()
            if x.state.linear_velocity_body_mps is not None:
                speed = float(np.linalg.norm(x.state.linear_velocity_body_mps))
            lost = (
                sigma is None
                or sigma > self.max_sigma
                or x.state.estimator_health is HealthLevel.FAULT
                or Reason.LOCALIZATION_LOST in x.estimator_reasons
            )
            if lost:
                add(SafetyState.HOLD, Reason.LOCALIZATION_LOST)
                untrusted = True
                if sigma is not None and sigma > self.max_sigma:
                    add(SafetyState.HOLD, Reason.POSE_SIGMA_HIGH)
                self._loss_since_ns = self._loss_since_ns if self._loss_since_ns is not None else x.now_ns
                if (x.now_ns - self._loss_since_ns) / NS_PER_S > c.relocalization_timeout_s:
                    add(SafetyState.RETURN, Reason.RELOCALIZATION_TIMEOUT)
            else:
                self._loss_since_ns = None
                if sigma is not None and sigma > c.degraded_sigma_fraction * self.max_sigma:
                    add(SafetyState.DEGRADED, Reason.POSE_SIGMA_ELEVATED)
                if x.state.estimator_health is HealthLevel.DEGRADED:
                    # estimator-declared inconsistency (NIS, gated fixes/depth) slows the vehicle even
                    # when the reported sigma still looks small: the sigma itself is then suspect.
                    add(SafetyState.DEGRADED, Reason.ESTIMATOR_DEGRADED)
                    for code in x.estimator_reasons:
                        add(SafetyState.DEGRADED, code)
            p = np.asarray(x.state.pose.position_m)
            depth, depth_sigma = c.water_surface_z_m - float(p[2]), (sigma or 0.0)
            if depth + c.depth_sigma_k * depth_sigma > self.max_depth:
                add(SafetyState.HOLD, Reason.DEPTH_LIMIT)
            if c.boundary is not None:
                # the boundary is judged on the WORST CASE the estimate admits, not on its mean: a drifting
                # estimate must stop the vehicle before the true position can be outside, not after.
                if not c.boundary.contains(p):
                    add(SafetyState.HOLD, Reason.BOUNDARY)
                    if untrusted and c.boundary_breach_escalates_to_estop:
                        # the declared operating volume is breached and the vehicle cannot navigate back on an
                        # estimate it may not use: latch, so no autonomy resumes motion without an operator.
                        self.emergency_stop(Reason.BOUNDARY_BREACH_UNLOCALIZED)
                elif sigma is None or not c.boundary.contains_ball(p, c.boundary_sigma_k * sigma):
                    add(SafetyState.HOLD, Reason.BOUNDARY_RISK)
        faulted: set[str] = set()
        if x.health is not None:
            if x.health.leak_detected and Reason.LEAK not in self._latched:
                self._latched.append(Reason.LEAK)
            for name, level in x.health.devices.items():
                if level is HealthLevel.FAULT:
                    code = Reason.DEGRADED_MANEUVERABILITY if name in self._thrusters else Reason.SENSOR_FAULT
                    add(SafetyState.DEGRADED, code)
                    if name in self._thrusters:
                        faulted.add(name)
                elif level is HealthLevel.DEGRADED and name in self._thrusters:
                    add(SafetyState.DEGRADED, Reason.DEGRADED_MANEUVERABILITY)
            faulted_devices = {n for n, lv in x.health.devices.items() if lv is HealthLevel.FAULT}
            # a FAULT fully explained by failed thrusters is DEGRADED_MANEUVERABILITY (ch20: the allocator
            # recomputes the feasible wrench), not an unexplained hardware fault
            explained = bool(faulted_devices) and faulted_devices <= self._thrusters
            if x.health.overall is HealthLevel.FAULT and not x.health.leak_detected and not explained:
                add(SafetyState.HOLD, Reason.HARDWARE_FAULT)
        faulted |= {t.thruster_id for t in x.thrusters if t.health is HealthLevel.FAULT}
        if x.battery is None:
            add(SafetyState.DEGRADED, Reason.BATTERY_UNKNOWN)
        elif x.battery.remaining_fraction < self.min_battery:
            add(SafetyState.RETURN, Reason.BATTERY_LOW)
        elif x.battery.remaining_fraction < self.reserve:
            add(SafetyState.DEGRADED, Reason.BATTERY_RESERVE)
        if x.altitude_m is None:
            info = [Reason.ALTITUDE_UNMONITORED]
        else:
            info = []
            if x.altitude_m < self.min_altitude:
                add(SafetyState.HOLD, Reason.ALTITUDE_LIMIT)
        d_safe = collision_envelope(self.vehicle_radius, speed, sigma, c)
        if x.obstacle_clearance_m is not None:
            hard = self.vehicle_radius + c.sigma_k * (sigma if sigma is not None else math.inf)
            if x.obstacle_clearance_m < hard:
                add(SafetyState.HOLD, Reason.COLLISION_ENVELOPE)
            elif x.obstacle_clearance_m < d_safe:
                add(SafetyState.DEGRADED, Reason.COLLISION_MARGIN)
        if x.tracking_error_m is not None and x.tracking_error_m > c.tracking_error_limit_m:
            add(SafetyState.DEGRADED, Reason.TRACKING_ERROR)
        for code in dict.fromkeys(self._latched):
            add(SafetyState.RECOVER if code == Reason.LEAK else SafetyState.EMERGENCY_STOP, code)
            if code == Reason.LEAK:
                add(SafetyState.RECOVER, Reason.RECOVERY_BEHAVIOUR_OPEN)

        state = max((st for st, _ in found), key=SEVERITY.__getitem__, default=SafetyState.NORMAL)
        if state is SafetyState.DEGRADED:
            self._degraded_until_ns = x.now_ns + round(c.degraded_dwell_s * NS_PER_S)
        elif state is SafetyState.NORMAL and x.now_ns < self._degraded_until_ns:
            state = SafetyState.DEGRADED
            found.append((state, Reason.DEGRADED_DWELL))
        reasons = tuple(dict.fromkeys([code for _, code in found] + info))
        hold = state is SafetyState.HOLD
        action = c.untrusted_state_hold_action if untrusted else self.safe_hold_action
        zero = state in (SafetyState.EMERGENCY_STOP, SafetyState.RECOVER) or (
            hold
            and (
                # no usable state at all: nothing may be commanded, whatever the configured actions say
                Reason.STATE_STALE in reasons
                or Reason.STATE_MISSING in reasons
                or action is SafeHoldAction.ZERO_THRUST
            )
        )
        if untrusted:
            reasons = (*reasons, Reason.STATE_NOT_TRUSTWORTHY)
        if hold:
            reasons = (*reasons, f"{Reason.SAFE_HOLD}:{action.value}")
        scale = 1.0 if state is SafetyState.NORMAL else c.degraded_speed_scale
        a = SafetyAssessment(
            state, reasons, scale, hold, zero, d_safe, frozenset(faulted), action, not untrusted
        )
        if state is not self.state or (self.last is not None and reasons != self.last.reason_codes):
            self.history.append((x.now_ns, state, reasons))
        self.state, self.last = state, a
        return a

    # -- per-command authorization ------------------------------------------------------------------------
    def authorize(
        self, command: AllocatedCommand, assessment: SafetyAssessment, now_ns: int, clock_domain: str
    ) -> SafetyDecision:
        refuse: list[str] = []
        if command.producer != PRODUCER:
            refuse.append(Reason.PRODUCER)
        if command.clock_domain != clock_domain:
            refuse.append(Reason.CLOCK)
        if command.deadline_ns <= now_ns:
            refuse.append(Reason.DEADLINE)
        if command.robot_config_digest != self._digest:
            refuse.append(Reason.DIGEST)
        values = command.thruster_commands
        if set(values) != self._thrusters:
            refuse.append(Reason.ACTUATOR_SET)
        if any(not math.isfinite(v) or abs(v) > 1.0 for v in values.values()):
            refuse.append(Reason.ACTUATOR_RANGE)
        if any(abs(values.get(t, 0.0)) > 0 for t in assessment.faulted_actuators):
            refuse.append(Reason.ACTUATOR_FAULTED)
        if assessment.zero_thrust_required and any(abs(v) > 0 for v in values.values()):
            refuse.append(Reason.ZERO_REQUIRED)
        all_zero = all(v == 0.0 for v in values.values())
        if assessment.state not in MOTION_STATES and not all_zero:
            # EMERGENCY_STOP / RECOVER / RETURN may still receive an explicit all-zero command.
            refuse.append(Reason.STATE_FORBIDS_MOTION)
        reasons = tuple(dict.fromkeys([*assessment.reason_codes, *refuse]))
        if refuse:
            return SafetyDecision(False, command, assessment.state, reasons)
        auth = SafetyAuthorization(
            authorization_id=self._ids.new(),
            command_id=command.command_id,
            supervisor_version=SUPERVISOR_VERSION,
            issued_time_ns=now_ns,
            safety_state=assessment.state.value,
            reason_codes=reasons,
        )
        return SafetyDecision(
            True, command.model_copy(update={"safety_authorization": auth}), assessment.state, reasons
        )
