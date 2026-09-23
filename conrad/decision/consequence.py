"""Consequence / utility representation (ch16 'Risk vs uncertainty', 'Utility representation').

C(A) = [mission, information, time, energy, risk, communication, compute]. Risk (consequence of
being wrong) and uncertainty (how little is known) are separate fields and are never summed into a
universal confidence. Hard constraints stay outside utility.

All numbers below are configuration (ENGINEERING_ESTIMATE defaults), not measurements.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from pydantic import Field

from conrad.decision.claims import ClaimGraph, RequirementAssessment
from conrad.decision.config import DecisionConfig, UtilityWeights
from conrad.decision.context import DecisionContext
from conrad.schemas.base import ConradModel
from conrad.schemas.decision import ActionProposal, ActionType, UncertaintyType
from conrad.schemas.robot import HealthLevel
from conrad.schemas.uncertainty import Uncertainty

# resolvability key used when U_E is over threshold only via the uncalibrated-source floor
CALIBRATION_CHECK = "CALIBRATION_CHECK"


class ConsequenceVector(ConradModel):
    mission: float = 0.0
    information: float = 0.0
    time: float = Field(default=0.0, ge=0)
    energy: float = Field(default=0.0, ge=0)
    risk: float = Field(default=0.0, ge=0, le=1)
    communication: float = Field(default=0.0, ge=0)
    compute: float = Field(default=0.0, ge=0)
    uncertainty_exposure: float = Field(
        default=0.0, ge=0, description="how little is known about what the action relies on; NOT risk"
    )

    def scalarize(self, w: UtilityWeights) -> float:
        return (
            w.mission * self.mission
            + w.information * self.information
            - w.time * self.time
            - w.energy * self.energy
            - w.risk * self.risk
            - w.communication * self.communication
            - w.compute * self.compute
        )


class ActionCost(ConradModel):
    time: float = 0.0
    energy: float = 0.0
    risk: float = 0.0
    communication: float = 0.0
    compute: float = 0.0


class ConsequenceConfig(ConradModel):
    """Normalised (0..1) nominal costs per action type and resolvability per uncertainty cause."""

    costs: dict[str, ActionCost] = Field(
        default_factory=lambda: {
            ActionType.CONTINUE_MISSION.value: ActionCost(time=0.05, energy=0.05, risk=0.02),
            ActionType.QUERY_BELIEF.value: ActionCost(time=0.01, compute=0.1),
            ActionType.REQUEST_INFORMATION.value: ActionCost(time=0.3, energy=0.3, risk=0.05, compute=0.1),
            ActionType.REVISIT_REGION.value: ActionCost(time=0.4, energy=0.4, risk=0.05),
            ActionType.CHANGE_SENSOR_MODE.value: ActionCost(time=0.05, energy=0.1),
            ActionType.REPLAN.value: ActionCost(time=0.1, compute=0.3),
            ActionType.WAIT.value: ActionCost(time=0.2),
            ActionType.TRANSMIT_INFORMATION.value: ActionCost(time=0.05, energy=0.05, communication=0.5),
            ActionType.STORE_AND_FORWARD.value: ActionCost(compute=0.02),
            ActionType.ESCALATE_TO_OPERATOR.value: ActionCost(time=0.5, communication=0.3),
            ActionType.RETURN_TO_SAFE_STATE.value: ActionCost(time=0.6, energy=0.5, risk=0.05),
            ActionType.ABORT_MISSION.value: ActionCost(time=0.6, energy=0.5, risk=0.05),
        }
    )
    resolvability: dict[str, float] = Field(
        default_factory=lambda: {
            UncertaintyType.OBSERVATIONAL.value: 0.8,
            UncertaintyType.CONTRADICTION.value: 0.7,
            UncertaintyType.ALEATORIC.value: 0.5,
            UncertaintyType.EPISTEMIC.value: 0.5,
            CALIBRATION_CHECK: 0.8,
        },
        description="prior that an autonomous observation can reduce this cause (EPISTEMIC: alternate modality only)",
    )
    attempt_decay: float = Field(default=0.5, gt=0, le=1)
    operator_value: float = Field(default=0.35, ge=0, le=1)
    operator_value_unreachable: float = Field(default=0.1, ge=0, le=1)
    operator_value_autonomous_factor: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="operator value multiplier while every open item has an autonomous path. 0 (I5 iteration "
        "2): the operator is not a resort until an item is a dead end (attempts exhausted or OOD without "
        "alternate evidence); 0.4 let a repeat-decayed request rank below ESCALATE while attempts remained",
    )
    report_value: float = Field(
        default=1.0, ge=0, description="value of reporting a finding, x its consequence"
    )
    default_finding_consequence: float = Field(default=0.5, ge=0, le=1)
    route_replan_value: float = Field(
        default=1.0, ge=0, description="mission value of replanning a blocked route"
    )
    blocked_route_risk: float = Field(
        default=0.9, ge=0, le=1, description="risk of continuing on a blocked route"
    )


def exposure_of(u: Uncertainty | None) -> float:
    """Largest single channel, clipped to [0, 1]. Used as exposure, never as a calibrated probability."""
    if u is None:
        return 1.0
    return min(1.0, max(u.as_tuple()))


class ConsequenceEstimator:
    def __init__(self, config: DecisionConfig, consequence_config: ConsequenceConfig | None = None) -> None:
        self.config = config
        self.cc = consequence_config or ConsequenceConfig()

    def estimate(self, action: ActionProposal, graph: ClaimGraph, ctx: DecisionContext) -> ConsequenceVector:
        cost = self.cc.costs.get(action.action_type.value, ActionCost())
        open_items = [a for a in graph.assessments if a.matters and not a.satisfied]
        total = max(1, len(graph.assessments))
        mission = 0.0
        information = 0.0
        risk = cost.risk
        exposure = 0.0
        kind = action.action_type
        if kind is ActionType.CONTINUE_MISSION:
            # An open requirement that MATTERS is a debt, not merely an unearned point: "does reducing this
            # uncertainty matter? if yes -> REQUEST_INFORMATION" (ch17 L7681-7695). Satisfied / low-consequence
            # requirements count +1, each open consequential one counts -consequence.
            if graph.assessments:
                mission = (total - sum(1.0 + a.consequence for a in open_items)) / total
            else:
                mission = 1.0
            for item in open_items:
                e = 1.0 if (item.unsupported_claim_ids or not item.grounded_claim_ids) else self._over(item)
                exposure = max(exposure, e)
                risk = max(risk, min(1.0, item.consequence * e))
            if graph.route_blocking_claim_ids:
                # continuing along a route a grounded belief says is occupied (ch17 'Replanning')
                exposure = 1.0
                risk = max(risk, self.cc.blocked_route_risk)
        elif kind is ActionType.REQUEST_INFORMATION:
            a = self._target(action, graph)
            if a is not None:
                cause = str(action.parameters.get("cause", UncertaintyType.OBSERVATIONAL.value))
                if action.parameters.get("calibration_check"):
                    # raw uncertainty is already low: one confirming look usually settles it
                    cause = CALIBRATION_CHECK
                tries = ctx.attempts_on(a.target_belief_ids, ActionType.REQUEST_INFORMATION)
                gain = self.cc.resolvability.get(cause, 0.3) * (self.cc.attempt_decay**tries)
                information = a.consequence * gain
                exposure = self._over(a)
        elif kind is ActionType.CHANGE_SENSOR_MODE:
            # ch16 L6899-6904 high U_A: repeat -> change modality. Switching modality earns value once a repeat
            # of the current measurement has been tried and U_A is still high (1 - decay^tries), and decays
            # with its own attempts. Deterministic sequencing, not a tuned weight.
            a = self._target(action, graph)
            if a is not None:
                cause = str(action.parameters.get("cause", UncertaintyType.ALEATORIC.value))
                repeats = ctx.attempts_on(a.target_belief_ids, ActionType.REQUEST_INFORMATION)
                switches = ctx.attempts_on(a.target_belief_ids, ActionType.CHANGE_SENSOR_MODE)
                gain = (
                    self.cc.resolvability.get(cause, 0.3)
                    * (1.0 - self.cc.attempt_decay**repeats)
                    * (self.cc.attempt_decay**switches)
                )
                information = a.consequence * gain
                exposure = self._over(a)
        elif kind in (ActionType.QUERY_BELIEF, ActionType.REVISIT_REGION):
            a = self._target(action, graph)
            if a is not None:
                tries = ctx.attempts_on(a.target_belief_ids, kind) + sum(
                    1
                    for d in ctx.previous_decisions
                    if d.action_type is kind
                    and d.executed is not False
                    and not a.target_belief_ids
                    and not d.target_belief_ids
                )
                base = 0.5 if kind is ActionType.QUERY_BELIEF else 0.7
                information = a.consequence * base * (self.cc.attempt_decay ** (2 * tries))
                exposure = 1.0
        elif kind is ActionType.ESCALATE_TO_OPERATOR:
            reachable = ctx.operator_reachable is not False
            value = self.cc.operator_value if reachable else self.cc.operator_value_unreachable
            worst = max((a.consequence for a in open_items), default=0.0)
            information = worst * value
            if worst >= self.config.escalate_consequence_above and self._no_autonomous_path(open_items, ctx):
                information = worst * min(1.0, value + 0.3)
            elif open_items and self._autonomous_path_for_all(open_items, ctx):
                # ch17 L7968-7979 escalation triggers: while every open item still has an autonomous, untried
                # resolution the operator is not the first resort (fixes ESCALATE dominating integrated runs)
                information = worst * value * self.cc.operator_value_autonomous_factor
        elif kind in (ActionType.RETURN_TO_SAFE_STATE, ActionType.ABORT_MISSION):
            mission = 1.0 if self._must_retreat(ctx) else -0.5
            if kind is ActionType.ABORT_MISSION:
                mission -= 0.1
        elif kind is ActionType.WAIT:
            # Once the runtime has explicitly exhausted the finite acquisition budget, holding position is
            # the bounded safe response. It neither invents evidence nor turns a nominal information gap into
            # an operator alert/retreat, and it cannot recreate the unexecutable REQUEST_INFORMATION loop.
            mission = 1.0 if ctx.unavailable_information_targets and not self._must_retreat(ctx) else 0.0
        elif kind in (ActionType.TRANSMIT_INFORMATION, ActionType.STORE_AND_FORWARD):
            # A grounded obstacle on the active route must be handled before a report is retried. Reporting
            # remains available and regains its normal value as soon as the detour removes the blocker.
            mission = 0.0 if graph.route_blocking_claim_ids else 0.3
            if action.parameters.get("intent") == "REPORT_FINDING":
                # an unreported finding delivered (or durably queued, ch18 'Store-and-forward') is worth its
                # requirement's consequence; the value decays once a report of it was already issued
                reported = ctx.attempts_on(
                    action.target_belief_ids, ActionType.TRANSMIT_INFORMATION
                ) + ctx.attempts_on(action.target_belief_ids, ActionType.STORE_AND_FORWARD)
                information = (
                    self._finding_consequence(action, ctx)
                    * self.cc.report_value
                    * (self.cc.attempt_decay ** (2 * reported))
                )
        elif kind is ActionType.REPLAN:
            mission = 0.2
            if action.parameters.get("reason") == "ROUTE_BLOCKED":
                mission = self.cc.route_replan_value
        return ConsequenceVector(
            mission=mission,
            information=information,
            time=cost.time,
            energy=cost.energy,
            risk=min(1.0, risk),
            communication=cost.communication,
            compute=cost.compute,
            uncertainty_exposure=exposure,
        )

    # ------------------------------------------------------------------ helpers
    def _over(self, a: RequirementAssessment) -> float:
        soft = {"DOMAIN_DEGRADED", "UNCALIBRATED_SOURCE", "EVIDENCE_CONFLICT"}
        if any(i not in soft for i in a.issues):
            return 1.0  # missing / stale / mis-associated / unobserved: nothing reliable is known
        return exposure_of(a.effective_uncertainty) if a.causes else 0.0

    @staticmethod
    def _target(action: ActionProposal, graph: ClaimGraph) -> RequirementAssessment | None:
        rid = action.parameters.get("requirement_id")
        for a in graph.assessments:
            if str(a.requirement_id) == str(rid):
                return a
        return None

    def _dead_end(self, a: RequirementAssessment, ctx: DecisionContext) -> bool:
        """No autonomous resolution left: information attempts exhausted, or OOD (U_E) without alternate evidence.

        A U_E excess that comes only from the uncalibrated-source floor is a calibration gap, not OOD: an
        independent confirming observation is an autonomous path (see ``claims.calibration_only_epistemic``).
        """
        if ctx.information_in_flight(a.target_belief_ids):
            return False
        tries = ctx.attempts_on(a.target_belief_ids, ActionType.REQUEST_INFORMATION)
        if tries >= self.config.max_information_attempts:
            return True
        used = set(ctx.mission.notes.get("modalities_used", []))
        alternates = [m for m in ctx.available_modalities if m not in used]
        return bool(
            a.causes
            and a.causes[0] is UncertaintyType.EPISTEMIC
            and not alternates
            and not a.calibration_only_epistemic
        )

    def _no_autonomous_path(self, open_items: list[RequirementAssessment], ctx: DecisionContext) -> bool:
        """True when a high-consequence open item is a dead end (see ``_dead_end``)."""
        return any(
            a.consequence >= self.config.escalate_consequence_above and self._dead_end(a, ctx)
            for a in open_items
        )

    def _autonomous_path_for_all(self, open_items: list[RequirementAssessment], ctx: DecisionContext) -> bool:
        return all(not self._dead_end(a, ctx) for a in open_items)

    def _finding_consequence(self, action: ActionProposal, ctx: DecisionContext) -> float:
        """Consequence of the requirement(s) the reported beliefs answer (by belief ID or registry identity)."""
        ids = set(action.target_belief_ids)
        entities = {m.world_entity_id for m in ctx.snapshot.messages if m.belief_id in ids} - {None}
        values = [
            r.consequence
            for r in ctx.requirements
            if ids.intersection(r.target_belief_ids) or entities.intersection(r.target_entity_ids)
        ]
        return max(values, default=self.cc.default_finding_consequence)

    def _must_retreat(self, ctx: DecisionContext) -> bool:
        if ctx.system_health is not None and (
            ctx.system_health.overall is HealthLevel.FAULT or ctx.system_health.leak_detected
        ):
            return True
        battery = None if ctx.resource_state is None else ctx.resource_state.battery_fraction
        if battery is not None and battery < self.config.constraints.battery_reserve_fraction:
            return True
        time_left = None if ctx.resource_state is None else ctx.resource_state.time_remaining_s
        return time_left is not None and time_left < self.config.constraints.time_reserve_s
