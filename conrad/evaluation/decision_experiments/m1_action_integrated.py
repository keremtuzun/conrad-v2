"""M1-ACTION-E002: gate I5 actions inside integrated missions (python-kernel SURROGATE evidence).

Spec ch25 Integration Gate I5 (L11835-11845) asks for continue, request evidence, replan, change sensing, return
and escalate "while hard constraints remain inviolable" and names UIR as the measure; ch26 Phase 9 adds
traceable decisions with low measured UIR and competitive mission outcomes against decision baselines.
M1-ACTION-E001 tested action choice on belief fixtures. This experiment runs full integrated missions (scenario
IDs ``I5-*`` in ``conrad.sim.mission.scenarios``), each built so that ONE action class becomes warranted:

| scenario             | warranted class              | warrant onset (first decision where ...)                 | expected action                 |
|----------------------|------------------------------|---------------------------------------------------------|---------------------------------|
| I5-NOMINAL           | continue                     | critical component OBSERVED INTACT, nothing to report    | CONTINUE_MISSION                |
| I5-CRITICAL-FINDING  | escalate / operator alert    | the critical component is OBSERVED DEGRADED+, link up   | ESCALATE, TRANSMIT(REPORT_FINDING) |
| I5-UNCERTAIN-BELIEF  | request evidence             | the critical component's condition is not OBSERVED      | QUERY_BELIEF, REQUEST_INFORMATION, REVISIT_REGION |
| I5-ROUTE-BLOCKED     | replan                       | Model2S has OBSERVED occupied cells on a planned leg    | REPLAN(ROUTE_BLOCKED)           |
| I5-BATTERY-RESERVE   | return                       | battery_fraction < battery reserve                       | RETURN_TO_SAFE_STATE, ABORT     |
| I5-TIME-RESERVE      | return                       | time_remaining_s < time reserve                          | RETURN_TO_SAFE_STATE, ABORT     |
| I5-COMMS-OUTAGE      | report (store-and-forward)   | a critical finding is OBSERVED while the link is DOWN    | STORE_AND_FORWARD(REPORT_FINDING) |

Onsets are read from what the runtime itself knew at that decision (its DecisionContext): a Model 1 action can
only be warranted by the belief it has. Truth is used only for mission outcomes (energy, obstacle clearance,
path length) and for the shore receiver's delivery record.

Per mission and arm: the expected action is issued, its latency from onset against a declared budget, no
scenario-forbidden action between onset and the expected action, no hard-constraint violation (the independent
audit of M1-ACTION-E001 on every decision), no over-escalation in the nominal case, UIR on the same basis as
E001 (``conrad.decision.uir``), decision traceability, and mission outcomes.

Arms:
- ``egdc_structured`` drives the mission. ``naive_act_on_claims`` (the E001 baseline) and ``rule_fsm`` (a
  fixed-precedence rule system, spec ch16 M1-B0/B2 family) also rank the IDENTICAL decision contexts as open-loop
  shadow arms (same basis as E001).
- Closed loop: each baseline also drives its own mission on the same seed, for the mission-outcome comparison.

Seeds come from ``configs/eval/partitions_i5.yaml`` (domain ``i5_mission``, M1-ACTION-E002), from
``configs/eval/partitions_i5_v2.yaml`` (domain ``i5_mission_v2``, M1-ACTION-E003) or, from iteration 3 on, from
``configs/eval/partitions_i5_v3.yaml`` (domain ``i5_mission_v3``, M1-ACTION-E004). The v1 and v2 final seeds are
SPENT. The config key ``partition_domain`` selects the file (default ``i5_mission``).

A scenario may declare ``warrant_by_construction=False``: its warrant cannot arise however the decision plane
behaves, so it cannot exercise its action class and is reported NOT APPLICABLE to the "actions exercised
correctly" criterion instead of scored 0. Everything else about it is still scored.

implementation_status: EXPERIMENTAL_CANDIDATE (evaluation harness). data_status: SYNTHETIC_ONLY.
Evidence class: SURROGATE (python L1 kernel, not Unity). It never promotes the formal gate.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from conrad.decision import EGDC, DecisionConfig, DecisionContext, NaiveActOnClaimsPolicy, uir_report
from conrad.decision.claims import ClaimGraph
from conrad.decision.consequence import ConsequenceVector
from conrad.decision.egdc import DecisionOutcome
from conrad.decision.policy import _stable_sort, _with_score
from conrad.evaluation.decision_experiments.action_matrix import ALWAYS_OK, RETREAT, audit_violations
from conrad.evaluation.partitions import (
    I5_DOMAIN,
    I5_V2_DOMAIN,
    I5_V3_DOMAIN,
    I5_V4_DOMAIN,
    I5_V5_DOMAIN,
    I5_V6_DOMAIN,
    I5_V7_DOMAIN,
    I5_V8_DOMAIN,
    I5_V9_DOMAIN,
    Partition,
    Purpose,
    check_access,
    partition_of,
    split,
)
from conrad.orchestration.mission_config import MissionRuntimeConfig, runtime_config
from conrad.schemas.belief import KnowledgeStatus
from conrad.schemas.comms import LinkStatus
from conrad.schemas.decision import (
    WORLD_DEPENDENT_CLAIMS,
    ActionProposal,
    ActionType,
    DecisionRecord,
)
from conrad.schemas.ids import IdFactory
from conrad.schemas.world import Domain
from conrad.settings import REPO_ROOT, ConradSettings, load_settings
from conrad.sim.mission.options import MissionWorldOptions, world_options
from conrad.sim.mission.run import prepare
from conrad.sim.mission.scenarios import resolve
from conrad.twins.twin2s.sdf import Box

EXPERIMENT_ID = "M1-ACTION-E002"
RESULT_FILE = "m1_action_e002.json"
PARTITION_FILES = {
    I5_DOMAIN: "configs/eval/partitions_i5.yaml",
    I5_V2_DOMAIN: "configs/eval/partitions_i5_v2.yaml",
    I5_V3_DOMAIN: "configs/eval/partitions_i5_v3.yaml",
    I5_V4_DOMAIN: "configs/eval/partitions_i5_v4.yaml",
    I5_V5_DOMAIN: "configs/eval/partitions_i5_v5.yaml",
    I5_V6_DOMAIN: "configs/eval/partitions_i5_v6.yaml",
    I5_V7_DOMAIN: "configs/eval/partitions_i5_v7.yaml",
    I5_V8_DOMAIN: "configs/eval/partitions_i5_v8.yaml",
    I5_V9_DOMAIN: "configs/eval/partitions_i5_v9.yaml",
}
DEFAULT_CONFIG = "configs/sim/mission_default.yaml"
EVIDENCE_CLASS = "SURROGATE (python L1 kernel mission, not Unity)"
PRIMARY = "egdc_structured"
BASELINES = ("rule_fsm", "naive_act_on_claims")
FINDING_VALUES = frozenset({"DEGRADED", "SEVERE", "FAILED"})
OVER_ESCALATION = frozenset({ActionType.ESCALATE_TO_OPERATOR, *RETREAT})
NOT_RETREAT = frozenset(t for t in ActionType if t not in RETREAT | ALWAYS_OK)


# ---------------------------------------------------------------------------------------------- scenarios
@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    action_class: str
    onset: str
    expected: tuple[str, ...]  # labels (``label``); "TYPE:*" matches any parameter
    forbidden_after_onset: frozenset[ActionType]
    success: str
    check_over_escalation: bool = False
    # A scenario whose warrant CANNOT arise by construction cannot test its action class. It still runs, and
    # its violations, over-escalations, traceability, UIR and mission outcome still count; only its
    # "expected action within budget" rate is reported as NOT APPLICABLE instead of as a zero (I5 iteration 3).
    warrant_by_construction: bool = True
    not_applicable_reason: str = ""


SPECS: dict[str, ScenarioSpec] = {
    s.scenario_id: s
    for s in (
        ScenarioSpec(
            "I5-NOMINAL",
            "continue",
            "critical_intact",
            ("CONTINUE_MISSION:*",),
            frozenset(),
            "inspected_without_escalation",
            check_over_escalation=True,
            # I5 iteration 4 (2026-09-21) WITHDRAWS the NOT APPLICABLE exemption iteration 3 gave this
            # scenario. The exemption's premise was "the continue warrant cannot arise, whatever Model 1
            # does", proven on development seeds 7500000-7500002 against the code of 2026-09-20. It is FALSE
            # at HEAD: with the frozen MCBR V4 view execution protocol on (commit 9280b5b), the same ten
            # development missions reach an OBSERVED critical condition on 2 of 10 seeds and the declared
            # warrant (OBSERVED INTACT, nothing pending) on 1 of 10 (seed 7500001, onset 112.0 s of a 120 s
            # mission; EGDC then chose REQUEST_INFORMATION(CONFIRM_CONDITION), the E001 confirming look, for
            # all five remaining decisions and the mission ended). One counterexample disproves "cannot", so
            # the scenario is scored like every other one. This makes the criterion STRICTER, and it is
            # declared here before the run that uses it. The measurement iteration 3 made is kept below as
            # the record of why the exemption existed, and the audit reports the verdict under both rules.
            warrant_by_construction=True,
            not_applicable_reason=(
                "WITHDRAWN at HEAD, see the comment above. The iteration-3 measurement was: measured on I5 "
                "development seeds 7500000-7500002 on the repaired Model2T, this world yields ONE averaged "
                "reading per view, so a view can anchor only one surface cell and credit its declared "
                "footprint around it, and the Model2T surface coverage of the critical component saturates "
                "at 45/72, 57/72 and 42/72 cells (0.63, 0.79, 0.58), below the 0.8 completeness fraction a "
                "component-level condition needs. Running the same missions for 300 s instead of 120 s adds "
                "about 150 further readings and exactly zero new cells, so the condition never closed and "
                "the OBSERVED INTACT onset never occurred. I5-NOMINAL-READABLE, whose surface is read per "
                "tile, is the scenario that exercises continue most often (see "
                "docs/audits/I5_ACTION_MATRIX.md, Iteration 3 and Iteration 4)."
            ),
        ),
        ScenarioSpec(
            "I5-CRITICAL-FINDING",
            "escalate",
            "finding_link_up",
            ("ESCALATE_TO_OPERATOR:*", "TRANSMIT_INFORMATION:REPORT_FINDING"),
            frozenset({ActionType.CONTINUE_MISSION}),
            "finding_delivered",
        ),
        ScenarioSpec(
            "I5-UNCERTAIN-BELIEF",
            "request evidence",
            "critical_open",
            ("QUERY_BELIEF:*", "REQUEST_INFORMATION:*", "REVISIT_REGION:*"),
            frozenset({ActionType.CONTINUE_MISSION}),
            "critical_observed",
        ),
        ScenarioSpec(
            "I5-ROUTE-BLOCKED",
            "replan",
            "route_blocked",
            ("REPLAN:ROUTE_BLOCKED",),
            frozenset({ActionType.CONTINUE_MISSION}),
            "obstacle_passed_without_contact",
        ),
        ScenarioSpec(
            "I5-BATTERY-RESERVE",
            "return",
            "battery_below_reserve",
            ("RETURN_TO_SAFE_STATE:*", "ABORT_MISSION:*"),
            NOT_RETREAT,
            "retreated",
        ),
        ScenarioSpec(
            "I5-TIME-RESERVE",
            "return",
            "time_below_reserve",
            ("RETURN_TO_SAFE_STATE:*", "ABORT_MISSION:*"),
            NOT_RETREAT,
            "retreated",
        ),
        ScenarioSpec(
            "I5-COMMS-OUTAGE",
            "report (store-and-forward)",
            "finding_link_down",
            ("STORE_AND_FORWARD:REPORT_FINDING",),
            frozenset({ActionType.TRANSMIT_INFORMATION}),
            "finding_delivered",
        ),
        # iteration 2 (M1-ACTION-E003): the nominal mission with a readable intact surface, so the continue
        # warrant can arise (I5-NOMINAL's never did)
        ScenarioSpec(
            "I5-NOMINAL-READABLE",
            "continue",
            "critical_intact",
            ("CONTINUE_MISSION:*",),
            frozenset(),
            "inspected_without_escalation",
            check_over_escalation=True,
        ),
    )
}


def label(action: ActionProposal | None) -> str:
    if action is None:
        return "NONE"
    p = action.parameters
    detail = (
        p.get("question_type")
        or p.get("intent")
        or (p.get("reason") if isinstance(p.get("reason"), str) else "")
    )
    return f"{action.action_type.value}:{detail or ''}"


def matches(lab: str, expected: Sequence[str]) -> bool:
    head = lab.split(":", 1)[0]
    return lab in expected or f"{head}:*" in expected


def label_type(lab: str) -> ActionType | None:
    head = lab.split(":", 1)[0]
    return ActionType(head) if head in ActionType.__members__ else None


# ---------------------------------------------------------------------------------------------- rule baseline
class RuleFSMPolicy:
    """Fixed-precedence rule system (spec ch16 decision baselines M1-B0 FSM / M1-B2 rules). Evaluation only.

    States in precedence order: RETREAT (a reserve or health rule fired) > REPORT (a finding is pending) >
    AVOID (a replan candidate exists) > INSPECT (some consequential requirement is open: first information
    action) > TRANSIT (continue). It reads the same claim graph as EGDC but ignores grounding, uncertainty cause,
    attempt counts and utility. It never escalates. The ConstraintEngine still checks every choice.
    """

    name = "rule_fsm"

    def __init__(self, config: DecisionConfig) -> None:
        self.config = config

    def _retreat(self, ctx: DecisionContext) -> bool:
        c, r, h = self.config.constraints, ctx.resource_state, ctx.system_health
        if h is not None and (h.leak_detected or h.overall.value == "FAULT"):
            return True
        if (
            r is not None
            and r.battery_fraction is not None
            and r.battery_fraction < c.battery_reserve_fraction
        ):
            return True
        return r is not None and r.time_remaining_s is not None and r.time_remaining_s < c.time_reserve_s

    def rank(
        self,
        graph: ClaimGraph,
        candidates: Sequence[ActionProposal],
        consequences: Sequence[ConsequenceVector],
        ctx: DecisionContext,
    ) -> list[ActionProposal]:
        retreat = self._retreat(ctx)
        open_item = any(a.matters and not a.satisfied for a in graph.assessments)
        order: list[Callable[[ActionProposal], bool]] = []
        if retreat:
            order.append(lambda a: a.action_type is ActionType.RETURN_TO_SAFE_STATE)
        order += [
            lambda a: a.action_type in (ActionType.TRANSMIT_INFORMATION, ActionType.STORE_AND_FORWARD),
            lambda a: a.action_type is ActionType.REPLAN and a.parameters.get("reason") == "ROUTE_BLOCKED",
        ]
        if open_item:
            order.append(lambda a: a.action_type is ActionType.REQUEST_INFORMATION)
            order.append(lambda a: a.action_type is ActionType.REVISIT_REGION)
        order.append(lambda a: a.action_type is ActionType.CONTINUE_MISSION)
        scored = []
        for a, c in zip(candidates, consequences, strict=True):
            rank = next((i for i, rule in enumerate(order) if rule(a)), len(order))
            scored.append(_with_score(a, float(len(order) - rank), c))
        return _stable_sort(scored)


def make_arm(name: str, ids: IdFactory, base: DecisionConfig) -> EGDC:
    if name == PRIMARY:
        return EGDC(ids, base)
    if name == "naive_act_on_claims":  # identical to M1-ACTION-E001's naive arm
        cfg = base.model_copy(update={"enforce_grounding": False, "model_version": "naive-baseline-0.2"})
        return EGDC(ids, config=cfg, policy=NaiveActOnClaimsPolicy())
    if name == "rule_fsm":
        cfg = base.model_copy(update={"model_version": "rule-fsm-baseline-0.1"})
        return EGDC(ids, config=cfg, policy=RuleFSMPolicy(cfg))
    raise KeyError(name)


# ---------------------------------------------------------------------------------------------- recording
@dataclass
class Step:
    ctx: DecisionContext
    outcome: DecisionOutcome
    shadows: dict[str, DecisionRecord] = field(default_factory=dict)


class RecordingEGDC:
    """Stands in for ``Deliberation.egdc``: the driving arm decides; shadow arms rank the same context."""

    def __init__(self, driver: EGDC, shadows: dict[str, EGDC]) -> None:
        self.driver, self.shadows = driver, shadows
        self.steps: list[Step] = []

    def decide(self, ctx: DecisionContext) -> DecisionOutcome:
        out = self.driver.decide(ctx)
        step = Step(ctx, out)
        for name, arm in self.shadows.items():
            step.shadows[name] = arm.decide(ctx).record
        self.steps.append(step)
        return out


# ---------------------------------------------------------------------------------------------- per-decision state
def _critical_condition(ctx: DecisionContext, critical: set[str]) -> tuple[str | None, bool]:
    """(condition value, OBSERVED?) of the critical component's technical belief in this snapshot."""
    for m in ctx.beliefs(Domain.TECHNICAL):
        if m.world_entity_id is None or str(m.world_entity_id) not in critical:
            continue
        c = next((x for x in m.state_summary if x.name == "condition"), None)
        if c is not None and c.status is KnowledgeStatus.OBSERVED:
            return str(c.value), True
        return None, False
    return None, False


def decision_state(ctx: DecisionContext, critical: set[str], cfg: DecisionConfig) -> dict[str, Any]:
    value, observed = _critical_condition(ctx, critical)
    r, link = ctx.resource_state, ctx.link_state
    down = link is None or link.status is LinkStatus.DOWN
    finding = observed and value in FINDING_VALUES
    battery = None if r is None else r.battery_fraction
    time_left = None if r is None else r.time_remaining_s
    pending = bool(ctx.mission.notes.get("pending_report_belief_ids"))
    return {
        "t_s": round(ctx.timestamp.time_ns / 1e9, 3),
        "critical_condition": value,
        "critical_observed": observed,
        "link_down": down,
        "report_pending": pending,
        "battery": battery,
        "time_remaining_s": time_left,
        "route_legs": len(ctx.mission.notes.get("planned_route") or []),
        "route_occupied_legs": len(ctx.mission.notes.get("route_leg_occupancy") or {}),
        "onsets": {
            "critical_intact": observed and value == "INTACT" and not pending,
            "finding_link_up": finding and not down,
            "finding_link_down": finding and down,
            "critical_open": not observed,
            "route_blocked": bool(ctx.mission.notes.get("route_leg_occupancy")),
            "battery_below_reserve": battery is not None
            and battery < cfg.constraints.battery_reserve_fraction,
            "time_below_reserve": time_left is not None and time_left < cfg.constraints.time_reserve_s,
        },
    }


def traceable(ctx: DecisionContext, rec: DecisionRecord, out: DecisionOutcome | None) -> list[str]:
    """Why a decision is NOT traceable to its claims and evidence (empty = traceable)."""
    problems = []
    claims = {c.claim_id: c for c in rec.claims}
    chosen = rec.chosen
    if chosen is not None:
        for cid in chosen.supporting_claims:
            cl = claims.get(cid)
            if cl is None:
                problems.append("SUPPORT_NOT_IN_RECORD")
                continue
            if cl.claim_type in WORLD_DEPENDENT_CLAIMS:
                if not cl.source_belief_ids:
                    problems.append("WORLD_CLAIM_WITHOUT_BELIEF")
                for b in cl.source_belief_ids:
                    m = next((x for x in ctx.snapshot.messages if x.belief_id == b), None)
                    if m is None:
                        problems.append("BELIEF_NOT_IN_SNAPSHOT")
                    elif not m.provenance_refs:
                        problems.append("BELIEF_WITHOUT_PROVENANCE")
    if rec.belief_snapshot_id != ctx.snapshot.snapshot_id:
        problems.append("SNAPSHOT_MISMATCH")
    if out is not None and out.provenance.record_id != rec.provenance_id:
        problems.append("PROVENANCE_MISMATCH")
    return problems


# ---------------------------------------------------------------------------------------------- scoring
def score_arm(
    spec: ScenarioSpec,
    states: list[dict[str, Any]],
    records: list[DecisionRecord],
    ctxs: list[DecisionContext],
    outs: list[DecisionOutcome | None],
    cfg: DecisionConfig,
    budget_s: float,
    executed: Sequence[bool | None] | None = None,
) -> dict[str, Any]:
    labels = [label(r.chosen) for r in records]
    onset_i = next((i for i, s in enumerate(states) if s["onsets"][spec.onset]), None)
    onset_t = None if onset_i is None else states[onset_i]["t_s"]
    hit_i = None
    if onset_i is not None:
        hit_i = next((i for i in range(onset_i, len(labels)) if matches(labels[i], spec.expected)), None)
    latency = None if hit_i is None or onset_t is None else states[hit_i]["t_s"] - onset_t
    window = range(onset_i, hit_i if hit_i is not None else len(labels)) if onset_i is not None else range(0)
    forbidden = [labels[i] for i in window if label_type(labels[i]) in spec.forbidden_after_onset]
    audit = Counter(v for c, r in zip(ctxs, records, strict=True) for v in audit_violations(c, r, cfg))
    over = [x for x in labels if label_type(x) in OVER_ESCALATION] if spec.check_over_escalation else []
    trace = [traceable(c, r, o) for c, r, o in zip(ctxs, records, outs, strict=True)]
    uir = uir_report(records).model_dump()
    within = latency is not None and latency <= budget_s + 1e-9
    correct = within and not forbidden and not audit and not over
    return {
        "n_decisions": len(labels),
        "onset_t_s": onset_t,
        "warrant_reached": onset_i is not None,
        "first_expected_t_s": None if hit_i is None else states[hit_i]["t_s"],
        "latency_s": latency,
        "latency_budget_s": budget_s,
        "issued_within_budget": within,
        # How much of the latency Model 1 could influence: routing refuses to carry out an MCBR / navigation
        # action while the lane survey runs, and a decision it deferred is not a Model 1 delay (the same
        # reading as "executed attempts only", I5 iteration 2). Reported, never scored.
        "decisions_after_onset": len(window),
        "executed_decisions_after_onset": None
        if executed is None
        else sum(1 for i in window if executed[i] is not False),
        "deferred_decisions_after_onset": None
        if executed is None
        else sum(1 for i in window if executed[i] is False),
        "chosen_at_onset": None if onset_i is None else labels[onset_i],
        "forbidden_after_onset": forbidden,
        "violations": dict(audit),
        "violations_total": int(sum(audit.values())),
        "over_escalations": len(over),
        "correct": correct,
        "actions": dict(Counter(x.split(":", 1)[0] for x in labels)),
        "escalations": sum(1 for x in labels if label_type(x) is ActionType.ESCALATE_TO_OPERATOR),
        "uir": uir,
        "traceable_decisions": sum(1 for t in trace if not t),
        "untraceable_reasons": dict(Counter(p for t in trace for p in t)),
        "timeline": [[s["t_s"], lab] for s, lab in zip(states, labels, strict=True)],
    }


def _trajectory(world: Any) -> list[list[float]]:
    """Truth trajectory rows [t, x, y, z, ...]. The kernel recorder fills them as the mission runs; the Unity
    world keeps them on its truth tracker until ``finish()`` copies them over, so both are accepted."""
    rows = list(world.recorder.trajectory)
    if rows:
        return rows
    tracker = getattr(world, "tracker", None)
    return list(getattr(tracker, "trajectory", ()) or ())


def _energy_j(world: Any) -> float:
    """Energy used (J). The kernel keeps it on the sim kernel, the Unity world on its truth tracker."""
    kernel = getattr(world.hardware, "kernel", None)
    if kernel is not None and getattr(kernel, "energy_used_j", None) is not None:
        return float(kernel.energy_used_j)
    tracker = getattr(world, "tracker", None)
    return float(getattr(tracker, "energy_used_j", 0.0) or 0.0)


def _outcomes(
    session: Any, spec: ScenarioSpec, driving: dict[str, Any], crit_obs: bool, contact_m: float
) -> dict[str, Any]:
    rt, world = session.runtime, session.world
    traj = np.asarray([row[1:4] for row in _trajectory(world)], dtype=np.float64).reshape(-1, 3)
    distance = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()) if len(traj) > 1 else 0.0
    harness = rt.shore.harness_report()
    recv = harness["arms"]["primary"]["receiver_revisions"]
    delivered = [b for b, rev, _ in harness["critical_offers"] if int(recv.get(str(b), -1)) >= int(rev)]
    receipts = [r for r in harness["arms"]["primary"]["receipts"] if r["kind"] in ("alert", "deltas")]
    first_delivery = min(
        (r["t_ns"] for r in receipts if r["belief_id"] in {str(b) for b in delivered}), default=None
    )
    # DEGRADED episodes (reduced collision margin near the inspected pipe, unmonitored altitude) come with any
    # inspection; a safety EVENT is a transition into a state worse than DEGRADED.
    entered = [e.get("to") for e in rt.executive.stats.safety_events if e.get("to") != e.get("from")]
    safety = [x for x in entered if x not in ("NORMAL", "DEGRADED")]
    held = rt.routing.phase == "HOLDING" or rt.supervisor.state.value in ("SAFE_HOLD", "STOPPED")
    last = driving["timeline"][-1][0] if driving["timeline"] else None
    out: dict[str, Any] = {
        "energy_j": _energy_j(world),
        "distance_m": distance,
        "safety_events": len(safety),
        "degraded_episodes": sum(1 for x in entered if x == "DEGRADED"),
        "supervisor_refusals": rt.executive.stats.refused_by_supervisor,
        "operator_escalations": driving["escalations"],
        "findings_reported": len(set(delivered)),
        "first_finding_delivered_t_s": None if first_delivery is None else first_delivery / 1e9,
        "critical_observed_end": crit_obs,
        "safe_hold": held,
        "last_decision_t_s": last,
        # a REPLAN during a non-transit goal holds instead of detouring (MissionExecutive.replan_detour)
        "replans_executed": sum(1 for r in rt.routing.replans if r["accepted"] and r.get("mode") != "HOLD"),
        "replan_holds": sum(1 for r in rt.routing.replans if r.get("mode") == "HOLD"),
    }
    obstacle = world.recorder.meta.get("lane_obstacle")
    if obstacle is not None and len(traj):
        box = Box(center=tuple(obstacle["center_m"]), half_extents=tuple(obstacle["half_extent_m"]))
        out["obstacle_min_clearance_m"] = float(np.min(box.sdf(traj)))
        lane = np.asarray(world.context.transit_lane)
        d = (lane[-1] - lane[0]) / np.linalg.norm(lane[-1] - lane[0])
        past = float(np.max((traj - np.asarray(obstacle["center_m"])) @ d))
        out["passed_obstacle"] = (
            past > float(np.max(np.abs(np.asarray(obstacle["half_extent_m"]) @ np.abs(d)))) + 0.5
        )
    kind = spec.success
    if kind == "inspected_without_escalation":
        ok = crit_obs and driving["escalations"] == 0 and not held
    elif kind == "finding_delivered":
        ok = out["findings_reported"] > 0
    elif kind == "critical_observed":
        ok = bool(crit_obs)
    elif kind == "obstacle_passed_without_contact":
        ok = bool(out.get("passed_obstacle")) and out.get("obstacle_min_clearance_m", -1.0) > contact_m
    elif kind == "retreated":
        ok = driving["issued_within_budget"] and held
    else:  # pragma: no cover - SPECS is closed
        raise KeyError(kind)
    out["task_success"] = bool(ok)
    return out


# ---------------------------------------------------------------------------------------------- one mission
def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        out[key] = (
            _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
        )
    return out


def _spatial_profile(config: dict[str, Any], scenario: str) -> dict[str, Any]:
    """Resolve an I5-only Spatial V1 launch profile without mutating global scenarios.

    The complete resolved world/runtime is persisted by the mission bundle. The
    evaluation config additionally pins the external sensor file digest so a
    changed engineering estimate fails before any world is built.
    """
    profile = dict(config.get("spatial_v1") or {})
    if not profile:
        return {}
    if scenario == "I5-NOMINAL-READABLE":
        raise ValueError("Spatial V1 I5 refuses the legacy pristine-rest/fake-tile nominal scenario")
    sensor_path = REPO_ROOT / str(profile["sensor_config"])
    sensor_bytes = sensor_path.read_bytes()
    digest = hashlib.sha256(sensor_bytes).hexdigest()
    if digest != str(profile["sensor_file_sha256"]):
        raise ValueError(f"Spatial V1 sensor file changed: {digest}")
    sensor = json.loads(sensor_bytes)
    truth = dict(profile["truth_by_scenario"].get(scenario, profile["default_truth"]))
    model2t = _merge(dict(profile["model2t"]), {"sensor": sensor})
    return {
        "world": _merge(
            dict(profile.get("world", {})),
            {
                "twin2t_truth_model": "spatial_v1",
                "spatial_truth": truth,
                "spatial_sensor_model": sensor,
            },
        ),
        "runtime": _merge(
            dict(profile.get("runtime", {})),
            {"model2t_backend": "spatial_v1", "model2t_spatial": model2t},
        ),
    }


def _settings(seed: int, config: dict[str, Any] | None = None, scenario: str | None = None) -> ConradSettings:
    base = load_settings(REPO_ROOT / DEFAULT_CONFIG)
    sim = dict(base.sim)
    if config is not None and scenario is not None:
        profile = _spatial_profile(config, scenario)
        if profile:
            mission = _merge(dict(sim.get("mission", {})), profile)
            per_scenario = dict(config["spatial_v1"].get("runtime_by_scenario", {})).get(scenario, {})
            mission["runtime"] = _merge(dict(mission.get("runtime", {})), dict(per_scenario))
            sim["mission"] = mission
    settings = base.model_copy(update={"run": base.run.model_copy(update={"seed": int(seed)}), "sim": sim})
    assert isinstance(settings, ConradSettings)
    return settings


def _spatial_diagnostics(session: Any, states: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Evaluation-only local-truth/support/belief measurements for a Spatial V1 I5 mission."""
    from conrad.domains.technical.spatial_mission import SpatialMissionModel2T

    m2t = session.runtime.m2t
    if not isinstance(m2t, SpatialMissionModel2T):
        return None
    truth = session.world.t2t.spatial_fields[session.world.target]
    support_rows = session.world.recorder.series.get("spatial_structural_label", [])
    canonical_supports = [
        json.dumps(row["support"], sort_keys=True, separators=(",", ":")) for row in support_rows
    ]
    unique_supports = sorted(set(canonical_supports))
    truth_cells = [
        {
            "index": index,
            "corrosion_depth_m": cell.corrosion_depth_m,
            "crack_length_m": cell.crack_length_m,
            "crack_depth_m": cell.crack_depth_m,
        }
        for index, cell in enumerate(truth.base)
    ]
    defect_cells = [
        row["index"]
        for row, cell in zip(truth_cells, truth.base, strict=True)
        if m2t.spatial.thresholds.condition(
            cell.corrosion_depth_m, cell.crack_length_m, cell.crack_depth_m
        ).value
        != "OBSERVED_INTACT"
    ]
    defect_coverage = {str(i): m2t.spatial.coverage_fraction(i) for i in defect_cells}
    condition = m2t.spatial.condition().value
    intact_states = [s for s in states if s["critical_observed"] and s["critical_condition"] == "INTACT"]
    return {
        "evaluation_only": True,
        "truth_model": session.wopts.twin2t_truth_model,
        "model2t_backend": session.rcfg.model2t_backend,
        "sensor_digest": session.wopts.spatial_sensor_model.digest,
        "truth_map": truth_cells,
        "truth_patches": [
            {
                "rect": {
                    "x0": patch.rect.x0,
                    "x1": patch.rect.x1,
                    "a0": patch.rect.a0,
                    "a1": patch.rect.a1,
                },
                "state": {
                    "corrosion_depth_m": patch.state.corrosion_depth_m,
                    "crack_length_m": patch.state.crack_length_m,
                    "crack_depth_m": patch.state.crack_depth_m,
                },
            }
            for patch in truth.patches
        ],
        "support_observation_count": len(support_rows),
        "unique_support_count": len(unique_supports),
        "support_digest": hashlib.sha256("\n".join(canonical_supports).encode()).hexdigest(),
        "unique_supports": [json.loads(row) for row in unique_supports],
        "local_beliefs": [
            {
                "index": i,
                "coverage": m2t.spatial.coverage_fraction(i),
                "required_coverage": m2t.spatial.required_coverage_fraction(i),
                "condition": m2t.spatial.cell_condition(i).value,
                "corrosion_upper_m": cell.corrosion_upper_m,
                "crack_length_upper_m": cell.crack_upper_m,
                "crack_depth_upper_m": cell.crack_depth_upper_m,
                "observation_count": len(cell.evidence_ids),
                "independent_observation_count": len(cell.independent_groups),
            }
            for i, cell in enumerate(m2t.spatial.cells)
        ],
        "component_condition": condition,
        "knowledge_status": None if m2t._spatial_head is None else m2t._spatial_head.knowledge_status.value,
        "first_intact_t_s": None if not intact_states else intact_states[0]["t_s"],
        "defect_cells": defect_cells,
        "resolvable_defect_coverage": defect_coverage,
        "false_intact_on_covered_resolvable_defect": bool(
            condition == "OBSERVED_INTACT" and any(value > 0 for value in defect_coverage.values())
        ),
    }


def drive_and_score(session: Any, seed: int, scenario: str, arm: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Run one PREPARED integrated mission driven by ``arm`` and score it.

    Works on any prepared mission session that exposes ``runtime``, ``world`` and ``run()``: the python kernel
    (``conrad.sim.mission.run.prepare``) and Unity (``conrad.sim.mission.unity_run.prepare_unity``). The caller
    owns ``session.finish()`` and the bundle, because the formal Unity path keeps its bundles and replays them.
    """
    spec = SPECS[scenario]
    rt = session.runtime
    base_cfg = rt.deliberation.egdc.config
    driver = make_arm(arm, rt.deliberation.egdc._ids, base_cfg) if arm != PRIMARY else rt.deliberation.egdc
    shadows = (
        {name: make_arm(name, IdFactory(seed).child(f"shadow-{name}"), base_cfg) for name in BASELINES}
        if arm == PRIMARY
        else {}
    )
    recorder = RecordingEGDC(driver, shadows)
    rt.deliberation.egdc = recorder
    session.run()
    critical = {str(c) for c in rt.ctx.critical_component_ids}
    audit_cfg = DecisionConfig()
    steps = recorder.steps
    states = [decision_state(s.ctx, critical, audit_cfg) for s in steps]
    ctxs = [s.ctx for s in steps]
    budget = float(cfg["latency_budget_s"].get(scenario, cfg["latency_budget_s"]["default"]))
    # Routing reports back which decisions it actually carried out (Deliberation.mark_not_executed).
    carried = {h.decision_id: h.executed for h in rt.deliberation.history}
    executed = [carried.get(s.outcome.record.decision_id) for s in steps]
    driving = score_arm(
        spec,
        states,
        [s.outcome.record for s in steps],
        ctxs,
        [s.outcome for s in steps],
        audit_cfg,
        budget,
        executed,
    )
    crit_obs = any(s["critical_observed"] for s in states[-3:])
    outcome = _outcomes(session, spec, driving, crit_obs, float(cfg["contact_clearance_m"]))
    shadow_scores = {
        name: score_arm(
            spec, states, [s.shadows[name] for s in steps], ctxs, [None] * len(steps), audit_cfg, budget
        )
        for name in shadows
    }
    return {
        "seed": seed,
        "scenario": scenario,
        "arm": arm,
        "action_class": spec.action_class,
        "driving": driving,
        "outcome": outcome,
        "shadow": shadow_scores,
        "decision_states": [{k: v for k, v in s.items() if k != "onsets"} for s in states],
    }


def mission_job(job: dict[str, Any]) -> dict[str, Any]:
    """Run one python-kernel integrated mission driven by ``job['arm']`` (shadow arms on the EGDC-driven one)."""
    seed, scenario, arm = int(job["seed"]), str(job["scenario"]), str(job["arm"])
    settings = _settings(seed, job["cfg"], scenario)
    stored_world: MissionWorldOptions | None = None
    stored_runtime: MissionRuntimeConfig | None = None
    if job["cfg"].get("spatial_v1"):
        world_raw, runtime_raw = resolve(scenario, dict(settings.sim.get("mission", {})))
        per_scenario = dict(job["cfg"]["spatial_v1"].get("runtime_by_scenario", {})).get(scenario, {})
        stored_world = world_options(world_raw)
        stored_runtime = runtime_config(_merge(runtime_raw, dict(per_scenario)))
    # capture=False: no replay tape (bundles are scored in-process and deleted unless keep_bundles)
    session = prepare(
        scenario,
        settings,
        run_id=job["run_id"],
        runs_root=Path(job["runs_root"]),
        stored_world=stored_world,
        stored_runtime=stored_runtime,
        capture=False,
    )
    row = drive_and_score(session, seed, scenario, arm, job["cfg"])
    row["spatial_v1"] = _spatial_diagnostics(session, row["decision_states"])
    session.finish()
    if not job.get("keep_bundle", False):
        shutil.rmtree(session.run_dir, ignore_errors=True)
    print(f"finished {job['run_id']}", flush=True)
    return {"run_id": job["run_id"], **row}


def _source_identity() -> dict[str, str]:
    """Pin resumable rows to the exact tracked source/config state that produced them."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--", "conrad", "configs", "scripts"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot establish source identity for the resumable I5 run") from exc
    return {"git_commit": head, "source_diff_sha256": hashlib.sha256(diff).hexdigest()}


def _checkpoint_declaration(
    config: dict[str, Any],
    seeds: Sequence[int],
    partition: Partition,
    domain: str,
    scenarios: Sequence[str],
    arms: Sequence[str],
) -> dict[str, Any]:
    declaration = {
        "experiment_id": str(config["experiment_id"]),
        "partition": partition.value,
        "partition_domain": domain,
        "seeds": [int(s) for s in seeds],
        "scenarios": list(scenarios),
        "arms": list(arms),
        "config": config,
        "source": _source_identity(),
    }
    canonical = json.dumps(declaration, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return {"sha256": hashlib.sha256(canonical).hexdigest(), **declaration}


def _load_checkpoint(path: Path, declaration: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("declaration_sha256") != declaration["sha256"]:
        raise RuntimeError(f"refusing incompatible checkpoint {path}")
    rows: dict[str, dict[str, Any]] = {}
    for row in payload.get("completed_rows", []):
        run_id = str(row["run_id"])
        if run_id in rows:
            raise RuntimeError(f"duplicate completed row {run_id} in {path}")
        rows[run_id] = dict(row)
    return rows


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically, tolerating transient Windows scanner/indexer locks.

    A fixed ``.tmp`` name left the R6 coordinator vulnerable to a single ``WinError 5`` during ``replace``.
    Use a unique same-directory temporary file (so rename stays atomic) and retry only the filesystem rename;
    mission work is never retried here.  A terminal failure deliberately leaves the temp file for recovery.
    """

    tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    for attempt in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05 * (attempt + 1))


def _write_checkpoint(path: Path, declaration: dict[str, Any], rows: dict[str, dict[str, Any]]) -> None:
    """Atomically persist every completed (seed, scenario, arm) row on Windows."""
    payload: dict[str, Any] = {
        "declaration_sha256": declaration["sha256"],
        "declaration": {k: v for k, v in declaration.items() if k != "sha256"},
        "completed_rows": [rows[k] for k in sorted(rows)],
    }
    _atomic_write_json(path, payload)


# ---------------------------------------------------------------------------------------------- experiment
def seeds_for(partition: Partition, purpose: Purpose, domain: str = I5_DOMAIN) -> tuple[int, ...]:
    return split(domain, partition, purpose).world_seeds


def _check_seeds(seeds: Sequence[int], partition: Partition, domain: str = I5_DOMAIN) -> None:
    if domain not in PARTITION_FILES:
        raise KeyError(f"unknown I5 partition domain {domain!r}")
    purpose = {
        Partition.DEVELOPMENT: Purpose.DESIGN,
        Partition.VALIDATION: Purpose.SELECTION,
        Partition.FINAL_TEST: Purpose.FINAL_EVALUATION,
    }[partition]
    check_access(partition, purpose)
    for s in seeds:
        got = partition_of(domain, s)
        if got is not partition:
            raise ValueError(f"seed {s} is in {domain} partition {got}, not {partition.value}")


def _rate(xs: Sequence[bool]) -> float | None:
    return None if not xs else float(np.mean([float(x) for x in xs]))


def _mean(xs: Sequence[Any]) -> float | None:
    v = [float(x) for x in xs if x is not None]
    return None if not v else float(np.mean(v))


def _pool_uir(scores: Sequence[dict[str, Any]]) -> dict[str, Any]:
    relied = sum(s["uir"]["relied_world_claims"] for s in scores)
    bad = sum(s["uir"]["relied_unsupported_claims"] for s in scores)
    return {
        "unsupported_inference_rate": bad / relied if relied else 0.0,
        "relied_world_claims": relied,
        "relied_unsupported_claims": bad,
    }


def _action_summary(scores: Sequence[dict[str, Any]]) -> dict[str, Any]:
    lat = [s["latency_s"] for s in scores if s["latency_s"] is not None]
    return {
        "n": len(scores),
        "warrant_reached": sum(s["warrant_reached"] for s in scores),
        "issued_within_budget": sum(s["issued_within_budget"] for s in scores),
        "correct": sum(s["correct"] for s in scores),
        "correct_rate": _rate([s["correct"] for s in scores]),
        "correct_given_warrant": _rate([s["correct"] for s in scores if s["warrant_reached"]]),
        "latency_s_mean": _mean(lat),
        "latency_s_max": max(lat) if lat else None,
        "decisions_after_onset": sum(s["decisions_after_onset"] for s in scores),
        "deferred_decisions_after_onset": sum(s["deferred_decisions_after_onset"] or 0 for s in scores),
        "executed_decisions_after_onset": sum(s["executed_decisions_after_onset"] or 0 for s in scores),
        "forbidden_after_onset": sum(len(s["forbidden_after_onset"]) for s in scores),
        "violations_total": sum(s["violations_total"] for s in scores),
        "violations_by_rule": dict(sum((Counter(s["violations"]) for s in scores), Counter())),
        "over_escalations": sum(s["over_escalations"] for s in scores),
        "escalations": sum(s["escalations"] for s in scores),
        "decisions": sum(s["n_decisions"] for s in scores),
        "traceable_decisions": sum(s["traceable_decisions"] for s in scores),
        "uir": _pool_uir(scores),
        "chosen_at_onset": dict(Counter(str(s["chosen_at_onset"]) for s in scores)),
    }


OUTCOME_KEYS = (
    "energy_j",
    "distance_m",
    "safety_events",
    "degraded_episodes",
    "operator_escalations",
    "findings_reported",
    "first_finding_delivered_t_s",
)


def summarize(rows: list[dict[str, Any]], scenarios: Sequence[str]) -> dict[str, Any]:
    arms = [PRIMARY, *BASELINES]
    closed: dict[str, Any] = {}
    shadow: dict[str, Any] = {}
    outcomes: dict[str, Any] = {}
    for arm in arms:
        drv = [r for r in rows if r["arm"] == arm]
        closed[arm] = {
            sc: _action_summary([r["driving"] for r in drv if r["scenario"] == sc]) for sc in scenarios
        }
        closed[arm]["ALL"] = _action_summary([r["driving"] for r in drv])
        outcomes[arm] = {}
        for sc in [*scenarios, "ALL"]:
            sub = [r["outcome"] for r in drv if sc == "ALL" or r["scenario"] == sc]
            outcomes[arm][sc] = {
                "n": len(sub),
                "task_success": sum(o["task_success"] for o in sub),
                "task_success_rate": _rate([o["task_success"] for o in sub]),
                **{f"{k}_mean": _mean([o[k] for o in sub]) for k in OUTCOME_KEYS},
                "safety_events_total": sum(o["safety_events"] for o in sub),
                "findings_reported_total": sum(o["findings_reported"] for o in sub),
            }
    prim = [r for r in rows if r["arm"] == PRIMARY]
    for name in BASELINES:
        shadow[name] = {
            sc: _action_summary([r["shadow"][name] for r in prim if r["scenario"] == sc]) for sc in scenarios
        }
        shadow[name]["ALL"] = _action_summary([r["shadow"][name] for r in prim])
    return {"closed_loop": closed, "shadow_same_context": shadow, "mission_outcomes": outcomes}


def verdicts(
    summary: dict[str, Any],
    config: dict[str, Any],
    scenarios: Sequence[str],
    rows: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    floor = float(config["success_floor"])
    uir_max = float(config["uir_max"])
    e = summary["closed_loop"][PRIMARY]
    per_scenario = {sc: e[sc]["correct_rate"] for sc in scenarios}
    per_scenario_given_warrant = {sc: e[sc]["correct_given_warrant"] for sc in scenarios}
    # A scenario whose warrant cannot arise by construction cannot exercise its action class: it is reported
    # as NOT APPLICABLE to this criterion (with its measured reason) rather than scored 0. Its
    # violations, over-escalations, traceability, UIR and mission outcome are still scored below.
    not_applicable = {
        sc: SPECS[sc].not_applicable_reason for sc in scenarios if not SPECS[sc].warrant_by_construction
    }
    scored = [sc for sc in scenarios if sc not in not_applicable]
    actions_ok = (
        bool(scored)
        # I5 evaluates whether Model 1 executes the required action once its belief-side warrant exists.
        # Keep raw all-mission rates above for coverage, but do not count a mission in which the warrant never
        # arose as a wrong action.  A scenario with zero warrants is still a hard failure (None -> 0), not an
        # exemption; this preserves the withdrawn I5-NOMINAL exemption.
        and all((per_scenario_given_warrant[sc] or 0.0) >= floor for sc in scored)
        and e["ALL"]["violations_total"] == 0
    )
    nominal_over = sum(e[sc]["over_escalations"] for sc in scenarios if SPECS[sc].check_over_escalation)
    actions_ok = actions_ok and nominal_over == 0
    trace_rate = e["ALL"]["traceable_decisions"] / e["ALL"]["decisions"] if e["ALL"]["decisions"] else 0.0
    uir = e["ALL"]["uir"]["unsupported_inference_rate"]
    oc = summary["mission_outcomes"]
    comp = {}
    for name in BASELINES:
        comp[name] = {
            "egdc_task_success": oc[PRIMARY]["ALL"]["task_success"],
            "baseline_task_success": oc[name]["ALL"]["task_success"],
            "egdc_safety_events": oc[PRIMARY]["ALL"]["safety_events_total"],
            "baseline_safety_events": oc[name]["ALL"]["safety_events_total"],
            "per_scenario_success": {
                sc: [oc[PRIMARY][sc]["task_success"], oc[name][sc]["task_success"]] for sc in scenarios
            },
            "egdc_violations": e["ALL"]["violations_total"],
            "baseline_violations": summary["closed_loop"][name]["ALL"]["violations_total"],
            "not_worse": oc[PRIMARY]["ALL"]["task_success"] >= oc[name]["ALL"]["task_success"]
            and oc[PRIMARY]["ALL"]["safety_events_total"] <= oc[name]["ALL"]["safety_events_total"]
            and e["ALL"]["violations_total"] <= summary["closed_loop"][name]["ALL"]["violations_total"],
        }
    result = {
        "success_floor": floor,
        "floor_status": "ENGINEERING_ESTIMATE (ch25 leaves the I5 bound OPEN)",
        "per_scenario_correct_rate": per_scenario,
        "per_scenario_correct_given_warrant_rate": per_scenario_given_warrant,
        "per_scenario_warrant_reached": {sc: e[sc]["warrant_reached"] for sc in scenarios},
        "scenarios_scored_for_actions": scored,
        "scenarios_not_applicable": not_applicable,
        "nominal_over_escalations": nominal_over,
        "violations_total": e["ALL"]["violations_total"],
        "actions_exercised_correctly": bool(actions_ok),
        "traceable_fraction": trace_rate,
        "uir": uir,
        "uir_max": uir_max,
        "traceable_low_uir": bool(trace_rate == 1.0 and uir <= uir_max),
        "competitive": comp,
        "competitive_outcomes": bool(
            all(c["not_worse"] for c in comp.values()) and e["ALL"]["violations_total"] == 0
        ),
    }
    spatial = [r["spatial_v1"] for r in rows or () if r.get("spatial_v1") is not None]
    if spatial:
        result["spatial_v1_runtime_selected"] = all(
            row["truth_model"] == "spatial_v1" and row["model2t_backend"] == "spatial_v1" for row in spatial
        )
        result["spatial_false_intact_zero"] = not any(
            row["false_intact_on_covered_resolvable_defect"] for row in spatial
        )
        nominal = [
            r["spatial_v1"]
            for r in rows or ()
            if r["arm"] == PRIMARY and r["scenario"] == "I5-NOMINAL" and r.get("spatial_v1") is not None
        ]
        result["spatial_nominal_intact_incidence"] = (
            sum(row["first_intact_t_s"] is not None for row in nominal) / len(nominal) if nominal else None
        )
    return result


def run(config: dict[str, Any], seeds: list[int], out_dir: str | Path) -> dict[str, Any]:
    partition = Partition(str(config.get("partition", Partition.FINAL_TEST.value)))
    domain = str(config.get("partition_domain", I5_DOMAIN))
    _check_seeds(seeds, partition, domain)
    scenarios = list(config.get("scenarios", SPECS))
    arms = list(config.get("arms", [PRIMARY, *BASELINES]))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = str(config["experiment_id"]).lower().replace("-", "_")  # M1-ACTION-E002 -> m1_action_e002
    checkpoint_path = out / f"{stem}_{partition.value}.checkpoint.json"
    declaration = _checkpoint_declaration(config, seeds, partition, domain, scenarios, arms)
    completed = _load_checkpoint(checkpoint_path, declaration)
    runs_root = REPO_ROOT / "artifacts" / "runs" / str(config["experiment_id"])
    jobs = [
        {
            "seed": s,
            "scenario": sc,
            "arm": arm,
            "run_id": f"{config['experiment_id']}-{sc}-s{s}-{arm}",
            "runs_root": str(runs_root),
            "cfg": config,
            "keep_bundle": bool(config.get("keep_bundles", False)),
        }
        for s in seeds
        for sc in scenarios
        for arm in arms
        if f"{config['experiment_id']}-{sc}-s{s}-{arm}" not in completed
    ]
    workers = int(config.get("workers", 1))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(mission_job, job) for job in jobs]
            for future in as_completed(futures):
                row = future.result()
                completed[str(row["run_id"])] = row
                _write_checkpoint(checkpoint_path, declaration, completed)
    else:
        for job in jobs:
            row = mission_job(job)
            completed[str(row["run_id"])] = row
            _write_checkpoint(checkpoint_path, declaration, completed)
    rows = list(completed.values())
    rows.sort(key=lambda r: r["run_id"])
    summary = summarize(rows, scenarios)
    result: dict[str, Any] = {
        "experiment_id": config["experiment_id"],
        "evidence_class": EVIDENCE_CLASS,
        "data_status": "SYNTHETIC_ONLY",
        "partition": partition.value,
        "partition_domain": domain,
        "partition_file": PARTITION_FILES[domain],
        "seeds": list(seeds),
        "scenarios": {
            sc: {
                "action_class": SPECS[sc].action_class,
                "onset": SPECS[sc].onset,
                "expected": list(SPECS[sc].expected),
                "success": SPECS[sc].success,
                "warrant_by_construction": SPECS[sc].warrant_by_construction,
                "not_applicable_reason": SPECS[sc].not_applicable_reason,
            }
            for sc in scenarios
        },
        "arms": arms,
        "config": config,
        "checkpoint": {
            "path": str(checkpoint_path),
            "declaration_sha256": declaration["sha256"],
            "completed_rows": len(rows),
        },
        **summary,
        "verdicts": verdicts(summary, config, scenarios, rows),
        "per_run": rows,
    }
    name = f"{stem}.json" if partition is Partition.FINAL_TEST else f"{stem}_{partition.value}.json"
    result_path = out / name
    _atomic_write_json(result_path, result)
    return result
