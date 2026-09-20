"""Gate I6 measurements (multi-domain intelligence) for one prepared mission session. EVALUATION PLANE.

Works for both drivers (duck-typed): the Python kernel ``MissionSession`` (surrogate) and the Unity ``UnitySession``
(formal). The deployment runtime is only read, never changed. Truth (Twin2E turbidity at the true target, Twin2E
optics constants) is used here for evaluation only.

Criteria (names as in ``conrad.evaluation.gates``) and what is measured for each:

* "one mission produces 2S, 2T and 2E beliefs": per domain, DIRECT revisions with consumed evidence, all in the same
  run and mission, all revised in place (revision > 0).
* "Model1 reasons across all three via the Belief Bus": EGDC decisions whose claim graph cites beliefs OWNED by all
  three domains (owner = the domain of the bus head of every cited belief); every cited (belief, revision) exists as a
  committed revision; the query engine asked all three routing tools; and the sensing-conditions gate (which reads
  only the decision record) recorded verdicts citing TECHNICAL, SPATIAL and ECOLOGICAL claims. Truth agreement of each
  gate verdict (true visibility at the true target from Twin2E) is reported.
* "children remain authoritative within domains": no bus rejection in the mission; every bus head is a revision its own
  child committed (the bus may lag the child, never lead or replace it; the lag is reported); every revision's provenance root was written by its own domain package; every DIRECT
  revision consumed only evidence from its own domain's sensors; every cross-domain delivery is
  CROSS_DOMAIN_CONTEXT and every CONTEXT revision consumes no evidence; a fresh bus loaded with the mission's heads
  rejects a foreign-domain write to every head.

implementation_status: EXPERIMENTAL_CANDIDATE (evaluation harness)
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any
from uuid import UUID

import numpy as np

from conrad.orchestration.belief_bus import REASON_FOREIGN_WRITE, BeliefBus
from conrad.orchestration.multidomain import DEFERRED, PROCEED, claim_domains
from conrad.schemas.belief import BeliefMessage, UpdateKind
from conrad.schemas.ids import IdFactory
from conrad.schemas.provenance import SourceType
from conrad.schemas.world import Domain

DOMAIN_PACKAGE = {
    Domain.SPATIAL: "conrad.domains.spatial",
    Domain.TECHNICAL: "conrad.domains.technical",
    Domain.ECOLOGICAL: "conrad.domains.ecological",
}
THREE = {d.value for d in Domain}


def target_station_truth(session: Any) -> np.ndarray:
    """TRUE mid-point of the target segment (Twin2S geometry), evaluation only."""
    w = session.world
    world = w.t2s.world
    i = world.index_of(w.target)
    prim = world.entities[i].primitive
    off = np.asarray(world.entity_offset(i), dtype=np.float64)
    return (np.asarray(prim.a, dtype=np.float64) + np.asarray(prim.b, dtype=np.float64)) / 2.0 + off


def true_visibility(session: Any, range_m: float) -> tuple[float, float]:
    """(true turbidity NTU at the true target, true beam transmission over ``range_m``) from Twin2E truth."""
    w = session.world
    p = target_station_truth(session)
    turb = float(w.t2e.fields.sample("turbidity", p[None])[0])
    o = w.t2e.cfg.observation
    return turb, math.exp(
        -(o.beam_attenuation_clear_per_m + o.beam_attenuation_per_m_per_ntu * turb) * range_m
    )


def run_with_truth(session: Any) -> list[dict[str, Any]]:
    """Step the whole mission; after each new sensing-gate record, sample the truth it should agree with."""
    rt = session.runtime
    gate = rt.sensing_gate
    truth: list[dict[str, Any]] = []
    seen = 0
    for _ in range(round(session.rcfg.duration_s / session.rcfg.control_period_s)):
        session.step()
        if gate is not None and len(gate.records) > seen:
            for rec in gate.records[seen:]:
                turb, vis = true_visibility(session, rec["inspection_range_m"])
                truth.append(
                    {
                        "decision_id": rec["decision_id"],
                        "true_turbidity_ntu": turb,
                        "true_visibility_at_range": vis,
                        "true_verdict": DEFERRED if vis < rec["min_visibility"] else PROCEED,
                    }
                )
            seen = len(gate.records)
    return truth


def _sensor_domains(session: Any) -> dict[UUID, Domain]:
    """Deployment sensor -> the domain its evidence may feed (mission context sensor declarations)."""
    ctx = session.world.context
    out: dict[UUID, Domain] = {}
    for s in ctx.sensors:
        if s.modality in ("DEPTH_RANGE", "SONAR"):
            out[s.sensor_id] = Domain.SPATIAL
        elif s.modality == "ENVIRONMENTAL" or s.frame_id in ("SENSOR_ENV", "SENSOR_SURVEY"):
            out[s.sensor_id] = Domain.ECOLOGICAL
    for sid in ctx.structural_sensor_ids:
        out[sid] = Domain.TECHNICAL
    return out


def beliefs_per_domain(session: Any) -> dict[str, Any]:
    rt, repo = session.runtime, session.repo
    revs = repo.all_revisions(rt.s.run_id)
    out: dict[str, Any] = {}
    for d in Domain:
        mine = [r for r in revs if r.cell.domain is d]
        direct = [r for r in mine if r.update_kind is UpdateKind.DIRECT]
        out[d.value] = {
            "revisions": len(mine),
            "direct": len(direct),
            "direct_with_evidence": sum(bool(r.consumed_evidence_ids) for r in direct),
            "beliefs": len({r.belief_id for r in mine}),
            "max_revision": max((r.revision for r in mine), default=-1),
            "update_kinds": dict(Counter(r.update_kind.value for r in mine)),
        }
    out["run_ids"] = sorted({str(rt.s.run_id)})
    out["other_run_revisions"] = len(repo.all_revisions()) - len(revs)
    return out


def model1_across_domains(session: Any, truth: list[dict[str, Any]]) -> dict[str, Any]:
    rt, repo = session.runtime, session.repo
    decisions = rt.deliberation.decisions
    cite3: list[str] = []
    missing_revisions = 0
    for out in decisions:
        doms = claim_domains(out.record.claims, rt.bus)
        if set(doms) == THREE:
            cite3.append(str(out.record.decision_id))
        for c in out.record.claims:
            for b, rev in zip(c.source_belief_ids, c.source_belief_revisions, strict=False):
                if not any(r.revision == rev for r in repo.revisions(b)):
                    missing_revisions += 1
    tools = Counter(tool for tool, _ in rt.deliberation.query.issued)
    gate = rt.sensing_gate
    records = [] if gate is None else list(gate.records)
    by_id = {t["decision_id"]: t for t in truth}
    verdicts = Counter(r["verdict"] for r in records)
    all3 = [
        r
        for r in records
        if all(r["claims"].get(d) for d in THREE) and all(r["beliefs"].get(d) for d in THREE)
    ]
    agree = [
        r
        for r in records
        if r["decision_id"] in by_id and by_id[r["decision_id"]]["true_verdict"] == r["verdict"]
    ]
    deferred = [r for r in records if r["verdict"] == DEFERRED]
    first_deferral = None
    if deferred:
        r = deferred[0]
        first_deferral = {
            "t_s": r["t_s"],
            "decision_id": r["decision_id"],
            "turbidity_ntu_belief": r["turbidity_ntu_belief"],
            "visibility_at_range": r["visibility_at_range"],
            "claims": {d: len(v) for d, v in r["claims"].items()},
            "beliefs": r["beliefs"],
            "truth": by_id.get(r["decision_id"]),
        }
    return {
        "decisions": len(decisions),
        "decisions_citing_all_three": len(cite3),
        "cited_revisions_not_committed": missing_revisions,
        "query_tools": dict(tools),
        "gate_records": len(records),
        "gate_verdicts": dict(verdicts),
        "gate_records_citing_all_three": len(all3),
        "gate_truth_agreement": None if not records else len(agree) / len(records),
        "gate_truth_records": len([r for r in records if r["decision_id"] in by_id]),
        "turbidity_belief_range_ntu": None
        if not records or all(r["turbidity_ntu_belief"] is None for r in records)
        else [
            min(r["turbidity_ntu_belief"] for r in records if r["turbidity_ntu_belief"] is not None),
            max(r["turbidity_ntu_belief"] for r in records if r["turbidity_ntu_belief"] is not None),
        ],
        "true_turbidity_range_ntu": None
        if not truth
        else [min(t["true_turbidity_ntu"] for t in truth), max(t["true_turbidity_ntu"] for t in truth)],
        "first_deferral": first_deferral,
        "inspection_goal_started_s": getattr(session, "inspection_s", None),
        "mcbr_plans": len(rt.deliberation.plans),
    }


def _foreign_write_probe(heads: list[BeliefMessage]) -> dict[str, int]:
    """Fresh bus loaded with the mission's heads; every head then gets a write from another domain."""
    bus = BeliefBus(IdFactory(0x16).child("i6_probe"))
    loaded = sum(r.accepted for r in bus.publish_many(heads))
    donors = {m.domain: m for m in heads}
    ids = IdFactory(0x16).child("i6_probe_msgs")
    attempts = rejected = unchanged = 0
    for m in heads:
        for d, donor in donors.items():
            if d is m.domain:
                continue
            fake = donor.model_copy(
                update={"message_id": ids.new(), "belief_id": m.belief_id, "revision": m.revision + 1}
            )
            attempts += 1
            receipt = bus.publish(fake)
            rejected += int(not receipt.accepted and receipt.reason_code == REASON_FOREIGN_WRITE)
            head = bus.head(m.belief_id)
            unchanged += int(head is not None and head.message_id == m.message_id)
    return {
        "heads_loaded": loaded,
        "attempts": attempts,
        "rejected_foreign": rejected,
        "heads_unchanged": unchanged,
    }


def children_authoritative(session: Any) -> dict[str, Any]:
    rt, repo = session.runtime, session.repo
    bus: BeliefBus = rt.bus
    heads = [m for m in (bus.head(b) for b in list(bus._heads)) if m is not None]  # read only
    head_mismatch = 0  # bus head is not a revision its own child committed (foreign, invented or ahead)
    head_lag = 0  # child committed later revisions that were not published (e.g. 2T CONTEXT revisions)
    for m in heads:
        committed = {r.revision: r for r in repo.revisions(m.belief_id)}
        same = committed.get(m.revision)
        if same is None or same.cell.domain is not m.domain:
            head_mismatch += 1
        elif max(committed) > m.revision:
            head_lag += 1
    revs = repo.all_revisions(rt.s.run_id)
    wrong_package: Counter[str] = Counter()
    for r in revs:
        rec = repo.provenance_record(r.provenance_root)
        if rec is None or not rec.module.startswith(DOMAIN_PACKAGE[r.cell.domain]):
            wrong_package[r.cell.domain.value] += 1
    sensors = _sensor_domains(session)
    foreign_evidence: Counter[str] = Counter()
    unknown_sensor = 0
    for r in revs:
        if r.update_kind is not UpdateKind.DIRECT:
            continue
        for e in r.consumed_evidence_ids:
            ev = repo.evidence(e)
            obs = None if ev is None else repo.observation(ev.source_observation_id)
            if obs is None:
                unknown_sensor += 1
                continue
            owner = sensors.get(obs.sensor_id)
            if owner is None:
                unknown_sensor += 1
            elif owner is not r.cell.domain:
                foreign_evidence[r.cell.domain.value] += 1
    deliveries = bus.context_deliveries
    bad_delivery = sum(
        1
        for d in deliveries
        if d.source_domain is d.target_domain or d.source_type is not SourceType.CROSS_DOMAIN_CONTEXT
    )
    context_revs = [r for r in revs if r.update_kind is UpdateKind.CONTEXT]
    return {
        "bus_rejected": len(bus.rejected),
        "bus_rejected_reasons": dict(Counter(r.reason_code for r in bus.rejected)),
        "bus_heads": len(heads),
        "bus_heads_per_domain": dict(Counter(m.domain.value for m in heads)),
        "bus_head_not_a_child_revision": head_mismatch,
        "bus_head_behind_child": head_lag,
        "revisions_checked": len(revs),
        "revisions_not_written_by_own_package": dict(wrong_package),
        "direct_evidence_from_other_domain_sensors": dict(foreign_evidence),
        "direct_evidence_unresolved": unknown_sensor,
        "context_deliveries": len(deliveries),
        "context_deliveries_by_pair": dict(
            Counter(f"{d.source_domain.value}->{d.target_domain.value}" for d in deliveries)
        ),
        "context_deliveries_not_context": bad_delivery,
        "context_revisions": dict(Counter(r.cell.domain.value for r in context_revs)),
        "context_revisions_with_evidence": sum(bool(r.consumed_evidence_ids) for r in context_revs),
        "foreign_write_probe": _foreign_write_probe(heads),
    }


def i6_measurements(session: Any, truth: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "one mission produces 2S, 2T and 2E beliefs": beliefs_per_domain(session),
        "Model1 reasons across all three via the Belief Bus": model1_across_domains(session, truth),
        "children remain authoritative within domains": children_authoritative(session),
    }


# ------------------------------------------------------------------------------------------------ pass rules
def beliefs_ok(m: dict[str, Any]) -> bool:
    return (
        all(
            m[d]["direct_with_evidence"] > 0
            and m[d]["direct"] == m[d]["direct_with_evidence"]
            and m[d]["max_revision"] > 0
            for d in THREE
        )
        and m["other_run_revisions"] == 0
    )


def reasoning_ok(m: dict[str, Any], arm: str) -> bool:
    """Both arms: decisions cite all three, nothing cited is uncommitted, all three tools queried, every gate record
    cites all three. TURBID: at least one deferral. CLEAR: no deferral and at least one PROCEED."""
    tools = {"query_model2s", "query_model2t", "query_model2e"}
    base = (
        m["decisions_citing_all_three"] > 0
        and m["cited_revisions_not_committed"] == 0
        and tools <= set(m["query_tools"])
        and m["gate_records"] > 0
        and m["gate_records_citing_all_three"] == m["gate_records"]
    )
    v = m["gate_verdicts"]
    if arm == "TURBID":
        return base and v.get(DEFERRED, 0) > 0
    return base and v.get(DEFERRED, 0) == 0 and v.get(PROCEED, 0) > 0


def authority_ok(m: dict[str, Any]) -> bool:
    p = m["foreign_write_probe"]
    return (
        m["bus_rejected"] == 0
        and m["bus_head_not_a_child_revision"] == 0
        and not m["revisions_not_written_by_own_package"]
        and not m["direct_evidence_from_other_domain_sensors"]
        and m["direct_evidence_unresolved"] == 0
        and m["context_deliveries"] > 0
        and m["context_deliveries_not_context"] == 0
        and m["context_revisions_with_evidence"] == 0
        and p["attempts"] > 0
        and p["rejected_foreign"] == p["attempts"]
        and p["heads_unchanged"] == p["attempts"]
    )
