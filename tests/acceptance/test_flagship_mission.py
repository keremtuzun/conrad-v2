"""FLAGSHIP-I4 closed active inspection: the full causal chain, measured on a real run (no fakes).

Truth is read only from the evaluation-only truth record. I4 thresholds are OPEN, so the acceptance record
is NOT_EVALUABLE; this test asserts the mechanism chain, not a performance claim.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest

from conrad.evaluation.acceptance import AcceptanceRecord, AcceptanceStatus, Measurements, evaluate
from conrad.persistence.db import make_engine
from conrad.persistence.repository import Repository
from conrad.runtime.event_log import read_events
from conrad.schemas.events import EventType
from conrad.schemas.provenance import SourceType
from tests.acceptance._runs import flagship


@pytest.fixture(scope="module")
def run():
    out = flagship()
    d = Path(out["run_dir"])
    engine = make_engine(d / "conrad.sqlite")
    repo = Repository(engine)
    truth = json.loads((d / "truth" / "truth_record.json").read_text(encoding="utf-8"))
    target = UUID(truth["meta"]["target_registry_id"])
    events = list(read_events(d / "events.jsonl"))
    decisions = [
        json.loads(x) for x in (d / "mission" / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    yield {"out": out, "dir": d, "repo": repo, "target": target, "events": events, "decisions": decisions}
    engine.dispose()


def _target_revs(run):
    return [
        r
        for r in run["repo"].all_revisions()
        if r.cell.registry_entity_id == run["target"] and r.cell.domain.value == "TECHNICAL"
    ]


def test_before_the_segment_condition_is_unknown_with_high_uo(run):
    before = run["out"]["report"]["target_before"]
    assert before["condition.status"] == "UNKNOWN" or before["U_O"] >= 0.5
    assert before["corrosion_depth_m"] is None  # UNKNOWN is not a guessed value


def test_egdc_raises_an_information_need_routed_to_mcbr(run):
    need = [
        e for e in run["events"] if e.event_type is EventType.DECISION_MADE and e.payload["route"] == "MCBR"
    ]
    assert need and need[0].payload["action"] == "REQUEST_INFORMATION"


def test_mcbr_plan_has_candidate_table_and_rejections(run):
    tables = json.loads((run["dir"] / "mission" / "mcbr_candidate_tables.json").read_text(encoding="utf-8"))
    planned = [t for t in tables if t["plan"]["status"] == "PLAN"]
    assert planned, "MCBR produced no ObservationPlan"
    t = planned[0]
    assert len(t["candidate_table"]) > 10
    assert t["plan"]["rejected"] and all(r["reason_codes"] for r in t["plan"]["rejected"])
    assert t["plan"]["primary_action"]["predicted_visibility"] > 0.0


def test_goal_then_commands_only_through_the_gateway(run):
    goals = [
        e
        for e in run["events"]
        if e.event_type is EventType.GOAL_ACCEPTED and e.payload["purpose"] == "INSPECT"
    ]
    assert goals and goals[0].payload["plan_id"] is not None
    sent = [e for e in run["events"] if e.event_type is EventType.COMMAND_SENT]
    acks = [e for e in run["events"] if e.event_type is EventType.COMMAND_ACK]
    rt = run["out"]["report"]["runtime"]
    assert len(sent) == len(acks) == rt["commands_accepted"] > 0
    assert all(e.module == "conrad.runtime.command_gateway" for e in sent)


def test_target_gets_a_direct_revision_after_the_mcbr_view(run):
    rep = run["out"]["report"]
    assert rep["target_direct_revisions_after_inspection"], "no DIRECT update of the target after inspection"
    direct = [r for r in _target_revs(run) if r.update_kind.value == "DIRECT"]
    ev = run["repo"].evidence(direct[-1].consumed_evidence_ids[0])
    obs = run["repo"].observation(ev.source_observation_id)
    assert obs is not None and obs.modality.value == "STRUCTURED"
    assert rep["target_after"]["condition.status"] == "OBSERVED"


def test_grounded_decision_on_the_target_after_the_update(run):
    last_direct = max(r.measurement_time_ns for r in _target_revs(run) if r.update_kind.value == "DIRECT")
    later = [d for d in run["decisions"] if d["timestamp"]["time_ns"] > last_direct]
    grounded = [
        c
        for d in later
        for c in d["claims"]
        if c["claim_type"] == "BELIEF_CLAIM" and c["grounding"] == "GROUNDED" and c["source_belief_ids"]
    ]
    assert grounded, "no grounded belief claim after the finding"


def test_baac_transmits_the_critical_finding(run):
    comm = run["out"]["report"]["runtime"]["communication"]
    assert any(e.event_type is EventType.TRANSMISSION for e in run["events"])
    assert comm["critical_offers"] >= 1 and comm["critical_delivered"] >= 1
    assert all(lat >= 0 for lat in comm["critical_alert_latency_s"])


def test_trace_from_last_command_to_a_raw_observation(run):
    import sqlite3

    con = sqlite3.connect(str(run["dir"] / "conrad.sqlite"))
    row = con.execute(
        "SELECT payload_json FROM commands WHERE accepted = 1 ORDER BY issued_time_ns DESC LIMIT 1"
    ).fetchone()
    con.close()
    rec_id = UUID(json.loads(row[0])["provenance_record_id"])
    closure = run["repo"].provenance_closure(rec_id)
    kinds = {r.source_type for r in closure.values()}
    assert {SourceType.COMMAND, SourceType.PLAN, SourceType.DECISION, SourceType.DIRECT_OBSERVATION} <= kinds
    leaves = [r for r in closure.values() if r.source_type is SourceType.DIRECT_OBSERVATION]
    repo = run["repo"]

    def raw(source):  # belief record -> evidence -> observation, or a direct observation reference
        if repo.observation(source) is not None:
            return True
        ev = repo.evidence(source)
        return ev is not None and repo.observation(ev.source_observation_id) is not None

    assert any(raw(s) for r in leaves for s in r.source_ids)


def test_i4_acceptance_record_is_not_evaluable(run):
    record = AcceptanceRecord(
        gate_id="I4", primary_metric="target_hidden_state_abs_error", baseline="FLAGSHIP-I4-FIXEDVIEW"
    )
    result = evaluate(record, Measurements(evidence_artifact=str(run["dir"])))
    assert result.status is AcceptanceStatus.NOT_EVALUABLE
