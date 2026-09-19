"""GOLDEN-V0.2 compact suite GS-01..GS-10 (spec ch35 Priority 3) on real components.

Truth is used only through the truth record or the twin oracles (Twin2S occupancy/visibility). Every
assertion is an invariant from the spec table, not a tuned performance threshold.
"""

from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from conrad.domains.technical import Model2T, PropagationMode, structured_evidence
from conrad.persistence.db import make_engine
from conrad.persistence.repository import Repository
from conrad.schemas.belief import UpdateKind
from conrad.schemas.decision import InformationNeed, NavigationGoal, PlanStatus, QuestionType
from conrad.schemas.events import EventType
from conrad.schemas.frames import WORLD, Pose, quat_from_euler
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import SensorHealth
from conrad.schemas.provenance import SourceType
from conrad.schemas.world import Domain
from tests.acceptance._runs import flagship, short_session, smoke_replay


@pytest.fixture(scope="module")
def gs():
    s = short_session("FLAGSHIP-I4", 22.0)
    yield s
    s.finish()


@pytest.fixture(scope="module")
def fl():
    out = flagship()
    engine = make_engine(Path(out["run_dir"]) / "conrad.sqlite")
    yield out, Repository(engine)
    engine.dispose()


def _target(s):
    comp = s.world.context.component(s.world.context.critical_component_ids[0])
    return comp, s.runtime.bus.head(
        next(m.belief_id for m in s.runtime.m2t.export_beliefs() if m.world_entity_id == comp.registry_id)
    )


def _never_seen(s, comp):
    """Truth oracle: probes around the target that no geometric sensor could see from any true pose so far."""
    a, b = np.asarray(comp.p0_m), np.asarray(comp.p1_m)
    c = (a + b) / 2
    g = np.linspace(-0.8, 0.8, 5)
    pts = np.array([c + np.array([dx, dy, dz]) for dx in g for dy in g for dz in g[1:4] * 0.6])
    pts = pts[~s.world.t2s.occupancy_truth_at(pts)]
    res = s.runtime.m2s.cfg.grid.base_voxel_m
    corners = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]) * res
    cells = np.floor(pts / res) * res + res / 2  # the map cell each probe falls in
    seen = np.zeros(len(pts), dtype=bool)
    for cap in s.world.recorder.series["geometric_capture"]:
        pose = Pose(
            frame_id=WORLD,
            position_m=tuple(cap["position_m"]),
            orientation_wxyz=tuple(cap["orientation_wxyz"]),
        )
        for sensor in s.world.hardware.suite.sensors.geometric:
            for off in corners:
                seen |= np.asarray(s.world.t2s.visibility(sensor, pose, cells + off).visible, dtype=bool)
    return pts[~seen]


def test_gs01_occluded_region_stays_unknown_not_invented(gs):
    comp, _ = _target(gs)
    hidden = _never_seen(gs, comp)
    assert len(hidden) >= 5, "scenario lost its occluded region"
    status = [x.value for x in gs.runtime.m2s.occupancy_status(hidden)]
    assert "OBSERVED" not in status  # never seen -> never claimed observed
    assert status.count("UNKNOWN") >= len(status) // 2  # mostly UNKNOWN (INFERRED gap-fill is allowed)


def _need(s, belief):
    return InformationNeed(
        need_id=s.runtime.s.ids.new(),
        trace_id=s.runtime.s.ids.new(),
        target_belief_ids=(belief.belief_id,),
        question_type=QuestionType.EXTEND_COVERAGE,
        target_properties=("condition",),
        priority=0.9,
        constraints={"target_region": s.runtime.deliberation.requirements[0].region.model_dump(mode="json")},
    )


def test_gs02_smoke_mcbr_returns_a_feasible_view_or_a_stop_status(gs, monkeypatch):
    _comp, belief = _target(gs)
    d = gs.runtime.deliberation
    decision = d.decisions[-1]
    now = gs.runtime._now()
    pose = gs.runtime.stack.estimator.get_state().pose
    plan = d.plan(_need(gs, belief), decision, [belief], pose, now, ())
    feasible = [r for r in plan.table if r["feasible"]]
    assert plan.plan.status in (PlanStatus.PLAN, PlanStatus.NOT_WORTH_COST, PlanStatus.NEED_SATISFIED)
    if plan.plan.status is PlanStatus.PLAN:
        best_vis = max(r["visibility"] for r in feasible)
        assert plan.plan.primary_action.predicted_visibility >= 0.5 * best_vis
    monkeypatch.setattr(d, "is_free", lambda pts: np.zeros(len(np.atleast_2d(pts)), dtype=bool))
    none = d.plan(_need(gs, belief), decision, [belief], pose, now, ())
    assert none.plan.status is PlanStatus.NO_FEASIBLE_OBSERVATION and none.plan.primary_action is None
    assert all(r["reason_codes"] for r in none.table)


def test_gs03_direct_evidence_improves_the_correct_belief_with_direct_provenance(fl):
    out, repo = fl
    err_before = out["report"]["target_error_before"]["corrosion_depth_m.prior_mean_abs_error"]
    err_after = out["report"]["target_error_after"]["corrosion_depth_m.abs_error"]
    assert err_after is not None and err_after < err_before
    target = out["report"]["target_registry_id"]
    direct = [
        r
        for r in repo.all_revisions()
        if str(r.cell.registry_entity_id) == target
        and r.cell.domain is Domain.TECHNICAL
        and r.update_kind is UpdateKind.DIRECT
    ]
    assert direct
    rec = repo.provenance_record(direct[-1].provenance_root)
    assert rec is not None and rec.source_type is SourceType.DIRECT_OBSERVATION


def _structured(gs, wall, surf, crack, health=SensorHealth.OK):
    obs = next(o for o in _park_and_observe(gs) if "twin2t_fidelity" in o.sensor_context)
    return obs.model_copy(
        update={
            "observation_id": gs.runtime.s.ids.new(),
            "inline_values": (wall, surf, crack),
            "sensor_health": health,
        }
    )


def _park_and_observe(gs):
    comp = gs.world.context.component(gs.world.context.critical_component_ids[0])
    c = (np.asarray(comp.p0_m) + np.asarray(comp.p1_m)) / 2
    gs.world.hardware.kernel.reset(
        c + np.array([0.0, 2.2, 0.6]), np.asarray(quat_from_euler(0.0, 0.0, np.pi))
    )
    gs.world.advance(1.2)
    return gs.world.hardware.get_payload_observations()


def test_gs04_reliable_contradiction_raises_uc_and_propagation_keeps_direct(gs):
    comp = gs.world.context.component(gs.world.context.critical_component_ids[0])
    m = Model2T(IdFactory(41).child("gs04"), mode=PropagationMode.TCDP)
    m.initialize({"asset_registry": gs.world.context.asset_registry})
    ids = IdFactory(41).child("ev")
    a = _structured(gs, 0.006, 0.6, 0.08)
    b = _structured(gs, 0.0003, 0.05, 0.0)
    ev_a, _ = structured_evidence(a, ids, comp.registry_id, independence_group="a")
    ev_b, _ = structured_evidence(b, ids, comp.registry_id, independence_group="b")
    m.ingest([ev_a])
    first = next(x for x in m.update_beliefs(a.timestamp) if x.world_entity_id == comp.registry_id)
    m.ingest([ev_b])
    second = next(x for x in m.update_beliefs(b.timestamp) if x.world_entity_id == comp.registry_id)
    assert second.uncertainty.contradiction > first.uncertainty.contradiction
    wall = next(c for c in second.state_summary if c.name == "corrosion_depth_m")
    assert wall.status.value == "OBSERVED" and wall.value is not None  # not erased by propagation


def test_gs05_duplicate_delivery_commits_one_contribution(gs):
    obs = next(o for o in _park_and_observe(gs) if "twin2t_fidelity" in o.sensor_context)
    gs.runtime.perception.process([obs, obs], obs.timestamp)
    evs = [
        e
        for e in gs.log.events
        if e.event_type is EventType.EVIDENCE_CREATED
        and e.payload["observation_id"] == str(obs.observation_id)
    ]
    assert len(evs) == 1
    eid = evs[0].payload["evidence_id"]
    holders = [r for r in gs.repo.all_revisions() if eid in {str(x) for x in r.consumed_evidence_ids}]
    assert len(holders) <= 1


def test_gs06_2t_separates_observed_from_propagated(fl):
    _, repo = fl
    tech = [r for r in repo.all_revisions() if r.cell.domain is Domain.TECHNICAL]
    direct = [r for r in tech if r.update_kind is UpdateKind.DIRECT]
    relational = [r for r in tech if r.update_kind is UpdateKind.RELATIONAL]
    assert direct and all(r.cell.claim("corrosion_depth_m").status.value == "OBSERVED" for r in direct)
    for r in relational:
        assert not r.consumed_evidence_ids
        assert {
            c.status.value for c in r.cell.claims if c.name in ("corrosion_depth_m", "crack_length_m")
        } <= {"INFERRED", "OBSERVED", "UNKNOWN", "PREDICTED"}
        assert r.cell.claim("corrosion_depth_m").status.value != "OBSERVED" or any(
            d.belief_id == r.belief_id and d.revision < r.revision for d in direct
        )


def test_gs07_degradation_lowers_reliability_and_unsupported_stays_bounded(gs):
    ids = IdFactory(7).child("gs07")
    ok = _structured(gs, 0.001, 0.1, 0.0, SensorHealth.OK)
    bad = _structured(gs, 0.001, 0.1, 0.0, SensorHealth.DEGRADED)
    e_ok, _ = structured_evidence(ok, ids, None)
    e_bad, _ = structured_evidence(bad, ids, None)
    assert e_bad.reliability < e_ok.reliability and e_bad.aleatoric_uncertainty > e_ok.aleatoric_uncertainty
    sess = short_session("INT-004", 16.0)
    try:
        drop = sess.world.recorder.series.get("geometric_dropout", [])
        assert drop  # truth-side dropout happened
        geo = [
            e
            for e in sess.log.events
            if e.event_type is EventType.OBSERVATION_RECEIVED
            and e.payload["modality"] in ("DEPTH_RANGE", "SONAR")
            and 10e9 <= e.envelope.measurement_time_ns < 25e9
        ]
        assert not geo  # nothing fabricated during the dropout
        from conrad.decision.uir import unsupported_inference_rate

        assert unsupported_inference_rate([d.record for d in sess.runtime.deliberation.decisions]) == 0.0
    finally:
        sess.finish()


def test_gs08_unsafe_viewpoint_rejected_before_the_gateway(gs, fl):
    comp = gs.world.context.component(gs.world.context.critical_component_ids[0])
    inside = (np.asarray(comp.p0_m) + np.asarray(comp.p1_m)) / 2
    goal = NavigationGoal(
        goal_id=gs.runtime.s.ids.new(),
        trace_id=gs.runtime.s.ids.new(),
        target_pose=Pose(frame_id=WORLD, position_m=tuple(float(v) for v in inside)),
        position_tolerance_m=0.3,
        orientation_tolerance_rad=0.3,
        risk_limit=0.2,
    )
    accepted = gs.runtime.executive.set_goal(goal, "INSPECT", None, gs.runtime._now())
    assert not accepted
    assert not [
        e for e in gs.log.events if e.event_type is EventType.COMMAND_SENT and e.trace_id == goal.trace_id
    ]
    tables = json.loads(
        (Path(fl[0]["run_dir"]) / "mission" / "mcbr_candidate_tables.json").read_text(encoding="utf-8")
    )
    unsafe = [r for t in tables for r in t["candidate_table"] if "POSE_NOT_FREE" in r["reason_codes"]]
    chosen = {
        tuple(t["plan"]["primary_action"]["pose"]["position_m"])
        for t in tables
        if t["plan"]["primary_action"]
    }
    assert unsafe and not {tuple(r["position_m"]) for r in unsafe} & chosen


def test_gs09_late_evidence_policy_and_chronology(fl):
    out, repo = fl
    per_belief = defaultdict(list)
    for r in repo.all_revisions():
        per_belief[r.belief_id].append(r)
    for revs in per_belief.values():
        for prev, cur in itertools.pairwise(revs):
            assert cur.revision == prev.revision + 1
            if cur.update_kind is UpdateKind.DIRECT and not cur.late:
                assert (
                    cur.measurement_time_ns >= prev.measurement_time_ns
                    or prev.update_kind is not UpdateKind.DIRECT
                )
    events = [
        json.loads(x)
        for x in (Path(out["run_dir"]) / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [e["sequence"] for e in events] == list(range(len(events)))


def test_gs10_replay_reproduces_revisions_lineage_and_decisions():
    report = smoke_replay()
    assert report["equal"] and report["revisions"]["equal"] and report["decisions"]["equal"]
