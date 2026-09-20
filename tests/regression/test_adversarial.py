"""Adversarial regression suite for the integrated runtime (real components, real run artifacts).

duplicate / late / corrupt observations, missing modality, huge temporal gap, stale beliefs, storage crash,
network outage, queue overload, localization degradation, thruster failure, low power, command expiry,
config mismatch, checkpoint mismatch and a missing replay artifact.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from conrad.orchestration.comms import ShoreLink
from conrad.orchestration.mission_config import runtime_config
from conrad.persistence.object_store import ObjectStore
from conrad.persistence.replay_store import ReplayIntegrityError, verify_bundle
from conrad.runtime.command_gateway import GatewayReason
from conrad.runtime.doctor import run_doctor
from conrad.schemas.belief import BeliefQuery
from conrad.schemas.events import EventType
from conrad.schemas.frames import quat_from_euler
from conrad.schemas.observation import Modality, PayloadRef
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.world import Domain
from conrad.settings import load_settings
from tests.acceptance._runs import short_session, smoke


@pytest.fixture(scope="module")
def s():
    session = short_session("FLAGSHIP-I4", 60.0, steps=40)
    yield session
    session.finish()


def _types(session, since=0):
    return [e.event_type for e in session.log.events[since:]]


def _observe_target(session):
    """Truth-side: park the vehicle on the far side and collect one round of payload observations."""
    world = session.world
    comp = world.context.component(world.context.critical_component_ids[0])
    c = (np.asarray(comp.p0_m) + np.asarray(comp.p1_m)) / 2
    pose = c + np.array([0.0, 2.2, 0.6])
    world.hardware.kernel.reset(
        pose, np.asarray(quat_from_euler(0.0, 0.0, 0.0))
    )  # side-looking payload faces +Y
    world.advance(1.2)
    return [o for o in world.hardware.get_payload_observations() if o.modality is Modality.STRUCTURED]


def test_duplicate_observation_counts_once(s):
    obs = _observe_target(s)
    assert obs
    n = len(s.log.events)
    before = s.runtime.perception.stats.duplicates
    s.runtime.perception.process([obs[0], obs[0]], obs[0].timestamp)
    assert s.runtime.perception.stats.duplicates == before + 1
    assert _types(s, n).count(EventType.OBSERVATION_RECEIVED) == 1
    assert EventType.EVIDENCE_DUPLICATE_DROPPED in _types(s, n)


def test_late_observation_follows_declared_policy(s):
    obs = _observe_target(s)
    late = obs[0].model_copy(
        update={
            "observation_id": s.runtime.s.ids.new(),
            "timestamp": TimeStamp(time_ns=obs[0].timestamp.time_ns - 5_000_000_000, clock_domain="SIM"),
        }
    )
    n = len(s.log.events)
    s.runtime.perception.process([*obs, late], obs[0].timestamp)
    handled = [e for e in s.log.events[n:] if e.event_type is EventType.LATE_EVIDENCE_HANDLED]
    assert handled and handled[0].payload["policy"] == "EXPLICIT_LATE"


def test_corrupt_observations_are_rejected_not_ingested(s):
    obs = _observe_target(s)[0]
    nan = obs.model_copy(
        update={"observation_id": s.runtime.s.ids.new(), "inline_values": (math.nan, 0.0, 0.0)}
    )
    bogus = PayloadRef(
        uri="sha256://" + "0" * 64, digest="0" * 64, media_type="application/x-npy", byte_length=8
    )
    ghost = obs.model_copy(update={"observation_id": s.runtime.s.ids.new(), "payload_ref": bogus})
    n = len(s.log.events)
    s.runtime.perception.process([nan, ghost], obs.timestamp)
    reasons = [e.payload["reason"] for e in s.log.events[n:] if e.event_type is EventType.FAULT_DETECTED]
    assert "NON_FINITE_INLINE_VALUES" in reasons and any(
        r.startswith("PAYLOAD_UNVERIFIABLE") for r in reasons
    )
    assert EventType.EVIDENCE_CREATED not in _types(s, n)


def test_huge_temporal_gap_keeps_uncertainty_finite(s):
    obs = _observe_target(s)
    day = 86_400 * 1_000_000_000
    far = [
        o.model_copy(
            update={
                "observation_id": s.runtime.s.ids.new(),
                "timestamp": TimeStamp(time_ns=o.timestamp.time_ns + day, clock_domain="SIM"),
            }
        )
        for o in obs
    ]
    msgs = s.runtime.perception.process(far, far[0].timestamp)
    assert all(all(math.isfinite(v) for v in m.uncertainty.as_tuple()) for m in msgs)


def test_stale_beliefs_block_continue(s):
    rt = s.runtime
    later = TimeStamp(time_ns=s.world.hardware.now_ns() + 10_000 * 1_000_000_000, clock_domain="SIM")
    out = rt.deliberation.decide(
        later,
        rt.stack.estimator.get_state(),
        None,
        None,
        s.world.hardware.get_health(),
        True,
        {"phase": "TRANSIT"},
    )
    assert out.record.chosen is None or out.record.chosen.action_type.value != "CONTINUE_MISSION"
    assert "STALE" in out.record.rationale


def test_storage_crash_marks_module_unavailable_without_half_commit():
    sess = short_session("GOLDEN-SMOKE", 30.0, steps=10)
    try:
        repo = sess.repo
        before = len(repo.all_revisions())
        repo.fault.fail_at = "after_revision"
        for _ in range(60):
            sess.step()
            if sess.runtime.runner.failed:
                break
        repo.fault.fail_at = None
        assert sess.runtime.runner.failed, "no module saw the storage fault"
        crashed = [
            e
            for e in sess.log.events
            if e.event_type is EventType.FAULT_DETECTED and "injected failure" in e.payload.get("error", "")
        ]
        assert crashed
        assert len(repo.all_revisions()) >= before  # nothing partial became visible; the DB stays consistent
        for rev in repo.all_revisions():
            assert repo.provenance_record(rev.provenance_root) is not None
    finally:
        sess.finish()


def _shore(s, **link):
    cfg = runtime_config({"link": {"bandwidth_bps": 1200.0, **link}})
    return ShoreLink(s.runtime.s, cfg, 7, s.world.context.critical_component_ids)


def _critical_message(s):
    return next(
        m
        for m in s.runtime.m2t.export_beliefs()
        if m.world_entity_id in s.world.context.critical_component_ids
    )


def test_network_outage_stores_and_forwards(s):
    shore = _shore(s, outages_s=[[0.0, 20.0]])
    msg = _critical_message(s)
    shore.offer([msg], msg.timestamp, floor=0.95)
    for t in range(0, 20):
        shore.step(float(t), 1.0)
    assert shore.link_state(5.0).status.value == "DOWN" and not shore.arrivals
    for t in range(20, 40):
        shore.step(float(t), 1.0)
    assert shore.arrivals and shore.receiver.revision(msg.belief_id) is not None


def test_queue_overload_never_drops_the_critical_unit(s):
    cfg = runtime_config({"link": {"bandwidth_bps": 50.0}, "baac": {"queue_capacity_bits": 30000}})
    shore = ShoreLink(s.runtime.s, cfg, 7, s.world.context.critical_component_ids)
    now = s.world.hardware.now_ns()
    routine = list(s.runtime.bus.query(BeliefQuery(domain=Domain.SPATIAL, max_results=80), now).messages)
    crit = _critical_message(s)
    shore.offer(routine, crit.timestamp)
    shore.offer([crit], crit.timestamp, floor=0.95)
    for t in range(5):
        shore.step(float(t), 1.0)
    critical_units = {u for u, (_, flag, _) in shore.sender.unit_created.items() if flag}
    assert shore.sender.queue.dropped, "the overload did not force any drop"
    assert critical_units and not {d.unit_id for d in shore.sender.queue.dropped} & critical_units


def test_missing_structural_modality_fabricates_nothing():
    sess = short_session(
        "GOLDEN-SMOKE", 16.0, world={"structural": {"degradation": {"missing_modality": 1.0}}}
    )
    try:
        assert sess.runtime.perception.stats.by_kind.get("STRUCTURED", 0) == 0
        direct = [
            r
            for r in sess.repo.all_revisions()
            if r.cell.domain is Domain.TECHNICAL and r.update_kind.value == "DIRECT"
        ]
        assert not direct
        assert sess.runtime.deliberation.decisions  # the loop keeps deciding on what it has
    finally:
        sess.finish()


@pytest.mark.parametrize(
    ("fault", "expect"),
    [
        ({"t_s": 2.0, "type": "THRUSTER_FAILURE", "target": "H1"}, "DEGRADED_MANEUVERABILITY"),
        ({"t_s": 1.0, "type": "FIX_OUTAGE", "duration_s": 200.0}, None),
    ],
)
def test_actuator_and_localization_faults_reach_safe_states(fault, expect):
    sess = short_session("GOLDEN-SMOKE", 20.0, world={"faults": [fault]})
    try:
        events = sess.runtime.executive.stats.safety_events
        assert [ev for ev in sess.log.events if ev.event_type is EventType.FAULT_INJECTED]
        if expect:
            assert any(expect in e["reasons"] for e in events)
        else:
            assert sess.runtime.stack.estimator.position_sigma_m > 0.15  # uncertainty grows without fixes
        assert sess.runtime.gateway.rejected == 0 or all(
            e.module == "conrad.runtime.command_gateway"
            for e in sess.log.events
            if e.event_type is EventType.COMMAND_REJECTED
        )
        assert sess.world.hardware.truth_access().collision_count == 0
    finally:
        sess.finish()


def test_low_power_reaches_return_and_safe_hold():
    sess = short_session(
        "GOLDEN-SMOKE", 8.0, world={"faults": [{"t_s": 0.5, "type": "LOW_POWER", "magnitude": 0.05}]}
    )
    try:
        events = sess.runtime.executive.stats.safety_events
        assert any(e["to"] == "RETURN" and "BATTERY_LOW" in e["reasons"] for e in events)
        chosen = [
            d.record.chosen.action_type.value for d in sess.runtime.deliberation.decisions if d.record.chosen
        ]
        assert "RETURN_TO_SAFE_STATE" in chosen
        assert sess.runtime.supervisor.state.value == "SAFE_HOLD"
        # RETURN motion is not authorized (EXT-HW-05). Since the I2 fault repair the stack no longer offers a
        # motion command the supervisor must refuse: it commands an explicit all-zero stop, which is accepted.
        # Either way the vehicle must not be driven, so assert the outcome rather than the refusal count.
        st = sess.runtime.executive.stats
        assert st.refused_by_supervisor > 0 or st.accepted > 0
        last = sess.runtime.stack.step()
        assert all(v == 0.0 for v in last.command.thruster_commands.values()), last.command
        truth = sess.world.hardware.truth_access().true_state()  # evaluation side only
        assert float(np.linalg.norm(truth.linear_velocity_body_mps)) < 0.05, truth.linear_velocity_body_mps
    finally:
        sess.finish()


def test_expired_command_and_config_mismatch_are_rejected_by_the_gateway(s):
    res = s.runtime.stack.step()
    assert res.decision.authorized
    s.world.advance(1.0)  # past the 0.25 s command deadline
    ack = s.runtime.gateway.submit(res.command)
    assert GatewayReason.EXPIRED in ack.reason_codes and not ack.accepted
    fresh = s.runtime.stack.step().command
    wrong = fresh.model_copy(update={"robot_config_digest": "f" * 64})
    ack2 = s.runtime.gateway.submit(wrong)
    assert GatewayReason.CONFIG_DIGEST in ack2.reason_codes


def test_checkpoint_mismatch_fails_doctor(tmp_path):
    bogus = tmp_path / "bad.ckpt"
    bogus.write_bytes(b"not a checkpoint")
    settings = load_settings("configs/sim/mission_test_small.yaml")
    settings = settings.model_copy(update={"model": {"checkpoints": [str(bogus)]}})
    report = run_doctor("configs/sim/mission_test_small.yaml", settings)
    check = next(c for c in report.checks if c.name == "checkpoints")
    assert check.status == "FAIL" and not report.ok


def test_missing_replay_artifact_fails_closed(tmp_path):
    src = Path(smoke()["run_dir"])
    broken = tmp_path / src.name
    shutil.copytree(src, broken)
    victim = next((broken / "objects").rglob("*.npy"), None) or next(
        p for p in (broken / "objects").rglob("*") if p.is_file()
    )
    victim.unlink()
    with pytest.raises(ReplayIntegrityError):
        verify_bundle(broken, ObjectStore(broken / "objects"))


def test_control_period_must_match_physics_step():
    from conrad.sim.mission.run import prepare

    settings = load_settings("configs/sim/mission_test_small.yaml")
    bad = settings.model_copy(
        update={"sim": {"mission": {"runtime": {"control_period_s": 0.05}, "world": {"physics_dt_s": 0.02}}}}
    )
    with pytest.raises(ValueError, match="integer multiple"):
        prepare("GOLDEN-SMOKE", bad)


def test_simulated_time_tracks_the_control_clock():
    sess = short_session("GOLDEN-SMOKE", 3.0)
    try:
        steps = round(sess.rcfg.duration_s / sess.rcfg.control_period_s)
        expected = 0.1 + steps * sess.rcfg.control_period_s
        assert abs(sess.world.t_s - expected) < 1e-6
    finally:
        sess.finish()
