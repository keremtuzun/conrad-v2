"""Integration Gate I0: the complete architectural loop executes with IDs, frames, timestamps,
provenance, uncertainty, versions and trace identity preserved; the run is stored and replayed."""

from __future__ import annotations

from pathlib import Path

from conrad.persistence.db import make_engine, migrate
from conrad.persistence.repository import Repository
from conrad.runtime.event_log import event_signature, read_events
from conrad.schemas.base import ARCHITECTURE_ID, STACK_ID
from conrad.schemas.belief import KnowledgeStatus
from conrad.schemas.decision import ActionType
from conrad.schemas.events import EventType
from conrad.schemas.provenance import SourceType
from tests.fixtures.fake_system import FakeFullSystem


def _run(tmp: Path, name: str, seed: int = 11):
    db = tmp / f"{name}.sqlite"
    migrate(db)
    repo = Repository(make_engine(db))
    system = FakeFullSystem(repo, seed=seed, log_path=tmp / f"{name}.events.jsonl")
    return system, repo, system.run()


def test_i0_loop_closes_and_belief_resolves(tmp_path: Path) -> None:
    _system, _repo, res = _run(tmp_path, "a")
    first, last = res.messages[0], res.messages[-1]
    assert first.knowledge_status is KnowledgeStatus.UNKNOWN and first.state_summary[0].value is None
    assert first.uncertainty.observational > 0.5
    assert res.decisions[0].chosen.action_type is ActionType.REQUEST_INFORMATION and res.plans
    assert res.commands, "navigation must have driven the robot through the gateway"
    assert last.knowledge_status is KnowledgeStatus.OBSERVED and last.uncertainty.observational < 0.5
    assert abs(float(last.state_summary[0].value) - res.true_severity) < 0.1
    assert res.decisions[-1].chosen.action_type is ActionType.TRANSMIT_INFORMATION
    assert res.transmissions and res.transmissions[-1].payload["new_revision"] == last.revision
    types = {e.event_type for e in res.events.events}
    assert {
        EventType.OBSERVATION_RECEIVED,
        EventType.EVIDENCE_CREATED,
        EventType.BELIEF_COMMITTED,
        EventType.DECISION_MADE,
        EventType.PLAN_PROPOSED,
        EventType.GOAL_ACCEPTED,
        EventType.COMMAND_SENT,
        EventType.COMMAND_ACK,
        EventType.TRANSMISSION,
    } <= types


def test_i0_boundaries_preserve_identity(tmp_path: Path) -> None:
    system, repo, res = _run(tmp_path, "b")
    for ev in res.events.events:
        assert ev.architecture_id == ARCHITECTURE_ID and ev.stack_id == STACK_ID
        assert ev.envelope.run_id == system.run_id and ev.envelope.mission_id == system.mission_id
        assert (
            ev.envelope.clock_domain == "SIM"
            and ev.envelope.created_time_ns >= ev.envelope.measurement_time_ns
        )
    for rev in repo.all_revisions(system.run_id):
        cell = rev.cell
        assert cell.spatial_support is not None and cell.spatial_support.frame_id == "WORLD"
        assert cell.model_version and cell.uncertainty is not None
        for eid in rev.consumed_evidence_ids:
            evidence = repo.evidence(eid)
            obs = repo.observation(evidence.source_observation_id)
            assert (
                obs is not None and obs.timestamp == evidence.timestamp
            )  # measurement time survives unchanged
            assert obs.robot_pose_estimate.frame_id == "WORLD" and obs.sensor_frame


def test_i0_full_causal_trace_from_command_to_raw_observation(tmp_path: Path) -> None:
    _system, repo, res = _run(tmp_path, "c")
    command = res.commands[-1]
    chain = repo.provenance_closure(command.provenance_root)
    kinds = {r.source_type for r in chain.values()}
    assert {SourceType.COMMAND, SourceType.PLAN, SourceType.DECISION, SourceType.DIRECT_OBSERVATION} <= kinds
    leaf_sources = {s for r in chain.values() if not r.parent_records for s in r.source_ids}
    assert any(repo.observation(s) is not None for s in leaf_sources), "trace must end at a raw observation"
    same_trace = res.events.by_trace(command.trace_id)
    assert {
        EventType.OBSERVATION_RECEIVED,
        EventType.DECISION_MADE,
        EventType.PLAN_PROPOSED,
        EventType.COMMAND_SENT,
    } <= {e.event_type for e in same_trace}


def test_i0_deterministic_replay(tmp_path: Path) -> None:
    _sys_a, repo_a, res_a = _run(tmp_path, "r1", seed=5)
    _sys_b, repo_b, res_b = _run(tmp_path, "r2", seed=5)
    assert event_signature(res_a.events.events) == event_signature(res_b.events.events)
    assert [r.canonical_json() for r in repo_a.all_revisions()] == [
        r.canonical_json() for r in repo_b.all_revisions()
    ]
    assert [d.canonical_json() for d in res_a.decisions] == [d.canonical_json() for d in res_b.decisions]
    stored = list(read_events(tmp_path / "r1.events.jsonl"))
    assert event_signature(stored) == event_signature(res_a.events.events)
    _, _, other = _run(tmp_path, "r3", seed=6)
    assert event_signature(other.events.events) != event_signature(res_a.events.events)
