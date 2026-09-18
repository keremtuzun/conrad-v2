"""SS-04 (missing digest fails closed), SS-08 (backup/restore/replay integrity), release lanes."""

from __future__ import annotations

from pathlib import Path

import pytest

from conrad.persistence.db import make_engine, migrate
from conrad.persistence.object_store import ObjectStore
from conrad.persistence.replay_store import (
    REQUIRED_REPLAY_KEYS,
    ReplayIntegrityError,
    backup,
    restore,
    verify_bundle,
    write_bundle_manifest,
)
from conrad.persistence.repository import Repository
from conrad.runtime.event_log import event_signature, read_events
from conrad.runtime.release import GateEvidence, ReleaseLane, build_release_manifest
from tests.fixtures.fake_system import FakeFullSystem


def _inputs() -> dict:
    return dict.fromkeys(REQUIRED_REPLAY_KEYS, "x")


def _run(tmp_path: Path):
    db_path = tmp_path / "run.sqlite"
    migrate(db_path)
    repo = Repository(make_engine(db_path))
    run_dir = tmp_path / "runs" / "r"
    system = FakeFullSystem(repo, seed=21, log_path=run_dir / "events.jsonl")
    res = system.run()
    store = ObjectStore(tmp_path / "objects")
    blob = store.put_bytes(b"raw sonar frame bytes", "application/octet-stream")
    return db_path, repo, run_dir, store, blob, res


def test_ss04_bundle_verification_fails_closed_and_names_digest(tmp_path: Path) -> None:
    _, _, run_dir, store, blob, _ = _run(tmp_path)
    write_bundle_manifest(run_dir, "r", _inputs(), [blob.digest])
    verify_bundle(run_dir, store)
    store.path_for(blob.digest).unlink()
    with pytest.raises(ReplayIntegrityError) as exc:
        verify_bundle(run_dir, store)
    assert blob.digest in str(exc.value)


def test_bundle_detects_tampered_event_log(tmp_path: Path) -> None:
    _, _, run_dir, store, blob, _ = _run(tmp_path)
    write_bundle_manifest(run_dir, "r", _inputs(), [blob.digest])
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as h:
        h.write("{}\n")
    with pytest.raises(ReplayIntegrityError, match=r"events.jsonl"):
        verify_bundle(run_dir, store)


def test_bundle_requires_complete_replay_inputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    with pytest.raises(ReplayIntegrityError, match="robot_config_digest"):
        write_bundle_manifest(run_dir, "r", {"scenario_seed": 1}, [])


def test_ss08_backup_restore_then_replay(tmp_path: Path) -> None:
    db_path, repo, run_dir, store, blob, _res = _run(tmp_path)
    before = [r.canonical_json() for r in repo.all_revisions()]
    report = backup(db_path, store, [blob.digest], tmp_path / "backup")
    assert report.object_count == 1
    restored = restore(
        tmp_path / "backup", tmp_path / "restored" / "db.sqlite", tmp_path / "restored" / "objects"
    )
    assert restored.restore_verified
    again = Repository(make_engine(tmp_path / "restored" / "db.sqlite"))
    assert [r.canonical_json() for r in again.all_revisions()] == before
    # registered replay fixture: same seed reproduces the stored event sequence
    db2 = tmp_path / "replay.sqlite"
    migrate(db2)
    replayed = FakeFullSystem(Repository(make_engine(db2)), seed=21).run()
    assert event_signature(replayed.events.events) == event_signature(
        list(read_events(run_dir / "events.jsonl"))
    )


def test_restore_rejects_corrupt_backup(tmp_path: Path) -> None:
    db_path, _, _, store, blob, _ = _run(tmp_path)
    backup(db_path, store, [blob.digest], tmp_path / "backup")
    (tmp_path / "backup" / "objects" / "sha256" / blob.digest).write_bytes(b"bitrot")
    with pytest.raises(ReplayIntegrityError, match="corrupt"):
        restore(tmp_path / "backup", tmp_path / "x" / "db.sqlite", tmp_path / "x" / "objects")


def test_release_lanes() -> None:
    digest = "a" * 64
    dev = build_release_manifest(ReleaseLane.DEV, digest, [], git=("abc", True))
    assert dev.releasable
    cand = build_release_manifest(
        ReleaseLane.CANDIDATE,
        digest,
        [GateEvidence(gate_id="I0", status="PASS", evidence_artifact="artifacts/runs/x")],
        git=("abc", False),
    )
    assert cand.releasable
    assert not build_release_manifest(ReleaseLane.CANDIDATE, digest, [], git=("abc", True)).releasable
    physical = build_release_manifest(
        ReleaseLane.PHYSICAL,
        digest,
        [GateEvidence(gate_id="I4", status="PASS", evidence_artifact="a")],
        git=("abc", False),
    )
    assert not physical.releasable and any("I8" in p for p in physical.problems)
