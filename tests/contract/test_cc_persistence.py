"""Contract-closure tests CC-01..CC-04 and software-closure SS-03/SS-04 on the real SQLite repository."""

from __future__ import annotations

import pytest

from conrad.persistence.object_store import DigestMismatchError, MissingObjectError, ObjectStore
from conrad.persistence.repository import (
    BeliefUpdate,
    CommitStatus,
    DuplicateEvidenceError,
    LifecycleError,
    PredecessorMismatchError,
    Repository,
    StaleUpdateError,
)
from conrad.schemas.belief import Lifecycle, UpdateKind
from conrad.schemas.ids import IdFactory
from conrad.schemas.provenance import ProvenanceError
from tests.conftest import make_evidence, make_revision


def _create(repo: Repository, ids: IdFactory, run_id, t: float = 1.0, group: str | None = None):
    ev, eprov = make_evidence(ids, run_id, t, group)
    repo.put_evidence(ev)
    bid = ids.new()
    rev, rprov = make_revision(ids, bid, 0, t, (ev.evidence_id,), (eprov.record_id,))
    upd = BeliefUpdate(
        message_id=ids.new(), producer_version="t0", run_id=run_id, revision=rev, provenance=[eprov, rprov]
    )
    return bid, ev, upd


def test_cc01_duplicate_delivery_is_one_contribution(repo: Repository, ids: IdFactory) -> None:
    run_id = ids.new()
    bid, ev, upd = _create(repo, ids, run_id)
    first = repo.commit_update(upd)
    assert first.status is CommitStatus.COMMITTED and first.independent_observation_count == 1
    again = repo.commit_update(upd)  # retried message
    assert again.status is CommitStatus.DUPLICATE_MESSAGE and again.independent_observation_count == 1
    assert repo.put_evidence(ev) is False  # retried evidence delivery
    # same evidence under a NEW message id must be refused, not double counted
    rev2, p2 = make_revision(ids, bid, 1, 1.0, (ev.evidence_id,), (upd.provenance[0].record_id,))
    with pytest.raises(DuplicateEvidenceError):
        repo.commit_update(
            BeliefUpdate(
                message_id=ids.new(), producer_version="t0", run_id=run_id, revision=rev2, provenance=[p2]
            )
        )
    assert len(repo.revisions(bid)) == 1 and repo.head(bid).independent_observation_count == 1


def test_cc01_same_independence_group_is_not_corroboration(repo: Repository, ids: IdFactory) -> None:
    run_id = ids.new()
    bid, _, upd = _create(repo, ids, run_id, group="camera-frame-17")
    repo.commit_update(upd)
    ev2, eprov2 = make_evidence(ids, run_id, 1.1, group="camera-frame-17")
    ev3, eprov3 = make_evidence(ids, run_id, 1.2, group="sonar-ping-3")
    repo.put_evidence(ev2)
    repo.put_evidence(ev3)
    rev, p = make_revision(
        ids, bid, 1, 1.2, (ev2.evidence_id, ev3.evidence_id), (eprov2.record_id, eprov3.record_id)
    )
    res = repo.commit_update(
        BeliefUpdate(
            message_id=ids.new(),
            producer_version="t0",
            run_id=run_id,
            revision=rev,
            provenance=[eprov2, eprov3, p],
        )
    )
    assert res.independent_observation_count == 2  # frame-17 counted once, sonar once


def test_cc02_late_observation_never_masquerades_as_current(repo: Repository, ids: IdFactory) -> None:
    run_id = ids.new()
    bid, _, upd = _create(repo, ids, run_id, t=10.0)
    repo.commit_update(upd)
    old_ev, old_prov = make_evidence(ids, run_id, t=4.0)
    repo.put_evidence(old_ev)
    rev, p = make_revision(ids, bid, 1, 4.0, (old_ev.evidence_id,), (old_prov.record_id,))
    with pytest.raises(StaleUpdateError):
        repo.commit_update(
            BeliefUpdate(
                message_id=ids.new(),
                producer_version="t0",
                run_id=run_id,
                revision=rev,
                provenance=[old_prov, p],
            )
        )
    late_rev, lp = make_revision(ids, bid, 1, 4.0, (old_ev.evidence_id,), (old_prov.record_id,), late=True)
    repo.commit_update(
        BeliefUpdate(
            message_id=ids.new(),
            producer_version="t0",
            run_id=run_id,
            revision=late_rev,
            provenance=[old_prov, lp],
        )
    )
    revs = repo.revisions(bid)
    assert revs[-1].late is True and revs[-1].measurement_time_ns < revs[0].measurement_time_ns
    # a stale writer holding an old predecessor cannot replace the newer head
    stale, sp = make_revision(ids, bid, 1, 11.0, (), (), kind=UpdateKind.PREDICTED)
    with pytest.raises(PredecessorMismatchError):
        repo.commit_update(
            BeliefUpdate(
                message_id=ids.new(), producer_version="t0", run_id=run_id, revision=stale, provenance=[sp]
            )
        )


@pytest.mark.parametrize(
    "point", ["before_writes", "after_provenance", "after_contributions", "after_revision", "before_commit"]
)
def test_cc03_ss03_failed_write_leaves_nothing_partial(repo: Repository, ids: IdFactory, point: str) -> None:
    run_id = ids.new()
    _, _, upd = _create(repo, ids, run_id)
    before = repo.counts()
    repo.fault.fail_at = point
    with pytest.raises(RuntimeError, match="injected failure"):
        repo.commit_update(upd)
    assert repo.counts() == before, (
        "a failed belief write must not leave any revision/provenance/contribution row"
    )
    repo.fault.fail_at = None
    assert repo.commit_update(upd).status is CommitStatus.COMMITTED  # restart: exactly one complete commit
    assert repo.commit_update(upd).status is CommitStatus.DUPLICATE_MESSAGE


def test_cc03_dangling_provenance_rejected(repo: Repository, ids: IdFactory) -> None:
    run_id = ids.new()
    _, _, upd = _create(repo, ids, run_id)
    broken = BeliefUpdate(
        message_id=upd.message_id,
        producer_version="t0",
        run_id=run_id,
        revision=upd.revision,
        provenance=[upd.provenance[1]],
    )
    before = repo.counts()
    with pytest.raises(ProvenanceError):
        repo.commit_update(broken)
    assert repo.counts() == before


def test_cc04_merge_preserves_lineage_to_raw_observations(repo: Repository, ids: IdFactory) -> None:
    run_id = ids.new()
    a, ev_a, upd_a = _create(repo, ids, run_id, 1.0)
    b, ev_b, upd_b = _create(repo, ids, run_id, 2.0)
    repo.commit_update(upd_a)
    repo.commit_update(upd_b)
    merged = ids.new()
    rev, p = make_revision(
        ids,
        merged,
        0,
        3.0,
        (),
        (upd_a.revision.provenance_root, upd_b.revision.provenance_root),
        kind=UpdateKind.LIFECYCLE,
        lifecycle=Lifecycle.CONFIRMED,
    )
    repo.commit_update(
        BeliefUpdate(
            message_id=ids.new(),
            producer_version="t0",
            run_id=run_id,
            revision=rev,
            provenance=[p],
            lineage_parents=[a, b],
            lineage_kind="MERGE",
        )
    )
    for old, upd in ((a, upd_a), (b, upd_b)):
        tomb, tp = make_revision(
            ids,
            old,
            1,
            3.0,
            (),
            (upd.revision.provenance_root,),
            kind=UpdateKind.LIFECYCLE,
            lifecycle=Lifecycle.MERGED,
        )
        repo.commit_update(
            BeliefUpdate(
                message_id=ids.new(), producer_version="t0", run_id=run_id, revision=tomb, provenance=[tp]
            )
        )
    assert repo.lineage_ancestors(merged) == {a, b}
    closure = repo.provenance_closure(rev.provenance_root)
    sources = {s for r in closure.values() for s in r.source_ids}
    assert {ev_a.source_observation_id, ev_b.source_observation_id} <= sources
    assert repo.head(a) is not None and repo.head(a).lifecycle is Lifecycle.MERGED  # old IDs stay resolvable
    # a merged belief is terminal
    zombie, zp = make_revision(ids, a, 2, 4.0, (), (), kind=UpdateKind.LIFECYCLE, lifecycle=Lifecycle.ACTIVE)
    with pytest.raises(LifecycleError):
        repo.commit_update(
            BeliefUpdate(
                message_id=ids.new(), producer_version="t0", run_id=run_id, revision=zombie, provenance=[zp]
            )
        )


def test_ss04_missing_or_corrupt_object_fails_closed(store: ObjectStore) -> None:
    ref = store.put_bytes(b"sonar-ping", "application/octet-stream")
    assert store.get_bytes(ref) == b"sonar-ping"
    store.path_for(ref.digest).write_bytes(b"tampered")
    with pytest.raises(DigestMismatchError) as exc:
        store.get_bytes(ref)
    assert ref.digest in str(exc.value)
    store.path_for(ref.digest).unlink()
    with pytest.raises(MissingObjectError) as missing:
        store.get_bytes(ref)
    assert ref.digest in str(missing.value)
    assert store.verify([ref.digest])
