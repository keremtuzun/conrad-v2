import pytest

from conrad.core.lifecycle import LifecycleEngine, LifecycleTransitionError, check_transition
from conrad.core.pmbl import PMBL
from conrad.persistence.repository import LifecycleError
from conrad.schemas.belief import (
    ALLOWED_LIFECYCLE_TRANSITIONS,
    BeliefQuery,
    KnowledgeStatus,
    Lifecycle,
    UpdateKind,
)
from conrad.schemas.provenance import SourceType
from conrad.schemas.timebase import stamp
from conrad.schemas.world import Domain

CLOCK = "sandbox_clock"


def _confirmed(pmbl, fac, archived, x=0.3, t0=1.0):
    e1 = archived(pmbl, fac, t0, {"p": x}, independence_group=f"a{t0}")
    cell = pmbl.create_candidate([e1], Domain.TECHNICAL, "weld")
    e2 = archived(pmbl, fac, t0 + 1, {"p": x}, independence_group=f"b{t0}")
    pmbl.apply_direct(cell.belief_id, [e2])
    return cell.belief_id, [e1, e2]


def test_every_listed_transition_is_checked():
    for old in Lifecycle:
        for new in Lifecycle:
            if new is old or new in ALLOWED_LIFECYCLE_TRANSITIONS[old]:
                check_transition(old, new)
            else:
                with pytest.raises(LifecycleTransitionError):
                    check_transition(old, new)


def test_lifecycle_candidate_confirm_active_dormant_reactivated(pmbl, fac, archived):
    bid, _ = _confirmed(pmbl, fac, archived)
    assert pmbl.node(bid).cell.lifecycle is Lifecycle.CONFIRMED
    pmbl.apply_direct(bid, [archived(pmbl, fac, 3.0, {"p": 0.3}, independence_group="c")])
    assert pmbl.node(bid).cell.lifecycle is Lifecycle.ACTIVE
    changes = pmbl.sweep(stamp(3.0 + pmbl.cfg.pmbl.dormant_after_s + 1, CLOCK))
    assert (bid, Lifecycle.ACTIVE, Lifecycle.DORMANT) in changes
    pmbl.apply_direct(bid, [archived(pmbl, fac, 5000.0, {"p": 0.31}, independence_group="d")])
    assert pmbl.node(bid).cell.lifecycle is Lifecycle.REACTIVATED
    kinds = [r.cell.lifecycle for r in pmbl.store.repo.revisions(bid)]
    assert kinds[0] is Lifecycle.CANDIDATE and Lifecycle.DORMANT in kinds  # every transition logged


def test_invalid_transition_rejected_before_write(pmbl, fac, archived):
    bid, _ = _confirmed(pmbl, fac, archived)
    pmbl.transition(bid, Lifecycle.RETIRED, stamp(10.0, CLOCK))
    n = len(pmbl.store.repo.revisions(bid))
    with pytest.raises((LifecycleTransitionError, LifecycleError)):
        pmbl.transition(bid, Lifecycle.ACTIVE, stamp(11.0, CLOCK))
    assert len(pmbl.store.repo.revisions(bid)) == n
    with pytest.raises(ValueError):
        pmbl.apply_direct(bid, [archived(pmbl, fac, 12.0, {"p": 0.3})])


def test_stale_candidate_is_rejected(pmbl, fac, archived):
    cell = pmbl.create_candidate([archived(pmbl, fac, 1.0, {"p": 0.9})], Domain.TECHNICAL, "weld")
    pmbl.sweep(stamp(1.0 + pmbl.cfg.pmbl.reject_after_s + 1, CLOCK))
    assert pmbl.store.repo.head(cell.belief_id).lifecycle is Lifecycle.REJECTED


def test_ablation_threshold_one_confirms_at_creation(pmbl):
    assert (
        LifecycleEngine(pmbl.cfg.pmbl.model_copy(update={"confirm_independent_observations": 1})).initial(1)
        is Lifecycle.CONFIRMED
    )
    assert LifecycleEngine(pmbl.cfg.pmbl).initial(1) is Lifecycle.CANDIDATE


def test_cc04_merge_preserves_lineage_evidence_and_provenance(pmbl, fac, archived):
    a, ev_a = _confirmed(pmbl, fac, archived, 0.30, 1.0)
    b, ev_b = _confirmed(pmbl, fac, archived, 0.32, 10.0)
    roots = {pmbl.node(a).cell.provenance_root, pmbl.node(b).cell.provenance_root}
    child = pmbl.merge(a, b, stamp(20.0, CLOCK))
    repo = pmbl.store.repo
    assert repo.lineage_ancestors(child.belief_id) == {a, b}
    assert repo.contributed_evidence(child.belief_id) == {e.evidence_id for e in ev_a + ev_b}
    closure = repo.provenance_closure(child.provenance_root)
    assert roots <= set(closure)
    assert all(e.provenance_id in closure for e in ev_a + ev_b)
    assert repo.head(a).lifecycle is Lifecycle.MERGED and repo.head(b).lifecycle is Lifecycle.MERGED
    assert child.independent_observation_count == 4
    assert closure[child.provenance_root].source_type is SourceType.RELATIONAL_INFERENCE


def test_cc04_split_redistributes_evidence_with_lineage(pmbl, fac, archived):
    bid, evs = _confirmed(pmbl, fac, archived)
    extra = archived(pmbl, fac, 3.0, {"p": 0.8}, independence_group="z")
    pmbl.apply_direct(bid, [extra])
    all_ids = [e.evidence_id for e in [*evs, extra]]
    with pytest.raises(ValueError):
        pmbl.split(bid, [all_ids[:1], all_ids[1:2]], stamp(4.0, CLOCK))  # evidence would be lost
    kids = pmbl.split(bid, [all_ids[:2], all_ids[2:]], stamp(4.0, CLOCK))
    repo = pmbl.store.repo
    assert [repo.lineage_ancestors(k.belief_id) for k in kids] == [{bid}, {bid}]
    assert set().union(*(repo.contributed_evidence(k.belief_id) for k in kids)) == set(all_ids)
    assert repo.head(bid).lifecycle is Lifecycle.SPLIT
    assert kids[1].claim("p").value == pytest.approx(0.8)


def test_cc02_late_evidence_is_explicit_or_rejected(pmbl, fac, repo, cfg, ids, archived):
    bid, _ = _confirmed(pmbl, fac, archived, t0=10.0)
    late = archived(pmbl, fac, 5.0, {"p": 0.35}, independence_group="late")
    out = pmbl.apply_direct(bid, [late])
    assert out.late_evidence_ids == (late.evidence_id,)
    last = repo.revisions(bid)[-1]
    assert (
        last.late
        and last.update_kind is UpdateKind.DIRECT
        and last.measurement_time_ns == late.timestamp.time_ns
    )
    assert last.cell.timestamp.time_ns > late.timestamp.time_ns  # belief time is not rolled back
    strict = PMBL(
        repo,
        fac.run_id,
        cfg.model_copy(update={"pmbl": cfg.pmbl.model_copy(update={"late_evidence_policy": "REJECT"})}),
        ids,
    )
    late2 = archived(strict, fac, 4.0, {"p": 0.35}, independence_group="late2")
    n = len(repo.revisions(bid))
    res = strict.apply_direct(bid, [late2])
    assert res.rejected_late_ids == (late2.evidence_id,) and len(repo.revisions(bid)) == n


def test_cc01_duplicate_delivery_single_contribution(pmbl, fac, archived):
    bid, evs = _confirmed(pmbl, fac, archived)
    count = pmbl.store.repo.head(bid).independent_observation_count
    n = len(pmbl.store.repo.revisions(bid))
    out = pmbl.apply_direct(bid, [evs[1]])
    assert out.result is not None and out.result.rejected_duplicate_ids == (evs[1].evidence_id,)
    assert len(pmbl.store.repo.revisions(bid)) == n
    assert pmbl.store.repo.head(bid).independent_observation_count == count


def test_cc09_reset_working_memory_preserves_persistence(pmbl, fac, archived):
    bid, _ = _confirmed(pmbl, fac, archived)
    pmbl.apply_direct(bid, [archived(pmbl, fac, 3.0, {"p": 0.9}, reliability=0.95, independence_group="k")])
    before_state = pmbl.node(bid).corrected
    before_counts = pmbl.store.repo.counts()
    pmbl.reset_working_memory()
    assert len(pmbl.graph) == 0
    assert pmbl.store.repo.counts() == before_counts
    rebuilt = pmbl.node(bid).corrected
    assert rebuilt.estimates["p"].mean == pytest.approx(before_state.estimates["p"].mean)
    assert rebuilt.estimates["p"].variance == pytest.approx(before_state.estimates["p"].variance)
    assert rebuilt.uc == pytest.approx(before_state.uc)
    assert {c.evidence_id for c in rebuilt.contradictions} == {
        c.evidence_id for c in before_state.contradictions
    }
    assert rebuilt.independence_groups == before_state.independence_groups


def test_query_api_returns_belief_cells(pmbl, fac, archived):
    bid, _ = _confirmed(pmbl, fac, archived)
    pmbl.reset_working_memory()
    got = pmbl.query(BeliefQuery(domain=Domain.TECHNICAL))
    assert [c.belief_id for c in got] == [bid]
    assert got[0].knowledge_status is KnowledgeStatus.OBSERVED
    assert pmbl.query(BeliefQuery(domain=Domain.SPATIAL)) == []
    assert pmbl.query(BeliefQuery(min_observational_uncertainty=0.99)) == []
