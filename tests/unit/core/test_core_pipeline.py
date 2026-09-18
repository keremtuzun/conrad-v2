"""Model2Core.forward_step end-to-end on SQLite: predict -> associate -> BUO -> RBP -> PMBL."""

import pytest
import torch

from conrad.core.buo import BeliefUpdateOperator
from conrad.core.pipeline import Model2Core
from conrad.core.pipeline_latent import LatentModules
from conrad.core.rbp import RelationalBeliefPropagation
from conrad.core.tbd import GruTBD, PropertyDynamics
from conrad.schemas.belief import KnowledgeStatus, Relationship, UpdateKind
from conrad.schemas.provenance import SourceType
from conrad.schemas.timebase import stamp
from conrad.schemas.uncertainty import unknown_uncertainty
from conrad.schemas.world import Domain

CLOCK = "sandbox_clock"


def _step(core, fac, t, items, **kw):
    batch = [fac.make(t, m, **extra) for m, extra in items]
    return core.forward_step([(e, [r]) for e, r in batch], stamp(t, CLOCK), **kw), [e for e, _ in batch]


def test_no_match_creates_candidate_then_matches(core, fac):
    res, _ = _step(core, fac, 1.0, [({"p": 0.4}, {"center_m": (0, 0, 0)})])
    assert len(res.created) == 1 and res.decisions[0].belief_id is None
    res2, _ = _step(
        core, fac, 2.0, [({"p": 0.41}, {"center_m": (0.1, 0, 0)}), ({"p": 0.9}, {"center_m": (30, 0, 0)})]
    )
    assert res2.updated == res.created and len(res2.created) == 1
    head = core.pmbl.store.repo.head(res.created[0])
    assert head.independent_observation_count == 2


def test_prediction_is_persisted_as_predicted_and_kept_apart(core, fac, repo):
    core.dynamics = {"p": PropertyDynamics(0.0, 1e-3)}
    res, _ = _step(core, fac, 1.0, [({"p": 0.4}, {})])
    bid = res.created[0]
    corrected = core.pmbl.node(bid).corrected
    core.forward_step([], stamp(100.0, CLOCK))
    revs = repo.revisions(bid)
    assert revs[-1].update_kind is UpdateKind.PREDICTED and not revs[-1].consumed_evidence_ids
    assert revs[-1].cell.knowledge_status is KnowledgeStatus.PREDICTED
    rec = repo.provenance_record(revs[-1].provenance_root)
    assert rec.source_type is SourceType.TEMPORAL_PREDICTION
    node = core.pmbl.node(bid)
    assert node.corrected is corrected and node.predicted is not None
    assert node.predicted.estimates["p"].variance > corrected.estimates["p"].variance


def test_sequence_index_is_irrelevant_only_physical_time_matters(repo, fac, cfg, ids, tmp_path):
    def run(seq_index):
        core = Model2Core(
            cfg,
            repo,
            ids.new(),
            ids,
            Domain.SPATIAL,
            f"obj{seq_index}",
            dynamics={"p": PropertyDynamics(0.0, 1e-3)},
        )
        ev, rec = fac.make(1.0, {"p": 0.4})
        ev = ev.model_copy(
            update={"timestamp": ev.timestamp.model_copy(update={"sequence_index": seq_index})}
        )
        core.forward_step([(ev, [rec])], stamp(1.0, CLOCK))
        core.forward_step([], stamp(50.0, CLOCK, sequence_index=seq_index * 7))
        return core.pmbl.node(core.beliefs()[0].belief_id).predicted.estimates["p"].variance

    assert run(1) == pytest.approx(run(900))


def test_redelivered_evidence_is_duplicate_not_reassociated(core, fac):
    batch = fac.make(1.0, {"p": 0.4})
    core.forward_step([(batch[0], [batch[1]])], stamp(1.0, CLOCK))
    res = core.forward_step([(batch[0], [batch[1]])], stamp(2.0, CLOCK))
    assert res.duplicates == [batch[0].evidence_id] and not res.created and not res.updated


def test_evidence_newer_than_now_is_refused(core, fac):
    ev, rec = fac.make(5.0, {"p": 0.4})
    with pytest.raises(ValueError):
        core.forward_step([(ev, [rec])], stamp(1.0, CLOCK))


def test_relational_step_is_inferred_with_provenance(core, fac, repo, ids):
    res, _ = _step(
        core, fac, 1.0, [({"p": 0.4}, {"center_m": (0, 0, 0)}), ({"q": 0.7}, {"center_m": (20, 0, 0)})]
    )
    src, dst = res.created
    rel = Relationship(
        relationship_id=ids.new(),
        relation_type="ATTACHED",
        source_belief_id=src,
        target_belief_id=dst,
        source_domain=Domain.TECHNICAL,
        target_domain=Domain.TECHNICAL,
        confidence=0.9,
        uncertainty=unknown_uncertainty(),
        provenance_id=ids.new(),
    )
    uo_before = repo.head(dst).uncertainty.observational
    res2, _ = _step(core, fac, 2.0, [({"p": 0.42}, {"center_m": (0, 0, 0)})], relationships=[rel])
    assert res2.inferred == [dst]
    last = repo.revisions(dst)[-1]
    assert last.update_kind is UpdateKind.RELATIONAL and not last.consumed_evidence_ids
    assert last.cell.claim("p").status is KnowledgeStatus.INFERRED
    assert last.cell.claim("q").status is not KnowledgeStatus.INFERRED
    assert last.cell.uncertainty.observational >= uo_before
    assert repo.provenance_record(last.provenance_root).source_type is SourceType.RELATIONAL_INFERENCE
    assert src in repo.provenance_record(last.provenance_root).source_ids


def test_latent_learned_path_moves_belief_latent_not_copying_evidence(repo, fac, cfg, ids):
    latent = LatentModules(BeliefUpdateOperator(cfg), RelationalBeliefPropagation(cfg), GruTBD(cfg))
    core = Model2Core(cfg, repo, fac.run_id, ids, Domain.TECHNICAL, "c", latent=latent)
    res, _ = _step(core, fac, 1.0, [({"p": 0.4}, {})])
    res2, evs2 = _step(core, fac, 2.0, [({"p": 0.41}, {"center_m": (0.05, 0, 0)})])
    bid = res.created[0]
    z = core.pmbl.node(bid).z
    assert res2.updated == [bid]
    assert not torch.allclose(z, torch.zeros_like(z))
    assert not torch.allclose(z, torch.tensor(evs2[0].embedding))  # evidence is not belief
    assert repo.head(bid).state_embedding == tuple(float(v) for v in z.tolist())
