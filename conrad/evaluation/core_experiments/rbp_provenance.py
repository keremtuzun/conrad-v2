"""Pipeline-level check that every relational revision is INFERRED with RELATIONAL_INFERENCE provenance
and never lowers observational uncertainty (used by CORE-RBP-E001)."""

from __future__ import annotations

from typing import Any

from conrad.core.config import CoreConfig
from conrad.core.pipeline import Model2Core
from conrad.evaluation.core_experiments.common import scratch_dir, temp_repository
from conrad.evaluation.core_experiments.evidence_factory import CLOCK, EvidenceFactory
from conrad.schemas.belief import KnowledgeStatus, Relationship, UpdateKind
from conrad.schemas.ids import IdFactory
from conrad.schemas.provenance import SourceType
from conrad.schemas.timebase import stamp
from conrad.schemas.uncertainty import unknown_uncertainty
from conrad.schemas.world import Domain


def provenance_check(cfg: CoreConfig, seed: int) -> dict[str, Any]:
    with scratch_dir() as tmp:
        repo = temp_repository(tmp)
        ids = IdFactory(seed)
        fac = EvidenceFactory(ids, cfg.evidence_dim)
        assert fac.run_id is not None
        core = Model2Core(cfg, repo, fac.run_id, ids, Domain.TECHNICAL, "component")
        a = fac.make(1.0, {"p0": 0.4}, center_m=(0.0, 0.0, 0.0), independence_group="a1")
        b = fac.make(1.0, {"p1": 0.7}, center_m=(10.0, 0.0, 0.0), independence_group="b1")
        step = core.forward_step([(a[0], [a[1]]), (b[0], [b[1]])], stamp(1.0, CLOCK))
        src, dst = step.created
        uo_before = core.pmbl.store.repo.head(dst)
        assert uo_before is not None
        rel = Relationship(
            relationship_id=ids.new(),
            relation_type="ATTACHED",
            source_belief_id=src,
            target_belief_id=dst,
            source_domain=Domain.TECHNICAL,
            target_domain=Domain.TECHNICAL,
            confidence=0.9,
            uncertainty=unknown_uncertainty(),
            provenance_id=uo_before.provenance_root,
        )
        a2 = fac.make(2.0, {"p0": 0.42}, center_m=(0.0, 0.0, 0.0), independence_group="a2")
        step2 = core.forward_step([(a2[0], [a2[1]])], stamp(2.0, CLOCK), relationships=[rel])
        revs = [r for r in repo.revisions(dst) if r.update_kind is UpdateKind.RELATIONAL]
        ok_prov = [
            (rec := repo.provenance_record(r.provenance_root)) is not None
            and rec.source_type is SourceType.RELATIONAL_INFERENCE
            for r in revs
        ]
        claim = revs[-1].cell.claim("p0") if revs else None
        prev = [r for r in repo.revisions(dst) if r.revision < (revs[-1].revision if revs else 0)]
        uo_prev = prev[-1].cell.uncertainty.observational if prev else float("nan")
        return {
            "relational_revisions": float(len(revs)),
            "inferred_targets_in_step": float(len(step2.inferred)),
            "provenance_relational_fraction": float(sum(ok_prov) / len(ok_prov)) if ok_prov else 0.0,
            "inferred_claim_status_ok": float(claim is not None and claim.status is KnowledgeStatus.INFERRED),
            "no_direct_evidence_claimed": float(all(not r.consumed_evidence_ids for r in revs)),
            "uo_not_lowered": float(bool(revs) and revs[-1].cell.uncertainty.observational >= uo_prev),
        }
