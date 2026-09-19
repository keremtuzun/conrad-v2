"""StructuralBeliefEngine: the analytic Model2T core (direct update, temporal prediction, TCDP, context).

No persistence and no messaging here (Model2T adds those), so baselines can reuse the same core.
Provenance records are minted per change and buffered per component until drained.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from uuid import UUID

from conrad.domains.technical.config import Model2TConfig, PropagationMode
from conrad.domains.technical.context import ContextEffect, apply_context
from conrad.domains.technical.coverage import SurfaceGeometry
from conrad.domains.technical.direct import DirectOutcome, apply_direct
from conrad.domains.technical.dynamics import predict_belief
from conrad.domains.technical.measurement import tail_moments
from conrad.domains.technical.registry import AssetRegistry
from conrad.domains.technical.state import ComponentBelief, Estimate, new_belief
from conrad.domains.technical.tcdp import EdgeKey, edge_key, infer
from conrad.schemas.belief import KnowledgeStatus, Relationship
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Evidence
from conrad.schemas.provenance import ProvenanceRecord, SourceType
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.uncertainty import Uncertainty
from conrad.schemas.world import Domain

MODULE = "conrad.domains.technical"


class StructuralBeliefEngine:
    def __init__(
        self,
        registry: AssetRegistry,
        config: Model2TConfig,
        ids: IdFactory,
        now: TimeStamp,
        mode: PropagationMode | None = None,
        geometry: Mapping[UUID, SurfaceGeometry] | None = None,
    ) -> None:
        if mode is not None:
            config = replace(config, tcdp=replace(config.tcdp, mode=mode))
        self.cfg, self.ids, self.registry = config, ids, registry
        self.now = now
        self.pending: dict[UUID, list[ProvenanceRecord]] = {}
        self.registry_record = ProvenanceRecord(
            record_id=ids.new(),
            source_type=SourceType.PRIOR,
            source_ids=(),
            operation="asset_registry_context",
            module=MODULE,
            model_version=config.model_version,
            timestamp=now,
        )
        self.beliefs: dict[UUID, ComponentBelief] = {}
        for rid in registry.stateful_ids:
            b = new_belief(
                ids.new(), registry.components[rid], config, now.time_ns, (geometry or {}).get(rid)
            )
            b.provenance_root = self._record(
                b, SourceType.PRIOR, (rid,), (self.registry_record.record_id,), "create_from_registry"
            )
            self.beliefs[rid] = b
        self.edge_ids: dict[EdgeKey, UUID] = {}
        self.relationships: dict[UUID, Relationship] = {}
        zero_u = Uncertainty(aleatoric=0.0, epistemic=0.0, contradiction=0.0, observational=0.0)
        for rel in registry.relations:
            src, dst = self.beliefs.get(rel.source), self.beliefs.get(rel.target)
            if src is None or dst is None:
                continue
            rel_id = ids.new()
            self.edge_ids[edge_key(rel.source, rel.target, rel.relation_type)] = rel_id
            self.relationships[rel_id] = Relationship(
                relationship_id=rel_id,
                relation_type=rel.relation_type,
                source_belief_id=src.belief_id,
                target_belief_id=dst.belief_id,
                source_domain=Domain.TECHNICAL,
                target_domain=Domain.TECHNICAL,
                confidence=1.0,
                uncertainty=zero_u,
                provenance_id=self.registry_record.record_id,
            )
            src.relationship_ids += (rel_id,)
            dst.relationship_ids += (rel_id,)
        for b in self.beliefs.values():
            if b.geometry is None:
                continue
            # read-surface part of the component (coverage, docs/audits/MODEL2T_REPAIR.md iteration 3)
            b.surface_id, rel_id = ids.new(), ids.new()
            b.surface_relationship = rel_id
            self.relationships[rel_id] = Relationship(
                relationship_id=rel_id,
                relation_type="PART_OF",
                source_belief_id=b.surface_id,
                target_belief_id=b.belief_id,
                source_domain=Domain.TECHNICAL,
                target_domain=Domain.TECHNICAL,
                confidence=1.0,
                uncertainty=zero_u,
                provenance_id=self.registry_record.record_id,
            )

    # ------------------------------------------------------------------ provenance
    def _record(
        self,
        belief: ComponentBelief,
        source_type: SourceType,
        source_ids: tuple[UUID, ...],
        extra_parents: tuple[UUID | None, ...],
        operation: str,
    ) -> UUID:
        parents = tuple(dict.fromkeys(p for p in (belief.provenance_root, *extra_parents) if p is not None))
        rec = ProvenanceRecord(
            record_id=self.ids.new(),
            source_type=source_type,
            source_ids=source_ids,
            operation=operation,
            module=MODULE,
            model_version=self.cfg.model_version,
            timestamp=self.now,
            parent_records=parents,
            subject_id=belief.belief_id,
        )
        self.pending.setdefault(belief.spec.registry_id, []).append(rec)
        belief.provenance_root = rec.record_id
        return rec.record_id

    def drain_provenance(self, registry_id: UUID) -> list[ProvenanceRecord]:
        return self.pending.pop(registry_id, [])

    # ------------------------------------------------------------------ association
    def associate(self, ev: Evidence) -> UUID | None:
        by_belief = {b.belief_id: rid for rid, b in self.beliefs.items()}
        best: tuple[float, UUID] | None = None
        for c in ev.entity_candidates:
            rid = c.registry_entity_id if c.registry_entity_id in self.beliefs else None
            if rid is None and c.belief_id is not None:
                rid = by_belief.get(c.belief_id)
            if rid is not None and (best is None or c.score > best[0]):
                best = (c.score, rid)
        return None if best is None else best[1]

    # ------------------------------------------------------------------ operators
    def apply_evidence(self, evidence: Sequence[Evidence], now: TimeStamp) -> dict[UUID, DirectOutcome]:
        self.now = now
        grouped: dict[UUID, list[Evidence]] = {}
        for ev in evidence:
            rid = self.associate(ev)
            if rid is not None:
                grouped.setdefault(rid, []).append(ev)
        out: dict[UUID, DirectOutcome] = {}
        for rid, evs in grouped.items():
            b = self.beliefs[rid]
            fresh = tuple(e.evidence_id for e in evs if e.evidence_id not in b.consumed)
            if not fresh:
                continue
            # Evidence provenance lives with the evidence archive; it is linked through source_ids.
            prov = self._record(b, SourceType.DIRECT_OBSERVATION, fresh, (), "direct_update")
            res = apply_direct(b, evs, prov, self.cfg)
            if res.changed:
                out[rid] = res
        return out

    def predict(self, delta_t_s: float, now: TimeStamp) -> set[UUID]:
        if delta_t_s < 0.0:
            raise ValueError("delta_t must be non-negative physical seconds")
        self.now = now
        changed: set[UUID] = set()
        if delta_t_s == 0.0:
            return changed
        for rid, b in self.beliefs.items():
            if not any(e.known for e in b.estimates.values()):
                predict_belief(b, delta_t_s, now.time_ns, self.registry_record.record_id, self.cfg)
                continue
            prov = self._record(b, SourceType.TEMPORAL_PREDICTION, (b.belief_id,), (), "temporal_prediction")
            if predict_belief(b, delta_t_s, now.time_ns, prov, self.cfg):
                changed.add(rid)
        return changed

    def propagate(self, now: TimeStamp) -> set[UUID]:
        self.now = now
        results = infer(self.beliefs, self.registry, self.edge_ids, self.cfg.tcdp, self._population_prior())
        changed: set[UUID] = set()
        support: dict[UUID, float] = {}
        for (rid, q), inf in sorted(results.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
            b = self.beliefs[rid]
            est = b.estimates[q]
            if inf is None:
                if est.known:
                    b.estimates[q] = b.prior[q].copy()
                    changed.add(rid)
                continue
            if est.status is KnowledgeStatus.INFERRED and abs(est.level - inf.mean) < 1e-12:
                support[rid] = max(support.get(rid, 0.0), inf.support)
                continue
            src_ids = tuple(s.belief_id for s in inf.sources) + tuple(
                s.relationship_id for s in inf.sources if s.relationship_id is not None
            )
            prov = self._record(
                b,
                SourceType.RELATIONAL_INFERENCE,
                src_ids,
                tuple(s.parent_provenance for s in inf.sources),
                f"tcdp_{self.cfg.tcdp.mode.value.lower()}:{q}",
            )
            b.estimates[q] = Estimate(
                level=inf.mean,
                level_var=inf.var,
                rate=inf.rate,
                rate_var=inf.rate_var,
                status=KnowledgeStatus.INFERRED,
                provenance_id=prov,
                direct_lineage=False,
                updated_ns=now.time_ns,
            )
            support[rid] = max(support.get(rid, 0.0), inf.support)
            changed.add(rid)
        for rid, b in self.beliefs.items():
            new = (
                support.get(rid, 0.0)
                if any(e.status is KnowledgeStatus.INFERRED for e in b.estimates.values())
                else 0.0
            )
            if new != b.propagated_support:
                b.propagated_support = new
                changed.add(rid)
        return changed

    def _population_prior(self) -> Callable[[str, Estimate], Estimate] | None:
        if self.cfg.direct.measurement_model != "SENSOR_CHARACTERISED":
            return None
        pc = self.cfg.prior

        def heavy(q: str, core: Estimate) -> Estimate:
            w, s = pc.tail_weight.get(q, 0.0), pc.tail_scale_m.get(q, 1.0)
            if w <= 0.0:
                return core
            m, v = tail_moments(core.level, core.level_var, w, s)
            return Estimate(level=m, level_var=v)

        return heavy

    def apply_context(self, effects: Sequence[ContextEffect], now: TimeStamp) -> set[UUID]:
        self.now = now
        level: dict[UUID, tuple[float, ContextEffect]] = {}
        for eff in effects:
            targets = (
                list(self.beliefs)
                if eff.target is None
                else [eff.target]
                if eff.target in self.beliefs
                else []
            )
            for rid in targets:
                if rid not in level or eff.obscuration > level[rid][0]:
                    level[rid] = (eff.obscuration, eff)
        changed: set[UUID] = set()
        for rid, (obs, eff) in level.items():
            b = self.beliefs[rid]
            if apply_context(b, obs, self.cfg.context):
                self._record(
                    b, SourceType.CROSS_DOMAIN_CONTEXT, (eff.message_id, *eff.provenance_refs), (), "context"
                )
                changed.add(rid)
        return changed
