"""Model 2 Core forward step (ch3 "Model 2 Core forward sequence", ch7 predict/correct).

predict (TBD, physical delta_t) -> associate (retrieval + scorer/NO_MATCH) -> BUO correct ->
RBP propagate (INFERRED) -> PMBL commit with provenance -> lifecycle sweep.

The analytic operators are the default runtime path. Learned modules are optional and only move
latent tensors (see :mod:`conrad.core.pipeline_latent`).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from uuid import UUID

from torch import Tensor

from conrad.core.association import AssociationDecision, AssociationEngine
from conrad.core.config import CoreConfig
from conrad.core.lifecycle import is_terminal
from conrad.core.pipeline_latent import LatentModules, latent_correct, latent_predict, latent_propagate
from conrad.core.pmbl import PMBL
from conrad.core.rbp_analytic import AnalyticRBP
from conrad.core.tbd_analytic import AnalyticTBD, PropertyDynamics
from conrad.persistence.repository import Repository
from conrad.schemas.belief import BeliefCell, Lifecycle, Relationship
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Evidence
from conrad.schemas.provenance import ProvenanceRecord
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.world import Domain


@dataclass
class StepResult:
    decisions: list[AssociationDecision] = field(default_factory=list)
    created: list[UUID] = field(default_factory=list)
    updated: list[UUID] = field(default_factory=list)
    inferred: list[UUID] = field(default_factory=list)
    predicted: list[UUID] = field(default_factory=list)
    duplicates: list[UUID] = field(default_factory=list)
    late_evidence: list[UUID] = field(default_factory=list)
    rejected_late: list[UUID] = field(default_factory=list)
    surprise: dict[UUID, float] = field(default_factory=dict)
    lifecycle_changes: list[tuple[UUID, Lifecycle, Lifecycle]] = field(default_factory=list)


class Model2Core:
    def __init__(
        self,
        cfg: CoreConfig,
        repo: Repository,
        run_id: UUID,
        ids: IdFactory,
        domain: Domain,
        entity_type: str,
        association: AssociationEngine | None = None,
        dynamics: Mapping[str, PropertyDynamics] | None = None,
        latent: LatentModules | None = None,
        persist_predictions: bool = True,
        use_rbp: bool = False,  # ADR-0006: off by default after CORE-RBP-E001 contamination result
    ) -> None:
        self.cfg = cfg
        self.domain = domain
        self.entity_type = entity_type
        self.pmbl = PMBL(repo, run_id, cfg, ids)
        self.association = association or AssociationEngine(cfg)
        self.tbd = AnalyticTBD(cfg.tbd)
        self.rbp = AnalyticRBP(cfg.rbp, cfg.analytic_buo.contradiction_sigma)
        self.dynamics = dict(dynamics or {})
        self.latent = latent or LatentModules()
        self.latent.eval()
        self.persist_predictions = persist_predictions
        self.use_rbp = use_rbp

    # ------------------------------------------------------------------ stages
    def _live_cells(self) -> list[BeliefCell]:
        return [
            c for c in self.pmbl.store.heads() if c.domain is self.domain and not is_terminal(c.lifecycle)
        ]

    def predict(self, target: TimeStamp, result: StepResult) -> None:
        for cell in self._live_cells():
            if cell.timestamp.clock_domain != target.clock_domain:
                continue
            dt = target.delta_s(cell.timestamp)
            if dt <= 0:
                continue  # never predict backwards; late evidence is handled by PMBL policy
            node = self.pmbl.node(cell.belief_id)
            # the head (cell.timestamp) is the predicted state when one exists; predict from THAT state
            base = (node.predicted or node.corrected) if self.persist_predictions else node.corrected
            pred = self.tbd.predict(base, dt, self.dynamics)
            if self.latent.tbd is not None:
                latent_predict(self.latent.tbd, node, dt)
            if self.persist_predictions and pred.state.estimates:
                self.pmbl.apply_prediction(cell.belief_id, pred.state, target)
            else:
                node.predicted = pred.state
            result.predicted.append(cell.belief_id)

    def associate(
        self, evidence: Sequence[Evidence], result: StepResult, registry: Mapping[UUID, UUID] | None = None
    ) -> dict[UUID, list[Evidence]]:
        """Sequential in canonical order so a candidate created from one item can absorb the next."""
        pending: dict[UUID, list[Evidence]] = {}
        cells = self._live_cells()
        for ev in sorted(evidence, key=lambda e: (e.timestamp.time_ns, e.evidence_id.int)):
            decision = self.association.associate(ev, cells, self.domain, self.entity_type)
            result.decisions.append(decision)
            if decision.belief_id is not None:
                pending.setdefault(decision.belief_id, []).append(ev)
                continue
            reg = (registry or {}).get(ev.evidence_id)
            cell = self.pmbl.create_candidate([ev], self.domain, self.entity_type, registry_entity_id=reg)
            result.created.append(cell.belief_id)
            cells.append(cell)
        return pending

    def correct(self, pending: Mapping[UUID, list[Evidence]], result: StepResult) -> None:
        for belief_id in sorted(pending, key=lambda b: b.int):
            evs = pending[belief_id]
            node = self.pmbl.node(belief_id)
            z: Tensor | None = None
            if self.latent.buo is not None:
                z = latent_correct(self.latent.buo, node, evs)
            outcome = self.pmbl.apply_direct(belief_id, evs, z=z)
            result.late_evidence += outcome.late_evidence_ids
            result.rejected_late += outcome.rejected_late_ids
            if outcome.result is not None:
                result.duplicates += outcome.result.rejected_duplicate_ids
                if outcome.result.changed:
                    result.updated.append(belief_id)
                    result.surprise[belief_id] = outcome.result.temporal_surprise

    def propagate(self, sources: Sequence[UUID], now: TimeStamp, result: StepResult) -> None:
        """Only edges out of beliefs directly updated in THIS step carry messages (one per source revision)."""
        src = set(sources)
        rels = [r for r in self.pmbl.graph.relationships.values() if r.source_belief_id in src]
        if not rels:
            return
        involved = {r.source_belief_id for r in rels} | {r.target_belief_id for r in rels}
        live = {c.belief_id for c in self._live_cells()}
        states = {
            b: self.pmbl.node(b).predicted or self.pmbl.node(b).corrected for b in involved if b in live
        }
        latent_z = latent_propagate(self.latent.rbp, self.pmbl.graph) if self.latent.rbp is not None else {}
        by_id = {r.relationship_id: r for r in rels}
        for inf in self.rbp.propagate(states, rels):
            if inf.belief_id in src:
                continue  # a directly observed belief this step keeps its direct revision
            z = latent_z.get(inf.belief_id)
            self.pmbl.apply_relational(
                inf.belief_id,
                inf.state,
                inf.source_belief_ids,
                [by_id[r] for r in inf.relationship_ids],
                now,
                z=z,
            )
            result.inferred.append(inf.belief_id)

    # ------------------------------------------------------------------ public
    def add_relationships(self, relationships: Sequence[Relationship]) -> None:
        for rel in relationships:
            self.pmbl.graph.add_relationship(rel)

    def forward_step(
        self,
        evidence: Sequence[tuple[Evidence, Sequence[ProvenanceRecord]]],
        now: TimeStamp,
        relationships: Sequence[Relationship] = (),
        registry: Mapping[UUID, UUID] | None = None,
        sweep: bool = True,
    ) -> StepResult:
        """One step. ``now`` is the step time; evidence must not be newer than ``now``.

        Prediction targets the earliest evidence time of the step (or ``now`` when there is none),
        so evidence is never mislabelled as late by the step's own prediction.
        """
        result = StepResult()
        if any(
            e.timestamp.clock_domain != now.clock_domain or e.timestamp.time_ns > now.time_ns
            for e, _ in evidence
        ):
            raise ValueError("evidence must share the step clock domain and not be newer than `now`")
        items: list[Evidence] = []
        for ev, prov in evidence:
            # CC-01: a re-delivered evidence object is already archived and is never associated again
            if self.pmbl.archive_evidence(ev, prov):
                items.append(ev)
            else:
                result.duplicates.append(ev.evidence_id)
        self.add_relationships(relationships)
        target = min((e.timestamp for e in items), key=lambda t: t.time_ns) if items else now
        self.predict(target, result)
        pending = self.associate(items, result, registry)
        self.correct(pending, result)
        if self.use_rbp:
            self.propagate([*result.updated, *result.created], now, result)
        if sweep:
            result.lifecycle_changes = self.pmbl.sweep(now)
        return result

    def reset_working_memory(self) -> None:
        self.pmbl.reset_working_memory()

    def beliefs(self) -> list[BeliefCell]:
        return self._live_cells()
