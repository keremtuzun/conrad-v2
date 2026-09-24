"""Production mission adapter for the versioned local structural belief ledger.

Only associated Evidence with measured spatial support updates the selected
component. The analytic Model2T remains responsible for other registry assets.
No truth-plane package is imported here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

from conrad.domains.technical.context import context_effects
from conrad.domains.technical.model import Model2T
from conrad.domains.technical.spatial_local import (
    LocalCondition,
    LocalThresholds,
    SpatialModel2T,
)
from conrad.persistence.repository import BeliefUpdate
from conrad.schemas.belief import (
    BeliefCell,
    BeliefMessage,
    BeliefQuery,
    BeliefRevision,
    KnowledgeStatus,
    PropertyClaim,
    TechnicalLocalCell,
    TechnicalPayload,
    UpdateKind,
    summarize_status,
)
from conrad.schemas.capsule_surface import CapsuleSurfaceGrid
from conrad.schemas.observation import Evidence
from conrad.schemas.provenance import ProvenanceRecord, SourceType
from conrad.schemas.structural_sensor import StructuralSensorModelV2
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.uncertainty import Uncertainty
from conrad.schemas.world import Domain

MODEL_VERSION = "model2t-spatial-v1"


class SpatialMissionModel2T(Model2T):
    """Model2T child with one explicitly selected spatial component."""

    def __init__(
        self,
        *args: Any,
        registry_id: UUID,
        grid: CapsuleSurfaceGrid,
        sensor: StructuralSensorModelV2,
        thresholds: LocalThresholds,
        required_looks: int = 1,
        required_axial_fraction: tuple[float, float] = (0.0, 1.0),
        required_sectors: tuple[int, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.spatial_registry_id = registry_id
        self.spatial = SpatialModel2T(
            grid,
            sensor,
            thresholds,
            registry_id,
            required_looks=required_looks,
            require_depth=True,
            required_axial_fraction=required_axial_fraction,
            required_sectors=required_sectors,
        )
        self._spatial_pending: list[Evidence] = []
        self._spatial_root: UUID | None = None
        self._spatial_head: BeliefMessage | None = None

    def initialize(self, context: dict[str, Any]) -> None:
        super().initialize(context)
        if self.spatial_registry_id not in self.beliefs:
            raise ValueError("spatial target missing from asset registry")
        now = context.get("timestamp") or TimeStamp(time_ns=0, clock_domain=self.config.clock_domain)
        self._commit_spatial(now, (), initial=True)

    def ingest(self, evidence: Sequence[Evidence]) -> None:
        legacy: list[Evidence] = []
        for ev in evidence:
            target = any(c.registry_entity_id == self.spatial_registry_id for c in ev.entity_candidates)
            if target:
                if ev.structural_support is not None and self.spatial.ingest(ev):
                    self._spatial_pending.append(ev)
                else:
                    self.spatial.unresolved.append(ev.evidence_id)
            elif ev.structural_support is not None:
                raise ValueError("spatial structural evidence associated with an unconfigured component")
            else:
                legacy.append(ev)
        super().ingest(legacy)

    def update_beliefs(self, now: TimeStamp) -> list[BeliefMessage]:
        out = [m for m in super().update_beliefs(now) if m.world_entity_id != self.spatial_registry_id]
        if self._spatial_pending:
            pending, self._spatial_pending = self._spatial_pending, []
            out.append(self._commit_spatial(now, pending))
        return out

    def predict(self, delta_t_s: float, now: TimeStamp) -> list[BeliefMessage]:
        changed = self.engine.predict(delta_t_s, now)
        return [
            self._commit(rid, UpdateKind.PREDICTED, now, prediction=f"predicted over {delta_t_s:.0f} s")
            for rid in sorted(changed - {self.spatial_registry_id}, key=str)
        ]

    def receive_context(self, messages: Sequence[BeliefMessage]) -> None:
        # Context may update other components. It cannot confer local direct
        # support or a component condition on the spatial target.
        effects = context_effects(messages, self.config.context)
        if effects:
            now = max((m.timestamp for m in messages), key=lambda t: t.time_ns)
            changed = self.engine.apply_context(effects, now)
            for rid in sorted(changed - {self.spatial_registry_id}, key=str):
                self._commit(rid, UpdateKind.CONTEXT, now, summary="cross-domain context")

    def export_beliefs(self) -> list[BeliefMessage]:
        out = [m for m in super().export_beliefs() if m.world_entity_id != self.spatial_registry_id]
        if self._spatial_head is not None:
            out.append(self._spatial_head)
        return out

    def query(self, query: BeliefQuery) -> list[BeliefMessage]:
        out = [m for m in super().query(query) if m.world_entity_id != self.spatial_registry_id]
        head = self._spatial_head
        eligible = (
            head is not None
            and query.domain in (None, Domain.TECHNICAL)
            and (not query.entity_ids or self.spatial_registry_id in query.entity_ids)
            and (not query.belief_ids or head.belief_id in query.belief_ids)
            and (
                query.time_range_ns is None
                or query.time_range_ns[0] <= head.timestamp.time_ns <= query.time_range_ns[1]
            )
            and (
                query.min_observational_uncertainty is None
                or head.uncertainty.observational >= query.min_observational_uncertainty
            )
            and (
                query.min_contradiction_uncertainty is None
                or head.uncertainty.contradiction >= query.min_contradiction_uncertainty
            )
        )
        if eligible and head is not None:
            filtered = self._filter_claims(head, query)
            if filtered is not None:
                out.append(filtered)
        return out[: query.max_results]

    def reset_working_memory(self) -> None:
        super().reset_working_memory()
        self._spatial_pending.clear()

    def _commit_spatial(
        self, now: TimeStamp, evidence: Sequence[Evidence], *, initial: bool = False
    ) -> BeliefMessage:
        b = self.beliefs[self.spatial_registry_id]
        previous_root = self._spatial_root or b.provenance_root
        root = self.ids.new()
        record = ProvenanceRecord(
            record_id=root,
            source_type=SourceType.PRIOR if initial else SourceType.DIRECT_OBSERVATION,
            source_ids=tuple(ev.evidence_id for ev in evidence),
            operation="spatial_model2t_initial" if initial else "spatial_model2t_update",
            module=__name__,
            model_version=MODEL_VERSION,
            timestamp=now,
            parent_records=() if previous_root is None else (previous_root,),
            subject_id=b.belief_id,
        )
        revision = b.revision + 1
        required = [i for i in range(self.spatial.grid.n_cells) if self.spatial.required_rect(i) is not None]
        coverage = sum(self.spatial.required_coverage_fraction(i) for i in required) / len(required)
        state = self.spatial.condition()
        condition = (
            None
            if state is LocalCondition.UNKNOWN
            else ("INTACT" if state is LocalCondition.OBSERVED_INTACT else state.value)
        )
        groups = set().union(*(c.independent_groups for c in self.spatial.cells))
        local_cells = tuple(
            TechnicalLocalCell(
                index=i,
                condition=(
                    None
                    if self.spatial.cell_condition(i) is LocalCondition.UNKNOWN
                    else "INTACT"
                    if self.spatial.cell_condition(i) is LocalCondition.OBSERVED_INTACT
                    else self.spatial.cell_condition(i).value
                ),
                coverage=self.spatial.coverage_fraction(i),
                corrosion_upper_m=c.corrosion_upper_m,
                crack_length_upper_m=c.crack_upper_m,
                crack_depth_upper_m=c.crack_depth_upper_m,
                observation_count=len(c.evidence_ids),
                independent_observation_count=len(c.independent_groups),
                last_direct_observation_ns=c.last_time_ns,
                evidence_ids=tuple(c.evidence_ids),
                knowledge_status=(
                    KnowledgeStatus.UNKNOWN
                    if self.spatial.cell_condition(i) is LocalCondition.UNKNOWN
                    else KnowledgeStatus.OBSERVED
                ),
            )
            for i, c in enumerate(self.spatial.cells)
        )
        corr = max((c.corrosion_upper_m or 0.0 for c in self.spatial.cells), default=0.0)
        crack = max((c.crack_upper_m or 0.0 for c in self.spatial.cells), default=0.0)
        depth = max((c.crack_depth_upper_m or 0.0 for c in self.spatial.cells), default=0.0)
        u = Uncertainty(
            aleatoric=self.spatial.sensor.noise_sigma_m,
            # A declared surface that has not been inspected is an observation gap,
            # not uncertainty about the spatial model itself. EGDC supplies its
            # separate uncalibrated-source epistemic floor when applicable.
            epistemic=0.0,
            contradiction=0.0,
            # Component condition is a worst-case claim: even a small unresolved
            # required patch can hide a defect. Keep that claim above the generic
            # information threshold until every local intact warrant is met.
            observational=1.0 if state is LocalCondition.UNKNOWN else max(0.0, 1.0 - coverage),
        )
        cond_status = KnowledgeStatus.UNKNOWN if condition is None else KnowledgeStatus.OBSERVED
        claims = (
            PropertyClaim(
                name="condition",
                value=condition,
                status=cond_status,
                uncertainty=u,
                provenance_id=None if condition is None else root,
            ),
            PropertyClaim(
                name="surface_coverage",
                value=coverage,
                units="1",
                status=KnowledgeStatus.OBSERVED if coverage > 0 else KnowledgeStatus.UNKNOWN,
                uncertainty=u,
                provenance_id=root if coverage > 0 else None,
            )
            if coverage > 0
            else PropertyClaim(
                name="surface_coverage",
                value=None,
                units="1",
                status=KnowledgeStatus.UNKNOWN,
                uncertainty=u,
            ),
        )
        cell = BeliefCell(
            belief_id=b.belief_id,
            domain=Domain.TECHNICAL,
            entity_type=b.spec.component_type,
            registry_entity_id=self.spatial_registry_id,
            lifecycle=b.lifecycle,
            revision=revision,
            timestamp=now,
            state_embedding=(corr, crack, depth, coverage),
            claims=claims,
            knowledge_status=summarize_status(claims),
            uncertainty=u,
            independent_observation_count=len(groups),
            provenance_root=root,
            model_version=MODEL_VERSION,
        )
        message_id = self.ids.new()
        if self.repository is not None:
            self.repository.commit_update(
                BeliefUpdate(
                    message_id=message_id,
                    producer_version=MODEL_VERSION,
                    run_id=self.run_id,
                    revision=BeliefRevision(
                        belief_id=b.belief_id,
                        revision=revision,
                        predecessor_revision=b.revision,
                        measurement_time_ns=max(
                            (ev.timestamp.time_ns for ev in evidence), default=now.time_ns
                        ),
                        consumed_evidence_ids=tuple(ev.evidence_id for ev in evidence),
                        provenance_root=root,
                        update_kind=UpdateKind.CREATE if initial else UpdateKind.DIRECT,
                        cell=cell,
                        late=bool(evidence and min(ev.timestamp.time_ns for ev in evidence) < b.head_time_ns),
                    ),
                    provenance=(record,),
                )
            )
        b.revision = revision
        b.head_time_ns = max(b.head_time_ns, now.time_ns)
        self._spatial_root = root
        self._spatial_head = BeliefMessage(
            message_id=message_id,
            belief_id=b.belief_id,
            revision=revision,
            independent_observation_count=len(groups),
            world_entity_id=self.spatial_registry_id,
            domain=Domain.TECHNICAL,
            timestamp=now,
            state_summary=claims,
            state_embedding=cell.state_embedding,
            knowledge_status=cell.knowledge_status,
            uncertainty=u,
            evidence_support=tuple(ev.evidence_id for ev in evidence),
            provenance_refs=(root,),
            lifecycle=b.lifecycle,
            model_version=MODEL_VERSION,
            technical=TechnicalPayload(
                condition=condition,
                corrosion_depth_m=corr if coverage > 0 else None,
                crack_length_m=crack if coverage > 0 else None,
                direct_support=coverage,
                local_cells=local_cells,
            ),
        )
        return self._spatial_head
