"""Model2T: the structural BELIEF child of Model 2 (ch10, ch33). BELIEF PLANE ONLY.

Default operators are analytic (sensor-characterised direct update, regime-mixture crack prediction,
engineering-prior wall-loss prediction, surface coverage from surveyed design geometry). Relational
propagation is OFF by default (ADR-0009: gate 2T-TCDP failed); analytic TCDP / GENERIC run only when a mode is
passed explicitly (EXPERIMENTAL arm). The learned TCDP (``learned_tcdp``) is never on this runtime path.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

from conrad.domains.base import Model2Child
from conrad.domains.technical.config import Model2TConfig, PropagationMode
from conrad.domains.technical.context import context_effects
from conrad.domains.technical.coverage import geometry_from_context
from conrad.domains.technical.engine import StructuralBeliefEngine
from conrad.domains.technical.persist import build_message, commit_revision
from conrad.domains.technical.registry import AssetRegistry
from conrad.domains.technical.state import ComponentBelief
from conrad.persistence.repository import Repository
from conrad.schemas.belief import (
    Availability,
    BeliefMessage,
    BeliefQuery,
    KnowledgeStatus,
    PropertyClaim,
    UpdateKind,
    summarize_status,
)
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Evidence
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.world import Domain


class Model2TNotInitialized(RuntimeError):
    pass


class Model2T(Model2Child):
    domain = Domain.TECHNICAL
    model_version = "model2t-analytic-0.1.0"

    def __init__(
        self,
        ids: IdFactory,
        config: Model2TConfig | None = None,
        *,
        repository: Repository | None = None,
        run_id: UUID | None = None,
        mode: PropagationMode | None = None,
    ) -> None:
        self.config = config or Model2TConfig()
        self.model_version = self.config.model_version
        self.ids = ids
        self.repository = repository
        self.run_id = run_id or ids.new()
        self._mode = mode
        self._engine: StructuralBeliefEngine | None = None
        self._pending: list[Evidence] = []
        self._unassociated = 0
        self._committed_relationships: set[UUID] = set()

    # ------------------------------------------------------------------ lifecycle
    @property
    def engine(self) -> StructuralBeliefEngine:
        if self._engine is None:
            raise Model2TNotInitialized("initialize(context) with an asset registry first")
        return self._engine

    @property
    def beliefs(self) -> dict[UUID, ComponentBelief]:
        """Persistent component beliefs keyed by asset-registry id."""
        return self.engine.beliefs

    def initialize(self, context: dict[str, Any]) -> None:
        registry = AssetRegistry.from_context(context)
        now = context.get("timestamp") or TimeStamp(time_ns=0, clock_domain=self.config.clock_domain)
        geometry = geometry_from_context(context, self.config.coverage)
        unknown = set(geometry) - set(registry.components)
        if unknown:
            raise ValueError(f"design_geometry names non-registry components: {sorted(map(str, unknown))}")
        self._engine = StructuralBeliefEngine(registry, self.config, self.ids, now, self._mode, geometry)
        occlusion = context.get("surface_occlusion")
        if occlusion is not None:
            # Belief-side occlusion test supplied by the deployment (Model2S map), never Twin truth: cells a
            # reading's footprint reaches but the sensor could not have seen are not credited as covered.
            if not callable(occlusion):
                raise TypeError("context['surface_occlusion'] must be callable")
            for belief in self._engine.beliefs.values():
                belief.occlusion = occlusion
        if self.repository is not None:
            self.repository.put_provenance(self.run_id, [self._engine.registry_record])
        for rid in self._engine.beliefs:
            self._commit(rid, UpdateKind.CREATE, now)

    def ingest(self, evidence: Sequence[Evidence]) -> None:
        engine = self.engine
        for ev in evidence:
            if engine.associate(ev) is None:
                self._unassociated += 1
                continue
            if self.repository is not None:
                self.repository.put_evidence(ev)
            self._pending.append(ev)

    def update_beliefs(self, now: TimeStamp) -> list[BeliefMessage]:
        engine = self.engine
        pending, self._pending = self._pending, []
        direct = engine.apply_evidence(pending, now)
        relational = engine.propagate(now) if engine.cfg.tcdp.mode is not PropagationMode.NONE else set()
        out: list[BeliefMessage] = []
        for rid, res in sorted(direct.items(), key=lambda kv: str(kv[0])):
            summary = f"direct:{len(res.accepted)} conflicts:{len(res.conflicts)}"
            b = engine.beliefs[rid]
            if b.surface_id is not None and b.condition_open():
                # Partial coverage: the readings determine the READ SURFACE, not the component worst case.
                # The surface part takes the DIRECT revision (it consumes the evidence); the component gets a
                # coverage revision through PART_OF, with its worst case still UNKNOWN.
                out.append(
                    self._commit(
                        rid,
                        UpdateKind.DIRECT,
                        now,
                        res.accepted,
                        res.max_time_ns,
                        summary=summary,
                        surface=True,
                    )
                )
                out.append(self._commit(rid, UpdateKind.RELATIONAL, now, summary="surface coverage"))
                continue
            out.append(
                self._commit(rid, UpdateKind.DIRECT, now, res.accepted, res.max_time_ns, summary=summary)
            )
        for rid in sorted(relational - set(direct), key=str):
            out.append(self._commit(rid, UpdateKind.RELATIONAL, now, summary="tcdp"))
        return out

    def predict(self, delta_t_s: float, now: TimeStamp) -> list[BeliefMessage]:
        changed = self.engine.predict(delta_t_s, now)
        note = f"predicted over {delta_t_s:.0f} s"
        return [
            self._commit(rid, UpdateKind.PREDICTED, now, prediction=note) for rid in sorted(changed, key=str)
        ]

    def receive_context(self, messages: Sequence[BeliefMessage]) -> None:
        effects = context_effects(messages, self.config.context)
        if not effects:
            return
        now = max((m.timestamp for m in messages), key=lambda t: t.time_ns)
        for rid in sorted(self.engine.apply_context(effects, now), key=str):
            self._commit(rid, UpdateKind.CONTEXT, now, summary="cross-domain context")

    def query(self, query: BeliefQuery) -> list[BeliefMessage]:
        if query.domain not in (None, Domain.TECHNICAL):
            return []
        out: list[BeliefMessage] = []
        for rid, b in sorted(self.engine.beliefs.items(), key=lambda kv: str(kv[0])):
            if query.entity_ids and rid not in query.entity_ids:
                continue
            if query.belief_ids and b.belief_id not in query.belief_ids:
                continue
            u = b.uncertainty()
            if (
                query.min_observational_uncertainty is not None
                and u.observational < query.min_observational_uncertainty
            ):
                continue
            if (
                query.min_contradiction_uncertainty is not None
                and u.contradiction < query.min_contradiction_uncertainty
            ):
                continue
            msg = self._message(b, self.engine.now)
            if (
                query.time_range_ns is not None
                and not query.time_range_ns[0] <= msg.timestamp.time_ns <= query.time_range_ns[1]
            ):
                continue
            filtered = self._filter_claims(msg, query)
            if filtered is not None:
                out.append(filtered)
            if len(out) >= query.max_results:
                break
        return out

    def export_beliefs(self) -> list[BeliefMessage]:
        """Every component belief, then every read-surface part that has been committed."""
        items = sorted(self.engine.beliefs.items(), key=lambda kv: str(kv[0]))
        out = [self._message(b, self.engine.now) for _, b in items]
        out += [self._message(b, self.engine.now, surface=True) for _, b in items if b.surface_revision >= 0]
        return out

    def reset_working_memory(self) -> None:
        """Drops un-applied evidence and counters; persistent beliefs, relationships and archive survive."""
        self._pending = []
        self._unassociated = 0

    def availability(self) -> Availability:
        if self._engine is None:
            return Availability.UNAVAILABLE
        return Availability.DEGRADED if self._unassociated else Availability.AVAILABLE

    # ------------------------------------------------------------------ internals
    def _message(self, b: ComponentBelief, now: TimeStamp, **kw: Any) -> BeliefMessage:
        return build_message(b, self.config, self.ids.new(), now, availability=self.availability(), **kw)

    @staticmethod
    def _filter_claims(msg: BeliefMessage, query: BeliefQuery) -> BeliefMessage | None:
        claims: tuple[PropertyClaim, ...] = msg.state_summary
        if not query.include_predictions:
            claims = tuple(c for c in claims if c.status is not KnowledgeStatus.PREDICTED)
        if query.requested_fields:
            claims = tuple(c for c in claims if c.name in query.requested_fields)
        if claims == msg.state_summary:
            return msg
        if not claims:
            return None
        status = summarize_status(claims)
        support = () if status is KnowledgeStatus.PREDICTED else msg.evidence_support
        return msg.model_copy(
            update={"state_summary": claims, "knowledge_status": status, "evidence_support": support}
        )

    def _commit(
        self,
        rid: UUID,
        kind: UpdateKind,
        now: TimeStamp,
        consumed: Sequence[UUID] = (),
        measurement_time_ns: int | None = None,
        *,
        summary: str | None = None,
        prediction: str | None = None,
        surface: bool = False,
    ) -> BeliefMessage:
        engine = self.engine
        b = engine.beliefs[rid]
        records = engine.drain_provenance(rid)
        message_id = self.ids.new()
        if self.repository is None:
            if surface:
                b.surface_revision += 1
            else:
                b.revision += 1
        else:
            rels = []
            if kind is UpdateKind.RELATIONAL or surface:
                own = (b.surface_relationship,) if surface and b.surface_relationship else b.relationship_ids
                rels = [engine.relationships[r] for r in own if r not in self._committed_relationships]
                self._committed_relationships.update(r.relationship_id for r in rels)
            commit_revision(
                self.repository,
                self.run_id,
                b,
                self.config,
                now,
                kind=kind,
                message_id=message_id,
                provenance=records,
                consumed=consumed,
                measurement_time_ns=measurement_time_ns,
                relationships=rels,
                surface=surface,
            )
        support = consumed if kind is UpdateKind.DIRECT else ()
        return build_message(
            b,
            self.config,
            message_id,
            now,
            evidence_support=support,
            change_summary=summary,
            prediction_summary=prediction,
            availability=self.availability(),
            surface=surface,
        )
