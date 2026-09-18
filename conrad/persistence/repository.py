"""Single-writer repository: atomic, idempotent belief + provenance persistence (ch28, ch34).

A crash yields zero complete commits or one complete commit; never half a belief update with
missing provenance (CC-03 / SS-03). Duplicate evidence never counts twice (CC-01). A stale
update cannot silently replace a newer state; late evidence is explicit (CC-02).

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID

from sqlalchemy import Connection, Engine, func, select

from conrad.persistence import db
from conrad.schemas.belief import (
    ALLOWED_LIFECYCLE_TRANSITIONS,
    BeliefCell,
    BeliefRevision,
    Lifecycle,
    Relationship,
    UpdateKind,
)
from conrad.schemas.events import RuntimeEvent
from conrad.schemas.observation import Evidence, Observation
from conrad.schemas.provenance import ProvenanceError, ProvenanceRecord, validate_provenance_dag


class RepositoryError(RuntimeError):
    pass


class DuplicateEvidenceError(RepositoryError):
    pass


class StaleUpdateError(RepositoryError):
    pass


class PredecessorMismatchError(RepositoryError):
    pass


class LifecycleError(RepositoryError):
    pass


class CommitStatus(str, Enum):
    COMMITTED = "COMMITTED"
    DUPLICATE_MESSAGE = "DUPLICATE_MESSAGE"


@dataclass(frozen=True)
class BeliefUpdate:
    message_id: UUID
    producer_version: str
    run_id: UUID
    revision: BeliefRevision
    provenance: Sequence[ProvenanceRecord]
    relationships: Sequence[Relationship] = ()
    lineage_parents: Sequence[UUID] = ()
    lineage_kind: str | None = None  # MERGE | SPLIT


@dataclass(frozen=True)
class CommitResult:
    status: CommitStatus
    belief_id: UUID
    revision: int
    independent_observation_count: int
    commit_sequence: int


@dataclass
class _FaultPlan:
    """Test-only crash injection points inside the write transaction."""

    fail_at: str | None = None
    hook: Callable[[str], None] | None = field(default=None)

    def check(self, point: str) -> None:
        if self.hook is not None:
            self.hook(point)
        if self.fail_at == point:
            raise RuntimeError(f"injected failure at {point}")


class Repository:
    def __init__(self, engine: Engine, require_migrated: bool = True) -> None:
        self.engine = engine
        if require_migrated:
            db.check_migration_state(engine)
        self.fault = _FaultPlan()

    # ---------------------------------------------------------------- simple inserts
    def put_observation(self, obs: Observation) -> bool:
        with self.engine.begin() as conn:
            if conn.execute(
                select(db.observations.c.observation_id).where(
                    db.observations.c.observation_id == str(obs.observation_id)
                )
            ).first():
                return False
            conn.execute(
                db.observations.insert().values(
                    observation_id=str(obs.observation_id),
                    run_id=str(obs.run_id),
                    mission_id=str(obs.mission_id),
                    sensor_id=str(obs.sensor_id),
                    modality=obs.modality.value,
                    measurement_time_ns=obs.timestamp.time_ns,
                    payload_digest=obs.payload_ref.digest if obs.payload_ref else None,
                    payload_json=obs.canonical_json(),
                )
            )
            return True

    def put_evidence(self, ev: Evidence) -> bool:
        """Idempotent on evidence_id. Returns False for a retried delivery."""
        with self.engine.begin() as conn:
            if conn.execute(
                select(db.evidence.c.evidence_id).where(db.evidence.c.evidence_id == str(ev.evidence_id))
            ).first():
                return False
            conn.execute(
                db.evidence.insert().values(
                    evidence_id=str(ev.evidence_id),
                    run_id=str(ev.run_id),
                    source_observation_id=str(ev.source_observation_id),
                    modality=ev.modality.value,
                    measurement_time_ns=ev.timestamp.time_ns,
                    independence_group=ev.independence_group,
                    payload_json=ev.canonical_json(),
                )
            )
            return True

    def put_provenance(self, run_id: UUID, records: Sequence[ProvenanceRecord]) -> None:
        with self.engine.begin() as conn:
            self._insert_provenance(conn, run_id, records)

    def put_event(self, ev: RuntimeEvent) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                db.events.insert().values(
                    event_id=str(ev.event_id),
                    run_id=str(ev.envelope.run_id),
                    sequence=ev.sequence,
                    event_type=ev.event_type.value,
                    trace_id=str(ev.trace_id),
                    measurement_time_ns=ev.envelope.measurement_time_ns,
                    payload_digest=ev.payload_digest,
                )
            )

    def put_command(
        self,
        run_id: UUID,
        command_id: UUID,
        trace_id: UUID,
        issued_ns: int,
        accepted: bool,
        reasons: Sequence[str],
        payload_json: str,
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                db.commands.insert().values(
                    command_id=str(command_id),
                    run_id=str(run_id),
                    trace_id=str(trace_id),
                    issued_time_ns=issued_ns,
                    accepted=accepted,
                    reason_codes=json.dumps(list(reasons)),
                    payload_json=payload_json,
                )
            )

    def put_artifact_ref(
        self, run_id: UUID, digest: str, role: str, media_type: str, byte_length: int
    ) -> None:
        with self.engine.begin() as conn:
            exists = conn.execute(
                select(db.artifact_refs.c.id).where(
                    (db.artifact_refs.c.run_id == str(run_id))
                    & (db.artifact_refs.c.digest == digest)
                    & (db.artifact_refs.c.role == role)
                )
            ).first()
            if not exists:
                conn.execute(
                    db.artifact_refs.insert().values(
                        run_id=str(run_id),
                        digest=digest,
                        role=role,
                        media_type=media_type,
                        byte_length=byte_length,
                    )
                )

    # ---------------------------------------------------------------- belief commits
    def _insert_provenance(self, conn: Connection, run_id: UUID, records: Sequence[ProvenanceRecord]) -> None:
        new = {r.record_id: r for r in records}
        parents_needed = {p for r in records for p in r.parent_records if p not in new}
        if parents_needed:
            found = {
                UUID(row[0])
                for row in conn.execute(
                    select(db.provenance_nodes.c.record_id).where(
                        db.provenance_nodes.c.record_id.in_([str(p) for p in parents_needed])
                    )
                )
            }
            missing = parents_needed - found
            if missing:
                raise ProvenanceError(f"dangling provenance parents: {sorted(str(m) for m in missing)}")
        # cycle check inside the new batch (parents already stored are acyclic by construction)
        closed = {
            rid: r.model_copy(update={"parent_records": tuple(p for p in r.parent_records if p in new)})
            for rid, r in new.items()
        }
        validate_provenance_dag(closed, list(closed))
        existing = {
            UUID(row[0])
            for row in conn.execute(
                select(db.provenance_nodes.c.record_id).where(
                    db.provenance_nodes.c.record_id.in_([str(r) for r in new])
                )
            )
        }
        for rid, rec in new.items():
            if rid in existing:
                continue
            conn.execute(
                db.provenance_nodes.insert().values(
                    record_id=str(rid),
                    run_id=str(run_id),
                    source_type=rec.source_type.value,
                    subject_id=str(rec.subject_id) if rec.subject_id else None,
                    payload_json=rec.canonical_json(),
                )
            )
            for parent in rec.parent_records:
                conn.execute(db.provenance_edges.insert().values(child_id=str(rid), parent_id=str(parent)))

    def commit_update(self, update: BeliefUpdate) -> CommitResult:
        rev = update.revision
        bid = str(rev.belief_id)
        with self.engine.begin() as conn:
            # 1. idempotency on (message_id, producer_version)
            prior = conn.execute(
                select(
                    db.belief_revisions.c.belief_id,
                    db.belief_revisions.c.revision,
                    db.belief_revisions.c.commit_sequence,
                ).where(
                    (db.belief_revisions.c.message_id == str(update.message_id))
                    & (db.belief_revisions.c.producer_version == update.producer_version)
                )
            ).first()
            if prior is not None:
                count = conn.execute(
                    select(db.beliefs.c.independent_observation_count).where(
                        db.beliefs.c.belief_id == prior[0]
                    )
                ).scalar_one()
                return CommitResult(
                    CommitStatus.DUPLICATE_MESSAGE, UUID(prior[0]), int(prior[1]), int(count), int(prior[2])
                )

            head = conn.execute(select(db.beliefs).where(db.beliefs.c.belief_id == bid)).mappings().first()

            # 2. predecessor / staleness
            if head is None:
                if rev.revision != 0:
                    raise PredecessorMismatchError(f"belief {bid} does not exist; first revision must be 0")
            else:
                if rev.predecessor_revision != head["head_revision"]:
                    raise PredecessorMismatchError(
                        f"belief {bid}: expected predecessor {head['head_revision']}, got {rev.predecessor_revision}"
                    )
                if (
                    rev.measurement_time_ns < head["head_measurement_time_ns"]
                    and not rev.late
                    and rev.update_kind is UpdateKind.DIRECT
                ):
                    raise StaleUpdateError(
                        f"belief {bid}: measurement {rev.measurement_time_ns} precedes head "
                        f"{head['head_measurement_time_ns']} and is not marked late"
                    )
                old = Lifecycle(head["lifecycle"])
                new = rev.cell.lifecycle
                if new is not old and new not in ALLOWED_LIFECYCLE_TRANSITIONS[old]:
                    raise LifecycleError(f"illegal lifecycle transition {old.value} -> {new.value}")

            # 3. duplicate evidence contributions
            ev_ids = [str(e) for e in rev.consumed_evidence_ids]
            if len(set(ev_ids)) != len(ev_ids):
                raise DuplicateEvidenceError("evidence listed twice within one update")
            if ev_ids:
                dup = conn.execute(
                    select(db.evidence_contributions.c.evidence_id).where(
                        (db.evidence_contributions.c.belief_id == bid)
                        & (db.evidence_contributions.c.evidence_id.in_(ev_ids))
                    )
                ).first()
                if dup is not None:
                    raise DuplicateEvidenceError(f"evidence {dup[0]} already contributed to belief {bid}")
                known = {
                    row[0]: row[1]
                    for row in conn.execute(
                        select(db.evidence.c.evidence_id, db.evidence.c.independence_group).where(
                            db.evidence.c.evidence_id.in_(ev_ids)
                        )
                    )
                }
                missing = set(ev_ids) - set(known)
                if missing:
                    raise RepositoryError(f"consumed evidence not archived: {sorted(missing)}")

            # 4. provenance DAG must resolve before anything is written
            root_known = any(r.record_id == rev.provenance_root for r in update.provenance)
            if (
                not root_known
                and not conn.execute(
                    select(db.provenance_nodes.c.record_id).where(
                        db.provenance_nodes.c.record_id == str(rev.provenance_root)
                    )
                ).first()
            ):
                raise ProvenanceError(f"provenance root {rev.provenance_root} is not resolvable")

            seq = (
                int(
                    conn.execute(
                        select(func.coalesce(func.max(db.belief_revisions.c.commit_sequence), 0))
                    ).scalar_one()
                )
                + 1
            )

            # 5. independent-observation accounting
            independent_total = int(head["independent_observation_count"]) if head else 0
            if ev_ids:
                prior_groups = {
                    (row[1] or row[0])
                    for row in conn.execute(
                        select(db.evidence.c.evidence_id, db.evidence.c.independence_group)
                        .join(
                            db.evidence_contributions,
                            db.evidence_contributions.c.evidence_id == db.evidence.c.evidence_id,
                        )
                        .where(db.evidence_contributions.c.belief_id == bid)
                    )
                }
            else:
                prior_groups = set()

            self.fault.check("before_writes")
            self._insert_provenance(conn, update.run_id, update.provenance)
            self.fault.check("after_provenance")

            if head is None:
                conn.execute(
                    db.beliefs.insert().values(
                        belief_id=bid,
                        run_id=str(update.run_id),
                        domain=rev.cell.domain.value,
                        lifecycle=rev.cell.lifecycle.value,
                        head_revision=0,
                        head_measurement_time_ns=rev.measurement_time_ns,
                        independent_observation_count=0,
                        registry_entity_id=str(rev.cell.registry_entity_id)
                        if rev.cell.registry_entity_id
                        else None,
                    )
                )
            contributions = []
            for eid in ev_ids:
                group = known[eid] or eid
                is_independent = group not in prior_groups
                prior_groups.add(group)
                independent_total += int(is_independent)
                contributions.append(
                    {
                        "belief_id": bid,
                        "evidence_id": eid,
                        "revision": rev.revision,
                        "independent": is_independent,
                    }
                )
            if contributions:
                conn.execute(db.evidence_contributions.insert(), contributions)
            self.fault.check("after_contributions")

            stored_cell = rev.cell.model_copy(update={"independent_observation_count": independent_total})
            stored = rev.model_copy(update={"cell": stored_cell})
            conn.execute(
                db.belief_revisions.insert().values(
                    belief_id=bid,
                    revision=rev.revision,
                    predecessor_revision=rev.predecessor_revision,
                    update_kind=rev.update_kind.value,
                    measurement_time_ns=rev.measurement_time_ns,
                    late=rev.late,
                    provenance_root=str(rev.provenance_root),
                    message_id=str(update.message_id),
                    producer_version=update.producer_version,
                    commit_sequence=seq,
                    payload_json=stored.canonical_json(),
                )
            )
            self.fault.check("after_revision")
            head_time = (
                rev.measurement_time_ns
                if head is None
                else max(int(head["head_measurement_time_ns"]), rev.measurement_time_ns)
            )
            conn.execute(
                db.beliefs.update()
                .where(db.beliefs.c.belief_id == bid)
                .values(
                    lifecycle=rev.cell.lifecycle.value,
                    head_revision=rev.revision,
                    head_measurement_time_ns=head_time,
                    independent_observation_count=independent_total,
                )
            )
            for parent in update.lineage_parents:
                conn.execute(
                    db.belief_lineage.insert().values(
                        child_belief_id=bid,
                        parent_belief_id=str(parent),
                        kind=update.lineage_kind or "MERGE",
                        commit_sequence=seq,
                    )
                )
            for rel in update.relationships:
                if not conn.execute(
                    select(db.relationships.c.relationship_id).where(
                        db.relationships.c.relationship_id == str(rel.relationship_id)
                    )
                ).first():
                    conn.execute(
                        db.relationships.insert().values(
                            relationship_id=str(rel.relationship_id),
                            run_id=str(update.run_id),
                            source_belief_id=str(rel.source_belief_id),
                            target_belief_id=str(rel.target_belief_id),
                            relation_type=rel.relation_type,
                            payload_json=rel.canonical_json(),
                        )
                    )
            self.fault.check("before_commit")
            return CommitResult(CommitStatus.COMMITTED, rev.belief_id, rev.revision, independent_total, seq)

    # ---------------------------------------------------------------- reads
    def contributed_evidence(self, belief_id: UUID) -> set[UUID]:
        with self.engine.connect() as conn:
            return {
                UUID(r[0])
                for r in conn.execute(
                    select(db.evidence_contributions.c.evidence_id).where(
                        db.evidence_contributions.c.belief_id == str(belief_id)
                    )
                )
            }

    def head(self, belief_id: UUID) -> BeliefCell | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(db.belief_revisions.c.payload_json)
                .where(db.belief_revisions.c.belief_id == str(belief_id))
                .order_by(db.belief_revisions.c.revision.desc())
                .limit(1)
            ).first()
        return None if row is None else BeliefRevision.model_validate_json(row[0]).cell

    def revisions(self, belief_id: UUID) -> list[BeliefRevision]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(db.belief_revisions.c.payload_json)
                .where(db.belief_revisions.c.belief_id == str(belief_id))
                .order_by(db.belief_revisions.c.revision)
            ).all()
        return [BeliefRevision.model_validate_json(r[0]) for r in rows]

    def all_revisions(self, run_id: UUID | None = None) -> list[BeliefRevision]:
        stmt = select(db.belief_revisions.c.payload_json).order_by(db.belief_revisions.c.commit_sequence)
        if run_id is not None:
            stmt = stmt.join(db.beliefs, db.beliefs.c.belief_id == db.belief_revisions.c.belief_id).where(
                db.beliefs.c.run_id == str(run_id)
            )
        with self.engine.connect() as conn:
            return [BeliefRevision.model_validate_json(r[0]) for r in conn.execute(stmt)]

    def heads(self, run_id: UUID | None = None) -> list[BeliefCell]:
        stmt = select(db.belief_revisions.c.payload_json).join(
            db.beliefs,
            (db.beliefs.c.belief_id == db.belief_revisions.c.belief_id)
            & (db.beliefs.c.head_revision == db.belief_revisions.c.revision),
        )
        if run_id is not None:
            stmt = stmt.where(db.beliefs.c.run_id == str(run_id))
        with self.engine.connect() as conn:
            return [
                BeliefRevision.model_validate_json(r[0]).cell
                for r in conn.execute(stmt.order_by(db.belief_revisions.c.commit_sequence))
            ]

    def evidence(self, evidence_id: UUID) -> Evidence | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(db.evidence.c.payload_json).where(db.evidence.c.evidence_id == str(evidence_id))
            ).first()
        return None if row is None else Evidence.model_validate_json(row[0])

    def observation(self, observation_id: UUID) -> Observation | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(db.observations.c.payload_json).where(
                    db.observations.c.observation_id == str(observation_id)
                )
            ).first()
        return None if row is None else Observation.model_validate_json(row[0])

    def provenance_record(self, record_id: UUID) -> ProvenanceRecord | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(db.provenance_nodes.c.payload_json).where(
                    db.provenance_nodes.c.record_id == str(record_id)
                )
            ).first()
        return None if row is None else ProvenanceRecord.model_validate_json(row[0])

    def provenance_closure(self, root: UUID) -> dict[UUID, ProvenanceRecord]:
        """All records reachable from ``root``. Raises if any link dangles."""
        out: dict[UUID, ProvenanceRecord] = {}
        stack = [root]
        while stack:
            rid = stack.pop()
            if rid in out:
                continue
            rec = self.provenance_record(rid)
            if rec is None:
                raise ProvenanceError(f"dangling provenance reference {rid}")
            out[rid] = rec
            stack.extend(rec.parent_records)
        validate_provenance_dag(out, [root])
        return out

    def lineage_ancestors(self, belief_id: UUID) -> set[UUID]:
        """Every predecessor belief reachable through merge/split lineage (CC-04)."""
        seen: set[UUID] = set()
        stack = [belief_id]
        with self.engine.connect() as conn:
            while stack:
                cur = stack.pop()
                for row in conn.execute(
                    select(db.belief_lineage.c.parent_belief_id).where(
                        db.belief_lineage.c.child_belief_id == str(cur)
                    )
                ):
                    parent = UUID(row[0])
                    if parent not in seen:
                        seen.add(parent)
                        stack.append(parent)
        return seen

    def referenced_digests(self, run_id: UUID) -> list[str]:
        with self.engine.connect() as conn:
            refs = {
                r[0]
                for r in conn.execute(
                    select(db.artifact_refs.c.digest).where(db.artifact_refs.c.run_id == str(run_id))
                )
            }
            refs |= {
                r[0]
                for r in conn.execute(
                    select(db.observations.c.payload_digest).where(
                        (db.observations.c.run_id == str(run_id))
                        & (db.observations.c.payload_digest.is_not(None))
                    )
                )
            }
        return sorted(refs)

    def counts(self) -> dict[str, int]:
        with self.engine.connect() as conn:
            return {
                t.name: int(conn.execute(select(func.count()).select_from(t)).scalar_one())
                for t in db.metadata.sorted_tables
            }
