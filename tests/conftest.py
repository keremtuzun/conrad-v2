from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from conrad.persistence.db import make_engine, migrate
from conrad.persistence.object_store import ObjectStore
from conrad.persistence.repository import Repository
from conrad.schemas.belief import BeliefCell, BeliefRevision, KnowledgeStatus, Lifecycle, PropertyClaim, UpdateKind
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Evidence, Modality
from conrad.schemas.provenance import ProvenanceRecord, SourceType
from conrad.schemas.timebase import TimeStamp, stamp
from conrad.schemas.uncertainty import Uncertainty
from conrad.schemas.world import Domain

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def ids() -> IdFactory:
    return IdFactory(seed=7)


@pytest.fixture
def repo(tmp_path: Path) -> Repository:
    db_path = tmp_path / "conrad.sqlite"
    migrate(db_path)
    return Repository(make_engine(db_path))


@pytest.fixture
def store(tmp_path: Path) -> ObjectStore:
    return ObjectStore(tmp_path / "objects")


def ts(t: float, seq: int = 0) -> TimeStamp:
    return stamp(t, "SIM", seq)


def make_evidence(ids: IdFactory, run_id: UUID, t: float = 1.0, group: str | None = None) -> tuple[Evidence, ProvenanceRecord]:
    eid, oid, pid = ids.new(), ids.new(), ids.new()
    prov = ProvenanceRecord(
        record_id=pid,
        source_type=SourceType.DIRECT_OBSERVATION,
        source_ids=(oid,),
        operation="encode",
        module="test.ecmer",
        model_version="t0",
        timestamp=ts(t),
        subject_id=eid,
    )
    ev = Evidence(
        evidence_id=eid,
        source_observation_id=oid,
        mission_id=ids.new(),
        run_id=run_id,
        trace_id=ids.new(),
        modality=Modality.STRUCTURED,
        timestamp=ts(t),
        created_time_ns=ts(t).time_ns + 1000,
        embedding=(0.1, 0.2),
        reliability=0.9,
        aleatoric_uncertainty=0.1,
        independence_group=group,
        provenance_id=pid,
        encoder_version="t0",
    )
    return ev, prov


def make_revision(
    ids: IdFactory,
    belief_id: UUID,
    revision: int,
    t: float,
    evidence_ids: tuple[UUID, ...],
    parents: tuple[UUID, ...],
    kind: UpdateKind = UpdateKind.DIRECT,
    lifecycle: Lifecycle = Lifecycle.CANDIDATE,
    late: bool = False,
) -> tuple[BeliefRevision, ProvenanceRecord]:
    pid = ids.new()
    source = {
        UpdateKind.DIRECT: SourceType.DIRECT_OBSERVATION,
        UpdateKind.CREATE: SourceType.DIRECT_OBSERVATION,
        UpdateKind.RELATIONAL: SourceType.RELATIONAL_INFERENCE,
        UpdateKind.PREDICTED: SourceType.TEMPORAL_PREDICTION,
        UpdateKind.CONTEXT: SourceType.CROSS_DOMAIN_CONTEXT,
        UpdateKind.LIFECYCLE: SourceType.PRIOR,
    }[kind]
    prov = ProvenanceRecord(
        record_id=pid,
        source_type=source,
        source_ids=evidence_ids or (belief_id,),
        operation=kind.value.lower(),
        module="test.buo",
        model_version="t0",
        timestamp=ts(t),
        parent_records=parents,
        subject_id=belief_id,
    )
    status = {
        UpdateKind.RELATIONAL: KnowledgeStatus.INFERRED,
        UpdateKind.PREDICTED: KnowledgeStatus.PREDICTED,
    }.get(kind, KnowledgeStatus.OBSERVED)
    unc = Uncertainty(aleatoric=0.1, epistemic=0.2, contradiction=0.0, observational=0.3)
    cell = BeliefCell(
        belief_id=belief_id,
        domain=Domain.TECHNICAL,
        entity_type="pipeline_segment",
        lifecycle=lifecycle,
        revision=revision,
        timestamp=ts(t),
        state_embedding=(0.0, 1.0),
        claims=(PropertyClaim(name="severity", value=0.4, units="unitless", status=status, uncertainty=unc, provenance_id=pid),),
        knowledge_status=status,
        uncertainty=unc,
        provenance_root=pid,
        model_version="t0",
    )
    rev = BeliefRevision(
        belief_id=belief_id,
        revision=revision,
        predecessor_revision=None if revision == 0 else revision - 1,
        measurement_time_ns=ts(t).time_ns,
        consumed_evidence_ids=evidence_ids,
        provenance_root=pid,
        update_kind=kind,
        cell=cell,
        late=late,
    )
    return rev, prov
