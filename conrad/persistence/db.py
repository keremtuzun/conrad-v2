"""SQLite (WAL, single writer) schema and engine (ch34 Persistence).

Stores UTC integer nanoseconds, UUID strings, SI values and immutable object digests only.
Schema changes happen through Alembic revisions; startup refuses a database newer than the code.

implementation_status: FROZEN_CONTRACT (backend choice recorded in ADR-0001)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Engine,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    text,
)

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

metadata = MetaData()


def _uuid(name: str, *args: Any, **kw: Any) -> Column[str]:
    return Column(name, String(36), *args, **kw)


runs = Table(
    "runs",
    metadata,
    _uuid("run_id", primary_key=True),
    _uuid("parent_run_id"),
    Column("architecture_id", String(64), nullable=False),
    Column("stack_id", String(64), nullable=False),
    Column("mode", String(16), nullable=False),
    Column("lane", String(16), nullable=False),
    Column("git_commit", String(64)),
    Column("git_dirty", Boolean, nullable=False),
    Column("config_digest", String(64), nullable=False),
    Column("seed", BigInteger, nullable=False),
    Column("status", String(24), nullable=False),
    Column("comparability", String(16), nullable=False),
    Column("created_time_ns", BigInteger, nullable=False),
    Column("finished_time_ns", BigInteger),
    Column("payload_json", Text, nullable=False),
)

scenarios = Table(
    "scenarios",
    metadata,
    _uuid("scenario_id", primary_key=True),
    Column("scenario_version", String(32), nullable=False),
    Column("seed", BigInteger, nullable=False),
    Column("digest", String(64), nullable=False),
    Column("payload_json", Text, nullable=False),
)

observations = Table(
    "observations",
    metadata,
    _uuid("observation_id", primary_key=True),
    _uuid("run_id", nullable=False, index=True),
    _uuid("mission_id", nullable=False),
    _uuid("sensor_id", nullable=False),
    Column("modality", String(24), nullable=False),
    Column("measurement_time_ns", BigInteger, nullable=False),
    Column("payload_digest", String(64)),
    Column("payload_json", Text, nullable=False),
)

evidence = Table(
    "evidence",
    metadata,
    _uuid("evidence_id", primary_key=True),
    _uuid("run_id", nullable=False, index=True),
    _uuid("source_observation_id", nullable=False, index=True),
    Column("modality", String(24), nullable=False),
    Column("measurement_time_ns", BigInteger, nullable=False),
    Column("independence_group", String(128)),
    Column("payload_json", Text, nullable=False),
)

beliefs = Table(
    "beliefs",
    metadata,
    _uuid("belief_id", primary_key=True),
    _uuid("run_id", nullable=False, index=True),
    Column("domain", String(16), nullable=False),
    Column("lifecycle", String(16), nullable=False),
    Column("head_revision", Integer, nullable=False),
    Column("head_measurement_time_ns", BigInteger, nullable=False),
    Column("independent_observation_count", Integer, nullable=False),
    _uuid("registry_entity_id"),
)

belief_revisions = Table(
    "belief_revisions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    _uuid("belief_id", ForeignKey("beliefs.belief_id"), nullable=False),
    Column("revision", Integer, nullable=False),
    Column("predecessor_revision", Integer),
    Column("update_kind", String(16), nullable=False),
    Column("measurement_time_ns", BigInteger, nullable=False),
    Column("late", Boolean, nullable=False),
    _uuid("provenance_root", nullable=False),
    _uuid("message_id", nullable=False),
    Column("producer_version", String(64), nullable=False),
    Column("commit_sequence", Integer, nullable=False),
    Column("payload_json", Text, nullable=False),
    UniqueConstraint("belief_id", "revision", name="uq_belief_revision"),
    UniqueConstraint("message_id", "producer_version", name="uq_revision_idempotency"),
)

evidence_contributions = Table(
    "evidence_contributions",
    metadata,
    _uuid("belief_id", ForeignKey("beliefs.belief_id"), primary_key=True),
    _uuid("evidence_id", primary_key=True),
    Column("revision", Integer, nullable=False),
    Column("independent", Boolean, nullable=False),
)

belief_lineage = Table(
    "belief_lineage",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    _uuid("child_belief_id", nullable=False, index=True),
    _uuid("parent_belief_id", nullable=False, index=True),
    Column("kind", String(8), nullable=False),  # MERGE | SPLIT
    Column("commit_sequence", Integer, nullable=False),
)

relationships = Table(
    "relationships",
    metadata,
    _uuid("relationship_id", primary_key=True),
    _uuid("run_id", nullable=False, index=True),
    _uuid("source_belief_id", nullable=False, index=True),
    _uuid("target_belief_id", nullable=False, index=True),
    Column("relation_type", String(48), nullable=False),
    Column("payload_json", Text, nullable=False),
)

provenance_nodes = Table(
    "provenance_nodes",
    metadata,
    _uuid("record_id", primary_key=True),
    _uuid("run_id", nullable=False, index=True),
    Column("source_type", String(32), nullable=False),
    _uuid("subject_id", index=True),
    Column("payload_json", Text, nullable=False),
)

provenance_edges = Table(
    "provenance_edges",
    metadata,
    _uuid("child_id", ForeignKey("provenance_nodes.record_id"), primary_key=True),
    _uuid("parent_id", primary_key=True),
)

commands = Table(
    "commands",
    metadata,
    _uuid("command_id", primary_key=True),
    _uuid("run_id", nullable=False, index=True),
    _uuid("trace_id", nullable=False, index=True),
    Column("issued_time_ns", BigInteger, nullable=False),
    Column("accepted", Boolean, nullable=False),
    Column("reason_codes", Text, nullable=False),
    Column("payload_json", Text, nullable=False),
)

events = Table(
    "events",
    metadata,
    _uuid("event_id", primary_key=True),
    _uuid("run_id", nullable=False),
    Column("sequence", Integer, nullable=False),
    Column("event_type", String(40), nullable=False),
    _uuid("trace_id", nullable=False),
    Column("measurement_time_ns", BigInteger, nullable=False),
    Column("payload_digest", String(64), nullable=False),
    UniqueConstraint("run_id", "sequence", name="uq_event_sequence"),
)
Index("ix_events_trace", events.c.trace_id)

artifact_refs = Table(
    "artifact_refs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    _uuid("run_id", nullable=False, index=True),
    Column("digest", String(64), nullable=False, index=True),
    Column("role", String(48), nullable=False),
    Column("media_type", String(64), nullable=False),
    Column("byte_length", BigInteger, nullable=False),
    UniqueConstraint("run_id", "digest", "role", name="uq_artifact_ref"),
)

dataset_manifests = Table(
    "dataset_manifests",
    metadata,
    Column("dataset_id", String(96), primary_key=True),
    Column("version", String(32), primary_key=True),
    Column("manifest_digest", String(64), nullable=False),
    Column("payload_json", Text, nullable=False),
)

experiments = Table(
    "experiments",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("experiment_id", String(64), nullable=False, index=True),
    _uuid("run_id", nullable=False),
    Column("claim_status", String(24), nullable=False),
    Column("payload_json", Text, nullable=False),
)

migration_history = Table(
    "migration_history",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("revision", String(32), nullable=False),
    Column("applied_time_ns", BigInteger, nullable=False),
    Column("code_version", String(32), nullable=False),
)


class DatabaseError(RuntimeError):
    pass


def make_engine(db_path: str | Path) -> Engine:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{path.as_posix()}", future=True)

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn: Any, _record: Any) -> None:
        # pysqlite's implicit transaction handling is disabled so that BEGIN IMMEDIATE is ours.
        dbapi_conn.isolation_level = None
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=FULL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.close()

    @event.listens_for(engine, "begin")
    def _begin_immediate(conn: Any) -> None:
        conn.exec_driver_sql("BEGIN IMMEDIATE")

    return engine


def _alembic_config(db_path: str | Path) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", MIGRATIONS_DIR.as_posix())
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{Path(db_path).as_posix()}")
    return cfg


def head_revision() -> str:
    head = ScriptDirectory.from_config(_alembic_config(":memory:")).get_current_head()
    if head is None:
        raise DatabaseError("no migration head found")
    return head


def current_revision(engine: Engine) -> str | None:
    with engine.connect() as conn:
        return MigrationContext.configure(conn).get_current_revision()


def migrate(db_path: str | Path) -> str:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    command.upgrade(_alembic_config(db_path), "head")
    return head_revision()


def check_migration_state(engine: Engine) -> None:
    """Refuse unmigrated, stale or newer-than-code databases. No automatic schema creation."""
    known = {s.revision for s in ScriptDirectory.from_config(_alembic_config(":memory:")).walk_revisions()}
    current = current_revision(engine)
    if current is None:
        raise DatabaseError("database is not migrated; run `conrad db migrate`")
    if current not in known:
        raise DatabaseError(f"database revision {current} is newer than this code supports")
    if current != head_revision():
        raise DatabaseError(
            f"database revision {current} is behind head {head_revision()}; run `conrad db migrate`"
        )


def integrity_check(engine: Engine) -> None:
    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA integrity_check")).scalar()
    if result != "ok":
        raise DatabaseError(f"sqlite integrity_check failed: {result}")
