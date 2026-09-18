"""initial Conrad V2 persistence schema

Revision ID: 0001_initial
Revises:
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "artifact_refs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("digest", sa.String(length=64), nullable=False),
        sa.Column("role", sa.String(length=48), nullable=False),
        sa.Column("media_type", sa.String(length=64), nullable=False),
        sa.Column("byte_length", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "digest", "role", name="uq_artifact_ref"),
    )
    op.create_index(op.f("ix_artifact_refs_digest"), "artifact_refs", ["digest"], unique=False)
    op.create_index(op.f("ix_artifact_refs_run_id"), "artifact_refs", ["run_id"], unique=False)
    op.create_table(
        "belief_lineage",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("child_belief_id", sa.String(length=36), nullable=False),
        sa.Column("parent_belief_id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=8), nullable=False),
        sa.Column("commit_sequence", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_belief_lineage_child_belief_id"), "belief_lineage", ["child_belief_id"], unique=False
    )
    op.create_index(
        op.f("ix_belief_lineage_parent_belief_id"), "belief_lineage", ["parent_belief_id"], unique=False
    )
    op.create_table(
        "beliefs",
        sa.Column("belief_id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("domain", sa.String(length=16), nullable=False),
        sa.Column("lifecycle", sa.String(length=16), nullable=False),
        sa.Column("head_revision", sa.Integer(), nullable=False),
        sa.Column("head_measurement_time_ns", sa.BigInteger(), nullable=False),
        sa.Column("independent_observation_count", sa.Integer(), nullable=False),
        sa.Column("registry_entity_id", sa.String(length=36), nullable=True),
        sa.PrimaryKeyConstraint("belief_id"),
    )
    op.create_index(op.f("ix_beliefs_run_id"), "beliefs", ["run_id"], unique=False)
    op.create_table(
        "commands",
        sa.Column("command_id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("trace_id", sa.String(length=36), nullable=False),
        sa.Column("issued_time_ns", sa.BigInteger(), nullable=False),
        sa.Column("accepted", sa.Boolean(), nullable=False),
        sa.Column("reason_codes", sa.Text(), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("command_id"),
    )
    op.create_index(op.f("ix_commands_run_id"), "commands", ["run_id"], unique=False)
    op.create_index(op.f("ix_commands_trace_id"), "commands", ["trace_id"], unique=False)
    op.create_table(
        "dataset_manifests",
        sa.Column("dataset_id", sa.String(length=96), nullable=False),
        sa.Column("version", sa.String(length=32), nullable=False),
        sa.Column("manifest_digest", sa.String(length=64), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("dataset_id", "version"),
    )
    op.create_table(
        "events",
        sa.Column("event_id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=40), nullable=False),
        sa.Column("trace_id", sa.String(length=36), nullable=False),
        sa.Column("measurement_time_ns", sa.BigInteger(), nullable=False),
        sa.Column("payload_digest", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint("event_id"),
        sa.UniqueConstraint("run_id", "sequence", name="uq_event_sequence"),
    )
    op.create_index("ix_events_trace", "events", ["trace_id"], unique=False)
    op.create_table(
        "evidence",
        sa.Column("evidence_id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("source_observation_id", sa.String(length=36), nullable=False),
        sa.Column("modality", sa.String(length=24), nullable=False),
        sa.Column("measurement_time_ns", sa.BigInteger(), nullable=False),
        sa.Column("independence_group", sa.String(length=128), nullable=True),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("evidence_id"),
    )
    op.create_index(op.f("ix_evidence_run_id"), "evidence", ["run_id"], unique=False)
    op.create_index(
        op.f("ix_evidence_source_observation_id"), "evidence", ["source_observation_id"], unique=False
    )
    op.create_table(
        "experiments",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("experiment_id", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("claim_status", sa.String(length=24), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_experiments_experiment_id"), "experiments", ["experiment_id"], unique=False)
    op.create_table(
        "migration_history",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("revision", sa.String(length=32), nullable=False),
        sa.Column("applied_time_ns", sa.BigInteger(), nullable=False),
        sa.Column("code_version", sa.String(length=32), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "observations",
        sa.Column("observation_id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("mission_id", sa.String(length=36), nullable=False),
        sa.Column("sensor_id", sa.String(length=36), nullable=False),
        sa.Column("modality", sa.String(length=24), nullable=False),
        sa.Column("measurement_time_ns", sa.BigInteger(), nullable=False),
        sa.Column("payload_digest", sa.String(length=64), nullable=True),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("observation_id"),
    )
    op.create_index(op.f("ix_observations_run_id"), "observations", ["run_id"], unique=False)
    op.create_table(
        "provenance_nodes",
        sa.Column("record_id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("subject_id", sa.String(length=36), nullable=True),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("record_id"),
    )
    op.create_index(op.f("ix_provenance_nodes_run_id"), "provenance_nodes", ["run_id"], unique=False)
    op.create_index(op.f("ix_provenance_nodes_subject_id"), "provenance_nodes", ["subject_id"], unique=False)
    op.create_table(
        "relationships",
        sa.Column("relationship_id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("source_belief_id", sa.String(length=36), nullable=False),
        sa.Column("target_belief_id", sa.String(length=36), nullable=False),
        sa.Column("relation_type", sa.String(length=48), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("relationship_id"),
    )
    op.create_index(op.f("ix_relationships_run_id"), "relationships", ["run_id"], unique=False)
    op.create_index(
        op.f("ix_relationships_source_belief_id"), "relationships", ["source_belief_id"], unique=False
    )
    op.create_index(
        op.f("ix_relationships_target_belief_id"), "relationships", ["target_belief_id"], unique=False
    )
    op.create_table(
        "runs",
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("parent_run_id", sa.String(length=36), nullable=True),
        sa.Column("architecture_id", sa.String(length=64), nullable=False),
        sa.Column("stack_id", sa.String(length=64), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("lane", sa.String(length=16), nullable=False),
        sa.Column("git_commit", sa.String(length=64), nullable=True),
        sa.Column("git_dirty", sa.Boolean(), nullable=False),
        sa.Column("config_digest", sa.String(length=64), nullable=False),
        sa.Column("seed", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("comparability", sa.String(length=16), nullable=False),
        sa.Column("created_time_ns", sa.BigInteger(), nullable=False),
        sa.Column("finished_time_ns", sa.BigInteger(), nullable=True),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_table(
        "scenarios",
        sa.Column("scenario_id", sa.String(length=36), nullable=False),
        sa.Column("scenario_version", sa.String(length=32), nullable=False),
        sa.Column("seed", sa.BigInteger(), nullable=False),
        sa.Column("digest", sa.String(length=64), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("scenario_id"),
    )
    op.create_table(
        "belief_revisions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("belief_id", sa.String(length=36), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("predecessor_revision", sa.Integer(), nullable=True),
        sa.Column("update_kind", sa.String(length=16), nullable=False),
        sa.Column("measurement_time_ns", sa.BigInteger(), nullable=False),
        sa.Column("late", sa.Boolean(), nullable=False),
        sa.Column("provenance_root", sa.String(length=36), nullable=False),
        sa.Column("message_id", sa.String(length=36), nullable=False),
        sa.Column("producer_version", sa.String(length=64), nullable=False),
        sa.Column("commit_sequence", sa.Integer(), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["belief_id"],
            ["beliefs.belief_id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("belief_id", "revision", name="uq_belief_revision"),
        sa.UniqueConstraint("message_id", "producer_version", name="uq_revision_idempotency"),
    )
    op.create_table(
        "evidence_contributions",
        sa.Column("belief_id", sa.String(length=36), nullable=False),
        sa.Column("evidence_id", sa.String(length=36), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("independent", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(
            ["belief_id"],
            ["beliefs.belief_id"],
        ),
        sa.PrimaryKeyConstraint("belief_id", "evidence_id"),
    )
    op.create_table(
        "provenance_edges",
        sa.Column("child_id", sa.String(length=36), nullable=False),
        sa.Column("parent_id", sa.String(length=36), nullable=False),
        sa.ForeignKeyConstraint(
            ["child_id"],
            ["provenance_nodes.record_id"],
        ),
        sa.PrimaryKeyConstraint("child_id", "parent_id"),
    )


def downgrade() -> None:
    op.drop_table("provenance_edges")
    op.drop_table("evidence_contributions")
    op.drop_table("belief_revisions")
    op.drop_table("scenarios")
    op.drop_table("runs")
    op.drop_index(op.f("ix_relationships_target_belief_id"), table_name="relationships")
    op.drop_index(op.f("ix_relationships_source_belief_id"), table_name="relationships")
    op.drop_index(op.f("ix_relationships_run_id"), table_name="relationships")
    op.drop_table("relationships")
    op.drop_index(op.f("ix_provenance_nodes_subject_id"), table_name="provenance_nodes")
    op.drop_index(op.f("ix_provenance_nodes_run_id"), table_name="provenance_nodes")
    op.drop_table("provenance_nodes")
    op.drop_index(op.f("ix_observations_run_id"), table_name="observations")
    op.drop_table("observations")
    op.drop_table("migration_history")
    op.drop_index(op.f("ix_experiments_experiment_id"), table_name="experiments")
    op.drop_table("experiments")
    op.drop_index(op.f("ix_evidence_source_observation_id"), table_name="evidence")
    op.drop_index(op.f("ix_evidence_run_id"), table_name="evidence")
    op.drop_table("evidence")
    op.drop_index("ix_events_trace", table_name="events")
    op.drop_table("events")
    op.drop_table("dataset_manifests")
    op.drop_index(op.f("ix_commands_trace_id"), table_name="commands")
    op.drop_index(op.f("ix_commands_run_id"), table_name="commands")
    op.drop_table("commands")
    op.drop_index(op.f("ix_beliefs_run_id"), table_name="beliefs")
    op.drop_table("beliefs")
    op.drop_index(op.f("ix_belief_lineage_parent_belief_id"), table_name="belief_lineage")
    op.drop_index(op.f("ix_belief_lineage_child_belief_id"), table_name="belief_lineage")
    op.drop_table("belief_lineage")
    op.drop_index(op.f("ix_artifact_refs_run_id"), table_name="artifact_refs")
    op.drop_index(op.f("ix_artifact_refs_digest"), table_name="artifact_refs")
    op.drop_table("artifact_refs")
