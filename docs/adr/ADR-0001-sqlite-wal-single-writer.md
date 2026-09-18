# ADR-0001 SQLite WAL single-writer persistence

- Date: 2026-09-18
- Packages: conrad.persistence, conrad.core (PMBL). Source: ch28 Belief Updates, ch34 Persistence.
- Status: ACCEPTED (selected by the V0.2 stack freeze, ch34)

## Problem
ch28 left the persistence backend OPEN; ch34 later froze SQLite (WAL, single writer) plus a content-addressed object store as the V0.1 reference.
## Options
SQLite WAL single writer; PostgreSQL; embedded KV store.
## Decision
SQLite WAL, `BEGIN IMMEDIATE` transactions, Alembic revisions, no automatic schema creation. Blobs >1 MB and all raw sensor data live in `artifacts/objects/sha256/<digest>`.
## Evidence
tests/contract/test_cc_persistence.py (CC-01..CC-04, SS-03, SS-04), tests/contract/test_ss09_ci_hygiene.py (migration drift).
## Failure modes
Not a fleet-scale store. Single writer only.
## Migration / rollback
Repository API hides the backend; a later ADR may swap it with a data migration.
## Approval
Kerem: PENDING REVIEW
