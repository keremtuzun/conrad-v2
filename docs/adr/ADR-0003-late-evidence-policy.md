# ADR-0003 Late-evidence policy

- Date: 2026-09-18
- Packages: conrad.persistence.repository, conrad.core.pmbl. Source: ch28 (delayed-measurement policy OPEN).
- Status: PROPOSED

## Options
(a) bounded rewind/replay of the belief from the late measurement time; (b) explicit late handling: the update is committed as a revision flagged `late=True` that keeps its true measurement time and never advances the head measurement time; (c) reject.
## Decision
V0.2 implements (b) as default `EXPLICIT_LATE` and (c) `REJECT` as a config alternative (`runtime.late_evidence_policy`). A DIRECT update older than the head without the late flag raises `StaleUpdateError`. Bounded rewind (a) stays OPEN pending an experiment comparing (a) vs (b) on delayed-observation episodes.
## Evidence
tests/contract/test_cc_persistence.py::test_cc02_late_observation_never_masquerades_as_current.
## Approval
Kerem: PENDING REVIEW
