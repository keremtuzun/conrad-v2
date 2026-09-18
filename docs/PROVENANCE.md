# Provenance

Every belief, decision, plan and command can be traced back to raw observations through a provenance DAG.

## Contract (`conrad/schemas/provenance.py`, FROZEN_CONTRACT)

`ProvenanceRecord`: `record_id`, `source_type`, `source_ids`, `operation`, `module`, `model_version`,
`timestamp`, `parent_records`, `subject_id`.

`SourceType`:

| Value | Meaning |
|---|---|
| DIRECT_OBSERVATION | Derived from an observation (ECMER and domain encoders) |
| RELATIONAL_INFERENCE | RBP / TCDP / spatial relational inference (INFERRED) |
| TEMPORAL_PREDICTION | TBD prediction (PREDICTED) |
| CROSS_DOMAIN_CONTEXT | Context another domain received through the Belief Bus |
| PRIOR | Prior; may have no sources |
| DECISION, PLAN, COMMAND | Upward chain from EGDC, MCBR and the executive |
| SENSOR_ARTIFACT | Raw sensor artifact; may have no sources |

Validation: a record may not list itself as a parent, and every type except PRIOR and SENSOR_ARTIFACT needs
sources or parents. `validate_provenance_dag` rejects cycles and dangling references; `trace_to_sources` returns
the chain from a root down to the leaves.

## Storage (`conrad/persistence/repository.py`)

- `provenance_nodes` and `provenance_edges` tables. `commit_update` validates the DAG before writing, in the same
  transaction as the belief revision, so a crash never leaves a revision without provenance (CC-03).
- `Repository.provenance_closure(root)` returns every record reachable from a root.

## Rules found in the code

- Evidence keeps its observation's measurement time unchanged.
- A PREDICTED `BeliefMessage` cannot list evidence support; `BeliefMessage` requires `provenance_refs`.
- A BELIEF_CLAIM in a decision cannot be GROUNDED without `source_belief_ids`.
- Model2T writes relational messages only to components without direct lineage.
- Cross-domain deliveries are logged as CROSS_DOMAIN_CONTEXT and never touch another child's state.

## Evidence

`tests/integration/test_i0_fake_full_system.py::test_i0_full_causal_trace_from_command_to_raw_observation`
checks that a command's closure contains COMMAND, PLAN, DECISION and DIRECT_OBSERVATION records and ends at a
stored observation, and that the same trace ID links the observation, decision, plan and command events. The
integrated mission (in progress) adds PLAN and COMMAND records in `conrad/orchestration/deliberation.py` and
`executive.py`.
