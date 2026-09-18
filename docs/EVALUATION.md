# Evaluation

`conrad/evaluation` runs registered experiments, records them in a hash-chained registry, and holds metrics,
oracles, the kill protocol, acceptance records and claim rules. Executed results:
[research/EXPERIMENT_RESULTS_2026-09-18.md](research/EXPERIMENT_RESULTS_2026-09-18.md).

## Running experiments

```
uv run conrad eval list
uv run conrad eval run --experiment CORE-BUO-E001 [--config PATH] [--seeds 1,2,3]
```

`conrad eval list` shows the ten IDs hard-coded in `conrad/evaluation/dispatch.py`: CORE-ASSOC-E001,
CORE-BUO-E001, CORE-UNC-E001, CORE-RBP-E001, CORE-TBD-E001, CORE-PERSIST-E001, CORE-FULL-E001, M1-UIR-E001,
ACTIVE-MCBR-E001, COM-BAAC-E001. `_discover()` also merges an `EXPERIMENTS` dict from the `__init__` of
`spatial_experiments`, `structural_experiments`, `ecological_experiments` and `int_benchmarks`, but none of those
defines one there today (and `int_benchmarks` does not exist yet), so the domain experiments run through their own
entry points:

```
python -m conrad.evaluation.structural_experiments.run_all [out_dir] [ids...]   # 2T-E001..E004
python -m conrad.evaluation.ecological_experiments.run_all [out_dir] [names...] # 2E-E001..E003
python -m conrad.evaluation.spatial_experiments.run_all [--only ...] [--out ...] # 2S-E001..E004
python -m conrad.evaluation.decision_experiments [out_dir]                       # M1, ACTIVE, COM
python -m conrad.evaluation.core_experiments.run_all [--only ...]                # CORE-*
```

The nav benchmarks NAV-001..NAV-008 run as `tests/simulation/nav/test_nav_benchmark_suite.py`
(`conrad.evaluation.nav_benchmarks.run_benchmark`, scenarios in `configs/sim/nav_benchmarks.yaml`).

`conrad eval run --run <id>` evaluates a stored integrated-mission run through
`conrad.orchestration.evaluation.evaluate_run`, which reads the run's SQLite, `mission/` artifacts and
`truth/truth_record.json` from disk. It is part of the integrated mission, which was still being written at the
time of writing; no integrated-mission result is recorded.

## Outputs and registry

- `run_experiment` writes to `artifacts/experiments/<ID>/` plus `dispatch_record.json`, and appends an
  `ExperimentRecord` to `artifacts/registry/experiments.jsonl` with tier `T1_DEVELOPMENT` and outcome
  **always** `INCONCLUSIVE`. Dispatch never grants SUPPORTS or REFUTES.
- `registry/experiments.py`: outcomes PLANNED, SUPPORTS, REFUTES, INCONCLUSIVE, FAILED_RUN, NOT_EVALUABLE. SUPPORTS
  needs a clean git tree, a paired-seed effect whose bootstrap CI excludes zero, and at least one baseline. The
  registry is a hash-chained JSONL file (`RegistryIntegrityError` on tampering).
- `registry/kill.py`: `assess` returns KILL_CANDIDATE on leakage, a safety, deadline or calibration regression, a
  task answer of NO or REGRESSION, or cost over budget. Only `retain_with_adr` keeps a KILL_CANDIDATE.
- `registry/hypotheses.py`: seeded hypotheses (`H-...-NN`) and statuses.
- `registry/notebook.py`: append-only `research_notebook/entries.jsonl` (not created yet).
- `acceptance.py`: gate records; a threshold still `OPEN` gives NOT_EVALUABLE.
- `claims.py`: claim ladder from NONE to VALIDATED_IN_REPRESENTATIVE_ENVIRONMENT; `check_claim` raises
  `ClaimRefused`. Twin usefulness is NOT_EVALUABLE without real data (EXT-DATA-01).
- `maturity.py`: D0..D8 levels, each needing evidence links that exist.

## Metrics (`conrad/evaluation/metrics`)

Percentile bootstrap and paired-seed comparison, ECE / NLL / Brier, masked metrics (a metric over zero labels has
no value, never 0.0), unsupported-confidence and confident-wrong rates (threshold supplied by the caller).

## Oracles (`conrad/evaluation/oracle`)

`decision_oracle.py` (M1 fault fixtures), `occlusion_world.py` (MCBR world), `comm_oracle.py` (BAAC retained
information). All worlds are SYNTHETIC_ONLY.

## Rules

- Seeds default to 2026201, 2026202, 2026203.
- Where a baseline wins, it is reported as winning.
- All data is synthetic; nothing is real-data or physical validation.
- Tests: `uv run pytest tests/unit/evaluation -q`.
