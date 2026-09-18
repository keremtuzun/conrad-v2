# Development

The binding rules are in [development/WORKSTREAM_BRIEF.md](development/WORKSTREAM_BRIEF.md) and, for the integrated
mission, [development/INTEGRATION_BRIEF.md](development/INTEGRATION_BRIEF.md). This page is the short version.

## Rules

- Truth stays in `conrad.twins`, `conrad.sim`, `conrad.training`, `conrad.evaluation`, `conrad.data` and tests.
  Deployment packages never import it (`tests/leakage`).
- Only `conrad.runtime.command_gateway` calls `RobotHardwareInterface.send`.
- Models subclass `ConradModel` / `VersionedModel` (frozen, `extra="forbid"`). IDs come from an injected
  `IdFactory`; randomness from a seeded `numpy.random.Generator` or `torch.Generator`; time is injected.
- Uncertainty is always the four-channel `Uncertainty`. Knowledge status is property-level.
- Dimensions and thresholds come from config, never from constants inside modules.
- Never fabricate measurements or results. Unmeasured values are `SourceKind.OPEN`, `SYNTHETIC_ONLY` or
  `ENGINEERING_ESTIMATE`.
- No legacy repository references, no vendor SDKs in core packages, no `Twin1` / `Model2A/B/C` identifiers.
- Each package `__init__.py` defines `IMPLEMENTATION_METADATA` (implementation_status, source_sections,
  configuration_keys, assumptions, baselines, acceptance_tests, claim_status). `claim_status` stays at most
  `IMPLEMENTED` unless an executed experiment supports more.

## Tooling

```
uv run ruff format --check .      # line length 110
uv run ruff check .
uv run mypy conrad tests          # strict for conrad/ (disallow_untyped_defs); tests may be untyped
uv run pytest -q
uv run python scripts/secret_scan.py
```

`scripts/lint.sh` and `scripts/test.sh` wrap these. On this Windows host use `python -m uv run ...`.

## Test layout

| Directory | What it checks |
|---|---|
| `tests/unit`, `tests/property` | Module behaviour; hypothesis property tests |
| `tests/contract` | Frozen contracts: persistence (CC-01..04), command gateway, P0 schemas, CI hygiene, Core, Unity parity and frames |
| `tests/leakage` | Static AST boundaries (see [ARCHITECTURE.md](ARCHITECTURE.md)) |
| `tests/integration` | Gate I0 fake full system |
| `tests/replay` | Bundle verification, backup and restore, release lanes |
| `tests/simulation` | Twin episodes, 2S/Twin2S loop, nav benchmarks |
| `tests/hardware_stub` | Mock Unity player over real ZeroMQ |
| `tests/regression`, `tests/acceptance` | Empty at the time of writing, although CI runs them |

Pytest markers: `slow`, `gpu`, `legacy_independence` (`--strict-markers`).

## CI (`.github/workflows`)

- `ci.yml`, job `cpu` (ubuntu-22.04, Python 3.11): legacy repository absent, `uv lock --check`,
  `uv sync --all-groups --locked`, ruff format and check, mypy, `secret_scan.py`, unit and property tests,
  contract tests, leakage tests, `conrad doctor`.
- `ci.yml`, job `main-integration` (main only): integration, replay, simulation, regression and acceptance tests.
- `contract.yml`: contract and leakage tests on PRs that touch schemas, persistence or contract tests.
- `nightly.yml`: full `pytest -q`, then the core smoke training run.

No CI job sets command_mode `hardware`.

## Changing things

- Schema changes: `conrad/schemas` is a frozen contract (`SCHEMA_VERSION = "1.0.0"`). Same-major versions are
  compatible (`check_schema_compatible`); a breaking change needs a major bump and a migration note.
- Database changes: add an Alembic revision under `conrad/persistence/migrations/versions`.
  `tests/contract/test_ss09_ci_hygiene.py` fails on migration drift.
- New experiments: register them in `conrad/evaluation/dispatch.py` or a package-level `EXPERIMENTS` dict (see
  [EVALUATION.md](EVALUATION.md)).
- After adding docs or implementation paths, refresh the ledger:
  `uv run python scripts/build_requirements_ledger.py`.
- The integrator commits; contributors and agents do not run git.
