# Runbook

Operator procedures that the code supports today. Everything runs in simulation; there is no physical operation.
On this Windows host prefix commands with `python -m uv run` instead of `uv run`.

## 1. Prepare a machine

```
uv sync --all-groups
uv run conrad db migrate
uv run conrad doctor          # must print RESULT: OK
```

For another lane: `uv run conrad doctor --config configs/runtime/<file>.yaml`. Machine-local paths go in
`configs/local.yaml` (only `paths` and `device`).

## 2. Health check before any run

- `conrad doctor --json` for a machine-readable report.
- `uv run python scripts/secret_scan.py` before sharing a tree.
- `uv run pytest tests/contract tests/leakage -q` for contracts and truth boundaries.

## 3. Run experiments and training

```
uv run conrad eval list
uv run conrad eval run --experiment <ID>
uv run conrad train run --config configs/train/core_smoke.yaml
```

Outputs: `artifacts/experiments/<ID>/`, `artifacts/registry/experiments.jsonl`, `artifacts/runs/train-*`. Record
the run ID and never edit a sealed run; add notes under `notes/`.

## 4. Review a run

```python
from conrad.console import build_console, serve_console

build_console("artifacts/runs/<run_id>", "artifacts/runs/<run_id>/notes/console.html")
```

The console verifies digests first and is read-only ([CONSOLE.md](CONSOLE.md)). Replay checks:
[REPLAY.md](REPLAY.md).

## 5. Runtime supervision

`RuntimeSupervisor` operator actions are `start`, `pause`, `safe_hold`, `resume`, `stop`; each is logged as
`OPERATOR_ACTION`. `resume` is refused while a critical module is unhealthy. A critical module failure moves the
runtime to SAFE_HOLD without a new command. There is no CLI for these yet: `conrad runtime start`
(`scripts/runtime.sh`) is not implemented at the time of writing (integrated mission in progress).

## 6. Backup and restore

`conrad.persistence.replay_store.backup(db_path, store, digests, dest)` and `restore(backup_dir, db_target,
store_root)`. A restore is valid only if every hash, the integrity check and the migration state pass.

## 7. Hardware

Not possible. The physical adapter is not supplied (EXT-HW-01), physical parameters are OPEN (EXT-HW-02,
EXT-HW-04), the frame contract (EXT-HW-03) and safety contract (EXT-HW-05) are OPEN, and target HIL is blocked
(EXT-HIL-01). The handoff procedure is in [PHYSICAL_INTEGRATION.md](PHYSICAL_INTEGRATION.md). Never set
`hardware_enable: true` in a shared config.

## 8. When something fails

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md). Record failed experiments too; the registry keeps failed hypotheses.
