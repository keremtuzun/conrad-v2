# Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `uv: command not found` on Windows | uv is not on PATH | Use `python -m uv ...` |
| doctor `WARN migrations ... runtime database not created yet` | Simulation lane before migration | `uv run conrad db migrate` |
| doctor `FAIL migrations ... does not exist` | HIL or physical lane without a database | `uv run conrad db migrate --config <same config>` |
| `database is not migrated` / `is behind head` / `newer than this code supports` | `check_migration_state` | Run `conrad db migrate`; a newer database needs newer code |
| doctor `FAIL python` | Not Python 3.11 | Use Python 3.11 |
| doctor `FAIL adapters unknown adapter 'unity_v2'` with `configs/runtime/hil_target.yaml` | The doctor's adapter table knows `sim_kernel`, `unity`, `physical` only | Known mismatch; target HIL is BLOCKED_EXTERNAL anyway (EXT-HIL-01) |
| doctor `FAIL adapters ... 'physical' is not installed (EXT-HW-01)` | No physical driver | Expected until EXT-HW-01 |
| doctor `FAIL robot_config ... not physically grounded` | Physical lane with OPEN or estimated parameters | Expected until EXT-HW-02 / EXT-HW-04 |
| doctor `FAIL command_mode hardware forbidden` | Missing hardware enable, HIL evidence or lane | Expected for `physical_example.yaml` (fails by design) |
| `ConfigError: local override may only contain ['device', 'paths']` | Other keys in `configs/local.yaml` | Move them to a real config |
| `binding a control endpoint to all interfaces is forbidden` | `bind_address` 0.0.0.0 or `::` | Use a loopback or private address |
| `conrad train run` prints `BLOCKED_EXTERNAL`, exit 3 | `scale: full` without CUDA | Use `core_smoke.yaml`; full needs EXT-COMPUTE-01 |
| `conrad train run` raises `ValueError` about the job | Config has no registered `job` (`core_default.yaml`, `core_tiny.yaml`) | Only `core_tbd_smoke` is registered |
| `unknown experiment '2T-E001'` from `conrad eval run` | Domain experiments are not in the dispatch registry | `python -m conrad.evaluation.structural_experiments.run_all` (see [EVALUATION.md](EVALUATION.md)) |
| `scripts/replay.sh`, `scripts/runtime.sh` fail with "No such command" | `replay`, `runtime` and `sim` groups have no commands yet | Integrated mission in progress |
| `scripts/eval.sh` fails on `configs/eval/default.yaml` | That file does not exist | Pass `--config` explicitly |
| `ReplayIntegrityError` | Bundle file or object missing or changed | Restore the original; never regenerate evidence |
| `StaleUpdateError` | DIRECT update older than the head, not flagged late (ADR-0003) | Mark it late or reject it |
| Gateway rejection with reason codes | See the table in [ARCHITECTURE.md](ARCHITECTURE.md) | Fix the named precondition |
| `RunSealedError` / read-only run files | Run directory is sealed | Start a new run; write notes under `notes/` |
| `conrad data verify` FAIL on a public manifest | Manifests are unverified templates | Expected until EXT-DATA-01 |
| Git-dependent code fails (dispatch, release) | `git` not available or not a repository | Run from a git checkout |

Legacy independence: `python scripts/legacy_independence_check.py --quick` runs the checks from a clean clone.
It calls `conrad sim run` and `conrad replay run`, which do not exist yet, so its full run cannot pass today.
