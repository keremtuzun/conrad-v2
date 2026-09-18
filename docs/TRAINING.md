# Training

`conrad/training` holds the trainer, curriculum runner, checkpoints and immutable run directories. Only CPU-scale
training has been executed.

## Command

```
uv run conrad train run --config configs/train/core_smoke.yaml
```

`run_training` (`conrad/training/entrypoints.py`) loads the YAML, looks up its `job` in `JOBS`, and runs it. The
only registered job is `core_tbd_smoke` (the GRU TBD). An unknown job raises `ValueError`, so
`core_default.yaml` and `core_tiny.yaml`, which are model configs without a `job` key, cannot be run this way.

| Config | Role |
|---|---|
| `configs/train/core_smoke.yaml` | Job `core_tbd_smoke`, scale smoke, tiny dims, 48 train / 16 val sequences of length 24, 6 epochs |
| `configs/train/core_full.yaml` | Same job, `scale: full`, full dims, 20000 / 2000 sequences, 200 epochs. Refused without CUDA. |
| `configs/train/core_default.yaml` | `model2_core:` block with ch33 defaults |
| `configs/train/core_tiny.yaml` | `model2_core:` block with test dims |
| `configs/train/base_trainer.yaml` | Trainer defaults (AdamW, lr 3e-4 new / 1e-4 pretrained, 5 % warm-up, batch 16, effective 64, patience 25, `val_loss`) |
| `configs/train/base_curriculum.yaml` | Curriculum C0..C9; C9 needs real data |

Compute gating: when `scale == "full"` and `inspect_compute()` finds no CUDA device, `ComputeBlockedError` is
raised and the CLI prints `BLOCKED_EXTERNAL: ...` and exits with code 3 (EXT-COMPUTE-01).

## Outputs

- Run directory `artifacts/runs/train-<job>-<time_ns>/` (`run_dir.py`): `config.resolved.yaml`,
  `environment.json`, `git.json`, `manifests.json`, `metrics.jsonl`, `events.jsonl`, `checkpoints/`, `reports/`
  (`training_result.json`), `figures/`, `replay/`, `logs/`, `notes/`, and `dirty_diff.patch` on a dirty tree.
- `seal()` hashes every file except `notes/`, writes `run_state.json` and makes files read-only. `verify_seal()`
  reports changes. A dirty tree makes the run NONCOMPARABLE; BENCHMARK, ACCEPTANCE and PHYSICAL runs require a
  clean tree (`DirtyGitError`).
- Checkpoints (`checkpoint.py`, `checkpoint_meta.py`): `<name>.pt` loaded with `weights_only=True` plus
  `<name>.pt.meta.json` (format `conrad-checkpoint-v1`) holding identity, config digest, manifest digests, split
  hash, git state, seeds, metrics and a validation selection metric. `load_checkpoint` refuses incompatible
  checkpoints with reason codes; `verify_checkpoint_metadata` is the doctor check.
- The smoke job reloads its best checkpoint and reports `reload_matches`.

## Other pieces

- `determinism.py`: `seed_everything` (random, numpy, torch, deterministic algorithms), `resolve_device` (CPU
  fallback is reported), `inspect_compute`.
- `curriculum.py`: stage-wise freezing by parameter-name pattern; a pattern that matches nothing is an error.
- `trainer.py`: cosine schedule, gradient clipping, accumulation, early stopping, validation-only selection, and a
  final-only test/OOD guard.
- `truncated.py`: truncated BPTT over windows.
- `mlflow_logger.py`: optional file-local MLflow (`artifacts/mlruns`), disabled by default.
- Other learned candidates train only inside experiment modules or their own training functions (see
  [MODEL_CARDS/](MODEL_CARDS/README.md)).

## Result so far

From [research/EXPERIMENT_RESULTS_2026-09-18.md](research/EXPERIMENT_RESULTS_2026-09-18.md): the smoke run (6
epochs, 102 steps, CPU) reached best val RMSE 1.034 against hold-last 0.758, so the smoke-scale model is worse
than the trivial baseline. Checkpoint reload reproduced the metric exactly.

## OPEN and blocked

- Research-scale training: BLOCKED_EXTERNAL (EXT-COMPUTE-01).
- Real-domain adaptation (curriculum C9) and real-data training: BLOCKED_EXTERNAL (EXT-DATA-01).
- Tests: `uv run pytest tests/unit/training -q`.
