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

## Universal V1.1 P4 10P Handoff

Universal V1.1 10P is not launched through the generic `conrad train run` path. Before Kerem or any operator starts
formal 10P work, run the fail-closed handoff gate:

```
uv run conrad train osfm-v11-10p-readiness --output artifacts/gates/V1.1/10P_KEREM_HANDOFF/readiness.json
```

The command never trains. It verifies the current branch/commit, 307M-class V1.1 architecture, 829-entry registry,
candidate configs, upstream checkpoint metadata and payload bytes, local dataset artifacts, gates, dependency files, launch protocols
and handoff documentation. It exits nonzero until every blocker is gone and prints `READY FOR KEREM`.

Large payloads are intentionally not stored in Git. A clean clone must be supplied with:

- `artifacts/runs/train-osfm_u1_sonar_research-1790575109765916596/checkpoints/osfm_u1_sonar_research_full.pt`, SHA-256 `b2dfc162c1d72e4363643b516b6566eb0f11ca4a874f9674fc5b06f1ddf85696`.
- `artifacts/data/public.subpipe/raw/SubPipeMini2.zip`, SHA-256 `a3068be28471786c726cd6100e0b1d92d1c17615a4dcfe7f5544ba758821188f`.

The machine-readable payload list is `artifacts/gates/V1.1/10P_KEREM_HANDOFF/required_payloads.json`. The remaining
open blocker is tracked in `docs/KEREM_P4_10P_OPEN_BLOCKERS.md`.

The original 8-L4 launch protocol remains documented, but the current approved handoff path is the 1-L4 fallback in
`configs/train/osfm/v11_p4_1l4_10p_fallback_launch_protocol.yaml`, approved by
`artifacts/gates/V1.1/10P_1L4/user_approved_fallback.json`. This is not equivalent to the 8-L4 protocol; the report
preserves the known 1-L4 benchmark limitations.

The current handoff note is [HANDOFF_KEREM_V11_10P.md](HANDOFF_KEREM_V11_10P.md). That document is subordinate to
the JSON readiness output; if they disagree, update the document or the gate before launch.

The guarded launch command is:

```
uv run conrad train osfm-v11-10p-launch --config configs/train/osfm/v11_p4_10p_829_semantic_candidate.yaml --readiness artifacts/gates/V1.1/10P_KEREM_HANDOFF/readiness.json
```

It refuses unless the readiness output is already `READY FOR KEREM`, then invokes the reviewed V1.1 10P trainer. The trainer is data-backed on verified SubPipe rendered side-scan sonar and must report non-SubPipe modalities as interface-covered/inactive rather than semantically mastered.

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
