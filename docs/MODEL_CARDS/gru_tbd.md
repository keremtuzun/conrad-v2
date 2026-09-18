# GRU TBD

- **Code:** `conrad/core/tbd.py` (`GruTBD`; also `MlpTBD`)
- **Config:** `model2_core.tbd.*`; training job `configs/train/core_smoke.yaml` (`core_tbd_smoke`)

## Architecture

Stem MLP, `GRUCell(Dz, Dz)`, latent projection, drift head for uncertainty growth. Inputs: z, u, temporal state,
log1p of physical `delta_t_s`, context. Loss: MSE + 0.5 x Gaussian NLL.

## Training data

Abstract Core sandbox sequences, synthetic; train and validation from disjoint generator seeds.

## Executed

- CORE-TBD-E001 (`tbd_e001.py`).
- `conrad train run --config configs/train/core_smoke.yaml` (6 epochs, 102 steps, CPU).

## Results

- CORE-TBD-E001 RMSE: GRU 0.140, MLP 0.230, GRU fed a sequence index 0.298, hold-last 0.409, linear 0.685.
  Spearman between delta_t and predicted variance 0.78.
- Training smoke: best val RMSE 1.034 vs hold-last 0.758 (the smoke-scale model is worse than the baseline).
  Checkpoint reload reproduced the metric exactly.

## Known failure modes

At smoke scale it loses to hold-last. Full-scale training was refused (EXT-COMPUTE-01).

## Claim status

IMPLEMENTED. The only learned Core module that beat its simple baselines in the sandbox; not promoted by any ADR.
Runtime uses `AnalyticTBD`.
