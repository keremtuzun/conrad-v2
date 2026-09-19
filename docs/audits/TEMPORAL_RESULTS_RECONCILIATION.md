# Temporal results reconciliation

Two earlier temporal results looked contradictory:
- **CORE-TBD-E001:** the GRU beats hold-last (RMSE 0.140 vs 0.409).
- **Training smoke:** the GRU loses to hold-last (1.034 vs 0.758).

They are different studies.

| | CORE-TBD-E001 | Training smoke (`train-core_tbd_smoke-*`) | CORE-TBD-E002 (new, matched) |
|---|---|---|---|
| Purpose | mechanism comparison | pipeline smoke: trainer, checkpoint, reload | reconcile the two |
| Generator | `tbd_e001.make_sequences` (fixed damped-oscillator linear dynamics, D=4, obs sd 0.05) | same | same |
| Process noise q | 0.001 /s | 0.02 /s | 0.02 /s |
| Data | 80 train / 30 test sequences, length 24 | 48 train / 16 val, length 24 | 80 / 30, length 24 |
| Dims / epochs | 32 / 25 | tiny (16) / 6 | 32 / 25 |
| Seeds | 2026201-3 (development) | 2026201 | 2026201-3 (development) |
| Config digest | `configs/eval/core_tbd_e001.yaml` | `configs/train/core_smoke.yaml` | `configs/eval/core_tbd_e002.yaml` |
| Metric | RMSE of the prediction vs the true next state | same | same |
| GRU RMSE | 0.140 | 1.034 | **0.364 ± 0.023** |
| Hold-last RMSE | 0.409 | 0.758 | **0.751 ± 0.069** |
| MLP / linear | 0.230 / 0.685 | n/a | 0.424 / 1.319 |
| GRU NLL vs analytic NLL | n/a | n/a | 0.161 vs 0.692 |

## Conclusions

1. At a matched training budget, the GRU beats hold-last in both noise regimes. The smoke "loss" came from undertraining (6 epochs, tiny dimensions), not from a regime reversal. The smoke result is not evaluation evidence.
2. Both studies use an abstract generator with learnable fixed linear dynamics. Neither is a comparison in the **runtime regime**: structural degradation (Model2T), spatial map aging (Model2S) and environmental fields (Model2E) each use their own analytic dynamics. The only runtime-regime temporal comparison is 2T-E004. There the analytic rate prior beat hold-last by +0.045 mm, and a zero-rate ablation still got +0.039 mm, so most of that gain is smoothing.
3. **Primary runtime temporal model: analytic** (domain dynamics with variance growth over physical Δt). The learned GRU stays EXPERIMENTAL_CANDIDATE. It has not been trained or evaluated on any runtime domain, so it cannot be selected for the runtime. It becomes eligible after a matched comparison on 2T/2S/2E temporal data under the partition protocol (`conrad/evaluation/partitions.py`).
4. All CORE-TBD numbers come from development seeds. None is final-test evidence.
