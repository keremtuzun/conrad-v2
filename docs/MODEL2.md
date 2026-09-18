# Model 2: the belief world

Model 2 turns observations into evidence, and evidence into persistent beliefs with property-level knowledge status
and a four-channel uncertainty. It has a shared Core (`conrad/core`) and three domain children
(`conrad/domains`). None of it imports truth (`tests/leakage/test_static_boundaries.py`).

Per-package contracts and status: [IMPLEMENTATION_MAP.md](IMPLEMENTATION_MAP.md). Learned candidates:
[MODEL_CARDS/](MODEL_CARDS/README.md).

## Core (`conrad/core`)

| Mechanism | Runtime default (analytic) | Learned candidate | Files |
|---|---|---|---|
| ECMER (evidence encoding) | none; encoders are the candidate | `EcmerModel`, `EcmerEncoder.encode` | `core/ecmer/*` |
| Association | `nearest_neighbour_decision` | `AssociationScorer` (pair MLP + NO_MATCH head) | `association.py`, `association_scorer.py` |
| BUO (direct update) | `AnalyticBUO` (Kalman-style, contradiction and independence handling) | `BeliefUpdateOperator` | `buo_analytic.py`, `buo.py` |
| Uncertainty | analytic channels in `AnalyticBeliefState` | `UncertaintyHeads` | `uncertainty.py` |
| TBD (temporal) | `AnalyticTBD` | `GruTBD`, `MlpTBD` (+ `HoldLastTBD`, `LinearExtrapolationTBD`) | `tbd_analytic.py`, `tbd.py` |
| RBP (relational) | `AnalyticRBP` (INFERRED only, never overwrites OBSERVED) | `RelationalBeliefPropagation` | `rbp_analytic.py`, `rbp.py` |
| PMBL (persistence, lifecycle) | `PMBL` (no learned variant) | none | `pmbl.py`, `pmbl_core.py`, `pmbl_store.py`, `lifecycle.py` |

`Model2Core` (`conrad/core/pipeline.py`) wires the analytic operators. Learned modules are optional
`LatentModules` (`pipeline_latent.py`) that only update latent tensors on a `WorkingNode`; all default to `None`
(ADR-0004).

Configuration: `CoreConfig` in `conrad/core/config.py`, loaded from the `model2_core:` block of
`configs/train/core_default.yaml` (ch33 defaults: De = Dz = 256, Du = 64, Dt = 64, Dh = 128, Dr = 64, Dm = 128,
8 heads, dropout 0.10) or `configs/train/core_tiny.yaml` (test dimensions). Validators require
`evidence_dim == belief_dim`, `belief_dim % heads == 0` and `uncertainty_dim % 4 == 0`.

### Invariants

- Masks: `True` means valid. Padding gets zero attention weight and adds nothing to loss denominators
  (`conrad/core/primitives.py`: `masked_softmax`, `masked_loss`).
- Temporal prediction takes physical `delta_t_s` in seconds. Negative or non-finite values raise.
- Evidence is not belief: the analytic BUO works on `Evidence.measurements`, not embeddings.
- Duplicate evidence never counts twice; a repeated independence group may move the mean but not shrink the
  variance (Model2T) or is collapsed (`collapse_independence_groups`).
- Property variances persist as `<name>.variance` `PropertyClaim`s, because the contract has no variance field.

## Domain children (`conrad/domains`)

All implement `Model2Child` (`conrad/domains/base.py`): `initialize`, `ingest(Sequence[Evidence])`,
`update_beliefs(now)`, `predict(delta_t_s, now)` (PREDICTED only), `receive_context` (logged as
CROSS_DOMAIN_CONTEXT), `query`, `export_beliefs`, `reset_working_memory`, `availability`.

| Child | Class | Runtime path | Learned candidate | Config |
|---|---|---|---|---|
| Technical (2T) | `Model2T` (`technical/model.py`) | per-quantity [level, rate] Kalman filter + analytic TCDP (`PropagationMode` NONE / TCDP / GENERIC) | `LearnedTCDP` (`learned_tcdp.py`), not on the runtime path | `Model2TConfig` (code; no YAML) |
| Ecological (2E) | `Model2E` (`ecological/model2e.py`) | entity + field beliefs, `AnalyticCEFD` coupling | `LearnedCEFD` (`ecological/learned/`) | `model2e:` block in `configs/eval/2e_e00*.yaml` |
| Spatial (2S) | `Model2S` (`spatial/model.py`), UAHSM | sparse hierarchical log-odds grid, pose-covariance propagation, relational gap inference | none (learned heads OPEN_BLOCKED) | `model:` block in `configs/eval/2s_e00*.yaml` |

Domain-specific rules found in the code:

- 2T: TCDP writes only to components without direct lineage and bounds the message shift to `max_shift_sd`
  (added after an unbounded run spread a failed crack). Sensor health maps to reliability OK 0.9 / DEGRADED 0.5 /
  FAULT 0.05.
- 2E: entity and field beliefs are never merged. `ecological_damage` is always UNKNOWN (no analytic decoder).
  Thermal stress is INFERRED only and never moves the cover mean.
- 2S: UNKNOWN is not occupied. RGB does not update geometry. A NaN range carves no free space. Log-odds magnitude
  is capped so a large pose sigma can never produce a confident cell.

## How to run

```
uv run pytest tests/unit/core tests/property/core tests/contract/core -q
uv run pytest tests/unit/domains tests/property/domains tests/simulation/spatial -q
uv run conrad eval run --experiment CORE-BUO-E001
python -m conrad.evaluation.structural_experiments.run_all          # 2T-E001..E004
python -m conrad.evaluation.ecological_experiments.run_all          # 2E-E001..E003
python -m conrad.evaluation.spatial_experiments.run_all             # 2S-E001..E004
```

(Prefix with `python -m uv run` on this Windows host.)

## Results and limits

See [research/EXPERIMENT_RESULTS_2026-09-18.md](research/EXPERIMENT_RESULTS_2026-09-18.md). In short, on
synthetic data only: learned association, learned BUO and learned RBP lost to their baselines (KILL_CANDIDATE);
the GRU TBD beat its simple baselines; PMBL identity and lineage behaved correctly with no accuracy benefit; 2T
corrosion estimation beat latest-observation but crack estimation failed; the 2E static-field baseline won
turbidity and CEFD coupling is a KILL_CANDIDATE; UAHSM made no unsupported hidden-geometry claims but has weaker
calibration than a plain grid.

OPEN: final neural architectures for every mechanism (REQ-OPEN-004), bounded late-evidence rewind (ADR-0003),
real-data training (EXT-DATA-01) and research-scale training (EXT-COMPUTE-01).
