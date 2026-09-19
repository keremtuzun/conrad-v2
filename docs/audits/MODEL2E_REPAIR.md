# Model2E field model repair (2E-E001 defect) and 2E / 2E-CEFD gate evidence

Date: 2026-09-19. Data: SYNTHETIC_ONLY (Twin2E small world, `configs/sim/twin2e_test_small.yaml`).
Model version: `model2e-uncoupled-analytic-0.3.0`. Production switches unchanged (uncoupled, ADR-0007).

## 1. Defect

`artifacts/experiments/ecological/2E-E001.json` (seeds 2026201-2026203, kept unchanged): turbidity RMSE on hidden
cells was 0.191 / 0.173 / 0.186 / 0.208 NTU for 1 / 2 / 4 / 8 sensors, while a static uniform field scored
0.05-0.08. Every arm was over-cautious (turbidity mean z^2 0.003-0.025).

## 2. Diagnosis

The v0.2 field model was `level + residual`, each an OU process whose stationary variance was the configured
prior (`prior_sd`, `local_sd`), updated reading-by-reading with a rank-1 kernel step.

1. **Over-dispersed local-deviation prior.** `local_sd` was 4 NTU (no stated source). The field's real local
   deviations are about 0.06 NTU. With OU relaxation over 6 h, the local process noise was about 1.3 NTU^2
   per 15 min step, against 0.04 NTU^2 sensor noise, so the Kalman gain stayed near 1. Each residual tracked
   the latest noisy reading, and the kernel spread that noise into the hidden cells. **More sensors meant more
   noise leaked into those cells.** This is the defect. The level had the same problem, with `prior_sd`
   acting as its stationary variance.
2. **No hierarchical pooling.** Readings from the same mooring were treated as independent evidence and never
   averaged. The split between level and deviation depended on which sensor reported last.
3. **Kernel scale and structure mismatched for temperature.** The dominant structure is a vertical gradient
   that a 6 m SE kernel cannot extrapolate.
4. **Sensor noise not learned.** The datasheet noise (0.2 NTU, 0.05 degC) overstates the simulator's noise.

## 3. Fix: hierarchical empirical-Bayes field model (`conrad/domains/ecological/field_belief.py`)

- **Stations.** Readings are pooled by sensor location (within `station_merge_radius_m`). Each station runs a
  local-level filter: a random walk with drift rate q, read through noise rho * r.
- **Temporal EB.** Both q and rho come from a temporal variogram at each station. The slope (lag-2 squared
  differences minus lag-1) gives q without depending on the noise. The lag-1 intercept gives rho. q is shrunk
  toward the prior 2 local_sd^2 / T (weight: one correlation time). rho is shrunk toward 1 (20 pseudo-pairs).
- **Hierarchical kriging.** Station values are combined as
  m ~ N(H b0, H P_b H^T + tau^2 K + diag(v)), where b is [domain level, optional linear depth trend].
  The level is estimated from all stations. Deviations are shrunk toward it according to their evidence.
  Cell mean and variance are the exact Gaussian posterior, and `resid_var` absorbs the level/deviation
  covariance.
- **Spatial EB.** tau^2 and the (horizontal, vertical) length-scale multipliers are the joint MAP of the station
  marginal likelihood. The priors are weak log-normals: 1 decade for tau^2, 1 octave for the length scales.
- **Per-field priors, with stated sources.**
  - turbidity `local_sd` 4.0 -> 0.063 NTU
  - temperature `local_sd` 2.0 -> 0.11 degC

  Both are the pooled EB tau^2 estimates at k = 8 on the DEV partition, computed from sensor readings only.
  temperature `depth_trend_sd` = 0.1 degC/m (ENGINEERING_ESTIMATE for stratified shelf seas).
  current and light keep their old, unsourced defaults because no 2E experiment exercises them.
- **Baselines unchanged in meaning.** `static_field` is a single uniform level with q = 0 and no variance
  growth. A new `production` baseline uses `CefdSwitches()`.

New config block: `Model2EConfig.field_model` (`FieldModelConfig`). New tests:
`tests/unit/domains/ecological/test_eco_field_eb.py`. Existing tests were adjusted for two reasons: the far-cell
`resid_var` now carries the posterior covariance (tolerance 2 %), and the variance-growth test uses turbidity
because temperature now has a depth trend. The property-test floor now matches the model's floors.

## 4. Partitions

`configs/eval/partitions.yaml` covers only MCBR and mission seeds, so the 2E configs define local, disjoint
partitions in `configs/eval/2e_e00{1,2,3}_r2.yaml`:

| partition | seeds |
|---|---|
| development | 2026201-2026203, 6100000-6100004 (8) |
| validation | 6200000-6200007 (8) |
| final | 6300000-6300011 (12) |

The design was iterated on DEV only. VALIDATION selected between two prior settings, on 2E-E001 production:

- **A (chosen):** DEV-EB priors. Turbidity RMSE 0.058 / 0.040 / 0.035 / 0.033.
- **B:** the original unsourced `local_sd`. Turbidity RMSE 0.37 / 0.13 / 0.12 / 0.15, significantly worse than
  static at every k.

FINAL was run once.

## 5. Before and after

### DEV partition (8 seeds, 2E-E001, hidden cells). Design data, not a result.

| arm | field | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|---|
| v0.2 cefd/field_only | turbidity RMSE | 0.337 | 0.237 | 0.196 | 0.189 |
| v0.2 static_field | turbidity RMSE | 0.129 | 0.068 | 0.045 | 0.039 |
| v0.3 production | turbidity RMSE | 0.084 | 0.055 | 0.042 | 0.037 |
| v0.2 cefd/field_only | temperature RMSE | 0.536 | 0.353 | 0.258 | 0.173 |
| v0.3 production | temperature RMSE | 0.507 | 0.161 | 0.072 | 0.045 |

v0.2 mean z^2 was 0.006-0.025 for turbidity and 0.09-0.18 for temperature. v0.3 is 0.39-0.69 and 0.42-0.45.

### FINAL partition (12 seeds), `artifacts/experiments/ecological/2E-E001-R2.json`

| production | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| turbidity RMSE (NTU) | 0.081 | 0.063 | 0.043 | 0.034 |
| static_field turbidity RMSE | 0.070 | 0.058 | 0.050 | 0.044 |
| turbidity coverage95 / mean z^2 | 0.995 / 0.40 | 0.983 / 0.61 | 0.965 / 0.73 | 0.985 / 0.60 |
| temperature RMSE (degC) | 0.587 | 0.155 | 0.068 | 0.045 |
| static_field temperature RMSE | 0.610 | 0.551 | 0.484 | 0.433 |
| temperature coverage95 / mean z^2 | 1.000 / 0.48 | 1.000 / 0.32 | 0.983 / 0.70 | 0.996 / 0.47 |

Paired per-seed 95 % CIs for the RMSE differences:

- **Turbidity, more sensors:** every step is significantly negative. k8 - k4 = -0.0095 [-0.017, -0.002].
- **Turbidity vs static:** k4 = -0.007 [-0.022, 0.008]; k8 = -0.010 [-0.020, 0.001]. The point estimate is
  better, but the difference is not significant.
- **Temperature, more sensors:** every step is negative. k4 - k2 has CI [-0.180, 0.007].
- **Temperature vs static:** about -0.4 degC at k >= 2, significant.
- `field_only` and `cefd` give the same field numbers as `production`. The CEFD filtration sink is negligible.

Acceptance, as asked:

- RMSE non-increasing in sensor count: **met** for both fields.
- Not worse than static at >= 4 sensors: **met** (turbidity point estimates are better, not significantly).
- Calibration: **partly met.** 95 % coverage is 0.965-1.0, but mean z^2 is 0.32-0.73, so the model is still
  over-cautious. Temperature at k = 2 (0.32) falls below the pre-stated 1/3 bound.

**Regression found on FINAL (not fixed, since FINAL must not be tuned on).** In the 2E-E003-R2 worlds, a
15 NTU spike appears abruptly at 6 h. There, production turbidity RMSE is 0.277 (v0.2: 0.271) but mean z^2 is
**35** (coverage 0.87), against v0.2's 0.026. The DEV-derived drift and local priors assume the E001 world's
small deviations, so a step change in that world makes the model over-confident. Follow-up: a
change-point/robust drift model, or DEV worlds that include large step events. This is OPEN.

### Other R2 results (FINAL)

- **2E-E002-R2, production cover** (RMSE / coverage95 / z^2):

  | turbidity | RMSE | coverage95 | z^2 |
  |---|---|---|---|
  | 1 NTU | 0.048 | 1.0 | 0.19 |
  | 4 NTU | 0.071 | 0.998 | 0.31 |
  | 10 NTU | 0.130 | 0.984 | 0.71 |
  | 25 NTU | 0.279 | 0.849 | 2.10 |

  At 25 NTU the turbidity-blind survey noise makes the production arm over-confident.
  `cefd` at 25 NTU: 0.188 / 1.0 / 0.39.
- **2E-E003-R2:** see section 6.

## 6. Gate outcomes (`artifacts/gates/2E*/evidence_formal.json`)

Thresholds were written into `scripts/record_gate_evidence.py` before the FINAL run.

**2E (functional): FAIL**

- **entity model works: FAIL.** The tests pass. The E002-R2 check requires coverage95 >= 0.85 and z^2 <= 2 at
  every turbidity level, plus RMSE <= 0.10 at <= 4 NTU. At 25 NTU, production has coverage 0.849 and
  z^2 2.10: the entity model is over-confident in very turbid water when uncoupled.
- **field model works: FAIL.** The tests pass. RMSE is non-increasing and not worse than static for both
  fields, and turbidity is calibrated. Temperature mean z^2 at k = 2 is 0.322, below the 1/3 bound.
  This is a marginal miss.
- **persistent inference works: PASS.** Covered by the reset, persistence/query, own-timestamp and
  unavailable-on-exception tests.

**2E-CEFD (research): FAIL**

- **Benefit.** Pooled paired CB (uncoupled RMSE - cefd RMSE, 48 seed-world pairs) is +0.0009
  [-0.0005, +0.0023]. The CI is not above 0. Same result against `production`.
  Per world:
  - confounded_disturbed: +0.0077 [0.0059, 0.0095], a real benefit
  - confounded_stable: -0.0021 [-0.0037, -0.0005], a real harm
- **Unsupported claims.** `cefd` made confident stress claims on healthy entities in 16 of 48 worlds
  (hot counterfactual 0.5, confounded_stable 0.67). Uncoupled and production made none. UEI = 0 for all arms.

CEFD stays off in production, consistent with ADR-0007.

## 7. Reproduce

```
python -m uv run python -m conrad.evaluation.ecological_experiments.run_all artifacts/experiments/ecological 2e_e001_r2
# likewise 2e_e002_r2, 2e_e003_r2  (do NOT run run_all without a name filter: it rewrites the old 2E-E00x.json)
python -m uv run python scripts/record_gate_evidence.py 2E 2E-CEFD
```

Wall time on this CPU with the three runs in parallel: E001-R2 470 s, E002-R2 929 s, E003-R2 1059 s.

## Iteration 2 (2026-09-19): observability context split out, change detection, fresh FINAL-2

Model version: `model2e-uncoupled-obsctx-analytic-0.4.0`. R2 artifacts are kept unchanged. The gate
thresholds in `scripts/record_gate_evidence.py` are **unchanged**: they were set before any FINAL data, and
there was no spec reason to change them.

### I2.1 Changes

1. **Switch split** (`CefdSwitches`; ADR-0007 addendum). `field_to_entity` is removed and replaced by two
   switches:
   - `observability_context`, **ON** in production. The turbidity belief sets the survey measurement variance,
     so it reaches UA/UO. It adds no process noise, makes no stress claim and has no term that shifts cover.
   - `ecological_coupling`, **OFF**. This is the stress likelihood and the stress-gated cover process noise.

   `entity_to_field` stays OFF. The `uncoupled` baseline is now fully turbidity-blind (the old production
   arm). `production` = `uncoupled` + observability context. `tests/contract/test_runtime_defaults.py`
   asserts `ecological_coupling`/`entity_to_field` False, `observability_context` True, and the exact switch
   set.
2. **Abrupt-change fix** (`field_belief.py`; `FieldModelConfig.change_detection`, `change_z`,
   `volatility_memory_s`). Priors are not widened. There are three parts, all driven by standardized
   station innovations:
   - **Change-point intervention.** When e^2 > change_z^2 S (3 sigma), the station state gets the
     unexplained variance e^2 - S before its update (West & Harrison intervention, method of moments).
     The station re-acquires the new level in one reading instead of averaging the jump away.
   - **Stale stations.** A change point at one station adds its intervention variance to every other
     station that has not reported since.
   - **Innovation-based process-noise inflation.** A per-station drift qa comes from covariance matching
     with 3 h exponential forgetting. It is unbiased at 0 under the model, and only the running estimate is
     clipped. It rises during a sustained, sub-threshold settling tail.

   Change points are kept out of the temporal variogram, so a jump no longer inflates q or the noise scale.
   The mechanism is disabled in the static baseline. New tests:
   - `test_abrupt_change_is_detected_and_not_over_confident` (reproduces z^2 > 20 without the mechanism)
   - `test_change_detection_is_quiet_on_a_stationary_field`
   - `test_static_baseline_never_detects_changes`
   - `test_observability_context_sets_survey_noise_but_never_couples_ecology`

### I2.2 Partitions (`configs/eval/2e_e00{1,2,3}_r3.yaml`)

`configs/eval/partitions.yaml` is pinned by digest and covers only MCBR/mission, so the local lists are kept.

| partition | seeds |
|---|---|
| development | 2026201-2026203, 6100000-6100004 (8) |
| validation | 6200000-6200007 + the spent R2 FINAL 6300000-6300011 (20) |
| final_2 | 6400000-6400019 (20), fresh, run once after freezing |

### I2.3 DEV (design data, not a result)

- **E003 base world, production turbidity mean z^2** (8 seeds):
  - R2 mechanism: 69 (seed 2026201: 138)
  - change points only: 1.08
  - change points + qa, 3 h memory: 0.79
  - change points + qa, 6 h memory: 0.80
- **E002, production at 25 NTU:** RMSE 0.188, coverage 1.0, z^2 0.39. The uncoupled arm gives 0.229, 0.909
  and 1.31.
- **E001 field check:** passes with and without change detection.

### I2.4 VALIDATION (selection)

The rule was fixed before the runs: keep the arms that pass the E001 field check, then pick the one whose
E003 production turbidity mean z^2 is closest to 1 in log terms.

| arm | E001 field check | E003 turbidity RMSE / coverage95 / mean z^2 |
|---|---|---|
| B: R2 behaviour (no change detection) | pass | 0.210 / 0.876 / **22.9** |
| C: change points only | pass | 0.125 / 0.918 / 1.50 |
| **A: change points + qa (3 h), chosen** | pass | 0.123 / 0.949 / 0.985 |

E002-R3 production, A: coverage >= 0.998 and z^2 0.35-0.41 at every turbidity level. At 25 NTU the RMSE is
0.193. The entity check passes. Validation temperature z^2 at k = 2 was 0.375, just above the 1/3 bound.
The design was frozen at A (the config defaults).

### I2.5 FINAL-2 (20 fresh seeds, run once)

Artifacts: `artifacts/experiments/ecological/2E-E00{1,2,3}-R3.json`.

**2E-E002-R3, production cover** (RMSE / coverage95 / z^2):

| turbidity | RMSE | coverage95 | z^2 |
|---|---|---|---|
| 1 NTU | 0.049 | 1.0 | 0.31 |
| 4 NTU | 0.069 | 0.999 | 0.33 |
| 10 NTU | 0.118 | 0.999 | 0.41 |
| 25 NTU | 0.183 | 1.0 | 0.37 |

At 25 NTU the turbidity-blind `uncoupled` arm gives 0.242 / 0.905 / 1.42. The observability context fixes
the R2 over-confidence. It is also somewhat over-cautious (z^2 about 0.3-0.4). The entity check has no lower
z^2 bound.

**2E-E003-R3, turbidity after the 15 NTU spike** (production, all worlds): RMSE 0.141, coverage 0.954,
mean z^2 **1.22**. R2 FINAL was 0.277 / 0.865 / 35.

**2E-E001-R3, production:**
- **Turbidity.** RMSE is 0.074 / 0.060 / 0.050 / 0.043 at 1 / 2 / 4 / 8 sensors. Every step down is
  significant. Against static, k4 is -0.002 [-0.013, 0.009] and k8 is -0.007 [-0.017, 0.004]: better, not
  significant. Coverage / z^2 are 1.0 / 0.37, 0.99 / 0.48, 0.97 / 0.67 and 0.95 / 0.87.
- **Temperature.** RMSE is 0.455 / 0.099 / 0.057 / 0.043, with every step significant. It is better than
  static at k >= 2 by about 0.3 degC, which is significant. z^2 is **0.323** / 0.409 / 0.532 / 0.403.
- **k = 1 is the problem.** The temperature z^2 at k = 1 is 0.323, just below the pre-stated 1/3 bound. This
  is the same marginal over-caution as R2, now at k = 1 instead of k = 2. Iteration 2 did not target it
  (change detection barely touches temperature: DEV/VAL z^2 are the same with or without it). It is left
  OPEN, with no tuning on FINAL-2.

### I2.6 Gate outcomes (re-recorded, `artifacts/gates/2E*/evidence_formal.json`)

**2E (functional): FAIL**
- **entity model works: PASS.** This was FAIL in R2.
- **field model works: FAIL**, only on temperature calibration at k = 1: z^2 0.323 < 1/3. Every other field
  criterion passes, and turbidity is calibrated.
- **persistent inference works: PASS.**

**2E-CEFD (research): FAIL**
- **Benefit.** Pooled CB over `uncoupled` is -0.0006 [-0.0015, +0.0003] (80 pairs). Over `production` it is
  -0.0004 [-0.0006, -0.0003]. Against production, the full ecological coupling is significantly *worse*.
- **Unsupported claims.** `cefd` made confident stress claims on healthy entities in 28 of 80 worlds (max
  1.0). `production` and `uncoupled` made none. UEI = 0 for all arms. Ecological coupling stays OFF.

### I2.7 Open items

- Temperature over-caution at small k. The mean z^2 is 0.32-0.45 across all partitions, and the k = 1 or
  k = 2 point sits on the 1/3 bound. It is the only reason the functional 2E gate fails. A principled next
  step is an EB check on the temperature `depth_trend_sd` and on the level/drift priors, designed on DEV only.
- `volatility_memory_s` (3 h) and `change_z` (3) are ENGINEERING_ESTIMATE, chosen on DEV/VAL.

### I2.8 Reproduce

```
python -m uv run python -m conrad.evaluation.ecological_experiments.run_all artifacts/experiments/ecological 2e_e001_r3
# likewise 2e_e002_r3, 2e_e003_r3 (always with a name filter)
python -m uv run python scripts/record_gate_evidence.py 2E 2E-CEFD
```

Wall time with the three runs in parallel: E001-R3 527 s, E002-R3 743 s, E003-R3 951 s.
