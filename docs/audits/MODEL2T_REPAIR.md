# Model2T repair for the realistic structural sensor

Date: 2026-09-19. Follows `docs/audits/STRUCTURAL_LINEAGE_AUDIT.md`. All data SYNTHETIC_ONLY (Twin2T MCDE
scenarios, T0 observations). Every sensor number below is an ENGINEERING_ESTIMATE, not a calibration.

## Diagnosis (software-owned defect)

Twin2T now defaults to a realistic T0 sensor: relative crack sizing (0.85 x in-view length, 20 % persistent
per-(component, sensor) bias, 25 % per-reading scatter), a length-dependent POD, sizing over the in-view part
only, and relative wall-loss error. Model2T still assumed a fixed 1.5 mm absolute crack sigma. Against ~30 %
relative error every large-crack reading was a "reliable contradiction": U_C saturated, the innovation-matched
inflation made the estimate chase the latest reading, and a single non-detection (a reading of ~0) was treated
as a precise measurement of a ~0 mm crack. On FLAGSHIP-I4 s2026202 this dragged the crack estimate to 0.8 mm
(error 79.2 mm, U_C 1.0).

## Fixes

### 1. Model2T measurement model (belief plane, no Twin2T import)

* `conrad/domains/technical/config.py`: new `SensorCharacteristics` (datasheet of the inspection sensor type as
  Model2T assumes it) and `DirectConfig.measurement_model` = `SENSOR_CHARACTERISED` (default) or
  `ABSOLUTE_GAUSSIAN` (pre-repair update, kept as an ablation arm). `PriorConfig.tail_weight/tail_scale_m` give
  the population prior a heavy tail (a few components carry real defects far outside the Gaussian core).
* `conrad/domains/technical/measurement.py` (new): per-reading Bayes update on a 1-D grid of the worst-case level,
  moment-matched back to the Gaussian [level, rate] state through an equivalent linear pseudo-measurement (so the
  rate still learns).
  * Relative sizing: log-normal around `median_factor * f * L` with relative scatter plus an absolute floor.
    All sigmas scale with `1 + noise_ua_gain * aleatoric` (reported sensor health).
  * Missed detection = censored evidence: a crack reading below `crack_call_threshold_m` has likelihood
    P(no call | L) = E_f[(1 - POD(fL)) * P(noise < call) + POD(fL) * P(sized below call)]. It falls slowly with
    L, so it lowers the odds of large cracks; it can never pin the length at ~0.
  * Partial visibility: the sensor never reports visibility. Every reading is a mixture over the in-view
    fraction f: P(full) = `full_view_prob * (0.5 + 0.5 * reliability)` (a degraded view is less likely to have
    covered everything), else f ~ U(f_min, 1). A low reading is a plausible lower bound, not a contradiction.
  * Measured geometry: `Evidence.spatial_support` (the measured surface point attached by association) is used
    for locality. Model2T remembers where the worst indication of a quantity was seen; a reading whose measured
    point lies more than `defect_locality_m` (0.3 m) + 2 sigma away views another region, so it only bounds the
    worst case from below (soft CDF likelihood) and a miss there is uninformative. Readings without a measured
    point are treated as whole-component views. Measured range is NOT used: the lineage test pins
    `QualityContext` to all-None, so the adapter copies nothing into it.
  * Contradiction: only when the belief puts < `contradiction_prior_mass` (1e-3) on the reading's plausible set
    (likelihood within 1 % of its maximum). Scatter within the relative noise, a partial view or a miss never
    qualifies; U_C rises only for credible (reliability >= 0.6) disagreement.
* Persistent bias: `structured_evidence` now writes `independence_group = "sensor:<sensor id>|<reading group>"`.
  Model2T counts distinct sensors per (component, quantity) and floors the level variance at
  `(bias * level)^2 / n_sensors` after each detection, so repeated same-sensor looks never average the bias away
  (a second sensor does help). DEV note: first tried an information-budget tempering of each reading; it froze
  estimates of cracks that later ran away (E001 capped-crack error 1.4 m), so the floor replaced it.
* Status: wall loss is OBSERVED from any valid reading. A crack is OBSERVED only after a detection; readings
  that never detected it update the latent estimate (direct lineage, so TCDP cannot overwrite it) and the claim
  stays UNKNOWN. A never-inspected component stays UNKNOWN.
* TCDP (`tcdp.py`, `engine.py`): messages compare posteriors against the moment-matched heavy-tailed prior on
  both edge ends, so the new prior does not create spurious shifts.

Declared characteristics vs the simulated sensor (documented, not read from Twin2T):

| Characteristic | Model2T declares | Twin2T REALISTIC | |
|---|---|---|---|
| crack median sizing | 0.90 | 0.85 | mismatch |
| crack per-reading scatter | 0.30 x (1+3 ua) | 0.25 x noise gain | mismatch |
| crack persistent bias | 0.25 | 0.20 | mismatch |
| crack POD a50 / log width | 8 mm / 0.6 (x (1+2 ua)) | 10 mm / 0.5 (x noise gain) | mismatch |
| crack call threshold | 2.5 mm | none (sensor always reports a number) | Model2T-only |
| crack absolute floor | 1.0 mm | 1.0 mm | match (generic floor) |
| wall median / scatter / bias / floor | 1.0 / 0.12 / 0.06 / 0.3 mm | 1.0 / 0.10 / 0.05 / 0.2 mm | median matches, rest mismatch |
| partial-view model | f=1 w.p. 0.8(0.5+0.5 rel), else U(0.2,1) crack, U(0.5,1) wall | in-view = visibility^0.5 (crack only) | different form |

DEV tuning (seeds 5100000-5100009, then 5100000-5100029): `full_view_prob` 0.6 -> 0.8, crack prior tail
0.1/20 mm -> 0.05/10 mm, `noise_ua_gain` 2 -> 3. A level-proportional crack process noise (1.0, 2.0 per sqrt yr)
was tried and rejected (near-static cracks drifted up more than run-away cracks gained); it stays in config at 0.

### 2. Near-side readings (truth plane)

* `structure.py`: Twin2T state is per component but the defect is local, so the target's non-defect surface is a
  truth-side Twin2T "surface region" component (only in the Twin2T copy of the scenario, sampled from the
  ordinary priors with its own RNG, `metadata.surface_region_of = target`). It never reaches the registry, the
  mapping or any Observation.
* `world.py` / `sensing.py`: a second `SurfaceTarget` covers the target's surface outside the patch and reads that
  region's state (`SurfaceTarget.twin_id`). A near-side view therefore reports "no crack detected here" with
  ordinary wall loss; the truth label still names the target, so association checks are unchanged.
* **Shipped OFF (`world.REGION_READINGS = False`).** With it on, lane views make the target's corrosion OBSERVED
  and its condition INTACT before any inspection. EGDC then raises no information need and MCBR plans nothing.
  Four tests that encode the "hidden target" premise fail:
  `test_i3_structural.py::test_hidden_target_stays_unknown_from_the_lane` and, in
  `test_flagship_mission.py`, `test_before_the_segment_condition_is_unknown_with_high_uo`,
  `test_smoke_mcbr_produces_a_plan_with_candidate_table_and_rejections` and
  `test_goal_then_commands_only_through_the_gateway`. Those tests are outside this workstream's paths and were not
  touched. The root cause is on the belief side: without coverage geometry (component extent in the registry,
  or view directions), Model2T cannot tell "near side seen" from "whole component seen". Treating near-side views
  as covering the component would be the overconfidence the audit warned about. Decision needed: add coverage
  to the registry/Model2T, then turn the switch on and restate those tests' premise.
* Leakage: all of `tests/leakage` pass with the switch on and off. With the switch on, the only failures in the
  requested suites were the four above. `test_static_boundaries` failed once, on a UTF-8 BOM in another agent's
  `conrad/sim/mission/unity_run.py`; that file has since been fixed and the test passes.

### 3. Experiments (R2, realistic sensor)

Registered in `conrad/evaluation/dispatch.py`: `2T-E00{1..4}-R2-DEV` (DEVELOPMENT, design) and `2T-E00{1..4}-R2`
(FINAL_TEST, reported once). Seeds come from `configs/eval/partitions.yaml`, mission domain (these are Twin2T
worlds, not mission worlds; the partition file only has abstract/mission domains). The runner checks every seed
against its declared partition and runs inside `purpose_scope`. DEV = 5100000-5100029 (E002/E004: first 10).
FINAL = 5300000-5300059 (E002: first 20, because B2 is refit per seed). The GRU baseline in E001-R2 is fitted
once on DEV seed 5100039. Old results in `artifacts/experiments/structural/` are untouched. Paired unit = seed.
CIs = percentile bootstrap (10 000). Gate rules and the coverage band [0.90, 0.99] were fixed in the configs
before the FINAL run.

## DEV results (design only, not evidence)

2T-E001-R2-DEV, 30 seeds, MAE gain = latest-observation MAE minus Model2T MAE (mm), 95 % CI:

| level | corrosion gain | corrosion cov95 | crack gain | crack cov95 |
|---|---|---|---|---|
| 0.0 | +0.065 [0.063, 0.067] | 0.996 | +0.60 [-0.94, 2.19] | 0.844 |
| 0.4 | +0.355 [0.346, 0.364] | 0.952 | +8.33 [1.42, 15.24] | 0.734 |

2T-E003-R2-DEV: corrosion RB_TCDP -0.114 [-0.121, -0.107], crack RB_TCDP +0.002 [-0.028, 0.034]. RC corrosion
TCDP 0.28 vs GENERIC 0.66. 2T-E004-R2-DEV: corrosion TB +0.034 [0.027, 0.041], crack TB -180 [-219, -144] mm.

2T-E002-R2-DEV was registered but not run: the run was cancelled to save CPU, since B2 is refit per seed at about 70 s.

FLAGSHIP-I4 s2026202 (DEV seed), final target crack error:

| Model2T / sensor | crack error | corrosion error |
|---|---|---|
| pre-repair, realistic sensor (audit) | 79.2 mm (dragged to 0.8 mm by one miss) | 0.13 mm |
| repaired, region readings OFF (shipped default) | **4.7 mm** (estimate 75.3 mm vs 80 mm) | 0.39 mm |
| repaired, region readings ON, before locality handling | 41.2 mm (near-side misses pulled it down) | 4.49 mm |
| repaired, region readings ON, with locality | 1.9 mm | 0.48 mm |

## FINAL results (final_test, run once)

**2T-E001-R2** (60 seeds; each seed 3 tier-3 episodes x 16 steps x 2 levels):

| level | quantity | Model2T MAE | latest MAE | gain [95 % CI] | cov95 | in band |
|---|---|---|---|---|---|---|
| 0.0 | corrosion | 0.106 mm | 0.172 mm | +0.066 [0.064, 0.068] | 0.995 | no (over-covers) |
| 0.0 | crack | 37.10 mm | 35.48 mm | **-1.63 [-3.13, -0.08]** | 0.825 | no |
| 0.4 | corrosion | 0.241 mm | 0.601 mm | +0.360 [0.353, 0.367] | 0.945 | yes |
| 0.4 | crack | 87.66 mm | 103.66 mm | +16.0 [10.3, 21.5] | 0.716 | no |

Against the pre-repair update (`MODEL2T_ABSOLUTE`): corrosion is slightly worse (-0.022 mm at both levels),
crack is -1.45 [-2.98, 0.13] at level 0 and +15.5 [9.9, 21.1] at level 0.4. Against the GRU the repaired model
wins on crack at both levels (+98 and +48 mm) and on corrosion at level 0, but loses on corrosion at level 0.4
(-0.013 [-0.019, -0.007]). Crack error is dominated by run-away cracks capped at 1 m, where the constant-rate
prediction and the partial-view lower-bound logic both undershoot.

**2T-E003-R2** (60 seeds): corrosion RB_TCDP **-0.104 [-0.110, -0.097] mm** (TCDP is worse than no propagation;
negative in every episode kind). Crack RB_TCDP -0.005 [-0.022, 0.013]. The only positive kind is coupled crack,
+0.032 [0.004, 0.062]. Contamination for corrosion: TCDP 0.286 vs GENERIC 0.662, difference +0.379
[0.353, 0.404]. Crack contamination is 0 for every arm (empty signal). The pre-declared rule required a positive
CI for every quantity with an RC pool, so crack's zero-vs-zero counts as "not lower". That rule is too strict,
but it was fixed before the FINAL run and does not change the outcome, because there is no benefit anyway.

**2T-E004-R2** (60 seeds): corrosion TB between inspections +0.036 [0.033, 0.039] mm, before the next inspection
+0.030 [0.026, 0.033]. Crack TB between -163 [-173, -153] mm, before the next inspection -202 [-213, -191] mm: the
engineering-prior crack rate cannot follow run-away growth over 6-12 month gaps.

**2T-E002-R2** (first 20 FINAL seeds; full Model2T path with TCDP; error on components hidden now but seen
before; gain = baseline MAE minus Model2T MAE, mm):

| coverage | corrosion vs latest | corrosion vs GRU | crack vs latest | crack vs GRU |
|---|---|---|---|---|
| 0.5 | +0.060 [0.052, 0.068] | +0.149 [0.141, 0.157] | -144 [-171, -118] | -47 [-73, -24] |
| 0.2 | +0.042 [0.032, 0.052] | +0.122 [0.111, 0.132] | -208 [-246, -173] | -119 [-159, -80] |
| 0.05 | +0.032 [0.017, 0.047] | +0.098 [0.080, 0.116] | -433 [-532, -344] | -345 [-448, -251] |

INDEPENDENT_COMPONENT UNKNOWN rate on never-observed components: 1.0 on every seed. Crack persistence loses for
the same reason as E004: between looks, run-away cracks outgrow the predicted state.

## Gates (`scripts/record_gate_evidence.py 2T 2T-TCDP`, FORMAL evidence in `artifacts/gates/`)

* **2T: FAIL.** Corrosion passes the MAE rule at both levels. Crack loses to latest-observation at level 0 (CI
  below 0). 95 % coverage is outside [0.90, 0.99] for crack at both levels (0.83, 0.72) and for corrosion at
  level 0 (0.995, over-covered).
* **2T-TCDP: FAIL.** There is no benefit versus no propagation. TCDP makes corrosion worse, with the CI entirely
  below 0. Contamination for corrosion is below GENERIC. The OPEN "excessive contamination" bound was never
  reached, because the benefit condition already fails.

## Open items

* Coverage geometry for Model2T (needed before region readings can be on by default; see section 2).
* Crack calibration and run-away cracks: constant-rate dynamics plus Gaussian moment matching under-cover long
  cracks. A level-dependent growth model (Paris-type, learned rate) is the next candidate. Test it on DEV and
  VALIDATION first; FINAL is now spent for this configuration.
* The E003 contamination rule should treat identical zero contamination as "not higher". Change it only with a
  fresh held-out split.
* Calibration coupling (audit row 12) still applies to the population prior.
