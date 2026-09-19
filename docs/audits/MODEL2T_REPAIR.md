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

## Iteration 3 (2026-09-19)

Scope: production default without propagation, crack temporal model and calibration, surface coverage with
region readings ON, and a fresh FINAL-3 split. All data SYNTHETIC_ONLY. All sensor and growth numbers are
Model2T's own ENGINEERING_ESTIMATEs.

### 1. Cause of INFERRED on a never-seen component (Unity I3)

The mission runtime built Model2T with `model2t_mode = "TCDP"` (the default in `mission_config.py`). After a
neighbour was read, `StructuralBeliefEngine.propagate` wrote an INFERRED corrosion estimate into the unseen
component, and its derived `condition` claim took that status. That is where U_O 1.0 with INFERRED came from.
No other Model2T operator sets INFERRED: the population prior and temporal prediction leave a never-read
quantity UNKNOWN (in `conrad/domains/technical`, only `engine.propagate` and `tcdp.py` write INFERRED).

Fix (ADR-0009): `TCDPConfig.mode` now defaults to `NONE`. The mission runtime (`conrad/orchestration/children.py`)
resolves its mode through `production_propagation_mode()`. That function returns NONE unless
`model2t.tcdp.experimental_enabled` is set, so the untouched `model2t_mode: "TCDP"` default in
`mission_config.py` (another workstream's file) has no effect. TCDP and GENERIC stay reachable only through an
explicit `mode=`. `tests/contract/test_runtime_defaults.py` checks the default, the resolver, the fact that
`children.py` uses the resolver, and that a default Model2T leaves a never-observed neighbour of a corroded
component UNKNOWN (no INFERRED claim, U_O >= 0.9). The E003 arms and the TCDP unit tests now pass `mode=TCDP`
explicitly. The strict xfail on `tests/unity_live/test_i3_unity.py::test_persistent_technical_belief` was
removed. That test was **not** re-run, because it needs Unity.

### 2. Crack temporal model and calibration

* `crack_filter.py` (new, default `crack_growth.model = REGIME_MIXTURE`): the crack belief is a discrete joint
  over ln(length) (240 cells, 0.1 mm to 3 m) and a growth regime. The regimes are STABLE (no growth, prior
  0.85) and six RUN-AWAY exponential rates (0.2 to 11 per yr). Regimes switch through initiation (0.02/yr),
  arrest (0.3/yr) and drift between neighbouring rates (0.5/yr), and ln L diffuses at 0.1/sqrt(yr).
  Prediction uses physical `delta_t`: mission ticks accumulate and are applied once they pass 1 h, and long
  gaps are split into sub-steps of at most 30 days. Uncertainty therefore grows with elapsed time and the
  predictive distribution is heavy-tailed. Readings use the unchanged sensor-characterised likelihood. The
  persistent-bias floor becomes a floor on the ln L spread. A reading the belief cannot explain mixes 2 % of
  log-uniform mass in before the update. Growth past the 3 m bound is refuted (renormalised away), not piled
  up at the top.
* The reported crack level is the posterior median; the variance is the posterior's.
* Healthy sensors: `noise_ua_reference = 0.1` (the declared sigmas hold at the OK-health aleatoric level),
  `noise_ua_gain` 3 -> 5, and `wall_abs_sigma_m` 0.30 -> 0.25 mm.
* The iteration-2 Gaussian model remains available as `crack_growth.model = CONSTANT_RATE` (ablation arm in
  E004-R3).

DEV record: these runs are design only and are not evidence. They use scratch scripts on DEV seeds
5100000-5100009 (E001 protocol, no GRU). Gain is latest minus Model2T MAE in mm, and cov is the pooled 95 %
coverage.

| Step | crack gain L0 | crack cov L0 / L0.4 | corrosion cov L0 / L0.4 |
|---|---|---|---|
| iteration 2 (reproduced) | -0.36 [-2.63, 2.19] | 0.838 / 0.727 | 0.997 / 0.950 |
| regime mixture, top cell absorbing | -20.8 [-27.8, -14.8] | 0.953 / 0.912 | 0.997 / 0.950 |
| + growth past the bound refuted (mean) | +0.89 [-2.72, 4.22] | 0.958 / 0.940 | 0.997 / 0.950 |
| + posterior median | +4.12 [1.29, 6.98] | 0.952 / 0.936 | 0.997 / 0.950 |
| + healthy-sensor noise reference, gain 5, wall floor 0.25 mm | +4.93 [2.41, 7.74] | 0.952 / 0.938 | 0.981 / 0.951 |

The corrosion variants tried and rejected on the same seeds (L0 / L0.4 corrosion coverage): wall floor
0.2 mm, 0.988 / 0.846; rel 0.10 with bias 0.05, 0.997 / 0.948; partial-view minimum 0.7, 0.997 / 0.954;
floor 0.2 mm with gain 5, 0.995 / 0.941.

Final design checks, including the 30-day sub-steps added after a unit test showed that one 365-day step
under-spread the prediction (0.0051 vs 0.0172 m^2 over 12 steps). Scratch E001 protocol, 10 seeds per block:

| Seeds | crack gain L0 | crack gain L0.4 | crack cov L0 / L0.4 | corrosion cov L0 / L0.4 |
|---|---|---|---|---|
| DEV 5100000-09 | +4.97 [2.60, 7.69] | +41.8 [36.0, 48.8] | 0.952 / 0.939 | 0.981 / 0.951 |
| DEV 5100010-19 | +5.74 [4.26, 7.25] | +40.6 [32.3, 47.7] | 0.955 / 0.949 | 0.985 / 0.963 |
| DEV 5100020-29 | +6.25 [4.29, 8.80] | +40.0 [31.4, 49.2] | 0.952 / 0.946 | 0.979 / 0.948 |
| VALIDATION 5200000-09 | +3.75 [1.31, 5.85] | +44.4 [34.0, 55.9] | 0.949 / 0.940 | 0.981 / 0.951 |
| VALIDATION 5200010-19 | +6.18 [4.22, 8.10] | +47.0 [37.0, 59.4] | 0.940 / 0.927 | 0.975 / 0.946 |
| VALIDATION 5200020-29 | +3.87 [2.39, 5.33] | +39.0 [28.6, 48.8] | 0.947 / 0.943 | 0.978 / 0.941 |

2T-E004-R3-DEV (10 DEV seeds, frozen design): crack TB between inspections -3.2 [-7.5, 0.8] mm, before the next
inspection +5.4 [1.3, 9.2] mm. Crack coverage between inspections: Model2T 0.975, iteration-2 constant rate
0.908, no growth prior 0.866, latest 0.676.

2T-E003-R3-DEV (30 DEV seeds, run before the sub-steps were added): corrosion RB_TCDP -0.111 [-0.118, -0.104].
TCDP still hurts corrosion, so it stays out of production.

**Evaluation defect found in E002/E004:** their R2 scoring compared each arm only on the items that arm
claimed. Model2T's crack claim is UNKNOWN until a detection, so Model2T was scored on detected (large)
cracks while latest-observation was scored on every crack. Much of the R2 "-163 mm" in E004 came from this
mismatch: per crack-size regime, the two arms were within about 10 %. R3 declares `scoring: paired_latent` in
the config before the run. Model2T arms are scored on their internal estimate, claimed or not (as E001-R2
already did), and an item counts only when every arm has an estimate. E004-R3 also adds a
`MODEL2T_CONSTANT_RATE` arm. The ablation `NO_RATE_PRIOR` now also removes crack growth (every crack STABLE,
no initiation).

### 3. Surface coverage and region readings ON

* `coverage.py` (new): Model2T takes the SURVEYED DESIGN geometry from the mission context. This is the
  geometry association already uses; it is not Twin truth. `children.py` passes `design_geometry` to
  `Model2T.initialize`. A capsule is divided into ceil(length / 0.5 m) axial bins x 8 sectors; a sphere or box
  into 6 faces. A reading covers only the cell that holds its MEASURED surface point
  (`Evidence.spatial_support`, frame WORLD). The belief side does not know the viewed area, so it never
  credits more than that cell.
* Each component with geometry has a READ-SURFACE part: a belief minted by Model2T, `registry_entity_id` None,
  entity type `<TYPE>_READ_SURFACE`, linked to the component by PART_OF. While the component's condition is
  open, a reading's DIRECT revision goes to the read-surface part, which consumes the evidence and has
  OBSERVED claims. "Open" means coverage below 0.8 and the read part alone below the worst band. The
  component itself gets a RELATIONAL coverage revision with no evidence: worst-case quantities, condition and
  severity UNKNOWN, `surface_coverage` and `observed_region_condition` INFERRED, and
  U_O >= 1 - coverage. Once the component is fully covered, or the read part already reaches the worst band
  (a worst case can only be worse), readings update the component DIRECTly and its condition is OBSERVED.
* Components without geometry keep the whole-component-view behaviour, so the 2T experiments are unchanged.
* `world.REGION_READINGS = True`. On the I3 surrogate the target's six near-side lane readings cover 6 of 64
  cells. The target stays UNKNOWN with U_O 0.906, EGDC still raises the need, MCBR plans, and the far-side
  patch (crack about 80 mm, severity 1.0) makes the condition OBSERVED/FAILED after inspection. The patch
  oracle still reports the far side unobserved (max visible fraction 0.0).
* The four tests that failed with the switch ON now pass unchanged:
  `test_hidden_target_stays_unknown_from_the_lane`, `test_before_the_segment_condition_is_unknown_with_high_uo`,
  `test_smoke_mcbr_produces_a_plan_with_candidate_table_and_rejections` and
  `test_goal_then_commands_only_through_the_gateway`. `gs06` (every DIRECT revision has OBSERVED corrosion)
  also passes unchanged, because DIRECT revisions from partial views belong to the read-surface part.

### 4. FINAL-3 (run once, after the design was frozen)

The R2 FINAL seeds 5300000-5300059 are SPENT and were not read. FINAL-3 = 6500000-6500059, declared in the R3
configs as a config-local `final_3` partition (`partitions.yaml` is digest-pinned). Before it was declared, all
of `configs/`, `conrad/`, `scripts/` and `tests/` were grepped for 65xxxxx; nothing used it. The used ranges
were 31/32/33/34xxxxx, 51/52/53/54xxxxx, 61-64xxxxx and 71-76xxxxx, and `partitions_i5.yaml` reserves nothing
there. `common.checked_seeds` enforces the split. A final split is read only with final_evaluation; it must be
disjoint from the other local partitions, from the spent list, and from every pinned mission/abstract seed.
The gate thresholds and the coverage band [0.90, 0.99] are unchanged.

**2T-E001-R3** (60 seeds, 3 tier-3 episodes x 16 steps x 2 levels each; gain = latest - Model2T MAE, mm):

| level | quantity | Model2T MAE | latest MAE | gain [95 % CI] | cov95 | in band |
|---|---|---|---|---|---|---|
| 0.0 | corrosion | 0.098 | 0.173 | +0.075 [0.073, 0.076] | 0.980 | yes |
| 0.0 | crack | 29.80 | 35.62 | **+5.81 [4.93, 6.72]** | 0.950 | yes |
| 0.4 | corrosion | 0.251 | 0.602 | +0.351 [0.344, 0.357] | 0.949 | yes |
| 0.4 | crack | 55.65 | 99.03 | +43.4 [39.3, 47.7] | 0.936 | yes |

Crack gain against the other arms: latest-debiased +5.01 [4.39, 5.65] at L0 and +51.5 [46.6, 56.6] at L0.4; GRU
+103 and +78; the pre-repair update +5.87 and +42.8. Corrosion against the GRU at L0.4 is still a loss:
-0.022 [-0.027, -0.016]. So is corrosion against the pre-repair absolute update: -0.014 [-0.015, -0.013] at L0
and -0.029 [-0.033, -0.025] at L0.4. The gate rule only compares against latest-observation.

**2T-E003-R3** (60 seeds): corrosion RB_TCDP -0.110 [-0.116, -0.103]; crack RB_TCDP -0.038 [-0.107, 0.033].
Contamination GENERIC - TCDP: corrosion +0.383 [0.364, 0.403], crack +0.212 [0.188, 0.236]. `tcdp_benefit`
False.

**2T-E004-R3** (60 seeds, paired scoring): corrosion TB between inspections +0.044 [0.042, 0.047] mm, before the
next inspection +0.039 [0.036, 0.041]. Crack TB between +2.48 [1.21, 3.77] mm, before the next inspection
+9.87 [8.72, 11.03]. Crack coverage between inspections: Model2T 0.975, constant rate 0.890, no growth prior
0.843, latest 0.654. Crack MAE between inspections: Model2T 72.6, constant rate 73.9, no growth prior 69.9,
latest 75.1. The no-growth ablation has a lower MAE but under-covers.

**2T-E002-R3** (first 20 FINAL-3 seeds, full Model2T path with production defaults, so no TCDP; paired
scoring; gain on components hidden now but seen before):

| coverage | corrosion vs latest | corrosion vs GRU | crack vs latest | crack vs GRU |
|---|---|---|---|---|
| 0.5 | +0.068 [0.063, 0.073] | +0.160 [0.151, 0.168] | -2.3 [-5.7, 1.5] | +89 [71, 109] |
| 0.2 | +0.051 [0.045, 0.056] | +0.139 [0.125, 0.152] | -3.2 [-7.3, 1.0] | +79 [66, 93] |
| 0.05 | +0.041 [0.031, 0.052] | +0.110 [0.096, 0.124] | +1.3 [-4.5, 7.5] | +69 [52, 87] |

The UNKNOWN rate on never-observed components is 1.0 on every seed, for both quantities. Hidden-crack
persistence is now level with latest-observation (every CI straddles 0). In R2 it looked like a loss of
144 to 433 mm, and that was mostly the scoring mismatch.

### 5. Gates (`scripts/record_gate_evidence.py 2T 2T-TCDP`, FORMAL, 2T entries point at the R3 artifacts)

* **2T: PASS.** On FINAL-3, corrosion and crack beat latest-observation at both degradation levels, with every
  CI above 0, and all four 95 % coverages are inside [0.90, 0.99]. The listed unit tests pass.
* **2T-TCDP: FAIL.** There is still no benefit versus no propagation; corrosion is worse, with the CI below 0.
  Production does not propagate (ADR-0009).

### 6. Tests (frozen code)

* `tests/unit/domains/technical`, `tests/property/domains/technical`, `tests/contract`,
  `tests/integration/test_i3_structural.py`, `tests/acceptance` and `tests/leakage`: 265 passed, 1 xfailed
  (the existing strict I4 xfail), 9 errors. All 9 errors are in `tests/acceptance/test_i5_integrated_missions.py`,
  which needs `artifacts/experiments/M1-ACTION-E002/m1_action_e002.json`. That artifact belongs to another
  workstream and is not produced yet. `tests/leakage` on its own: 27 passed.
* New: `tests/unit/domains/technical/test_m2t_iteration3.py`, which covers crack uncertainty growing with
  physical time, tick accumulation, regime learning, the constant-rate ablation, read-surface coverage, a
  worst-band finding, the no-geometry fallback, geometry for an unregistered component, and the guarded
  FINAL-3 split. Three contract tests were added.
* The Unity I3 test was not run.

### Open items

* Unity I3 should be re-run to confirm the fix and to remove the gate FAIL on record.
* The owner of `mission_config.py` should change `model2t_mode` to "NONE" so that the config matches ADR-0009.
  Today the resolver makes the old default inert.
* Coverage credits one cell per reading. A real footprint (range and FOV reaching the belief side) would credit
  honest views faster. The 0.8 completeness fraction is an ENGINEERING_ESTIMATE.
* The read-surface part is only committed on readings, not on temporal prediction.
* The crack growth prior (rates, hazards, 3 m bound) is uncalibrated. The E004 no-growth ablation has a lower
  crack MAE between inspections but under-covers.
