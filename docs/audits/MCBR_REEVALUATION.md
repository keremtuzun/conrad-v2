# MCBR re-evaluation and production planner selection (2026-09-19)

All data is SYNTHETIC_ONLY. Nothing here is real-data or physical validation.

## Verdict

**I4 is FAIL** (re-confirmed 2026-09-20 with the belief-side predictive model and a formal Unity run: see
section 8). In the abstract occlusion world (ACTIVE-MCBR-E002), the frozen production planner clearly beats fixed, random and coverage views on the final partition. It does the same on the held-out OOD families. In the integrated surrogate mission (ACTIVE-MCBR-E003, FLAGSHIP-I4 family), it is **worse than fixed views and coverage**, and the paired 95 % CIs lie entirely below 0. The behavioral test `tests/acceptance/test_i4_matched_policy.py` fixed its criterion before any final result existed. That test fails on E003.

The old MCBR (A-B10_mcbr_full) remains a failed hypothesis. It loses to every simple baseline in both settings, except fixed views and uncertainty-NBV in the abstract world.

## 1. Partitions

- `configs/eval/partitions.yaml` (version `partitions-2026-09-19-v1`). Canonical-content SHA-256 `074ae11310252217315b2ac3324fd914d1964a3e34b76b57a65f0ca3a3acb904`, pinned in `conrad/evaluation/partitions.py` (`PARTITIONS_SHA256`). If the file changes, loading refuses it.
- Seeds and families are disjoint across partitions, and this is checked on load.

| domain | development | validation | final_test | ood_test |
|---|---|---|---|---|
| abstract world seeds | 2026201-3, 1000, 1001, 3100000-039 | 3200000-039 | 3300000-039 | 3400000-029 |
| abstract families | open_field, standard, cluttered | same | same | ood_dense_occluders, ood_wide_weld (held out) |
| mission world seeds | 2026201-3, 1000, 1001, 5100000-039 | 5200000-039 | 5300000-059 | 5400000-039 |
| mission families | straight_pipeline | same | same | pipeline_with_supports (held out) |

- Seed 2026201 is `DEVELOPMENT / CONTAMINATED_FOR_FINAL_EVALUATION`. Seeds 2026202/3 and the scratch seeds 1000/1001 are development too.
- Access guard: every seed request declares a purpose.
  - design and tuning may read development and validation only.
  - selection may read validation only.
  - final_evaluation may read final_test and ood_test only.
- `purpose_scope("design")` marks a code path. A held-out request inside it raises `PartitionAccessError`, even if the caller claims another purpose.
- The experiment runner also refuses caller-supplied seeds that fall outside its stage's partitions.
- A note was appended to `docs/research/EXPERIMENT_RESULTS_2026-09-18.md`. It marks E001 and the s2026201/2/3 flagship runs as development evidence and records the previous I4 evidence as FAIL. Nothing was deleted.

## 2. Candidates

All candidates sit behind the unchanged `InformationNeed -> candidates -> feasibility filter -> value ranking -> ObservationPlan` pipeline (`MCBRPlanner`). They share the candidate generator and the filter.

- Existing: A-B0 random, A-B1 fixed inspection, A-B2 coverage, A-B3 frontier, A-B4 geometric NBV, A-B5 entropy NBV, A-B6 standard EIG (analytic channel sum), A-B7 uncertainty NBV, A-B9, A-B10 current MCBR, and A-B10 with the stop rule off (diagnostic).
- New (`conrad/active/predictive.py`, `conrad/active/rankers.py`). All three use an optional `PlanningRequest.predictive`, which is a belief-side Gaussian/binary model plus a nominal sensor model:
  - **A-B6b standard Bayesian EIG**: expected entropy reduction, generic.
  - **A-B11 mission-conditioned**: expected reduction of the mission-relevant absolute belief error.
  - **A-B12 hypothesis-discrimination**: expected reduction of P(misclassify). It covers two things:
    - whether each weld sector's severity is above the action threshold 0.5 (Gauss-Hermite preposterior);
    - the binary H1/H2 hypothesis.
- Optional cost handling: `none`, `subtract` or `ratio`. There is also an optional minimum-value stop rule.
- When `predictive` is absent, all three fall back to the analytic MCBR mission value.

### Matched budgets (abstract world)

- Every planner gets:
  - at most 4 observations;
  - an energy cap of 900 J and a time cap of 90 s, enforced for every planner through the shared feasibility filter (`MissionBounds.energy_budget_j`, need deadline);
  - the same two sensors;
  - common random numbers.
- Baselines never stop early. MCBR variants may stop, but they stay within the same caps.

### Matched budgets (mission)

- Same 100 s duration, the same `max_plans_per_need=4`, the same single structural sensor, and the same config.
- Energy and travel are measured, not capped.

## 3. Diagnosis: why the current MCBR loses

The numbers come from the design run on 24 DEVELOPMENT worlds (4 scenario types × 3 replicates each).

1. **The NOT_WORTH_COST stop rule.** A-B10 stopped in 49.7 % of episodes.
   - In the OOD (turbid) scenario it stops at once, giving a mission-error reduction of 0.000. The same ranking without the stop gets 0.491.
   - Overall: 0.297 with the stop vs 0.434 without it.
   - The cost weights put value below cost for useful views.
2. **The value estimate is poorly aligned with actual error reduction.**
   - Spearman correlation between planner score and oracle one-step error reduction: 0.37 for A-B10, against 0.56 for A-B11 and 0.44 for A-B6b.
   - In the coverage scenario, every analytic-gain planner (A-B2/4/5/6/9/10) gets about 0.05, against 0.29-0.41 for the belief-predictive rankers.
   - Cause: the channel gains are region-level visibility × uncertainty level. They do not predict which sectors a given sensor can measure, at what noise, or whether a hypothesis vote is returned.
3. **Visibility was sensor-agnostic.** The shared `predicted_visibility` ignores sensor range. I added a sensor-aware option (`MCBRConfig.sensor_aware_visibility`, on by default when a predictive belief exists). It changes A-B10 by less than 0.001, so it is not the cause. Causes (1) and (2) are.
4. **The integrated mission.** Model2T messages carry no variances, so no predictive belief exists there, and every "smart" ranker degrades to the analytic value.
   - On the final surrogate partition, A-B10 took **0 observations** in all 30 worlds (all NOT_WORTH_COST).
   - The target stayed UNKNOWN in 10 of 30 worlds.

## 4. Selection (VALIDATION only, ACTIVE-MCBR-SEL001)

- 40 validation worlds. Primary metric: actual mission-relevant hidden-state error reduction.
- 22 arms: 14 candidates plus 8 variants tuned on development.
- Winner: **V-bayes_eig_ratio**, a conventional Bayesian EIG per unit cost (`value / (0.05 + cost)`), with mean 0.505.
- Its lead over A-B6b (0.500), A-B11 (0.499) and the mission-ratio variants (0.494) is **not significant**: paired CI [-0.012, +0.021] vs A-B6b.
- It beats A-B10 by +0.192 [0.171, 0.212].
- As the brief requires, a conventional EIG method that wins becomes the production planner under the MCBR interface.

### Frozen config

- File: `configs/active/mcbr_frozen.yaml`.
- Planner section: `{selected: V-bayes_eig_ratio, ranker: {kind: bayes_eig, cost_mode: ratio, cost_weight: 1.0}}`.
- **config_digest `8eca16cc896e560b26cbe9814828fe9f75801d843e0e38bedb5442299805d903`**, verified on load.
- partitions digest recorded.
- git_commit `0918f5547e9f9f75f159f9194428f121a436e4e8` is the working-tree parent. The frozen code is uncommitted.
- The mission runtime default is now `planner: PRODUCTION` (`MissionRuntimeConfig`, `Deliberation`). Runs report `PRODUCTION[V-bayes_eig_ratio]`.

## 5. Final evaluation (after freezing)

Both experiments are registered in `conrad/evaluation/dispatch.py`. Artifacts are in `artifacts/experiments/ACTIVE-MCBR-E00{2,3}/`. CIs are 95 % paired percentile bootstraps over worlds (`conrad.evaluation.metrics.paired_seed_comparison`, 4000 resamples). A positive benefit means PRODUCTION is better.

### ACTIVE-MCBR-E002: abstract occlusion world

- FINAL_TEST: **40 worlds × 3 noise seeds × 4 scenario types**.
- OOD_TEST: **30 worlds × 3 × 4** (held-out families).

Final test, means:

| planner | mission err. red. | hidden-state err. red. | obs | redundant | travel m | energy J | info/s | info/kJ | collisions | success |
|---|---|---|---|---|---|---|---|---|---|---|
| PRODUCTION | 0.516 | 0.407 | 2.40 | 0.45 | 9.9 | 468 | 0.021 | 1.10 | 0.019 | 0.58 |
| A-B11 mission-cond. | 0.511 | 0.391 | 1.94 | 0.32 | 14.9 | 652 | 0.015 | 0.78 | 0.006 | 0.63 |
| A-B6b Bayes EIG | 0.509 | 0.390 | 1.93 | 0.31 | 15.2 | 665 | 0.015 | 0.77 | 0.006 | 0.63 |
| A-B12 hyp.-discrim. | 0.480 | 0.368 | 2.26 | 0.56 | 15.7 | 689 | 0.013 | 0.70 | 0.029 | 0.62 |
| A-B0 random | 0.351 | 0.285 | 2.82 | 0.88 | 17.1 | 740 | 0.009 | 0.50 | 0.029 | 0.49 |
| A-B2 coverage | 0.333 | 0.238 | 1.14 | 0.21 | 19.2 | 795 | 0.008 | 0.39 | 0.008 | 0.47 |
| A-B10 MCBR (old) | 0.318 | 0.246 | 1.67 | 0.32 | 11.5 | 503 | 0.008 | 0.44 | 0.006 | 0.48 |
| A-B1 fixed | 0.039 | 0.011 | 3.28 | 1.47 | 18.0 | 753 | 0.001 | 0.06 | 0.042 | 0.32 |

Paired benefit of PRODUCTION on the final test:

| vs | mission err. red. | hidden-state err. red. | success |
|---|---|---|---|
| fixed | +0.477 [0.448, 0.506] | +0.396 [0.374, 0.417] | +0.26 [0.20, 0.31] |
| random | +0.165 [0.140, 0.192] | +0.122 [0.102, 0.144] | +0.08 [0.04, 0.13] |
| coverage | +0.184 [0.163, 0.203] | +0.170 [0.150, 0.188] | +0.11 [0.07, 0.15] |
| A-B10 | +0.198 [0.176, 0.219] | +0.161 [0.142, 0.178] | +0.10 [0.05, 0.14] |

- Against A-B6b and A-B11, the mission-error difference is not significant: +0.007 [-0.007, 0.020].
- PRODUCTION also uses less energy and travel than every I4 comparator.

**OOD test (30 held-out-family worlds).** PRODUCTION's mission-error reduction is 0.509 (hidden-state 0.411). Paired benefit on mission error:

| vs | benefit |
|---|---|
| fixed | +0.466 [0.437, 0.495] |
| random | +0.159 [0.122, 0.198] |
| coverage | +0.096 [0.058, 0.130] |
| A-B10 | +0.147 [0.116, 0.177] |

There are also caveats:

- PRODUCTION had more collisions (with occluders the belief map missed) than fixed, random and coverage in OOD: +0.058, +0.025, +0.033 per episode. The CI for each excludes 0.
- On the final test it had more collisions than fixed: +0.023 [0.004, 0.044].
- The fixed baseline takes more observations than PRODUCTION. Coverage takes fewer, because it stops finding novel views.

### ACTIVE-MCBR-E003: integrated SURROGATE mission

- FLAGSHIP-I4 on the python kernel. This is **SURROGATE evidence**; the formal Unity run belongs to the integrator.
- **30 worlds × 1 run each** per partition. Each world has one seed, so there are no separate noise seeds.
- Planners: PRODUCTION, fixed, random, coverage, A-B10.

Final test (straight_pipeline, 30 worlds). HSE improvement = 1 − error/prior-error, averaged over corrosion and crack.

| planner | HSE improvement | obs | energy J | travel m | collisions | target OBSERVED |
|---|---|---|---|---|---|---|
| A-B2 coverage | 0.714 | 0.37 | 19631 | 22.6 | 0.033 | 1.00 |
| A-B1 fixed | 0.702 | 0.33 | 19407 | 22.2 | 0.033 | 1.00 |
| PRODUCTION | 0.664 | 0.47 | 19842 | 22.7 | 0.033 | 1.00 |
| A-B0 random | 0.634 | 0.40 | 19895 | 22.8 | 0.033 | 0.93 |
| A-B10 MCBR (old) | 0.442 | 0.00 | 16868 | 19.3 | 0.033 | 0.67 |

Paired benefit of PRODUCTION on HSE improvement:

| vs | benefit |
|---|---|
| fixed | **−0.039 [−0.082, −0.003]** |
| random | +0.030 [−0.028, +0.097] |
| coverage | **−0.050 [−0.095, −0.014]** |
| A-B10 | +0.222 [+0.112, +0.345] |

- Information per time and per energy follow the same ordering.
- The one collision per 30 runs is common to all planners.

**OOD (pipeline_with_supports, 30 worlds).** All five planners produced *identical* runs:

- 0 inspection goals and HSE improvement 0.156;
- the target was OBSERVED from the transit lane in every run.

MCBR never acted, so this partition is **non-informative** about view selection. No OOD claim can be made for the mission.

## 6. Tests

- `tests/acceptance/test_i4_matched_policy.py` reads the stored final artifacts and checks:
  - frozen-planner digest and partition digest;
  - at least 20 worlds;
  - a paired CI lower bound above 0 vs fixed, random and coverage.
- Current result: the E002 test passes and the **E003 test fails**, so I4 = FAIL.
- Renamed smoke-level tests:
  - `test_flagship_mission.py::test_smoke_mcbr_produces_a_plan_with_candidate_table_and_rejections`
  - `::test_integration_i4_open_threshold_record_stays_not_evaluable`
  - `test_golden_suite.py::test_gs02_smoke_mcbr_returns_a_feasible_view_or_a_stop_status`
- The flagship docstring now states that seed 2026201 is contaminated and not I4 evidence.

## 7. What would be needed for I4 (not done here)

The integrated runtime gives the planner no predictive belief. It would need a belief-side predictive model: Model2T variances for the target quantities, plus a sensor model of the structural payload through the 2S map. Without that, the production ranker falls back to the analytic value, which loses to fixed views in the mission.

Building it touches perception/twin-side files owned by another agent, and it would need a **new** selection round on development/validation. The current final partitions have now been seen, so an I4 re-test would need a new partition version with fresh final seeds.

## 8. I4 repair: the belief-side predictive model (2026-09-19/20)

Section 7 named the missing piece: in missions MCBR had no predictive belief, so the frozen ranker fell back to
the analytic mission value. Model2T iteration 3 (surface coverage geometry, calibrated crack/corrosion
variances, region readings) supplies what a predictive model needs. All data is SYNTHETIC_ONLY.

### 8.1 The model

`conrad/active/surface_predictive.py` (belief plane, generic) plus `conrad/orchestration/mission_predictive.py`
(builds it from the mission's own Model2T / Model2S). For the need's target component:

- Cells: the SURVEYED design surface of the component (the geometry association already uses) is divided
  exactly as Model2T's coverage divides it (`capsule_cells` reproduces `SurfaceGeometry.cell_of`).
- P(the unread worst case lies in cell c): uniform over cells, zero on cells a reading already covered
  (Model2T `covered`), and multiplied by `1 - 0.8 * w`, where `w` is the best observation weight that cell had
  from the robot's OWN estimated past positions (its track, the sensor range, the incidence cosine and the
  Model2S clear-ray probability). Model2T credits only the one cell that holds a reading's measured point, so
  without this term the lane pass leaves the near side as "unread" as the far side.
- Per quantity (corrosion depth, crack length) a scalar `q:unread` with the Model2T population prior as its
  belief and a one-reading noise std built from the declared `SensorCharacteristics`: relative sizing scatter,
  absolute floor, the partial-view mixture and the persistent per-sensor bias. For cracks the value is also
  multiplied by the declared probability of detection at that level.
- A candidate's value for that scalar is the EXPECTED entropy reduction of an erasure channel,
  `phi(v) * 0.5 ln(var / var_post)`, with `phi(v) = sum_c P(defect in c) * w_c(v)` computed by Model2S
  ray-casting through the belief map at the candidate pose. It reaches the unchanged rankers as an
  information-equivalent noise std, so the production ranker's `entropy` value IS this conventional EIG.
- An optional `q:read` scalar (refining the worst indication at its Model2T locus, capped by the persistent-bias
  floor) was implemented and tested, and DEVELOPMENT selected it OFF.
- Motion cost stays where it was: the shared `navigation_cost` of the request.
- Every candidate always reports every scalar, so `predicted_coverage` is 1 and the feasibility filter is
  identical for every planner. The model changes ranking only. The same `PlanningRequest` (predictive included)
  goes to every planner, so budgets and inputs stay matched.
- No truth: `tests/leakage` (27 tests) passes, including the static import guard for `conrad.orchestration`.

Runtime hook (minimal, additive): `Deliberation.predictive_provider` (default None) is set by `MissionRuntime`
from `mission_predictive_provider(...)`, which reads the frozen config; the provider also receives the runtime's
own estimated track.

### 8.2 Selection (DEVELOPMENT only)

Arms were compared on 26 DEVELOPMENT worlds where MCBR actually plans (8 of the `unity_gate` development worlds
7810000-7810019 and 18 of the `mission` development worlds; in the other worlds the lane pass already closes the
target and every planner ties). Mean hidden-state error improvement:

| arm | dev mean |
|---|---|
| C: track discount 0.8, read channel off (SELECTED) | 0.730 |
| A: same, read channel on above a 5 % worst-band probability | 0.727 |
| B: same, read channel always on | 0.547 |
| D: PRODUCTION without any predictive model (the v1 fallback) | 0.636 |
| A-B2 coverage | 0.736 |
| A-B1 fixed views | 0.756 |

Paired against C: fixed -0.026 [-0.064, +0.011], coverage -0.006 [-0.151, +0.136], no-predictive
+0.094 [-0.075, +0.335]. So on development the model helps against the old fallback, and it does NOT beat fixed
or coverage. That was known before any final world was touched.

### 8.3 Frozen production planner v2

`configs/active/mcbr_frozen_v2.yaml` (`conrad.active.production.FROZEN_PATH` now points at it). The `planner`
section is identical to v1, so `config_digest` is still
`8eca16cc896e560b26cbe9814828fe9f75801d843e0e38bedb5442299805d903` and the E002 artifacts stay valid. The new
`mission_predictive` section carries the selected model and its configuration, digest
`2fa5b75d6e9618c0a1274eff284d014a30df4c0ba17a8d83969c2c8609d41431`, verified on load.

### 8.4 I4 worlds

Declared in `configs/eval/active_mcbr_e004.yaml` BEFORE any run: `unity_gate` final_test seeds 7800002-7800013
(12 worlds), the range the partition file reserves for the next formal gates; 7800000 / 7800001 are the I1 / I3
roles. `configs/eval/partitions_unity_gates.yaml` was not edited (it is digest-pinned). The same 12 worlds carry
the surrogate and the formal run. Decision rule, also pre-declared: paired percentile bootstrap over worlds
(4000 resamples), "beats" means the 95 % CI lower bound is above 0.

### 8.5 ACTIVE-MCBR-E004 (surrogate, python kernel, run once)

12 worlds x 5 planners, 100 s mission, `max_plans_per_need` 4, one structural sensor, energy and travel measured.

| planner | HSE improvement | obs | energy J | info/kJ | target OBSERVED |
|---|---|---|---|---|---|
| A-B1 fixed | 0.806 | 0.92 | 15847 | 0.055 | 1.00 |
| A-B2 coverage | 0.790 | 0.83 | 16910 | 0.052 | 1.00 |
| PRODUCTION | 0.707 | 1.33 | 16228 | 0.049 | 0.83 |
| A-B10 MCBR (old) | 0.630 | 0.50 | 15394 | 0.045 | 0.75 |
| A-B0 random | 0.592 | 1.25 | 16903 | 0.042 | 0.75 |

Paired benefit of PRODUCTION on HSE improvement: vs fixed **-0.099 [-0.273, +0.012]**, vs random
+0.116 [-0.067, +0.342], vs coverage **-0.083 [-0.261, +0.067]**, vs A-B10 +0.077 [-0.007, +0.237]. On
information per time and per kJ, PRODUCTION is below fixed (info/kJ -0.006 [-0.014, -0.000]). In 5 of the 12
worlds the lane pass already closed the target and every planner tied.

**Surrogate I4 = FAIL** (`artifacts/gates/I4/evidence_surrogate.json`, recorded with
`scripts/record_gate_evidence.py I4 --surrogate`).

### 8.6 Formal Unity run (I4-UNITY, run once, sequential)

`tests/unity_live/test_i4_unity.py`: the same 12 worlds, 4 planners, 48 sequential flights through the built
Unity player (1:08 h), recorded by `scripts/record_unity_gate_evidence.py I4`. Means over the 12 worlds:

| planner | HSE improvement | info/s | info/kJ |
|---|---|---|---|
| PRODUCTION | 0.857 | 0.0086 | 0.232 |
| A-B1 fixed | 0.792 | 0.0079 | 0.218 |
| A-B0 random | 0.609 | 0.0061 | 0.171 |
| A-B2 coverage | 0.472 | 0.0047 | 0.139 |

Paired benefit of PRODUCTION (95 % CI over worlds):

| vs | HSE improvement | info/s | info/kJ |
|---|---|---|---|
| fixed | **+0.066 [-0.043, +0.236]** | +0.0007 [-0.0004, +0.0024] | +0.014 [-0.013, +0.055] |
| random | +0.249 [+0.074, +0.453] | +0.0025 [+0.0007, +0.0045] | +0.061 [+0.019, +0.110] |
| coverage | +0.386 [+0.017, +1.009] | +0.0039 [+0.0002, +0.0101] | +0.093 [+0.005, +0.240] |

Closed loop: MCBR planned in 7 of the 12 worlds, and every one of those 7 produced new target evidence; the
target ended OBSERVED in 12 of 12. The bundle of the first world replays bit-exactly, and the leakage scan over
all 12 PRODUCTION bundles is clean.

**Formal I4 = FAIL.** Three of the five criteria pass (closed loop, beats random views, beats coverage-only).
The two that fail are "beats fixed views on actual hidden-state reconstruction" and "beats simple views on
information/time/energy", both because the fixed inspection route is not beaten with a CI above 0.

### 8.7 Reading of the result

- The predictive model is what turned the mission comparison around against random and coverage views: in E003
  MCBR lost to both or tied, and in the formal Unity run it beats both with CIs above 0. It also beats the old
  A-B10 on the surrogate.
- Fixed views remain unbeaten. In this scenario family the fixed route's first candidate is already an oblique
  far-side view of the inspection station, and the outcome is dominated by whether one reading of the far-side
  patch lands in Model2T's worst condition band. That is a Model2T calibration question (partial-view lower
  bounds under-size a crack seen at a 0.4 to 0.9 in-view fraction), not a view-selection question. The same
  effect makes 5 of the 12 surrogate worlds ties.
- The surrogate and the formal run disagree in size (-0.099 vs +0.066 against fixed). Both CIs include 0, so
  neither supports a claim. The python kernel is not the formal path; only the Unity numbers are formal evidence.
- Nothing was tuned on the 12 I4 worlds: the model and its configuration were frozen before the first run on
  them, and the surrogate and the formal run were each executed once.

### 8.8 What would be needed next (not done here)

- Model2T crack sizing under partial views: the worst-band decision, and therefore the whole mission metric,
  hangs on it. A view-quality aware likelihood (in-view fraction from the planner's own geometry) is the
  candidate, and it needs its own development/validation round.
- A scenario family where a fixed route cannot see the defect (the current family rewards it), so that view
  selection is what the metric measures. **Taken up in section 9.**
- The predictive model ranks one component's surface. Multi-component needs, and the read-surface refinement
  channel, stay open.

## 9. The I4 world family (2026-09-20): ACTIVE_INSPECTION_OCCLUDED_V1

Section 8.7 and the remediation audit both named the first honest route: a world family where the defect is
genuinely not visible from the nominal route. It is declared in `docs/audits/I4_WORLD_FAMILY.md` (design and
justification written before the split file existed, and therefore before any final or OOD seed was drawn),
frozen in `configs/eval/i4_occluded_family.yaml` (digest
`f77c5dbb7d665e3740e76cd4dfd97f37d0f0060bc2b336af6391ff0758877bdd`), and split in
`configs/eval/partitions_i4_occluded.yaml` (digest
`d929beb67c051ea1116e33ebb8e51cd638f5894ef54989a0a3f2ec3a9221b557`, domain `i4_occluded`).

Reading the generator while building it produced the mechanism behind "5 of 12 worlds tied outright", which
was not previously written down: `_patch_mask` puts the `far` defect on the WORLD +Y normal of the pipe
heading, while `transit_lane` puts the lane on the right-hand normal of that heading after a possible axis
reversal. The two rules agree only for some headings, so for a share of the sampled worlds the "hidden"
defect is on the lane side. Measured on development world 7810000: patch visible fraction 0.854 during the
lane pass with zero accepted inspection goals. The new family replaces that rule with `defect.side: off_lane`,
computed from the lane the mission actually flies, and adds an unregistered two-panel occluder that leaves one
angular window whose direction is sampled per world.

The old `straight_pipeline` results in sections 5 and 8 are NOT replaced. They stay as the record of what MCBR
does on a family whose nominal route already sees the defect. The new family is reported next to them.

`ACTIVE-MCBR-E005` (`conrad/evaluation/decision_experiments/active_mcbr_i4_occluded.py`) runs the family in
four stages: `design` (development), `selection` (validation), `e005` (final test) and `e005_ood` (held-out
structural family). The comparison is complete: random, the fixed nominal route, coverage-only, geometric NBV,
entropy NBV, uncertainty NBV, a conventional Bayesian EIG, the old A-B10 and PRODUCTION, all on the identical
`PlanningRequest` under matched budgets.

### 9.1 Development and validation

Full tables in `docs/audits/I4_WORLD_FAMILY.md` sections 7 and 8. Both were run once, on the code that
includes Model2T iteration 4. The final and OOD splits are untouched. Mean hidden-state error improvement:

| planner | development (24) | validation (24) |
|---|---|---|
| `A-B5b_entropy_nbv_predictive` | 0.350 | 0.424 |
| `PRODUCTION` (V-bayes_eig_ratio) | 0.365 | 0.406 |
| `A-B6b_bayes_eig` | 0.394 | 0.361 |
| `A-B4_geometric_nbv` | 0.183 | 0.351 |
| `A-B10_mcbr_full` | 0.144 | 0.288 |
| `A-B7_uncertainty_nbv` | 0.215 | 0.280 |
| `A-B0_random` | 0.225 | 0.279 |
| `A-B2_coverage` = `A-B5_entropy_nbv` | 0.265 | 0.240 |
| `A-B1_fixed_inspection` | 0.064 | 0.000 |

The comparison against fixed views, which is what section 8.6 could not win on `straight_pipeline`, is now
decisive: PRODUCTION beats the fixed nominal route by +0.301 [+0.123, +0.470] on development and
+0.406 [+0.242, +0.578] on validation, and also on information per time and per kJ. It beats coverage-only on
the point estimate on both splits (+0.100 and +0.166). It does NOT beat random views with a CI above 0 on
either split (+0.140 [-0.051, +0.325] and +0.127 [-0.002, +0.271]), and neither does any other arm.

Under the predeclared selection rule no arm is eligible, so nothing is swapped in: the production planner
stays `V-bayes_eig_ratio` plus the frozen mission predictive model, recorded in
`configs/active/mcbr_frozen_v3.yaml` with an unchanged `config_digest`
`8eca16cc896e560b26cbe9814828fe9f75801d843e0e38bedb5442299805d903`. `A-B5b` does not beat it on either split
and the sign of the difference flips between them.

Two findings worth carrying forward:

- `A-B5_entropy_nbv` multiplies a candidate-independent uncertainty level, so in every integrated mission it
  ranks identically to `A-B2_coverage` (identical means on every metric on both splits). It was never an
  independent entropy baseline. `A-B5b_entropy_nbv_predictive` was added as the one that is.
- The binding constraint on this family is statistical power, not the size of the effect: the per-world
  outcome is close to all-or-nothing, so 24 paired worlds give a CI half width of about 0.20 against random.
  The declared final split has 20 worlds.
