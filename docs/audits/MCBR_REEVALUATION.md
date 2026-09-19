# MCBR re-evaluation and production planner selection (2026-09-19)

All data is SYNTHETIC_ONLY. Nothing here is real-data or physical validation.

## Verdict

**I4 is FAIL.** In the abstract occlusion world (ACTIVE-MCBR-E002), the frozen production planner clearly beats fixed, random and coverage views on the final partition. It does the same on the held-out OOD families. In the integrated surrogate mission (ACTIVE-MCBR-E003, FLAGSHIP-I4 family), it is **worse than fixed views and coverage**, and the paired 95 % CIs lie entirely below 0. The behavioral test `tests/acceptance/test_i4_matched_policy.py` fixed its criterion before any final result existed. That test fails on E003.

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
