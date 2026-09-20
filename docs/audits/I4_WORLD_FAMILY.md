# I4 world family: ACTIVE_INSPECTION_OCCLUDED_V1 (predeclared 2026-09-20)

All data is SYNTHETIC_ONLY. Nothing here is real-data or physical validation.

This document is the pre-declaration of a new integrated-mission world family for gate I4, and of what will be
measured on it. Sections 1 to 6 were written BEFORE the split file `configs/eval/partitions_i4_occluded.yaml`
existed and therefore before any final or OOD seed was drawn. Sections 7 and 8 are filled from development and
validation runs only. The final and OOD partitions stay untouched until the integrator sequences the formal run.

**Code baseline.** Model2T iteration 4 (a reading credits its whole declared footprint, sizing is per surface
region so a partial view moves the estimate locally, and the intact band is recalibrated) changes how much a
chosen viewpoint pays off. Every development and validation number in sections 7 and 8 was produced on the
code that includes it. An earlier development sweep on this family predates that change and is kept, clearly
labelled and unused, under
`artifacts/experiments/ACTIVE-MCBR-E005-DESIGN/stale_pre_model2t_iteration4/`, together with two 6-world
development probes from the same period (one of them at a larger observation budget than the declared one).
Its `README.json` states what each file is and that none of it was used. Nothing in the family definition, the
splits, the budgets or the decision rule was changed between the two runs: only the belief-side code moved.

## 1. Why a new family is needed

The ch25 I4 criterion begins "critical structure partly hidden -> uncertain -> MCBR view -> navigation -> new
evidence -> belief improves". The formal Unity run of 2026-09-19/20 (`docs/audits/MCBR_REEVALUATION.md` section
8.6) beat random views (+0.249 [+0.074, +0.453]) and coverage-only (+0.386 [+0.017, +1.009]) but not the fixed
nominal route (+0.066 [-0.043, +0.236]), and lost to fixed on information per kJ.

The remediation audit named the first cause: in the `straight_pipeline` family the gate premise is barely
exercised. Reading the generator confirms it and gives the mechanism, which was not previously written down:

* `conrad/sim/mission/world.py::_patch_mask` places the hidden defect patch on the side of the target segment
  whose outward normal has a positive WORLD +Y component (`left[1] < 0 -> left = -left`), and `side: far`
  keeps that side.
* `conrad/sim/mission/registry.py::transit_lane` places the transit lane on the RIGHT-HAND normal of the pipe
  HEADING, `right = (d_y, -d_x, 0)`, after possibly reversing the axis.

Those two rules agree only for some headings. For a pipeline whose sampled heading points into -X with a small
-Y component (no axis reversal, `d_x < 0`), `right` has a positive Y component, so the lane is on the same side
as the "far" defect. The nominal transit pass then looks straight at the supposedly hidden patch.

Measured on development world 7810000 (`unity_gate` development split, purpose design): with the fixed-view
planner and zero accepted inspection goals, the patch reached a visible fraction of 0.854 during the lane pass,
first becoming visible at t = 13.3 s with the vehicle at y = +1.3 m and the patch centre at y = -0.72 m. That
world does not contain a hidden critical region at all.

This is the honest explanation of "5 of 12 worlds tied outright" and of "the nominal fixed route already
includes an oblique far-side look". It is a property of the world generator, not a property of MCBR, and no
amount of planner work can fix it. The gate needs a family in which the premise actually holds.

## 2. What the new family is

`ACTIVE_INSPECTION_OCCLUDED_V1` is the pair

* mission scenario `I4-OCCLUDED` (`conrad/sim/mission/scenarios.py`), and
* the world-option block it sets: `defect.side: off_lane`, `defect_variation`, `occlusion`.

on the existing `pipeline_with_supports` Twin2S geometry family. Nothing about the robot, the payload, the
navigation stack, the belief plane or the planner interface changes. The only changes are truth-side world
construction, and they are additive: every existing scenario keeps its current behaviour.

### 2.1 The critical surface is defined relative to the nominal route, not to world +Y

`defect.side: off_lane` places the defect patch on the side of the target segment that faces AWAY from the
transit lane, computed from the lane the mission actually flies. In every world of the family the nominal
inspection route therefore leaves the mission-critical region unresolved: the side-looking structural payload
sees only the lane-facing surface, Model2T credits only the cells it reads, and the component's condition stays
UNKNOWN with high U_O. This is the ch25 premise, made true by construction rather than by luck of the heading.

`side: lane` is added as its own value for completeness and is the explicit near-side case; `far` and `near`
keep their exact current meaning, so `straight_pipeline`, the I5 scenarios, I7 and the flagship run are
unchanged.

### 2.2 A physical occluder means not every off-lane view resolves the ambiguity

A single unregistered structure is placed alongside the target segment on the off-lane side: a longitudinal
rack panel (the subsea case is a cable tray, a pipeline bundle neighbour, a protective cover or a clamp
flange), plus, when the geometry allows it, two vertical posts down to the seabed. It is a Twin2S entity only:
it is not in the asset registry, it has no Twin2T or Twin2E state, and it appears in no Observation. The
deployment can learn about it only through its own geometric sensors, exactly as for the I5 lane obstacle.

Its position is sampled per world, in the frame of the target segment:

| parameter | range | meaning |
|---|---|---|
| `azimuth_offset_deg` | see `configs/sim/i4_occluded.yaml` | angle of the rack around the pipe axis, measured from the patch direction |
| `standoff_m` | " | radial gap between the pipe surface and the rack's near face |
| `half_width_m` | " | tangential half-width of the panel |
| `half_length_fraction` | " | axial half-length as a fraction of the target segment length |

The panel casts an angular shadow of roughly `atan(half_width / standoff)` around its own azimuth. Because the
azimuth offset is sampled, in some worlds the shadow covers the part of the patch a generic ring-walking route
would look at, and in other worlds it does not. That is the source of the required variation, and it is
geometric: nothing in the sampling refers to any planner, to any candidate index, or to which planner wins.

### 2.3 What varies between worlds

1. **Occlusion direction**: the rack azimuth relative to the patch, plus its standoff, width and axial length.
2. **Defect severity**: corrosion depth and crack length are sampled per world. In the current family every
   world carries the identical defect (0.006 m and 0.08 m), which removes a whole axis of difficulty and makes
   ties more likely.
3. **Patch geometry**: tilt above the horizontal, angular half-width and axial fraction.
4. **Worlds the fixed route can solve**: no world is built to be unsolvable by the fixed route. With the rack
   azimuth sampled over a wide interval, a share of the worlds leave the generic route's views clear. Section 7
   reports the measured share on development. If that share were zero the family would be rejected and
   redesigned, because a family only MCBR can solve proves nothing.

### 2.4 What MCBR is NOT told

* The occluder is not in the asset registry, not in the mission context, and carries no metadata.
* No candidate, no `SensorOption`, no `SpatialSupport` and no `InformationNeed` field names the defect, the
  patch, the occluder or the hidden condition. The planner's only geometric input remains the injected
  `is_free` / `predicted_visibility` / `navigation_cost` callables and (for the predictive rankers) the
  Model2S belief map plus the Model2T coverage and datasheet. Model2S learns about the rack the same way it
  learns about a rock: by ray-casting its own geometric observations.
* `tests/leakage` must stay green, including the static import guard for `conrad.active`,
  `conrad.orchestration` and `conrad.decision`.

### 2.5 Why the baselines stay honest

* **Fixed nominal route** (`A-B1_fixed_inspection`): the lane transit plus a deterministic walk of the
  candidate ring in generation order, one stride per view taken. It is the same rule as today. It is a
  reasonable inspection route for a pipe: it circles the component at valid standoffs and both elevations. It
  is not modified, not reordered and not de-tuned for this family, and the occluder is placed by geometry, not
  by candidate index.
* **Coverage-only** (`A-B2_coverage`) and **random** (`A-B0_random`) are unchanged.
* Every planner receives the identical `PlanningRequest` (including the predictive model when one exists),
  the identical candidate generator, the identical feasibility filter and the identical budgets.

## 3. Why this is the intended Conrad use case

Conrad's stated deployment is autonomous inspection of infrastructure whose critical surfaces are not visible
from a nominal transit. Subsea pipelines are inspected from a survey lane offset to one side; the far side, the
underside, the surfaces behind clamps, anodes, spans, supports, cable trays and neighbouring bundles are the
ones that are not read by the lane pass, and they are exactly where the operator needs a condition claim. A
transit pass that already sees everything does not need a view planner; the question "which extra view is worth
taking" only exists when part of the asset is hidden, when several extra views are possible, and when they are
not equally informative. That is the situation this family builds, and it is the only situation in which the I4
criterion "beats fixed views on actual hidden-state reconstruction" is a meaningful test rather than a
tautology about a route that happened to already look at the defect.

Choosing this family is legitimate only under the conditions the remediation audit set, and all of them hold:
the family is declared and pinned before any run on its held-out seeds; the design is the same for every
planner; and the old `straight_pipeline` result is reported next to the new one rather than replaced. No result
on any world of this family existed when the design above was fixed.

## 4. Splits

`configs/eval/partitions_i4_occluded.yaml`, digest-pinned in `conrad/evaluation/partitions.py`
(`I4_OCCLUDED_PARTITIONS_SHA256`, domain `i4_occluded`). Seeds come from the free 8 000 000 block; the file
records every range reserved by another partition file or agent and the loader refuses an overlap.

| partition | seeds | purpose allowed |
|---|---|---|
| development | 8000000-8000023 | design, tuning |
| validation | 8000100-8000123 | design, tuning, selection |
| final_test | 8000200-8000219 | final evaluation only |
| ood_test | 8000300-8000311 | final evaluation only |

The OOD split holds out a structural family: its worlds are built on `bent_pipeline` geometry with a wider
occluder standoff range, so the held-out worlds differ in geometry family, not only in seed.

## 5. Planner baselines compared

All ten take the identical `PlanningRequest` and run under matched budgets (same mission duration and time
budget, same `max_plans_per_need` observation budget, same single structural sensor and sensor capability, same
candidate generator, same feasibility filter; travel allowance, energy and travel are measured on the same
scale for every planner and reported).

| id | rule |
|---|---|
| `A-B0_random` | uniform random over feasible candidates |
| `A-B1_fixed_inspection` | fixed nominal route (candidate ring in generation order) |
| `A-B2_coverage` | visibility x novelty (coverage-only) |
| `A-B4_geometric_nbv` | geometric next best view: visibility x novelty discounted by travel |
| `A-B5_entropy_nbv` | entropy NBV as it already existed: visibility x total belief uncertainty x novelty |
| `A-B5b_entropy_nbv_predictive` | uncertainty NBV that is actually independent (see below) |
| `A-B7_uncertainty_nbv` | max-channel uncertainty NBV minus cost |
| `A-B6b_bayes_eig` | conventional Bayesian expected information gain (expected entropy reduction) |
| `PRODUCTION` | the frozen production planner under the MCBR contract |
| `A-B10_mcbr_full` | the old MCBR (diagnostic arm, reported, never a gate comparator) |

`A-B5_entropy_nbv` multiplies a candidate-INDEPENDENT quantity (the knowledge gap's own uncertainty level, one
number for the whole need) by visibility and novelty, so in every integrated mission it produces exactly the
same ranking as `A-B2_coverage`. Reporting it as an independent entropy baseline would be misleading, so it is
kept for continuity and `A-B5b_entropy_nbv_predictive` is added: the same NBV shape, with the uncertainty term
taken from the candidate itself (its predicted entropy reduction of the belief, falling back to the analytic
mission value when no predictive belief exists). That makes it distinct from `A-B6b` (value, no novelty) and
from `A-B2` (novelty, no uncertainty). Both properties are asserted in `tests/unit/active/test_mcbr.py`.

MCBR is a contract, not an obligation to keep one ranker. If a conventional EIG or NBV method wins on
development and validation, that method becomes the production MCBR implementation and is frozen as such.

## 6. What will be measured (declared before the final run)

Per world, per planner, paired on the world:

| metric | direction | definition |
|---|---|---|
| `hidden_state_error_improvement` | higher | 1 - mean over corrosion depth and crack length of (final absolute belief error / prior-mean absolute error) against evaluation-only truth. UNKNOWN at the end scores as the prior mean. |
| `hidden_state_error_final` | lower | the same ratio |
| `corrosion_abs_error_m`, `crack_abs_error_m` | lower | raw hidden-state reconstruction errors |
| `mission_error_reduction` | higher | mission-relevant belief error reduction (the target segment is the mission-relevant belief) |
| `info_per_time` | higher | improvement per second of mission time |
| `info_per_kj` | higher | improvement per kJ of measured vehicle energy |
| `travel_m` | lower | measured travelled distance |
| `observations` | lower | accepted MCBR inspection goals |
| `redundant_observations` | lower | accepted inspection goals that produced no new direct target revision |
| `collisions` | lower | measured vehicle collisions |
| `mission_success` | higher | target OBSERVED and the critical finding delivered |
| `target_observed` | higher | target condition OBSERVED at the end |

Decision rule, pre-declared: the statistical unit is one world. The benefit of the selected planner over a
baseline is compared with a paired percentile bootstrap over worlds (95 %, 4000 resamples,
`conrad.evaluation.metrics.paired_seed_comparison`). "Beats" means the lower bound of the 95 % CI is above 0.
The ch25 criteria map as:

* "critical structure partly hidden -> ... -> belief improves": the selected planner's mean
  `hidden_state_error_improvement` is above 0 and at least half of the worlds that still needed a view produced
  an accepted inspection goal that yielded a new direct target revision;
* "beats fixed views on actual hidden-state reconstruction": `hidden_state_error_improvement` vs
  `A-B1_fixed_inspection`;
* "beats random views": the same vs `A-B0_random`;
* "beats coverage-only": the same vs `A-B2_coverage`;
* "beats simple views on information/time/energy": `info_per_time` AND `info_per_kj` vs each of those three.

ch25 leaves the numeric margin OPEN. No margin is invented here. The directional criterion above is applied,
and every comparison is reported with its paired CI whatever the sign, including the ones that fail.

## 7. Development results

`ACTIVE-MCBR-E005-DESIGN`, 24 development worlds (8000000-8000023), purpose design, partition digest
`d929beb67c051ea1`, on the code that includes Model2T iteration 4. Run once. Means over worlds:

| planner | hidden-state err. impr. | target OBSERVED | obs | travel m | energy J | info/kJ | collisions |
|---|---|---|---|---|---|---|---|
| `A-B6b_bayes_eig` | 0.394 | 0.50 | 3.00 | 24.3 | 17252 | 0.0217 | 0.21 |
| `PRODUCTION` | 0.365 | 0.50 | 3.08 | 24.6 | 17517 | 0.0215 | 0.33 |
| `A-B5b_entropy_nbv_predictive` | 0.350 | 0.46 | 3.08 | 26.5 | 19350 | 0.0190 | 0.50 |
| `A-B2_coverage` | 0.265 | 0.38 | 3.50 | 29.0 | 20570 | 0.0135 | 0.00 |
| `A-B5_entropy_nbv` | 0.265 | 0.38 | 3.50 | 29.0 | 20570 | 0.0135 | 0.00 |
| `A-B0_random` | 0.225 | 0.29 | 3.58 | 30.8 | 22445 | 0.0112 | 0.21 |
| `A-B7_uncertainty_nbv` | 0.215 | 0.29 | 2.92 | 24.8 | 16554 | 0.0124 | 0.04 |
| `A-B4_geometric_nbv` | 0.183 | 0.25 | 3.33 | 27.6 | 19355 | 0.0108 | 0.00 |
| `A-B10_mcbr_full` | 0.144 | 0.21 | 0.58 | 20.0 | 13263 | 0.0091 | 0.00 |
| `A-B1_fixed_inspection` | 0.064 | 0.08 | 3.62 | 27.5 | 21472 | 0.0036 | 0.00 |

Paired benefit of `PRODUCTION` (95 % paired percentile bootstrap over the 24 worlds, 4000 resamples):

| vs | hidden-state err. impr. | info/time | info/kJ |
|---|---|---|---|
| fixed | **+0.301 [+0.123, +0.470]** | **+0.0030 [+0.0012, +0.0047]** | **+0.0178 [+0.0076, +0.0277]** |
| random | +0.140 [-0.051, +0.325] | +0.0014 [-0.0005, +0.0032] | +0.0102 [+0.0002, +0.0203] |
| coverage | +0.100 [-0.070, +0.257] | +0.0010 [-0.0007, +0.0026] | +0.0080 [-0.0012, +0.0165] |

What this says about the family, all of it decided before the run:

1. **The premise holds.** The fixed nominal route resolves the hidden defect in 2 of 24 worlds (`target
   OBSERVED` 0.08) against 12 of 24 for the two best information-seeking planners. In `straight_pipeline` the
   same fixed route reached 1.00. The gate criterion "beats fixed views on actual hidden-state
   reconstruction" is now a real test rather than a tautology.
2. **The family is not a straw man and not only-MCBR-solvable.** Coverage-only reaches 0.38 and random 0.29,
   so the competitors do find the window in a fair share of worlds. No world is unsolvable: the unit test
   `test_the_panels_block_most_of_the_candidate_ring_but_never_all_of_it` asserts a resolving candidate exists
   in every development world, and 6 of the 24 worlds were closed by no planner within the 4-observation
   budget, which is difficulty, not impossibility.
3. **`A-B5_entropy_nbv` is exactly `A-B2_coverage`**, as predicted from the code: identical means on every
   metric. That is why `A-B5b` was added.
4. **Power is the binding constraint, not the effect.** The outcome is close to all-or-nothing per world (the
   target is either OBSERVED or not), so at n = 24 the paired CI half width against coverage is about 0.16
   while the effect is about 0.10. On development no arm clears the predeclared eligibility rule of beating
   fixed AND random AND coverage with a CI lower bound above 0; every arm clears it against fixed alone. The
   final split has 20 worlds, which is less power again. This is reported to the integrator as a property of
   the declared design, not fixed after the fact by moving a threshold or a split.
5. **Cost of the information-seeking arms**: they use less energy and travel than coverage and random, but
   they collide more (0.21 to 0.50 per episode against 0.00 for coverage, fixed and geometric NBV). The
   occluding structure is unregistered, so the belief map only learns it by sensing, and a planner that flies
   into the window before the panels are mapped can clip one. This is a real cost and it is reported.

## 8. Validation results and selection

`ACTIVE-MCBR-E005-SELECTION`, 24 validation worlds (8000100-8000123), purpose selection, same partition file
and digest, same code baseline. Run once. Means over worlds:

| planner | hidden-state err. impr. | target OBSERVED | obs | travel m | energy J | info/kJ | collisions |
|---|---|---|---|---|---|---|---|
| `A-B5b_entropy_nbv_predictive` | 0.424 | 0.50 | 3.04 | 27.7 | 19773 | 0.0231 | 0.21 |
| `PRODUCTION` | 0.406 | 0.50 | 2.75 | 25.8 | 17979 | 0.0233 | 0.25 |
| `A-B6b_bayes_eig` | 0.361 | 0.42 | 3.17 | 24.6 | 17747 | 0.0196 | 0.25 |
| `A-B4_geometric_nbv` | 0.351 | 0.42 | 3.00 | 27.9 | 19206 | 0.0204 | 0.08 |
| `A-B10_mcbr_full` | 0.288 | 0.33 | 0.83 | 21.8 | 14572 | 0.0163 | 0.04 |
| `A-B7_uncertainty_nbv` | 0.280 | 0.33 | 2.67 | 25.4 | 16931 | 0.0155 | 0.08 |
| `A-B0_random` | 0.279 | 0.38 | 3.29 | 28.6 | 21763 | 0.0141 | 0.17 |
| `A-B2_coverage` | 0.240 | 0.29 | 3.33 | 29.2 | 20833 | 0.0126 | 0.25 |
| `A-B5_entropy_nbv` | 0.240 | 0.29 | 3.33 | 29.2 | 20833 | 0.0126 | 0.25 |
| `A-B1_fixed_inspection` | 0.000 | 0.00 | 3.67 | 26.5 | 20288 | 0.0000 | 0.04 |

Paired benefit on `hidden_state_error_improvement` against the three required comparators:

| arm | vs fixed | vs random | vs coverage |
|---|---|---|---|
| `A-B5b_entropy_nbv_predictive` | **+0.424 [+0.252, +0.591]** | +0.145 [-0.052, +0.346] | **+0.184 [+0.022, +0.351]** |
| `PRODUCTION` | **+0.406 [+0.242, +0.578]** | +0.127 [-0.002, +0.271] | +0.166 [-0.001, +0.343] |
| `A-B6b_bayes_eig` | **+0.361 [+0.206, +0.539]** | +0.082 [-0.138, +0.298] | +0.121 [-0.059, +0.302] |
| `A-B4_geometric_nbv` | **+0.351 [+0.187, +0.523]** | +0.072 [-0.102, +0.249] | **+0.111 [+0.010, +0.234]** |
| `A-B10_mcbr_full` | **+0.288 [+0.138, +0.462]** | +0.009 [-0.142, +0.168] | +0.048 [-0.122, +0.222] |
| `A-B7_uncertainty_nbv` | **+0.280 [+0.135, +0.450]** | +0.001 [-0.148, +0.155] | +0.040 [-0.126, +0.203] |

### 8.1 Selection outcome: no change, and the reason

The rule was declared in `configs/eval/active_mcbr_e005_selection.yaml` before the run and is applied by
`select_validation_winner` (unit-tested in `tests/unit/evaluation/test_i4_occluded_selection.py`).

**No arm is eligible.** Every arm beats the fixed nominal route with a CI far above 0. Two arms beat
coverage-only. **No arm beats random views** with a CI above 0. The declared rule for that case is to record
the honest conclusion rather than manufacture a selection, and that is what is recorded.

The incumbent production planner is therefore **retained unchanged**: `V-bayes_eig_ratio` (conventional
Bayesian EIG per unit cost) with the frozen mission predictive model. The brief's instruction to replace a
losing production ranker with a conventional EIG or NBV method does not apply here, because the incumbent is
not losing:

* it is 2nd of 10 on development (0.365) and 2nd of 10 on validation (0.406), and best or joint best on
  information per kJ on both splits;
* the nominally leading arm `A-B5b` does not beat it on either split, and the sign of the difference flips:
  `PRODUCTION - A-B5b` is +0.014 [-0.148, +0.158] on development and -0.018 [-0.183, +0.150] on validation.

Replacing a production config on a sign that flips between splits and a CI that spans zero would be churn, not
selection. Frozen as `configs/active/mcbr_frozen_v3.yaml`: the `planner` and `mission_predictive` sections are
byte-identical to v2, so `config_digest` stays
`8eca16cc896e560b26cbe9814828fe9f75801d843e0e38bedb5442299805d903` and
`mission_predictive_digest` stays `2fa5b75d6e9618c0a1274eff284d014a30df4c0ba17a8d83969c2c8609d41431`, which
keeps the ACTIVE-MCBR-E002 and E004 artifacts valid. v3 adds the family digest, the I4 partition digest and
this selection round.

### 8.2 The honest read, before the final run

* The family does what it was built to do. The fixed nominal route scores 0.000 on validation and 0.064 on
  development, against 0.41 and 0.37 for the production planner. On `straight_pipeline` the same fixed route
  was unbeaten. The gate premise "critical structure partly hidden" is now real.
* "Beats fixed views on actual hidden-state reconstruction" and "beats simple views on information per time
  and per energy" (against fixed) pass comfortably on both non-held-out splits.
* "Beats random views" is the one that does not clear the bar, on either split, for any arm. The point
  estimate is positive on both (+0.140 development, +0.127 validation) and the direction is consistent, but
  the interval includes 0.
* The cause is power, not a missing effect. The per-world outcome is close to all-or-nothing, so at n = 24 the
  paired CI half width against random is about 0.20. The declared final split has 20 worlds, which gives less
  power again. A final run on 20 worlds is very likely to reproduce "beats fixed, not random" and therefore to
  record I4 as FAIL on the random-views criterion.
* That is reported to the integrator as a property of the declared design. The split is digest-pinned and is
  not being enlarged after seeing these numbers, and no threshold is being moved.

## 9. Formal run 1 (Unity, 2026-09-20): gate I4 = FAIL

`configs/eval/i4_occluded_unity.yaml`, final split 8000200-8000219 (20 worlds), declared reduced arm set of
four (PRODUCTION, fixed, random, coverage), 80 sequential flights at a measured 82 s each, Twin2E off as in
the previous formal I4 run. Recorded in `artifacts/gates/I4/unity_i4_occluded_results.json` and
`artifacts/gates/I4/evidence_formal.json`. Three of five criteria PASS:

| criterion | result |
|---|---|
| closed loop: hidden -> uncertain -> MCBR view -> new evidence -> belief improves | PASS |
| beats fixed views on actual hidden-state reconstruction | PASS, +0.232 [+0.081, +0.399] |
| beats random views | PASS, +0.206 [+0.030, +0.382] |
| beats coverage-only | FAIL, +0.163 [-0.045, +0.369] |
| beats simple views on information/time/energy | FAIL (the coverage half) |

Means over the 20 worlds:

| arm | hidden-state err. impr. | obs | redundant | target OBSERVED | patch seen | collisions |
|---|---|---|---|---|---|---|
| PRODUCTION | 0.392 | 1.95 | 1.45 | 0.50 | 0.467 | 0.00 |
| A-B2_coverage | 0.229 | 2.90 | 2.55 | 0.30 | 0.282 | 0.00 |
| A-B0_random | 0.186 | 2.45 | 2.15 | 0.25 | 0.322 | 0.35 |
| A-B1_fixed_inspection | 0.161 | 2.35 | 2.05 | 0.25 | 0.222 | 0.05 |

### 9.1 Why coverage-only does so well here

Both mechanisms the question offers are present, and the measurement separates them.

**MCBR is not spending its budget.** The observation budget is `max_plans_per_need` 4. PRODUCTION averages
1.95 accepted observations per world against coverage's 2.90, and takes 0 or 1 observation in 6 of the 20
worlds. Counting planner calls rather than accepted plans: PRODUCTION is asked 2.6 times per world and returns
39 PLAN / 2 NEED_SATISFIED / 11 NO_FEASIBLE_OBSERVATION; coverage is asked 3.8 times and returns 58 PLAN /
6 NEED_SATISFIED / 12 NO_FEASIBLE_OBSERVATION. The gap is in how many planning cycles each arm fits inside the
100 s mission, not in refusals: PRODUCTION picks the window, which is on the far side of the structure, so
each of its views costs more mission time to reach and fewer cycles fit. The budget is matched on observation
count and on mission time, but the binding constraint is mission time, and the two arms spend it differently.

**Coverage is sweeping the window by volume, not by aim.** Per view, PRODUCTION is clearly better: mean patch
visible fraction 0.467 against coverage's 0.282, and it ends with the target OBSERVED in 10 of 20 worlds
against coverage's 6. Coverage does not aim at the window; it walks the candidate ring for novelty and crosses
the window in some worlds because it takes more shots. The per-world split is the clearest statement of this:
PRODUCTION observes the target and coverage does not in 6 worlds, coverage does and PRODUCTION does not in 2,
both do in 4, neither does in 8.

So the honest summary of run 1 is: MCBR chooses better views and takes fewer of them, and a systematic sweep
closes most of that gap by volume inside the same mission time. That is a real finding about the mechanism and
it does not depend on how the replication turns out. It also names the concrete follow-up, which is NOT part
of this gate: the cost of reaching a chosen view is not reflected in how many planning cycles the runtime
grants, so an arm that picks far, high-value views is charged twice.

## 10. Pre-registered replication (run 2), declared before the first flight

Run 1 recorded FAIL. The power warning in section 8.2 was written before run 1, not after it, so the response
is more worlds under an identical design rather than a reinterpretation of the same ones.

Declared in `configs/eval/i4_occluded_unity_rep2.yaml` and pinned in
`configs/eval/partitions_i4_occluded_v2.yaml` (digest
`ad4f97312ea19eaeef47995b2d80299c4b9cd49a59b448c365bb545c5f363fb6`, domain `i4_occluded_v2`) BEFORE any
flight on it:

* **Worlds**: fresh 8001000-8001059, 60 worlds, disjoint from run 1's spent 20 and from every other partition
  file. Read exactly once. Sized from run 1's measured 82 s per flight so 120 flights fit in about 2.7 h.
* **Arms**: PRODUCTION and coverage-only only. The fixed-views and random-views criteria PASSED in run 1 on
  their own held-out worlds and are not re-decided. Re-running a criterion that already passed, on fresh
  worlds, until it passes again is exactly the shopping this declaration exists to prevent. The reduction and
  its reason are recorded the same way the four-arm reduction was, and the harness asserts it.
* **Unchanged**: the frozen planner `configs/active/mcbr_frozen_v3.yaml` (config digest
  `8eca16cc896e560b26cbe9814828fe9f75801d843e0e38bedb5442299805d903`), the family digest
  `f77c5dbb7d665e3740e76cd4dfd97f37d0f0060bc2b336af6391ff0758877bdd`, the scenario, the candidate generator,
  the feasibility filter, the budgets, the metric, and the bootstrap with its resample count and seed.
* **Decides**: "beats coverage-only" on `hidden_state_error_improvement`, and the coverage half of "beats
  simple views on information/time/energy" on `info_per_time` and `info_per_kj`.
* **Rule**: unchanged. PRODUCTION beats coverage on a metric iff the lower bound of the 95 % paired
  percentile-bootstrap CI over worlds is above 0.
* **Stopping rule, declared now**: if the interval still includes 0, gate I4 stays FAIL and the investigation
  stops. Two honest attempts are enough, and "MCBR does not beat a systematic coverage sweep in this family"
  is a legitimate result. The split is not enlarged, no threshold is moved, no baseline is dropped and nothing
  is re-run after the outcome is seen. A tie is reported as a tie. Both runs are reported side by side,
  never the better one alone.

### 10.1 Result of run 2: a tie, and gate I4 stays FAIL

Run once, 60 worlds, 120 sequential flights, 2 h 20 m. Recorded in
`artifacts/gates/I4/unity_i4_occluded_rep2_results.json`; both runs side by side in
`artifacts/gates/I4/i4_replication_summary.json`.

| metric | benefit of PRODUCTION over coverage-only | verdict |
|---|---|---|
| `hidden_state_error_improvement` | +0.0094 [-0.0828, +0.0979] | not beaten |
| `info_per_time` | +0.00009 [-0.00083, +0.00098] | not beaten |
| `info_per_kj` | +0.0036 [-0.0179, +0.0244] | not beaten |

| arm | hidden-state err. impr. | obs | redundant | target OBSERVED | patch seen | collisions |
|---|---|---|---|---|---|---|
| PRODUCTION | 0.365 | 2.65 | 2.17 | 0.43 | 0.456 | 0.03 |
| A-B2_coverage | 0.355 | 2.77 | 2.30 | 0.45 | 0.409 | 0.07 |

Target OBSERVED, per world: both 21, PRODUCTION only 5, coverage only 6, neither 28. The benefit is positive
in 28 % of worlds.

**This is a tie, not an underpowered interval, and that distinction matters.** At n = 60 the interval on the
primary metric is about +-0.09 and it is centred on +0.009. Run 1's point estimate of +0.163 was not
reproduced; the replication did not find a smaller effect than expected, it found no effect. Under the
stopping rule declared before the run, the investigation stops here.

**Gate I4 = FAIL.** Three criteria PASS (closed loop, beats fixed views, beats random views, all decided on
run 1's own held-out worlds); "beats coverage-only" and the coverage half of "beats simple views on
information/time/energy" FAIL. The recorded verdict is unchanged from run 1; the replication removes the
possibility that it was a sample-size artefact.

The result worth stating plainly: **on a family where the nominal route genuinely cannot see the defect, the
production MCBR planner clearly beats a fixed route and random views, and does not beat a systematic coverage
sweep.** That is a legitimate scientific result, not a gap to be closed by another run.

### 10.2 Correction to section 9.1: the run 1 diagnosis did not replicate

Section 9.1 was written from run 1 (n = 20) and two of its three claims do not survive run 2 (n = 60). The
correction is recorded rather than the original quietly revised.

| quantity | run 1 (n = 20) | run 2 (n = 60) |
|---|---|---|
| PRODUCTION observations | 1.95 | 2.65 |
| coverage observations | 2.90 | 2.77 |
| PRODUCTION patch visible fraction | 0.467 | 0.456 |
| coverage patch visible fraction | 0.282 | 0.409 |
| PRODUCTION hidden-state err. impr. | 0.392 | 0.365 |
| coverage hidden-state err. impr. | 0.229 | 0.355 |

1. **"MCBR is not spending its budget" does NOT replicate.** The observation gap was 0.95 views per world in
   run 1 and 0.12 in run 2. PRODUCTION now uses 2.65 of its 4 allowed observations against coverage's 2.77.
   The run 1 gap was a small-sample artefact.
2. **"Coverage sweeps the window by volume rather than by aim" is much weaker than it looked.** Coverage's
   per-view patch visibility rose from 0.282 to 0.409, close to PRODUCTION's 0.456. Coverage is not merely
   getting lucky with more shots; on 60 worlds its views are nearly as good.
3. **PRODUCTION is the stable arm across both runs** (0.392 then 0.365; patch 0.467 then 0.456). Coverage is
   what moved (0.229 then 0.355). Run 1's 20-world coverage sample was on the low side, which is what produced
   run 1's +0.163 point estimate and the apparent near-miss.

So the honest mechanism statement after both runs is narrower than section 9.1 claimed: MCBR's chosen views
are slightly better per view, it takes about as many views as coverage does, and the two arms end up
resolving the hidden defect about equally often. The follow-up named in section 9.1, that travel cost is not
reflected in how many planning cycles the runtime grants, is no longer supported by the data and should not be
pursued on the strength of this evidence.

### 10.3 OOD split: not run

The held-out OOD split 8000300-8000311 (`I4-OCCLUDED-OOD-UNITY`, bent pipeline geometry) was declared out of
the pass decision from the start and is still unrun and unread. With the gate decided it adds no decision
value, so it was left for the integrator to sequence. It is 48 flights, about 1.1 h, and the harness is in
place.
