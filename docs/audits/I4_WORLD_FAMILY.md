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
