# MCBR V3 failure analysis: why better views did not become a better belief (2026-09-20)

All data is SYNTHETIC_ONLY. Nothing here is real-data or physical validation.

## 0. What this document is, and what it may not be used for

Gate I4 is decided and its verdict stands unchanged. On the world family `ACTIVE_INSPECTION_OCCLUDED_V1` the
frozen production MCBR planner **beats a fixed inspection route, beats random views, and does not beat a
systematic coverage sweep**. Formal run 1 (20 worlds, 8000200-8000219) and the pre-registered replication run
2 (60 worlds, 8001000-8001059) are recorded in `artifacts/gates/I4/` and read out in
`docs/audits/I4_WORLD_FAMILY.md` sections 9 and 10. The primary replication result is
`hidden_state_error_improvement` +0.0094 with a 95 % paired CI of [-0.0828, +0.0979], means 0.365 against
0.355: a tie, and formal I4 = FAIL.

This document is a **post-hoc analysis of those already-spent traces**. Every world it reads has already been
consumed as evidence, so:

* it may falsify a stated mechanism, and it may motivate a hypothesis for a possible V4;
* it may **not** select, calibrate, tune or justify any parameter, ranker, threshold or world distribution.
  Nothing in it feeds back into a planner. A V4 mechanism, if one is built, is designed on the fresh
  development worlds of `configs/eval/partitions_i4_mcbr_v4.yaml` and on nothing else.
* it does not reinterpret, soften or re-score the gate result. Where a number here differs in framing from
  section 9.1 or 10.2 of `I4_WORLD_FAMILY.md`, the gate artifact is the record and this is commentary.

The companion measurement that decides whether a V4 is worth building at all is `ACTIVE-MCBR-E006` (section
10 and `docs/audits/MCBR_V4_ORACLE_HEADROOM.md`), which runs on **fresh development worlds** and never
touches a held-out split.

## 1. Two provenance facts, recorded and not yet changed

1. `conrad/active/production.py` sets `FROZEN_PATH = configs/active/mcbr_frozen_v2.yaml`, while the audit
   trail, the run-2 harness and `artifacts/gates/I4/unity_i4_occluded_rep2_results.json` all name
   `configs/active/mcbr_frozen_v3.yaml`. The two files have byte-identical `planner` and `mission_predictive`
   sections (`config_digest 8eca16cc...5d903`, `mission_predictive_digest 2fa5b75d...41431`), so the planner
   that flew is the planner the audit describes and no result is affected. The pointer is still wrong and is
   left alone here: changing it would change what `load_frozen()` reports inside the recorded artifacts.
2. `configs/active/mcbr_frozen_v3.yaml` has `mission_predictive.config.read_channel: false`. That is the
   DEVELOPMENT selection recorded in `MCBR_REEVALUATION.md` section 8.1, and it is the subject of Limitation C
   below. It is not changed here either.

## 2. Method

`scripts/analyze_i4_v3_failure.py` reads the 120 finished run-2 bundles (60 worlds x 2 arms) and the 80
run-1 bundles under `artifacts/unity/gate_runs/`. Per bundle it reads the mission artifacts (accepted
inspection goals, the MCBR candidate tables, runtime metrics), the belief revisions of the target component
from the run's own SQLite, and the evaluation-only `truth/truth_record.json`, which carries the sampled
defect, the patch centre and direction, and the exact geometry of the unregistered rack panels
(`view_occlusion`). Surface cells are rebuilt exactly as Model2T partitions the surveyed design capsule
(`cell_m` 0.5, 8 sectors), so "which cells a view sees" is on the same grid the belief uses.

Outputs: `artifacts/analysis/MCBR_V3_FAILURE/{per_world,per_view,candidates,redundancy,summary}_run{1,2}.json`
and the per-world table `per_world_table_run2.md` (appendix A).

Three definitions are used throughout:

| term | definition |
|---|---|
| commanded patch fraction | fraction of the defect patch cells that the COMMANDED view pose could see: facing, in sensor range, and not blocked by the world's real rack panels |
| realised patch fraction | the same quantity computed at the vehicle's point of closest approach to that commanded pose while that goal was active |
| defect read | the view window contains at least one DIRECT belief revision of the target component |

**Validation of the pipeline.** Recomputed from the bundles, the arm means reproduce the recorded gate
numbers: run 2 `hidden_state_error_improvement` 0.3648 / 0.3553 (recorded 0.365 / 0.355), observations
2.65 / 2.77 (recorded 2.65 / 2.77), patch seen 0.456 / 0.409 (recorded 0.456 / 0.409); run 1 0.3922 / 0.2287 /
0.1863 / 0.1607 for PRODUCTION / coverage / random / fixed (recorded 0.392 / 0.229 / 0.186 / 0.161). The
paired bootstrap reproduces +0.0094 [-0.0845, +0.1019] against the recorded +0.0094 [-0.0828, +0.0979].

## 3. The decomposition that explains the tie

The per-world outcome on this family is not close to all-or-nothing, it is **exactly** all-or-nothing. Over
all 120 run-2 missions, a world whose defect was read scored a mean `hidden_state_error_improvement` of 0.78,
and a world whose defect was not read scored **0.000, with no intermediate case**:

| arm | worlds where the defect was read | mean when read | mean when not read | overall mean |
|---|---|---|---|---|
| PRODUCTION | 28 / 60 | 0.7816 | 0.0000 | 0.3648 |
| A-B2_coverage | 28 / 60 | 0.7614 | 0.0000 | 0.3553 |

so the entire paired difference is

    (28/60) x (0.7816 - 0.7614) = +0.0094

which is the recorded gate number to four decimals. **The two arms read the defect in exactly the same number
of worlds.** The whole of MCBR's measured benefit is the slightly better reading it gets in the worlds where
both succeed (crack absolute error 0.0474 m against 0.0519 m), and none of it is a higher success rate.

Per-world agreement: both arms read the defect in 21 worlds, neither in 25, PRODUCTION only in 7, coverage
only in 7. That is a symmetric tie, not a small effect hidden by noise.

## 4. Per-world and per-arm numbers

Means over the 60 run-2 worlds, with paired 95 % percentile-bootstrap CIs over worlds (4000 resamples,
PRODUCTION minus coverage):

| quantity | PRODUCTION | coverage | paired difference |
|---|---|---|---|
| `hidden_state_error_improvement` | 0.3648 | 0.3553 | +0.0094 [-0.0845, +0.1019] |
| defect read (0/1) | 0.467 | 0.467 | 0.000 [-0.133, +0.117] |
| target condition OBSERVED | 0.433 | 0.450 | -0.017 [-0.133, +0.100] |
| accepted inspection views | 2.65 | 2.77 | -0.117 [-0.400, +0.167] |
| commanded patch fraction, per view | 0.624 | 0.496 | **+0.091 [+0.027, +0.156]** |
| realised patch fraction, per view | 0.407 | 0.369 | +0.045 [-0.039, +0.130] |
| truth patch max visible fraction | 0.456 | 0.409 | +0.048 [-0.050, +0.145] |
| read-surface proxy `1 - U_O` | 0.446 | 0.462 | -0.016 [-0.127, +0.093] |
| mean angle between chosen view directions | 24.0 deg | 56.5 deg | |
| new surface cells per view (1 - overlap) | 0.433 | 0.480 | |
| overlap of a view with all earlier views | 0.567 | 0.520 | |
| corrosion absolute error | 0.00366 m | 0.00347 m | |
| crack absolute error | 0.0474 m | 0.0519 m | |
| final `U_O` / `U_C` / `U_E` | 0.554 / 0.046 / 0.200 | 0.538 / 0.039 / 0.200 | |
| travel | 24.6 m | 26.2 m | |
| energy | 4042 J | 4192 J | |
| mission time | 100 s | 100 s | |
| collisions | 0.033 | 0.067 | |

The one comparison whose CI excludes 0 is the **commanded** patch fraction: MCBR genuinely aims better. It
loses half of that advantage on the way to the realised pose (+0.091 commanded, +0.045 realised), and all of
the rest by the time it reaches the belief (0.000 on defect read).

Per-world detail for every world and both arms, including the selected azimuths and elevations, the predicted
value of the first chosen view, the detection view index and the per-world errors, is in appendix A and in
`artifacts/analysis/MCBR_V3_FAILURE/per_world_run2.json`. Per-view records, including the closest approach
and the direct revisions in each view window, are in `per_view_run2.json`.

### 4.1 Where the chain breaks

| step | PRODUCTION | coverage |
|---|---|---|
| views whose COMMANDED pose sees more than half the patch | 93 of 159 | 75 of 166 |
| of those, views that produced a direct belief revision | 25.8 % | 21.3 % |
| views whose REALISED pose sees more than half the patch | 60 of 159 | 59 of 166 |
| of those, views that produced a direct belief revision | 45.0 % | 44.1 % |
| views where the vehicle came within 0.5 m of the commanded pose | 54.7 % | 54.2 % |
| median closest approach | 0.14 m | 0.15 m |
| worlds that used the full 4-view budget | 27 of 60 | 31 of 60 |
| planner calls per world | 3.18 | 3.38 |

Per-world rank correlation with the outcome: realised patch fraction 0.71 (PRODUCTION) and 0.68 (coverage),
commanded patch fraction 0.54 and 0.66. **What the view was aimed at predicts the outcome less well than
where the vehicle actually got to.** The distribution of the closest approach is bimodal: when a view is
reached it is reached to within 0.15 m, and when it is not reached it is missed by metres, because the mission
ends or replans first. Roughly 45 % of chosen views are never flown.

## 5. The four questions, answered with numbers

### 5.1 How often does MCBR detect the defect earlier than coverage?

It does not. Of the 21 worlds where both arms read the defect, PRODUCTION was first in 7, coverage was first
in 9, and the two were simultaneous in 5. Mean time of first direct revision: 61.7 s (PRODUCTION) against
60.2 s (coverage), out of a 100 s mission. Counting worlds rather than times: PRODUCTION only 7, coverage only
7, both 21, neither 25.

The view at which detection happens is early in the budget, not late: PRODUCTION detects on view 0 in 11
worlds, view 1 in 10, view 2 in 2, view 3 in 5; coverage on view 0 in 12, view 1 in 3, view 2 in 8, view 3 in
5.

### 5.2 After detection, how much does each planner improve severity?

Essentially not at all, because there is almost nothing left of the mission. Of the 28 worlds each arm
detected in, **PRODUCTION takes zero further views in 26 and coverage takes zero further views in 27**. Mean
post-detection views: 0.107 (PRODUCTION) and 0.036 (coverage).

Mean change in `hidden_state_error_improvement` between the detection revision and the end of the mission:
**+0.0084** for PRODUCTION and **-0.0258** for coverage. Coverage's belief is on average slightly *worse* at
the end than at the moment of detection. Both numbers are noise next to the 0.78 step that detection itself
produces.

### 5.3 How many post-detection views does each take, and how much new geometric information does each
subsequent view carry?

Post-detection views are covered above (0.107 and 0.036 per detected world). The information content of
successive views, measured as the fraction of the surface cells a view sees that no earlier view of that
mission saw:

| view index | 1st | 2nd | 3rd | 4th |
|---|---|---|---|---|
| PRODUCTION, new cell fraction | 1.000 | 0.214 | 0.101 | 0.037 |
| coverage, new cell fraction | 1.000 | 0.225 | 0.258 | 0.175 |
| PRODUCTION, commanded patch fraction | 0.747 | 0.593 | 0.531 | 0.537 |
| coverage, commanded patch fraction | 0.682 | 0.599 | 0.316 | 0.247 |

By its fourth view PRODUCTION is looking at surface that is 96 % already seen, while coverage is still adding
18 % new surface. The mean angle between chosen view directions is 24.0 deg for PRODUCTION and 56.5 deg for
coverage. Run 1 shows the same shape (PRODUCTION 1.000 / 0.413 / 0.208 / 0.300 against coverage 1.000 / 0.200
/ 0.516 / 0.150, mean pair angle 33.2 deg against 69.8 deg).

MCBR keeps re-taking the same good view. Coverage keeps moving. In the worlds where the first view is right,
MCBR's repetition costs nothing; in the worlds where it is wrong, MCBR has no mechanism that makes it move.

### 5.4 Does predicted EIG correlate with actual hidden-state improvement?

Yes, and this is the part of MCBR that is working. Because only the chosen candidate's outcome is observed,
the per-candidate label is supplied by an evaluation-only geometric oracle over the recorded candidate tables:
the fraction of the true defect patch each listed candidate pose would actually see, with the world's real
occluders. `PRODUCTION`'s recorded `score` is `value / (0.05 + cost)`, so the predictive EIG itself is
recovered exactly as `score x (0.05 + cost)`.

| quantity | PRODUCTION | coverage |
|---|---|---|
| plans scored | 159 | 166 |
| within-plan Spearman(score, oracle patch fraction), mean | **+0.351** (86 non-degenerate plans) | +0.002 (90 plans) |
| within-plan Spearman(recovered EIG, oracle patch fraction) | +0.337 | +0.002 |
| top-1 regret (oracle best minus chosen), mean | **0.061** | 0.186 |
| plans where the chosen view IS the oracle best | **88.1 %** | 74.1 % |
| top-5 recall of useful candidates | 0.981 | 0.884 |
| top-10 recall of useful candidates | 0.990 | 0.978 |
| oracle patch fraction of the chosen view | 0.624 | 0.496 |
| oracle patch fraction of the best available candidate | 0.685 | 0.641 |
| oracle patch fraction averaged over all feasible candidates | 0.536 | 0.526 |

Calibration by score bin (candidates binned by their score percentile within their own plan; the label is the
oracle patch fraction):

| score percentile | 0-20 | 20-40 | 40-60 | 60-80 | 80-100 |
|---|---|---|---|---|---|
| PRODUCTION, mean oracle patch fraction | 0.559 | 0.648 | 0.647 | 0.725 | **0.745** |
| PRODUCTION, fraction of candidates that see any patch | 0.688 | 0.747 | 0.730 | 0.791 | 0.793 |
| coverage, mean oracle patch fraction | 0.590 | 0.737 | 0.692 | 0.651 | 0.601 |
| coverage, fraction of candidates that see any patch | 0.688 | 0.823 | 0.739 | 0.717 | 0.672 |

PRODUCTION's calibration is monotone and coverage's is not: coverage's top score bin is no better than its
bottom one, which is what "novelty, not aim" looks like in a number.

Correlating the chosen view's predicted value with the world's final outcome gives Spearman +0.388 over
PRODUCTION's 159 chosen views (coverage's score against the same outcome gives +0.451, but coverage's score is
visibility times novelty and is not an information estimate, so the two are not comparable).

One methodological note that matters for anyone repeating this: the oracle label is heavily tied, so rank
correlations must use average ranks. Plain `argsort` ranks, which is what the helper `_spearman` in
`conrad/evaluation/decision_experiments/active_mcbr_reeval.py` uses, break ties in index order, and candidate
index order is candidate-generation order, which correlates with azimuth and therefore with the score. With
`argsort` ranks the same data gives Spearman -0.503 instead of +0.351. The sign flips. The analysis script
uses average ranks; the repository helper is a latent reporting hazard elsewhere and is left alone here
because changing it would alter previously recorded diagnostics.

**Conclusion of 5.4: the ranking is not the failure.** The frozen ranker picks the best available candidate in
88 % of plans and its residual selection regret is 0.061 of a patch. There is at most 6 % of patch coverage
available to a better ranking rule, and 0 % available to it in the worlds where no feasible candidate sees the
patch at all (46 of 159 PRODUCTION plans had no useful candidate in the whole feasible set).

## 6. Limitation C, tested rather than assumed

**Claim.** With `read_channel: false`, the production predictive model values finding unread worst-case
surface but not refining an indication it has already detected, and that is what separates the better per-view
patch visibility (0.456 against 0.409) from the tied final reconstruction.

**The traces do not support it.** The read channel can only act after the component's worst indication has
been located, that is, after detection. Post-detection views essentially do not exist in these missions: 26 of
28 detected PRODUCTION worlds and 27 of 28 detected coverage worlds take **zero** views after the first direct
revision, and the mean post-detection view count is 0.107. A channel that only changes the ranking of a view
that is never taken cannot be responsible for a 0.0094 difference in the outcome.

Two further measurements point the same way:

* the total post-detection change in the metric is +0.008 (PRODUCTION) and -0.026 (coverage), so refinement
  after detection contributes nothing to either arm's score, in either direction;
* the published per-view patch visibility gap (0.456 against 0.409) is `patch_max_visible_fraction`, the
  maximum over the whole trajectory of the truth-side visible fraction of the patch, and it is not per view.
  Its paired CI is +0.048 [-0.050, +0.145], which does not exclude 0. The quantity whose CI does exclude 0 is
  the commanded per-view patch fraction, +0.091 [+0.027, +0.156], and that gap is created at plan time, before
  any belief update and therefore before any question about the read channel arises.

What actually separates better aim from the tied reconstruction is measured in section 4.1: about 45 % of
chosen views are never flown, and a commanded view that sees more than half the patch produces a belief
revision only 26 % of the time.

Turning the read channel on may still be worth testing on the fresh development split for a different reason
(a detected-but-badly-sized crack is the dominant residual error: crack absolute error 0.047 m against a
0.138 m defect), but it is not the explanation of the I4 tie, and it must not be justified by these numbers.

## 7. Limitation B, tested rather than assumed

**Claim.** Either the production belief update already captures the redundancy between overlapping views, or
overlapping candidates are effectively double counted.

**The traces support the second.** The probe matches candidates across consecutive plans of the same mission
by pose, computes each candidate's surface-cell overlap with the view that was just taken, and compares the
candidate's predicted value before and after that view:

| arm (run 2) | candidates | value ratio after / before, overlap >= 0.5 | value ratio, overlap <= 0.1 | Spearman(overlap, ratio) |
|---|---|---|---|---|
| PRODUCTION | 763 | **0.963** (n = 709) | 1.171 (n = 42) | **-0.114** |
| A-B2_coverage | 739 | 0.683 (n = 696) | 1.304 (n = 26) | -0.511 |

Run 1 agrees (PRODUCTION 0.973 against 1.258, Spearman -0.270; coverage 0.670 against 1.010, Spearman -0.589).

A candidate that duplicates the view just taken keeps 96 % of its predicted value. Coverage's explicit novelty
term cuts the same candidate to 68 %. The mechanism is visible in the model: `SurfaceCellPredictive` handles
redundancy only through `cell_prior`, which is zeroed on cells that Model2T records as *covered by a reading*
and discounted by `track_discount` along the vehicle's estimated track. A planned view that is commanded but
not flown, or flown but whose reading does not land on the target, changes neither, so the same candidate
scores the same on the next cycle. That is exactly the 45 % of views that are never reached, and exactly the
degenerate azimuth sequences in appendix A (world 8001000: PRODUCTION commands azimuth 90 deg, elevation 20
deg, four times in a row, with recovered EIG 0.0718, 0.0718, 0.0768, 0.0765).

Whether this is "double counting" in the strict Bayesian sense depends on whether the first view produced a
reading. When it did, Model2T's coverage update does remove those cells and the discount is real. When it did
not, the belief has no way to know that the view was already attempted, so the value is re-offered in full.
The observable consequence is the same either way: MCBR's views are 2.3 times less angularly diverse than
coverage's (24.0 deg against 56.5 deg mean pair angle) and carry a quarter as much new surface by the fourth
view (0.037 against 0.175).

## 8. Limitations A and D

The brief names B and C explicitly. The two other standing limitation claims in the audit trail are:

**A: the predicted value is poorly aligned with actual error reduction** (`MCBR_REEVALUATION.md` section 3.2,
raised against the old A-B10 planner). **Not supported for the frozen V3 planner.** Section 5.4: within-plan
Spearman +0.351, top-1 regret 0.061, the chosen view is the oracle best in 88 % of plans, and calibration by
score bin is monotone. Whatever V3 is doing wrong, mis-ranking the candidates it is given is not it.

**D: the runtime charges an arm that picks far, high-value views twice, because travel cost is not reflected
in how many planning cycles it is granted** (`I4_WORLD_FAMILY.md` section 9.1, already withdrawn in section
10.2 after run 2). **Still not supported, and the withdrawal was correct.** Planner calls per world 3.18
(PRODUCTION) against 3.38 (coverage); accepted views 2.65 against 2.77, paired -0.117 [-0.400, +0.167];
worlds that spend the full 4-view budget 27 against 31; travel 24.6 m against 26.2 m and energy 4042 J against
4192 J, both in PRODUCTION's favour. There is a small residual budget gap, but it is a fifth of the size run 1
reported and it cannot carry the result.

What the traces do support, and what no lettered limitation named, is a fifth mechanism:

**E: the commanded view and the flown view are different views.** 45 % of accepted inspection goals are never
reached within 0.5 m; the realised pose's patch fraction predicts the outcome better than the commanded pose's
(Spearman 0.71 against 0.54); and PRODUCTION's aiming advantage decays monotonically along the chain
(commanded +0.091 with a CI above 0, realised +0.045, truth patch max +0.048, defect read 0.000). A planner
that ranks candidates without a model of whether the mission will actually arrive at them is optimising a
quantity that is only half-transmitted to the belief.

## 9. Summary of what the traces do and do not support

| claim | verdict on these traces |
|---|---|
| A: the ranker's predicted value is misaligned with actual error reduction | NOT supported. Spearman +0.351 within plan, top-1 regret 0.061, 88 % oracle-best picks, monotone calibration |
| B: overlapping candidates are effectively double counted | SUPPORTED. Overlapping candidates keep 96 % of their value against coverage's 68 %; views are 2.3 times less diverse; the fourth view is 96 % redundant |
| C: `read_channel: false` is what separates per-view aim from final reconstruction | NOT supported. 26 of 28 detected worlds take zero post-detection views; post-detection gain +0.008 and -0.026; the published 0.456 / 0.409 gap is a trajectory maximum with a CI spanning 0 |
| D: far, high-value views are charged twice by the planning-cycle budget | NOT supported, consistent with its withdrawal in I4_WORLD_FAMILY 10.2. 3.18 against 3.38 planner calls, 2.65 against 2.77 views, less travel and less energy |
| E (new): commanded views are frequently not flown | SUPPORTED. 45 % of chosen views are never reached; the realised pose predicts the outcome better than the commanded one; the aiming advantage decays to zero along the chain |

The honest one-line reading: **MCBR V3 already picks very nearly the best view on the list. It loses the
benefit to two things that are not ranking, namely that it does not move on after an unproductive view and
that half of its chosen views are never flown.**

## 10. Whether any of this is worth building on

Section 5.4 bounds what a better ranking rule can win from inside these traces: 0.061 of a patch of selection
regret, in the 71 % of plans where a useful candidate exists at all. That is an upper bound on a re-ranking
V4, and it is small. `ACTIVE-MCBR-E006` measures the same ceiling end to end, with an evaluation-only oracle
that sees truth, on the **fresh development worlds** of `configs/eval/partitions_i4_mcbr_v4.yaml`. It agrees:
the best truth-seeing ranker reads the defect in 16 of 40 worlds, the frozen V3 planner in 15 and
coverage-only in 11, so the whole remaining ranking headroom above V3 is +0.0168 [-0.0821, +0.1269], one world
in forty. The result and the recommendation are in `docs/audits/MCBR_V4_ORACLE_HEADROOM.md`.

That also settles the practical weight of Limitation B. The oracle scores marginal rather than total patch
coverage, so it already ranks with perfect redundancy handling, and it upper-bounds any redundancy-aware
re-ranking. Limitation B is real (section 7) and it is worth at most +0.017.

## 11. Fresh partitions for any V4 work

`configs/eval/partitions_i4_mcbr_v4.yaml`, canonical-content SHA-256
`17490d3dbcc85673ebe624e6f04357f42b87a2316d431d2982af58e40e864457`, pinned in
`conrad/evaluation/partitions.py` as `I4_MCBR_V4_PARTITIONS_SHA256`, domain `i4_mcbr_v4`. It was written
before any run on any of its seeds.

| partition | seeds | n | purpose allowed |
|---|---|---|---|
| development | 8002000-8002039 | 40 | design, tuning |
| validation | 8002100-8002139 | 40 | design, tuning, selection |
| final_test | 8002200-8002259 | 60 | final evaluation only |
| ood_test | 8002300-8002319 | 20 | final evaluation only |

Every seed is disjoint from every other partition file and from every range the handoff marks SPENT,
including 8000200-8000219 and 8001000-8001059; the loader refuses the file otherwise. The final split is 60
worlds because run 2 measured a paired CI half width of about 0.09 at n = 60 and the per-world outcome is
binary, so a smaller final split could not separate a V4 from V3. **The final and OOD splits have not been
read and must not be read until a V4 mechanism is frozen.** The diagnostic phase read the development split
only, through `purpose_scope("design")`.

## Appendix A: per-world table, run 2

Columns: accepted inspection views; the selected azimuth and elevation of each view in degrees, in the world
frame about the target component centre (the same parameterisation the candidate generator uses); the mean
angle between chosen view directions (diversity); the mean surface-cell overlap of a view with all earlier
views of that mission; the mean commanded patch fraction; the recovered predicted value of the first chosen
view (for PRODUCTION this is the predictive EIG, for coverage it is visibility times novelty and is not an
information estimate); the view index at which the defect was first read; post-detection views; the final
absolute errors; `hidden_state_error_improvement`; final `U_O` and `U_C`; travel; energy; and the final
condition classification.

The table is generated by `scripts/analyze_i4_v3_failure.py` and kept beside the JSON records at
`artifacts/analysis/MCBR_V3_FAILURE/per_world_table_run2.md`.
