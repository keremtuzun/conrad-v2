# Is MCBR V4 worth building? The oracle headroom test (ACTIVE-MCBR-E006, 2026-09-20/21)

All data is SYNTHETIC_ONLY. This is a DEVELOPMENT measurement on the python sim kernel. It is SURROGATE
evidence, it promotes no gate, and it changes no recorded result. Gate I4 stays FAIL exactly as
`artifacts/gates/I4/evidence_formal.json` records it: on `ACTIVE_INSPECTION_OCCLUDED_V1` the production MCBR
planner beats a fixed inspection route and random views, and does not beat a systematic coverage sweep.

## 1. The question

`docs/audits/MCBR_V3_FAILURE_ANALYSIS.md` shows, from the spent gate I4 traces, that the frozen V3 planner
already picks the oracle-best candidate in 88 % of its plans, with a mean top-1 selection regret of 0.061 of a
defect patch. That bounds what a better ranking rule could win from inside those traces, and the bound is
small. This experiment measures the same ceiling end to end, on worlds that have never been used:

> replace the ranking rule with an evaluation-only oracle that sees truth and picks the candidate that would
> actually reveal most of the hidden defect. If the ORACLE cannot beat a systematic coverage sweep, no ranking
> rule can. If it beats coverage but not the incumbent, then V3 already captures the available headroom and a
> V4 that re-ranks views is still not worth building.

## 2. Design, declared before the run

* **Worlds**: `configs/eval/partitions_i4_mcbr_v4.yaml` DEVELOPMENT split, seeds 8002000-8002039, 40 worlds,
  digest `17490d3dbcc85673ebe624e6f04357f42b87a2316d431d2982af58e40e864457`. Fresh: disjoint from both spent
  gate I4 final splits and from every other partition file. The final and OOD splits of that file were not
  read, and the runner executes inside `purpose_scope("design")` and refuses any seed outside development.
* **Scenario**: `I4-OCCLUDED` on `ACTIVE_INSPECTION_OCCLUDED_V1`, 100 s mission, `max_plans_per_need` 4,
  Twin2E off, one structural sensor. Identical candidate generator, feasibility filter, `PlanningRequest`,
  budgets and metric for every arm. Only the ranking rule differs. Runs are deterministic per seed: the
  coverage arm reproduces bit-identically across the two invocations, which is asserted when the two result
  files are merged.
* **The oracle's objective is not a proxy for the metric.** On this family the outcome is exactly binary:
  over the 120 spent run-2 missions a world whose defect was read scored a mean
  `hidden_state_error_improvement` of 0.78 and a world whose defect was not read scored 0.000, with no
  intermediate case. Maximising the defect surface a view delivers is maximising the actual hidden-state error
  improvement.
* **Where the oracle lives**: `conrad/evaluation/oracle/i4_view_oracle.py`, on the evaluation side. The
  deployment packages cannot import it: `tests/leakage`'s static guard forbids `conrad.active`,
  `conrad.orchestration`, `conrad.decision`, `conrad.domains` and `conrad.runtime` from importing
  `conrad.evaluation` at all, and `tests/unit/evaluation/test_i4_mcbr_v4_partitions.py` adds a direct check
  for this module by name. The experiment injects it by rebinding
  `conrad.orchestration.deliberation.production_planner` inside its own worker process, the same mechanism
  ACTIVE-MCBR-E005 uses for candidate production planners. `tests/leakage` passes, 27 tests.
* The oracle's world model is harvested from each world's own finished run bundle: the surveyed design capsule
  from the mission context, and from the evaluation-only truth record the sampled defect patch and the exact
  geometry of the unregistered rack panels.

### 2.1 Arms

| arm | ranking rule |
|---|---|
| `A-B2_coverage` | visibility times novelty, the sweep gate I4 could not beat. Also the truth-harvest pass |
| `PRODUCTION` | the frozen V3 planner (`V-bayes_eig_ratio` plus the frozen predictive model) |
| `ORACLE_1STEP` | truth-side share of the defect patch cells this view would newly SEE (binary visibility) |
| `ORACLE_2STEP` | the best two-view union that starts with this view, immediate gain breaking ties |
| `ORACLE_1STEP_RATIO` | the binary oracle under the frozen planner's own cost rule, `value / (0.05 + cost)` |
| `ORACLE_1STEP_WEIGHTED` | the same, but cells count by `cos(incidence)` rather than 0 or 1 |
| `ORACLE_1STEP_WEIGHTED_RATIO` | the weighted oracle under the frozen planner's cost rule |

The weighted arms were added after the binary ones were measured, for a stated reason (section 3.1), and they
were run on the same declared development split with everything else unchanged.

## 3. Result

40 development worlds, one run per world per arm, paired on the world. "Worlds read" counts worlds where any
view produced a direct belief revision of the target.

| arm | HSE improvement | worlds read | views | target OBSERVED | patch seen | travel m | energy J |
|---|---|---|---|---|---|---|---|
| `ORACLE_1STEP_WEIGHTED_RATIO` | **0.3052** | 16 / 40 | 2.58 | 0.400 | 0.380 | 26.4 | 18327 |
| `PRODUCTION` (V3) | 0.2885 | 15 / 40 | 2.58 | 0.375 | 0.304 | 25.5 | 17746 |
| `ORACLE_1STEP_WEIGHTED` | 0.2840 | 15 / 40 | 2.60 | 0.375 | 0.356 | 26.4 | 18415 |
| `A-B2_coverage` | 0.2187 | 11 / 40 | 2.95 | 0.275 | 0.272 | 26.9 | 18949 |
| `ORACLE_1STEP_RATIO` | 0.1547 | 8 / 40 | 2.90 | 0.200 | 0.201 | 25.4 | 18630 |
| `ORACLE_1STEP` | 0.1449 | 7 / 40 | 2.70 | 0.175 | 0.183 | 23.9 | 16702 |
| `ORACLE_2STEP` | 0.1449 | 7 / 40 | 2.70 | 0.175 | 0.183 | 23.9 | 16702 |

Paired 95 % percentile-bootstrap benefit on `hidden_state_error_improvement` (4000 resamples over the 40
worlds), against coverage-only:

| arm | benefit over coverage |
|---|---|
| `ORACLE_1STEP_WEIGHTED_RATIO` | **+0.0865 [+0.0050, +0.1812]** |
| `PRODUCTION` | +0.0698 [-0.0177, +0.1633] |
| `ORACLE_1STEP_WEIGHTED` | +0.0653 [-0.0078, +0.1571] |
| `ORACLE_1STEP_RATIO` | -0.0640 [-0.1853, +0.0536] |
| `ORACLE_1STEP` / `ORACLE_2STEP` | -0.0737 [-0.1735, +0.0206] |

And the comparison the decision actually turns on:

| comparison | benefit | worlds |
|---|---|---|
| best oracle minus `PRODUCTION` | **+0.0168 [-0.0821, +0.1269]** | 16 read against 15 |
| best oracle minus coverage | +0.0865 [+0.0050, +0.1812] | 16 against 11 |
| best oracle minus binary oracle | +0.1603 [+0.0440, +0.2834] | 16 against 7 |

### 3.1 The binary oracles are not oracles, and that is itself a finding

`ORACLE_1STEP` and `ORACLE_2STEP` rank by whether a candidate pose can see a patch cell at all: facing, in
range, unoccluded. They finished **below coverage-only** and far below the incumbent, and their realised
truth-side patch visibility was the worst of any arm (0.183 against coverage's 0.272). An oracle that loses
while maximising its own objective is not measuring a ceiling, it is measuring a misspecified objective.

The cause is visible in the spent traces: a commanded view whose binary visibility exceeds half the patch
produced a belief revision only 26 % of the time, while a view that actually delivered that patch at the
realised pose produced one 45 % of the time. Binary visibility over-credits a grazing look and a look from
the far end of the sensor's range, which a real reading does not deliver.

The weighted oracle replaces the binary test with `cos(incidence)` on the unoccluded cells, which is the
truth-side twin of the production predictive model's own cell weight (`MissionPredictive`'s `CellWeightFn`:
clear-ray probability times `cos(incidence) ** incidence_power`, with `incidence_power` 1.0). That single
change moves the oracle from 7 worlds read to 15, a paired +0.160 [+0.044, +0.283]. Adding the frozen
planner's cost rule adds one more world.

Two smaller observations, reported because they are visible in the data:

* `ORACLE_2STEP` is identical to `ORACLE_1STEP` on every metric in all 40 worlds. Two-step lookahead never
  changed a selection here: the family leaves one angular window, so the best two-view union is reachable from
  the best single view, and the tie-break by immediate gain then reproduces the greedy choice.
* the oracle's advantage is not bought with resources: it travels no further than coverage (26.4 m against
  26.9 m) and takes fewer views (2.58 against 2.95).

## 4. Reading, and the recommendation

**There is exploitable view-selection headroom in this family, and MCBR V3 has already taken almost all of
it.** The best truth-seeing ranker reads the defect in 16 of 40 worlds. The frozen V3 planner reads it in 15.
Coverage-only reads it in 11. In `hidden_state_error_improvement`, the whole interval between coverage and a
perfect ranker is 0.0865, and V3 already occupies 0.0698 of it, leaving **+0.0168, one world in forty**, with
a CI of [-0.0821, +0.1269] that spans zero.

For scale: the formal replication measured a paired CI half width of about 0.09 on 60 worlds. A V4 that
captured the entire remaining ranking headroom would move the metric by about a fifth of the noise floor of
the experiment that would have to certify it. It could not be distinguished from V3 on any final split this
project can afford to run.

**Recommendation: do not build a V4 that re-ranks candidate views.** That includes the redundancy repair that
Limitation B in the failure analysis points at. The oracle already accounts for what earlier views delivered
(it scores marginal, not total, patch coverage), so it upper-bounds any redundancy-aware re-ranking too, and
that upper bound is +0.017 over the incumbent.

What this experiment does **not** bound, and where the remaining difference lives, is everything outside the
ranking rule under the current runtime protocol:

* which candidates the feasibility filter offers in the first place. In 46 of PRODUCTION's 159 plans in the
  spent traces, no feasible candidate saw any part of the patch. No ranking can fix that;
* whether a chosen view is ever flown. About 45 % of accepted inspection goals never came within 0.5 m of the
  commanded pose, and the realised pose predicts the outcome better than the commanded one (Spearman 0.71
  against 0.54). A planner that re-plans when a view is abandoned, or that prefers a view it will actually
  reach, is changing the protocol, not the ranking, and is outside this bound;
* how many planning cycles the mission grants inside its 100 s.

If MCBR work continues, it should target those and be evaluated on the validation split of
`configs/eval/partitions_i4_mcbr_v4.yaml`, with the final split kept closed until a mechanism is frozen.
Nothing in this document, and nothing in the failure analysis, may be used to tune such a mechanism: both are
measurements, and the development split has now been read once.

## 5. Artifacts

| file | content |
|---|---|
| `artifacts/experiments/ACTIVE-MCBR-E006/i4_oracle_headroom_binary_oracles.json` | coverage, the three binary oracle arms and PRODUCTION, 40 worlds |
| `artifacts/experiments/ACTIVE-MCBR-E006-W/i4_oracle_headroom_weighted.json` | coverage and the two incidence-weighted oracle arms, the same 40 worlds. Its `experiment_id` field reads `ACTIVE-MCBR-E006` because the runner did not yet take the id from the config; the directory and the config name it as the `-W` run |
| `artifacts/experiments/ACTIVE-MCBR-E006/per_world_merged.json` | both runs merged per world, after asserting that the shared coverage arm reproduces exactly |
| `configs/eval/i4_oracle_headroom.yaml`, `configs/eval/i4_oracle_headroom_weighted.yaml` | the declared configurations |
| `conrad/evaluation/oracle/i4_view_oracle.py` | the evaluation-only oracle |
| `conrad/evaluation/decision_experiments/i4_oracle_headroom.py` | the runner, registered as `ACTIVE-MCBR-E006` and `ACTIVE-MCBR-E006-W` |
