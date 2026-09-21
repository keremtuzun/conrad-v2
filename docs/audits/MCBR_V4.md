# MCBR V4: the chosen view is now the flown view (2026-09-21)

All data is SYNTHETIC_ONLY and every run in this document is on the python sim kernel. It is SURROGATE
evidence: it promotes no gate and it changes no recorded result.

**Gate I4 stands exactly as recorded.** On `ACTIVE_INSPECTION_OCCLUDED_V1` the frozen production MCBR
planner beats a fixed inspection route, beats random views, and does not beat a systematic coverage sweep.
The pre-registered replication (60 worlds, 8001000-8001059) measured `hidden_state_error_improvement`
+0.0094 [-0.0828, +0.0979], a tie, and formal I4 = FAIL. Nothing here re-scores that, and nothing here has
been evaluated on a held-out split.

## 1. Why V4 does not re-rank views

`docs/audits/MCBR_V3_FAILURE_ANALYSIS.md` and `docs/audits/MCBR_V4_ORACLE_HEADROOM.md` established three
things that together closed off the obvious V4:

* the per-world outcome on this family is exactly binary. Over the 120 spent replication missions a world
  whose defect was read scored a mean of 0.78 and a world whose defect was not read scored 0.000, with no
  intermediate case, and **both arms read the defect in exactly 28 of 60 worlds**. The entire gate
  difference is a success-rate difference of zero;
* V3's ranking is already near optimal: the chosen view is the oracle best in 88 % of plans and the mean
  top-1 selection regret is 0.061 of a defect patch;
* an evaluation-only oracle that SEES TRUTH beats coverage by +0.0865 [+0.0050, +0.1812] but beats V3 by
  only +0.0168 [-0.0821, +0.1269]. Better ranking, read-channel refinement, complementarity and two-step
  lookahead are together bounded at about +0.017, a fifth of the noise floor of the run that would certify
  them.

So V4 targets the two things that bound does not cover: whether a chosen view is ever flown, and which
candidates the filter offers at all. Each extra world in which the defect is read is worth about 0.78,
roughly eighty times the residual ranking gain.

## 2. What was measured, and how

A **view execution ledger** (`conrad/orchestration/view_execution.py`) was added to the deployment plane.
Every accepted inspection goal opens a record; every control tick updates the closest approach to the
commanded pose and the aim error; and the goal is closed with exactly one reason code. It reads the
runtime's own EKF estimate and the commanded pose from the plan. It holds no truth, and with the V4
switches off it only records, so the incumbent behaviour is unchanged.

Two definitions are used throughout, and they are the ones the runtime itself declares on the
`NavigationGoal`:

| term | definition |
|---|---|
| reached | the estimated position came within the declared `position_tolerance_m` (0.35 m) of the commanded pose |
| flown | reached, AND the sensor boresight was within the declared `orientation_tolerance_rad` (0.3 rad) of the aim point, AND that state was held for the declared dwell (6 s) |

An observation is credited to a plan only when it was captured inside both tolerances. `reached` and
`flown` are reported separately so that a view that arrived but pointed away is not silently counted as a
success.

The reason codes separate the candidate explanations the brief named:

| code | what it means |
|---|---|
| `FLOWN` | the view was flown to the declared tolerance and the dwell was held |
| `ABANDONED_ROUTE_BLOCKED` | a REPLAN(ROUTE_BLOCKED) decision pre-empted the view while it was in flight |
| `ABANDONED_TIMEOUT` | `inspection_timeout_s` elapsed with the vehicle still in transit |
| `ABANDONED_MISSION_END` | the mission clock ran out with the view still executing |
| `ABANDONED_PREEMPTED` | some other goal replaced the view goal |
| `ABANDONED_SAFE_HOLD` | the safety supervisor's hold replaced it |
| `ABANDONED_NAV_REJECTED` | the navigation stack refused the view goal itself |
| `ABANDONED_TRAJECTORY_LOST` | a later goal was refused, which cleared the running view's objective |
| `COMPLETED_OFF_POSE` | navigation declared the goal COMPLETE outside the declared pose tolerance |
| `ARRIVED_DWELL_SHORT` | inside both tolerances, but the goal ended before the dwell was complete |

Experiment `ACTIVE-MCBR-E007-DEV` ran the incumbent and coverage, plus five V4 variants, on the
DEVELOPMENT split of `configs/eval/partitions_i4_mcbr_v4.yaml` (seeds 8002000-8002039, 40 worlds, partition
digest `17490d3dbcc85673ebe624e6f04357f42b87a2316d431d2982af58e40e864457`), inside
`purpose_scope("design")`, which refuses any seed outside that split. Every arm gets the same mission, the
same 100 s duration, the same `max_plans_per_need` budget of 4, the same sensor and the same metric; the
arms differ only in the runtime configuration named in `configs/eval/i4_v4_development.yaml`.

Both reference arms reproduce the independent measurement of `ACTIVE-MCBR-E006` on the same 40 worlds to
four decimals: V3 0.2885 with 15 worlds read, coverage 0.2187 with 11 worlds read. That is the pipeline
validation for this document.

One earlier invocation is recorded for completeness. `configs/eval/i4_v4_diagnostic.yaml` ran the two
reference arms on the same development worlds and then crashed in its own statistics step, on a metric that
is not defined in a world where no view was accepted. The runs finished but no artifact was written, the
runner now persists the per-world record before any statistic is computed, and a metric that is not finite
in every world is reported as not paired rather than imputed. Those 40 worlds are development worlds and
were simply re-run as part of `ACTIVE-MCBR-E007-DEV`.

## 3. The measured execution-failure breakdown

40 development worlds, per arm, summed over worlds. The incumbent accepted 103 view goals and flew 48.

| outcome | V3 PRODUCTION | share | A-B2 coverage | share |
|---|---|---|---|---|
| `FLOWN` | 48 | 46.6 % | 60 | 50.8 % |
| `ABANDONED_ROUTE_BLOCKED` | **45** | **43.7 %** | **55** | **46.6 %** |
| `ABANDONED_MISSION_END` | 10 | 9.7 % | 3 | 2.5 % |
| `ABANDONED_TIMEOUT` | 0 | 0 % | 0 | 0 % |
| `ABANDONED_PREEMPTED` | 0 | 0 % | 0 | 0 % |
| `ABANDONED_SAFE_HOLD` | 0 | 0 % | 0 | 0 % |
| `ABANDONED_NAV_REJECTED` | 0 | 0 % | 0 | 0 % |
| `ABANDONED_TRAJECTORY_LOST` | 0 | 0 % | 0 | 0 % |
| `COMPLETED_OFF_POSE` | 0 | 0 % | 0 | 0 % |
| `ARRIVED_DWELL_SHORT` | 0 | 0 % | 0 | 0 % |
| accepted views | 103 | | 118 | |

**One mechanism accounts for essentially every view that was not flown, and it is not any of the ones the
brief listed as likely.** It is not the mission clock in transit, not the navigation goal tolerance, not a
different controller acceptance criterion, not an observation captured away from the intended pose, and not
a late safety or feasibility refusal: those are all exactly zero. It is a pre-empting decision, and the
pre-empting decision is always the same one.

### 3.1 The mechanism, traced

On world 8002000 the incumbent accepts four inspection views and flies none of them. The ledger and the
decision log agree tick for tick:

| t (s) | event |
|---|---|
| 32 | lane survey ends; MCBR plans; INSPECT goal accepted, commanded pose 8.41 m away |
| 34-40 | the goal executes; the vehicle closes from 7.68 m to 5.32 m |
| 40 | EGDC decides REPLAN(ROUTE_BLOCKED); the view is abandoned at 5.32 m |
| 42 | MCBR plans again; INSPECT accepted, 5.41 m away |
| 44 | REPLAN(ROUTE_BLOCKED) again; abandoned at 4.88 m |
| 46-48 | third view, abandoned at 4.59 m |
| 64-66 | fourth and last view, abandoned at 4.20 m; `max_plans_per_need` is now spent |
| 66-100 | station keeping; the defect is never read |

The cause is a single line of the executive. `DecisionRouting._route` guards MCBR and NAVIGATION requests
with a `busy` test, so an inspection already in progress is treated as the attempt and a new request is
dropped. The MISSION_EXECUTIVE branch has no such guard, and
`MissionExecutive.replan_detour` re-routes only a TRANSIT goal: for any other goal it calls
`hold_goal(...)`, which is a new station-keep goal, which replaces the running view.

The world family makes this systematic rather than incidental. `ACTIVE_INSPECTION_OCCLUDED_V1` places two
unregistered rack panels flanking the target pipe on the defect side, leaving one angular window through
which the defect can be seen (`conrad/sim/mission/occlusion.py`). Any view that resolves the defect must be
approached past those panels. Once Model2S has OBSERVED them, the planned-route corridor of that very
approach contains occupied cells that are not part of the surveyed design, EGDC correctly reports
ROUTE_BLOCKED, and the executive answers by abandoning the approach. **The better a view is, the more
reliably it is cancelled.** That is why V3's aiming advantage decays to nothing along the chain, and why
both arms read the defect in the same number of worlds.

### 3.2 Realised versus commanded

Pooled over the development worlds in which each arm took at least one view:

| quantity | V3 PRODUCTION | A-B2 coverage |
|---|---|---|
| mean closest approach to the commanded pose | 1.731 m | 2.252 m |
| angle at the aim point between the commanded and the realised view direction | 0.557 rad (31.9 deg) | 0.769 rad (44.1 deg) |
| angle between the realised boresight and the aim point | 0.409 rad (23.4 deg) | 0.470 rad (26.9 deg) |
| fraction of accepted views that reached the pose tolerance (pooled over views) | 0.563 | 0.534 |
| fraction of accepted views flown to pose AND aim, for the dwell (pooled over views) | 0.466 | 0.508 |

### 3.3 Candidate supply

| quantity | V3 PRODUCTION | A-B2 coverage |
|---|---|---|
| plans | 118 | 143 |
| plans with an action | 103 | 118 |
| plans that returned NO_FEASIBLE_OBSERVATION | 12 | 16 |
| plans that returned NEED_SATISFIED | 3 | 9 |
| mean feasible candidates per plan | 7.38 | 6.86 |
| navigation goals refused outright | 19 | 17 |

About one plan in ten comes back with no feasible candidate at all, out of a ring of 48 candidates of which
only about seven survive the shared filter.

## 4. What was changed

Everything below is inside the existing contracts. MCBR still outputs an `ObservationPlan` and navigation
still moves the robot; `conrad.runtime.command_gateway` is still the only path to hardware; the safety
supervisor and the local planner are untouched; and every V4 switch defaults to OFF, so the recorded V3
behaviour is reproducible byte for byte (verified: the V3 rows of the second development round reproduce
the first round exactly on all 40 worlds).

| file | change |
|---|---|
| `conrad/orchestration/view_execution.py` | NEW. The view execution ledger: `ViewCommand`, `ViewExecutionRecord`, `ViewLedger`, the reason-code vocabulary and `summarize`. Deployment plane, no truth |
| `conrad/active/gap.py` | NEW `AbandonedView`: the belief-side record of a view the mission accepted and did not fly, beside the existing `PriorView` |
| `conrad/active/planner.py` | `PlanningRequest` gains `abandoned_views`, `time_remaining_s`, `energy_remaining_j`, all optional and empty by default. `MCBRPlanner` gains an optional `extra_filter` hook, evaluated with the shared `FeasibilityFilter` and BEFORE any ranking |
| `conrad/active/v4.py` | NEW. `V4Config`, `ExecutionFilter` and `v4_planner`: V3's ranking rule behind two extra feasibility tests, `APPROACH_ABANDONED` and `TIME_BUDGET_EXCEEDED` |
| `conrad/orchestration/executive.py` | `set_goal` opens and closes view records, `_close_reason` decides the outcome from the state the view actually reached, `control_tick` updates the ledger, `close_views` closes a view still running at mission end |
| `conrad/orchestration/routing.py` | the ROUTE_BLOCKED guard for a view in flight; `reconcile_views`; attempt refunds and the `max_view_attempts` cap; abandoned views and budgets passed into the plan; `empty_plan_row`, so a plan with no action is recorded instead of dropped |
| `conrad/orchestration/mission.py` | `_nav_is_free` / `_nav_distance`: with the V4 switch on, the navigation stack's global and local planners also see Model2S OBSERVED occupancy, not only the surveyed registry design |
| `conrad/orchestration/deliberation.py` | builds the V4 planner; `plan()` carries the abandoned views and the budgets; `corridor_blocked` charges a candidate whose believed corridor crosses observed unregistered structure |
| `conrad/orchestration/mission_config.py` | `ViewExecutionSettings` (every switch off by default), `v4`, `view_position_tolerance_m`, `view_orientation_tolerance_rad` |
| `conrad/orchestration/artifacts.py` | the ledger, the empty plans, the view outcomes, the deferred replans and per-plan feasible counts and latency reach `mission/view_execution.json` and `runtime_metrics.json` |
| `conrad/evaluation/decision_experiments/i4_view_execution.py` | NEW. `ACTIVE-MCBR-E007`, the experiment runner |
| `scripts/run_mcbr_v4_experiment.py` | NEW. Runs one V4 experiment config; it reads the commit from `.git/HEAD` and executes no git command |

The single behavioural change that carries the result is the first one in `routing.py`: while an INSPECT or
REVISIT goal is running, a REPLAN(ROUTE_BLOCKED) decision is deferred and recorded rather than answered
with a hold. This is the same rule the `busy` guard already applies to MCBR and NAVIGATION requests, and it
takes nothing away from safety: collision avoidance is the local planner's and the safety supervisor's job,
and neither was touched.

## 5. Development results

`ACTIVE-MCBR-E007-DEV`, 40 worlds, seeds 8002000-8002039, one run per world per arm, paired on the world.
`read/40` counts worlds where the defect produced a direct belief revision of the target. `aim` is the
angle at the aim point between the commanded and the realised view direction, `bore` the angle between the
realised boresight and the aim point, both at the closest approach and pooled over worlds with at least one
view. `views_bd` and `t_det` are over all worlds, with an unread world counted at its full view count and
at the mission duration. `flown` is pooled over views; `reached` is the per-world mean of the per-world
fraction, and pooled over views it is 0.563 (V3), 0.534 (coverage), 1.000 (`V4_full`, `V4_protocol_only`,
`V4_no_belief_map`, `V4_route_cost`) and 0.989 (`V4_supply`).

| arm | HSE | read/40 | flown | reached | aim rad | bore rad | closest m | empty plans | views | views_bd | t_det s | redundant | travel m | energy J | latency ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `V4_supply` | **0.4764** | **23** | 0.789 | 0.994 | 0.0229 | 0.0270 | 0.056 | **0.000** | 2.38 | 1.70 | 76.7 | 1.75 | 27.5 | 19598 | 4464 |
| `V4_route_cost` | 0.4062 | 21 | 0.821 | 0.950 | 0.0112 | 0.0130 | 0.035 | 0.081 | 1.95 | 1.38 | 78.0 | 1.38 | 26.6 | 18103 | 1677 |
| `V4_full` | 0.4001 | 20 | 0.813 | 0.950 | 0.0117 | 0.0171 | 0.035 | 0.081 | 1.88 | 1.35 | 77.5 | 1.35 | 26.6 | 18167 | 1630 |
| `V4_protocol_only` | 0.4001 | 20 | 0.813 | 0.950 | 0.0117 | 0.0171 | 0.035 | 0.081 | 1.88 | 1.35 | 77.5 | 1.35 | 26.6 | 18167 | 1635 |
| `V4_no_belief_map` | 0.3978 | 20 | 0.840 | 0.950 | 0.0124 | 0.0119 | 0.037 | 0.081 | 1.88 | 1.35 | 77.3 | 1.35 | 26.7 | 18044 | 1629 |
| `V3_PRODUCTION` | 0.2885 | 15 | 0.466 | 0.658 | 0.5566 | 0.4090 | 1.731 | 0.081 | 2.58 | 2.17 | 85.5 | 2.17 | 25.5 | 17746 | 1687 |
| `A-B2_coverage` | 0.2187 | 11 | 0.508 | 0.577 | 0.7692 | 0.4700 | 2.252 | 0.100 | 2.95 | 2.67 | 88.7 | 2.67 | 26.9 | 18949 | 1643 |

Restricted to the worlds where the defect WAS read: mean time to first direct revision 61.3 s (V3), 59.0 s
(coverage), 55.6 s (`V4_full`), 60.9 s (`V4_supply`); mean views before detection 1.00, 1.09, 0.30 and 0.65.

Paired 95 % percentile-bootstrap benefit over the incumbent V3, 4000 resamples over the 40 worlds. A
positive benefit means the V4 arm is better on that metric in its declared direction.

| arm | `hidden_state_error_improvement` | defect read | fraction of views flown | empty-plan rate |
|---|---|---|---|---|
| `V4_supply` | **+0.1880 [+0.0387, +0.3307]** | +0.200 [+0.025, +0.375] | +0.285 [+0.138, +0.431] | +0.081 [+0.013, +0.169] |
| `V4_route_cost` | +0.1178 [+0.0351, +0.2094] | +0.150 [+0.050, +0.275] | +0.256 [+0.138, +0.377] | 0.000 [0.000, 0.000] |
| `V4_full` | +0.1117 [+0.0110, +0.2184] | +0.125 [+0.000, +0.250] | +0.256 [+0.138, +0.377] | 0.000 [0.000, 0.000] |
| `V4_protocol_only` | +0.1117 [+0.0110, +0.2184] | +0.125 [+0.000, +0.250] | +0.256 [+0.138, +0.377] | 0.000 [0.000, 0.000] |
| `V4_no_belief_map` | +0.1093 [+0.0079, +0.2145] | +0.125 [+0.000, +0.250] | +0.273 [+0.156, +0.394] | 0.000 [0.000, 0.000] |
| `A-B2_coverage` | -0.0698 [-0.1633, +0.0177] | -0.100 [-0.225, +0.000] | +0.002 [-0.106, +0.098] | -0.019 [-0.050, +0.000] |

And against the coverage sweep that gate I4 could not beat:

| arm | benefit over coverage on `hidden_state_error_improvement` | defect read |
|---|---|---|
| `V4_supply` | **+0.2578 [+0.1287, +0.3923]** | +0.300 [+0.150, +0.450] |
| `V4_route_cost` | +0.1876 [+0.0865, +0.2991] | +0.250 [+0.125, +0.400] |
| `V4_full` | +0.1814 [+0.0804, +0.2947] | +0.225 [+0.100, +0.350] |
| `V4_no_belief_map` | +0.1791 [+0.0784, +0.2907] | +0.225 [+0.100, +0.350] |
| `V3_PRODUCTION` | +0.0698 [-0.0177, +0.1633] | +0.100 [+0.000, +0.225] |

Costs, paired against V3 (positive = the V4 arm uses less): `V4_full` travel -1.11 m [-1.80, -0.49] and
energy -421 J [-1001, +97]; `V4_supply` travel +1.92 m more [0.84, 3.12] and energy +1852 J more
[759, 3048], with planning latency 2777 ms per plan higher [2380, 3107]. Mission time is 100 s for every
arm by construction.

### 5.1 The ablations, reported including the ones that changed nothing

* **`V4_protocol_only` is identical to `V4_full` on every metric in all 40 worlds.** The V4 planner's two
  extra feasibility tests never fired: once a view in flight is protected, no view is abandoned for an
  approach reason, so `APPROACH_ABANDONED` has nothing to refuse; and this scenario declares no operator
  time budget, so `TIME_BUDGET_EXCEEDED` is never evaluated. **On development, the entire effect is the
  execution protocol and none of it is the new planner.** That is stated here rather than hidden because it
  is the honest reading of the ablation.
* **`V4_no_belief_map` is within noise of `V4_full`** (0.3978 against 0.4001, same 20 worlds read). Giving
  the navigation stack the observed occupancy of the belief map is the right thing to do and it did not
  change the outcome on this family: once the view is not cancelled, the local planner and the trajectory
  controller already get the vehicle to the commanded pose.
* **`V4_route_cost` adds one world** (21 against 20) at no measurable cost. Its effect is a ranking
  preference, and the oracle bound says a ranking preference cannot be worth much here; one world in forty
  is consistent with that bound rather than evidence against it.
* **A second development round (`ACTIVE-MCBR-E007-DEV2`) tested a mission-deadline budget and it did
  nothing.** With `budget_from_mission_duration` on, the planner declares the mission's own remaining
  duration as the horizon and the shared filter may refuse a view that cannot be reached and dwelled on in
  time. `V4_deadline` and `V4_supply_deadline` produced output identical to `V4_full` and `V4_supply` on all
  40 worlds. The reason is visible in the numbers: the planner's travel-time estimate is straight-line
  distance over `cruise_speed_mps`, which is optimistic relative to the flown trajectory, so the test never
  binds. The variants are kept in the record and carried to validation, where the budget could bind on a
  world it did not bind on here.
* **Candidate supply is the one thing besides the protocol that moved the read rate.** A denser ring
  (16 azimuths, 3 elevations, 3 standoffs, thinned deterministically to the 128-candidate cap) raises the
  mean feasible candidate count from 7.06 to 21.36, eliminates NO_FEASIBLE_OBSERVATION entirely (12 plans
  to 0) and adds three worlds over `V4_full`. Ring candidates remain the baseline and no belief-conditioned
  candidate generator was added, because the development evidence did not require one. Every candidate
  still passes the shared `FeasibilityFilter` before any score is computed.

### 5.2 Has the mechanism changed?

Yes, and the evidence is mechanistic rather than only distributional:

* views abandoned for a route-blocked replan: **45 of 103 for V3, 0 of 75 for every V4 arm**;
* fraction of accepted views flown to the declared pose and aim tolerance for the declared dwell: **0.466
  for V3, 0.789 to 0.840 for the V4 arms**;
* mean closest approach to the commanded pose: **1.731 m for V3, 0.035 m to 0.056 m for the V4 arms**;
* realised-versus-commanded aim error: **0.557 rad for V3, 0.011 to 0.023 rad for the V4 arms**;
* worlds where the defect is read: **15 of 40 for V3, 20 to 23 of 40 for the V4 arms**, against 11 for
  coverage and 16 for the truth-seeing ranking oracle of `ACTIVE-MCBR-E006`.

The last line is the point of the whole exercise. The oracle bounded what RANKING can win on this family
and the bound was one world in forty. Execution was outside that bound, and it is worth eight.

## 6. Validation and selection

The selection rule was declared in `configs/eval/i4_v4_validation.yaml` **before the validation split was
read**: primary metric `hidden_state_error_improvement`, higher is better; secondaries `defect_read`,
`fraction_views_flown`, `empty_plan_rate`, `views_before_detection`, `time_to_detection_s`,
`redundant_observations`, `travel_m`, `energy_j`, `planning_latency_ms`; unit = one world, paired across
arms, 95 % paired percentile bootstrap with 4000 resamples. Among the V4 variants whose paired benefit over
`V3_PRODUCTION` on the primary metric has a CI lower bound above 0, take the highest primary mean, breaking
a tie by the higher `defect_read`, then by the lower planning latency. If no variant clears that bar,
nothing is frozen and no final run is spent.

`ACTIVE-MCBR-E007-VAL`, seeds 8002100-8002139, 40 worlds, purpose `selection`, which the partition loader
restricts to the validation split. Nine arms, one run per world per arm, 3332 s of wall clock.

| arm | HSE | read/40 | flown | reached | aim rad | bore rad | closest m | empty plans | views | views_bd | t_det s | redundant | travel m | energy J | latency ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `V4_protocol_only` | **0.4971** | **25** | 0.828 | 0.983 | 0.0111 | 0.0292 | 0.035 | 0.175 | 1.45 | 0.72 | 71.4 | 0.72 | 25.7 | 17927 | 1447 |
| `V4_full` | 0.4971 | 25 | 0.828 | 0.983 | 0.0111 | 0.0292 | 0.035 | 0.175 | 1.45 | 0.72 | 71.4 | 0.72 | 25.7 | 17927 | 1461 |
| `V4_deadline` | 0.4971 | 25 | 0.828 | 0.983 | 0.0111 | 0.0292 | 0.035 | 0.175 | 1.45 | 0.72 | 71.4 | 0.72 | 25.7 | 17927 | 1463 |
| `V4_route_cost` | 0.4941 | 25 | 0.839 | 0.984 | 0.0109 | 0.0275 | 0.035 | 0.175 | 1.55 | 0.80 | 71.7 | 0.80 | 25.8 | 17823 | 1521 |
| `V4_no_belief_map` | 0.4526 | 23 | 0.846 | 1.000 | 0.0093 | 0.0225 | 0.030 | 0.169 | 1.62 | 0.90 | 72.7 | 0.97 | 25.9 | 17373 | 1516 |
| `V4_supply` | 0.4462 | 22 | 0.841 | 0.988 | 0.0132 | 0.0238 | 0.039 | **0.075** | 2.05 | 1.27 | 73.1 | 1.32 | 26.9 | 18793 | 4539 |
| `V4_supply_deadline` | 0.4462 | 22 | 0.841 | 0.988 | 0.0132 | 0.0238 | 0.039 | 0.075 | 2.05 | 1.27 | 73.1 | 1.32 | 26.9 | 18793 | 4572 |
| `A-B2_coverage` | 0.2565 | 13 | 0.364 | 0.390 | 1.0795 | 0.5920 | 3.210 | 0.144 | 2.95 | 2.52 | 86.6 | 2.60 | 26.1 | 18581 | 1602 |
| `V3_PRODUCTION` | 0.1937 | 10 | 0.269 | 0.327 | 0.8714 | 0.6334 | 2.927 | 0.156 | 2.60 | 2.23 | 87.8 | 2.25 | 23.2 | 16160 | 1562 |

Restricted to the worlds where the defect WAS read, mean time to first direct revision: 55.6 s (V3), 62.3 s
(coverage), 54.3 s (the selected arm), 55.4 s (`V4_supply`); mean views before detection 0.50, 1.54, 0.24
and 0.32.

The validation worlds are harder for the incumbent than the development worlds. V3 reads the defect in 10 of
40 and flies 28 of its 104 accepted views, and it now finishes **below** the coverage sweep (0.1937 against
0.2565, a paired -0.0628 [-0.1609, +0.0286]). The V4 arms do not move the same way: 25 of 40 worlds read and
48 of 58 accepted views flown.

Execution breakdown on validation, summed over the 40 worlds:

| outcome | V3 PRODUCTION | A-B2 coverage | selected V4 | `V4_supply` |
|---|---|---|---|---|
| `FLOWN` | 28 | 43 | 48 | 69 |
| `ABANDONED_ROUTE_BLOCKED` | 70 | 72 | **0** | **0** |
| `ABANDONED_MISSION_END` | 5 | 3 | 9 | 13 |
| `ARRIVED_DWELL_SHORT` | 1 | 0 | 1 | 0 |
| accepted views | 104 | 118 | 58 | 82 |
| plans | 137 | 147 | 104 | 101 |
| plans with NO_FEASIBLE_OBSERVATION | 25 | 23 | 28 | **12** |
| mean feasible candidates per plan | 7.6 | 7.7 | 6.6 | **21.8** |
| navigation goals refused outright | 52 | 30 | 12 | 6 |

Paired 95 % percentile-bootstrap benefit over the incumbent V3 on validation. Positive means better in the
metric's declared direction, so a positive travel, energy, latency or detection-time entry means less of it.

| arm | `hidden_state_error_improvement` | defect read | fraction of views flown | views before detection | time to detection s | redundant obs | travel m | energy J | latency ms |
|---|---|---|---|---|---|---|---|---|---|
| `V4_protocol_only` (selected) | **+0.3034 [+0.1808, +0.4279]** | +0.375 [+0.225, +0.525] | +0.431 [+0.300, +0.563] | +1.500 [+0.975, +2.025] | +16.35 [+9.61, +23.28] | +1.525 [+1.000, +2.050] | -2.55 [-3.64, -1.50] | -1768 [-3028, -725] | +114 [+6, +241] |
| `V4_full` | +0.3034 [+0.1808, +0.4279] | +0.375 [+0.225, +0.525] | +0.431 [+0.300, +0.563] | +1.500 [+0.975, +2.025] | +16.35 [+9.61, +23.28] | +1.525 [+1.000, +2.050] | -2.55 [-3.64, -1.50] | -1768 [-3028, -725] | +100 [-11, +234] |
| `V4_deadline` | +0.3034 [+0.1808, +0.4279] | +0.375 [+0.225, +0.525] | +0.431 [+0.300, +0.563] | +1.500 [+0.975, +2.025] | +16.35 [+9.61, +23.28] | +1.525 [+1.000, +2.050] | -2.55 [-3.64, -1.50] | -1768 [-3028, -725] | +99 [-15, +234] |
| `V4_route_cost` | +0.3004 [+0.1762, +0.4267] | +0.375 [+0.225, +0.525] | +0.433 [+0.302, +0.567] | +1.425 [+0.900, +1.950] | +16.05 [+9.44, +22.88] | +1.450 [+0.925, +1.975] | -2.58 [-3.74, -1.46] | -1663 [-2966, -633] | +41 [-97, +190] |
| `V4_no_belief_map` | +0.2589 [+0.1254, +0.3957] | +0.325 [+0.175, +0.475] | +0.438 [+0.302, +0.573] | +1.325 [+0.800, +1.850] | +15.13 [+8.72, +21.83] | +1.275 [+0.775, +1.801] | -2.74 [-3.74, -1.78] | -1214 [-1991, -547] | +46 [-41, +152] |
| `V4_supply` | +0.2525 [+0.0946, +0.4049] | +0.300 [+0.100, +0.475] | +0.494 [+0.346, +0.633] | +0.950 [+0.250, +1.625] | +14.70 [+6.05, +22.69] | +0.925 [+0.225, +1.600] | -3.71 [-5.10, -2.39] | -2634 [-3973, -1460] | **-2977 [-3209, -2721]** |
| `V4_supply_deadline` | +0.2525 [+0.0946, +0.4049] | +0.300 [+0.100, +0.475] | +0.494 [+0.346, +0.633] | +0.950 [+0.250, +1.625] | +14.70 [+6.05, +22.69] | +0.925 [+0.225, +1.600] | -3.71 [-5.10, -2.39] | -2634 [-3973, -1460] | -3010 [-3253, -2739] |
| `A-B2_coverage` | +0.0628 [-0.0286, +0.1609] | +0.075 [-0.025, +0.200] | +0.027 [-0.073, +0.119] | -0.300 [-0.801, +0.200] | +1.18 [-4.46, +6.78] | -0.350 [-0.800, +0.075] | -2.90 [-3.91, -1.93] | -2421 [-3275, -1634] | -41 [-145, +54] |

The `empty_plan_rate` benefit over V3 is -0.019 [-0.056, +0.000] for the protocol arms and +0.081
[-0.013, +0.188] for the supply arms: only the denser candidate ring reduces it, and even there the CI
includes 0 on validation.

Against the coverage sweep that gate I4 could not beat, on the primary metric: `V4_protocol_only`,
`V4_full` and `V4_deadline` +0.2406 [+0.1072, +0.3757]; `V4_route_cost` +0.2376 [+0.1032, +0.3744];
`V4_no_belief_map` +0.1961 [+0.0543, +0.3391]; `V4_supply` and `V4_supply_deadline` +0.1896
[+0.0309, +0.3475]; `V3_PRODUCTION` -0.0628 [-0.1609, +0.0286].

### 6.1 Applying the rule

All seven V4 variants clear the bar: every one of them beats V3 on the primary metric with a CI lower bound
above 0, and every one of them also beats coverage. The highest primary mean is an **exact three-way tie at
0.4971** between `V4_protocol_only`, `V4_full` and `V4_deadline`, which also tie on `defect_read` at 25 of
40. The third criterion, planning latency, gives 1447 ms, 1461 ms and 1463 ms, so the rule names
**`V4_protocol_only`**: the frozen V3 ranking rule with the V4 view execution protocol closed around it.

That tie-break has to be reported for what it is. Those three arms are **per-world identical on every
behavioural metric across all 40 validation worlds** (checked field by field: zero differences), because the
`conrad.active.v4` planner's two extra feasibility tests never fire on this family. `APPROACH_ABANDONED` has
nothing to refuse once a view in flight is protected, and `TIME_BUDGET_EXCEEDED` is never evaluated, because
this scenario declares no operator time budget and the mission-duration horizon of `V4_deadline` does not
bind (the planner's straight-line travel-time estimate is optimistic relative to the flown trajectory). The
latency difference between behaviourally identical arms is wall-clock noise measured under a twelve-way
parallel load, about 1 % of the mean. The rule was declared in advance and is applied as written, but what
was actually selected is the **execution protocol**, and `V4_full` and `V4_deadline` are the same mechanism
wearing a different planner object.

### 6.2 Variants that did not win, kept in the record

| variant | validation HSE | worlds read | why it was not selected |
|---|---|---|---|
| `V4_full` | 0.4971 | 25 | identical behaviour to the selected arm; lost the declared latency tie-break by 14 ms of wall-clock noise |
| `V4_deadline` | 0.4971 | 25 | identical behaviour to the selected arm; the mission-duration horizon never binds |
| `V4_route_cost` | 0.4941 | 25 | within noise of the selected arm; its effect is a ranking preference, which `ACTIVE-MCBR-E006` bounds at about +0.017 |
| `V4_no_belief_map` | 0.4526 | 23 | lower mean and two fewer worlds read; giving navigation the observed occupancy of the belief map is kept |
| `V4_supply` | 0.4462 | 22 | best on development (0.4764, 23 of 40) and best on candidate supply everywhere (12 NO_FEASIBLE_OBSERVATION plans against 28, mean feasible candidates 21.8 against 6.6), but it did NOT carry to validation on the primary metric and it costs about three times the planning latency (4539 ms against 1447 ms). A development-best arm that does not reproduce on validation is exactly what a validation split is for |
| `V4_supply_deadline` | 0.4462 | 22 | identical to `V4_supply` |

## 7. Freeze

`configs/active/mcbr_frozen_v4.yaml`, created after selection, SHA-256 of the file
`19a47687abf49ac56c28d02ec0cda8f38254fb7dfd781e4713f240899c0d2d75`. `configs/active/mcbr_frozen_v2.yaml`
and `mcbr_frozen_v3.yaml` are untouched on disk as historical evidence.

| section | digest | content |
|---|---|---|
| `planner` | `8eca16cc896e560b26cbe9814828fe9f75801d843e0e38bedb5442299805d903` | byte-identical to v2 and v3: `V-bayes_eig_ratio`. V4 does not re-rank views |
| `mission_predictive` | `2fa5b75d6e9618c0a1274eff284d014a30df4c0ba17a8d83969c2c8609d41431` | byte-identical to v2 and v3, `read_channel: false` |
| `view_execution` | `8eab12e582523ad1597f151abcd049dac4da8b709f56ba45b776b7b26950ab08` | NEW: the selected execution protocol |

`conrad.active.production.load_frozen` now verifies all three digests and raises `FrozenConfigError` on any
edit, and `runtime_config` adopts the frozen `view_execution` block whenever the planner is `PRODUCTION` and
the mission configuration does not declare one of its own. An experiment can still run the old protocol as a
control arm by declaring `view_execution` explicitly, which is what the V3 reference arms above do.

**The `FROZEN_PATH` ambiguity is now resolved and recorded.** `conrad/active/production.py` set
`FROZEN_PATH = configs/active/mcbr_frozen_v2.yaml` while the audit trail, the gate I4 run-2 harness and
`artifacts/gates/I4/unity_i4_occluded_rep2_results.json` all named `mcbr_frozen_v3.yaml`. No recorded result
was affected, because v2 and v3 have byte-identical `planner` and `mission_predictive` sections, so the
planner that flew was the planner the audit describes. `FROZEN_PATH` now points at v4, whose `planner` and
`mission_predictive` sections are byte-identical again, so `load_frozen()` still reports the same planner
digest inside every previously recorded artifact. What changes is that the pointer names the file that
actually describes production, and that file carries the execution protocol as well. The v4 file also
carries forward the `selection` block of v2 (the `ACTIVE-MCBR-SEL001` validation selection of the ranking
rule), which v3 had dropped.

## 8. What remains for the final run

Nothing in this work has touched `final_test` (8002200-8002259, 60 worlds) or `ood_test`
(8002300-8002319, 20 worlds). Both are still closed. The partition loader refuses them for any purpose other
than `final_evaluation`, and no run here used that purpose.

For the one final run, when you sequence it:

* **Arms.** The frozen production planner, which is now V4 (`planner: PRODUCTION` with no `view_execution`
  override, so the frozen protocol is adopted), against `A-B2_coverage`, and against the V3 protocol as a
  control arm (`planner: PRODUCTION` with `view_execution: {enabled: false}`), so the final run measures the
  comparison gate I4 measured plus the mechanism change.
* **Split.** `final_test`, 8002200-8002259, 60 worlds, chosen because run 2 measured a paired CI half width
  of about 0.09 at n = 60 and the per-world outcome is binary. OOD (`ood_test`, the `bent_pipeline` family
  under `I4-OCCLUDED-OOD`) is read in the same pass or not at all.
* **Primary metric and rule.** `hidden_state_error_improvement`, higher is better, paired on the world, 95 %
  paired percentile bootstrap with 4000 resamples, as in run 2. The question is whether the production stack
  now beats the systematic coverage sweep.
* **Expected effect and power.** The validation benefit over coverage is +0.2406 [+0.1072, +0.3757] and over
  the V3 protocol +0.3034 [+0.1808, +0.4279]. At n = 60 the measured half width is about 0.09, so an effect
  of this size is separable, unlike the +0.017 a re-ranking V4 would have carried.
* **Lane.** Development and validation ran on the python sim kernel. Whether the final is the python kernel
  or the Unity player is your call; the Unity player was not launched in this work.
* **Not yet done.** No final or OOD seed has been read, no gate evidence has been written, and gate I4's
  recorded verdict is unchanged.

Two limitations to carry into that run:

* the whole effect measured here is the execution protocol. The `conrad.active.v4` planner is implemented,
  tested and frozen-compatible, but its two extra feasibility tests never fired on this family, so nothing
  here is evidence for or against them;
* `ABANDONED_MISSION_END` is the failure mode V4 leaves standing: 9 of 58 accepted views on validation are
  still in flight when the 100 s clock runs out. The mission-duration horizon that would refuse them does
  not bind, because the planner's travel-time model is straight-line distance over `cruise_speed_mps` and is
  optimistic relative to the flown trajectory. Calibrating that model against the realised trajectory is the
  obvious next mechanism, and it was not attempted here.

## 9. Artifacts

| file | content |
|---|---|
| `configs/eval/i4_v4_development.yaml` | development round 1, seven arms |
| `configs/eval/i4_v4_development_r2.yaml` | development round 2, the mission-deadline variants |
| `configs/eval/i4_v4_validation.yaml` | the validation round, with the selection rule declared before the split was read |
| `artifacts/experiments/ACTIVE-MCBR-E007-DEV/i4_v4_development.json` | per world, per arm, with the execution breakdown and the paired CIs |
| `artifacts/experiments/ACTIVE-MCBR-E007-DEV2/i4_v4_development_r2.json` | the same for round 2 |
| `artifacts/experiments/ACTIVE-MCBR-E007-VAL/i4_v4_validation.json` | the validation round |
| `artifacts/experiments/ACTIVE-MCBR-E007-*/provenance.json` | config, commit, partition digest and wall clock of each round |
| `configs/active/mcbr_frozen_v4.yaml` | the frozen selection, SHA-256 `19a47687abf49ac56c28d02ec0cda8f38254fb7dfd781e4713f240899c0d2d75` |
| `conrad/orchestration/view_execution.py`, `conrad/active/v4.py` | the ledger and the V4 planner |
| `tests/unit/active/test_mcbr_v4.py` | sequence value, abandonment and re-planning, filter-before-rank, budgets, the no-truth boundary, empty-plan reporting, the v3 and v4 frozen configs |
| `tests/unit/orchestration/test_view_execution.py` | the ledger: abandonment is visible, and a view is credited only inside the declared pose AND aim tolerance |
| `tests/unit/orchestration/test_frozen_view_execution.py` | the frozen protocol is adopted by PRODUCTION, an explicit override wins, and an edited section fails closed |
| `scripts/run_mcbr_v4_experiment.py` | the runner (reads the commit from `.git/HEAD`, executes no git command) |
| `<run_dir>/mission/view_execution.json` | the per-view ledger of one mission |
| `<run_dir>/mission/runtime_metrics.json` | `view_execution`, `empty_plans`, `view_outcomes`, `deferred_replans`, per-plan feasible counts and planning latency |

`ACTIVE-MCBR-E007-DEV`, `-DEV2` and `-VAL` are registered in `conrad/evaluation/dispatch.py`. Every round ran
at commit `4ec7193a6c6f478469233619ed7fc14e8fa3845c`, on partition digest
`17490d3dbcc85673ebe624e6f04357f42b87a2316d431d2982af58e40e864457`, in 2689 s, 1216 s and 3332 s of wall
clock respectively.

## 10. Tests added

| test | what it pins |
|---|---|
| `test_sequence_value_of_overlapping_views_is_below_the_naive_sum` | the value of a sequence of overlapping views is not the sum of their values; the disjoint case is the sum |
| `test_a_view_that_is_never_reached_is_abandoned_with_a_visible_reason` | an accepted view that cannot be reached is closed with a reason code and appears in `abandoned()` |
| `test_unreachable_view_is_refused_with_a_visible_reason_and_a_different_view_is_planned` | the planner refuses that pose with `APPROACH_ABANDONED` and re-plans a different one |
| `test_an_observation_is_credited_only_inside_the_declared_pose_and_aim_tolerance` | at the commanded position but pointing away is reached and NOT flown |
| `test_empty_plan_is_reported_with_its_rejection_reasons` | a plan with no action is recorded with its reason codes instead of dropped |
| `test_extra_filter_runs_before_ranking_and_cannot_rescue_a_refused_candidate` | filter-before-rank holds with the V4 extra filter in place |
| `test_v4_modules_import_no_truth_side_package` | no Twin truth reaches runtime MCBR, checked on the AST of the new modules |
| `test_v3_frozen_config_still_loads` | the v3 frozen config still loads and its digest still matches |
| `test_v4_frozen_config_loads_and_fails_closed_when_edited` | an edited v4 frozen config is refused |
| `test_an_edited_view_execution_section_fails_closed` | the same for the new `view_execution` section |

`tests/leakage` passes unchanged, and the full suite is green: 1170 unit, contract and leakage tests, plus
198 acceptance, integration, regression, replay, simulation and property tests with 4 pre-existing xfails
(the recorded gate I5 and I7 failures) and 1 skip.
