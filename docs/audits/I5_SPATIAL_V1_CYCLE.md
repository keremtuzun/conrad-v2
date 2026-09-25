# I5 Spatial V1 cycle

Status: **DEVELOPMENT DECLARED; NO VALIDATION OR FINAL WORLD OPENED**.

The historical I5 formal result remains **FAIL, 9/10 criteria**. This is a
new versioned cycle after Spatial Structural Architecture V1 passed A-J and
was frozen. It does not rewrite or relabel any earlier evidence.

## Fixed architecture and semantics

The cycle uses the frozen `spatial_v1` Twin2T truth,
`RESOLUTION_CELL_SAMPLES`, pose-derived conservative visibility support,
`model2t-spatial-v1`, and spatial MCBR prediction. The sensor file is pinned
by SHA-256 `c7f68b08a008135d5a86961135c05930170c19f86d772b10094926f207877efe`;
the parsed sensor digest is
`034f949a14ded197ecca8cb1a8ab714b9872c98fac94a099414e77645fdd2104`.
All parameters remain `ENGINEERING_ESTIMATE` and `SYNTHETIC_ONLY`.

The legacy `I5-NOMINAL-READABLE` control is excluded and explicitly refused:
it uses `pristine_rest` and duplicated tile truth, which Spatial V1 forbids.
Ordinary `I5-NOMINAL` instead has an explicit healthy 2 by 4 local truth map.
Critical-finding and outage cases use a heterogeneous covered resolvable
corrosion map; other scenarios use healthy local truth. The warrant is
unchanged: critical component `OBSERVED INTACT` and nothing pending.

The required inspection domain and condition thresholds are the frozen
Spatial V1 values. Nominal duration is 480 simulated seconds because the
architecture acceptance mission first reached qualified intact at 291.3 s;
all other I5 scenarios retain 120 s. The longer duration is declared before
opening development and is not a threshold or warrant change.

## Partition and order

`configs/eval/partitions_i5_v8.yaml` is digest-pinned as
`f0a7007deda3470a742a13427ab3618a1c237d1712fcc11df48d55aa15752d29`.
Development is `8500000..8500009`, validation is `8500100..8500109`, and the
sealed final pool is `8500200..8500239`. The legacy fake-tile scenario is not
in the partition. Validation and final are not to be opened during mechanism
development.

The development screen starts with seeds 8500000 and 8500001 across the seven
legitimate action scenarios and all three frozen arms. It must demonstrate:

- frozen Spatial V1 truth and belief backends on every mission;
- at least one legitimate nominal intact warrant without nominal escalation;
- zero false intact results when a resolvable defect cell has measured support;
- no hard-constraint violation, complete traceability, and UIR zero; and
- no regression that makes another I5 action class structurally untestable.

The harness records an evaluation-only local truth map, support count and
digest, local coverage/beliefs, component condition, knowledge status, first
intact time, and defect-cell coverage. None of this truth data enters the
runtime decision path.

Validation, its selection rule, the warrant-incidence power calculation, final
N, minimum warrant count, and all final thresholds will be committed before
validation or final worlds are opened. If validation fails, final remains
sealed. If the one-shot surrogate final fails, no formal Unity run is allowed.

## Development iteration 1: failed attempt-budget reproduction

Seeds 8500000 and 8500001 completed all 42 missions (seven scenarios, three
arms) in 537.5 wall seconds. Spatial truth/backend selection and the frozen
sensor digest matched in every row. The six non-nominal action scenarios were
2/2 correct given warrant, hard violations were zero, traceability was 1.0,
UIR was 0, nominal over-escalation was zero, and every covered resolvable
defect stayed non-intact. Ordinary nominal nevertheless reached zero intact
warrants: both 480 s worlds ended `UNKNOWN` with incomplete required-cell
coverage. This is a **failed development hypothesis**, not gate evidence.

Fresh diagnostic development seed 8500002 reproduced the cause. It executed
only four views, then the default `max_plans_per_need=4` marked acquisition
unavailable while required cells remained incomplete. The architecture's
recorded successful 480 s development mission used fifteen views after its
earlier investigation had declared a larger Model1/MCBR attempt budget. The
I5 profile had accidentally inherited the generic four-plan default instead
of carrying that known spatial mission requirement forward.

Iteration 2 therefore changes only finite software acquisition capacity:
`max_plans_per_need`, Model1 `max_information_attempts`, and
`max_view_attempts` are each prospectively set to 24. Sensor, support,
detectability, truth maps, required inspection domain, condition thresholds,
warrant, baselines, mission durations, and scoring are unchanged. Fresh
development seeds 8500003 and 8500004 are used; iteration-1 seeds are not
rerun. Validation and final remain unopened.

The immutable R1 result is
`artifacts/experiments/M1-ACTION-SPATIAL-V1-DEV/m1_action_spatial_v1_dev_development.json`
with SHA-256
`5bf6b83bf80e1b6fc0bd645174246d24fc9c3f2bdd9c47c035844ae82a534b0b`.

## Development iteration 2: mixed reachability, competitiveness fail

Seeds 8500003 and 8500004 completed all 42 missions in 537.4 wall seconds.
One of two primary nominal missions reached qualified `OBSERVED INTACT` at
254 s with all four required cells at coverage 1.0 and selected CONTINUE
within budget. The other ended `UNKNOWN`: three required cells were complete,
but one remained at 0.841 coverage. Every non-nominal action was 2/2 correct
given warrant, violations and nominal over-escalations were zero, traceability
was 1.0, UIR was 0, and covered resolvable defects again had zero false intact.

The frozen competitiveness criterion nevertheless failed: EGDC had 8 safety
events versus the rule baseline's 6. The largest path-specific differences
were one extra nominal event and one extra outage event; no hard decision
constraint was violated. R2 is therefore development evidence, not a selected
candidate or gate result. Its result SHA-256 is
`d531d8ab84e82b63d9fc76453f8d54703a27e07e6db5814acb2a2225107a65eb`.

Iteration 3 is an unchanged-candidate replication on the five remaining fresh
development seeds 8500005..8500009. No parameter, threshold, scenario, arm,
duration or decision rule changes. It must retain zero false intact and the
complete action/grounding results while determining whether nominal warrant
incidence and the R2 safety comparison generalize. Validation and final remain
unopened.

## Development iteration 3: mechanism retained, boundary-feasibility defect found

Seeds 8500005..8500009 completed all 105 missions in 757.6 wall seconds. Four
of five primary nominal missions reached a legitimate intact warrant and were
correct given warrant. Every non-nominal action scenario was 5/5 correct,
violations and nominal over-escalations were zero, traceability was 1.0, UIR
was 0, and covered resolvable defect cells again produced zero false intact.

The unchanged candidate nevertheless failed the competitive criterion. EGDC
and the rule baseline each recorded 216 safety-state entries, versus 166 for
the naive baseline. Seed 8500008 contributed 208 entries in the nominal
mission and never reached an intact warrant; all other primary nominal worlds
recorded at most one safety entry and reached intact. The immutable R3 result
is
`artifacts/experiments/M1-ACTION-SPATIAL-V1-DEV-R3/m1_action_spatial_v1_dev_r3_development.json`
with SHA-256
`971ac50f503a1e8a0e06e2dc5848b6c66d0afa553f0346b4af5ceb4c3bcda942`.

A retained diagnostic rerun of seed 8500008 established that this was not a
collision: truth recorded zero vehicle collisions and minimum clearance was
0.969 m. The eleventh selected MCBR candidate placed the sensor boresight at
z=4.825 m, but routing applied the declared payload mount transform and
commanded the vehicle at z=4.965 m inside a mission whose upper boundary is
z=5.0 m. The planner's feasibility checks considered the sensor pose, did not
declare the mission bounds on the `PlanningRequest`, and did not reserve the
safety supervisor's uncertainty margin for the transformed vehicle goal. As
the estimator covariance changed, the supervisor correctly alternated between
motion and `MISSION_BOUNDARY_RISK` HOLD; the view timed out and the required
cell remained incomplete.

R3 is therefore a failed development hypothesis and exposes a software
contract gap between active-view feasibility and navigation safety. The next
candidate must reject any sensor pose whose corresponding vehicle pose and
current uncertainty ball do not fit inside the declared mission boundary.
That correction must be covered by a focused regression and evaluated on a
new prospectively declared partition. No v8 validation or final seed has been
opened.

## Boundary-feasibility correction

The active-planning request now carries the declared mission bounds and the
current EKF position uncertainty. Candidate filtering applies the navigation
safety supervisor's configured `boundary_sigma_k` margin to the vehicle pose
that routing will actually execute after the payload mount transform. Missing
pose uncertainty fails closed. Navigation independently refuses every planned
route whose waypoints do not contain that same uncertainty ball, including
belief-map detours created after goal admission.

Focused active-planning, navigation, orchestration, and decision tests pass
(252 tests). A diagnostic replay of the already-spent seed 8500008 is not new
selection evidence, but confirms the repaired mechanism: the former z=4.965 m
vehicle goal was filtered, maximum commanded z was 4.643 m, every one of 24
selected views completed, and scored HOLD entries fell from 208 to 2. The
world still did not reach intact, so no success is inferred from this reused
seed. A fresh partition is required for candidate assessment.

## Version 9 prospective corrected-candidate cycle

Before opening any corrected-candidate world,
`configs/eval/partitions_i5_v9.yaml` declares development
8600000..8600009, validation 8600100..8600109, and a sealed 40-world final
pool 8600200..8600239. Its canonical digest is
`c5b890be75e80872d3ccc8c6f730b567bba6449d0f7ba3f743cf41e1c97ff8fc`.
The future formal Unity range 8601000..8601999 is separately reserved.

R4 will open only the first five development worlds, 8600000..8600004. The
candidate advances only if every covered resolvable defect remains non-intact,
every non-nominal action class is correct given warrant, decision violations
and nominal over-escalations are zero, traceability is 1.0, UIR is 0, at least
three of five nominal worlds reach a legitimate intact warrant, and aggregate
task success, safety entries, and decision violations are no worse than both
frozen baselines. Any safety outlier is inspected before validation. Sensor,
truth, support, Model2T, warrant, thresholds, scenarios, arms, durations,
attempt bounds, and scoring remain identical to R3. Validation and final stay
sealed during R4.
