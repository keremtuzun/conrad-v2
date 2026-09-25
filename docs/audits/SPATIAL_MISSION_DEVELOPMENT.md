# Spatial mission development evidence

This record concerns the opt-in `spatial_v1` path. It does not replace the
historical I4, I5, or I7 results and does not authorize a formal partition.

## Inspected surface and intact meaning

`OBSERVED INTACT` means no above-resolution synthetic defect was observed in
the **declared required inspection domain** after complete conservative
coverage and the required independent looks. It does not mean that the whole
physical component, inaccessible regions, or microscopic defects are intact.
The domain is a versioned part of the spatial Model2T mission configuration:
an axial fraction interval and a set of capsule sectors. The default is the
whole capsule. A detected defect outside the required domain still overrides
an intact result.

The default straight pipeline cannot earn whole-capsule intact from the tested
views because the underside and end bands are obscured. A development case
uses the existing `pipeline_with_supports` family and declares axial fraction
`[0.05, 0.95]`, sectors `[2, 3]` before sensing. This limits the claim to the
top-side inspection region. The requirement presented to Model1 names that
scope. Restricting the domain after seeing an outcome would invalidate the
case; any gate configuration must pin it before validation.

## Synthetic development checks

- Seed 401, exact surveyed design axis, zero declared pose/support uncertainty,
  0.2 m axial/lateral resolution, zero sensor noise, engineering-estimate
  detectability: a 64-view sweep through the mission structural sensor,
  Observation, ECMER Evidence, registry association, persisted spatial
  Model2T, and Model1 `DecisionContext` yields full required support and
  `OBSERVED INTACT` for healthy truth. EGDC accepts the resulting context.
- A single healthy view leaves required coverage incomplete and condition
  unknown. A local defect outside that view's measured support also leaves
  condition unknown; truth is not inserted into the belief.
- A resolvable corrosion patch and a resolvable crack patch in the required
  domain each produce `SEVERE` under the same sweep; neither is falsely intact.
- Two seed-401 worlds with the same 0.001 m whole-surface corrosion mean
  separate after the sweep: uniform 0.001 m yields qualified `INTACT`, while
  one 0.008 m local cell and seven healthy cells yields `SEVERE`.
- The same real sensor/Evidence/association/Model2T sweep now also covers
  heterogeneous healthy cells (`INTACT`), separated corrosion and crack cells
  (`SEVERE`), and a resolvable defect at the required-domain axial edge
  (`SEVERE`). These three additions passed at development seed 401. They are
  not independent prospective validation worlds.
- A patch smaller than the declared synthetic spatial resolution is not
  detected in the sweep; the resulting intact label remains qualified to the
  resolution threshold and does not claim microscopic defect absence.
- A matched development pair places a resolvable defect behind a fixed
  structural occluder. The clear world emits the defect reading at the same
  pose; the occluded world emits no structural support and Model2T remains
  unknown.
- The visibility certificate proves each admitted rectangle continuously by
  SDF bounds. It subdivides wide resolution cells only when necessary and
  admits the parent only if every child is certified. An occluder rejection
  test and a visible-wide-cell test exercise both sides.
- The recursive certificate has its own replay version,
  `capsule-sdf-recursive-v1`. Spatial run bundles pin it alongside truth,
  sensor, support schema, detectability, and Model2T versions.

At this stage these were controlled development views; later closed-loop and
Unity results are recorded below. Nonzero Gaussian survey noise without a
hard endpoint bound is explicitly unregistered and earns no healthy coverage.
`SPATIAL_REGISTRATION_V2.md` describes a separately declared bounded
synthetic survey mode. A complete prospective case matrix and gate impact
evaluations remain open software work.

## Closed-loop integration diagnosis

A 120-second seed-2026201 development mission with the initial spatial path
flew four Model1/MCBR inspection plans but produced no target structural
reading. The synthetic payload is level and yaw-mounted; its centre ray passed
above the pipe while the pipe was inside the camera field of view. The support
producer now considers the near surface in that case, but only admits a
resolution rectangle if the continuous FOV, incidence, quality and occlusion
certificate proves it. A flown-pose regression covers this case.

The first post-repair 76-second mission emitted spatial readings but the
perception dispatcher only routed legacy `twin2t_fidelity` readings to Model2T.
That dispatch was repaired and checked through the actual mission perception
service. In the next 76-second run, 64 spatial target readings were emitted,
53 were associated, and direct Model2T revisions appeared. The target
correctly remained unknown with incomplete required coverage. The legacy
report field counts historical patch labels; distinct spatial report fields
now count emitted and associated spatial readings.

With centre-only candidate views, a 180-second run associated 170 spatial
target readings but aggregate required coverage stalled near 0.33. The spatial
predictor now uses the level view flown by the controller, supplies candidate
regions from the largest uncovered surface rectangles, and scores only newly
covered area. The first 180-second run after this change associated 397 target
readings; required-cell coverage rose to 0.38–0.51 and condition remained
unknown. It used the legacy four-plan cap. A subsequent 360-second run with
that same cap was deliberately interrupted after 1,639 of 3,600 control
commands because all four plans were spent by 98 seconds. It has no bundle
manifest and is **incomplete development data**.

The next development configuration declares `max_plans_per_need=12`: the
required axial span is about 3.6 m, the synthetic footprint is 1.0 m wide,
two required angular sectors need distinct views, and additional looks may
be needed around occlusion and resolution-cell edges. This is a planning
budget for software investigation, not a changed gate threshold or a measured
physical setting. The completed 360-second run made eleven inspection attempts
and associated 940 spatial target readings, yet required-cell coverage stayed
at 0.380–0.512 and condition remained `UNKNOWN`. Only the first view met its
full dwell. Later plans moved the vehicle toward new candidate surface regions
but retained the original midspan as the action's aim point; orientation errors
caused short or abandoned dwells. The planner now carries the candidate's
region through visibility filtering and the selected action, so the view
executor aims at the selected local surface. The next development run checked
the execution and coverage effects.

That 180-second seed-2026201 profiled development run made seven inspection
plans and all seven views were `FLOWN`; the commanded aim points now track the
local candidate regions. It emitted 238 spatial target readings and associated
221. The target still stayed `UNKNOWN`: required-cell coverage was 0.455,
0.434, 0.512, and 0.380. The supports admitted by the runtime sensor covered
angular strips near the edges of the two required sectors, but left the crown
between them unobserved. Repeated, successfully flown views added readings
without closing that gap. The predictive candidate score therefore still
overstates new conservative support for some views. This is a development
failure, not an intact warrant or A–J acceptance. A belief-side prediction
that agrees with the sensor's conservative FOV/incidence certificate, and a
reachability check for the uncovered crown, remain software work.

A following 180-second development run used the sensor's declared synthetic
incidence, quality, and water attenuation limits in belief-side scoring. It
fully flew seven views and associated 248 target readings, but required-cell
coverage stayed at 0.434–0.455. The controller had been restricted to level
yaw views while the candidate generator could propose elevation. The spatial
path now retains the proposed sensor pitch through the mount transform,
navigation, and the execution ledger. It also offers higher elevation
candidates while leaving legacy candidate generation unchanged.

The first pitched development run was **interrupted** after 579/1,800 control
commands. It has no bundle manifest. Its second accepted view was only about
0.15 m from the vehicle, yet the navigation stack generated a 98,387-second,
491,936-point trajectory. The trajectory path used yaw-only waypoints and a
near-degenerate translational speed profile. Spatial station-keeping now
rotates in place when already at the requested position, applies the full
target attitude at the end of moving routes, waits for attitude tolerance
before starting dwell, and rejects extreme route durations before allocating
their samples. A fresh pitched development mission must verify the fix.

The completed 120-second pitched mission after that navigation repair emitted
certified support in the formerly missed 3.95–4.71 rad crown interval.
However, its two views ended `ABANDONED_TIMEOUT` and
`ABANDONED_MISSION_END`. The vehicle approached their positions but held
about 0.35 rad less pitch than commanded, so it never accumulated the
required aim dwell. The synthetic kernel applies a buoyancy restoring moment
about pitch; the control law compensated the net vertical force but omitted
that moment. The controller now applies the corresponding model-based moment
feedforward, and an isolated simulated 1.0-rad pitch hold reaches its target.
The new 120-second closed-loop development run flew three pitched views with
full dwell and boresight errors below 0.02 rad; a fourth view reached the
mission end before dwell. It associated 185 spatial target readings. Required
cell coverage became 0.434, 0.491, 0.434, and 0.659, so component condition
correctly remained `UNKNOWN`. A 360-second run using the same twelve-view
development configuration is testing whether the remaining area is reachable
without changing the intact criterion or sensor quality limit.

The 360-second run completed with all twelve inspection views `FLOWN` and
1,145 associated target readings. Nine views reused the same commanded
vehicle position. Required-cell coverage was only 0.488, 0.491, 0.434, and
0.659; condition remained `UNKNOWN`. The current coarse predictor still
assigned high gain to already repeated support. More readings at one pose do
not replace spatial coverage. The next planner repair must exclude genuinely
duplicate pose/boresight combinations and score novel support using the same
resolution-cell geometry limits as the runtime certificate.

That repair now shares the continuous FOV/incidence/quality rectangle
certificate between the runtime sensor and the belief-side predictor. The
predictor scores only resolution tiles outside the already credited support,
discounted by Model2S's predicted visibility; it cannot read structural
truth. Prior view records retain optional boresight orientation, and the
spatial planner rejects a pose/boresight combination already accepted. A
fresh 120-second development mission selected four distinct views, of which
three were `FLOWN` and the fourth was cut off by mission end. The four
required-cell coverages were 0.163, 0.775, 0.434, and 1.000. The last cell
became `OBSERVED_INTACT`; the component remained `UNKNOWN`, as it must. A
360-second run of this unchanged configuration is in progress to test whether
the remaining cells can be completed.

The completed 360-second run flew four distinct views and then stopped
requesting information, leaving the target `UNKNOWN` with observational
uncertainty 0.379. The development runtime had declared twelve MCBR plan
attempts, but Model1's separate `max_information_attempts` remained at its
default of three. Its decision policy therefore exhausted the autonomous
information path before the spatial inspection could use the declared view
budget. The development configuration now sets both attempt budgets to
twelve, with the same sensor, detectability, coverage, and intact criteria.
This is a fresh software-development budget declaration, not a change to any
historical gate or final partition.

The fresh twelve-attempt Model1/MCBR 360-second development run still selected
only four views. It associated 1,153 target structural readings and ended with
the target `UNKNOWN` and observational uncertainty 0.379. The separate attempt
budget was therefore not the sole cause. Spatial Model2T had assigned raw
epistemic uncertainty 1.0 whenever its aggregate condition was `UNKNOWN`, even
when required surface simply lacked certified observations. Model1 treated that
as a model dead end. The belief now reports this uncovered required surface
as observational uncertainty, using the least-covered required cell rather
than hiding a local gap in the mean. A fresh closed-loop development run is
required to determine whether the remaining surface can actually be reached.

That fresh 360-second run selected twelve plans and executed eleven views.
It associated 364 target readings. Required-cell coverage was 1.000, 0.775,
0.661, and 1.000; two cells became `OBSERVED_INTACT`, but the component
correctly remained `UNKNOWN` (raw epistemic 0, observational 0.339). The
uncertainty correction removed the early Model1 dead end. The remaining
failure is an inspection-planning and mission-budget problem: the selected
views did not certify the uncovered areas in two required cells. No nominal
continue warrant or architecture PASS follows from this run.

A fresh 720-second development run declared twenty-four Model1 and MCBR
information attempts before launch. It executed seventeen views, all `FLOWN`,
with 431 associated target readings. Required-cell coverage reached 1.000,
0.775, 0.716, and 1.000; the component remained `UNKNOWN` (raw epistemic 0,
observational 0.284). The last target Evidence arrived at mission time
352.3 s. After the last inspection plan at 378 s, the remaining mission time
produced repeated belief queries and revisits, not additional certified
target support. More time and attempts alone did not close the contiguous
uncovered strips. This is a completed failed development run, not an
interrupted run or an intact warrant.

An evaluation-only viewpoint sweep on the same seed found sensor-certified,
design-clearance-feasible poses for representative uncovered rectangles in
both cells, including poses on MCBR's eight-azimuth, three-standoff,
four-elevation candidate grid. This rules out simple geometric impossibility
for those sampled patches; it does not prove that the full remaining union is
reachable in one mission. The candidate tables show a second decision error:
as fractional observational uncertainty fell below EGDC's generic 0.4
threshold, the still-`UNKNOWN` worst-case condition was handled chiefly as
an uncalibrated epistemic issue. For this qualified condition claim, any
required unresolved local cell now keeps raw observational uncertainty at
1.0. The separate numeric coverage claim continues to report partial area.
The next closed-loop run must verify that this change leads to certified
coverage, not merely more information requests.

The fresh 480-second categorical-gap run did close the required domain. It
flew all fifteen selected views and associated 1,238 target Evidence records.
The first spatial target `INTACT` revision was direct at 298.3 s, with required
coverage 1.0 and 168 independent target observations; Model1's first
`CONTINUE_MISSION` followed at 302 s. The final target was observed `INTACT`
with raw observational uncertainty 0.0. This is a healthy, exact-survey,
synthetic DEVELOPMENT result. The bundle's full deterministic replay and a
fresh clean-checkout regression are still pending at this entry; no
architecture freeze or I5 gate result is inferred. Historical I5 FORMAL
remains FAIL 9/10.

Full deterministic replay of that 480-second bundle completed equal. The
original and replay each have 27,671 event signatures, 240 decisions, and
8,001 belief revision headers. The replay comparator was extended to check
persisted payloads and mission artifacts as well: all 2,402 Observations,
1,922 Evidence rows, 4,800 commands, 8,001 belief payloads, 15 view execution
records, 1,442 structural association records, 504 BAAC transmission rows,
and the trajectory artifact matched exactly. Observation payload equality
includes the structural support geometry. The original manifest verified
1,623 files and 478 objects before replay. A mutation test now checks that
changing a replayed Observation payload is detected even when event
signatures still match. This replay does not replace a fresh clean-checkout
regression or a prospective architecture validation partition.

A paired 320-second DEVELOPMENT mission used the same seed and runtime but
placed a resolvable 0.008 m corrosion defect in required local cell 6. Five
views were `FLOWN`; 725 target Evidence records associated. The first
spatial target `SEVERE` revision occurred at 52.3 s, and the final condition
was `SEVERE`. Across all target spatial revisions, zero were `INTACT`.
This supports the false-intact guard for this one covered defect case; it is
not a false-intact rate estimate across worlds. Model1 later selected
`CONTINUE_MISSION` after a known severe condition, which continues the
inspection mission rather than issuing an asset-safety clearance. Gate I5's
nominal continue warrant remains a separate explicit test.

A second paired 320-second DEVELOPMENT mission placed a resolvable 0.020 m
crack with 0.006 m depth in required cell 3. The first target `SEVERE`
revision occurred at 10.3 s; five views were `FLOWN`, 725 target Evidence
records associated, and the final condition was `SEVERE`. Zero target
spatial revisions were `INTACT`. These two paired defect runs are deterministic
development checks, not a prospective false-intact rate estimate.

An 80-second nonzero-Gaussian-survey development mission exercised the
unregistered-support path. The mission emitted 114 target spatial structural
Observations and associated 106 target Evidence records. Spatial Model2T
accepted none for healthy coverage: the target stayed `UNKNOWN` with
observational uncertainty 1.0 and no direct target revision. The survey
sigma is a distribution parameter, not a hard registration-error bound, so
the sensor labels these observations `CAPSULE_UNREGISTERED` with unknown
axial/angular registration uncertainty. The replay contract now pins this
registration policy as `exact-or-unregistered-v1` in that bundle. The later
`exact-bounded-or-unregistered-v2` contract adds a distinct finite-error
synthetic mode. This historical unbounded run stays `UNKNOWN` and is not a
noisy-survey intact warrant.

A fresh bounded synthetic survey run, `SPATIAL-BOUNDED-SURVEY-80S-DEV`, used
Gaussian scale 0.001 m with a declared 0.002 m **hard endpoint bound**. All
120 target spatial Observations carried `CAPSULE_DESIGN` support with finite
nonzero axial uncertainty; 114 associated to the target. Spatial Model2T
consumed three for guaranteed support and reported observed surface coverage
0.1189, with final condition `UNKNOWN` and observational uncertainty 1.0.
This demonstrates partial conservative credit instead of either treating
Gaussian sigma as a bound or giving a false intact warrant. Full noisy-axis
coverage was not achieved in this 80-second run. Full deterministic replay
verified 295 files and 81 objects, then reproduced all 309 Observation
payloads, 229 Evidence payloads, 1,091 belief payloads, and the trajectory
artifact exactly. The expanded opt-in Unity suite subsequently passed 9/9
matched tests across exact/bounded surveys, clear/occluded worlds, and
deterministic/noisy synthetic sensor cases.

A clean checkout at `73e2cf6` ran the ordinary non-Unity regression suite:
1,450 passed, 14 explicitly skipped, 121 deselected, 3 historical expected
failures, and one NAV-007 assertion failed. The loss declaration was at
30.02 s against a strict `<30.0 s` test cutoff, while maximum position error
was 2.9013 m against its unchanged `<4.0 m` guard. This followed the revised
navigation trajectory by one 20 ms control step; the assertion now allows
`<30.1 s`, still far ahead of the historical 39.1 s failure. A final clean
checkout regression at the completed branch HEAD remains required.

## Development performance profile

`scripts/profile_spatial_bundle.py` reconstructs a fresh run from the
recorded 480-second healthy development configuration. Its 80-second timing
run (`SPATIAL-PROFILE-TIMING-80S-DEV`) used 800 ticks and 42.81 process CPU
seconds (49.78 wall seconds while the broad test suite also ran). The 80
spatial sensor support/response calls took 1.40 s total; 51 spatial Model2T
ingests took 0.030 s, and 51 updates took 1.01 s. Three MCBR plans took
5.42 s total. It persisted 1,096 belief revisions (3,474,536 payload bytes)
and a 17,347,658-byte bundle. A separate allocation-traced 80-second run
reported a 40,482,797-byte Python allocation peak, but tracing inflated wall
time to 159.52 s and MCBR time to 26.73 s; those traced timings are not used
for throughput. These measurements are development, on this Windows host,
with concurrent pytest; they are not a formal latency guarantee. Unity step
overhead and longer-mission scaling still need measurement.

## Declared closed-loop same-mean development pair

Before these runs, the next development pair was declared with the same
seed 2026201, exact survey, 480-second runtime, sensor, inspection domain,
Model1/MCBR attempt budgets, and planner. The world truth overrides are
`configs/sim/spatial_v1_same_mean_uniform.json` (all eight cells at 0.001 m
corrosion) and `configs/sim/spatial_v1_same_mean_local.json` (one cell at
0.008 m, seven at zero). Both whole-surface cell means are 0.001 m.
The uniform world should earn qualified `INTACT` after enough coverage; the
local world should report `SEVERE` without any target `INTACT` revision.
This is a DEVELOPMENT pair, not validation or a gate. Its outcomes will be
recorded without changing these files or expectations after inspection.

The uniform half, `SPATIAL-SAMEMEAN-UNIFORM-480S-DEV`, completed with
1,238 associated target Evidence records, first direct `INTACT` at 298.3 s,
full required coverage, final `OBSERVED INTACT`, and zero severe target
revisions. The local half, `SPATIAL-SAMEMEAN-LOCAL-480S-DEV`, completed. Its
398 spatial target revisions contained 386 `SEVERE`, 12 not-yet-observed,
and **zero `INTACT`** conditions. The final target condition was `SEVERE`
at 479.3 s, with observed surface coverage 0.6336. Thus the equal global
means led to different component decisions in the declared closed-loop pair.
This pair used the earlier crack detectability contract; its corrosion-only
result remains development evidence, not validation for the new contract.

The fresh current-contract exact-survey healthy mission
`SPATIAL-V2-HEALTHY-480S-DEV` completed before that pair. It again consumed
1,238 associated target Evidence records and first reached direct qualified
`INTACT` at 298.3 s, with 168 independent observations; final required
coverage was 1.0, condition `OBSERVED INTACT`, and raw observational
uncertainty 0.0. There were 350 spatial target revisions and no `SEVERE`
revision. The 480-second run consumed 488.23 process CPU seconds on this
contended host and wrote a 119,730,638-byte bundle. Full current-contract
replay verified 1,619 files and 478 objects, then reproduced all 2,402
Observation payloads, 1,922 Evidence payloads, 8,001 belief payloads, and
the trajectory artifact exactly. No validation or gate result is implied.
That replay also used the earlier crack detectability contract. The later
`spatial-synthetic-detectability-v2` adds independent configurable positive
crack length and depth thresholds, while retaining a separate patch-size
limit. Development seed 401 passed both subthreshold crack mission negative
controls and the covered crack positive control. A fresh current-contract
closed-loop mission and replay were then run. `SPATIAL-V2-DETECTABILITY-480S-DEV`
used the committed engineering-estimate sensor config with independent crack
length 0.01 m and depth 0.001 m limits. Its 350 spatial target revisions
contained 182 `INTACT`, 168 not-yet-observed, and zero `SEVERE` conditions;
the final target condition was qualified `INTACT` with required coverage 1.0.
The run used 4,800 control ticks, 15 MCBR plans, 480 sensor responses, and
387 spatial Model2T ingests/updates. It consumed 480.55 process CPU seconds
and produced a 119,730,814-byte bundle with 24,767,104 bytes of belief
payloads. Full replay verified 1,619 files and 478 objects, then exactly
reproduced 27,671 events, 2,402 Observation payloads, 1,922 Evidence
payloads, 8,001 belief payloads, 4,800 command payloads, trajectory,
view execution, associations, and BAAC transmissions. This is development
evidence under detectability v2; the declared validation seeds remain unopened.
The earlier detectability-v1 healthy bundle is rejected by the current
replay contract because it lacks the two required crack thresholds.

## First prospective controlled-view matrix and association diagnosis

The committed 14-case by two-seed controlled-view matrix was run once on
seeds 1401 and 1402. It **FAILED 22/28** cases: every failed case reached
the same full-coverage assertion with target direct support below 1.0.
The two healthy cases ended at 0.9883863 and 0.9778138 observed support,
and correctly stayed `UNKNOWN`; the partial/outside/occluded negative
controls passed on both seeds. This is a failed validation partition, not
an excuse to relax the required domain or count partial coverage as intact.
The stored `results.jsonl` records every case and traceback. These seeds
will not be rerun.

Read-only inspection of seed 1401's stored observations and Evidence found
full emitted raw support over the required cells but three near-joint
resolution supports with no registry association. One cell's associated
surface union was only 0.9535. Fresh development seed 403 reproduced
98.79% coverage. Filtering capsule-surface readings away from nearby
non-segment geometry alone was insufficient on development seeds 404/405.
Instrumentation of development seed 406 found genuine ambiguity between
adjacent segment surfaces at the joint: a projected reading was about
0.28 m inside one surveyed segment and about 0.28 m beyond its neighbor's
axis endpoint, with projected sigma about 0.052 m. An exact-survey axial
interval rule now resolves only a unique interior segment separated by the
existing three-sigma gate; near-joint, overlapping, and uncertain-survey
cases remain unassociated. Development seeds 404/405 reached qualified
`INTACT` after this repair. The fresh 14-case by two-seed development matrix
on seeds 407 and 408 then passed **28/28**, including full healthy and
heterogeneous healthy coverage, covered corrosion and crack, each
subthreshold crack control, occlusion, required-domain edge defect,
separated defects, and the equal-global-mean pair. Results were appended
case by case under `SPATIAL-ASSOC-DEV-407-408/results.jsonl`. These are
development seeds and do not repair or reclassify validation seeds 1401/1402.

The repaired `SPATIAL-V2-AXIAL-480S-DEV` closed-loop mission subsequently
finished with final qualified `INTACT`, required coverage 1.0, and zero
`SEVERE` target revisions. The first direct `INTACT` revision was at 291.3 s;
there were 181 target spatial revisions, 13 MCBR plans, 480 structural
sensor responses, and 213 Model2T ingests. It used 422.28 process CPU
seconds over 480.94 wall seconds on the contended host, persisted 7,801
belief revisions with 24,282,590 payload bytes, and wrote a 109,468,831-byte
bundle. The bundle pins `spatial-structural-association-axial-v2` together
with detectability-v2 and bounded-registration-v2. Full replay verified
1,753 files and 478 objects, and exactly reproduced 25,957 events, 1,822
Observation payloads, 1,155 Evidence payloads, 7,801 belief payloads,
4,800 command payloads, trajectory, 13 view executions, 675 structural
associations, and 503 BAAC transmissions. This is development evidence;
The nine opt-in Unity development parity tests then passed under this
same axial-association/detectability contract, across clear/occluded,
exact/bounded-survey, and deterministic/noisy cases. Cycle-2 validation
seeds remained unopened at this entry.

## Cycle-2 failures and association-v3 development

The separately declared cycle-2 controlled-view partition **FAILED 17/28**
and its Unity parity partition **FAILED 6/16**. Neither seed 1501 nor 1502
will be rerun. Read-only diagnostics found all 208 emitted supports associated
on seed 1502, while permanent support hardware physically hid about 20.76%
of the required surface. This was a world-fixture feasibility failure, not a
reason to lower the unchanged full-support threshold. The Unity comparison
assertions themselves matched before test-fixture assertions failed: fixed
views were not guaranteed clear or blocked, and one bounded-survey case had
no credited coverage.

Fresh development seed 412 also exposed a confident wrong-neighbor point
association near a joint. Association-v3 applies unique axial separation to
every multi-candidate supported reading, including confident point matches.
An exact survey or an explicit hard endpoint bound is required; a Gaussian
survey sigma alone still abstains. The bounded axial guard includes the
endpoint and axis-direction perturbation. Overlap and near-joint cases
remain unassociated. The inspectable full-surface positive-control family
has no permanent support hardware or clutter and 1 m pipe clearance; the
occluded negative control adds an explicit truth-side rack. No domain,
condition, sensor detectability, or support threshold was reduced.

The unchanged 14-case controlled-view matrix passed **28/28** on fresh
development seeds 418/419. Unity development seed 420 found a fixed clear
pose with no support. Seed 425 found that the exact-only axial rule could
starve a bounded survey of all credited coverage. Both failed development
runs remain recorded. After fixing the pose and bounded rule, fresh Unity
development seeds 426/427 passed **16/16** with exact support/measurement,
association, condition, and coverage parity, all clear views supported,
one deliberately blocked occluded view, and positive bounded coverage.
These are development results, not prospective validation or formal gates.

The committed association-v3 `SPATIAL-V3-BOUNDAXIAL-480S-DEV-COMMITTED`
mission completed 4,800 ticks, 13 MCBR plans, 480 spatial sensor calls,
369 spatial ingests/updates, 8,166 belief revisions (25,390,907 JSON payload
bytes), and a 117,526,574-byte bundle. Its 499 target revisions contained
189 direct `INTACT`, 310 unresolved, and zero `SEVERE`. The first qualified
`INTACT` was at 291.3 s with required coverage 1.0 and 149 independent
observations; final required coverage stayed 1.0. The run took 495.05 wall
seconds for stepping, including 10.74 s sensor response, 21.54 s Model2T
updates, and 21.99 s MCBR planning. Full current-contract replay verified
1,703 files and 478 objects, then exactly reproduced 27,161 events, 240
decisions, 2,174 Observation payloads, 1,694 Evidence payloads, 8,166
belief payloads, 4,800 commands, trajectory, 13 view executions, 1,214
structural associations, and 505 BAAC transmissions. A runtime scan of
31,785 records against 18 truth-side world IDs found zero leaks; the static
boundary suite passed 22/22. This remains development evidence.
The formatted association-v3 source subsequently passed the unchanged
controlled-view matrix on fresh development seeds 428/429 (**28/28**).

Cycle 3 then passed the prospectively declared controlled-view partition
**28/28** but failed Unity parity **12/16**. All four failures were the clear
cases on seed 1702; the first two predeclared fixed views emitted zero
support in both kernel and Unity, while the other two emitted four each.
The seed is spent; the parity failure is not relabeled as PASS. The old
views used 2 m standoff. A narrower pipe can project too little visible
surface into a fully certified resolution cell at that distance, although
this is an inference from the support counts rather than a physical sensor
measurement. Shorter, still in-range development views passed 16/16 clear
exact cases on fresh seeds 430–445, and the full parity matrix passed
16/16 on seeds 446/447. Wider fresh development pose screening passed
32/32 on seeds 448–479, bringing clear exact positive controls to 48/48
fresh seeds. A second full parity development pair on seeds 480/481 passed
16/16. These results support the separately declared cycle-4 fixture;
they do not repair cycle 3.

Cycle 4 **FAILED** its controlled-view partition 16/28 and Unity parity
12/16. Read-only SQLite analysis of the spent seed-1802 healthy case found
271 observations, 240 with capsule support, 31 unassociated, and target
direct support 0.4297. Each of four large angular cells spanned about
1.152 rad but had only two 0.2 m axial resolution supports per 1 m-spaced
view station, leaving gaps. The deterministic 16-angle sweep had views at
all 64 stations, so the missing direct support was a station-density issue,
not permission to credit unseen surface. The occluded negative control's
clear counterpart also failed to see the declared defect from the one
fixed 2 m standoff. Four bounded Unity cases on seed 1802 had zero credited
coverage. These are immutable validation failures. Denser, independently
specified development view geometry will be evaluated before another cycle.

The development repair uses ten 0.4 m-spaced axial view stations for a
full-surface sweep while retaining all 16 angular directions, unchanged
sensor thresholds, and required direct support 1.0. The occlusion negative
control now places a resolvable 2 rad synthetic corrosion patch in the
predeclared inspection domain and directs its one test ray through an
evaluation-only rack panel; its clear counterpart must detect the patch.
The original 0.3 rad patch could be narrower than the 0.05 m declared
corrosion resolution on small-radius worlds. Fresh seeds 485–490 passed
12/12 targeted healthy and occluded cases; 555–564 passed 20/20 more;
565/566 passed the complete unchanged-condition case matrix **28/28**;
and 567–586 passed 20/20 additional full-surface healthy cases.

The Unity parity fixture now includes two clear views centred on broad
circumferential resolution cells, chosen from the declared radius and sensor
lateral resolution. Those cells can retain positive coverage after the
declared hard registration erosion. The prior four-view bounded fixture
failed 5/20 fresh development seeds 491–510. With one broad view, clear
bounded cases passed 20/20 on 511–530, but the view was rack-occluded in
seed 531. A second broad view in a different sector passed full parity
16/16 on 533/534, occluded bounded screening 20/20 on 535–554, clear
bounded screening 20/20 on 587–606, and full parity 16/16 on 607/608.
These are fixture changes and development evidence, not a repair of cycle 4.
