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

These are controlled development views, not closed-loop Model1 mission success
or a complete case matrix. Nonzero design survey noise is rejected at launch
until a conservative support transform exists. Broader Unity parity,
closed-loop Model1 decisions, performance profiling, and gate impact
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
