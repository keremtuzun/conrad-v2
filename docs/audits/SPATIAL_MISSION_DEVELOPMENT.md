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
