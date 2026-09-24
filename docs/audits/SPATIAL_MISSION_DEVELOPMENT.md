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
