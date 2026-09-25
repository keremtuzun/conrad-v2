# Spatial structural association at segment joints

Version: `spatial-structural-association-axial-v3`. This is a synthetic
deployment-plane inference rule and is pinned in each spatial mission replay
contract. It does not use a Twin world ID, hidden condition, or truth-plane
mapping.

The ordinary association projects a structural reading through its **estimated**
pose and measured range/bearing/elevation, then compares the projected WORLD
point with surveyed registry geometry. Ambiguous candidates remain `NO_MATCH`.
At a joint, neighboring segment capsules can have similar nearest-surface
distances even when the projected point lies well inside only one segment's
axial interval. This left small uncovered strips in the first validation
partition, correctly preventing a qualified intact claim.

For an Observation with a declared capsule-surface support, only surveyed
`SEGMENT` capsules are eligible; a nearby support box or weld cannot explain
that measurement type. For every multi-candidate spatial reading, the axial
rule is applied even if nearest-surface distance would select one candidate.
Development seed 412 showed that the nearest-surface choice can be confidently
wrong near a joint. The rule may assign exactly one segment only when:

1. Every candidate segment has an exact surveyed axis or a declared hard
   endpoint error bound. A Gaussian survey sigma alone does not qualify.
2. The point projects farther than `3 * projected_sigma_m` plus a
   conservative endpoint and axis-direction error margin inside one
   segment's open axial interval.
3. The point projects farther than its own corresponding margin beyond every other
   candidate segment's axial endpoints.

The three-sigma margin uses the existing association configuration and is a
confidence heuristic, **not** a hard Gaussian error bound. A point near a
joint, overlapping segments, an uncertain survey, or multiple interior
candidates remains unassociated, including when point-distance association
appears confident. The directional margin uses endpoint displacement at
most `2 * bound`, axis length at least `length - 2 * bound`, and unit-axis
change at most `4 * bound / (length - 2 * bound)`; each endpoint plane is
guarded by its own distance from the projected point. Bounded-survey support
uncertainty is handled separately for coverage; this rule does not convert
a bounded survey into an exact one. Registry IDs in the result come solely from the
mission's design registry.

Development tests cover adjacent-segment interior points, joint points,
overlapping axes, and uncertain surveys. The full mission path must also
demonstrate that heterogeneous, defect, occlusion, and same-mean cases retain
their intended conservative condition semantics before another prospective
validation partition is opened.
