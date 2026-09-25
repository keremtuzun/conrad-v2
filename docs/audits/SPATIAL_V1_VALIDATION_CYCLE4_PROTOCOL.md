# Spatial V1 prospective validation, cycle 4

Cycles 1, 2, and 3 are immutable failures. Seeds 1401/1402, 1501/1502,
and 1701/1702 are spent and will not be rerun. This cycle declares fresh
seeds **1801 and 1802** before opening either partition. The association,
sensor, truth, support, registration, Model2T, MCBR, and replay contracts
remain as committed before cycle 3. Only the Unity parity harness's four
clear positive-control camera poses were brought closer to the target while
remaining inside the declared sensor working range. The deliberately blocked
occlusion ray is unchanged.

The cycle-3 controlled-view matrix passed 28/28, but cycle-3 Unity parity
failed 12/16: on seed 1702 the two 2 m clear poses produced zero supported
resolution cells in both backends. This is a fixture feasibility issue,
not a backend discrepancy. Fresh development clear-view screening passed
48/48 seeds (430–445 and 448–479) with the closer poses. Full parity
development passed 16/16 on seeds 446/447 and again 16/16 on 480/481.
Those development seeds are distinct from this partition. The current
committed 480 s mission has an exact full replay; the ordinary preflight
passed 1,473 tests, 14 explicit skips, and three historical xfails.

`scripts/validate_spatial_case_matrix_v4.py` imports the original unchanged
14-case matrix. The two seeds require **28/28 PASS**. Required domain stays
axial fraction `[0.05, 0.95]`, sectors `[2, 3]`, with direct support 1.0
for each full sweep. Healthy, heterogeneous healthy, subresolution,
subthreshold crack length/depth, and uniform same-mean cases must produce
qualified synthetic `INTACT`; covered corrosion/crack, multiple separated
defects, edge defect, and local same-mean must produce `SEVERE`; partial
healthy, outside-support, and occluded defect remain `UNKNOWN`. A covered
resolvable defect must never receive a false `INTACT`.

`scripts/validate_spatial_unity_parity_v4.py` runs the pinned player SHA-256
`36c5c9f13481406382a8e9ef8fc0ea7cdf055c43bb12fc8fd545b07c199cd277`
for **16** cases across two seeds, clear/occluded, deterministic/noisy,
and exact/bounded survey. Structural supports and inline measurements must
match exactly, as must registry association decisions and local/component
conditions. Coverage must match within absolute `1e-12`; no kernel-visible
probe may be Unity-hidden. Every clear view must emit support, at least one
deliberately blocked occluded view must emit none, and bounded survey must
earn positive coverage. Uncertain association may abstain, but never select
a wrong component. **All 16** must pass. Run one Unity player at a time.

Runners write a protocol before the first case and append each result
immediately. Any failure is immutable. A started case interrupted before
its result is recorded stays `INTERRUPTED`; `--resume` may run only cases
never started. This cycle is synthetic architecture validation, not a
formal I4/I5/I6/I7 gate or physical sensor calibration.
