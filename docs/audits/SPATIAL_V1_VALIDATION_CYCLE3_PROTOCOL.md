# Spatial V1 prospective validation, cycle 3

Cycles 1 and 2 are immutable failures. Seeds 1401/1402 and 1501/1502 are
spent and will not be rerun. This cycle uses fresh seeds **1701 and 1702**,
declared here before either is opened. The cycle may start only after a
committed association-v3 closed-loop 480-second development mission replays
exactly and the committed source passes the relevant ordinary and Unity
development tests. These preconditions are evidence checks, not acceptance
results.

`scripts/validate_spatial_case_matrix_v3.py` imports the original unchanged
14-case matrix and runs **28** controlled-view cases. Required domain remains
axial fraction `[0.05, 0.95]`, sectors `[2, 3]`. Sensor detectability,
condition thresholds, and direct full-sweep support **1.0** remain unchanged.
Full healthy, heterogeneous healthy, uniform same-mean, subresolution, and
each subthreshold crack must yield qualified synthetic `INTACT`. Covered
corrosion, covered crack, separated defects, required-domain edge defect,
and local same-mean must yield `SEVERE`. Partial healthy, outside-support
defect, and occluded defect must stay `UNKNOWN`. False `INTACT` on a covered
resolvable defect must be zero; equal-global-mean cases must differ. The
full-surface positive control uses a declared inspectable pipe without
permanent support hardware or clutter; the occluded negative control adds
a truth-side rack. **All 28** must pass.

`scripts/validate_spatial_unity_parity_v3.py` runs **16** cases: both seeds
times clear/occluded, deterministic/noisy, exact/bounded survey. The Unity
player SHA-256 remains
`36c5c9f13481406382a8e9ef8fc0ea7cdf055c43bb12fc8fd545b07c199cd277`.
Kernel and Unity structural support and measured inline values must match
exactly. Their registry association decisions and local/component conditions
must match; local coverage must agree within absolute `1e-12`, with zero
kernel-visible/Unity-hidden probes. Every clear test view must emit support,
at least one deliberately panel-blocked view must emit none in occluded
cases, and bounded-survey cases must earn positive coverage. An association
may abstain under a hard-bounded survey but may not select a wrong registry
component. **All 16** must pass. Run only one Unity player at a time.

The replay contract pins `spatial-structural-association-axial-v3`,
`spatial-synthetic-detectability-v2`, and
`exact-bounded-or-unregistered-v2`. Each runner writes its protocol and
appends each result immediately. Any failure is immutable. A started case
interrupted before its result is recorded remains `INTERRUPTED`; `--resume`
may run only never-started cases. This cycle does not itself close any
I4/I5/I6/I7 formal gate or establish physical sensor performance.
