# Spatial V1 prospective validation, cycle 5

Cycles 1–4 are immutable failures. Seeds 1401/1402, 1501/1502,
1701/1702, and 1801/1802 are spent. This protocol declares **fresh seeds
1901 and 1902** before either partition is opened. The deployment source,
sensor config, direct-coverage threshold, belief logic, association rule,
Unity player, and replay contract remain unchanged from cycle 4. The
controlled-view and Unity parity *test fixtures* have changed as described
below. No failed validation case is rerun.

Cycle 4 failed controlled-view **16/28** and Unity parity **12/16**.
Read-only seed-1802 evidence showed the four 1 m-spaced full-surface view
stations left axial gaps in wide angular resolution cells; the target's
direct support was 0.4297. Its occlusion control did not produce a
detectable clear-world reading from the one fixed pose. The four bounded
Unity failures had zero credited coverage. No backend support/measurement
discrepancy was observed in the failing parity cases.

For this cycle a full controlled-view sweep uses ten declared 0.4 m-spaced
axial stations and the same 16 angular directions. The occlusion negative
control uses one ray aimed through a known test rack panel, with an
evaluation-only truth-side pose construction; the deployment receives only
the resulting simulated observation. Its clear counterpart contains a
declared, resolution-sized synthetic corrosion patch. This makes the
positive detection control feasible across the generator's radius range.
The full-sweep required axial domain stays `[0.05, 0.95]`, sectors `[2, 3]`,
and every full case still requires **direct support 1.0**. Partial and
occluded cases must remain `UNKNOWN`. Healthy, heterogeneous healthy,
uniform same-mean, subresolution, and subthreshold crack cases must yield
qualified synthetic `INTACT`. Covered corrosion/crack, separated defects,
edge defect, and local same-mean must yield `SEVERE`, with zero false
`INTACT` on covered resolvable defects. `scripts/validate_spatial_case_matrix_v5.py`
runs the original 14 case names and expected conditions on both seeds;
**28/28 PASS** is required.

Unity parity uses six clear fixture views: the four close geometric views
from cycle 4 plus two views centred on broad resolution cells in distinct
sectors. They are calculated from the declared sensor resolution and
capsule geometry so the fixed 0.002 m endpoint bound can leave positive
credited area. The explicitly panel-blocked ray is unchanged. The player
SHA-256 is
`36c5c9f13481406382a8e9ef8fc0ea7cdf055c43bb12fc8fd545b07c199cd277`.
`scripts/validate_spatial_unity_parity_v5.py` runs two seeds times
clear/occluded, deterministic/noisy, and exact/bounded survey: **16/16
PASS** is required. Kernel and Unity support and inline measurements must
match exactly, registry association decisions and local/component conditions
must match, coverage must agree within absolute `1e-12`, and there must be
zero kernel-visible/Unity-hidden probes. All clear fixture views must emit
support; each occluded case must include a zero-support blocked ray; each
bounded case must earn positive coverage. Uncertain association may abstain
but never select a wrong component. Run one Unity player at a time.

Fresh development evidence before declaration: targeted healthy/occluded
20/20 on seeds 555–564, complete controlled-view 28/28 on 565/566,
healthy full-sweep 20/20 on 567–586, bounded Unity clear 20/20 on 587–606,
bounded Unity occluded 20/20 on 535–554, and full Unity parity 16/16 on
607/608. The completed association-v3 480 s mission exactly replayed all
recorded payloads. The changed test fixture passed 19/19 focused spatial
integration tests and 9/9 live Unity development tests. All numerical sensor values remain synthetic engineering
estimates, with physical calibration pending.

Each runner writes its protocol before the first case and appends results
immediately. A failure is immutable. An interrupted started case stays
`INTERRUPTED`; `--resume` may run only cases never started. This is software
architecture validation, not a formal I4/I5/I6/I7 gate.
