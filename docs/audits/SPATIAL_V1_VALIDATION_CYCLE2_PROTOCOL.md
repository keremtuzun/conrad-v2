# Spatial V1 prospective validation, cycle 2

The first controlled-view partition on seeds 1401/1402 is immutable **FAIL
6/28**. All 22 failures were caused by incomplete registry-associated
coverage at segment joints. Its cases, expected conditions, required domain,
and results are unchanged. Those seeds must not be rerun.

This new partition uses **fresh seeds 1501 and 1502**. These seeds were not
used in development or the first validation. Run only after the axial
association repair is committed, a current-association closed-loop mission
replays exactly, and the nine Unity development parity tests pass under that
same code. `scripts/validate_spatial_case_matrix_v2.py` runs the unchanged
14-case `CASES` tuple imported from the first protocol: 28 controlled-view
cases. The required domain is unchanged at axial fraction `[0.05, 0.95]`,
sectors `[2, 3]`; the sensor and condition thresholds are unchanged. Full
healthy, heterogeneous healthy, uniform same-mean, subresolution, and each
subthreshold crack must yield qualified synthetic `INTACT`. Covered corrosion,
covered crack, separated defects, required-domain edge defect, and local
same-mean must yield `SEVERE`. Partial healthy, outside-support defect, and
occluded defect must remain `UNKNOWN`. Required full sweeps need direct
support 1.0. False `INTACT` on covered resolvable defects must be zero, and
the equal-global-mean pair must differ. **All 28** must pass.

`scripts/validate_spatial_unity_parity_v2.py` runs the same pinned player
SHA-256 `36c5c9f13481406382a8e9ef8fc0ea7cdf055c43bb12fc8fd545b07c199cd277`
for 16 cases (two seeds times clear/occluded, deterministic/noisy, and
exact/bounded survey). Matching remains exact structural support and inline
values, identical registry association and local/component conditions, local
coverage within absolute `1e-12`, zero kernel-visible/Unity-hidden probes,
support in every clear view, at least one hidden occluded view, and positive
guaranteed bounded-survey coverage. **All 16** must pass. Only one Unity player
may run at a time.

The repair is pinned as `spatial-structural-association-axial-v2`; sensor
detectability remains `spatial-synthetic-detectability-v2`, and registration
remains `exact-bounded-or-unregistered-v2`. Both runners write a protocol file
and append each result immediately. A failure is immutable; interrupted
started cases are marked `INTERRUPTED`, never rerun. `--resume` may complete
only cases never started. A failed cycle requires new development and a new,
predeclared partition. Neither cycle is an I4/I5/I6/I7 gate or physical
sensor validation.
