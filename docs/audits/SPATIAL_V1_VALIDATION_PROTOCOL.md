# Spatial V1 controlled-view validation protocol

Declared before opening these worlds. The runner is
`scripts/validate_spatial_case_matrix.py`, with exactly two new seeds:
`1401`, `1402`. It runs the 14 fixed cases in its `CASES` tuple once per seed,
for 28 cases. The required inspection domain remains axial fraction
`[0.05, 0.95]`, sectors `[2, 3]`, with the same versioned sensor,
detectability, and condition thresholds used in development. The runner
records each result immediately and refuses an existing output directory.
After an interrupted process, `--resume` verifies the identical protocol,
skips recorded cases, and marks any created case directory without a result
`INTERRUPTED`; it never reruns that case or turns it into PASS.

Expected result by case: fully covered healthy, heterogeneous healthy,
uniform same-mean, subresolution, and each below-limit crack yield
**qualified synthetic INTACT**;
covered corrosion, covered crack, separated defects, required-domain edge
defect, and local same-mean yield **SEVERE**; partial healthy, a defect outside
measured support, and an occluded defect remain **UNKNOWN**. For every covered
resolvable defect, the false-INTACT count must be zero. The uniform and local
same-mean pair must differ despite equal whole-surface corrosion mean.
All 28 cases must pass their predeclared expectation for this matrix to PASS.
The sensor contract is `spatial-synthetic-detectability-v2`, with independent
crack length 0.01 m and depth 0.001 m limits, in addition to the 0.05 m crack
patch-size limit. All are `ENGINEERING_ESTIMATE`. The covered crack case
exceeds each limit; the subresolution case remains below a spatial feature
size limit, while two crack negative controls fall just below one of the
independent crack limits. These expectations are declared before these seeds.
No failed case may be rerun, dropped, or edited into PASS; a failure requires
diagnosis on fresh development worlds and a separately declared later cycle.

This matrix exercises actual mission Twin2T truth, pose-driven structural
sensing, Observation, ECMER Evidence, registry association, spatial Model2T,
and Model1 decision context under controlled views. It does not by itself
validate autonomous view completion, long-horizon communication, or Unity
mission outcomes. Those have separate development and gate requirements.
