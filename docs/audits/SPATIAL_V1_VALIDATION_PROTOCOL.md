# Spatial V1 controlled-view validation protocol

Declared before opening these worlds. The runner is
`scripts/validate_spatial_case_matrix.py`, with exactly two new seeds:
`1401`, `1402`. It runs the 12 fixed cases in its `CASES` tuple once per seed,
for 24 cases. The required inspection domain remains axial fraction
`[0.05, 0.95]`, sectors `[2, 3]`, with the same versioned sensor,
detectability, and condition thresholds used in development. The runner
records each result immediately and refuses an existing output directory.

Expected result by case: fully covered healthy, heterogeneous healthy,
uniform same-mean, and subresolution yield **qualified synthetic INTACT**;
covered corrosion, covered crack, separated defects, required-domain edge
defect, and local same-mean yield **SEVERE**; partial healthy, a defect outside
measured support, and an occluded defect remain **UNKNOWN**. For every covered
resolvable defect, the false-INTACT count must be zero. The uniform and local
same-mean pair must differ despite equal whole-surface corrosion mean.
All 24 cases must pass their predeclared expectation for this matrix to PASS.
No failed case may be rerun, dropped, or edited into PASS; a failure requires
diagnosis on fresh development worlds and a separately declared later cycle.

This matrix exercises actual mission Twin2T truth, pose-driven structural
sensing, Observation, ECMER Evidence, registry association, spatial Model2T,
and Model1 decision context under controlled views. It does not by itself
validate autonomous view completion, long-horizon communication, or Unity
mission outcomes. Those have separate development and gate requirements.
