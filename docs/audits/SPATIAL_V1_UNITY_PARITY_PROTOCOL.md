# Spatial V1 prospective Unity parity protocol

Declared before opening these Unity worlds. Use
`scripts/validate_spatial_unity_parity.py` once with a new output directory.
Seeds are `1401` and `1402`. For each seed, run the Cartesian product of:
clear/view-occluder, deterministic/noisy synthetic structural response, and
exact/bounded design survey, for 16 matched cases. The bounded survey uses
Gaussian scale 0.001 m with an explicit 0.002 m hard endpoint bound; it does
not claim a measured physical survey. One Unity player runs at a time.
This validation uses `spatial-synthetic-detectability-v2` and the same
explicit crack length and depth limits as the controlled-view matrix.

Pinned player: Unity 6000.5.9f1, SHA-256
`36c5c9f13481406382a8e9ef8fc0ea7cdf055c43bb12fc8fd545b07c199cd277`.
The runner refuses another hash before opening a case. The predeclared
matching rule is exact structural support and inline values, identical
registry association and local/component conditions, local coverage within
absolute `1e-12`, and zero kernel-visible/Unity-hidden probes. Each case must
emit spatial support; clear worlds must emit support at every tested view,
occluded worlds must hide at least one view, and bounded worlds must earn
some guaranteed local coverage. All 16 cases must pass. Results are written
after each case; failed cases cannot be rerun or reclassified as passing.
After a process interruption, `--resume` verifies the same protocol and player
hash, skips recorded cases, and marks a started case with no result
`INTERRUPTED` rather than flying it again.

This is synthetic development architecture validation, not an I5 formal
world or a calibrated real-world parity claim. It does not open any historical
I4, I5, I6, or I7 gate partition.
