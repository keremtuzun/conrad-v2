# Spatial Structural Architecture V1 freeze

Status: **SOFTWARE VALIDATED, SYNTHETIC_ONLY**. This freezes the Spatial V1
software contract represented by source commit
`8c79b7127dc1424fc093e9c0814e41ca64e66b00` and the acceptance ledger
in `../audits/SPATIAL_V1_ACCEPTANCE.md`. It does not establish physical
sensor performance, vehicle endurance, or underwater communication quality.

## Pinned contract

| Element | Frozen identifier or digest |
|---|---|
| Mission truth | `twin2t-spatial-v1` |
| Local evolution | `twin2t-spatial-evolution-v1` |
| Structural sensor | `structural-sensor-v2` |
| Synthetic sensor configuration | `034f949a14ded197ecca8cb1a8ab714b9872c98fac94a099414e77645fdd2104` |
| Detectability | `spatial-synthetic-detectability-v2` |
| Visibility certificate | `capsule-sdf-recursive-v1` |
| Support schema | `structural-observation-v2` |
| Registration | `exact-bounded-or-unregistered-v2` |
| Association | `spatial-structural-association-axial-v3` |
| Mission Model2T | `model2t-spatial-v1` |
| Canonical mission Model2T configuration | `afdd935d2171ceaaa0bfe586746ceaaf987d1afbe759ba2606fa6350d023230e` |
| Unity Windows player SHA-256 | `36c5c9f13481406382a8e9ef8fc0ea7cdf055c43bb12fc8fd545b07c199cd277` |

The sensor configuration is
`configs/sim/spatial_v1_synthetic_sensor_detectability_v2.json` with
`RESOLUTION_CELL_SAMPLES`, 0.2 m axial/lateral synthetic resolution,
0.05 m minimum corrosion/crack patch scale, and separate 0.01 m crack
length and 0.001 m crack depth thresholds. Every numerical sensor value is
`ENGINEERING_ESTIMATE`. The Model2T digest is SHA-256 of sorted, compact
UTF-8 JSON for `mission_runtime_config.model2t_spatial` in the verified
mission bundle. It pins 2 axial cells, 4 sectors, required axial fraction
`[0.05, 0.95]`, sectors `[2, 3]`, required independent looks 1, and the
declared condition thresholds. Replay compares the complete spatial version
contract and sensor digest and refuses a mismatch.

The three evaluated truth configurations use the same version with canonical
sorted, compact UTF-8 JSON digests: healthy
`ec5885b948a4a002c743fa018d344c24627691cd5538ffdd0009cc28444ce3c7`,
uniform same-mean
`cba6b551c69328500e36ff7787290461fb70050493df78ae5021783bd41b55af`,
and local-defect same-mean
`00b41db2f2404655a6d7fabfbeed57aae5ffb2a3b22b44c7dc16034d1c414200`.
These are world instances, not alternative truth-model versions.

## Acceptance evidence

The predeclared cycle-5 controlled-view matrix passed 28/28 on seeds
1901/1902. Its `results.jsonl` SHA-256 is
`8cf693be70bf86819af451b99459344f6c95ccc11c528e1f905455935e779050`.
Matched kernel/Unity parity passed 16/16 on the same fresh seeds and pinned
player; its `results.jsonl` SHA-256 is
`898453e2ceb8c585e6953f1e5abe940d5319776a423ba97c5d1d78dbad84dd73`.
All earlier failed validation cycles and their spent seeds remain recorded in
the acceptance ledger.

The autonomous 480 s healthy mission reached qualified `INTACT` with full
required coverage, then replayed exactly across 1,703 files, 478 objects,
and 8,166 belief payloads. The current-contract 480 s equal-global-mean
pair distinguished uniform and local corrosion: 189 intact/zero severe
revisions versus zero intact/387 severe. Both bundles passed manifest and
object verification. Their manifest SHA-256 digests are
`b968cf3f3b8f7559e83ca243d16cd2979040ad3d9a44f89002dd83770929cc6a`
and `16c036ec11698ea4222295172d79c054cff7d2b8dd0a00bab831c8d310d67bfb`,
respectively. Static truth-boundary tests passed 22/22; a runtime scan found
zero leaks in 31,785 records. Spatial MCBR candidate predictions are tested
against unresolved, resolved, occluded, far-side, subresolution, and disjoint
support cases.

A fresh worktree at the source commit passed the ordinary full regression:
1,473 passed, 14 documented skips, 125 opt-in Unity tests deselected, and
3 historical expected failures in 33:04, with no setup errors. Ruff
format/lint, mypy over 643 files, and the secret scan also passed. The
14 skips include historical evidence that is gitignored in a clean checkout;
no spent historical final seed was regenerated.

This freeze applies to the opt-in `spatial_v1` mission truth and Model2T
backend. Historical legacy missions and the historical I4/I5/I7 results are
unchanged. A change to the frozen model, sensor assumptions, support,
association, or belief contract needs a new version and independent evidence.
Physical parameter replacement and real-world validation remain listed in
`../hardware/PENDING_PHYSICAL_CALIBRATION.md`.
