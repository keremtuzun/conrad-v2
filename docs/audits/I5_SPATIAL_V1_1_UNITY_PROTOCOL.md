# I5 Spatial V1.1 formal Unity protocol

Status: **FORMAL FAIL — 7/10 criteria passed**

The Spatial V1.1 surrogate final passed under its preregistered rule. Its exact
result SHA-256 is
`0dbc96d699a698ed95011330692533312094d941b5e3fb9c9bf25a965e10e3d1`.
This authorizes a separate formal Unity cycle; it does not alter or replace the
historical I5 formal 9/10 FAIL.

## Frozen inputs

- config: `configs/eval/i5_spatial_v1_1_unity_formal.yaml`, SHA-256
  `9051df7d1d7d409e53c4749faa31a16a99a703f7505b47128150a34c1f8f9e51`;
- partition: `configs/eval/partitions_i5_unity_v3.yaml`, canonical digest
  `cb8afdb68cec7e95a013a75b46b31491a382440ef0f37a48cc6e469bf026c31f`;
- player: exact macOS native executable SHA-256
  `7e42f64199b043204d62737d06aabc6b6ceb1f3c1c699f158a612b938ea87ac6`;
- worlds: 8701000 and 8701001; and
- unused suffix: 8701002..8701009, barred from post-result supplementation.

The run is exactly 2 worlds x 7 Spatial V1.1 I5 scenarios x 3 arms = 42
sequential Unity flights. All final surrogate scenarios are included, including
battery reserve. The mission-level `LOW_POWER` event maps to Unity's existing
`BATTERY_DEGRADATION` bridge fault with the same capacity factor; it is not a
mocked battery reading.

## Immutable execution result

The exact declared grid ran once, sequentially, on worlds 8701000 and 8701001.
No unused suffix world was opened and no flight was repeated. The exact player
hash matched, bundle replay passed with zero trajectory difference, the
declared grid was complete, and the matrix's seven criteria passed.

The integrated action criterion failed because `I5-BATTERY-RESERVE` reached
its warrant on 0/2 worlds. Every other scenario reached its warrant on 2/2
worlds with correct-given-warrant 1.0. The primary arm had zero decision
violations, zero false-intact cases, zero nominal over-escalations,
traceability 1.0, and UIR 0.0. It was not worse than either baseline, but those
results cannot compensate for the missing battery warrants.

The cause is a cross-runtime semantic mismatch in the declared bridge:
kernel `LOW_POWER` sets energy used so remaining battery becomes the requested
fraction, whereas Unity `BATTERY_DEGRADATION` only scales capacity and leaves
energy used low. Unity accepted the injected 0.12 event, but remaining battery
was still about 0.96 at mission end instead of 0.12. This is a software defect,
not permission to reinterpret or rerun the result.

The defect was subsequently corrected and verified on development world
8701100 with a newly built player. See
`docs/audits/I5_LOW_POWER_POST_FORMAL_CORRECTION.md`. That post-result
development evidence does not alter this formal result.

The formal leakage node also errored before completing because the test tried
to label a scored row with absent `run_id`. A separately versioned read-only
supplement scans the exact stored bundles. Its outcome does not alter that
formal node or the formal verdict.

Immutable artifacts:

- result SHA-256: `bde617d70a617f13758cf06a3bdac6bf323c63d7a5093cb14da3194f20b5ffc1`;
- measured SHA-256: `563985f3fac47a01defa03e040aefc1b5736ea1a216c1507781e48b1fb764f90`;
- JUnit SHA-256: `3f114cfb40ad9aac2618122d9147a086ef8ea5ff8c65f696af524f69b51a1b11`;
- log SHA-256: `8ce15dd3a7639a31411c05fede24af122bf896709c2eb83fe316178b208f8ce8`; and
- I5 evidence SHA-256: `08caaf1d3745bfd3c5d9a5cdf078130f4da17d5c5121a55d15b9c82ac2f05822`.

## Frozen rule

All ten existing I5 criteria must pass. The seven belief-fixture criteria use
the immutable M1-ACTION-E001 final artifact. The three integrated-mission
criteria require:

1. every scenario reaches its warrant on both worlds and correct-given-warrant
   is at least 0.9;
2. zero false intact on covered resolvable defects, zero primary decision
   violations, and zero nominal over-escalations;
3. traceability 1.0 and UIR 0.0;
4. pooled EGDC task success at least each baseline, safety entries at most each
   baseline, and violations at most each baseline; and
5. the exact grid, exact player, Spatial V1.1 runtime, leakage scan, and replay
   all pass.

Completed bundles may be resumed but never re-flown. Any other missing,
duplicate, interrupted, player-mismatched, leaked, or failed evidence is a
formal FAIL. Thresholds, N, and the unused suffix cannot change after any
formal result is observed.

This is synthetic Unity L1 approximate-physics evidence. It is not physical,
ROS, HIL, sensor-calibration, or deployment evidence.
