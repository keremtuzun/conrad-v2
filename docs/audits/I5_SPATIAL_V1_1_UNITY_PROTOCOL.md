# I5 Spatial V1.1 formal Unity protocol

Status: **DECLARED / SEALED BEFORE EXECUTION**

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
