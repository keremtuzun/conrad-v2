# Gates I1, I2, I3 on the formal Unity path

Date: 2026-09-19. ADR-0008: only this evidence can promote I1, I2 and I3. The Python L1 kernel runs stay SURROGATE.
Evidence files: `artifacts/gates/{I1,I2,I3}/evidence_formal.json`, written by `scripts/record_unity_gate_evidence.py`.
Each gate folder also holds the measured numbers (`unity_measured.json`), the JUnit XML and the pytest log
(`unity_pytest.xml` and `unity_pytest.txt`). Validity is **L1_APPROXIMATE_PHYSICS**, and every physical and sensor
parameter is SYNTHETIC_ONLY.

## Result

| Gate | Formal status | Official status (`conrad gates status`) |
|---|---|---|
| I1 | **PASS**, all 9 criteria | **PASS** (U0 and 2S-FIRST are PASS) |
| I2 | **FAIL**: "station keep" and "NAV-001..NAV-006" fail on NAV-005 | FAIL |
| I3 | **FAIL**: "persistent technical belief" fails because one hidden component is INFERRED | FAIL (it also needs I2 and 2T) |

## What was built

| Piece | File |
|---|---|
| Twin2S primitives converted to Unity colliders, with measured conversion error | `conrad/sim/unity/twin_scene.py` |
| Truth-side world: MissionWorld scenario, twins, registry and payload suite, with Unity as the vehicle | `conrad/sim/mission/unity_world.py` |
| Driver, replay, NAV-001..006 on Unity, and the scenarios `I1-UNITY`, `I3-UNITY`, `I2-UNITY-NAV` | `conrad/sim/mission/unity_run.py` |
| CLI: `conrad sim run --backend unity [--seed]`; `conrad replay run` now dispatches Unity bundles | `conrad/cli/commands.py` (appended) |
| Gate tests | `tests/unity_live/test_i{1,2,3}_unity.py` and `unity_gate_support.py` |
| C#: `RangeImageSensor` (DEPTH_RANGE, `range_f32_hw_v1`), optional sonar elevation fan, `range_imagers` capability, scenario-declared current `gusts` | `unity/ConradUnityV2/Assets/Conrad/...` |
| Bridge: `RANGE_LAYOUT`, the `WireCapabilities.range_imagers` field, `UnityRobotHardware.get_range_image()` | `conrad/adapters/unity/*` |

The player was rebuilt with `BuildScript.BuildWindows64Player`. The build finished with 0 `error CS` and 0 `warning CS`
(log: `unity/ConradUnityV2/Logs/i1_i3_build.log`).

### Data flow

- **Truth side.** `MissionWorld.build` produces the scenario, Twin2S, Twin2T, the registry and the payload suite.
  `twin_scene` turns the Twin2S primitives into Unity colliders, which are loaded with `CONFIGURE_SCENE`. The player
  starts from a RobotConfig variant: the range imager, and the sonar geometry declared in the scenario's SensorSpecs.
- **From Unity.** Physics, IMU (AHRS), pressure depth, power, thrusters, the range imager, the sonar and the camera
  all come from Unity.
- **From the twins.** Structural readings come from Twin2T. Their visibility is computed by the Twin2S oracle at the
  Unity TRUE pose, which is read from the truth endpoint on the truth side only. The USBL-like fix is the Unity true
  pose plus noise. Both reuse `MissionSensorSuite` unchanged, with its Twin2S geometric channel removed.
- **Adapters.** Unity range and sonar frames are converted to the Model2S payload formats by pure functions:
  `range_image_to_model2s` is a shape check, and `sonar_to_model2s` re-bins and flips the beams. Model2S semantics
  are unchanged. Each derived observation names the raw wire digest, which stays in the object store.
- **Deployment side.** `MissionRuntime` is unchanged. It receives only the RHI object and the mission context.

### Scene conversion error (measured, seed 5300059)

| Primitive | Conversion | Error |
|---|---|---|
| capsule, box | exact | 0 |
| blob | sphere-capsules | 0.5 mm |
| torus | 24-segment ring | 3.0 mm |
| heightfield | 0.25 m grid | 0.4 mm vertical |
| ellipsoid (3 rocks) | one capsule | ≤ 57 mm |

Other seeds give larger ellipsoid errors, up to 181 mm on development seed 5100000. Dynamic entities are refused.
The `straight_pipeline` family has none.

## I1 (seed 5300059, mission FINAL partition)

Bundle: `artifacts/unity/gate_runs/I1/I1-UNITY-s5300059-3fdf2d30`. Replay: `artifacts/unity/gate_runs/replay/I1/`.
Two independent recordings gave identical numbers.

| Criterion | Measured | Declared bound | Result |
|---|---|---|---|
| robot moves through synthetic world | true path 21.54 m in 60 s; 600 commands accepted via `unity_v2`; 0 collisions; min clearance 0.53 m | ≥ 5 m, 0 collisions | PASS |
| persistent map | 390 spatial revisions (361 DIRECT with evidence); max revision 17; 69 beliefs | revised, WORLD frame | PASS |
| geometry | 395 occupied-claimed cells; median \|Twin2S sdf\| 0.152 m, p90 0.177 m; 100 % within 0.5 m | ≥ 90 % within 0.5 m | PASS |
| occupancy | 134 of 1200 probes claimed free; wrong 0.0 | < 0.05 | PASS |
| coverage | lane region 0.146; target segment 0.763; 1591 observed cells | > 0 | PASS |
| observed/inferred/unknown | cells: OBSERVED 1591, INFERRED 54, UNKNOWN 4788; 500 of 500 points 0.8 m below the seabed are UNKNOWN and not free | all three present; hidden = UNKNOWN | PASS |
| uncertainty | mean U_O: OBSERVED 0.056, INFERRED 0.727, UNKNOWN 0.824; per-channel std (UA, UE, UC, UO) = 0.141, 0.008, 0.030, 0.402 | U_O(UNKNOWN) > U_O(OBSERVED) + 0.3; ≥ 2 channels vary | PASS |
| provenance | 21 of 21 evidence items feeding DIRECT spatial revisions re-derive bit-exactly from their raw Unity wire frames (`adapter = unity_v2`) | all | PASS |
| no Twin truth leakage | 2455 runtime texts scanned (events, `mission/`, SQLite payloads); 0 world-entity UUIDs, 0 twin-only keys, 0 `visibility:` keys; registry IDs on cells ⊂ registry | 0 | PASS |
| replay | digests verified (152 files, 97 objects); events 2638 = 2638, decisions 30 = 30, revisions 456 = 456; true trajectory max \|Δ\| = 0.0 | tolerance 1e-9 m | equal |

## I2 (NAV-001..006 on Unity, noise seeds 7300001..7300006)

The NAV set has no partition domain. These seeds lie outside every range in `configs/eval/partitions.yaml`.
Bundles: `artifacts/unity/gate_runs/I2/NAV-00x/`, each containing `result.json`, `trajectory.json`,
`commands.json`, `nav_manifest.json` and `unity/`.

| Benchmark | Final error (m) | Other metric | Estimation RMS (m) | Result |
|---|---|---|---|---|
| NAV-001 | 0.086 | | 0.123 | PASS |
| NAV-002 | 0.240 | | 0.129 | PASS |
| NAV-003 | 0.117 | min clearance 0.469 m, 0 collisions | 0.115 | PASS |
| NAV-004 | 0.080 | standoff RMS 0.011 m (bound 0.25) | 0.120 | PASS |
| NAV-005 | 0.072 | **station RMS 0.1534 m (bound 0.15)** | 0.140 | **FAIL** |
| NAV-006 | 0.118 | gust 15–20 s as a scenario current gust | 0.109 | PASS |

Final-error bound: 0.3 m, from `nav_benchmarks.yaml`.

Criterion results:

- **Reach waypoint, avoid obstacle, follow pipeline:** PASS.
- **Station keep:** FAIL.
- **NAV-001..NAV-006:** FAIL (5 of 6 pass).
- **Uses estimated state:** PASS. The stack gets only `UnityRobotHardware` and the synthetic fix. As a counterfactual,
  NAV-001 was rerun with the fix biased by +0.5 m in y. The true end point moved −0.495 m in y (tolerance 0.15 m),
  so control follows the estimate.

The NAV-005 failure is not Unity-specific. On the same seed, the Python kernel gives 0.1574 m and also fails. On
development seed 7100001, both backends pass: Unity 0.1163 m, kernel 0.1167 m. The seed was not changed after the
result was seen.

Replay: NAV-003 and NAV-006 were re-run from their manifests. In both, the true trajectory max |Δ| was 0.0 and the
3000 gateway acks were identical.

## I3 (seed 5300058, mission FINAL partition)

Bundle: `artifacts/unity/gate_runs/I3/I3-UNITY-s5300058-b43c66d9`. Replay: `artifacts/unity/gate_runs/replay/I3/`.

| Criterion | Measured | Result |
|---|---|---|
| Twin2T + Twin2S -> Unity -> ECMER -> 2S + 2T | 19 structural readings; 12 associated, 0 wrong, 7 NO_MATCH; Unity frames forwarded: range 9, sonar 9, camera 29; 133 Model2S and 12 Model2T DIRECT revisions; leakage scan 1229 texts, 0 violations | PASS |
| robot sees only partial infrastructure | far-side defect patch max visible fraction 0.0 over 30 checks; 3 of 6 components observed | PASS |
| persistent technical belief | 12 DIRECT revisions, all with evidence and corrosion OBSERVED; max revision 13; target condition UNKNOWN, U_O 1.0. The 3 never-seen components have conditions **INFERRED** (U_O 1.0), UNKNOWN (U_O 1.0) and NO_BELIEF | **FAIL**: the criterion requires hidden components to be UNKNOWN |

Replay: events 1322 = 1322, decisions 15 = 15, revisions 214 = 214, trajectory max |Δ| = 0.0.

## Honest notes

- **Changed test logic.** During this work another agent changed `conrad/sim/mission/world.py` (16:49) and
  `conrad/domains/technical/*`. One intermediate run showed near-side readings of the I3 target, which gave a
  condition of OBSERVED/DEGRADED. The final run shows none. I rewrote the "hidden components UNKNOWN" check to use the
  truth labels: any component never seen by a structural reading must be UNKNOWN. Before, it checked only the target
  segment. No threshold was changed. I1 was re-run after these changes and its numbers were identical.
- **Pressure-depth bias.** The mission WORLD puts the surface at z = +30 m, while the EKF assumes depth = −z. Unity
  keeps the physical surface at +30 m, so buoyancy is real. The depth sensor is given a declared bias of −30 m, so it
  reads −z, which is the kernel's convention. The estimator should take the surface datum from the mission context.
  That is a gap for the robotics owner.
- **I2 bridge timeout.** One earlier I2 recording ran 3 players at once, plus other agents' jobs. NAV-004 hit a 30 s
  bridge timeout in that run. The recorded I2 evidence comes from a sequential rerun.
- **Scope.** The camera is forwarded as a raw RGB observation (stored with provenance) and is not used by Model2S.
  Scheduled faults, Twin2E, currents and dynamic entities are refused on the Unity mission path.
