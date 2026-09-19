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

## I2 repair (2026-09-19)

Formal I2 is now **PASS** on fresh held-out seeds. All 7 criteria pass, including the new ch26 Phase 6 criterion
"faults reach defined safe states". `conrad gates status` shows I2 official=PASS formal=PASS. This section
supersedes the I2 table and the "Pressure-depth bias" note above.

### Cause of the NAV-005 failure

- **The 0.15 m bound is an engineering estimate.** The spec (ch20 NAV-005, ch25 I2, ch26 Phase 6) names station
  keeping but gives no number. The bound comes from `configs/sim/nav_benchmarks.yaml`. It was kept unchanged.
- **The cause was estimator noise at hover.** It was not the controller, the current or the fix rate.
  - On 4 development seeds the station error (true position against target) was 0.140 to 0.147 m.
  - The estimation error alone was 0.078 to 0.101 m per horizontal axis, close to the 0.10 m sigma of one fix.
- **Mechanism.** The EKF was tuned conservatively for outage honesty (NAV-CAL-E001).
  - It used a 0.25 m/s white velocity-prior error and a 0.02 m/s/sqrt(s) random walk on the mismatch state c.
  - At hover this kept the velocity sigma near 0.07 m/s and the c sigma near 0.086 m/s.
  - Before each 1 Hz fix the position sigma was 0.12 m, more than the fix sigma. Each fix moved the estimate
    about 60 % of the way.
  - The estimate followed the fix noise, and the controller tracked the estimate.
- **What was ruled out (development seeds).**
  - Halving the position gain: 0.136 m.
  - Lowering only the accel noise: 0.138 m.
  - Lowering only the walk: 0.142 m.
  - Turning the velocity prior off: 0.146 m.
  - Only a tight prior combined with a small walk while fixes arrive helped.

### Fix

- **Quiet velocity prior** (`EkfConfig.velocity_prior_quiet_sigma_mps` = 0.03/0.03/0.4 m/s).
  - The white prior error rises linearly to the full 0.25 m/s at a body turn rate of 0.5 rad/s. The static
    thrust/drag model misses turning coupling, not the steady state.
  - Scheduling on model acceleration was tried and dropped. The hover control loop alone commands about
    0.17 m/s^2.
- **Walk of c while fixes arrive** (`mismatch_walk_observed_mps_per_sqrt_s` = 0.005 m/s/sqrt(s)).
  - After 3 s without an accepted fix (`blind_after_s`), the full 0.02 m/s/sqrt(s) applies again.
  - While fixes arrive, a faster change shows up in the NIS, which scales the walk (existing covariance matching).
- `EkfConfig.legacy_baseline()` sets both new fields to None, so EST-B0 is unchanged. All values are
  SYNTHETIC_ONLY configured defaults.
- **Cost: LOCALIZATION_LOST now fires later.** After a fix outage it fires after about 28.7 s on the development
  and final runs. NAV-CAL-E001 reported 18.6 s with the old defaults.
- **NAV-CAL development check.** Seeds 101 to 105, all outage lengths, IMU noise 1x and 5x:
  - Missed danger stayed at 0 s.
  - Max error/sigma was 0.93 to 0.96 (1.06 before).
  - Mean outage error roughly halved.
- **Not re-run:** the held-out NAV-CAL table in ESTIMATOR_CALIBRATION.md. It still describes the old defaults.

### Depth convention

- **Surface height from the mission context.**
  - `MissionContext.water_surface_z_m` is filled from the scenario in `conrad/sim/mission/registry.py`.
  - `EkfConfig.water_surface_z_m` and `SafetyConfig.water_surface_z_m` replace the hard-coded 0. The depth residual
    is (s - depth) - z. The depth limit, the goal envelope check and the SURFACE hold height all use s.
  - `NavigationStackConfig.for_surface(s, ...)` sets both fields, and a validator refuses a mismatch.
    `MissionRuntime` (`conrad/orchestration/mission.py`) uses it.
  - The kernel (`SimKernelConfig.water_surface_z_m`) reports true depth s - z.
- **world.py: one minimal change.** `MissionWorld.build` passes the scenario surface to `SimKernelConfig`.
- **Unity.** The -30 m pressure-sensor bias is removed from `unity_robot_config`. Unity now reports true depth
  below its physical surface at +30 m.
- **Check on the Unity mission path.** `I1-UNITY`, development seed 5100001, `artifacts/runs/i2repair-depthcheck-s5100001`:
  - True z was 1.24 to 1.32 m, which is about 28.7 m below the surface.
  - The estimated z error had a mean of 3 mm and a maximum of 30 mm.
  - There were no DEPTH_REJECTED events.
- The NAV benchmarks keep s = 0.

### Faults reach defined safe states (new I2 criterion)

`configs/sim/nav_fault_cases.yaml` defines five cases on the NAV-001 transit. The fault is injected at 10 s and each
run lasts 50 s. The runner is `run_unity_nav(faults=...)` and the checker is `conrad/sim/mission/unity_faults.py`.

Injection is on the adapter side:

- Unity faults go through the bridge INJECT_FAULT.
- Loss of localization is an outage of the synthetic USBL-like fix.

Each case must meet all of these:

- nothing worse than DEGRADED before the fault;
- the declared state and reason within a deadline, held to the end of the run;
- a motion rule;
- no collision.

**Defects found on development seeds and fixed.** In the first development runs every case reached the right
supervisor state, but the vehicle kept executing its last accepted command. In the LOW_BATTERY run it circled at
about 0.37 m/s.

- **Unity player.** The C# was changed and the player rebuilt once. The build log
  (`unity/ConradUnityV2/Logs/i2_repair_build.log`) shows 0 `error CS` and 0 `warning CS`.
  - It refused even all-zero stop commands in RETURN, RECOVER and EMERGENCY_STOP. It now accepts them, as the
    Command Gateway does.
  - It had no command-timeout watchdog. It now zeroes the thrusters at the last accepted command's deadline. This
    is the same rule as the Python kernel.
- **NavigationStack.** In any state that forbids motion it now sends an explicit all-zero command. Before, it
  sent a motion command that was then refused.
- **Unity adapter (`health_from_state`).**
  - Device names such as `thruster:H1` and `sensor:imu` are mapped to the RobotConfig IDs.
  - A FAULT caused only by failed thrusters is reported as overall DEGRADED, as on the kernel. Before, the gateway
    refused every command (INVALID_HEALTH_STATE), so graceful degradation (ch20) never happened.
  - The SafetySupervisor no longer raises a HARDWARE_FAULT HOLD for a FAULT that failed thrusters fully explain.
- **Checker, tightened after this finding and before any final run.**
  - "zero" now also requires the vehicle to stop: mean true speed over the last 10 s of at most 0.1 m/s.
  - "avoid_target" requires at least 90 % of commands executed.
  - LOW_BATTERY changed from "refused" to "zero".

### Seed partition

- **Old seeds contaminated.** 7300001 to 7300006 are CONTAMINATED_FOR_FINAL_EVALUATION and are now development
  seeds.
- **New pinned file.** `configs/eval/partitions_nav.yaml` (version partitions-nav-2026-09-19-v1) is pinned by
  `NAV_PARTITIONS_SHA256` in `conrad/evaluation/partitions.py`. It was written and pinned before any run.
  - development: 7100001, 7300001 to 7300006, 7410001 to 7410020.
  - final_test: 7400001 to 7400006, one per benchmark. Fault case k uses final seed k.
- **Why a separate file.** Adding a domain to `configs/eval/partitions.yaml` would change its v1 digest. The I4
  artifacts and `configs/active/mcbr_frozen.yaml` pin that digest. `partitions.yaml` only got a comment, and its
  digest is unchanged.
- **Collision check.** No 74xxxxx seed was in use anywhere. The NAV final seeds are also disjoint from the I5
  action-matrix seeds (7100000 to 7100009 and 7300000 to 7300009). The old I2 seeds overlapped the I5 final seeds.

### Development numbers (development seeds only)

**NAV-005 station RMS, kernel, 10 seeds (7410001 to 7410010).**

| Defaults | Mean | Max |
|---|---|---|
| Before the fix | 0.131 m | 0.147 m |
| After the fix | 0.092 m | 0.110 m |

**NAV-005 station RMS, Unity.**

| Seed | Station RMS |
|---|---|
| 7410001 | 0.099 m |
| 7410002 | 0.094 m |
| 7410003 | 0.075 m |
| 7410004 | 0.093 m |
| 7410016 | 0.097 m |
| 7410017 | 0.094 m |

**Other development checks.**

| Check | Seeds | Result |
|---|---|---|
| Kernel NAV-001..NAV-008, old and new defaults | 7410011 to 7410013 | all 21 runs pass with both |
| Unity NAV-001..NAV-006 | 7410016, 7410017 (rebuilt player) | 12 of 12 pass |
| Fault cases, Unity, rebuilt player | 7410011 to 7410015 | 5 of 5 pass |

With the new defaults the kernel final errors are mostly lower. NAV-006 estimation RMS rose slightly, from
0.110-0.124 m to 0.117-0.133 m.

### Final run

The final run used seeds 7400001 to 7400006 on the rebuilt player and was recorded with
`record_unity_gate_evidence.py I2`. Result: 13 of 13 tests passed. The strict-xfail markers were removed before
this run.

| Benchmark | Final error (m) | Other metric | Estimation RMS (m) | Result |
|---|---|---|---|---|
| NAV-001 | 0.085 | | 0.112 | PASS |
| NAV-002 | 0.055 | | 0.112 | PASS |
| NAV-003 | 0.070 | min clearance 0.442 m, 0 collisions | 0.104 | PASS |
| NAV-004 | 0.023 | standoff RMS 0.013 m | 0.095 | PASS |
| NAV-005 | 0.020 | **station RMS 0.072 m (bound 0.15)** | 0.075 | PASS |
| NAV-006 | 0.076 | | 0.126 | PASS |

- **Uses estimated state.** A +0.5 m fix bias moved the true end point by -0.509 m.
- **Replay.** For NAV-003 and NAV-006 the trajectory max |d| was 0.0, and the acks matched (3000 = 3000).

| Fault case (seed) | Safe state reached | Time after fault | Motion rule evidence |
|---|---|---|---|
| LOSS_OF_LOCALIZATION (7400001) | HOLD, LOCALIZATION_LOST | 28.64 s (deadline 35 s) | hold-point drift 0.019 m, speed 0.021 m/s |
| THRUSTER_FAULT on H1 (7400002) | DEGRADED, DEGRADED_MANEUVERABILITY | 0.02 s | 0 commands to H1, 100 % executed |
| STALE_STATE, IMU dropout (7400003) | HOLD, STATE_STALE, zero thrust | 0.52 s | all-zero commands, stopped by the watchdog (speed 4e-5 m/s) |
| LOW_BATTERY (7400004) | RETURN, BATTERY_LOW | 0.02 s | all-zero commands, speed 2e-5 m/s |
| LEAK (7400005) | RECOVER, LEAK_DETECTED | 0.02 s | all-zero commands, speed 2e-5 m/s |

### Honest notes

- **The final seeds were touched once before the final run.**
  - The first evidence command was run with `--help`. The script does not parse that flag, so it started
    recording all gates.
  - I stopped it after I1 had finished. By then the I2 module had completed NAV-001 on seed 7400001 and had
    started NAV-002.
  - No I2 result was read or recorded, and nothing was changed afterwards. The complete final run then executed
    once on the same seeds.
- **I1 was re-recorded by that accidental run.** It ran on its final seed 5300059 with the new code and passed 10
  of 10. `artifacts/gates/I1/evidence_formal.json` is now from that run.
- **I3 evidence was not re-run.** It was recorded before this change, with the -30 m bias.
- **LOW_BATTERY uses a smaller battery.** The case declares a 20 kJ SYNTHETIC_ONLY battery, because the player
  clamps the capacity factor at 0.01.
- **RETURN has no return behaviour yet.** RETURN and RECOVER are stops, as the gateway already states
  (EXT-HW-05 is OPEN).
- **Thresholds are evaluation allowances, not spec numbers.** The deadlines were set before any run. The 0.1 m/s
  stop threshold was added after the development finding and before the final run.
