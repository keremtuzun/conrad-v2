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

## Safe-hold under lost localization (2026-09-20)

The flagship mission (`docs/audits/FLAGSHIP_UNITY.md`, section 7) found a safety defect in the HOLD behaviour.
It is fixed here, in the supervisor/monitors layer, and I1 and I2 were re-recorded on the current player and the
current code. Every number in this section comes from runs executed for it.

### The defect

The position fix was lost at 90.0 s of the 150 s flagship mission and never returned. The supervisor behaved as
designed on the state machine: DEGRADED 12.0 s after the outage, LOCALIZATION_LOST and HOLD after 23.6 s, held to
the end. The vehicle did not stop. The RobotConfig safe-hold action is STATION_KEEP
(`configs/robot/sim_reference.yaml`), and station keeping runs a position controller against the vehicle's own
estimate. With no fixes the EKF dead-reckoned, the estimate walked 4.74 m away from the hold point and the
controller chased it: true mean speed 0.207 m/s over the last 10 s, the TRUE position outside the mission
boundary from 122.6 s (276 of 1500 control steps, hard constraint `MISSION_BOUNDARY_VIOLATION`), a
`COLLISION_ENVELOPE` violation at 138.4 s and a minimum true clearance of 0.058 m. Final estimator sigma 4.63 m
against a true error of 0.53 m: the filter was honest, the hold action was not.

Holding station against an estimate that has just been declared untrustworthy is unsafe, because the reference
and the feedback come from the same failed source. The `MISSION_BOUNDARY_VIOLATION` was raised while the state
was already HOLD, so nothing escalated either.

Two corrections to the flagship write-up, both checked here:

- the I2 NAV fault case did not pass because of a ZERO_THRUST configuration. The NAV runs use the same
  `sim_reference` RobotConfig with STATION_KEEP. It passed because the fault fires 10 s into a 50 s run, so the
  vehicle station kept on the lost estimate for only about 21 s and drifted 0.019 m in that time;
- the boundary rules below are inert for the NAV benchmarks, whose `SafetyConfig.boundary` is None.

### The rule

**A HOLD may not command a closed loop on an untrustworthy estimate.** In
`conrad/robotics/safety/supervisor.py` the assessment now carries the safe-hold action the consumer must execute,
and that action is not read from RobotConfig when the estimate is bad. The estimate is "not trustworthy" when any
of these holds, which are exactly the conditions that already force HOLD:

| Condition | Reason code |
|---|---|
| no state at all | `STATE_MISSING` |
| state older than `safety.state_stale_after_s`, or the wrong clock domain | `STATE_STALE` |
| estimator health FAULT, or the estimator declares it | `LOCALIZATION_LOST` |
| no position covariance, or sigma above `safety.max_pose_sigma_m` | `LOCALIZATION_LOST`, `POSE_SIGMA_HIGH` |

When one of them holds, the assessment reports `STATE_ESTIMATE_NOT_TRUSTWORTHY`, sets `state_trustworthy = False`
and replaces the configured action by `SafetyConfig.untrusted_state_hold_action`, whose default is
`ZERO_THRUST`. The reason code that already named the action (`SAFE_HOLD_ACTION:<action>`) now names the
**effective** action, so the event log shows `SAFE_HOLD_ACTION:ZERO_THRUST` rather than the configured
`STATION_KEEP`. `SafetyAssessment.safe_hold_action` and `.state_trustworthy` are new public fields, so every
consumer inherits the rule; `NavigationStack._reference` reads the assessment instead of
`supervisor.safe_hold_action`.

Why zero thrust and not a station keep with a wider dead band or a slower gain: there is no trustworthy reference
to hold. A dead band would only change how far the estimate has to walk before the chase starts. Zero thrust is
the one action whose safety does not depend on the estimate being right, it is already the defined response to
`STATE_STALE`, and it is in the ch20 `safe_hold_action` vocabulary, so no new hardware semantics are introduced.

It stays configurable. `untrusted_state_hold_action` accepts `ZERO_THRUST` (default) or `SURFACE`, and a
validator refuses `STATION_KEEP` with the reason, because `STATION_KEEP` is the one action in the vocabulary that
closes a loop on the estimate (`ESTIMATE_CLOSED_LOOP_ACTIONS` in `monitors.py`). `SURFACE` is a declared
ascend-and-hold: with an untrustworthy estimate the stack drives only the depth channel, which the pressure
sensor observes directly and which a position-fix outage does not touch, and it follows the estimate
horizontally instead of chasing a frozen point. A missing or stale state is still a full stop whatever the
configuration says, which is the pre-existing guarantee, kept.

`STATION_KEEP` is unchanged while localization is valid. A HOLD for `COLLISION_ENVELOPE`, `DEPTH_LIMIT`,
`ALTITUDE_LIMIT` or a boundary breach with a good estimate still station keeps and still authorizes thrust.

### Boundary enforcement under lost localization

Decision: **the mission boundary is judged on the worst case the estimate admits, and a real breach that the
vehicle cannot navigate out of escalates to a latched emergency stop.** Implemented in `assess`:

1. **Worst case, not mean.** The containment test inflates the estimate by `boundary_sigma_k * sigma_pose`
   (`SafetyConfig.boundary_sigma_k`, default 3.0, the same k the depth limit already uses). If the ball is not
   fully inside the box, the supervisor raises `MISSION_BOUNDARY_RISK` and holds. A drifting estimate therefore
   stops the vehicle before the true position can be outside, instead of after. This is what the flagship needed:
   the nominal estimate crossed at 122.7 s, while sigma had been above 2 m since about 120 s.
2. **Breach plus lost estimate escalates.** If the nominal estimate is outside the boundary while the state is
   untrustworthy, the supervisor latches EMERGENCY_STOP with `MISSION_BOUNDARY_BREACH_UNLOCALIZED`. The reasoning
   is that the vehicle has left its declared operating volume and, by rule 1 above, may not use its estimate to
   come back, so no autonomy should resume motion until an operator acknowledges. The latch is cleared only by
   `reset_latch`, as for leak and operator E-stop.
3. **A breach with a good estimate does not escalate.** It stays HOLD with `MISSION_BOUNDARY_VIOLATION` and
   station keeping, which is the behaviour that can actually recover the vehicle.

`boundary_breach_escalates_to_estop` (default true) makes rule 2 configurable; with it off the breach stays a
HOLD, and rule 1 of the safe-hold section still stops the vehicle. Rules 1 and 2 only apply where a boundary is
configured: `MissionRuntime` sets it from `MissionContext.spec`, and the NAV benchmarks have none.

### Tests

| Test | What it fixes in place |
|---|---|
| `tests/unit/robotics/test_nav_safety.py::test_hold_with_lost_localization_commands_no_motion` | STATION_KEEP config, sigma above the limit: HOLD carries `SAFE_HOLD_ACTION:ZERO_THRUST` and `STATE_ESTIMATE_NOT_TRUSTWORTHY`, a thrusting command is refused with `ZERO_THRUST_REQUIRED` and an all-zero command is authorized |
| `...::test_every_untrustworthy_state_refuses_station_keeping` | the same for estimator FAULT, an estimator-declared `LOCALIZATION_LOST`, a missing state and a stale state |
| `...::test_station_keep_survives_a_hold_while_localization_is_valid` | `COLLISION_ENVELOPE`, `DEPTH_LIMIT` and `ALTITUDE_LIMIT` holds still station keep and still authorize thrust |
| `...::test_untrusted_state_hold_action_refuses_a_closed_loop_action` | the validator, the ZERO_THRUST default, the SURFACE alternative, and that a missing state stops anyway |
| `...::test_worst_case_boundary_holds_before_the_estimate_crosses` | 3 sigma inflation: NORMAL at sigma 0.1 m, `MISSION_BOUNDARY_RISK` HOLD at sigma 0.5 m, same position |
| `...::test_boundary_breach_with_a_lost_estimate_escalates_to_emergency_stop` | breach plus lost estimate latches EMERGENCY_STOP, survives the vehicle being back inside, clears on `reset_latch` |
| `...::test_boundary_breach_escalation_is_configurable` | with escalation off the breach is a HOLD that still commands zero thrust |
| `tests/simulation/nav/test_safe_hold_lost_localization.py` (5 tests) | the flagship condition on the Python kernel |

The simulation test runs the flagship condition on the L1 kernel, not Unity, so it is fast (22 s): 150 s at a
0.1 s control period, 1 Hz USBL-like fixes until 90.0 s and none afterwards, a 5x IMU noise fault at the outage
so the dead reckoning really drifts (as NAV-007 does), a mission boundary, and the commands actually submitted
through the CommandGateway. Measured on seed 9: HOLD with LOCALIZATION_LOST at 112.9 s (22.9 s after the
outage), then 371 control steps in which the maximum absolute authorized thruster command is **0.0**, true mean
speed over the last 10 s **0.0447 m/s** (threshold 0.1), **0 of 1500** steps with the true position outside the
boundary, 0 collisions. Over the same window the estimate ran away by **1.935 m**, so this is the flagship
condition and not a quiet run: the vehicle simply did not follow it, the true horizontal drift after HOLD being
**0.302 m**.

**One existing test asserted the old behaviour and was updated.**
`tests/unit/robotics/test_nav_estimator_calibration.py::test_growing_sigma_slows_then_holds_with_configured_policy`
asserted that a STATION_KEEP RobotConfig kept `zero_thrust_required` false in a LOCALIZATION_LOST HOLD. That is
the defect. It is now
`test_growing_sigma_slows_then_stops_whatever_the_configured_hold_action` and asserts the safe outcome for both
parametrized configurations: the effective action is ZERO_THRUST and every thruster command is 0.0. The rest of
the test (the NORMAL, DEGRADED, HOLD sequence, the sigma bands, the speed scale, the fixed hold reference) is
unchanged. No threshold was moved anywhere, and no other test was touched.

Suites run on the changed paths: `tests/leakage`, `tests/contract`, `tests/unit/robotics` and
`tests/simulation/nav`, **317 passed**. ruff and mypy are clean on `conrad/robotics`.

### Before and after on the kernel (scratch measurement, not a test)

To show that the rule changes what the vehicle does and not only what the log says, the same 150 s kernel run was
executed once with the pre-2026-09-20 rule restored (a scratch script that allows `untrusted_state_hold_action`
to be STATION_KEEP, with `boundary_sigma_k = 0` and escalation off). Both runs reach HOLD with
LOCALIZATION_LOST at 112.9 s.

| | Old rule (station keep on the lost estimate) | New rule |
|---|---|---|
| max absolute authorized thruster command after HOLD | 0.3054 | **0.0** |
| true horizontal drift from the hold point | 1.479 m | **0.302 m** |
| true vertical drift from the hold point | 0.044 m | 1.440 m (buoyant ascent, see below) |
| true mean speed over the last 10 s | 0.0674 m/s | 0.0447 m/s |

The kernel drift is much smaller than the flagship's 4.74 m, so this is a demonstration of the mechanism, not a
reproduction of its magnitude. The magnitude evidence is the flagship run itself.

### Re-recorded I1 and I2

The Unity player was **not** rebuilt: no C# was changed. It is the binary the flagship run produced
(`ConradSim.exe`, sha256 `36c5c9f13481406382a8e9ef8fc0ea7cdf055c43bb12fc8fd545b07c199cd277`, Unity 6000.5.9f1,
built 2026-09-20T03:51:10Z, 0 errors, 0 warnings), so `same_player_binary` is true in every replay below. Both
gates were recorded sequentially with `scripts/record_unity_gate_evidence.py`, with no other player running. Git
commit at record: `025b302faff03a7dd6037f6349e5894a2b449809`. I3 and I4 were deliberately **not** re-recorded:
Model2T is being repaired in parallel and I3 has to wait for it.

**I2: PASS, all 7 criteria, 13 of 13 tests in 928.60 s.** Seeds 7400001 to 7400006
(`configs/eval/partitions_nav.yaml` final_test), unchanged.

| Benchmark | Final error (m) | Other metric | Estimation RMS (m) | Result |
|---|---|---|---|---|
| NAV-001 | 0.085 | | 0.112 | PASS |
| NAV-002 | 0.055 | | 0.112 | PASS |
| NAV-003 | 0.070 | min clearance 0.442 m, 0 collisions | 0.104 | PASS |
| NAV-004 | 0.023 | standoff RMS 0.013 m (bound 0.25) | 0.095 | PASS |
| NAV-005 | 0.020 | station RMS 0.072 m (bound 0.15) | 0.075 | PASS |
| NAV-006 | 0.076 | | 0.126 | PASS |

Uses estimated state: a +0.5 m fix bias moved the true end point by -0.508 m (tolerance 0.15 m). Replay: NAV-003
and NAV-006 both give a true-trajectory max difference of 0.0 and 3000 = 3000 identical gateway acks, over 427
verified bundle files each.

| Fault case (seed) | Safe state | Time after fault | Motion rule evidence |
|---|---|---|---|
| LOSS_OF_LOCALIZATION (7400001) | HOLD, LOCALIZATION_LOST | 22.10 s (deadline 35 s) | hold-point drift **0.242 m** (radius 1.0 m), true speed **0.00087 m/s** |
| THRUSTER_FAULT on H1 (7400002) | DEGRADED, DEGRADED_MANEUVERABILITY | 0.02 s | 0 commands to H1, 100 % executed |
| STALE_STATE, IMU dropout (7400003) | HOLD, STATE_STALE, zero thrust | 0.52 s | all-zero commands, speed 3.8e-5 m/s |
| LOW_BATTERY (7400004) | RETURN, BATTERY_LOW | 0.02 s | all-zero commands, speed 2.2e-5 m/s |
| LEAK (7400005) | RECOVER, LEAK_DETECTED | 0.02 s | all-zero commands, speed 1.7e-5 m/s |

The LOSS_OF_LOCALIZATION case is the one the change touches. Its recorded reason codes are now
`LOCALIZATION_LOST`, `POSE_SIGMA_HIGH`, `ALTITUDE_UNMONITORED`, `STATE_ESTIMATE_NOT_TRUSTWORTHY`,
`SAFE_HOLD_ACTION:ZERO_THRUST`. All 900 commands issued after the safe state was reached were authorized and
executed, and the true speed fell from 0.0209 m/s on the previous recording to **0.00087 m/s**, a factor of 24.
The estimated hold-point drift rose from 0.019 m to 0.242 m, well inside the declared 1.0 m radius, because
nothing is correcting the estimate any more. The declared expectation in `configs/sim/nav_fault_cases.yaml` was
**not** changed, and neither was any threshold; only the comment that described "hold" as station keeping.

**I1: PASS, all 9 criteria, 10 of 10 tests in 148.89 s.** Seed 7800000
(`configs/eval/partitions_unity_gates.yaml` final_test).

| Criterion | Measured | Result |
|---|---|---|
| robot moves through synthetic world | true path 15.93 m in 60 s, 600 commands via `unity_v2`, 0 collisions, min clearance 0.695 m | PASS |
| persistent map | 275 spatial revisions, all DIRECT; max revision 16; 50 beliefs | PASS |
| geometry | 206 occupied-claimed cells; median \|Twin2S sdf\| 0.101 m, p90 0.165 m; 100 % within 0.5 m | PASS |
| occupancy | 78 of 1200 probes claimed free; wrong 0.0 | PASS |
| coverage | lane region 0.101; target segment 0.457; 913 observed cells | PASS |
| observed/inferred/unknown | OBSERVED 913, INFERRED 12, UNKNOWN 3655; 500 of 500 points below the seabed UNKNOWN | PASS |
| uncertainty | mean U_O: OBSERVED 0.052, INFERRED 0.856, UNKNOWN 0.852; per-channel std 0.211, 0.018, 0.035, 0.404 | PASS |
| provenance | 25 of 25 evidence items re-derive bit-exactly from their raw Unity frames | PASS |
| no Twin truth leakage | 2226 runtime texts scanned, 0 violations | PASS |
| replay | events 2520 = 2520, decisions 30 = 30, revisions 343 = 343, trajectory max \|d\| 0.0; 156 files, 97 objects | equal |

`conrad gates status` after both recordings: U0 PASS, 2S-FIRST PASS, I1 PASS, I2 PASS, 2T PASS, I3 PASS,
I4 FAIL. Unchanged by this work except that I1 and I2 now rest on current evidence.

### Honest notes

- **Zero thrust is not a hover on the kernel vehicle.** `sim_reference` is 11.5 kg with a displaced volume of
  0.01125 m3, so at 1025 kg/m3 it is positively buoyant by about 0.31 N and rises at roughly 0.045 m/s with the
  thrusters off. That is the 1.440 m of vertical drift in the table above, and it is the declared fail-safe
  behaviour of a positively buoyant vehicle, not a chase. It does mean that a long enough zero-thrust hold will
  eventually reach the top face of a mission boundary or the surface: about 22 s per metre of headroom at that
  rate. The Unity player's vehicle does not show it (0.00087 m/s over the I2 fault run). Recorded as OPEN: a
  depth-holding variant that keeps the vertical loop, which a position-fix outage does not invalidate, and drops
  only the horizontal one, would be strictly better than either option in the vocabulary. `SURFACE` is the
  nearest available approximation and is implemented, but no gate exercises it.
- **The boundary rules are untested on the integrated mission path.** They are covered by unit tests and by the
  kernel simulation test. Re-running the flagship mission is not part of this workstream, so the claim that they
  would have prevented the 122.6 s excursion is an argument from the recorded sigma trace, not a measurement.
- **`MISSION_BOUNDARY_RISK` can hold a mission that plans close to its boundary.** With the default k of 3 and a
  healthy sigma of about 0.06 m the margin is 0.18 m, so this only bites in the last 20 cm. It is a new way for a
  mission to stop, and it is deliberate.
- **The before/after kernel table is a scratch measurement**, produced by a throwaway script that monkeypatches
  the closed-loop guard away. It is not in the test suite and cannot be reproduced by running the tests.
- **Deadlines and the 0.1 m/s stop threshold are unchanged evaluation allowances**, as the I2 repair section
  already states.
- **LOCALIZATION_LOST now fires at 22.10 s on seed 7400001, where the 2026-09-19 I2 repair table reports
  28.64 s.** The evidence file that was in the tree before this work (recorded earlier on 2026-09-20, after the
  flagship player rebuild and not by this workstream) already read 22.1 s, so the change predates the safe-hold
  rule and is not caused by it. I did not investigate which change moved it. Both are inside the declared 35 s
  deadline.
- **I1 and I2 evidence had already been re-recorded earlier on 2026-09-20** by another run, on the same seeds and
  the same player. The numbers in the I1 and I2 tables above are from my own runs and are the ones in
  `artifacts/gates/{I1,I2}/`. The NAV benchmark numbers are unchanged by the safe-hold rule, as expected: it only
  acts once the estimate is declared untrustworthy, which on the NAV benchmarks happens only in the
  LOSS_OF_LOCALIZATION fault case.
