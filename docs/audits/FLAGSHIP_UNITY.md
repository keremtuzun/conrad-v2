# FLAGSHIP-UNITY: one integrated mission through Unity with all three stressors

Date: 2026-09-20. Declaration: `configs/eval/flagship_unity.yaml` (written before the final world ran).
Entry point: `scripts/run_flagship_unity.py`. Artifacts: `artifacts/flagship/` (small JSON, tracked) and
`artifacts/falsification/flagship_unity__FLAGSHIP-UNITY-s7800017/report.json`.

**This is a demonstration run, not a gate.** No criterion in `conrad/evaluation/gates.py` reads it and it
promotes nothing. I4 still FAILS, so I5, I6 and I7 stay blocked upstream. Evidence validity is
`L1_APPROXIMATE_PHYSICS`; every physical, sensor and link number is SYNTHETIC_ONLY.

Every number below comes from runs executed for this audit.

## 1. Result in one paragraph

The mission flew the full declared 150 s on the held-out world 7800017 through the built Unity player and
carried all three stressors. The critical structural finding was made at 14.3 s while the link was down, and
it reached the shore receiver 1.25 s (alert) and 6.97 s (belief delta) after reconnection, ahead of every
routine delta. The second structural payload produced a credible disagreement that drove U_C from 0.35 to
1.00 instead of being averaged away, and it also pulled the corrosion estimate 7.6 mm away from truth. The
position-fix outage drove the estimator to DEGRADED after 12.0 s and to LOCALIZATION_LOST after 23.6 s, and
the supervisor declared HOLD and held it to the end. **The vehicle did not stop.** In HOLD it kept station
keeping against a dead-reckoning estimate that was drifting, so the true position left the mission boundary
at 122.6 s and the true mean speed over the last 10 s was 0.207 m/s. That is the main negative finding of
this run. The bundle replays bit-exactly, both from observations alone with all truth removed
(falsification verdict PASS) and by full re-execution through a fresh player (13 of 13 compared items equal,
Unity true trajectory max difference 0.0).

## 2. Pinned identities

| | |
|---|---|
| Scenario | `FLAGSHIP-UNITY` (kernel definition in `conrad/sim/mission/scenarios.py`, Unity entry in `conrad/sim/mission/unity_run.py`) |
| Scenario version / UUID | `twin2s-ocpwe-0.1.0` / `018bcfe5-680d-7145-8f1d-02aded2438d0` |
| World family / seed | `straight_pipeline` / **7800017** (unity_gate final_test, partition digest `ad753291...f12f88c`) |
| Config | `configs/sim/mission_test_small.yaml`, resolved digest `e4a3c1c3f1397584da2c7be83654079763d08cfb17ce7e649c77e8e1ece7fc1a` |
| Sensor configuration digest | `2f8c7f8b6308b4b99d807458b8bec05d13bd09c5def67af0cf40afede6086edf` |
| RobotConfig | `artifacts/unity/robot/sim_reference_unity_mission-2028d46fe2ac.yaml`, digest `2028d46fe2ac345ccc626365c74907b7786630acfc50c1b92e7bcc6a530ba87a` |
| Twin versions | twin2s / twin2e `twin2s-ocpwe-0.1.0`, twin2t `twin2t-mcde-0.1.0` |
| Model versions | Model2T `model2t-analytic-0.1.0`, Model2S `uahsm-v1-analytic`, Model2E `model2e-uncoupled-obsctx-analytic-0.5.0`, planner `PRODUCTION[V-bayes_eig_ratio]` |
| git commit at record | `757f352a1cfa235d38ff4483c9e6d8f6d3f23842` |
| Architecture / stack | `conrad_v2_impl_freeze_v0_2` / `conrad_v2_stack_freeze_v0_2` |
| Player | `ConradSim.exe`, sha256 `36c5c9f13481406382a8e9ef8fc0ea7cdf055c43bb12fc8fd545b07c199cd277`, Unity 6000.5.9f1, built 2026-09-20T03:51:10Z, 0 errors, 0 warnings |
| Unity scene digest | `2b163298e764ea65e8aea4c108bc3ee712c057002e863614ff407ac255252a94` |
| Bundle | `artifacts/runs_capture/flagship_unity/FLAGSHIP-UNITY-s7800017` (338 files, 233 objects; not committed) |
| Code digest at record and at replay | `68bfade5aaaaad7e2ba5fd1d84f94e25b65b0bcb1977d300c9afc82cd415a9a2` (identical) |

Runtime, checked before the flight and refused if it differs: planner `PRODUCTION`, `model2t_mode = NONE`
(TCDP off), Model2S refinement disabled (RBP off), Model2E enabled with `observability_context = true` and
`ecological_coupling = false`, multi-domain sensing gate off, analytic operators on all three children,
duration 150 s, control period 0.1 s.

## 3. The scenario

One mission, three stressors, each at a time declared before the final world ran.

| Stressor | Declared | Mechanism |
|---|---|---|
| Communication outage | link down `[6 s, 70 s)` on the mission link (1200 bps, 1 s latency, 5 % loss, BER 1e-6, 1024-bit packets) | `MissionRuntimeConfig.link.outages_s`. The defect sits on the LANE side (`defect.side = near`, as in `I7-OUTAGE-CRITICAL`), so the lane pass itself makes the critical finding while the link is down. |
| Contradiction | second structural payload from 16.0 s, wall-loss offset +10 mm, `corruption = 0`, surface-appearance offset -0.8 | `ContradictionOptions`: the existing `SENSOR_INSPECTION_AUX` Twin2T sensor. Unlike INT-003 it is **not** corrupted, so Model2T rates it reliable (`direct.conflict_min_reliability = 0.6`) and the disagreement is credible rather than dismissed. |
| Localization degradation | position-fix outage from 90.0 s, duration 90 s (so it never returns inside the mission) | `FaultSpec` type `FIX_OUTAGE`, realised by `UnityMissionWorld.due_faults` on the Python-rendered synthetic USBL-like fix, exactly as on the kernel. |

### How the times were chosen

Design ran on the unity_gate **development** split only. Seven kernel dry runs (worlds 7810007 to 7810013)
gave the first DIRECT target revision and the maximum U_C reached on the target belief:

| World | first DIRECT target revision | DIRECT target revisions | max U_C |
|---|---|---|---|
| 7810007 | 14.3 s | 7 | 0.350 |
| 7810008 | 16.3 s | 52 | 0.350 |
| 7810009 | 19.3 s | 3 | 0.000 |
| 7810010 | 15.3 s | 8 | 0.350 |
| 7810011 | 13.3 s | 8 | 0.871 |
| 7810012 | 15.3 s | 8 | 0.529 |
| 7810013 | 55.3 s | 47 | 0.588 |

16.0 s is the value that leaves at least one single-payload reading before the second payload on the worlds
whose lane pass reaches the target at 13.3 to 15.3 s. It is a compromise, not a guarantee: on 7810009 the
target was first read at 19.3 s, so there was no single-payload phase and U_C never rose. That failure mode
is a property of this design and is reported, not hidden. The fix-outage start (90 s) leaves 60 s, which the
I2 repair showed is about twice the time LOCALIZATION_LOST needs (28.6 s on the NAV transit). One Unity
development flight (7810007) then exercised the whole path before the final world ran; its numbers are in
`artifacts/flagship/FLAGSHIP-UNITY-s7810007.json` and are quoted in section 9.

## 4. Structural estimate against truth

Twin2T's **realistic** sensor model is in force (`docs/audits/STRUCTURAL_LINEAGE_AUDIT.md`): crack sizing
over the in-view part only, log-logistic POD, multiplicative error and a persistent per (component, sensor)
bias that repeated looks cannot average away. The old optimistic additive model, which produced the 0.24 mm
crack error of the 2026-09-18 flagship, is not used.

| Quantity | Twin2T truth at 150.1 s | Final Model2T belief | Absolute error |
|---|---|---|---|
| `corrosion_depth_m` | 0.0060027 m | 0.0135866 m | **0.007584 m** |
| `crack_length_m` | 0.0800026 m | 0.0706061 m | **0.009397 m** |

Final belief (revision 82, CONTEXT at 150.0 s): knowledge status OBSERVED, condition **FAILED** (OBSERVED),
U_A 0.292, U_E 0.200, U_C 0.490, U_O 0.288.

The corrosion estimate is 2.3x the truth. That is the contradiction working as designed: the second payload
carries a declared +10 mm wall-loss offset, and Model2T keeps both hypotheses' spread instead of picking the
first payload, so the belief sits between a 6 mm reading and a 16 mm reading. The crack error of 9.4 mm on an
80 mm crack is in the band the realistic sensor model produces (24.4 mm on the old flagship world, 24.7 mm on
development world 7810007 here).

Readings on the target component: 22 structural labels, 12 of them on the defect patch (7 from the primary
payload at 14.3 to 20.3 s, 5 from the second payload at 16.3 to 20.3 s) and 10 on the rest of its surface.
The patch visibility rose from 0.229 at 13.3 s to a maximum of 0.896 at 18.3 s and fell back to 0.104 by
22.3 s. The target belief has 83 revisions in total, 6 of them DIRECT. **No INSPECT goal was ever accepted**
(`inspection_goal_started_s = null`): the lane pass alone produced the finding, and all four MCBR plans came
back `NEED_SATISFIED` with 0 rejected candidates.

## 5. The contradiction and what U_C did

The second payload starts at 16.0 s, so the 16.3 s view is the first that yields two readings per surface.

| t | revision | evidence | U_C | U_A | corrosion (m) | crack (m) |
|---|---|---|---|---|---|---|
| 14.3 s | 14 | 2 | **0.350** | 0.277 | 0.049678 | 0.018852 |
| 16.3 s | 18 | 4 | **0.595** | 0.289 | 0.018450 | 0.043599 |
| 17.3 s | 19 | 4 | **0.945** | 0.291 | 0.011261 | 0.053149 |
| 18.3 s | 22 | 4 | **1.000** | 0.232 | 0.011050 | 0.058338 |
| 19.3 s | 23 | 2 | 0.700 | 0.232 | 0.013230 | 0.065915 |
| 20.3 s | 25 | 2 | 0.490 | 0.292 | 0.013587 | 0.070606 |
| 150.0 s | 82 (CONTEXT) | 0 | 0.490 | 0.292 | 0.013587 | 0.070606 |

- **Before the second payload:** the last single-payload DIRECT revision is 14 at 14.3 s, U_C 0.350.
  U_C was **already non-zero** there, and this is worth stating plainly: the target component is read through
  two truth-side surfaces (the defect patch and the rest of its surface), which disagree with each other by
  construction, so two readings reach the same belief in the same tick from 14.3 s on. The injected
  contradiction is therefore the *second* source of disagreement on this belief, not the first.
- **After:** U_C rises to 0.595, then 0.945, then saturates at 1.000 at 18.3 s. That is `direct.uc_gain`
  (0.35) applied on three consecutive reliable conflicts.
- **The belief was not averaged.** The corrosion claim swings 49.7 mm, 18.5 mm, 11.3 mm, 11.1 mm, 13.2 mm,
  13.6 mm while U_C stays high, which is `conrad/domains/technical/direct.py` widening `level_var` to
  `max(level_var, 0.5*before + 0.5*r + 0.25*innov^2)` on each reliable conflict instead of collapsing to a
  weighted mean.
- **U_C then decays**, to 0.700 at 19.3 s and 0.490 at 20.3 s, because the last two views yield two
  consistent readings each (the second payload leaves the rest-of-surface view first) and
  `direct.uc_resolve_factor = 0.7` applies. The mission ends with U_C 0.490 on a belief whose corrosion
  claim is 7.6 mm from truth. The uncertainty channel reports the disagreement honestly; it does not repair
  the estimate.

## 6. Communications timeline

| Event | Mission time |
|---|---|
| Link drops | 6.0 s (link status DOWN observed at 6.2 s) |
| **Critical finding made** (target condition OBSERVED FAILED, revision 14) | **14.3 s, link DOWN** |
| First critical offer to BAAC | 15.1 s |
| Link returns | 70.0 s (link status UP observed at 70.2 s) |
| **F0 alert arrives at the shore receiver** | **71.253 s** (1.25 s after reconnection) |
| **Critical F1 belief delta applied at the receiver** | **76.971 s** (6.97 s after reconnection) |
| First routine delta after reconnection | 80.658 s, after the critical delta |

Latency from the finding: 56.95 s for the alert, 62.67 s for the delta. The floor is the outage itself: the
finding happens 55.7 s before the link returns. Policy `C-B10_baac`. At mission end the receiver holds
revision 25, equal to the sender's revision 25, with 0 duplicate contributions and 0 resync requests. Queue
maximum 18 units; 23.58 Mbit still queued at the end (BAAC carries full-fidelity evidence it never gets to
send on a 1200 bps link, as I7 already recorded). 90 transmissions, 24 delivered, 105,795 bits, 10.58 J.
83 offers, 0 dropped. Receiver holds 12 beliefs and 1 alert.

## 7. Localization degradation, safety states and whether the vehicle stopped

Position fixes stop at 90.0 s (first refused fix recorded at 90.1 s) and never return.

| t | estimator error (m) | position sigma (m) | estimator health | safety state |
|---|---|---|---|---|
| 89.2 s | 0.108 | 0.056 | OK | NORMAL |
| 90.0 s | 0.116 | 0.069 | OK | NORMAL |
| 95.0 s | 0.147 | 0.236 | OK | NORMAL |
| 100.0 s | 0.165 | 0.603 | OK | NORMAL |
| **102.0 s** | 0.175 | 0.743 | **DEGRADED** (`POSE_SIGMA_ELEVATED`, `ESTIMATOR_DEGRADED`) | **DEGRADED** |
| 110.0 s | 0.233 | 1.267 | DEGRADED | DEGRADED |
| **113.5 s** | 0.257 | 1.498 | **FAULT** (`LOCALIZATION_LOST`) | **HOLD** (`LOCALIZATION_LOST`, `POSE_SIGMA_HIGH`, `SAFE_HOLD_ACTION:STATION_KEEP`) |
| 122.7 s | 0.335 | 2.16 | FAULT | HOLD, now also `MISSION_BOUNDARY_VIOLATION` |
| 138.4 s | 0.463 | 3.49 | FAULT | HOLD, now also `COLLISION_ENVELOPE` |
| 150.0 s (end) | **0.529** | **4.631** | FAULT | HOLD |

- Time from the fix outage to DEGRADED: **12.0 s**. To LOCALIZATION_LOST and HOLD: **23.6 s**, inside the
  declared 40 s deadline. HOLD is held to the end of the mission.
- The estimator is **honest about the outage**: maximum error over sigma during the outage is 1.688, and it
  reaches that only in the first second (0.116 m error against 0.069 m sigma at 90.0 s, before the covariance
  has grown). By the end the error is 0.53 m against a 4.63 m sigma, so the filter over-reports its own
  ignorance rather than under-reporting it. Before the outage, error peaked at 0.268 m and sigma at 0.131 m.
- **The vehicle did not stop, and this is the run's main negative finding.** `conrad.sim.mission.unity_faults.check_safe_state`,
  applied with the declared expectation (HOLD, `LOCALIZATION_LOST`, `motion = hold`, hold radius 0.5 m),
  returns `passed: false`: `nominal_before_fault` true, `reached_in_time` true, `held_to_end` true,
  `no_collision` true, **`motion_rule` false**. The estimated position drifted 4.74 m from the hold point and
  the true mean speed over the last 10 s was **0.207 m/s** (threshold 0.1 m/s); the true mean speed after
  LOCALIZATION_LOST was 0.188 m/s.
- **Why.** The HOLD reason is `SAFE_HOLD_ACTION:STATION_KEEP`, not a zero-thrust stop. The last accepted
  navigation goal was a HOLD at 38.0 s. With no fixes, the EKF dead-reckons, the estimate walks away from the
  hold point, and the controller thrusts to chase it. This is not the NAV fault-case behaviour of the I2
  repair (`LOSS_OF_LOCALIZATION` there gives hold-point drift 0.019 m and speed 0.021 m/s on a 50 s run):
  there the fault fires 10 s into a transit and the run ends 40 s later, whereas here the vehicle is left
  station keeping on a drifting estimate for 36.5 s after HOLD. The gap is that STATION_KEEP in HOLD is
  driven by the same estimate the supervisor has just declared lost.
- **Hard-constraint violation.** The mission boundary is `[-10, -6, -1.5]` to `[10, 6, 5]` m. The
  supervisor raised `MISSION_BOUNDARY_VIOLATION` at 122.7 s, and this was not an artefact of the bad
  estimate: the **true** position was outside the boundary from 122.6 s, for 276 of 1500 control steps
  (the estimated position was outside for 274). `COLLISION_ENVELOPE` followed at 138.4 s. There were
  **0 collisions**, but the minimum true clearance over the whole mission was **0.058 m**.
- **The mission did not end early.** It ran all 1500 control steps to the declared 150 s. The runtime state
  history is BOOT, SELF_TEST, CONFIGURED, CONNECTING, READY, RUNNING, SAFE_HOLD, STOPPING, STOPPED, all three
  terminal entries with reason "mission duration reached". The SAFE_HOLD goal issued at 150.1 s was itself
  **refused** (`GOAL_OUTSIDE_ENVELOPE: waypoint outside the mission boundary`), because by then the estimate
  was outside the envelope. So the mission ended in HOLD/LOCALIZATION_LOST at its declared duration, and the
  final safe-hold manoeuvre could not be planned.

## 8. Decisions, commands and constraints

| | |
|---|---|
| EGDC decisions | 75 (QUERY_BELIEF 34, REQUEST_INFORMATION 22, STORE_AND_FORWARD 14, REVISIT_REGION 3, TRANSMIT_INFORMATION 2); 14 abstentions |
| Information needs (requirements) | 5 TECHNICAL, one at consequence 0.9 (the critical component) and four at 0.2 |
| MCBR plans | 4, all `NEED_SATISFIED`, planner `PRODUCTION[V-bayes_eig_ratio]`, 0 rejected candidates |
| Navigation goals | 6: TRANSIT at 0.1 s accepted, HOLD at 38.0 s accepted, REVISIT at 54.0 / 76.0 / 98.0 s all **refused** (`PlanningError: goal lies in space the active policy forbids`), SAFE_HOLD at 150.1 s refused (`GOAL_OUTSIDE_ENVELOPE`) |
| Commands | **1500 accepted, 0 rejected by the gateway, 0 refused by the safety supervisor** |
| Gateway rejection reasons | none |
| Collisions / minimum true clearance | **0** / **0.058 m** |
| Hard-constraint violations | `MISSION_BOUNDARY_VIOLATION` from 122.7 s (true position outside from 122.6 s, 276 of 1500 steps), `COLLISION_ENVELOPE` from 138.4 s |
| UIR | **0.0** (0 relied-on unsupported claims; 314 of 745 world claims are unsupported but none was relied on) |
| Module health | all 8 AVAILABLE, 0 failed modules |
| Beliefs published | 932 |
| Unity frames forwarded | range imager 49, sonar 49, camera 149; 150 Twin2E optics updates; 0 out-of-order packets |
| Energy | 5926 J (vehicle), 10.58 J (link) |

Three refused REVISIT goals are worth naming: EGDC asked to revisit the target three times after the lane
pass, and navigation refused all three because the requested point lies inside the space the planner policy
forbids. The mission therefore never re-observed the target after 20.3 s, which is why the contradiction is
never resolved by new evidence and U_C ends at 0.490.

## 9. Replay and falsification

### Truth-removal falsification (observation-only replay)

`python -m uv run python scripts/falsify_truth_removal.py artifacts/runs_capture/flagship_unity/FLAGSHIP-UNITY-s7800017`

Verdict: **PASS**. Report:
`artifacts/falsification/flagship_unity__FLAGSHIP-UNITY-s7800017/report.json`.

| Check | Result |
|---|---|
| `identical_without_truth` | true |
| `no_truth_path_opened_no_truth_module_imported` | true |
| `truth_corruption_has_no_effect` | true (control A) |
| `observation_change_is_detected` | true (controls B and C) |
| `unsealed_edit_refused` | true (control D) |
| `same_code_as_recording` | true |

Playback: 1500 ticks, 75,134 tape records, 322 verified files, 6 manifest files absent (the removed truth),
0 tape divergences, **1500 of 1500 commands checked with 0 mismatches**. Comparison against the original:
events 6609 = 6609 and byte-identical, all 18 SQLite tables identical, all 8 `mission/*` files identical,
commands 1500 = 1500 equal.

**Every path the playback process opened** (fresh `sys.addaudithook` on `open`, installed before `conrad` is
imported), 2885 distinct paths:

- bundle, 323 paths: `bundle_manifest.json`, `config.resolved.yaml`, the four `capture/*` files
  (`rhi_tape.jsonl.gz`, `mission_context.json`, `robot_config.json`, `capture_meta.json`) and 317
  `objects/sha256/*` payload objects;
- repository non-code: **`configs/active/mcbr_frozen_v2.yaml` only** (the frozen production planner config);
- the playback output directory, 327 paths (its own `events.jsonl`, `mission/*` (8), 317 objects and the
  captured RobotConfig copy);
- outside the repo: the null device `\\.\nul` only;
- Python code and libraries: 2233 files.

**Truth paths opened: none. Truth-side modules imported: none** (`conrad.twins`, `conrad.sim.kernel`,
`conrad.sim.unity`, `conrad.adapters.unity`, `conrad.sim.mission.{world,unity_world,truth,sensing,structure,run,unity_run}`,
`conrad.evaluation`, `conrad.orchestration.evaluation`). No Unity player is started to replay a Unity bundle.

### Full CC-10 replay (re-execution through a fresh player)

`python -m uv run python scripts/run_flagship_unity.py --replay artifacts/runs_capture/flagship_unity/FLAGSHIP-UNITY-s7800017`

Bundle verified: 338 files, 233 objects. Same player binary. Result:
`artifacts/flagship/replay_FLAGSHIP-UNITY-s7800017.json`.

| Item compared | Original | Replayed | Equal |
|---|---|---|---|
| Event signature | 6609 | 6609 | yes |
| Observation identities (id, sensor, modality, time, payload digest) | 505 | 505 | yes |
| Evidence digests (id, source observation, modality, time, independence group, payload) | 266 | 266 | yes |
| Belief revision order (belief, revision, predecessor, kind, time, late, provenance root, commit sequence) | 1346 | 1346 | yes |
| Uncertainty (U_A, U_E, U_C, U_O) per revision | 1346 | 1346 | yes, max abs difference **0.0** (declared tolerance 0.0) |
| Model1 claims and decisions (decision id, action, abstention, claim graph) | 75 | 75 | yes |
| Information needs | 5 | 5 | yes |
| MCBR candidate ranking and selected plan | 4 tables | 4 tables | yes |
| Navigation goals | 6 | 6 | yes |
| Command sequence (`COMMAND_SENT` payloads) | 1500 | 1500 | yes |
| BAAC deltas (`baac_transmissions.jsonl`) | 90 | 90 | yes |
| Receiver state | 1 | 1 | yes |
| Terminal mission status (runtime state history) | 9 | 9 | yes |
| Unity true trajectory | 151 samples | 151 samples | max abs difference **0.0**, tolerance 1e-9 m |

13 of 13 items equal.

### Development flight (Unity, world 7810007)

`artifacts/flagship/FLAGSHIP-UNITY-s7810007.json`, falsification
`artifacts/falsification/unitydev__DEV-FLAGSHIP-UNITY-s7810007/report.json`, verdict **PASS**.
Corrosion 0.01419 m against truth 0.00600 m (error 0.00819 m), crack 0.05527 m against truth 0.08000 m
(error 0.02473 m). U_C 0.000 before the second payload, 0.350 after, 0.084 at the end. Finding at 15.3 s
during the outage, alert at 71.25 s, delta at 76.67 s. DEGRADED at 102.0 s, LOCALIZATION_LOST and HOLD at
113.5 s (23.6 s after the fix outage), same `motion_rule` failure (estimated drift 4.33 m, true speed
0.202 m/s over the last 10 s), true position outside the boundary from 123.0 s. 1500 commands accepted,
0 rejected, 0 collisions, UIR 0.0.

## 10. What was changed

| File | Change |
|---|---|
| `conrad/sim/mission/scenarios.py` | **appended** `FLAGSHIP-UNITY` with the three stressors and the `FLAGSHIP_UNITY_*` time constants. No existing scenario touched. |
| `conrad/sim/mission/unity_run.py` | **appended** `UNITY_SCENARIOS["FLAGSHIP-UNITY"]` (base `FLAGSHIP-UNITY`, control period 0.1 s). |
| `conrad/sim/mission/unity_world.py` | `FIX_OUTAGE` faults are now realised on the Unity mission path: the build refuses only non-`FIX_OUTAGE` types (before, it refused every scheduled fault), the neutral payload suite is kept on the world as `UnityMissionWorld.suite`, and `due_faults` appends the outage window to `suite.dynamic_fix_outages`, exactly as `MissionWorld.due_faults` does. Unity-side faults (thruster, sensor, power, leak) still go through `run_unity_nav`'s bridge injection and are still refused here. |
| `configs/eval/flagship_unity.yaml` | new declaration (world, production-runtime expectations, stressor times, what is reported). |
| `scripts/run_flagship_unity.py` | new entry point: partition and runtime pre-flight checks, the flight with per-step sampling, the analysis, the falsification call and the full CC-10 replay comparison. |
| `.gitignore` | one whitelist line for `artifacts/flagship/`, excluding the per-step `trace_*.json` (about 560 kB each). |
| `unity/ConradUnityV2/Builds/Win64/ConradSim.exe` | rebuilt (see the note below). Gitignored. |

## 11. Honest notes

- **The Unity player had to be rebuilt, and that invalidates a binary identity in older evidence.** The
  shipped player was built on 2026-09-19 (`i2_repair_build.log`), before the I6 agent added
  `EcologyScene.cs`. The first flagship attempt failed with `BAD_REQUEST: unknown primitive kind
  'optics_grid'`, because Model2E is a production default and the Twin2E optics grid therefore reaches
  `CONFIGURE_SCENE`. The player was rebuilt from the unchanged checked-in C# with
  `BuildScript.BuildWindows64Player` (log `unity/ConradUnityV2/Logs/flagship_unity_build.log`, result
  Succeeded, 0 `error CS`, 0 `warning CS`). No C# was edited. The new exe sha256 is
  `36c5c9f1...c199cd277`, so `same_player_binary` will now read false for every bundle recorded with the
  previous binary (the I1, I2, I3 and I4 gate bundles). Their measured numbers are unchanged; only the
  identity check is.
- **The vehicle does not stop in HOLD on the integrated mission path.** Reported in section 7. It is a real
  defect of the flagship configuration, not of the checker: the checker was used unchanged and no threshold
  was moved. It is also the reason the mission ends with a `MISSION_BOUNDARY_VIOLATION` on the true position
  and a refused SAFE_HOLD goal. This belongs to the robotics owner and is not fixed here.
- **The safety log fed to `check_safe_state` is reconstructed.** The integrated mission path records safety
  state changes as events (`MissionExecutive.control_tick` emits on a state change or on a newly seen reason
  code), not a per-step row as `run_unity_nav` does. The per-step log passed to `check_safe_state` carries
  the state sampled every control step and the reasons of the most recent event. A reason code that
  disappears and returns without a state change would not re-appear in the reconstruction. The per-step
  states, the true positions and the estimated positions are sampled directly and are exact.
- **U_C was not zero before the injected contradiction** (0.350 at 14.3 s), because the target is read
  through two disagreeing truth-side surfaces. The "before" column of section 5 is therefore "before the
  second payload", not "before any disagreement".
- **The contradiction design is world-sensitive.** It needs the lane pass to reach the target before 16.0 s.
  On development world 7810009 it did not, and U_C stayed at 0. The final world happened to read the target
  first at 14.3 s.
- **The second payload makes the corrosion estimate worse**, by 7.6 mm. That is the intended behaviour of a
  credible disagreement, not a bug, but it should not be read as "the mission estimated corrosion well".
- **`min_clearance_m` is 0.058 m.** No collision was recorded, but the vehicle came within 6 cm of a
  structure while station keeping on a lost estimate. The margin is small enough that a slightly different
  world could have produced a collision.
- **One MCBR consequence worth naming:** all four plans returned `NEED_SATISFIED` and no INSPECT goal was
  ever accepted, so this run exercises the lane-pass path, not the closed active-inspection path. It says
  nothing about I4, which still FAILS.
- **The tree moved under this work, after the evidence was taken.** Another agent edited
  `conrad/domains/technical/config.py` and `conrad/domains/technical/measurement.py` while this audit was
  being written (file mtime 2026-09-20 10:16, repeatedly). The flagship run, its falsification and its full
  replay all completed before that, all at code digest
  `68bfade5aaaaad7e2ba5fd1d84f94e25b65b0bcb1977d300c9afc82cd415a9a2`, which the bundle records and which the
  falsification confirmed (`same_code: true`). The live tree is now at `ca55e442...`. Re-running the
  falsification on this bundle against the changed tree will report `INCONCLUSIVE_CODE_CHANGED`, not a leak.
  The bundle itself is intact: all 338 manifest files still match their digests.
- **One pre-existing test fails on the live tree for the same reason.**
  `tests/replay/test_mission_replay.py::test_ss02_same_config_gives_byte_equivalent_manifests` compares the
  bundle manifests of a GOLDEN-SMOKE run and its replay. The only differing entry is
  `capture/capture_meta.json`, and the only differing field inside it is `code_digest`: the two runs happen
  minutes apart in one pytest process, and `conrad/domains/technical/{config,measurement}.py` changed on
  disk in between. Everything else in that file (tape version, 12,330 records, run UUID, producer,
  RobotConfig digest) is identical. This is not caused by anything in this workstream, and no test was
  weakened for it. It is reported for the integrator: SS-02 as written can only hold on a static source
  tree, which is also why `docs/audits/FLAGSHIP_REPLAY_AND_FALSIFICATION.md` froze a copy of the tree before
  measuring. `tests/leakage` passes (27 of 27).
- **Not a gate.** `conrad gates status` is unaffected by this document.

## 12. Reproduce

```
# once, on the declared held-out world (launches the Unity player; run players sequentially)
python -m uv run python scripts/run_flagship_unity.py --partition final_test

# design dry run on a development world
python -m uv run python scripts/run_flagship_unity.py --partition development --seed 7810007

# full CC-10 replay of a bundle (launches a fresh player)
python -m uv run python scripts/run_flagship_unity.py --replay artifacts/runs_capture/flagship_unity/FLAGSHIP-UNITY-s7800017

# truth-removal falsification on its own
python -m uv run python scripts/falsify_truth_removal.py artifacts/runs_capture/flagship_unity/FLAGSHIP-UNITY-s7800017
```

Run bundles are not committed. They live under `artifacts/runs_capture/flagship_unity/` (final world) and
the replay output under `artifacts/runs_capture/flagship_unity/replay/` unless `--replay-scratch` moves it.
