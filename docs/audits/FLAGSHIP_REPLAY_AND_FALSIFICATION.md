# Flagship replay (CC-10) and truth-removal falsification

Date: 2026-09-19. Reports: `artifacts/falsification/<parent>__<bundle>/report.json` (small JSON, tracked).
Harness: `scripts/falsify_truth_removal.py`, `conrad/sim/mission/capture.py`. Test:
`tests/replay/test_truth_removal_falsification.py`. All numbers below come from runs executed for this audit.

## 1. The recorded flagship run

**Provenance.** The flagship result reported in `docs/research/EXPERIMENT_RESULTS_2026-09-18.md` (crack error
78.5 -> 0.24 mm) and re-checked in `docs/audits/STRUCTURAL_LINEAGE_AUDIT.md` (0.2423 mm) is the bundle
`artifacts/runs/FLAGSHIP-I4-s2026201-3fc605f1`: scenario `FLAGSHIP-I4`, seed 2026201 (DEVELOPMENT partition),
config digest `3fc605f1...`, recorded at git commit `4478cd98`, planner `A-B10_mcbr_full`, Python L1 kernel,
120 s. Recorded headline: crack estimate 80.24 mm (truth 80.00 mm, error 0.24 mm), corrosion 5.995 mm,
57 target structural readings, 60 decisions, 2400 commands, 9998 events, 2132 belief revisions.

**Result: it does not replay with the current code.** The bundle itself is intact: `verify_bundle` checked 436 files
and 120 objects. The re-execution diverges (`artifacts/falsification/runs__FLAGSHIP-I4-s2026201-3fc605f1/cc10_replay_current_code.json`):

| Replay | Events | Decisions (60) | Revisions | First difference | Crack estimate (error) |
|---|---|---|---|---|---|
| `conrad replay run` as shipped | 9978 vs 9998 | differ from #1 | 2117 vs 2132 | event 5: `RUN_STARTED` planner `PRODUCTION` vs recorded `A-B10_mcbr_full` | 50.07 mm (29.9 mm) |
| after the replay fix below | 9476 vs 9998 | differ from #1 | 1800 vs 2132 | event 12, the first `COMMAND_SENT`: H1 -0.09328 vs recorded -0.09250 | no target reading, UNKNOWN |

Two separate causes:

1. **Replay bug (fixed).** `replay_run` re-resolved the scenario through the current scenario table and code defaults
   instead of using the options stored in the bundle. The `MissionRuntimeConfig.planner` default changed from
   `A-B10_mcbr_full` to `PRODUCTION`, so the "replay" silently ran a different planner. `replay_run` and
   `replay_unity_run` now pass the stored `mission_world_options` / `mission_runtime_config` into
   `prepare` / `prepare_unity` (new `stored_world` / `stored_runtime` parameters). They fail closed if the stored options
   no longer validate. Stored options validate today; the three fields added since then take their defaults
   (`route`, `time_budget_s`, `lane_obstacle`).
2. **Legitimate code drift.** With the stored options, the first thruster command already differs. Navigation,
   estimation, Model2T and Twin2T code changed after `4478cd98` (MODEL2T_REPAIR, the estimator work and the realistic
   Twin2T sensor). Under current code with the recorded planner, the robot never gets a structural reading of the
   target, so the target stays UNKNOWN. The recorded 0.24 mm is therefore **not reproducible by the current tree**. It
   remains a property of commit `4478cd98` only. A re-execution replay cannot check bundles across code changes; that
   is expected and is reported as a mismatch, not hidden.

**Fresh flagship bundle (current code, Python L1 kernel, labelled as such).** Same scenario, seed and resolved config:
`artifacts/runs_capture/FLAGSHIP-I4-s2026201-3fc605f1` (not committed, 435 files). It holds the new observation
capture (section 2). Headline: crack estimate 53.24 mm (error 26.76 mm), corrosion 5.77 mm, 9893 events,
2400 commands. Its observation-only replay is bit-exact (section 3), and the headline numbers recomputed from the
replayed beliefs match exactly (17 of 17 metric keys equal).

## 2. Why the existing CC-10 replay could not answer the truth question

- `replay_run` / `replay_unity_run` re-simulate the whole mission. They rebuild `MissionWorld` / `UnityMissionWorld`,
  including Twin2S, Twin2T and a Unity player, and `verify_bundle` hashes every manifest file first. Audit-hook result:
  `verify_bundle` opens `truth/truth_record.json` and `reports/metrics.json` (and, for Unity, all four `unity/*`
  files). With truth removed, the existing replay fails closed before running. It cannot show truth independence.
- **Unity replay did need Unity.** `replay_unity_run` launches a fresh player. The I1 gate bundle stores only 177
  `OBSERVATION_RECEIVED` payloads (range, sonar, camera, structural). It does not store the IMU, depth, power,
  thruster, health, clock and command-ack stream that the runtime reads every tick. The old bundles
  (`I1-UNITY-s5300059-3fdf2d30`, `FLAGSHIP-I4-s2026201-3fc605f1`) are therefore **not replayable from observations**.
  Both reports say so (`verdict: NOT_REPLAYABLE_FROM_OBSERVATIONS`).

**What was added.** The runtime sees the robot only through `RobotHardwareInterface`, so that boundary is where
capture happens. `RecordingHardware` (in `capture.py`, wired into `run.prepare` and `unity_run.prepare_unity`) wraps
the world's hardware object and appends every call the runtime makes to `capture/rhi_tape.jsonl.gz`: name, arguments
and return value, in order. The file is deterministic gzip. Also stored: tick, start, finish and artifact markers, the
driver's `FAULT_INJECTED` events, `capture/mission_context.json`, the RobotConfig the runtime loaded, and a digest of
`conrad/**/*.py`. `replay_inference` rebuilds the unchanged `MissionRuntime` on `PlaybackHardware`, which serves the
tape and raises `TapeDivergence` on any call the tape does not hold. It also checks every `send` command against the
recorded one. No twin, no world and no player is constructed.

One codec trap was found and fixed. `Pose.orientation_wxyz` re-normalizes in a validator, which is not idempotent
at the last ulp. On the first fresh flagship, 11 of 423 `observations` rows differed by 1 ulp in
`robot_pose_estimate`. Nothing downstream was affected (events and revisions were identical). The recorder now checks
that each JSON round trip is exact, and when it is not, it stores the exact object as a pickle next to the JSON.

## 3. Falsification results

Both recordings and all replays ran from one frozen copy of the source tree (code digest recorded = code digest at
replay, `same_code: true`). Other agents were editing `conrad/robotics/estimation/ekf.py` and
`conrad/domains/technical/*` during this work. Before freezing, a live-tree replay of an 8 s bundle diverged at the
first command (80 of 80 command mismatches). That was code drift, not a leak, which is why the code digest is now
recorded and the verdict distinguishes `INCONCLUSIVE_CODE_CHANGED`.

Truth layout moved out of the copy: `truth/truth_record.json` (trajectory, target states, registry-to-world map,
world entity IDs, scene), `reports/metrics.json` (truth-scored evaluation), and for Unity `unity/{scenario,experiment,
robot_config}.json` and `unity/player.log`.

| | Fresh FLAGSHIP-I4 s2026201 (Python kernel) | Fresh I1-UNITY s5300059 (Unity player) |
|---|---|---|
| Ticks / tape records | 2400 / 118085 | 600 / 29401 |
| Events (bytes) | 9893 = 9893, identical | 2685 = 2685, identical |
| SQLite tables (beliefs, revisions, evidence, observations, provenance, commands...) | 18 of 18 identical | 18 of 18 identical |
| `mission/*` (decisions, MCBR tables, trajectories, BAAC, associations...) | 8 of 8 identical | 8 of 8 identical |
| Commands | 2400 = 2400, 0 mismatches against the tape | 600 = 600, 0 mismatches |
| Truth paths opened | **none** | **none** |
| Truth-side modules imported (`conrad.twins`, `conrad.sim.kernel`, `conrad.sim.unity`, `sim.mission.world`/`truth`/`sensing`/`structure`/`run`, `conrad.evaluation`, `orchestration.evaluation`, `adapters.unity`) | **none** | **none** |
| Headline after truth restored (post-hoc) | crack 53.24 mm, 17 of 17 metric keys equal | 17 of 17 metric keys equal |
| Verdict | **PASS** | **PASS** |

Unity is **not** needed to replay a Unity bundle once it carries the capture. The playback process never imported
`conrad.sim.unity` or `conrad.adapters.unity` and started no player.

**Every path the playback process opened** (fresh `sys.addaudithook` on `open`, installed before `conrad` is
imported):

- bundle: `bundle_manifest.json`, `config.resolved.yaml`, the four `capture/*` files, and `objects/sha256/*`
  (418 flagship and 135 Unity payload objects; these are the raw observation payloads);
- repository non-code: `configs/active/mcbr_frozen.yaml` only (the frozen production planner config);
- the playback output directory (the new `events.jsonl`, `mission/*`, objects, and the captured RobotConfig copy);
- outside the repo: the null device `\\.\nul`;
- Python code and libraries: 2215 files (`.py`/`.pyc`/`.pyd`/`.dll` and the interpreter prefix).

## 4. Controls

| Control | Flagship | Unity I1 |
|---|---|---|
| A. truth corrupted in place (all truth/report/unity files overwritten, kept in the bundle) | output fingerprint identical to the truth-removed replay; no truth path opened | identical; no truth path opened |
| B. one depth sample +0.5 m (tape record 58866 / 14847, manifest resealed) | output changes: event stream differs from #4821; runtime asks for a call the tape does not hold at record 58874; 8 SQLite tables and 8 mission files differ | differs from event #1416; divergence at record 14855; 8 tables, 8 files differ |
| C. one captured STRUCTURED observation dropped (tape record 35) | differs from event #9; divergence at record 45; 9 tables, 8 files differ | differs from event #9; divergence at record 44 |
| D. the edit from B without resealing the manifest | refused: `required file corrupt: capture/rhi_tape.jsonl.gz` | refused (same) |

B and C show the test can detect a dependence. A single altered observation changes what the runtime does next,
including which hardware calls it makes. A tape cannot answer counterfactual calls, so playback stops at the first
unmatched call, and the truncated output is itself the detected difference.

## 5. Findings

- **No LEAK found.** The inference and decision planes (Model2S/2T, perception, association, MCBR/deliberation,
  routing, executive, navigation, command gateway, BAAC) re-run bit-exactly with every truth file removed. The
  process opened no truth path and imported no truth-side module.
- **Replay bug fixed.** CC-10 replay re-resolved options from current code (`conrad/sim/mission/replay.py`,
  `conrad/sim/mission/unity_run.py::replay_unity_run`). It now uses the stored options.
- **The recorded flagship number is not reproducible by the current tree.** It diverges at the first thruster command.
  With the recorded planner, the target is never read and stays UNKNOWN. The current-code flagship crack error is
  26.76 mm, not 0.24 mm.
- **Legacy bundles lack an observation stream.** Every bundle recorded before today (including all gate bundles) can
  only be replayed by re-simulating truth. New bundles carry `capture/` (flagship tape 118k records).
- **Note (not a leak):** `conrad/orchestration/evaluation.py::evaluate_run_dir` lives in the deployment-plane
  package but reads `truth/truth_record.json` by path. It is called only by the truth-side driver after the mission.
  The playback never imported it. Moving it under `conrad.evaluation` would make the static boundary test cover it.
  That file is outside this workstream.
- **Unity I1 numbers moved.** The fresh I1-UNITY run has 2685 events, against 2638 in the gate bundle, because the
  code changed since the gate recording. The gate evidence is tied to its own commit.

- **Capture crash reported by the I5 agent (fixed).** For a few minutes during this work, `MissionSession.finish`
  closed the tape before `write_mission_artifacts`. That call still queries the clock, so it failed with "I/O
  operation on closed file". The tape now closes after the artifacts are written. As a second fix, any hardware call
  after the tape is sealed is passed through without recording (`RecordingHardware.calls_after_close`), so a harness
  that scores the runtime after `finish()` cannot crash it. Verified with capture forced on through
  `m1_action_integrated.mission_job` (seed 7500000, arm `egdc_structured`): `I5-COMMS-OUTAGE` and `I5-ROUTE-BLOCKED`
  finish, and both bundles replay from observations bit-exactly (2400 ticks, 0 command mismatches). The I5 harness
  can drop `capture=False`. That file belongs to the I5 agent and was not edited here.

## 6. Reproduce

```
python -m uv run python scripts/falsify_truth_removal.py <bundle> [<bundle> ...] [--scratch DIR]
python -m uv run pytest tests/replay/test_truth_removal_falsification.py -q
```

The Unity test variant is skipped when no captured `artifacts/runs_capture/I1-UNITY-s5300059-*` bundle exists. It is
also skipped when the source changed since that recording.
