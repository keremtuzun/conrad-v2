# Integrated mission brief (Phases 5-11, gates I1-I7, flagship I4)

Binding design for `conrad/sim/mission/**` (truth side) and `conrad/orchestration/**` (deployment side).
The rules in [WORKSTREAM_BRIEF.md](WORKSTREAM_BRIEF.md) still apply.

## Two sides, one boundary

```
TRUTH SIDE (conrad/sim/mission)                    DEPLOYMENT SIDE (conrad/orchestration)
SharedScenario -> Twin2S / Twin2T / Twin2E          MissionRuntime: sees ONLY the RobotHardwareInterface,
SimRobotHardware (6-DOF kernel, Twin2S SDF)           mission context (asset registry, mission spec) and config
MissionSensorSuite: renders camera/sonar/depth  ==>   Observation objects  ==> ECMER/adapters -> Model2S/2T/2E
  from Twin2S, structural inspection from Twin2T                             -> Belief Bus -> EGDC -> MCBR
  (visibility from the Twin2S oracle), ecological                            -> NavigationStack -> Safety
  surveys and environmental sensors from Twin2E                              -> CommandGateway -> RHI.send
USBL-like position fix (true pose + noise)                                   -> BAAC -> ChannelSim -> receiver
MissionTruthRecorder: truth for evaluation only
```

- `conrad/orchestration/**` must never import `conrad.twins`, `conrad.sim` (except through the RHI object it is
  handed), `conrad.schemas.truth` or `conrad.evaluation`. `tests/leakage` enforces this.
- Upper layers never branch on adapter type. Extra sensing (structural inspection payload, environmental probes, the
  position fix) is exposed as `get_payload_observations() -> list[Observation]` on the hardware and declared in
  `RobotCapabilities.environmental_sensors`. The runtime calls it only if the capability is declared.
- The asset registry given to the runtime (mission context) uses **registry IDs that are not world-entity IDs**.
  The truth side keeps the mapping. Design geometry in the registry carries survey noise, and hidden condition is
  never included.
- Structural STRUCTURED observations carry sensor-shaped `measured_range_m` / `measured_bearing_rad` /
  `measured_elevation_rad` in `sensor_context`: noisy, and in the sensor frame. The runtime associates them with
  registry components by projecting through the ESTIMATED pose and gating. NO_MATCH is allowed.

## Flagship scenario FLAGSHIP-I4

- Shared pipeline-inspection scenario. One pipeline segment has a significant hidden defect (corrosion wall loss
  plus a crack) on its far (+Y) side. The robot's transit lane is on the -Y side, so the defect patch is invisible
  from the lane (Twin2S visibility oracle on the patch samples).
- Injected conditions: partial occlusion (the defect side), sensor noise, mission-relevant uncertainty (the segment's
  condition matters to the mission requirement), and a constrained acoustic link. Optional ones: credible
  contradiction (a second reading from a degraded sensor), localization degradation (a fix outage), a link outage,
  and an actuator fault.
- Expected causal chain:
  1. Transit observations leave the segment's condition UNKNOWN or weakly supported with high U_O.
  2. EGDC raises an InformationNeed.
  3. MCBR produces an ObservationPlan with a feasible +Y view (its candidate table and rejected candidates are stored).
  4. Navigation executes through the gateway.
  5. The new structural observation is associated and Model2T updates with DIRECT provenance.
  6. EGDC makes a grounded decision.
  7. BAAC transmits the critical finding under the constrained link.
  8. Replay reproduces the event signature.
- Metrics: before/after hidden-state error on the target segment (truth recorder), U_O/U_A/U_C before and after,
  decisions, UIR, commands accepted/rejected, bits sent, critical alert latency, and the causal trace from a command
  back to the raw observation. A **fixed-view baseline** runs the same scenario with MCBR replaced by a fixed
  inspection pattern. Acceptance thresholds are OPEN, so the I4 acceptance record is NOT_EVALUABLE; the report
  gives the measured numbers only.

## Run bundle

`artifacts/runs/<run_id>/`: config.resolved.yaml, events.jsonl, the SQLite DB with revisions, provenance and
commands, `mission/` (MCBR candidate tables, decisions, trajectories, BAAC transmissions, receiver state),
`truth/` (evaluation-only truth record), `reports/metrics.json`, and bundle_manifest.json (replay inputs +
digests). `conrad replay run --run <id>` verifies digests and then re-executes with the stored seed and config;
the event signatures must be equal.
