# Twins: the truth world

The twins generate synthetic world truth and the observations a robot would get from it. They are truth-plane
code: only `conrad.sim`, `conrad.training`, `conrad.evaluation`, `conrad.data` and tests may use them. Twin1 does
not exist.

## Common interface (`conrad/twins/base.py`)

`Twin(ids, store)`:

- `initialize(Scenario)`, `step(dt_s, events)`, `reset(seed)`, `export_domain_state()`
- `get_truth(TimeStamp) -> list[TruthState]`
- `generate_observation(SensingContext) -> list[TwinSample]`

`SensingContext` holds the true pose (never leaves the truth plane), the estimated pose, the `SensorSpec` and a
degradation map. `TwinSample(observation, supervision)` keeps the `SupervisionLabel`
(`conrad/schemas/truth.py`: targets, target masks, true world-entity ID, lineage) on a separate channel that only
training and evaluation read. Observations carry only the estimated pose.

## The three twins

| Twin | Package | Engine | Observations | Config |
|---|---|---|---|---|
| Twin2T | `conrad/twins/twin2t` | MCDE (`mcde.py`): corrosion -> fatigue -> coupling -> events per tick | abstract structural observations with visibility and fidelity levels | `configs/sim/twin2t_default.yaml` |
| Twin2E | `conrad/twins/twin2e` | MEIFE: advected-diffused temperature/turbidity (upwind finite volume), prescribed current profile, diagnostic light, ecological entities | E0_ABSTRACT and E1_FEATURE; E2 sensor level is refused (renderer-owned) | `configs/sim/twin2e_default.yaml`, `twin2e_test_small.yaml` |
| Twin2S | `conrad/twins/twin2s` | SDF geometry, sparse voxel octree, ray casting, OCPWE observation schedules and counterfactual pairs | DEPTH_RANGE, RGB, SONAR, POINT_CLOUD, PRESSURE_DEPTH, IMU | `configs/sim/twin2s_default.yaml` |

Baselines and ablations:

- Twin2T: `GeneratorKind` (CONFIGURED, ENGINEERING, ENGINEERING_STOCHASTIC, ENGINEERING_TOPOLOGY,
  INDEPENDENT_STOCHASTIC, MCDE_NO_COUPLING, FULL_MCDE, RANDOM_WALK, RANDOM_STATIC_DEFECTS);
  `configs/sim/twin2t_independent_baseline.yaml`, `twin2t_random_walk_baseline.yaml`.
- Twin2E: `BASELINE_SWITCHES` and `ABLATIONS` in `twin2e/baselines.py`; `make_counterfactual_pair`,
  `make_confounded_variant`.
- Twin2S: OCPWE styles `coverage`, `redundant`, `poor`, `information_seeking`.

## Shared scenario

`conrad/sim/scenarios/pipeline_inspection.py`, `build_pipeline_inspection_scenario(seed, ids, ...)` builds one
`Scenario` with one set of world-entity IDs shared by all three twins (structural, ecological and spatial state, a
robot with five sensors and a `PIPELINE_INSPECTION` mission spec). `conrad/sim/mission/structure.py` maps it onto
each twin's contract for the integrated mission (in progress).

## Honesty rules in the code

- Every coefficient is ENGINEERING_ESTIMATE or SYNTHETIC_ONLY and uncalibrated (package metadata). Twin2E states
  it does not claim accurate marine ecosystem simulation.
- Twin2T has no crack initiation model; concrete and HDPE have no valid V1 mechanism and are fully masked.
- Twin2E has no learned component; its residual hook raises `LearnedResidualNotCalibratedError` if enabled.
- Twin2S marks an unobserved point INFERABLE only when an explicit counterfactual consensus mask says so,
  otherwise UNKNOWN.
- Per-component random streams come from (seed, entity UUID) in Twin2T.

## Tests

```
uv run pytest tests/unit/twins tests/property/twins tests/simulation/twin2s tests/simulation/twin2t tests/simulation/twin2e -q
```

## Limits and OPEN items

- Target-material priors, stochastic calibration and learned residuals (Twin2T); field solver or surrogate and
  target-environment priors (Twin2E); target assets, sensor priors and realism calibration (Twin2S).
- A twin-usefulness claim (twin-assisted vs real-only) needs real data: BLOCKED_EXTERNAL (EXT-DATA-01,
  `conrad/evaluation/claims.py`).
- Simulation validity is L1 at most.
