# Structural (2T) truth-leakage and proxy-leakage audit

Date: 2026-09-19. Scope: every input that reaches Model2T's `corrosion_depth_m` and `crack_length_m` estimates
in the integrated mission (FLAGSHIP-I4), from Twin2T truth to the evaluation metric.

**Why.** The flagship run reported crack error falling from 78.5 mm to 0.24 mm. Isolated experiment 2T-E001 lists
crack estimation as a failed hypothesis (model mean error 1.43 mm vs 0.84 mm for latest-observation). That
combination was treated as possible contamination until proven otherwise.

**Short answer.** No truth leak reaches Model2T. No world-entity UUID, hidden label, truth-derived confidence or
hidden noise level crosses the boundary. The flagship crack number came from an over-optimistic sensor model:
a 1 mm additive error on an 80 mm crack, the full crack length reported when only a third of the defect patch was
in view, deterministic detection, and 57 independent looks that averaged the error away. The metric was computed
correctly against the right truth quantity. Verdict: **VALID_BUT_OPTIMISTIC_SENSOR_MODEL**. The sensor model is
now fixed (configurable, realistic by default). Under it the same flagship crack error is 24.4 mm, not 0.24 mm.

## Method

- Read the full path: `conrad/twins/twin2t/{observation,emission,state,config,priors,twin}.py`,
  `conrad/sim/mission/{sensing,structure,registry,truth,world,options}.py`,
  `conrad/orchestration/{perception,association,evaluation}.py` and
  `conrad/domains/technical/{evidence,engine,direct,model,config}.py`.
- Queried the flagship bundle `artifacts/runs/FLAGSHIP-I4-s2026201-3fc605f1/`. It holds 416 observations and 296
  evidence rows in `conrad.sqlite`. It also has 9998 events, the `mission/*` files and `reports/metrics.json`.
  None contains any of the 19 world-entity UUIDs listed in the truth record. The structural `sensor_context` key
  set is exactly `{twin2t_fidelity, measurements, units, measured_range_m, measured_bearing_rad,
  measured_elevation_rad, range_sigma_m, angle_sigma_rad}`.
- Checked association against the truth mapping: 68 of 68 structural readings were matched, all correctly (57 on
  the target, 11 on other components). None was wrong and none was forced.
- Re-ran the flagship before any change. It reproduced the bundle exactly (57 target readings, crack error
  0.2423 mm). Re-ran it under the fixed sensor model, and re-ran 2T-E001 with a per-regime breakdown.

## Per-input lineage table

"Truth used" means truth consumed on the truth side to synthesize a sensor value. That is legitimate when a real
sensor would see the same physics. What matters is whether anything reaches Model2T that a real sensor could not
produce.

| # | Input reaching Model2T | Computed in | Derived from | Contains hidden value / UUID / truth confidence? | Verdict |
|---|---|---|---|---|---|
| 1 | `crack_indication_length` (T0 value 3) | `twin2t/observation.py::ObservationModel.t0` | true `crack_length_m` + N(0, 1 mm·g); step detection at `2 mm·g`; **full length at any visibility** | No exact value (noise). But 1 mm sd on 80 mm (1.25 %), iid per look, blind to how much of the crack is in view: at 33 % patch visibility it still reported 80.2 mm. It carried information about the unseen part of the crack. | **SENSOR_MODEL_TOO_OPTIMISTIC** (fixed) |
| 2 | `apparent_wall_loss` (T0 value 1) | same | true `corrosion_depth_m` + N(0, 0.2 mm·g) (+ `sensor_bias_m` in INT-003) | No. But 0.2 mm iid error on 6 mm wall loss from a visual payload, averaged over 57 looks, gives a 6 µm final error | **SENSOR_MODEL_TOO_OPTIMISTIC** (fixed) |
| 3 | `surface_anomaly_score` (T0 value 2) | same | 0.7·rust + 0.3·coating breakdown, ×(1−biofouling), + N(0, 0.05·g), clipped | No. A plausible visual appearance score. | CLEAN |
| 4 | Noise gain `g` (scales noise; never sent) | `ObservationModel.noise_gain` | Twin2E turbidity and biofouling at the observed point, curriculum corruption | Only environment, never component state. Model2T never receives `g`. | CLEAN |
| 5 | `sensor_health` → Evidence `reliability`, `aleatoric_uncertainty`, `validity` | `ObservationModel.health` → `domains/technical/evidence.py::HEALTH_RELIABILITY` | OK / DEGRADED / FAULT from `g` and corruption; Model2T's own reliability table | A coarse 3-level self-assessment a real payload can make from image contrast and fault flags. No truth confidence. | CLEAN |
| 6 | Existence and timing of a target reading | `sim/mission/sensing.py::_structural`, `world.py` target list | Twin2S visibility oracle on the defect patch samples; threshold 0.3 | The target segment is observable **only through its defect patch** (`world.py` excludes the target's own surface from the non-patch targets). The model therefore never receives near-side "no crack here" readings of the target. This is a selection effect in the scenario, not a value leak. | **SENSOR_MODEL_TOO_OPTIMISTIC** (not fixed, see Open items) |
| 7 | `measured_range_m / bearing / elevation` (+ sigmas) | `sensing.py::_with_geometry` | true pose → centroid of the visible patch points, + N(0, 5 cm) and N(0, 0.02 rad) | Sensor-frame geometry with noise. Standard synthetic sensor. | CLEAN |
| 8 | `robot_pose_estimate` | runtime EKF via `hardware.set_estimated_pose_provider` | estimator output with covariance | Not the true pose | CLEAN |
| 9 | Association hint `entity_candidates[].registry_entity_id` | `orchestration/association.py::StructuralAssociator` | projection through the ESTIMATED pose and the sensor mount from the mission context; gating on survey-noised design geometry (`sim/mission/registry.py::_design`, σ 5 cm) | Fresh registry IDs, never world IDs (`RegistryMapping` stays truth-side). No truth used. | CLEAN |
| 10 | `independence_group = obs:<observation_id>` | `orchestration/perception.py::_structural` | observation ID | Every look counts as independent. Together with #1 and #2 (iid noise), this is what let 57 looks average to 0.24 mm. | CLEAN (IDs); optimistic only through #1/#2 |
| 11 | Model2T asset registry (component type, material, coating, wall thickness, relations) | `sim/mission/registry.py::build_mission_context` | design keys only (`DESIGN_KEYS`); initial corrosion, cracks and the defect are not copied | Design data a real operator has | CLEAN |
| 12 | Model2T prior, dynamics and measurement σ | `domains/technical/config.py` | Model2T's own constants. `conrad.domains` cannot import `conrad.twins` (static test). | No code path, but numerical **calibration coupling**: prior means (0.5 mm corrosion, 1.5 mm crack) are the exact midpoints of Twin2T's uniform initial-state ranges. The corrosion rate 1e-4 m/yr is the geometric mean of Twin2T's `corrosion_A` range. The measurement σ is exactly 1.5× the old Twin2T σ. Not per-instance truth. It cannot touch the flagship crack (the defect sits 52 prior sd outside the prior), but it flatters E00x on non-defect components. | CLEAN (calibration coupling flagged) |
| 13 | Evaluation truth quantity | `orchestration/evaluation.py::evaluate_run_dir` | truth record `target_states.end.crack_length_m` for the registry→world mapped target vs the last Model2T revision | Right quantity (length, not depth), right component, right time. "Before" is the *prior mean* vs truth: a measure of ignorance, not a competing estimator. | CLEAN (framing caveat) |

Not on the integrated path, but noted: T1 feature vectors (`ObservationModel.t1`) embed `crack_depth_m / wall`,
which no visual payload measures. Per spec, T1 is "truth-conditioned features". The flagship uses T0 only. The
2T-E00x harness (`evaluation/structural_experiments/common.py`) sets the association hint from
`supervision.true_world_entity_id`. That is oracle association inside an evaluation package: acceptable there, it
favours E001 rather than the flagship, and it is not a runtime path.

### Answers to the specific questions

- **Is the T0 crack length just truth plus small noise?** Yes, before the fix: `true + N(0, 1 mm)` above a step
  threshold. At the stated fidelity (an abstract visual inspection payload at 0.3 to 4 m range) that is not
  realistic. Visual crack sizing is crude: tight tips are missed, only the in-view part can be sized, and the
  error is relative and largely systematic for one crack and one viewing setup.
- **Does the noise depend on true state in a leaky way?** No. Before the fix the noise was state-independent.
  Detection compared the true length to a threshold, which is a POD and is legitimate. After the fix the error is
  multiplicative in the true length, which is how sizing error behaves. The noise level is never sent to Model2T.
- **Does association use truth?** No. It uses the estimated pose, the mission-context sensor mount and
  survey-noised design geometry. It was 68/68 correct in the bundle.
- **Does `sensor_context` carry anything a sensor would not know?** No. There are no visibility fractions, no
  component type and no entity id. The Twin2S visibility fraction stays truth-side (degradation key and truth
  record only).
- **Do Model2T's prior or dynamics share parameters with Twin2T?** Not by code. They do by calibration (row 12).
- **Is the metric comparing against the right truth?** Yes (row 13).

## Verdict on the flagship crack metric

**VALID_BUT_OPTIMISTIC_SENSOR_MODEL.** The computation is correct and uncontaminated by truth. Its size comes from
the sensor model:

| FLAGSHIP-I4 s2026201 (mission_default.yaml) | Old sensor model (bundle) | Realistic sensor model (this audit) |
|---|---|---|
| Target readings | 57 | 56 |
| Crack readings, mean ± sd (true 80.0 mm) | 80.22 ± 1.25 mm | 36.5 ± 14.7 mm; 3 non-detections |
| Single-reading crack MAE (the latest-observation baseline) | 1.01 mm | 43.5 mm |
| Model2T crack error after 1 / 2 / 57 DIRECT updates | 45.8 / 0.74 / **0.242 mm** | final **24.4 mm** |
| Single-reading wall-loss MAE | 0.20 mm | 0.68 mm |
| Model2T corrosion error, final | 0.006 mm | 0.332 mm |
| Final U_C / condition | 0.0 / FAILED | 0.70 / FAILED |

The same code path on the other table rows (realistic model, current tree):

| Run | Old crack error (results doc) | New crack error | New corrosion error | New final condition |
|---|---|---|---|---|
| FIXED-VIEW s2026201 | 0.26 mm | 44.0 mm | 0.079 mm | SEVERE |
| FLAGSHIP-I4 s2026203 | 0.32 mm | 17.3 mm | 1.75 mm | FAILED |
| FIXED-VIEW s2026203 | 0.27 mm | 35.4 mm | 0.049 mm | FAILED |
| FLAGSHIP-I4 s2026202 | 0.18 mm | 79.2 mm (one late non-detection drove the estimate to 0.8 mm, U_C 1.0) | 0.13 mm | DEGRADED |

The old model re-run through the new code (`measurement_model: IDEALISED_ADDITIVE`) gives 0.247 mm on s2026201. The
difference from 0.242 mm comes from concurrent changes in other workstreams (EKF, MCBR) that shift the trajectory
(56 vs 57 looks). Before any edit, the untouched tree reproduced the bundle bit for bit.

Also note: the task framing "after one observation" does not match the run. After one DIRECT update the crack
error was still 45.8 mm (the 1.5 mm prior pulls the first reading halfway). The 0.24 mm came from 57 looks.

## The gap between 2T-E001 and the flagship

Both paths used the same Twin2T T0 sensor model and the same `Model2T` direct update, so the noise per reading was
comparable (single-reading crack MAE 0.85 mm in E001 vs 1.01 mm in the flagship). The gap comes from the
**regime**:

| | 2T-E001 (configs/eval/2t_e001.yaml) | FLAGSHIP-I4 |
|---|---|---|
| Time between looks | 60 days × 16 steps | 1 s × 57 looks within ~56 s |
| Looks per component | ≤ 1 per step | 57 on the target |
| Crack between looks | grows; 14.6 % of scored samples sit at the 1 m run-away cap | static (80.000 → 80.002 mm) |
| Association | oracle (true world id) | geometric, 68/68 correct |
| Baseline in the headline | latest observation | prior mean (78.5 mm) |

E001 crack MAE by true-length regime (legacy sensor, level 0 and 0.4 pooled, 3 seeds × 3 episodes, re-run here;
level 0 reproduces the doc's 1.43 vs 0.845 exactly):

| True crack | Share | Model2T | Latest obs. | Mean growth per step |
|---|---|---|---|---|
| < 3 mm | 70.5 % | 0.75 mm | 1.28 mm | 0.03 mm |
| 3 to 10 mm | 10.8 % | 1.77 | 1.96 | 0.66 |
| 10 to 50 mm | 2.9 % | 3.21 | 1.93 | 11.0 |
| 50 to 999 mm | 1.3 % | 1.90 | 1.88 | 154 |
| capped at 1 m | 14.6 % | **7.60** | 1.70 | 91 |
| static-like (< 5 mm) | 79 % | **0.81** | 1.34 | |

On near-static cracks the model beats latest-observation in E001 too. It loses on the mean only because of
run-away and capped cracks: its rate estimate keeps extrapolating after the crack stops at the 1 m cap, and a
growth prior of 0.5 mm/yr cannot follow 150 mm steps. The flagship crack is the static case with 57 iid looks,
exactly where averaging wins. So the two results are consistent, and neither is contaminated. The flagship
number showed averaging of an idealised sensor, not structural inference.

With the realistic sensor model, E001 becomes: level 0, model 46.6 mm vs latest 46.3 mm; level 0.4, 129.2 vs 129.3;
static-like cracks 1.04 vs 1.60 mm. The mean is now dominated by relative sizing error on large cracks.

## Fixes made

1. **Twin2T T0 sensor model** (`conrad/twins/twin2t/config.py`, `observation.py`, `emission.py`, `__init__.py`
   metadata). `ObservationConfig.measurement_model` defaults to `REALISTIC`; `IDEALISED_ADDITIVE` reproduces the
   old model with the same RNG order. REALISTIC defaults (all ENGINEERING_ESTIMATE, uncalibrated, configurable):
   - in-view crack length = true length × visibility^0.5 (`crack_visibility_exponent`): an area fraction in view
     maps to roughly its square root as a length fraction;
   - detection: log-logistic POD in the in-view length, a50 = 10 mm × noise gain (`crack_pod_a50_m`), width
     0.5 in ln-length (`crack_pod_log_width`). Turbidity and biofouling raise a50. A miss reports only the
     additive indication noise;
   - sizing: median 0.85 × in-view length (`crack_sizing_median_factor`, tips missed), a persistent per
     (component, sensor) log-normal bias with sd 0.20 (`crack_systematic_rel_sigma`) that repeated looks cannot
     average away, and per-reading log-normal scatter with sd 0.25 × gain (`crack_rel_sigma`). The additive 1 mm
     floor stays;
   - wall loss: persistent 5 % and per-reading 10 % × gain relative error, plus the 0.2 mm additive floor.
   - The persistent bias is drawn from a dedicated stream in first-observation order, never from the UUID value.
2. **Regression test** `tests/leakage/test_structural_lineage.py`:
   - (a) The serialized STRUCTURED Observations and their Evidence from a real FLAGSHIP-I4 run contain no
     world-entity or scenario UUID, no hidden Twin2T value within rtol 1e-9, no `visibility` / `component_type` /
     `supervision`, and only whitelisted `sensor_context` keys. Evidence quality context is not fabricated.
   - (b) The deployment-side structural pipeline (`structured_evidence` → `StructuralAssociator` → `Model2T`) is
     re-run on the captured payloads after the rebuilt truth world's hidden state and registry mapping are
     randomized (twice). The beliefs come out bit-identical. A positive control (scaled payloads) changes them.
   - (c) At Twin2T level, crack depth and wall thickness (not visible to T0) leave the payload unchanged, while
     crack length changes it.
3. **Unit tests** `tests/unit/twins/twin2t/test_t2t_sensor_model.py`: multiplicative scatter, persistent
   per-sensor bias, partial-view undersizing, POD in length and turbidity, relative wall-loss error, and legacy
   reproduction.

No LEAK was found, so nothing in `conrad/domains/technical`, `perception.py` or `association.py` was changed.

## Consequences and open items

- **Model2T's measurement model is now mis-specified (not a leak, out of scope here).** It assumes a 1.5 mm
  absolute crack σ. Against ~30 % relative error, every reading of a large crack is a "reliable contradiction":
  U_C saturates and the innovation-matched inflation makes the estimate chase the latest reading. On s2026202 a
  single non-detection dropped the estimate from about 100 mm to 0.8 mm. Model2T needs a length-relative sizing
  likelihood and a non-detection (POD) likelihood. That is a Model2T change for its owner.
- **Target selection effect (row 6).** The target segment only ever produces readings from its defect patch.
  Fixing it needs a near-side target and a local defect location in Twin2T (whose state is per-component). It
  touches `world.py`, which is outside this audit's paths.
- **Calibration coupling (row 12).** Model2T's population prior and rates are set at Twin2T's generator ranges.
  Any E00x claim on never-defected components should be read with that in mind.
- `configs/sim/twin2t_default.yaml` does not list the new keys. The mission and E00x paths construct `Twin2T` with
  code defaults, so the realistic model applies to them. A YAML that sets only the old keys also gets REALISTIC.
- Acceptance thresholds for I4 remain OPEN. None of these numbers is a performance claim.
